#!/usr/bin/env python3
"""Build a real B/D Cosmos3 SFT condition root from joint overfit samples.

The output is an overfit data-interface artifact for Cosmos future/action
training. It derives action/state channels from recorded traces; it does not
generate rollouts, edit simulator state, or claim method evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


VECTOR_NAMES = [
    "action_0",
    "action_1",
    "action_2",
    "action_3",
    "action_4",
    "action_5",
    "action_6",
    "task_tcp_x",
    "task_tcp_y",
    "task_tcp_z",
    "task_peg_x",
    "task_peg_y",
    "task_peg_z",
    "task_hole_x",
    "task_hole_y",
    "task_hole_z",
    "task_peg_head_hole_x",
    "task_peg_head_hole_y",
    "task_peg_head_hole_z",
    "task_hole_velocity_x",
    "task_hole_velocity_y",
    "task_hole_velocity_z",
    "task_grasped",
    "task_inserted",
    "task_hole_delta_cumulative_x",
    "task_hole_delta_cumulative_y",
    "task_hole_delta_cumulative_z",
    "task_peg_delta_applied_x",
    "task_peg_delta_applied_y",
    "task_peg_delta_applied_z",
    "action_time_fraction",
    "task_perturb_triggered",
]


def require_slurm_unless_allowed(allow_login: bool) -> None:
    if allow_login:
        return
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_STEP_ID") == "extern":
        print(
            "refusing_login_node_execution=true\n"
            "reason=Run this converter inside a compute-node srun step.",
            file=sys.stderr,
        )
        raise SystemExit(30)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), sort_keys=True) + "\n")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def vec3(value: Any) -> np.ndarray:
    arr = np.asarray(value if value is not None else [0.0, 0.0, 0.0], dtype=np.float32).reshape(-1)
    out = np.zeros((3,), dtype=np.float32)
    out[: min(3, arr.shape[0])] = arr[:3]
    return out


def latent_prefix_indexes(prefix_frames: int, temporal_compression_factor: int) -> list[int]:
    latent_prefix = 1 + (int(prefix_frames) - 1) // int(temporal_compression_factor)
    return list(range(max(1, latent_prefix)))


def video_metadata(path: Path) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        framerate = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    if width <= 0 or height <= 0 or nb_frames <= 0:
        raise RuntimeError(f"bad video metadata for {path}: {width}x{height} frames={nb_frames}")
    return {
        "width": width,
        "height": height,
        "nb_frames": nb_frames,
        "framerate": framerate,
    }


def filter_rows(trace: dict[str, Any], key: str, entity_id: int, row_key: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in trace.get(row_key, [])
        if isinstance(row, dict) and int(row.get(key, -1)) == int(entity_id)
    ]
    return sorted(rows, key=lambda row: int(row.get("step", row.get("env_step", row.get("action_idx", 0)))))


def task_step(row: dict[str, Any]) -> int:
    return int(row.get("step", row.get("env_step", 0)))


def rebase_rows_for_window(rows: list[dict[str, Any]], start: int, count: int) -> list[dict[str, Any]]:
    """Select absolute-step rows inside a window and rebase step-like fields to 0."""
    end = int(start) + int(count)
    out: list[dict[str, Any]] = []
    for row in rows:
        absolute_step = task_step(row)
        if absolute_step < int(start) or absolute_step >= end:
            continue
        rebased = dict(row)
        for key in ("step", "env_step", "action_idx"):
            if key in rebased:
                rebased[key] = int(rebased[key]) - int(start)
        out.append(rebased)
    return sorted(out, key=task_step)


def materialize_video_window(source_path: Path, output_path: Path, start_frame: int, frame_count: int) -> dict[str, Any]:
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open source video: {source_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if width <= 0 or height <= 0 or total <= 0:
            raise RuntimeError(f"bad source video metadata for {source_path}: {width}x{height} frames={total}")
        if int(start_frame) < 0 or int(start_frame) + int(frame_count) > total:
            raise RuntimeError(
                f"window outside video: start={start_frame} frames={frame_count} total={total} path={source_path}"
            )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer: {output_path}")
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
            written = 0
            while written < int(frame_count):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(
                        f"short source video while writing window: written={written} requested={frame_count}"
                    )
                writer.write(frame)
                written += 1
        finally:
            writer.release()
    finally:
        cap.release()
    return video_metadata(output_path)


def build_action_state_array(
    action_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    motion_rows: list[dict[str, Any]],
    action_steps: int,
    state_frames: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if len(action_rows) < action_steps or len(task_rows) < state_frames:
        raise RuntimeError(
            f"not enough rows: actions={len(action_rows)} task={len(task_rows)} "
            f"action_steps={action_steps} state_frames={state_frames}"
        )
    task_by_step = {task_step(row): row for row in task_rows}
    first_task = task_rows[0]
    first_hole = vec3(first_task.get("hole_p"))
    first_peg = vec3(first_task.get("peg_p"))
    motion_steps = {int(row.get("step", -1)) for row in motion_rows}
    state_array = np.zeros((state_frames, len(VECTOR_NAMES)), dtype=np.float32)
    labels: list[dict[str, Any]] = []
    prev_hole: np.ndarray | None = None
    last_task = first_task
    for i in range(state_frames):
        if i < len(action_rows):
            action = np.asarray(action_rows[i].get("action", []), dtype=np.float32).reshape(-1)
        else:
            action = np.zeros((7,), dtype=np.float32)
        if action.shape[0] < 7:
            raise RuntimeError(f"action_dim_lt_7 at step {i}: {action.shape}")
        if i < len(action_rows):
            action_step = int(action_rows[i].get("step", action_rows[i].get("env_step", i)))
            task = task_by_step.get(i, task_by_step.get(action_step, last_task))
        else:
            task = task_by_step.get(i, last_task)
        last_task = task
        tcp = vec3(task.get("tcp_p"))
        peg = vec3(task.get("peg_p"))
        hole = vec3(task.get("hole_p"))
        peg_head = vec3(task.get("peg_head_p"))
        peg_head_at_hole = vec3(task.get("peg_head_at_hole", peg_head - hole))
        hole_velocity = np.zeros((3,), dtype=np.float32) if prev_hole is None else hole - prev_hole
        prev_hole = hole
        hole_delta = hole - first_hole
        peg_delta = peg - first_peg
        row = np.concatenate(
            [
                action[:7],
                tcp,
                peg,
                hole,
                peg_head_at_hole,
                hole_velocity,
                np.asarray([float(parse_bool(task.get("grasped", False))), float(parse_bool(task.get("inserted", False)))], dtype=np.float32),
                hole_delta,
                peg_delta,
                np.asarray([float(i) / float(max(1, state_frames - 1)), float(i in motion_steps)], dtype=np.float32),
            ]
        ).astype(np.float32)
        if row.shape[0] != len(VECTOR_NAMES):
            raise RuntimeError(f"bad vector width {row.shape[0]} != {len(VECTOR_NAMES)}")
        state_array[i] = row
        labels.append(
            {
                "frame": i,
                "grasped": bool(parse_bool(task.get("grasped", False))),
                "inserted": bool(parse_bool(task.get("inserted", False))),
                "peg_head_l2": float(task.get("peg_head_l2", np.linalg.norm(peg_head_at_hole))),
                "peg_head_at_hole": peg_head_at_hole.tolist(),
                "target_object": "hole",
                "tool_object": "peg",
            }
        )
    return state_array[:action_steps].copy(), state_array, labels


def make_record(sample: dict[str, Any], output_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    source_class = str(sample.get("source_class"))
    if source_class not in {"B", "D"}:
        raise ValueError(f"unsupported source_class for Cosmos SFT: {source_class}")
    source_video_path = Path(str(sample["video"])).resolve()
    source_vmeta = video_metadata(source_video_path)
    window_start = int(sample.get("window_start", 0))
    if int(source_vmeta["nb_frames"]) != int(args.source_video_frames):
        raise RuntimeError(
            f"video frame count mismatch for {sample.get('sample_id')}: "
            f"{source_vmeta['nb_frames']} != {args.source_video_frames}"
        )
    trace = read_json(Path(str(sample["trace"])))
    entity_key = str(sample["entity_key"])
    entity_id = int(sample["entity_id"])
    action_rows = rebase_rows_for_window(
        filter_rows(trace, entity_key, entity_id, "action_rows"),
        window_start,
        int(args.expected_steps),
    )
    task_rows = rebase_rows_for_window(
        filter_rows(trace, entity_key, entity_id, "task_rows"),
        window_start,
        int(args.expected_video_frames),
    )
    motion_rows = rebase_rows_for_window(
        filter_rows(trace, entity_key, entity_id, "motion_rows"),
        window_start,
        int(args.expected_video_frames),
    )
    action_array, state_array, frame_labels = build_action_state_array(
        action_rows,
        task_rows,
        motion_rows,
        int(args.expected_steps),
        int(args.expected_video_frames),
    )

    raw_sample_id = str(sample["sample_id"])
    sample_id = f"{source_class}_{raw_sample_id}"
    sample_dir = output_root / "samples" / sample_id
    window_video_path = sample_dir / "window_rgb.mp4"
    action_path = sample_dir / "action_state_32.npy"
    state_path = sample_dir / "state_targets.json"
    label_path = sample_dir / "task_labels.json"
    action_path.parent.mkdir(parents=True, exist_ok=True)
    vmeta = materialize_video_window(
        source_video_path,
        window_video_path,
        window_start,
        int(args.expected_video_frames),
    )
    np.save(action_path, action_array.astype(np.float32, copy=False), allow_pickle=False)
    write_json(state_path, {"states": state_array.tolist(), "state_vector_names": VECTOR_NAMES})
    write_json(
        label_path,
        {
            "summary": {
                "frame_labels": frame_labels,
                "target_object": "hole",
                "tool_object": "peg",
                "success_once": bool(sample.get("success_once")),
                "success_at_end": bool(sample.get("success_at_end")),
            },
        "source_sample": sample_id,
        "raw_source_sample": raw_sample_id,
        },
    )
    prefix_frame = max(0, min(int(sample.get("obs_horizon", 2)) - 1, int(args.expected_video_frames) - 1))
    prefix_frames = prefix_frame + 1
    prefix_role = "dynamic_future_observation" if source_class == "B" else "future_teacher_adapter"
    framerate = float(vmeta["framerate"])
    duration = float(args.expected_video_frames) / framerate if framerate > 0.0 else 0.0
    caption = (
        "Observed PegInsertionSide history up to the causal prefix. Predict future RGB, "
        "task state, and action/state channels without future target labels."
    )
    metadata = {
        "source_class": source_class,
        "dataset_class": sample.get("dataset_class"),
        "sample_id": sample_id,
        "raw_sample_id": raw_sample_id,
        "success_once": bool(sample.get("success_once")),
        "success_at_end": bool(sample.get("success_at_end")),
        "positive_dp_bc_allowed": bool(sample.get("positive_dp_bc_allowed")),
        "teacher_adapter_action_allowed": bool(sample.get("teacher_adapter_action_allowed")),
        "method_evidence_allowed": False,
        "prefix_causal_state": {
            "prefix_frame_index": prefix_frame,
            "source_window_start_frame": window_start,
            "source_window_end_frame": window_start + int(args.expected_video_frames) - 1,
            "condition_prefix_frames": prefix_frames,
            "prefix_role": prefix_role,
            "sampled_prefix_role": prefix_role,
            "mode": prefix_role,
            "grasped": bool(frame_labels[min(prefix_frame, len(frame_labels) - 1)]["grasped"]),
            "inserted": bool(frame_labels[min(prefix_frame, len(frame_labels) - 1)]["inserted"]),
            "target_motion_observed": bool(prefix_frame > 0 and np.linalg.norm(state_array[:prefix_frames, 24:27], axis=1).max() > 0.0),
            "peg_head_at_hole_xyz": frame_labels[min(prefix_frame, len(frame_labels) - 1)]["peg_head_at_hole"],
            "causal_boundary": "Only observed prefix task/action rows are described in metadata.",
        },
    }
    return {
        "uuid": f"joint_overfit_{sample_id}",
        "source_uuid": sample_id,
        "scenario": sample.get("dataset_role", source_class),
        "prefix_role": prefix_role,
        "sampled_prefix_role": prefix_role,
        "physical_mode": prefix_role,
        "target_object": "hole",
        "tool_object": "peg",
        "vision_path": str(window_video_path.resolve()),
        "source_vision_path": str(source_video_path),
        "source_window_start_frame": window_start,
        "source_window_end_frame": window_start + int(args.expected_video_frames) - 1,
        "width": int(vmeta["width"]),
        "height": int(vmeta["height"]),
        "nb_frames": int(vmeta["nb_frames"]),
        "framerate": framerate,
        "duration": duration,
        "source_video_frames": int(args.source_video_frames),
        "exported_video_frames": int(vmeta["nb_frames"]),
        "action_path": str(action_path.resolve()),
        "state_target_path": str(state_path.resolve()),
        "task_state_target_path": str(state_path.resolve()),
        "task_label_path": str(label_path.resolve()),
        "task_switch_label_path": str(label_path.resolve()),
        "num_video_frames": int(args.expected_video_frames),
        "num_action_steps": int(args.expected_steps),
        "action_chunk_size": int(args.expected_steps),
        "raw_action_dim": len(VECTOR_NAMES),
        "max_action_dim": 64,
        "fps": 30,
        "model_mode": "video_action_policy",
        "condition_prefix_frames": prefix_frames,
        "condition_frame_indexes_vision": latent_prefix_indexes(prefix_frames, int(args.temporal_compression_factor)),
        "condition_frame_indexes_action": list(range(max(0, prefix_frames - 1))),
        "condition_policy": "causal_prefix_only_no_future_target_labels",
        "t2w_windows": [{"caption": caption, "start_frame": 0, "end_frame": int(args.expected_video_frames) - 1, "temporal_interval": 1}],
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-dataset-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-video-frames", type=int, default=300)
    parser.add_argument("--expected-video-frames", type=int, default=300)
    parser.add_argument("--expected-steps", type=int, default=300)
    parser.add_argument("--temporal-compression-factor", type=int, default=4)
    parser.add_argument("--allow-login", action="store_true")
    args = parser.parse_args()

    require_slurm_unless_allowed(args.allow_login)
    joint_dir = Path(args.joint_dataset_dir).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        print("refusing_existing_nonempty_output_root=true", file=sys.stderr)
        print(f"output_root={output_root}", file=sys.stderr)
        raise SystemExit(31)
    samples = read_jsonl(joint_dir / "joint_overfit_samples.jsonl")
    records = [make_record(sample, output_root, args) for sample in samples if sample.get("source_class") in {"B", "D"}]
    if not records:
        raise SystemExit("no B/D records selected for Cosmos SFT condition root")
    for split in ("train", "val"):
        write_jsonl(output_root / f"{split}/video_action_dataset_file.jsonl", records)
        write_jsonl(output_root / f"{split}/video_dataset_file.jsonl", records)
    counts = Counter(str(row["metadata"]["source_class"]) for row in records)
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "schema": "joint_dp_cosmos_sft_condition_root_v3",
        "joint_dataset_dir": joint_dir,
        "output_root": output_root,
        "num_rows": len(records),
        "num_train": len(records),
        "num_val": len(records),
        "source_class_counts": dict(counts),
        "num_video_frames": int(args.expected_video_frames),
        "source_video_frames": int(args.source_video_frames),
        "num_action_steps": int(args.expected_steps),
        "raw_action_dim": len(VECTOR_NAMES),
        "vector_names": VECTOR_NAMES,
        "method_evidence_allowed": False,
        "training_started": False,
        "data_generation_started": False,
        "uses_toy_model": False,
        "contract": "B/D overfit Cosmos SFT condition root derived from real RGB videos and trace rows; no future target labels in conditions.",
        "cosmos_recipe_default_jsonl": {
            "train": str(output_root / "train/video_dataset_file.jsonl"),
            "val": str(output_root / "val/video_dataset_file.jsonl"),
        },
        "explicit_action_jsonl": {
            "train": str(output_root / "train/video_action_dataset_file.jsonl"),
            "val": str(output_root / "val/video_action_dataset_file.jsonl"),
        },
        "excluded_classes": {
            "A": "used for protected DP objective in joint overfit, not this dynamic Cosmos condition root",
            "C": "used for outcome/discrepancy labels, not positive Cosmos/DP action imitation",
        },
    }
    write_json(output_root / "manifest.json", manifest)
    print(json.dumps(jsonable(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
