#!/usr/bin/env python3
"""Build one causal live-prefix Cosmos3 WAM policy inference sample.

This prepares the input format needed by ``cosmos_framework.scripts.inference``
for a receding controller step. It does not run Cosmos inference, does not
construct ManiSkill, and does not provide future object state as a condition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_cosmos3_maniskill_full_episode_wam_conditions import (  # noqa: E402
    FULL_EPISODE_VECTOR_NAMES,
    _caption,
    _load_episode_arrays,
    _prefix_latent_indexes,
    _prefix_payload,
    _raw_vectors_from_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prefix-video", required=True)
    parser.add_argument("--normalization-stats", required=True)
    parser.add_argument("--prefix-frame-index", type=int, required=True)
    parser.add_argument("--sample-name", default="live_prefix_receding_step")
    parser.add_argument("--source-h5", default=None)
    parser.add_argument("--scenario", default="live_dynamic")
    parser.add_argument("--prefix-role", default="live_receding")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--condition-root", default="")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--history-action-path", default=None)
    parser.add_argument("--total-video-frames", type=int, default=301)
    parser.add_argument("--total-action-steps", type=int, default=300)
    parser.add_argument("--raw-action-dim", type=int, default=32)
    parser.add_argument("--robot-action-dim", type=int, default=7)
    parser.add_argument("--temporal-compression-factor", type=int, default=4)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-future-frames-in-prefix-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic override only. By default the prefix video must contain "
            "exactly prefix_frame_index+1 frames so future RGB cannot leak into "
            "controller-facing Cosmos inference."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def count_media_frames(path: Path) -> int:
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return 1
    import imageio.v2 as imageio

    reader = imageio.get_reader(path)
    try:
        meta_count = reader.count_frames()
        if isinstance(meta_count, int) and meta_count > 0:
            return meta_count
    except Exception:
        pass
    count = 0
    try:
        for _ in reader:
            count += 1
    finally:
        reader.close()
    return count


def load_history_action(path: Path, expected_steps: int, raw_action_dim: int) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    else:
        payload = read_json(path)
        arr = np.asarray(payload.get("action", payload), dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"history action must be rank 2, got shape {arr.shape}")
    if arr.shape[0] > expected_steps:
        raise ValueError(f"history action has too many rows: {arr.shape[0]} > {expected_steps}")
    if arr.shape[1] not in {7, raw_action_dim}:
        raise ValueError(f"history action dim must be 7 or {raw_action_dim}, got {arr.shape[1]}")
    out = np.zeros((expected_steps, raw_action_dim), dtype=np.float32)
    out[: arr.shape[0], : arr.shape[1]] = arr
    return out


def prefix_payload_from_history(raw: np.ndarray, prefix_frame: int, prefix_role: str) -> dict[str, Any]:
    if raw.ndim != 2 or raw.shape[1] != len(FULL_EPISODE_VECTOR_NAMES):
        raise ValueError(f"live history raw action/state shape {raw.shape} is invalid")
    if prefix_frame <= 0:
        row_idx = 0
    else:
        row_idx = min(prefix_frame - 1, raw.shape[0] - 1)
    row = np.asarray(raw[row_idx], dtype=np.float32)
    values = {name: float(row[idx]) for idx, name in enumerate(FULL_EPISODE_VECTOR_NAMES)}
    hole_history_end = max(0, min(prefix_frame, raw.shape[0]))
    hole_delta = 0.0
    if hole_history_end > 0:
        hole = np.asarray(raw[:hole_history_end, 13:16], dtype=np.float32)
        if hole.size:
            hole_delta = float(np.max(np.linalg.norm(hole - hole[0:1], axis=1)))

    def xyz(prefix: str) -> list[float]:
        return [values[f"{prefix}_{axis}"] for axis in ("x", "y", "z")]

    hole_velocity = xyz("task_hole_velocity")
    return {
        "prefix_frame_index": int(prefix_frame),
        "condition_prefix_frames": int(prefix_frame + 1),
        "prefix_role": prefix_role,
        "mode": prefix_role,
        "mode_id": -1,
        "tcp_xyz": xyz("task_tcp"),
        "peg_xyz": xyz("task_peg"),
        "target_hole_xyz": xyz("task_hole"),
        "peg_head_at_hole_xyz": xyz("task_peg_head_hole"),
        "hole_velocity_xyz": hole_velocity,
        "target_motion_observed": bool(
            hole_delta > 0.002 or float(np.linalg.norm(np.asarray(hole_velocity, dtype=np.float32))) > 0.001
        ),
        "peg_needs_recovery": bool(values["task_grasped"] <= 0.5),
        "grasped": bool(values["task_grasped"] > 0.5),
        "inserted": bool(values["task_inserted"] > 0.5),
        "causal_boundary": (
            "Live-prefix caption reconstructed from observed history row "
            "prefix_frame_index-1. It contains current observed task state only; "
            "future rows remain unconditioned."
        ),
    }


def build_raw_action_state(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    prefix_frame = int(args.prefix_frame_index)
    prefix_frames = prefix_frame + 1
    raw = np.zeros((args.total_action_steps, args.raw_action_dim), dtype=np.float32)
    source_summary: dict[str, Any] = {
        "source": "empty_future_unconditioned_rows",
        "history_rows_conditioned": prefix_frame,
    }

    if args.history_action_path:
        history = load_history_action(Path(args.history_action_path), args.total_action_steps, args.raw_action_dim)
        raw[:prefix_frame] = history[:prefix_frame]
        source_summary.update(
            {
                "source": "provided_live_history_action_rows_only",
                "history_action_path": str(Path(args.history_action_path).resolve()),
                "source_h5_provenance": str(Path(args.source_h5).resolve()) if args.source_h5 else None,
                "prefix_payload": prefix_payload_from_history(raw, prefix_frame, args.prefix_role),
                "precedence": (
                    "history_action_path overrides source_h5 for controller-facing "
                    "live inference. Source H5 rows must not replace real executed "
                    "history after the first live action chunk."
                ),
            }
        )
    elif args.source_h5:
        arrays = _load_episode_arrays(Path(args.source_h5), args)
        full_raw = _raw_vectors_from_arrays(
            arrays,
            args.total_video_frames,
            prefix_frames,
            "future_aligned_state",
        )
        if full_raw.shape != raw.shape:
            raise ValueError(f"source raw shape {full_raw.shape} != expected {raw.shape}")
        # Only observed history rows are clean conditions. Future rows remain
        # zero and unconditioned so source H5 future object states are not
        # available to controller-facing inference.
        raw[:prefix_frame] = full_raw[:prefix_frame]
        payload = _prefix_payload(arrays, {"frame_labels": [{"mode": args.prefix_role, "mode_id": -1, "target_started": False, "peg_needs_recovery": False, "insert_resume_candidate": False, "grasped": False, "inserted": False}] * args.total_video_frames}, prefix_frame, args.prefix_role)
        source_summary.update(
            {
                "source": "source_h5_observed_history_rows_only",
                "source_h5": str(Path(args.source_h5).resolve()),
                "prefix_payload": payload,
            }
        )
    else:
        source_summary.update(
            {
                "warning": "No source_h5/history_action_path was provided, so action/state history conditions are all zero.",
            }
        )

    denom = max(1, args.total_action_steps - 1)
    raw[:, args.raw_action_dim - 2] = np.arange(args.total_action_steps, dtype=np.float32) / float(denom)
    return raw, source_summary


def normalize_action(raw: np.ndarray, stats: dict[str, Any], expected_dim: int) -> np.ndarray:
    if stats.get("type") != "zscore":
        raise ValueError(f"expected zscore normalization stats, got {stats.get('type')!r}")
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if mean.shape != (expected_dim,) or std.shape != (expected_dim,):
        raise ValueError(f"normalization stats shape mismatch: mean={mean.shape}, std={std.shape}")
    std = np.where(np.abs(std) < 1.0e-6, 1.0, std)
    return ((raw - mean[None, :]) / std[None, :]).astype(np.float32)


def prompt_from_args(args: argparse.Namespace, source_summary: dict[str, Any]) -> str:
    if args.prompt.strip():
        return args.prompt.strip()
    payload = source_summary.get("prefix_payload")
    if isinstance(payload, dict):
        return _caption(payload)
    return (
        "PegInsertionSide full-episode world/action model live-prefix sample. "
        "Roles: TARGET_OBJECT=hole, TOOL_OBJECT=peg, ACTOR=robot_gripper_tcp. "
        f"Observed prefix role={args.prefix_role} through frame {args.prefix_frame_index}. "
        "Predict target motion, future RGB/task state, and executable actions in the changed world."
    )


def main() -> int:
    args = parse_args()
    if args.total_video_frames != 301 or args.total_action_steps != 300:
        raise SystemExit("active contract requires 301 video frames and 300 action steps")
    if args.raw_action_dim != len(FULL_EPISODE_VECTOR_NAMES):
        raise SystemExit(
            f"raw_action_dim={args.raw_action_dim} does not match WAM vector names {len(FULL_EPISODE_VECTOR_NAMES)}"
        )
    prefix_frame = max(0, min(int(args.prefix_frame_index), args.total_video_frames - 2))
    args.prefix_frame_index = prefix_frame

    prefix_video = Path(args.prefix_video).resolve()
    if not prefix_video.is_file():
        raise FileNotFoundError(prefix_video)
    prefix_video_frames = count_media_frames(prefix_video)
    expected_prefix_video_frames = prefix_frame + 1
    if (
        prefix_video_frames != expected_prefix_video_frames
        and not args.allow_future_frames_in_prefix_video
    ):
        raise ValueError(
            f"prefix_video frame count {prefix_video_frames} != observed prefix "
            f"frames {expected_prefix_video_frames}. Provide a prefix-only video "
            "or pass --allow-future-frames-in-prefix-video for a non-method diagnostic."
        )
    stats_path = Path(args.normalization_stats).resolve()
    stats = read_json(stats_path)
    raw, source_summary = build_raw_action_state(args)
    action = normalize_action(raw, stats, args.raw_action_dim)
    if not np.isfinite(action).all():
        raise ValueError("normalized action/state condition contains non-finite values")

    output_root = Path(args.output_root).resolve()
    input_dir = output_root / "inputs"
    action_path = input_dir / "actions" / f"{args.sample_name}.json"
    write_json(action_path, {"action": action.astype(float).tolist()})

    condition_action = list(range(prefix_frame))
    condition_vision = _prefix_latent_indexes(prefix_frame + 1, args.temporal_compression_factor)
    row = {
        "name": args.sample_name,
        "model_mode": "policy",
        "prompt": prompt_from_args(args, source_summary),
        "vision_path": str(prefix_video),
        "action_path": str(action_path),
        "domain_name": "maniskill_peg_insertion",
        "view_point": "third_person_view",
        "fps": args.fps,
        "num_frames": args.total_video_frames,
        "image_size": args.image_size,
        "aspect_ratio": "1,1",
        "action_chunk_size": args.total_action_steps,
        "raw_action_dim": args.raw_action_dim,
        "condition_frame_indexes_vision": condition_vision,
        "condition_frame_indexes_action": condition_action,
        "num_steps": 30,
        "guidance": 1.0,
        "seed": int(args.seed),
        "extra": {
            "condition_root": args.condition_root,
            "checkpoint_path": args.checkpoint_path,
            "scenario": args.scenario,
            "prefix_role": args.prefix_role,
            "prefix_frame_index": prefix_frame,
            "condition_prefix_frames": prefix_frame + 1,
            "expected_video_frames": args.total_video_frames,
            "expected_action_steps": args.total_action_steps,
            "expected_action_dim": args.raw_action_dim,
            "target_object": "hole",
            "tool_object": "peg",
            "actor": "robot_gripper_tcp",
            "source_summary": source_summary,
            "prefix_video_frames": prefix_video_frames,
            "causal_boundary": (
                "Only observed RGB prefix latent frames and action/state rows "
                "strictly before prefix_frame_index are clean conditions. "
                "Future target/peg/TCP states are not provided."
            ),
        },
    }
    input_jsonl = input_dir / "live_prefix_wam_policy_samples.jsonl"
    write_jsonl(input_jsonl, [row])

    manifest = {
        "boundary": (
            "Live-prefix inference input only. This is the per-reobservation "
            "input needed for a real receding Cosmos controller; it is not "
            "controller evidence until inference, env.step, metrics, and video "
            "review run inside a compute allocation."
        ),
        "input_jsonl": str(input_jsonl),
        "output_root": str(output_root),
        "sample_name": args.sample_name,
        "scenario": args.scenario,
        "prefix_role": args.prefix_role,
        "prefix_frame_index": prefix_frame,
        "prefix_video": str(prefix_video),
        "action_json": str(action_path),
        "sample": row["extra"] | {"name": args.sample_name, "prefix_video": str(prefix_video), "action_json": str(action_path)},
        "normalization_stats": str(stats_path),
        "vector_names": FULL_EPISODE_VECTOR_NAMES,
        "condition_frame_indexes_vision": condition_vision,
        "condition_frame_indexes_action": condition_action,
        "condition_frame_indexes_action_len": len(condition_action),
        "condition_action_max_index": max(condition_action) if condition_action else None,
        "prefix_video_frames": prefix_video_frames,
        "expected_prefix_video_frames": expected_prefix_video_frames,
        "allow_future_frames_in_prefix_video": bool(args.allow_future_frames_in_prefix_video),
        "strict_live_prefix_input_ok": True,
    }
    write_json(output_root / "live_prefix_input_manifest.json", manifest)
    print(json.dumps({"input_jsonl": str(input_jsonl), "manifest": str(output_root / "live_prefix_input_manifest.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
