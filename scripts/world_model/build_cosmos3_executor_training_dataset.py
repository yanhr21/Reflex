#!/usr/bin/env python3
"""Build a first-pass executor dataset from Cosmos3 dense full-episode rows.

This is an interface/preflight builder for the low-frequency-WM plus
high-frequency-executor branch. It deliberately records whether the task path
and DP-prior inputs are real controller-facing inputs or debug placeholders.
Full executor training is allowed only after those inputs are non-privileged.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROLES = (
    "target_motion_observed",
    "target_post_motion",
    "insert_resume",
    "peg_recovery",
)

TASK_PATH_NAMES = (
    "hole_pose_x",
    "hole_pose_y",
    "hole_pose_z",
    "peg_head_at_hole_x",
    "peg_head_at_hole_y",
    "peg_head_at_hole_z",
    "tcp_pose_x",
    "tcp_pose_y",
    "tcp_pose_z",
    "hole_velocity_step_x",
    "hole_velocity_step_y",
    "hole_velocity_step_z",
    "grasped",
    "inserted",
)

CURRENT_STATE_NAMES = (
    "tcp_pose_x",
    "tcp_pose_y",
    "tcp_pose_z",
    "peg_pose_x",
    "peg_pose_y",
    "peg_pose_z",
    "hole_pose_x",
    "hole_pose_y",
    "hole_pose_z",
    "qpos_0",
    "qpos_1",
    "qpos_2",
    "qpos_3",
    "qpos_4",
    "qpos_5",
    "qpos_6",
    "qpos_7",
    "qpos_8",
    "qvel_0",
    "qvel_1",
    "qvel_2",
    "qvel_3",
    "qvel_4",
    "qvel_5",
    "qvel_6",
    "qvel_7",
    "qvel_8",
    "peg_head_at_hole_x",
    "peg_head_at_hole_y",
    "peg_head_at_hole_z",
    "hole_velocity_step_x",
    "hole_velocity_step_y",
    "hole_velocity_step_z",
    "grasped",
    "inserted",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--include-roles", default=",".join(DEFAULT_ROLES))
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--max-rows-per-split", type=int, default=0)
    parser.add_argument("--require-prefix-grasped", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-gt-task-path-debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Write future GT task paths as debug conditioning fields. This is "
            "allowed for schema/overfit debugging only and never makes the "
            "dataset ready for formal executor training."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


class StateCache:
    def __init__(self, max_items: int = 32) -> None:
        self.max_items = max_items
        self._items: OrderedDict[str, tuple[list[str], np.ndarray]] = OrderedDict()

    def load(self, path: Path) -> tuple[list[str], np.ndarray]:
        key = str(path)
        if key in self._items:
            value = self._items.pop(key)
            self._items[key] = value
            return value
        payload = read_json(path)
        names = [str(name) for name in payload["state_vector_names"]]
        states = np.asarray(payload["states"], dtype=np.float32)
        if states.ndim != 2 or states.shape[0] != 301:
            raise RuntimeError(f"{path} has invalid states shape {states.shape}")
        value = (names, states)
        self._items[key] = value
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)
        return value


def column_indices(names: list[str], required: tuple[str, ...], path: Path) -> list[int]:
    by_name = {name: idx for idx, name in enumerate(names)}
    missing = [name for name in required if name not in by_name]
    if missing:
        raise RuntimeError(f"{path} missing state columns: {missing}")
    return [by_name[name] for name in required]


def denormalize_action_sidecar(action_path: Path, stats: dict[str, Any]) -> np.ndarray:
    arr = np.load(action_path, allow_pickle=False).astype(np.float32)
    if arr.ndim != 2 or arr.shape != (300, 32):
        raise RuntimeError(f"{action_path} has invalid action sidecar shape {arr.shape}")
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(1, -1)
    if mean.shape[1] != arr.shape[1] or std.shape[1] != arr.shape[1]:
        raise RuntimeError(f"normalization stats shape mismatch for {action_path}")
    return arr * std + mean


def rel_error(row: np.ndarray, names: list[str]) -> float:
    idxs = column_indices(
        names,
        ("peg_head_at_hole_x", "peg_head_at_hole_y", "peg_head_at_hole_z"),
        Path("<state_target>"),
    )
    return float(np.linalg.norm(row[idxs].astype(np.float32)))


def safe_bool(value: Any) -> bool:
    return bool(float(value) > 0.5)


def build_sample(
    *,
    row: dict[str, Any],
    split: str,
    row_index: int,
    condition_root: Path,
    output_root: Path,
    stats: dict[str, Any],
    state_cache: StateCache,
    chunk_size: int,
    allow_gt_task_path_debug: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    prefix_frame = int(row.get("prefix_frame_index", -1))
    if prefix_frame < 0:
        return None, "missing_prefix_frame_index"
    start_step = prefix_frame
    end_step = start_step + int(chunk_size)
    if end_step > 300:
        return None, "chunk_exceeds_episode_horizon"

    state_path = Path(str(row.get("state_target_path", ""))).resolve()
    action_path = Path(str(row.get("action_path", ""))).resolve()
    if not state_path.is_file():
        return None, "missing_state_target_path"
    if not action_path.is_file():
        return None, "missing_action_path"

    state_names, states = state_cache.load(state_path)
    current_idxs = column_indices(state_names, CURRENT_STATE_NAMES, state_path)
    task_idxs = column_indices(state_names, TASK_PATH_NAMES, state_path)
    prefix_state = states[prefix_frame]
    current_state = prefix_state[current_idxs].astype(np.float32)
    task_path_gt = states[prefix_frame + 1 : prefix_frame + 1 + chunk_size, task_idxs].astype(np.float32)
    if task_path_gt.shape != (chunk_size, len(TASK_PATH_NAMES)):
        return None, "invalid_task_path_shape"
    task_path = (
        task_path_gt
        if allow_gt_task_path_debug
        else np.zeros((0, len(TASK_PATH_NAMES)), dtype=np.float32)
    )

    state_by_name = {name: float(prefix_state[idx]) for idx, name in enumerate(state_names)}
    if not safe_bool(state_by_name["grasped"]):
        return None, "prefix_not_grasped"
    if safe_bool(state_by_name["inserted"]):
        return None, "prefix_already_inserted"

    raw_action_sidecar = denormalize_action_sidecar(action_path, stats)
    teacher_robot_actions = raw_action_sidecar[start_step:end_step, :7].astype(np.float32)
    if teacher_robot_actions.shape != (chunk_size, 7):
        return None, "invalid_teacher_action_shape"

    dp_prior_actions = np.zeros((0, 7), dtype=np.float32)
    task_path_source = "gt_state_targets_debug" if allow_gt_task_path_debug else "missing_cosmos_prediction"
    sample_rel = Path("samples") / split / f"{row_index:06d}_{row['uuid']}.npz"
    sample_path = output_root / sample_rel
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sample_path,
        current_state=current_state,
        current_state_names=np.asarray(CURRENT_STATE_NAMES),
        task_path=task_path,
        task_path_names=np.asarray(TASK_PATH_NAMES),
        teacher_robot_actions=teacher_robot_actions,
        dp_prior_actions=dp_prior_actions,
        prefix_frame=np.asarray([prefix_frame], dtype=np.int32),
        action_start_step=np.asarray([start_step], dtype=np.int32),
        action_end_step=np.asarray([end_step], dtype=np.int32),
    )

    start_error = rel_error(states[prefix_frame], state_names)
    end_error = rel_error(states[prefix_frame + chunk_size], state_names)
    blockers = ["missing_dp_prior_actions", "missing_cosmos_predicted_task_path"]
    if allow_gt_task_path_debug:
        blockers = ["missing_dp_prior_actions", "gt_task_path_debug_not_formal_evidence"]

    record = {
        "uuid": row["uuid"],
        "split": split,
        "sample_npz": str(sample_path),
        "sample_npz_rel": str(sample_rel),
        "source_uuid": row.get("source_uuid"),
        "source_h5": row.get("source_h5") or (row.get("metadata") or {}).get("source_h5"),
        "rgb_video": row.get("vision_path") or row.get("rgb_video"),
        "scenario": row.get("scenario"),
        "prefix_role": row.get("prefix_role"),
        "physical_mode": row.get("physical_mode"),
        "sampled_prefix_role": row.get("sampled_prefix_role"),
        "prefix_frame_index": prefix_frame,
        "chunk_size": int(chunk_size),
        "action_start_step": start_step,
        "action_end_step": end_step,
        "current_peg_head_at_hole": [
            state_by_name["peg_head_at_hole_x"],
            state_by_name["peg_head_at_hole_y"],
            state_by_name["peg_head_at_hole_z"],
        ],
        "current_grasped": safe_bool(state_by_name["grasped"]),
        "current_inserted": safe_bool(state_by_name["inserted"]),
        "start_peg_head_error_norm": start_error,
        "end_peg_head_error_norm": end_error,
        "teacher_error_delta": float(end_error - start_error),
        "task_path_source": task_path_source,
        "dp_prior_actions_available": False,
        "ready_for_debug_overfit": bool(allow_gt_task_path_debug),
        "ready_for_formal_executor_training": False,
        "formal_blockers": blockers,
        "boundary": (
            "GT task path is a debug/training scaffold only. Formal controller "
            "evidence must replace it with Cosmos-predicted causal task path "
            "and must provide frozen-DP prior actions."
        ),
    }
    return record, None


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Executor Dataset Preflight",
        "",
        f"condition_root: `{summary['condition_root']}`",
        f"output_root: `{summary['output_root']}`",
        f"chunk_size: `{summary['chunk_size']}`",
        "",
        "## Result",
        "",
        f"- schema_ok: `{summary['schema_ok']}`",
        f"- ready_for_debug_overfit: `{summary['ready_for_debug_overfit']}`",
        f"- ready_for_formal_executor_training: `{summary['ready_for_formal_executor_training']}`",
        f"- samples_written: `{summary['samples_written_total']}`",
        "",
        "## Formal Blockers",
        "",
    ]
    for blocker, count in summary["formal_blocker_counts"].items():
        lines.append(f"- `{blocker}`: `{count}`")
    lines.extend(["", "## Excluded Rows", ""])
    for reason, count in summary["excluded_reason_counts"].items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This starts the executor interface. It does not prove a controller.",
            "Full executor training needs causal Cosmos task-path predictions and",
            "frozen-DP prior action chunks, then closed-loop video/final-state eval.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    condition_root = Path(args.condition_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stats_path = condition_root / "normalization_stats.json"
    manifest_path = condition_root / "manifest.json"
    if not stats_path.is_file():
        raise SystemExit(f"missing normalization stats: {stats_path}")
    if not manifest_path.is_file():
        raise SystemExit(f"missing condition manifest: {manifest_path}")

    stats = read_json(stats_path)
    manifest = read_json(manifest_path)
    if int(manifest.get("num_video_frames", -1)) != 301 or int(manifest.get("num_action_steps", -1)) != 300:
        raise SystemExit("condition root does not satisfy 301/300 full-episode contract")
    if int(stats.get("raw_action_dim", -1)) != 32:
        raise SystemExit("normalization stats do not describe 32-D WAM sidecars")

    include_roles = {item.strip() for item in args.include_roles.split(",") if item.strip()}
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    state_cache = StateCache()
    all_records: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()

    for split in splits:
        rows = iter_jsonl(condition_root / split / "video_action_dataset_file.jsonl")
        used = 0
        for row_idx, row in enumerate(rows):
            role = str(row.get("prefix_role", ""))
            if include_roles and role not in include_roles:
                excluded["role_not_selected"] += 1
                continue
            if args.max_rows_per_split > 0 and used >= args.max_rows_per_split:
                excluded["max_rows_per_split_reached"] += 1
                continue
            record, reason = build_sample(
                row=row,
                split=split,
                row_index=row_idx,
                condition_root=condition_root,
                output_root=output_root,
                stats=stats,
                state_cache=state_cache,
                chunk_size=int(args.chunk_size),
                allow_gt_task_path_debug=bool(args.allow_gt_task_path_debug),
            )
            if record is None:
                excluded[str(reason)] += 1
                continue
            all_records.append(record)
            split_counts[split] += 1
            role_counts[role] += 1
            for blocker in record["formal_blockers"]:
                blocker_counts[blocker] += 1
            used += 1

    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}
    for record in all_records:
        by_split.setdefault(str(record["split"]), []).append(record)
    for split, rows in by_split.items():
        write_jsonl(output_root / split / "executor_dataset_file.jsonl", rows)

    schema_ok = bool(all_records)
    ready_for_debug_overfit = bool(args.allow_gt_task_path_debug and len(all_records) >= 2)
    summary = {
        "schema": "cosmos3_lowfreq_wm_executor_dataset_v1",
        "condition_root": str(condition_root),
        "output_root": str(output_root),
        "chunk_size": int(args.chunk_size),
        "include_roles": sorted(include_roles),
        "splits": splits,
        "samples_written_total": len(all_records),
        "samples_written_by_split": dict(sorted(split_counts.items())),
        "samples_written_by_role": dict(sorted(role_counts.items())),
        "excluded_reason_counts": dict(sorted(excluded.items())),
        "formal_blocker_counts": dict(sorted(blocker_counts.items())),
        "schema_ok": schema_ok,
        "ready_for_debug_overfit": ready_for_debug_overfit,
        "ready_for_formal_executor_training": False,
        "formal_training_blocker": (
            "No frozen-DP prior chunks and no Cosmos-predicted task path are present yet. "
            "GT task paths, when enabled, are debug scaffolds only."
        ),
        "full_training_resource_rule": "at least 2 GPUs for at least 3 hours after gates pass",
        "short_overfit_rule": "1-2 GPUs, about 50-100 steps, no 3-hour minimum, debug only",
    }
    write_json(output_root / "executor_dataset_preflight_summary.json", summary)
    write_markdown(output_root / "executor_dataset_preflight_summary.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if schema_ok else 64


if __name__ == "__main__":
    raise SystemExit(main())
