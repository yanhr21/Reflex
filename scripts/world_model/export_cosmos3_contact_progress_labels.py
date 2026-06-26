#!/usr/bin/env python3
"""Export contact/progress labels for the Cosmos3 executor branch.

This is a data-label bridge for the DDP/HDP-inspired executor repair. It does
not train a controller and does not use future labels as controller-facing
conditions. The labels are supervision/evaluation targets for a future
contact-conditioned executor.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PHASE_NAMES = np.asarray(
    [
        "lost_grasp",
        "far",
        "lateral_align",
        "preinsert_aligned",
        "dp_continuable",
        "inserted",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-rows-per-split", type=int, default=0)
    parser.add_argument("--continuability-stats-json", default="")
    parser.add_argument("--continuability-profile", default="within_32_steps_to_first_success")
    parser.add_argument("--max-rel-x", type=float, default=0.04)
    parser.add_argument("--max-hole-speed", type=float, default=0.01)
    parser.add_argument("--fallback-min-rel-x", type=float, default=-0.13425659396282583)
    parser.add_argument("--fallback-max-abs-y", type=float, default=0.00984170909184218)
    parser.add_argument("--fallback-max-abs-z", type=float, default=0.003884278655338463)
    parser.add_argument("--insert-x-min", type=float, default=-0.015)
    parser.add_argument("--progress-far-x", type=float, default=-0.25)
    parser.add_argument("--lateral-align-radius", type=float, default=0.05)
    parser.add_argument("--preinsert-yz-multiplier", type=float, default=2.0)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _resolve_path(value: Any, *, base: Path) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def source_h5_for_row(row: dict[str, Any], condition_root: Path) -> Path | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in ("source_h5", "h5_path"):
        path = _resolve_path(row.get(key) or metadata.get(key), base=condition_root)
        if path is not None:
            return path
    state_path = _resolve_path(row.get("state_target_path"), base=condition_root)
    if state_path is not None:
        return state_path
    return None


def _first_trajectory(h5: h5py.File) -> h5py.Group:
    names = sorted([name for name in h5.keys() if name.startswith("traj_")], key=lambda x: int(x.split("_", 1)[1]))
    if len(names) != 1:
        raise RuntimeError(f"expected one trajectory, found {names}")
    return h5[names[0]]


def _array(group: h5py.Group, name: str, dtype: Any) -> np.ndarray:
    if name not in group:
        raise RuntimeError(f"missing dataset {group.name}/{name}")
    return np.asarray(group[name], dtype=dtype)


def _optional(group: h5py.Group | None, name: str, default: np.ndarray, dtype: Any) -> np.ndarray:
    if group is not None and name in group:
        return np.asarray(group[name], dtype=dtype)
    return np.asarray(default, dtype=dtype)


def load_episode(h5_path: Path) -> dict[str, Any]:
    with h5py.File(h5_path, "r") as h5:
        group = _first_trajectory(h5)
        slots = group["slots"]
        perturb = group["perturb"] if "perturb" in group else None
        summary = {}
        for attr_name in ("summary_json", "source_summary_json"):
            if attr_name in group.attrs:
                try:
                    summary = json.loads(group.attrs[attr_name])
                    break
                except Exception:
                    summary = {}
        n = int(_array(slots, "peg_head_at_hole", np.float32).shape[0])
        action_steps = max(0, n - 1)
        return {
            "summary": summary,
            "actions": _array(group, "actions", np.float32),
            "peg_head_at_hole": _array(slots, "peg_head_at_hole", np.float32),
            "hole_pose": _array(slots, "hole_pose", np.float32),
            "peg_pose": _array(slots, "peg_pose", np.float32),
            "tcp_pose": _array(slots, "tcp_pose", np.float32),
            "hole_velocity_step": _array(slots, "hole_velocity_step", np.float32),
            "grasped": _array(slots, "grasped", bool),
            "inserted": _array(slots, "inserted", bool),
            "robust_held": _optional(slots, "robust_held", np.zeros((n,), dtype=bool), bool),
            "hole_delta_cumulative": _optional(perturb, "hole_delta_cumulative", np.zeros((action_steps, 3), dtype=np.float32), np.float32),
            "peg_delta_applied": _optional(perturb, "peg_delta_applied", np.zeros((action_steps, 3), dtype=np.float32), np.float32),
            "triggered": _optional(perturb, "triggered", np.zeros((action_steps,), dtype=bool), bool),
        }


def load_continuability_thresholds(args: argparse.Namespace) -> dict[str, float]:
    out = {
        "min_rel_x": float(args.fallback_min_rel_x),
        "max_rel_x": float(args.max_rel_x),
        "max_abs_y": float(args.fallback_max_abs_y),
        "max_abs_z": float(args.fallback_max_abs_z),
        "max_hole_speed": float(args.max_hole_speed),
    }
    if args.continuability_stats_json:
        stats = read_json(Path(args.continuability_stats_json).resolve())
        profile = stats.get(str(args.continuability_profile), {})
        if isinstance(profile, dict):
            xq = profile.get("x_quantiles", {})
            yq = profile.get("y_abs_quantiles", {})
            zq = profile.get("z_abs_quantiles", {})
            out["min_rel_x"] = float(xq.get("0.01", out["min_rel_x"]))
            out["max_abs_y"] = float(yq.get("0.95", out["max_abs_y"]))
            out["max_abs_z"] = float(zq.get("0.95", out["max_abs_z"]))
    return out


def compute_labels(arrays: dict[str, Any], args: argparse.Namespace, thresholds: dict[str, float]) -> dict[str, np.ndarray]:
    head = np.asarray(arrays["peg_head_at_hole"], dtype=np.float32)
    if head.ndim != 2 or head.shape[1] < 3:
        raise RuntimeError(f"invalid peg_head_at_hole shape {head.shape}")
    n = int(head.shape[0])
    if n != 301:
        raise RuntimeError(f"expected 301 state frames, got {n}")
    for name in ("hole_pose", "peg_pose", "tcp_pose", "hole_velocity_step", "grasped", "inserted"):
        if int(np.asarray(arrays[name]).shape[0]) != n:
            raise RuntimeError(f"{name} has invalid length {np.asarray(arrays[name]).shape[0]}")

    x = head[:, 0].astype(np.float32)
    y = head[:, 1].astype(np.float32)
    z = head[:, 2].astype(np.float32)
    lateral = np.sqrt(np.square(y) + np.square(z)).astype(np.float32)
    hole_speed = np.linalg.norm(np.asarray(arrays["hole_velocity_step"], dtype=np.float32)[:, :3], axis=1).astype(np.float32)
    grasped = np.asarray(arrays["grasped"], dtype=bool)
    robust_held = np.asarray(arrays["robust_held"], dtype=bool)
    inserted = np.asarray(arrays["inserted"], dtype=bool)
    dp_continuable = (
        grasped
        & (x >= float(thresholds["min_rel_x"]))
        & (x <= float(thresholds["max_rel_x"]))
        & (np.abs(y) <= float(thresholds["max_abs_y"]))
        & (np.abs(z) <= float(thresholds["max_abs_z"]))
        & (hole_speed <= float(thresholds["max_hole_speed"]))
    )

    preinsert = (
        grasped
        & (x >= float(args.progress_far_x))
        & (x <= float(thresholds["max_rel_x"]))
        & (np.abs(y) <= float(args.preinsert_yz_multiplier) * float(thresholds["max_abs_y"]))
        & (np.abs(z) <= float(args.preinsert_yz_multiplier) * float(thresholds["max_abs_z"]))
    )
    lateral_align = grasped & (lateral <= float(args.lateral_align_radius)) & ~preinsert

    phase = np.full((n,), 1, dtype=np.int16)  # far
    phase[lateral_align] = 2
    phase[preinsert] = 3
    phase[dp_continuable] = 4
    phase[inserted] = 5
    phase[~grasped] = 0

    denom = max(float(args.insert_x_min) - float(args.progress_far_x), 1e-6)
    insertion_progress = np.clip((x - float(args.progress_far_x)) / denom, 0.0, 1.0).astype(np.float32)
    insertion_progress[inserted] = 1.0
    lateral_progress = np.clip(1.0 - lateral / max(float(args.lateral_align_radius), 1e-6), 0.0, 1.0).astype(np.float32)
    hold_progress = grasped.astype(np.float32)
    contact_progress = (0.45 * insertion_progress + 0.45 * lateral_progress + 0.10 * hold_progress).astype(np.float32)
    contact_progress[inserted] = 1.0

    first_insert = int(np.flatnonzero(inserted)[0]) if inserted.any() else -1
    steps_to_first_insert = np.full((n,), -1, dtype=np.int16)
    if first_insert >= 0:
        for i in range(n):
            steps_to_first_insert[i] = max(0, first_insert - i)

    action_steps = n - 1
    peg_delta = np.asarray(arrays["peg_delta_applied"], dtype=np.float32)
    hole_delta = np.asarray(arrays["hole_delta_cumulative"], dtype=np.float32)
    triggered = np.asarray(arrays["triggered"], dtype=bool)
    if peg_delta.shape[0] != action_steps:
        peg_delta = np.zeros((action_steps, 3), dtype=np.float32)
    if hole_delta.shape[0] != action_steps:
        hole_delta = np.zeros((action_steps, 3), dtype=np.float32)
    if triggered.shape[0] != action_steps:
        triggered = np.zeros((action_steps,), dtype=bool)

    return {
        "phase_id": phase,
        "phase_names": PHASE_NAMES,
        "peg_head_at_hole": head[:, :3].astype(np.float32),
        "lateral_error": lateral.astype(np.float32),
        "hole_speed": hole_speed.astype(np.float32),
        "grasped": grasped.astype(np.bool_),
        "robust_held": robust_held.astype(np.bool_),
        "inserted": inserted.astype(np.bool_),
        "dp_continuable": dp_continuable.astype(np.bool_),
        "insertion_progress": insertion_progress,
        "lateral_progress": lateral_progress,
        "contact_progress": contact_progress,
        "steps_to_first_insert": steps_to_first_insert,
        "hole_delta_cumulative": hole_delta.astype(np.float32),
        "peg_delta_applied": peg_delta.astype(np.float32),
        "perturb_triggered": triggered.astype(np.bool_),
    }


def row_chunk_record(
    *,
    row: dict[str, Any],
    split: str,
    source_h5: Path,
    episode_rel: Path,
    labels: dict[str, np.ndarray],
    chunk_size: int,
) -> dict[str, Any] | None:
    prefix = int(row.get("prefix_frame_index", -1))
    if prefix < 0 or prefix >= 300:
        return None
    end = min(300, prefix + int(chunk_size))
    future_slice = slice(prefix + 1, end + 1)
    future_inserted = bool(np.asarray(labels["inserted"])[future_slice].any())
    future_dp = bool(np.asarray(labels["dp_continuable"])[future_slice].any())
    progress = np.asarray(labels["contact_progress"], dtype=np.float32)
    phase = np.asarray(labels["phase_id"], dtype=np.int16)
    current_head = np.asarray(labels["peg_head_at_hole"], dtype=np.float32)[prefix]
    end_frame = end
    return {
        "uuid": row.get("uuid"),
        "split": split,
        "source_uuid": row.get("source_uuid"),
        "source_h5": str(source_h5),
        "scenario": row.get("scenario"),
        "prefix_role": row.get("prefix_role"),
        "prefix_frame_index": prefix,
        "chunk_end_frame": int(end_frame),
        "contact_label_npz_rel": str(episode_rel),
        "current_phase_id": int(phase[prefix]),
        "current_phase": str(PHASE_NAMES[int(phase[prefix])]),
        "current_peg_head_at_hole": current_head.astype(float).tolist(),
        "current_contact_progress": float(progress[prefix]),
        "chunk_end_contact_progress": float(progress[end_frame]),
        "chunk_contact_progress_delta": float(progress[end_frame] - progress[prefix]),
        "future_inserted_within_chunk": future_inserted,
        "future_dp_continuable_within_chunk": future_dp,
        "current_dp_continuable": bool(np.asarray(labels["dp_continuable"])[prefix]),
        "current_grasped": bool(np.asarray(labels["grasped"])[prefix]),
        "current_inserted": bool(np.asarray(labels["inserted"])[prefix]),
        "boundary": (
            "Contact/progress labels are supervision and evaluation targets. "
            "Controller-facing execution must use only causal live state and "
            "Cosmos-predicted task/contact paths, not future ground truth labels."
        ),
    }


def main() -> int:
    args = parse_args()
    condition_root = Path(args.condition_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    thresholds = load_continuability_thresholds(args)
    splits = [item.strip() for item in str(args.splits).split(",") if item.strip()]

    rows_by_split: dict[str, list[dict[str, Any]]] = {
        split: iter_jsonl(condition_root / split / "video_action_dataset_file.jsonl")
        for split in splits
    }

    source_order: OrderedDict[str, Path] = OrderedDict()
    for split, rows in rows_by_split.items():
        used_rows = 0
        for row in rows:
            if args.max_rows_per_split > 0 and used_rows >= int(args.max_rows_per_split):
                break
            used_rows += 1
            source_h5 = source_h5_for_row(row, condition_root)
            if source_h5 is None:
                continue
            key = str(source_h5)
            if key not in source_order:
                source_order[key] = source_h5
                if args.max_episodes > 0 and len(source_order) >= int(args.max_episodes):
                    break
        if args.max_episodes > 0 and len(source_order) >= int(args.max_episodes):
            break

    episode_records: list[dict[str, Any]] = []
    episode_label_by_source: dict[str, tuple[Path, dict[str, np.ndarray], dict[str, Any]]] = {}
    phase_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    success_count = 0

    for idx, (source_key, source_h5) in enumerate(source_order.items()):
        arrays = load_episode(source_h5)
        labels = compute_labels(arrays, args, thresholds)
        scenario = str((arrays["summary"] or {}).get("scenario", source_h5.parent.name.split("_seed", 1)[0]))
        source_uuid = str((arrays["summary"] or {}).get("source_uuid", source_h5.parent.name))
        rel = Path("episodes") / f"{idx:06d}_{source_h5.stem}.contact_progress.npz"
        out_path = output_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **labels)
        final_success = bool(np.asarray(labels["inserted"])[-1])
        success_count += int(final_success)
        scenario_counts[scenario] += 1
        unique, counts = np.unique(np.asarray(labels["phase_id"], dtype=np.int16), return_counts=True)
        for phase_id, count in zip(unique, counts):
            phase_counts[str(PHASE_NAMES[int(phase_id)])] += int(count)
        record = {
            "source_h5": str(source_h5),
            "source_uuid": source_uuid,
            "scenario": scenario,
            "contact_label_npz": str(out_path),
            "contact_label_npz_rel": str(rel),
            "num_frames": int(np.asarray(labels["phase_id"]).shape[0]),
            "final_success": final_success,
            "first_insert_frame": int(np.flatnonzero(np.asarray(labels["inserted"]))[0])
            if np.asarray(labels["inserted"]).any()
            else -1,
            "final_phase": str(PHASE_NAMES[int(np.asarray(labels["phase_id"])[-1])]),
            "final_peg_head_at_hole": np.asarray(labels["peg_head_at_hole"])[-1].astype(float).tolist(),
            "max_contact_progress": float(np.asarray(labels["contact_progress"]).max()),
            "thresholds": thresholds,
        }
        episode_records.append(record)
        episode_label_by_source[source_key] = (rel, labels, record)

    write_jsonl(output_root / "contact_progress_episode_labels.jsonl", episode_records)

    row_records_by_split: dict[str, list[dict[str, Any]]] = {}
    row_excluded: Counter[str] = Counter()
    for split, rows in rows_by_split.items():
        out_rows: list[dict[str, Any]] = []
        used_rows = 0
        for row in rows:
            if args.max_rows_per_split > 0 and used_rows >= int(args.max_rows_per_split):
                row_excluded["max_rows_per_split_reached"] += 1
                break
            used_rows += 1
            source_h5 = source_h5_for_row(row, condition_root)
            if source_h5 is None:
                row_excluded["missing_source_h5"] += 1
                continue
            item = episode_label_by_source.get(str(source_h5))
            if item is None:
                row_excluded["source_not_exported"] += 1
                continue
            rel, labels, _episode = item
            record = row_chunk_record(
                row=row,
                split=split,
                source_h5=source_h5,
                episode_rel=rel,
                labels=labels,
                chunk_size=int(args.chunk_size),
            )
            if record is None:
                row_excluded["invalid_prefix"] += 1
                continue
            out_rows.append(record)
        row_records_by_split[split] = out_rows
        write_jsonl(output_root / split / "contact_progress_row_labels.jsonl", out_rows)

    summary = {
        "schema": "cosmos3_contact_progress_labels_v1",
        "condition_root": str(condition_root),
        "output_root": str(output_root),
        "splits": splits,
        "chunk_size": int(args.chunk_size),
        "num_episodes": len(episode_records),
        "num_success_episodes": int(success_count),
        "num_row_labels": int(sum(len(v) for v in row_records_by_split.values())),
        "row_labels_by_split": {split: len(rows) for split, rows in row_records_by_split.items()},
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "phase_frame_counts": dict(sorted(phase_counts.items())),
        "row_excluded_counts": dict(sorted(row_excluded.items())),
        "continuability_thresholds": thresholds,
        "ready_for_contact_executor_dataset": bool(episode_records and sum(len(v) for v in row_records_by_split.values()) > 0),
        "boundary": (
            "Labels are for supervision, scoring, and diagnosis. They must not "
            "be used as future privileged controller inputs during live eval."
        ),
    }
    write_json(output_root / "contact_progress_label_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ready_for_contact_executor_dataset"] else 64


if __name__ == "__main__":
    raise SystemExit(main())
