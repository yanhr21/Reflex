#!/usr/bin/env python3
"""Audit full301 task-state labels for external-target readout.

This is a label/readout-readiness check. It does not evaluate model outputs,
run inference, train a readout, or launch controller code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_STATE_NAMES = [
    "hole_pose_x",
    "hole_pose_y",
    "hole_pose_z",
    "peg_head_at_hole_x",
    "peg_head_at_hole_y",
    "peg_head_at_hole_z",
    "hole_velocity_step_x",
    "hole_velocity_step_y",
    "hole_velocity_step_z",
    "inserted",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-state-frames", type=int, default=301)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--motion-delta-threshold-m", type=float, default=0.002)
    parser.add_argument("--motion-velocity-threshold-m", type=float, default=1e-5)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def state_array_from_json(path: Path) -> tuple[list[str], np.ndarray, int | None]:
    data = json.loads(path.read_text())
    names = data.get("state_vector_names") or data.get("vector_names")
    if not isinstance(names, list):
        raise ValueError(f"{path}: missing state_vector_names")
    values = data.get("state") or data.get("state_targets") or data.get("states") or data.get("state_vectors")
    if values is None:
        raise ValueError(f"{path}: missing state vector values")
    arr = np.asarray(values, dtype=float)
    action_steps = data.get("num_action_steps")
    return names, arr, int(action_steps) if action_steps is not None else None


def scenario_from_row(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get("scenario"):
        return str(metadata["scenario"])
    uuid = str(row.get("uuid", "unknown"))
    return uuid.split("_seed")[0] if "_seed" in uuid else "unknown"


def audit_split(args: argparse.Namespace, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    jsonl = args.condition_root / split / "video_action_dataset_file.jsonl"
    rows = load_jsonl(jsonl)
    split_rows: list[dict[str, Any]] = []
    bad_rows: list[dict[str, Any]] = []
    motion_onsets: list[int] = []
    final_hole_deltas: list[float] = []
    final_insert_dists: list[float] = []
    scenarios: dict[str, int] = {}

    for idx, row in enumerate(rows):
        uuid = row.get("uuid")
        scenario = scenario_from_row(row)
        scenarios[scenario] = scenarios.get(scenario, 0) + 1
        state_path = Path(row.get("state_target_path") or row.get("task_state_target_path") or "")
        if not state_path.exists():
            bad_rows.append({"idx": idx, "uuid": uuid, "reason": "missing_state_target", "path": str(state_path)})
            continue
        try:
            names, arr, action_steps_from_file = state_array_from_json(state_path)
        except Exception as exc:  # noqa: BLE001 - report malformed row, continue audit
            bad_rows.append({"idx": idx, "uuid": uuid, "reason": "state_target_parse_error", "error": str(exc)})
            continue

        row_action_steps = row.get("action_chunk_size")
        action_steps = action_steps_from_file if action_steps_from_file is not None else row_action_steps
        if arr.shape != (args.expected_state_frames, len(names)):
            bad_rows.append(
                {
                    "idx": idx,
                    "uuid": uuid,
                    "reason": "state_shape_mismatch",
                    "shape": list(arr.shape),
                    "num_names": len(names),
                }
            )
            continue
        if action_steps != args.expected_action_steps:
            bad_rows.append(
                {
                    "idx": idx,
                    "uuid": uuid,
                    "reason": "action_step_mismatch",
                    "action_steps": action_steps,
                }
            )
            continue

        name_to_i = {name: i for i, name in enumerate(names)}
        missing = [name for name in REQUIRED_STATE_NAMES if name not in name_to_i]
        if missing:
            bad_rows.append({"idx": idx, "uuid": uuid, "reason": "missing_required_state_names", "missing": missing})
            continue

        hole = arr[:, [name_to_i["hole_pose_x"], name_to_i["hole_pose_y"], name_to_i["hole_pose_z"]]]
        velocity = arr[
            :,
            [
                name_to_i["hole_velocity_step_x"],
                name_to_i["hole_velocity_step_y"],
                name_to_i["hole_velocity_step_z"],
            ],
        ]
        peg_head_at_hole = arr[
            :,
            [
                name_to_i["peg_head_at_hole_x"],
                name_to_i["peg_head_at_hole_y"],
                name_to_i["peg_head_at_hole_z"],
            ],
        ]
        inserted = arr[:, name_to_i["inserted"]]
        hole_delta = np.linalg.norm(hole - hole[0], axis=1)
        hole_speed = np.linalg.norm(velocity, axis=1)
        onset_candidates = np.flatnonzero(
            (hole_delta > args.motion_delta_threshold_m) | (hole_speed > args.motion_velocity_threshold_m)
        )
        onset_frame = int(onset_candidates[0]) if onset_candidates.size else None
        final_hole_delta = float(hole_delta[-1])
        final_insert_dist = float(np.linalg.norm(peg_head_at_hole[-1]))

        if onset_frame is not None:
            motion_onsets.append(onset_frame)
        final_hole_deltas.append(final_hole_delta)
        final_insert_dists.append(final_insert_dist)
        split_rows.append(
            {
                "idx": idx,
                "uuid": uuid,
                "scenario": scenario,
                "onset_frame": onset_frame,
                "final_hole_delta_m": final_hole_delta,
                "final_insert_dist_m": final_insert_dist,
                "inserted_any": bool(np.nanmax(inserted) > 0.5),
                "num_frames": int(arr.shape[0]),
                "num_action_steps": int(action_steps),
            }
        )

    rows_with_motion = sum(1 for row in split_rows if row["onset_frame"] is not None)
    summary = {
        "jsonl": str(jsonl),
        "num_rows": len(rows),
        "num_valid_rows": len(split_rows),
        "num_bad_rows": len(bad_rows),
        "bad_rows": bad_rows[:20],
        "scenarios": scenarios,
        "rows_with_target_motion": rows_with_motion,
        "rows_without_target_motion": len(split_rows) - rows_with_motion,
        "motion_onset_min": min(motion_onsets) if motion_onsets else None,
        "motion_onset_max": max(motion_onsets) if motion_onsets else None,
        "motion_onset_mean": float(np.mean(motion_onsets)) if motion_onsets else None,
        "final_hole_delta_m_min": float(np.min(final_hole_deltas)) if final_hole_deltas else None,
        "final_hole_delta_m_max": float(np.max(final_hole_deltas)) if final_hole_deltas else None,
        "final_hole_delta_m_mean": float(np.mean(final_hole_deltas)) if final_hole_deltas else None,
        "final_insert_dist_m_min": float(np.min(final_insert_dists)) if final_insert_dists else None,
        "final_insert_dist_m_max": float(np.max(final_insert_dists)) if final_insert_dists else None,
        "final_insert_dist_m_mean": float(np.mean(final_insert_dists)) if final_insert_dists else None,
        "sample_rows": split_rows[:10],
    }
    return summary, split_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "condition_root": str(args.condition_root),
        "expected": {
            "state_frames": args.expected_state_frames,
            "action_steps": args.expected_action_steps,
            "motion_delta_threshold_m": args.motion_delta_threshold_m,
            "motion_velocity_threshold_m": args.motion_velocity_threshold_m,
        },
        "physical_reason": (
            "verify full301 labels contain external target-hole motion, final pose, "
            "and insertion geometry for post-SFT readout evaluation"
        ),
        "boundary": "label audit only; not model evidence and not controller evidence",
        "splits": {},
        "failures": [],
        "strict_external_target_label_ok": True,
    }

    for split in ["train", "val"]:
        split_summary, split_rows = audit_split(args, split)
        report["splits"][split] = split_summary
        (args.output_dir / f"{split}_rows.json").write_text(json.dumps(split_rows, indent=2, sort_keys=True) + "\n")
        if split_summary["num_bad_rows"]:
            report["strict_external_target_label_ok"] = False
            report["failures"].append(f"{split}_bad_rows:{split_summary['num_bad_rows']}")
        if split == "val" and split_summary["rows_with_target_motion"] == 0:
            report["strict_external_target_label_ok"] = False
            report["failures"].append("val_has_no_target_motion_rows")

    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and not report["strict_external_target_label_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
