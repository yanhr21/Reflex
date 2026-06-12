#!/usr/bin/env python3
"""Audit full301 Cosmos3 WAM action targets.

This validates the 300-step structured action/state target sidecars used by
the joint-policy WAM SFT. It is a data-contract/readiness check only: it does
not run inference, train, or evaluate controller behavior.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
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
    "prefix_tcp_x",
    "prefix_tcp_y",
    "prefix_tcp_z",
    "prefix_peg_x",
    "prefix_peg_y",
    "prefix_peg_z",
    "prefix_hole_x",
    "prefix_hole_y",
    "prefix_hole_z",
    "prefix_peg_head_hole_x",
    "prefix_peg_head_hole_y",
    "prefix_peg_head_hole_z",
    "prefix_hole_velocity_x",
    "prefix_hole_velocity_y",
    "prefix_hole_velocity_z",
    "prefix_grasped",
    "prefix_inserted",
    "prefix_hole_delta_cumulative_x",
    "prefix_hole_delta_cumulative_y",
    "prefix_hole_delta_cumulative_z",
    "prefix_peg_delta_applied_x",
    "prefix_peg_delta_applied_y",
    "prefix_peg_delta_applied_z",
    "action_time_fraction",
    "prefix_perturb_triggered",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--expected-action-dim", type=int, default=32)
    parser.add_argument("--robot-action-dim", type=int, default=7)
    parser.add_argument("--nonzero-threshold", type=float, default=1e-6)
    parser.add_argument("--variation-threshold", type=float, default=1e-6)
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


def scenario_from_row(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get("scenario"):
        return str(metadata["scenario"])
    uuid = str(row.get("uuid", "unknown"))
    return uuid.split("_seed")[0] if "_seed" in uuid else "unknown"


def load_action(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        for key in ("actions", "action", "action_targets"):
            if key in payload:
                payload = payload[key]
                break
    return np.asarray(payload, dtype=np.float32)


def vector_names_from_row(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        structured = metadata.get("structured_action_state_condition")
        if isinstance(structured, dict) and isinstance(structured.get("vector_names"), list):
            return [str(name) for name in structured["vector_names"]]
    return list(VECTOR_NAMES)


def sidecar_target_mode_from_row(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        structured = metadata.get("structured_action_state_condition")
        if isinstance(structured, dict) and structured.get("sidecar_target_mode"):
            return str(structured["sidecar_target_mode"])
    return "unknown"


def audit_split(args: argparse.Namespace, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    jsonl = args.condition_root / split / "video_action_dataset_file.jsonl"
    rows = load_jsonl(jsonl)
    bad_rows: list[dict[str, Any]] = []
    row_reports: list[dict[str, Any]] = []
    scenarios: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    robot_abs_means: list[float] = []
    robot_abs_maxes: list[float] = []
    robot_std_means: list[float] = []
    full_abs_maxes: list[float] = []
    time_stds: list[float] = []
    task_sidecar_stds: list[float] = []
    sidecar_modes: Counter[str] = Counter()

    for idx, row in enumerate(rows):
        uuid = row.get("uuid")
        scenario = scenario_from_row(row)
        scenarios[scenario] += 1
        action_path = Path(str(row.get("action_path", "")))
        if not action_path.exists():
            bad_rows.append({"idx": idx, "uuid": uuid, "reason": "missing_action_path", "path": str(action_path)})
            continue
        try:
            action = load_action(action_path)
        except Exception as exc:  # noqa: BLE001 - record malformed row and continue
            bad_rows.append({"idx": idx, "uuid": uuid, "reason": "action_parse_error", "error": str(exc)})
            continue

        shapes["x".join(str(x) for x in action.shape)] += 1
        if action.shape != (args.expected_action_steps, args.expected_action_dim):
            bad_rows.append(
                {
                    "idx": idx,
                    "uuid": uuid,
                    "reason": "action_shape_mismatch",
                    "shape": list(action.shape),
                }
            )
            continue
        if not np.isfinite(action).all():
            bad_rows.append({"idx": idx, "uuid": uuid, "reason": "nonfinite_action"})
            continue

        vector_names = vector_names_from_row(row)
        sidecar_mode = sidecar_target_mode_from_row(row)
        sidecar_modes[sidecar_mode] += 1
        if len(vector_names) != args.expected_action_dim:
            bad_rows.append(
                {
                    "idx": idx,
                    "uuid": uuid,
                    "reason": "vector_name_count_mismatch",
                    "num_vector_names": len(vector_names),
                }
            )
            continue
        expected_prefix = [f"action_{i}" for i in range(args.robot_action_dim)]
        if vector_names[: args.robot_action_dim] != expected_prefix:
            bad_rows.append(
                {
                    "idx": idx,
                    "uuid": uuid,
                    "reason": "robot_action_prefix_names_mismatch",
                    "prefix": vector_names[: args.robot_action_dim],
                }
            )
            continue

        robot = action[:, : args.robot_action_dim]
        robot_abs_mean = float(np.mean(np.abs(robot)))
        robot_abs_max = float(np.max(np.abs(robot)))
        robot_std_mean = float(np.mean(np.std(robot, axis=0)))
        full_abs_max = float(np.max(np.abs(action)))
        time_fraction = action[:, vector_names.index("action_time_fraction")]
        time_std = float(np.std(time_fraction))
        sidecar_variable_indices = [
            i
            for i, name in enumerate(vector_names[args.robot_action_dim :], start=args.robot_action_dim)
            if name not in {"action_time_fraction", "prefix_perturb_triggered", "task_perturb_triggered"}
        ]
        task_sidecar_std = (
            float(np.mean(np.std(action[:, sidecar_variable_indices], axis=0))) if sidecar_variable_indices else 0.0
        )

        robot_abs_means.append(robot_abs_mean)
        robot_abs_maxes.append(robot_abs_max)
        robot_std_means.append(robot_std_mean)
        full_abs_maxes.append(full_abs_max)
        time_stds.append(time_std)
        task_sidecar_stds.append(task_sidecar_std)
        row_reports.append(
            {
                "idx": idx,
                "uuid": uuid,
                "scenario": scenario,
                "shape": list(action.shape),
                "robot_action_abs_mean": robot_abs_mean,
                "robot_action_abs_max": robot_abs_max,
                "robot_action_std_mean": robot_std_mean,
                "full_abs_max": full_abs_max,
                "action_time_fraction_std": time_std,
                "sidecar_target_mode": sidecar_mode,
                "task_state_sidecar_std_mean": task_sidecar_std,
                "robot_action_nonzero": robot_abs_max > args.nonzero_threshold,
                "robot_action_varies_over_time": robot_std_mean > args.variation_threshold,
                "time_fraction_varies": time_std > args.variation_threshold,
                "task_state_sidecar_varies_over_time": task_sidecar_std > args.variation_threshold,
            }
        )

    rows_with_nonzero_robot = sum(1 for row in row_reports if row["robot_action_nonzero"])
    rows_with_robot_variation = sum(1 for row in row_reports if row["robot_action_varies_over_time"])
    rows_with_time_variation = sum(1 for row in row_reports if row["time_fraction_varies"])
    rows_with_task_sidecar_variation = sum(1 for row in row_reports if row["task_state_sidecar_varies_over_time"])
    future_sidecar_rows = sum(1 for row in row_reports if row["sidecar_target_mode"] == "future_aligned_state")
    future_sidecar_variation_rows = sum(
        1
        for row in row_reports
        if row["sidecar_target_mode"] == "future_aligned_state" and row["task_state_sidecar_varies_over_time"]
    )
    summary = {
        "jsonl": str(jsonl),
        "num_rows": len(rows),
        "num_valid_rows": len(row_reports),
        "num_bad_rows": len(bad_rows),
        "bad_rows": bad_rows[:20],
        "scenarios": dict(scenarios),
        "sidecar_target_modes": dict(sidecar_modes),
        "action_shapes": dict(shapes),
        "rows_with_nonzero_robot_action": rows_with_nonzero_robot,
        "rows_with_robot_action_variation": rows_with_robot_variation,
        "rows_with_time_fraction_variation": rows_with_time_variation,
        "rows_with_task_state_sidecar_variation": rows_with_task_sidecar_variation,
        "future_aligned_state_rows": future_sidecar_rows,
        "future_aligned_state_rows_with_task_sidecar_variation": future_sidecar_variation_rows,
        "robot_action_abs_mean_mean": float(np.mean(robot_abs_means)) if robot_abs_means else None,
        "robot_action_abs_max_max": float(np.max(robot_abs_maxes)) if robot_abs_maxes else None,
        "robot_action_std_mean_mean": float(np.mean(robot_std_means)) if robot_std_means else None,
        "full_abs_max_max": float(np.max(full_abs_maxes)) if full_abs_maxes else None,
        "time_fraction_std_mean": float(np.mean(time_stds)) if time_stds else None,
        "task_state_sidecar_std_mean": float(np.mean(task_sidecar_stds)) if task_sidecar_stds else None,
        "sample_rows": row_reports[:10],
    }
    return summary, row_reports


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "condition_root": str(args.condition_root),
        "expected": {
            "action_steps": args.expected_action_steps,
            "action_dim": args.expected_action_dim,
            "robot_action_dim": args.robot_action_dim,
        },
        "physical_reason": (
            "verify full301 WAM has nondegenerate 300-step structured robot/action "
            "targets before post-SFT action prediction metrics"
        ),
        "boundary": "action-target label audit only; not model evidence and not controller evidence",
        "splits": {},
        "failures": [],
        "strict_action_target_ok": True,
    }
    for split in ["train", "val"]:
        split_summary, row_reports = audit_split(args, split)
        report["splits"][split] = split_summary
        (args.output_dir / f"{split}_rows.json").write_text(json.dumps(row_reports, indent=2, sort_keys=True) + "\n")
        if split_summary["num_bad_rows"]:
            report["strict_action_target_ok"] = False
            report["failures"].append(f"{split}_bad_rows:{split_summary['num_bad_rows']}")
        if split_summary["rows_with_nonzero_robot_action"] != split_summary["num_valid_rows"]:
            report["strict_action_target_ok"] = False
            report["failures"].append(
                f"{split}_zero_robot_action_rows:"
                f"{split_summary['num_valid_rows'] - split_summary['rows_with_nonzero_robot_action']}"
            )
        if split_summary["rows_with_robot_action_variation"] != split_summary["num_valid_rows"]:
            report["strict_action_target_ok"] = False
            report["failures"].append(
                f"{split}_nonvarying_robot_action_rows:"
                f"{split_summary['num_valid_rows'] - split_summary['rows_with_robot_action_variation']}"
            )
        if split_summary["rows_with_time_fraction_variation"] != split_summary["num_valid_rows"]:
            report["strict_action_target_ok"] = False
            report["failures"].append(
                f"{split}_nonvarying_time_fraction_rows:"
                f"{split_summary['num_valid_rows'] - split_summary['rows_with_time_fraction_variation']}"
            )
        if split_summary["future_aligned_state_rows"]:
            missing = (
                split_summary["future_aligned_state_rows"]
                - split_summary["future_aligned_state_rows_with_task_sidecar_variation"]
            )
            if missing:
                report["strict_action_target_ok"] = False
                report["failures"].append(f"{split}_nonvarying_future_task_sidecar_rows:{missing}")

    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and not report["strict_action_target_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
