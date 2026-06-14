#!/usr/bin/env python3
"""Read-only action/rebind diagnostics for Cosmos3 hard-case panels.

The script compares executed Cosmos robot-action chunks against the matching
source-teacher action rows from the same H5 trajectory. This is diagnostic
only: it must not be used to execute teacher actions or to weaken the live
closed-loop success gate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosmos-panel-root", required=True)
    parser.add_argument("--pure-dp-panel-root", default="")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def sample_index_from_name(path: Path) -> int | None:
    match = re.match(r"sample_(\d+)_", path.name)
    if not match:
        return None
    return int(match.group(1))


def load_teacher_robot_actions(source_h5: Path) -> np.ndarray:
    with h5py.File(source_h5, "r") as h5:
        traj_names = sorted([name for name in h5.keys() if name.startswith("traj_")])
        if len(traj_names) != 1:
            raise RuntimeError(f"{source_h5} expected one traj group, found {traj_names}")
        actions = np.asarray(h5[traj_names[0]]["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise RuntimeError(f"{source_h5} action shape is invalid: {actions.shape}")
    return actions[:, :7].astype(np.float32)


def gate_fail_counts(summary: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for iteration in summary.get("iterations") or []:
        for key in (
            "pre_controller_continuability_gate",
            "post_cosmos_continuability_gate",
            "after_dp_handoff_continuability_gate",
        ):
            gate = iteration.get(key)
            if not isinstance(gate, dict) or gate.get("ok"):
                continue
            checks = gate.get("checks") or {}
            for check_name, ok in checks.items():
                if not ok:
                    counts[str(check_name)] = counts.get(str(check_name), 0) + 1
    return counts


def summarize_chunks(summary: dict[str, Any], sample_dir: Path, teacher_actions: np.ndarray) -> dict[str, Any]:
    chunk_records: list[dict[str, Any]] = []
    for iteration in summary.get("iterations") or []:
        if iteration.get("controller_step_type") != "cosmos_rebind_short_chunk":
            continue
        chunk_path_text = iteration.get("action_chunk_json")
        if not chunk_path_text:
            continue
        chunk_path = Path(chunk_path_text)
        if not chunk_path.is_file():
            chunk_path = sample_dir / chunk_path_text
        if not chunk_path.is_file():
            chunk_records.append(
                {
                    "iteration": iteration.get("iteration"),
                    "prefix_frame_index": iteration.get("prefix_frame_index"),
                    "missing_action_chunk_json": str(chunk_path_text),
                }
            )
            continue
        chunk = read_json(chunk_path)
        pred = np.asarray(chunk.get("denormalized_robot_action_chunk") or [], dtype=np.float32)
        if pred.ndim != 2 or pred.shape[1] < 7 or pred.shape[0] == 0:
            chunk_records.append(
                {
                    "iteration": iteration.get("iteration"),
                    "prefix_frame_index": iteration.get("prefix_frame_index"),
                    "invalid_predicted_action_shape": list(pred.shape),
                }
            )
            continue
        pred = pred[:, :7]
        start = int(chunk.get("chunk_start", iteration.get("prefix_frame_index", 0)))
        teacher = teacher_actions[start : start + pred.shape[0]]
        if teacher.shape != pred.shape:
            chunk_records.append(
                {
                    "iteration": iteration.get("iteration"),
                    "prefix_frame_index": iteration.get("prefix_frame_index"),
                    "chunk_start": start,
                    "invalid_teacher_action_shape": list(teacher.shape),
                    "predicted_action_shape": list(pred.shape),
                }
            )
            continue
        pred_xyz_mean = np.mean(pred[:, :3], axis=0)
        teacher_xyz_mean = np.mean(teacher[:, :3], axis=0)
        pred_xyz_abs = np.mean(np.abs(pred[:, :3]), axis=0)
        teacher_xyz_abs = np.mean(np.abs(teacher[:, :3]), axis=0)
        chunk_records.append(
            {
                "iteration": iteration.get("iteration"),
                "prefix_frame_index": iteration.get("prefix_frame_index"),
                "prefix_role": iteration.get("prefix_role"),
                "chunk_start": start,
                "chunk_steps": int(pred.shape[0]),
                "rmse_robot_action_7d": float(np.sqrt(np.mean((pred - teacher) ** 2))),
                "pred_mean_abs_xyz": pred_xyz_abs,
                "teacher_mean_abs_xyz": teacher_xyz_abs,
                "pred_over_teacher_abs_xyz": pred_xyz_abs / np.maximum(teacher_xyz_abs, 1e-6),
                "pred_mean_xyz": pred_xyz_mean,
                "teacher_mean_xyz": teacher_xyz_mean,
                "mean_sign_agreement_xyz": np.sign(pred_xyz_mean) * np.sign(teacher_xyz_mean),
                "after_eval": iteration.get("after_eval"),
                "post_cosmos_continuability_gate": iteration.get("post_cosmos_continuability_gate"),
            }
        )
    valid = [row for row in chunk_records if "rmse_robot_action_7d" in row]
    if not valid:
        return {"records": chunk_records, "valid_chunk_count": 0}
    rmse = np.asarray([row["rmse_robot_action_7d"] for row in valid], dtype=np.float32)
    ratio = np.stack([np.asarray(row["pred_over_teacher_abs_xyz"], dtype=np.float32) for row in valid], axis=0)
    sign = np.stack([np.asarray(row["mean_sign_agreement_xyz"], dtype=np.float32) for row in valid], axis=0)
    return {
        "records": chunk_records,
        "valid_chunk_count": len(valid),
        "mean_rmse_robot_action_7d": float(np.mean(rmse)),
        "median_rmse_robot_action_7d": float(np.median(rmse)),
        "max_rmse_robot_action_7d": float(np.max(rmse)),
        "mean_pred_over_teacher_abs_xyz": np.mean(ratio, axis=0),
        "mean_sign_agreement_xyz": np.mean(sign, axis=0),
        "worst_chunks_by_rmse": sorted(valid, key=lambda row: row["rmse_robot_action_7d"], reverse=True)[:3],
    }


def load_pure_dp_by_index(root: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if not root:
        return out
    for summary_path in sorted(root.glob("sample_*/pure_dp_full_episode_summary.json")):
        sample_dir = summary_path.parent
        summary = read_json(summary_path)
        idx = summary.get("sample_index")
        if idx is None:
            idx = sample_index_from_name(sample_dir)
        if idx is None:
            continue
        out[int(idx)] = {
            "sample_dir": str(sample_dir),
            "scenario": summary.get("scenario"),
            "final_success": bool((summary.get("final_eval") or {}).get("success", False)),
            "final_peg_head_at_hole": (summary.get("final_eval") or {}).get("peg_head_pos_at_hole"),
            "target_motion_detector": summary.get("target_motion_detector"),
            "controller_frame_counts": summary.get("controller_frame_counts"),
        }
    return out


def main() -> None:
    args = parse_args()
    cosmos_root = Path(args.cosmos_panel_root).resolve()
    pure_dp = load_pure_dp_by_index(Path(args.pure_dp_panel_root).resolve()) if args.pure_dp_panel_root else {}
    samples: list[dict[str, Any]] = []
    for summary_path in sorted(cosmos_root.glob("sample_*/live_receding_loop_summary.json")):
        sample_dir = summary_path.parent
        summary = read_json(summary_path)
        idx = summary.get("sample_index")
        if idx is None:
            idx = sample_index_from_name(sample_dir)
        teacher_actions = load_teacher_robot_actions(Path(summary["source_h5"]))
        chunk_summary = summarize_chunks(summary, sample_dir, teacher_actions)
        samples.append(
            {
                "sample_index": idx,
                "sample_dir": str(sample_dir),
                "scenario": summary.get("scenario"),
                "source_h5": summary.get("source_h5"),
                "final_success": bool((summary.get("final_eval") or {}).get("success", False)),
                "final_peg_head_at_hole": (summary.get("final_eval") or {}).get("peg_head_pos_at_hole"),
                "full_episode_length_ok": summary.get("full_episode_length_ok"),
                "prefix_selection": summary.get("prefix_selection"),
                "controller_frame_counts": summary.get("controller_frame_counts"),
                "continuability_gate_fail_counts": gate_fail_counts(summary),
                "cosmos_action_vs_teacher": chunk_summary,
                "pure_dp_baseline": pure_dp.get(int(idx)) if idx is not None else None,
            }
        )
    output = {
        "boundary": (
            "Read-only diagnostic. Cosmos chunks are compared against matching "
            "source-teacher action rows to localize action/rebind errors. "
            "Teacher actions are not executed and do not affect controller "
            "success, which remains the real live final state plus video review."
        ),
        "cosmos_panel_root": str(cosmos_root),
        "pure_dp_panel_root": str(Path(args.pure_dp_panel_root).resolve()) if args.pure_dp_panel_root else None,
        "sample_count": len(samples),
        "cosmos_final_success_count": sum(1 for sample in samples if sample["final_success"]),
        "pure_dp_final_success_count_on_matched": sum(
            1 for sample in samples if (sample.get("pure_dp_baseline") or {}).get("final_success")
        ),
        "samples": samples,
    }
    write_json(Path(args.output_json).resolve(), output)
    print(json.dumps(to_jsonable({k: output[k] for k in ("sample_count", "cosmos_final_success_count", "pure_dp_final_success_count_on_matched")}), sort_keys=True))


if __name__ == "__main__":
    main()
