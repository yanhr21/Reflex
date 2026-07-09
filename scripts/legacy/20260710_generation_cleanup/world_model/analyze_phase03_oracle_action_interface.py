#!/usr/bin/env python3
"""Read-only Phase 03 Cosmos/action-interface diagnostic.

This script compares Cosmos actions executed during a Phase 03 Oracle run
against the matching approved source-H5 teacher action rows. It is diagnostic
only: teacher actions are not executed and the output must not be reported as
method or Oracle success evidence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--offset-window",
        type=int,
        default=16,
        help="Read-only teacher-action temporal offset sweep window in env steps.",
    )
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


def load_teacher_actions(source_h5: Path) -> np.ndarray:
    with h5py.File(source_h5, "r") as h5:
        traj_names = sorted(name for name in h5.keys() if name.startswith("traj_"))
        if len(traj_names) != 1:
            raise RuntimeError(f"{source_h5} expected one traj group, found {traj_names}")
        actions = np.asarray(h5[traj_names[0]]["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise RuntimeError(f"{source_h5} invalid action shape {actions.shape}")
    return actions[:, :7]


def vector_stats(arr: np.ndarray) -> dict[str, Any]:
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.shape[0]),
        "mean": np.mean(arr, axis=0),
        "mean_abs": np.mean(np.abs(arr), axis=0),
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
    }


def fit_axis_gain(pred_mat: np.ndarray, teacher_mat: np.ndarray, nonnegative: bool) -> np.ndarray:
    denom = np.sum(pred_mat * pred_mat, axis=0)
    numer = np.sum(pred_mat * teacher_mat, axis=0)
    gain = np.ones((pred_mat.shape[1],), dtype=np.float32)
    valid = denom > 1.0e-12
    gain[valid] = (numer[valid] / denom[valid]).astype(np.float32)
    if nonnegative:
        gain = np.maximum(gain, 0.0)
    return gain


def candidate_metrics(name: str, pred_mat: np.ndarray, teacher_mat: np.ndarray, note: str = "") -> dict[str, Any]:
    diff = pred_mat - teacher_mat
    sign_xyz = np.sign(pred_mat[:, :3]) * np.sign(teacher_mat[:, :3])
    sign_rot = np.sign(pred_mat[:, 3:6]) * np.sign(teacher_mat[:, 3:6])
    return {
        "name": name,
        "note": note,
        "count": int(pred_mat.shape[0]),
        "rmse_7d": float(np.sqrt(np.mean(diff**2))),
        "rmse_xyz": float(np.sqrt(np.mean(diff[:, :3] ** 2))),
        "rmse_rot": float(np.sqrt(np.mean(diff[:, 3:6] ** 2))),
        "mean_xyz_sign_agreement": np.mean(sign_xyz, axis=0),
        "mean_rot_sign_agreement": np.mean(sign_rot, axis=0),
        "mean_gripper_sign_agreement": float(np.mean(np.sign(pred_mat[:, 6]) * np.sign(teacher_mat[:, 6]))),
        "pred_stats": vector_stats(pred_mat),
        "diff_stats": vector_stats(diff),
    }


def apply_motion_sign_guard(
    pred_mat: np.ndarray,
    target_delta_mat: np.ndarray,
    mode: str,
) -> np.ndarray:
    guarded = pred_mat.copy()
    signs = np.sign(target_delta_mat[:, :3]).astype(np.float32)
    for row_idx in range(guarded.shape[0]):
        for axis in range(3):
            sign = float(signs[row_idx, axis])
            if sign == 0.0:
                continue
            if float(guarded[row_idx, axis]) * sign < 0.0:
                if mode == "clip_opposite":
                    guarded[row_idx, axis] = 0.0
                elif mode == "rectify_opposite":
                    guarded[row_idx, axis] = abs(float(guarded[row_idx, axis])) * sign
                else:
                    raise RuntimeError(f"unsupported_motion_guard_mode={mode}")
    return guarded


def summarize_adapter_candidates(
    raw_mat: np.ndarray,
    executed_mat: np.ndarray,
    teacher_mat: np.ndarray,
    target_delta_mat: np.ndarray,
) -> dict[str, Any]:
    """Future-label diagnostic only; never use this as method evidence."""
    candidates: list[dict[str, Any]] = [
        candidate_metrics(
            "executed_trace",
            executed_mat,
            teacher_mat,
            "Actions actually executed by the archived run.",
        ),
        candidate_metrics(
            "raw_cosmos",
            raw_mat,
            teacher_mat,
            "Raw denormalized Cosmos actions before any local adapter/guard.",
        ),
    ]

    for mode in ("clip_opposite", "rectify_opposite"):
        candidates.append(
            candidate_metrics(
                f"target_delta_sign_{mode}",
                apply_motion_sign_guard(raw_mat, target_delta_mat, mode),
                teacher_mat,
                (
                    "Offline simulation of the current source-motion-sign guard "
                    "using each row's logged target-motion delta sign."
                ),
            )
        )

    for nonnegative in (True, False):
        gain = fit_axis_gain(raw_mat, teacher_mat, nonnegative=nonnegative)
        scaled = raw_mat * gain.reshape(1, -1)
        candidates.append(
            {
                **candidate_metrics(
                    "axis_gain_nonnegative_fit_7d" if nonnegative else "axis_gain_signed_fit_7d",
                    scaled,
                    teacher_mat,
                    (
                        "Future-label least-squares per-axis gain fitted on this same "
                        "trace; diagnostic headroom only, not a deployable adapter."
                    ),
                ),
                "axis_gain": gain,
            }
        )

    xyz_gain = fit_axis_gain(raw_mat[:, :3], teacher_mat[:, :3], nonnegative=True)
    xyz_scaled = raw_mat.copy()
    xyz_scaled[:, :3] *= xyz_gain.reshape(1, -1)
    candidates.append(
        {
            **candidate_metrics(
                "xyz_gain_nonnegative_fit",
                xyz_scaled,
                teacher_mat,
                "Future-label xyz-only least-squares gain; rotational/gripper axes unchanged.",
            ),
            "xyz_gain": xyz_gain,
        }
    )

    ranked = sorted(candidates, key=lambda item: float(item["rmse_xyz"]))
    return {
        "boundary": (
            "Offline future-label adapter comparison. It can diagnose whether "
            "a simple local adapter has headroom, but it is not method evidence "
            "and must not be used to claim Oracle success."
        ),
        "rank_by_rmse_xyz": ranked,
    }


def summarize_temporal_offsets(
    rows: list[dict[str, Any]],
    teacher_actions: np.ndarray,
    max_offset: int,
) -> dict[str, Any]:
    step_pred_pairs: list[tuple[int, np.ndarray, np.ndarray]] = []
    for row in rows:
        step = int(row.get("env_step", -1))
        pred = np.asarray(row.get("action") or row.get("raw_cosmos_action"), dtype=np.float32).reshape(-1)[:7]
        raw = np.asarray(row.get("raw_cosmos_action") or pred, dtype=np.float32).reshape(-1)[:7]
        if step >= 0 and pred.size >= 7 and raw.size >= 7:
            step_pred_pairs.append((step, pred, raw))
    offset_rows: list[dict[str, Any]] = []
    for offset in range(-int(max_offset), int(max_offset) + 1):
        pred_rows: list[np.ndarray] = []
        raw_rows: list[np.ndarray] = []
        teacher_rows: list[np.ndarray] = []
        used_steps: list[int] = []
        for step, pred, raw in step_pred_pairs:
            teacher_step = step + offset
            if teacher_step < 0 or teacher_step >= teacher_actions.shape[0]:
                continue
            pred_rows.append(pred)
            raw_rows.append(raw)
            teacher_rows.append(np.asarray(teacher_actions[teacher_step], dtype=np.float32).reshape(-1)[:7])
            used_steps.append(teacher_step)
        if not pred_rows:
            offset_rows.append({"offset": int(offset), "count": 0})
            continue
        pred_mat = np.stack(pred_rows, axis=0)
        raw_mat = np.stack(raw_rows, axis=0)
        teacher_mat = np.stack(teacher_rows, axis=0)
        pred_diff = pred_mat - teacher_mat
        raw_diff = raw_mat - teacher_mat
        offset_rows.append(
            {
                "offset": int(offset),
                "count": int(pred_mat.shape[0]),
                "teacher_step_min": int(min(used_steps)),
                "teacher_step_max": int(max(used_steps)),
                "executed_rmse_xyz": float(np.sqrt(np.mean(pred_diff[:, :3] ** 2))),
                "executed_rmse_7d": float(np.sqrt(np.mean(pred_diff**2))),
                "raw_rmse_xyz": float(np.sqrt(np.mean(raw_diff[:, :3] ** 2))),
                "raw_rmse_7d": float(np.sqrt(np.mean(raw_diff**2))),
                "executed_mean_xyz_sign_agreement": np.mean(
                    np.sign(pred_mat[:, :3]) * np.sign(teacher_mat[:, :3]), axis=0
                ),
                "raw_mean_xyz_sign_agreement": np.mean(
                    np.sign(raw_mat[:, :3]) * np.sign(teacher_mat[:, :3]), axis=0
                ),
            }
        )
    valid = [row for row in offset_rows if int(row.get("count", 0)) > 0]
    return {
        "boundary": (
            "Read-only future-label temporal alignment diagnostic. It compares "
            "executed/raw Cosmos actions to source-H5 teacher actions at "
            "neighboring time offsets. It is not method evidence and cannot "
            "count as Oracle success."
        ),
        "max_offset": int(max_offset),
        "best_by_executed_rmse_xyz": min(valid, key=lambda row: float(row["executed_rmse_xyz"]))
        if valid
        else None,
        "best_by_raw_rmse_xyz": min(valid, key=lambda row: float(row["raw_rmse_xyz"])) if valid else None,
        "offset_rows": offset_rows,
    }


def summarize_rows(rows: list[dict[str, Any]], teacher_actions: np.ndarray) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    pred_rows: list[np.ndarray] = []
    raw_rows: list[np.ndarray] = []
    teacher_rows: list[np.ndarray] = []
    target_delta_rows: list[np.ndarray] = []
    l2_values: list[float] = []
    for row in rows:
        step = int(row.get("env_step", -1))
        if step < 0 or step >= teacher_actions.shape[0]:
            records.append({"env_step": step, "error": "teacher_step_out_of_range"})
            continue
        pred = np.asarray(row.get("action") or row.get("raw_cosmos_action"), dtype=np.float32).reshape(-1)[:7]
        raw = np.asarray(row.get("raw_cosmos_action") or pred, dtype=np.float32).reshape(-1)[:7]
        teacher = np.asarray(teacher_actions[step], dtype=np.float32).reshape(-1)[:7]
        target_delta = np.asarray(row.get("target_motion_delta_xyz") or [0.0, 0.0, 0.0], dtype=np.float32)
        if target_delta.size < 3:
            target_delta = np.zeros((3,), dtype=np.float32)
        state = row.get("live_eval") or {}
        l2 = state.get("peg_head_l2")
        if l2 is not None:
            l2_values.append(float(l2))
        pred_rows.append(pred)
        raw_rows.append(raw)
        teacher_rows.append(teacher)
        target_delta_rows.append(target_delta[:3])
        records.append(
            {
                "env_step": step,
                "cosmos_round": row.get("cosmos_round"),
                "cosmos_action_index": row.get("cosmos_action_index"),
                "peg_head_l2": l2,
                "target_motion_delta_xyz": row.get("target_motion_delta_xyz"),
                "pred_action": pred,
                "raw_cosmos_action": raw,
                "teacher_action": teacher,
                "diff": pred - teacher,
                "xyz_sign_agreement": np.sign(pred[:3]) * np.sign(teacher[:3]),
                "rot_sign_agreement": np.sign(pred[3:6]) * np.sign(teacher[3:6]),
                "gripper_sign_agreement": float(np.sign(pred[6]) * np.sign(teacher[6])),
            }
        )
    if not pred_rows:
        return {"records": records, "valid_count": 0}

    pred_mat = np.stack(pred_rows, axis=0)
    raw_mat = np.stack(raw_rows, axis=0)
    teacher_mat = np.stack(teacher_rows, axis=0)
    target_delta_mat = np.stack(target_delta_rows, axis=0)
    diff = pred_mat - teacher_mat
    sign_xyz = np.sign(pred_mat[:, :3]) * np.sign(teacher_mat[:, :3])
    sign_rot = np.sign(pred_mat[:, 3:6]) * np.sign(teacher_mat[:, 3:6])
    l2_delta = None
    if l2_values:
        l2_delta = float(l2_values[-1] - l2_values[0])
    return {
        "records": records,
        "valid_count": int(pred_mat.shape[0]),
        "rmse_7d": float(np.sqrt(np.mean(diff**2))),
        "rmse_xyz": float(np.sqrt(np.mean(diff[:, :3] ** 2))),
        "rmse_rot": float(np.sqrt(np.mean(diff[:, 3:6] ** 2))),
        "pred_stats": vector_stats(pred_mat),
        "teacher_stats": vector_stats(teacher_mat),
        "diff_stats": vector_stats(diff),
        "mean_xyz_sign_agreement": np.mean(sign_xyz, axis=0),
        "mean_rot_sign_agreement": np.mean(sign_rot, axis=0),
        "mean_gripper_sign_agreement": float(np.mean(np.sign(pred_mat[:, 6]) * np.sign(teacher_mat[:, 6]))),
        "l2_first": l2_values[0] if l2_values else None,
        "l2_last": l2_values[-1] if l2_values else None,
        "l2_delta": l2_delta,
        "adapter_candidate_diagnostics": summarize_adapter_candidates(
            raw_mat=raw_mat,
            executed_mat=pred_mat,
            teacher_mat=teacher_mat,
            target_delta_mat=target_delta_mat,
        ),
        "worst_rows_by_abs_xyz_error": sorted(
            records,
            key=lambda rec: float(np.linalg.norm(np.asarray(rec.get("diff", [0, 0, 0])[:3], dtype=np.float32)))
            if "diff" in rec
            else -1.0,
            reverse=True,
        )[:8],
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    summary = read_json(run_dir / "summary.json")
    trace = read_json(run_dir / "action_trace.json")
    source_h5 = Path(summary.get("source_h5_path") or (summary.get("source_protocol") or {}).get("source_h5_path"))
    if not source_h5.is_file():
        raise FileNotFoundError(source_h5)
    teacher_actions = load_teacher_actions(source_h5)
    dynamic_rows = [row for row in trace if row.get("stage") == "cosmos_dynamic_control"]
    finisher_rows = [row for row in trace if "finisher" in str(row.get("stage", ""))]
    output = {
        "boundary": (
            "Read-only diagnostic comparing executed Cosmos dynamic actions "
            "against matching source-H5 teacher rows. Teacher actions are not "
            "executed; this cannot count as Oracle success."
        ),
        "run_dir": str(run_dir),
        "source_h5": str(source_h5.resolve()),
        "classification": summary.get("classification"),
        "simulator_success_metric": summary.get("simulator_success_metric"),
        "target_motion_trigger_frame": summary.get("target_motion_trigger_frame"),
        "target_motion_complete_before_finisher": summary.get("target_motion_complete_before_finisher"),
        "cosmos_dynamic_action_count": summary.get("cosmos_dynamic_action_count"),
        "finisher_row_count": len(finisher_rows),
        "teacher_action_shape": list(teacher_actions.shape),
        "teacher_temporal_offset_sweep": summarize_temporal_offsets(
            dynamic_rows,
            teacher_actions,
            max_offset=max(0, int(args.offset_window)),
        ),
        "cosmos_dynamic_vs_teacher": summarize_rows(dynamic_rows, teacher_actions),
    }
    write_json(Path(args.output_json).resolve(), output)
    print(
        json.dumps(
            {
                "ok": True,
                "valid_count": output["cosmos_dynamic_vs_teacher"].get("valid_count"),
                "rmse_7d": output["cosmos_dynamic_vs_teacher"].get("rmse_7d"),
                "l2_delta": output["cosmos_dynamic_vs_teacher"].get("l2_delta"),
                "output_json": str(Path(args.output_json).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
