#!/usr/bin/env python3
"""Audit whether selected live chunks can be calibrated from predicted features.

This is a diagnostic bridge between the live receding panel and the next
selector/executor training step. It uses only causal/predicted fields as model
inputs and uses the real after-state as a supervised target. The output is not
method success evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-outcome-jsonl", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--min-rows", type=int, default=8)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.95)
    parser.add_argument("--require-compute-step", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def require_compute_step() -> None:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    step_id = os.environ.get("SLURM_STEP_ID", "")
    if not job_id or not step_id or step_id == "extern":
        raise SystemExit(
            "refusing_login_node_execution=true\n"
            "reason=Run live outcome calibration audit only inside a compute-node srun step."
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), sort_keys=True))
            f.write("\n")
    os.replace(tmp, path)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def float_or(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def bool_float(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def vec3(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        vals = [float_or(item, 0.0) for item in list(value)[:3]]
    else:
        vals = []
    while len(vals) < 3:
        vals.append(0.0)
    return np.asarray(vals, dtype=np.float32)


def candidate_features(name: str) -> tuple[list[str], np.ndarray]:
    text = str(name or "")
    short_match = re.match(r"short(?P<steps>\d+)_(?P<base>.+)", text)
    short_steps = 0.0
    base = text
    if short_match:
        short_steps = float(short_match.group("steps"))
        base = short_match.group("base")
    names = [
        "cand_is_dp_prior",
        "cand_is_mean",
        "cand_is_scale",
        "cand_is_sample",
        "cand_is_diffusion",
        "cand_is_short_prefix",
        "cand_short_steps_norm24",
    ]
    vals = [
        float(base == "dp_prior"),
        float(base == "mean" or base == "model_mean"),
        float("scale" in base),
        float("sample" in base),
        float("diffusion" in base),
        float(short_steps > 0.0),
        float(short_steps / 24.0) if short_steps > 0.0 else 0.0,
    ]
    return names, np.asarray(vals, dtype=np.float32)


def build_feature(row: dict[str, Any]) -> tuple[list[str], np.ndarray] | None:
    before = vec3(row.get("before_peg_head_at_hole"))
    scorer_state = vec3(row.get("scorer_predicted_state"))
    executor_state = vec3(row.get("executor_predicted_next_state"))
    progress = row.get("outcome_scorer_predicted_progress")
    progress_vec = np.asarray(
        [
            float_or(progress[0], 0.0) if isinstance(progress, list) and len(progress) > 0 else 0.0,
            float_or(progress[1], 0.0) if isinstance(progress, list) and len(progress) > 1 else 0.0,
        ],
        dtype=np.float32,
    )
    cand_names, cand_vals = candidate_features(str(row.get("selected_candidate_name") or ""))
    names = [
        "before_x",
        "before_y",
        "before_z",
        "before_abs_l1",
        "before_abs_yz",
        "iteration_norm300",
        "scorer_state_x",
        "scorer_state_y",
        "scorer_state_z",
        "executor_state_x",
        "executor_state_y",
        "executor_state_z",
        "scorer_score",
        "scorer_handoff_prob",
        "scorer_continuable_prob",
        "scorer_inserted_prob",
        "scorer_progress",
        "scorer_progress_delta",
    ] + cand_names
    vals = [
        float(before[0]),
        float(before[1]),
        float(before[2]),
        float_or(row.get("before_abs_l1"), float(np.sum(np.abs(before)))),
        float_or(row.get("before_abs_yz"), float(abs(before[1]) + abs(before[2]))),
        float_or(row.get("iteration"), 0.0) / 300.0,
        float(scorer_state[0]),
        float(scorer_state[1]),
        float(scorer_state[2]),
        float(executor_state[0]),
        float(executor_state[1]),
        float(executor_state[2]),
        float_or(row.get("outcome_scorer_score"), 0.0),
        float_or(row.get("outcome_scorer_predicted_handoff_success_probability"), 0.0),
        float_or(row.get("outcome_scorer_predicted_continuable_probability"), 0.0),
        float_or(row.get("outcome_scorer_predicted_inserted_probability"), 0.0),
        float(progress_vec[0]),
        float(progress_vec[1]),
    ]
    vals.extend(float(v) for v in cand_vals)
    return names, np.asarray(vals, dtype=np.float32)


def target_vec(row: dict[str, Any]) -> np.ndarray | None:
    after = row.get("after_peg_head_at_hole")
    if not isinstance(after, list) or len(after) < 3:
        return None
    after_v = vec3(after)
    before_abs_yz = float_or(row.get("before_abs_yz"), 0.0)
    after_abs_yz = float_or(row.get("after_abs_yz"), float(abs(after_v[1]) + abs(after_v[2])))
    return np.asarray(
        [
            float(after_v[0]),
            float(after_v[1]),
            float(after_v[2]),
            float(after_abs_yz),
            float(after_abs_yz - before_abs_yz),
            float(after_abs_yz > before_abs_yz),
            bool_float(row.get("gate_ok")),
        ],
        dtype=np.float32,
    )


def ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1.0e-6, 1.0, std)
    xtr = (x_train - mean) / std
    xte = (x_test - mean) / std
    xtr_b = np.concatenate([xtr, np.ones((xtr.shape[0], 1), dtype=np.float32)], axis=1)
    xte_b = np.concatenate([xte, np.ones((xte.shape[0], 1), dtype=np.float32)], axis=1)
    reg = np.eye(xtr_b.shape[1], dtype=np.float32) * float(alpha)
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(xtr_b.T @ xtr_b + reg, xtr_b.T @ y_train)
    return xte_b @ coef


def regression_stats(pred: np.ndarray, target: np.ndarray, prefix: str) -> dict[str, Any]:
    err = pred - target
    out: dict[str, Any] = {"count": int(target.shape[0])}
    target_names = ["x", "y", "z", "abs_yz", "delta_abs_yz", "worsened_abs_yz", "gate_ok"]
    for idx, name in enumerate(target_names[: target.shape[1]]):
        vals = err[:, idx]
        out[f"{prefix}_{name}_mae"] = float(np.mean(np.abs(vals)))
        out[f"{prefix}_{name}_rmse"] = float(np.sqrt(np.mean(vals * vals)))
        out[f"{prefix}_{name}_bias"] = float(np.mean(vals))
    if target.shape[1] >= 6:
        pred_bin = pred[:, 5] >= 0.5
        true_bin = target[:, 5] >= 0.5
        out[f"{prefix}_worsened_abs_yz_accuracy_at_0p5"] = float(np.mean(pred_bin == true_bin))
    if target.shape[1] >= 7:
        pred_gate = pred[:, 6] >= 0.5
        true_gate = target[:, 6] >= 0.5
        out[f"{prefix}_gate_ok_accuracy_at_0p5"] = float(np.mean(pred_gate == true_gate))
    return out


def baseline_prediction(rows: list[dict[str, Any]], kind: str) -> np.ndarray:
    preds: list[np.ndarray] = []
    for row in rows:
        if kind == "identity":
            state = vec3(row.get("before_peg_head_at_hole"))
        elif kind == "scorer":
            state = vec3(row.get("scorer_predicted_state"))
        elif kind == "executor":
            state = vec3(row.get("executor_predicted_next_state"))
        else:
            raise ValueError(kind)
        abs_yz_val = float(abs(state[1]) + abs(state[2]))
        before_abs_yz = float_or(row.get("before_abs_yz"), 0.0)
        preds.append(
            np.asarray(
                [
                    float(state[0]),
                    float(state[1]),
                    float(state[2]),
                    abs_yz_val,
                    abs_yz_val - before_abs_yz,
                    float(abs_yz_val > before_abs_yz),
                    0.0,
                ],
                dtype=np.float32,
            )
        )
    return np.stack(preds).astype(np.float32)


def summarize_high_risk(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    high_conf = [
        r
        for r in rows
        if float_or(r.get("outcome_scorer_predicted_handoff_success_probability"), 0.0) >= threshold
    ]
    worsened = [
        r
        for r in high_conf
        if float_or(r.get("after_abs_yz"), 0.0) > float_or(r.get("before_abs_yz"), 0.0)
    ]
    gate_fail = [r for r in high_conf if r.get("gate_ok") is False]
    by_name = Counter(str(r.get("selected_candidate_name") or "") for r in high_conf)
    worsened_by_name = Counter(str(r.get("selected_candidate_name") or "") for r in worsened)
    return {
        "high_confidence_threshold": float(threshold),
        "high_confidence_count": int(len(high_conf)),
        "high_confidence_worsened_abs_yz_count": int(len(worsened)),
        "high_confidence_gate_fail_count": int(len(gate_fail)),
        "high_confidence_candidate_counts": dict(sorted(by_name.items())),
        "high_confidence_worsened_candidate_counts": dict(sorted(worsened_by_name.items())),
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Cosmos3 Live Outcome Calibration Audit",
        "",
        "This is a diagnostic audit, not method success evidence.",
        "",
        "## Key Result",
        "",
        f"- rows: `{summary.get('num_rows')}`",
        f"- samples: `{summary.get('num_samples')}`",
        f"- live final success samples: `{summary.get('final_success_sample_count')}`",
        f"- selected chunks that worsened y/z: `{summary.get('worsened_abs_yz_count')}`",
        f"- high-confidence worsened y/z count: `{summary.get('high_risk', {}).get('high_confidence_worsened_abs_yz_count')}`",
        "",
        "## Calibration Metrics",
        "",
    ]
    for key in ("identity", "scorer", "executor", "ridge_leave_one_sample_out"):
        metrics = summary.get("metrics", {}).get(key, {})
        if not metrics:
            continue
        lines.extend(
            [
                f"### {key}",
                "",
                f"- after y MAE: `{metrics.get(f'{key}_y_mae')}`",
                f"- after z MAE: `{metrics.get(f'{key}_z_mae')}`",
                f"- after abs_yz MAE: `{metrics.get(f'{key}_abs_yz_mae')}`",
                f"- delta abs_yz MAE: `{metrics.get(f'{key}_delta_abs_yz_mae')}`",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if bool(args.require_compute_step):
        require_compute_step()

    raw_rows: list[dict[str, Any]] = []
    for item in args.live_outcome_jsonl:
        path = Path(item).resolve()
        raw_rows.extend(read_jsonl(path))

    rows: list[dict[str, Any]] = []
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    feature_names: list[str] | None = None
    for row in raw_rows:
        if row.get("schema") != "cosmos3_live_receding_selected_outcome_v1":
            continue
        if row.get("chunk_type") != "candidate_executor":
            continue
        target = target_vec(row)
        feature_pair = build_feature(row)
        if target is None or feature_pair is None:
            continue
        names, feat = feature_pair
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("nonuniform feature names")
        rows.append(row)
        feature_rows.append(feat)
        target_rows.append(target)

    if len(rows) < int(args.min_rows):
        raise SystemExit(f"not enough candidate rows: {len(rows)} < {args.min_rows}")

    x = np.stack(feature_rows).astype(np.float32)
    y = np.stack(target_rows).astype(np.float32)
    groups = [str(r.get("sample_dir") or r.get("sample_name") or idx) for idx, r in enumerate(rows)]
    group_names = sorted(set(groups))
    ridge_pred = np.zeros_like(y)
    for group in group_names:
        test_idx = np.asarray([idx for idx, value in enumerate(groups) if value == group], dtype=np.int64)
        train_idx = np.asarray([idx for idx, value in enumerate(groups) if value != group], dtype=np.int64)
        if len(train_idx) == 0:
            ridge_pred[test_idx] = y[train_idx] if len(train_idx) else y[test_idx]
            continue
        ridge_pred[test_idx] = ridge_fit_predict(x[train_idx], y[train_idx], x[test_idx], float(args.ridge_alpha))

    metrics = {
        "identity": regression_stats(baseline_prediction(rows, "identity"), y, "identity"),
        "scorer": regression_stats(baseline_prediction(rows, "scorer"), y, "scorer"),
        "executor": regression_stats(baseline_prediction(rows, "executor"), y, "executor"),
        "ridge_leave_one_sample_out": regression_stats(
            ridge_pred, y, "ridge_leave_one_sample_out"
        ),
    }

    high_risk = summarize_high_risk(rows, float(args.high_confidence_threshold))
    worsened_count = sum(
        1 for r in rows if float_or(r.get("after_abs_yz"), 0.0) > float_or(r.get("before_abs_yz"), 0.0)
    )
    sample_success: dict[str, bool] = defaultdict(bool)
    for r in raw_rows:
        sample_success[str(r.get("sample_dir") or r.get("sample_name") or "")] = bool(
            sample_success[str(r.get("sample_dir") or r.get("sample_name") or "")]
            or r.get("final_sample_success")
        )

    scored_rows: list[dict[str, Any]] = []
    for row, pred in zip(rows, ridge_pred):
        before_abs_yz = float_or(row.get("before_abs_yz"), 0.0)
        after_abs_yz = float_or(row.get("after_abs_yz"), 0.0)
        scored_rows.append(
            {
                "sample_name": row.get("sample_name"),
                "scenario": row.get("scenario"),
                "iteration": row.get("iteration"),
                "selected_candidate_name": row.get("selected_candidate_name"),
                "before_abs_yz": before_abs_yz,
                "after_abs_yz": after_abs_yz,
                "actual_delta_abs_yz": after_abs_yz - before_abs_yz,
                "actual_worsened_abs_yz": after_abs_yz > before_abs_yz,
                "scorer_handoff_probability": row.get(
                    "outcome_scorer_predicted_handoff_success_probability"
                ),
                "scorer_continuable_probability": row.get(
                    "outcome_scorer_predicted_continuable_probability"
                ),
                "scorer_score": row.get("outcome_scorer_score"),
                "ridge_pred_after_abs_yz": float(pred[3]),
                "ridge_pred_delta_abs_yz": float(pred[4]),
                "ridge_pred_worsened_abs_yz_score": float(pred[5]),
                "gate_ok": row.get("gate_ok"),
                "final_sample_success": row.get("final_sample_success"),
            }
        )
    scored_rows.sort(
        key=lambda r: (
            -float_or(r.get("scorer_handoff_probability"), 0.0),
            -float_or(r.get("actual_delta_abs_yz"), 0.0),
        )
    )

    summary = {
        "schema": "cosmos3_live_outcome_calibration_audit_v1",
        "input_jsonl": [str(Path(item).resolve()) for item in args.live_outcome_jsonl],
        "num_rows": int(len(rows)),
        "num_samples": int(len(group_names)),
        "feature_names": feature_names,
        "ridge_alpha": float(args.ridge_alpha),
        "candidate_name_counts": dict(sorted(Counter(str(r.get("selected_candidate_name") or "") for r in rows).items())),
        "scenario_counts": dict(sorted(Counter(str(r.get("scenario") or "") for r in rows).items())),
        "worsened_abs_yz_count": int(worsened_count),
        "improved_or_equal_abs_yz_count": int(len(rows) - worsened_count),
        "final_success_sample_count": int(sum(1 for value in sample_success.values() if value)),
        "metrics": metrics,
        "high_risk": high_risk,
        "plain_interpretation": (
            "If scorer/executor baseline state errors are high and many high-confidence "
            "handoff chunks worsen y/z, the blocker is live consequence calibration, "
            "not episode length or another scalar gate."
        ),
        "evidence_boundary": (
            "This audit uses real after-state labels only to diagnose and calibrate "
            "future selectors. It is not task-completion evidence."
        ),
    }

    output_root = Path(args.output_root).resolve()
    write_json(output_root / "live_outcome_calibration_summary.json", summary)
    write_jsonl(output_root / "live_outcome_calibration_rows.jsonl", scored_rows)
    write_markdown(output_root / "live_outcome_calibration_summary.md", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
