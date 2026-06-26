#!/usr/bin/env python3
"""Decide whether a formal contact-executor run is allowed to launch live eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--max-action-mse-ratio", type=float, default=1.0)
    parser.add_argument("--max-progress-mse", type=float, default=0.05)
    parser.add_argument("--min-inserted-acc", type=float, default=0.75)
    parser.add_argument("--min-dp-continuable-acc", type=float, default=0.75)
    parser.add_argument("--min-positive-scale-gain", type=float, default=0.0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    root = Path(args.training_root).resolve()
    summary_path = root / "training_summary.json"
    group_path = root / "post_training_group_metrics.json"
    missing = [str(path) for path in (summary_path, group_path) if not path.is_file()]
    if missing:
        payload = {
            "schema": "cosmos3_contact_executor_formal_gate_v1",
            "training_root": str(root),
            "live_eval_allowed": False,
            "missing_files": missing,
            "failure_reasons": ["missing_summary_or_group_metrics"],
        }
        output = Path(args.output_json).resolve() if args.output_json else root / "formal_live_eval_gate.json"
        write_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    summary = read_json(summary_path)
    groups = read_json(group_path)
    metrics = dict(summary.get("final_metrics") or {})
    overall = dict(groups.get("overall") or {})
    sweep = dict(groups.get("residual_scale_sweep") or {})
    global_best = dict(sweep.get("global_best") or {})
    selector_diagnostics = dict(sweep.get("selector_diagnostics") or {})
    phase_selector = dict(selector_diagnostics.get("by_current_phase") or {})
    traincal_diagnostics = dict(sweep.get("train_calibrated_selector_diagnostics") or {})
    traincal_phase_selector = dict(traincal_diagnostics.get("by_current_phase") or {})
    traincal_global_selector = dict(traincal_diagnostics.get("global") or {})
    group_checkpoint = Path(str(groups.get("checkpoint") or ""))
    final_checkpoint = root / "checkpoint_final.pt"

    action_mse = float(metrics.get("eval_action_mse", overall.get("action_mse", float("inf"))))
    prior_mse = float(metrics.get("eval_baseline_dp_prior_mse", overall.get("dp_prior_mse", float("inf"))))
    progress_mse = float(metrics.get("eval_progress_mse", overall.get("progress_mse", float("inf"))))
    inserted_acc = float(metrics.get("eval_inserted_acc", overall.get("inserted_acc", -1.0)))
    dp_acc = float(metrics.get("eval_dp_continuable_acc", overall.get("dp_continuable_acc", -1.0)))
    formal_floor_met = bool(summary.get("formal_training_floor_met"))
    best_scale = float(global_best.get("scale", 0.0))
    best_scale_mse = float(global_best.get("action_mse", float("inf")))
    positive_scale_gain = prior_mse - best_scale_mse if best_scale > 0 else 0.0
    phase_selector_mse = phase_selector.get("weighted_action_mse")
    phase_selector_mse = float(phase_selector_mse) if phase_selector_mse is not None else None
    traincal_phase_selector_mse = traincal_phase_selector.get("weighted_eval_action_mse")
    traincal_phase_selector_mse = (
        float(traincal_phase_selector_mse) if traincal_phase_selector_mse is not None else None
    )
    traincal_global_selector_mse = traincal_global_selector.get("eval_action_mse")
    traincal_global_selector_mse = (
        float(traincal_global_selector_mse) if traincal_global_selector_mse is not None else None
    )

    reasons: list[str] = []
    if not formal_floor_met:
        reasons.append("formal_training_floor_not_met")
    if not final_checkpoint.is_file():
        reasons.append("missing_checkpoint_final")
    if group_checkpoint.name != "checkpoint_final.pt":
        reasons.append("group_metrics_not_from_checkpoint_final")
    elif group_checkpoint.resolve() != final_checkpoint.resolve():
        reasons.append("group_metrics_checkpoint_path_mismatch")
    if not (action_mse <= prior_mse * float(args.max_action_mse_ratio)):
        reasons.append("unscaled_action_mse_worse_than_dp_prior")
    if not (progress_mse <= float(args.max_progress_mse)):
        reasons.append("progress_mse_too_high")
    if not (inserted_acc >= float(args.min_inserted_acc)):
        reasons.append("inserted_accuracy_too_low")
    if not (dp_acc >= float(args.min_dp_continuable_acc)):
        reasons.append("dp_continuable_accuracy_too_low")
    if best_scale > 0 and positive_scale_gain <= float(args.min_positive_scale_gain):
        reasons.append("positive_residual_scale_gain_too_small")

    # A positive scale that barely beats DP can be useful for later candidate
    # selection, but it is not enough to execute the unscaled action head.
    live_eval_allowed = len(reasons) == 0
    payload = {
        "schema": "cosmos3_contact_executor_formal_gate_v1",
        "training_root": str(root),
        "summary_path": str(summary_path),
        "group_metrics_path": str(group_path),
        "group_metrics_checkpoint": str(group_checkpoint),
        "live_eval_allowed": live_eval_allowed,
        "failure_reasons": reasons,
        "formal_training_floor_met": formal_floor_met,
        "action_mse": action_mse,
        "dp_prior_mse": prior_mse,
        "action_mse_ratio_to_prior": action_mse / prior_mse if prior_mse > 0 else None,
        "progress_mse": progress_mse,
        "inserted_acc": inserted_acc,
        "dp_continuable_acc": dp_acc,
        "best_global_residual_scale": best_scale,
        "best_global_residual_scale_mse": best_scale_mse,
        "positive_scale_gain_over_prior": positive_scale_gain,
        "oracle_phase_selector_mse": phase_selector_mse,
        "oracle_phase_selector_gain_over_prior": (
            prior_mse - phase_selector_mse if phase_selector_mse is not None else None
        ),
        "train_calibrated_global_selector_mse": traincal_global_selector_mse,
        "train_calibrated_phase_selector_mse": traincal_phase_selector_mse,
        "train_calibrated_phase_selector_gain_over_prior": (
            prior_mse - traincal_phase_selector_mse if traincal_phase_selector_mse is not None else None
        ),
        "boundary": (
            "This gate only allows launching live eval. It does not prove "
            "method success. If false, stop and report the concrete offline "
            "blocker instead of running closed-loop videos."
        ),
    }
    output = Path(args.output_json).resolve() if args.output_json else root / "formal_live_eval_gate.json"
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
