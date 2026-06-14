#!/usr/bin/env python3
"""Summarize current Cosmos3 closed-loop effectiveness failure modes.

This is a read-only evidence reducer. It does not run simulation, rendering,
training, inference, or controller code. Its purpose is to keep the current
boundary explicit: full-episode closed-loop mechanics may pass while method
effectiveness still fails against pure DP and hard-case comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective-gate-json", required=True)
    parser.add_argument("--hard-action-rebind-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--large-action-scale-ratio", type=float, default=2.0)
    parser.add_argument("--small-action-scale-ratio", type=float, default=0.5)
    parser.add_argument("--low-sign-agreement", type=float, default=0.5)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def as_float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return []
    return out


def gate_fail_totals(samples: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for sample in samples:
        if sample.get("final_success"):
            continue
        for name, count in (sample.get("continuability_gate_fail_counts") or {}).items():
            totals[str(name)] = totals.get(str(name), 0) + int(count)
    return dict(sorted(totals.items(), key=lambda item: (-item[1], item[0])))


def sample_action_flags(sample: dict[str, Any], args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    summary = sample.get("cosmos_action_vs_teacher") or {}
    ratios = as_float_list(summary.get("mean_pred_over_teacher_abs_xyz"))
    signs = as_float_list(summary.get("mean_sign_agreement_xyz"))
    axes = ["x", "y", "z"]
    for axis, ratio in zip(axes, ratios):
        if ratio > float(args.large_action_scale_ratio):
            flags.append(f"pred_{axis}_scale_gt_{args.large_action_scale_ratio:g}x_teacher")
        if ratio < float(args.small_action_scale_ratio):
            flags.append(f"pred_{axis}_scale_lt_{args.small_action_scale_ratio:g}x_teacher")
    for axis, agreement in zip(axes, signs):
        if agreement < float(args.low_sign_agreement):
            flags.append(f"pred_{axis}_sign_agreement_lt_{args.low_sign_agreement:g}")
    return flags


def summarize_action_flags(samples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, int]:
    totals: dict[str, int] = {}
    for sample in samples:
        if sample.get("final_success"):
            continue
        for flag in sample_action_flags(sample, args):
            totals[flag] = totals.get(flag, 0) + 1
    return dict(sorted(totals.items(), key=lambda item: (-item[1], item[0])))


def controller_counts_sum(samples: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for sample in samples:
        for name, count in (sample.get("controller_frame_counts") or {}).items():
            totals[str(name)] = totals.get(str(name), 0) + int(count)
    return dict(sorted(totals.items()))


def final_error_rows(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        rel = as_float_list(sample.get("final_peg_head_at_hole"))
        rows.append(
            {
                "sample_index": sample.get("sample_index"),
                "scenario": sample.get("scenario"),
                "final_success": bool(sample.get("final_success", False)),
                "final_peg_head_at_hole": rel,
                "abs_y": abs(rel[1]) if len(rel) >= 2 else None,
                "abs_z": abs(rel[2]) if len(rel) >= 3 else None,
                "controller_frame_counts": sample.get("controller_frame_counts"),
                "continuability_gate_fail_counts": sample.get("continuability_gate_fail_counts"),
                "action_flags": sample_action_flags(sample, args),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    gate = read_json(Path(args.objective_gate_json))
    hard = read_json(Path(args.hard_action_rebind_json))
    hard_samples = hard.get("samples") or []
    val_samples = ((gate.get("val_cosmos") or {}).get("samples") or [])
    hard_gate_samples = ((gate.get("hard_cosmos_on_pure_dp_failures") or {}).get("samples") or [])

    val_cosmos_success = int((gate.get("val_cosmos") or {}).get("final_success_count", 0) or 0)
    val_cosmos_total = int((gate.get("val_cosmos") or {}).get("completed_samples", len(val_samples)) or len(val_samples))
    val_dp_success = int((gate.get("val_pure_dp") or {}).get("final_success_count", 0) or 0)
    val_dp_total = int((gate.get("val_pure_dp") or {}).get("sample_count", 0) or 0)
    hard_success = int((gate.get("hard_cosmos_on_pure_dp_failures") or {}).get("final_success_count", 0) or 0)
    hard_total = int((gate.get("hard_cosmos_on_pure_dp_failures") or {}).get("completed_samples", len(hard_gate_samples)) or len(hard_gate_samples))
    hard_dp_success = int((gate.get("hard_action_rebind_analysis") or {}).get("pure_dp_final_success_count_on_matched", -1))

    failure_samples = [sample for sample in hard_samples if not sample.get("final_success")]
    rel_errors = np.asarray(
        [as_float_list(sample.get("final_peg_head_at_hole"))[:3] for sample in failure_samples],
        dtype=np.float32,
    ) if failure_samples else np.zeros((0, 3), dtype=np.float32)

    output = {
        "boundary": (
            "Read-only failure-mode summary. This report explains why the "
            "current checkpoint is not method evidence even though the "
            "full-episode closed-loop implementation contract passes."
        ),
        "objective_gate_json": str(Path(args.objective_gate_json).resolve()),
        "hard_action_rebind_json": str(Path(args.hard_action_rebind_json).resolve()),
        "implementation_contract_ok": bool(gate.get("implementation_contract_ok", False)),
        "method_effectiveness_ok": bool(gate.get("method_effectiveness_ok", False)),
        "method_effectiveness_failures": gate.get("method_effectiveness_failures") or [],
        "comparison": {
            "val_cosmos": f"{val_cosmos_success}/{val_cosmos_total}",
            "val_pure_dp": f"{val_dp_success}/{val_dp_total}",
            "hard_cosmos_on_pure_dp_failures": f"{hard_success}/{hard_total}",
            "hard_matched_pure_dp_success": f"{hard_dp_success}/{hard.get('sample_count')}",
        },
        "not_the_primary_current_failure": [
            "video_length_or_missing_301_frames",
            "manual_target_onset_disclosure",
            "missing_cosmos_takeover_annotation",
            "static_no_motion_special_branch",
        ],
        "primary_current_failure": "direct_raw_cosmos_action_rebind_and_dp_continuability_are_unreliable",
        "hard_failure_sample_count": len(failure_samples),
        "hard_failure_gate_fail_totals": gate_fail_totals(hard_samples),
        "hard_failure_action_flag_totals": summarize_action_flags(hard_samples, args),
        "hard_controller_frame_count_totals": controller_counts_sum(hard_samples),
        "hard_failure_final_abs_error_mean": (
            np.mean(np.abs(rel_errors), axis=0).astype(float).tolist()
            if rel_errors.size
            else None
        ),
        "hard_failure_final_abs_error_max": (
            np.max(np.abs(rel_errors), axis=0).astype(float).tolist()
            if rel_errors.size
            else None
        ),
        "sample_rows": final_error_rows(hard_samples, args),
        "next_aligned_action": (
            "Do not keep broad-evaluating the old checkpoint as evidence. "
            "Use the clean-role/dense-receding condition repair path, starting "
            "with preflight and two-sample overfit after explicit user approval; "
            "if direct raw Cosmos actions remain unstable, move to a learned "
            "short-chunk executor or DP-prior policy conditioned on Cosmos-"
            "predicted task state."
        ),
    }
    write_json(Path(args.output_json).resolve(), output)
    if args.output_md:
        lines = [
            "# Cosmos3 Closed-Loop Failure Modes",
            "",
            f"- implementation_contract_ok: `{output['implementation_contract_ok']}`",
            f"- method_effectiveness_ok: `{output['method_effectiveness_ok']}`",
            f"- val Cosmos: `{output['comparison']['val_cosmos']}`",
            f"- val pure DP: `{output['comparison']['val_pure_dp']}`",
            f"- hard Cosmos on pure-DP failures: `{output['comparison']['hard_cosmos_on_pure_dp_failures']}`",
            f"- hard matched pure-DP success: `{output['comparison']['hard_matched_pure_dp_success']}`",
            f"- primary_current_failure: `{output['primary_current_failure']}`",
            "",
            "## Method Effectiveness Failures",
            "",
            *(f"- `{item}`" for item in output["method_effectiveness_failures"]),
            "",
            "## Hard Failure Gate Blocks",
            "",
            *(f"- `{key}`: `{value}`" for key, value in output["hard_failure_gate_fail_totals"].items()),
            "",
            "## Hard Failure Action Flags",
            "",
            *(f"- `{key}`: `{value}`" for key, value in output["hard_failure_action_flag_totals"].items()),
            "",
            "## Not The Current Primary Failure",
            "",
            *(f"- `{item}`" for item in output["not_the_primary_current_failure"]),
            "",
            "## Next Aligned Action",
            "",
            output["next_aligned_action"],
            "",
        ]
        Path(args.output_md).resolve().write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "implementation_contract_ok": output["implementation_contract_ok"],
                "method_effectiveness_ok": output["method_effectiveness_ok"],
                "primary_current_failure": output["primary_current_failure"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
