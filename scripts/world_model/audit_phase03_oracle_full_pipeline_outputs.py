#!/usr/bin/env python3
"""Audit Phase 03 full-pipeline Oracle outputs.

This is a read-only artifact checker. It does not run ManiSkill, Cosmos,
rollouts, rendering, training, or evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--require-source-h5", action="store_true")
    parser.add_argument(
        "--allow-diagnostic-action-row-offset",
        action="store_true",
        help=(
            "Allow a nonzero Cosmos action row offset only as diagnostic evidence. "
            "Such runs must not be validation-key success evidence."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def exists_file(path_text: str | None) -> bool:
    return bool(path_text) and Path(path_text).is_file()


def fail_if(condition: bool, failures: list[str], name: str) -> None:
    if condition:
        failures.append(name)


def maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_premotion_report(report: dict[str, Any]) -> bool:
    stage_name = str(report.get("stage_name", ""))
    prefix_role = str(report.get("prefix_role", ""))
    return (
        stage_name.startswith("premotion_")
        or stage_name.startswith("pre/")
        or prefix_role == "target_pre_motion"
    )


def is_postmotion_report(report: dict[str, Any]) -> bool:
    stage_name = str(report.get("stage_name", ""))
    prefix_role = str(report.get("prefix_role", ""))
    return (
        stage_name.startswith("postmotion_")
        or stage_name.startswith("post/")
        or prefix_role == "target_motion_observed"
    )


def vector3(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


def norm3(value: list[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in value[:3]))


def float_list(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def cosine3(a: list[float], b: list[float]) -> float | None:
    an = norm3(a)
    bn = norm3(b)
    if an <= 1.0e-12 or bn <= 1.0e-12:
        return None
    return sum(float(x) * float(y) for x, y in zip(a[:3], b[:3])) / (an * bn)


def observed_peg_delta_from_trace(trace: list[Any]) -> list[float] | None:
    trigger_row = next(
        (
            row
            for row in trace
            if isinstance(row, dict) and row.get("stage") == "peg_perturb_trigger_zero_robot_action"
        ),
        None,
    )
    if not isinstance(trigger_row, dict):
        return None
    trigger_state = trigger_row.get("live_eval") or {}
    trigger_peg = vector3(trigger_state.get("peg_xyz"))
    first_delta = vector3(trigger_row.get("peg_perturb_observed_delta_xyz"))
    if trigger_peg is None or first_delta is None:
        return None
    reference = [trigger_peg[i] - first_delta[i] for i in range(3)]
    last_forced_peg = trigger_peg
    for row in trace:
        if not isinstance(row, dict):
            continue
        force = vector3(row.get("peg_perturb_force_xyz"))
        state = row.get("live_eval") or {}
        peg = vector3(state.get("peg_xyz"))
        if force is not None and norm3(force) > 1.0e-9 and peg is not None:
            last_forced_peg = peg
    return [last_forced_peg[i] - reference[i] for i in range(3)]


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    failures: list[str] = []

    summary_path = run_dir / "summary.json"
    action_trace_path = run_dir / "action_trace.json"
    manifest_path = run_dir / "manifest.json"
    classification_path = run_dir / "classification.txt"

    fail_if(not summary_path.is_file(), failures, "missing_summary_json")
    fail_if(not action_trace_path.is_file(), failures, "missing_action_trace_json")
    fail_if(not manifest_path.is_file(), failures, "missing_manifest_json")
    fail_if(not classification_path.is_file(), failures, "missing_classification_txt")
    if failures:
        report = {"ok": False, "run_dir": str(run_dir), "failures": failures}
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    summary = read_json(summary_path)
    trace = read_json(action_trace_path)
    manifest = read_json(manifest_path)
    cosmos_reports = summary.get("cosmos_policy_reports") or []
    adapter = summary.get("cosmos_action_adapter") or manifest.get("cosmos_action_adapter") or {}
    action_row_offset = int(adapter.get("action_row_offset") or 0)
    action_row_offset_diagnostic = bool(adapter.get("action_row_offset_diagnostic") or action_row_offset != 0)

    fail_if(summary.get("method_evidence_allowed") is not False, failures, "method_evidence_allowed_not_false")
    fail_if(summary.get("physical_insertion_success_claimed") is not False, failures, "physical_success_claimed")
    if action_row_offset_diagnostic:
        fail_if(
            not args.allow_diagnostic_action_row_offset,
            failures,
            "diagnostic_action_row_offset_without_audit_override",
        )
        fail_if(action_row_offset == 0, failures, "action_row_offset_diagnostic_missing_nonzero_offset")
        fail_if(
            not adapter.get("action_row_offset_source"),
            failures,
            "action_row_offset_source_missing",
        )
        fail_if(
            summary.get("validation_key_success_allowed") is not False,
            failures,
            "diagnostic_action_row_offset_validation_success_allowed",
        )
    if args.require_source_h5:
        fail_if(not summary.get("source_h5_path"), failures, "missing_required_source_h5_path")
        fail_if(not summary.get("source_key"), failures, "missing_required_source_key")
        fail_if(
            summary.get("target_motion_source") != "fix3_733_source_h5_protocol",
            failures,
            "target_motion_not_from_fix3_733_source_h5_protocol",
        )
        if not action_row_offset_diagnostic:
            fail_if(
                summary.get("validation_key_success_allowed") is not True,
                failures,
                "validation_key_success_not_allowed",
            )
    fail_if(summary.get("set_pose_used_on_peg") is not False, failures, "set_pose_used_on_peg_not_false")
    fail_if(summary.get("forbidden_peg_state_intervention_used") is not False, failures, "forbidden_peg_state_intervention")
    peg_state_guard = summary.get("peg_state_guard") or {}
    fail_if(peg_state_guard.get("ok") is not True, failures, "peg_state_guard_not_ok")
    fail_if(not exists_file(summary.get("video")), failures, "missing_raw_attempt_video")
    fail_if(not exists_file(summary.get("annotated_video")), failures, "missing_annotated_attempt_video")
    fail_if(not isinstance(trace, list) or len(trace) == 0, failures, "empty_action_trace")
    fail_if(not isinstance(cosmos_reports, list) or len(cosmos_reports) == 0, failures, "missing_cosmos_policy_reports")
    premotion_reports = [
        report
        for report in cosmos_reports
        if isinstance(report, dict) and is_premotion_report(report)
    ]
    postmotion_reports = [
        report
        for report in cosmos_reports
        if isinstance(report, dict) and is_postmotion_report(report)
    ]
    required_premotion_reports = maybe_int(
        summary.get("required_premotion_cosmos_predictions")
        or manifest.get("required_premotion_cosmos_predictions")
    )
    if required_premotion_reports is None:
        configured_premotion_reports = maybe_int(
            summary.get("max_premotion_cosmos_predictions")
            or manifest.get("max_premotion_cosmos_predictions")
        )
        required_premotion_reports = max(2, configured_premotion_reports or 0)
    fail_if(not premotion_reports, failures, "missing_premotion_cosmos_prediction")
    fail_if(
        len(premotion_reports) < required_premotion_reports,
        failures,
        "insufficient_repeated_premotion_cosmos_predictions",
    )
    fail_if(not postmotion_reports, failures, "missing_postmotion_cosmos_prediction")

    stages = [row.get("stage") for row in trace if isinstance(row, dict)]
    dynamic_protocol_type = str(summary.get("dynamic_protocol_type") or manifest.get("dynamic_protocol_type") or "target_motion")
    trigger_stage = (
        "peg_perturb_trigger_zero_robot_action"
        if dynamic_protocol_type == "peg_perturb"
        else "target_motion_trigger_no_robot_action"
    )
    fail_if("dp_static_prefix" not in stages, failures, "missing_dp_static_prefix_stage")
    fail_if(trigger_stage not in stages, failures, f"missing_{trigger_stage}_stage")
    fail_if("cosmos_dynamic_control" not in stages, failures, "missing_cosmos_dynamic_control_stage")
    trigger_index = (
        stages.index(trigger_stage)
        if trigger_stage in stages
        else None
    )
    first_dynamic_index = (
        stages.index("cosmos_dynamic_control") if "cosmos_dynamic_control" in stages else None
    )
    last_dp_index = (
        max(idx for idx, stage in enumerate(stages) if stage == "dp_static_prefix")
        if "dp_static_prefix" in stages
        else None
    )
    if trigger_index is not None and first_dynamic_index is not None:
        fail_if(trigger_index > first_dynamic_index, failures, "target_motion_trigger_after_cosmos_dynamic")
    if trigger_index is not None and last_dp_index is not None:
        fail_if(last_dp_index > trigger_index, failures, "dp_prefix_after_target_motion_trigger")
    fail_if(
        any(
            row.get("target_motion_applied") is True
            for row in trace
            if isinstance(row, dict) and row.get("stage") == "dp_static_prefix"
        ),
        failures,
        "diffusion_policy_crossed_target_motion_boundary",
    )
    fail_if(
        bool(summary.get("cosmos_dynamic_actions_executed")) is not True,
        failures,
        "summary_cosmos_dynamic_actions_not_true",
    )
    dynamic_rows = [row for row in trace if isinstance(row, dict) and row.get("stage") == "cosmos_dynamic_control"]
    fail_if(not dynamic_rows, failures, "no_dynamic_rows")
    min_dynamic_before_finisher = int(summary.get("min_cosmos_dynamic_actions_before_finisher") or 4)
    fail_if(
        min_dynamic_before_finisher < 4,
        failures,
        "min_cosmos_dynamic_actions_before_finisher_below_active_protocol_floor_4",
    )
    fail_if(
        int(summary.get("cosmos_dynamic_action_count") or 0) < min_dynamic_before_finisher,
        failures,
        "insufficient_cosmos_dynamic_actions_before_finisher",
    )
    fail_if(
        any(row.get("action_source") != "cosmos3_policy_output" for row in dynamic_rows),
        failures,
        "non_cosmos_action_source_in_dynamic_stage",
    )
    fail_if(
        any("raw_cosmos_action" not in row or "action" not in row for row in dynamic_rows),
        failures,
        "dynamic_rows_missing_raw_or_executed_action",
    )
    postmotion_by_round: dict[int, dict[str, Any]] = {}
    for report in postmotion_reports:
        stage_name = str(report.get("stage_name", ""))
        try:
            round_idx = int(stage_name.rsplit("/", 1)[-1])
        except ValueError:
            continue
        postmotion_by_round[round_idx] = report
    dynamic_report_mismatch_steps: list[Any] = []
    dynamic_raw_action_mismatch_steps: list[Any] = []
    for row in dynamic_rows:
        round_idx = maybe_int(row.get("cosmos_round"))
        local_idx = maybe_int(row.get("cosmos_action_index"))
        if round_idx is None or local_idx is None or round_idx not in postmotion_by_round:
            dynamic_report_mismatch_steps.append(row.get("env_step"))
            continue
        report = postmotion_by_round[round_idx]
        chunk = report.get("denormalized_robot_action_chunk")
        if not isinstance(chunk, list) or local_idx < 0 or local_idx >= len(chunk):
            dynamic_report_mismatch_steps.append(row.get("env_step"))
            continue
        expected_raw = float_list(chunk[local_idx])
        actual_raw = float_list(row.get("raw_cosmos_action"))
        if expected_raw is None or actual_raw is None or len(expected_raw) != len(actual_raw):
            dynamic_raw_action_mismatch_steps.append(row.get("env_step"))
            continue
        max_abs_diff = max((abs(a - b) for a, b in zip(expected_raw, actual_raw)), default=0.0)
        if max_abs_diff > 1.0e-5:
            dynamic_raw_action_mismatch_steps.append(row.get("env_step"))
        report_chunk_start = maybe_int(report.get("chunk_start"))
        report_chunk_end = maybe_int(report.get("chunk_end_exclusive"))
        row_chunk_start = maybe_int(row.get("cosmos_action_chunk_start"))
        row_chunk_end = maybe_int(row.get("cosmos_action_chunk_end_exclusive"))
        if (
            report_chunk_start is not None
            and row_chunk_start is not None
            and report_chunk_start != row_chunk_start
        ):
            dynamic_report_mismatch_steps.append(row.get("env_step"))
        if (
            report_chunk_end is not None
            and row_chunk_end is not None
            and report_chunk_end != row_chunk_end
        ):
            dynamic_report_mismatch_steps.append(row.get("env_step"))
    fail_if(
        bool(dynamic_report_mismatch_steps),
        failures,
        "cosmos_dynamic_rows_do_not_reference_valid_postmotion_report_chunk",
    )
    fail_if(
        bool(dynamic_raw_action_mismatch_steps),
        failures,
        "cosmos_dynamic_raw_action_not_equal_report_chunk_row",
    )
    row_offset_trace_values = sorted(
        {
            value
            for value in (maybe_int(row.get("cosmos_action_row_offset")) for row in dynamic_rows)
            if value is not None
        }
    )
    row_offset_prediction_mismatches = []
    if action_row_offset_diagnostic:
        required_row_offset_fields = {
            "cosmos_action_chunk_start",
            "cosmos_action_chunk_end_exclusive",
            "cosmos_action_row_offset",
            "cosmos_action_row_offset_source",
            "cosmos_raw_chunk_start",
            "cosmos_predicted_action_row_index",
        }
        fail_if(
            any(not required_row_offset_fields.issubset(row.keys()) for row in dynamic_rows),
            failures,
            "diagnostic_action_row_offset_dynamic_rows_missing_trace_fields",
        )
        fail_if(
            any(maybe_int(row.get("cosmos_action_row_offset")) != action_row_offset for row in dynamic_rows),
            failures,
            "diagnostic_action_row_offset_dynamic_rows_wrong_offset",
        )
        fail_if(
            any(row.get("cosmos_action_row_offset_source") != adapter.get("action_row_offset_source") for row in dynamic_rows),
            failures,
            "diagnostic_action_row_offset_dynamic_rows_wrong_source",
        )
        for row in dynamic_rows:
            chunk_start = maybe_int(row.get("cosmos_action_chunk_start"))
            chunk_end = maybe_int(row.get("cosmos_action_chunk_end_exclusive"))
            local_idx = maybe_int(row.get("cosmos_action_index"))
            predicted_idx = maybe_int(row.get("cosmos_predicted_action_row_index"))
            if chunk_start is None or chunk_end is None or local_idx is None or predicted_idx is None:
                row_offset_prediction_mismatches.append(row.get("env_step"))
                continue
            if predicted_idx != chunk_start + local_idx or not (chunk_start <= predicted_idx < chunk_end):
                row_offset_prediction_mismatches.append(row.get("env_step"))
        fail_if(
            bool(row_offset_prediction_mismatches),
            failures,
            "diagnostic_action_row_offset_dynamic_row_index_mismatch",
        )
    fail_if(
        any("target_motion_cumulative_xyz" not in row for row in trace if isinstance(row, dict)),
        failures,
        "trace_missing_target_motion_cumulative_xyz",
    )
    fail_if(
        any("target_motion_delta_xyz" not in row for row in trace if isinstance(row, dict)),
        failures,
        "trace_missing_target_motion_delta_xyz",
    )
    target_motion_per_step = float_or_none(summary.get("target_motion_per_step"))
    target_motion_xyz = vector3(summary.get("target_motion_xyz"))
    target_motion_delta_norms: list[float] = []
    target_motion_unparseable_steps: list[Any] = []
    target_motion_cumulative_mismatches: list[Any] = []
    target_motion_completion_error: float | None = None
    previous_cumulative = [0.0, 0.0, 0.0]
    for row in trace:
        if not isinstance(row, dict):
            continue
        delta = vector3(row.get("target_motion_delta_xyz"))
        cumulative = vector3(row.get("target_motion_cumulative_xyz"))
        if delta is None or cumulative is None:
            target_motion_unparseable_steps.append(row.get("env_step"))
            continue
        target_motion_delta_norms.append(norm3(delta))
        expected_cumulative = [previous_cumulative[i] + delta[i] for i in range(3)]
        if norm3([cumulative[i] - expected_cumulative[i] for i in range(3)]) > 1.0e-5:
            target_motion_cumulative_mismatches.append(row.get("env_step"))
        previous_cumulative = cumulative
    max_target_motion_step_delta = max(target_motion_delta_norms) if target_motion_delta_norms else 0.0
    if target_motion_per_step is not None and target_motion_per_step > 0.0:
        fail_if(
            max_target_motion_step_delta > abs(target_motion_per_step) * 1.05 + 1.0e-6,
            failures,
            "target_motion_step_delta_exceeds_protocol_per_step",
        )
    fail_if(
        bool(target_motion_unparseable_steps),
        failures,
        "trace_unparseable_target_motion_delta_or_cumulative_xyz",
    )
    fail_if(
        bool(target_motion_cumulative_mismatches),
        failures,
        "target_motion_cumulative_not_equal_sum_of_logged_deltas",
    )
    if target_motion_xyz is not None and bool(summary.get("target_motion_complete_before_finisher")):
        target_motion_completion_error = norm3(
            [previous_cumulative[i] - target_motion_xyz[i] for i in range(3)]
        )
        fail_if(
            target_motion_completion_error > 1.0e-5,
            failures,
            "target_motion_complete_flag_but_cumulative_motion_mismatch",
        )
    peg_expected_delta: list[float] | None = None
    peg_observed_delta: list[float] | None = None
    peg_observed_fraction: float | None = None
    peg_observed_cosine: float | None = None
    if dynamic_protocol_type == "peg_perturb":
        source_protocol = summary.get("source_protocol") or {}
        peg_expected_delta = vector3(source_protocol.get("peg_delta_sum_xyz")) or vector3(
            source_protocol.get("peg_delta_first_xyz")
        )
        peg_observed_delta = (
            vector3(summary.get("peg_physical_perturb_observed_force_window_delta_xyz"))
            or vector3(summary.get("peg_physical_perturb_observed_cumulative_delta_xyz"))
            or observed_peg_delta_from_trace(trace)
            or vector3(summary.get("peg_physical_perturb_observed_delta_xyz"))
        )
        fail_if(peg_expected_delta is None, failures, "missing_source_peg_perturb_delta")
        fail_if(peg_observed_delta is None, failures, "missing_observed_physical_peg_perturb_delta")
        if peg_expected_delta is not None and peg_observed_delta is not None:
            expected_norm = norm3(peg_expected_delta)
            observed_norm = norm3(peg_observed_delta)
            peg_observed_fraction = observed_norm / expected_norm if expected_norm > 1.0e-12 else None
            peg_observed_cosine = cosine3(peg_expected_delta, peg_observed_delta)
            fail_if(
                peg_observed_fraction is None or peg_observed_fraction < 0.2,
                failures,
                "peg_perturb_observed_delta_too_small_for_source_key",
            )
            fail_if(
                peg_observed_cosine is None or peg_observed_cosine < 0.5,
                failures,
                "peg_perturb_observed_delta_wrong_direction_for_source_key",
            )

    for idx, report in enumerate(cosmos_reports):
        prefix = f"cosmos_report_{idx}"
        fail_if(not exists_file(report.get("prefix_video")), failures, f"{prefix}_missing_prefix_video")
        fail_if(
            not exists_file(report.get("cosmos_rgb_prediction_video")),
            failures,
            f"{prefix}_missing_cosmos_rgb_prediction_video",
        )
        fail_if(not exists_file(report.get("sample_output_json")), failures, f"{prefix}_missing_sample_output_json")
        fail_if(not exists_file(report.get("cosmos_action_chunk_json")), failures, f"{prefix}_missing_action_chunk_json")
        fail_if(report.get("ok") is not True, failures, f"{prefix}_action_chunk_not_ok")

    audit = summary.get("discontinuity_audit") or {}
    fail_if(audit.get("snap_detected") is True, failures, "snap_detected")
    fail_if("max_peg_step_displacement" not in audit, failures, "missing_peg_discontinuity_audit")
    fail_if("max_hole_step_displacement" not in audit, failures, "missing_hole_discontinuity_audit")

    finisher_stages = {
        "oracle_physical_dp_finisher",
        "oracle_physical_manual_finisher",
    }
    finisher_rows = [
        row
        for row in trace
        if isinstance(row, dict) and row.get("stage") in finisher_stages
    ]
    finisher_attempted = bool(finisher_rows)
    if finisher_attempted:
        fail_if(
            summary.get("near_target_before_finisher") is not True,
            failures,
            "finisher_started_without_near_target_gate",
        )
    fail_if(
        summary.get("near_target_before_finisher") is True and not finisher_attempted,
        failures,
        "near_target_gate_true_but_no_physical_finisher_rows",
    )

    simulator_success_metric = bool(summary.get("simulator_success_metric"))
    fail_if(
        simulator_success_metric and not finisher_attempted,
        failures,
        "simulator_success_without_physical_finisher_attempt",
    )
    visual_full_insertion_confirmed = bool(summary.get("visual_full_insertion_confirmed"))
    visual_review_required_for_success = simulator_success_metric and not visual_full_insertion_confirmed

    report = {
        "ok": not failures,
        "run_dir": str(run_dir),
        "failures": failures,
        "classification": summary.get("classification"),
        "manifest_contract": manifest.get("contract"),
        "source_h5_path": summary.get("source_h5_path"),
        "source_key": summary.get("source_key"),
        "target_motion_source": summary.get("target_motion_source"),
        "validation_key_success_allowed": bool(summary.get("validation_key_success_allowed")),
        "action_row_offset": action_row_offset,
        "action_row_offset_diagnostic": action_row_offset_diagnostic,
        "action_row_offset_source": adapter.get("action_row_offset_source"),
        "action_row_offset_trace_values": row_offset_trace_values,
        "action_row_offset_trace_index_mismatch_steps": row_offset_prediction_mismatches,
        "target_motion_per_step": target_motion_per_step,
        "max_target_motion_step_delta": max_target_motion_step_delta,
        "target_motion_unparseable_steps": target_motion_unparseable_steps,
        "target_motion_cumulative_mismatch_steps": target_motion_cumulative_mismatches,
        "target_motion_completion_error": target_motion_completion_error,
        "dynamic_protocol_type": dynamic_protocol_type,
        "trigger_stage": trigger_stage,
        "peg_physical_perturb_applied": bool(summary.get("peg_physical_perturb_applied")),
        "peg_expected_delta_xyz": peg_expected_delta,
        "peg_observed_delta_xyz": peg_observed_delta,
        "peg_observed_delta_fraction": peg_observed_fraction,
        "peg_observed_delta_cosine": peg_observed_cosine,
        "cosmos_policy_report_count": len(cosmos_reports),
        "premotion_cosmos_report_count": len(premotion_reports),
        "required_premotion_cosmos_report_count": required_premotion_reports,
        "postmotion_cosmos_report_count": len(postmotion_reports),
        "cosmos_dynamic_rows": len(dynamic_rows),
        "cosmos_dynamic_report_mismatch_steps": dynamic_report_mismatch_steps,
        "cosmos_dynamic_raw_action_mismatch_steps": dynamic_raw_action_mismatch_steps,
        "cosmos_dynamic_action_count": int(summary.get("cosmos_dynamic_action_count") or 0),
        "min_cosmos_dynamic_actions_before_finisher": min_dynamic_before_finisher,
        "finisher_controller": summary.get("finisher_controller"),
        "finisher_attempted": finisher_attempted,
        "finisher_row_count": len(finisher_rows),
        "finisher_start_step": summary.get("finisher_start_step"),
        "near_target_before_finisher": bool(summary.get("near_target_before_finisher")),
        "simulator_success_metric": simulator_success_metric,
        "physical_insertion_success_claimed": bool(summary.get("physical_insertion_success_claimed")),
        "visual_full_insertion_confirmed": visual_full_insertion_confirmed,
        "visual_review_required_for_success": visual_review_required_for_success,
        "active_robot_insertion_visual_review_required": simulator_success_metric,
        "target_assisted_insertion_must_be_rejected": True,
        "note": (
            "This audit checks protocol artifacts only. Physical success still "
            "requires visual review of the annotated video. If target/hole "
            "motion creates insertion by moving onto the peg or wooden stick, "
            "the run is not a valid Oracle success even when simulator_success_metric=true."
        ),
    }
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
