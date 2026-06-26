#!/usr/bin/env python3
"""Extract real live outcomes for selected receding-controller chunks."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "cosmos3_live_receding_selected_outcome_v1"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")


def as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return None
    return out


def vector_delta(after: list[float] | None, before: list[float] | None) -> list[float] | None:
    if after is None or before is None or len(after) != len(before):
        return None
    return [a - b for a, b in zip(after, before)]


def abs_l1(vec: list[float] | None) -> float | None:
    if vec is None:
        return None
    return float(sum(abs(v) for v in vec))


def abs_yz(vec: list[float] | None) -> float | None:
    if vec is None or len(vec) < 3:
        return None
    return float(abs(vec[1]) + abs(vec[2]))


def error_vec(pred: list[float] | None, actual: list[float] | None) -> list[float] | None:
    if pred is None or actual is None or len(pred) != len(actual):
        return None
    return [p - a for p, a in zip(pred, actual)]


def find_iter_dir(sample_dir: Path, iteration: int) -> Path | None:
    matches = sorted(sample_dir.glob(f"iter_{iteration:02d}_prefix_f*"))
    return matches[0] if matches else None


def load_candidate_payload(sample_dir: Path, iteration: int) -> dict[str, Any] | None:
    iter_dir = find_iter_dir(sample_dir, iteration)
    if iter_dir is None:
        return None
    path = iter_dir / "candidate_executor_action_chunk.json"
    if not path.exists():
        return None
    payload = read_json(path)
    payload["_path"] = str(path)
    payload["_iter_dir"] = str(iter_dir)
    return payload


def get_after_state(iter_record: dict[str, Any]) -> list[float] | None:
    after_eval = iter_record.get("after_eval")
    if isinstance(after_eval, dict):
        vec = as_float_list(after_eval.get("peg_head_pos_at_hole"))
        if vec is not None:
            return vec
    after_handoff = iter_record.get("after_dp_handoff_eval")
    if isinstance(after_handoff, dict):
        return as_float_list(after_handoff.get("peg_head_pos_at_hole"))
    return None


def get_after_success(iter_record: dict[str, Any]) -> bool | None:
    for key in ("after_eval", "after_dp_handoff_eval"):
        payload = iter_record.get(key)
        if isinstance(payload, dict) and payload.get("success") is not None:
            return bool(payload.get("success"))
    return None


def get_gate(iter_record: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("post_cosmos_continuability_gate", "after_dp_handoff_continuability_gate"):
        payload = iter_record.get(key)
        if isinstance(payload, dict):
            return payload
    return None


def extract_records(panel_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    panel_summary_path = panel_root / "live_receding_panel_summary.json"
    panel_summary = read_json(panel_summary_path) if panel_summary_path.exists() else {}

    for sample_dir in sorted(panel_root.glob("sample_*")):
        summary_path = sample_dir / "live_receding_loop_summary.json"
        if not summary_path.exists():
            continue
        sample_summary = read_json(summary_path)
        final_eval = sample_summary.get("final_eval") or {}
        final_state = as_float_list(final_eval.get("peg_head_pos_at_hole"))
        sample_name = str(sample_summary.get("sample_name") or sample_dir.name)
        scenario = str(sample_summary.get("scenario") or sample_dir.name)

        for iter_record in sample_summary.get("iterations", []):
            if not isinstance(iter_record, dict):
                continue
            iteration = int(iter_record.get("iteration", len(records)))
            candidate_name = iter_record.get("candidate_executor_selected_candidate_name")
            chunk_type = "candidate_executor" if candidate_name else "dp_handoff"
            before = None
            before_eval = iter_record.get("before_eval")
            if isinstance(before_eval, dict):
                before = as_float_list(before_eval.get("peg_head_pos_at_hole"))
            after = get_after_state(iter_record)
            gate = get_gate(iter_record)
            checks = gate.get("checks", {}) if isinstance(gate, dict) else {}

            candidate_payload = (
                load_candidate_payload(sample_dir, iteration) if chunk_type == "candidate_executor" else None
            )
            selected = candidate_payload.get("selected_candidate", {}) if isinstance(candidate_payload, dict) else {}

            scorer_pred_state = as_float_list(selected.get("outcome_scorer_predicted_state"))
            executor_pred_state = as_float_list(selected.get("predicted_next_state_peg_head_at_hole"))
            scorer_error = error_vec(scorer_pred_state, after)
            executor_error = error_vec(executor_pred_state, after)

            record = {
                "schema": SCHEMA,
                "panel_root": str(panel_root),
                "panel_summary": str(panel_summary_path) if panel_summary_path.exists() else None,
                "contact_sheet": (panel_summary.get("contact_sheet") or {}).get("contact_sheet"),
                "sample_dir": str(sample_dir),
                "sample_name": sample_name,
                "scenario": scenario,
                "iteration": iteration,
                "chunk_type": chunk_type,
                "selected_candidate_name": candidate_name or "dp_handoff",
                "candidate_payload_path": candidate_payload.get("_path") if isinstance(candidate_payload, dict) else None,
                "iter_dir": candidate_payload.get("_iter_dir") if isinstance(candidate_payload, dict) else None,
                "before_peg_head_at_hole": before,
                "after_peg_head_at_hole": after,
                "after_minus_before": vector_delta(after, before),
                "before_abs_l1": abs_l1(before),
                "after_abs_l1": abs_l1(after),
                "before_abs_yz": abs_yz(before),
                "after_abs_yz": abs_yz(after),
                "after_success": get_after_success(iter_record),
                "gate_ok": gate.get("ok") if isinstance(gate, dict) else None,
                "gate_checks": checks if isinstance(checks, dict) else {},
                "cosmos_chunk_stop_reason": iter_record.get("cosmos_chunk_stop_reason"),
                "iteration_stop_reason": iter_record.get("stop_reason"),
                "dp_handoff_executed": iter_record.get("dp_handoff_executed"),
                "final_sample_success": bool(final_eval.get("success")) if final_eval.get("success") is not None else None,
                "final_peg_head_at_hole": final_state,
                "full_episode_length_ok": sample_summary.get("full_episode_length_ok"),
                "final_observed_frames": sample_summary.get("final_observed_frames"),
                "scorer_predicted_state": scorer_pred_state,
                "scorer_predicted_state_error": scorer_error,
                "scorer_predicted_state_abs_l1_error": abs_l1(scorer_error),
                "executor_predicted_next_state": executor_pred_state,
                "executor_predicted_next_state_error": executor_error,
                "executor_predicted_next_state_abs_l1_error": abs_l1(executor_error),
                "outcome_scorer_score": selected.get("outcome_scorer_score"),
                "outcome_scorer_predicted_handoff_success_probability": selected.get(
                    "outcome_scorer_predicted_handoff_success_probability"
                ),
                "outcome_scorer_predicted_continuable_probability": selected.get(
                    "outcome_scorer_predicted_continuable_probability"
                ),
                "outcome_scorer_predicted_inserted_probability": selected.get(
                    "outcome_scorer_predicted_inserted_probability"
                ),
                "outcome_scorer_predicted_progress": selected.get("outcome_scorer_predicted_progress"),
                "candidate_count": candidate_payload.get("candidate_count") if isinstance(candidate_payload, dict) else None,
                "controller_boundary": (
                    "Real executed selected chunks from the live receding controller. "
                    "These rows are calibration/debug labels for live action consequences, "
                    "not method success evidence."
                ),
            }
            records.append(record)
    return records


def axis_stats(errors: list[list[float]]) -> dict[str, Any]:
    if not errors:
        return {"count": 0}
    dim = max(len(e) for e in errors)
    names = ["x", "y", "z"][:dim]
    out: dict[str, Any] = {"count": len(errors)}
    for axis, name in enumerate(names):
        vals = [e[axis] for e in errors if len(e) > axis and math.isfinite(e[axis])]
        if not vals:
            continue
        mae = sum(abs(v) for v in vals) / len(vals)
        rmse = math.sqrt(sum(v * v for v in vals) / len(vals))
        bias = sum(vals) / len(vals)
        out[name] = {"mae": mae, "rmse": rmse, "bias": bias}
    return out


def summarize(records: list[dict[str, Any]], panel_root: Path) -> dict[str, Any]:
    selected = [r for r in records if r.get("chunk_type") == "candidate_executor"]
    handoff = [r for r in records if r.get("chunk_type") == "dp_handoff"]
    names = Counter(str(r.get("selected_candidate_name")) for r in records)
    scenarios = Counter(str(r.get("scenario")) for r in records)

    def worsened_yz(r: dict[str, Any]) -> bool:
        before = r.get("before_abs_yz")
        after = r.get("after_abs_yz")
        return isinstance(before, (int, float)) and isinstance(after, (int, float)) and after > before

    scorer_errors = [
        r["scorer_predicted_state_error"]
        for r in selected
        if isinstance(r.get("scorer_predicted_state_error"), list)
    ]
    executor_errors = [
        r["executor_predicted_next_state_error"]
        for r in selected
        if isinstance(r.get("executor_predicted_next_state_error"), list)
    ]

    gate_fail_checks: Counter[str] = Counter()
    for r in records:
        checks = r.get("gate_checks")
        if isinstance(checks, dict):
            for key, value in checks.items():
                if value is False:
                    gate_fail_checks[key] += 1

    final_success_samples = {
        str(r.get("sample_name"))
        for r in records
        if r.get("final_sample_success") is True
    }
    full_episode_samples = {
        str(r.get("sample_name"))
        for r in records
        if r.get("full_episode_length_ok") is True
    }
    all_samples = {str(r.get("sample_name")) for r in records}

    return {
        "schema": "cosmos3_live_receding_selected_outcome_summary_v1",
        "panel_root": str(panel_root),
        "records": len(records),
        "candidate_executor_records": len(selected),
        "dp_handoff_records": len(handoff),
        "sample_count": len(all_samples),
        "final_success_sample_count": len(final_success_samples),
        "full_episode_length_ok_sample_count": len(full_episode_samples),
        "selected_candidate_name_counts": dict(sorted(names.items())),
        "scenario_counts": dict(sorted(scenarios.items())),
        "worsened_abs_yz_selected_count": sum(1 for r in selected if worsened_yz(r)),
        "improved_abs_yz_selected_count": sum(1 for r in selected if not worsened_yz(r)),
        "gate_ok_count": sum(1 for r in records if r.get("gate_ok") is True),
        "gate_fail_check_counts": dict(sorted(gate_fail_checks.items())),
        "scorer_predicted_state_error": axis_stats(scorer_errors),
        "executor_predicted_next_state_error": axis_stats(executor_errors),
        "plain_interpretation": (
            "This file measures whether the selected live chunks did what the scorer "
            "expected in the real rollout. Large y/z errors or worsened abs_yz mean "
            "the offline candidate-outcome scorer is not calibrated to live contact "
            "execution."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = extract_records(args.panel_root)
    summary = summarize(records, args.panel_root)
    write_jsonl(args.output_jsonl, records)
    write_json(args.summary_json, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if records else 64


if __name__ == "__main__":
    raise SystemExit(main())
