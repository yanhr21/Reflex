#!/usr/bin/env python3
"""Summarize the candidate-executor no-sample -> diffusion gate chain."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--diffusion-smoke-root", required=True)
    parser.add_argument("--formal-diffusion-root", required=True)
    parser.add_argument("--watch-log", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def maybe_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def maybe_json_list(path: Path) -> list[Any]:
    if not path.is_file():
        return []
    payload = read_json(path)
    return payload if isinstance(payload, list) else []


def read_key_value_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def source_count_summary(source_counts: Any) -> dict[str, Any]:
    source_counts = source_counts if isinstance(source_counts, dict) else {}
    non_dp = 0
    diffusion = 0
    total = 0
    for name, count_value in source_counts.items():
        try:
            count = int(count_value)
        except (TypeError, ValueError):
            continue
        total += count
        if str(name) != "dp_prior":
            non_dp += count
        if str(name).startswith("diffusion_"):
            diffusion += count
    return {
        "total": total,
        "non_dp": non_dp,
        "non_dp_fraction": float(non_dp / max(1, total)) if total else None,
        "diffusion": diffusion,
    }


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size": int(path.stat().st_size) if path.is_file() else 0,
    }


def summarize_training_history(root: Path) -> dict[str, Any]:
    history_path = root / "training_history.json"
    history = maybe_json_list(history_path)
    latest = history[-1] if history and isinstance(history[-1], dict) else {}
    latest_eval = latest.get("eval") if isinstance(latest.get("eval"), dict) else {}
    latest_sources = latest_eval.get("candidate_source_counts")
    latest_source_summary = source_count_summary(latest_sources)
    recent_rows = [row for row in history[-24:] if isinstance(row, dict)]
    recent_deltas: list[float] = []
    for row in recent_rows:
        eval_payload = row.get("eval") if isinstance(row.get("eval"), dict) else {}
        try:
            selected = float(eval_payload.get("selected_action_mse"))
            dp = float(eval_payload.get("dp_prior_action_mse"))
        except (TypeError, ValueError):
            continue
        recent_deltas.append(selected - dp)
    recent_worse = sum(1 for value in recent_deltas if value > 0.0)
    return {
        "history": file_info(history_path),
        "num_records": len(history),
        "latest_step": latest.get("step"),
        "latest_elapsed_seconds": latest.get("elapsed_seconds"),
        "latest_selected_action_mse": latest_eval.get("selected_action_mse"),
        "latest_dp_prior_action_mse": latest_eval.get("dp_prior_action_mse"),
        "latest_selected_minus_dp_prior_mse": latest_eval.get("selected_minus_dp_prior_mse"),
        "latest_teacher_progress_mse": latest_eval.get("teacher_progress_mse"),
        "latest_teacher_value_mse": latest_eval.get("teacher_value_mse"),
        "latest_teacher_inserted_acc": latest_eval.get("teacher_inserted_acc"),
        "latest_teacher_dp_continuable_acc": latest_eval.get("teacher_dp_continuable_acc"),
        "latest_candidate_source_counts": latest_sources,
        "latest_selected_non_dp_candidate_count": latest_eval.get(
            "selected_non_dp_candidate_count", latest_source_summary["non_dp"]
        ),
        "latest_selected_non_dp_candidate_fraction": latest_eval.get(
            "selected_non_dp_candidate_fraction", latest_source_summary["non_dp_fraction"]
        ),
        "latest_selected_diffusion_candidate_count": latest_eval.get(
            "selected_diffusion_candidate_count", latest_source_summary["diffusion"]
        ),
        "recent_window_records": len(recent_deltas),
        "recent_window_selected_worse_than_dp_count": recent_worse,
        "recent_window_selected_not_worse_than_dp_count": len(recent_deltas) - recent_worse,
        "recent_window_delta_mean": float(sum(recent_deltas) / len(recent_deltas)) if recent_deltas else None,
        "recent_window_delta_min": float(min(recent_deltas)) if recent_deltas else None,
        "recent_window_delta_max": float(max(recent_deltas)) if recent_deltas else None,
    }


def summarize_training_root(root: Path) -> dict[str, Any]:
    summary_path = root / "training_summary.json"
    checkpoint_path = root / "checkpoint_final.pt"
    payload = maybe_json(summary_path)
    live_eval_checkpoint_raw = payload.get("formal_live_eval_checkpoint") if payload else None
    live_eval_checkpoint_path = Path(str(live_eval_checkpoint_raw)) if live_eval_checkpoint_raw else None
    post_gate_path = root / "post_gate_status.json"
    manifest_path = root / "run_manifest.txt"
    post_gate = maybe_json(post_gate_path)
    final_eval = (payload.get("final_metrics") or {}).get("eval") if payload else {}
    final_eval = final_eval if isinstance(final_eval, dict) else {}
    final_sources = final_eval.get("candidate_source_counts")
    final_source_summary = source_count_summary(final_sources)
    history = summarize_training_history(root)
    manifest = read_key_value_file(manifest_path)
    summary_output_root = payload.get("output_root") if payload else None
    summary_output_root_matches = None
    if summary_path.is_file():
        summary_output_root_matches = Path(str(summary_output_root)).resolve() == root.resolve() if summary_output_root else False
    configured_min_wall_seconds = maybe_float(manifest.get("min_wall_seconds"))
    latest_elapsed = maybe_float(history.get("latest_elapsed_seconds"))
    final_elapsed = maybe_float(payload.get("elapsed_seconds")) if payload else None
    elapsed_for_floor = final_elapsed if final_elapsed is not None else latest_elapsed
    formal_floor_remaining = None
    if configured_min_wall_seconds is not None and elapsed_for_floor is not None:
        formal_floor_remaining = max(0.0, configured_min_wall_seconds - elapsed_for_floor)
    selected_mse = maybe_float(final_eval.get("selected_action_mse"))
    dp_mse = maybe_float(final_eval.get("dp_prior_action_mse"))
    try:
        num_candidate_sources = int(final_eval.get("num_candidate_sources"))
    except (TypeError, ValueError):
        num_candidate_sources = 0
    try:
        candidate_samples = int(payload.get("candidate_samples"))
    except (TypeError, ValueError):
        candidate_samples = 0
    try:
        rank_diffusion_count = int(payload.get("candidate_rank_diffusion_count"))
    except (TypeError, ValueError):
        rank_diffusion_count = 0
    diffusion_interface_ready = bool(
        summary_path.is_file()
        and checkpoint_path.is_file()
        and payload.get("generator_type") == "diffusion"
        and candidate_samples > 0
        and rank_diffusion_count > 0
        and selected_mse is not None
        and dp_mse is not None
        and math.isfinite(selected_mse)
        and math.isfinite(dp_mse)
        and num_candidate_sources > 2
    )
    return {
        "root": str(root),
        "run_manifest": file_info(manifest_path),
        "summary": file_info(summary_path),
        "summary_output_root": summary_output_root,
        "summary_output_root_matches": summary_output_root_matches,
        "checkpoint_final": file_info(checkpoint_path),
        "formal_live_eval_checkpoint": str(live_eval_checkpoint_path) if live_eval_checkpoint_path else None,
        "formal_live_eval_checkpoint_file": file_info(live_eval_checkpoint_path) if live_eval_checkpoint_path else None,
        "post_gate": {
            "file": file_info(post_gate_path),
            "status": post_gate.get("status"),
            "formal_gate_ok_seen": post_gate.get("formal_gate_ok_seen"),
            "refusal_reasons": post_gate.get("refusal_reasons"),
            "live_rc": post_gate.get("live_rc"),
            "live_output_root": post_gate.get("live_output_root"),
            "live_summary_exists": post_gate.get("live_summary_exists"),
            "live_final_success_count": post_gate.get("live_final_success_count"),
            "live_requested_samples": post_gate.get("live_requested_samples"),
            "live_completed_samples": post_gate.get("live_completed_samples"),
            "live_failed_process_count": post_gate.get("live_failed_process_count"),
            "live_panel_contract_ok": post_gate.get("live_panel_contract_ok"),
            "live_contact_sheet": post_gate.get("live_contact_sheet"),
            "live_contact_sheet_ok": post_gate.get("live_contact_sheet_ok"),
            "live_visual_review_status": post_gate.get("live_visual_review_status"),
        },
        "status": (
            "summary_available"
            if summary_path.is_file() and checkpoint_path.is_file()
            else "waiting_training_final_artifacts"
        ),
        "formal_training_floor_met": payload.get("formal_training_floor_met"),
        "ready_for_offline_gate": payload.get("ready_for_offline_gate"),
        "best_gate_ready_for_offline_gate": payload.get("best_gate_ready_for_offline_gate"),
        "ready_for_formal_live_eval": payload.get("ready_for_formal_live_eval"),
        "formal_live_eval_metrics_source": payload.get("formal_live_eval_metrics_source"),
        "diffusion_interface_ready": diffusion_interface_ready,
        "feature_dim": payload.get("feature_dim"),
        "target_dim": payload.get("target_dim"),
        "action_horizon": payload.get("action_horizon"),
        "generator_type": payload.get("generator_type"),
        "diffusion_steps": payload.get("diffusion_steps"),
        "candidate_samples": payload.get("candidate_samples"),
        "candidate_scales": payload.get("candidate_scales"),
        "candidate_rank_loss_weight": payload.get("candidate_rank_loss_weight"),
        "candidate_rank_random_count": payload.get("candidate_rank_random_count"),
        "candidate_rank_diffusion_count": payload.get("candidate_rank_diffusion_count"),
        "offline_gate_thresholds": payload.get("offline_gate_thresholds"),
        "configured_min_wall_seconds": configured_min_wall_seconds,
        "formal_floor_remaining_seconds": formal_floor_remaining,
        "final_selected_action_mse": selected_mse,
        "final_dp_prior_action_mse": dp_mse,
        "final_selected_minus_dp_prior_mse": final_eval.get("selected_minus_dp_prior_mse"),
        "final_teacher_progress_mse": final_eval.get("teacher_progress_mse"),
        "final_teacher_value_mse": final_eval.get("teacher_value_mse"),
        "final_teacher_inserted_acc": final_eval.get("teacher_inserted_acc"),
        "final_teacher_dp_continuable_acc": final_eval.get("teacher_dp_continuable_acc"),
        "final_candidate_source_counts": final_sources,
        "final_selected_non_dp_candidate_count": final_eval.get(
            "selected_non_dp_candidate_count", final_source_summary["non_dp"]
        ),
        "final_selected_non_dp_candidate_fraction": final_eval.get(
            "selected_non_dp_candidate_fraction", final_source_summary["non_dp_fraction"]
        ),
        "final_selected_diffusion_candidate_count": final_eval.get(
            "selected_diffusion_candidate_count", final_source_summary["diffusion"]
        ),
        "training_history": history,
    }


def summarize_watch_log(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    lines = path.read_text(errors="replace").splitlines()
    return {
        "path": str(path),
        "exists": True,
        "line_count": len(lines),
        "tail": lines[-12:],
        "mentions_launching_diffusion_smoke": any("launching_diffusion_smoke" in line for line in lines),
        "mentions_launching_formal_diffusion": any("launching_formal_diffusion" in line for line in lines),
        "mentions_launching_live": any("launching_formal_diffusion_gated_live_panel" in line for line in lines),
    }


def summarize_prelaunch_audit(root: Path) -> dict[str, Any]:
    path = root / "prelaunch_audit.json"
    payload = maybe_json(path)
    if not payload:
        return {"path": str(path), "exists": path.is_file()}
    training_input = payload.get("training_input") if isinstance(payload.get("training_input"), dict) else {}
    feature_contract = (
        training_input.get("feature_contract") if isinstance(training_input.get("feature_contract"), dict) else {}
    )
    return {
        "path": str(path),
        "exists": True,
        "ready_for_allocation_launch": payload.get("ready_for_allocation_launch"),
        "blocker_class": payload.get("blocker_class"),
        "training_input_ok": payload.get("training_input_ok"),
        "static_dependencies_ok": payload.get("static_dependencies_ok"),
        "planned_formal_config_ok": payload.get("planned_formal_config_ok"),
        "rows": training_input.get("rows"),
        "task_path_sources": training_input.get("task_path_sources"),
        "feature_contract_ok": feature_contract.get("ok"),
        "actual_feature_dim": feature_contract.get("actual_feature_dim"),
        "expected_feature_dim": feature_contract.get("expected_feature_dim"),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def write_md(path: Path, payload: dict[str, Any]) -> None:
    current = payload["current_formal"]
    smoke = payload["diffusion_smoke"]
    formal = payload["formal_diffusion"]
    prelaunch = payload["prelaunch_audit"]
    lines = [
        "# Candidate Executor Diffusion Chain Status",
        "",
        f"- timestamp: `{payload['timestamp']}`",
        f"- overall_status: `{payload['overall_status']}`",
        f"- prelaunch_audit_ready: `{prelaunch.get('ready_for_allocation_launch')}`",
        f"- prelaunch_blocker_class: `{prelaunch.get('blocker_class')}`",
        f"- prelaunch_rows: `{prelaunch.get('rows')}`",
        f"- prelaunch_feature_contract_ok: `{prelaunch.get('feature_contract_ok')}`",
        f"- current_formal_status: `{current['status']}`",
        f"- current_formal_summary_root_matches: `{current.get('summary_output_root_matches')}`",
        f"- current_post_gate_status: `{current['post_gate'].get('status')}`",
        f"- current_post_gate_live_rc: `{current['post_gate'].get('live_rc')}`",
        f"- current_post_gate_live_success_count: `{current['post_gate'].get('live_final_success_count')}`",
        f"- current_post_gate_live_contract_ok: `{current['post_gate'].get('live_panel_contract_ok')}`",
        f"- current_post_gate_live_contact_sheet: `{current['post_gate'].get('live_contact_sheet')}`",
        f"- current_formal_ready_for_live: `{current.get('ready_for_formal_live_eval')}`",
        f"- current_formal_live_eval_checkpoint: `{current.get('formal_live_eval_checkpoint')}`",
        f"- current_formal_live_eval_metrics_source: `{current.get('formal_live_eval_metrics_source')}`",
        f"- current_best_gate_ready_for_offline_gate: `{current.get('best_gate_ready_for_offline_gate')}`",
        f"- current_formal_feature_dim: `{current.get('feature_dim')}`",
        f"- current_formal_target_dim: `{current.get('target_dim')}`",
        f"- current_formal_action_horizon: `{current.get('action_horizon')}`",
        f"- current_formal_selected_mse: `{current.get('final_selected_action_mse')}`",
        f"- current_formal_dp_mse: `{current.get('final_dp_prior_action_mse')}`",
        f"- current_formal_teacher_progress_mse: `{current.get('final_teacher_progress_mse')}`",
        f"- current_formal_teacher_value_mse: `{current.get('final_teacher_value_mse')}`",
        f"- current_formal_teacher_inserted_acc: `{current.get('final_teacher_inserted_acc')}`",
        f"- current_formal_teacher_dp_continuable_acc: `{current.get('final_teacher_dp_continuable_acc')}`",
        f"- current_formal_gate_thresholds: `{current.get('offline_gate_thresholds')}`",
        f"- current_formal_latest_step: `{current['training_history'].get('latest_step')}`",
        f"- current_formal_floor_remaining_seconds: `{current.get('formal_floor_remaining_seconds')}`",
        f"- current_formal_latest_selected_mse: `{current['training_history'].get('latest_selected_action_mse')}`",
        f"- current_formal_latest_dp_mse: `{current['training_history'].get('latest_dp_prior_action_mse')}`",
        f"- current_formal_latest_teacher_progress_mse: `{current['training_history'].get('latest_teacher_progress_mse')}`",
        f"- current_formal_latest_teacher_value_mse: `{current['training_history'].get('latest_teacher_value_mse')}`",
        f"- current_formal_latest_teacher_inserted_acc: `{current['training_history'].get('latest_teacher_inserted_acc')}`",
        f"- current_formal_latest_teacher_dp_continuable_acc: `{current['training_history'].get('latest_teacher_dp_continuable_acc')}`",
        f"- current_formal_latest_source_counts: `{current['training_history'].get('latest_candidate_source_counts')}`",
        f"- current_formal_latest_non_dp_selected: `{current['training_history'].get('latest_selected_non_dp_candidate_count')}`",
        f"- current_recent_worse_than_dp: `{current['training_history'].get('recent_window_selected_worse_than_dp_count')}/{current['training_history'].get('recent_window_records')}`",
        f"- diffusion_smoke_status: `{smoke['status']}`",
        f"- diffusion_smoke_summary_root_matches: `{smoke.get('summary_output_root_matches')}`",
        f"- diffusion_smoke_interface_ready: `{smoke.get('diffusion_interface_ready')}`",
        f"- diffusion_smoke_ready_for_offline_gate: `{smoke.get('ready_for_offline_gate')}`",
        f"- diffusion_smoke_latest_step: `{smoke['training_history'].get('latest_step')}`",
        f"- formal_diffusion_status: `{formal['status']}`",
        f"- formal_diffusion_ready_for_live: `{formal.get('ready_for_formal_live_eval')}`",
        f"- formal_diffusion_live_eval_checkpoint: `{formal.get('formal_live_eval_checkpoint')}`",
        f"- formal_diffusion_live_eval_metrics_source: `{formal.get('formal_live_eval_metrics_source')}`",
        f"- formal_diffusion_best_gate_ready_for_offline_gate: `{formal.get('best_gate_ready_for_offline_gate')}`",
        f"- formal_diffusion_teacher_progress_mse: `{formal.get('final_teacher_progress_mse')}`",
        f"- formal_diffusion_teacher_value_mse: `{formal.get('final_teacher_value_mse')}`",
        f"- formal_diffusion_teacher_inserted_acc: `{formal.get('final_teacher_inserted_acc')}`",
        f"- formal_diffusion_teacher_dp_continuable_acc: `{formal.get('final_teacher_dp_continuable_acc')}`",
        f"- formal_diffusion_gate_thresholds: `{formal.get('offline_gate_thresholds')}`",
        f"- formal_diffusion_latest_step: `{formal['training_history'].get('latest_step')}`",
        f"- watch_log: `{payload['watch_log']['path']}`",
        "",
        "Boundary: this is a chain-status summary only. Method evidence still "
        "requires final closed-loop metrics plus inspected video/contact sheet.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text("\n".join(lines))
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    formal_root = Path(args.formal_root).resolve()
    smoke_root = Path(args.diffusion_smoke_root).resolve()
    formal_diffusion_root = Path(args.formal_diffusion_root).resolve()
    watch_log = Path(args.watch_log).resolve() if args.watch_log else formal_root / "diffusion_smoke_after_gate_watch.log"
    output_json = Path(args.output_json).resolve() if args.output_json else formal_root / "diffusion_chain_status.json"
    output_md = Path(args.output_md).resolve() if args.output_md else formal_root / "diffusion_chain_status.md"

    current = summarize_training_root(formal_root)
    smoke = summarize_training_root(smoke_root)
    formal = summarize_training_root(formal_diffusion_root)
    prelaunch = summarize_prelaunch_audit(formal_diffusion_root)
    direct_diffusion_chain = formal_root == formal_diffusion_root
    if prelaunch.get("exists") and prelaunch.get("ready_for_allocation_launch") is False:
        overall = "prelaunch_audit_failed"
    elif direct_diffusion_chain:
        if smoke["status"] == "waiting_training_final_artifacts":
            overall = "waiting_diffusion_smoke"
        elif smoke.get("diffusion_interface_ready") is not True:
            overall = "diffusion_smoke_interface_failed"
        elif formal["status"] == "waiting_training_final_artifacts":
            overall = "diffusion_smoke_ready_waiting_formal_diffusion"
        elif formal.get("ready_for_formal_live_eval") is not True:
            overall = "formal_diffusion_gate_failed"
        else:
            overall = "formal_diffusion_gate_passed_waiting_or_running_live"
    elif current.get("ready_for_formal_live_eval") is True:
        overall = "current_formal_gate_passed_waiting_or_running_live"
    elif current["status"] == "waiting_training_final_artifacts":
        overall = "waiting_current_formal_gate"
    elif current.get("ready_for_formal_live_eval") is not True and smoke["status"] == "waiting_training_final_artifacts":
        overall = "current_formal_gate_failed_waiting_diffusion_smoke"
    elif smoke.get("diffusion_interface_ready") is not True:
        overall = "diffusion_smoke_interface_failed"
    elif formal["status"] == "waiting_training_final_artifacts":
        overall = "diffusion_smoke_ready_waiting_formal_diffusion"
    elif formal.get("ready_for_formal_live_eval") is not True:
        overall = "formal_diffusion_gate_failed"
    else:
        overall = "formal_diffusion_gate_passed_waiting_or_running_live"

    payload = {
        "schema": "candidate_executor_diffusion_chain_status_v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "overall_status": overall,
        "chain_mode": "direct_diffusion" if direct_diffusion_chain else "after_current_formal_gate",
        "prelaunch_audit": prelaunch,
        "current_formal": current,
        "diffusion_smoke": smoke,
        "formal_diffusion": formal,
        "watch_log": summarize_watch_log(watch_log),
    }
    write_json(output_json, payload)
    write_md(output_md, payload)
    print(json.dumps({"overall_status": overall, "output_json": str(output_json)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
