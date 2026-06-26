#!/usr/bin/env python3
"""Record candidate-executor post-gate/live status from existing artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--watch-log", default="")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Candidate Executor Post-Gate Status",
        "",
        f"- status: `{payload.get('status')}`",
        f"- formal_root: `{payload.get('formal_root')}`",
        f"- summary_exists: `{payload.get('summary_exists')}`",
        f"- checkpoint_final_exists: `{payload.get('checkpoint_final_exists')}`",
        f"- formal_live_eval_checkpoint: `{payload.get('formal_live_eval_checkpoint')}`",
        f"- formal_live_eval_checkpoint_exists: `{payload.get('formal_live_eval_checkpoint_exists')}`",
        f"- formal_live_eval_metrics_source: `{payload.get('formal_live_eval_metrics_source')}`",
        f"- formal_training_floor_met: `{payload.get('formal_training_floor_met')}`",
        f"- ready_for_formal_live_eval: `{payload.get('ready_for_formal_live_eval')}`",
        f"- final_selected_action_mse: `{payload.get('final_selected_action_mse')}`",
        f"- final_dp_prior_action_mse: `{payload.get('final_dp_prior_action_mse')}`",
        f"- final_teacher_progress_mse: `{payload.get('final_teacher_progress_mse')}`",
        f"- final_teacher_value_mse: `{payload.get('final_teacher_value_mse')}`",
        f"- final_teacher_inserted_acc: `{payload.get('final_teacher_inserted_acc')}`",
        f"- final_teacher_dp_continuable_acc: `{payload.get('final_teacher_dp_continuable_acc')}`",
        f"- offline_gate_thresholds: `{payload.get('offline_gate_thresholds')}`",
        f"- metric_gate_ok_seen: `{payload.get('metric_gate_ok_seen')}`",
        f"- gate_metrics: `{payload.get('gate_metrics')}`",
        f"- gate_thresholds: `{payload.get('gate_thresholds')}`",
        f"- watch_log: `{payload.get('watch_log')}`",
        f"- live_rc: `{payload.get('live_rc')}`",
        f"- live_output_root: `{payload.get('live_output_root')}`",
        f"- live_summary_exists: `{payload.get('live_summary_exists')}`",
        f"- live_final_success_count: `{payload.get('live_final_success_count')}`",
        f"- live_panel_contract_ok: `{payload.get('live_panel_contract_ok')}`",
        f"- live_contact_sheet: `{payload.get('live_contact_sheet')}`",
        "",
        "Boundary: this file only records artifacts. It is not a success claim "
        "without direct video/contact-sheet inspection.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text("\n".join(lines))
    tmp.replace(path)


def parse_watch_log(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"watch_log_exists": False}
    text = path.read_text(errors="replace")
    output_roots = re.findall(r"^output_root=(.+)$", text, flags=re.MULTILINE)
    rc_matches = re.findall(r"(?:candidate_after_gate_live_rc|formal_diffusion_gated_live_rc)=([0-9]+)", text)
    reasons = re.findall(r"^reason=(.+)$", text, flags=re.MULTILINE)
    gate_metrics_matches = re.findall(r"^gate_metrics=(\{.+\})$", text, flags=re.MULTILINE)
    gate_threshold_matches = re.findall(r"^gate_thresholds=(\{.+\})$", text, flags=re.MULTILINE)
    gate_metrics = None
    gate_thresholds = None
    if gate_metrics_matches:
        try:
            gate_metrics = json.loads(gate_metrics_matches[-1])
        except json.JSONDecodeError:
            gate_metrics = {"parse_error": gate_metrics_matches[-1]}
    if gate_threshold_matches:
        try:
            gate_thresholds = json.loads(gate_threshold_matches[-1])
        except json.JSONDecodeError:
            gate_thresholds = {"parse_error": gate_threshold_matches[-1]}
    return {
        "watch_log_exists": True,
        "live_output_root": output_roots[-1].strip() if output_roots else None,
        "live_rc": int(rc_matches[-1]) if rc_matches else None,
        "refusal_reasons": reasons,
        "formal_gate_ok_seen": "candidate_executor_formal_gate_ok=true" in text,
        "metric_gate_ok_seen": "candidate_executor_metric_gate_ok=true" in text,
        "gate_metrics": gate_metrics,
        "gate_thresholds": gate_thresholds,
    }


def summarize_live(output_root: str | None) -> dict[str, Any]:
    if not output_root:
        return {"live_summary_exists": False}
    root = Path(output_root)
    summary_path = root / "live_receding_panel_summary.json"
    out: dict[str, Any] = {
        "live_output_root": str(root),
        "live_summary_path": str(summary_path),
        "live_summary_exists": summary_path.is_file(),
    }
    if not summary_path.is_file():
        return out
    data = read_json(summary_path)
    contact = data.get("contact_sheet") if isinstance(data.get("contact_sheet"), dict) else {}
    out.update(
        {
            "live_final_success_count": data.get("final_success_count"),
            "live_requested_samples": data.get("requested_samples", data.get("num_samples")),
            "live_completed_samples": data.get("completed_samples"),
            "live_panel_contract_ok": data.get("panel_full_episode_contract_ok"),
            "live_failed_process_count": data.get("failed_process_count"),
            "live_contact_sheet": contact.get("contact_sheet"),
            "live_contact_sheet_ok": contact.get("ok"),
            "live_visual_review_status": data.get("visual_review_status"),
            "live_method_evidence_allowed": data.get("method_evidence_allowed"),
            "live_summary_boundary": data.get("boundary"),
        }
    )
    return out


def build_payload(formal_root: Path, watch_log: Path) -> dict[str, Any]:
    summary_path = formal_root / "training_summary.json"
    checkpoint_path = formal_root / "checkpoint_final.pt"
    live_eval_checkpoint = None
    payload: dict[str, Any] = {
        "formal_root": str(formal_root),
        "training_summary": str(summary_path),
        "checkpoint_final": str(checkpoint_path),
        "summary_exists": summary_path.is_file(),
        "checkpoint_final_exists": checkpoint_path.is_file(),
        "watch_log": str(watch_log),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if summary_path.is_file():
        summary = read_json(summary_path)
        live_eval_checkpoint = summary.get("formal_live_eval_checkpoint")
        live_eval_checkpoint_path = Path(str(live_eval_checkpoint)) if live_eval_checkpoint else None
        final_eval = (summary.get("final_metrics") or {}).get("eval") or {}
        payload.update(
            {
                "formal_training_floor_met": summary.get("formal_training_floor_met"),
                "ready_for_formal_live_eval": summary.get("ready_for_formal_live_eval"),
                "ready_for_offline_gate": summary.get("ready_for_offline_gate"),
                "best_gate_ready_for_offline_gate": summary.get("best_gate_ready_for_offline_gate"),
                "formal_live_eval_checkpoint": live_eval_checkpoint,
                "formal_live_eval_checkpoint_exists": live_eval_checkpoint_path.is_file()
                if live_eval_checkpoint_path
                else False,
                "formal_live_eval_metrics_source": summary.get("formal_live_eval_metrics_source"),
                "training_steps": summary.get("steps"),
                "training_elapsed_seconds": summary.get("elapsed_seconds"),
                "training_stop_reason": summary.get("stop_reason"),
                "final_selected_action_mse": final_eval.get("selected_action_mse"),
                "final_dp_prior_action_mse": final_eval.get("dp_prior_action_mse"),
                "final_selected_minus_dp_prior_mse": final_eval.get("selected_minus_dp_prior_mse"),
                "final_teacher_progress_mse": final_eval.get("teacher_progress_mse"),
                "final_teacher_value_mse": final_eval.get("teacher_value_mse"),
                "final_teacher_inserted_acc": final_eval.get("teacher_inserted_acc"),
                "final_teacher_dp_continuable_acc": final_eval.get("teacher_dp_continuable_acc"),
                "offline_gate_thresholds": summary.get("offline_gate_thresholds"),
                "final_candidate_source_counts": final_eval.get("candidate_source_counts"),
                "final_candidate_scales": final_eval.get("candidate_scales"),
                "final_num_candidate_sources": final_eval.get("num_candidate_sources"),
            }
        )
    watch = parse_watch_log(watch_log)
    payload.update(watch)
    payload.update(summarize_live(watch.get("live_output_root")))
    if not payload["summary_exists"] or not payload["checkpoint_final_exists"]:
        payload["status"] = "waiting_training_final_artifacts"
    elif payload.get("ready_for_formal_live_eval") is not True:
        payload["status"] = "formal_gate_failed_live_not_allowed"
    elif payload.get("live_rc") is None:
        payload["status"] = "formal_gate_passed_waiting_live_completion"
    elif payload.get("live_summary_exists"):
        payload["status"] = "live_panel_summary_available_needs_video_review"
    else:
        payload["status"] = "live_finished_without_panel_summary"
    return payload


def main() -> int:
    args = parse_args()
    formal_root = Path(args.formal_root).resolve()
    watch_log = Path(args.watch_log).resolve() if args.watch_log else formal_root / "candidate_after_gate_watch_envsafe.log"
    output_json = Path(args.output_json).resolve() if args.output_json else formal_root / "post_gate_status.json"
    output_md = Path(args.output_md).resolve() if args.output_md else formal_root / "post_gate_status.md"
    while True:
        payload = build_payload(formal_root, watch_log)
        write_json(output_json, payload)
        write_md(output_md, payload)
        status = str(payload.get("status"))
        print(json.dumps({"status": status, "output_json": str(output_json)}, sort_keys=True), flush=True)
        if status in {
            "formal_gate_failed_live_not_allowed",
            "live_panel_summary_available_needs_video_review",
            "live_finished_without_panel_summary",
        }:
            return 0
        time.sleep(max(5, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
