#!/usr/bin/env python3
"""Check whether a requested Cosmos3 next action is allowed by status JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ACTION_DESCRIPTIONS = {
    "resume_current_condition_sft": (
        "Resume SFT from the current condition root/checkpoint. This is "
        "disallowed after corrected live closed-loop failure unless a future "
        "status file explicitly marks it safe."
    ),
    "launch_broad_panel_current_checkpoint": (
        "Launch a broad closed-loop panel from the current checkpoint. This is "
        "disallowed while generated/live evidence is controller-blocked."
    ),
    "clean_dense_preflight_after_user_approval": (
        "Run the clean-role/dense-receding condition preflight only after user "
        "approval. This action must still use RUN_SFT=false and a compute-node "
        "allocation."
    ),
    "clean_dense_overfit_sft_after_user_approval": (
        "Run clean/dense two-sample overfit SFT only after preflight has "
        "ready_for_overfit=true and the user explicitly approves training."
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def check_gate(
    status: dict[str, Any],
    action: str,
    user_approved: bool,
    requirement_audit: dict[str, Any] | None = None,
    clean_dense_preflight_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    flags = status.get("status_flags", {})
    active = bool(flags.get("training_or_eval_process_active"))
    latest_eval_blocked = bool(flags.get("latest_generated_eval_controller_blocked"))
    safe_resume = bool(flags.get("safe_to_resume_current_condition_sft_without_user_approval"))
    safe_panel = bool(flags.get("safe_to_launch_broad_panel_from_current_checkpoint"))
    requirement_audit_present = isinstance(requirement_audit, dict)
    requirement_goal_achieved = bool((requirement_audit or {}).get("current_goal_achieved", False))
    requirement_status_counts = (requirement_audit or {}).get("status_counts") or {}
    requirement_failed_ids = [
        str(item.get("id"))
        for item in (requirement_audit or {}).get("requirements", [])
        if item.get("status") in {"failed", "partial", "missing"}
    ]
    clean_dense_preflight_present = isinstance(clean_dense_preflight_summary, dict)
    clean_dense_ready_for_overfit = bool(
        (clean_dense_preflight_summary or {}).get("ready_for_overfit", False)
    )
    clean_dense_failed_checks = (
        (clean_dense_preflight_summary or {}).get("failed_checks") or []
        if clean_dense_preflight_present
        else None
    )
    clean_dense_failed_check_count = (
        len(clean_dense_failed_checks)
        if isinstance(clean_dense_failed_checks, list)
        else None
    )
    clean_dense_diagnostic_not_ready_reason = (
        (clean_dense_preflight_summary or {}).get("diagnostic_not_ready_reason")
        if clean_dense_preflight_present
        else None
    )

    reasons: list[str] = []
    allowed = False

    if active:
        reasons.append("training_or_eval_process_active")

    if action == "resume_current_condition_sft":
        allowed = safe_resume and not active and (not requirement_audit_present or requirement_goal_achieved)
        if not safe_resume:
            reasons.append("current_condition_sft_marked_unsafe_without_user_approval")
        if latest_eval_blocked:
            reasons.append("latest_generated_eval_controller_blocked")
        if requirement_audit_present and not requirement_goal_achieved:
            reasons.append("closed_loop_requirement_audit_not_achieved")
    elif action == "launch_broad_panel_current_checkpoint":
        allowed = safe_panel and not active and (not requirement_audit_present or requirement_goal_achieved)
        if not safe_panel:
            reasons.append("broad_panel_marked_unsafe_from_current_checkpoint")
        if latest_eval_blocked:
            reasons.append("latest_generated_eval_controller_blocked")
        if requirement_audit_present and not requirement_goal_achieved:
            reasons.append("closed_loop_requirement_audit_not_achieved")
    elif action == "clean_dense_preflight_after_user_approval":
        allowed = user_approved and not active
        if not user_approved:
            reasons.append("missing_user_approval")
        if active:
            reasons.append("wait_for_active_process_to_stop")
    elif action == "clean_dense_overfit_sft_after_user_approval":
        allowed = (
            user_approved
            and not active
            and clean_dense_preflight_present
            and clean_dense_ready_for_overfit
            and clean_dense_failed_check_count == 0
            and not clean_dense_diagnostic_not_ready_reason
        )
        if not user_approved:
            reasons.append("missing_user_approval")
        if active:
            reasons.append("wait_for_active_process_to_stop")
        if not clean_dense_preflight_present:
            reasons.append("requires_clean_dense_preflight_summary_ready_for_overfit")
        elif not clean_dense_ready_for_overfit:
            reasons.append("clean_dense_preflight_summary_not_ready_for_overfit")
        elif clean_dense_failed_check_count != 0:
            reasons.append("clean_dense_preflight_summary_has_failed_checks")
        elif clean_dense_diagnostic_not_ready_reason:
            reasons.append("clean_dense_preflight_summary_is_diagnostic_not_ready")
    else:
        raise ValueError(f"unknown action: {action}")

    if allowed:
        reasons = []

    return {
        "action": action,
        "description": ACTION_DESCRIPTIONS[action],
        "allowed": allowed,
        "reasons": sorted(set(reasons)),
        "status_generated_at": status.get("generated_at"),
        "latest_checkpoint": (status.get("training", {}).get("latest_checkpoint") or {}).get("name"),
        "latest_eval": (status.get("generated_eval", {}).get("latest_eval") or {}).get("name"),
        "latest_eval_closed_loop_allowed": (
            status.get("generated_eval", {}).get("latest_eval") or {}
        ).get("closed_loop_allowed"),
        "active_matching_process_count": status.get("processes", {}).get("active_matching_process_count"),
        "requirement_audit_present": requirement_audit_present,
        "requirement_goal_achieved": requirement_goal_achieved if requirement_audit_present else None,
        "requirement_status_counts": requirement_status_counts if requirement_audit_present else None,
        "requirement_failed_or_partial_ids": requirement_failed_ids if requirement_audit_present else None,
        "clean_dense_preflight_summary_present": clean_dense_preflight_present,
        "clean_dense_ready_for_overfit": clean_dense_ready_for_overfit if clean_dense_preflight_present else None,
        "clean_dense_failed_checks": clean_dense_failed_checks,
        "clean_dense_failed_check_count": clean_dense_failed_check_count,
        "clean_dense_diagnostic_not_ready_reason": clean_dense_diagnostic_not_ready_reason,
        "boundary": (
            "This gate is a safety check over existing status artifacts only. "
            "It does not start Slurm jobs, training, rendering, inference, or eval."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--action", choices=sorted(ACTION_DESCRIPTIONS), required=True)
    parser.add_argument("--requirement-audit-json", type=Path, default=None)
    parser.add_argument("--clean-dense-preflight-summary-json", type=Path, default=None)
    parser.add_argument("--user-approved", action="store_true")
    parser.add_argument("--allow-nonpassing-exit-zero", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    requirement_audit = load_json(args.requirement_audit_json) if args.requirement_audit_json else None
    clean_dense_preflight_summary = (
        load_json(args.clean_dense_preflight_summary_json)
        if args.clean_dense_preflight_summary_json
        else None
    )
    verdict = check_gate(
        load_json(args.status_json),
        args.action,
        args.user_approved,
        requirement_audit,
        clean_dense_preflight_summary,
    )
    text = json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text)
    print(text, end="")
    if not verdict["allowed"] and not args.allow_nonpassing_exit_zero:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
