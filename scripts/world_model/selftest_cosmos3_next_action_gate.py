#!/usr/bin/env python3
"""Lightweight self-test for Cosmos3 next-action gate logic."""

from __future__ import annotations

from check_cosmos3_next_action_gate import check_gate


def make_status(*, active: bool = False, blocked: bool = True, safe_resume: bool = False, safe_panel: bool = False) -> dict:
    return {
        "generated_at": "selftest",
        "status_flags": {
            "training_or_eval_process_active": active,
            "latest_generated_eval_controller_blocked": blocked,
            "safe_to_resume_current_condition_sft_without_user_approval": safe_resume,
            "safe_to_launch_broad_panel_from_current_checkpoint": safe_panel,
        },
        "training": {"latest_checkpoint": {"name": "iter_000002700"}},
        "generated_eval": {
            "latest_eval": {
                "name": "eval_full_episode_wam_iter_000002700",
                "closed_loop_allowed": False,
            }
        },
        "processes": {"active_matching_process_count": 1 if active else 0},
    }


def requirement_audit(*, achieved: bool) -> dict:
    return {
        "current_goal_achieved": achieved,
        "status_counts": {"passed": 8} if achieved else {"passed": 6, "partial": 1, "failed": 1},
        "requirements": []
        if achieved
        else [
            {"id": "dp_handoff_available_but_not_proven_reliable", "status": "partial"},
            {"id": "method_effectiveness_against_pure_dp", "status": "failed"},
        ],
    }


def assert_gate(
    status: dict,
    action: str,
    user_approved: bool,
    expected_allowed: bool,
    audit: dict | None = None,
    preflight_summary: dict | None = None,
) -> None:
    verdict = check_gate(status, action, user_approved, audit, preflight_summary)
    if verdict["allowed"] is not expected_allowed:
        raise AssertionError(
            f"{action} user_approved={user_approved} expected {expected_allowed}, got {verdict}"
        )


def main() -> None:
    idle_blocked = make_status(active=False, blocked=True)
    assert_gate(idle_blocked, "resume_current_condition_sft", False, False)
    assert_gate(idle_blocked, "launch_broad_panel_current_checkpoint", False, False)
    assert_gate(idle_blocked, "clean_dense_preflight_after_user_approval", False, False)
    assert_gate(idle_blocked, "clean_dense_preflight_after_user_approval", True, True)
    assert_gate(idle_blocked, "clean_dense_overfit_sft_after_user_approval", True, False)
    assert_gate(
        idle_blocked,
        "clean_dense_overfit_sft_after_user_approval",
        True,
        False,
        preflight_summary={"ready_for_overfit": False, "failed_checks": [{"name": "example"}]},
    )
    assert_gate(
        idle_blocked,
        "clean_dense_overfit_sft_after_user_approval",
        False,
        False,
        preflight_summary={"ready_for_overfit": True, "failed_checks": []},
    )
    assert_gate(
        idle_blocked,
        "clean_dense_overfit_sft_after_user_approval",
        True,
        True,
        audit=requirement_audit(achieved=False),
        preflight_summary={"ready_for_overfit": True, "failed_checks": []},
    )
    assert_gate(
        idle_blocked,
        "clean_dense_overfit_sft_after_user_approval",
        True,
        False,
        preflight_summary={"ready_for_overfit": True, "failed_checks": [{"name": "bad"}]},
    )
    assert_gate(
        idle_blocked,
        "clean_dense_overfit_sft_after_user_approval",
        True,
        False,
        preflight_summary={
            "ready_for_overfit": True,
            "failed_checks": [],
            "diagnostic_not_ready_reason": "live_query_coverage_audit_skipped_by_diagnostic_override",
        },
    )

    active_blocked = make_status(active=True, blocked=True)
    assert_gate(active_blocked, "clean_dense_preflight_after_user_approval", True, False)
    assert_gate(
        active_blocked,
        "clean_dense_overfit_sft_after_user_approval",
        True,
        False,
        preflight_summary={"ready_for_overfit": True, "failed_checks": []},
    )

    status_claims_safe = make_status(active=False, blocked=False, safe_resume=True, safe_panel=True)
    incomplete_audit = requirement_audit(achieved=False)
    assert_gate(status_claims_safe, "resume_current_condition_sft", False, False, incomplete_audit)
    assert_gate(status_claims_safe, "launch_broad_panel_current_checkpoint", False, False, incomplete_audit)
    assert_gate(status_claims_safe, "clean_dense_preflight_after_user_approval", True, True, incomplete_audit)
    complete_audit = requirement_audit(achieved=True)
    assert_gate(status_claims_safe, "resume_current_condition_sft", False, True, complete_audit)
    assert_gate(status_claims_safe, "launch_broad_panel_current_checkpoint", False, True, complete_audit)

    print("cosmos3_next_action_gate_selftest=passed")


if __name__ == "__main__":
    main()
