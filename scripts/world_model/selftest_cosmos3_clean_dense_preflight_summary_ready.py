#!/usr/bin/env python3
"""Self-test for clean/dense preflight SFT-entry validator."""

from __future__ import annotations

from pathlib import Path
import tempfile

from check_cosmos3_clean_dense_preflight_summary_ready import validate_summary


def assert_ok(summary: dict, condition_root: Path, **kwargs) -> None:
    verdict = validate_summary(summary, condition_root=condition_root, **kwargs)
    if not verdict["ok"]:
        raise AssertionError(verdict)


def assert_reason(summary: dict, condition_root: Path, reason: str, **kwargs) -> None:
    verdict = validate_summary(summary, condition_root=condition_root, **kwargs)
    if verdict["ok"] or reason not in verdict["reasons"]:
        raise AssertionError(f"expected reason {reason}, got {verdict}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cosmos3_preflight_summary_ready_selftest_") as tmp_text:
        root = Path(tmp_text) / "condition_root"
        root.mkdir(parents=True)
        good = {
            "condition_root": str(root.resolve()),
            "ready_for_overfit": True,
            "failed_checks": [],
            "diagnostic_not_ready_reason": None,
        }
        assert_ok(good, root)

        failed = dict(good)
        failed["failed_checks"] = [{"name": "bad"}]
        assert_reason(failed, root, "clean_dense_preflight_summary_has_failed_checks")

        diagnostic = dict(good)
        diagnostic["diagnostic_not_ready_reason"] = "live_query_coverage_audit_skipped_by_diagnostic_override"
        assert_reason(diagnostic, root, "clean_dense_preflight_summary_is_diagnostic_not_ready")

        not_ready = dict(good)
        not_ready["ready_for_overfit"] = False
        assert_reason(not_ready, root, "clean_dense_preflight_summary_not_ready_for_overfit")

        wrong_root = dict(good)
        wrong_root["condition_root"] = str((Path(tmp_text) / "other").resolve())
        assert_reason(wrong_root, root, "clean_dense_preflight_summary_condition_root_mismatch")

        override_checks = [
            {"name": "condition_manifest_exists", "ok": True},
            {"name": "full_episode_preflight_exists", "ok": True},
            {"name": "receding_distribution_audit_exists", "ok": True},
            {"name": "live_query_coverage_audit_exists", "ok": True},
            {"name": "role_weighted_manifest_exists", "ok": True},
            {"name": "source_episode_count", "ok": True},
            {"name": "episode_length_contract", "ok": True},
            {"name": "prefix_role_source", "ok": True},
            {"name": "dense_receding_enabled", "ok": True},
            {"name": "manifest_role_mode_clean", "ok": True},
            {"name": "strict_full_episode_preflight", "ok": True},
            {"name": "receding_audit_strict_ok", "ok": True},
            {"name": "receding_role_mode_clean", "ok": True},
            {"name": "late_rebind_coverage", "ok": True},
            {"name": "weighted_jsonl_expands_or_preserves_rows", "ok": True},
            {"name": "late_rebind_weight_recorded", "ok": True},
            {"name": "live_query_coverage_condition_root_matches", "ok": True},
            {"name": "live_query_coverage_has_queries", "ok": True},
            {"name": "live_query_coverage_role_mode_clean", "ok": True},
        ]
        coverage_gap = dict(good)
        coverage_gap["ready_for_overfit"] = False
        coverage_gap["failed_checks"] = [
            {"name": "live_query_coverage_undercovered_count"},
            {"name": "live_query_coverage_undercovered_fraction"},
        ]
        coverage_gap["checks"] = override_checks + [
            {"name": "live_query_coverage_undercovered_count", "ok": False},
            {"name": "live_query_coverage_undercovered_fraction", "ok": False},
        ]
        assert_reason(coverage_gap, root, "clean_dense_preflight_summary_not_ready_for_overfit")
        assert_ok(
            coverage_gap,
            root,
            allow_user_overridden_live_query_coverage_gap=True,
            override_note="selftest",
        )

        unsafe_override = dict(coverage_gap)
        unsafe_override["checks"] = [dict(item) for item in coverage_gap["checks"]]
        for item in unsafe_override["checks"]:
            if item["name"] == "episode_length_contract":
                item["ok"] = False
        assert_reason(
            unsafe_override,
            root,
            "user_override_required_safe_checks_failed",
            allow_user_overridden_live_query_coverage_gap=True,
        )

        wrong_failure_override = dict(coverage_gap)
        wrong_failure_override["failed_checks"] = [{"name": "strict_full_episode_preflight"}]
        assert_reason(
            wrong_failure_override,
            root,
            "clean_dense_preflight_summary_not_ready_for_overfit",
            allow_user_overridden_live_query_coverage_gap=True,
        )

    print("cosmos3_clean_dense_preflight_summary_ready_selftest=passed")


if __name__ == "__main__":
    main()
