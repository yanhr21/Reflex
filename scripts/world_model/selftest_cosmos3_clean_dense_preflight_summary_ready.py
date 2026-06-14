#!/usr/bin/env python3
"""Self-test for clean/dense preflight SFT-entry validator."""

from __future__ import annotations

from pathlib import Path
import tempfile

from check_cosmos3_clean_dense_preflight_summary_ready import validate_summary


def assert_ok(summary: dict, condition_root: Path) -> None:
    verdict = validate_summary(summary, condition_root=condition_root)
    if not verdict["ok"]:
        raise AssertionError(verdict)


def assert_reason(summary: dict, condition_root: Path, reason: str) -> None:
    verdict = validate_summary(summary, condition_root=condition_root)
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

    print("cosmos3_clean_dense_preflight_summary_ready_selftest=passed")


if __name__ == "__main__":
    main()
