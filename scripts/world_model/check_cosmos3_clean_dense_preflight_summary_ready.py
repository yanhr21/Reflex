#!/usr/bin/env python3
"""Validate that a clean/dense preflight summary may unlock SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LIVE_QUERY_COVERAGE_FAILED_CHECKS = {
    "live_query_coverage_undercovered_count",
    "live_query_coverage_undercovered_fraction",
}

REQUIRED_OVERRIDE_SAFE_CHECKS = {
    "condition_manifest_exists",
    "full_episode_preflight_exists",
    "receding_distribution_audit_exists",
    "live_query_coverage_audit_exists",
    "role_weighted_manifest_exists",
    "source_episode_count",
    "episode_length_contract",
    "prefix_role_source",
    "dense_receding_enabled",
    "manifest_role_mode_clean",
    "strict_full_episode_preflight",
    "receding_audit_strict_ok",
    "receding_role_mode_clean",
    "late_rebind_coverage",
    "weighted_jsonl_expands_or_preserves_rows",
    "late_rebind_weight_recorded",
    "live_query_coverage_condition_root_matches",
    "live_query_coverage_has_queries",
    "live_query_coverage_role_mode_clean",
}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"summary is not a JSON object: {path}")
    return payload


def failed_check_names(summary: dict[str, Any]) -> list[str]:
    names: list[str] = []
    failed_checks = summary.get("failed_checks") or []
    if not isinstance(failed_checks, list):
        return ["failed_checks_not_list"]
    for item in failed_checks:
        if isinstance(item, dict):
            names.append(str(item.get("name", "")))
        else:
            names.append(str(item))
    return names


def check_map(summary: dict[str, Any]) -> dict[str, bool]:
    checks = summary.get("checks") or []
    result: dict[str, bool] = {}
    if not isinstance(checks, list):
        return result
    for item in checks:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if name is None:
            continue
        result[str(name)] = bool(item.get("ok", False))
    return result


def validate_summary(
    summary: dict[str, Any],
    *,
    condition_root: Path,
    allow_user_overridden_live_query_coverage_gap: bool = False,
    override_note: str | None = None,
) -> dict[str, Any]:
    summary_condition_root = Path(str(summary.get("condition_root", ""))).resolve()
    reasons: list[str] = []
    failed_names = failed_check_names(summary)
    failed_checks = summary.get("failed_checks") or []
    coverage_gap_only = (
        bool(failed_names)
        and set(failed_names).issubset(LIVE_QUERY_COVERAGE_FAILED_CHECKS)
    )
    user_override_applied = False
    missing_override_safe_checks: list[str] = []
    failed_override_safe_checks: list[str] = []

    if allow_user_overridden_live_query_coverage_gap and coverage_gap_only:
        checks = check_map(summary)
        missing_override_safe_checks = sorted(REQUIRED_OVERRIDE_SAFE_CHECKS - set(checks))
        failed_override_safe_checks = sorted(
            name for name in REQUIRED_OVERRIDE_SAFE_CHECKS if name in checks and not checks[name]
        )
        if missing_override_safe_checks:
            reasons.append("user_override_missing_required_safe_checks")
        if failed_override_safe_checks:
            reasons.append("user_override_required_safe_checks_failed")
        user_override_applied = not missing_override_safe_checks and not failed_override_safe_checks

    if not bool(summary.get("ready_for_overfit", False)) and not user_override_applied:
        reasons.append("clean_dense_preflight_summary_not_ready_for_overfit")
    if failed_checks and not user_override_applied:
        reasons.append("clean_dense_preflight_summary_has_failed_checks")
    diagnostic_reason = summary.get("diagnostic_not_ready_reason")
    if diagnostic_reason:
        reasons.append("clean_dense_preflight_summary_is_diagnostic_not_ready")
    if summary_condition_root != condition_root.resolve():
        reasons.append("clean_dense_preflight_summary_condition_root_mismatch")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "summary_condition_root": str(summary_condition_root),
        "requested_condition_root": str(condition_root.resolve()),
        "ready_for_overfit": bool(summary.get("ready_for_overfit", False)),
        "failed_check_count": len(failed_checks) if isinstance(failed_checks, list) else None,
        "failed_check_names": failed_names,
        "diagnostic_not_ready_reason": diagnostic_reason,
        "allow_user_overridden_live_query_coverage_gap": allow_user_overridden_live_query_coverage_gap,
        "user_override_applied": user_override_applied,
        "user_override_note": override_note,
        "missing_override_safe_checks": missing_override_safe_checks,
        "failed_override_safe_checks": failed_override_safe_checks,
        "boundary": (
            "This is an SFT-entry readiness validator. It accepts only a "
            "non-diagnostic clean/dense preflight summary with ready_for_overfit=true, "
            "zero failed checks, and a matching condition root, except for the "
            "explicit 2026-06-14 user override that allows only live-query "
            "coverage undercoverage after structural checks pass."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--allow-user-overridden-live-query-coverage-gap", action="store_true")
    parser.add_argument("--override-note", type=str, default=None)
    args = parser.parse_args()
    verdict = validate_summary(
        load_json(args.summary_json.resolve()),
        condition_root=args.condition_root.resolve(),
        allow_user_overridden_live_query_coverage_gap=args.allow_user_overridden_live_query_coverage_gap,
        override_note=args.override_note,
    )
    text = json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text)
    print(text, end="")
    if not verdict["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
