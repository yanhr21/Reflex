#!/usr/bin/env python3
"""Validate that a clean/dense preflight summary may unlock SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"summary is not a JSON object: {path}")
    return payload


def validate_summary(summary: dict[str, Any], *, condition_root: Path) -> dict[str, Any]:
    summary_condition_root = Path(str(summary.get("condition_root", ""))).resolve()
    reasons: list[str] = []
    if not bool(summary.get("ready_for_overfit", False)):
        reasons.append("clean_dense_preflight_summary_not_ready_for_overfit")
    failed_checks = summary.get("failed_checks") or []
    if failed_checks:
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
        "diagnostic_not_ready_reason": diagnostic_reason,
        "boundary": (
            "This is an SFT-entry readiness validator. It accepts only a "
            "non-diagnostic clean/dense preflight summary with ready_for_overfit=true, "
            "zero failed checks, and a matching condition root."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    verdict = validate_summary(load_json(args.summary_json.resolve()), condition_root=args.condition_root.resolve())
    text = json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text)
    print(text, end="")
    if not verdict["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
