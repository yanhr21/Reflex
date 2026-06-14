#!/usr/bin/env python3
"""Summarize clean/dense Cosmos3 condition preflight readiness.

This is a read-only gate summary. It combines the condition manifest, strict
full-episode preflight, receding-distribution audit, and optional weighted
train JSONL manifest into a single ready_for_overfit flag.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-source-episodes", type=int, default=733)
    parser.add_argument("--expected-video-frames", type=int, default=301)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--expected-prefix-role-source", default="physical_mode")
    parser.add_argument("--min-late-rebind-candidates", type=int, default=1)
    parser.add_argument("--live-query-coverage-audit-json", type=Path, default=None)
    parser.add_argument("--max-live-query-undercovered-count", type=int, default=0)
    parser.add_argument("--max-live-query-undercovered-fraction", type=float, default=0.0)
    parser.add_argument(
        "--diagnostic-not-ready-reason",
        default="",
        help=(
            "If non-empty, force ready_for_overfit=false and record why this "
            "preflight is diagnostic-only rather than method/training evidence."
        ),
    )
    parser.add_argument("--require-weighted-train-jsonl", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing {path}"
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - diagnostic path
        return None, f"failed to read {path}: {exc}"
    if not isinstance(payload, dict):
        return None, f"not a JSON object: {path}"
    return payload, None


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: Any) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def main() -> None:
    args = parse_args()
    condition_root = args.condition_root.resolve()
    output_root = args.output_root.resolve()

    manifest, manifest_err = read_json(condition_root / "manifest.json")
    preflight, preflight_err = read_json(output_root / "full_episode_wam_preflight.json")
    receding, receding_err = read_json(output_root / "receding_training_distribution_audit.json")
    weighted, weighted_err = read_json(condition_root / "train" / "role_weighted_manifest.json")
    coverage: dict[str, Any] | None = None
    coverage_err: str | None = None
    if args.live_query_coverage_audit_json is not None:
        coverage, coverage_err = read_json(args.live_query_coverage_audit_json.resolve())

    checks: list[dict[str, Any]] = []
    add_check(checks, "condition_manifest_exists", manifest_err is None, manifest_err or str(condition_root / "manifest.json"))
    add_check(checks, "full_episode_preflight_exists", preflight_err is None, preflight_err or str(output_root / "full_episode_wam_preflight.json"))
    add_check(checks, "receding_distribution_audit_exists", receding_err is None, receding_err or str(output_root / "receding_training_distribution_audit.json"))
    if args.live_query_coverage_audit_json is not None:
        add_check(
            checks,
            "live_query_coverage_audit_exists",
            coverage_err is None,
            coverage_err or str(args.live_query_coverage_audit_json.resolve()),
        )
    if args.require_weighted_train_jsonl:
        add_check(checks, "role_weighted_manifest_exists", weighted_err is None, weighted_err or str(condition_root / "train" / "role_weighted_manifest.json"))
    if args.diagnostic_not_ready_reason.strip():
        add_check(
            checks,
            "diagnostic_not_method_ready",
            False,
            {"reason": args.diagnostic_not_ready_reason.strip()},
        )

    if manifest is not None:
        add_check(
            checks,
            "source_episode_count",
            int(manifest.get("num_source_episodes", -1)) == int(args.expected_source_episodes),
            {"actual": manifest.get("num_source_episodes"), "expected": args.expected_source_episodes},
        )
        add_check(
            checks,
            "episode_length_contract",
            int(manifest.get("num_video_frames", -1)) == int(args.expected_video_frames)
            and int(manifest.get("num_action_steps", -1)) == int(args.expected_action_steps),
            {
                "video_frames": manifest.get("num_video_frames"),
                "action_steps": manifest.get("num_action_steps"),
                "expected_video_frames": args.expected_video_frames,
                "expected_action_steps": args.expected_action_steps,
            },
        )
        add_check(
            checks,
            "prefix_role_source",
            str(manifest.get("prefix_role_source")) == str(args.expected_prefix_role_source),
            {"actual": manifest.get("prefix_role_source"), "expected": args.expected_prefix_role_source},
        )
        add_check(
            checks,
            "dense_receding_enabled",
            int(manifest.get("dense_receding_prefix_stride", 0)) > 0,
            {"dense_receding_prefix_stride": manifest.get("dense_receding_prefix_stride")},
        )
        add_check(
            checks,
            "manifest_role_mode_clean",
            int(manifest.get("prefix_role_mode_mismatch_count", -1)) == 0,
            {"prefix_role_mode_mismatch_count": manifest.get("prefix_role_mode_mismatch_count")},
        )

    if preflight is not None:
        add_check(
            checks,
            "strict_full_episode_preflight",
            bool(preflight.get("strict_alignment_ok", False)),
            {"strict_alignment_ok": preflight.get("strict_alignment_ok"), "failures": preflight.get("failures", [])},
        )

    if receding is not None:
        add_check(
            checks,
            "receding_audit_strict_ok",
            bool(receding.get("strict_ok", False)),
            {"strict_ok": receding.get("strict_ok"), "hard_failures": receding.get("hard_failures", [])},
        )
        add_check(
            checks,
            "receding_role_mode_clean",
            int(receding.get("role_mode_mismatch_count", -1)) == 0,
            {"role_mode_mismatch_count": receding.get("role_mode_mismatch_count")},
        )
        add_check(
            checks,
            "late_rebind_coverage",
            int(receding.get("late_rebind_candidate_total", -1)) >= int(args.min_late_rebind_candidates),
            {
                "late_rebind_candidate_total": receding.get("late_rebind_candidate_total"),
                "min_late_rebind_candidates": args.min_late_rebind_candidates,
            },
        )

    if weighted is not None:
        add_check(
            checks,
            "weighted_jsonl_expands_or_preserves_rows",
            int(weighted.get("num_output_rows", -1)) >= int(weighted.get("num_input_rows", 0)),
            {"num_input_rows": weighted.get("num_input_rows"), "num_output_rows": weighted.get("num_output_rows")},
        )
        add_check(
            checks,
            "late_rebind_weight_recorded",
            int(weighted.get("late_rebind_weight", 1)) >= 1,
            {"late_rebind_weight": weighted.get("late_rebind_weight")},
        )

    if coverage is not None:
        coverage_root = coverage.get("condition_root")
        try:
            coverage_root_matches = Path(str(coverage_root)).resolve() == condition_root
        except Exception:
            coverage_root_matches = False
        strict_cov = coverage.get("strict_local_coverage") if isinstance(coverage.get("strict_local_coverage"), dict) else {}
        undercovered_count = int(strict_cov.get("undercovered_query_count", -1))
        undercovered_fraction = float(strict_cov.get("undercovered_query_fraction", 1.0))
        live_query_count = int(coverage.get("live_cosmos_query_count", 0) or 0)
        role_mode_mismatch_count = int(coverage.get("train_role_mode_mismatch_count", -1))
        add_check(
            checks,
            "live_query_coverage_condition_root_matches",
            coverage_root_matches,
            {"coverage_condition_root": coverage_root, "expected_condition_root": str(condition_root)},
        )
        add_check(
            checks,
            "live_query_coverage_has_queries",
            live_query_count > 0,
            {"live_cosmos_query_count": live_query_count},
        )
        add_check(
            checks,
            "live_query_coverage_role_mode_clean",
            role_mode_mismatch_count == 0,
            {"train_role_mode_mismatch_count": role_mode_mismatch_count},
        )
        add_check(
            checks,
            "live_query_coverage_undercovered_count",
            undercovered_count <= int(args.max_live_query_undercovered_count),
            {
                "undercovered_query_count": undercovered_count,
                "max_live_query_undercovered_count": args.max_live_query_undercovered_count,
            },
        )
        add_check(
            checks,
            "live_query_coverage_undercovered_fraction",
            undercovered_fraction <= float(args.max_live_query_undercovered_fraction),
            {
                "undercovered_query_fraction": undercovered_fraction,
                "max_live_query_undercovered_fraction": args.max_live_query_undercovered_fraction,
            },
        )

    failed = [item for item in checks if not item["ok"]]
    report = {
        "condition_root": str(condition_root),
        "output_root": str(output_root),
        "live_query_coverage_audit_json": (
            str(args.live_query_coverage_audit_json.resolve())
            if args.live_query_coverage_audit_json is not None
            else None
        ),
        "diagnostic_not_ready_reason": args.diagnostic_not_ready_reason.strip() or None,
        "ready_for_overfit": not failed,
        "failed_checks": failed,
        "checks": checks,
        "boundary": (
            "This summary proves only that the clean/dense condition export is structurally ready "
            "for a user-approved overfit SFT. It is not training, generated-video, or controller evidence."
        ),
    }

    output_json = args.output_json or (output_root / "clean_dense_preflight_summary.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    output_md = args.output_md or (output_root / "clean_dense_preflight_summary.md")
    lines = [
        "# Clean Dense Preflight Summary",
        "",
        f"- condition root: `{condition_root}`",
        f"- output root: `{output_root}`",
        f"- ready_for_overfit: `{str(report['ready_for_overfit']).lower()}`",
        "",
        "## Checks",
    ]
    for item in checks:
        status = "PASS" if item["ok"] else "FAIL"
        lines.append(f"- {status}: `{item['name']}` - `{json.dumps(item['detail'], sort_keys=True)}`")
    output_md.write_text("\n".join(lines) + "\n")

    print(json.dumps({"ready_for_overfit": report["ready_for_overfit"], "failed_checks": failed}, sort_keys=True))
    if failed:
        raise SystemExit("clean dense preflight summary failed")


if __name__ == "__main__":
    main()
