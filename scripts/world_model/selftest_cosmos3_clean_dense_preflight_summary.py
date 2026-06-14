#!/usr/bin/env python3
"""Self-test for clean/dense preflight summary readiness logic."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("summarize_cosmos3_clean_dense_preflight.py")


def write_json(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


def write_ready_inputs(tmp: Path, *, coverage_undercovered: int = 0) -> tuple[Path, Path, Path]:
    condition_root = tmp / "condition_root"
    output_root = tmp / "output_root"
    write_json(
        condition_root / "manifest.json",
        {
            "num_source_episodes": 2,
            "num_video_frames": 301,
            "num_action_steps": 300,
            "prefix_role_source": "physical_mode",
            "dense_receding_prefix_stride": 8,
            "prefix_role_mode_mismatch_count": 0,
        },
    )
    write_json(
        output_root / "full_episode_wam_preflight.json",
        {"strict_alignment_ok": True, "failures": []},
    )
    write_json(
        output_root / "receding_training_distribution_audit.json",
        {
            "strict_ok": True,
            "hard_failures": [],
            "role_mode_mismatch_count": 0,
            "late_rebind_candidate_total": 3,
        },
    )
    coverage = write_json(
        output_root / "live_query_training_coverage_audit.json",
        {
            "condition_root": str(condition_root.resolve()),
            "train_role_mode_mismatch_count": 0,
            "live_cosmos_query_count": 4,
            "strict_local_coverage": {
                "undercovered_query_count": coverage_undercovered,
                "undercovered_query_fraction": coverage_undercovered / 4.0,
            },
        },
    )
    return condition_root, output_root, coverage


def run_summary(
    condition_root: Path,
    output_root: Path,
    coverage: Path,
    *,
    diagnostic_reason: str = "",
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--condition-root",
        str(condition_root),
        "--output-root",
        str(output_root),
        "--expected-source-episodes",
        "2",
        "--expected-video-frames",
        "301",
        "--expected-action-steps",
        "300",
        "--expected-prefix-role-source",
        "physical_mode",
        "--min-late-rebind-candidates",
        "1",
        "--live-query-coverage-audit-json",
        str(coverage),
        "--max-live-query-undercovered-count",
        "0",
        "--max-live-query-undercovered-fraction",
        "0.0",
    ]
    if diagnostic_reason:
        cmd.extend(["--diagnostic-not-ready-reason", diagnostic_reason])
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cosmos3_clean_dense_summary_selftest_") as tmp_text:
        tmp = Path(tmp_text)
        condition_root, output_root, coverage = write_ready_inputs(tmp / "ok", coverage_undercovered=0)
        ok = run_summary(condition_root, output_root, coverage)
        if ok.returncode != 0:
            raise AssertionError(f"expected ready summary, got {ok.returncode}\n{ok.stdout}\n{ok.stderr}")
        report = json.loads((output_root / "clean_dense_preflight_summary.json").read_text())
        if not report.get("ready_for_overfit"):
            raise AssertionError(report.get("failed_checks"))

        bad_condition, bad_output, bad_coverage = write_ready_inputs(tmp / "bad", coverage_undercovered=1)
        bad = run_summary(bad_condition, bad_output, bad_coverage)
        if bad.returncode == 0:
            raise AssertionError("coverage-undercovered summary unexpectedly passed")
        bad_report = json.loads((bad_output / "clean_dense_preflight_summary.json").read_text())
        failed_names = {item["name"] for item in bad_report.get("failed_checks", [])}
        if "live_query_coverage_undercovered_count" not in failed_names:
            raise AssertionError(f"missing coverage count failure: {failed_names}")
        if "live_query_coverage_undercovered_fraction" not in failed_names:
            raise AssertionError(f"missing coverage fraction failure: {failed_names}")

        diag_condition, diag_output, diag_coverage = write_ready_inputs(tmp / "diagnostic", coverage_undercovered=0)
        diag = run_summary(
            diag_condition,
            diag_output,
            diag_coverage,
            diagnostic_reason="live_query_coverage_audit_skipped_by_diagnostic_override",
        )
        if diag.returncode == 0:
            raise AssertionError("diagnostic-not-ready summary unexpectedly passed")
        diag_report = json.loads((diag_output / "clean_dense_preflight_summary.json").read_text())
        diag_failed_names = {item["name"] for item in diag_report.get("failed_checks", [])}
        if "diagnostic_not_method_ready" not in diag_failed_names:
            raise AssertionError(f"missing diagnostic_not_method_ready failure: {diag_failed_names}")

    print("cosmos3_clean_dense_preflight_summary_selftest=passed")


if __name__ == "__main__":
    main()
