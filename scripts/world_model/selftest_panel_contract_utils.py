#!/usr/bin/env python3
"""Local checks for panel-level full-episode contract utilities."""

from __future__ import annotations

from panel_contract_utils import panel_contract_rows, panel_full_episode_contract_ok


def valid_row() -> dict:
    return {
        "ok": True,
        "sample_index": 0,
        "scenario": "hole_late_move_stop",
        "summary_path": "/tmp/summary.json",
        "full_episode_length_ok": True,
        "final_prefix_frame_index": 300,
        "final_observed_frames": 301,
        "video_file_contract_ok": True,
        "controller_frame_counts": {"INIT_OBS": 1, "DP_SCAN_TARGET": 100, "WM_ACTIVE": 200},
    }


def valid_pure_dp_row() -> dict:
    row = valid_row()
    row["controller_frame_counts"] = {"INIT_OBS": 1, "PURE_DP": 300}
    return row


def assert_failure(row: dict, expected: str, *, require_pure_dp: bool = False) -> None:
    contract = panel_contract_rows(
        [row],
        expected_frames=301,
        expected_actions=300,
        require_pure_dp=require_pure_dp,
    )
    failures = contract[0]["contract_failures"]
    if expected not in failures:
        raise AssertionError(f"expected {expected!r} in failures={failures}")


def main() -> None:
    good = panel_contract_rows([valid_row()], expected_frames=301, expected_actions=300)
    if good[0]["contract_failures"]:
        raise AssertionError(f"valid Cosmos panel row failed: {good}")
    if not panel_full_episode_contract_ok(good, expected_samples=1):
        raise AssertionError("valid Cosmos panel should pass")

    short = valid_row()
    short["final_observed_frames"] = 131
    assert_failure(short, "final_observed_frames_not_expected_frames")

    bad_video = valid_row()
    bad_video["video_file_contract_ok"] = False
    assert_failure(bad_video, "video_file_contract_not_ok")

    bad_counts = valid_row()
    bad_counts["controller_frame_counts"] = {"INIT_OBS": 1, "WM_ACTIVE": 100}
    assert_failure(bad_counts, "controller_frame_counts_sum_not_expected_frames")

    pure = panel_contract_rows(
        [valid_pure_dp_row()],
        expected_frames=301,
        expected_actions=300,
        require_pure_dp=True,
    )
    if pure[0]["contract_failures"]:
        raise AssertionError(f"valid pure-DP panel row failed: {pure}")

    wm_in_pure = valid_pure_dp_row()
    wm_in_pure["controller_frame_counts"] = {"INIT_OBS": 1, "PURE_DP": 299, "WM_ACTIVE": 1}
    assert_failure(wm_in_pure, "pure_dp_frame_count_not_expected_actions", require_pure_dp=True)
    assert_failure(wm_in_pure, "pure_dp_has_wm_active", require_pure_dp=True)

    if panel_full_episode_contract_ok(good, expected_samples=2):
        raise AssertionError("missing requested sample should fail panel contract")

    print("panel_contract_utils_selftest=passed")


if __name__ == "__main__":
    main()
