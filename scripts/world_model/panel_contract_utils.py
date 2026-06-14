#!/usr/bin/env python3
"""Shared panel-level contract checks for full-episode rollouts."""

from __future__ import annotations

from typing import Any


def sample_full_episode_contract_failures(
    row: dict[str, Any],
    *,
    expected_frames: int,
    expected_actions: int,
    require_pure_dp: bool = False,
) -> list[str]:
    failures: list[str] = []
    if not row.get("ok"):
        failures.append("sample_process_or_summary_not_ok")
    if row.get("full_episode_length_ok") is not True:
        failures.append("full_episode_length_not_ok")
    if int(row.get("final_prefix_frame_index") or -1) != int(expected_actions):
        failures.append("final_prefix_frame_index_not_expected_actions")
    if int(row.get("final_observed_frames") or -1) != int(expected_frames):
        failures.append("final_observed_frames_not_expected_frames")
    if row.get("video_file_contract_ok") is not True:
        failures.append("video_file_contract_not_ok")
    counts = row.get("controller_frame_counts") or {}
    if not isinstance(counts, dict) or sum(int(value or 0) for value in counts.values()) != int(expected_frames):
        failures.append("controller_frame_counts_sum_not_expected_frames")
    if require_pure_dp:
        if isinstance(counts, dict) and int(counts.get("PURE_DP", 0) or 0) != int(expected_actions):
            failures.append("pure_dp_frame_count_not_expected_actions")
        if isinstance(counts, dict) and int(counts.get("WM_ACTIVE", 0) or 0) != 0:
            failures.append("pure_dp_has_wm_active")
    return failures


def panel_contract_rows(
    rows: list[dict[str, Any]],
    *,
    expected_frames: int,
    expected_actions: int,
    require_pure_dp: bool = False,
) -> list[dict[str, Any]]:
    return [
        {
            "sample_index": row.get("sample_index"),
            "scenario": row.get("scenario"),
            "summary_path": row.get("summary_path"),
            "contract_failures": sample_full_episode_contract_failures(
                row,
                expected_frames=expected_frames,
                expected_actions=expected_actions,
                require_pure_dp=require_pure_dp,
            ),
        }
        for row in rows
    ]


def panel_full_episode_contract_ok(contract_rows: list[dict[str, Any]], *, expected_samples: int) -> bool:
    return bool(
        len(contract_rows) == int(expected_samples)
        and not [row for row in contract_rows if row.get("contract_failures")]
    )
