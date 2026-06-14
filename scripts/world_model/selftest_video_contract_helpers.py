#!/usr/bin/env python3
"""Local checks for video artifact contract helpers."""

from __future__ import annotations

from video_contract_utils import video_inspections_match_contract


def inspection(frames: int = 301) -> dict:
    return {
        "path": "/tmp/fake.mp4",
        "exists": True,
        "decoder": "synthetic",
        "decoder_errors": [],
        "decoded_frame_count": frames,
        "fps": 30.0,
        "duration_seconds": frames / 30.0,
        "error": None,
    }


def main() -> None:
    if not video_inspections_match_contract(
        [inspection(), inspection()],
        expected_video_frames=301,
        expected_inspection_count=2,
    ):
        raise AssertionError("valid raw+annotated inspections should pass")
    if video_inspections_match_contract(
        [inspection()],
        expected_video_frames=301,
        expected_inspection_count=2,
    ):
        raise AssertionError("missing annotated inspection should fail")
    if video_inspections_match_contract(
        [inspection(300), inspection()],
        expected_video_frames=301,
        expected_inspection_count=2,
    ):
        raise AssertionError("wrong frame count should fail")
    bad = inspection()
    bad["error"] = "decode_failed"
    if video_inspections_match_contract(
        [bad, inspection()],
        expected_video_frames=301,
        expected_inspection_count=2,
    ):
        raise AssertionError("decode error should fail")
    missing = inspection()
    missing["exists"] = False
    if video_inspections_match_contract(
        [missing, inspection()],
        expected_video_frames=301,
        expected_inspection_count=2,
    ):
        raise AssertionError("missing video should fail")
    if not video_inspections_match_contract(
        [inspection()],
        expected_video_frames=301,
        expected_inspection_count=1,
    ):
        raise AssertionError("valid raw-only inspection should pass when annotation is disabled")
    print("video_contract_helpers_selftest=passed")


if __name__ == "__main__":
    main()
