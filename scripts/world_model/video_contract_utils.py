#!/usr/bin/env python3
"""Lightweight video artifact contract utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def inspect_video_file(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "decoder": None,
        "decoder_errors": [],
        "decoded_frame_count": None,
        "fps": None,
        "duration_seconds": None,
        "error": None,
    }
    if not path.is_file():
        report["error"] = "missing_file"
        return report
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover - environment dependent
        report["decoder_errors"].append(f"imageio_import_failed:{exc}")
    else:
        try:
            reader = imageio.get_reader(path)
            try:
                meta = reader.get_meta_data() or {}
                fps = meta.get("fps")
                if fps is not None:
                    report["fps"] = float(fps)
                count = 0
                for _frame in reader:
                    count += 1
                report["decoder"] = "imageio"
                report["decoded_frame_count"] = int(count)
                if report["fps"]:
                    report["duration_seconds"] = float(count) / float(report["fps"])
                return report
            finally:
                reader.close()
        except Exception as exc:
            report["decoder_errors"].append(f"imageio_decode_failed:{exc}")

    try:
        import cv2
    except Exception as exc:
        report["decoder_errors"].append(f"cv2_import_failed:{exc}")
        report["error"] = ";".join(report["decoder_errors"])
        return report
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            report["decoder_errors"].append("cv2_open_failed")
            report["error"] = ";".join(report["decoder_errors"])
            return report
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            report["fps"] = float(fps)
        count = 0
        while True:
            ok, _frame = cap.read()
            if not ok:
                break
            count += 1
        report["decoder"] = "cv2"
        report["decoded_frame_count"] = int(count)
        if report["fps"]:
            report["duration_seconds"] = float(count) / float(report["fps"])
        report["error"] = None
    except Exception as exc:
        report["decoder_errors"].append(f"cv2_decode_failed:{exc}")
        report["error"] = ";".join(report["decoder_errors"])
    finally:
        cap.release()
    return report


def video_inspections_match_contract(
    inspections: list[Any],
    *,
    expected_video_frames: int,
    expected_inspection_count: int,
) -> bool:
    if len(inspections) != int(expected_inspection_count):
        return False
    return bool(
        inspections
        and all(
            isinstance(item, dict)
            and item.get("exists") is True
            and item.get("error") is None
            and int(item.get("decoded_frame_count") or -1) == int(expected_video_frames)
            for item in inspections
        )
    )
