#!/usr/bin/env python3
"""Inspect the corrected Cosmos3 SFT dataset before training.

This is a structural/data-quality check. It verifies that the full1000 dataset
uses readable ManiSkill-default videos and that the JSONL records carry the
video-prefix and robot/object state conditioning required by the active method.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as handle:
        for line_i, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_i}: invalid JSONL") from exc
    return rows


def _caption(record: dict[str, Any]) -> str:
    windows = record.get("t2w_windows") or []
    if not windows:
        return ""
    return str(windows[0].get("caption", ""))


def _video_report(path: Path, max_scan_frames: int) -> dict[str, Any]:
    reader = imageio.get_reader(path)
    frames = 0
    first_shape = None
    means = []
    stds = []
    meta = {}
    try:
        try:
            meta = dict(reader.get_meta_data())
        except Exception:
            meta = {}
        for frame_i, frame in enumerate(reader):
            if frame_i >= max_scan_frames:
                break
            arr = np.asarray(frame)
            if first_shape is None:
                first_shape = list(arr.shape)
            rgb = arr[..., :3].astype(np.float32)
            means.append(float(rgb.mean()))
            stds.append(float(rgb.std()))
            frames += 1
    finally:
        reader.close()
    return {
        "path": str(path),
        "readable": bool(frames > 0),
        "frames_scanned": frames,
        "first_shape": first_shape,
        "fps_meta": meta.get("fps"),
        "duration_meta": meta.get("duration"),
        "mean_rgb": float(np.mean(means)) if means else None,
        "std_rgb": float(np.mean(stds)) if stds else None,
        "nonblank_basic": bool(stds and float(np.mean(stds)) > 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--expected-count", type=int, default=1000)
    parser.add_argument("--expected-width", type=int, default=1024)
    parser.add_argument("--expected-height", type=int, default=1024)
    parser.add_argument("--expected-fps", type=float, default=30.0)
    parser.add_argument("--video-sample-count", type=int, default=10)
    parser.add_argument("--max-scan-frames", type=int, default=400)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    split_records = {
        "train": _read_jsonl(root / "train" / "video_dataset_file.jsonl"),
        "val": _read_jsonl(root / "val" / "video_dataset_file.jsonl"),
    }
    records = [(split, record) for split, rows in split_records.items() for record in rows]
    final_mp4s = sorted(path for path in root.rglob("*.mp4") if not path.name.endswith(".tmp.mp4"))
    tmp_mp4s = sorted(root.rglob("*.tmp.mp4"))

    scenario_counts = Counter()
    missing_videos = []
    bad_size_records = []
    bad_fps_records = []
    missing_state_caption = []
    missing_state_metadata = []
    bad_conditioning_policy = []
    for split, record in records:
        metadata = record.get("metadata") or {}
        scenario_counts[str(metadata.get("scenario", "unknown"))] += 1
        rel = str(record.get("vision_path", ""))
        video_path = root / split / rel
        if not video_path.exists():
            missing_videos.append({"split": split, "uuid": record.get("uuid"), "path": str(video_path)})
        if int(record.get("width", -1)) != args.expected_width or int(record.get("height", -1)) != args.expected_height:
            bad_size_records.append(record.get("uuid"))
        if abs(float(metadata.get("fps", -1.0)) - args.expected_fps) > 1e-6:
            bad_fps_records.append(record.get("uuid"))
        caption = _caption(record)
        if "Robot and object state condition:" not in caption:
            missing_state_caption.append(record.get("uuid"))
        state = metadata.get("task_state_condition") or {}
        if not state or not any(key.endswith("_xyz_start") for key in state):
            missing_state_metadata.append(record.get("uuid"))
        if metadata.get("conditioning_policy") != "video_prefix_not_single_image_i2v":
            bad_conditioning_policy.append(record.get("uuid"))

    video_reports = []
    for video_path in final_mp4s[: max(0, args.video_sample_count)]:
        video_reports.append(_video_report(video_path, args.max_scan_frames))

    bad_video_samples = [
        item
        for item in video_reports
        if not item["readable"]
        or not item["nonblank_basic"]
        or item["first_shape"] is None
        or item["first_shape"][0] != args.expected_height
        or item["first_shape"][1] != args.expected_width
        or item["frames_scanned"] < 300
    ]

    checks = {
        "manifest_num_videos": int(manifest.get("num_videos", -1)) == args.expected_count,
        "jsonl_record_count": len(records) == args.expected_count,
        "final_mp4_count": len(final_mp4s) == args.expected_count,
        "tmp_mp4_count_zero": len(tmp_mp4s) == 0,
        "all_record_videos_exist": len(missing_videos) == 0,
        "all_record_sizes_match": len(bad_size_records) == 0,
        "all_record_fps_match": len(bad_fps_records) == 0,
        "all_records_have_state_caption": len(missing_state_caption) == 0,
        "all_records_have_state_metadata": len(missing_state_metadata) == 0,
        "all_records_have_video_prefix_policy": len(bad_conditioning_policy) == 0,
        "sampled_videos_readable_nonblank_1024": len(bad_video_samples) == 0,
    }
    report = {
        "dataset_root": str(root),
        "expected_count": args.expected_count,
        "num_manifest_videos": manifest.get("num_videos"),
        "num_train": len(split_records["train"]),
        "num_val": len(split_records["val"]),
        "num_records": len(records),
        "num_final_mp4s": len(final_mp4s),
        "num_tmp_mp4s": len(tmp_mp4s),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "checks": checks,
        "valid": all(checks.values()),
        "failures": {
            "missing_videos": missing_videos[:20],
            "bad_size_records": bad_size_records[:20],
            "bad_fps_records": bad_fps_records[:20],
            "missing_state_caption": missing_state_caption[:20],
            "missing_state_metadata": missing_state_metadata[:20],
            "bad_conditioning_policy": bad_conditioning_policy[:20],
            "bad_video_samples": bad_video_samples[:20],
        },
        "video_reports": video_reports,
        "boundary": (
            "This validates corrected Cosmos3 SFT data structure and conditioning. "
            "It is not a controller or task-success result."
        ),
    }

    output_json = Path(args.output_json) if args.output_json else root / "cosmos3_sft_dataset_inspection.json"
    output_json.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n")
    output_md = Path(args.output_md) if args.output_md else root / "cosmos3_sft_dataset_inspection.md"
    lines = [
        "# Cosmos3 SFT Dataset Inspection",
        "",
        f"- dataset root: `{root}`",
        f"- records: `{len(records)}`",
        f"- final MP4s: `{len(final_mp4s)}`",
        f"- tmp MP4s: `{len(tmp_mp4s)}`",
        f"- valid: `{report['valid']}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    output_md.write_text("\n".join(lines) + "\n")

    print(json.dumps({"inspection": str(output_json), "valid": report["valid"], "checks": checks}, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
