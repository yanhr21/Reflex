#!/usr/bin/env python3
"""Audit mp4 frame-count/duration contracts under one or more roots.

This is a read-only artifact checker. It is intended to make old short
closed-loop videos visibly invalid while proving current full-episode videos
from the file contents, not from summaries alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from video_contract_utils import inspect_video_file


def parse_root(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise argparse.ArgumentTypeError(f"empty root label in {value!r}")
        return label, Path(path).expanduser().resolve()
    path = Path(value).expanduser().resolve()
    return path.name or "root", path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        action="append",
        type=parse_root,
        required=True,
        help="Root to scan, either LABEL=/path or /path. Can be repeated.",
    )
    parser.add_argument(
        "--include-glob",
        action="append",
        default=[],
        help="Relative glob under each root. Defaults to **/*.mp4.",
    )
    parser.add_argument("--expected-frames", type=int, default=301)
    parser.add_argument("--expected-fps", type=float, default=30.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    return parser.parse_args()


def read_video(path: Path) -> dict[str, Any]:
    report = inspect_video_file(path)
    return {
        **report,
        "frame_count": report.get("decoded_frame_count"),
    }


def unique_paths(root: Path, patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns or ["**/*.mp4"]:
        for path in sorted(root.glob(pattern)):
            resolved = path.resolve()
            if resolved not in seen and path.is_file():
                seen.add(resolved)
                paths.append(path)
    return paths


def summarize(label: str, root: Path, videos: list[dict[str, Any]], expected_frames: int, expected_fps: float) -> dict[str, Any]:
    frame_counts = [
        int(video["frame_count"])
        for video in videos
        if video.get("frame_count") is not None
    ]
    durations = [
        float(video["duration_seconds"])
        for video in videos
        if video.get("duration_seconds") is not None
    ]
    errors = [video for video in videos if video.get("error")]
    frame_failures = [
        video
        for video in videos
        if video.get("frame_count") is not None and int(video["frame_count"]) != int(expected_frames)
    ]
    fps_failures = [
        video
        for video in videos
        if video.get("fps") is not None and abs(float(video["fps"]) - float(expected_fps)) > 1e-6
    ]
    return {
        "label": label,
        "root": str(root),
        "video_count": len(videos),
        "scan_error_count": len(errors),
        "expected_frames": int(expected_frames),
        "expected_fps": float(expected_fps),
        "all_videos_match_expected_frames": bool(videos) and not frame_failures and not errors,
        "all_videos_match_expected_fps": bool(videos) and not fps_failures and not errors,
        "min_frame_count": min(frame_counts) if frame_counts else None,
        "max_frame_count": max(frame_counts) if frame_counts else None,
        "min_duration_seconds": min(durations) if durations else None,
        "max_duration_seconds": max(durations) if durations else None,
        "frame_failure_count": len(frame_failures),
        "fps_failure_count": len(fps_failures),
        "frame_failures": frame_failures,
        "fps_failures": fps_failures,
        "errors": errors,
        "videos": videos,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Video Length Contract Audit",
        "",
        f"- expected_frames: `{report['expected_frames']}`",
        f"- expected_fps: `{report['expected_fps']}`",
        f"- include_globs: `{report['include_globs']}`",
        "",
        "## Roots",
        "",
    ]
    for item in report["roots"]:
        lines.extend(
            [
                f"### {item['label']}",
                "",
                f"- root: `{item['root']}`",
                f"- video_count: `{item['video_count']}`",
                f"- all_videos_match_expected_frames: `{item['all_videos_match_expected_frames']}`",
                f"- frame_count_range: `{item['min_frame_count']}..{item['max_frame_count']}`",
                f"- duration_seconds_range: `{item['min_duration_seconds']}..{item['max_duration_seconds']}`",
                f"- frame_failure_count: `{item['frame_failure_count']}`",
                f"- scan_error_count: `{item['scan_error_count']}`",
                "",
            ]
        )
        if item["frame_failures"]:
            lines.append("Frame failures:")
            for video in item["frame_failures"][:20]:
                rel = Path(video["path"])
                lines.append(f"- `{rel}`: `{video.get('frame_count')}` frames, `{video.get('duration_seconds')}` seconds")
            if len(item["frame_failures"]) > 20:
                lines.append(f"- ... {len(item['frame_failures']) - 20} more")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    include_globs = args.include_glob or ["**/*.mp4"]
    roots = []
    for label, root in args.root:
        paths = unique_paths(root, include_globs)
        videos = [read_video(path) for path in paths]
        roots.append(summarize(label, root, videos, args.expected_frames, args.expected_fps))
    report = {
        "boundary": (
            "Read-only video artifact audit. This checks mp4 files directly "
            "and does not run simulation, rendering, training, or inference."
        ),
        "expected_frames": int(args.expected_frames),
        "expected_fps": float(args.expected_fps),
        "include_globs": include_globs,
        "roots": roots,
    }
    write_json(Path(args.output_json).resolve(), report)
    if args.output_md:
        write_md(Path(args.output_md).resolve(), report)
    print(
        json.dumps(
            {
                item["label"]: {
                    "video_count": item["video_count"],
                    "all_videos_match_expected_frames": item["all_videos_match_expected_frames"],
                    "frame_count_range": [item["min_frame_count"], item["max_frame_count"]],
                    "frame_failure_count": item["frame_failure_count"],
                }
                for item in roots
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
