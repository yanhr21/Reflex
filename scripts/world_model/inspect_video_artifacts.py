#!/usr/bin/env python3
"""Prepare review sheets for controller video artifacts.

This script checks whether videos exist and are readable, then samples frames
into padded contact sheets. It does not judge task success; semantic success
still requires human/agent visual inspection of the generated sheets together
with metrics and event logs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import tyro


@dataclass
class Args:
    run_dir: str
    output_dir: str | None = None
    output_json: str | None = None
    output_md: str | None = None
    sample_count: int = 20
    max_frames_to_scan: int = 2000
    thumb_width: int = 480
    require_video: bool = True
    require_nonblank: bool = True


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return list(value)
    return value


def _video_paths(run_dir: Path) -> list[Path]:
    video_dir = run_dir / "videos"
    if not video_dir.exists():
        return []
    return sorted(path for path in video_dir.glob("*.mp4") if path.is_file())


def _frame_stats(frame: np.ndarray) -> dict[str, float]:
    arr = np.asarray(frame[..., :3], dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _sample_video(path: Path, sample_count: int, max_frames_to_scan: int) -> tuple[list[tuple[int, np.ndarray]], dict[str, Any]]:
    reader = imageio.get_reader(path)
    frames: list[tuple[int, np.ndarray]] = []
    stats = []
    try:
        for frame_i, frame in enumerate(reader):
            if frame_i >= max_frames_to_scan:
                break
            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            if arr.shape[-1] > 3:
                arr = arr[..., :3]
            arr = arr.astype(np.uint8, copy=False)
            frames.append((frame_i, arr.copy()))
            stats.append(_frame_stats(arr))
    finally:
        reader.close()
    if not frames:
        return [], {"frame_count_scanned": 0, "sample_indices": [], "frame_stats": {}}
    desired = min(sample_count, len(frames))
    indices = np.linspace(0, len(frames) - 1, desired).round().astype(np.int64)
    sampled = [frames[int(idx)] for idx in np.unique(indices)]
    means = np.asarray([item["mean"] for item in stats], dtype=np.float32)
    stds = np.asarray([item["std"] for item in stats], dtype=np.float32)
    return sampled, {
        "frame_count_scanned": len(frames),
        "sample_indices": [int(item[0]) for item in sampled],
        "frame_stats": {
            "mean_min": float(means.min()),
            "mean_mean": float(means.mean()),
            "mean_max": float(means.max()),
            "std_min": float(stds.min()),
            "std_mean": float(stds.mean()),
            "std_max": float(stds.max()),
        },
        "nonblank_basic": bool(float(stds.mean()) > 1.0),
    }


def _draw_sheet(samples: list[tuple[int, np.ndarray]], output_path: Path, thumb_width: int):
    if not samples:
        return
    thumbs = []
    for frame_i, frame in samples:
        image = Image.fromarray(frame[..., :3]).convert("RGB")
        scale = thumb_width / float(image.width)
        image = image.resize((thumb_width, max(1, int(round(image.height * scale)))))
        thumbs.append((frame_i, image))
    cols = min(5, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    label_h = 24
    cell_h = max(img.height for _idx, img in thumbs) + label_h
    sheet = Image.new("RGB", (cols * thumb_width, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (frame_i, image) in enumerate(thumbs):
        row, col = divmod(idx, cols)
        x = col * thumb_width
        y = row * cell_h
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + image.height + 4), f"frame {frame_i}", fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _write_md(report: dict[str, Any], path: Path):
    lines = [
        "# Video Artifact Inspection",
        "",
        f"- run dir: `{report['run_dir']}`",
        f"- videos found: `{report['num_videos']}`",
        f"- readable videos: `{report['num_readable_videos']}`",
        f"- basic nonblank videos: `{report['num_nonblank_basic_videos']}`",
        f"- review sheets: `{len(report['review_sheets'])}`",
        f"- valid video artifacts: `{report['valid_video_artifacts']}`",
        "",
        "## Boundary",
        "",
        "This report checks that video artifacts are readable, basically nonblank when required, and prepares sampled review sheets. It does not judge task success.",
        "",
        "## Videos",
        "",
        "| video | readable | frames scanned | nonblank basic | review sheet |",
        "|---|---:|---:|---:|---|",
    ]
    for item in report["videos"]:
        lines.append(
            "| `{video}` | {readable} | {frames} | {nonblank} | `{sheet}` |".format(
                video=item["path"],
                readable=item["readable"],
                frames=item.get("frame_count_scanned", 0),
                nonblank=item.get("nonblank_basic"),
                sheet=item.get("review_sheet", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    args = tyro.cli(Args)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "video_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_reports = []
    review_sheets = []
    for video_path in _video_paths(run_dir):
        review_sheet = output_dir / f"{video_path.stem}_review_sheet.png"
        try:
            samples, sample_report = _sample_video(video_path, args.sample_count, args.max_frames_to_scan)
            if samples:
                _draw_sheet(samples, review_sheet, args.thumb_width)
                review_sheets.append(str(review_sheet))
            video_reports.append(
                {
                    "path": str(video_path),
                    "readable": bool(samples),
                    "review_sheet": str(review_sheet) if samples else None,
                    **sample_report,
                }
            )
        except Exception as exc:  # noqa: BLE001
            video_reports.append(
                {
                    "path": str(video_path),
                    "readable": False,
                    "error": repr(exc),
                    "review_sheet": None,
                    "frame_count_scanned": 0,
                    "nonblank_basic": False,
                }
            )
    valid = bool(video_reports) and all(
        item.get("readable", False)
        and item.get("review_sheet")
        and (not args.require_nonblank or item.get("nonblank_basic", False))
        for item in video_reports
    )
    report = {
        "args": asdict(args),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "num_videos": len(video_reports),
        "num_readable_videos": int(sum(bool(item.get("readable", False)) for item in video_reports)),
        "num_nonblank_basic_videos": int(sum(bool(item.get("nonblank_basic", False)) for item in video_reports)),
        "review_sheets": review_sheets,
        "valid_video_artifacts": valid,
        "videos": video_reports,
        "boundary": "Readable video artifacts are necessary but not sufficient for task success.",
    }
    output_json = Path(args.output_json) if args.output_json else output_dir / "video_artifact_inspection.json"
    output_md = Path(args.output_md) if args.output_md else output_dir / "video_artifact_inspection.md"
    output_json.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    _write_md(report, output_md)
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    if args.require_video and not valid:
        raise SystemExit(65)


if __name__ == "__main__":
    main()
