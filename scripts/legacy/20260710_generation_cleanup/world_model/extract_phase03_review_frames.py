#!/usr/bin/env python3
"""Extract fixed review frames from a Phase 03 Oracle video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def read_frame(video_path: Path, frame_idx: int) -> Image.Image:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"cannot read frame {frame_idx} from {video_path}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def make_sheet(video_path: Path, output_path: Path, frames: list[int], labels: list[str]) -> None:
    thumbs: list[Image.Image] = []
    for frame_idx, label in zip(frames, labels):
        img = read_frame(video_path, frame_idx)
        img.thumbnail((480, 270))
        tile = Image.new("RGB", (480, 310), "white")
        tile.paste(img, ((480 - img.width) // 2, 24))
        ImageDraw.Draw(tile).text((8, 6), f"f{frame_idx} {label}", fill=(0, 0, 0))
        thumbs.append(tile)

    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 480, rows * 310), "white")
    for idx, tile in enumerate(thumbs):
        sheet.paste(tile, ((idx % cols) * 480, (idx // cols) * 310))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    args = parser.parse_args()
    if len(args.frames) != len(args.labels):
        raise ValueError("--frames and --labels must have the same length")

    review_dir = args.run_dir / "review"
    make_sheet(args.run_dir / "videos" / "annotated.mp4", review_dir / "annotated_keyframes.jpg", args.frames, args.labels)
    make_sheet(args.run_dir / "videos" / "raw.mp4", review_dir / "raw_keyframes.jpg", args.frames, args.labels)
    print(review_dir / "annotated_keyframes.jpg")
    print(review_dir / "raw_keyframes.jpg")


if __name__ == "__main__":
    main()
