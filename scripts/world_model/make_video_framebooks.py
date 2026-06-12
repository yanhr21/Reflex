#!/usr/bin/env python3
"""Create paginated all-frame contact sheets for rendered videos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import cv2
from PIL import Image, ImageDraw, ImageFont
import tyro


@dataclass
class Args:
    videos_dir: str
    output_dir: str
    manifest_json: str = ""
    frames_per_page: int = 20
    columns: int = 5
    thumb_width: int = 256
    thumb_height: int = 256
    overwrite: bool = False


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text(xy, text, fill=(0, 0, 0), font=font)


def _video_paths(args: Args) -> list[Path]:
    if args.manifest_json:
        manifest = json.loads(Path(args.manifest_json).read_text())
        videos = manifest.get("videos") or []
        paths = [Path(item["video"]) for item in videos]
    else:
        paths = sorted(Path(args.videos_dir).glob("*.mp4"))
    if not paths:
        raise FileNotFoundError(f"no videos found under {args.videos_dir}")
    return paths


def _make_pages(video_path: Path, out_dir: Path, args: Args) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {video_path}")

    rows = (args.frames_per_page + args.columns - 1) // args.columns
    cell_w = int(args.thumb_width)
    cell_h = int(args.thumb_height) + 24
    header_h = 34
    page_w = args.columns * cell_w
    page_h = header_h + rows * cell_h
    frames: list[Image.Image] = []
    page_paths: list[str] = []
    frame_idx = 0
    video_out = out_dir / video_path.stem
    video_out.mkdir(parents=True, exist_ok=True)

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((cell_w, int(args.thumb_height)), Image.Resampling.LANCZOS)
        frames.append(img)
        if len(frames) == args.frames_per_page:
            page_paths.append(str(_write_page(video_out, video_path.name, frames, frame_idx - len(frames) + 1, page_w, page_h, header_h, cell_w, cell_h, args)))
            frames = []
        frame_idx += 1

    if frames:
        page_paths.append(str(_write_page(video_out, video_path.name, frames, frame_idx - len(frames), page_w, page_h, header_h, cell_w, cell_h, args)))
    cap.release()
    return {
        "video": str(video_path),
        "num_frames": int(frame_idx),
        "num_pages": len(page_paths),
        "page_paths": page_paths,
    }


def _write_page(
    out_dir: Path,
    video_name: str,
    frames: list[Image.Image],
    start_frame: int,
    page_w: int,
    page_h: int,
    header_h: int,
    cell_w: int,
    cell_h: int,
    args: Args,
) -> Path:
    end_frame = start_frame + len(frames) - 1
    page_idx = start_frame // int(args.frames_per_page)
    out_path = out_dir / f"page_{page_idx:03d}_frames_{start_frame:03d}_{end_frame:03d}.png"
    if out_path.exists() and not args.overwrite:
        return out_path
    canvas = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(canvas)
    _draw_text(draw, (6, 6), f"{video_name} | frames {start_frame:03d}-{end_frame:03d}")
    for local_idx, img in enumerate(frames):
        frame_id = start_frame + local_idx
        row = local_idx // int(args.columns)
        col = local_idx % int(args.columns)
        x = col * cell_w
        y = header_h + row * cell_h
        canvas.paste(img, (x, y))
        _draw_text(draw, (x + 4, y + int(args.thumb_height) + 4), f"frame {frame_id:03d}")
    canvas.save(out_path)
    return out_path


def main() -> None:
    args = tyro.cli(Args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [_make_pages(video_path, out_dir, args) for video_path in _video_paths(args)]
    summary = {
        "args": vars(args),
        "num_videos": len(records),
        "records": records,
    }
    (out_dir / "framebook_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "framebooks_done", "output_dir": str(out_dir), "num_videos": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
