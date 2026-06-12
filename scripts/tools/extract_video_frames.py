#!/usr/bin/env python3
"""Extract a few debug frames from rollout videos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import imageio.v2 as imageio
import tyro


@dataclass
class Args:
    video_dir: str
    output_dir: str
    max_videos: int = 16
    frame_indices: str = ""


def safe_indices(reader) -> list[int]:
    try:
        n = reader.count_frames()
        if n <= 0:
            return [0]
        return sorted(set([0, n // 2, n - 1]))
    except Exception:
        return [0]


def parse_frame_indices(raw: str) -> list[int] | None:
    raw = raw.strip()
    if not raw:
        return None
    indices = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        indices.append(max(0, int(part)))
    return sorted(set(indices))


def main():
    args = tyro.cli(Args)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.rglob("*.mp4"))[: args.max_videos]
    manifest = []
    for video in videos:
        reader = imageio.get_reader(video)
        try:
            requested_indices = parse_frame_indices(args.frame_indices)
            indices = requested_indices if requested_indices is not None else safe_indices(reader)
            for idx in indices:
                try:
                    frame = reader.get_data(idx)
                except Exception:
                    continue
                rel = video.relative_to(video_dir)
                stem = "_".join(rel.with_suffix("").parts)
                out_path = output_dir / f"{stem}_frame_{idx:05d}.png"
                imageio.imwrite(out_path, frame)
                manifest.append({"video": str(video), "frame": idx, "output": str(out_path)})
        finally:
            reader.close()

    manifest_path = output_dir / "frames_manifest.jsonl"
    with manifest_path.open("w") as f:
        for item in manifest:
            f.write(json.dumps(item, sort_keys=True) + "\n")
    print(f"videos={len(videos)} frames={len(manifest)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
