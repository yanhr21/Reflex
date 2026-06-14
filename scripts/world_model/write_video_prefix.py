#!/usr/bin/env python3
"""Write a causal prefix-only video from a longer reference/observed video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--prefix-frame-index", type=int, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import imageio.v2 as imageio

    input_video = Path(args.input_video).resolve()
    output_video = Path(args.output_video).resolve()
    if not input_video.is_file():
        raise FileNotFoundError(input_video)
    keep = max(1, int(args.prefix_frame_index) + 1)

    reader = imageio.get_reader(input_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_video, fps=max(1, int(args.fps)))
    written = 0
    try:
        for frame in reader:
            if written >= keep:
                break
            writer.append_data(frame)
            written += 1
    finally:
        writer.close()
        reader.close()

    if written != keep:
        raise ValueError(f"wrote {written} frames, expected {keep} from {input_video}")
    manifest = {
        "input_video": str(input_video),
        "output_video": str(output_video),
        "prefix_frame_index": int(args.prefix_frame_index),
        "num_frames": written,
        "fps": int(args.fps),
        "causal_boundary": "Output contains frames [0, prefix_frame_index] only.",
    }
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
