#!/usr/bin/env python3
"""Sanitize causal full-episode WAM captions without touching targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPLACEMENTS = {
    "Predict target motion/final target pose, future RGB/task state, and executable actions in the changed world.": (
        "Predict target motion, future RGB/task state, and executable actions in the changed world."
    ),
    "final target pose": "future task-frame state",
    "target final xyz": "target settled xyz label",
    "target_final_xyz": "target_settled_xyz_label",
    "target_final": "target_settled_label",
}


def sanitize_text(text: str) -> str:
    out = text
    for old, new in REPLACEMENTS.items():
        out = out.replace(old, new)
    return out


def sanitize_row(row: dict) -> tuple[dict, int]:
    edits = 0
    windows = row.get("t2w_windows") or []
    for window in windows:
        if isinstance(window, dict) and isinstance(window.get("caption"), str):
            old = window["caption"]
            new = sanitize_text(old)
            if new != old:
                window["caption"] = new
                edits += 1
    return row, edits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-root", required=True)
    args = parser.parse_args()
    root = Path(args.condition_root)
    total_rows = 0
    total_edits = 0
    for split in ("train", "val"):
        path = root / split / "video_action_dataset_file.jsonl"
        rows = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row, edits = sanitize_row(json.loads(line))
            rows.append(row)
            total_rows += 1
            total_edits += edits
        path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    print(json.dumps({"condition_root": str(root), "rows": total_rows, "caption_edits": total_edits}, sort_keys=True))


if __name__ == "__main__":
    main()
