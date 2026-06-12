#!/usr/bin/env python3
"""Build a two-sample full-episode Cosmos3 overfit condition root.

The output keeps the original 301-frame / 300-action rows and only narrows the
train/val JSONLs to the same two rows. It is a training-chain sanity dataset,
not method evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PAIRS = (
    ("target_motion_observed", "hole_late_move_stop"),
    ("peg_recovery", "peg_drop"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-condition-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--expected-video-frames", type=int, default=301)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--expected-action-dim", type=int, default=32)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty jsonl: {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def row_scenario(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("scenario") or metadata.get("scenario") or "unknown")


def validate_row(row: dict[str, Any], args: argparse.Namespace) -> None:
    failures = []
    if int(row.get("num_video_frames", -1)) != args.expected_video_frames:
        failures.append(f"num_video_frames={row.get('num_video_frames')}")
    if int(row.get("num_action_steps", row.get("action_chunk_size", -1))) != args.expected_action_steps:
        failures.append(f"num_action_steps/action_chunk_size={row.get('num_action_steps')}/{row.get('action_chunk_size')}")
    if int(row.get("raw_action_dim", -1)) != args.expected_action_dim:
        failures.append(f"raw_action_dim={row.get('raw_action_dim')}")
    for key in ("vision_path", "action_path", "state_target_path", "task_label_path"):
        value = row.get(key)
        if not value or not Path(str(value)).is_file():
            failures.append(f"missing_{key}:{value}")
    cond_vision = row.get("condition_frame_indexes_vision") or []
    cond_action = row.get("condition_frame_indexes_action") or []
    if not cond_vision:
        failures.append("missing_condition_frame_indexes_vision")
    if cond_action and max(int(x) for x in cond_action) >= args.expected_action_steps:
        failures.append("condition_frame_indexes_action_out_of_range")
    if failures:
        raise RuntimeError(f"row {row.get('uuid')} failed: {failures}")


def choose_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = []
    used_sources = set()
    for role, scenario in DEFAULT_PAIRS:
        candidates = [row for row in rows if row.get("prefix_role") == role and row_scenario(row) == scenario]
        if not candidates:
            raise RuntimeError(f"no candidate for role={role} scenario={scenario}")
        for row in candidates:
            source_uuid = str(row.get("source_uuid") or row.get("uuid"))
            if source_uuid in used_sources and len(candidates) > 1:
                continue
            validate_row(row, args)
            selected.append(row)
            used_sources.add(source_uuid)
            break
    if len(selected) != 2:
        raise RuntimeError(f"expected 2 selected rows, got {len(selected)}")
    return selected


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_condition_root)
    output_root = Path(args.output_root)
    source_jsonl = source_root / args.split / "video_action_dataset_file.jsonl"
    manifest_path = source_root / "manifest.json"
    rows = read_jsonl(source_jsonl)
    selected = choose_rows(rows, args)

    for split in ("train", "val"):
        write_jsonl(output_root / split / "video_action_dataset_file.jsonl", selected)

    source_manifest = json.loads(manifest_path.read_text())
    manifest = dict(source_manifest)
    manifest.update(
        {
            "source_condition_root": str(source_root),
            "overfit_boundary": "Two-sample overfit sanity root; train and val contain the same two full-episode rows.",
            "num_source_episodes": len({str(row.get("source_uuid") or row.get("uuid")) for row in selected}),
            "num_rows": len(selected),
            "num_train_rows": len(selected),
            "num_val_rows": len(selected),
            "selected_rows": [
                {
                    "uuid": row.get("uuid"),
                    "source_uuid": row.get("source_uuid"),
                    "prefix_role": row.get("prefix_role"),
                    "scenario": row_scenario(row),
                    "prefix_frame_index": row.get("prefix_frame_index"),
                    "vision_path": row.get("vision_path"),
                    "action_path": row.get("action_path"),
                }
                for row in selected
            ],
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_root / "OVERFIT_SANITY_ONLY.md").write_text(
        "# Two-Sample Overfit Sanity Root\n\n"
        "Train and val JSONLs intentionally contain the same two full-episode rows. "
        "This root tests whether the Cosmos3 SFT/inference chain can memorize two samples; "
        "it is not method evidence.\n"
    )
    print(json.dumps({"output_root": str(output_root), "selected_rows": manifest["selected_rows"]}, sort_keys=True))


if __name__ == "__main__":
    main()
