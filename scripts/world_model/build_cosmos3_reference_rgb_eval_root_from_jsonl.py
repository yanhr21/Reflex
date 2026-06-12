#!/usr/bin/env python3
"""Build a reference-RGB eval root from full-episode condition JSONL rows.

This is used to run the task-state readout on ground-truth RGB videos for a
larger calibration set. Samples are unique by source_uuid so the same full
episode is not repeated through multiple prefix rows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


SCENARIO_ORDER = [
    "hole_continuous_insert_large",
    "hole_move_stop_large",
    "hole_constant_large",
    "hole_reverse_large",
    "hole_sine_large",
    "hole_late_shift_large",
    "none",
    "hole_move_stop",
    "hole_constant",
    "hole_reverse",
    "peg_disturb",
    "peg_drop",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--expected-video-frames", type=int, default=301)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def scenario(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("scenario") or metadata.get("scenario") or "unknown")


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")[:180] or "sample"


def first_unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique = []
    for row in rows:
        source_uuid = str(row.get("source_uuid") or row.get("source_sample_id") or row.get("uuid") or "")
        if not source_uuid or source_uuid in seen:
            continue
        seen.add(source_uuid)
        unique.append(row)
    return unique


def balanced_subset(rows: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    if max_samples <= 0 or len(rows) <= max_samples:
        return rows
    buckets: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in rows:
        buckets[scenario(row)].append(row)
    selected: list[dict[str, Any]] = []
    order = [name for name in SCENARIO_ORDER if buckets.get(name)] + sorted(
        name for name in buckets if name not in SCENARIO_ORDER
    )
    while len(selected) < max_samples and any(buckets.values()):
        for name in order:
            if buckets.get(name):
                selected.append(buckets[name].popleft())
                if len(selected) >= max_samples:
                    break
    return selected


def replace_symlink(link_path: Path, target_path: Path, overwrite: bool) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite existing path without --overwrite: {link_path}")
        if link_path.is_dir() and not link_path.is_symlink():
            raise IsADirectoryError(f"refusing to replace directory: {link_path}")
        link_path.unlink()
    os.symlink(target_path, link_path)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def main() -> None:
    args = parse_args()
    condition_root = Path(args.condition_root)
    output_root = Path(args.output_root)
    source_jsonl = condition_root / args.split / "video_action_dataset_file.jsonl"
    if not source_jsonl.is_file():
        raise FileNotFoundError(f"missing condition JSONL: {source_jsonl}")

    rows = balanced_subset(first_unique_rows(read_jsonl(source_jsonl)), int(args.max_samples))
    failures: list[str] = []
    samples: list[dict[str, Any]] = []
    scenario_counts: dict[str, int] = defaultdict(int)
    for index, row in enumerate(rows):
        source_uuid = str(row.get("source_uuid") or row.get("source_sample_id") or row.get("uuid") or f"row_{index}")
        name = f"{index:05d}_{sanitize(source_uuid)}"
        vision_path = Path(str(row.get("vision_path") or ""))
        state_target_path = Path(str(row.get("task_state_target_path") or row.get("state_target_path") or ""))
        if int(row.get("num_video_frames", -1)) != int(args.expected_video_frames):
            failures.append(f"{name}:num_video_frames:{row.get('num_video_frames')}")
        if int(row.get("num_action_steps", row.get("action_chunk_size", -1))) != int(args.expected_action_steps):
            failures.append(f"{name}:num_action_steps:{row.get('num_action_steps', row.get('action_chunk_size'))}")
        if not vision_path.is_file():
            failures.append(f"{name}:missing_vision:{vision_path}")
        if not state_target_path.is_file():
            failures.append(f"{name}:missing_state_target:{state_target_path}")
        link_path = output_root / "inference" / name / "vision.mp4"
        if vision_path.is_file():
            replace_symlink(link_path, vision_path, bool(args.overwrite))
        row_scenario = scenario(row)
        scenario_counts[row_scenario] += 1
        samples.append(
            {
                "actor": row.get("actor", "robot_gripper_tcp"),
                "condition_root": str(condition_root),
                "evidence_boundary": (
                    "Reference-RGB readout calibration sample. GT RGB is used "
                    "only to calibrate the readout/target-motion head."
                ),
                "expected_action_steps": int(args.expected_action_steps),
                "expected_video_frames": int(args.expected_video_frames),
                "name": name,
                "prefix_frame_index": -1,
                "prefix_role": "reference_full_episode",
                "reference_rgb_calibration_video_path": str(vision_path),
                "reference_video_path": str(vision_path),
                "scenario": row_scenario,
                "source_jsonl": str(source_jsonl),
                "source_row_uuid": row.get("uuid"),
                "source_uuid": source_uuid,
                "state_target_path": str(state_target_path),
                "target_object": row.get("target_object", "hole"),
                "task_label_path": row.get("task_label_path"),
                "task_state_target_path": str(state_target_path),
                "tool_object": row.get("tool_object", "peg"),
            }
        )

    manifest = {
        "boundary": (
            "Reference-RGB readout calibration root from full-episode condition "
            "JSONL. It is not Cosmos3 generated-video evidence."
        ),
        "condition_root": str(condition_root),
        "expected_action_steps": int(args.expected_action_steps),
        "expected_video_frames": int(args.expected_video_frames),
        "input_jsonl": str(source_jsonl),
        "num_selected_samples": len(samples),
        "output_root": str(output_root),
        "reference_rgb_calibration": True,
        "role_counts": {"reference_full_episode": len(samples)},
        "samples": samples,
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "split": args.split,
        "strict_failures": failures,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "eval_input_manifest.json").write_text(json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n")
    (output_root / "reference_rgb_calibration_manifest.json").write_text(
        json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"num_selected_samples": len(samples), "scenario_counts": dict(scenario_counts), "failures": failures}, sort_keys=True))
    if failures:
        raise SystemExit("reference RGB eval root build failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
