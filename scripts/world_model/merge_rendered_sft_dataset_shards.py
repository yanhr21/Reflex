#!/usr/bin/env python3
"""Merge sharded render outputs from render_cosmos3_maniskill_sft_dataset.py.

Each render shard writes:
  - shards/<shard_id>_manifest.json
  - train/video_dataset_file.<shard_id>.jsonl
  - val/video_dataset_file.<shard_id>.jsonl

This script writes the canonical:
  - manifest.json
  - train/video_dataset_file.jsonl
  - val/video_dataset_file.jsonl
without changing rendered videos.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"manifest is not a dict: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--expected-frames", type=int, default=301)
    parser.add_argument("--expected-fps", type=int, default=30)
    parser.add_argument("--require-contiguous-indexes", action="store_true")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    shard_dir = root / "shards"
    manifests = sorted(shard_dir.glob("*_manifest.json"))
    if not manifests:
        raise RuntimeError(f"no shard manifests found under {shard_dir}")

    merged_videos: list[dict[str, Any]] = []
    records_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    shard_payloads = []
    for manifest_path in manifests:
        payload = _load_manifest(manifest_path)
        shard_id = str(payload.get("shard_id") or manifest_path.name.replace("_manifest.json", ""))
        shard_payloads.append(payload)
        videos = payload.get("videos") or []
        if not isinstance(videos, list):
            raise RuntimeError(f"videos is not a list in {manifest_path}")
        merged_videos.extend(videos)
        for split in ("train", "val"):
            records_by_split[split].extend(_read_jsonl(root / split / f"video_dataset_file.{shard_id}.jsonl"))

    indexes = [int(item["index"]) for item in merged_videos]
    duplicate_indexes = [index for index, count in Counter(indexes).items() if count > 1]
    if duplicate_indexes:
        raise RuntimeError(f"duplicate video indexes: {duplicate_indexes[:20]}")
    merged_videos.sort(key=lambda item: int(item["index"]))

    if args.expected_count and len(merged_videos) != args.expected_count:
        raise RuntimeError(f"merged video count {len(merged_videos)} != expected {args.expected_count}")
    if args.require_contiguous_indexes:
        expected_indexes = list(range(len(merged_videos)))
        if indexes and [int(item["index"]) for item in merged_videos] != expected_indexes:
            raise RuntimeError("merged video indexes are not contiguous from zero")

    failures: list[dict[str, Any]] = []
    for item in merged_videos:
        if int(item.get("num_video_frames", -1)) != int(args.expected_frames):
            failures.append({"index": item.get("index"), "reason": f"num_video_frames={item.get('num_video_frames')}"})
        if int(item.get("fps", -1)) != int(args.expected_fps):
            failures.append({"index": item.get("index"), "reason": f"fps={item.get('fps')}"})
        video_path = Path(str(item.get("video", "")))
        if not video_path.exists():
            failures.append({"index": item.get("index"), "reason": f"missing_video={video_path}"})
        h5_path = Path(str(item.get("input_h5", "")))
        if not h5_path.exists():
            failures.append({"index": item.get("index"), "reason": f"missing_input_h5={h5_path}"})
    if failures:
        raise RuntimeError(f"merge validation failed: {failures[:20]}")

    scenario_counts = Counter(str(item.get("scenario", "unknown")) for item in merged_videos)
    num_train = sum(1 for item in merged_videos if str(item.get("split")) == "train")
    num_val = sum(1 for item in merged_videos if str(item.get("split")) == "val")
    base = dict(shard_payloads[0])
    base.update(
        {
            "schema": "rendered_sft_dataset_shard_merge_v1",
            "canonical_metadata_written": True,
            "merged_from_shards": [str(path) for path in manifests],
            "output_root": str(root),
            "num_total_paths": len(merged_videos),
            "start_index": 0,
            "end_index": len(merged_videos),
            "num_videos": len(merged_videos),
            "num_train": num_train,
            "num_val": num_val,
            "num_skipped_existing": sum(int(payload.get("num_skipped_existing", 0)) for payload in shard_payloads),
            "scenario_counts": dict(sorted(scenario_counts.items())),
            "videos": merged_videos,
        }
    )
    base.pop("shard_id", None)
    base.pop("canonical_metadata_written", None)
    base["canonical_metadata_written"] = True

    _write_jsonl(root / "train" / "video_dataset_file.jsonl", records_by_split["train"])
    _write_jsonl(root / "val" / "video_dataset_file.jsonl", records_by_split["val"])
    (root / "manifest.json").write_text(json.dumps(base, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "event": "render_shards_merged",
                "dataset_root": str(root),
                "num_videos": len(merged_videos),
                "num_train": num_train,
                "num_val": num_val,
                "scenario_counts": dict(sorted(scenario_counts.items())),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
