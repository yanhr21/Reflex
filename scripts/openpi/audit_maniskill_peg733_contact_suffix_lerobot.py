#!/usr/bin/env python3
"""Audit the contact-suffix LeRobot dataset for OpenPI pi0.5 training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import tyro


DEFAULT_DATASET_ROOT = (
    "experiments/world_model_task_rebinding/openpi/lerobot_home/"
    "yanhongru/maniskill_peg733_openpi_contact_suffix16"
)
DEFAULT_CONVERSION_MANIFEST = (
    "experiments/world_model_task_rebinding/openpi/"
    "pi05_peg733_contact_suffix_lerobot_full733_20260625_alloc150773/"
    "conversion_manifest.json"
)


@dataclass(frozen=True)
class Args:
    dataset_root: str = DEFAULT_DATASET_ROOT
    conversion_manifest: str = DEFAULT_CONVERSION_MANIFEST
    output_json: str | None = None
    expected_source_episodes: int = 733
    expected_suffix_episodes: int = 5853
    expected_episode_length: int = 16
    expected_total_frames: int = 93648
    expected_image_shape: tuple[int, int, int] = (256, 256, 3)
    expected_state_dim: int = 8
    expected_action_dim: int = 7
    allow_login_node: bool = False


def _refuse_login_node(args: Args) -> None:
    if args.allow_login_node:
        return
    step_id = os.environ.get("SLURM_STEP_ID", "")
    if not os.environ.get("SLURM_JOB_ID") or step_id in {"", "extern"}:
        raise SystemExit(
            "refusing_login_node_execution=true\n"
            "reason=Run contact-suffix LeRobot audit only inside a compute-node srun step."
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _shape_from_feature(feature: dict[str, Any]) -> list[int] | None:
    shape = feature.get("shape")
    if isinstance(shape, list):
        return [int(x) for x in shape]
    return None


def _is_float_vector(type_text: str, expected_dim: int) -> bool:
    return (
        f"fixed_size_list<item: float>[{expected_dim}]" in type_text
        or f"fixed_size_list<element: float>[{expected_dim}]" in type_text
        or f"list<item: float>" in type_text
        or f"list<element: float>" in type_text
    )


def main(args: Args) -> None:
    _refuse_login_node(args)
    root = Path(args.dataset_root).resolve()
    manifest_path = Path(args.conversion_manifest).resolve()
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    tasks_path = root / "meta" / "tasks.jsonl"

    failures: list[str] = []
    if not info_path.exists():
        failures.append(f"missing info.json: {info_path}")
    if not manifest_path.exists():
        failures.append(f"missing conversion manifest: {manifest_path}")
    if failures:
        summary = {"failures": failures, "passed": False}
        out = Path(args.output_json).resolve() if args.output_json else root / "meta" / "contact_suffix_audit.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
        raise SystemExit(1)

    info = json.loads(info_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    episodes = _read_jsonl(episodes_path)
    tasks = _read_jsonl(tasks_path)
    parquet_files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    video_files = sorted((root / "videos").glob("chunk-*/*/episode_*.mp4"))

    total_rows = 0
    observed_lengths: list[int] = []
    first_parquet_summaries: list[dict[str, Any]] = []
    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path)
        total_rows += table.num_rows
        observed_lengths.append(table.num_rows)
        names = set(table.column_names)
        required = {"state", "actions", "episode_index", "frame_index", "task_index"}
        if all(info.get("features", {}).get(key, {}).get("dtype") == "image" for key in ("image", "wrist_image")):
            required.update({"image", "wrist_image"})
        missing = sorted(required - names)
        if missing:
            failures.append(f"{parquet_path}: missing columns {missing}")
        for column, expected_dim in (("state", args.expected_state_dim), ("actions", args.expected_action_dim)):
            if column in names:
                col_type = str(table.schema.field(column).type)
                if not _is_float_vector(col_type, expected_dim):
                    failures.append(f"{parquet_path}: unexpected {column} type {col_type}")
        if len(first_parquet_summaries) < 3:
            first_parquet_summaries.append(
                {"path": str(parquet_path), "rows": table.num_rows, "columns": table.column_names}
            )

    features = info.get("features", {})
    camera_dtypes = {image_key: features.get(image_key, {}).get("dtype") for image_key in ("image", "wrist_image")}
    for image_key in ("image", "wrist_image"):
        shape = _shape_from_feature(features.get(image_key, {}))
        if shape != list(args.expected_image_shape):
            failures.append(f"{image_key} shape expected {args.expected_image_shape}, got {shape}")
        if camera_dtypes[image_key] not in {"image", "video"}:
            failures.append(f"{image_key} dtype expected image or video, got {camera_dtypes[image_key]}")
    if any(dtype == "video" for dtype in camera_dtypes.values()):
        if set(camera_dtypes.values()) != {"video"}:
            failures.append(f"mixed camera storage is unsupported: {camera_dtypes}")
        expected_video_files = int(args.expected_suffix_episodes) * 2
        if len(video_files) != expected_video_files:
            failures.append(f"video file count expected {expected_video_files}, got {len(video_files)}")
        if int(info.get("total_videos", -1)) != expected_video_files:
            failures.append(f"info.total_videos expected {expected_video_files}, got {info.get('total_videos')}")
    if _shape_from_feature(features.get("state", {})) != [args.expected_state_dim]:
        failures.append(f"state feature shape mismatch: {features.get('state')}")
    if _shape_from_feature(features.get("actions", {})) != [args.expected_action_dim]:
        failures.append(f"actions feature shape mismatch: {features.get('actions')}")

    checks = {
        "info.total_episodes": (info.get("total_episodes"), args.expected_suffix_episodes),
        "info.total_frames": (info.get("total_frames"), args.expected_total_frames),
        "episodes_jsonl_rows": (len(episodes), args.expected_suffix_episodes),
        "parquet_file_count": (len(parquet_files), args.expected_suffix_episodes),
        "total_parquet_rows": (total_rows, args.expected_total_frames),
        "manifest.num_source_episodes_read": (
            manifest.get("num_source_episodes_read"),
            args.expected_source_episodes,
        ),
        "manifest.num_suffix_episodes_written": (
            manifest.get("num_suffix_episodes_written"),
            args.expected_suffix_episodes,
        ),
        "manifest.frames_per_suffix_episode": (
            manifest.get("frames_per_suffix_episode"),
            args.expected_episode_length,
        ),
        "manifest.total_frames_written": (manifest.get("total_frames_written"), args.expected_total_frames),
    }
    for key, (actual, expected) in checks.items():
        if int(actual if actual is not None else -1) != int(expected):
            failures.append(f"{key} expected {expected}, got {actual}")

    bad_lengths = sorted({x for x in observed_lengths if x != args.expected_episode_length})
    if bad_lengths:
        failures.append(f"episode length mismatch values: {bad_lengths}")

    records = manifest.get("records", [])
    end_inserted_count = sum(bool(record.get("end_inserted")) for record in records if isinstance(record, dict))
    start_grasped_count = sum(bool(record.get("start_grasped")) for record in records if isinstance(record, dict))

    summary = {
        "schema": "openpi_lerobot_maniskill_peg733_contact_suffix16_audit_v1",
        "dataset_root": str(root),
        "conversion_manifest": str(manifest_path),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "host": os.uname().nodename,
        "expected": asdict(args),
        "info_total_episodes": info.get("total_episodes"),
        "info_total_frames": info.get("total_frames"),
        "episodes_jsonl_rows": len(episodes),
        "tasks_jsonl_rows": len(tasks),
        "parquet_file_count": len(parquet_files),
        "video_file_count": len(video_files),
        "total_parquet_rows": total_rows,
        "unique_episode_lengths": sorted(set(observed_lengths)),
        "features": features,
        "manifest_summary": {
            key: manifest.get(key)
            for key in (
                "repo_id",
                "num_source_episodes_read",
                "num_suffix_episodes_written",
                "frames_per_suffix_episode",
                "total_frames_written",
                "offsets_before_insert",
                "skipped_no_insert",
                "skipped_no_window",
                "camera_storage",
            )
        },
        "camera_dtypes": camera_dtypes,
        "record_end_inserted_count": end_inserted_count,
        "record_start_grasped_count": start_grasped_count,
        "first_parquet_files": first_parquet_summaries,
        "failures": failures,
        "passed": not failures,
    }
    out = Path(args.output_json).resolve() if args.output_json else root / "meta" / "contact_suffix_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    tyro.cli(main)
