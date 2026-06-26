#!/usr/bin/env python3
"""Audit the converted ManiSkill peg-insertion LeRobot dataset.

This is a data-preflight check only. It must run inside a Slurm compute step
because reading hundreds of parquet files and image records is project compute.
"""

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
    "yanhongru/maniskill_peg733_openpi_libero"
)


@dataclass(frozen=True)
class Args:
    dataset_root: str = DEFAULT_DATASET_ROOT
    conversion_manifest: str | None = None
    output_json: str | None = None
    expected_episodes: int = 733
    expected_episode_length: int = 300
    expected_total_frames: int = 219900
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
            "reason=Run LeRobot dataset audit only inside a compute-node srun step."
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _shape_from_feature(feature: dict[str, Any]) -> list[int] | None:
    shape = feature.get("shape")
    if isinstance(shape, list):
        return [int(x) for x in shape]
    return None


def _is_fixed_float_vector(type_text: str, expected_dim: int) -> bool:
    return (
        f"fixed_size_list<item: float>[{expected_dim}]" in type_text
        or f"fixed_size_list<element: float>[{expected_dim}]" in type_text
        or f"list<item: float>" in type_text
        or f"list<element: float>" in type_text
    )


def main(args: Args) -> None:
    _refuse_login_node(args)

    root = Path(args.dataset_root).resolve()
    meta_dir = root / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    tasks_path = meta_dir / "tasks.jsonl"
    if not info_path.exists():
        raise RuntimeError(f"missing LeRobot info.json: {info_path}")

    info = json.loads(info_path.read_text())
    episodes = _read_jsonl(episodes_path)
    tasks = _read_jsonl(tasks_path)
    parquet_files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))

    failures: list[str] = []
    total_rows = 0
    observed_lengths: list[int] = []
    parquet_summaries: list[dict[str, Any]] = []

    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path)
        total_rows += table.num_rows
        observed_lengths.append(table.num_rows)
        names = set(table.column_names)
        required = {"image", "wrist_image", "state", "actions", "episode_index", "frame_index", "task_index"}
        missing = sorted(required - names)
        if missing:
            failures.append(f"{parquet_path}: missing columns {missing}")
        for column, expected_dim in (("state", args.expected_state_dim), ("actions", args.expected_action_dim)):
            if column in names:
                col_type = str(table.schema.field(column).type)
                if not _is_fixed_float_vector(col_type, expected_dim):
                    failures.append(f"{parquet_path}: unexpected {column} type {col_type}")
        parquet_summaries.append(
            {
                "path": str(parquet_path),
                "rows": table.num_rows,
                "columns": table.column_names,
            }
        )

    features = info.get("features", {})
    for image_key in ("image", "wrist_image"):
        shape = _shape_from_feature(features.get(image_key, {}))
        if shape != list(args.expected_image_shape):
            failures.append(f"{image_key} shape mismatch: expected {args.expected_image_shape}, got {shape}")
    if _shape_from_feature(features.get("state", {})) != [args.expected_state_dim]:
        failures.append(f"state feature shape mismatch: {features.get('state')}")
    if _shape_from_feature(features.get("actions", {})) != [args.expected_action_dim]:
        failures.append(f"actions feature shape mismatch: {features.get('actions')}")

    if int(info.get("total_episodes", -1)) != args.expected_episodes:
        failures.append(f"info.total_episodes expected {args.expected_episodes}, got {info.get('total_episodes')}")
    if int(info.get("total_frames", -1)) != args.expected_total_frames:
        failures.append(f"info.total_frames expected {args.expected_total_frames}, got {info.get('total_frames')}")
    if len(episodes) != args.expected_episodes:
        failures.append(f"episodes.jsonl rows expected {args.expected_episodes}, got {len(episodes)}")
    if len(parquet_files) != args.expected_episodes:
        failures.append(f"parquet files expected {args.expected_episodes}, got {len(parquet_files)}")
    if total_rows != args.expected_total_frames:
        failures.append(f"total parquet rows expected {args.expected_total_frames}, got {total_rows}")
    bad_lengths = sorted({x for x in observed_lengths if x != args.expected_episode_length})
    if bad_lengths:
        failures.append(f"episode length mismatch values: {bad_lengths}")

    conversion_manifest: dict[str, Any] | None = None
    if args.conversion_manifest:
        manifest_path = Path(args.conversion_manifest).resolve()
        if not manifest_path.exists():
            failures.append(f"missing conversion manifest: {manifest_path}")
        else:
            conversion_manifest = json.loads(manifest_path.read_text())
            if int(conversion_manifest.get("num_episodes", -1)) != args.expected_episodes:
                failures.append(
                    "conversion manifest num_episodes expected "
                    f"{args.expected_episodes}, got {conversion_manifest.get('num_episodes')}"
                )
            if int(conversion_manifest.get("frames_per_episode_written", -1)) != args.expected_episode_length:
                failures.append(
                    "conversion manifest frames_per_episode_written expected "
                    f"{args.expected_episode_length}, got {conversion_manifest.get('frames_per_episode_written')}"
                )

    summary = {
        "schema": "openpi_lerobot_maniskill_peg733_audit_v1",
        "dataset_root": str(root),
        "conversion_manifest": args.conversion_manifest,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "host": os.uname().nodename,
        "expected": asdict(args),
        "info_total_episodes": info.get("total_episodes"),
        "info_total_frames": info.get("total_frames"),
        "episodes_jsonl_rows": len(episodes),
        "tasks_jsonl_rows": len(tasks),
        "parquet_file_count": len(parquet_files),
        "total_parquet_rows": total_rows,
        "unique_episode_lengths": sorted(set(observed_lengths)),
        "features": features,
        "conversion_manifest_summary": {
            k: conversion_manifest.get(k)
            for k in ("num_episodes", "frames_per_episode_written", "source_frames_per_episode", "fps", "features")
        }
        if conversion_manifest
        else None,
        "first_parquet_files": parquet_summaries[:3],
        "failures": failures,
        "passed": not failures,
    }

    out_path = Path(args.output_json).resolve() if args.output_json else root / "meta" / "maniskill_peg733_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    tyro.cli(main)
