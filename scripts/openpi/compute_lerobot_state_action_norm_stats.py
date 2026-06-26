#!/usr/bin/env python3
"""Compute OpenPI state/action norm stats directly from LeRobot parquet files.

This is a data-loading fallback for cases where the official
scripts/compute_norm_stats.py hangs before reaching its state/action iterator.
It deliberately uses OpenPI's own normalize.RunningStats and normalize.save so
the produced asset has the same on-disk schema as the official script.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
OPENPI_ROOT = Path(os.environ.get("OPENPI_ROOT", "/public/home/yanhongru/ICLR2027/openpi")).resolve()
if str(OPENPI_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(OPENPI_ROOT / "src"))

from openpi.shared import normalize  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default=str(
            ROOT
            / "experiments/world_model_task_rebinding/openpi/lerobot_home"
            / "yanhongru/maniskill_peg733_openpi_contact_suffix16_object17"
        ),
    )
    parser.add_argument("--config-name", default="pi05_maniskill_peg733_contact_suffix16_object17")
    parser.add_argument("--repo-id", default="yanhongru/maniskill_peg733_openpi_contact_suffix16_object17")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--expected-state-dim", type=int, default=17)
    parser.add_argument("--expected-action-dim", type=int, default=7)
    parser.add_argument("--expected-total-frames", type=int, default=93648)
    parser.add_argument("--allow-login-node", action="store_true")
    return parser.parse_args()


def refuse_login_node(args: argparse.Namespace) -> None:
    if args.allow_login_node:
        return
    step_id = os.environ.get("SLURM_STEP_ID", "")
    if not os.environ.get("SLURM_JOB_ID") or step_id in {"", "extern"}:
        raise SystemExit(
            "refusing_login_node_execution=true\n"
            "reason=Run LeRobot norm-stat fallback only inside a compute-node srun step."
        )


def fixed_list_to_numpy(column: pa.ChunkedArray, expected_dim: int, name: str) -> np.ndarray:
    arr = column.combine_chunks()
    if pa.types.is_fixed_size_list(arr.type):
        dim = int(arr.type.list_size)
        if dim != expected_dim:
            raise RuntimeError(f"{name} dim expected {expected_dim}, got {dim}")
        return np.asarray(arr.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1, dim)
    values = np.asarray(arr.to_pylist(), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != expected_dim:
        raise RuntimeError(f"{name} expected [N,{expected_dim}], got {values.shape}")
    return values


def main() -> int:
    args = parse_args()
    refuse_login_node(args)

    dataset_root = Path(args.dataset_root).resolve()
    parquet_files = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"no parquet files found under {dataset_root / 'data'}")

    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    total_rows = 0
    first_files: list[dict[str, Any]] = []
    for path in tqdm(parquet_files, desc="Computing LeRobot state/action norm stats"):
        table = pq.read_table(path, columns=["state", "actions"])
        state = fixed_list_to_numpy(table["state"], int(args.expected_state_dim), "state")
        actions = fixed_list_to_numpy(table["actions"], int(args.expected_action_dim), "actions")
        if state.shape[0] != actions.shape[0]:
            raise RuntimeError(f"{path}: state rows {state.shape[0]} != action rows {actions.shape[0]}")
        stats["state"].update(state)
        stats["actions"].update(actions)
        total_rows += int(state.shape[0])
        if len(first_files) < 3:
            first_files.append({"path": str(path), "rows": int(state.shape[0])})

    if int(args.expected_total_frames) and total_rows != int(args.expected_total_frames):
        raise RuntimeError(f"total rows expected {args.expected_total_frames}, got {total_rows}")

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    output_dir = (
        Path(args.output_dir).resolve()
        if str(args.output_dir).strip()
        else OPENPI_ROOT / "assets" / str(args.config_name) / str(args.repo_id)
    )
    normalize.save(output_dir, norm_stats)

    summary = {
        "schema": "openpi_lerobot_state_action_norm_stats_fallback_v1",
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "norm_stats_json": str(output_dir / "norm_stats.json"),
        "config_name": str(args.config_name),
        "repo_id": str(args.repo_id),
        "parquet_file_count": len(parquet_files),
        "total_rows": total_rows,
        "state_dim": int(args.expected_state_dim),
        "action_dim": int(args.expected_action_dim),
        "first_files": first_files,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "host": os.uname().nodename,
        "boundary": (
            "Fallback normalization path for OpenPI data assets. It uses OpenPI "
            "normalize.RunningStats/save on LeRobot parquet state/actions only; "
            "no model, policy, or action representation is changed."
        ),
    }
    summary_path = Path(args.output_summary).resolve() if str(args.output_summary).strip() else output_dir / "norm_stats_fallback_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"norm_stats_json": str(output_dir / "norm_stats.json"), "total_rows": total_rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
