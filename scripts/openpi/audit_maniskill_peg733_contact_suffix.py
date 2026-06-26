#!/usr/bin/env python3
"""Audit direct insertion/contact suffix windows in the accepted 733 H5 data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_H5_ROOT = (
    ROOT
    / "experiments/world_model_task_rebinding/cosmos3/"
    "fix3_v7_dp_user_override_sft_source_20260612_733/canonical_h5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-root", default=str(DEFAULT_H5_ROOT))
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--expected-episodes", type=int, default=733)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--max-start-before-insert", type=int, default=96)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--allow-login-node", action="store_true")
    return parser.parse_args()


def refuse_login_node(args: argparse.Namespace) -> None:
    if bool(args.allow_login_node):
        return
    step_id = os.environ.get("SLURM_STEP_ID", "")
    if not os.environ.get("SLURM_JOB_ID") or step_id in {"", "extern"}:
        raise SystemExit(
            "refusing_login_node_execution=true\n"
            "reason=Run contact-suffix H5 audit only inside a compute-node Slurm step."
        )


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def scenario_from_path(path: Path) -> str:
    name = path.stem
    if "_seed" in name:
        return name.split("_seed", 1)[0]
    return path.parent.name.split("_seed", 1)[0]


def read_one(path: Path, window: int, max_start_before_insert: int) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        traj = "traj_0" if "traj_0" in h5 else next(iter(h5.keys()))
        group = h5[traj]
        slots = group["slots"]
        inserted = np.asarray(slots["inserted"], dtype=bool)
        grasped = np.asarray(slots["grasped"], dtype=bool)
        rel = np.asarray(slots["peg_head_at_hole"], dtype=np.float32)
        actions = np.asarray(group["actions"], dtype=np.float32)
    first_insert = int(np.flatnonzero(inserted)[0]) if bool(np.any(inserted)) else -1
    final_inserted = bool(inserted[-1])
    final_success_like = bool(final_inserted and grasped[-1])
    candidate_starts: list[int] = []
    if first_insert >= 0:
        lo = max(0, first_insert - int(max_start_before_insert))
        hi = min(first_insert, actions.shape[0] - int(window) + 1)
        if hi >= lo:
            candidate_starts = list(range(lo, hi + 1))
    close_yz = np.sum(np.abs(rel[:, 1:3]), axis=1)
    close_contactish = (close_yz < 0.03) & grasped
    return {
        "path": str(path),
        "scenario": scenario_from_path(path),
        "actions_shape": list(actions.shape),
        "frame_count": int(inserted.shape[0]),
        "final_inserted": final_inserted,
        "final_grasped": bool(grasped[-1]),
        "final_success_like": final_success_like,
        "inserted_frame_count": int(np.sum(inserted)),
        "first_insert_frame": first_insert,
        "first_grasp_frame": int(np.flatnonzero(grasped)[0]) if bool(np.any(grasped)) else -1,
        "grasped_frame_count": int(np.sum(grasped)),
        "contactish_frame_count": int(np.sum(close_contactish)),
        "min_abs_yz_sum": float(np.min(close_yz)) if close_yz.size else None,
        "final_peg_head_at_hole": rel[-1, :3].astype(float).tolist(),
        "eligible_suffix_window_count": int(len(candidate_starts)),
        "eligible_suffix_start_min": int(candidate_starts[0]) if candidate_starts else -1,
        "eligible_suffix_start_max": int(candidate_starts[-1]) if candidate_starts else -1,
    }


def main() -> int:
    args = parse_args()
    refuse_login_node(args)
    h5_root = Path(args.h5_root).resolve()
    files = sorted(h5_root.rglob("*.h5"))
    if int(args.max_files) > 0:
        files = files[: int(args.max_files)]
    if int(args.expected_episodes) > 0 and int(args.max_files) <= 0 and len(files) != int(args.expected_episodes):
        raise RuntimeError(f"expected {args.expected_episodes} H5 files, found {len(files)} under {h5_root}")
    rows = [read_one(path, int(args.window), int(args.max_start_before_insert)) for path in files]
    scenario_counts: dict[str, int] = {}
    scenario_success_counts: dict[str, int] = {}
    for row in rows:
        scenario = str(row["scenario"])
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        scenario_success_counts[scenario] = scenario_success_counts.get(scenario, 0) + int(bool(row["final_success_like"]))
    summary = {
        "schema": "openpi_pi05_peg733_contact_suffix_audit_v1",
        "h5_root": str(h5_root),
        "episode_count": len(rows),
        "window": int(args.window),
        "max_start_before_insert": int(args.max_start_before_insert),
        "final_success_like_count": sum(bool(r["final_success_like"]) for r in rows),
        "final_inserted_count": sum(bool(r["final_inserted"]) for r in rows),
        "any_inserted_count": sum(int(r["inserted_frame_count"]) > 0 for r in rows),
        "eligible_episode_count": sum(int(r["eligible_suffix_window_count"]) > 0 for r in rows),
        "eligible_suffix_window_count": sum(int(r["eligible_suffix_window_count"]) for r in rows),
        "scenario_counts": scenario_counts,
        "scenario_success_like_counts": scenario_success_counts,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "boundary": (
            "Audit only. It identifies direct insertion/contact-positive windows "
            "inside the accepted 733 H5 data for a future OpenPI-native dataset; "
            "it trains no model and introduces no custom intermediate model."
        ),
        "rows": rows,
    }
    out = Path(args.output_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in [
        "episode_count",
        "final_success_like_count",
        "any_inserted_count",
        "eligible_episode_count",
        "eligible_suffix_window_count",
    ]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
