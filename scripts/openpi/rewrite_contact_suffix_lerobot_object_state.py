#!/usr/bin/env python3
"""Clone contact-suffix LeRobot data and replace qpos state with object state.

This is a data-conditioning repair for the OpenPI/pi0.5 contact-action branch.
It reuses the already converted RGB/action LeRobot suffix dataset and rewrites
only the low-dimensional `state` column to a causal object/task-frame vector.
No model, policy, VAE, MLP, or diffusion component is introduced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro


DEFAULT_SOURCE_ROOT = (
    "experiments/world_model_task_rebinding/openpi/lerobot_home/"
    "yanhongru/maniskill_peg733_openpi_contact_suffix16"
)
DEFAULT_SOURCE_MANIFEST = (
    "experiments/world_model_task_rebinding/openpi/"
    "pi05_peg733_contact_suffix_lerobot_full733_20260625_alloc150773/"
    "conversion_manifest.json"
)
DEFAULT_REPO_ID = "yanhongru/maniskill_peg733_openpi_contact_suffix16_object17"


@dataclass(frozen=True)
class Args:
    source_dataset_root: str = DEFAULT_SOURCE_ROOT
    source_conversion_manifest: str = DEFAULT_SOURCE_MANIFEST
    lerobot_home: str = "experiments/world_model_task_rebinding/openpi/lerobot_home"
    repo_id: str = DEFAULT_REPO_ID
    output_manifest: str | None = None
    overwrite: bool = False
    allow_login_node: bool = False


def _refuse_login_node(args: Args) -> None:
    if args.allow_login_node:
        return
    step_id = os.environ.get("SLURM_STEP_ID", "")
    if not os.environ.get("SLURM_JOB_ID") or step_id in {"", "extern"}:
        raise SystemExit(
            "refusing_login_node_execution=true\n"
            "reason=Run LeRobot state rewrite only inside a compute-node srun step."
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _copy_dataset_tree(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise SystemExit(f"output dataset exists; pass --overwrite to replace: {dst}")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _load_slots(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        traj_names = sorted([k for k in h5.keys() if k.startswith("traj_")], key=lambda x: int(x.split("_", 1)[1]))
        if len(traj_names) != 1:
            raise RuntimeError(f"{path} expected one trajectory, found {traj_names}")
        slots = h5[traj_names[0]]["slots"]
        return {
            "tcp_pose": np.asarray(slots["tcp_pose"], dtype=np.float32),
            "peg_pose": np.asarray(slots["peg_pose"], dtype=np.float32),
            "hole_pose": np.asarray(slots["hole_pose"], dtype=np.float32),
            "peg_head_at_hole": np.asarray(slots["peg_head_at_hole"], dtype=np.float32),
            "hole_velocity_step": np.asarray(slots["hole_velocity_step"], dtype=np.float32),
            "grasped": np.asarray(slots["grasped"], dtype=bool),
            "inserted": np.asarray(slots["inserted"], dtype=bool),
        }


def _pad_first(row: np.ndarray, dim: int) -> np.ndarray:
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    out = np.zeros((dim,), dtype=np.float32)
    out[: min(dim, row.shape[0])] = row[:dim]
    return out


def _object_state17(slots: dict[str, np.ndarray], frame: int) -> np.ndarray:
    t = max(0, min(int(frame), int(slots["tcp_pose"].shape[0]) - 1))
    out = np.concatenate(
        [
            _pad_first(slots["tcp_pose"][t], 3),
            _pad_first(slots["peg_pose"][t], 3),
            _pad_first(slots["hole_pose"][t], 3),
            _pad_first(slots["peg_head_at_hole"][t], 3),
            _pad_first(slots["hole_velocity_step"][t], 3),
            np.asarray([float(bool(slots["grasped"][t])), float(bool(slots["inserted"][t]))], dtype=np.float32),
        ]
    ).astype(np.float32)
    if out.shape != (17,):
        raise RuntimeError(f"object_state17 has shape {out.shape}")
    return out


def _state_stats(states: np.ndarray) -> dict[str, Any]:
    return {
        "min": np.min(states, axis=0).astype(float).tolist(),
        "max": np.max(states, axis=0).astype(float).tolist(),
        "mean": np.mean(states, axis=0).astype(float).tolist(),
        "std": np.std(states, axis=0).astype(float).tolist(),
        "count": [int(states.shape[0])],
    }


def _episode_path_by_index(root: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for path in root.glob("data/chunk-*/episode_*.parquet"):
        idx = int(path.stem.split("_", 1)[1])
        out[idx] = path
    return out


def _fixed_size_state_array(states: np.ndarray) -> pa.FixedSizeListArray:
    flat = pa.array(states.reshape(-1).astype(np.float32), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, 17)


def main(args: Args) -> None:
    _refuse_login_node(args)
    root = Path.cwd()
    source_root = Path(args.source_dataset_root).resolve()
    source_manifest_path = Path(args.source_conversion_manifest).resolve()
    lerobot_home = Path(args.lerobot_home).resolve()
    output_root = lerobot_home / args.repo_id
    output_manifest = (
        Path(args.output_manifest).resolve()
        if args.output_manifest
        else root
        / "experiments/world_model_task_rebinding/openpi"
        / f"{args.repo_id.replace('/', '_')}_rewrite_manifest.json"
    )

    source_manifest = json.loads(source_manifest_path.read_text())
    records = list(source_manifest.get("records") or [])
    if not records:
        raise RuntimeError(f"source manifest has no records: {source_manifest_path}")

    _copy_dataset_tree(source_root, output_root, bool(args.overwrite))
    episode_paths = _episode_path_by_index(output_root)
    if len(episode_paths) != len(records):
        raise RuntimeError(f"episode parquet count {len(episode_paths)} != manifest records {len(records)}")

    stats_rows = _read_jsonl(output_root / "meta" / "episodes_stats.jsonl")
    stats_by_episode = {int(row["episode_index"]): row for row in stats_rows if "episode_index" in row}
    h5_cache: dict[str, dict[str, np.ndarray]] = {}
    rewritten = 0
    first_rows: list[dict[str, Any]] = []
    for episode_idx, record in enumerate(records):
        path = episode_paths[episode_idx]
        h5_path = str(Path(record["input_h5"]).resolve())
        slots = h5_cache.get(h5_path)
        if slots is None:
            slots = _load_slots(Path(h5_path))
            h5_cache[h5_path] = slots
        start = int(record["suffix_start_frame"])
        table = pq.read_table(path)
        states = np.stack([_object_state17(slots, start + i) for i in range(table.num_rows)], axis=0)
        state_col = _fixed_size_state_array(states)
        state_idx = table.column_names.index("state")
        table = table.set_column(state_idx, "state", state_col).replace_schema_metadata(None)
        pq.write_table(table, path)
        if episode_idx in stats_by_episode:
            stats_by_episode[episode_idx].setdefault("stats", {})["state"] = _state_stats(states)
        rewritten += 1
        if len(first_rows) < 5:
            first_rows.append(
                {
                    "episode_index": episode_idx,
                    "parquet": str(path),
                    "source_h5": h5_path,
                    "suffix_start_frame": start,
                    "state_first": states[0].astype(float).tolist(),
                    "state_last": states[-1].astype(float).tolist(),
                }
            )

    if stats_rows:
        _write_jsonl(output_root / "meta" / "episodes_stats.jsonl", [stats_by_episode[i] for i in sorted(stats_by_episode)])

    info_path = output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["repo_id"] = args.repo_id
    info["features"]["state"] = {"dtype": "float32", "shape": [17], "names": ["state"]}
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")

    manifest = dict(source_manifest)
    manifest.update(
        {
            "schema": "openpi_lerobot_maniskill_peg733_contact_suffix16_object17_rewrite_v1",
            "repo_id": args.repo_id,
            "output_path": str(output_root),
            "source_dataset_root": str(source_root),
            "source_conversion_manifest": str(source_manifest_path),
            "state_mode": "object_state17",
            "state_semantics": (
                "object_state17=tcp_xyz,peg_xyz,hole_xyz,peg_head_at_hole_xyz,"
                "hole_velocity_step_xyz,grasped,inserted"
            ),
            "features": {
                **dict(source_manifest.get("features") or {}),
                "state": [17],
            },
            "rewritten_episode_count": rewritten,
            "unique_h5_loaded": len(h5_cache),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "boundary": (
                "Rewrites only the LeRobot low-dimensional state column from "
                "causal source H5 slots. Images, actions, task prompts, and "
                "official OpenPI/LeRobot format are preserved. No custom model "
                "or privileged future runtime condition is introduced."
            ),
            "args": asdict(args),
            "first_rows": first_rows,
        }
    )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "output_path": str(output_root),
                "rewritten_episode_count": rewritten,
                "state_dim": 17,
                "output_manifest": str(output_manifest),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    tyro.cli(main)
