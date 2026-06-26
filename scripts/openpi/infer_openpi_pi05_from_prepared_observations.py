#!/usr/bin/env python3
"""Run official OpenPI pi0.5 inference on prepared snapshot observations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

OPENPI_ROOT = Path(os.environ.get("OPENPI_ROOT", "/public/home/yanhongru/ICLR2027/openpi")).resolve()
if str(OPENPI_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(OPENPI_ROOT / "src"))


DEFAULT_CHECKPOINT = (
    Path("/public/home/yanhongru/ICLR2027/Reflex")
    / "experiments/world_model_task_rebinding/openpi/checkpoints_local_preserved/pi05_maniskill_peg733/"
    "pi05_peg733_1gpu1h_20260625_after_nfs_enolck_localckpt_1600_skipnorm_alloc150773/1599"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-name", default="pi05_maniskill_peg733")
    parser.add_argument("--robot-action-dim", type=int, default=7)
    parser.add_argument("--execute-steps", type=int, default=8)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_openpi_policy(args: argparse.Namespace) -> Any:
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    train_config = openpi_config.get_config(str(args.config_name))
    return policy_config.create_trained_policy(train_config, Path(args.checkpoint_dir).resolve())


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    action_dir = output_root / "action_chunks"
    action_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(Path(args.prepared_manifest).resolve())
    rows = list(manifest.get("rows") or [])
    policy = load_openpi_policy(args)
    action_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        obs_npz = np.load(str(row["observation_npz"]))
        obs = {
            "observation/state": np.asarray(obs_npz["state"], dtype=np.float32),
            "observation/image": np.asarray(obs_npz["image"], dtype=np.uint8),
            "observation/wrist_image": np.asarray(obs_npz["wrist_image"], dtype=np.uint8),
            "prompt": str(row["prompt"]),
        }
        result = policy.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < int(args.robot_action_dim):
            raise RuntimeError(f"OpenPI returned invalid action shape {actions.shape}")
        actions = actions[:, : int(args.robot_action_dim)]
        execute_steps = min(int(args.execute_steps), int(actions.shape[0]))
        out = {
            "schema": "openpi_pi05_snapshot_action_chunk_v1",
            "prepared_observation": row,
            "source_h5": row["source_h5"],
            "snapshot_state_h5": row["snapshot_state_h5"],
            "history_action_state_json": row["history_action_state_json"],
            "prefix_frame_index": int(row["prefix_frame_index"]),
            "iteration": int(row["iteration"]),
            "prompt": str(row["prompt"]),
            "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
            "config_name": str(args.config_name),
            "policy_timing": result.get("policy_timing"),
            "denormalized_robot_action_chunk": actions.astype(float).tolist(),
            "action_shape": list(actions.shape),
            "execute_steps": int(execute_steps),
        }
        path = action_dir / f"openpi_pi05_{index:04d}_iter_{int(row['iteration']):03d}_f{int(row['prefix_frame_index']):03d}.action_chunk.json"
        write_json(path, out)
        action_rows.append({"action_chunk_json": str(path), **out})

    summary = {
        "schema": "openpi_pi05_prepared_observation_inference_summary_v1",
        "prepared_manifest": str(Path(args.prepared_manifest).resolve()),
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "output_root": str(output_root),
        "action_count": len(action_rows),
        "actions": action_rows,
    }
    write_json(output_root / "openpi_pi05_prepared_inference_summary.json", summary)
    print(json.dumps({"action_count": len(action_rows), "summary": str(output_root / "openpi_pi05_prepared_inference_summary.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
