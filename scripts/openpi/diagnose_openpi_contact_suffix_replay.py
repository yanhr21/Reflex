#!/usr/bin/env python3
"""Diagnose OpenPI contact-suffix replay failures from saved artifacts.

This is an evidence summarizer only. It reads action chunks, replay labels, and
source H5 slots to quantify whether the trained OpenPI suffix policy improves
the peg-hole relative state from dynamic snapshots.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np


DEFAULT_REPLAY_ROOT = (
    Path("/public/home/yanhongru/ICLR2027/Reflex")
    / "experiments/world_model_task_rebinding/openpi/"
    "openpi_pi05_snapshot_replay_20260626_contact_suffix1699_panel4_exec16_alloc150773"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", default=str(DEFAULT_REPLAY_ROOT))
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--allow-login-node", action="store_true")
    return parser.parse_args()


def refuse_login_node(args: argparse.Namespace) -> None:
    if args.allow_login_node:
        return
    step_id = os.environ.get("SLURM_STEP_ID", "")
    if not os.environ.get("SLURM_JOB_ID") or step_id in {"", "extern"}:
        raise SystemExit(
            "refusing_login_node_execution=true\n"
            "reason=Run OpenPI replay diagnostics only inside a compute-node Slurm step."
        )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


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


def load_source_slots(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        traj_names = sorted([k for k in h5.keys() if k.startswith("traj_")], key=lambda x: int(x.split("_", 1)[1]))
        traj = traj_names[0] if traj_names else next(iter(h5.keys()))
        group = h5[traj]
        slots = group["slots"]
        return {
            "actions": np.asarray(group["actions"], dtype=np.float32),
            "inserted": np.asarray(slots["inserted"], dtype=bool),
            "grasped": np.asarray(slots["grasped"], dtype=bool),
            "peg_head_at_hole": np.asarray(slots["peg_head_at_hole"], dtype=np.float32),
        }


def action_stats(actions: np.ndarray) -> dict[str, Any]:
    if actions.size == 0:
        return {}
    xyz = actions[:, :3]
    rot = actions[:, 3:6] if actions.shape[1] >= 6 else np.zeros((actions.shape[0], 0), dtype=np.float32)
    grip = actions[:, 6] if actions.shape[1] >= 7 else np.zeros((actions.shape[0],), dtype=np.float32)
    return {
        "shape": list(actions.shape),
        "xyz_mean": np.mean(xyz, axis=0),
        "xyz_sum": np.sum(xyz, axis=0),
        "xyz_abs_mean": np.mean(np.abs(xyz), axis=0),
        "xyz_l2_mean": float(np.mean(np.linalg.norm(xyz, axis=1))),
        "rot_mean": np.mean(rot, axis=0) if rot.size else [],
        "gripper_mean": float(np.mean(grip)) if grip.size else None,
        "gripper_min": float(np.min(grip)) if grip.size else None,
        "gripper_max": float(np.max(grip)) if grip.size else None,
    }


def cosine_flat(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.shape != b.shape or a.size == 0:
        return None
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom <= 1e-12:
        return None
    return float(np.dot(af, bf) / denom)


def label_for_action(replay_root: Path, action_path: Path) -> Path:
    base = action_path.name.replace(".action_chunk.json", "")
    return replay_root / "replay" / base / "policy_droid_snapshot_action_replay_label.json"


def main() -> int:
    args = parse_args()
    refuse_login_node(args)
    replay_root = Path(args.replay_root).resolve()
    action_paths = sorted((replay_root / "action_chunks").glob("*.action_chunk.json"))
    if not action_paths:
        raise RuntimeError(f"no action chunks found under {replay_root / 'action_chunks'}")

    rows: list[dict[str, Any]] = []
    for action_path in action_paths:
        action = read_json(action_path)
        label_path = label_for_action(replay_root, action_path)
        if not label_path.exists():
            raise RuntimeError(f"missing label for {action_path}: {label_path}")
        label = read_json(label_path)
        source_h5 = Path(action["source_h5"]).resolve()
        source = load_source_slots(source_h5)
        prefix = int(action["prefix_frame_index"])
        execute_steps = int(action.get("execute_steps") or label.get("execute_steps_actual") or 0)
        pred = np.asarray(action["denormalized_robot_action_chunk"], dtype=np.float32)[:execute_steps, :7]
        source_actions = source["actions"]
        source_window = source_actions[prefix : min(prefix + execute_steps, source_actions.shape[0]), :7]
        if source_window.shape[0] != pred.shape[0]:
            source_window = np.zeros_like(pred)
            same_time_source_available = False
        else:
            same_time_source_available = True

        inserted = source["inserted"]
        inserted_idx = np.flatnonzero(inserted)
        first_insert = int(inserted_idx[0]) if inserted_idx.size else -1
        offset_to_first_insert = first_insert - prefix if first_insert >= 0 else None
        rel = source["peg_head_at_hole"]
        rel_prefix = rel[min(prefix, rel.shape[0] - 1), :3]
        rel_first_insert = rel[min(first_insert, rel.shape[0] - 1), :3] if first_insert >= 0 else np.full((3,), np.nan)

        before = label.get("before_rel_metrics") or {}
        after = label.get("after_rel_metrics") or {}
        before_yz = float(before.get("abs_yz_sum", np.nan))
        after_yz = float(after.get("abs_yz_sum", np.nan))
        before_x = float(before.get("abs_x", np.nan))
        after_x = float(after.get("abs_x", np.nan))

        row = {
            "action_chunk_json": str(action_path),
            "label_path": str(label_path),
            "sample": action_path.stem.replace(".action_chunk", ""),
            "source_h5": str(source_h5),
            "prefix_frame_index": prefix,
            "first_insert_frame": first_insert,
            "offset_to_first_insert": offset_to_first_insert,
            "is_training_offset": offset_to_first_insert in {64, 48, 32, 24, 16, 12, 8, 4}
            if offset_to_first_insert is not None
            else False,
            "execute_steps": execute_steps,
            "after_success": bool(label.get("after_success")),
            "after_inserted_live_pose": bool(label.get("after_inserted_live_pose")),
            "after_contact_stable_proxy": bool(label.get("after_contact_stable_proxy")),
            "after_grasped": bool(label.get("after_grasped")),
            "dp96_success": bool((label.get("dp_rollout_continuability") or {}).get("success")),
            "dp96_continuable": bool((label.get("dp_rollout_continuability") or {}).get("continuable")),
            "before_abs_x": before_x,
            "after_abs_x": after_x,
            "delta_abs_x": after_x - before_x,
            "before_abs_yz_sum": before_yz,
            "after_abs_yz_sum": after_yz,
            "delta_abs_yz_sum": after_yz - before_yz,
            "source_rel_prefix": rel_prefix,
            "source_rel_first_insert": rel_first_insert,
            "pred_action_stats": action_stats(pred),
            "source_same_time_action_stats": action_stats(source_window),
            "same_time_source_available": same_time_source_available,
            "pred_vs_source_same_time_l2_mean": float(np.mean(np.linalg.norm(pred - source_window, axis=1)))
            if same_time_source_available
            else None,
            "pred_vs_source_same_time_cosine": cosine_flat(pred, source_window) if same_time_source_available else None,
        }
        rows.append(row)

    summary = {
        "schema": "openpi_contact_suffix_replay_diagnosis_v1",
        "replay_root": str(replay_root),
        "label_count": len(rows),
        "after_success_count": sum(r["after_success"] for r in rows),
        "after_inserted_live_pose_count": sum(r["after_inserted_live_pose"] for r in rows),
        "after_contact_stable_proxy_count": sum(r["after_contact_stable_proxy"] for r in rows),
        "after_grasped_count": sum(r["after_grasped"] for r in rows),
        "dp96_success_count": sum(r["dp96_success"] for r in rows),
        "dp96_continuable_count": sum(r["dp96_continuable"] for r in rows),
        "yz_worsened_count": sum(float(r["delta_abs_yz_sum"]) > 0 for r in rows),
        "x_worsened_count": sum(float(r["delta_abs_x"]) > 0 for r in rows),
        "training_offset_count": sum(bool(r["is_training_offset"]) for r in rows),
        "offset_to_first_insert_values": [r["offset_to_first_insert"] for r in rows],
        "mean_delta_abs_yz_sum": float(np.mean([float(r["delta_abs_yz_sum"]) for r in rows])),
        "mean_delta_abs_x": float(np.mean([float(r["delta_abs_x"]) for r in rows])),
        "mean_pred_vs_source_same_time_l2": float(
            np.mean([float(r["pred_vs_source_same_time_l2_mean"]) for r in rows if r["pred_vs_source_same_time_l2_mean"] is not None])
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "boundary": (
            "Diagnostic only. It reads completed OpenPI replay artifacts and source H5 labels; "
            "it trains no model and introduces no custom intermediate policy."
        ),
        "rows": rows,
    }
    out = Path(args.output_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in [
        "label_count",
        "after_success_count",
        "after_inserted_live_pose_count",
        "after_contact_stable_proxy_count",
        "after_grasped_count",
        "dp96_continuable_count",
        "yz_worsened_count",
        "x_worsened_count",
        "training_offset_count",
        "offset_to_first_insert_values",
        "mean_delta_abs_yz_sum",
        "mean_delta_abs_x",
        "mean_pred_vs_source_same_time_l2",
    ]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
