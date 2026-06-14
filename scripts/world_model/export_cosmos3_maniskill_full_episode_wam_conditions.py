#!/usr/bin/env python3
"""Export full-episode Cosmos3 WAM conditions for 300-step ManiSkill rollouts.

This exporter keeps each training row tied to the complete 300-step episode:
301 RGB/state frames and 300 action steps. Multiple rows may reference the same
episode with different causal prefix masks, but no row is a sliced 128-action
or 129-frame clip.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
import tyro

from export_cosmos3_maniskill_action_conditions import (
    STATE_TARGET_VECTOR_NAMES,
    _apply_stats,
    _write_jsonl,
    _zscore_stats,
)

FULL_EPISODE_VECTOR_NAMES = [
    "action_0",
    "action_1",
    "action_2",
    "action_3",
    "action_4",
    "action_5",
    "action_6",
    "task_tcp_x",
    "task_tcp_y",
    "task_tcp_z",
    "task_peg_x",
    "task_peg_y",
    "task_peg_z",
    "task_hole_x",
    "task_hole_y",
    "task_hole_z",
    "task_peg_head_hole_x",
    "task_peg_head_hole_y",
    "task_peg_head_hole_z",
    "task_hole_velocity_x",
    "task_hole_velocity_y",
    "task_hole_velocity_z",
    "task_grasped",
    "task_inserted",
    "task_hole_delta_cumulative_x",
    "task_hole_delta_cumulative_y",
    "task_hole_delta_cumulative_z",
    "task_peg_delta_applied_x",
    "task_peg_delta_applied_y",
    "task_peg_delta_applied_z",
    "action_time_fraction",
    "task_perturb_triggered",
]

PREFIX_REPEATED_VECTOR_NAMES = [
    name.replace("task_", "prefix_") if name.startswith("task_") else name
    for name in FULL_EPISODE_VECTOR_NAMES
]


MODE_IDS = {
    "static_monitor": 0,
    "target_pre_motion": 1,
    "target_motion_observed": 2,
    "target_post_motion": 3,
    "peg_recovery": 4,
    "insert_resume": 5,
}

@dataclass
class Args:
    dataset_root: str
    output_root: str
    total_video_frames: int = 301
    total_action_steps: int = 300
    prefix_policy: str = "multi_mode"
    fixed_prefix_frames: int = 81
    min_prefix_frames: int = 12
    max_prefix_frames: int = 260
    condition_latent_frames: int = 0
    temporal_compression_factor: int = 4
    action_condition_mode: str = "joint_policy_history_action"
    sidecar_target_mode: str = "future_aligned_state"
    prefix_role_source: str = "sampled"
    dense_receding_prefix_stride: int = 0
    require_video_frames: int = 301
    max_records: int = 0
    normalize: bool = True
    target_motion_epsilon_m: float = 0.002
    target_velocity_epsilon_m: float = 0.0007
    progress_every: int = 50


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _safe_prefix_frame(frame: int, args: Args) -> int:
    lo = max(1, int(args.min_prefix_frames) - 1)
    hi = min(int(args.max_prefix_frames), int(args.total_video_frames) - 2)
    return max(lo, min(hi, int(frame)))


def _first_true(mask: np.ndarray) -> int:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if hits.size else -1


def _optional_array(group: h5py.Group | None, name: str, default: np.ndarray, dtype: Any) -> np.ndarray:
    if group is not None and name in group:
        return np.asarray(group[name], dtype=dtype)
    return np.asarray(default, dtype=dtype)


def _load_episode_arrays(h5_path: Path, args: Args) -> dict[str, Any]:
    with h5py.File(h5_path, "r") as h5:
        traj_names = sorted([k for k in h5.keys() if k.startswith("traj_")], key=lambda x: int(x.split("_", 1)[1]))
        if len(traj_names) != 1:
            raise RuntimeError(f"{h5_path} expected one trajectory, found {traj_names}")
        group = h5[traj_names[0]]
        slots = group["slots"]
        perturb = group["perturb"]
        return {
            "traj_name": traj_names[0],
            "actions": np.asarray(group["actions"], dtype=np.float32),
            "source_frame_indices": _optional_array(
                group,
                "source_frame_indices",
                np.arange(args.total_video_frames, dtype=np.float32),
                np.float32,
            ),
            "hole_pose": np.asarray(slots["hole_pose"], dtype=np.float32),
            "peg_pose": np.asarray(slots["peg_pose"], dtype=np.float32),
            "tcp_pose": np.asarray(slots["tcp_pose"], dtype=np.float32),
            "qpos": _optional_array(slots, "qpos", np.zeros((args.total_video_frames, 9), dtype=np.float32), np.float32),
            "qvel": _optional_array(slots, "qvel", np.zeros((args.total_video_frames, 9), dtype=np.float32), np.float32),
            "peg_head_at_hole": np.asarray(slots["peg_head_at_hole"], dtype=np.float32),
            "hole_velocity_step": np.asarray(slots["hole_velocity_step"], dtype=np.float32),
            "grasped": np.asarray(slots["grasped"], dtype=bool),
            "inserted": np.asarray(slots["inserted"], dtype=bool),
            "hole_delta_cumulative": _optional_array(
                perturb,
                "hole_delta_cumulative",
                np.zeros((args.total_action_steps, 3), dtype=np.float32),
                np.float32,
            ),
            "peg_delta_applied": _optional_array(
                perturb,
                "peg_delta_applied",
                np.zeros((args.total_action_steps, 3), dtype=np.float32),
                np.float32,
            ),
            "triggered": _optional_array(perturb, "triggered", np.zeros((args.total_action_steps,), dtype=bool), bool),
            "trigger_step": _optional_array(perturb, "trigger_step", np.full((args.total_action_steps,), -1), np.int32),
        }


def _vec_at(array: np.ndarray, idx: int, dim: int) -> np.ndarray:
    out = np.zeros((dim,), dtype=np.float32)
    if array.size == 0:
        return out
    safe_idx = max(0, min(int(idx), int(array.shape[0]) - 1))
    value = np.asarray(array[safe_idx], dtype=np.float32).reshape(-1)
    out[: min(dim, value.shape[0])] = value[:dim]
    return out


def _bool_at(array: np.ndarray, idx: int) -> float:
    if array.size == 0:
        return 0.0
    safe_idx = max(0, min(int(idx), int(array.shape[0]) - 1))
    return float(bool(np.asarray(array[safe_idx]).reshape(-1)[0]))


def _scalar_at(array: np.ndarray, idx: int, default: float) -> float:
    if array.size == 0:
        return float(default)
    safe_idx = max(0, min(int(idx), int(array.shape[0]) - 1))
    return float(np.asarray(array[safe_idx]).reshape(-1)[0])


def _vector_names_for_mode(sidecar_target_mode: str) -> list[str]:
    if sidecar_target_mode == "future_aligned_state":
        return list(FULL_EPISODE_VECTOR_NAMES)
    if sidecar_target_mode == "prefix_repeated":
        return list(PREFIX_REPEATED_VECTOR_NAMES)
    raise ValueError(
        "sidecar_target_mode must be future_aligned_state or prefix_repeated, "
        f"got {sidecar_target_mode!r}"
    )


def _sidecar_vectors_for_step(
    arrays: dict[str, Any],
    *,
    step: int,
    total_video_frames: int,
    condition_prefix_frames: int,
    sidecar_target_mode: str,
) -> list[np.ndarray]:
    if sidecar_target_mode == "prefix_repeated":
        state_idx = max(0, min(int(condition_prefix_frames) - 1, total_video_frames - 1))
        perturb_idx = min(state_idx, max(0, arrays["actions"].shape[0] - 1))
    elif sidecar_target_mode == "future_aligned_state":
        # Action token i is temporally aligned to vision frame i + 1 in the
        # Cosmos3 action packer. History rows are observed/conditioned; rows
        # after the prefix are future state targets, not future conditions.
        state_idx = max(0, min(int(step) + 1, total_video_frames - 1))
        perturb_idx = min(max(0, int(step)), max(0, arrays["actions"].shape[0] - 1))
    else:
        raise ValueError(
            "sidecar_target_mode must be future_aligned_state or prefix_repeated, "
            f"got {sidecar_target_mode!r}"
        )

    return [
        _vec_at(arrays["tcp_pose"], state_idx, 3),
        _vec_at(arrays["peg_pose"], state_idx, 3),
        _vec_at(arrays["hole_pose"], state_idx, 3),
        _vec_at(arrays["peg_head_at_hole"], state_idx, 3),
        _vec_at(arrays["hole_velocity_step"], state_idx, 3),
        np.asarray(
            [
                _bool_at(arrays["grasped"], state_idx),
                _bool_at(arrays["inserted"], state_idx),
            ],
            dtype=np.float32,
        ),
        _vec_at(arrays["hole_delta_cumulative"], perturb_idx, 3),
        _vec_at(arrays["peg_delta_applied"], perturb_idx, 3),
    ]


def _raw_vectors_from_arrays(
    arrays: dict[str, Any],
    total_video_frames: int,
    condition_prefix_frames: int,
    sidecar_target_mode: str,
) -> np.ndarray:
    if total_video_frames <= 1:
        raise ValueError(f"total_video_frames must be at least 2, got {total_video_frames}")
    actions = arrays["actions"]
    action_steps = total_video_frames - 1
    vector_names = _vector_names_for_mode(sidecar_target_mode)

    rows = []
    denom = max(1, action_steps - 1)
    for step in range(action_steps):
        perturb_idx = min(max(0, step), max(0, actions.shape[0] - 1))
        action = np.zeros((7,), dtype=np.float32)
        if step < actions.shape[0]:
            src = np.asarray(actions[step], dtype=np.float32).reshape(-1)
            action[: min(7, src.shape[0])] = src[:7]
        sidecar = _sidecar_vectors_for_step(
            arrays,
            step=step,
            total_video_frames=total_video_frames,
            condition_prefix_frames=condition_prefix_frames,
            sidecar_target_mode=sidecar_target_mode,
        )
        rows.append(
            np.concatenate(
                [
                    action,
                    *sidecar,
                    np.asarray(
                        [
                            float(step) / float(denom),
                            _bool_at(arrays["triggered"], perturb_idx),
                        ],
                        dtype=np.float32,
                    ),
                ]
            )
        )
    out = np.stack(rows, axis=0).astype(np.float32)
    if out.shape != (action_steps, len(vector_names)):
        raise RuntimeError(f"unexpected action/state shape {out.shape}")
    return out


def _state_targets_from_arrays(arrays: dict[str, Any], total_video_frames: int) -> np.ndarray:
    rows = []
    denom = max(1, total_video_frames - 1)
    for frame_idx in range(total_video_frames):
        perturb_idx = min(frame_idx, total_video_frames - 2)
        row = np.concatenate(
            [
                np.asarray(
                    [
                        _scalar_at(arrays["source_frame_indices"], frame_idx, default=float(frame_idx)),
                        float(frame_idx) / float(denom),
                    ],
                    dtype=np.float32,
                ),
                _vec_at(arrays["tcp_pose"], frame_idx, 7),
                _vec_at(arrays["peg_pose"], frame_idx, 7),
                _vec_at(arrays["hole_pose"], frame_idx, 7),
                _vec_at(arrays["qpos"], frame_idx, 9),
                _vec_at(arrays["qvel"], frame_idx, 9),
                _vec_at(arrays["peg_head_at_hole"], frame_idx, 3),
                _vec_at(arrays["hole_velocity_step"], frame_idx, 3),
                np.asarray(
                    [
                        _bool_at(arrays["grasped"], frame_idx),
                        _bool_at(arrays["inserted"], frame_idx),
                    ],
                    dtype=np.float32,
                ),
                _vec_at(arrays["hole_delta_cumulative"], perturb_idx, 3),
                _vec_at(arrays["peg_delta_applied"], perturb_idx, 3),
                np.asarray([_bool_at(arrays["triggered"], perturb_idx)], dtype=np.float32),
            ]
        )
        rows.append(row)
    out = np.stack(rows, axis=0).astype(np.float32)
    if out.shape != (total_video_frames, len(STATE_TARGET_VECTOR_NAMES)):
        raise RuntimeError(f"unexpected state target shape {out.shape}")
    return out


def _episode_summary(arrays: dict[str, Any], scenario: str, args: Args) -> dict[str, Any]:
    actions = arrays["actions"]
    hole_xyz = arrays["hole_pose"][:, :3]
    peg_xyz = arrays["peg_pose"][:, :3]
    tcp_xyz = arrays["tcp_pose"][:, :3]
    grasped = arrays["grasped"]
    inserted = arrays["inserted"]
    velocity = arrays["hole_velocity_step"]
    triggered = arrays["triggered"]

    if actions.shape[0] != args.total_action_steps:
        raise RuntimeError(f"action length {actions.shape[0]} != {args.total_action_steps}")
    for name in ("hole_pose", "peg_pose", "tcp_pose", "peg_head_at_hole", "hole_velocity_step", "grasped", "inserted"):
        arr = arrays[name]
        if arr.shape[0] != args.total_video_frames:
            raise RuntimeError(f"{name} length {arr.shape[0]} != {args.total_video_frames}")

    displacement = np.linalg.norm(hole_xyz - hole_xyz[0:1], axis=1)
    onset_from_disp = _first_true(displacement > args.target_motion_epsilon_m)
    onset_from_trigger = _first_true(triggered)
    if onset_from_disp >= 0 and onset_from_trigger >= 0:
        target_onset_frame = min(onset_from_disp, onset_from_trigger + 1)
    elif onset_from_disp >= 0:
        target_onset_frame = onset_from_disp
    elif onset_from_trigger >= 0:
        target_onset_frame = onset_from_trigger + 1
    else:
        target_onset_frame = -1

    moving_mask = np.linalg.norm(velocity, axis=1) > args.target_velocity_epsilon_m
    moving_hits = np.flatnonzero(moving_mask)
    target_last_motion_frame = int(moving_hits[-1]) if moving_hits.size else -1
    target_settle_frame = target_last_motion_frame + 1 if target_last_motion_frame >= 0 else -1
    first_grasp_frame = _first_true(grasped)
    first_ungrasp_after_grasp = -1
    if first_grasp_frame >= 0:
        later_false = np.flatnonzero(~grasped[first_grasp_frame + 1 :])
        if later_false.size:
            first_ungrasp_after_grasp = int(first_grasp_frame + 1 + later_false[0])
    first_inserted_frame = _first_true(inserted)

    frame_labels = []
    for frame in range(args.total_video_frames):
        target_started = bool(target_onset_frame >= 0 and frame >= target_onset_frame)
        target_currently_moving = bool(moving_mask[frame])
        peg_needs_recovery = bool(first_ungrasp_after_grasp >= 0 and frame >= first_ungrasp_after_grasp and not grasped[frame])
        insert_resume = bool(grasped[frame] and not peg_needs_recovery and np.linalg.norm(arrays["peg_head_at_hole"][frame, :3]) < 0.05)
        if peg_needs_recovery:
            mode = "peg_recovery"
        elif insert_resume:
            mode = "insert_resume"
        elif target_started and target_currently_moving:
            mode = "target_motion_observed"
        elif target_started:
            mode = "target_post_motion"
        elif target_onset_frame >= 0:
            mode = "target_pre_motion"
        else:
            mode = "static_monitor"
        frame_labels.append(
            {
                "frame_index": frame,
                "mode": mode,
                "mode_id": MODE_IDS[mode],
                "target_started": target_started,
                "target_currently_moving": target_currently_moving,
                "peg_needs_recovery": peg_needs_recovery,
                "insert_resume_candidate": insert_resume,
                "grasped": bool(grasped[frame]),
                "inserted": bool(inserted[frame]),
            }
        )

    return {
        "scenario": scenario,
        "target_object": "hole",
        "tool_object": "peg",
        "actor": "robot_gripper_tcp",
        "goal": "insert peg into target hole",
        "target_onset_frame": int(target_onset_frame),
        "target_last_motion_frame": int(target_last_motion_frame),
        "target_settle_frame": int(target_settle_frame),
        "first_grasp_frame": int(first_grasp_frame),
        "first_ungrasp_after_grasp_frame": int(first_ungrasp_after_grasp),
        "first_inserted_frame": int(first_inserted_frame),
        "target_initial_xyz": hole_xyz[0].astype(float).tolist(),
        "target_final_xyz": hole_xyz[-1].astype(float).tolist(),
        "peg_initial_xyz": peg_xyz[0].astype(float).tolist(),
        "peg_final_xyz": peg_xyz[-1].astype(float).tolist(),
        "tcp_initial_xyz": tcp_xyz[0].astype(float).tolist(),
        "tcp_final_xyz": tcp_xyz[-1].astype(float).tolist(),
        "frame_labels": frame_labels,
        "label_boundary": (
            "These future fields are supervision/evaluation labels only. "
            "They must not be injected as controller-facing conditions."
        ),
    }


def _prefix_specs(summary: dict[str, Any], args: Args) -> list[dict[str, Any]]:
    if args.prefix_policy == "fixed":
        frame = _safe_prefix_frame(args.fixed_prefix_frames - 1, args)
        return [{"role": "fixed_prefix", "prefix_frame_index": frame, "reason": "fixed full-episode prefix"}]
    if args.prefix_policy != "multi_mode":
        raise ValueError(f"unknown prefix_policy={args.prefix_policy!r}")

    specs: list[dict[str, Any]] = []

    def add(role: str, frame: int, reason: str, sampled_prefix_role: str | None = None) -> None:
        safe = _safe_prefix_frame(frame, args)
        sampled = sampled_prefix_role or role
        key = (role, safe, sampled)
        if key not in {
            (item["role"], item["prefix_frame_index"], item.get("sampled_prefix_role", item["role"]))
            for item in specs
        }:
            specs.append(
                {
                    "role": role,
                    "prefix_frame_index": safe,
                    "sampled_prefix_role": sampled,
                    "reason": reason,
                }
            )

    base = args.fixed_prefix_frames - 1
    onset = int(summary["target_onset_frame"])
    settle = int(summary["target_settle_frame"])
    first_drop = int(summary["first_ungrasp_after_grasp_frame"])
    first_grasp = int(summary["first_grasp_frame"])

    if onset >= 0:
        add("target_pre_motion", onset - 8, "target has not visibly moved yet; monitor onset/final pose")
        add("target_motion_observed", onset + 8, "target motion has been observed; predict future path and actions")
        if settle >= 0:
            add("target_post_motion", settle + 4, "target motion history is available; predict final insertion frame")
        else:
            add("target_post_motion", onset + 40, "post-motion continuation prefix")
    else:
        add("static_monitor", base, "target has not moved; preserve DP/static skill while monitoring")
        add("static_late_monitor", max(base + 60, 140), "late static monitor and insertion resume sanity")

    if first_drop >= 0:
        add("peg_recovery", first_drop + 8, "peg drop/loss observed; predict regrasp/recovery continuation")
    elif str(summary["scenario"]) in {"peg_drop", "peg_disturb"} and first_grasp >= 0:
        add("peg_recovery", first_grasp + 20, "peg disturbance/drop scenario recovery context")

    if first_grasp >= 0:
        add("insert_resume", first_grasp + 40, "peg is likely held; predict insertion resume")

    stride = int(args.dense_receding_prefix_stride)
    if stride > 0 and onset >= 0:
        end = int(summary["first_inserted_frame"])
        if end < 0:
            end = int(args.total_video_frames) - 2
        start = _safe_prefix_frame(onset + 1, args)
        stop = _safe_prefix_frame(end, args)
        for frame in range(start, stop + 1, stride):
            physical_role = str(summary["frame_labels"][frame]["mode"])
            add(
                physical_role,
                frame,
                f"dense receding physical-mode prefix every {stride} frames after target onset",
                sampled_prefix_role="dense_receding",
            )

    specs = sorted(specs, key=lambda item: (item["prefix_frame_index"], item["role"]))
    return specs


def _controller_role_for_spec(spec: dict[str, Any], label: dict[str, Any], args: Args) -> str:
    if args.prefix_role_source == "sampled":
        return str(spec["role"])
    if args.prefix_role_source == "physical_mode":
        return str(label["mode"])
    raise ValueError("prefix_role_source must be sampled or physical_mode")


def _prefix_payload(
    arrays: dict[str, Any],
    summary: dict[str, Any],
    prefix_frame: int,
    role: str,
    sampled_prefix_role: str,
    prefix_role_source: str,
) -> dict[str, Any]:
    label = summary["frame_labels"][prefix_frame]
    return {
        "prefix_frame_index": int(prefix_frame),
        "condition_prefix_frames": int(prefix_frame + 1),
        "prefix_role": role,
        "sampled_prefix_role": sampled_prefix_role,
        "prefix_role_source": prefix_role_source,
        "mode": label["mode"],
        "mode_id": int(label["mode_id"]),
        "target_motion_observed": bool(label["target_started"]),
        "peg_needs_recovery": bool(label["peg_needs_recovery"]),
        "insert_resume_candidate": bool(label["insert_resume_candidate"]),
        "grasped": bool(label["grasped"]),
        "inserted": bool(label["inserted"]),
        "tcp_xyz": arrays["tcp_pose"][prefix_frame, :3].astype(float).tolist(),
        "peg_xyz": arrays["peg_pose"][prefix_frame, :3].astype(float).tolist(),
        "target_hole_xyz": arrays["hole_pose"][prefix_frame, :3].astype(float).tolist(),
        "peg_head_at_hole_xyz": arrays["peg_head_at_hole"][prefix_frame, :3].astype(float).tolist(),
        "hole_velocity_xyz": arrays["hole_velocity_step"][prefix_frame, :3].astype(float).tolist(),
        "causal_boundary": "Prefix/current state only; no future target final pose is exposed here.",
    }


def _prefix_latent_indexes(prefix_frames: int, temporal_compression_factor: int) -> list[int]:
    if temporal_compression_factor <= 0:
        raise ValueError("temporal_compression_factor must be positive")
    latent_prefix = 1 + (int(prefix_frames) - 1) // int(temporal_compression_factor)
    return list(range(max(1, latent_prefix)))


def _fmt(values: list[float]) -> str:
    return "[" + ", ".join(f"{float(v):.4f}" for v in values) + "]"


def _caption(payload: dict[str, Any]) -> str:
    return (
        "PegInsertionSide full-episode world/action model sample. "
        "Roles: TARGET_OBJECT=hole, TOOL_OBJECT=peg, ACTOR=robot_gripper_tcp. "
        f"Observed prefix role={payload['prefix_role']} mode={payload['mode']} "
        f"through frame {payload['prefix_frame_index']}. "
        f"Current target/hole xyz={_fmt(payload['target_hole_xyz'])}; "
        f"current peg xyz={_fmt(payload['peg_xyz'])}; "
        f"current TCP xyz={_fmt(payload['tcp_xyz'])}; "
        f"peg-head relative-to-hole xyz={_fmt(payload['peg_head_at_hole_xyz'])}; "
        f"observed target velocity xyz={_fmt(payload['hole_velocity_xyz'])}; "
        f"target_motion_observed={payload['target_motion_observed']}; "
        f"peg_needs_recovery={payload['peg_needs_recovery']}; "
        f"grasped={payload['grasped']}; inserted={payload['inserted']}. "
        "Predict target motion, future RGB/task state, and executable actions in the changed world."
    )


def main() -> None:
    args = tyro.cli(Args)
    if args.total_video_frames != 301 or args.total_action_steps != 300:
        raise ValueError("active contract requires total_video_frames=301 and total_action_steps=300")
    if args.require_video_frames != 301:
        raise ValueError("active contract requires require_video_frames=301")
    if args.action_condition_mode not in {"joint_policy_history_action", "forward_dynamics"}:
        raise ValueError("action_condition_mode must be joint_policy_history_action or forward_dynamics")
    if args.prefix_role_source not in {"sampled", "physical_mode"}:
        raise ValueError("prefix_role_source must be sampled or physical_mode")
    if int(args.dense_receding_prefix_stride) < 0:
        raise ValueError("dense_receding_prefix_stride must be >= 0")
    vector_names = _vector_names_for_mode(args.sidecar_target_mode)

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    manifest = json.loads((dataset_root / "manifest.json").read_text())
    videos = list(manifest.get("videos", []))
    if args.max_records > 0:
        videos = videos[: args.max_records]
    if not videos:
        raise ValueError("no videos found in source manifest")

    source_rows: dict[str, dict[str, Any]] = {}
    for split in ("train", "val"):
        for row in _read_jsonl(dataset_root / split / "video_dataset_file.jsonl"):
            source_rows[row["uuid"]] = row

    raw_rows_for_stats: list[np.ndarray] = []
    planned: list[dict[str, Any]] = []
    state_targets: dict[str, np.ndarray] = {}
    arrays_by_sample: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}

    _progress(f"full_episode_export_plan_start records={len(videos)} output_root={output_root}")
    for index, item in enumerate(videos, start=1):
        if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0):
            _progress(f"full_episode_export_reading {index}/{len(videos)} sample_id={item['sample_id']}")
        if int(item.get("num_video_frames", -1)) != 301:
            raise RuntimeError(f"{item.get('sample_id')} has num_video_frames={item.get('num_video_frames')}")
        h5_path = Path(item["input_h5"])
        if not h5_path.exists():
            raise FileNotFoundError(h5_path)
        arrays = _load_episode_arrays(h5_path, args)
        summary = _episode_summary(arrays, str(item.get("scenario")), args)
        specs = _prefix_specs(summary, args)
        arrays_by_sample[item["sample_id"]] = arrays
        summaries[item["sample_id"]] = summary
        state_targets[item["sample_id"]] = _state_targets_from_arrays(arrays, args.total_video_frames)
        for spec in specs:
            raw = _raw_vectors_from_arrays(
                arrays,
                args.total_video_frames,
                spec["prefix_frame_index"] + 1,
                args.sidecar_target_mode,
            )
            raw_rows_for_stats.append(raw)
            planned.append({"item": item, "summary": summary, "prefix": spec, "raw": raw})
        if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0 or index == len(videos)):
            _progress(f"full_episode_export_planned {index}/{len(videos)} sample_id={item['sample_id']} prefixes={len(specs)}")

    stats = _zscore_stats(raw_rows_for_stats) if args.normalize else None
    if stats is not None:
        # The helper is shared with the older prefix-repeated exporter, whose
        # vector names use prefix_* labels. Full-episode WAM rows use task_*
        # names in the same column order, so keep the statistics but publish
        # the correct names for downstream diagnostics and sidecar readout.
        stats["vector_names"] = vector_names
        stats["raw_action_dim"] = len(vector_names)
    records_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    condition_rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    written_state_targets: dict[str, Path] = {}
    written_summary_labels: dict[str, Path] = {}
    row_uuid_seen: Counter[str] = Counter()
    role_mode_counts: dict[str, int] = {}
    sampled_role_counts: dict[str, int] = {}

    for row_index, plan in enumerate(planned, start=1):
        item = plan["item"]
        summary = plan["summary"]
        spec = plan["prefix"]
        raw = plan["raw"]
        split = str(item["split"])
        sample_id = str(item["sample_id"])
        prefix_frame = int(spec["prefix_frame_index"])
        prefix_frames = prefix_frame + 1
        condition_frame_indexes_vision = _prefix_latent_indexes(prefix_frames, args.temporal_compression_factor)
        label = summary["frame_labels"][prefix_frame]
        role = _controller_role_for_spec(spec, label, args)
        sampled_prefix_role = str(spec.get("sampled_prefix_role", spec["role"]))
        row_uuid_base = f"{sample_id}__{role}_f{prefix_frame:03d}"
        row_uuid_seen[row_uuid_base] += 1
        if row_uuid_seen[row_uuid_base] == 1:
            row_uuid = row_uuid_base
        else:
            row_uuid = f"{row_uuid_base}__sampled_{sampled_prefix_role}_{row_uuid_seen[row_uuid_base]}"
        arrays = arrays_by_sample[sample_id]
        payload = _prefix_payload(
            arrays,
            summary,
            prefix_frame,
            role,
            sampled_prefix_role,
            args.prefix_role_source,
        )
        role_mode_key = f"{role}|{payload['mode']}"
        role_mode_counts[role_mode_key] = role_mode_counts.get(role_mode_key, 0) + 1
        sampled_role_counts[sampled_prefix_role] = sampled_role_counts.get(sampled_prefix_role, 0) + 1

        action = _apply_stats(raw, stats) if stats is not None else raw
        action_rel = Path("actions") / split / f"{row_index:05d}_{row_uuid}.npy"
        action_path = output_root / action_rel
        action_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(action_path, np.asarray(action, dtype=np.float32), allow_pickle=False)

        if sample_id not in written_state_targets:
            state_rel = Path("state_targets") / split / f"{sample_id}.json"
            state_path = output_root / state_rel
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "uuid": sample_id,
                        "num_frames": args.total_video_frames,
                        "num_action_steps": args.total_action_steps,
                        "state_vector_names": STATE_TARGET_VECTOR_NAMES,
                        "states": state_targets[sample_id].tolist(),
                        "boundary": (
                            "Full future state sequence is a target/readout label. "
                            "In future_aligned_state action sidecars, rows after the causal "
                            "prefix also supervise generated future task-state tokens."
                        ),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            written_state_targets[sample_id] = state_path.resolve()
        state_path = written_state_targets[sample_id]

        if sample_id not in written_summary_labels:
            summary_rel = Path("task_labels") / split / f"{sample_id}__summary.json"
            summary_path = output_root / summary_rel
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "uuid": sample_id,
                        "summary": summary,
                        "label_boundary": (
                            "Future labels supervise/read out the world model. They must not be "
                            "used as controller-facing conditions."
                        ),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            written_summary_labels[sample_id] = summary_path.resolve()
        summary_path = written_summary_labels[sample_id]

        label_rel = Path("task_labels") / split / f"{row_index:05d}_{row_uuid}.json"
        label_path = output_root / label_rel
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "uuid": row_uuid,
                    "prefix": payload,
                    "summary_path": str(summary_path),
                    "sampled_prefix_role": sampled_prefix_role,
                    "prefix_reason": spec["reason"],
                    "label_boundary": (
                        "Future labels supervise/read out the world model. They must not be "
                        "used as controller-facing conditions."
                    ),
                },
                separators=(",", ":"),
            )
            + "\n"
        )

        video_path = Path(item["video"]).resolve()
        source_row = source_rows.get(sample_id)
        if source_row is None:
            raise KeyError(f"missing source JSONL row for {sample_id}")
        sft_row = dict(source_row)
        sft_row.update(
            {
                "uuid": row_uuid,
                "source_uuid": sample_id,
                "vision_path": str(video_path),
                "action_path": str(action_path.resolve()),
                "state_target_path": str(state_path.resolve()),
                "task_state_target_path": str(state_path.resolve()),
                "task_label_path": str(label_path.resolve()),
                "task_switch_label_path": str(label_path.resolve()),
                "model_mode": "policy" if args.action_condition_mode == "joint_policy_history_action" else "forward_dynamics",
                "domain_name": "maniskill_peg_insertion",
                "domain_id": 21,
                "raw_action_dim": len(vector_names),
                "max_action_dim": 64,
                "action_chunk_size": args.total_action_steps,
                "num_video_frames": args.total_video_frames,
                "num_action_steps": args.total_action_steps,
                "condition_prefix_frames": prefix_frames,
                "prefix_frame_index": prefix_frame,
                "prefix_role": role,
                "sampled_prefix_role": sampled_prefix_role,
                "physical_mode": payload["mode"],
                "prefix_role_source": args.prefix_role_source,
                "target_object": "hole",
                "tool_object": "peg",
                "actor": "robot_gripper_tcp",
                "condition_frame_indexes_action": list(range(prefix_frame)) if args.action_condition_mode == "joint_policy_history_action" else list(range(args.total_action_steps)),
                "condition_frame_indexes_vision": condition_frame_indexes_vision,
                "observed_video_frame_indexes": list(range(prefix_frames)),
                "t2w_windows": [
                    {
                        "caption": _caption(payload),
                        "start_frame": 0,
                        "end_frame": args.total_video_frames - 1,
                        "temporal_interval": 1,
                    }
                ],
                "metadata": {
                    "split": split,
                    "scenario": item.get("scenario"),
                    "source_h5": item.get("input_h5"),
                    "source_sample_id": sample_id,
                    "visual_input": "RGB only; depth is not used",
                    "episode_contract": {
                        "num_video_frames": args.total_video_frames,
                        "num_action_steps": args.total_action_steps,
                        "frame_start": 0,
                        "frame_end": args.total_video_frames - 1,
                    },
                    "role_binding": {
                        "target_object": "hole",
                        "tool_object": "peg",
                        "actor": "robot_gripper_tcp",
                        "goal": "insert peg into target hole",
                    },
                    "prefix_causal_state": payload,
                    "sampled_prefix_role": sampled_prefix_role,
                    "physical_mode": payload["mode"],
                    "prefix_role_source": args.prefix_role_source,
                    "supervision_available": {
                        "target_onset_frame": summary["target_onset_frame"],
                        "target_final_xyz": summary["target_final_xyz"],
                        "task_labels_path": str(label_path.resolve()),
                        "task_summary_path": str(summary_path),
                        "state_target_path": str(state_path.resolve()),
                    },
                    "future_label_boundary": (
                        "Future target/peg/TCP labels are targets only. In future_aligned_state "
                        "mode they appear only on unconditioned future action/state rows; "
                        "history rows before prefix_frame are the only structured action/state "
                        "conditions."
                    ),
                    "structured_action_state_condition": {
                        "domain_name": "maniskill_peg_insertion",
                        "domain_id": 21,
                        "raw_action_dim": len(vector_names),
                        "vector_names": vector_names,
                        "normalization": "zscore" if stats is not None else "none",
                        "action_condition_mode": args.action_condition_mode,
                        "sidecar_target_mode": args.sidecar_target_mode,
                        "condition_prefix_frames": prefix_frames,
                        "condition_frame_indexes_vision_are_latent_indexes": True,
                        "temporal_compression_factor": args.temporal_compression_factor,
                        "causal_boundary": (
                            "Action/state sidecar is a rank-2 Cosmos-readable NumPy float32 array. "
                            "condition_frame_indexes_action marks observed history rows as clean; "
                            "later rows are generated future action/task-state targets."
                        ),
                    },
                },
            }
        )
        condition_row = {
            "uuid": row_uuid,
            "source_uuid": sample_id,
            "split": split,
            "scenario": item.get("scenario"),
            "source_h5": item.get("input_h5"),
            "rgb_video": str(video_path),
            "frame_start": 0,
            "frame_end": args.total_video_frames - 1,
            "num_rgb_frames": args.total_video_frames,
            "num_action_steps": args.total_action_steps,
            "condition_prefix_frames": prefix_frames,
            "prefix_frame_index": prefix_frame,
            "prefix_role": role,
            "sampled_prefix_role": sampled_prefix_role,
            "physical_mode": payload["mode"],
            "prefix_role_source": args.prefix_role_source,
            "target_object": "hole",
            "tool_object": "peg",
            "actor": "robot_gripper_tcp",
            "action_path": str(action_path.resolve()),
            "state_target_path": str(state_path.resolve()),
            "task_label_path": str(label_path.resolve()),
            "task_summary_path": str(summary_path),
            "condition_frame_indexes_action": sft_row["condition_frame_indexes_action"],
            "condition_frame_indexes_vision": sft_row["condition_frame_indexes_vision"],
            "observed_video_frame_indexes": sft_row["observed_video_frame_indexes"],
            "sidecar_target_mode": args.sidecar_target_mode,
            "boundary": "Full episode row with causal prefix mask; not a 128/129 sliced chunk.",
        }
        records_by_split[split].append(sft_row)
        condition_rows_by_split[split].append(condition_row)
        if args.progress_every > 0 and (row_index == 1 or row_index % args.progress_every == 0 or row_index == len(planned)):
            _progress(f"full_episode_export_wrote {row_index}/{len(planned)} uuid={row_uuid}")

    for split in ("train", "val"):
        _write_jsonl(output_root / split / "video_action_dataset_file.jsonl", records_by_split[split])
        _write_jsonl(output_root / split / "action_condition_dataset_file.jsonl", condition_rows_by_split[split])
    if stats is not None:
        (output_root / "normalization_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")

    manifest_out = {
        "args": asdict(args),
        "source_dataset_root": str(dataset_root),
        "num_source_episodes": len(videos),
        "num_rows": len(planned),
        "num_train_rows": len(records_by_split["train"]),
        "num_val_rows": len(records_by_split["val"]),
        "num_video_frames": args.total_video_frames,
        "num_action_steps": args.total_action_steps,
        "raw_action_dim": len(vector_names),
        "vector_names": vector_names,
        "sidecar_target_mode": args.sidecar_target_mode,
        "state_target_vector_names": STATE_TARGET_VECTOR_NAMES,
        "mode_ids": MODE_IDS,
        "prefix_role_source": args.prefix_role_source,
        "dense_receding_prefix_stride": args.dense_receding_prefix_stride,
        "sampled_prefix_role_counts": dict(sorted(sampled_role_counts.items())),
        "prefix_role_mode_counts": dict(sorted(role_mode_counts.items())),
        "prefix_role_mode_mismatch_count": int(
            sum(count for key, count in role_mode_counts.items() if key.split("|", 1)[0] != key.split("|", 1)[1])
        ),
        "normalization": "zscore" if stats is not None else "none",
        "visual_input": "RGB only; depth is not used",
        "contract": (
            "full episode 301 RGB/state frames and 300 action/state rows; no 129/128 "
            "chunking; future_aligned_state sidecars supervise future task state only "
            "on unconditioned rows"
        ),
        "shared_state_target_files": len(written_state_targets),
        "shared_task_summary_files": len(written_summary_labels),
        "sft_jsonl": {
            "train": str(output_root / "train" / "video_action_dataset_file.jsonl"),
            "val": str(output_root / "val" / "video_action_dataset_file.jsonl"),
        },
        "condition_jsonl": {
            "train": str(output_root / "train" / "action_condition_dataset_file.jsonl"),
            "val": str(output_root / "val" / "action_condition_dataset_file.jsonl"),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest_out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "output_root": str(output_root), "num_rows": len(planned)}, sort_keys=True))


if __name__ == "__main__":
    main()
