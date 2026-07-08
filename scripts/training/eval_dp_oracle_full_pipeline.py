#!/usr/bin/env python3
"""Phase 03 full Oracle pipeline attempt.

This keeps one live ManiSkill environment in memory while it:

1. runs the active DP checkpoint before target motion,
2. calls official Cosmos3 policy inference for RGB/action prediction after a
   live RGB prefix,
3. executes only Cosmos-derived actions during the dynamic stage, and
4. switches to a physical finisher only after a logged near-target gate.

It must not use peg set_pose, saved-state replay, or source-state restore.
"""

from __future__ import annotations

import json
import importlib
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import tyro
from gymnasium.vector import SyncVectorEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack

REVIEW_CAMERA_UIDS: list[str] = []


@dataclass
class Args:
    ckpt_path: str
    output_dir: str
    cosmos_config_file: str
    cosmos_checkpoint_path: str
    cosmos_normalization_stats: str
    cosmos_python: str
    project_python: str
    cosmos_framework_path: str
    cosmos_tokenizer_dir: str
    wan_vae_path: str
    source_h5_path: str = ""
    source_key: str = ""
    source_h5_require_live_motion_gate: bool = True
    source_h5_gate_x_margin: float = 0.03
    source_h5_gate_yz_margin: float = 0.015
    source_h5_peg_perturb_mode: str = "block"
    source_h5_peg_force_scale: float = 25.0
    source_h5_peg_force_steps: int = 8
    env_id: str = "PegInsertionSide-v1"
    control_mode: str = "pd_ee_delta_pose"
    sim_backend: str = "physx_cpu"
    scenario_name: str = "target_y_positive"
    max_episode_steps: int = 300
    seed: int = 2
    target_motion_step: int = 84
    target_motion_x: float = 0.0
    target_motion_y: float = 0.025
    target_motion_z: float = 0.0
    target_motion_per_step: float = 0.00125
    target_motion_during_finisher: bool = True
    require_target_motion_complete_before_finisher: bool = False
    motion_detect_threshold: float = 0.003
    premotion_cosmos_step: int = 20
    premotion_cosmos_interval: int = 16
    max_premotion_cosmos_predictions: int = 0
    cosmos_action_horizon: int = 8
    max_cosmos_rounds: int = 4
    cosmos_build_timeout_s: int = 180
    cosmos_inference_timeout_s: int = 900
    cosmos_extract_timeout_s: int = 180
    cosmos_action_row_offset: int = 0
    cosmos_action_row_offset_source: str = ""
    cosmos_action_scale_x: float = 1.0
    cosmos_action_scale_y: float = 1.0
    cosmos_action_scale_z: float = 1.0
    cosmos_action_scale_rot: float = 1.0
    cosmos_action_scale_gripper: float = 1.0
    cosmos_action_direction_guard: str = "none"
    cosmos_action_direction_guard_mode: str = "clip_opposite"
    dynamic_controller: str = "cosmos3_policy"
    min_cosmos_dynamic_actions_before_finisher: int = 4
    near_target_l2: float = 0.08
    max_finisher_steps: int = 80
    finisher_controller: str = "manual_oracle_servo"
    manual_forward_bias: float = 0.0
    manual_forward_gain: float = 1.6
    manual_forward_limit: float = 0.28
    manual_lateral_gain: float = 1.2
    manual_lateral_limit: float = 0.08
    manual_vertical_gain: float = 1.2
    manual_vertical_limit: float = 0.08
    manual_yaw_action: float = 0.22
    manual_yaw_stop_l2: float = -1.0
    manual_gripper_action: float = -1.0
    manual_align_threshold: float = 0.012
    manual_yz_abort_threshold: float = 0.045
    manual_soft_insert_threshold: float = -1.0
    manual_soft_insert_scale: float = 0.35
    manual_insert_speed: float = 0.035
    manual_retreat_speed: float = 0.015
    manual_insert_roll_action: float = 0.0
    manual_insert_pitch_action: float = 0.0
    manual_insert_yaw_action: float = 0.0
    manual_pose_rot_gain: float = 1.0
    manual_pose_rot_limit: float = 0.25
    manual_pose_rot_yz_threshold: float = 0.025
    manual_dp_to_manual_l2: float = 0.04
    source_h5_teacher_action_start_offset: int = 0
    source_h5_teacher_dynamic_action_start_offset: int = 0
    cosmos_experiment: str = "vision_sft_nano"
    render_shader_pack: str = "minimal"
    fps: int = 30
    use_ema: bool = True
    cuda: bool = True


def read_h5_scalar_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "asstr"):
        arr = value.asstr()[()]
    else:
        arr = value[()] if hasattr(value, "__getitem__") else value
    if isinstance(arr, bytes):
        return arr.decode("utf-8")
    if isinstance(arr, np.ndarray):
        arr = arr.reshape(-1)[0]
        if isinstance(arr, bytes):
            return arr.decode("utf-8")
    return str(arr)


def find_h5_item(group: Any, name: str) -> Any | None:
    if name in group:
        return group[name]
    for key in group:
        item = group[key]
        if hasattr(item, "keys"):
            found = find_h5_item(item, name)
            if found is not None:
                return found
    return None


def find_h5_attr(group: Any, name: str) -> Any | None:
    if name in getattr(group, "attrs", {}):
        return group.attrs[name]
    for key in group:
        item = group[key]
        if hasattr(item, "keys"):
            found = find_h5_attr(item, name)
            if found is not None:
                return found
    return None


def load_source_h5_protocol(source_h5_path: str) -> dict[str, Any]:
    if not source_h5_path:
        return {}
    import h5py

    path = Path(source_h5_path).resolve()
    with h5py.File(path, "r") as h5:
        summary_item = find_h5_item(h5, "summary_json")
        summary_text = read_h5_scalar_text(summary_item) if summary_item is not None else None
        if summary_text is None:
            summary_attr = find_h5_attr(h5, "summary_json")
            summary_text = read_h5_scalar_text(summary_attr) if summary_attr is not None else None
        if summary_text is None:
            raise RuntimeError(f"source_h5_missing_summary_json={path}")
        summary = json.loads(summary_text)
        peg_delta_item = find_h5_item(h5, "peg_delta_applied")
        peg_delta_applied = np.asarray(peg_delta_item[()], dtype=np.float64) if peg_delta_item is not None else None
    first_motion = int(summary.get("first_target_motion_step", -1))
    last_motion = int(summary.get("last_target_motion_frame", first_motion))
    trigger_step = int(summary.get("trigger_step", first_motion if first_motion >= 0 else 84))
    motion_xyz = np.asarray(summary.get("target_motion_xyz") or [0.0, 0.0, 0.0], dtype=np.float64).reshape(3)
    first_peg_perturb_step = int(summary.get("first_peg_perturb_step", -1))
    peg_delta_first = np.zeros((3,), dtype=np.float64)
    peg_delta_sum = np.zeros((3,), dtype=np.float64)
    if peg_delta_applied is not None and peg_delta_applied.size:
        peg_delta_applied = peg_delta_applied.reshape((-1, peg_delta_applied.shape[-1]))
        peg_delta_xyz = peg_delta_applied[:, :3]
        norms = np.linalg.norm(peg_delta_xyz, axis=1)
        nonzero = np.flatnonzero(norms > 1.0e-9)
        if nonzero.size:
            first_peg_perturb_step = int(nonzero[0])
            peg_delta_first = peg_delta_xyz[nonzero[0]].astype(np.float64)
            peg_delta_sum = peg_delta_xyz[nonzero].sum(axis=0).astype(np.float64)
    if first_motion < 0:
        first_motion = trigger_step
        last_motion = trigger_step
    motion_steps = max(1, last_motion - first_motion + 1)
    motion_norm = float(np.linalg.norm(motion_xyz))
    source_defaults = (
        summary.get("original_protocol_provenance", {})
        .get("source_manifest_defaults", {})
    )
    return {
        "source_h5_path": str(path),
        "summary": summary,
        "sample_id": summary.get("sample_id") or path.stem,
        "scenario": summary.get("scenario"),
        "seed": int(summary["seed"]),
        "trigger_step": trigger_step,
        "first_target_motion_step": first_motion,
        "last_target_motion_frame": last_motion,
        "target_motion_xyz": motion_xyz.astype(float).tolist(),
        "target_motion_steps": motion_steps,
        "target_motion_per_step": float(motion_norm / motion_steps) if motion_norm > 0 else 0.0,
        "first_peg_perturb_step": first_peg_perturb_step,
        "peg_delta_first_xyz": peg_delta_first.astype(float).tolist(),
        "peg_delta_sum_xyz": peg_delta_sum.astype(float).tolist(),
        "source_motion_gate": {
            "trigger_reason": summary.get("trigger_reason"),
            "trigger_mode": source_defaults.get("trigger_mode"),
            "trigger_x_min": source_defaults.get("trigger_x_min"),
            "trigger_x_max": source_defaults.get("trigger_x_max"),
            "trigger_yz_threshold": source_defaults.get("trigger_yz_threshold"),
            "fallback_requires_robust_hold_and_relaxed_preinsert": source_defaults.get(
                "fallback_requires_robust_hold_and_relaxed_preinsert"
            ),
        },
    }


def load_source_h5_actions(source_h5_path: str) -> np.ndarray:
    if not source_h5_path:
        raise RuntimeError("source_h5_teacher_suffix_requires_source_h5_path")
    import h5py

    path = Path(source_h5_path).resolve()
    with h5py.File(path, "r") as h5:
        traj_names = sorted(name for name in h5.keys() if name.startswith("traj_"))
        if len(traj_names) != 1:
            raise RuntimeError(f"{path} expected one traj group, found {traj_names}")
        actions = np.asarray(h5[traj_names[0]]["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise RuntimeError(f"{path} invalid action shape {actions.shape}")
    return actions[:, :7]


def source_protocol_is_peg_perturb(source_protocol: dict[str, Any]) -> bool:
    return str(source_protocol.get("scenario") or "").strip() in {"peg_drop", "peg_disturb"}


VECTOR_NAMES = [
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


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def require_allocation() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("refusing_login_node_execution=true; run inside a tmux-held Slurm allocation")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def progress_marker(run_dir: Path, stage: str, **fields: Any) -> None:
    payload = {
        "time_unix": time.time(),
        "stage": stage,
        **fields,
    }
    line = json.dumps(jsonable(payload), sort_keys=True)
    with (run_dir / "progress.jsonl").open("a") as handle:
        handle.write(line + "\n")
    write_json(run_dir / "progress_last.json", payload)
    print(f"progress_stage={stage}", flush=True)


def write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def annotate_frame(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)[..., :3]).convert("RGB")
    draw = ImageDraw.Draw(image)
    pad = 5
    line_h = 14
    box_h = pad * 2 + line_h * len(lines)
    draw.rectangle((0, 0, image.width, box_h), fill=(0, 0, 0))
    for idx, line in enumerate(lines):
        draw.text((pad, pad + idx * line_h), line[:130], fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def annotated_frames(frames: list[np.ndarray], action_trace: list[dict[str, Any]]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for idx, frame in enumerate(frames):
        if idx == 0 or idx - 1 >= len(action_trace):
            lines = ["frame=0 reset", "stage=initial"]
        else:
            row = action_trace[idx - 1]
            state = row.get("live_eval") or {}
            lines = [
                f"frame={idx} env_step={row.get('env_step')} stage={row.get('stage')}",
                f"source={row.get('action_source')} peg_head_l2={state.get('peg_head_l2')}",
            ]
            if row.get("stage") == "cosmos_dynamic_control":
                lines.append(f"cosmos_round={row.get('cosmos_round')} action_idx={row.get('cosmos_action_index')}")
            if "target_motion_cumulative_xyz" in row:
                lines.append(f"target_motion_cumulative_xyz={row.get('target_motion_cumulative_xyz')}")
        out.append(annotate_frame(frame, lines))
    return out


def as_frame(frame: Any) -> np.ndarray:
    arr = np.asarray(jsonable(frame))
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max(initial=0) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


def camera_pose(eye: list[float], target: list[float], up: list[float] | None = None) -> Pose:
    kwargs = {"up": up} if up is not None else {}
    return sapien_utils.look_at(eye, target, **kwargs)


def oracle_human_render_camera_configs(shader_pack: str = "minimal") -> list[CameraConfig]:
    target = [0.0, 0.24, 0.10]
    return [
        CameraConfig("render_camera", camera_pose([0.4, 0.4, 0.8], [0.0, 0.0, 0.4]), 512, 512, 1.0, 0.01, 100, shader_pack=shader_pack),
        CameraConfig("oracle_hole_close", camera_pose([0.24, 0.55, 0.24], target), 512, 512, 0.62, 0.01, 100, shader_pack=shader_pack),
        CameraConfig("oracle_hole_side", camera_pose([-0.32, 0.42, 0.20], target), 512, 512, 0.58, 0.01, 100, shader_pack=shader_pack),
        CameraConfig("oracle_hole_top", camera_pose([0.0, 0.24, 0.72], target, up=[1.0, 0.0, 0.0]), 512, 512, 0.72, 0.01, 100, shader_pack=shader_pack),
    ]


def render_frame(env: Any, camera_uid: str | None = None) -> np.ndarray:
    if camera_uid is None:
        return as_frame(env.render())
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    return as_frame(common.to_numpy(base_env.render_rgb_array(camera_uid)))


def append_render_frames(
    env: Any,
    frames: list[np.ndarray],
    review_frames: dict[str, list[np.ndarray]],
    review_render_failures: list[dict[str, Any]],
) -> None:
    frames.append(render_frame(env))
    for camera_uid in REVIEW_CAMERA_UIDS:
        try:
            review_frames[camera_uid].append(render_frame(env, camera_uid))
        except Exception as exc:
            review_render_failures.append(
                {
                    "camera_uid": camera_uid,
                    "frame_index": len(frames) - 1,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )


def build_train_args(saved: dict[str, Any], args: Args, ms_train: Any) -> Any:
    train_args = ms_train.Args()
    for key, value in saved.items():
        if hasattr(train_args, key):
            setattr(train_args, key, value)
    train_args.env_id = args.env_id
    train_args.control_mode = args.control_mode
    train_args.sim_backend = args.sim_backend
    train_args.max_episode_steps = args.max_episode_steps
    train_args.num_eval_envs = 1
    train_args.num_eval_episodes = 1
    train_args.seed = args.seed
    train_args.capture_video = False
    train_args.render_shader_pack = args.render_shader_pack
    return train_args


def make_env(train_args: Any):
    assert train_args.sim_backend == "physx_cpu"

    def thunk():
        env = gym.make(
            train_args.env_id,
            reconfiguration_freq=1,
            control_mode=train_args.control_mode,
            reward_mode="sparse",
            obs_mode="state",
            render_mode="rgb_array",
            human_render_camera_configs=dict(shader_pack=train_args.render_shader_pack),
            max_episode_steps=train_args.max_episode_steps,
        )
        env = FrameStack(env, num_stack=train_args.obs_horizon)
        env = CPUGymWrapper(env, ignore_terminations=False, record_metrics=True)
        env.action_space.seed(train_args.seed)
        env.observation_space.seed(train_args.seed)
        return env

    return SyncVectorEnv([thunk])


def eval_state(base_env: Any) -> dict[str, Any]:
    info = base_env.evaluate()
    rel = np.asarray(jsonable(info["peg_head_pos_at_hole"]), dtype=np.float64).reshape(-1)[:3]

    def pose_p(obj: Any) -> list[float]:
        return np.asarray(jsonable(obj.pose.p), dtype=np.float64).reshape(-1)[:3].astype(float).tolist()

    tcp = [0.0, 0.0, 0.0]
    try:
        tcp = np.asarray(jsonable(base_env.agent.tcp.pose.p), dtype=np.float64).reshape(-1)[:3].astype(float).tolist()
    except Exception:
        pass

    return {
        "success": bool(np.asarray(info["success"]).reshape(-1)[0]),
        "peg_head_at_hole": rel.astype(float).tolist(),
        "peg_head_l2": float(np.linalg.norm(rel)),
        "tcp_xyz": tcp,
        "peg_xyz": pose_p(base_env.peg),
        "hole_xyz": np.asarray(jsonable(base_env.box_hole_pose.p), dtype=np.float64).reshape(-1)[:3].astype(float).tolist(),
        "peg_pose": jsonable(base_env.peg.pose.raw_pose),
        "box_hole_pose": jsonable(base_env.box_hole_pose.raw_pose),
        "goal_pose": jsonable(base_env.goal_pose.raw_pose),
    }


def source_target_motion_live_gate(args: Args, source_protocol: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    if not source_protocol:
        return {"ok": True, "reason": "no_source_h5_protocol_synthetic_diagnostic"}
    if not args.source_h5_require_live_motion_gate:
        return {"ok": True, "reason": "source_h5_live_motion_gate_disabled"}

    gate = source_protocol.get("source_motion_gate") or {}
    x_min_raw = gate.get("trigger_x_min")
    x_max_raw = gate.get("trigger_x_max")
    yz_raw = gate.get("trigger_yz_threshold")
    x_min = (float(x_min_raw) if x_min_raw is not None else -0.14) - float(args.source_h5_gate_x_margin)
    x_max = (float(x_max_raw) if x_max_raw is not None else 0.05) + float(args.source_h5_gate_x_margin)
    yz_max = (float(yz_raw) if yz_raw is not None else 0.035) + float(args.source_h5_gate_yz_margin)

    rel = np.asarray(state["peg_head_at_hole"], dtype=np.float64).reshape(-1)[:3]
    yz = float(np.linalg.norm(rel[1:3]))
    ok = bool(x_min <= float(rel[0]) <= x_max and yz <= yz_max)
    return {
        "ok": ok,
        "reason": "approved_source_h5_preinsert_gate" if ok else "live_state_not_at_source_h5_preinsert_gate",
        "peg_head_at_hole": rel.astype(float).tolist(),
        "peg_head_l2": float(state["peg_head_l2"]),
        "yz": yz,
        "x_min": x_min,
        "x_max": x_max,
        "yz_max": yz_max,
        "source_key": source_protocol.get("sample_id"),
        "trigger_reason": gate.get("trigger_reason"),
        "trigger_mode": gate.get("trigger_mode"),
    }


def hole_xyz(base_env: Any) -> np.ndarray:
    return np.asarray(jsonable(base_env.box_hole_pose.p), dtype=np.float64).reshape(-1)[:3]


def shift_box(base_env: Any, delta_xyz: np.ndarray) -> None:
    current = base_env.box.pose
    p = current.p + torch.as_tensor(delta_xyz, dtype=current.p.dtype, device=current.p.device).reshape(1, 3)
    base_env.box.set_pose(Pose.create_from_pq(p=p, q=current.q))


def peg_perturb_force_vector(args: Args, source_protocol: dict[str, Any]) -> np.ndarray:
    delta = np.asarray(source_protocol.get("peg_delta_first_xyz") or [0.0, 0.0, 0.0], dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(delta))
    if norm <= 1.0e-9:
        delta = np.asarray(source_protocol.get("peg_delta_sum_xyz") or [0.0, 0.0, 0.0], dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(delta))
    if norm <= 1.0e-9:
        return np.zeros((3,), dtype=np.float64)
    return delta / norm * float(args.source_h5_peg_force_scale)


def apply_physical_peg_perturb(base_env: Any, force_xyz: np.ndarray) -> None:
    base_env.peg.apply_force(force_xyz.astype(np.float32))


def install_forbidden_peg_state_guard(base_env: Any) -> dict[str, Any]:
    """Fail immediately if this runner or imported glue tries to edit peg state."""

    peg = getattr(base_env, "peg", None)
    report: dict[str, Any] = {"ok": peg is not None, "guarded_methods": [], "failures": []}
    if peg is None:
        report["failures"].append("base_env_has_no_peg")
        return report

    def blocked(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("forbidden_peg_state_intervention_guard_triggered")

    for name in ("set_pose", "set_state", "set_state_dict"):
        if not hasattr(peg, name):
            continue
        try:
            setattr(peg, name, blocked)
            report["guarded_methods"].append(name)
        except Exception as exc:
            report["failures"].append(
                {
                    "method": name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    report["ok"] = bool(report["guarded_methods"]) and not report["failures"]
    return report


def target_motion_vector(args: Args) -> np.ndarray:
    return np.asarray([args.target_motion_x, args.target_motion_y, args.target_motion_z], dtype=np.float64)


def apply_cosmos_action_direction_guard(
    action_vec: np.ndarray,
    args: Args,
) -> tuple[np.ndarray, dict[str, Any]]:
    guarded = action_vec.astype(np.float32).copy()
    report: dict[str, Any] = {
        "enabled": False,
        "guard": args.cosmos_action_direction_guard,
        "mode": args.cosmos_action_direction_guard_mode,
        "changed_axes": [],
    }
    if args.cosmos_action_direction_guard == "none":
        return guarded, report
    if args.cosmos_action_direction_guard != "source_motion_sign":
        raise RuntimeError(f"unsupported_cosmos_action_direction_guard={args.cosmos_action_direction_guard}")
    motion_sign = np.sign(target_motion_vector(args)[:3]).astype(np.float32)
    before = guarded[:3].copy()
    changed_axes: list[int] = []
    for axis, sign in enumerate(motion_sign):
        if sign == 0.0:
            continue
        if guarded[axis] * sign < 0.0:
            if args.cosmos_action_direction_guard_mode == "clip_opposite":
                guarded[axis] = 0.0
            elif args.cosmos_action_direction_guard_mode == "rectify_opposite":
                guarded[axis] = float(abs(float(guarded[axis])) * float(sign))
            else:
                raise RuntimeError(
                    "unsupported_cosmos_action_direction_guard_mode="
                    f"{args.cosmos_action_direction_guard_mode}"
                )
            changed_axes.append(axis)
    report.update(
        {
            "enabled": True,
            "target_motion_sign_xyz": motion_sign.astype(float).tolist(),
            "changed_axes": changed_axes,
            "before_xyz": before.astype(float).tolist(),
            "after_xyz": guarded[:3].astype(float).tolist(),
        }
    )
    return guarded, report


def next_target_motion_delta(args: Args, applied_xyz: np.ndarray) -> np.ndarray:
    remaining = target_motion_vector(args) - np.asarray(applied_xyz, dtype=np.float64).reshape(3)
    remaining_norm = float(np.linalg.norm(remaining))
    if remaining_norm <= 1.0e-6:
        return np.zeros((3,), dtype=np.float64)
    step = min(abs(float(args.target_motion_per_step)), remaining_norm)
    return remaining / remaining_norm * step


def target_motion_complete(args: Args, applied_xyz: np.ndarray) -> bool:
    remaining = target_motion_vector(args) - np.asarray(applied_xyz, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(remaining)) <= 1.0e-6


def target_motion_allows_finisher(args: Args, applied_xyz: np.ndarray) -> bool:
    return (not args.require_target_motion_complete_before_finisher) or target_motion_complete(args, applied_xyz)


def wam_row(
    action: np.ndarray,
    state: dict[str, Any],
    initial_hole: np.ndarray,
    step: int,
    total_steps: int,
    peg_delta_applied: np.ndarray | None = None,
    perturb_triggered: bool = False,
) -> list[float]:
    row = np.zeros((len(VECTOR_NAMES),), dtype=np.float32)
    row[:7] = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
    row[7:10] = np.asarray(state["tcp_xyz"], dtype=np.float32)
    row[10:13] = np.asarray(state["peg_xyz"], dtype=np.float32)
    hole = np.asarray(state["hole_xyz"], dtype=np.float32)
    row[13:16] = hole
    row[16:19] = np.asarray(state["peg_head_at_hole"], dtype=np.float32)
    row[19:22] = 0.0
    row[22] = 1.0
    row[23] = 1.0 if state["success"] else 0.0
    row[24:27] = hole - initial_hole.astype(np.float32)
    if peg_delta_applied is not None:
        row[27:30] = np.asarray(peg_delta_applied, dtype=np.float32).reshape(-1)[:3]
    else:
        row[27:30] = 0.0
    row[30] = float(step) / float(max(1, total_steps - 1))
    row[31] = 1.0 if perturb_triggered else 0.0
    return row.astype(float).tolist()


def discontinuity_audit(action_trace: list[dict[str, Any]], threshold: float = 0.08) -> dict[str, Any]:
    max_peg_step = 0.0
    max_hole_step = 0.0
    max_peg_head_l2_delta = 0.0
    max_peg_step_at = None
    max_hole_step_at = None
    prev_peg = None
    prev_hole = None
    prev_l2 = None
    for row in action_trace:
        state = row.get("live_eval") or {}
        peg = np.asarray(state.get("peg_xyz", []), dtype=np.float64).reshape(-1)
        hole = np.asarray(state.get("hole_xyz", []), dtype=np.float64).reshape(-1)
        l2 = state.get("peg_head_l2")
        if peg.size >= 3 and prev_peg is not None:
            delta = float(np.linalg.norm(peg[:3] - prev_peg[:3]))
            if delta > max_peg_step:
                max_peg_step = delta
                max_peg_step_at = row.get("env_step")
        if hole.size >= 3 and prev_hole is not None:
            delta = float(np.linalg.norm(hole[:3] - prev_hole[:3]))
            if delta > max_hole_step:
                max_hole_step = delta
                max_hole_step_at = row.get("env_step")
        if l2 is not None and prev_l2 is not None:
            max_peg_head_l2_delta = max(max_peg_head_l2_delta, abs(float(l2) - float(prev_l2)))
        if peg.size >= 3:
            prev_peg = peg[:3]
        if hole.size >= 3:
            prev_hole = hole[:3]
        if l2 is not None:
            prev_l2 = float(l2)
    return {
        "max_peg_step_displacement": max_peg_step,
        "max_peg_step_displacement_at": max_peg_step_at,
        "max_hole_step_displacement": max_hole_step,
        "max_hole_step_displacement_at": max_hole_step_at,
        "max_peg_head_l2_delta": max_peg_head_l2_delta,
        "snap_detected": bool(max_peg_step > threshold or max_hole_step > threshold),
        "snap_threshold": threshold,
        "boundary": "Automatic audit for discontinuous peg or target/hole motion; any snap requires invalid classification and visual review.",
    }


def clip_to_action_space(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, bool]:
    raw = np.asarray(action, dtype=np.float32).reshape(-1)
    clipped = np.clip(raw, low.reshape(-1)[: raw.size], high.reshape(-1)[: raw.size])
    return clipped.astype(np.float32), bool(np.max(np.abs(clipped - raw)) > 1.0e-6)


def manual_oracle_servo_action(
    args: Args,
    state: dict[str, Any],
    action_dim: int,
) -> np.ndarray:
    """Physical upper-bound finisher: live error in, controller action out."""

    rel = np.asarray(state["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
    action = np.zeros((action_dim,), dtype=np.float32)
    if action_dim >= 1:
        action[0] = np.clip(
            args.manual_lateral_gain * rel[1],
            -args.manual_lateral_limit,
            args.manual_lateral_limit,
        )
    if action_dim >= 2:
        forward = args.manual_forward_bias + args.manual_forward_gain * float(-rel[0])
        action[1] = np.clip(forward, -args.manual_forward_limit, args.manual_forward_limit)
    if action_dim >= 3:
        action[2] = np.clip(
            -args.manual_vertical_gain * rel[2],
            -args.manual_vertical_limit,
            args.manual_vertical_limit,
        )
    if action_dim >= 6:
        action[5] = float(args.manual_yaw_action)
    if action_dim >= 7:
        action[6] = float(args.manual_gripper_action)
    return action


def quat_wxyz_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(-1)[:4]
    v = np.asarray(v, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-8:
        return v
    w, x, y, z = q / norm
    q_vec = np.asarray([x, y, z], dtype=np.float32)
    return v + 2.0 * np.cross(q_vec, np.cross(q_vec, v) + w * v)


def quat_wxyz_error_vector(target_q: np.ndarray, current_q: np.ndarray) -> np.ndarray:
    """Small-angle rotation vector that rotates current orientation toward target."""

    target = np.asarray(target_q, dtype=np.float32).reshape(-1)[:4]
    current = np.asarray(current_q, dtype=np.float32).reshape(-1)[:4]
    target_norm = float(np.linalg.norm(target))
    current_norm = float(np.linalg.norm(current))
    if target_norm <= 1e-6 or current_norm <= 1e-6:
        return np.zeros((3,), dtype=np.float32)
    target = target / target_norm
    current = current / current_norm
    tw, tx, ty, tz = target
    cw, cx, cy, cz = current
    err = np.asarray(
        [
            tw * cw + tx * cx + ty * cy + tz * cz,
            -tw * cx + tx * cw - ty * cz + tz * cy,
            -tw * cy + tx * cz + ty * cw - tz * cx,
            -tw * cz - tx * cy + ty * cx + tz * cw,
        ],
        dtype=np.float32,
    )
    if err[0] < 0.0:
        err *= -1.0
    return 2.0 * err[1:4]


def manual_hole_frame_servo_action(
    args: Args,
    state: dict[str, Any],
    action_dim: int,
) -> np.ndarray:
    """Compute local hole-frame correction, then rotate it into action space."""

    rel = np.asarray(state["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
    local_delta = np.zeros((3,), dtype=np.float32)
    local_delta[0] = np.clip(
        args.manual_forward_bias - args.manual_forward_gain * float(rel[0]),
        -args.manual_forward_limit,
        args.manual_forward_limit,
    )
    local_delta[1] = np.clip(
        -args.manual_lateral_gain * float(rel[1]),
        -args.manual_lateral_limit,
        args.manual_lateral_limit,
    )
    local_delta[2] = np.clip(
        -args.manual_vertical_gain * float(rel[2]),
        -args.manual_vertical_limit,
        args.manual_vertical_limit,
    )

    pose = np.asarray(state["box_hole_pose"], dtype=np.float32).reshape(-1)
    q_wxyz = pose[3:7] if pose.size >= 7 else np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    world_delta = quat_wxyz_rotate(q_wxyz, local_delta)

    action = np.zeros((action_dim,), dtype=np.float32)
    action[: min(3, action_dim)] = world_delta[: min(3, action_dim)]
    if action_dim >= 6:
        yaw_action = float(args.manual_yaw_action)
        if float(args.manual_yaw_stop_l2) >= 0.0 and float(state["peg_head_l2"]) <= float(args.manual_yaw_stop_l2):
            yaw_action = 0.0
        action[5] = yaw_action
    if action_dim >= 7:
        action[6] = float(args.manual_gripper_action)
    return action


def manual_staged_hole_servo_action(
    args: Args,
    state: dict[str, Any],
    action_dim: int,
) -> np.ndarray:
    """Align in the hole frame before making slow physical insertion progress."""

    rel = np.asarray(state["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
    yz_error = float(np.linalg.norm(rel[1:3]))
    local_delta = np.zeros((3,), dtype=np.float32)

    soft_insert_enabled = (
        float(args.manual_soft_insert_threshold) >= 0.0
        and yz_error <= float(args.manual_soft_insert_threshold)
    )

    if yz_error > args.manual_yz_abort_threshold:
        local_delta[0] = -min(float(args.manual_retreat_speed), float(args.manual_forward_limit))
    elif yz_error <= args.manual_align_threshold or soft_insert_enabled:
        if rel[0] < 0.0:
            insert_scale = 1.0 if yz_error <= args.manual_align_threshold else float(args.manual_soft_insert_scale)
            insert_scale = float(np.clip(insert_scale, 0.0, 1.0))
            remaining_forward = min(
                float(args.manual_insert_speed) * insert_scale,
                float(-args.manual_forward_gain * rel[0]) * insert_scale,
            )
            local_delta[0] = min(remaining_forward, float(args.manual_forward_limit))
        else:
            local_delta[0] = -min(float(args.manual_retreat_speed), float(args.manual_forward_limit))

    local_delta[1] = np.clip(
        -args.manual_lateral_gain * float(rel[1]),
        -args.manual_lateral_limit,
        args.manual_lateral_limit,
    )
    local_delta[2] = np.clip(
        -args.manual_vertical_gain * float(rel[2]),
        -args.manual_vertical_limit,
        args.manual_vertical_limit,
    )

    pose = np.asarray(state["box_hole_pose"], dtype=np.float32).reshape(-1)
    q_wxyz = pose[3:7] if pose.size >= 7 else np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    world_delta = quat_wxyz_rotate(q_wxyz, local_delta)

    action = np.zeros((action_dim,), dtype=np.float32)
    action[: min(3, action_dim)] = world_delta[: min(3, action_dim)]
    if action_dim >= 6:
        yaw_action = float(args.manual_yaw_action)
        if float(args.manual_yaw_stop_l2) >= 0.0 and float(state["peg_head_l2"]) <= float(args.manual_yaw_stop_l2):
            yaw_action = 0.0
        action[5] = yaw_action
    if action_dim >= 7:
        action[6] = float(args.manual_gripper_action)
    return action


def manual_staged_twist_insert_action(
    args: Args,
    state: dict[str, Any],
    action_dim: int,
) -> np.ndarray:
    """Staged insertion with DP-like wrist twist during the physical insert."""

    action = manual_staged_hole_servo_action(args, state, action_dim)
    rel = np.asarray(state["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
    yz_error = float(np.linalg.norm(rel[1:3]))
    in_insert_stage = bool(yz_error <= args.manual_align_threshold and rel[0] < 0.0)
    if in_insert_stage:
        if action_dim >= 4:
            action[3] = float(args.manual_insert_roll_action)
        if action_dim >= 5:
            action[4] = float(args.manual_insert_pitch_action)
        if action_dim >= 6:
            action[5] = float(args.manual_insert_yaw_action)
    return action


def manual_staged_pose_servo_action(
    args: Args,
    state: dict[str, Any],
    action_dim: int,
) -> np.ndarray:
    """Staged insertion with live quaternion-error wrist correction near the hole."""

    action = manual_staged_hole_servo_action(args, state, action_dim)
    if action_dim < 6:
        return action
    rel = np.asarray(state["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
    yz_error = float(np.linalg.norm(rel[1:3]))
    if yz_error > float(args.manual_pose_rot_yz_threshold) or rel[0] >= 0.0:
        return action
    peg_pose = np.asarray(state["peg_pose"], dtype=np.float32).reshape(-1)
    hole_pose = np.asarray(state["box_hole_pose"], dtype=np.float32).reshape(-1)
    if peg_pose.size < 7 or hole_pose.size < 7:
        return action
    rot_vec = quat_wxyz_error_vector(hole_pose[3:7], peg_pose[3:7])
    rot_cmd = np.clip(
        float(args.manual_pose_rot_gain) * rot_vec,
        -float(args.manual_pose_rot_limit),
        float(args.manual_pose_rot_limit),
    )
    action[3:6] = rot_cmd[:3]
    return action


def manual_regrasp_then_hole_servo_action(
    args: Args,
    state: dict[str, Any],
    action_dim: int,
) -> np.ndarray:
    """Move the TCP back to a dropped peg before using the hole-frame insert servo."""

    tcp = np.asarray(state["tcp_xyz"], dtype=np.float32).reshape(-1)[:3]
    peg = np.asarray(state["peg_xyz"], dtype=np.float32).reshape(-1)[:3]
    target_tcp = peg + np.asarray([0.0, 0.0, 0.045], dtype=np.float32)
    tcp_error = target_tcp - tcp
    tcp_error_l2 = float(np.linalg.norm(tcp_error))

    if tcp_error_l2 > max(float(args.manual_align_threshold), 0.025):
        action = np.zeros((action_dim,), dtype=np.float32)
        limits = np.asarray(
            [
                float(args.manual_lateral_limit),
                float(args.manual_lateral_limit),
                float(args.manual_vertical_limit),
            ],
            dtype=np.float32,
        )
        gains = np.asarray(
            [
                float(args.manual_lateral_gain),
                float(args.manual_lateral_gain),
                float(args.manual_vertical_gain),
            ],
            dtype=np.float32,
        )
        delta = np.clip(gains * tcp_error, -limits, limits)
        action[: min(3, action_dim)] = delta[: min(3, action_dim)]
        if action_dim >= 7:
            action[6] = 1.0
        return action

    action = manual_hole_frame_servo_action(args, state, action_dim)
    if action_dim >= 7:
        action[6] = float(args.manual_gripper_action)
    return action


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_s: int | None = None,
) -> None:
    with log_path.open("a") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        if timeout_s is not None:
            log.write(f"timeout_s={timeout_s}\n")
        log.flush()
        subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout_s,
        )


def cosmos_env(args: Args) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.cosmos_framework_path}:{env.get('PYTHONPATH', '')}"
    env["WAN_VAE_PATH"] = args.wan_vae_path
    env["COSMOS3_LOCAL_TOKENIZER_DIR"] = args.cosmos_tokenizer_dir
    env["AWS_EC2_METADATA_DISABLED"] = "true"
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)
    env.pop("ALL_PROXY", None)
    env.pop("all_proxy", None)
    env.pop("NO_PROXY", None)
    env.pop("no_proxy", None)
    return env


def run_cosmos_policy(
    args: Args,
    *,
    root: Path,
    run_dir: Path,
    stage_name: str,
    prefix_role: str,
    prefix_frame_index: int,
    frames: list[np.ndarray],
    wam_history: list[list[float]],
) -> dict[str, Any]:
    stats_path = Path(args.cosmos_normalization_stats)
    if not stats_path.is_file():
        raise FileNotFoundError(f"missing Cosmos normalization stats: {stats_path}")

    stage_dir = run_dir / "cosmos_policy" / stage_name
    prefix_video = stage_dir / "prefix_rgb.mp4"
    history_json = stage_dir / "history_wam_actions.json"
    write_mp4(prefix_video, frames, args.fps)
    write_json(history_json, {"action": wam_history})

    sample_name = "sample"
    log_path = stage_dir / "cosmos_policy.log"
    project_env = os.environ.copy()
    progress_marker(
        run_dir,
        "cosmos_build_start",
        stage_name=stage_name,
        prefix_frame_index=prefix_frame_index,
        timeout_s=args.cosmos_build_timeout_s,
    )
    run_cmd(
        [
            args.project_python,
            "-u",
            "scripts/world_model/build_cosmos3_live_prefix_wam_input.py",
            "--output-root",
            str(stage_dir / "input"),
            "--prefix-video",
            str(prefix_video),
            "--normalization-stats",
            str(stats_path),
            "--prefix-frame-index",
            str(prefix_frame_index),
            "--expected-prefix-video-frames",
            str(len(frames)),
            "--sample-name",
            sample_name,
            "--scenario",
            "phase03_live_oracle",
            "--prefix-role",
            prefix_role,
            "--history-action-path",
            str(history_json),
            "--condition-root",
            str(stats_path.parent),
            "--checkpoint-path",
            args.cosmos_checkpoint_path,
            "--fps",
            str(args.fps),
        ],
        cwd=root,
        env=project_env,
        log_path=log_path,
        timeout_s=int(args.cosmos_build_timeout_s),
    )
    progress_marker(run_dir, "cosmos_build_done", stage_name=stage_name)

    input_jsonl = stage_dir / "input" / "inputs" / "live_prefix_wam_policy_samples.jsonl"
    cosmos_out = stage_dir / "outputs"
    progress_marker(
        run_dir,
        "cosmos_inference_start",
        stage_name=stage_name,
        timeout_s=args.cosmos_inference_timeout_s,
    )
    run_cmd(
        [
            args.cosmos_python,
            "-m",
            "cosmos_framework.scripts.inference",
            "--parallelism-preset=latency",
            "--no-guardrails",
            f"--config-file={args.cosmos_config_file}",
            f"--experiment={args.cosmos_experiment}",
            "-i",
            str(input_jsonl),
            "-o",
            str(cosmos_out),
            "--checkpoint-path",
            args.cosmos_checkpoint_path,
            "--seed=0",
        ],
        cwd=root,
        env=cosmos_env(args),
        log_path=log_path,
        timeout_s=int(args.cosmos_inference_timeout_s),
    )
    progress_marker(run_dir, "cosmos_inference_done", stage_name=stage_name)

    sample_output = cosmos_out / sample_name / "sample_outputs.json"
    cosmos_vision = cosmos_out / sample_name / "vision.mp4"
    action_json = stage_dir / "cosmos_action_chunk.json"
    progress_marker(
        run_dir,
        "cosmos_extract_start",
        stage_name=stage_name,
        timeout_s=args.cosmos_extract_timeout_s,
    )
    run_cmd(
        [
            args.project_python,
            "-u",
            "scripts/world_model/extract_cosmos3_policy_action_chunk.py",
            "--sample-output-json",
            str(sample_output),
            "--normalization-stats",
            str(stats_path),
            "--prefix-frame-index",
            str(prefix_frame_index),
            "--action-row-offset",
            str(args.cosmos_action_row_offset),
            "--action-exec-horizon",
            str(args.cosmos_action_horizon),
            "--output-json",
            str(action_json),
        ],
        cwd=root,
        env=project_env,
        log_path=log_path,
        timeout_s=int(args.cosmos_extract_timeout_s),
    )
    progress_marker(run_dir, "cosmos_extract_done", stage_name=stage_name)
    return json.loads(action_json.read_text()) | {
        "stage_name": stage_name,
        "prefix_role": prefix_role,
        "prefix_frame_index": int(prefix_frame_index),
        "stage_dir": str(stage_dir),
        "prefix_video": str(prefix_video),
        "cosmos_rgb_prediction_video": str(cosmos_vision),
        "sample_output_json": str(sample_output),
        "cosmos_action_chunk_json": str(action_json),
    }


def main() -> int:
    require_allocation()
    args = tyro.cli(Args)
    if "MANUAL_DP_TO_MANUAL_L2" in os.environ:
        args.manual_dp_to_manual_l2 = float(os.environ["MANUAL_DP_TO_MANUAL_L2"])
    root = Path(__file__).resolve().parents[2]
    run_dir = Path(args.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_marker(run_dir, "main_start", source_h5_path=args.source_h5_path, source_key=args.source_key)
    source_protocol: dict[str, Any] = {}
    if args.source_h5_path:
        try:
            progress_marker(run_dir, "source_h5_load_start")
            source_protocol = load_source_h5_protocol(args.source_h5_path)
            if args.source_key and args.source_key != source_protocol["sample_id"]:
                raise RuntimeError(
                    f"source_key_mismatch requested={args.source_key} h5={source_protocol['sample_id']}"
                )
            args.source_key = str(source_protocol["sample_id"])
            if args.scenario_name == "target_y_positive":
                args.scenario_name = f"source_{args.source_key}"
            args.seed = int(source_protocol["seed"])
            args.target_motion_step = int(source_protocol["first_target_motion_step"])
            motion_xyz = np.asarray(source_protocol["target_motion_xyz"], dtype=np.float64).reshape(3)
            args.target_motion_x = float(motion_xyz[0])
            args.target_motion_y = float(motion_xyz[1])
            args.target_motion_z = float(motion_xyz[2])
            args.target_motion_per_step = float(source_protocol["target_motion_per_step"])
            progress_marker(
                run_dir,
                "source_h5_load_done",
                source_key=args.source_key,
                scenario=source_protocol.get("scenario"),
                seed=args.seed,
                target_motion_step=args.target_motion_step,
            )
        except Exception as exc:
            summary = {
                "schema": "phase03_oracle_full_pipeline_v1",
                "method_evidence_allowed": False,
                "physical_insertion_success_claimed": False,
                "classification": "blocked_source_h5_protocol_load_failed_no_rollout",
                "source_h5_path": args.source_h5_path,
                "source_key": args.source_key,
                "source_protocol_failure": {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            }
            write_json(run_dir / "summary.json", summary)
            (run_dir / "classification.txt").write_text(
                "phase03_status=blocked_source_h5_protocol_load_failed_no_rollout\n"
                "method_evidence_allowed=false\nphysical_insertion_success=false\n"
            )
            print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
            return 45
    target_motion_source = (
        "fix3_733_source_h5_protocol" if source_protocol else "synthetic_runner_args_diagnostic_only"
    )
    dynamic_protocol_type = (
        "peg_perturb" if source_protocol_is_peg_perturb(source_protocol)
        else "target_motion"
    )
    source_h5_teacher_dynamic_enabled = args.dynamic_controller == "source_h5_teacher_dynamic"
    if args.dynamic_controller not in {"cosmos3_policy", "source_h5_teacher_dynamic"}:
        raise RuntimeError(f"unsupported_dynamic_controller={args.dynamic_controller}")
    source_h5_teacher_actions: np.ndarray | None = None
    source_h5_teacher_suffix_enabled = args.finisher_controller == "source_h5_teacher_suffix"
    if source_h5_teacher_suffix_enabled or source_h5_teacher_dynamic_enabled:
        try:
            progress_marker(run_dir, "source_h5_teacher_actions_load_start")
            source_h5_teacher_actions = load_source_h5_actions(args.source_h5_path)
            progress_marker(
                run_dir,
                "source_h5_teacher_actions_load_done",
                rows=int(source_h5_teacher_actions.shape[0]),
            )
        except Exception as exc:
            summary = {
                "schema": "phase03_oracle_full_pipeline_v1",
                "method_evidence_allowed": False,
                "physical_insertion_success_claimed": False,
                "classification": "blocked_source_h5_teacher_actions_load_failed_no_rollout",
                "source_h5_path": args.source_h5_path,
                "source_key": args.source_key,
                "future_label_teacher_suffix_diagnostic": bool(source_h5_teacher_suffix_enabled),
                "future_label_teacher_dynamic_diagnostic": bool(source_h5_teacher_dynamic_enabled),
                "source_h5_teacher_actions_failure": {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            }
            write_json(run_dir / "summary.json", summary)
            (run_dir / "classification.txt").write_text(
                "phase03_status=blocked_source_h5_teacher_actions_load_failed_no_rollout\n"
                "method_evidence_allowed=false\nphysical_insertion_success=false\n"
                f"future_label_teacher_suffix_diagnostic={str(bool(source_h5_teacher_suffix_enabled)).lower()}\n"
                f"future_label_teacher_dynamic_diagnostic={str(bool(source_h5_teacher_dynamic_enabled)).lower()}\n"
            )
            print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
            return 45

    action_row_offset_diagnostic = int(args.cosmos_action_row_offset) != 0
    manifest = {
        "schema": "phase03_oracle_full_pipeline_v1",
        "phase": "03_oracle",
        "method_evidence_allowed": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.uname().nodename,
        "ckpt_path": args.ckpt_path,
        "cosmos_config_file": args.cosmos_config_file,
        "cosmos_checkpoint_path": args.cosmos_checkpoint_path,
        "cosmos_normalization_stats": args.cosmos_normalization_stats,
        "cosmos_python": args.cosmos_python,
        "project_python": args.project_python,
        "source_h5_path": args.source_h5_path,
        "source_key": args.source_key,
        "source_protocol": source_protocol,
        "target_motion_source": target_motion_source,
        "dynamic_protocol_type": dynamic_protocol_type,
        "source_h5_peg_perturb_mode": args.source_h5_peg_perturb_mode,
        "source_h5_peg_force_scale": float(args.source_h5_peg_force_scale),
        "source_h5_peg_force_steps": int(args.source_h5_peg_force_steps),
        "synthetic_target_motion_diagnostic_only": not bool(source_protocol),
        "validation_key_success_allowed": bool(source_protocol) and not action_row_offset_diagnostic,
        "scenario_name": args.scenario_name,
        "seed": int(args.seed),
        "target_motion_step": int(args.target_motion_step),
        "target_motion_xyz": target_motion_vector(args).astype(float).tolist(),
        "target_motion_per_step": float(args.target_motion_per_step),
        "target_motion_during_finisher": bool(args.target_motion_during_finisher),
        "require_target_motion_complete_before_finisher": bool(
            args.require_target_motion_complete_before_finisher
        ),
        "premotion_cosmos_step": int(args.premotion_cosmos_step),
        "premotion_cosmos_interval": int(args.premotion_cosmos_interval),
        "max_premotion_cosmos_predictions": int(args.max_premotion_cosmos_predictions),
        "required_premotion_cosmos_predictions": max(2, int(args.max_premotion_cosmos_predictions)),
        "cosmos_action_adapter": {
            "boundary": (
                "Optional documented adapter applied only to Cosmos-derived "
                "dynamic-stage actions before action-space clipping. Defaults "
                "are identity. Non-identity settings are diagnostic and are not "
                "method evidence."
            ),
            "scale_x": float(args.cosmos_action_scale_x),
            "scale_y": float(args.cosmos_action_scale_y),
            "scale_z": float(args.cosmos_action_scale_z),
            "scale_rot": float(args.cosmos_action_scale_rot),
            "scale_gripper": float(args.cosmos_action_scale_gripper),
            "direction_guard": str(args.cosmos_action_direction_guard),
            "direction_guard_mode": str(args.cosmos_action_direction_guard_mode),
            "action_row_offset": int(args.cosmos_action_row_offset),
            "action_row_offset_source": str(args.cosmos_action_row_offset_source),
            "action_row_offset_diagnostic": bool(action_row_offset_diagnostic),
            "identity": bool(
                abs(float(args.cosmos_action_scale_x) - 1.0) < 1.0e-9
                and abs(float(args.cosmos_action_scale_y) - 1.0) < 1.0e-9
                and abs(float(args.cosmos_action_scale_z) - 1.0) < 1.0e-9
                and abs(float(args.cosmos_action_scale_rot) - 1.0) < 1.0e-9
                and abs(float(args.cosmos_action_scale_gripper) - 1.0) < 1.0e-9
                and args.cosmos_action_direction_guard == "none"
                and int(args.cosmos_action_row_offset) == 0
            ),
        },
        "dynamic_controller": {
            "name": str(args.dynamic_controller),
            "future_label_teacher_dynamic_diagnostic": bool(source_h5_teacher_dynamic_enabled),
            "boundary": (
                "Dynamic stage normally executes Cosmos-3 policy actions. "
                "source_h5_teacher_dynamic is a future-label diagnostic only: "
                "Cosmos RGB/action reports are still produced, but executed "
                "dynamic actions come from the matching approved source H5 "
                "through env.step. It is not method evidence or success."
            ),
            "teacher_action_start_offset": int(args.source_h5_teacher_dynamic_action_start_offset),
        },
        "contract": "DP static prefix -> Cosmos RGB/action dynamic control -> physical finisher.",
        "finisher_controller": args.finisher_controller,
        "source_h5_teacher_suffix": {
            "enabled": bool(source_h5_teacher_suffix_enabled),
            "boundary": (
                "Future-label teacher action suffix diagnostic only. It may "
                "execute source-H5 actions through env.step after the "
                "near-target gate, but it must not be counted as method "
                "evidence or physical insertion success."
            ),
            "action_start_offset": int(args.source_h5_teacher_action_start_offset),
            "action_rows": int(source_h5_teacher_actions.shape[0])
            if source_h5_teacher_actions is not None
            else None,
        },
        "manual_oracle_servo": {
            "forward_bias": args.manual_forward_bias,
            "forward_gain": args.manual_forward_gain,
            "forward_limit": args.manual_forward_limit,
            "lateral_gain": args.manual_lateral_gain,
            "lateral_limit": args.manual_lateral_limit,
            "vertical_gain": args.manual_vertical_gain,
            "vertical_limit": args.manual_vertical_limit,
            "yaw_action": args.manual_yaw_action,
            "yaw_stop_l2": args.manual_yaw_stop_l2,
            "gripper_action": args.manual_gripper_action,
            "align_threshold": args.manual_align_threshold,
            "yz_abort_threshold": args.manual_yz_abort_threshold,
            "soft_insert_threshold": args.manual_soft_insert_threshold,
            "soft_insert_scale": args.manual_soft_insert_scale,
            "insert_speed": args.manual_insert_speed,
            "retreat_speed": args.manual_retreat_speed,
            "insert_roll_action": args.manual_insert_roll_action,
            "insert_pitch_action": args.manual_insert_pitch_action,
            "insert_yaw_action": args.manual_insert_yaw_action,
            "pose_rot_gain": args.manual_pose_rot_gain,
            "pose_rot_limit": args.manual_pose_rot_limit,
            "pose_rot_yz_threshold": args.manual_pose_rot_yz_threshold,
            "dp_to_manual_l2": args.manual_dp_to_manual_l2,
        },
    }
    write_json(run_dir / "manifest.json", manifest)

    summary: dict[str, Any] = {
        "schema": "phase03_oracle_full_pipeline_v1",
        "method_evidence_allowed": False,
        "physical_insertion_success_claimed": False,
        "visual_full_insertion_confirmed": False,
        "simulator_success_metric": False,
        "forbidden_peg_state_intervention_used": False,
        "set_pose_used_on_peg": False,
        "target_motion_state_intervention_used": False,
        "peg_physical_perturb_applied": False,
        "peg_physical_perturb_force_xyz": [0.0, 0.0, 0.0],
        "source_h5_path": args.source_h5_path,
        "source_key": args.source_key,
        "source_protocol": source_protocol,
        "target_motion_source": target_motion_source,
        "dynamic_protocol_type": dynamic_protocol_type,
        "source_h5_peg_perturb_mode": args.source_h5_peg_perturb_mode,
        "source_h5_peg_force_scale": float(args.source_h5_peg_force_scale),
        "source_h5_peg_force_steps": int(args.source_h5_peg_force_steps),
        "synthetic_target_motion_diagnostic_only": not bool(source_protocol),
        "validation_key_success_allowed": bool(source_protocol) and not action_row_offset_diagnostic,
        "scenario_name": args.scenario_name,
        "seed": int(args.seed),
        "target_motion_step": int(args.target_motion_step),
        "target_motion_xyz": target_motion_vector(args).astype(float).tolist(),
        "target_motion_per_step": float(args.target_motion_per_step),
        "target_motion_during_finisher": bool(args.target_motion_during_finisher),
        "target_motion_frozen_during_finisher": not bool(args.target_motion_during_finisher),
        "require_target_motion_complete_before_finisher": bool(
            args.require_target_motion_complete_before_finisher
        ),
        "cosmos_action_adapter": manifest["cosmos_action_adapter"],
        "action_row_offset_diagnostic": bool(action_row_offset_diagnostic),
        "dynamic_controller": manifest["dynamic_controller"],
        "source_h5_teacher_suffix": manifest["source_h5_teacher_suffix"],
        "future_label_teacher_suffix_diagnostic": bool(source_h5_teacher_suffix_enabled),
        "future_label_teacher_dynamic_diagnostic": bool(source_h5_teacher_dynamic_enabled),
        "classification": "started",
        "finisher_controller": args.finisher_controller,
    }

    if source_protocol_is_peg_perturb(source_protocol) and args.source_h5_peg_perturb_mode != "force":
        summary.update(
            {
                "classification": "blocked_source_h5_peg_perturb_requires_physical_force_mode_no_rollout",
                "reason": "Approved peg_drop/peg_disturb keys require a physical perturbation path. The runner refuses to treat them as target-motion cases or to use peg state edits.",
            }
        )
        write_json(run_dir / "summary.json", summary)
        (run_dir / "classification.txt").write_text(
            "phase03_status=blocked_source_h5_peg_perturb_requires_physical_force_mode_no_rollout\n"
            "method_evidence_allowed=false\n"
            "physical_insertion_success=false\n"
            "cosmos_dynamic_actions_executed=false\n"
            "oracle_set_pose_used=false\n"
            "peg_state_edit_used=false\n"
        )
        print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
        return 45

    if not Path(args.cosmos_normalization_stats).is_file():
        summary.update(
            {
                "classification": "blocked_missing_cosmos_normalization_stats_no_action_execution",
                "missing_file": args.cosmos_normalization_stats,
                "reason": "Cannot denormalize official Cosmos policy action output without WAM normalization stats.",
            }
        )
        write_json(run_dir / "summary.json", summary)
        (run_dir / "classification.txt").write_text(
            "phase03_status=blocked_missing_cosmos_normalization_stats_no_action_execution\n"
            "method_evidence_allowed=false\nphysical_insertion_success=false\n"
        )
        return 44

    try:
        progress_marker(run_dir, "initialization_start")
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        progress_marker(run_dir, "dp_train_import_start")
        ms_train = importlib.import_module("train")
        progress_marker(run_dir, "dp_train_import_done")
        progress_marker(run_dir, "checkpoint_load_start", device=str(device), ckpt_path=args.ckpt_path)
        ckpt = torch.load(args.ckpt_path, map_location=device)
        progress_marker(run_dir, "checkpoint_load_done")
        train_args = build_train_args(ckpt.get("args", {}), args, ms_train)
        ms_train.args = train_args
        ms_train.device = device
        progress_marker(run_dir, "make_env_start", env_id=train_args.env_id, seed=train_args.seed)
        envs = make_env(train_args)
        progress_marker(run_dir, "make_env_done")
        progress_marker(run_dir, "agent_init_start")
        agent = ms_train.Agent(envs, train_args).to(device)
        progress_marker(run_dir, "agent_init_done")
        state_key = "ema_agent" if args.use_ema else "agent"
        progress_marker(run_dir, "agent_state_load_start", state_key=state_key)
        agent.load_state_dict(ckpt[state_key])
        agent.eval()
        progress_marker(run_dir, "initialization_done")
    except Exception as exc:
        summary.update(
            {
                "classification": "blocked_initialization_failed_no_rollout",
                "initialization_failure": {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            }
        )
        write_json(run_dir / "summary.json", summary)
        (run_dir / "classification.txt").write_text(
            "phase03_status=blocked_initialization_failed_no_rollout\n"
            "method_evidence_allowed=false\n"
            "physical_insertion_success=false\n"
            "cosmos_dynamic_actions_executed=false\n"
            "oracle_set_pose_used=false\n"
        )
        print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
        return 45

    frames: list[np.ndarray] = []
    review_frames: dict[str, list[np.ndarray]] = {camera_uid: [] for camera_uid in REVIEW_CAMERA_UIDS}
    review_render_failures: list[dict[str, Any]] = []
    action_trace: list[dict[str, Any]] = []
    wam_history: list[list[float]] = []
    cosmos_reports: list[dict[str, Any]] = []
    cosmos_failure: dict[str, Any] | None = None
    runtime_failure: dict[str, Any] | None = None
    source_h5_teacher_suffix_exhausted: dict[str, Any] | None = None
    source_h5_teacher_dynamic_exhausted: dict[str, Any] | None = None
    source_motion_gate_rejections: list[dict[str, Any]] = []
    final_success = False
    target_assisted_metric_true_before_motion_complete: dict[str, Any] | None = None
    manual_finisher_controllers = {
        "manual_oracle_servo",
        "manual_hole_frame_servo",
        "manual_staged_hole_servo",
        "manual_staged_twist_insert",
        "manual_staged_pose_servo",
        "manual_staged_dp_rot",
        "manual_align_then_dp",
        "dp_then_manual_close",
        "manual_regrasp_then_hole_servo",
    }

    try:
        with torch.no_grad():
            progress_marker(run_dir, "env_reset_start", seed=args.seed)
            obs, reset_info = envs.reset(seed=args.seed)
            progress_marker(run_dir, "env_reset_done")
            base_env = envs.envs[0].unwrapped
            progress_marker(run_dir, "peg_state_guard_start")
            peg_state_guard = install_forbidden_peg_state_guard(base_env)
            summary["peg_state_guard"] = peg_state_guard
            if not peg_state_guard["ok"]:
                summary.update(
                    {
                        "classification": "blocked_peg_state_guard_install_failed_no_rollout",
                        "guard_failure": peg_state_guard,
                    }
                )
                write_json(run_dir / "summary.json", summary)
                (run_dir / "classification.txt").write_text(
                    "phase03_status=blocked_peg_state_guard_install_failed_no_rollout\n"
                    "method_evidence_allowed=false\n"
                    "physical_insertion_success=false\n"
                    "cosmos_dynamic_actions_executed=false\n"
                    "oracle_set_pose_used=false\n"
                    "peg_state_guard_ok=false\n"
                )
                envs.close()
                print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
                return 45
            progress_marker(run_dir, "peg_state_guard_done", ok=peg_state_guard["ok"])
            action_low = np.asarray(envs.single_action_space.low, dtype=np.float32).reshape(-1)
            action_high = np.asarray(envs.single_action_space.high, dtype=np.float32).reshape(-1)
            progress_marker(run_dir, "initial_render_start")
            append_render_frames(envs.envs[0], frames, review_frames, review_render_failures)
            progress_marker(run_dir, "initial_render_done", frames=len(frames))
            initial_hole = hole_xyz(base_env)
            initial_eval = eval_state(base_env)
            target_motion_applied = False
            target_motion_applied_xyz = np.zeros((3,), dtype=np.float64)
            peg_perturb_triggered = False
            peg_perturb_force_xyz = peg_perturb_force_vector(args, source_protocol)
            peg_perturb_force_steps_remaining = 0
            peg_perturb_reference_xyz: np.ndarray | None = None
            peg_perturb_observed_cumulative_delta = np.zeros((3,), dtype=np.float64)
            trigger_frame = None
            premotion_cosmos_count = 0
            next_premotion_cosmos_step = int(args.premotion_cosmos_step)

            for chunk_idx in range(train_args.max_episode_steps):
                obs_tensor = common.to_tensor(obs, device)
                dp_actions = agent.get_action(obs_tensor).detach().cpu().numpy()
                for action_idx in range(dp_actions.shape[1]):
                    step_idx = len(action_trace)
                    if (
                        step_idx >= next_premotion_cosmos_step
                        and step_idx < args.target_motion_step
                        and (
                            args.max_premotion_cosmos_predictions <= 0
                            or premotion_cosmos_count < args.max_premotion_cosmos_predictions
                        )
                    ):
                        try:
                            cosmos_reports.append(
                                run_cosmos_policy(
                                    args,
                                    root=root,
                                    run_dir=run_dir,
                                    stage_name=f"pre/{premotion_cosmos_count:02d}",
                                    prefix_role="target_pre_motion",
                                    prefix_frame_index=len(wam_history),
                                    frames=frames,
                                    wam_history=wam_history,
                                )
                            )
                        except Exception as exc:
                            cosmos_failure = {
                                "stage": "premotion",
                                "stage_name": f"pre/{premotion_cosmos_count:02d}",
                                "step_index": int(step_idx),
                                "prefix_frame_index": len(wam_history),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                            summary.update(
                                {
                                    "classification": "blocked_premotion_cosmos_policy_failed_partial_evidence_written",
                                    "cosmos_policy_failure": cosmos_failure,
                                }
                            )
                            raise StopIteration
                        premotion_cosmos_count += 1
                        next_premotion_cosmos_step = step_idx + max(1, int(args.premotion_cosmos_interval))

                    target_motion_delta = np.zeros((3,), dtype=np.float64)
                    if (
                        source_protocol_is_peg_perturb(source_protocol)
                        and (not peg_perturb_triggered)
                        and step_idx >= args.target_motion_step
                    ):
                        gate_state = eval_state(base_env)
                        gate_report = source_target_motion_live_gate(args, source_protocol, gate_state)
                        if gate_report["ok"]:
                            if not np.any(peg_perturb_force_xyz):
                                summary.update(
                                    {
                                        "classification": "blocked_source_h5_peg_perturb_missing_force_vector_partial_evidence_written",
                                        "target_motion_live_gate": gate_report,
                                    }
                                )
                                raise StopIteration
                            state_before_perturb = eval_state(base_env)
                            peg_perturb_reference_xyz = np.asarray(
                                state_before_perturb["peg_xyz"], dtype=np.float64
                            ).reshape(3)
                            apply_physical_peg_perturb(base_env, peg_perturb_force_xyz)
                            zero_action = np.zeros_like(action_low, dtype=np.float32)
                            obs, reward, terminated, truncated, info = envs.step(zero_action.reshape(1, -1))
                            state = eval_state(base_env)
                            peg_delta_observed = (
                                np.asarray(state["peg_xyz"], dtype=np.float64)
                                - np.asarray(state_before_perturb["peg_xyz"], dtype=np.float64)
                            )
                            peg_perturb_observed_cumulative_delta = (
                                np.asarray(state["peg_xyz"], dtype=np.float64) - peg_perturb_reference_xyz
                            )
                            peg_perturb_triggered = True
                            peg_perturb_force_steps_remaining = max(0, int(args.source_h5_peg_force_steps) - 1)
                            summary["target_motion_live_gate"] = gate_report
                            summary["peg_physical_perturb_applied"] = True
                            summary["peg_physical_perturb_force_xyz"] = peg_perturb_force_xyz.astype(float).tolist()
                            summary["peg_physical_perturb_observed_delta_xyz"] = peg_delta_observed.astype(float).tolist()
                            summary["peg_physical_perturb_reference_xyz"] = peg_perturb_reference_xyz.astype(float).tolist()
                            summary["peg_physical_perturb_observed_cumulative_delta_xyz"] = (
                                peg_perturb_observed_cumulative_delta.astype(float).tolist()
                            )
                            summary["peg_physical_perturb_observed_force_window_delta_xyz"] = (
                                peg_perturb_observed_cumulative_delta.astype(float).tolist()
                            )
                            append_render_frames(envs.envs[0], frames, review_frames, review_render_failures)
                            wam_history.append(
                                wam_row(
                                    zero_action,
                                    state,
                                    initial_hole,
                                    len(wam_history),
                                    300,
                                    peg_delta_applied=peg_delta_observed,
                                    perturb_triggered=True,
                                )
                            )
                            action_trace.append(
                                {
                                    "env_step": step_idx,
                                    "stage": "peg_perturb_trigger_zero_robot_action",
                                    "action_source": "physical_peg_apply_force_zero_robot_action",
                                    "action": zero_action.astype(float).tolist(),
                                    "peg_perturb_force_xyz": peg_perturb_force_xyz.astype(float).tolist(),
                                    "peg_perturb_observed_delta_xyz": peg_delta_observed.astype(float).tolist(),
                                    "peg_perturb_observed_cumulative_delta_xyz": (
                                        peg_perturb_observed_cumulative_delta.astype(float).tolist()
                                    ),
                                    "hole_delta_from_start": float(np.linalg.norm(hole_xyz(base_env) - initial_hole)),
                                    "target_motion_applied": False,
                                    "target_motion_delta_xyz": target_motion_delta.astype(float).tolist(),
                                    "target_motion_cumulative_xyz": target_motion_applied_xyz.astype(float).tolist(),
                                    "peg_perturb_triggered": True,
                                    "live_eval": state,
                                    "reward": jsonable(reward),
                                    "terminated": jsonable(terminated),
                                    "truncated": jsonable(truncated),
                                }
                            )
                            while peg_perturb_force_steps_remaining > 0:
                                apply_physical_peg_perturb(base_env, peg_perturb_force_xyz)
                                obs, reward, terminated, truncated, info = envs.step(zero_action.reshape(1, -1))
                                state = eval_state(base_env)
                                peg_perturb_force_steps_remaining -= 1
                                peg_perturb_observed_cumulative_delta = (
                                    np.asarray(state["peg_xyz"], dtype=np.float64).reshape(3)
                                    - peg_perturb_reference_xyz
                                )
                                summary["peg_physical_perturb_observed_cumulative_delta_xyz"] = (
                                    peg_perturb_observed_cumulative_delta.astype(float).tolist()
                                )
                                summary["peg_physical_perturb_observed_force_window_delta_xyz"] = (
                                    peg_perturb_observed_cumulative_delta.astype(float).tolist()
                                )
                                append_render_frames(envs.envs[0], frames, review_frames, review_render_failures)
                                wam_history.append(
                                    wam_row(
                                        zero_action,
                                        state,
                                        initial_hole,
                                        len(wam_history),
                                        300,
                                        peg_delta_applied=peg_perturb_observed_cumulative_delta,
                                        perturb_triggered=True,
                                    )
                                )
                                action_trace.append(
                                    {
                                        "env_step": step_idx,
                                        "stage": "peg_perturb_force_window_zero_robot_action",
                                        "action_source": "physical_peg_apply_force_zero_robot_action",
                                        "action": zero_action.astype(float).tolist(),
                                        "peg_perturb_force_xyz": peg_perturb_force_xyz.astype(float).tolist(),
                                        "peg_perturb_observed_cumulative_delta_xyz": (
                                            peg_perturb_observed_cumulative_delta.astype(float).tolist()
                                        ),
                                        "peg_perturb_force_steps_remaining": int(peg_perturb_force_steps_remaining),
                                        "hole_delta_from_start": float(
                                            np.linalg.norm(hole_xyz(base_env) - initial_hole)
                                        ),
                                        "target_motion_applied": False,
                                        "target_motion_delta_xyz": target_motion_delta.astype(float).tolist(),
                                        "target_motion_cumulative_xyz": target_motion_applied_xyz.astype(float).tolist(),
                                        "peg_perturb_triggered": True,
                                        "live_eval": state,
                                        "reward": jsonable(reward),
                                        "terminated": jsonable(terminated),
                                        "truncated": jsonable(truncated),
                                    }
                                )
                            trigger_frame = step_idx
                            raise StopIteration
                        else:
                            if len(source_motion_gate_rejections) < 32:
                                source_motion_gate_rejections.append(
                                    {
                                        "env_step": step_idx,
                                        "gate": gate_report,
                                    }
                                )
                            summary["source_motion_gate_rejections"] = source_motion_gate_rejections
                    elif (not target_motion_applied) and step_idx >= args.target_motion_step:
                        gate_state = eval_state(base_env)
                        gate_report = source_target_motion_live_gate(args, source_protocol, gate_state)
                        if gate_report["ok"]:
                            target_motion_applied = True
                            summary["target_motion_live_gate"] = gate_report
                        else:
                            if len(source_motion_gate_rejections) < 32:
                                source_motion_gate_rejections.append(
                                    {
                                        "env_step": step_idx,
                                        "gate": gate_report,
                                    }
                                )
                            summary["source_motion_gate_rejections"] = source_motion_gate_rejections
                    if target_motion_applied:
                        target_motion_delta = next_target_motion_delta(args, target_motion_applied_xyz)
                    if np.any(target_motion_delta):
                        shift_box(base_env, target_motion_delta)
                        target_motion_applied_xyz += target_motion_delta
                        target_motion_applied = True
                        summary["target_motion_state_intervention_used"] = True
                        state = eval_state(base_env)
                        append_render_frames(envs.envs[0], frames, review_frames, review_render_failures)
                        zero_action = np.zeros_like(action_low, dtype=np.float32)
                        wam_history.append(wam_row(zero_action, state, initial_hole, len(wam_history), 300))
                        action_trace.append(
                            {
                                "env_step": step_idx,
                                "stage": "target_motion_trigger_no_robot_action",
                                "action_source": "scene_target_motion_only_no_robot_action",
                                "action": zero_action.astype(float).tolist(),
                                "hole_delta_from_start": float(np.linalg.norm(hole_xyz(base_env) - initial_hole)),
                                "target_motion_applied": True,
                                "target_motion_delta_xyz": target_motion_delta.astype(float).tolist(),
                                "target_motion_cumulative_xyz": target_motion_applied_xyz.astype(float).tolist(),
                                "live_eval": state,
                                "reward": None,
                                "terminated": False,
                                "truncated": False,
                            }
                        )
                        trigger_frame = step_idx
                        raise StopIteration

                    action = dp_actions[:, action_idx]
                    obs, reward, terminated, truncated, info = envs.step(action)
                    state = eval_state(base_env)
                    append_render_frames(envs.envs[0], frames, review_frames, review_render_failures)
                    wam_history.append(wam_row(action.reshape(-1), state, initial_hole, len(wam_history), 300))
                    hole_delta = float(np.linalg.norm(hole_xyz(base_env) - initial_hole))
                    action_trace.append(
                        {
                            "env_step": step_idx,
                            "stage": "dp_static_prefix",
                            "action_source": "diffusion_policy",
                            "action": action.reshape(-1).astype(float).tolist(),
                            "hole_delta_from_start": hole_delta,
                            "target_motion_applied": bool(target_motion_applied),
                            "target_motion_delta_xyz": target_motion_delta.astype(float).tolist(),
                            "target_motion_cumulative_xyz": target_motion_applied_xyz.astype(float).tolist(),
                            "live_eval": state,
                            "reward": jsonable(reward),
                            "terminated": jsonable(terminated),
                            "truncated": jsonable(truncated),
                        }
                    )
    except StopIteration:
        pass
    except Exception as exc:
        runtime_failure = {
            "stage": "dp_static_prefix_or_render",
            "action_trace_steps": len(action_trace),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        summary.update(
            {
                "classification": "blocked_runtime_exception_partial_evidence_written",
                "runtime_failure": runtime_failure,
                "action_trace_steps": len(action_trace),
                "cosmos_policy_reports": cosmos_reports,
                "cosmos_dynamic_actions_executed": any(
                    row.get("stage") == "cosmos_dynamic_control" for row in action_trace
                ),
                "dp_actions_used_during_dynamic_stage": False,
            }
        )

    if runtime_failure is not None:
        pass
    elif cosmos_failure is not None:
        summary.update(
            {
                "action_trace_steps": len(action_trace),
                "cosmos_policy_reports": cosmos_reports,
                "cosmos_policy_failure": cosmos_failure,
                "cosmos_dynamic_actions_executed": any(
                    row.get("stage") == "cosmos_dynamic_control" for row in action_trace
                ),
                "dp_actions_used_during_dynamic_stage": False,
            }
        )
    elif trigger_frame is None:
        no_trigger_classification = (
            "protocol_mismatch_source_h5_motion_gate_never_reached"
            if source_protocol and source_motion_gate_rejections
            else "failed_no_causal_target_motion_trigger"
        )
        summary.update(
            {
                "classification": no_trigger_classification,
                "action_trace_steps": len(action_trace),
                "cosmos_policy_reports": cosmos_reports,
                "cosmos_dynamic_actions_executed": False,
                "dp_actions_used_during_dynamic_stage": False,
                "peg_perturb_triggered": bool(peg_perturb_triggered),
                "source_motion_gate_rejections": source_motion_gate_rejections,
            }
        )
    else:
        obs_current = obs
        cosmos_dynamic_action_count = 0
        for round_idx in range(args.max_cosmos_rounds):
            state_before = eval_state(base_env)
            if (
                state_before["peg_head_l2"] <= args.near_target_l2
                and cosmos_dynamic_action_count >= args.min_cosmos_dynamic_actions_before_finisher
                and target_motion_allows_finisher(args, target_motion_applied_xyz)
            ):
                break
            stage_name = f"post/{round_idx:02d}"
            try:
                report = run_cosmos_policy(
                    args,
                    root=root,
                    run_dir=run_dir,
                    stage_name=stage_name,
                    prefix_role="target_motion_observed",
                    prefix_frame_index=len(wam_history),
                    frames=frames,
                    wam_history=wam_history,
                )
            except Exception as exc:
                cosmos_failure = {
                    "stage": "postmotion",
                    "stage_name": stage_name,
                    "round_index": int(round_idx),
                    "prefix_frame_index": len(wam_history),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                summary.update(
                    {
                        "classification": "blocked_postmotion_cosmos_policy_failed_partial_evidence_written",
                        "target_motion_trigger_frame": trigger_frame,
                        "source_motion_gate_rejections": source_motion_gate_rejections,
                        "cosmos_policy_reports": cosmos_reports,
                        "cosmos_policy_failure": cosmos_failure,
                        "cosmos_dynamic_actions_executed": any(
                            row.get("stage") == "cosmos_dynamic_control" for row in action_trace
                        ),
                        "dp_actions_used_during_dynamic_stage": False,
                    }
                )
                break
            cosmos_reports.append(report)
            chunk = np.asarray(report["denormalized_robot_action_chunk"], dtype=np.float32)
            chunk_start = int(report.get("chunk_start", 0))
            chunk_end_exclusive = int(report.get("chunk_end_exclusive", chunk_start + len(chunk)))
            action_row_offset = int(report.get("action_row_offset", args.cosmos_action_row_offset))
            raw_chunk_start = int(report.get("raw_chunk_start", chunk_start))
            cosmos_action_adapter = np.asarray(
                [
                    float(args.cosmos_action_scale_x),
                    float(args.cosmos_action_scale_y),
                    float(args.cosmos_action_scale_z),
                    float(args.cosmos_action_scale_rot),
                    float(args.cosmos_action_scale_rot),
                    float(args.cosmos_action_scale_rot),
                    float(args.cosmos_action_scale_gripper),
                ],
                dtype=np.float32,
            )
            for local_idx, action_vec in enumerate(chunk):
                target_motion_delta = next_target_motion_delta(args, target_motion_applied_xyz)
                if np.any(target_motion_delta):
                    shift_box(base_env, target_motion_delta)
                    target_motion_applied_xyz += target_motion_delta
                    summary["target_motion_state_intervention_used"] = True
                peg_force_this_step = np.zeros((3,), dtype=np.float64)
                if peg_perturb_force_steps_remaining > 0:
                    peg_force_this_step = peg_perturb_force_xyz
                    apply_physical_peg_perturb(base_env, peg_force_this_step)
                    peg_perturb_force_steps_remaining -= 1
                adapted_action_vec = action_vec.astype(np.float32).copy()
                adapted_action_vec[:7] = adapted_action_vec[:7] * cosmos_action_adapter[: adapted_action_vec[:7].size]
                guarded_action_vec, guard_report = apply_cosmos_action_direction_guard(adapted_action_vec, args)
                dynamic_action_source = "cosmos3_policy_output"
                source_h5_teacher_dynamic_action_index: int | None = None
                final_dynamic_action_vec = guarded_action_vec
                if source_h5_teacher_dynamic_enabled:
                    dynamic_action_source = "source_h5_teacher_dynamic_future_label_diagnostic"
                    source_h5_teacher_dynamic_action_index = (
                        len(action_trace) + int(args.source_h5_teacher_dynamic_action_start_offset)
                    )
                    if (
                        source_h5_teacher_actions is None
                        or source_h5_teacher_dynamic_action_index < 0
                        or source_h5_teacher_dynamic_action_index >= source_h5_teacher_actions.shape[0]
                    ):
                        source_h5_teacher_dynamic_exhausted = {
                            "stage": "source_h5_teacher_dynamic",
                            "action_trace_steps": len(action_trace),
                            "source_h5_teacher_action_index": source_h5_teacher_dynamic_action_index,
                            "source_h5_teacher_action_rows": int(source_h5_teacher_actions.shape[0])
                            if source_h5_teacher_actions is not None
                            else None,
                            "reason": "source H5 teacher dynamic actions ran out of rows before the dynamic stage completed",
                        }
                        break
                    teacher_dynamic_action = np.zeros((action_low.size,), dtype=np.float32)
                    source_action = source_h5_teacher_actions[source_h5_teacher_dynamic_action_index]
                    copy_dim = min(teacher_dynamic_action.size, source_action.size)
                    teacher_dynamic_action[:copy_dim] = source_action[:copy_dim]
                    final_dynamic_action_vec = teacher_dynamic_action
                exec_action, clipped = clip_to_action_space(final_dynamic_action_vec, action_low, action_high)
                obs_current, reward, terminated, truncated, info = envs.step(exec_action.reshape(1, -1))
                state = eval_state(base_env)
                if peg_perturb_reference_xyz is not None:
                    peg_perturb_observed_cumulative_delta = (
                        np.asarray(state["peg_xyz"], dtype=np.float64).reshape(3) - peg_perturb_reference_xyz
                    )
                    summary["peg_physical_perturb_observed_current_cumulative_delta_xyz"] = (
                        peg_perturb_observed_cumulative_delta.astype(float).tolist()
                    )
                    if np.any(peg_force_this_step):
                        summary["peg_physical_perturb_observed_force_window_delta_xyz"] = (
                            peg_perturb_observed_cumulative_delta.astype(float).tolist()
                        )
                append_render_frames(envs.envs[0], frames, review_frames, review_render_failures)
                wam_history.append(wam_row(exec_action, state, initial_hole, len(wam_history), 300))
                action_trace.append(
                    {
                        "env_step": len(action_trace),
                        "stage": "cosmos_dynamic_control",
                        "action_source": dynamic_action_source,
                        "cosmos_round": round_idx,
                        "cosmos_action_index": local_idx,
                        "cosmos_action_chunk_start": chunk_start,
                        "cosmos_action_chunk_end_exclusive": chunk_end_exclusive,
                        "cosmos_action_row_offset": action_row_offset,
                        "cosmos_action_row_offset_source": str(args.cosmos_action_row_offset_source),
                        "cosmos_raw_chunk_start": raw_chunk_start,
                        "cosmos_predicted_action_row_index": int(chunk_start + local_idx),
                        "raw_cosmos_action": action_vec.astype(float).tolist(),
                        "adapted_cosmos_action": adapted_action_vec.astype(float).tolist(),
                        "guarded_cosmos_action": guarded_action_vec.astype(float).tolist(),
                        "source_h5_teacher_dynamic_action_index": source_h5_teacher_dynamic_action_index,
                        "future_label_teacher_dynamic_diagnostic": bool(source_h5_teacher_dynamic_enabled),
                        "cosmos_action_adapter_scale": cosmos_action_adapter.astype(float).tolist(),
                        "cosmos_action_direction_guard": guard_report,
                        "action": exec_action.astype(float).tolist(),
                        "action_clipped_to_space": clipped,
                        "peg_perturb_force_xyz": peg_force_this_step.astype(float).tolist(),
                        "peg_perturb_observed_cumulative_delta_xyz": (
                            peg_perturb_observed_cumulative_delta.astype(float).tolist()
                            if peg_perturb_reference_xyz is not None
                            else [0.0, 0.0, 0.0]
                        ),
                        "peg_perturb_triggered": bool(peg_perturb_triggered),
                        "target_motion_delta_xyz": target_motion_delta.astype(float).tolist(),
                        "target_motion_cumulative_xyz": target_motion_applied_xyz.astype(float).tolist(),
                        "live_eval": state,
                        "reward": jsonable(reward),
                        "terminated": jsonable(terminated),
                        "truncated": jsonable(truncated),
                    }
                )
                cosmos_dynamic_action_count += 1
                if bool(state["success"]):
                    final_success = True
                    if (
                        args.require_target_motion_complete_before_finisher
                        and not target_motion_complete(args, target_motion_applied_xyz)
                    ):
                        target_assisted_metric_true_before_motion_complete = {
                            "env_step": len(action_trace) - 1,
                            "stage": "cosmos_dynamic_control",
                            "cosmos_round": int(round_idx),
                            "cosmos_action_index": int(local_idx),
                            "target_motion_cumulative_xyz": target_motion_applied_xyz.astype(float).tolist(),
                            "peg_head_l2": float(state["peg_head_l2"]),
                            "reason": (
                                "simulator success became true before source-H5 target motion completed; "
                                "treat as target-assisted metric-true counterexample, not physical success"
                            ),
                        }
                    break
                done = bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0])
                if (
                    final_success
                    or done
                    or source_h5_teacher_dynamic_exhausted is not None
                    or (
                        state["peg_head_l2"] <= args.near_target_l2
                        and cosmos_dynamic_action_count >= args.min_cosmos_dynamic_actions_before_finisher
                        and target_motion_allows_finisher(args, target_motion_applied_xyz)
                    )
                ):
                    break
            if (
                final_success
                or (
                    eval_state(base_env)["peg_head_l2"] <= args.near_target_l2
                    and cosmos_dynamic_action_count >= args.min_cosmos_dynamic_actions_before_finisher
                    and target_motion_allows_finisher(args, target_motion_applied_xyz)
                )
                or cosmos_failure is not None
                or source_h5_teacher_dynamic_exhausted is not None
            ):
                break

        if cosmos_failure is None:
            finisher_start = len(action_trace)
            enough_cosmos_dynamic_actions = (
                cosmos_dynamic_action_count >= args.min_cosmos_dynamic_actions_before_finisher
            )
            near_before_finisher = enough_cosmos_dynamic_actions and eval_state(base_env)["peg_head_l2"] <= args.near_target_l2
            target_motion_complete_before_finisher = target_motion_complete(args, target_motion_applied_xyz)
            finisher_allowed_by_target_motion = target_motion_allows_finisher(args, target_motion_applied_xyz)
            near_before_finisher = near_before_finisher and finisher_allowed_by_target_motion
            best_finisher_eval: dict[str, Any] | None = None
            best_finisher_step: int | None = None
            if near_before_finisher and not final_success:
                for _ in range(args.max_finisher_steps):
                    stop = False
                    source_h5_teacher_action_index: int | None = None
                    finisher_stage = (
                        "oracle_physical_manual_finisher"
                        if args.finisher_controller in manual_finisher_controllers
                        else "oracle_physical_dp_finisher"
                    )
                    finisher_action_source = (
                        f"{args.finisher_controller}_after_near_target_gate"
                        if args.finisher_controller in manual_finisher_controllers
                        else "diffusion_policy_finisher_after_near_target_gate"
                    )
                    if args.finisher_controller == "source_h5_teacher_suffix":
                        finisher_stage = "oracle_physical_manual_finisher"
                        finisher_action_source = "source_h5_teacher_suffix_future_label_after_near_target_gate"
                        source_h5_teacher_action_index = (
                            len(action_trace) + int(args.source_h5_teacher_action_start_offset)
                        )
                        if (
                            source_h5_teacher_actions is None
                            or source_h5_teacher_action_index < 0
                            or source_h5_teacher_action_index >= source_h5_teacher_actions.shape[0]
                        ):
                            source_h5_teacher_suffix_exhausted = {
                                "stage": "source_h5_teacher_suffix_finisher",
                                "action_trace_steps": len(action_trace),
                                "source_h5_teacher_action_index": source_h5_teacher_action_index,
                                "source_h5_teacher_action_rows": int(source_h5_teacher_actions.shape[0])
                                if source_h5_teacher_actions is not None
                                else None,
                                "reason": "source H5 teacher action suffix ran out of rows before simulator success",
                            }
                            stop = True
                            break
                        teacher_action = np.zeros((action_low.size,), dtype=np.float32)
                        source_action = source_h5_teacher_actions[source_h5_teacher_action_index]
                        copy_dim = min(teacher_action.size, source_action.size)
                        teacher_action[:copy_dim] = source_action[:copy_dim]
                        dp_actions = teacher_action.reshape(1, 1, -1)
                    elif args.finisher_controller == "manual_oracle_servo":
                        state_before_action = eval_state(base_env)
                        dp_actions = manual_oracle_servo_action(
                            args,
                            state_before_action,
                            action_low.size,
                        ).reshape(1, 1, -1)
                    elif args.finisher_controller == "manual_hole_frame_servo":
                        state_before_action = eval_state(base_env)
                        dp_actions = manual_hole_frame_servo_action(
                            args,
                            state_before_action,
                            action_low.size,
                        ).reshape(1, 1, -1)
                    elif args.finisher_controller == "manual_staged_hole_servo":
                        state_before_action = eval_state(base_env)
                        dp_actions = manual_staged_hole_servo_action(
                            args,
                            state_before_action,
                            action_low.size,
                        ).reshape(1, 1, -1)
                    elif args.finisher_controller == "manual_staged_twist_insert":
                        state_before_action = eval_state(base_env)
                        dp_actions = manual_staged_twist_insert_action(
                            args,
                            state_before_action,
                            action_low.size,
                        ).reshape(1, 1, -1)
                    elif args.finisher_controller == "manual_staged_pose_servo":
                        state_before_action = eval_state(base_env)
                        dp_actions = manual_staged_pose_servo_action(
                            args,
                            state_before_action,
                            action_low.size,
                        ).reshape(1, 1, -1)
                    elif args.finisher_controller == "manual_regrasp_then_hole_servo":
                        state_before_action = eval_state(base_env)
                        dp_actions = manual_regrasp_then_hole_servo_action(
                            args,
                            state_before_action,
                            action_low.size,
                        ).reshape(1, 1, -1)
                    elif args.finisher_controller == "manual_align_then_dp":
                        state_before_action = eval_state(base_env)
                        rel = np.asarray(state_before_action["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
                        yz_error = float(np.linalg.norm(rel[1:3]))
                        if yz_error <= float(args.manual_align_threshold) and rel[0] < 0.0:
                            obs_tensor = common.to_tensor(obs_current, device)
                            dp_actions = agent.get_action(obs_tensor).detach().cpu().numpy()
                            finisher_stage = "oracle_physical_dp_finisher"
                            finisher_action_source = "diffusion_policy_after_manual_align_gate"
                        else:
                            dp_actions = manual_staged_hole_servo_action(
                                args,
                                state_before_action,
                                action_low.size,
                            ).reshape(1, 1, -1)
                            finisher_stage = "oracle_physical_manual_finisher"
                            finisher_action_source = "manual_staged_align_before_dp_gate"
                    elif args.finisher_controller == "dp_then_manual_close":
                        state_before_action = eval_state(base_env)
                        rel = np.asarray(state_before_action["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
                        yz_error = float(np.linalg.norm(rel[1:3]))
                        if (
                            float(state_before_action["peg_head_l2"]) <= float(args.manual_dp_to_manual_l2)
                            and yz_error <= float(args.manual_soft_insert_threshold)
                            and rel[0] < 0.0
                        ):
                            dp_actions = manual_staged_hole_servo_action(
                                args,
                                state_before_action,
                                action_low.size,
                            ).reshape(1, 1, -1)
                            finisher_stage = "oracle_physical_manual_finisher"
                            finisher_action_source = "manual_close_after_dp_gate"
                        else:
                            obs_tensor = common.to_tensor(obs_current, device)
                            dp_actions = agent.get_action(obs_tensor).detach().cpu().numpy()
                            finisher_stage = "oracle_physical_dp_finisher"
                            finisher_action_source = "diffusion_policy_before_manual_close_gate"
                    elif args.finisher_controller == "manual_staged_dp_rot":
                        state_before_action = eval_state(base_env)
                        manual_action = manual_staged_hole_servo_action(
                            args,
                            state_before_action,
                            action_low.size,
                        )
                        obs_tensor = common.to_tensor(obs_current, device)
                        dp_action = agent.get_action(obs_tensor).detach().cpu().numpy()[0, 0]
                        hybrid_action = manual_action.copy()
                        if action_low.size >= 4:
                            hybrid_action[3:] = dp_action.reshape(-1)[3:action_low.size]
                        dp_actions = hybrid_action.reshape(1, 1, -1)
                    elif args.finisher_controller == "diffusion_policy":
                        obs_tensor = common.to_tensor(obs_current, device)
                        dp_actions = agent.get_action(obs_tensor).detach().cpu().numpy()
                    else:
                        raise ValueError(f"unsupported_finisher_controller={args.finisher_controller}")
                    for action_vec in dp_actions[0]:
                        target_motion_delta = (
                            next_target_motion_delta(args, target_motion_applied_xyz)
                            if args.target_motion_during_finisher
                            else np.zeros((3,), dtype=np.float64)
                        )
                        if np.any(target_motion_delta):
                            shift_box(base_env, target_motion_delta)
                            target_motion_applied_xyz += target_motion_delta
                            summary["target_motion_state_intervention_used"] = True
                        peg_force_this_step = np.zeros((3,), dtype=np.float64)
                        if peg_perturb_force_steps_remaining > 0:
                            peg_force_this_step = peg_perturb_force_xyz
                            apply_physical_peg_perturb(base_env, peg_force_this_step)
                            peg_perturb_force_steps_remaining -= 1
                        exec_action, clipped = clip_to_action_space(action_vec, action_low, action_high)
                        obs_current, reward, terminated, truncated, info = envs.step(exec_action.reshape(1, -1))
                        state = eval_state(base_env)
                        if (
                            best_finisher_eval is None
                            or float(state["peg_head_l2"]) < float(best_finisher_eval["peg_head_l2"])
                        ):
                            best_finisher_eval = state
                            best_finisher_step = len(action_trace)
                        if peg_perturb_reference_xyz is not None:
                            peg_perturb_observed_cumulative_delta = (
                                np.asarray(state["peg_xyz"], dtype=np.float64).reshape(3) - peg_perturb_reference_xyz
                            )
                            summary["peg_physical_perturb_observed_current_cumulative_delta_xyz"] = (
                                peg_perturb_observed_cumulative_delta.astype(float).tolist()
                            )
                            if np.any(peg_force_this_step):
                                summary["peg_physical_perturb_observed_force_window_delta_xyz"] = (
                                    peg_perturb_observed_cumulative_delta.astype(float).tolist()
                                )
                        append_render_frames(envs.envs[0], frames, review_frames, review_render_failures)
                        action_trace.append(
                            {
                                "env_step": len(action_trace),
                                "stage": finisher_stage,
                                "action_source": finisher_action_source,
                                "action": exec_action.astype(float).tolist(),
                                "action_clipped_to_space": clipped,
                                "finisher_controller": args.finisher_controller,
                                "source_h5_teacher_action_index": source_h5_teacher_action_index,
                                "future_label_teacher_suffix_diagnostic": bool(source_h5_teacher_suffix_enabled),
                                "peg_perturb_force_xyz": peg_force_this_step.astype(float).tolist(),
                                "peg_perturb_observed_cumulative_delta_xyz": (
                                    peg_perturb_observed_cumulative_delta.astype(float).tolist()
                                    if peg_perturb_reference_xyz is not None
                                    else [0.0, 0.0, 0.0]
                                ),
                                "peg_perturb_triggered": bool(peg_perturb_triggered),
                                "target_motion_delta_xyz": target_motion_delta.astype(float).tolist(),
                                "target_motion_cumulative_xyz": target_motion_applied_xyz.astype(float).tolist(),
                                "live_eval": state,
                                "reward": jsonable(reward),
                                "terminated": jsonable(terminated),
                                "truncated": jsonable(truncated),
                            }
                        )
                        if bool(state["success"]):
                            final_success = True
                            stop = True
                            break
                        if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
                            stop = True
                            break
                    if stop:
                        break
            final_state = eval_state(base_env)
            if runtime_failure is not None:
                classification = "blocked_runtime_exception_partial_evidence_written"
            elif not enough_cosmos_dynamic_actions:
                classification = "blocked_insufficient_cosmos_dynamic_actions_no_finisher"
            elif target_assisted_metric_true_before_motion_complete is not None:
                classification = "invalid_target_assisted_metric_true_before_target_motion_complete"
            elif final_success and source_h5_teacher_dynamic_enabled:
                classification = "diagnostic_future_label_teacher_dynamic_metric_true_not_success"
            elif source_h5_teacher_dynamic_exhausted is not None:
                classification = "diagnostic_future_label_teacher_dynamic_exhausted_without_success"
            elif final_success and source_h5_teacher_suffix_enabled:
                classification = "diagnostic_future_label_teacher_suffix_metric_true_not_success"
            elif source_h5_teacher_suffix_exhausted is not None:
                classification = "diagnostic_future_label_teacher_suffix_exhausted_without_success"
            elif final_success and action_row_offset_diagnostic:
                classification = "diagnostic_action_row_offset_metric_true_not_success"
            elif action_row_offset_diagnostic and cosmos_reports:
                classification = "diagnostic_action_row_offset_physical_failure_not_inserted_full_pipeline_attempted"
            elif final_success:
                classification = "simulator_success_metric_true_visual_review_required_not_physical_success"
            elif cosmos_reports:
                classification = "physical_failure_not_inserted_full_pipeline_attempted"
            else:
                classification = "blocked_before_cosmos_action_execution"
            summary.update(
                {
                    "classification": classification,
                    "target_motion_trigger_frame": trigger_frame,
                    "source_motion_gate_rejections": source_motion_gate_rejections,
                    "cosmos_policy_reports": cosmos_reports,
                    "cosmos_dynamic_actions_executed": any(row.get("stage") == "cosmos_dynamic_control" for row in action_trace),
                    "cosmos_dynamic_action_count": cosmos_dynamic_action_count,
                    "max_premotion_cosmos_predictions": int(args.max_premotion_cosmos_predictions),
                    "required_premotion_cosmos_predictions": max(
                        2, int(args.max_premotion_cosmos_predictions)
                    ),
                    "peg_perturb_triggered": bool(peg_perturb_triggered),
                    "peg_physical_perturb_applied": bool(summary.get("peg_physical_perturb_applied", False)),
                    "min_cosmos_dynamic_actions_before_finisher": int(args.min_cosmos_dynamic_actions_before_finisher),
                    "dp_actions_used_during_dynamic_stage": False,
                    "near_target_before_finisher": bool(near_before_finisher),
                    "target_motion_complete_before_finisher": bool(target_motion_complete_before_finisher),
                    "finisher_allowed_by_target_motion": bool(finisher_allowed_by_target_motion),
                    "finisher_start_step": finisher_start if near_before_finisher else None,
                    "future_label_teacher_suffix_diagnostic": bool(source_h5_teacher_suffix_enabled),
                    "future_label_teacher_dynamic_diagnostic": bool(source_h5_teacher_dynamic_enabled),
                    "target_assisted_metric_true_before_motion_complete": (
                        target_assisted_metric_true_before_motion_complete
                    ),
                    "dynamic_controller": manifest["dynamic_controller"],
                    "source_h5_teacher_suffix": manifest["source_h5_teacher_suffix"],
                    "source_h5_teacher_suffix_exhausted": source_h5_teacher_suffix_exhausted,
                    "source_h5_teacher_dynamic_exhausted": source_h5_teacher_dynamic_exhausted,
                    "runtime_failure": runtime_failure,
                    "finisher_parameters": {
                        "manual_dp_to_manual_l2": float(args.manual_dp_to_manual_l2),
                        "manual_soft_insert_threshold": float(args.manual_soft_insert_threshold),
                        "manual_align_threshold": float(args.manual_align_threshold),
                        "manual_insert_speed": float(args.manual_insert_speed),
                        "manual_forward_gain": float(args.manual_forward_gain),
                        "manual_forward_limit": float(args.manual_forward_limit),
                        "manual_yaw_stop_l2": float(args.manual_yaw_stop_l2),
                    },
                    "final_success": bool(final_success),
                    "simulator_success_metric": bool(final_success),
                    "physical_insertion_success_claimed": False,
                    "visual_full_insertion_confirmed": False,
                    "initial_eval": initial_eval,
                    "final_eval": final_state,
                    "best_finisher_step": best_finisher_step,
                    "best_finisher_eval": best_finisher_eval,
                }
            )

    video_path = run_dir / "videos" / "raw.mp4"
    annotated_video_path = run_dir / "videos" / "annotated.mp4"
    review_videos: dict[str, dict[str, str | None]] = {}
    if frames:
        write_mp4(video_path, frames, args.fps)
        write_mp4(annotated_video_path, annotated_frames(frames, action_trace), args.fps)
    for camera_uid, camera_frames in review_frames.items():
        camera_dir = run_dir / "videos" / camera_uid
        raw_path = camera_dir / "raw.mp4"
        annotated_path = camera_dir / "annotated.mp4"
        if camera_frames:
            write_mp4(raw_path, camera_frames, args.fps)
            write_mp4(annotated_path, annotated_frames(camera_frames, action_trace), args.fps)
        review_videos[camera_uid] = {
            "video": str(raw_path) if raw_path.exists() else None,
            "annotated_video": str(annotated_path) if annotated_path.exists() else None,
            "num_frames": len(camera_frames),
        }
    write_json(run_dir / "action_trace.json", action_trace)
    audit = discontinuity_audit(action_trace)
    summary["discontinuity_audit"] = audit
    if audit["snap_detected"]:
        summary["classification"] = "invalid_discontinuous_peg_motion_review_required"
        summary["physical_insertion_success_claimed"] = False
        summary["visual_full_insertion_confirmed"] = False
    summary["video"] = str(video_path) if video_path.exists() else None
    summary["annotated_video"] = str(annotated_video_path) if annotated_video_path.exists() else None
    summary["review_videos"] = review_videos
    summary["review_render_failures"] = review_render_failures
    summary["num_video_frames"] = len(frames)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "classification.txt").write_text(
        "\n".join(
            [
                f"phase03_status={summary['classification']}",
                "method_evidence_allowed=false",
                "physical_insertion_success=false",
                f"simulator_success_metric={str(bool(summary.get('simulator_success_metric'))).lower()}",
                "visual_full_insertion_confirmed=false",
                f"cosmos_dynamic_actions_executed={str(bool(summary.get('cosmos_dynamic_actions_executed'))).lower()}",
                "oracle_set_pose_used=false",
            ]
        )
        + "\n"
    )
    envs.close()
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    classification = str(summary.get("classification") or "")
    if classification == "blocked_missing_cosmos_normalization_stats_no_action_execution":
        return 44
    if classification.startswith("blocked_"):
        return 45
    if classification.startswith("invalid_"):
        return 46
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
