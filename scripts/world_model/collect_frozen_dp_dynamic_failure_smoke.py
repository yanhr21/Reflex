#!/usr/bin/env python3
"""Collect C-class frozen-DP dynamic failure smoke data inside Slurm.

The policy is the official state Diffusion Policy checkpoint loaded through the
existing ManiSkill DP training code. The target motion is commanded through
`active_dynamic_peg_adapter.py`; failed chunks are negative/diagnostic data,
not positive BC.

The historical class name contains "failure", but success is allowed and is
recorded as an outcome label. A C rollout is invalid only for forbidden
mechanisms such as state intervention, snap / teleport, or target-assisted
self-insertion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import mani_skill.envs  # noqa: F401
import numpy as np
import sapien
import torch
from gymnasium.vector import SyncVectorEnv
from mani_skill.utils import common
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack

import train as ms_train
from scripts.world_model.active_dynamic_peg_adapter import (
    DynamicMotionSpec,
    build_motion_command,
    command_target_from_motion,
    manifest_fields,
    validate_trace_rows,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist() if value.ndim else value.item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _tensor_to_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _pose_arrays(actor: Any) -> tuple[np.ndarray, np.ndarray]:
    pose = actor.pose
    p = _tensor_to_np(pose.p).reshape(-1, 3)[0].astype(np.float64)
    q = _tensor_to_np(pose.q).reshape(-1, 4)[0].astype(np.float64)
    return p, q


def _frame_from_render(rendered: Any) -> np.ndarray:
    if isinstance(rendered, dict):
        for key in ("rgb", "Color", "color", "image"):
            if key in rendered:
                rendered = rendered[key]
                break
    frame = np.asarray(rendered)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame


def _scalar_bool(info: dict[str, Any], key: str) -> bool:
    if key not in info:
        return False
    return bool(np.asarray(info[key]).reshape(-1)[0])


def _task_state(base_env: Any) -> dict[str, Any]:
    tcp_p, _ = _pose_arrays(base_env.agent.tcp)
    peg_p, _ = _pose_arrays(base_env.peg)
    peg_head_at_hole = _tensor_to_np((base_env.box_hole_pose.inv() * base_env.peg_head_pose).p).reshape(-1, 3)[
        0
    ].astype(np.float64)
    grasped = bool(_tensor_to_np(base_env.agent.is_grasping(base_env.peg, max_angle=20)).reshape(-1)[0])
    inserted = bool(_tensor_to_np(base_env.has_peg_inserted()[0]).reshape(-1)[0])
    return {
        "tcp_p": tcp_p,
        "peg_p": peg_p,
        "peg_head_at_hole": peg_head_at_hole,
        "peg_head_l2": float(np.linalg.norm(peg_head_at_hole)),
        "grasped": grasped,
        "inserted": inserted,
    }


def _quality_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    if not rows:
        failures.append("no_task_state_rows")
        return {
            "quality_gate_passed": False,
            "quality_gate_failures": failures,
            "min_tcp_motion_m": args.min_tcp_motion_m,
            "min_peg_motion_m": args.min_peg_motion_m,
            "require_grasp": args.require_grasp,
        }
    tcp = np.asarray([row["tcp_p"] for row in rows], dtype=np.float64)
    peg = np.asarray([row["peg_p"] for row in rows], dtype=np.float64)
    peg_head_l2 = np.asarray([row["peg_head_l2"] for row in rows], dtype=np.float64)
    tcp_motion_m = float(np.max(np.linalg.norm(tcp - tcp[0], axis=1)))
    peg_motion_m = float(np.max(np.linalg.norm(peg - peg[0], axis=1)))
    peg_lift_m = float(np.max(peg[:, 2] - peg[0, 2]))
    grasp_once = any(bool(row["grasped"]) for row in rows)
    if tcp_motion_m < args.min_tcp_motion_m:
        failures.append("tcp_motion_too_small")
    if peg_motion_m < args.min_peg_motion_m:
        failures.append("peg_motion_too_small")
    if args.require_grasp and not grasp_once:
        failures.append("grasp_never_detected")
    return {
        "tcp_motion_m": tcp_motion_m,
        "peg_motion_m": peg_motion_m,
        "peg_lift_m": peg_lift_m,
        "min_peg_head_l2": float(np.min(peg_head_l2)),
        "final_peg_head_l2": float(peg_head_l2[-1]),
        "grasp_once": bool(grasp_once),
        "quality_gate_passed": not failures,
        "quality_gate_failures": failures,
        "min_tcp_motion_m": args.min_tcp_motion_m,
        "min_peg_motion_m": args.min_peg_motion_m,
        "require_grasp": args.require_grasp,
    }


def _add_pre_insert_timing_quality(
    quality: dict[str, Any],
    rows: list[dict[str, Any]],
    motion_trigger_step: int | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = dict(quality)
    first_inserted_step: int | None = None
    for row in rows:
        if bool(row.get("inserted")):
            first_inserted_step = int(row["env_step"])
            break
    payload["first_inserted_step"] = first_inserted_step
    payload["first_motion_step"] = motion_trigger_step
    payload["min_trigger_to_insert_steps"] = args.min_trigger_to_insert_steps
    payload["pre_insert_motion_required"] = args.require_pre_insert_motion
    if motion_trigger_step is None or not args.require_pre_insert_motion:
        payload["trigger_to_insert_steps"] = None
        return payload
    if first_inserted_step is None:
        payload["trigger_to_insert_steps"] = None
        return payload
    lead_steps = int(first_inserted_step) - int(motion_trigger_step)
    payload["trigger_to_insert_steps"] = lead_steps
    if lead_steps < args.min_trigger_to_insert_steps:
        failures = list(payload.get("quality_gate_failures", []))
        failures.append("motion_started_too_close_to_or_after_insertion")
        payload["quality_gate_failures"] = failures
        payload["quality_gate_passed"] = False
    return payload


def build_train_args(saved: dict[str, Any], args: argparse.Namespace) -> ms_train.Args:
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
    return train_args


def _str_bool(value: str) -> bool:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    raise ValueError(f"expected true/false, got {value!r}")


def _maybe_trigger_motion(
    *,
    step: int,
    task_state: dict[str, Any],
    current_trigger_step: int | None,
    args: argparse.Namespace,
) -> int | None:
    if args.motion_trigger_mode == "fixed_step":
        return args.motion_start_step
    if args.motion_trigger_mode == "inserted":
        if current_trigger_step is not None:
            return current_trigger_step
        if step < args.motion_trigger_min_step:
            return None
        if args.motion_trigger_require_grasp and not bool(task_state.get("grasped")):
            return None
        if bool(task_state.get("inserted")):
            return step
        return None
    if args.motion_trigger_mode == "pre_insert_l2":
        if current_trigger_step is not None:
            return current_trigger_step
        if step < args.motion_trigger_min_step:
            return None
        if args.motion_trigger_require_grasp and not bool(task_state.get("grasped")):
            return None
        if bool(task_state.get("inserted")):
            return None
        if float(task_state.get("peg_head_l2", float("inf"))) <= args.motion_trigger_threshold_m:
            return step
        return None
    if args.motion_trigger_mode != "peg_head_l2":
        raise RuntimeError(f"unknown motion trigger mode: {args.motion_trigger_mode}")
    if current_trigger_step is not None:
        return current_trigger_step
    if step < args.motion_trigger_min_step:
        return None
    if args.motion_trigger_require_grasp and not bool(task_state.get("grasped")):
        return None
    if float(task_state.get("peg_head_l2", float("inf"))) <= args.motion_trigger_threshold_m:
        return step
    return None


def _maybe_apply_peg_disturbance(
    *,
    base_env: Any,
    env_step: int,
    motion_trigger_step: int | None,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if args.scenario != "peg_disturb" or motion_trigger_step is None:
        return None
    local_step = env_step - motion_trigger_step
    if local_step < 0 or local_step >= args.peg_disturb_duration_steps:
        return None
    force = np.asarray(
        (args.peg_disturb_force_x, args.peg_disturb_force_y, args.peg_disturb_force_z),
        dtype=np.float32,
    )
    base_env.peg.apply_force(force)
    force_norm = float(np.linalg.norm(force.astype(np.float64)))
    return {
        "step": int(env_step),
        "scenario": args.scenario,
        "command_kind": "peg_physical_force",
        "peg_perturb_trace": True,
        "peg_perturb_force_xyz": force.astype(float).tolist(),
        "instantaneous_delta_m": force_norm,
        "state_intervention": False,
        "snap_or_teleport": False,
    }


def make_env(train_args: ms_train.Args):
    def thunk():
        env = gym.make(
            train_args.env_id,
            reconfiguration_freq=1,
            control_mode=train_args.control_mode,
            reward_mode="sparse",
            obs_mode="state",
            render_mode="rgb_array",
            human_render_camera_configs=dict(shader_pack="default"),
            max_episode_steps=train_args.max_episode_steps,
        )
        env = FrameStack(env, num_stack=train_args.obs_horizon)
        env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
        return env

    return SyncVectorEnv([thunk])


def collect(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    trace_dir = output_dir / "trace"
    review_dir = output_dir / "review" / "frames"
    video_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    train_args = build_train_args(ckpt.get("args", {}), args)
    ms_train.args = train_args
    ms_train.device = device

    envs = make_env(train_args)
    agent = ms_train.Agent(envs, train_args).to(device)
    state_key = "ema_agent" if args.use_ema else "agent"
    agent.load_state_dict(ckpt[state_key])
    agent.eval()

    spec = DynamicMotionSpec(
        scenario=args.scenario,
        start_step=args.motion_start_step,
        duration_steps=args.motion_duration_steps,
        delta_xyz=(args.delta_x, args.delta_y, args.delta_z),
        reverse_fraction=args.reverse_fraction,
        sine_cycles=args.sine_cycles,
        max_step_delta_m=args.max_step_delta_m,
    )

    action_rows: list[dict[str, Any]] = []
    motion_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    quality_by_rollout: list[dict[str, Any]] = []
    skipped_attempts: list[dict[str, Any]] = []
    video_paths: list[Path] = []
    total_frames = 0
    failure: str | None = None
    success_once = False
    final_info: dict[str, Any] = {}
    max_rollout_attempts = args.max_rollout_attempts
    if max_rollout_attempts <= 0:
        max_rollout_attempts = max(args.num_rollouts * 5, args.num_rollouts + 20)
    attempt_idx = 0
    try:
        with torch.no_grad():
            accepted_rollout_idx = 0
            while accepted_rollout_idx < args.num_rollouts and attempt_idx < max_rollout_attempts:
                rollout_idx = accepted_rollout_idx
                frames: list[np.ndarray] = []
                obs, reset_info = envs.reset(seed=args.seed + attempt_idx)
                base_env = envs.envs[0].unwrapped
                initial_p, initial_q = _pose_arrays(base_env.box)
                previous_p: np.ndarray | None = initial_p.copy()
                env_step = 0
                done = False
                rollout_task_rows: list[dict[str, Any]] = []
                rollout_action_rows: list[dict[str, Any]] = []
                rollout_motion_rows: list[dict[str, Any]] = []
                motion_trigger_step: int | None = None

                while env_step < args.max_episode_steps:
                    obs_tensor = common.to_tensor(obs, device)
                    action_seq = agent.get_action(obs_tensor).detach().cpu().numpy()
                    for action_idx in range(action_seq.shape[1]):
                        pre_task_state = _task_state(base_env)
                        motion_trigger_step = _maybe_trigger_motion(
                            step=env_step,
                            task_state=pre_task_state,
                            current_trigger_step=motion_trigger_step,
                            args=args,
                        )
                        command = None
                        if motion_trigger_step is not None and args.scenario != "peg_disturb":
                            effective_spec = DynamicMotionSpec(
                                scenario=args.scenario,
                                start_step=motion_trigger_step,
                                duration_steps=args.motion_duration_steps,
                                delta_xyz=(args.delta_x, args.delta_y, args.delta_z),
                                reverse_fraction=args.reverse_fraction,
                                sine_cycles=args.sine_cycles,
                                max_step_delta_m=args.max_step_delta_m,
                            )
                            command = build_motion_command(
                                step=env_step,
                                spec=effective_spec,
                                initial_p=initial_p,
                                initial_q=initial_q,
                                previous_p=previous_p,
                            )
                        if command is not None:
                            motion_row = command_target_from_motion(base_env.box, sapien.Pose, command)
                            motion_row["rollout"] = int(rollout_idx)
                            motion_row["attempt_idx"] = int(attempt_idx)
                            motion_row["motion_trigger_mode"] = args.motion_trigger_mode
                            motion_row["motion_trigger_step"] = int(motion_trigger_step)
                            motion_row["motion_trigger_threshold_m"] = args.motion_trigger_threshold_m
                            motion_row["pre_step_peg_head_l2"] = float(pre_task_state["peg_head_l2"])
                            motion_row["pre_step_grasped"] = bool(pre_task_state["grasped"])
                            rollout_motion_rows.append(motion_row)
                            previous_p = np.asarray(command.target_p, dtype=np.float64)
                        peg_force_row = _maybe_apply_peg_disturbance(
                            base_env=base_env,
                            env_step=env_step,
                            motion_trigger_step=motion_trigger_step,
                            args=args,
                        )
                        if peg_force_row is not None:
                            peg_force_row["rollout"] = int(rollout_idx)
                            peg_force_row["attempt_idx"] = int(attempt_idx)
                            peg_force_row["motion_trigger_mode"] = args.motion_trigger_mode
                            peg_force_row["motion_trigger_step"] = int(motion_trigger_step)
                            peg_force_row["pre_step_peg_head_l2"] = float(pre_task_state["peg_head_l2"])
                            peg_force_row["pre_step_grasped"] = bool(pre_task_state["grasped"])
                            rollout_motion_rows.append(peg_force_row)

                        action = action_seq[:, action_idx]
                        obs, reward, terminated, truncated, info = envs.step(action)
                        success = _scalar_bool(info, "success")
                        fail = _scalar_bool(info, "fail")
                        success_once = success_once or success
                        final_info = _jsonable(info)
                        frames.append(_frame_from_render(envs.envs[0].render()))
                        task_state = _task_state(base_env)
                        task_row = {
                            "rollout": int(rollout_idx),
                            "attempt_idx": int(attempt_idx),
                            "env_step": int(env_step),
                            **_jsonable(task_state),
                        }
                        rollout_task_rows.append(task_row)
                        rollout_action_rows.append(
                            {
                                "rollout": int(rollout_idx),
                                "attempt_idx": int(attempt_idx),
                                "env_step": int(env_step),
                                "action_idx": int(action_idx),
                                "action": action.reshape(-1).astype(float).tolist(),
                                "reward": _jsonable(reward),
                                "success": bool(success),
                                "fail": bool(fail),
                                "terminated": _jsonable(terminated),
                                "truncated": _jsonable(truncated),
                                "positive_policy_data_allowed": False,
                            }
                        )
                        env_step += 1
                        done = bool(np.asarray(terminated).reshape(-1)[0]) or bool(
                            np.asarray(truncated).reshape(-1)[0]
                        )
                        if done or env_step >= args.max_episode_steps:
                            break
                    if done or env_step >= args.max_episode_steps:
                        break

                quality = _add_pre_insert_timing_quality(
                    _quality_summary(rollout_task_rows, args),
                    rollout_task_rows,
                    motion_trigger_step,
                    args,
                )
                quality["rollout"] = int(rollout_idx)
                quality["attempt_idx"] = int(attempt_idx)
                quality["motion_trigger_step"] = motion_trigger_step
                quality["motion_trigger_mode"] = args.motion_trigger_mode
                if motion_trigger_step is None:
                    quality["accepted"] = False
                    quality["skip_reason"] = "motion_trigger_never_reached"
                    skipped_attempts.append(quality)
                    attempt_idx += 1
                    continue
                if not quality["quality_gate_passed"]:
                    quality["accepted"] = False
                    quality["skip_reason"] = "task_motion_quality_gate_failed"
                    skipped_attempts.append(quality)
                    attempt_idx += 1
                    continue

                quality["accepted"] = True
                quality_by_rollout.append(quality)
                task_rows.extend(rollout_task_rows)
                action_rows.extend(rollout_action_rows)
                motion_rows.extend(rollout_motion_rows)

                if frames:
                    video_path = video_dir / f"rollout_{rollout_idx:06d}.mp4"
                    imageio.mimsave(video_path, frames, fps=args.fps)
                    video_paths.append(video_path)
                    total_frames += len(frames)
                    for idx in sorted(set([0, args.grasp_review_step, len(frames) // 2, len(frames) - 1])):
                        if 0 <= idx < len(frames):
                            imageio.imwrite(review_dir / f"rollout_{rollout_idx:06d}_frame_{idx:04d}.png", frames[idx])
                accepted_rollout_idx += 1
                attempt_idx += 1
            if accepted_rollout_idx < args.num_rollouts:
                raise RuntimeError(
                    "insufficient accepted rollouts: "
                    f"accepted {accepted_rollout_idx} of {args.num_rollouts} "
                    f"after {attempt_idx} attempts"
                )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        envs.close()

    trace_payload = {
        "dataset_class": "C_frozen_dp_dynamic_failure",
        "ckpt_path": args.ckpt_path,
        "state_key": state_key,
        "scenario": args.scenario,
        "motion_trigger_mode": args.motion_trigger_mode,
        "motion_trigger_threshold_m": args.motion_trigger_threshold_m,
        "motion_trigger_require_grasp": args.motion_trigger_require_grasp,
        "motion_trigger_min_step": args.motion_trigger_min_step,
        "seed": args.seed,
        "num_rollouts": args.num_rollouts,
        "max_rollout_attempts": max_rollout_attempts,
        "attempt_count": attempt_idx,
        "reset_info": locals().get("reset_info", {}),
        "motion_rows": motion_rows,
        "action_rows": action_rows,
        "task_rows": task_rows,
        "motion_trace_validation": validate_trace_rows(motion_rows),
        "peg_disturbance_physical_force": args.scenario == "peg_disturb",
        "quality_by_rollout": quality_by_rollout,
        "skipped_attempts": skipped_attempts,
        "positive_policy_data_allowed": False,
    }
    trace_path = trace_dir / "frozen_dp_trace.json"
    trace_path.write_text(json.dumps(_jsonable(trace_payload), indent=2, sort_keys=True), encoding="utf-8")

    loss_fields = manifest_fields("C_frozen_dp_dynamic_failure")
    dataset_smoke_only = _str_bool(args.dataset_smoke_only)
    human_review_required = _str_bool(args.human_review_required)
    large_scale_production_allowed = _str_bool(args.large_scale_production_allowed)
    all_quality_passed = (
        len(quality_by_rollout) == args.num_rollouts
        and all(row["quality_gate_passed"] for row in quality_by_rollout)
    )
    status = (
        ("smoke_complete" if dataset_smoke_only else "production_complete")
        if failure is None and len(video_paths) == args.num_rollouts and all_quality_passed
        else "failed"
    )
    summary = {
        "phase": "01_dataset",
        "dataset_class": "C_frozen_dp_dynamic_failure",
        "status": status,
        "failure": failure,
        "ckpt_path": args.ckpt_path,
        "state_key": state_key,
        "scenario": args.scenario,
        "motion_trigger_mode": args.motion_trigger_mode,
        "motion_trigger_threshold_m": args.motion_trigger_threshold_m,
        "motion_trigger_require_grasp": args.motion_trigger_require_grasp,
        "motion_trigger_min_step": args.motion_trigger_min_step,
        "peg_disturbance_physical_force": args.scenario == "peg_disturb",
        "peg_disturb_force_xyz": [args.peg_disturb_force_x, args.peg_disturb_force_y, args.peg_disturb_force_z],
        "peg_disturb_duration_steps": args.peg_disturb_duration_steps,
        "seed": args.seed,
        "num_rollouts_requested": args.num_rollouts,
        "max_rollout_attempts": max_rollout_attempts,
        "attempt_count": attempt_idx,
        "skipped_attempt_count": len(skipped_attempts),
        "rollout_count": len(video_paths),
        "max_episode_steps": args.max_episode_steps,
        "frame_count": total_frames,
        "video_count": len(video_paths),
        "video_bytes": sum(path.stat().st_size for path in video_paths),
        "rgb_required": True,
        "human_review_required": human_review_required,
        "large_scale_production_allowed": large_scale_production_allowed,
        "dataset_smoke_only": dataset_smoke_only,
        "method_evidence_allowed": False,
        "teacher_evidence_allowed": False,
        "positive_policy_data_allowed": False,
        "success_once": bool(success_once),
        "target_assisted": False,
        "state_intervention": False,
        "snap_or_teleport": False,
        "task_motion_quality_gate_passed": all_quality_passed,
        "quality_by_rollout": quality_by_rollout,
        "skipped_attempts": skipped_attempts,
        "allowed_losses": loss_fields["allowed_losses"],
        "disallowed_losses": loss_fields["disallowed_losses"],
        "output_dir": str(output_dir),
        "rgb_video": str(video_paths[0]) if video_paths else "",
        "rgb_videos": [str(path) for path in video_paths],
        "trace": str(trace_path),
        "final_info": final_info,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if failure is not None:
        raise RuntimeError(failure)
    if not video_paths:
        raise RuntimeError("no RGB frames were rendered")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-id", default="PegInsertionSide-v1")
    parser.add_argument("--control-mode", default="pd_ee_delta_pose")
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--max-rollout-attempts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=710001)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scenario", default="constant_lr")
    parser.add_argument("--motion-start-step", type=int, default=20)
    parser.add_argument(
        "--motion-trigger-mode",
        choices=["fixed_step", "peg_head_l2", "inserted", "pre_insert_l2"],
        default="pre_insert_l2",
    )
    parser.add_argument("--motion-trigger-threshold-m", type=float, default=0.20)
    parser.add_argument("--motion-trigger-min-step", type=int, default=0)
    parser.add_argument("--motion-trigger-require-grasp", action="store_true", default=True)
    parser.add_argument("--require-pre-insert-motion", action="store_true", default=True)
    parser.add_argument("--min-trigger-to-insert-steps", type=int, default=8)
    parser.add_argument("--motion-duration-steps", type=int, default=40)
    parser.add_argument("--delta-x", type=float, default=0.0)
    parser.add_argument("--delta-y", type=float, default=0.08)
    parser.add_argument("--delta-z", type=float, default=0.0)
    parser.add_argument("--reverse-fraction", type=float, default=0.5)
    parser.add_argument("--sine-cycles", type=float, default=0.5)
    parser.add_argument("--max-step-delta-m", type=float, default=0.004)
    parser.add_argument("--peg-disturb-force-x", type=float, default=0.0)
    parser.add_argument("--peg-disturb-force-y", type=float, default=-25.0)
    parser.add_argument("--peg-disturb-force-z", type=float, default=0.0)
    parser.add_argument("--peg-disturb-duration-steps", type=int, default=18)
    parser.add_argument("--min-tcp-motion-m", type=float, default=0.05)
    parser.add_argument("--min-peg-motion-m", type=float, default=0.03)
    parser.add_argument("--require-grasp", action="store_true", default=True)
    parser.add_argument("--grasp-review-step", type=int, default=90)
    parser.add_argument("--dataset-smoke-only", default="true")
    parser.add_argument("--human-review-required", default="true")
    parser.add_argument("--large-scale-production-allowed", default="false")
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--cuda", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    try:
        collect(parse_args())
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"frozen_dp_dynamic_smoke_failed": str(exc)}, sort_keys=True), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
