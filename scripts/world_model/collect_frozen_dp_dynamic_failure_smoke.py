#!/usr/bin/env python3
"""Collect C-class frozen-DP dynamic failure smoke data inside Slurm.

The policy is the official state Diffusion Policy checkpoint loaded through the
existing ManiSkill DP training code. The target motion is commanded through
`active_dynamic_peg_adapter.py`; failed chunks are negative/diagnostic data,
not positive BC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
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
    return {
        "tcp_p": tcp_p,
        "peg_p": peg_p,
        "peg_head_at_hole": peg_head_at_hole,
        "peg_head_l2": float(np.linalg.norm(peg_head_at_hole)),
        "grasped": grasped,
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
    video_paths: list[Path] = []
    total_frames = 0
    failure: str | None = None
    success_once = False
    final_info: dict[str, Any] = {}
    try:
        with torch.no_grad():
            for rollout_idx in range(args.num_rollouts):
                frames: list[np.ndarray] = []
                obs, reset_info = envs.reset(seed=args.seed + rollout_idx)
                base_env = envs.envs[0].unwrapped
                initial_p, initial_q = _pose_arrays(base_env.box)
                previous_p: np.ndarray | None = initial_p.copy()
                env_step = 0
                done = False
                rollout_task_rows: list[dict[str, Any]] = []
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
                        if motion_trigger_step is not None:
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
                            motion_row["motion_trigger_mode"] = args.motion_trigger_mode
                            motion_row["motion_trigger_step"] = int(motion_trigger_step)
                            motion_row["motion_trigger_threshold_m"] = args.motion_trigger_threshold_m
                            motion_row["pre_step_peg_head_l2"] = float(pre_task_state["peg_head_l2"])
                            motion_row["pre_step_grasped"] = bool(pre_task_state["grasped"])
                            motion_rows.append(motion_row)
                            previous_p = np.asarray(command.target_p, dtype=np.float64)

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
                            "env_step": int(env_step),
                            **_jsonable(task_state),
                        }
                        task_rows.append(task_row)
                        rollout_task_rows.append(task_row)
                        action_rows.append(
                            {
                                "rollout": int(rollout_idx),
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

                quality = _quality_summary(rollout_task_rows, args)
                quality["rollout"] = int(rollout_idx)
                quality["motion_trigger_step"] = motion_trigger_step
                quality["motion_trigger_mode"] = args.motion_trigger_mode
                quality_by_rollout.append(quality)
                if motion_trigger_step is None:
                    raise RuntimeError(f"motion trigger never reached for rollout {rollout_idx}")
                if not quality["quality_gate_passed"]:
                    raise RuntimeError(f"task motion quality gate failed: {quality['quality_gate_failures']}")

                if frames:
                    video_path = video_dir / f"rollout_{rollout_idx:06d}.mp4"
                    imageio.mimsave(video_path, frames, fps=args.fps)
                    video_paths.append(video_path)
                    total_frames += len(frames)
                    for idx in sorted(set([0, args.grasp_review_step, len(frames) // 2, len(frames) - 1])):
                        if 0 <= idx < len(frames):
                            imageio.imwrite(review_dir / f"rollout_{rollout_idx:06d}_frame_{idx:04d}.png", frames[idx])
    except Exception as exc:  # noqa: BLE001
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
        "reset_info": locals().get("reset_info", {}),
        "motion_rows": motion_rows,
        "action_rows": action_rows,
        "task_rows": task_rows,
        "motion_trace_validation": validate_trace_rows(motion_rows),
        "quality_by_rollout": quality_by_rollout,
        "positive_policy_data_allowed": False,
    }
    trace_path = trace_dir / "frozen_dp_trace.json"
    trace_path.write_text(json.dumps(_jsonable(trace_payload), indent=2, sort_keys=True), encoding="utf-8")

    loss_fields = manifest_fields("C_frozen_dp_dynamic_failure")
    dataset_smoke_only = _str_bool(args.dataset_smoke_only)
    human_review_required = _str_bool(args.human_review_required)
    large_scale_production_allowed = _str_bool(args.large_scale_production_allowed)
    all_quality_passed = bool(quality_by_rollout) and all(row["quality_gate_passed"] for row in quality_by_rollout)
    status = (
        ("smoke_complete" if dataset_smoke_only else "production_complete")
        if failure is None and frames and all_quality_passed
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
        "seed": args.seed,
        "num_rollouts_requested": args.num_rollouts,
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
    parser.add_argument("--seed", type=int, default=710001)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scenario", default="constant_lr")
    parser.add_argument("--motion-start-step", type=int, default=20)
    parser.add_argument("--motion-trigger-mode", choices=["fixed_step", "peg_head_l2"], default="peg_head_l2")
    parser.add_argument("--motion-trigger-threshold-m", type=float, default=0.12)
    parser.add_argument("--motion-trigger-min-step", type=int, default=0)
    parser.add_argument("--motion-trigger-require-grasp", action="store_true", default=True)
    parser.add_argument("--motion-duration-steps", type=int, default=40)
    parser.add_argument("--delta-x", type=float, default=0.0)
    parser.add_argument("--delta-y", type=float, default=0.08)
    parser.add_argument("--delta-z", type=float, default=0.0)
    parser.add_argument("--reverse-fraction", type=float, default=0.5)
    parser.add_argument("--sine-cycles", type=float, default=0.5)
    parser.add_argument("--max-step-delta-m", type=float, default=0.004)
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
