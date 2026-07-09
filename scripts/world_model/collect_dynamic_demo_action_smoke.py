#!/usr/bin/env python3
"""Collect dynamic RGB smoke by replaying official pd_ee_delta_pose actions.

This collector is used for B observation-only and D teacher-only smoke. It
does not restore source simulator states. It resets by the official episode
seed, executes legal controller actions through env.step, commands continuous
target / hole motion through the active adapter, and rejects videos where the
robot does not visibly perform the peg-insertion task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import mani_skill.envs  # noqa: F401
import numpy as np
import sapien

from scripts.world_model.active_dynamic_peg_adapter import (
    DynamicMotionSpec,
    build_motion_command,
    command_target_from_motion,
    manifest_fields,
    validate_trace_rows,
)

B_CLASS = "B_dynamic_rgb_observation"
D_CLASS = "D_future_frame_cooperation_teacher"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist() if value.ndim else value.item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
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
    frame = _tensor_to_np(rendered)
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
    return bool(_tensor_to_np(info[key]).reshape(-1)[0])


def _str_bool(value: str) -> bool:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    raise ValueError(f"expected true/false, got {value!r}")


def _teacher_evidence_allowed(dataset_class: str) -> bool:
    teacher_evidence_allowed = True if dataset_class == D_CLASS else False
    return teacher_evidence_allowed


def _read_episodes(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    episodes = data.get("episodes")
    if not isinstance(episodes, list):
        raise RuntimeError(f"{path} missing episodes list")
    return episodes


def _load_actions(h5: h5py.File, episode_id: int) -> np.ndarray:
    group_name = f"traj_{episode_id}"
    if group_name not in h5:
        raise RuntimeError(f"missing H5 group {group_name}")
    actions = np.asarray(h5[group_name]["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise RuntimeError(f"{group_name} invalid action shape {actions.shape}")
    return actions[:, :7]


def _select_episodes(args: argparse.Namespace) -> list[dict[str, Any]]:
    episodes = _read_episodes(Path(args.source_json))
    selected: list[dict[str, Any]] = []
    max_attempts = args.max_source_episode_attempts
    if max_attempts <= 0:
        max_attempts = max(args.num_episodes * 4, args.num_episodes + 200)
    for episode in episodes[args.episode_start :]:
        if args.require_source_success and not bool(episode.get("success")):
            continue
        selected.append(episode)
        if len(selected) >= max_attempts:
            break
    if len(selected) < args.num_episodes:
        raise RuntimeError(
            f"requested {args.num_episodes} source episodes, found {len(selected)} after start {args.episode_start}"
        )
    return selected


def _hold_action(last_action: np.ndarray | None, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    action = np.zeros(shape, dtype=np.float32)
    if last_action is not None and last_action.size:
        flat = action.reshape(-1)
        prev = np.asarray(last_action, dtype=np.float32).reshape(-1)
        if flat.size and prev.size:
            flat[-1] = prev[-1]
    return action.astype(dtype)


def _source_action_for_space(source_action: np.ndarray, env: Any) -> np.ndarray:
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    flat = action.reshape(-1)
    source_flat = np.asarray(source_action, dtype=np.float32).reshape(-1)
    if source_flat.size > flat.size:
        raise RuntimeError(
            f"source action dim {source_flat.size} exceeds env action dim {flat.size}"
        )
    flat[: source_flat.size] = source_flat
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    np.clip(flat, low, high, out=flat)
    return action.astype(env.action_space.dtype)


def _planned_motion_command(
    *,
    step: int,
    spec: DynamicMotionSpec,
    initial_p: np.ndarray,
    initial_q: np.ndarray,
) -> Any | None:
    previous_p: np.ndarray | None = initial_p.copy()
    command = None
    for planned_step in range(step + 1):
        command = build_motion_command(
            step=planned_step,
            spec=spec,
            initial_p=initial_p,
            initial_q=initial_q,
            previous_p=previous_p,
        )
        if command is not None:
            previous_p = np.asarray(command.target_p, dtype=np.float64)
    return command


def _apply_future_teacher_residual(
    *,
    action: np.ndarray,
    env: Any,
    step: int,
    motion_trigger_step: int | None,
    initial_p: np.ndarray,
    initial_q: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    payload: dict[str, Any] = {
        "teacher_future_target_source": "motion_trigger_not_reached",
        "future_tau_steps": int(args.future_tau_steps),
        "future_target_delta_xyz": [0.0, 0.0, 0.0],
        "future_action_residual_xyz": [0.0, 0.0, 0.0],
        "future_residual_gain": float(args.future_residual_gain),
    }
    if motion_trigger_step is None:
        return action, payload
    effective_spec = DynamicMotionSpec(
        scenario=args.scenario,
        start_step=motion_trigger_step,
        duration_steps=args.motion_duration_steps,
        delta_xyz=(args.delta_x, args.delta_y, args.delta_z),
        reverse_fraction=args.reverse_fraction,
        sine_cycles=args.sine_cycles,
        max_step_delta_m=args.max_step_delta_m,
    )
    future_command = _planned_motion_command(
        step=step + args.future_tau_steps,
        spec=effective_spec,
        initial_p=initial_p,
        initial_q=initial_q,
    )
    if future_command is None:
        return action, payload

    low = np.asarray(env.action_space.low, dtype=np.float64).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float64).reshape(-1)
    action_shape = action.shape
    flat = np.asarray(action, dtype=np.float64).reshape(-1)
    future_delta = np.asarray(future_command.target_delta_xyz, dtype=np.float64).reshape(3)
    residual = future_delta * float(args.future_residual_gain)
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm > args.max_future_residual_m and residual_norm > 1e-9:
        residual = residual / residual_norm * args.max_future_residual_m
    flat[:3] = np.clip(flat[:3] + residual[:3], low[:3], high[:3])
    payload.update(
        {
            "teacher_future_target_source": "ground_truth_future_motion_plan",
            "future_target_step": int(step + args.future_tau_steps),
            "future_target_p": [float(x) for x in future_command.target_p],
            "future_target_delta_xyz": future_delta.astype(float).tolist(),
            "future_action_residual_xyz": residual.astype(float).tolist(),
        }
    )
    return flat.reshape(action_shape).astype(env.action_space.dtype), payload


def _task_state(base_env: Any) -> dict[str, Any]:
    tcp_p, _ = _pose_arrays(base_env.agent.tcp)
    peg_p, _ = _pose_arrays(base_env.peg)
    box_p, _ = _pose_arrays(base_env.box)
    peg_head = _tensor_to_np(base_env.peg_head_pos).reshape(-1, 3)[0].astype(np.float64)
    hole = _tensor_to_np(base_env.box_hole_pose.p).reshape(-1, 3)[0].astype(np.float64)
    peg_head_at_hole = _tensor_to_np((base_env.box_hole_pose.inv() * base_env.peg_head_pose).p).reshape(-1, 3)[
        0
    ].astype(np.float64)
    grasped = bool(_tensor_to_np(base_env.agent.is_grasping(base_env.peg, max_angle=20)).reshape(-1)[0])
    inserted = bool(_tensor_to_np(base_env.has_peg_inserted()[0]).reshape(-1)[0])
    return {
        "tcp_p": tcp_p,
        "peg_p": peg_p,
        "box_p": box_p,
        "peg_head_p": peg_head,
        "hole_p": hole,
        "peg_head_at_hole": peg_head_at_hole,
        "peg_head_l2": float(np.linalg.norm(peg_head_at_hole)),
        "grasped": grasped,
        "inserted": inserted,
    }


def _quality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"quality_gate_passed": False, "reason": "no_task_state_rows"}
    tcp = np.asarray([row["tcp_p"] for row in rows], dtype=np.float64)
    peg = np.asarray([row["peg_p"] for row in rows], dtype=np.float64)
    peg_head_l2 = np.asarray([row["peg_head_l2"] for row in rows], dtype=np.float64)
    grasp_once = any(bool(row["grasped"]) for row in rows)
    return {
        "tcp_motion_m": float(np.max(np.linalg.norm(tcp - tcp[0], axis=1))),
        "peg_motion_m": float(np.max(np.linalg.norm(peg - peg[0], axis=1))),
        "peg_lift_m": float(np.max(peg[:, 2] - peg[0, 2])),
        "min_peg_head_l2": float(np.min(peg_head_l2)),
        "final_peg_head_l2": float(peg_head_l2[-1]),
        "grasp_once": bool(grasp_once),
    }


def _check_quality(quality: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    if float(quality.get("tcp_motion_m", 0.0)) < args.min_tcp_motion_m:
        failures.append("tcp_motion_too_small")
    if float(quality.get("peg_motion_m", 0.0)) < args.min_peg_motion_m:
        failures.append("peg_motion_too_small")
    if args.require_grasp and not bool(quality.get("grasp_once")):
        failures.append("grasp_never_detected")
    payload = dict(quality)
    payload["quality_gate_passed"] = not failures
    payload["quality_gate_failures"] = failures
    payload["min_tcp_motion_m"] = args.min_tcp_motion_m
    payload["min_peg_motion_m"] = args.min_peg_motion_m
    payload["require_grasp"] = args.require_grasp
    return payload


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
            first_inserted_step = int(row["step"])
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
    step: int,
    motion_trigger_step: int | None,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if args.scenario != "peg_disturb" or motion_trigger_step is None:
        return None
    local_step = step - motion_trigger_step
    if local_step < 0 or local_step >= args.peg_disturb_duration_steps:
        return None
    force = np.asarray(
        (args.peg_disturb_force_x, args.peg_disturb_force_y, args.peg_disturb_force_z),
        dtype=np.float32,
    )
    base_env.peg.apply_force(force)
    force_norm = float(np.linalg.norm(force.astype(np.float64)))
    return {
        "step": int(step),
        "scenario": args.scenario,
        "command_kind": "peg_physical_force",
        "peg_perturb_trace": True,
        "peg_perturb_force_xyz": force.astype(float).tolist(),
        "instantaneous_delta_m": force_norm,
        "state_intervention": False,
        "snap_or_teleport": False,
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset_class not in {B_CLASS, D_CLASS}:
        raise RuntimeError(f"unsupported dataset class for demo-action collector: {args.dataset_class}")

    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    trace_dir = output_dir / "trace"
    review_dir = output_dir / "review" / "frames"
    video_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    spec = DynamicMotionSpec(
        scenario=args.scenario,
        start_step=args.motion_start_step,
        duration_steps=args.motion_duration_steps,
        delta_xyz=(args.delta_x, args.delta_y, args.delta_z),
        reverse_fraction=args.reverse_fraction,
        sine_cycles=args.sine_cycles,
        max_step_delta_m=args.max_step_delta_m,
    )

    selected = _select_episodes(args)
    env = gym.make(
        args.env_id,
        obs_mode="state",
        control_mode=args.control_mode,
        render_mode="rgb_array",
        reward_mode="dense",
        sim_backend=args.sim_backend,
        max_episode_steps=args.max_episode_steps,
    )

    motion_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    video_paths: list[Path] = []
    total_frames = 0
    success_once = False
    final_info: dict[str, Any] = {}
    failure: str | None = None
    quality_by_episode: list[dict[str, Any]] = []
    skipped_source_episodes: list[dict[str, Any]] = []
    source_episode_attempt_count = 0
    try:
        with h5py.File(args.source_h5, "r") as h5:
            accepted_episode_idx = 0
            for source_attempt_idx, episode in enumerate(selected):
                if accepted_episode_idx >= args.num_episodes:
                    break
                source_episode_attempt_count += 1
                local_idx = accepted_episode_idx
                episode_id = int(episode["episode_id"])
                seed = int(episode.get("episode_seed", episode_id))
                source_actions = _load_actions(h5, episode_id)
                frames: list[np.ndarray] = []
                env.reset(seed=seed)
                base_env = env.unwrapped
                initial_p, initial_q = _pose_arrays(base_env.box)
                previous_p: np.ndarray | None = initial_p.copy()
                episode_task_rows: list[dict[str, Any]] = []
                episode_motion_rows: list[dict[str, Any]] = []
                episode_action_rows: list[dict[str, Any]] = []
                last_action: np.ndarray | None = None
                motion_trigger_step: int | None = None
                episode_success_once = False
                episode_final_info: dict[str, Any] = {}

                for step in range(args.max_episode_steps):
                    pre_task_state = _task_state(base_env)
                    motion_trigger_step = _maybe_trigger_motion(
                        step=step,
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
                            step=step,
                            spec=effective_spec,
                            initial_p=initial_p,
                            initial_q=initial_q,
                            previous_p=previous_p,
                        )
                    if command is not None:
                        motion_row = command_target_from_motion(base_env.box, sapien.Pose, command)
                        motion_row["episode"] = int(local_idx)
                        motion_row["source_episode_id"] = episode_id
                        motion_row["source_attempt_idx"] = int(source_attempt_idx)
                        motion_row["motion_trigger_mode"] = args.motion_trigger_mode
                        motion_row["motion_trigger_step"] = int(motion_trigger_step)
                        motion_row["motion_trigger_threshold_m"] = args.motion_trigger_threshold_m
                        motion_row["pre_step_peg_head_l2"] = float(pre_task_state["peg_head_l2"])
                        motion_row["pre_step_grasped"] = bool(pre_task_state["grasped"])
                        episode_motion_rows.append(motion_row)
                        previous_p = np.asarray(command.target_p, dtype=np.float64)
                    peg_force_row = _maybe_apply_peg_disturbance(
                        base_env=base_env,
                        step=step,
                        motion_trigger_step=motion_trigger_step,
                        args=args,
                    )
                    if peg_force_row is not None:
                        peg_force_row["episode"] = int(local_idx)
                        peg_force_row["source_episode_id"] = episode_id
                        peg_force_row["source_attempt_idx"] = int(source_attempt_idx)
                        peg_force_row["motion_trigger_mode"] = args.motion_trigger_mode
                        peg_force_row["motion_trigger_step"] = int(motion_trigger_step)
                        peg_force_row["pre_step_peg_head_l2"] = float(pre_task_state["peg_head_l2"])
                        peg_force_row["pre_step_grasped"] = bool(pre_task_state["grasped"])
                        episode_motion_rows.append(peg_force_row)

                    if step < source_actions.shape[0]:
                        action = _source_action_for_space(source_actions[step], env)
                    else:
                        action = _hold_action(last_action, env.action_space.shape, env.action_space.dtype)
                    teacher_payload: dict[str, Any] = {}
                    if args.dataset_class == D_CLASS:
                        action, teacher_payload = _apply_future_teacher_residual(
                            action=action,
                            env=env,
                            step=step,
                            motion_trigger_step=motion_trigger_step,
                            initial_p=initial_p,
                            initial_q=initial_q,
                            args=args,
                        )
                    last_action = np.asarray(action, dtype=np.float32).copy()

                    _obs, reward, terminated, truncated, info = env.step(action)
                    rendered = env.render()
                    frames.append(_frame_from_render(rendered))
                    task_state = _task_state(base_env)
                    task_row = {
                        "episode": int(local_idx),
                        "source_episode_id": episode_id,
                        "source_attempt_idx": int(source_attempt_idx),
                        "step": int(step),
                        **_jsonable(task_state),
                    }
                    episode_task_rows.append(task_row)
                    success = _scalar_bool(info, "success")
                    fail = _scalar_bool(info, "fail")
                    episode_success_once = episode_success_once or success
                    episode_final_info = _jsonable(info)
                    episode_action_rows.append(
                        {
                            "episode": int(local_idx),
                            "source_episode_id": episode_id,
                            "source_attempt_idx": int(source_attempt_idx),
                            "step": int(step),
                            "source_action_available": bool(step < source_actions.shape[0]),
                            "action": np.asarray(action).reshape(-1).astype(float).tolist(),
                            "reward": _jsonable(reward),
                            "success": bool(success),
                            "fail": bool(fail),
                            "terminated": _jsonable(terminated),
                            "truncated": _jsonable(truncated),
                            "method_evidence_allowed": False,
                            "teacher_evidence_allowed": _teacher_evidence_allowed(args.dataset_class),
                            "positive_policy_data_allowed": False,
                            **_jsonable(teacher_payload),
                        }
                    )

                quality = _add_pre_insert_timing_quality(
                    _check_quality(_quality_summary(episode_task_rows), args),
                    episode_task_rows,
                    motion_trigger_step,
                    args,
                )
                quality["episode"] = int(local_idx)
                quality["source_episode_id"] = episode_id
                quality["source_attempt_idx"] = int(source_attempt_idx)
                quality["motion_trigger_step"] = motion_trigger_step
                quality["motion_trigger_mode"] = args.motion_trigger_mode
                if motion_trigger_step is None:
                    quality["accepted"] = False
                    quality["skip_reason"] = "motion_trigger_never_reached"
                    skipped_source_episodes.append(quality)
                    continue
                if not quality["quality_gate_passed"]:
                    quality["accepted"] = False
                    quality["skip_reason"] = "task_motion_quality_gate_failed"
                    skipped_source_episodes.append(quality)
                    continue

                quality["accepted"] = True
                quality_by_episode.append(quality)
                motion_rows.extend(episode_motion_rows)
                action_rows.extend(episode_action_rows)
                task_rows.extend(episode_task_rows)
                success_once = success_once or episode_success_once
                final_info = episode_final_info

                video_path = video_dir / f"episode_{local_idx:06d}.mp4"
                imageio.mimsave(video_path, frames, fps=args.fps)
                video_paths.append(video_path)
                total_frames += len(frames)
                review_indices = sorted(set([0, args.grasp_review_step, len(frames) // 2, len(frames) - 1]))
                for idx in review_indices:
                    if 0 <= idx < len(frames):
                        imageio.imwrite(review_dir / f"episode_{local_idx:06d}_frame_{idx:04d}.png", frames[idx])
                accepted_episode_idx += 1
            if accepted_episode_idx < args.num_episodes:
                raise RuntimeError(
                    "insufficient accepted source episodes: "
                    f"accepted {accepted_episode_idx} of {args.num_episodes} "
                    f"after {source_episode_attempt_count} attempts"
                )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()

    loss_fields = manifest_fields(args.dataset_class)
    trace_path = trace_dir / "demo_action_trace.json"
    trace_payload = {
        "dataset_class": args.dataset_class,
        "source_h5": args.source_h5,
        "source_json": args.source_json,
        "source_controller": "official_pd_ee_delta_pose_demo_actions",
        "scenario": args.scenario,
        "motion_trigger_mode": args.motion_trigger_mode,
        "motion_trigger_threshold_m": args.motion_trigger_threshold_m,
        "motion_trigger_require_grasp": args.motion_trigger_require_grasp,
        "motion_trigger_min_step": args.motion_trigger_min_step,
        "selected_source_episodes": _jsonable(selected),
        "source_episode_attempt_count": source_episode_attempt_count,
        "accepted_episode_count": len(video_paths),
        "skipped_source_episodes": skipped_source_episodes,
        "motion_rows": motion_rows,
        "action_rows": action_rows,
        "task_rows": task_rows,
        "motion_trace_validation": validate_trace_rows(motion_rows),
        "peg_disturbance_physical_force": args.scenario == "peg_disturb",
        "quality_by_episode": quality_by_episode,
        "method_evidence_allowed": False,
        "teacher_evidence_allowed": _teacher_evidence_allowed(args.dataset_class),
        "positive_policy_data_allowed": False,
        "teacher_future_target_source": (
            "ground_truth_future_motion_plan" if args.dataset_class == D_CLASS else "not_teacher_data"
        ),
    }
    trace_path.write_text(json.dumps(_jsonable(trace_payload), indent=2, sort_keys=True), encoding="utf-8")

    dataset_smoke_only = _str_bool(args.dataset_smoke_only)
    human_review_required = _str_bool(args.human_review_required)
    large_scale_production_allowed = _str_bool(args.large_scale_production_allowed)
    all_quality_passed = (
        len(quality_by_episode) == args.num_episodes
        and all(row["quality_gate_passed"] for row in quality_by_episode)
    )
    status = (
        ("smoke_complete" if dataset_smoke_only else "production_complete")
        if failure is None and len(video_paths) == args.num_episodes and all_quality_passed
        else "failed"
    )
    summary = {
        "phase": "01_dataset",
        "dataset_class": args.dataset_class,
        "status": status,
        "failure": failure,
        "source_h5": args.source_h5,
        "source_json": args.source_json,
        "source_controller": "official_pd_ee_delta_pose_demo_actions",
        "teacher_future_target_source": (
            "ground_truth_future_motion_plan" if args.dataset_class == D_CLASS else "not_teacher_data"
        ),
        "teacher_action_adapter": (
            "official_demo_actions_plus_gt_future_residual" if args.dataset_class == D_CLASS else "not_teacher_data"
        ),
        "scenario": args.scenario,
        "motion_trigger_mode": args.motion_trigger_mode,
        "motion_trigger_threshold_m": args.motion_trigger_threshold_m,
        "motion_trigger_require_grasp": args.motion_trigger_require_grasp,
        "motion_trigger_min_step": args.motion_trigger_min_step,
        "peg_disturbance_physical_force": args.scenario == "peg_disturb",
        "peg_disturb_force_xyz": [args.peg_disturb_force_x, args.peg_disturb_force_y, args.peg_disturb_force_z],
        "peg_disturb_duration_steps": args.peg_disturb_duration_steps,
        "episode_start": args.episode_start,
        "num_episodes_requested": args.num_episodes,
        "max_source_episode_attempts": args.max_source_episode_attempts,
        "source_episode_attempt_count": source_episode_attempt_count,
        "skipped_source_episode_count": len(skipped_source_episodes),
        "episode_count": len(video_paths),
        "max_episode_steps": args.max_episode_steps,
        "frame_count": total_frames,
        "video_count": len(video_paths),
        "video_bytes": sum(path.stat().st_size for path in video_paths),
        "rgb_required": True,
        "human_review_required": human_review_required,
        "large_scale_production_allowed": large_scale_production_allowed,
        "dataset_smoke_only": dataset_smoke_only,
        "method_evidence_allowed": False,
        "teacher_evidence_allowed": _teacher_evidence_allowed(args.dataset_class),
        "positive_policy_data_allowed": False,
        "success_once": bool(success_once),
        "target_assisted": False,
        "state_intervention": False,
        "snap_or_teleport": False,
        "task_motion_quality_gate_passed": all_quality_passed,
        "quality_by_episode": quality_by_episode,
        "skipped_source_episodes": skipped_source_episodes,
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
    if not all_quality_passed:
        raise RuntimeError("task motion quality gate failed")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-class", required=True, choices=[B_CLASS, D_CLASS])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-h5", required=True)
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--env-id", default="PegInsertionSide-v1")
    parser.add_argument("--control-mode", default="pd_ee_delta_pose")
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-source-episode-attempts", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scenario", default="constant_lr")
    parser.add_argument("--motion-start-step", type=int, default=120)
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
    parser.add_argument("--motion-duration-steps", type=int, default=150)
    parser.add_argument("--delta-x", type=float, default=0.0)
    parser.add_argument("--delta-y", type=float, default=0.08)
    parser.add_argument("--delta-z", type=float, default=0.0)
    parser.add_argument("--reverse-fraction", type=float, default=0.5)
    parser.add_argument("--sine-cycles", type=float, default=0.5)
    parser.add_argument("--max-step-delta-m", type=float, default=0.004)
    parser.add_argument("--future-tau-steps", type=int, default=12)
    parser.add_argument("--future-residual-gain", type=float, default=0.5)
    parser.add_argument("--max-future-residual-m", type=float, default=0.03)
    parser.add_argument("--peg-disturb-force-x", type=float, default=0.0)
    parser.add_argument("--peg-disturb-force-y", type=float, default=-25.0)
    parser.add_argument("--peg-disturb-force-z", type=float, default=0.0)
    parser.add_argument("--peg-disturb-duration-steps", type=int, default=18)
    parser.add_argument("--min-tcp-motion-m", type=float, default=0.05)
    parser.add_argument("--min-peg-motion-m", type=float, default=0.03)
    parser.add_argument("--require-grasp", action="store_true", default=True)
    parser.add_argument("--require-source-success", action="store_true", default=True)
    parser.add_argument("--grasp-review-step", type=int, default=90)
    parser.add_argument("--dataset-smoke-only", default="true")
    parser.add_argument("--human-review-required", default="true")
    parser.add_argument("--large-scale-production-allowed", default="false")
    return parser.parse_args()


def main() -> None:
    try:
        collect(parse_args())
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"dynamic_demo_action_smoke_failed": str(exc)}, sort_keys=True), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
