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
    for episode in episodes[args.episode_start :]:
        if args.require_source_success and not bool(episode.get("success")):
            continue
        selected.append(episode)
        if len(selected) >= args.num_episodes:
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
    return {
        "tcp_p": tcp_p,
        "peg_p": peg_p,
        "box_p": box_p,
        "peg_head_p": peg_head,
        "hole_p": hole,
        "peg_head_at_hole": peg_head_at_hole,
        "peg_head_l2": float(np.linalg.norm(peg_head_at_hole)),
        "grasped": grasped,
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
    try:
        with h5py.File(args.source_h5, "r") as h5:
            for local_idx, episode in enumerate(selected):
                episode_id = int(episode["episode_id"])
                seed = int(episode.get("episode_seed", episode_id))
                source_actions = _load_actions(h5, episode_id)
                frames: list[np.ndarray] = []
                env.reset(seed=seed)
                base_env = env.unwrapped
                initial_p, initial_q = _pose_arrays(base_env.box)
                previous_p: np.ndarray | None = initial_p.copy()
                episode_task_rows: list[dict[str, Any]] = []
                last_action: np.ndarray | None = None
                motion_trigger_step: int | None = None

                for step in range(args.max_episode_steps):
                    pre_task_state = _task_state(base_env)
                    motion_trigger_step = _maybe_trigger_motion(
                        step=step,
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
                        motion_row["motion_trigger_mode"] = args.motion_trigger_mode
                        motion_row["motion_trigger_step"] = int(motion_trigger_step)
                        motion_row["motion_trigger_threshold_m"] = args.motion_trigger_threshold_m
                        motion_row["pre_step_peg_head_l2"] = float(pre_task_state["peg_head_l2"])
                        motion_row["pre_step_grasped"] = bool(pre_task_state["grasped"])
                        motion_rows.append(motion_row)
                        previous_p = np.asarray(command.target_p, dtype=np.float64)

                    if step < source_actions.shape[0]:
                        action = np.asarray(source_actions[step], dtype=env.action_space.dtype)
                    else:
                        action = _hold_action(last_action, env.action_space.shape, env.action_space.dtype)
                    last_action = np.asarray(action, dtype=np.float32).copy()

                    _obs, reward, terminated, truncated, info = env.step(action)
                    rendered = env.render()
                    frames.append(_frame_from_render(rendered))
                    task_state = _task_state(base_env)
                    task_row = {
                        "episode": int(local_idx),
                        "source_episode_id": episode_id,
                        "step": int(step),
                        **_jsonable(task_state),
                    }
                    task_rows.append(task_row)
                    episode_task_rows.append(task_row)
                    success = _scalar_bool(info, "success")
                    fail = _scalar_bool(info, "fail")
                    success_once = success_once or success
                    final_info = _jsonable(info)
                    action_rows.append(
                        {
                            "episode": int(local_idx),
                            "source_episode_id": episode_id,
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
                        }
                    )

                quality = _check_quality(_quality_summary(episode_task_rows), args)
                quality["episode"] = int(local_idx)
                quality["source_episode_id"] = episode_id
                quality["motion_trigger_step"] = motion_trigger_step
                quality["motion_trigger_mode"] = args.motion_trigger_mode
                quality_by_episode.append(quality)
                if motion_trigger_step is None:
                    raise RuntimeError(f"motion trigger never reached for source episode {episode_id}")
                if not quality["quality_gate_passed"]:
                    raise RuntimeError(
                        f"task motion quality gate failed for source episode {episode_id}: {quality['quality_gate_failures']}"
                    )

                video_path = video_dir / f"episode_{local_idx:06d}.mp4"
                imageio.mimsave(video_path, frames, fps=args.fps)
                video_paths.append(video_path)
                total_frames += len(frames)
                review_indices = sorted(set([0, args.grasp_review_step, len(frames) // 2, len(frames) - 1]))
                for idx in review_indices:
                    if 0 <= idx < len(frames):
                        imageio.imwrite(review_dir / f"episode_{local_idx:06d}_frame_{idx:04d}.png", frames[idx])
    except Exception as exc:  # noqa: BLE001
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
        "motion_rows": motion_rows,
        "action_rows": action_rows,
        "task_rows": task_rows,
        "motion_trace_validation": validate_trace_rows(motion_rows),
        "quality_by_episode": quality_by_episode,
        "method_evidence_allowed": False,
        "teacher_evidence_allowed": _teacher_evidence_allowed(args.dataset_class),
        "positive_policy_data_allowed": False,
    }
    trace_path.write_text(json.dumps(_jsonable(trace_payload), indent=2, sort_keys=True), encoding="utf-8")

    dataset_smoke_only = _str_bool(args.dataset_smoke_only)
    human_review_required = _str_bool(args.human_review_required)
    large_scale_production_allowed = _str_bool(args.large_scale_production_allowed)
    all_quality_passed = bool(quality_by_episode) and all(row["quality_gate_passed"] for row in quality_by_episode)
    status = (
        ("smoke_complete" if dataset_smoke_only else "production_complete")
        if failure is None and video_paths and all_quality_passed
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
        "scenario": args.scenario,
        "motion_trigger_mode": args.motion_trigger_mode,
        "motion_trigger_threshold_m": args.motion_trigger_threshold_m,
        "motion_trigger_require_grasp": args.motion_trigger_require_grasp,
        "motion_trigger_min_step": args.motion_trigger_min_step,
        "episode_start": args.episode_start,
        "num_episodes_requested": args.num_episodes,
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
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scenario", default="constant_lr")
    parser.add_argument("--motion-start-step", type=int, default=120)
    parser.add_argument("--motion-trigger-mode", choices=["fixed_step", "peg_head_l2"], default="peg_head_l2")
    parser.add_argument("--motion-trigger-threshold-m", type=float, default=0.12)
    parser.add_argument("--motion-trigger-min-step", type=int, default=0)
    parser.add_argument("--motion-trigger-require-grasp", action="store_true", default=True)
    parser.add_argument("--motion-duration-steps", type=int, default=150)
    parser.add_argument("--delta-x", type=float, default=0.0)
    parser.add_argument("--delta-y", type=float, default=0.08)
    parser.add_argument("--delta-z", type=float, default=0.0)
    parser.add_argument("--reverse-fraction", type=float, default=0.5)
    parser.add_argument("--sine-cycles", type=float, default=0.5)
    parser.add_argument("--max-step-delta-m", type=float, default=0.004)
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
