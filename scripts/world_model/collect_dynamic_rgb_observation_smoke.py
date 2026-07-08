#!/usr/bin/env python3
"""Collect B-class dynamic RGB observation smoke data inside Slurm.

This script must run only after a shell runner has sourced
`require_dataset_runtime_context.sh`. It uses the active dynamic adapter to
command continuous target motion and records RGB video plus trace artifacts.
It does not mark robot actions as positive expert data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import sapien

import mani_skill.envs  # noqa: F401
from scripts.world_model.active_dynamic_peg_adapter import (
    DynamicMotionSpec,
    build_motion_command,
    command_target_from_motion,
    manifest_fields,
    validate_trace_rows,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
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


def _zero_action(env: Any) -> np.ndarray:
    action_space = env.action_space
    return np.zeros(action_space.shape, dtype=action_space.dtype)


def _write_text_manifest(path: Path, fields: dict[str, Any]) -> None:
    lines = []
    for key, value in fields.items():
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(_jsonable(value), sort_keys=True)
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _str_bool(value: str) -> bool:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    raise ValueError(f"expected true/false, got {value!r}")


def collect(args: argparse.Namespace) -> dict[str, Any]:
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

    env = gym.make(
        args.env_id,
        obs_mode="state",
        control_mode=args.control_mode,
        render_mode="rgb_array",
        reward_mode="dense",
        sim_backend=args.sim_backend,
        max_episode_steps=args.steps_per_episode,
    )

    all_trace_rows: list[dict[str, Any]] = []
    video_paths: list[Path] = []
    total_frames = 0
    failure: str | None = None
    try:
        for episode_idx in range(args.num_episodes):
            frames: list[np.ndarray] = []
            env.reset(seed=args.seed + episode_idx)
            base_env = env.unwrapped
            initial_p, initial_q = _pose_arrays(base_env.box)
            previous_p: np.ndarray | None = initial_p.copy()

            for step in range(args.steps_per_episode):
                command = build_motion_command(
                    step=step,
                    spec=spec,
                    initial_p=initial_p,
                    initial_q=initial_q,
                    previous_p=previous_p,
                )
                if command is not None:
                    row = command_target_from_motion(base_env.box, sapien.Pose, command)
                    row["episode"] = int(episode_idx)
                    all_trace_rows.append(row)
                    previous_p = np.asarray(command.target_p, dtype=np.float64)

                _obs, _reward, terminated, truncated, info = env.step(_zero_action(env))
                rendered = env.render()
                frames.append(_frame_from_render(rendered))
                if bool(terminated) or bool(truncated):
                    all_trace_rows.append(
                        {
                            "motion_trace": True,
                            "episode": int(episode_idx),
                            "step": int(step),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "info": _jsonable(info),
                            "state_intervention": False,
                            "snap_or_teleport": False,
                        }
                    )
                    break

            if frames:
                video_path = video_dir / f"episode_{episode_idx:06d}.mp4"
                imageio.mimsave(video_path, frames, fps=args.fps)
                video_paths.append(video_path)
                total_frames += len(frames)
                if episode_idx == 0:
                    for idx in sorted(set([0, len(frames) // 2, len(frames) - 1])):
                        imageio.imwrite(review_dir / f"episode_000000_frame_{idx:04d}.png", frames[idx])
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()

    trace_path = trace_dir / "motion_trace.json"
    trace_payload = {
        "dataset_class": "B_dynamic_rgb_observation",
        "scenario": args.scenario,
        "seed": args.seed,
        "num_episodes": args.num_episodes,
        "steps_per_episode": args.steps_per_episode,
        "rows": all_trace_rows,
        "trace_validation": validate_trace_rows(all_trace_rows),
    }
    trace_path.write_text(json.dumps(_jsonable(trace_payload), indent=2, sort_keys=True), encoding="utf-8")

    loss_fields = manifest_fields("B_dynamic_rgb_observation")
    dataset_smoke_only = _str_bool(args.dataset_smoke_only)
    human_review_required = _str_bool(args.human_review_required)
    large_scale_production_allowed = _str_bool(args.large_scale_production_allowed)
    status = ("smoke_complete" if dataset_smoke_only else "production_complete") if failure is None and frames else "failed"
    summary = {
        "phase": "01_dataset",
        "dataset_class": "B_dynamic_rgb_observation",
        "status": status,
        "failure": failure,
        "scenario": args.scenario,
        "seed": args.seed,
        "num_episodes_requested": args.num_episodes,
        "episode_count": len(video_paths),
        "steps_per_episode": args.steps_per_episode,
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
        "state_intervention": False,
        "snap_or_teleport": False,
        "allowed_losses": loss_fields["allowed_losses"],
        "disallowed_losses": loss_fields["disallowed_losses"],
        "output_dir": str(output_dir),
        "rgb_video": str(video_paths[0]) if video_paths else "",
        "rgb_videos": [str(path) for path in video_paths],
        "motion_trace": str(trace_path),
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-id", default="PegInsertionSide-v1")
    parser.add_argument("--control-mode", default="pd_ee_delta_pose")
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--seed", type=int, default=700001)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--steps-per-episode", type=int, default=80)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scenario", default="constant_lr")
    parser.add_argument("--motion-start-step", type=int, default=20)
    parser.add_argument("--motion-duration-steps", type=int, default=40)
    parser.add_argument("--delta-x", type=float, default=0.0)
    parser.add_argument("--delta-y", type=float, default=0.08)
    parser.add_argument("--delta-z", type=float, default=0.0)
    parser.add_argument("--reverse-fraction", type=float, default=0.5)
    parser.add_argument("--sine-cycles", type=float, default=0.5)
    parser.add_argument("--max-step-delta-m", type=float, default=0.004)
    parser.add_argument("--dataset-smoke-only", default="true")
    parser.add_argument("--human-review-required", default="true")
    parser.add_argument("--large-scale-production-allowed", default="false")
    return parser.parse_args()


def main() -> None:
    try:
        collect(parse_args())
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"dynamic_rgb_smoke_failed": str(exc)}, sort_keys=True), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
