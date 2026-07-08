#!/usr/bin/env python3
"""Collect E-class Cosmos-predicted cooperation smoke data inside Slurm.

This collector consumes precomputed Cosmos/readout prediction JSONL and uses
those predicted future target frames to generate legal `pd_ee_delta_pose`
actions. It does not run Cosmos, does not use ground-truth future target labels
as controller input, and does not mark artifacts as method evidence.
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


def _str_bool(value: str) -> bool:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    raise ValueError(f"expected true/false, got {value!r}")


def _action_bounds(env: Any) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], Any]:
    space = env.action_space
    low = np.asarray(space.low, dtype=np.float64)
    high = np.asarray(space.high, dtype=np.float64)
    return low, high, space.shape, space.dtype


def _tcp_position(base_env: Any) -> np.ndarray:
    tcp = getattr(getattr(base_env, "agent", None), "tcp", None)
    if tcp is None:
        raise RuntimeError("PegInsertionSide-v1 agent has no tcp pose for Cosmos-predicted action generation")
    tcp_p, _tcp_q = _pose_arrays(tcp)
    return tcp_p


def _load_prediction_rows(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Cosmos/readout prediction JSONL missing: {path}")
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rollout = int(row.get("rollout", row.get("episode", row.get("sample_index", 0))))
            if "step" not in row:
                raise ValueError(f"prediction row {line_no} missing step")
            step = int(row["step"])
            rows[(rollout, step)] = row
    if not rows:
        raise ValueError(f"Cosmos/readout prediction JSONL is empty: {path}")
    return rows


def _prediction_target_p(row: dict[str, Any], initial_p: np.ndarray) -> tuple[np.ndarray, str]:
    for key in ("cosmos_predicted_future_target_p", "predicted_future_target_p", "predicted_target_p"):
        if key in row:
            return np.asarray(row[key], dtype=np.float64).reshape(3), key
    for key in ("cosmos_predicted_future_target_delta", "predicted_future_target_delta"):
        if key in row:
            return np.asarray(initial_p, dtype=np.float64).reshape(3) + np.asarray(row[key], dtype=np.float64).reshape(3), key
    raise ValueError("prediction row lacks a predicted future target position or delta")


def _cosmos_predicted_action(
    *,
    env: Any,
    base_env: Any,
    predicted_target_p: np.ndarray,
    approach_offset: np.ndarray,
    max_translation: float,
) -> np.ndarray:
    low, high, shape, dtype = _action_bounds(env)
    action = np.zeros(shape, dtype=np.float64).reshape(-1)
    tcp_p = _tcp_position(base_env)
    desired_delta = np.asarray(predicted_target_p, dtype=np.float64).reshape(3) + approach_offset - tcp_p
    dist = float(np.linalg.norm(desired_delta))
    if dist > max_translation and dist > 1e-9:
        desired_delta = desired_delta / dist * max_translation
    action[:3] = desired_delta[:3]
    action = np.clip(action.reshape(shape), low, high)
    return action.astype(dtype)


def collect(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    trace_dir = output_dir / "trace"
    review_dir = output_dir / "review" / "frames"
    video_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    prediction_jsonl = Path(args.prediction_jsonl)
    prediction_rows = _load_prediction_rows(prediction_jsonl)

    spec = DynamicMotionSpec(
        scenario=args.scenario,
        start_step=args.motion_start_step,
        duration_steps=args.motion_duration_steps,
        delta_xyz=(args.delta_x, args.delta_y, args.delta_z),
        reverse_fraction=args.reverse_fraction,
        sine_cycles=args.sine_cycles,
        max_step_delta_m=args.max_step_delta_m,
    )
    approach_offset = np.asarray(
        (args.approach_offset_x, args.approach_offset_y, args.approach_offset_z),
        dtype=np.float64,
    )

    env = gym.make(
        args.env_id,
        obs_mode="state",
        control_mode=args.control_mode,
        render_mode="rgb_array",
        reward_mode="dense",
        sim_backend=args.sim_backend,
    )

    motion_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    video_paths: list[Path] = []
    total_frames = 0
    success_once = False
    final_info: dict[str, Any] = {}
    failure: str | None = None
    try:
        for rollout_idx in range(args.num_rollouts):
            frames: list[np.ndarray] = []
            env.reset(seed=args.seed + rollout_idx)
            base_env = env.unwrapped
            initial_p, initial_q = _pose_arrays(base_env.box)
            previous_p: np.ndarray | None = initial_p.copy()

            for step in range(args.max_episode_steps):
                command = build_motion_command(
                    step=step,
                    spec=spec,
                    initial_p=initial_p,
                    initial_q=initial_q,
                    previous_p=previous_p,
                )
                if command is not None:
                    motion_row = command_target_from_motion(base_env.box, sapien.Pose, command)
                    motion_row["rollout"] = int(rollout_idx)
                    motion_rows.append(motion_row)
                    previous_p = np.asarray(command.target_p, dtype=np.float64)

                pred_row = prediction_rows.get((rollout_idx, step)) or prediction_rows.get((0, step))
                if pred_row is None:
                    raise RuntimeError(f"missing Cosmos/readout prediction for rollout={rollout_idx} step={step}")
                predicted_target_p, predicted_key = _prediction_target_p(pred_row, initial_p)
                action = _cosmos_predicted_action(
                    env=env,
                    base_env=base_env,
                    predicted_target_p=predicted_target_p,
                    approach_offset=approach_offset,
                    max_translation=args.max_action_translation,
                )
                _obs, reward, terminated, truncated, info = env.step(action)
                rendered = env.render()
                frames.append(_frame_from_render(rendered))
                success = _scalar_bool(info, "success")
                fail = _scalar_bool(info, "fail")
                success_once = success_once or success
                final_info = _jsonable(info)
                tcp_p = _tcp_position(base_env)
                action_rows.append(
                    {
                        "rollout": int(rollout_idx),
                        "step": int(step),
                        "prediction_key": predicted_key,
                        "prediction_source": pred_row.get("source_artifact", str(prediction_jsonl)),
                        "prediction_uncertainty": _jsonable(pred_row.get("uncertainty", pred_row.get("sigma", None))),
                        "predicted_target_p": predicted_target_p.astype(float).tolist(),
                        "tcp_p_after_step": tcp_p.astype(float).tolist(),
                        "action": np.asarray(action).reshape(-1).astype(float).tolist(),
                        "reward": _jsonable(reward),
                        "success": bool(success),
                        "fail": bool(fail),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "teacher_evidence_allowed": False,
                        "method_evidence_allowed": False,
                        "positive_policy_data_allowed": False,
                        "hidden_ground_truth_future_used": False,
                    }
                )
                if bool(terminated) or bool(truncated):
                    break

            if frames:
                video_path = video_dir / f"rollout_{rollout_idx:06d}.mp4"
                imageio.mimsave(video_path, frames, fps=args.fps)
                video_paths.append(video_path)
                total_frames += len(frames)
                if rollout_idx == 0:
                    for idx in sorted(set([0, len(frames) // 2, len(frames) - 1])):
                        imageio.imwrite(review_dir / f"rollout_000000_frame_{idx:04d}.png", frames[idx])
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()

    trace_path = trace_dir / "cosmos_predicted_trace.json"
    trace_payload = {
        "dataset_class": "E_cosmos_predicted_cooperation",
        "scenario": args.scenario,
        "seed": args.seed,
        "num_rollouts": args.num_rollouts,
        "prediction_jsonl": str(prediction_jsonl),
        "motion_rows": motion_rows,
        "action_rows": action_rows,
        "motion_trace_validation": validate_trace_rows(motion_rows),
        "teacher_evidence_allowed": False,
        "method_evidence_allowed": False,
        "positive_policy_data_allowed": False,
        "hidden_ground_truth_future_used": False,
    }
    trace_path.write_text(json.dumps(_jsonable(trace_payload), indent=2, sort_keys=True), encoding="utf-8")

    loss_fields = manifest_fields("E_cosmos_predicted_cooperation")
    dataset_smoke_only = _str_bool(args.dataset_smoke_only)
    human_review_required = _str_bool(args.human_review_required)
    large_scale_production_allowed = _str_bool(args.large_scale_production_allowed)
    status = (
        "smoke_complete" if dataset_smoke_only else "production_complete"
    ) if failure is None and video_paths else "failed"
    summary = {
        "phase": "01_dataset",
        "dataset_class": "E_cosmos_predicted_cooperation",
        "status": status,
        "failure": failure,
        "scenario": args.scenario,
        "seed": args.seed,
        "num_rollouts_requested": args.num_rollouts,
        "rollout_count": len(video_paths),
        "max_episode_steps": args.max_episode_steps,
        "frame_count": total_frames,
        "video_count": len(video_paths),
        "video_bytes": sum(path.stat().st_size for path in video_paths),
        "prediction_jsonl": str(prediction_jsonl),
        "rgb_required": True,
        "human_review_required": human_review_required,
        "large_scale_production_allowed": large_scale_production_allowed,
        "dataset_smoke_only": dataset_smoke_only,
        "method_evidence_allowed": False,
        "teacher_evidence_allowed": False,
        "positive_policy_data_allowed": False,
        "success_once": bool(success_once),
        "hidden_ground_truth_future_used": False,
        "state_intervention": False,
        "snap_or_teleport": False,
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
    parser.add_argument("--prediction-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-id", default="PegInsertionSide-v1")
    parser.add_argument("--control-mode", default="pd_ee_delta_pose")
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--seed", type=int, default=730001)
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=80)
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
    parser.add_argument("--max-action-translation", type=float, default=0.01)
    parser.add_argument("--approach-offset-x", type=float, default=-0.03)
    parser.add_argument("--approach-offset-y", type=float, default=0.0)
    parser.add_argument("--approach-offset-z", type=float, default=0.0)
    parser.add_argument("--dataset-smoke-only", default="true")
    parser.add_argument("--human-review-required", default="true")
    parser.add_argument("--large-scale-production-allowed", default="false")
    return parser.parse_args()


def main() -> None:
    try:
        collect(parse_args())
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"cosmos_predicted_coop_smoke_failed": str(exc)}, sort_keys=True), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
