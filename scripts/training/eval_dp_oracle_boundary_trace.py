#!/usr/bin/env python3
"""Trace a Phase 03 Oracle boundary run with RecordEpisode video.

This follows the Phase 01 DP trace harness as closely as possible, then adds
only the Phase 03 diagnostic pieces: controlled target motion and an explicit
Oracle boundary / decision marker. It must not teleport the peg into the hole.
The result is not method evidence.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import tyro
from gymnasium.vector import SyncVectorEnv
from mani_skill.utils import common
from mani_skill.utils.structs import Pose
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack, RecordEpisode

import train as ms_train


@dataclass
class OracleTraceArgs:
    ckpt_path: str
    output_dir: str
    phase02_run: str
    phase02_sample: str = "hole_late_fast_shift_seed10300001_idx5000"
    env_id: str = "PegInsertionSide-v1"
    control_mode: str = "pd_ee_delta_pose"
    sim_backend: str = "physx_cpu"
    max_episode_steps: int = 300
    seed: int = 2
    target_motion_step: int = 84
    target_motion_y: float = 0.025
    motion_detect_threshold: float = 0.003
    max_oracle_step_displacement: float = 0.05
    use_ema: bool = True
    cuda: bool = True


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return to_jsonable(value.item())
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def require_allocation() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("refusing_login_node_execution=true; run inside a tmux-held Slurm allocation")


def build_train_args(saved: dict, trace_args: OracleTraceArgs) -> ms_train.Args:
    args = ms_train.Args()
    for key, value in saved.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.env_id = trace_args.env_id
    args.control_mode = trace_args.control_mode
    args.sim_backend = trace_args.sim_backend
    args.max_episode_steps = trace_args.max_episode_steps
    args.num_eval_envs = 1
    args.num_eval_episodes = 1
    args.seed = trace_args.seed
    args.capture_video = True
    return args


def make_trace_env(args: ms_train.Args, video_dir: Path):
    assert args.sim_backend == "physx_cpu"

    def thunk():
        env = gym.make(
            args.env_id,
            reconfiguration_freq=1,
            control_mode=args.control_mode,
            reward_mode="sparse",
            obs_mode="state",
            render_mode="rgb_array",
            human_render_camera_configs=dict(shader_pack="default"),
            max_episode_steps=args.max_episode_steps,
        )
        env = FrameStack(env, num_stack=args.obs_horizon)
        env = CPUGymWrapper(env, ignore_terminations=False, record_metrics=True)
        env = RecordEpisode(
            env,
            output_dir=str(video_dir),
            save_trajectory=False,
            save_video=True,
            info_on_video=True,
            source_type="phase03_oracle",
            source_desc="phase03 oracle boundary diagnostic; no peg final-seat state intervention",
        )
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)
        return env

    return SyncVectorEnv([thunk])


def first_info_value(info: dict, key: str) -> Any:
    if key not in info:
        return None
    value = to_jsonable(info[key])
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def eval_state(base_env: Any) -> dict[str, Any]:
    info = base_env.evaluate()
    rel = np.asarray(to_jsonable(info["peg_head_pos_at_hole"]), dtype=np.float64).reshape(-1)[:3]
    return {
        "success": bool(np.asarray(info["success"]).reshape(-1)[0]),
        "peg_head_at_hole": rel.astype(float).tolist(),
        "peg_head_l2": float(np.linalg.norm(rel)),
        "peg_pose": to_jsonable(base_env.peg.pose.raw_pose),
        "box_hole_pose": to_jsonable(base_env.box_hole_pose.raw_pose),
        "goal_pose": to_jsonable(base_env.goal_pose.raw_pose),
    }


def hole_xyz(base_env: Any) -> np.ndarray:
    return np.asarray(to_jsonable(base_env.box_hole_pose.p), dtype=np.float64).reshape(-1)[:3]


def shift_box(base_env: Any, delta_xyz: np.ndarray) -> None:
    current = base_env.box.pose
    p = current.p + torch.as_tensor(delta_xyz, dtype=current.p.dtype, device=current.p.device).reshape(1, 3)
    base_env.box.set_pose(Pose.create_from_pq(p=p, q=current.q))


def main() -> None:
    require_allocation()
    trace_args = tyro.cli(OracleTraceArgs)
    output_dir = Path(trace_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"

    phase02 = Path(trace_args.phase02_run)
    cosmos_video = phase02 / "cosmos_outputs" / trace_args.phase02_sample / "vision.mp4"
    if not cosmos_video.exists() or cosmos_video.stat().st_size <= 0:
        raise FileNotFoundError(f"missing Cosmos video: {cosmos_video}")

    device = torch.device("cuda" if torch.cuda.is_available() and trace_args.cuda else "cpu")
    ckpt = torch.load(trace_args.ckpt_path, map_location=device)
    args = build_train_args(ckpt.get("args", {}), trace_args)

    ms_train.args = args
    ms_train.device = device

    envs = make_trace_env(args, video_dir)
    agent = ms_train.Agent(envs, args).to(device)
    state_key = "ema_agent" if trace_args.use_ema else "agent"
    agent.load_state_dict(ckpt[state_key])
    agent.eval()

    action_trace: list[dict[str, Any]] = []
    target_motion_trigger_frame = None
    before_oracle = None
    after_oracle = None
    oracle_jump_distance = None
    target_motion_applied = False

    try:
        with torch.no_grad():
            obs, reset_info = envs.reset(seed=trace_args.seed)
            base_env = envs.envs[0].unwrapped
            initial_hole = hole_xyz(base_env)
            initial_eval = eval_state(base_env)

            for chunk_idx in range(args.max_episode_steps):
                obs_tensor = common.to_tensor(obs, device)
                action_seq = agent.get_action(obs_tensor).detach().cpu().numpy()
                for action_idx in range(action_seq.shape[1]):
                    step_idx = len(action_trace)
                    if (not target_motion_applied) and step_idx >= trace_args.target_motion_step:
                        shift_box(base_env, np.asarray([0.0, trace_args.target_motion_y, 0.0], dtype=np.float64))
                        target_motion_applied = True

                    action = action_seq[:, action_idx]
                    obs, reward, terminated, truncated, info = envs.step(action)
                    live_eval = eval_state(base_env)
                    hole_delta = float(np.linalg.norm(hole_xyz(base_env) - initial_hole))
                    row = {
                        "env_step": step_idx,
                        "chunk_idx": chunk_idx,
                        "action_idx": action_idx,
                        "action": action.reshape(-1).astype(float).tolist(),
                        "reward": to_jsonable(reward),
                        "terminated": to_jsonable(terminated),
                        "truncated": to_jsonable(truncated),
                        "peg_head_pos_at_hole": first_info_value(info, "peg_head_pos_at_hole"),
                        "hole_delta_from_start": hole_delta,
                        "target_motion_applied": bool(target_motion_applied),
                        "live_eval": live_eval,
                    }
                    action_trace.append(row)

                    if hole_delta >= trace_args.motion_detect_threshold:
                        target_motion_trigger_frame = step_idx
                        before_oracle = live_eval
                        before_peg_pose = np.asarray(before_oracle["peg_pose"], dtype=np.float64).reshape(-1)[:3]
                        # One extra action-driven step lets RecordEpisode capture the
                        # Oracle decision moment without any peg state edit.
                        obs, reward, terminated, truncated, info = envs.step(action)
                        after_oracle = eval_state(base_env)
                        after_peg_pose = np.asarray(after_oracle["peg_pose"], dtype=np.float64).reshape(-1)[:3]
                        oracle_jump_distance = float(np.linalg.norm(after_peg_pose - before_peg_pose))
                        action_trace.append(
                            {
                                "env_step": len(action_trace),
                                "chunk_idx": chunk_idx,
                                "action_idx": action_idx,
                                "action": action.reshape(-1).astype(float).tolist(),
                                "oracle_decision_record_step": True,
                                "oracle_peg_state_intervention_used": False,
                                "reward": to_jsonable(reward),
                                "terminated": to_jsonable(terminated),
                                "truncated": to_jsonable(truncated),
                                "peg_head_pos_at_hole": first_info_value(info, "peg_head_pos_at_hole"),
                                "live_eval": after_oracle,
                            }
                        )
                        raise StopIteration

                    done = bool(np.asarray(terminated).reshape(-1)[0]) or bool(
                        np.asarray(truncated).reshape(-1)[0]
                    )
                    if done:
                        raise StopIteration
    except StopIteration:
        pass
    finally:
        envs.close()

    videos = [str(path) for path in sorted(video_dir.glob("*.mp4"))]
    snap_detected = (
        oracle_jump_distance is not None
        and oracle_jump_distance > float(trace_args.max_oracle_step_displacement)
    )
    summary = {
        "schema": "phase03_oracle_boundary_trace_v2_no_peg_teleport",
        "phase": "03_oracle",
        "evidence_type": "oracle_boundary_decision_diagnostic_no_peg_teleport_with_recordepisode_video",
        "method_evidence_allowed": False,
        "physical_insertion_success_claimed": False,
        "target_motion_state_intervention_used": bool(target_motion_applied),
        "oracle_state_intervention_used": False,
        "oracle_peg_state_intervention_used": False,
        "oracle_set_pose_used": False,
        "snap_detected_by_displacement_threshold": bool(snap_detected),
        "max_oracle_step_displacement": float(trace_args.max_oracle_step_displacement),
        "ckpt_path": trace_args.ckpt_path,
        "state_key": state_key,
        "output_dir": str(output_dir),
        "video_dir": str(video_dir),
        "videos": videos,
        "phase02_run": str(phase02),
        "cosmos_video": str(cosmos_video),
        "seed": trace_args.seed,
        "target_motion_trigger_frame": target_motion_trigger_frame,
        "dp_prefix_steps": target_motion_trigger_frame + 1 if target_motion_trigger_frame is not None else len(action_trace),
        "initial_eval": initial_eval,
        "before_oracle": before_oracle,
        "after_oracle": after_oracle,
        "oracle_jump_distance": oracle_jump_distance,
        "classification": (
            "invalid_oracle_snap_detected"
            if snap_detected
            else "oracle_boundary_no_teleport_video_diagnostic_complete_not_method_success"
            if videos and after_oracle is not None
            else "oracle_boundary_video_missing_or_incomplete"
        ),
        "notes": [
            "DP prefix uses active DP checkpoint and pd_ee_delta_pose actions.",
            "Target motion is a controlled diagnostic perturbation.",
            "Oracle records a boundary / decision point only and does not call peg.set_pose.",
            "Do not report this as physical insertion success or method evidence.",
        ],
    }
    (output_dir / "action_trace.json").write_text(json.dumps(to_jsonable(action_trace), indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")
    (output_dir / "classification.txt").write_text(
        "\n".join(
            [
                f"phase03_status={summary['classification']}",
                "method_evidence_allowed=false",
                "physical_insertion_success=false",
                "oracle_state_intervention_used=false",
                "oracle_peg_state_intervention_used=false",
                "oracle_set_pose_used=false",
                f"snap_detected={str(bool(snap_detected)).lower()}",
                f"target_motion_trigger_frame={target_motion_trigger_frame}",
                f"video_count={len(videos)}",
            ]
        )
        + "\n"
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
