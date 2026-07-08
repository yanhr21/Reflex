#!/usr/bin/env python3
"""Trace DP state-policy rollouts with videos and per-step actions.

This is an evaluation/diagnostic harness only. It executes the checkpoint
through the original Diffusion Policy action path and never writes simulator
state back into the environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import tyro
from gymnasium.vector import SyncVectorEnv
from mani_skill.utils import common
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack, RecordEpisode

import train as ms_train


@dataclass
class TraceArgs:
    ckpt_path: str
    output_dir: str
    env_id: Optional[str] = None
    control_mode: Optional[str] = None
    sim_backend: Optional[str] = None
    max_episode_steps: Optional[int] = None
    num_episodes: int = 2
    seed: int = 1
    use_ema: bool = True
    cuda: bool = True
    stop_on_success: bool = True


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


def build_train_args(saved: dict, trace_args: TraceArgs) -> ms_train.Args:
    args = ms_train.Args()
    for key, value in saved.items():
        if hasattr(args, key):
            setattr(args, key, value)

    if trace_args.env_id is not None:
        args.env_id = trace_args.env_id
    if trace_args.control_mode is not None:
        args.control_mode = trace_args.control_mode
    if trace_args.sim_backend is not None:
        args.sim_backend = trace_args.sim_backend
    if trace_args.max_episode_steps is not None:
        args.max_episode_steps = trace_args.max_episode_steps
    args.num_eval_envs = 1
    args.num_eval_episodes = trace_args.num_episodes
    args.seed = trace_args.seed
    args.capture_video = True
    return args


def make_trace_env(args: ms_train.Args, video_dir: Path):
    assert args.sim_backend == "physx_cpu", "Trace harness is for CPU ManiSkill eval."

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
            source_type="diffusion_policy",
            source_desc="phase01 static DP physical rollout trace",
        )
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)
        return env

    return SyncVectorEnv([thunk])


def scalar_bool_from_info(info: dict, key: str) -> bool:
    if key not in info:
        return False
    arr = np.asarray(info[key])
    return bool(arr.reshape(-1)[0])


def first_info_value(info: dict, key: str) -> Any:
    if key not in info:
        return None
    value = to_jsonable(info[key])
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def main():
    trace_args = tyro.cli(TraceArgs)
    output_dir = Path(trace_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"

    device = torch.device("cuda" if torch.cuda.is_available() and trace_args.cuda else "cpu")
    ckpt = torch.load(trace_args.ckpt_path, map_location=device)
    args = build_train_args(ckpt.get("args", {}), trace_args)
    assert args.max_episode_steps is not None

    ms_train.args = args
    ms_train.device = device

    envs = make_trace_env(args, video_dir)
    agent = ms_train.Agent(envs, args).to(device)
    state_key = "ema_agent" if trace_args.use_ema else "agent"
    agent.load_state_dict(ckpt[state_key])
    agent.eval()

    episodes = []
    all_actions = []
    final_state = None

    with torch.no_grad():
        for episode_idx in range(trace_args.num_episodes):
            obs, reset_info = envs.reset(seed=trace_args.seed + episode_idx)
            step_rows = []
            success_once = False
            final_info = {}

            for chunk_idx in range(args.max_episode_steps):
                obs_tensor = common.to_tensor(obs, device)
                action_seq = agent.get_action(obs_tensor).detach().cpu().numpy()
                for action_idx in range(action_seq.shape[1]):
                    action = action_seq[:, action_idx]
                    obs, reward, terminated, truncated, info = envs.step(action)
                    success = scalar_bool_from_info(info, "success")
                    fail = scalar_bool_from_info(info, "fail")
                    success_once = success_once or success
                    row = {
                        "episode": episode_idx,
                        "env_step": len(step_rows),
                        "chunk_idx": chunk_idx,
                        "action_idx": action_idx,
                        "action": action.reshape(-1).astype(float).tolist(),
                        "reward": to_jsonable(reward),
                        "terminated": to_jsonable(terminated),
                        "truncated": to_jsonable(truncated),
                        "success": success,
                        "fail": fail,
                        "peg_head_pos_at_hole": first_info_value(info, "peg_head_pos_at_hole"),
                    }
                    step_rows.append(row)
                    all_actions.append(row["action"])
                    final_info = to_jsonable(info)

                    done = bool(np.asarray(terminated).reshape(-1)[0]) or bool(
                        np.asarray(truncated).reshape(-1)[0]
                    )
                    if done or (trace_args.stop_on_success and success):
                        break
                if done or (trace_args.stop_on_success and success_once):
                    break

            final_state = to_jsonable(envs.envs[0].unwrapped.get_state_dict())
            episode_summary = {
                "episode": episode_idx,
                "reset_seed": trace_args.seed + episode_idx,
                "reset_info": to_jsonable(reset_info),
                "num_steps": len(step_rows),
                "success_once": bool(success_once),
                "success_at_end": bool(step_rows[-1]["success"]) if step_rows else False,
                "stopped_on_success": bool(trace_args.stop_on_success and success_once),
                "final_info": final_info,
            }
            episodes.append(episode_summary)
            trace_path = output_dir / f"episode_{episode_idx:03d}_action_trace.json"
            with trace_path.open("w") as f:
                json.dump({"summary": episode_summary, "steps": step_rows}, f, indent=2)

    envs.close()

    actions = np.asarray(all_actions, dtype=np.float32) if all_actions else np.zeros((0, 0))
    action_stats = {
        "count": int(actions.shape[0]),
        "shape": list(actions.shape),
        "min": actions.min(axis=0).tolist() if actions.size else [],
        "max": actions.max(axis=0).tolist() if actions.size else [],
        "mean": actions.mean(axis=0).tolist() if actions.size else [],
        "std": actions.std(axis=0).tolist() if actions.size else [],
        "abs_max": np.abs(actions).max(axis=0).tolist() if actions.size else [],
    }

    result = {
        "phase": "01_dp_static",
        "evidence_type": "static_dp_physical_rollout_trace",
        "method_evidence_allowed": True,
        "ckpt_path": trace_args.ckpt_path,
        "state_key": state_key,
        "output_dir": str(output_dir),
        "video_dir": str(video_dir),
        "args": vars(args),
        "episodes": episodes,
        "mean_success_once": float(np.mean([ep["success_once"] for ep in episodes])) if episodes else 0.0,
        "mean_success_at_end": float(np.mean([ep["success_at_end"] for ep in episodes])) if episodes else 0.0,
        "action_stats": action_stats,
        "final_state_dict": final_state,
        "forbidden_state_intervention_used": False,
        "notes": [
            "No set_pose, set_state, set_state_dict, source-state restore, saved-state replay, or Oracle final seat is used by this harness.",
            "final_state_dict is read after rollout for audit only.",
        ],
    }
    with (output_dir / "trace_metrics.json").open("w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps({k: result[k] for k in ["mean_success_once", "mean_success_at_end", "video_dir"]}, indent=2))
    print(f"wrote {output_dir / 'trace_metrics.json'}")


if __name__ == "__main__":
    main()
