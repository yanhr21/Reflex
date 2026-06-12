#!/usr/bin/env python3
"""Official-metric eval for ManiSkill state Diffusion Policy checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro

import train as ms_train
from diffusion_policy.make_env import make_eval_envs
from dp_eval_compat import evaluate


@dataclass
class EvalArgs:
    ckpt_path: str
    output_dir: str
    env_id: Optional[str] = None
    control_mode: Optional[str] = None
    sim_backend: Optional[str] = None
    max_episode_steps: Optional[int] = None
    num_eval_episodes: int = 100
    num_eval_envs: int = 10
    seed: int = 1
    use_ema: bool = True
    capture_video: bool = False
    video_name: Optional[str] = None
    cuda: bool = True


def build_train_args(saved: dict, eval_args: EvalArgs) -> ms_train.Args:
    args = ms_train.Args()
    for key, value in saved.items():
        if hasattr(args, key):
            setattr(args, key, value)

    if eval_args.env_id is not None:
        args.env_id = eval_args.env_id
    if eval_args.control_mode is not None:
        args.control_mode = eval_args.control_mode
    if eval_args.sim_backend is not None:
        args.sim_backend = eval_args.sim_backend
    if eval_args.max_episode_steps is not None:
        args.max_episode_steps = eval_args.max_episode_steps
    args.num_eval_episodes = eval_args.num_eval_episodes
    args.num_eval_envs = eval_args.num_eval_envs
    args.seed = eval_args.seed
    args.capture_video = eval_args.capture_video
    return args


def main():
    eval_args = tyro.cli(EvalArgs)
    output_dir = Path(eval_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and eval_args.cuda else "cpu")
    ckpt = torch.load(eval_args.ckpt_path, map_location=device)
    args = build_train_args(ckpt.get("args", {}), eval_args)
    assert args.max_episode_steps is not None

    ms_train.args = args
    ms_train.device = device

    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode="state",
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default"),
        max_episode_steps=args.max_episode_steps,
    )
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    video_dir = None
    if eval_args.capture_video:
        video_dir = output_dir / (eval_args.video_name or "videos")
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=str(video_dir) if video_dir is not None else None,
    )

    agent = ms_train.Agent(envs, args).to(device)
    state_key = "ema_agent" if eval_args.use_ema else "agent"
    agent.load_state_dict(ckpt[state_key])
    metrics = evaluate(args.num_eval_episodes, agent, envs, device, args.sim_backend)
    envs.close()

    raw_metrics = {key: np.asarray(value).tolist() for key, value in metrics.items()}
    mean_metrics = {key: float(np.mean(value)) for key, value in metrics.items()}
    result = {
        "ckpt_path": eval_args.ckpt_path,
        "state_key": state_key,
        "args": vars(args),
        "mean_metrics": mean_metrics,
        "raw_metrics": raw_metrics,
        "video_dir": str(video_dir) if video_dir is not None else None,
    }
    out_path = output_dir / "metrics.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(mean_metrics, indent=2, sort_keys=True))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
