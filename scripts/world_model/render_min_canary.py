#!/usr/bin/env python3
"""Minimal ManiSkill render canary with stage logs."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import time
from typing import Any

os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ["DISPLAY"] = ""

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import tyro

import mani_skill.envs  # noqa: F401
from mani_skill.utils import common, sapien_utils


@dataclass
class Args:
    output_dir: str
    env_id: str = "PegInsertionSide-v1"
    obs_mode: str = "state"
    control_mode: str = "pd_ee_delta_pose"
    sim_backend: str = "physx_cpu"
    width: int = 256
    height: int = 256
    shader_pack: str = "minimal"
    render_api: str = "gym"
    camera_eye: tuple[float, float, float] = (0.5, -0.5, 0.8)
    camera_target: tuple[float, float, float] = (0.05, -0.1, 0.4)
    camera_up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    camera_fov: float = 1.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _log(**payload: Any) -> None:
    payload.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    print(json.dumps(_jsonable(payload), sort_keys=True), flush=True)


def main() -> None:
    args = tyro.cli(Args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(
        event="python_start",
        args=asdict(args),
        vk=os.environ.get("VK_ICD_FILENAMES"),
        display=os.environ.get("DISPLAY"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    try:
        import sapien.render as sr

        _log(event="sapien_device_summary", summary=sr.get_device_summary())
    except Exception as exc:  # noqa: BLE001
        _log(event="sapien_device_summary_failed", error=repr(exc))

    camera_pose = sapien_utils.look_at(args.camera_eye, args.camera_target, args.camera_up)
    pose = camera_pose.raw_pose.detach().cpu().numpy().reshape(-1, 7)[0].tolist()
    camera_config = {
        "render_camera": {
            "pose": pose,
            "width": int(args.width),
            "height": int(args.height),
            "fov": float(args.camera_fov),
            "near": 0.01,
            "far": 100,
            "shader_pack": args.shader_pack,
        }
    }
    env_kwargs = {
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "reward_mode": "sparse",
        "render_mode": "rgb_array",
        "sim_backend": args.sim_backend,
        "human_render_camera_configs": camera_config,
    }
    _log(event="gym_make_start", env_kwargs=env_kwargs)
    env = gym.make(args.env_id, **env_kwargs)
    _log(event="gym_make_done")
    try:
        _log(event="reset_start")
        env.reset(seed=[2022], options={"reconfigure": True})
        _log(event="reset_done")
        if args.render_api == "gym":
            _log(event="render_gym_start")
            frame = common.to_numpy(env.render())
        elif args.render_api == "render_rgb_array":
            _log(event="render_rgb_array_start")
            frame = common.to_numpy(env.unwrapped.render_rgb_array("render_camera"))
        else:
            raise ValueError(f"unsupported_render_api={args.render_api}")
        frame = np.asarray(frame)
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]
        frame = np.clip(frame[..., :3], 0, 255).astype(np.uint8)
        _log(event="render_done", render_api=args.render_api, shape=list(frame.shape), mean=float(frame.mean()))
        out_path = output_dir / "frame.png"
        imageio.imwrite(out_path, frame)
        _log(event="write_done", output=out_path, bytes=out_path.stat().st_size)
    finally:
        env.close()
        _log(event="env_closed")


if __name__ == "__main__":
    main()
