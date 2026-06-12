#!/usr/bin/env python3
"""Render saved ManiSkill env_states without stepping the trajectory."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import tyro

import mani_skill.envs  # noqa: F401
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import io_utils


@dataclass
class Args:
    traj_path: str
    output_dir: str
    episode_index: int = 0
    frame_indices: str = ""
    stride: int = 10
    max_frames: int = 80
    sim_backend: Optional[str] = None
    render_mode: str = "rgb_array"
    shader: Optional[str] = None
    make_video: bool = True
    video_fps: int = 30
    thumb_width: int = 240


def _parse_indices(raw: str, n: int, stride: int, max_frames: int) -> list[int]:
    raw = raw.strip()
    if raw:
        out = []
        for part in raw.split(","):
            part = part.strip()
            if part:
                out.append(min(max(0, int(part)), n - 1))
        return sorted(set(out))
    if n <= 0:
        return []
    indices = list(range(0, n, max(1, int(stride))))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    if max_frames > 0 and len(indices) > max_frames:
        keep = np.linspace(0, len(indices) - 1, max_frames).round().astype(int)
        indices = [indices[i] for i in sorted(set(keep.tolist()))]
    return indices


def _make_contact_sheet(items: list[dict], sheet_path: Path, thumb_width: int):
    if not items:
        return
    thumbs = []
    for item in items:
        img = Image.open(item["output"]).convert("RGB")
        scale = thumb_width / img.width
        img = img.resize((thumb_width, max(1, int(img.height * scale))))
        thumbs.append((img, f"frame {item['frame']}"))
    label_h = 24
    cols = min(5, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    cell_h = thumbs[0][0].height + label_h
    sheet = Image.new("RGB", (cols * thumb_width, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (img, label) in enumerate(thumbs):
        x = (i % cols) * thumb_width
        y = (i // cols) * cell_h
        sheet.paste(img, (x, y))
        draw.text((x + 4, y + img.height + 4), label, fill=(0, 0, 0))
    sheet.save(sheet_path)


def main():
    args = tyro.cli(Args)
    traj_path = Path(args.traj_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_data = io_utils.load_json(str(traj_path).replace(".h5", ".json"))
    env_info = json_data["env_info"]
    episodes = json_data["episodes"]
    episode = episodes[int(args.episode_index)]
    traj_id = f"traj_{episode['episode_id']}"

    env_kwargs = dict(env_info["env_kwargs"])
    if args.sim_backend is not None:
        env_kwargs["sim_backend"] = args.sim_backend
    env_kwargs["render_mode"] = args.render_mode
    if args.shader is not None:
        env_kwargs["shader_dir"] = args.shader
    env = gym.make(env_info["env_id"], **env_kwargs)

    manifest = []
    try:
        reset_kwargs = dict(episode.get("reset_kwargs") or {})
        env.reset(**reset_kwargs)
        with h5py.File(traj_path, "r") as f:
            env_states = trajectory_utils.dict_to_list_of_dicts(
                f[traj_id]["env_states"]
            )
        indices = _parse_indices(
            args.frame_indices,
            len(env_states),
            args.stride,
            args.max_frames,
        )
        frames = []
        for idx in indices:
            env.unwrapped.set_state_dict(env_states[idx])
            frame = env.render()
            if isinstance(frame, list):
                frame = frame[0]
            frame = np.asarray(frame)
            out_path = output_dir / f"frame_{idx:05d}.png"
            imageio.imwrite(out_path, frame)
            frames.append(frame)
            manifest.append(
                {
                    "frame": int(idx),
                    "output": str(out_path),
                    "traj_path": str(traj_path),
                    "trajectory_id": traj_id,
                }
            )
        if args.make_video and frames:
            imageio.mimsave(output_dir / "state_replay.mp4", frames, fps=args.video_fps)
    finally:
        env.close()

    manifest_path = output_dir / "frames_manifest.jsonl"
    with manifest_path.open("w") as f:
        for item in manifest:
            f.write(json.dumps(item, sort_keys=True) + "\n")
    _make_contact_sheet(manifest[:50], output_dir / "contact_sheet.png", args.thumb_width)
    print(f"frames={len(manifest)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
