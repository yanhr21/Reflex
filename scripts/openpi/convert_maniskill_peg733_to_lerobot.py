#!/usr/bin/env python3
"""Convert the accepted 733 ManiSkill peg-insertion episodes to LeRobot.

This script intentionally creates a Libero-style OpenPI dataset:

- image: approved ManiSkill default human-render RGB frame.
- wrist_image: duplicate of image, because the 733 export has no real wrist
  camera. This is recorded in the conversion manifest.
- state: first seven qpos values plus one gripper scalar.
- actions: seven-dimensional pd_ee_delta_pose action from the source H5.
- task: language instruction for OpenPI prompt_from_task.

Run only inside a tmux-held Slurm allocation. The script refuses login-node
execution by default because video decoding and H5 conversion are project
compute for this repository.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro


DEFAULT_RENDER_MANIFEST = (
    "experiments/world_model_task_rebinding/cosmos3/"
    "sft_dataset_fix3_v7_user_override_733_rgb_512_20260612/manifest.json"
)
DEFAULT_REPO_ID = "yanhongru/maniskill_peg733_openpi_libero"
DEFAULT_TASK = "insert the peg into the current target hole"


@dataclass(frozen=True)
class Args:
    render_manifest: str = DEFAULT_RENDER_MANIFEST
    repo_id: str = DEFAULT_REPO_ID
    output_manifest: str | None = None
    fps: int = 30
    image_height: int = 256
    image_width: int = 256
    expected_episodes: int = 733
    expected_frames: int = 301
    expected_actions: int = 300
    max_episodes: int = 0
    overwrite: bool = False
    allow_login_node: bool = False
    task_prompt: str = DEFAULT_TASK
    duplicate_base_as_wrist: bool = True


def _refuse_login_node(args: Args) -> None:
    if args.allow_login_node:
        return
    step_id = os.environ.get("SLURM_STEP_ID", "")
    if not os.environ.get("SLURM_JOB_ID") or step_id in {"", "extern"}:
        raise SystemExit(
            "refusing_login_node_execution=true\n"
            "reason=Run 733-to-LeRobot conversion only inside a compute-node srun step."
        )


def _resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[0] == height and image.shape[1] == width:
        return np.asarray(image, dtype=np.uint8)
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    return np.asarray(pil.resize((width, height), resample=Image.BICUBIC), dtype=np.uint8)


def _read_video_frames(path: Path, expected_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    frames: list[np.ndarray] = []
    try:
        while len(frames) < expected_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(frame_bgr[..., ::-1].copy())
    finally:
        cap.release()
    if len(frames) != expected_frames:
        raise RuntimeError(f"{path} expected {expected_frames} frames, got {len(frames)}")
    return frames


def _load_h5_arrays(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        traj_names = sorted([k for k in h5.keys() if k.startswith("traj_")], key=lambda x: int(x.split("_", 1)[1]))
        if len(traj_names) != 1:
            raise RuntimeError(f"{path} expected one trajectory, found {traj_names}")
        group = h5[traj_names[0]]
        slots = group["slots"]
        arrays = {
            "actions": np.asarray(group["actions"], dtype=np.float32),
            "qpos": np.asarray(slots["qpos"], dtype=np.float32) if "qpos" in slots else np.zeros((301, 9), np.float32),
        }
    return arrays


def _state_from_qpos(qpos: np.ndarray, t: int) -> np.ndarray:
    row = np.asarray(qpos[min(t, qpos.shape[0] - 1)], dtype=np.float32).reshape(-1)
    state = np.zeros((8,), dtype=np.float32)
    state[: min(7, row.shape[0])] = row[:7]
    if row.shape[0] >= 9:
        state[7] = float(np.mean(row[7:9]))
    elif row.shape[0] >= 8:
        state[7] = float(row[7])
    return state


def _action_from_source(actions: np.ndarray, t: int) -> np.ndarray:
    if t >= actions.shape[0]:
        raise RuntimeError(f"action index {t} out of range for shape {actions.shape}")
    row = np.asarray(actions[t], dtype=np.float32).reshape(-1)
    out = np.zeros((7,), dtype=np.float32)
    out[: min(7, row.shape[0])] = row[:7]
    return out


def _load_render_manifest(path: Path, expected_episodes: int) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    videos = data.get("videos")
    if not isinstance(videos, list):
        raise RuntimeError(f"{path} has no videos list")
    if expected_episodes and len(videos) != expected_episodes:
        raise RuntimeError(f"{path} expected {expected_episodes} videos, got {len(videos)}")
    return videos


def main(args: Args) -> None:
    _refuse_login_node(args)

    render_manifest = Path(args.render_manifest).resolve()
    videos = _load_render_manifest(render_manifest, args.expected_episodes)
    if args.max_episodes > 0:
        videos = videos[: args.max_episodes]

    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        if not args.overwrite:
            raise SystemExit(f"output dataset exists; pass --overwrite to replace: {output_path}")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="panda",
        fps=args.fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (args.image_height, args.image_width, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (args.image_height, args.image_width, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    records: list[dict[str, Any]] = []
    scenario_counts: dict[str, int] = {}
    for item in tqdm(videos, desc="Converting 733 ManiSkill episodes"):
        video_path = Path(item["video"]).resolve()
        h5_path = Path(item["input_h5"]).resolve()
        scenario = str(item.get("scenario", "unknown"))
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1

        frames = _read_video_frames(video_path, args.expected_frames)
        arrays = _load_h5_arrays(h5_path)
        actions = arrays["actions"]
        qpos = arrays["qpos"]
        if actions.shape[0] != args.expected_actions:
            raise RuntimeError(f"{h5_path} expected {args.expected_actions} actions, got {actions.shape}")
        if qpos.shape[0] != args.expected_frames:
            raise RuntimeError(f"{h5_path} expected {args.expected_frames} qpos frames, got {qpos.shape}")

        task = f"{args.task_prompt}; scenario {scenario}"
        # LeRobot actions are aligned to observations at t, so use frames
        # 0..299 and leave frame 300 as episode-end provenance in the manifest.
        for t in range(args.expected_actions):
            image = _resize_rgb(frames[t], args.image_width, args.image_height)
            wrist = image.copy() if args.duplicate_base_as_wrist else np.zeros_like(image)
            dataset.add_frame(
                {
                    "image": image,
                    "wrist_image": wrist,
                    "state": _state_from_qpos(qpos, t),
                    "actions": _action_from_source(actions, t),
                    "task": task,
                }
            )
        dataset.save_episode()
        records.append(
            {
                "sample_id": item.get("sample_id"),
                "scenario": scenario,
                "split": item.get("split"),
                "video": str(video_path),
                "input_h5": str(h5_path),
                "source_video_frames": args.expected_frames,
                "converted_observation_action_frames": args.expected_actions,
                "action_semantics": "source_h5_first_7_pd_ee_delta_pose_values",
                "state_semantics": "qpos_first_7_plus_mean_finger_qpos",
                "wrist_image_policy": "duplicate_base_image" if args.duplicate_base_as_wrist else "zeros",
            }
        )

    manifest = {
        "schema": "openpi_lerobot_maniskill_peg733_conversion_v1",
        "repo_id": args.repo_id,
        "output_path": str(output_path),
        "render_manifest": str(render_manifest),
        "num_episodes": len(records),
        "frames_per_episode_written": args.expected_actions,
        "source_frames_per_episode": args.expected_frames,
        "fps": args.fps,
        "image_shape": [args.image_height, args.image_width, 3],
        "features": {
            "image": [args.image_height, args.image_width, 3],
            "wrist_image": [args.image_height, args.image_width, 3],
            "state": [8],
            "actions": [7],
        },
        "scenario_counts": scenario_counts,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "boundary": (
            "Official OpenPI/LeRobot data conversion only. No custom action model, "
            "VAE, MLP, or non-OpenPI weight path is introduced."
        ),
        "args": asdict(args),
        "records": records,
    }
    out_manifest = Path(args.output_manifest).resolve() if args.output_manifest else output_path / "maniskill_peg733_manifest.json"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    tyro.cli(main)
