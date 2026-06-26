#!/usr/bin/env python3
"""Convert contact suffix windows to LeRobot with causal task-frame rel state.

This is an OpenPI data-adapter branch, not a custom action model.  It keeps the
same official pi0.5/LeRobot layout as the qpos8 contact-suffix dataset, but
sets observation/state to:

    qpos_first_7 + mean_finger_qpos + current peg_head_at_hole rel3

The rel3 term is the current observed/source slot at timestep t.  Future
inserted labels are used only to select positive training suffix windows.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
from tqdm import tqdm


DEFAULT_RENDER_MANIFEST = (
    "experiments/world_model_task_rebinding/cosmos3/"
    "sft_dataset_fix3_v7_user_override_733_rgb_512_20260612/manifest.json"
)
DEFAULT_REPO_ID = "yanhongru/maniskill_peg733_openpi_contact_suffix16_relstate11"
DEFAULT_TASK = "insert the grasped peg into the current target hole"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-manifest", default=DEFAULT_RENDER_MANIFEST)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output-manifest", default="")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--expected-episodes", type=int, default=733)
    parser.add_argument("--expected-frames", type=int, default=301)
    parser.add_argument("--expected-actions", type=int, default=300)
    parser.add_argument("--suffix-length", type=int, default=16)
    parser.add_argument("--offsets-before-insert", default="64,48,32,24,16,12,8,4")
    parser.add_argument("--max-source-episodes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-login-node", action="store_true")
    parser.add_argument("--task-prompt", default=DEFAULT_TASK)
    parser.add_argument("--duplicate-base-as-wrist", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def refuse_login_node(args: argparse.Namespace) -> None:
    if args.allow_login_node:
        return
    step_id = os.environ.get("SLURM_STEP_ID", "")
    if not os.environ.get("SLURM_JOB_ID") or step_id in {"", "extern"}:
        raise SystemExit(
            "refusing_login_node_execution=true\n"
            "reason=Run relstate LeRobot conversion only inside a compute-node srun step."
        )


def parse_offsets(text: str) -> list[int]:
    offsets = sorted({int(part.strip()) for part in text.split(",") if part.strip()}, reverse=True)
    if not offsets or any(x < 0 for x in offsets):
        raise ValueError(f"invalid offsets-before-insert: {text!r}")
    return offsets


def load_render_manifest(path: Path, expected_episodes: int) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    videos = data.get("videos")
    if not isinstance(videos, list):
        raise RuntimeError(f"{path} has no videos list")
    if expected_episodes and len(videos) != expected_episodes:
        raise RuntimeError(f"{path} expected {expected_episodes} videos, got {len(videos)}")
    return videos


def read_video_frames(path: Path, expected_frames: int) -> list[np.ndarray]:
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


def resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return np.asarray(image, dtype=np.uint8)
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    return np.asarray(pil.resize((width, height), resample=Image.BICUBIC), dtype=np.uint8)


def load_h5_arrays(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        traj_names = sorted([k for k in h5.keys() if k.startswith("traj_")], key=lambda x: int(x.split("_", 1)[1]))
        if len(traj_names) != 1:
            raise RuntimeError(f"{path} expected one trajectory, found {traj_names}")
        group = h5[traj_names[0]]
        slots = group["slots"]
        return {
            "actions": np.asarray(group["actions"], dtype=np.float32),
            "qpos": np.asarray(slots["qpos"], dtype=np.float32),
            "inserted": np.asarray(slots["inserted"], dtype=bool),
            "grasped": np.asarray(slots["grasped"], dtype=bool),
            "peg_head_at_hole": np.asarray(slots["peg_head_at_hole"], dtype=np.float32),
        }


def qpos8(qpos: np.ndarray, t: int) -> np.ndarray:
    row = np.asarray(qpos[min(t, qpos.shape[0] - 1)], dtype=np.float32).reshape(-1)
    state = np.zeros((8,), dtype=np.float32)
    state[: min(7, row.shape[0])] = row[:7]
    if row.shape[0] >= 9:
        state[7] = float(np.mean(row[7:9]))
    elif row.shape[0] >= 8:
        state[7] = float(row[7])
    return state


def relstate11(qpos: np.ndarray, rel: np.ndarray, t: int) -> np.ndarray:
    row_index = min(int(t), rel.shape[0] - 1)
    rel3 = np.asarray(rel[row_index, :3], dtype=np.float32).reshape(3)
    return np.concatenate([qpos8(qpos, t), rel3], axis=0).astype(np.float32)


def action7(actions: np.ndarray, t: int) -> np.ndarray:
    row = np.asarray(actions[t], dtype=np.float32).reshape(-1)
    out = np.zeros((7,), dtype=np.float32)
    out[: min(7, row.shape[0])] = row[:7]
    return out


def window_starts(first_insert: int, action_count: int, suffix_length: int, offsets: list[int]) -> list[int]:
    starts = []
    for offset in offsets:
        start = int(first_insert) - int(offset)
        if start >= 0 and start + int(suffix_length) <= int(action_count):
            starts.append(start)
    return sorted(set(starts))


def main() -> int:
    args = parse_args()
    refuse_login_node(args)
    offsets = parse_offsets(args.offsets_before_insert)

    render_manifest = Path(args.render_manifest).resolve()
    videos = load_render_manifest(render_manifest, int(args.expected_episodes))
    if int(args.max_source_episodes) > 0:
        videos = videos[: int(args.max_source_episodes)]

    output_path = HF_LEROBOT_HOME / str(args.repo_id)
    if output_path.exists():
        if not args.overwrite:
            raise SystemExit(f"output dataset exists; pass --overwrite to replace: {output_path}")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=str(args.repo_id),
        robot_type="panda",
        fps=int(args.fps),
        features={
            "image": {
                "dtype": "image",
                "shape": (int(args.image_height), int(args.image_width), 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (int(args.image_height), int(args.image_width), 3),
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": (11,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    records: list[dict[str, Any]] = []
    scenario_counts: dict[str, int] = {}
    scenario_window_counts: dict[str, int] = {}
    skipped_no_insert = 0
    skipped_no_window = 0

    for item in tqdm(videos, desc="Converting relstate11 contact suffix windows"):
        video_path = Path(item["video"]).resolve()
        h5_path = Path(item["input_h5"]).resolve()
        scenario = str(item.get("scenario", "unknown"))
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1

        arrays = load_h5_arrays(h5_path)
        actions = arrays["actions"]
        qpos = arrays["qpos"]
        inserted = arrays["inserted"]
        grasped = arrays["grasped"]
        rel = arrays["peg_head_at_hole"]
        if actions.shape[0] != int(args.expected_actions):
            raise RuntimeError(f"{h5_path} expected {args.expected_actions} actions, got {actions.shape}")
        if qpos.shape[0] != int(args.expected_frames):
            raise RuntimeError(f"{h5_path} expected {args.expected_frames} qpos frames, got {qpos.shape}")
        if rel.shape[0] != int(args.expected_frames):
            raise RuntimeError(f"{h5_path} expected {args.expected_frames} rel frames, got {rel.shape}")
        if inserted.shape[0] != int(args.expected_frames) or grasped.shape[0] != int(args.expected_frames):
            raise RuntimeError(f"{h5_path} has inconsistent inserted/grasped frame counts")

        first_insert = int(np.flatnonzero(inserted)[0]) if bool(np.any(inserted)) else -1
        if first_insert < 0:
            skipped_no_insert += 1
            continue
        starts = window_starts(first_insert, actions.shape[0], int(args.suffix_length), offsets)
        if not starts:
            skipped_no_window += 1
            continue

        frames = read_video_frames(video_path, int(args.expected_frames))
        for start in starts:
            steps_before = int(first_insert - start)
            task = f"{args.task_prompt}; scenario {scenario}; contact suffix starts {steps_before} steps before first insertion"
            for t in range(start, start + int(args.suffix_length)):
                image = resize_rgb(frames[t], int(args.image_width), int(args.image_height))
                wrist = image.copy() if bool(args.duplicate_base_as_wrist) else np.zeros_like(image)
                dataset.add_frame(
                    {
                        "image": image,
                        "wrist_image": wrist,
                        "state": relstate11(qpos, rel, t),
                        "actions": action7(actions, t),
                        "task": task,
                    }
                )
            dataset.save_episode()
            scenario_window_counts[scenario] = scenario_window_counts.get(scenario, 0) + 1
            end_index = min(start + int(args.suffix_length), rel.shape[0] - 1)
            records.append(
                {
                    "source_sample_id": item.get("sample_id"),
                    "scenario": scenario,
                    "split": item.get("split"),
                    "video": str(video_path),
                    "input_h5": str(h5_path),
                    "first_insert_frame": int(first_insert),
                    "suffix_start_frame": int(start),
                    "suffix_end_action_exclusive": int(start + int(args.suffix_length)),
                    "steps_before_first_insert": steps_before,
                    "start_grasped": bool(grasped[start]),
                    "end_grasped": bool(grasped[end_index]),
                    "end_inserted": bool(inserted[end_index]),
                    "start_peg_head_at_hole": rel[start, :3].astype(float).tolist(),
                    "end_peg_head_at_hole": rel[end_index, :3].astype(float).tolist(),
                    "action_semantics": "source_h5_first_7_pd_ee_delta_pose_values",
                    "state_semantics": "qpos_first_7_plus_mean_finger_qpos_plus_current_peg_head_at_hole_rel3",
                    "state_dim": 11,
                    "wrist_image_policy": "duplicate_base_image" if bool(args.duplicate_base_as_wrist) else "zeros",
                }
            )

    manifest = {
        "schema": "openpi_lerobot_maniskill_peg733_contact_suffix16_relstate11_conversion_v1",
        "repo_id": str(args.repo_id),
        "output_path": str(output_path),
        "render_manifest": str(render_manifest),
        "num_source_episodes_read": len(videos),
        "num_suffix_episodes_written": len(records),
        "frames_per_suffix_episode": int(args.suffix_length),
        "total_frames_written": len(records) * int(args.suffix_length),
        "source_frames_per_episode": int(args.expected_frames),
        "source_actions_per_episode": int(args.expected_actions),
        "offsets_before_insert": offsets,
        "skipped_no_insert": skipped_no_insert,
        "skipped_no_window": skipped_no_window,
        "fps": int(args.fps),
        "image_shape": [int(args.image_height), int(args.image_width), 3],
        "features": {"image": [int(args.image_height), int(args.image_width), 3], "wrist_image": [int(args.image_height), int(args.image_width), 3], "state": [11], "actions": [7]},
        "scenario_source_counts": scenario_counts,
        "scenario_suffix_counts": scenario_window_counts,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "boundary": (
            "Official OpenPI/LeRobot data conversion from accepted 733 H5/RGB data. "
            "The additional rel3 state is current-timestep causal task-frame geometry "
            "from source slots, used to diagnose missing object/task-frame conditioning. "
            "Future inserted labels select positive suffix windows only; no custom action "
            "model, scorer-only selector, VAE, MLP, or diffusion substitute is introduced."
        ),
        "args": vars(args),
        "records": records,
    }
    out_manifest = Path(args.output_manifest).resolve() if str(args.output_manifest).strip() else output_path / "maniskill_peg733_contact_suffix16_relstate11_manifest.json"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "repo_id": str(args.repo_id),
                "num_source_episodes_read": len(videos),
                "num_suffix_episodes_written": len(records),
                "total_frames_written": len(records) * int(args.suffix_length),
                "skipped_no_insert": skipped_no_insert,
                "skipped_no_window": skipped_no_window,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
