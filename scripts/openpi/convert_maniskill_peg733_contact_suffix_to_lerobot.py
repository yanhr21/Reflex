#!/usr/bin/env python3
"""Convert insertion/contact suffix windows from the accepted 733 data to LeRobot.

This creates an OpenPI-native dataset for the current contact-action reset:
short causal observations immediately before insertion/contact, with the
recorded successful suffix actions as targets. It uses the same official
LeRobot field layout as the full-episode pi0.5 dataset and introduces no
custom model or intermediate action representation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import cv2
import h5py
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset_module
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.video_utils import encode_video_frames as lerobot_encode_video_frames
import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro


DEFAULT_RENDER_MANIFEST = (
    "experiments/world_model_task_rebinding/cosmos3/"
    "sft_dataset_fix3_v7_user_override_733_rgb_512_20260612/manifest.json"
)
DEFAULT_REPO_ID = "yanhongru/maniskill_peg733_openpi_contact_suffix16"
DEFAULT_TASK = "insert the grasped peg into the current target hole"
STATE_MODE_QPOS8 = "qpos8"
STATE_MODE_OBJECT17 = "object_state17"


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
    suffix_length: int = 16
    offsets_before_insert: str = "64,48,32,24,16,12,8,4"
    max_source_episodes: int = 0
    overwrite: bool = False
    allow_login_node: bool = False
    task_prompt: str = DEFAULT_TASK
    duplicate_base_as_wrist: bool = True
    state_mode: str = STATE_MODE_QPOS8
    camera_storage: str = "image"
    video_codec: str = "libsvtav1"


def _refuse_login_node(args: Args) -> None:
    if args.allow_login_node:
        return
    step_id = os.environ.get("SLURM_STEP_ID", "")
    if not os.environ.get("SLURM_JOB_ID") or step_id in {"", "extern"}:
        raise SystemExit(
            "refusing_login_node_execution=true\n"
            "reason=Run contact-suffix LeRobot conversion only inside a compute-node srun step."
        )


def _parse_offsets(text: str) -> list[int]:
    offsets = sorted({int(part.strip()) for part in text.split(",") if part.strip()}, reverse=True)
    if not offsets:
        raise ValueError("offsets_before_insert must contain at least one integer offset")
    if any(offset < 0 for offset in offsets):
        raise ValueError(f"offsets_before_insert must be nonnegative: {offsets}")
    return offsets


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
        return {
            "actions": np.asarray(group["actions"], dtype=np.float32),
            "qpos": np.asarray(slots["qpos"], dtype=np.float32),
            "qvel": np.asarray(slots["qvel"], dtype=np.float32),
            "tcp_pose": np.asarray(slots["tcp_pose"], dtype=np.float32),
            "peg_pose": np.asarray(slots["peg_pose"], dtype=np.float32),
            "hole_pose": np.asarray(slots["hole_pose"], dtype=np.float32),
            "hole_velocity_step": np.asarray(slots["hole_velocity_step"], dtype=np.float32),
            "inserted": np.asarray(slots["inserted"], dtype=bool),
            "grasped": np.asarray(slots["grasped"], dtype=bool),
            "peg_head_at_hole": np.asarray(slots["peg_head_at_hole"], dtype=np.float32),
        }


def _state_from_qpos(qpos: np.ndarray, t: int) -> np.ndarray:
    row = np.asarray(qpos[min(t, qpos.shape[0] - 1)], dtype=np.float32).reshape(-1)
    state = np.zeros((8,), dtype=np.float32)
    state[: min(7, row.shape[0])] = row[:7]
    if row.shape[0] >= 9:
        state[7] = float(np.mean(row[7:9]))
    elif row.shape[0] >= 8:
        state[7] = float(row[7])
    return state


def _pad_first(row: np.ndarray, dim: int) -> np.ndarray:
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    out = np.zeros((dim,), dtype=np.float32)
    out[: min(dim, row.shape[0])] = row[:dim]
    return out


def _object_state17_from_arrays(arrays: dict[str, np.ndarray], t: int) -> np.ndarray:
    frame = min(int(t), int(arrays["qpos"].shape[0]) - 1)
    state = np.concatenate(
        [
            _pad_first(arrays["tcp_pose"][frame], 3),
            _pad_first(arrays["peg_pose"][frame], 3),
            _pad_first(arrays["hole_pose"][frame], 3),
            _pad_first(arrays["peg_head_at_hole"][frame], 3),
            _pad_first(arrays["hole_velocity_step"][frame], 3),
            np.asarray(
                [
                    float(bool(arrays["grasped"][frame])),
                    float(bool(arrays["inserted"][frame])),
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)
    if state.shape != (17,):
        raise RuntimeError(f"object_state17 has invalid shape {state.shape}")
    return state


def _state_from_arrays(arrays: dict[str, np.ndarray], t: int, state_mode: str) -> np.ndarray:
    if state_mode == STATE_MODE_QPOS8:
        return _state_from_qpos(arrays["qpos"], t)
    if state_mode == STATE_MODE_OBJECT17:
        return _object_state17_from_arrays(arrays, t)
    raise ValueError(f"unsupported state_mode={state_mode!r}; expected {STATE_MODE_QPOS8!r} or {STATE_MODE_OBJECT17!r}")


def _action_from_source(actions: np.ndarray, t: int) -> np.ndarray:
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


def _configure_video_codec(video_codec: str) -> None:
    if video_codec not in {"libsvtav1", "h264", "hevc"}:
        raise ValueError("video_codec must be one of: libsvtav1, h264, hevc")
    if video_codec == "libsvtav1":
        return

    def encode_video_frames_with_codec(imgs_dir: Path, video_path: Path, fps: int, overwrite: bool = False) -> None:
        lerobot_encode_video_frames(
            imgs_dir,
            video_path,
            fps,
            vcodec=video_codec,
            pix_fmt="yuv420p",
            g=2,
            crf=30,
            overwrite=overwrite,
        )

    lerobot_dataset_module.encode_video_frames = encode_video_frames_with_codec


def _window_starts(first_insert: int, action_count: int, suffix_length: int, offsets: list[int]) -> list[int]:
    starts: list[int] = []
    for offset in offsets:
        start = int(first_insert) - int(offset)
        if start < 0:
            continue
        if start + int(suffix_length) > action_count:
            continue
        starts.append(start)
    return sorted(set(starts))


def main(args: Args) -> None:
    _refuse_login_node(args)
    offsets = _parse_offsets(args.offsets_before_insert)
    if args.state_mode not in {STATE_MODE_QPOS8, STATE_MODE_OBJECT17}:
        raise ValueError(
            f"unsupported state_mode={args.state_mode!r}; expected {STATE_MODE_QPOS8!r} or {STATE_MODE_OBJECT17!r}"
        )
    if args.camera_storage not in {"image", "video"}:
        raise ValueError("camera_storage must be 'image' or 'video'")
    if args.camera_storage == "video":
        _configure_video_codec(args.video_codec)
    state_dim = 8 if args.state_mode == STATE_MODE_QPOS8 else 17
    camera_dtype = str(args.camera_storage)

    render_manifest = Path(args.render_manifest).resolve()
    videos = _load_render_manifest(render_manifest, args.expected_episodes)
    if args.max_source_episodes > 0:
        videos = videos[: args.max_source_episodes]

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
                "dtype": camera_dtype,
                "shape": (args.image_height, args.image_width, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": camera_dtype,
                "shape": (args.image_height, args.image_width, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (state_dim,),
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
    scenario_window_counts: dict[str, int] = {}
    skipped_no_insert = 0
    skipped_no_window = 0
    for item in tqdm(videos, desc="Converting contact/insertion suffix windows"):
        video_path = Path(item["video"]).resolve()
        h5_path = Path(item["input_h5"]).resolve()
        scenario = str(item.get("scenario", "unknown"))
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1

        arrays = _load_h5_arrays(h5_path)
        actions = arrays["actions"]
        qpos = arrays["qpos"]
        inserted = arrays["inserted"]
        grasped = arrays["grasped"]
        rel = arrays["peg_head_at_hole"]
        if actions.shape[0] != args.expected_actions:
            raise RuntimeError(f"{h5_path} expected {args.expected_actions} actions, got {actions.shape}")
        if qpos.shape[0] != args.expected_frames:
            raise RuntimeError(f"{h5_path} expected {args.expected_frames} qpos frames, got {qpos.shape}")
        if inserted.shape[0] != args.expected_frames or grasped.shape[0] != args.expected_frames:
            raise RuntimeError(f"{h5_path} has inconsistent inserted/grasped frame counts")
        first_insert = int(np.flatnonzero(inserted)[0]) if bool(np.any(inserted)) else -1
        if first_insert < 0:
            skipped_no_insert += 1
            continue
        starts = _window_starts(first_insert, actions.shape[0], args.suffix_length, offsets)
        if not starts:
            skipped_no_window += 1
            continue

        frames = _read_video_frames(video_path, args.expected_frames)
        for start in starts:
            task = (
                f"{args.task_prompt}; scenario {scenario}; "
                f"contact suffix starts {first_insert - start} steps before first insertion"
            )
            for t in range(start, start + args.suffix_length):
                image = _resize_rgb(frames[t], args.image_width, args.image_height)
                wrist = image.copy() if args.duplicate_base_as_wrist else np.zeros_like(image)
                dataset.add_frame(
                    {
                        "image": image,
                        "wrist_image": wrist,
                        "state": _state_from_arrays(arrays, t, args.state_mode),
                        "actions": _action_from_source(actions, t),
                        "task": task,
                    }
                )
            dataset.save_episode()
            scenario_window_counts[scenario] = scenario_window_counts.get(scenario, 0) + 1
            records.append(
                {
                    "source_sample_id": item.get("sample_id"),
                    "scenario": scenario,
                    "split": item.get("split"),
                    "video": str(video_path),
                    "input_h5": str(h5_path),
                    "first_insert_frame": first_insert,
                    "suffix_start_frame": int(start),
                    "suffix_end_action_exclusive": int(start + args.suffix_length),
                    "steps_before_first_insert": int(first_insert - start),
                    "start_grasped": bool(grasped[start]),
                    "end_grasped": bool(grasped[min(start + args.suffix_length, grasped.shape[0] - 1)]),
                    "end_inserted": bool(inserted[min(start + args.suffix_length, inserted.shape[0] - 1)]),
                    "start_peg_head_at_hole": rel[start, :3].astype(float).tolist(),
                    "end_peg_head_at_hole": rel[min(start + args.suffix_length, rel.shape[0] - 1), :3].astype(float).tolist(),
                    "action_semantics": "source_h5_first_7_pd_ee_delta_pose_values",
                    "state_semantics": args.state_mode,
                    "wrist_image_policy": "duplicate_base_image" if args.duplicate_base_as_wrist else "zeros",
                }
            )

    manifest = {
        "schema": "openpi_lerobot_maniskill_peg733_contact_suffix16_conversion_v1",
        "repo_id": args.repo_id,
        "output_path": str(output_path),
        "render_manifest": str(render_manifest),
        "num_source_episodes_read": len(videos),
        "num_suffix_episodes_written": len(records),
        "frames_per_suffix_episode": args.suffix_length,
        "total_frames_written": len(records) * args.suffix_length,
        "source_frames_per_episode": args.expected_frames,
        "source_actions_per_episode": args.expected_actions,
        "offsets_before_insert": offsets,
        "skipped_no_insert": skipped_no_insert,
        "skipped_no_window": skipped_no_window,
        "fps": args.fps,
        "camera_storage": args.camera_storage,
        "video_codec": args.video_codec if args.camera_storage == "video" else None,
        "image_shape": [args.image_height, args.image_width, 3],
        "features": {
            "image": [args.image_height, args.image_width, 3],
            "wrist_image": [args.image_height, args.image_width, 3],
            "state": [state_dim],
            "actions": [7],
        },
        "state_mode": args.state_mode,
        "state_semantics": (
            "qpos_first_7_plus_mean_finger_qpos"
            if args.state_mode == STATE_MODE_QPOS8
            else (
                "object_state17=tcp_xyz,peg_xyz,hole_xyz,peg_head_at_hole_xyz,"
                "hole_velocity_step_xyz,grasped,inserted"
            )
        ),
        "scenario_source_counts": scenario_counts,
        "scenario_suffix_counts": scenario_window_counts,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "boundary": (
            "OpenPI/LeRobot contact-suffix data conversion from accepted 733 H5/RGB data. "
            "Window selection uses source inserted/grasped labels only as offline "
            "training-data selection metadata; policy observations remain causal RGB/state "
            "at the suffix timestep. No scorer, VAE, MLP, custom diffusion, or oracle "
            "runtime condition is introduced."
        ),
        "args": asdict(args),
        "records": records,
    }
    out_manifest = Path(args.output_manifest).resolve() if args.output_manifest else output_path / "maniskill_peg733_contact_suffix16_manifest.json"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "num_source_episodes_read": len(videos),
                "num_suffix_episodes_written": len(records),
                "total_frames_written": len(records) * args.suffix_length,
                "skipped_no_insert": skipped_no_insert,
                "skipped_no_window": skipped_no_window,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    tyro.cli(main)
