#!/usr/bin/env python3
"""Render a Cosmos3 SFT video dataset from saved ManiSkill env states.

This regenerates videos with the ManiSkill3 PegInsertionSide-v1 default human
render camera. It is the approved replacement for the old full1000 preview-link
Cosmos3 dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import tyro

import mani_skill.envs  # noqa: F401
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import common, sapien_utils


SCENARIO_CAPTIONS = {
    "none": (
        "A robot gripper performs a side peg insertion task on a static scene. "
        "The red and white peg, tan hole block, table, and camera remain "
        "consistent while the robot moves the peg toward the side hole."
    ),
    "hole_move_stop": (
        "A robot gripper manipulates a red and white peg while the tan hole "
        "block moves to a new task frame and then stops. The future video "
        "should preserve the changed hole position and the robot motion toward "
        "task completion."
    ),
    "hole_constant": (
        "A robot gripper manipulates a red and white peg while the tan hole "
        "block moves at approximately constant velocity. The scene tests "
        "prediction of a moving target hole and task-frame rebinding."
    ),
    "hole_reverse": (
        "A robot gripper manipulates a red and white peg while the tan hole "
        "block reverses motion during the task. The future video should "
        "reflect the changing target frame rather than restoring the original "
        "layout."
    ),
    "peg_disturb": (
        "A robot gripper manipulates a red and white peg after the peg is "
        "physically disturbed. The future video should preserve the disturbed "
        "peg state and the robot's attempt to recover the side insertion task."
    ),
    "peg_drop": (
        "A robot gripper manipulates a red and white peg after a drop or "
        "regrasp event. The future video should show the robot and peg "
        "continuing in the changed physical scene toward the side insertion "
        "task."
    ),
    "hole_move_stop_large": (
        "A robot gripper manipulates a red and white peg while the tan target "
        "hole block moves a large distance to a new task frame and then stops. "
        "The robot trajectory anticipates the future target pose and finishes "
        "the side insertion."
    ),
    "hole_constant_large": (
        "A robot gripper manipulates a red and white peg while the tan target "
        "hole block moves over a large range at roughly constant velocity. "
        "The future video should predict the moving target path and final "
        "insertion frame."
    ),
    "hole_reverse_large": (
        "A robot gripper manipulates a red and white peg while the tan target "
        "hole block moves a large distance, overshoots, and reverses before "
        "the final insertion."
    ),
    "hole_sine_large": (
        "A robot gripper manipulates a red and white peg while the tan target "
        "hole block follows a large curved or sinusoidal path before the peg "
        "insertion is completed."
    ),
    "hole_continuous_insert_large": (
        "A robot gripper manipulates a red and white peg while the tan target "
        "hole block keeps moving until the insertion moment. The robot must "
        "aim at the future target location rather than the current hole pose."
    ),
    "hole_late_shift_large": (
        "A robot gripper manipulates a red and white peg while the tan target "
        "hole block makes a late large shift before settling into the final "
        "successful insertion pose."
    ),
    "hole_late_move_stop": (
        "A robot gripper holds the red and white peg near the current side "
        "hole before the tan target block quickly shifts to a new reachable "
        "task frame and stops. The robot must redirect from the late target "
        "motion and complete a physically valid insertion."
    ),
    "hole_late_constant": (
        "A robot gripper holds the red and white peg near the current side "
        "hole before the tan target block begins a late constant-velocity "
        "motion. The robot must anticipate the moving hole frame and insert "
        "the peg at the final reachable pose."
    ),
    "hole_late_reverse": (
        "A robot gripper holds the red and white peg near the current side "
        "hole before the tan target block makes a late shift, overshoots, and "
        "reverses. The robot must rebind to the final hole pose and insert the "
        "peg without colliding with the wall."
    ),
    "hole_late_sine": (
        "A robot gripper holds the red and white peg near the current side "
        "hole before the tan target block follows a late curved motion. The "
        "robot must predict the changed task frame and complete the side "
        "insertion."
    ),
    "hole_late_continuous_insert": (
        "A robot gripper holds the red and white peg near the current side "
        "hole before the tan target block starts a late motion that continues "
        "toward the insertion moment. The robot must aim ahead rather than "
        "waiting for the block to find the peg."
    ),
    "hole_late_fast_shift": (
        "A robot gripper holds the red and white peg near the current side "
        "hole before the tan target block makes a brief fast shift to a new "
        "reachable task frame. The robot must recover from the abrupt change "
        "and complete a valid side insertion."
    ),
}


@dataclass
class Args:
    paths_file: str
    output_root: str
    env_id: str = "PegInsertionSide-v1"
    obs_mode: str = "state"
    control_mode: str = "pd_ee_delta_pose"
    sim_backend: str = "physx_cpu"
    width: int = 1024
    height: int = 1024
    fps: int = 30
    frame_stride: int = 1
    max_frames: int = 0
    val_fraction: float = 0.1
    camera_eye: tuple[float, float, float] = (0.5, -0.5, 0.8)
    camera_target: tuple[float, float, float] = (0.05, -0.1, 0.4)
    camera_up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    camera_fov: float = 1.0
    shader_pack: str = "default"
    sheet_limit: int = 10
    sheet_frames: int = 12
    sheet_thumb_width: int = 512
    resume: bool = True
    overwrite: bool = False
    start_index: int = 0
    end_index: int = 0
    shard_id: str = ""
    write_canonical_metadata: bool = True
    metadata_only: bool = False


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_paths(path: Path) -> list[Path]:
    out = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            out.append(Path(stripped))
    return out


def _trajectory_names(h5: h5py.File) -> list[str]:
    names = [key for key in h5.keys() if key.startswith("traj_")]
    return sorted(names, key=lambda name: int(name.split("_", 1)[1]))


def _sample_id(path: Path, traj_name: str) -> str:
    parent = path.parent.name.replace(".rgbd", "")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", parent)
    return f"{safe}_{traj_name}"


def _scenario(sample_id: str) -> str:
    for scenario in sorted(SCENARIO_CAPTIONS, key=len, reverse=True):
        if sample_id.startswith(scenario + "_"):
            return scenario
    return "unknown"


def _stable_val(sample_id: str, val_fraction: float) -> bool:
    if val_fraction <= 0:
        return False
    if val_fraction >= 1:
        return True
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    return (seed % 10000) < int(round(val_fraction * 10000))


def _seed(group: h5py.Group, fallback: int) -> int:
    if "seed" in group.attrs:
        return int(group.attrs["seed"])
    raw = group.attrs.get("source_summary_json") or group.attrs.get("summary_json")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if raw:
        try:
            summary = json.loads(raw)
            return int(summary.get("seed", fallback))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return int(fallback)


def _select_indices(n: int, stride: int, max_frames: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    indices = np.arange(0, n, max(1, int(stride)), dtype=np.int64)
    if indices[-1] != n - 1:
        indices = np.concatenate([indices, np.asarray([n - 1], dtype=np.int64)])
    if max_frames > 0 and len(indices) > max_frames:
        keep = np.linspace(0, len(indices) - 1, max_frames).round().astype(np.int64)
        indices = indices[np.unique(keep)]
    return indices


def _render_frame(env, camera_name: str = "render_camera") -> np.ndarray:
    frame = env.unwrapped.render_rgb_array(camera_name)
    frame = common.to_numpy(frame)
    frame = np.asarray(frame)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    return np.clip(frame[..., :3], 0, 255).astype(np.uint8)


def _sheet_indices(num_frames: int, count: int) -> set[int]:
    if count <= 0 or num_frames <= 0:
        return set()
    return set(np.linspace(0, num_frames - 1, min(count, num_frames)).round().astype(int).tolist())


def _make_contact_sheet(
    frames: list[tuple[int, int, np.ndarray]],
    output_path: Path,
    thumb_width: int,
) -> None:
    if not frames:
        return
    label_h = 28
    thumbs = []
    for video_i, source_i, frame in frames:
        img = Image.fromarray(frame).convert("RGB")
        scale = thumb_width / max(1, img.width)
        img = img.resize((thumb_width, max(1, int(round(img.height * scale)))))
        thumbs.append((video_i, source_i, img))
    cols = min(4, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    cell_h = thumbs[0][2].height + label_h
    sheet = Image.new("RGB", (cols * thumb_width, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (video_i, source_i, img) in enumerate(thumbs):
        x = (i % cols) * thumb_width
        y = (i // cols) * cell_h
        sheet.paste(img, (x, y))
        draw.text((x + 4, y + img.height + 4), f"video frame {video_i}, source frame {source_i}", fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _fmt_vec(values: np.ndarray, ndigits: int = 4) -> str:
    return "[" + ", ".join(f"{float(x):.{ndigits}f}" for x in np.asarray(values).reshape(-1).tolist()) + "]"


def _task_state_condition(group: h5py.Group, indices: np.ndarray) -> dict[str, Any]:
    """Summarize robot/object state so Cosmos sees more than pixels/text only."""
    if len(indices) == 0:
        return {"caption": "", "metadata": {}}

    first_i = int(indices[0])
    last_i = int(indices[-1])
    slots = group.get("slots")
    perturb = group.get("perturb")
    out: dict[str, Any] = {
        "source_frame_start": first_i,
        "source_frame_end": last_i,
    }

    def slot_xyz(name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        if slots is None or name not in slots:
            return None, None
        arr = np.asarray(slots[name])
        if arr.ndim < 2 or arr.shape[0] <= max(first_i, last_i):
            return None, None
        return arr[first_i, :3].astype(float), arr[last_i, :3].astype(float)

    for key, dataset_name in (
        ("hole", "hole_pose"),
        ("peg", "peg_pose"),
        ("tcp", "tcp_pose"),
    ):
        start, end = slot_xyz(dataset_name)
        if start is None or end is None:
            continue
        out[f"{key}_xyz_start"] = start.tolist()
        out[f"{key}_xyz_end"] = end.tolist()
        out[f"{key}_xyz_delta"] = (end - start).tolist()

    if slots is not None:
        for key in ("grasped", "inserted"):
            if key in slots:
                arr = np.asarray(slots[key])
                if arr.shape[0] > max(first_i, last_i):
                    out[f"{key}_start"] = bool(arr[first_i])
                    out[f"{key}_end"] = bool(arr[last_i])

    if perturb is not None:
        for key in ("trigger_step", "triggered"):
            if key in perturb:
                arr = np.asarray(perturb[key])
                if arr.size:
                    flat = arr.reshape(-1)
                    if key == "trigger_step":
                        nonneg = flat[flat >= 0]
                        out[key] = int(nonneg[0]) if nonneg.size else int(flat[0])
                    else:
                        out[key] = bool(flat.any())
        for key in ("hole_delta_cumulative", "peg_delta_applied"):
            if key in perturb:
                arr = np.asarray(perturb[key])
                if arr.ndim >= 2 and arr.shape[0] > 0:
                    out[f"{key}_final"] = arr[-1, :3].astype(float).tolist()

    parts = []
    if "hole_xyz_start" in out:
        parts.append(
            "hole block xyz moves from "
            f"{_fmt_vec(np.asarray(out['hole_xyz_start']))} to {_fmt_vec(np.asarray(out['hole_xyz_end']))}"
        )
    if "peg_xyz_start" in out:
        parts.append(
            "peg xyz moves from "
            f"{_fmt_vec(np.asarray(out['peg_xyz_start']))} to {_fmt_vec(np.asarray(out['peg_xyz_end']))}"
        )
    if "tcp_xyz_start" in out:
        parts.append(
            "robot tool-center xyz moves from "
            f"{_fmt_vec(np.asarray(out['tcp_xyz_start']))} to {_fmt_vec(np.asarray(out['tcp_xyz_end']))}"
        )
    if "hole_delta_cumulative_final" in out:
        parts.append(f"cumulative hole perturbation xyz is {_fmt_vec(np.asarray(out['hole_delta_cumulative_final']))}")
    if "peg_delta_applied_final" in out:
        parts.append(f"final peg perturbation xyz is {_fmt_vec(np.asarray(out['peg_delta_applied_final']))}")
    if "grasped_start" in out and "grasped_end" in out:
        parts.append(f"grasped changes from {out['grasped_start']} to {out['grasped_end']}")
    if "inserted_start" in out and "inserted_end" in out:
        parts.append(f"inserted changes from {out['inserted_start']} to {out['inserted_end']}")

    caption = ""
    if parts:
        caption = " Robot and object state condition: " + "; ".join(parts) + "."
    return {"caption": caption, "metadata": out}


def _record(
    video_rel: str,
    sample_id: str,
    scenario: str,
    duration: float,
    args: Args,
    task_state_condition: dict[str, Any],
) -> dict[str, Any]:
    caption = SCENARIO_CAPTIONS.get(
        scenario,
        "A robot gripper manipulates a red and white peg and a tan side hole block in a dynamic peg insertion task.",
    )
    caption = caption + task_state_condition.get("caption", "")
    num_frames = int(round(duration * float(args.fps)))
    return {
        "uuid": sample_id,
        "duration": float(duration),
        "width": int(args.width),
        "height": int(args.height),
        "vision_path": video_rel,
        "t2w_windows": [
            {
                "start_frame": 0,
                "end_frame": max(0, num_frames - 1),
                "temporal_interval": 1,
                "caption": caption,
            }
        ],
        "metadata": {
            "scenario": scenario,
            "source": "maniskill_default_human_render_from_env_states",
            "fps": int(args.fps),
            "camera": "PegInsertionSide-v1_default_human_render",
            "conditioning_policy": "video_prefix_not_single_image_i2v",
            "task_state_condition": task_state_condition.get("metadata", {}),
        },
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> None:
    args = tyro.cli(Args)
    paths_file = Path(args.paths_file)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _read_paths(paths_file)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[:10])
    total_paths = len(paths)
    start_index = max(0, int(args.start_index))
    end_index = total_paths if int(args.end_index) <= 0 else min(total_paths, int(args.end_index))
    if start_index > end_index:
        raise ValueError(f"start_index={start_index} is greater than end_index={end_index}")

    camera_pose = sapien_utils.look_at(args.camera_eye, args.camera_target, args.camera_up)
    camera_pose_list = camera_pose.raw_pose.detach().cpu().numpy().reshape(-1, 7)[0].tolist()
    render_camera_config = {
        "render_camera": {
            "pose": camera_pose_list,
            "width": int(args.width),
            "height": int(args.height),
            "fov": float(args.camera_fov),
            "near": 0.01,
            "far": 100,
            "shader_pack": args.shader_pack,
        }
    }
    env = None
    startup_event = {
        "event": "metadata_only_start" if args.metadata_only else "env_create_start",
        "env_id": args.env_id,
        "num_paths": total_paths,
        "start_index": start_index,
        "end_index": end_index,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "frame_stride": args.frame_stride,
        "max_frames": args.max_frames,
        "camera_eye": list(args.camera_eye),
        "camera_target": list(args.camera_target),
        "metadata_only": bool(args.metadata_only),
    }
    print(json.dumps(startup_event, sort_keys=True), flush=True)
    if not args.metadata_only:
        env = gym.make(
            args.env_id,
            obs_mode=args.obs_mode,
            control_mode=args.control_mode,
            reward_mode="sparse",
            render_mode="rgb_array",
            sim_backend=args.sim_backend,
            human_render_camera_configs=render_camera_config,
        )
        print(json.dumps({"event": "env_create_done"}, sort_keys=True), flush=True)

    records: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    videos: list[dict[str, Any]] = []
    scenario_counts: dict[str, int] = {}
    skipped = 0

    try:
        for sample_i, h5_path in enumerate(paths):
            if sample_i < start_index or sample_i >= end_index:
                continue
            with h5py.File(h5_path, "r") as h5:
                names = _trajectory_names(h5)
                if len(names) != 1:
                    raise ValueError(f"{h5_path} expected one trajectory, found {names}")
                traj_name = names[0]
                group = h5[traj_name]
                seed = _seed(group, sample_i)
                sample_id = _sample_id(h5_path, traj_name)
                scenario = _scenario(sample_id)
                split = "val" if _stable_val(sample_id, args.val_fraction) else "train"
                scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
                video_rel = f"videos/{sample_i:04d}_{sample_id}.mp4"
                video_path = output_root / split / video_rel
                sheet_path = output_root / "review_sheets" / f"{sample_i:04d}_{sample_id}_review_sheet.png"
                env_states = trajectory_utils.dict_to_list_of_dicts(group["env_states"])
                indices = _select_indices(len(env_states), args.frame_stride, args.max_frames)
                duration = len(indices) / float(args.fps)
                task_state_condition = _task_state_condition(group, indices)
                item = {
                    "index": sample_i,
                    "scenario": scenario,
                    "split": split,
                    "sample_id": sample_id,
                    "input_h5": str(h5_path),
                    "trajectory": traj_name,
                    "seed": seed,
                    "num_source_states": len(env_states),
                    "num_video_frames": len(indices),
                    "first_source_frame": int(indices[0]) if len(indices) else None,
                    "last_source_frame": int(indices[-1]) if len(indices) else None,
                    "fps": args.fps,
                    "duration_seconds": duration,
                    "video": str(video_path),
                    "review_sheet": str(sheet_path) if sample_i < args.sheet_limit else None,
                    "task_state_condition": task_state_condition.get("metadata", {}),
                }
                records[split].append(_record(video_rel, sample_id, scenario, duration, args, task_state_condition))
                videos.append(item)

                if args.metadata_only:
                    if not video_path.exists():
                        raise FileNotFoundError(f"metadata-only mode requires existing video: {video_path}")
                    skipped += 1
                    print(json.dumps({"event": "sample_metadata_existing", **item}, sort_keys=True), flush=True)
                    continue
                if video_path.exists() and args.resume and not args.overwrite:
                    skipped += 1
                    print(json.dumps({"event": "sample_skip_existing", **item}, sort_keys=True), flush=True)
                    continue
                if video_path.exists() and not args.overwrite:
                    raise FileExistsError(video_path)
                if env is None:
                    raise RuntimeError("render environment is unavailable outside metadata-only mode")

                video_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_video_path = video_path.with_name(video_path.name + ".tmp.mp4")
                if tmp_video_path.exists():
                    tmp_video_path.unlink()
                env.reset(seed=seed)
                sheet_keep = _sheet_indices(len(indices), args.sheet_frames) if sample_i < args.sheet_limit else set()
                sheet_frames: list[tuple[int, int, np.ndarray]] = []
                print(
                    json.dumps(
                        {
                            "event": "sample_start",
                            "index": sample_i,
                            "sample_id": sample_id,
                            "scenario": scenario,
                            "split": split,
                            "num_source_states": len(env_states),
                            "num_selected_frames": len(indices),
                            "video": str(video_path),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                with imageio.get_writer(tmp_video_path, fps=args.fps, macro_block_size=1) as writer:
                    for video_i, source_i in enumerate(indices):
                        env.unwrapped.set_state_dict(env_states[int(source_i)])
                        frame = _render_frame(env)
                        writer.append_data(frame)
                        if video_i in sheet_keep:
                            sheet_frames.append((video_i, int(source_i), frame.copy()))
                        if video_i == 0 or (video_i + 1) % 60 == 0 or video_i == len(indices) - 1:
                            print(
                                json.dumps(
                                    {
                                        "event": "frame_rendered",
                                        "index": sample_i,
                                        "sample_id": sample_id,
                                        "video_frame": video_i,
                                        "source_frame": int(source_i),
                                    },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                tmp_video_path.replace(video_path)
                if sample_i < args.sheet_limit:
                    _make_contact_sheet(sheet_frames, sheet_path, args.sheet_thumb_width)
                print(json.dumps({"event": "sample_done", **item}, sort_keys=True), flush=True)
    finally:
        if env is not None:
            env.close()

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "args": _jsonable(asdict(args)),
        "paths_file": str(paths_file),
        "output_root": str(output_root),
        "num_total_paths": total_paths,
        "start_index": start_index,
        "end_index": end_index,
        "num_videos": len(videos),
        "num_train": len(records["train"]),
        "num_val": len(records["val"]),
        "num_skipped_existing": skipped,
        "scenario_counts": scenario_counts,
        "camera": {
            "source": "ManiSkill3 PegInsertionSide-v1 _default_human_render_camera_configs",
            "eye": list(args.camera_eye),
            "target": list(args.camera_target),
            "up": list(args.camera_up),
            "fov": args.camera_fov,
            "width": args.width,
            "height": args.height,
            "shader_pack": args.shader_pack,
        },
        "environment": {
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "SLURM_JOB_NODELIST": os.environ.get("SLURM_JOB_NODELIST"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "VK_ICD_FILENAMES": os.environ.get("VK_ICD_FILENAMES"),
            "DISPLAY": os.environ.get("DISPLAY"),
        },
        "boundary": (
            "Approved full1000 Cosmos3 SFT video dataset generated from saved "
            "env_states with ManiSkill default human render camera. Old preview "
            "videos are not used as training input."
        ),
        "videos": videos,
    }
    if args.write_canonical_metadata:
        for split in ("train", "val"):
            _write_jsonl(output_root / split / "video_dataset_file.jsonl", records[split])
        manifest_path = output_root / "manifest.json"
    else:
        shard_id = args.shard_id or f"range_{start_index}_{end_index}"
        shard_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", shard_id)
        for split in ("train", "val"):
            _write_jsonl(output_root / split / f"video_dataset_file.{shard_id}.jsonl", records[split])
        manifest_path = output_root / "shards" / f"{shard_id}_manifest.json"
        manifest["canonical_metadata_written"] = False
        manifest["shard_id"] = shard_id
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "manifest_written", "manifest": str(manifest_path), "num_videos": len(videos)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
