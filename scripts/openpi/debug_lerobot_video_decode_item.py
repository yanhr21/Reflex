#!/usr/bin/env python3
"""Stage LeRobot video decode and item retrieval inside a Slurm step."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", default="/public/home/yanhongru/ICLR2027/openpi")
    parser.add_argument(
        "--repo-id",
        default="yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_video_clean_20260626",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--mode",
        choices=("metadata", "direct_decode", "dataset_item", "manual_constructor_tail"),
        required=True,
    )
    parser.add_argument("--backend", default="pyav")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--camera-key", default="image")
    parser.add_argument("--with-delta", action="store_true")
    parser.add_argument("--full-dataset", action="store_true")
    parser.add_argument("--decode-all-frames", action="store_true")
    return parser.parse_args()


def require_compute_step() -> None:
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_STEP_ID") in (None, "extern"):
        raise SystemExit("refusing_login_node_execution: run inside a Slurm compute step")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def summarize_value(value: Any) -> Any:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return {"shape": list(shape), "dtype": str(dtype)}
    if isinstance(value, dict):
        return {str(k): summarize_value(v) for k, v in value.items()}
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def main() -> int:
    args = parse_args()
    require_compute_step()
    openpi_root = Path(args.openpi_root).resolve()
    sys.path.insert(0, str(openpi_root / "src"))

    import datasets  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: PLC0415
    from lerobot.common.datasets.utils import check_timestamps_sync  # noqa: PLC0415
    from lerobot.common.datasets.utils import get_episode_data_index  # noqa: PLC0415
    from lerobot.common.datasets.utils import hf_transform_to_torch  # noqa: PLC0415
    from lerobot.common.datasets.video_utils import decode_video_frames  # noqa: PLC0415

    output_json = Path(args.output_json).resolve()
    payload: dict[str, Any] = {
        "schema": "lerobot_video_decode_item_debug_v1",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "repo_id": args.repo_id,
        "mode": args.mode,
        "backend": args.backend,
        "episode_index": args.episode_index,
        "index": args.index,
        "camera_key": args.camera_key,
        "with_delta": args.with_delta,
        "full_dataset": args.full_dataset,
        "decode_all_frames": args.decode_all_frames,
        "openpi_root": str(openpi_root),
    }
    write_json(output_json, {**payload, "status": "started"})

    t0 = time.time()
    meta = LeRobotDatasetMetadata(args.repo_id)
    payload.update(
        {
            "metadata_seconds": time.time() - t0,
            "dataset_root": str(meta.root),
            "fps": meta.fps,
            "total_episodes": meta.total_episodes,
            "total_frames": meta.total_frames,
            "video_keys": list(meta.video_keys),
            "camera_keys": list(meta.camera_keys),
            "features": meta.features,
        }
    )
    video_path = meta.root / meta.get_video_file_path(args.episode_index, args.camera_key)
    payload["video_path"] = str(video_path)
    payload["video_exists"] = video_path.is_file()
    if video_path.is_file():
        payload["video_size_bytes"] = video_path.stat().st_size
    write_json(output_json, {**payload, "status": "metadata_loaded"})

    if args.mode == "metadata":
        payload["total_seconds"] = time.time() - t0
        write_json(output_json, {**payload, "status": "ok"})
        print(json.dumps({"ok": True, "output_json": str(output_json)}))
        return 0

    if args.decode_all_frames:
        timestamps = [i / meta.fps for i in range(16)]
    else:
        timestamps = [0.0]

    if args.mode == "direct_decode":
        t1 = time.time()
        frames = decode_video_frames(video_path, timestamps, 1.0 / meta.fps, args.backend)
        payload["decode_seconds"] = time.time() - t1
        payload["decoded_frames"] = summarize_value(frames)
        payload["total_seconds"] = time.time() - t0
        write_json(output_json, {**payload, "status": "ok"})
        print(json.dumps({"ok": True, "output_json": str(output_json)}))
        return 0

    if args.mode == "manual_constructor_tail":
        t_load = time.time()
        if args.full_dataset:
            hf_dataset = datasets.load_dataset("parquet", data_dir=str(meta.root / "data"), split="train")
        else:
            data_file = str(meta.root / meta.get_data_file_path(args.episode_index))
            hf_dataset = datasets.load_dataset("parquet", data_files=[data_file], split="train")
        hf_dataset.set_transform(hf_transform_to_torch)
        payload["hf_load_seconds"] = time.time() - t_load
        payload["hf_len"] = len(hf_dataset)
        write_json(output_json, {**payload, "status": "hf_loaded"})

        t_index = time.time()
        episodes = None if args.full_dataset else [args.episode_index]
        episode_data_index = get_episode_data_index(meta.episodes, episodes)
        payload["episode_data_index_seconds"] = time.time() - t_index
        payload["episode_data_index"] = {k: v.tolist() for k, v in episode_data_index.items()}
        write_json(output_json, {**payload, "status": "episode_data_index_done"})

        t_ts = time.time()
        timestamp_column = hf_dataset["timestamp"]
        payload["timestamp_column_seconds"] = time.time() - t_ts
        payload["timestamp_column_type"] = type(timestamp_column).__name__
        payload["timestamp_column_len"] = len(timestamp_column)
        write_json(output_json, {**payload, "status": "timestamp_column_done"})

        t_ts_stack = time.time()
        timestamps_np = torch.stack(timestamp_column).numpy()
        payload["timestamp_stack_seconds"] = time.time() - t_ts_stack
        payload["timestamps_shape"] = list(timestamps_np.shape)
        write_json(output_json, {**payload, "status": "timestamp_stack_done"})

        t_ep = time.time()
        episode_column = hf_dataset["episode_index"]
        payload["episode_column_seconds"] = time.time() - t_ep
        payload["episode_column_len"] = len(episode_column)
        write_json(output_json, {**payload, "status": "episode_column_done"})

        t_ep_stack = time.time()
        episode_indices_np = torch.stack(episode_column).numpy()
        payload["episode_stack_seconds"] = time.time() - t_ep_stack
        payload["episode_indices_shape"] = list(episode_indices_np.shape)
        write_json(output_json, {**payload, "status": "episode_stack_done"})

        t_check = time.time()
        check_timestamps_sync(
            timestamps_np,
            episode_indices_np,
            {k: v.numpy() for k, v in episode_data_index.items()},
            meta.fps,
            1.0 / meta.fps,
        )
        payload["check_timestamps_seconds"] = time.time() - t_check
        payload["total_seconds"] = time.time() - t0
        write_json(output_json, {**payload, "status": "ok"})
        print(json.dumps({"ok": True, "output_json": str(output_json)}))
        return 0

    delta_timestamps = None
    if args.with_delta:
        delta_timestamps = {"actions": [i / meta.fps for i in range(16)]}
    episodes = None if args.full_dataset else [args.episode_index]
    t2 = time.time()
    dataset = LeRobotDataset(
        args.repo_id,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        video_backend=args.backend,
    )
    payload["dataset_construct_seconds"] = time.time() - t2
    payload["dataset_len"] = len(dataset)
    payload["dataset_video_backend"] = dataset.video_backend
    write_json(output_json, {**payload, "status": "dataset_constructed"})

    t3 = time.time()
    item = dataset[args.index]
    payload["item_seconds"] = time.time() - t3
    payload["item_summary"] = summarize_value(item)
    payload["total_seconds"] = time.time() - t0
    write_json(output_json, {**payload, "status": "ok"})
    print(json.dumps({"ok": True, "output_json": str(output_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
