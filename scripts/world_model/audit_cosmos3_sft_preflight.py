#!/usr/bin/env python3
"""Strict preflight for Cosmos3 ManiSkill SFT JSONL/video/action/state lengths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import cv2


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def read_rows(path: Path, limit: int | None) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise RuntimeError(f"empty jsonl: {path}")
    return rows


def video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


def effective_video_frame_count(row: dict, path: Path, source_frames: int | None = None) -> tuple[int, dict]:
    """Return the training clip length, honoring JSONL t2w window slices."""
    if source_frames is None:
        source_frames = video_frame_count(path)
    windows = row.get("t2w_windows") or []
    if not windows:
        return source_frames, {"source_video_frames": source_frames, "windowed": False}
    if len(windows) != 1 or not isinstance(windows[0], dict):
        raise RuntimeError(f"expected exactly one t2w window for {row.get('uuid')}, got {windows!r}")
    window = windows[0]
    start = int(window.get("start_frame", 0))
    end = int(window.get("end_frame", source_frames - 1))
    interval = int(window.get("temporal_interval", 1))
    if interval <= 0:
        raise RuntimeError(f"invalid temporal_interval for {row.get('uuid')}: {interval}")
    if start < 0 or end < start or end >= source_frames:
        raise RuntimeError(
            f"invalid t2w window for {row.get('uuid')}: start={start} end={end} "
            f"source_frames={source_frames}"
        )
    count = 1 + (end - start) // interval
    return count, {
        "source_video_frames": source_frames,
        "windowed": True,
        "start_frame": start,
        "end_frame": end,
        "temporal_interval": interval,
    }


def resolve_path(value: str, jsonl_path: Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = jsonl_path.parent / path
    return path


def load_action(path: Path):
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    payload = json.loads(path.read_text())
    return payload["action"] if isinstance(payload, dict) and "action" in payload else payload


def action_shape(path: Path) -> tuple[int, int]:
    action = load_action(path)
    if isinstance(action, np.ndarray):
        if action.ndim != 2:
            return (int(action.shape[0]) if action.ndim >= 1 else 0, -1)
        return int(action.shape[0]), int(action.shape[1])
    if not action:
        return 0, -1
    return len(action), len(action[0])


def check_rows(
    path: Path,
    *,
    num_video_frames: int,
    action_conditioned: bool,
    wam_sft_mode: str,
    require_state_targets: bool,
    strict_full_preflight: bool,
) -> dict:
    rows = read_rows(path, None if strict_full_preflight else 8)
    action_lengths: set[int] = set()
    action_dims: set[int] = set()
    state_lengths: set[int] = set()
    state_dims: set[int] = set()
    video_frames_seen: set[int] = set()
    video_frame_cache: dict[Path, int] = {}
    action_shape_cache: dict[Path, tuple[int, int]] = {}
    state_cache: dict[Path, tuple[int, int, list[str]]] = {}
    for row in rows:
        if any("depth" in str(key).lower() for key in row.keys()):
            raise RuntimeError(f"depth input is not allowed in active SFT row: {row.get('uuid')}")
        video_path = resolve_path(str(row["vision_path"]), path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        if video_path not in video_frame_cache:
            video_frame_cache[video_path] = video_frame_count(video_path)
        frames, frame_info = effective_video_frame_count(row, video_path, video_frame_cache[video_path])
        video_frames_seen.add(frames)
        if frames != num_video_frames:
            raise RuntimeError(
                f"video length mismatch for {row.get('uuid')}: frames={frames}, "
                f"expected={num_video_frames}, frame_info={frame_info}"
            )
        if not action_conditioned:
            continue
        action_chunk_size = int(row.get("action_chunk_size", -1))
        if action_chunk_size != num_video_frames - 1:
            raise RuntimeError(
                f"action_chunk_size mismatch for {row.get('uuid')}: "
                f"{action_chunk_size} != {num_video_frames - 1}"
            )
        action_path = resolve_path(str(row["action_path"]), path)
        if not action_path.exists():
            raise FileNotFoundError(action_path)
        if action_path not in action_shape_cache:
            action_shape_cache[action_path] = action_shape(action_path)
        action_len, action_dim = action_shape_cache[action_path]
        if action_len != num_video_frames - 1:
            raise RuntimeError(
                f"action length mismatch for {row.get('uuid')}: {action_len} != {num_video_frames - 1}"
            )
        action_lengths.add(action_len)
        action_dims.add(action_dim)
        cond_action = [int(x) for x in row.get("condition_frame_indexes_action", [])]
        if cond_action and max(cond_action) >= action_len:
            raise RuntimeError(f"condition_frame_indexes_action out of range for {row.get('uuid')}")
        if "joint_policy" in wam_sft_mode and cond_action:
            allow_history_action_condition = "history_action" in wam_sft_mode
            if not allow_history_action_condition:
                raise RuntimeError(
                    f"joint_policy SFT must predict actions, not condition on every action: {row.get('uuid')}"
                )
            expected_history = int(row.get("condition_prefix_frames", 0)) - 1
            if expected_history <= 0 or cond_action != list(range(expected_history)):
                raise RuntimeError(
                    f"chunked joint_policy history action condition mismatch for {row.get('uuid')}: "
                    f"{cond_action[:5]}... len={len(cond_action)} expected_history={expected_history}"
                )
        if require_state_targets:
            state_path_value = row.get("state_target_path") or row.get("task_state_target_path")
            if not state_path_value:
                raise RuntimeError(f"missing state_target_path for {row.get('uuid')}")
            state_path = resolve_path(str(state_path_value), path)
            if not state_path.exists():
                raise FileNotFoundError(state_path)
            if state_path not in state_cache:
                payload = json.loads(state_path.read_text())
                states = payload.get("states")
                names = payload.get("state_vector_names") or row.get("state_target_vector_names") or []
                if not isinstance(states, list):
                    state_cache[state_path] = (0, 0, list(names))
                else:
                    state_cache[state_path] = (len(states), len(states[0]) if states else 0, list(names))
            state_len, state_dim, names = state_cache[state_path]
            if state_len != num_video_frames:
                raise RuntimeError(
                    f"state target length mismatch for {row.get('uuid')}: "
                    f"{state_len} != {num_video_frames}"
                )
            required = {"hole_pose_x", "hole_pose_y", "hole_pose_z", "peg_head_at_hole_x", "tcp_pose_x"}
            if not required.issubset(set(names)):
                raise RuntimeError(f"state target missing required task fields for {row.get('uuid')}")
            state_lengths.add(state_len)
            if state_len:
                state_dims.add(state_dim)
                if state_dim != len(names):
                    raise RuntimeError(
                        f"state target dim/name mismatch for {row.get('uuid')}: "
                        f"{state_dim} != {len(names)}"
                    )
    return {
        "jsonl": str(path),
        "rows_checked": len(rows),
        "strict_full_preflight": strict_full_preflight,
        "video_frames_seen": sorted(video_frames_seen),
        "action_lengths": sorted(action_lengths),
        "action_dims": sorted(action_dims),
        "state_lengths": sorted(state_lengths),
        "state_dims": sorted(state_dims),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--num-video-frames", type=int, required=True)
    parser.add_argument("--action-conditioned-sft", type=str_to_bool, required=True)
    parser.add_argument("--wam-sft-mode", required=True)
    parser.add_argument("--require-state-targets", type=str_to_bool, required=True)
    parser.add_argument("--strict-full-preflight", type=str_to_bool, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "chunked_pickup" in args.wam_sft_mode:
        raise RuntimeError(
            "chunked_pickup WAM mode was rejected by the 2026-06-09 "
            "300-step reset; use a full-episode/equal-length contract instead"
        )
    train_report = check_rows(
        args.train_jsonl,
        num_video_frames=args.num_video_frames,
        action_conditioned=args.action_conditioned_sft,
        wam_sft_mode=args.wam_sft_mode,
        require_state_targets=args.require_state_targets,
        strict_full_preflight=args.strict_full_preflight,
    )
    val_report = check_rows(
        args.val_jsonl,
        num_video_frames=args.num_video_frames,
        action_conditioned=args.action_conditioned_sft,
        wam_sft_mode=args.wam_sft_mode,
        require_state_targets=args.require_state_targets,
        strict_full_preflight=args.strict_full_preflight,
    )
    print(
        "strict_length_preflight=passed "
        f"num_video_frames={args.num_video_frames} action_conditioned={args.action_conditioned_sft} "
        f"wam_sft_mode={args.wam_sft_mode} require_state_targets={args.require_state_targets} "
        f"strict_full_preflight={args.strict_full_preflight}"
    )
    print("strict_length_preflight_report=" + json.dumps({"train": train_report, "val": val_report}, sort_keys=True))


if __name__ == "__main__":
    main()
