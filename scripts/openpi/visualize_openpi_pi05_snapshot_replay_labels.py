#!/usr/bin/env python3
"""Render videos/contact sheets for saved OpenPI pi0.5 snapshot replay labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
WORLD_MODEL_DIR = ROOT / "scripts" / "world_model"
if str(WORLD_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(WORLD_MODEL_DIR))

from replay_cosmos3_live_action_bank_from_snapshots import (  # noqa: E402
    load_env_states,
    load_snapshot_state,
)
from run_cosmos3_live_receding_loop import (  # noqa: E402
    apply_external_target_pose,
    live_pose_row,
    require_compute_step,
)
from run_cosmos3_receding_closed_loop import (  # noqa: E402
    _action_space_bounds,
    _get_base_env,
    _import_live_control_stack,
    _make_live_env,
    _parse_seed_from_text,
    _prepare_step_action,
    _render_frame,
    jsonable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--dp-manifest", default=str(ROOT / "experiments/dp_peg1000/run_90201/manifest.json"))
    parser.add_argument("--robot-action-dim", type=int, default=7)
    parser.add_argument("--max-labels", type=int, default=0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--sheet-cols", type=int, default=4)
    parser.add_argument("--sheet-frame-width", type=int, default=256)
    parser.add_argument("--clip-live-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--external-target-mode", choices=("source_env_state", "none"), default="source_env_state")
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260625)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def find_label_paths(replay_root: Path) -> list[Path]:
    labels = sorted(replay_root.rglob("policy_droid_snapshot_action_replay_label.json"))
    if labels:
        return labels
    list_path = replay_root / "replay_labels.list"
    if list_path.exists():
        out = [Path(line).resolve() for line in list_path.read_text().splitlines() if line.strip()]
        return [path for path in out if path.exists()]
    return []


def normalize_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size and float(arr.max()) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def write_video(path: Path, frames_rgb: list[np.ndarray], fps: float) -> None:
    if not frames_rgb:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(w), int(h)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    try:
        for frame in frames_rgb:
            writer.write(cv2.cvtColor(normalize_rgb(frame), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def resize_for_sheet(frame: np.ndarray, width: int) -> np.ndarray:
    rgb = normalize_rgb(frame)
    h, w = rgb.shape[:2]
    scale = float(width) / float(max(1, w))
    return cv2.resize(rgb, (int(width), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def write_contact_sheet(path: Path, frames_rgb: list[np.ndarray], captions: list[str], cols: int, width: int) -> None:
    if not frames_rgb:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tiles = []
    caption_h = 34
    for frame, caption in zip(frames_rgb, captions):
        tile = resize_for_sheet(frame, int(width))
        canvas = np.zeros((tile.shape[0] + caption_h, tile.shape[1], 3), dtype=np.uint8)
        canvas[: tile.shape[0]] = tile
        cv2.putText(
            canvas,
            str(caption)[:64],
            (6, tile.shape[0] + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(canvas)
    cols = max(1, int(cols))
    rows = []
    max_h = max(tile.shape[0] for tile in tiles)
    max_w = max(tile.shape[1] for tile in tiles)
    for start in range(0, len(tiles), cols):
        row_tiles = []
        for tile in tiles[start : start + cols]:
            padded = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            padded[: tile.shape[0], : tile.shape[1]] = tile
            row_tiles.append(padded)
        while len(row_tiles) < cols:
            row_tiles.append(np.zeros((max_h, max_w, 3), dtype=np.uint8))
        rows.append(np.concatenate(row_tiles, axis=1))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def select_sheet_indices(n: int, max_tiles: int = 8) -> list[int]:
    if n <= max_tiles:
        return list(range(n))
    return sorted(set(int(round(x)) for x in np.linspace(0, n - 1, max_tiles)))


def render_one(
    *,
    label_path: Path,
    output_root: Path,
    stack: dict[str, Any],
    env: Any,
    base_env: Any,
    low: Any,
    high: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    label = read_json(label_path)
    action_path = Path(str(label["action_chunk_json"])).resolve()
    action_chunk = read_json(action_path)
    source_h5 = Path(str(label["source_h5"])).resolve()
    snapshot_h5 = Path(str(label["snapshot_state_h5"])).resolve()
    prefix_frame = int(label["prefix_frame_index"])
    execute_steps = int(label.get("execute_steps_actual") or label.get("execute_steps_requested") or args.max_episode_steps)
    actions = np.asarray(action_chunk["denormalized_robot_action_chunk"], dtype=np.float32)
    actions = actions[:, : int(args.robot_action_dim)]
    execute_steps = min(execute_steps, actions.shape[0])

    env_states = load_env_states(source_h5, stack["trajectory_utils"])
    snapshot_state, _snapshot_attrs = load_snapshot_state(snapshot_h5)
    reset_seed = _parse_seed_from_text(" ".join([source_h5.name, str(action_chunk.get("sample_output_json", ""))]))
    env.reset(seed=int(reset_seed if reset_seed is not None else args.seed))
    base_env.set_state_dict(snapshot_state)

    sample_name = label_path.parent.name
    out_dir = output_root / sample_name
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = [normalize_rgb(_render_frame(env))]
    captions: list[str] = ["start"]
    step_rows: list[dict[str, Any]] = []
    previous_hole_xyz = np.asarray(live_pose_row(base_env, stack, None)["hole_xyz"], dtype=np.float32).copy()
    for local_i, action in enumerate(actions[:execute_steps]):
        step_action, action_record = _prepare_step_action(action, low, high, bool(args.clip_live_actions))
        _obs, reward, terminated, truncated, _info = env.step(step_action)
        source_frame = min(int(prefix_frame) + local_i + 1, len(env_states) - 1)
        external_target = apply_external_target_pose(
            base_env=base_env,
            stack=stack,
            env_states=env_states,
            source_frame=source_frame,
            args=args,
        )
        live = live_pose_row(base_env, stack, previous_hole_xyz)
        previous_hole_xyz = np.asarray(live["hole_xyz"], dtype=np.float32).copy()
        frame = normalize_rgb(_render_frame(env))
        frames.append(frame)
        rel = np.asarray(live["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
        inserted = bool(live.get("inserted", False))
        grasped = bool(live.get("grasped", False))
        captions.append(f"s{local_i:02d} g{int(grasped)} i{int(inserted)} x{rel[0]:+.2f} y{rel[1]:+.2f} z{rel[2]:+.2f}")
        step_rows.append(
            {
                "local_step": int(local_i),
                "source_frame": int(source_frame),
                "action": action_record,
                "external_target": external_target,
                "reward": jsonable(reward),
                "terminated": jsonable(terminated),
                "truncated": jsonable(truncated),
                "grasped": grasped,
                "inserted": inserted,
                "peg_head_at_hole": rel.astype(float).tolist(),
            }
        )

    video_path = out_dir / "openpi_chunk_replay.mp4"
    sheet_path = out_dir / "openpi_chunk_contact_sheet.jpg"
    write_video(video_path, frames, float(args.fps))
    sheet_indices = select_sheet_indices(len(frames))
    write_contact_sheet(
        sheet_path,
        [frames[i] for i in sheet_indices],
        [captions[i] for i in sheet_indices],
        int(args.sheet_cols),
        int(args.sheet_frame_width),
    )
    summary = {
        "schema": "openpi_pi05_snapshot_replay_visual_evidence_v1",
        "label_path": str(label_path),
        "action_chunk_json": str(action_path),
        "source_h5": str(source_h5),
        "snapshot_state_h5": str(snapshot_h5),
        "prefix_frame_index": int(prefix_frame),
        "execute_steps_visualized": int(execute_steps),
        "frame_count": int(len(frames)),
        "video_path": str(video_path),
        "contact_sheet_path": str(sheet_path),
        "label_after_success": bool(label.get("after_success", False)),
        "label_after_inserted_live_pose": bool(label.get("after_inserted_live_pose", False)),
        "label_after_grasped": bool(label.get("after_grasped", False)),
        "label_after_contact_stable_proxy": bool(label.get("after_contact_stable_proxy", False)),
        "label_dp96_success": bool((label.get("dp_rollout_continuability") or {}).get("success", False)),
        "step_rows": step_rows,
        "boundary": (
            "Visual evidence regenerated from the saved live snapshot and the "
            "already selected OpenPI action chunk. It does not change action "
            "selection, policy inputs, or replay metrics."
        ),
    }
    write_json(out_dir / "visual_evidence_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    require_compute_step()
    replay_root = Path(args.replay_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else replay_root / "visual_evidence"
    output_root.mkdir(parents=True, exist_ok=True)

    labels = find_label_paths(replay_root)
    if int(args.max_labels) > 0:
        labels = labels[: int(args.max_labels)]
    if not labels:
        raise RuntimeError(f"no replay label files found under {replay_root}")

    stack = _import_live_control_stack(ROOT)
    env = _make_live_env(stack, Path(args.dp_manifest).resolve(), args)
    base_env = _get_base_env(env)
    low, high = _action_space_bounds(env, int(args.robot_action_dim))

    rows = [
        render_one(
            label_path=label_path,
            output_root=output_root,
            stack=stack,
            env=env,
            base_env=base_env,
            low=low,
            high=high,
            args=args,
        )
        for label_path in labels
    ]
    manifest = {
        "schema": "openpi_pi05_snapshot_replay_visual_evidence_manifest_v1",
        "replay_root": str(replay_root),
        "output_root": str(output_root),
        "label_count": int(len(labels)),
        "rows": rows,
    }
    write_json(output_root / "visual_evidence_manifest.json", manifest)
    print(json.dumps({"visualized": len(rows), "manifest": str(output_root / "visual_evidence_manifest.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
