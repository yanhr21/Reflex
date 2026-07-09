#!/usr/bin/env python3
"""Render close-up review videos by replaying a Phase 03 action trace.

This script is for visual review only. It starts from reset and replays the
recorded controller actions plus the recorded target-motion increments. It must
not edit peg state, restore simulator state, or create success evidence by
state intervention.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Pose


CAMERA_VIEWS = {
    "front": ([0.0, 0.72, 0.28], [0.0, 0.25, 0.10], [0.0, 0.0, 1.0], 0.78),
    "oblique": ([0.34, 0.66, 0.30], [0.0, 0.25, 0.10], [0.0, 0.0, 1.0], 0.72),
    "top": ([0.0, 0.24, 0.72], [0.0, 0.24, 0.10], [1.0, 0.0, 0.0], 0.72),
    "hole_front_close": ([0.04, 0.56, 0.18], [0.035, 0.27, 0.155], [0.0, 0.0, 1.0], 0.28),
    "hole_oblique_close": ([0.20, 0.48, 0.22], [0.035, 0.27, 0.155], [0.0, 0.0, 1.0], 0.34),
    "peg_side_close": ([0.28, 0.20, 0.20], [0.035, 0.22, 0.155], [0.0, 0.0, 1.0], 0.34),
    "goal_front_zoom": ([0.04, 0.78, 0.19], [0.04, 0.16, 0.16], [0.0, 0.0, 1.0], 0.24),
    "goal_oblique_zoom": ([0.24, 0.70, 0.23], [0.04, 0.16, 0.16], [0.0, 0.0, 1.0], 0.28),
    "goal_side_zoom": ([0.38, 0.24, 0.20], [0.04, 0.16, 0.16], [0.0, 0.0, 1.0], 0.30),
    "goal_back_zoom": ([0.04, 0.02, 0.20], [0.04, 0.16, 0.16], [0.0, 0.0, 1.0], 0.34),
    "goal_back_oblique_zoom": ([-0.18, 0.04, 0.22], [0.04, 0.16, 0.16], [0.0, 0.0, 1.0], 0.36),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--views", nargs="*", default=None)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def require_allocation() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("refusing_login_node_execution=true; run inside a tmux-held Slurm allocation")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def as_frame(frame: Any) -> np.ndarray:
    arr = np.asarray(jsonable(frame))
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max(initial=0) <= 1.5:
            arr *= 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


def annotate_frame(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)[..., :3]).convert("RGB")
    draw = ImageDraw.Draw(image)
    line_h = 14
    pad = 5
    box_h = pad * 2 + line_h * len(lines)
    draw.rectangle((0, 0, image.width, box_h), fill=(0, 0, 0))
    for idx, line in enumerate(lines):
        draw.text((pad, pad + idx * line_h), line[:130], fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def pose_list(eye: list[float], target: list[float], up: list[float]) -> list[float]:
    pose = sapien_utils.look_at(eye, target, up=up)
    return pose.raw_pose.detach().cpu().numpy().reshape(-1, 7)[0].tolist()


def make_env(view_name: str, args: argparse.Namespace):
    eye, target, up, fov = CAMERA_VIEWS[view_name]
    camera_config = {
        "render_camera": {
            "pose": pose_list(eye, target, up),
            "width": int(args.width),
            "height": int(args.height),
            "fov": float(fov),
            "near": 0.01,
            "far": 100,
            "shader_pack": "default",
        }
    }
    return gym.make(
        "PegInsertionSide-v1",
        reconfiguration_freq=1,
        control_mode="pd_ee_delta_pose",
        reward_mode="sparse",
        obs_mode="state",
        render_mode="rgb_array",
        human_render_camera_configs=camera_config,
        max_episode_steps=420,
    )


def eval_state(base_env: Any) -> dict[str, Any]:
    info = base_env.evaluate()
    rel = np.asarray(jsonable(info["peg_head_pos_at_hole"]), dtype=np.float64).reshape(-1)[:3]
    return {
        "success": bool(np.asarray(info["success"]).reshape(-1)[0]),
        "peg_head_at_hole": rel.astype(float).tolist(),
        "peg_head_l2": float(np.linalg.norm(rel)),
        "peg_pose": jsonable(base_env.peg.pose.raw_pose),
        "box_hole_pose": jsonable(base_env.box_hole_pose.raw_pose),
    }


def shift_box(base_env: Any, delta_xyz: np.ndarray) -> None:
    current = base_env.box.pose
    p = current.p + torch.as_tensor(delta_xyz, dtype=current.p.dtype, device=current.p.device).reshape(1, 3)
    base_env.box.set_pose(Pose.create_from_pq(p=p, q=current.q))


def apply_recorded_peg_force(base_env: Any, force_xyz: np.ndarray) -> None:
    if not np.any(force_xyz):
        return
    base_env.peg.apply_force(force_xyz.astype(np.float32))


def install_forbidden_peg_state_guard(base_env: Any) -> dict[str, Any]:
    peg = getattr(base_env, "peg", None)
    report: dict[str, Any] = {"ok": peg is not None, "guarded_methods": [], "failures": []}
    if peg is None:
        report["failures"].append("base_env_has_no_peg")
        return report

    def blocked(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("forbidden_peg_state_intervention_guard_triggered")

    for name in ("set_pose", "set_state", "set_state_dict"):
        if not hasattr(peg, name):
            continue
        try:
            setattr(peg, name, blocked)
            report["guarded_methods"].append(name)
        except Exception as exc:
            report["failures"].append({"method": name, "error_type": type(exc).__name__, "error": str(exc)})
    report["ok"] = bool(report["guarded_methods"]) and not report["failures"]
    return report


def render(env: Any) -> np.ndarray:
    return as_frame(common.to_numpy(env.unwrapped.render_rgb_array("render_camera")))


def replay_view(
    view_name: str,
    trace: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    env = make_env(view_name, args)
    frames: list[np.ndarray] = []
    annotated: list[np.ndarray] = []
    replay_trace: list[dict[str, Any]] = []
    try:
        env.reset(seed=args.seed)
        base_env = env.unwrapped
        guard = install_forbidden_peg_state_guard(base_env)
        if not guard["ok"]:
            raise RuntimeError(f"peg_state_guard_failed={guard}")
        frames.append(render(env))
        annotated.append(annotate_frame(frames[-1], [f"view={view_name} frame=0 reset"]))

        for row in trace:
            delta = np.asarray(row.get("target_motion_delta_xyz") or [0.0, 0.0, 0.0], dtype=np.float64).reshape(3)
            peg_force = np.asarray(row.get("peg_perturb_force_xyz") or [0.0, 0.0, 0.0], dtype=np.float64).reshape(3)
            if np.any(delta):
                shift_box(base_env, delta)
            apply_recorded_peg_force(base_env, peg_force)
            if row.get("stage") != "target_motion_trigger_no_robot_action":
                action = np.asarray(row["action"], dtype=np.float32)
                env.step(action)
            state = eval_state(base_env)
            frame = render(env)
            frames.append(frame)
            annotated.append(
                annotate_frame(
                    frame,
                    [
                        f"view={view_name} replay_frame={len(frames)-1} env_step={row.get('env_step')} stage={row.get('stage')}",
                        f"source={row.get('action_source')} peg_head_l2={state['peg_head_l2']:.6f}",
                    ],
                )
            )
            replay_trace.append(
                {
                    "source_env_step": row.get("env_step"),
                    "source_stage": row.get("stage"),
                    "source_action": row.get("action"),
                    "target_motion_delta_xyz": delta.astype(float).tolist(),
                    "peg_perturb_force_xyz": peg_force.astype(float).tolist(),
                    "replay_eval": state,
                }
            )
            if bool(state["success"]):
                break

        view_dir = output_dir / view_name
        raw_path = view_dir / "raw.mp4"
        annotated_path = view_dir / "annotated.mp4"
        trace_path = view_dir / "trace.json"
        write_mp4(raw_path, frames, args.fps)
        write_mp4(annotated_path, annotated, args.fps)
        write_json(trace_path, replay_trace)
        final_state = replay_trace[-1]["replay_eval"] if replay_trace else eval_state(base_env)
        return {
            "view": view_name,
            "ok": True,
            "video": raw_path,
            "annotated_video": annotated_path,
            "trace": trace_path,
            "num_frames": len(frames),
            "final_eval": final_state,
            "peg_state_guard": guard,
        }
    finally:
        env.close()


def main() -> int:
    require_allocation()
    args = parse_args()
    source_run_dir = Path(args.source_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else source_run_dir / "review"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = read_json(source_run_dir / "action_trace.json")
    summary = read_json(source_run_dir / "summary.json")

    report: dict[str, Any] = {
        "schema": "phase03_visual_review_replay_v1",
        "source_run_dir": source_run_dir,
        "output_dir": output_dir,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.uname().nodename,
        "seed": args.seed,
        "source_classification": summary.get("classification"),
        "source_simulator_success_metric": summary.get("simulator_success_metric"),
        "method_evidence_allowed": False,
        "physical_insertion_success_claimed": False,
        "visual_full_insertion_confirmed": False,
        "state_intervention_note": "Recorded target/hole motion increments and recorded peg perturb forces are replayed; peg state is guarded against set_pose/set_state/set_state_dict.",
        "views": [],
    }
    view_names = args.views if args.views else list(CAMERA_VIEWS)
    unknown_views = sorted(set(view_names) - set(CAMERA_VIEWS))
    if unknown_views:
        raise ValueError(f"unknown_views={unknown_views} available={sorted(CAMERA_VIEWS)}")
    for view_name in view_names:
        report["views"].append(replay_view(view_name, trace, output_dir, args))
    write_json(output_dir / "visual_review_replay_manifest.json", report)
    print(json.dumps(jsonable(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
