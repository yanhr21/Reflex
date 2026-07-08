#!/usr/bin/env python3
"""Render approved H5 env states to RGB for Phase 02 Cosmos diagnostics.

This is a simulator-state audit renderer, not a controller or success path. It
uses H5 env states to render RGB history when the approved H5 files do not
already contain RGB frames. Outputs must be labeled as diagnostic simulator
state audit evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ["DISPLAY"] = ""

import gymnasium as gym
import h5py
import mani_skill.envs  # noqa: F401
import numpy as np
from PIL import Image, ImageDraw

from mani_skill.trajectory import utils as trajectory_utils


DEFAULT_SCENARIOS = [
    "none",
    "hole_late_fast_shift",
    "hole_late_reverse",
    "hole_late_continuous_insert",
    "peg_drop",
    "peg_disturb",
]


def require_slurm_step() -> None:
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise SystemExit("refusing_login_node_execution=true; run inside a compute-node srun step")
    if os.environ.get("SLURM_STEP_ID") == "extern":
        raise SystemExit("refusing_extern_step=true; run inside an active compute-node srun step")


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def select_h5s(data_root: Path, scenarios: list[str], samples_per_scenario: int) -> list[Path]:
    h5_root = data_root / "canonical_h5"
    all_h5 = sorted(h5_root.glob("*/*.h5"))
    selected: list[Path] = []
    for scenario in scenarios:
        matches = [path for path in all_h5 if path.name.startswith(scenario)]
        selected.extend(matches[:samples_per_scenario])
    return selected


def scenario_for_path(path: Path) -> str:
    name = path.name
    for scenario in sorted(DEFAULT_SCENARIOS, key=len, reverse=True):
        if name.startswith(scenario):
            return scenario
    return name.split("_seed", 1)[0]


def write_mp4(frames: np.ndarray, out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found; cannot write mp4")
    with tempfile.TemporaryDirectory(prefix="phase02_state_render_") as tmp:
        tmp_path = Path(tmp)
        for idx, frame in enumerate(frames):
            Image.fromarray(frame).save(tmp_path / f"frame_{idx:05d}.png")
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_path / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)


def as_rgb_frame(frame: Any) -> np.ndarray:
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = frame.astype(np.float32)
        if frame.max(initial=0) <= 1.5:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return np.ascontiguousarray(frame)


def save_overlay_frames(frames: np.ndarray, out_dir: Path, label: str, limit: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    indexes = np.linspace(0, len(frames) - 1, num=min(limit, len(frames)), dtype=int)
    for idx in indexes:
        image = Image.fromarray(frames[idx]).convert("RGB")
        draw = ImageDraw.Draw(image)
        text = f"{label} state-audit frame={idx}"
        draw.rectangle((4, 4, min(image.width - 4, 4 + 8 * len(text)), 24), fill=(0, 0, 0))
        draw.text((8, 8), text, fill=(255, 255, 255))
        path = out_dir / f"frame_{idx:05d}.png"
        image.save(path)
        written.append(str(path))
    return written


def read_slot(group: h5py.Group, name: str) -> np.ndarray | None:
    path = f"traj_0/slots/{name}"
    if path not in group:
        return None
    return group[path][()]


def write_task_state_chart(h5_path: Path, out_path: Path, max_frames: int) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as h5:
        peg_head = read_slot(h5, "peg_head_at_hole")
        hole_pose = read_slot(h5, "hole_pose")
        peg_pose = read_slot(h5, "peg_pose")
        inserted = read_slot(h5, "inserted")
        triggered = h5["traj_0/perturb/triggered"][()] if "traj_0/perturb/triggered" in h5 else None
        hole_delta = (
            h5["traj_0/perturb/hole_delta_cumulative"][()]
            if "traj_0/perturb/hole_delta_cumulative" in h5
            else None
        )
        peg_delta = (
            h5["traj_0/perturb/peg_delta_applied"][()]
            if "traj_0/perturb/peg_delta_applied" in h5
            else None
        )

    n = max(
        len(x)
        for x in [peg_head, hole_pose, peg_pose, inserted]
        if x is not None
    )
    n = min(n, max_frames)
    trigger_frame = None
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "source",
                "peg_head_at_hole_x",
                "peg_head_at_hole_y",
                "peg_head_at_hole_z",
                "hole_x",
                "hole_y",
                "hole_z",
                "peg_x",
                "peg_y",
                "peg_z",
                "inserted_state_label",
                "perturb_triggered",
                "hole_delta_x",
                "hole_delta_y",
                "hole_delta_z",
                "peg_delta_x",
                "peg_delta_y",
                "peg_delta_z",
            ],
        )
        writer.writeheader()
        for idx in range(n):
            trig = bool(triggered[idx]) if triggered is not None and idx < len(triggered) else False
            if trig and trigger_frame is None:
                trigger_frame = idx
            ph = peg_head[idx] if peg_head is not None and idx < len(peg_head) else [np.nan] * 3
            hp = hole_pose[idx, :3] if hole_pose is not None and idx < len(hole_pose) else [np.nan] * 3
            pp = peg_pose[idx, :3] if peg_pose is not None and idx < len(peg_pose) else [np.nan] * 3
            hd = hole_delta[idx] if hole_delta is not None and idx < len(hole_delta) else [np.nan] * 3
            pd = peg_delta[idx] if peg_delta is not None and idx < len(peg_delta) else [np.nan] * 3
            writer.writerow(
                {
                    "frame": idx,
                    "source": "diagnostic_simulator_state_audit",
                    "peg_head_at_hole_x": float(ph[0]),
                    "peg_head_at_hole_y": float(ph[1]),
                    "peg_head_at_hole_z": float(ph[2]),
                    "hole_x": float(hp[0]),
                    "hole_y": float(hp[1]),
                    "hole_z": float(hp[2]),
                    "peg_x": float(pp[0]),
                    "peg_y": float(pp[1]),
                    "peg_z": float(pp[2]),
                    "inserted_state_label": bool(inserted[idx]) if inserted is not None and idx < len(inserted) else "",
                    "perturb_triggered": trig,
                    "hole_delta_x": float(hd[0]),
                    "hole_delta_y": float(hd[1]),
                    "hole_delta_z": float(hd[2]),
                    "peg_delta_x": float(pd[0]),
                    "peg_delta_y": float(pd[1]),
                    "peg_delta_z": float(pd[2]),
                }
            )
    return {"chart_csv": str(out_path), "trigger_frame": trigger_frame, "num_rows": n}


def prompt_for_scenario(scenario: str) -> str:
    return (
        "A simulated robot arm performs side peg insertion in ManiSkill. "
        f"The scenario is {scenario}. Continue the state-audit RGB prefix consistently. "
        "Preserve the peg, gripper, moving hole target, table, camera viewpoint, and contact geometry. "
        "Avoid teleportation, penetration, object disappearance, or discontinuous motion."
    )


def render_one(env: Any, h5_path: Path, out_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    sample_name = h5_path.stem
    scenario = scenario_for_path(h5_path)
    sample_dir = out_root / "state_audit_rgb" / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as h5:
        states = trajectory_utils.dict_to_list_of_dicts(h5["traj_0/env_states"])
        states = states[: args.max_frames]

        reset_seed = 0
        env.reset(seed=reset_seed)
        frames = []
        for state in states:
            env.unwrapped.set_state_dict(state)
            frames.append(as_rgb_frame(env.unwrapped.render_rgb_array()))
        frames_arr = np.stack(frames, axis=0)

    observed_mp4 = sample_dir / "observed_state_audit_rgb.mp4"
    prefix_mp4 = sample_dir / "prefix_state_audit_rgb.mp4"
    chart_csv = sample_dir / "task_state_chart.csv"
    write_mp4(frames_arr, observed_mp4, args.fps)
    write_mp4(frames_arr[: args.prefix_frames], prefix_mp4, args.fps)
    overlays = save_overlay_frames(frames_arr, sample_dir / "overlays", sample_name, args.overlay_limit)
    chart_info = write_task_state_chart(h5_path, chart_csv, args.max_frames)

    sample_json = {
        "name": sample_name,
        "model_mode": "video2video",
        "prompt": prompt_for_scenario(scenario),
        "vision_path": str(prefix_mp4),
        "fps": args.fps,
        "num_frames": args.cosmos_num_frames,
        "resolution": args.cosmos_resolution,
        "aspect_ratio": args.cosmos_aspect_ratio,
        "condition_video_keep": "first",
        "condition_frame_indexes_vision": [0, 1],
        "seed": 0,
        "prompt_upsampling": False,
    }
    input_dir = out_root / "cosmos_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    sample_json_path = input_dir / f"{sample_name}.json"
    sample_json_path.write_text(json.dumps(sample_json, indent=2, sort_keys=True))

    return {
        "name": sample_name,
        "scenario": scenario,
        "h5_path": str(h5_path),
        "status": "state_audit_rgb_ready",
        "source_label": "diagnostic_simulator_state_audit",
        "uses_set_state_dict_for_rendering": True,
        "method_evidence_allowed": False,
        "num_frames": int(len(frames_arr)),
        "observed_rgb_mp4": str(observed_mp4),
        "prefix_rgb_mp4": str(prefix_mp4),
        "cosmos_sample_json": str(sample_json_path),
        "overlay_frames": overlays,
        "task_state_chart": chart_info,
        "reset_seed_for_renderer_initialization": reset_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="experiments/maniskill/data/fix3_733")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--samples-per-scenario", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--prefix-frames", type=int, default=16)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--cosmos-num-frames", type=int, default=121)
    parser.add_argument("--cosmos-resolution", default="256")
    parser.add_argument("--cosmos-aspect-ratio", default="1,1")
    parser.add_argument("--overlay-limit", type=int, default=8)
    parser.add_argument("--render-source-label", default="diagnostic_simulator_state_audit")
    args = parser.parse_args()

    require_slurm_step()

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    h5_paths = select_h5s(data_root, scenarios, args.samples_per_scenario)

    manifest: dict[str, Any] = {
        "phase": "02_cosmos_imagination",
        "evidence_type": "state_audit_rgb_render_for_cosmos_v2v",
        "method_evidence_allowed": False,
        "uses_set_state_dict_for_rendering": True,
        "controller_facing_state_intervention_used": False,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "node": os.uname().nodename,
        "samples": [],
        "notes": [
            "State setting is used only to render historical approved H5 states into RGB for Phase02 diagnostic Cosmos input.",
            "This output is not physical insertion success, not Oracle success, and not method evidence.",
            "Controller/live phases must not use this state-audit render path as execution evidence.",
        ],
    }

    env = gym.make(
        "PegInsertionSide-v1",
        reconfiguration_freq=1,
        control_mode="pd_ee_delta_pose",
        reward_mode="sparse",
        obs_mode="state",
        render_mode="rgb_array",
        human_render_camera_configs=dict(
            shader_pack="default",
            width=args.width,
            height=args.height,
        ),
        max_episode_steps=300,
    )
    try:
        for h5_path in h5_paths:
            try:
                manifest["samples"].append(render_one(env, h5_path, output_dir, args))
            except Exception as exc:
                manifest["samples"].append(
                    {
                        "name": h5_path.stem,
                        "h5_path": str(h5_path),
                        "status": "error",
                        "error": repr(exc),
                    }
                )
    finally:
        env.close()

    (report_dir / "phase02_state_audit_rgb_manifest.json").write_text(
        json.dumps(jsonable(manifest), indent=2, sort_keys=True)
    )
    ready = sum(1 for sample in manifest["samples"] if sample.get("status") == "state_audit_rgb_ready")
    print(json.dumps({"selected": len(h5_paths), "state_audit_rgb_ready": ready}, indent=2))
    if ready == 0:
        raise SystemExit(44)


if __name__ == "__main__":
    main()
