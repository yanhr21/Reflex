#!/usr/bin/env python3
"""Prepare Phase 02 RGB video2video inputs from approved ManiSkill H5 files.

This script is an in-allocation diagnostic utility. It reads approved H5
artifacts, extracts existing RGB frames when present, and writes Cosmos-3 V2V
sample JSON files plus visual review artifacts. It does not replay control,
edit simulator state, or claim insertion success.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


DEFAULT_SCENARIOS = [
    "none",
    "hole_late_fast_shift",
    "hole_late_reverse",
    "hole_late_continuous_insert",
    "peg_drop",
    "peg_disturb",
]


@dataclass
class RgbCandidate:
    path: str
    shape: tuple[int, ...]
    dtype: str


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


def scenario_for_path(path: Path) -> str:
    name = path.name
    for scenario in sorted(DEFAULT_SCENARIOS, key=len, reverse=True):
        if name.startswith(scenario):
            return scenario
    if name.startswith("hole_late_move_stop"):
        return "hole_late_move_stop"
    if name.startswith("hole_late_sine"):
        return "hole_late_sine"
    if name.startswith("hole_late_constant"):
        return "hole_late_constant"
    return name.split("_seed", 1)[0]


def select_h5s(data_root: Path, scenarios: list[str], samples_per_scenario: int) -> list[Path]:
    h5_root = data_root / "canonical_h5"
    all_h5 = sorted(h5_root.glob("*/*.h5"))
    selected: list[Path] = []
    for scenario in scenarios:
        matches = [path for path in all_h5 if path.name.startswith(scenario)]
        selected.extend(matches[:samples_per_scenario])
    return selected


def collect_h5_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"path": str(path), "datasets": [], "groups": []}
    with h5py.File(path, "r") as h5:
        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                summary["datasets"].append(
                    {"name": name, "shape": list(obj.shape), "dtype": str(obj.dtype)}
                )
            elif isinstance(obj, h5py.Group):
                summary["groups"].append(name)

        h5.visititems(visitor)
    return summary


def find_rgb_candidates(path: Path) -> list[RgbCandidate]:
    candidates: list[RgbCandidate] = []
    with h5py.File(path, "r") as h5:
        def visitor(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            lname = name.lower()
            if not (lname.endswith("/rgb") or lname == "rgb" or "/rgb/" in lname):
                return
            shape = tuple(int(x) for x in obj.shape)
            if len(shape) < 4:
                return
            if shape[-1] in (3, 4) or (len(shape) >= 4 and shape[-3] in (3, 4)):
                candidates.append(RgbCandidate(path=name, shape=shape, dtype=str(obj.dtype)))
    return candidates


def normalize_rgb_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 5 and arr.shape[-1] in (3, 4):
        # Common layouts are [T, N, H, W, C] or [N, T, H, W, C].
        arr = arr[:, 0] if arr.shape[0] >= arr.shape[1] else arr[0]
    elif arr.ndim == 5 and arr.shape[-3] in (3, 4):
        arr = arr[:, 0] if arr.shape[0] >= arr.shape[1] else arr[0]
        arr = np.moveaxis(arr, -3, -1)
    elif arr.ndim == 4 and arr.shape[-1] in (3, 4):
        pass
    elif arr.ndim == 4 and arr.shape[1] in (3, 4):
        arr = np.moveaxis(arr, 1, -1)
    else:
        raise ValueError(f"Unsupported RGB dataset shape: {arr.shape}")

    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max(initial=0) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def read_rgb_frames(h5_path: Path, dataset_path: str, max_frames: int) -> np.ndarray:
    with h5py.File(h5_path, "r") as h5:
        ds = h5[dataset_path]
        if max_frames > 0 and len(ds.shape) > 0:
            raw = ds[:max_frames]
        else:
            raw = ds[()]
    frames = normalize_rgb_array(raw)
    if max_frames > 0:
        frames = frames[:max_frames]
    return frames


def write_mp4(frames: np.ndarray, out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        with imageio.get_writer(out_path, fps=fps, codec="libx264", macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame, dtype=np.uint8))
        return
    with tempfile.TemporaryDirectory(prefix="phase02_frames_") as tmp:
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


def save_overlay_frames(frames: np.ndarray, out_dir: Path, label: str, limit: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    if len(frames) == 0:
        return written
    indexes = np.linspace(0, len(frames) - 1, num=min(limit, len(frames)), dtype=int)
    for idx in indexes:
        image = Image.fromarray(frames[idx]).convert("RGB")
        draw = ImageDraw.Draw(image)
        text = f"{label} frame={idx}"
        draw.rectangle((4, 4, min(image.width - 4, 4 + 8 * len(text)), 24), fill=(0, 0, 0))
        draw.text((8, 8), text, fill=(255, 255, 255))
        path = out_dir / f"frame_{idx:05d}.png"
        image.save(path)
        written.append(str(path))
    return written


def write_frame_stats_csv(frames: np.ndarray, out_path: Path, source_label: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "source",
                "rgb_mean_r",
                "rgb_mean_g",
                "rgb_mean_b",
                "task_state_status",
                "notes",
            ],
        )
        writer.writeheader()
        for idx, frame in enumerate(frames):
            mean = frame.reshape(-1, 3).mean(axis=0)
            writer.writerow(
                {
                    "frame": idx,
                    "source": source_label,
                    "rgb_mean_r": float(mean[0]),
                    "rgb_mean_g": float(mean[1]),
                    "rgb_mean_b": float(mean[2]),
                    "task_state_status": "rgb_available_no_geometry_extractor_yet",
                    "notes": "Phase02 visual/RGB evidence only; not simulator-state success evidence.",
                }
            )


def prompt_for_scenario(scenario: str) -> str:
    return (
        "A simulated robot arm performs side peg insertion in ManiSkill. "
        f"The scenario is {scenario}. Continue the RGB video consistently from the observed prefix, "
        "preserving the peg, gripper, hole target, table, camera viewpoint, and contact geometry. "
        "Do not introduce teleportation, object disappearance, or discontinuous motion."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="experiments/maniskill/data/fix3_733")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--samples-per-scenario", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--prefix-frames", type=int, default=16)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--cosmos-num-frames", type=int, default=121)
    parser.add_argument("--cosmos-resolution", default="256")
    parser.add_argument("--cosmos-aspect-ratio", default="1,1")
    parser.add_argument("--overlay-limit", type=int, default=8)
    args = parser.parse_args()

    require_slurm_step()

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    selected_dir = output_dir / "selected_rgb"
    input_dir = output_dir / "cosmos_inputs"
    report_dir = output_dir / "reports"
    selected_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    h5_paths = select_h5s(data_root, scenarios, args.samples_per_scenario)
    (report_dir / "selected_h5_paths.txt").write_text("\n".join(str(p) for p in h5_paths) + "\n")

    manifest: dict[str, Any] = {
        "phase": "02_cosmos_imagination",
        "evidence_type": "rgb_h5_to_cosmos_v2v_input_preparation",
        "method_evidence_allowed": False,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "node": os.uname().nodename,
        "scenarios": scenarios,
        "samples": [],
        "forbidden_state_intervention_used": False,
        "notes": [
            "This script reads existing RGB datasets from H5 only.",
            "It does not call set_pose, set_state, set_state_dict, source-state restore, saved-state replay, or Oracle final-seat.",
            "Frame-stat charts are RGB diagnostics, not task-state extraction claims.",
        ],
    }

    h5_summaries = []
    jsonl_rows = []
    blockers = []
    for h5_path in h5_paths:
        scenario = scenario_for_path(h5_path)
        sample_name = h5_path.stem
        try:
            h5_summaries.append(collect_h5_summary(h5_path))
            candidates = find_rgb_candidates(h5_path)
            sample_record: dict[str, Any] = {
                "name": sample_name,
                "scenario": scenario,
                "h5_path": str(h5_path),
                "rgb_candidates": [candidate.__dict__ for candidate in candidates],
            }
            if not candidates:
                sample_record["status"] = "blocked_no_rgb_dataset_found"
                blockers.append(sample_record)
                manifest["samples"].append(sample_record)
                continue

            candidate = candidates[0]
            frames = read_rgb_frames(h5_path, candidate.path, args.max_frames)
            if len(frames) == 0:
                sample_record["status"] = "blocked_empty_rgb_dataset"
                blockers.append(sample_record)
                manifest["samples"].append(sample_record)
                continue

            sample_dir = selected_dir / sample_name
            observed_mp4 = sample_dir / "observed_rgb.mp4"
            prefix_mp4 = sample_dir / "prefix_rgb.mp4"
            overlay_dir = sample_dir / "overlays"
            stats_csv = sample_dir / "rgb_frame_stats.csv"
            write_mp4(frames, observed_mp4, args.fps)
            write_mp4(frames[: args.prefix_frames], prefix_mp4, args.fps)
            overlays = save_overlay_frames(frames, overlay_dir, sample_name, args.overlay_limit)
            write_frame_stats_csv(frames, stats_csv, "h5_rgb_extracted")

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
            sample_json_path = input_dir / f"{sample_name}.json"
            sample_json_path.write_text(json.dumps(sample_json, indent=2, sort_keys=True))
            jsonl_rows.append(sample_json)

            sample_record.update(
                {
                    "status": "rgb_ready",
                    "chosen_rgb_dataset": candidate.__dict__,
                    "num_frames": int(len(frames)),
                    "observed_rgb_mp4": str(observed_mp4),
                    "prefix_rgb_mp4": str(prefix_mp4),
                    "cosmos_sample_json": str(sample_json_path),
                    "overlay_frames": overlays,
                    "rgb_frame_stats_csv": str(stats_csv),
                }
            )
            manifest["samples"].append(sample_record)
        except Exception as exc:  # Keep diagnostics for every selected sample.
            record = {
                "name": sample_name,
                "scenario": scenario,
                "h5_path": str(h5_path),
                "status": "error",
                "error": repr(exc),
            }
            blockers.append(record)
            manifest["samples"].append(record)

    (report_dir / "h5_structure_summary.json").write_text(json.dumps(h5_summaries, indent=2))
    (report_dir / "phase02_rgb_input_manifest.json").write_text(json.dumps(jsonable(manifest), indent=2, sort_keys=True))
    if jsonl_rows:
        with (input_dir / "samples.jsonl").open("w") as f:
            for row in jsonl_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    if blockers:
        (report_dir / "blockers.json").write_text(json.dumps(jsonable(blockers), indent=2, sort_keys=True))

    ready = sum(1 for sample in manifest["samples"] if sample.get("status") == "rgb_ready")
    print(json.dumps({"selected": len(h5_paths), "rgb_ready": ready, "blockers": len(blockers)}, indent=2))
    if ready == 0:
        raise SystemExit(42)


if __name__ == "__main__":
    main()
