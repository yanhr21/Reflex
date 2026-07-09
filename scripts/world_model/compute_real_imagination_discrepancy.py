#!/usr/bin/env python3
"""Compute real-vs-imagined video discrepancy for Cosmos policy checks.

This is an evidence reducer. It does not run simulation, train a model,
generate Cosmos outputs, or decide task success. It compares an observed
reference video against an imagined/predicted video for the same causal window
and writes a trust signal that a closed-loop runner can log alongside actions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-video", required=True)
    parser.add_argument("--imagined-video", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--expected-frames", type=int, default=0)
    parser.add_argument("--trust-mae-threshold", type=float, default=0.08)
    parser.add_argument("--trust-psnr-threshold", type=float, default=18.0)
    parser.add_argument(
        "--allow-length-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If false, length mismatch makes the discrepancy report fail.",
    )
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def load_frames(path: Path, start_frame: int, max_frames: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    reader = imageio.get_reader(path)
    meta = reader.get_meta_data() or {}
    frames: list[np.ndarray] = []
    try:
        for idx, frame in enumerate(reader):
            if idx < start_frame:
                continue
            if max_frames > 0 and len(frames) >= max_frames:
                break
            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            if arr.shape[-1] == 4:
                arr = arr[:, :, :3]
            frames.append(arr.astype(np.float32) / 255.0)
    finally:
        reader.close()
    return frames, {
        "path": path,
        "fps": meta.get("fps"),
        "source_size": meta.get("source_size"),
        "loaded_frames": len(frames),
        "start_frame": int(start_frame),
        "max_frames": int(max_frames),
    }


def compare_frames(reference: list[np.ndarray], imagined: list[np.ndarray]) -> dict[str, Any]:
    count = min(len(reference), len(imagined))
    if count <= 0:
        return {
            "compared_frames": 0,
            "per_frame": [],
            "mean_mse_rgb01": None,
            "mean_mae_rgb01": None,
            "mean_psnr_db": None,
            "max_mae_rgb01": None,
        }
    per_frame: list[dict[str, Any]] = []
    mses: list[float] = []
    maes: list[float] = []
    psnrs: list[float] = []
    for idx in range(count):
        ref = reference[idx]
        pred = imagined[idx]
        if ref.shape != pred.shape:
            raise ValueError(f"frame_shape_mismatch at {idx}: {ref.shape} != {pred.shape}")
        diff = ref - pred
        mse = float(np.mean(diff * diff))
        mae = float(np.mean(np.abs(diff)))
        psnr = float("inf") if mse <= 0.0 else float(20.0 * math.log10(1.0 / math.sqrt(mse)))
        mses.append(mse)
        maes.append(mae)
        psnrs.append(psnr)
        per_frame.append(
            {
                "frame": idx,
                "mse_rgb01": mse,
                "mae_rgb01": mae,
                "psnr_db": psnr,
            }
        )
    finite_psnr = [value for value in psnrs if math.isfinite(value)]
    return {
        "compared_frames": count,
        "per_frame": per_frame,
        "mean_mse_rgb01": float(np.mean(mses)),
        "mean_mae_rgb01": float(np.mean(maes)),
        "mean_psnr_db": float(np.mean(finite_psnr)) if finite_psnr else float("inf"),
        "max_mae_rgb01": float(np.max(maes)),
    }


def write_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Real-Imagination Discrepancy",
        "",
        f"- status: `{report['status']}`",
        f"- trust state: `{report['trust_state']}`",
        f"- compared frames: `{report['metrics']['compared_frames']}`",
        f"- mean MAE RGB01: `{report['metrics']['mean_mae_rgb01']}`",
        f"- mean PSNR dB: `{report['metrics']['mean_psnr_db']}`",
        f"- length match: `{report['length_match']}`",
        "",
        "Boundary: this is a visual discrepancy/trust reducer only. It is not "
        "closed-loop control evidence or insertion success evidence.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    reference_path = Path(args.reference_video).resolve()
    imagined_path = Path(args.imagined_video).resolve()
    reference, reference_meta = load_frames(reference_path, int(args.start_frame), int(args.max_frames))
    imagined, imagined_meta = load_frames(imagined_path, int(args.start_frame), int(args.max_frames))
    length_match = len(reference) == len(imagined)
    metrics = compare_frames(reference, imagined)
    compared = int(metrics["compared_frames"])
    failures: list[str] = []
    if int(args.expected_frames) > 0 and compared != int(args.expected_frames):
        failures.append("compared_frame_count_mismatch")
    if not length_match and not args.allow_length_mismatch:
        failures.append("video_length_mismatch")
    if compared <= 0:
        failures.append("no_frames_compared")

    mean_mae = metrics["mean_mae_rgb01"]
    mean_psnr = metrics["mean_psnr_db"]
    trust_ok = (
        not failures
        and mean_mae is not None
        and float(mean_mae) <= float(args.trust_mae_threshold)
        and mean_psnr is not None
        and float(mean_psnr) >= float(args.trust_psnr_threshold)
    )
    report = {
        "status": "ok" if not failures else "failed",
        "trust_state": "trusted" if trust_ok else "untrusted",
        "trust_ok": bool(trust_ok),
        "failures": failures,
        "reference_video": reference_path,
        "imagined_video": imagined_path,
        "reference_meta": reference_meta,
        "imagined_meta": imagined_meta,
        "length_match": bool(length_match),
        "expected_frames": int(args.expected_frames),
        "thresholds": {
            "trust_mae_threshold": float(args.trust_mae_threshold),
            "trust_psnr_threshold": float(args.trust_psnr_threshold),
        },
        "metrics": metrics,
        "method_evidence_allowed": False,
        "closed_loop_evidence": False,
        "boundary": (
            "Real-imagination discrepancy for a matched window only. This does "
            "not execute controller actions and does not prove task success."
        ),
    }
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        write_md(report, Path(args.output_md).resolve())
    print(json.dumps({"status": report["status"], "trust_state": report["trust_state"], "output_json": str(output_json)}, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
