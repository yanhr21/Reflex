#!/usr/bin/env python3
"""Run task-state readout inspection over full-episode Cosmos3 eval videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--readout-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python-bin", default=str(ROOT / ".venv" / "bin" / "python"))
    parser.add_argument("--num-frames", type=int, default=301)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--motion-threshold-m", type=float, default=0.002)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def run_command(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    return int(proc.returncode)


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = eval_root / "eval_input_manifest.json"
    manifest = read_json(manifest_path)
    reference_rgb_calibration = bool(manifest.get("reference_rgb_calibration"))
    samples = list(manifest.get("samples") or [])
    if not samples:
        raise ValueError(f"no eval samples found in {manifest_path}")

    reports: list[dict[str, Any]] = []
    strict_failures: list[str] = []
    for sample in samples:
        name = str(sample["name"])
        sample_root = output_root / "samples" / name
        pred_video = eval_root / "inference" / name / "vision.mp4"
        sample_manifest_path = sample_root / "sample_manifest.json"
        prediction_json = sample_root / "readout_prediction.json"
        inspection_json = sample_root / "readout_inspection.json"
        inspection_md = sample_root / "readout_inspection.md"
        trajectory_json = sample_root / "readout_trajectory.json"
        sample_root.mkdir(parents=True, exist_ok=True)
        sample_payload = dict(sample)
        sample_payload["prefix_boundary_frame"] = int(sample.get("prefix_frame_index", -1))
        sample_payload["predicted_video_path"] = str(pred_video)
        sample_payload["readout_input_kind"] = (
            "reference_rgb_calibration" if reference_rgb_calibration else "cosmos3_generated_rgb"
        )
        sample_manifest_path.write_text(json.dumps(jsonable(sample_payload), indent=2, sort_keys=True) + "\n")

        failures: list[str] = []
        if not pred_video.is_file():
            failures.append("missing_prediction_video")
        if not Path(args.readout_checkpoint).is_file():
            failures.append("missing_readout_checkpoint")
        if failures:
            reports.append({"name": name, "failures": failures, "strict_readout_ok": False})
            strict_failures.extend(f"{name}:{failure}" for failure in failures)
            continue

        predict_cmd = [
            args.python_bin,
            str(ROOT / "scripts" / "world_model" / "train_cosmos3_task_state_readout.py"),
            "--dataset-manifest",
            str(ROOT / "experiments" / "world_model_task_rebinding" / "cosmos3" / "sft_dataset_full1000_maniskill_default_regen_20260606_0055" / "manifest.json"),
            "--output-dir",
            str(sample_root),
            "--checkpoint-path",
            str(args.readout_checkpoint),
            "--predict-video",
            str(pred_video),
            "--predict-output-json",
            str(prediction_json),
            "--num-frames",
            str(args.num_frames),
            "--image-size",
            str(args.image_size),
            "--require-exact-video-frames",
            "--require-cuda",
        ]
        predict_rc = run_command(predict_cmd, sample_root / "predict_readout.log")
        if predict_rc != 0:
            failures.append(f"predict_readout_rc:{predict_rc}")

        future_start = max(0, int(sample.get("prefix_frame_index", -1)) + 1)
        inspect_cmd = [
            args.python_bin,
            str(ROOT / "scripts" / "world_model" / "inspect_cosmos3_task_state_prediction.py"),
            "--prediction-json",
            str(prediction_json),
            "--output-json",
            str(inspection_json),
            "--output-md",
            str(inspection_md),
            "--trajectory-json",
            str(trajectory_json),
            "--future-start-frame",
            str(future_start),
            "--sample-manifest",
            str(sample_manifest_path),
            "--motion-threshold-m",
            str(args.motion_threshold_m),
            "--expected-num-frames",
            str(args.num_frames),
            "--require-finite",
            "--require-reference-metrics",
            "--require-external-target-event-metrics",
        ]
        inspect_rc = run_command(inspect_cmd, sample_root / "inspect_readout.log")
        if inspect_rc != 0:
            failures.append(f"inspect_readout_rc:{inspect_rc}")

        inspection = read_json(inspection_json) if inspection_json.is_file() else {}
        failures.extend(str(x) for x in inspection.get("strict_readout_failures", []))
        reports.append(
            {
                "name": name,
                "prefix_role": sample.get("prefix_role"),
                "scenario": sample.get("scenario"),
                "prefix_frame_index": sample.get("prefix_frame_index"),
                "prediction_json": str(prediction_json),
                "inspection_json": str(inspection_json),
                "inspection_md": str(inspection_md),
                "trajectory_json": str(trajectory_json),
                "strict_readout_ok": bool(inspection.get("strict_readout_ok")) and not failures,
                "reference_metrics": inspection.get("reference_metrics"),
                "external_target_event_metrics": inspection.get("external_target_event_metrics"),
                "failures": failures,
            }
        )
        strict_failures.extend(f"{name}:{failure}" for failure in failures)

    aggregate = {
        "num_samples": len(reports),
        "num_strict_ok": sum(1 for row in reports if row.get("strict_readout_ok") is True),
    }
    final_hole_errors = [
        float(row["external_target_event_metrics"]["final_hole_pos_error_m"])
        for row in reports
        if isinstance(row.get("external_target_event_metrics"), dict)
        and row["external_target_event_metrics"].get("final_hole_pos_error_m") is not None
    ]
    if final_hole_errors:
        aggregate["mean_final_hole_pos_error_m"] = sum(final_hole_errors) / len(final_hole_errors)

    boundary = (
        "Readout inspection over ground-truth reference RGB calibration videos. "
        "This decodes GT videos to calibrate task-state/onset diagnostics; it is "
        "not Cosmos3 world-model evidence and not controller success evidence."
        if reference_rgb_calibration
        else (
            "Readout inspection over Cosmos3-generated RGB. This decodes generated "
            "videos into task-state diagnostics; it is not a replacement world model "
            "and not controller success evidence."
        )
    )
    report = {
        "boundary": boundary,
        "eval_root": str(eval_root),
        "reference_rgb_calibration": reference_rgb_calibration,
        "readout_checkpoint": str(args.readout_checkpoint),
        "output_root": str(output_root),
        "strict_readout_eval_ok": not strict_failures,
        "strict_failures": strict_failures,
        "aggregate": aggregate,
        "samples": reports,
    }
    report_path = output_root / "readout_eval_summary.json"
    report_path.write_text(json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"strict_readout_eval_ok": report["strict_readout_eval_ok"], "strict_failures": strict_failures}, sort_keys=True))
    if args.strict and strict_failures:
        raise SystemExit("strict readout eval failed")


if __name__ == "__main__":
    main()
