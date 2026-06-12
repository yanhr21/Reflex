#!/usr/bin/env python3
"""Inspect a Cosmos3 video -> task-state readout export.

This is controller-interface tooling. It checks that the readout prediction is
finite, computes all/future task-state reconstruction metrics when reference
labels are available, and writes a compact trajectory JSON that a controller
adapter can consume. It is not a task-success gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
WORLD_MODEL_DIR = ROOT / "scripts" / "world_model"
if str(WORLD_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(WORLD_MODEL_DIR))

import train_cosmos3_task_state_readout as readout  # noqa: E402


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _rmse(pred: np.ndarray, target: np.ndarray, cols: slice | list[int]) -> float:
    pred_t = torch.as_tensor(pred[:, cols], dtype=torch.float32)
    target_t = torch.as_tensor(target[:, cols], dtype=torch.float32)
    return float(torch.sqrt(torch.mean((pred_t - target_t) ** 2)).item())


def _accuracy(prob: np.ndarray, target: np.ndarray, col: int) -> float:
    pred_b = torch.as_tensor(prob[:, col] >= 0.5)
    target_b = torch.as_tensor(target[:, col] >= 0.5)
    return float((pred_b == target_b).float().mean().item())


def _metrics(pred_cont: np.ndarray, pred_bin: np.ndarray, target_cont: np.ndarray, target_bin: np.ndarray) -> dict[str, Any]:
    return {
        "num_frames": int(pred_cont.shape[0]),
        "hole_pos_rmse_m": _rmse(pred_cont, target_cont, slice(0, 3)),
        "peg_pos_rmse_m": _rmse(pred_cont, target_cont, slice(7, 10)),
        "tcp_pos_rmse_m": _rmse(pred_cont, target_cont, slice(14, 17)),
        "peg_head_hole_rmse_m": _rmse(pred_cont, target_cont, slice(21, 24)),
        "grasped_accuracy": _accuracy(pred_bin, target_bin, 0),
        "inserted_accuracy": _accuracy(pred_bin, target_bin, 1),
    }


def _first_motion_frame(hole_pos: np.ndarray, threshold_m: float) -> int | None:
    if hole_pos.shape[0] == 0:
        return None
    displacement = np.linalg.norm(hole_pos - hole_pos[0:1], axis=1)
    hits = np.flatnonzero(displacement >= float(threshold_m))
    if hits.size == 0:
        return None
    return int(hits[0])


def _motion_event_metrics(
    pred_cont: np.ndarray,
    target_cont: np.ndarray,
    reference_start: int,
    future_start: int,
    threshold_m: float,
) -> dict[str, Any]:
    pred_hole = np.asarray(pred_cont[:, 0:3], dtype=np.float32)
    target_hole = np.asarray(target_cont[:, 0:3], dtype=np.float32)
    pred_onset = _first_motion_frame(pred_hole, threshold_m)
    target_onset = _first_motion_frame(target_hole, threshold_m)
    final_hole_error = float(np.linalg.norm(pred_hole[-1] - target_hole[-1]))
    final_insert_point_error = float(np.linalg.norm(pred_cont[-1, 21:24] - target_cont[-1, 21:24]))
    report: dict[str, Any] = {
        "threshold_m": float(threshold_m),
        "predicted_motion_onset_frame": None if pred_onset is None else int(reference_start + pred_onset),
        "target_motion_onset_frame": None if target_onset is None else int(reference_start + target_onset),
        "predicted_motion_onset_index": pred_onset,
        "target_motion_onset_index": target_onset,
        "motion_onset_abs_error_frames": (
            None if pred_onset is None or target_onset is None else int(abs(pred_onset - target_onset))
        ),
        "predicted_final_hole_pos": pred_hole[-1].astype(float).tolist(),
        "target_final_hole_pos": target_hole[-1].astype(float).tolist(),
        "final_hole_pos_error_m": final_hole_error,
        "predicted_final_peg_head_at_hole": pred_cont[-1, 21:24].astype(float).tolist(),
        "target_final_peg_head_at_hole": target_cont[-1, 21:24].astype(float).tolist(),
        "final_insert_point_error_m": final_insert_point_error,
        "future_start_frame": int(reference_start + future_start),
        "boundary": (
            "Advisory external-target diagnostics: motion onset, final hole "
            "position, and peg-head/insertion-point geometry are used to inspect "
            "whether the world model/readout saw the target object move. They are "
            "not standalone task success or controller evidence."
        ),
    }
    if future_start < pred_cont.shape[0]:
        pred_future = pred_hole[future_start:]
        target_future = target_hole[future_start:]
        report["future_final_hole_pos_error_m"] = final_hole_error
        report["future_hole_path_rmse_m"] = float(np.sqrt(np.mean((pred_future - target_future) ** 2)))
        report["future_insert_point_rmse_m"] = float(
            np.sqrt(np.mean((pred_cont[future_start:, 21:24] - target_cont[future_start:, 21:24]) ** 2))
        )
    return report


def _trajectory(pred_cont: np.ndarray, pred_bin: np.ndarray, frame_start: int = 0) -> list[dict[str, Any]]:
    rows = []
    for frame_idx in range(pred_cont.shape[0]):
        row = pred_cont[frame_idx]
        bins = pred_bin[frame_idx]
        rows.append(
            {
                "frame": int(frame_start) + int(frame_idx),
                "hole_pose": row[0:7].astype(float).tolist(),
                "peg_pose": row[7:14].astype(float).tolist(),
                "tcp_pose": row[14:21].astype(float).tolist(),
                "peg_head_at_hole": row[21:24].astype(float).tolist(),
                "hole_radius": float(row[24]),
                "grasped_probability": float(bins[0]),
                "inserted_probability": float(bins[1]),
            }
        )
    return rows


def _state_targets_from_json(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    names = list(data.get("state_vector_names") or [])
    states = np.asarray(data.get("states"), dtype=np.float32)
    if states.ndim != 2:
        raise ValueError(f"state target JSON states must be rank-2: {path}, shape={states.shape}")
    if states.shape[1] != len(names):
        raise ValueError(
            f"state target JSON width mismatch: {path}, states_shape={states.shape}, num_names={len(names)}"
        )
    index = {name: idx for idx, name in enumerate(names)}

    def cols(required: list[str]) -> np.ndarray:
        missing = [name for name in required if name not in index]
        if missing:
            raise ValueError(f"state target JSON missing columns {missing}: {path}")
        return states[:, [index[name] for name in required]]

    hole = cols(["hole_pose_x", "hole_pose_y", "hole_pose_z", "hole_pose_qw", "hole_pose_qx", "hole_pose_qy", "hole_pose_qz"])
    peg = cols(["peg_pose_x", "peg_pose_y", "peg_pose_z", "peg_pose_qw", "peg_pose_qx", "peg_pose_qy", "peg_pose_qz"])
    tcp = cols(["tcp_pose_x", "tcp_pose_y", "tcp_pose_z", "tcp_pose_qw", "tcp_pose_qx", "tcp_pose_qy", "tcp_pose_qz"])
    peg_head = cols(["peg_head_at_hole_x", "peg_head_at_hole_y", "peg_head_at_hole_z"])
    if "hole_radius" in index:
        hole_radius = states[:, [index["hole_radius"]]]
    else:
        hole_radius = np.zeros((states.shape[0], 1), dtype=np.float32)
    bins = cols(["grasped", "inserted"])
    cont = np.concatenate([hole, peg, tcp, peg_head, hole_radius], axis=1).astype(np.float32)
    return cont, bins.astype(np.float32)


def _write_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Cosmos3 Task-State Prediction Inspection",
        "",
        f"- prediction_json: `{report['prediction_json']}`",
        f"- video_path: `{report.get('video_path', '')}`",
        f"- num_frames: `{report['num_frames']}`",
        f"- finite_prediction: `{report['finite_prediction']}`",
        f"- strict readout failures: `{report.get('strict_readout_failures')}`",
        f"- boundary: `{report['boundary']}`",
    ]
    if "reference_metrics" in report:
        lines += ["", "## Metrics", ""]
        for name in ("all", "future"):
            metrics = report["reference_metrics"][name]
            lines.append(
                f"- {name}: hole `{metrics['hole_pos_rmse_m']:.6f}` m, "
                f"peg `{metrics['peg_pos_rmse_m']:.6f}` m, "
                f"tcp `{metrics['tcp_pos_rmse_m']:.6f}` m, "
                f"peg_head_hole `{metrics['peg_head_hole_rmse_m']:.6f}` m, "
                f"grasp_acc `{metrics['grasped_accuracy']:.6f}`"
            )
    if "external_target_event_metrics" in report:
        event = report["external_target_event_metrics"]
        lines += [
            "",
            "## External Target Event",
            "",
            f"- motion threshold: `{event['threshold_m']:.6f}` m",
            f"- predicted onset frame: `{event['predicted_motion_onset_frame']}`",
            f"- target onset frame: `{event['target_motion_onset_frame']}`",
            f"- onset error: `{event['motion_onset_abs_error_frames']}` frames",
            f"- final hole position error: `{event['final_hole_pos_error_m']:.6f}` m",
            f"- final insertion geometry error: `{event['final_insert_point_error_m']:.6f}` m",
            f"- future hole path RMSE: `{event.get('future_hole_path_rmse_m', float('nan')):.6f}` m",
        ]
    lines += [
        "",
        "## Controller Interface",
        "",
        "- source: Cosmos3 generated video decoded by the task-state readout.",
        "- use: candidate adapter input for controller development.",
        "- non-use: not task success, not a hard gate, and not oracle-state evidence.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--trajectory-json", default=None)
    parser.add_argument("--future-start-frame", type=int, default=29)
    parser.add_argument("--reference-start-frame", type=int, default=-1)
    parser.add_argument("--sample-manifest", default=None)
    parser.add_argument("--motion-threshold-m", type=float, default=0.002)
    parser.add_argument("--expected-num-frames", type=int, default=0)
    parser.add_argument("--require-finite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-reference-metrics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-external-target-event-metrics", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    prediction_path = Path(args.prediction_json)
    data = json.loads(prediction_path.read_text())
    pred_cont = np.asarray(data["pred_cont"], dtype=np.float32)
    pred_bin = np.asarray(data["pred_bin_probability"], dtype=np.float32)
    if pred_cont.ndim != 2 or pred_cont.shape[1] != len(readout.TARGET_CONT_NAMES):
        raise ValueError(f"pred_cont shape {pred_cont.shape} does not match target contract")
    if pred_bin.ndim != 2 or pred_bin.shape[1] != len(readout.TARGET_BIN_NAMES):
        raise ValueError(f"pred_bin_probability shape {pred_bin.shape} does not match target contract")
    if pred_cont.shape[0] != pred_bin.shape[0]:
        raise ValueError(f"frame mismatch: pred_cont={pred_cont.shape} pred_bin={pred_bin.shape}")

    reference_start = (
        int(data.get("reference_start_frame", 0))
        if int(args.reference_start_frame) < 0
        else int(args.reference_start_frame)
    )
    future_start = min(max(0, int(args.future_start_frame)), int(pred_cont.shape[0]))
    sample_manifest: dict[str, Any] = {}
    if args.sample_manifest:
        sample_manifest_path = Path(args.sample_manifest)
        sample_manifest = json.loads(sample_manifest_path.read_text())
    prefix_boundary = int(
        sample_manifest.get(
            "prefix_boundary_frame",
            reference_start + future_start - 1,
        )
    )
    report: dict[str, Any] = {
        "prediction_json": str(prediction_path),
        "video_path": data.get("video_path"),
        "num_frames": int(pred_cont.shape[0]),
        "reference_start_frame": int(reference_start),
        "target_cont_names": list(data.get("target_cont_names", [])),
        "target_bin_names": list(data.get("target_bin_names", [])),
        "finite_prediction": bool(np.isfinite(pred_cont).all() and np.isfinite(pred_bin).all()),
        "future_start_frame": int(future_start),
        "prefix_boundary_frame": int(prefix_boundary),
        "valid_prediction_start_frame": int(reference_start + future_start),
        "sample_manifest": _jsonable(sample_manifest) if sample_manifest else {},
        "boundary": (
            "Controller-facing Cosmos3 task-state export. Metrics are advisory "
            "diagnostics for integration; this is not a hard gate or dynamic "
            "task-completion evidence."
        ),
        "controller_interface": {
            "source": "cosmos3_generated_video_decoded_by_task_state_readout",
            "uses_oracle_state_at_controller_time": False,
            "trajectory_fields": [
                "hole_pose",
                "peg_pose",
                "tcp_pose",
                "peg_head_at_hole",
                "hole_radius",
                "grasped_probability",
                "inserted_probability",
            ],
        },
    }

    reference_source = None
    target_cont = None
    target_bin = None
    if data.get("reference_h5"):
        target_cont, target_bin = readout._slot_targets(  # noqa: SLF001
            Path(data["reference_h5"]),
            None,
            list(range(reference_start, reference_start + pred_cont.shape[0])),
        )
        report["reference_h5"] = data["reference_h5"]
        reference_source = "reference_h5"
    elif sample_manifest.get("state_target_path") or sample_manifest.get("task_state_target_path"):
        state_target_path = Path(str(sample_manifest.get("state_target_path") or sample_manifest["task_state_target_path"]))
        target_full_cont, target_full_bin = _state_targets_from_json(state_target_path)
        reference_end = reference_start + pred_cont.shape[0]
        if reference_start < 0 or reference_end > target_full_cont.shape[0]:
            raise ValueError(
                "state target frame range mismatch: "
                f"reference_start={reference_start}, requested_end={reference_end}, "
                f"available={target_full_cont.shape[0]}, path={state_target_path}"
            )
        target_cont = target_full_cont[reference_start:reference_end]
        target_bin = target_full_bin[reference_start:reference_end]
        report["reference_state_target_json"] = str(state_target_path)
        reference_source = "state_target_json"

    if target_cont is not None and target_bin is not None:
        report["reference_source"] = reference_source
        report["reference_metrics"] = {
            "all": _metrics(pred_cont, pred_bin, target_cont, target_bin),
            "future": _metrics(
                pred_cont[future_start:],
                pred_bin[future_start:],
                target_cont[future_start:],
                target_bin[future_start:],
            )
            if future_start < pred_cont.shape[0]
            else {},
        }
        report["external_target_event_metrics"] = _motion_event_metrics(
            pred_cont,
            target_cont,
            reference_start,
            future_start,
            float(args.motion_threshold_m),
        )

    strict_failures: list[str] = []
    expected_num_frames = int(args.expected_num_frames)
    if expected_num_frames > 0 and int(pred_cont.shape[0]) != expected_num_frames:
        strict_failures.append(f"num_frames_mismatch:{pred_cont.shape[0]}!={expected_num_frames}")
    if expected_num_frames > 0 and data.get("require_exact_video_frames") is not True:
        strict_failures.append("prediction_was_not_exact_video_frame_decode")
    if expected_num_frames > 0 and data.get("reference_end_exclusive") is not None:
        reference_span = int(data["reference_end_exclusive"]) - int(reference_start)
        if reference_span != expected_num_frames:
            strict_failures.append(f"reference_span_mismatch:{reference_span}!={expected_num_frames}")
    if bool(args.require_finite) and not bool(report["finite_prediction"]):
        strict_failures.append("nonfinite_prediction")
    if bool(args.require_reference_metrics) and "reference_metrics" not in report:
        strict_failures.append("missing_reference_metrics")
    if bool(args.require_external_target_event_metrics):
        event = report.get("external_target_event_metrics")
        if not isinstance(event, dict):
            strict_failures.append("missing_external_target_event_metrics")
        else:
            required_event_keys = [
                "predicted_motion_onset_frame",
                "target_motion_onset_frame",
                "final_hole_pos_error_m",
                "final_insert_point_error_m",
                "future_hole_path_rmse_m",
                "future_insert_point_rmse_m",
            ]
            for key in required_event_keys:
                if key not in event:
                    strict_failures.append(f"missing_external_target_event_key:{key}")
            if event.get("target_motion_onset_frame") is not None and event.get("predicted_motion_onset_frame") is None:
                strict_failures.append("moving_target_without_predicted_motion_onset")
    report["strict_readout_requirements"] = {
        "expected_num_frames": expected_num_frames,
        "require_finite": bool(args.require_finite),
        "require_reference_metrics": bool(args.require_reference_metrics),
        "require_external_target_event_metrics": bool(args.require_external_target_event_metrics),
    }
    report["strict_readout_failures"] = strict_failures
    report["strict_readout_ok"] = not strict_failures

    if args.trajectory_json:
        trajectory_path = Path(args.trajectory_json)
        trajectory_payload = {
            "prediction_json": str(prediction_path),
            "video_path": data.get("video_path"),
            "boundary": report["boundary"],
            "frame_start": int(reference_start),
            "frame_end": int(reference_start + pred_cont.shape[0] - 1),
            "future_start_frame": int(future_start),
            "prefix_boundary_frame": int(prefix_boundary),
            "valid_prediction_start_frame": int(reference_start + future_start),
            "sample_manifest": _jsonable(sample_manifest) if sample_manifest else {},
            "trajectory": _trajectory(pred_cont, pred_bin, reference_start),
        }
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_path.write_text(json.dumps(_jsonable(trajectory_payload), indent=2, sort_keys=True) + "\n")
        report["trajectory_json"] = str(trajectory_path)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n")
    if args.output_md:
        _write_md(Path(args.output_md), report)
    print(json.dumps(_jsonable(report), sort_keys=True))
    if strict_failures:
        raise SystemExit("strict task-state readout inspection failed: " + "; ".join(strict_failures))


if __name__ == "__main__":
    main()
