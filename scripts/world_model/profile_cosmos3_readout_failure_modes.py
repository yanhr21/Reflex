#!/usr/bin/env python3
"""Profile generated-RGB readout failure modes for Cosmos3 full-episode evals.

The readout is a diagnostic decoder on generated RGB. This script summarizes
where target/peg/TCP trajectories drift and how sensitive target-motion onset
is to displacement thresholds. It does not change evaluation gates and is not a
controller-success metric.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_THRESHOLDS_M = [0.002, 0.005, 0.01, 0.02, 0.05]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readout-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument(
        "--thresholds-m",
        default=",".join(str(x) for x in DEFAULT_THRESHOLDS_M),
        help="Comma-separated target-motion displacement thresholds in meters.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def first_onset(pos: np.ndarray, threshold: float) -> int | None:
    if pos.size == 0:
        return None
    disp = np.linalg.norm(pos - pos[0:1], axis=1)
    hits = np.flatnonzero(disp > float(threshold))
    return int(hits[0]) if hits.size else None


def rmse(a: np.ndarray, b: np.ndarray, start: int = 0) -> float | None:
    end = min(a.shape[0], b.shape[0])
    start = max(0, min(int(start), end))
    if start >= end:
        return None
    diff = a[start:end] - b[start:end]
    return float(np.sqrt(np.mean(diff * diff)))


def final_error(a: np.ndarray, b: np.ndarray) -> float | None:
    end = min(a.shape[0], b.shape[0])
    if end <= 0:
        return None
    return float(np.linalg.norm(a[end - 1] - b[end - 1]))


def idx(names: list[str], key: str) -> int:
    try:
        return names.index(key)
    except ValueError as exc:
        raise KeyError(f"missing state vector key {key!r}") from exc


def state_xyz(states: np.ndarray, names: list[str], prefix: str) -> np.ndarray:
    return states[:, [idx(names, f"{prefix}_x"), idx(names, f"{prefix}_y"), idx(names, f"{prefix}_z")]]


def sample_dirs(readout_root: Path) -> list[Path]:
    samples_root = readout_root / "samples"
    if not samples_root.is_dir():
        raise FileNotFoundError(f"missing samples directory: {samples_root}")
    return sorted(p for p in samples_root.iterdir() if p.is_dir())


def profile_sample(sample_dir: Path, thresholds: list[float]) -> dict[str, Any]:
    traj_path = sample_dir / "readout_trajectory.json"
    if not traj_path.is_file():
        return {"sample_dir": str(sample_dir), "strict_profile_ok": False, "failures": ["missing_readout_trajectory"]}
    traj_payload = read_json(traj_path)
    manifest = traj_payload.get("sample_manifest") if isinstance(traj_payload.get("sample_manifest"), dict) else {}
    state_target_path = Path(str(manifest.get("task_state_target_path") or manifest.get("state_target_path") or ""))
    failures: list[str] = []
    if not state_target_path.is_file():
        failures.append(f"missing_state_target:{state_target_path}")
        states_payload = {}
    else:
        states_payload = read_json(state_target_path)

    traj = traj_payload.get("trajectory") or []
    if not traj:
        failures.append("empty_trajectory")
    pred_hole = np.asarray([row.get("hole_pose", [math.nan, math.nan, math.nan])[:3] for row in traj], dtype=np.float64)
    pred_peg = np.asarray([row.get("peg_pose", [math.nan, math.nan, math.nan])[:3] for row in traj], dtype=np.float64)
    pred_tcp = np.asarray([row.get("tcp_pose", [math.nan, math.nan, math.nan])[:3] for row in traj], dtype=np.float64)
    pred_peg_head = np.asarray(
        [row.get("peg_head_at_hole", [math.nan, math.nan, math.nan])[:3] for row in traj],
        dtype=np.float64,
    )

    states = np.asarray(states_payload.get("states") or [], dtype=np.float64)
    names = list(states_payload.get("state_vector_names") or [])
    if states.ndim != 2 or not names:
        failures.append("invalid_state_targets")
        gt_hole = gt_peg = gt_tcp = gt_peg_head = np.empty((0, 3), dtype=np.float64)
    else:
        gt_hole = state_xyz(states, names, "hole_pose")
        gt_peg = state_xyz(states, names, "peg_pose")
        gt_tcp = state_xyz(states, names, "tcp_pose")
        gt_peg_head = state_xyz(states, names, "peg_head_at_hole")

    prefix_frame = int(manifest.get("prefix_frame_index", traj_payload.get("prefix_boundary_frame", -1)) or -1)
    future_start = max(0, prefix_frame + 1)
    threshold_rows = []
    for threshold in thresholds:
        pred_onset = first_onset(pred_hole, threshold)
        gt_onset = first_onset(gt_hole, threshold)
        threshold_rows.append(
            {
                "threshold_m": float(threshold),
                "predicted_onset_frame": pred_onset,
                "target_onset_frame": gt_onset,
                "onset_abs_error_frames": None
                if pred_onset is None or gt_onset is None
                else int(abs(pred_onset - gt_onset)),
            }
        )

    finite = all(np.isfinite(arr).all() for arr in (pred_hole, pred_peg, pred_tcp, pred_peg_head))
    if not finite:
        failures.append("nonfinite_prediction_trajectory")

    return {
        "sample_dir": str(sample_dir),
        "name": manifest.get("name", sample_dir.name),
        "scenario": manifest.get("scenario"),
        "prefix_role": manifest.get("prefix_role"),
        "prefix_frame_index": prefix_frame,
        "future_start_frame": future_start,
        "num_pred_frames": int(pred_hole.shape[0]),
        "num_target_frames": int(gt_hole.shape[0]),
        "threshold_onsets": threshold_rows,
        "errors_m": {
            "initial_hole_pos_error": final_error(pred_hole[:1], gt_hole[:1]) if pred_hole.shape[0] and gt_hole.shape[0] else None,
            "final_hole_pos_error": final_error(pred_hole, gt_hole),
            "final_peg_pos_error": final_error(pred_peg, gt_peg),
            "final_tcp_pos_error": final_error(pred_tcp, gt_tcp),
            "all_hole_rmse": rmse(pred_hole, gt_hole, 0),
            "all_peg_rmse": rmse(pred_peg, gt_peg, 0),
            "all_tcp_rmse": rmse(pred_tcp, gt_tcp, 0),
            "all_peg_head_hole_rmse": rmse(pred_peg_head, gt_peg_head, 0),
            "future_hole_rmse": rmse(pred_hole, gt_hole, future_start),
            "future_peg_rmse": rmse(pred_peg, gt_peg, future_start),
            "future_tcp_rmse": rmse(pred_tcp, gt_tcp, future_start),
            "future_peg_head_hole_rmse": rmse(pred_peg_head, gt_peg_head, future_start),
        },
        "strict_profile_ok": not failures,
        "failures": failures,
    }


def write_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        f"# {report['title']}",
        "",
        f"- readout root: `{report['readout_root']}`",
        f"- input kind: `{report['input_kind']}`",
        f"- samples: `{report['aggregate']['num_samples']}`",
        f"- strict profile ok: `{report['strict_profile_ok']}`",
        f"- failures: `{report['failures']}`",
        "",
        report["boundary"],
        "",
        "## Aggregate",
        "",
        f"- mean final hole error: `{report['aggregate']['mean_final_hole_pos_error_m']}` m",
        f"- mean future hole RMSE: `{report['aggregate']['mean_future_hole_rmse_m']}` m",
        f"- mean future peg RMSE: `{report['aggregate']['mean_future_peg_rmse_m']}` m",
        f"- mean future TCP RMSE: `{report['aggregate']['mean_future_tcp_rmse_m']}` m",
        f"- mean future peg-head-hole RMSE: `{report['aggregate']['mean_future_peg_head_hole_rmse_m']}` m",
        "",
        "## Threshold Onsets",
        "",
        "| sample | role | scenario | prefix | threshold | pred onset | target onset | abs err | final hole err | future hole rmse |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sample in report["samples"]:
        errors = sample.get("errors_m", {})
        for onset in sample.get("threshold_onsets", []):
            lines.append(
                "| `{name}` | `{role}` | `{scenario}` | {prefix} | {thr:.3f} | {pred} | {target} | {err} | {final} | {future} |".format(
                    name=sample.get("name"),
                    role=sample.get("prefix_role"),
                    scenario=sample.get("scenario"),
                    prefix=sample.get("prefix_frame_index"),
                    thr=float(onset["threshold_m"]),
                    pred=onset.get("predicted_onset_frame"),
                    target=onset.get("target_onset_frame"),
                    err=onset.get("onset_abs_error_frames"),
                    final=errors.get("final_hole_pos_error"),
                    future=errors.get("future_hole_rmse"),
                )
            )
    path.write_text("\n".join(lines) + "\n")


def mean_present(samples: list[dict[str, Any]], key: str) -> float | None:
    vals = [s.get("errors_m", {}).get(key) for s in samples]
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def main() -> None:
    args = parse_args()
    thresholds = [float(x) for x in args.thresholds_m.split(",") if x.strip()]
    readout_root = Path(args.readout_root)
    eval_manifest_path = readout_root.parent / "eval_input_manifest.json"
    eval_manifest = read_json(eval_manifest_path) if eval_manifest_path.is_file() else {}
    reference_rgb_calibration = bool(eval_manifest.get("reference_rgb_calibration"))
    input_kind = "reference_rgb_calibration" if reference_rgb_calibration else "cosmos3_generated_rgb"
    title = (
        "Cosmos3 Reference-RGB Readout Calibration Profile"
        if reference_rgb_calibration
        else "Cosmos3 Generated-RGB Readout Failure Profile"
    )
    boundary = (
        "Reference-RGB readout calibration profile only. It diagnoses the "
        "task-state readout/onset decoder on ground-truth RGB and does not "
        "prove Cosmos3 world-model or controller success."
        if reference_rgb_calibration
        else (
            "Generated-RGB readout failure profile only. It diagnoses failure "
            "modes on top of Cosmos3-generated RGB and does not replace the "
            "world model, change thresholds, or prove controller success."
        )
    )
    samples = [profile_sample(path, thresholds) for path in sample_dirs(readout_root)]
    failures = [f"{sample.get('name')}:{failure}" for sample in samples for failure in sample.get("failures", [])]
    report = {
        "boundary": boundary,
        "input_kind": input_kind,
        "readout_root": str(readout_root),
        "reference_rgb_calibration": reference_rgb_calibration,
        "strict_profile_ok": not failures,
        "title": title,
        "failures": failures,
        "aggregate": {
            "num_samples": len(samples),
            "mean_final_hole_pos_error_m": mean_present(samples, "final_hole_pos_error"),
            "mean_future_hole_rmse_m": mean_present(samples, "future_hole_rmse"),
            "mean_future_peg_rmse_m": mean_present(samples, "future_peg_rmse"),
            "mean_future_tcp_rmse_m": mean_present(samples, "future_tcp_rmse"),
            "mean_future_peg_head_hole_rmse_m": mean_present(samples, "future_peg_head_hole_rmse"),
        },
        "samples": samples,
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_md(report, output_md)
    print(json.dumps(report["aggregate"], sort_keys=True))
    if failures:
        raise SystemExit("readout failure profile strict checks failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
