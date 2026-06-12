#!/usr/bin/env python3
"""Score target-motion onset from task-state readout trajectories.

This script is diagnostic only. It treats the readout-predicted hole
displacement from frame 0 as a continuous target-motion score and compares it
against the ground-truth hole displacement labels from the state target JSON.
It reports AUROC/F1/onset timing without changing any controller gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readout-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--motion-threshold-m", type=float, default=0.002)
    parser.add_argument("--score-threshold-m", type=float, default=None)
    parser.add_argument("--input-kind", default="")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sample_dirs(readout_root: Path) -> list[Path]:
    samples_root = readout_root / "samples"
    if not samples_root.is_dir():
        raise FileNotFoundError(f"missing samples directory: {samples_root}")
    return sorted(p for p in samples_root.iterdir() if p.is_dir())


def idx(names: list[str], key: str) -> int:
    try:
        return names.index(key)
    except ValueError as exc:
        raise KeyError(f"missing state vector key {key!r}") from exc


def gt_hole_xyz(path: Path) -> np.ndarray:
    payload = read_json(path)
    names = list(payload.get("state_vector_names") or [])
    states = np.asarray(payload.get("states"), dtype=np.float64)
    if states.ndim != 2:
        raise ValueError(f"invalid state target shape {states.shape}: {path}")
    cols = [idx(names, key) for key in ("hole_pose_x", "hole_pose_y", "hole_pose_z")]
    return states[:, cols]


def first_true(values: np.ndarray) -> int | None:
    hits = np.flatnonzero(values)
    return int(hits[0]) if hits.size else None


def auc_roc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[start:end] = avg_rank
        start = end
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    pos_rank_sum = float(ranks[inverse][labels].sum())
    return (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def f1_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = scores >= float(threshold)
    labels = labels.astype(bool)
    tp = int(np.logical_and(pred, labels).sum())
    fp = int(np.logical_and(pred, ~labels).sum())
    fn = int(np.logical_and(~pred, labels).sum())
    tn = int(np.logical_and(~pred, ~labels).sum())
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "threshold_m": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def best_f1(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    candidates = np.unique(scores[np.isfinite(scores)])
    if candidates.size == 0:
        return {"threshold_m": None, "f1": None}
    # Include just-above max for all-negative predictions.
    candidates = np.concatenate([candidates, np.asarray([float(candidates.max()) + 1e-9])])
    best: dict[str, Any] | None = None
    for threshold in candidates:
        row = f1_at_threshold(labels, scores, float(threshold))
        if best is None or (row.get("f1") or -1.0) > (best.get("f1") or -1.0):
            best = row
    return best or {"threshold_m": None, "f1": None}


def profile_sample(sample_dir: Path, motion_threshold: float, score_threshold: float) -> dict[str, Any]:
    traj_path = sample_dir / "readout_trajectory.json"
    if not traj_path.is_file():
        return {"name": sample_dir.name, "strict_ok": False, "failures": ["missing_readout_trajectory"]}
    traj_payload = read_json(traj_path)
    manifest = traj_payload.get("sample_manifest") if isinstance(traj_payload.get("sample_manifest"), dict) else {}
    state_target_path = Path(str(manifest.get("task_state_target_path") or manifest.get("state_target_path") or ""))
    failures: list[str] = []
    if not state_target_path.is_file():
        failures.append(f"missing_state_target:{state_target_path}")
        gt_hole = np.empty((0, 3), dtype=np.float64)
    else:
        gt_hole = gt_hole_xyz(state_target_path)
    traj = traj_payload.get("trajectory") or []
    pred_hole = np.asarray([row.get("hole_pose", [math.nan, math.nan, math.nan])[:3] for row in traj], dtype=np.float64)
    n = min(pred_hole.shape[0], gt_hole.shape[0])
    pred_hole = pred_hole[:n]
    gt_hole = gt_hole[:n]
    if n == 0:
        failures.append("empty_or_mismatched_trajectory")
        scores = np.empty((0,), dtype=np.float64)
        labels = np.empty((0,), dtype=bool)
    else:
        scores = np.linalg.norm(pred_hole - pred_hole[0:1], axis=1)
        labels = np.linalg.norm(gt_hole - gt_hole[0:1], axis=1) >= float(motion_threshold)
    pred_onset = first_true(scores >= float(score_threshold)) if scores.size else None
    target_onset = first_true(labels) if labels.size else None
    return {
        "name": manifest.get("name", sample_dir.name),
        "scenario": manifest.get("scenario"),
        "prefix_role": manifest.get("prefix_role"),
        "prefix_frame_index": manifest.get("prefix_frame_index"),
        "num_frames": int(n),
        "target_has_motion": target_onset is not None,
        "predicted_onset_frame": pred_onset,
        "target_onset_frame": target_onset,
        "onset_abs_error_frames": None
        if pred_onset is None or target_onset is None
        else int(abs(pred_onset - target_onset)),
        "false_positive_on_static": bool(target_onset is None and pred_onset is not None),
        "score_threshold_m": float(score_threshold),
        "motion_threshold_m": float(motion_threshold),
        "max_score_m": None if scores.size == 0 else float(np.max(scores)),
        "strict_ok": not failures,
        "failures": failures,
        "_labels": labels,
        "_scores": scores,
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_md(report: dict[str, Any], path: Path) -> None:
    agg = report["aggregate"]
    lines = [
        "# Target-Onset Readout Score",
        "",
        f"- readout root: `{report['readout_root']}`",
        f"- input kind: `{report['input_kind']}`",
        f"- motion threshold: `{report['motion_threshold_m']}` m",
        f"- fixed score threshold: `{report['score_threshold_m']}` m",
        f"- strict ok: `{report['strict_ok']}`",
        f"- failures: `{report['failures']}`",
        "",
        "Diagnostic only: this scores target-motion detection from readout displacement. It is not controller evidence.",
        "",
        "## Aggregate",
        "",
        f"- frame AUROC: `{agg['frame_auroc']}`",
        f"- fixed-threshold F1: `{agg['fixed_threshold']['f1']}`",
        f"- fixed precision/recall: `{agg['fixed_threshold']['precision']}` / `{agg['fixed_threshold']['recall']}`",
        f"- best F1: `{agg['best_threshold']['f1']}` at threshold `{agg['best_threshold']['threshold_m']}` m",
        f"- moving onset mean abs error: `{agg['moving_onset_mean_abs_error_frames']}` frames",
        f"- static false-positive samples: `{agg['static_false_positive_samples']}` / `{agg['static_samples']}`",
        "",
        "## Samples",
        "",
        "| sample | role | scenario | target onset | pred onset | abs err | static FP | max score |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for sample in report["samples"]:
        lines.append(
            "| `{name}` | `{role}` | `{scenario}` | {target} | {pred} | {err} | {fp} | {score} |".format(
                name=sample.get("name"),
                role=sample.get("prefix_role"),
                scenario=sample.get("scenario"),
                target=sample.get("target_onset_frame"),
                pred=sample.get("predicted_onset_frame"),
                err=sample.get("onset_abs_error_frames"),
                fp=sample.get("false_positive_on_static"),
                score=sample.get("max_score_m"),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    motion_threshold = float(args.motion_threshold_m)
    score_threshold = float(args.score_threshold_m if args.score_threshold_m is not None else motion_threshold)
    readout_root = Path(args.readout_root)
    eval_manifest_path = readout_root.parent / "eval_input_manifest.json"
    eval_manifest = read_json(eval_manifest_path) if eval_manifest_path.is_file() else {}
    input_kind = args.input_kind or (
        "reference_rgb_calibration" if eval_manifest.get("reference_rgb_calibration") else "cosmos3_generated_rgb"
    )

    samples = [profile_sample(path, motion_threshold, score_threshold) for path in sample_dirs(readout_root)]
    failures = [f"{s.get('name')}:{failure}" for s in samples for failure in s.get("failures", [])]
    labels = np.concatenate([s["_labels"] for s in samples if s.get("_labels") is not None and len(s["_labels"])])
    scores = np.concatenate([s["_scores"] for s in samples if s.get("_scores") is not None and len(s["_scores"])])
    moving_errors = [
        float(s["onset_abs_error_frames"])
        for s in samples
        if s.get("target_has_motion") and s.get("onset_abs_error_frames") is not None
    ]
    static_samples = [s for s in samples if not s.get("target_has_motion")]
    aggregate = {
        "num_samples": len(samples),
        "num_frames": int(labels.size),
        "frame_auroc": auc_roc(labels, scores) if labels.size else None,
        "fixed_threshold": f1_at_threshold(labels, scores, score_threshold) if labels.size else {},
        "best_threshold": best_f1(labels, scores) if labels.size else {},
        "moving_onset_mean_abs_error_frames": float(np.mean(moving_errors)) if moving_errors else None,
        "static_false_positive_samples": int(sum(1 for s in static_samples if s.get("false_positive_on_static"))),
        "static_samples": len(static_samples),
    }
    report = {
        "aggregate": aggregate,
        "boundary": (
            "Target-onset readout score diagnostic only. It evaluates whether "
            "readout displacement can detect target motion; it is not a world-model "
            "success gate or controller evidence."
        ),
        "failures": failures,
        "input_kind": input_kind,
        "motion_threshold_m": motion_threshold,
        "readout_root": str(readout_root),
        "samples": samples,
        "score_threshold_m": score_threshold,
        "strict_ok": not failures,
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n")
    write_md(jsonable(report), output_md)
    print(json.dumps(jsonable({"aggregate": aggregate, "input_kind": input_kind}), sort_keys=True))
    if failures:
        raise SystemExit("target-onset readout scoring failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
