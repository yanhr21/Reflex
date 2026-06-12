#!/usr/bin/env python3
"""Train/evaluate a calibrated target-motion head from RGB-derived readout.

The inputs are task-state readout trajectories decoded from RGB videos. This is
a target-monitor diagnostic and not controller or Cosmos3 success evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-readout-root", required=True)
    parser.add_argument("--eval-readout-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--extra-eval-root", action="append", default=[], help="NAME=READOUT_ROOT")
    parser.add_argument("--motion-threshold-m", type=float, default=0.002)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def idx(names: list[str], key: str) -> int:
    try:
        return names.index(key)
    except ValueError as exc:
        raise KeyError(f"missing state vector key {key!r}") from exc


def gt_hole_xyz(path: Path) -> np.ndarray:
    payload = read_json(path)
    names = list(payload.get("state_vector_names") or [])
    states = np.asarray(payload.get("states"), dtype=np.float64)
    cols = [idx(names, key) for key in ("hole_pose_x", "hole_pose_y", "hole_pose_z")]
    return states[:, cols]


def sample_dirs(root: Path) -> list[Path]:
    samples = root / "samples"
    if not samples.is_dir():
        raise FileNotFoundError(f"missing samples directory: {samples}")
    return sorted(p for p in samples.iterdir() if p.is_dir())


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if x.size == 0:
        return x
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        start = max(0, i - window + 1)
        out[i] = np.mean(x[start : i + 1])
    return out


def build_features(pred_hole: np.ndarray) -> np.ndarray:
    n = pred_hole.shape[0]
    if n == 0:
        return np.empty((0, 10), dtype=np.float32)
    disp = np.linalg.norm(pred_hole - pred_hole[0:1], axis=1)
    delta1 = np.zeros(n, dtype=np.float64)
    delta5 = np.zeros(n, dtype=np.float64)
    delta15 = np.zeros(n, dtype=np.float64)
    if n > 1:
        delta1[1:] = np.linalg.norm(pred_hole[1:] - pred_hole[:-1], axis=1)
    for offset, dest in ((5, delta5), (15, delta15)):
        if n > offset:
            dest[offset:] = np.linalg.norm(pred_hole[offset:] - pred_hole[:-offset], axis=1)
    cummax = np.maximum.accumulate(disp)
    time = np.linspace(0.0, 1.0, n, dtype=np.float64)
    xyz_delta = pred_hole - pred_hole[0:1]
    feats = np.column_stack(
        [
            disp,
            cummax,
            delta1,
            rolling_mean(delta1, 5),
            rolling_mean(delta1, 15),
            delta5,
            delta15,
            xyz_delta,
            time,
        ]
    )
    return feats.astype(np.float32)


def load_dataset(root: Path, motion_threshold: float) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    sample_reports: list[dict[str, Any]] = []
    for sample_dir in sample_dirs(root):
        traj_path = sample_dir / "readout_trajectory.json"
        if not traj_path.is_file():
            continue
        traj_payload = read_json(traj_path)
        manifest = traj_payload.get("sample_manifest") if isinstance(traj_payload.get("sample_manifest"), dict) else {}
        state_path = Path(str(manifest.get("task_state_target_path") or manifest.get("state_target_path") or ""))
        if not state_path.is_file():
            continue
        traj = traj_payload.get("trajectory") or []
        pred_hole = np.asarray([row.get("hole_pose", [math.nan, math.nan, math.nan])[:3] for row in traj], dtype=np.float64)
        gt_hole = gt_hole_xyz(state_path)
        n = min(pred_hole.shape[0], gt_hole.shape[0])
        pred_hole = pred_hole[:n]
        gt_hole = gt_hole[:n]
        if n == 0 or not np.isfinite(pred_hole).all():
            continue
        labels = (np.linalg.norm(gt_hole - gt_hole[0:1], axis=1) >= float(motion_threshold)).astype(np.float32)
        features = build_features(pred_hole)
        all_features.append(features)
        all_labels.append(labels)
        sample_reports.append(
            {
                "name": manifest.get("name", sample_dir.name),
                "scenario": manifest.get("scenario"),
                "num_frames": int(n),
                "target_onset_frame": first_true(labels >= 0.5),
            }
        )
    if not all_features:
        raise ValueError(f"no usable readout samples in {root}")
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0), sample_reports


def first_true(values: np.ndarray) -> int | None:
    hits = np.flatnonzero(values)
    return int(hits[0]) if hits.size else None


class MotionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


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
        ranks[start:end] = 0.5 * (start + 1 + end)
        start = end
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    pos_rank_sum = float(ranks[inverse][labels].sum())
    return (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def f1_at(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = probs >= float(threshold)
    labels_b = labels.astype(bool)
    tp = int(np.logical_and(pred, labels_b).sum())
    fp = int(np.logical_and(pred, ~labels_b).sum())
    fn = int(np.logical_and(~pred, labels_b).sum())
    tn = int(np.logical_and(~pred, ~labels_b).sum())
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def best_f1(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    candidates = np.unique(probs[np.isfinite(probs)])
    if candidates.size == 0:
        return {"threshold": None, "f1": None}
    best = None
    for threshold in candidates:
        row = f1_at(labels, probs, float(threshold))
        if best is None or (row.get("f1") or -1.0) > (best.get("f1") or -1.0):
            best = row
    return best or {"threshold": None, "f1": None}


def evaluate_dataset(
    name: str,
    model: MotionHead,
    features: np.ndarray,
    labels: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    with torch.no_grad():
        x = torch.as_tensor((features - mean) / std, dtype=torch.float32, device=device)
        probs = torch.sigmoid(model(x)).cpu().numpy()
    return {
        "name": name,
        "num_frames": int(labels.size),
        "positive_frames": int(labels.sum()),
        "frame_auroc": auc_roc(labels, probs),
        "f1_at_0_5": f1_at(labels, probs, 0.5),
        "best_f1": best_f1(labels, probs),
    }


def jsonable(value: Any) -> Any:
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
    return value


def write_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Target-Motion Head Calibration",
        "",
        f"- train root: `{report['train_readout_root']}`",
        f"- eval root: `{report['eval_readout_root']}`",
        f"- motion threshold: `{report['motion_threshold_m']}` m",
        f"- boundary: `{report['boundary']}`",
        "",
        "## Metrics",
        "",
        "| split | frames | positives | AUROC | F1@0.5 | best F1 | best threshold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["results"]:
        lines.append(
            "| `{name}` | {frames} | {pos} | {auc} | {f1} | {best} | {thr} |".format(
                name=result["name"],
                frames=result["num_frames"],
                pos=result["positive_frames"],
                auc=result.get("frame_auroc"),
                f1=result.get("f1_at_0_5", {}).get("f1"),
                best=result.get("best_f1", {}).get("f1"),
                thr=result.get("best_f1", {}).get("threshold"),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y, train_samples = load_dataset(Path(args.train_readout_root), float(args.motion_threshold_m))
    eval_x, eval_y, eval_samples = load_dataset(Path(args.eval_readout_root), float(args.motion_threshold_m))
    mean = train_x.mean(axis=0).astype(np.float32)
    std = np.maximum(train_x.std(axis=0), 1e-6).astype(np.float32)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    model = MotionHead(train_x.shape[1], int(args.hidden_dim)).to(device)
    pos = float(train_y.sum())
    neg = float(train_y.size - pos)
    pos_weight = torch.as_tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    x = torch.as_tensor((train_x - mean) / std, dtype=torch.float32, device=device)
    y = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    for step in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(x), y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
    results = [
        evaluate_dataset("train_reference_rgb", model, train_x, train_y, mean, std, device),
        evaluate_dataset("eval_reference_rgb", model, eval_x, eval_y, mean, std, device),
    ]
    for raw in args.extra_eval_root:
        if "=" not in raw:
            raise ValueError(f"--extra-eval-root must be NAME=PATH, got {raw!r}")
        name, path = raw.split("=", 1)
        extra_x, extra_y, _ = load_dataset(Path(path), float(args.motion_threshold_m))
        results.append(evaluate_dataset(name, model, extra_x, extra_y, mean, std, device))
    report = {
        "boundary": (
            "Calibrated target-motion head over RGB-derived task-state readout. "
            "This is a target-monitor diagnostic and does not prove controller "
            "or Cosmos3 world-model success."
        ),
        "eval_readout_root": str(args.eval_readout_root),
        "extra_eval_roots": list(args.extra_eval_root),
        "feature_names": [
            "hole_displacement",
            "hole_displacement_cummax",
            "hole_delta_1",
            "hole_delta_1_roll5",
            "hole_delta_1_roll15",
            "hole_delta_5",
            "hole_delta_15",
            "hole_delta_x",
            "hole_delta_y",
            "hole_delta_z",
            "time_fraction",
        ],
        "motion_threshold_m": float(args.motion_threshold_m),
        "results": results,
        "train_loss_final": float(loss.detach().cpu().item()),
        "train_readout_root": str(args.train_readout_root),
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
    }
    (output_dir / "target_motion_head_metrics.json").write_text(json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n")
    write_md(jsonable(report), output_dir / "target_motion_head_metrics.md")
    torch.save({"state_dict": model.state_dict(), "feature_mean": mean, "feature_std": std, "report": jsonable(report)}, output_dir / "target_motion_head.pt")
    print(json.dumps(jsonable({"results": results, "train_loss_final": report["train_loss_final"]}), sort_keys=True))


if __name__ == "__main__":
    main()
