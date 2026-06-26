#!/usr/bin/env python3
"""Calibrate a residual-executor scale on the held-out split.

This is not training. It checks whether the learned residual direction helps
when conservatively scaled before being added to the frozen DP prior.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_cosmos3_live_receding_loop import require_compute_step  # noqa: E402
from train_cosmos3_executor_residual import build_model, load_arrays, read_jsonl, split_indices  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--executor-jsonl", required=True)
    parser.add_argument("--dp-prior-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--training-summary-json", default="")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument(
        "--scales",
        default="0,0.01,0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.75,1.0",
    )
    parser.add_argument("--formal-min-gpus", type=int, default=2)
    parser.add_argument("--formal-min-wall-seconds", type=int, default=10800)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_scales(value: str) -> list[float]:
    scales = sorted({float(part.strip()) for part in value.split(",") if part.strip()})
    if not scales or scales[0] < 0.0 or scales[-1] > 1.0:
        raise ValueError(f"scales must be within [0, 1], got {scales}")
    if 0.0 not in scales:
        scales.insert(0, 0.0)
    return scales


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def group_summary(
    *,
    rows: list[dict[str, Any]],
    val_idx: np.ndarray,
    y: np.ndarray,
    prior: np.ndarray,
    pred_residual: np.ndarray,
    scale: float,
    key: str,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    val_set = [int(i) for i in val_idx]
    for idx in val_set:
        groups[str(rows[idx].get(key) or "unknown")].append(idx)
    out: dict[str, Any] = {}
    for name, idxs in sorted(groups.items()):
        arr_idx = np.asarray(idxs, dtype=np.int64)
        pred = prior[arr_idx] + float(scale) * pred_residual[arr_idx]
        out[name] = {
            "n": int(len(idxs)),
            "dp_prior_mse": mse(prior[arr_idx], y[arr_idx]),
            "scaled_residual_mse": mse(pred, y[arr_idx]),
        }
    return out


def main() -> int:
    args = parse_args()
    require_compute_step()

    import torch

    checkpoint_path = Path(args.checkpoint).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    payload = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
    ckpt_args = payload.get("args") or {}
    executor_rows = read_jsonl(Path(args.executor_jsonl).resolve())
    prior_rows = read_jsonl(Path(args.dp_prior_jsonl).resolve())
    x_raw, y_raw, prior_raw, used_rows = load_arrays(executor_rows, prior_rows, 0)
    task_sources = sorted({str(row.get("task_path_source") or "unknown") for row in used_rows})
    if any("gt" in source and "debug" in source for source in task_sources):
        raise SystemExit(f"refusing GT debug task paths: {task_sources}")

    train_idx, val_idx = split_indices(len(used_rows), float(args.val_fraction), int(args.seed))
    x_mean = np.asarray(payload["x_mean"], dtype=np.float32).reshape(1, -1)
    x_std = np.asarray(payload["x_std"], dtype=np.float32).reshape(1, -1)
    residual_mean = np.asarray(payload["residual_mean"], dtype=np.float32).reshape(1, -1)
    residual_std = np.asarray(payload["residual_std"], dtype=np.float32).reshape(1, -1)
    x_norm = ((x_raw - x_mean) / x_std).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        int(payload["feature_dim"]),
        int(payload["target_dim"]),
        int(ckpt_args.get("hidden_dim", 1024)),
        int(ckpt_args.get("num_layers", 4)),
        float(ckpt_args.get("dropout", 0.05)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.as_tensor(x_norm, device=device, dtype=torch.float32)).detach().cpu().numpy()
    pred_residual = (pred_norm * residual_std + residual_mean).astype(np.float32)

    scales = parse_scales(args.scales)
    scale_rows: list[dict[str, Any]] = []
    baseline_val = mse(prior_raw[val_idx], y_raw[val_idx])
    baseline_train = mse(prior_raw[train_idx], y_raw[train_idx])
    for scale in scales:
        pred_train = prior_raw[train_idx] + float(scale) * pred_residual[train_idx]
        pred_val = prior_raw[val_idx] + float(scale) * pred_residual[val_idx]
        scale_rows.append(
            {
                "scale": float(scale),
                "train_action_mse": mse(pred_train, y_raw[train_idx]),
                "val_action_mse": mse(pred_val, y_raw[val_idx]),
                "val_delta_vs_dp_prior": mse(pred_val, y_raw[val_idx]) - baseline_val,
            }
        )

    best_any = min(scale_rows, key=lambda row: float(row["val_action_mse"]))
    positive_rows = [row for row in scale_rows if float(row["scale"]) > 0.0]
    best_positive = min(positive_rows, key=lambda row: float(row["val_action_mse"])) if positive_rows else None

    training_summary = None
    if args.training_summary_json:
        path = Path(args.training_summary_json).resolve()
        if path.is_file():
            training_summary = json.loads(path.read_text())

    formal_floor_met = bool(
        training_summary
        and int(training_summary.get("world_size", 0)) >= int(args.formal_min_gpus)
        and float(training_summary.get("elapsed_seconds", 0.0)) >= float(args.formal_min_wall_seconds)
        and bool(training_summary.get("formal_training_floor_met", False))
    )
    ready_positive = bool(
        best_positive
        and float(best_positive["val_action_mse"]) < baseline_val
        and formal_floor_met
    )

    chosen_scale = float(best_positive["scale"]) if ready_positive and best_positive else float(best_any["scale"])
    summary = {
        "schema": "cosmos3_executor_residual_scale_calibration_v1",
        "checkpoint": str(checkpoint_path),
        "executor_jsonl": str(Path(args.executor_jsonl).resolve()),
        "dp_prior_jsonl": str(Path(args.dp_prior_jsonl).resolve()),
        "output_root": str(output_root),
        "num_samples": int(len(used_rows)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "task_path_sources": task_sources,
        "baseline_train_dp_prior_mse": baseline_train,
        "baseline_val_dp_prior_mse": baseline_val,
        "scales": scale_rows,
        "best_any_scale": best_any,
        "best_positive_scale": best_positive,
        "formal_training_floor_met_from_summary": formal_floor_met,
        "ready_for_scaled_closed_loop_eval": ready_positive,
        "chosen_scale_for_live_eval": chosen_scale if ready_positive else None,
        "by_val_role_at_best_any": group_summary(
            rows=used_rows,
            val_idx=val_idx,
            y=y_raw,
            prior=prior_raw,
            pred_residual=pred_residual,
            scale=float(best_any["scale"]),
            key="prefix_role",
        ),
        "by_val_scenario_at_best_any": group_summary(
            rows=used_rows,
            val_idx=val_idx,
            y=y_raw,
            prior=prior_raw,
            pred_residual=pred_residual,
            scale=float(best_any["scale"]),
            key="scenario",
        ),
        "boundary": (
            "Scale calibration is validation-only model selection. scale=0 is "
            "the frozen DP prior baseline and must not be reported as residual "
            "executor method success. Live eval is allowed only if a positive "
            "scale beats the DP-prior baseline after the formal training floor."
        ),
    }
    write_json(output_root / "residual_scale_calibration.json", summary)
    print(json.dumps({
        "output_json": str(output_root / "residual_scale_calibration.json"),
        "baseline_val_dp_prior_mse": baseline_val,
        "best_any_scale": best_any,
        "best_positive_scale": best_positive,
        "ready_for_scaled_closed_loop_eval": ready_positive,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
