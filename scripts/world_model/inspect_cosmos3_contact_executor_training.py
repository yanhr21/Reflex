#!/usr/bin/env python3
"""Inspect a trained contact/progress executor by scenario and phase."""

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
from train_cosmos3_contact_executor import (  # noqa: E402
    ContactExecutorNet,
    load_arrays,
    read_jsonl,
    split_indices,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--allow-latest", action="store_true")
    parser.add_argument("--require-compute-step", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": int(len(rows)),
        "action_mse": mean_or_none([float(row["action_mse"]) for row in rows]),
        "dp_prior_mse": mean_or_none([float(row["dp_prior_mse"]) for row in rows]),
        "progress_mse": mean_or_none([float(row["progress_mse"]) for row in rows]),
        "inserted_acc": mean_or_none([float(row["inserted_correct"]) for row in rows]),
        "dp_continuable_acc": mean_or_none([float(row["dp_continuable_correct"]) for row in rows]),
        "action_mse_minus_prior": mean_or_none(
            [float(row["action_mse"]) - float(row["dp_prior_mse"]) for row in rows]
        ),
    }


def scale_sweep_for_indices(
    *,
    pred_resid: np.ndarray,
    prior: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    scales: list[float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if len(indices) == 0:
        return out
    for scale in scales:
        action = prior[indices] + float(scale) * pred_resid[indices]
        mse = float(np.mean((action - target[indices]) ** 2))
        out.append({"scale": float(scale), "action_mse": mse})
    return out


def best_scale_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None
    return min(entries, key=lambda item: float(item["action_mse"]))


def main() -> int:
    args = parse_args()
    if args.require_compute_step:
        require_compute_step()

    import torch

    training_root = Path(args.training_root).resolve()
    manifest = load_json(training_root / "training_manifest.json")
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else training_root / "checkpoint_final.pt"
    if not checkpoint_path.is_file() and args.allow_latest:
        checkpoint_path = training_root / "checkpoint_latest.pt"
    if not checkpoint_path.is_file():
        raise SystemExit(f"missing checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    rows = read_jsonl(Path(manifest["contact_executor_jsonl"]).resolve())
    x_raw, y_raw, prior_raw, progress_raw, binary_raw, used_rows = load_arrays(rows, 0)
    seed = int(ckpt.get("args", {}).get("seed", 20260615))
    val_fraction = float(ckpt.get("args", {}).get("val_fraction", 0.15))
    train_idx, val_idx = split_indices(len(used_rows), val_fraction, seed)
    eval_idx = val_idx if len(val_idx) else train_idx

    x_mean = ckpt["x_mean"]
    x_std = ckpt["x_std"]
    residual_mean = ckpt["residual_mean"]
    residual_std = ckpt["residual_std"]
    x_norm = ((x_raw - x_mean) / x_std).astype(np.float32)

    model = ContactExecutorNet(
        feature_dim=int(ckpt["feature_dim"]),
        target_dim=int(ckpt["target_dim"]),
        hidden_dim=int(ckpt["args"]["hidden_dim"]),
        num_layers=int(ckpt["args"]["num_layers"]),
        dropout=float(ckpt["args"].get("dropout", 0.0)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        pred_resid_norm_all, _, _ = model(torch.from_numpy(x_norm))
        pred_resid_all = pred_resid_norm_all.numpy() * residual_std + residual_mean
        pred_resid_norm, pred_progress, pred_logits = model(torch.from_numpy(x_norm[eval_idx]))
        pred_resid = pred_resid_all[eval_idx]
        pred_action = prior_raw[eval_idx] + pred_resid
        pred_progress_np = pred_progress.numpy()
        pred_binary = (torch.sigmoid(pred_logits).numpy() >= 0.5).astype(np.float32)

    per_row: list[dict[str, Any]] = []
    for local_i, idx in enumerate(eval_idx):
        row_meta = used_rows[int(idx)]
        action_mse = float(np.mean((pred_action[local_i] - y_raw[idx]) ** 2))
        prior_mse = float(np.mean((prior_raw[idx] - y_raw[idx]) ** 2))
        progress_mse = float(np.mean((pred_progress_np[local_i] - progress_raw[idx]) ** 2))
        per_row.append(
            {
                "uuid": row_meta.get("uuid"),
                "scenario": row_meta.get("scenario"),
                "prefix_role": row_meta.get("prefix_role"),
                "current_phase": row_meta.get("current_phase"),
                "action_mse": action_mse,
                "dp_prior_mse": prior_mse,
                "action_mse_minus_prior": action_mse - prior_mse,
                "progress_mse": progress_mse,
                "inserted_correct": float(pred_binary[local_i, 0] == binary_raw[idx, 0]),
                "dp_continuable_correct": float(pred_binary[local_i, 1] == binary_raw[idx, 1]),
            }
        )

    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "scenario": defaultdict(list),
        "prefix_role": defaultdict(list),
        "current_phase": defaultdict(list),
    }
    group_local_indices: dict[str, dict[str, list[int]]] = {
        "scenario": defaultdict(list),
        "prefix_role": defaultdict(list),
        "current_phase": defaultdict(list),
    }
    for row in per_row:
        for key in groups:
            groups[key][str(row.get(key) or "unknown")].append(row)
    for local_i, idx in enumerate(eval_idx):
        row_meta = used_rows[int(idx)]
        group_local_indices["scenario"][str(row_meta.get("scenario") or "unknown")].append(local_i)
        group_local_indices["prefix_role"][str(row_meta.get("prefix_role") or "unknown")].append(local_i)
        group_local_indices["current_phase"][str(row_meta.get("current_phase") or "unknown")].append(local_i)

    scales = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    local_all = np.arange(len(eval_idx), dtype=np.int64)
    global_scale_sweep = scale_sweep_for_indices(
        pred_resid=pred_resid,
        prior=prior_raw[eval_idx],
        target=y_raw[eval_idx],
        indices=local_all,
        scales=scales,
    )

    def grouped_scale_sweep(key: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, indices in sorted(group_local_indices[key].items()):
            entries = scale_sweep_for_indices(
                pred_resid=pred_resid,
                prior=prior_raw[eval_idx],
                target=y_raw[eval_idx],
                indices=np.asarray(indices, dtype=np.int64),
                scales=scales,
            )
            out[name] = {"sweep": entries, "best": best_scale_entry(entries)}
        return out

    phase_scale_sweep = grouped_scale_sweep("current_phase")
    prefix_role_scale_sweep = grouped_scale_sweep("prefix_role")
    scenario_scale_sweep = grouped_scale_sweep("scenario")

    def selector_summary(group_key: str, group_sweep: dict[str, Any]) -> dict[str, Any]:
        weighted_mse = 0.0
        total = 0
        choices: dict[str, Any] = {}
        for name, indices in sorted(group_local_indices[group_key].items()):
            best = dict((group_sweep.get(name) or {}).get("best") or {})
            count = int(len(indices))
            if not best or count <= 0:
                continue
            weighted_mse += count * float(best["action_mse"])
            total += count
            choices[name] = {"count": count, **best}
        return {
            "weighted_action_mse": float(weighted_mse / total) if total > 0 else None,
            "count": int(total),
            "choices": choices,
            "boundary": (
                "Oracle offline selector diagnostic only. It uses eval-set "
                "group best scales to test whether grouped candidate selection "
                "has signal; it is not live-controller evidence."
            ),
        }

    def global_train_calibrated_selector() -> dict[str, Any]:
        train_entries = scale_sweep_for_indices(
            pred_resid=pred_resid_all,
            prior=prior_raw,
            target=y_raw,
            indices=train_idx,
            scales=scales,
        )
        best = best_scale_entry(train_entries) or {"scale": 0.0, "action_mse": None}
        scale = float(best.get("scale", 0.0))
        eval_action = prior_raw[eval_idx] + scale * pred_resid_all[eval_idx]
        eval_mse = float(np.mean((eval_action - y_raw[eval_idx]) ** 2))
        return {
            "scale": scale,
            "train_action_mse": best.get("action_mse"),
            "eval_action_mse": eval_mse,
            "num_train": int(len(train_idx)),
            "num_eval": int(len(eval_idx)),
            "boundary": (
                "Offline calibration diagnostic only. The residual scale is "
                "selected on the train split and evaluated on the validation "
                "split; it still does not prove live execution."
            ),
        }

    def group_indices_for(indices: np.ndarray, group_key: str) -> dict[str, list[int]]:
        out: dict[str, list[int]] = defaultdict(list)
        for idx in indices:
            row_meta = used_rows[int(idx)]
            out[str(row_meta.get(group_key) or "unknown")].append(int(idx))
        return out

    def grouped_train_calibrated_selector(group_key: str) -> dict[str, Any]:
        train_groups = group_indices_for(train_idx, group_key)
        eval_groups = group_indices_for(eval_idx, group_key)
        weighted_mse = 0.0
        total = 0
        choices: dict[str, Any] = {}
        for name, eval_indices_list in sorted(eval_groups.items()):
            train_indices = np.asarray(train_groups.get(name, []), dtype=np.int64)
            eval_indices = np.asarray(eval_indices_list, dtype=np.int64)
            if len(train_indices) > 0:
                train_entries = scale_sweep_for_indices(
                    pred_resid=pred_resid_all,
                    prior=prior_raw,
                    target=y_raw,
                    indices=train_indices,
                    scales=scales,
                )
                best = best_scale_entry(train_entries) or {"scale": 0.0, "action_mse": None}
            else:
                best = {"scale": 0.0, "action_mse": None}
            scale = float(best.get("scale", 0.0))
            eval_action = prior_raw[eval_indices] + scale * pred_resid_all[eval_indices]
            eval_mse = float(np.mean((eval_action - y_raw[eval_indices]) ** 2))
            weighted_mse += len(eval_indices) * eval_mse
            total += len(eval_indices)
            choices[name] = {
                "scale": scale,
                "train_action_mse": best.get("action_mse"),
                "eval_action_mse": eval_mse,
                "num_train": int(len(train_indices)),
                "num_eval": int(len(eval_indices)),
            }
        return {
            "weighted_eval_action_mse": float(weighted_mse / total) if total > 0 else None,
            "num_eval": int(total),
            "choices": choices,
            "boundary": (
                "Offline calibration diagnostic only. Each group scale is "
                "selected on the train split and evaluated on the validation "
                "split; it is not live-controller evidence."
            ),
        }

    payload = {
        "schema": "cosmos3_contact_executor_training_inspection_v1",
        "training_root": str(training_root),
        "checkpoint": str(checkpoint_path),
        "num_eval_rows": int(len(per_row)),
        "overall": summarize_group(per_row),
        "by_scenario": {key: summarize_group(value) for key, value in sorted(groups["scenario"].items())},
        "by_prefix_role": {key: summarize_group(value) for key, value in sorted(groups["prefix_role"].items())},
        "by_current_phase": {key: summarize_group(value) for key, value in sorted(groups["current_phase"].items())},
        "residual_scale_sweep": {
            "scales": scales,
            "global": global_scale_sweep,
            "global_best": best_scale_entry(global_scale_sweep),
            "by_current_phase": phase_scale_sweep,
            "by_prefix_role": prefix_role_scale_sweep,
            "by_scenario": scenario_scale_sweep,
            "selector_diagnostics": {
                "by_current_phase": selector_summary("current_phase", phase_scale_sweep),
                "by_prefix_role": selector_summary("prefix_role", prefix_role_scale_sweep),
                "by_scenario": selector_summary("scenario", scenario_scale_sweep),
            },
            "train_calibrated_selector_diagnostics": {
                "global": global_train_calibrated_selector(),
                "by_current_phase": grouped_train_calibrated_selector("current_phase"),
                "by_prefix_role": grouped_train_calibrated_selector("prefix_role"),
                "by_scenario": grouped_train_calibrated_selector("scenario"),
            },
            "boundary": (
                "Diagnostic only. Scale 0 is the frozen DP prior and cannot be "
                "reported as contact-executor success. Positive-scale results "
                "only show whether the learned residual has offline signal."
            ),
        },
        "worst_action_mse_minus_prior_rows": sorted(
            per_row, key=lambda item: float(item["action_mse_minus_prior"]), reverse=True
        )[:20],
        "boundary": (
            "Offline checkpoint inspection only. It does not prove live task "
            "success without closed-loop final-state metrics and inspected video."
        ),
    }
    output_json = Path(args.output_json).resolve() if args.output_json else training_root / "post_training_group_metrics.json"
    write_json(output_json, payload)
    print(json.dumps(payload["overall"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
