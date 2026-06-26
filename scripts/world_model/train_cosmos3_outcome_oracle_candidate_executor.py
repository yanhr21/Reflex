#!/usr/bin/env python3
"""Train a candidate executor from oracle selections over real outcome labels."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_cosmos3_live_receding_loop import require_compute_step  # noqa: E402
from train_cosmos3_candidate_executor import (  # noqa: E402
    CandidateExecutorNet,
    candidate_eval,
    compute_phase_residual_l2_caps,
    fixed_width_vector,
    value_target,
)
from train_cosmos3_candidate_outcome_scorer import (  # noqa: E402
    build_base_feature,
    is_dp_prior_candidate,
    load_base_rows,
    load_candidate_actions,
    load_outcome_rows,
)
from train_cosmos3_contact_executor import split_indices  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-executor-jsonl", required=True)
    parser.add_argument("--outcome-jsonl", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-base-rows", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--oracle-min-improvement", type=float, default=0.005)
    parser.add_argument(
        "--hard-phases",
        default="far,lateral_align,preinsert_aligned",
        help="Comma-separated phases where candidate generation should be emphasized.",
    )
    parser.add_argument(
        "--hard-phase-oracle-min-improvement",
        type=float,
        default=-1.0,
        help="Override oracle-min-improvement for hard phases when non-negative.",
    )
    parser.add_argument(
        "--prefer-success-targets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When true, choose successful/inserted replay candidates before lower-error non-success candidates.",
    )
    parser.add_argument(
        "--phase-sampling",
        choices=("none", "balanced", "hard_boost"),
        default="none",
        help="Optional train sampler reweighting so hard phases do not get washed out by easy rows.",
    )
    parser.add_argument("--hard-phase-sample-weight", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--nll-loss-weight", type=float, default=0.25)
    parser.add_argument("--mean-action-loss-weight", type=float, default=1.0)
    parser.add_argument("--progress-loss-weight", type=float, default=0.5)
    parser.add_argument("--binary-loss-weight", type=float, default=0.3)
    parser.add_argument("--value-loss-weight", type=float, default=0.5)
    parser.add_argument("--next-state-loss-weight", type=float, default=0.5)
    parser.add_argument("--logstd-min", type=float, default=-4.5)
    parser.add_argument("--logstd-max", type=float, default=1.0)
    parser.add_argument("--generator-type", choices=("gaussian",), default="gaussian")
    parser.add_argument("--candidate-samples", type=int, default=0)
    parser.add_argument("--candidate-temps", default="1.0")
    parser.add_argument("--candidate-scales", default="0.2,0.5,1.0")
    parser.add_argument("--diffusion-steps", type=int, default=0)
    parser.add_argument("--diffusion-beta-start", type=float, default=1e-4)
    parser.add_argument("--diffusion-beta-end", type=float, default=2e-2)
    parser.add_argument("--candidate-rank-loss-weight", type=float, default=0.0)
    parser.add_argument("--candidate-rank-random-count", type=int, default=0)
    parser.add_argument("--candidate-rank-diffusion-count", type=int, default=0)
    parser.add_argument("--candidate-rank-temperature", type=float, default=1.0)
    parser.add_argument("--score-inserted-weight", type=float, default=0.4)
    parser.add_argument("--score-dp-continuable-weight", type=float, default=0.2)
    parser.add_argument("--score-value-weight", type=float, default=0.8)
    parser.add_argument("--score-next-state-weight", type=float, default=0.8)
    parser.add_argument("--score-next-state-axis-weights", default="1.0,2.0,4.0")
    parser.add_argument("--score-next-state-target", default="0.0,0.0,0.0")
    parser.add_argument("--score-logprob-weight", type=float, default=0.05)
    parser.add_argument("--score-residual-l2-penalty", type=float, default=0.02)
    parser.add_argument("--score-mean-source-penalty", type=float, default=0.0)
    parser.add_argument("--score-scale-source-penalty", type=float, default=0.0)
    parser.add_argument("--score-large-scale-source-penalty", type=float, default=0.0)
    parser.add_argument("--score-stochastic-source-penalty", type=float, default=0.25)
    parser.add_argument("--dp-fallback-phases", default="all")
    parser.add_argument("--dp-fallback-score-margin", type=float, default=0.3)
    parser.add_argument("--selector-residual-l2-cap-quantile", type=float, default=0.95)
    parser.add_argument("--selector-residual-l2-cap-min", type=float, default=1e-4)
    parser.add_argument("--selector-residual-l2-cap-max", type=float, default=0.2)
    parser.add_argument("--selector-residual-l2-cap-multiplier", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--min-steps", type=int, default=0)
    parser.add_argument("--min-wall-seconds", type=int, default=0)
    parser.add_argument("--max-wall-seconds", type=int, default=0)
    parser.add_argument("--formal-min-gpus", type=int, default=1)
    parser.add_argument("--eval-every-steps", type=int, default=50)
    parser.add_argument("--save-every-steps", type=int, default=200)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=20260618)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    torch.save(payload, tmp)
    os.replace(tmp, path)


def csv_set(text: str) -> set[str]:
    return {item.strip() for item in str(text).split(",") if item.strip()}


def outcome_success(row: dict[str, Any]) -> bool:
    return bool(row.get("final_success", False)) or bool(row.get("final_inserted_live_pose", False))


def choose_target(
    rows: list[dict[str, Any]],
    *,
    min_improvement: float,
    prefer_success_targets: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    dp_rows = [row for row in rows if is_dp_prior_candidate(str(row.get("candidate_name") or ""))]
    dp_row = dp_rows[0] if dp_rows else None
    oracle_pool = rows
    if bool(prefer_success_targets):
        success_rows = [row for row in rows if outcome_success(row)]
        if success_rows:
            oracle_pool = success_rows
    oracle_row = min(oracle_pool, key=lambda row: float(row.get("final_abs_task_error_weighted", float("inf"))))
    if dp_row is None:
        return oracle_row, None, oracle_row
    dp_error = float(dp_row["final_abs_task_error_weighted"])
    oracle_error = float(oracle_row["final_abs_task_error_weighted"])
    if oracle_error <= dp_error - float(min_improvement):
        return oracle_row, dp_row, oracle_row
    return dp_row, dp_row, oracle_row


def build_dataset(
    *,
    base_rows: dict[str, dict[str, Any]],
    outcome_rows: list[dict[str, Any]],
    min_improvement: float,
    hard_phases: set[str],
    hard_phase_min_improvement: float,
    prefer_success_targets: bool,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcome_rows:
        grouped[str(row.get("uuid") or "")].append(row)

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    priors: list[np.ndarray] = []
    progress_targets: list[np.ndarray] = []
    binary_targets: list[np.ndarray] = []
    next_state_targets: list[np.ndarray] = []
    used_rows: list[dict[str, Any]] = []
    for uuid, base_row in base_rows.items():
        candidates = grouped.get(str(uuid), [])
        if not candidates:
            continue
        base_feature, prior_flat, horizon, base_meta = build_base_feature(base_row)
        phase = str(base_row.get("current_phase") or "unknown")
        phase_min_improvement = (
            float(hard_phase_min_improvement)
            if phase in hard_phases and float(hard_phase_min_improvement) >= 0.0
            else float(min_improvement)
        )
        selected_row, dp_row, oracle_row = choose_target(
            candidates,
            min_improvement=phase_min_improvement,
            prefer_success_targets=bool(prefer_success_targets),
        )
        actions = load_candidate_actions(selected_row, horizon)
        target_flat = actions.reshape(-1).astype(np.float32)
        if target_flat.shape != prior_flat.shape:
            continue
        selected_error = float(selected_row["final_abs_task_error_weighted"])
        dp_error = float(dp_row["final_abs_task_error_weighted"]) if dp_row is not None else selected_error
        oracle_error = float(oracle_row["final_abs_task_error_weighted"])
        improvement = float(dp_error - selected_error)
        final_state = np.asarray(selected_row["final_peg_head_at_hole"], dtype=np.float32).reshape(3)
        inserted_or_success = float(
            bool(selected_row.get("final_success", False))
            or bool(selected_row.get("final_inserted_live_pose", False))
        )
        non_harmful = float(selected_error <= dp_error + 1e-6)
        features.append(base_feature.astype(np.float32))
        targets.append(target_flat)
        priors.append(prior_flat.astype(np.float32))
        progress_targets.append(np.asarray([max(0.0, improvement), improvement], dtype=np.float32))
        binary_targets.append(np.asarray([inserted_or_success, non_harmful], dtype=np.float32))
        next_state_targets.append(final_state.astype(np.float32))
        used_rows.append(
            {
                **base_meta,
                "target_candidate_name": selected_row.get("candidate_name"),
                "target_candidate_source": selected_row.get("candidate_source"),
                "target_final_abs_task_error_weighted": selected_error,
                "dp_prior_final_abs_task_error_weighted": dp_error,
                "oracle_final_abs_task_error_weighted": oracle_error,
                "target_minus_dp_error": float(selected_error - dp_error),
                "oracle_minus_dp_error": float(oracle_error - dp_error),
                "oracle_min_improvement_for_phase": float(phase_min_improvement),
                "target_final_peg_head_at_hole": final_state.astype(float).tolist(),
                "target_final_success": bool(selected_row.get("final_success", False)),
                "target_final_inserted_live_pose": bool(selected_row.get("final_inserted_live_pose", False)),
                "target_final_grasped": bool(selected_row.get("final_grasped", False)),
                "outcome_jsonl": selected_row.get("_outcome_jsonl"),
                "boundary": (
                    "Outcome-oracle executor target. The target is DP by default "
                    "unless a real replayed candidate beats DP by the configured "
                    "margin for the same observed state."
                ),
            }
        )
    if not features:
        raise RuntimeError("no outcome-oracle executor samples")
    widths = {
        "feature": {item.shape[0] for item in features},
        "target": {item.shape[0] for item in targets},
        "prior": {item.shape[0] for item in priors},
    }
    if any(len(value) != 1 for value in widths.values()) or widths["target"] != widths["prior"]:
        raise RuntimeError(f"nonuniform outcome-oracle widths: {widths}")
    return {
        "x": np.stack(features).astype(np.float32),
        "y": np.stack(targets).astype(np.float32),
        "prior": np.stack(priors).astype(np.float32),
        "progress": np.stack(progress_targets).astype(np.float32),
        "binary": np.stack(binary_targets).astype(np.float32),
        "next_state": np.stack(next_state_targets).astype(np.float32),
        "used_rows": used_rows,
    }


def phase_sample_weights(
    used_rows: list[dict[str, Any]],
    train_idx: np.ndarray,
    *,
    phase_sampling: str,
    hard_phases: set[str],
    hard_phase_sample_weight: float,
) -> np.ndarray | None:
    mode = str(phase_sampling)
    if mode == "none":
        return None
    train_idx = np.asarray(train_idx, dtype=np.int64)
    phases = [str(used_rows[int(i)].get("current_phase") or "unknown") for i in train_idx]
    counts = Counter(phases)
    weights: list[float] = []
    for phase in phases:
        if mode == "balanced":
            weights.append(1.0 / max(1, int(counts[phase])))
        elif mode == "hard_boost":
            weights.append(float(hard_phase_sample_weight) if phase in hard_phases else 1.0)
        else:
            raise ValueError(f"unknown phase sampling mode: {mode}")
    arr = np.asarray(weights, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else 1.0
    if mean > 0:
        arr = arr / mean
    return arr.astype(np.float64)


def phase_sampling_summary(
    used_rows: list[dict[str, Any]],
    train_idx: np.ndarray,
    weights: np.ndarray | None,
) -> dict[str, Any]:
    train_idx = np.asarray(train_idx, dtype=np.int64)
    phases = [str(used_rows[int(i)].get("current_phase") or "unknown") for i in train_idx]
    counts = Counter(phases)
    summary: dict[str, Any] = {"phase_counts": dict(sorted(counts.items()))}
    if weights is not None:
        weight_by_phase: dict[str, list[float]] = defaultdict(list)
        for phase, weight in zip(phases, weights):
            weight_by_phase[phase].append(float(weight))
        summary["mean_weight_by_phase"] = {
            phase: float(np.mean(values)) for phase, values in sorted(weight_by_phase.items())
        }
        summary["min_weight"] = float(np.min(weights)) if len(weights) else None
        summary["max_weight"] = float(np.max(weights)) if len(weights) else None
    return summary


def summarize_targets(used_rows: list[dict[str, Any]], indices: np.ndarray) -> dict[str, Any]:
    selected_delta = np.asarray(
        [float(used_rows[int(i)]["target_minus_dp_error"]) for i in indices],
        dtype=np.float32,
    )
    oracle_delta = np.asarray(
        [float(used_rows[int(i)]["oracle_minus_dp_error"]) for i in indices],
        dtype=np.float32,
    )
    names = [str(used_rows[int(i)].get("target_candidate_name") or "") for i in indices]
    phases = [str(used_rows[int(i)].get("current_phase") or "unknown") for i in indices]
    non_dp = sum(1 for name in names if not is_dp_prior_candidate(name))
    return {
        "num_rows": int(len(indices)),
        "target_minus_dp_error_mean": float(selected_delta.mean()) if selected_delta.size else None,
        "oracle_minus_dp_error_mean": float(oracle_delta.mean()) if oracle_delta.size else None,
        "target_non_dp_count": int(non_dp),
        "target_non_dp_fraction": float(non_dp / max(1, len(indices))),
        "target_candidate_counts": dict(sorted(Counter(names).items())),
        "phase_counts": dict(sorted(Counter(phases).items())),
    }


def main() -> int:
    args = parse_args()
    require_compute_step()

    import torch
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("require_cuda=true but torch.cuda.is_available() is false")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.set_float32_matmul_precision("high")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base_rows = load_base_rows(Path(args.contact_executor_jsonl).resolve(), int(args.max_base_rows))
    outcome_rows = load_outcome_rows(list(args.outcome_jsonl), dedupe=True)
    hard_phases = csv_set(str(args.hard_phases))
    dataset = build_dataset(
        base_rows=base_rows,
        outcome_rows=outcome_rows,
        min_improvement=float(args.oracle_min_improvement),
        hard_phases=hard_phases,
        hard_phase_min_improvement=float(args.hard_phase_oracle_min_improvement),
        prefer_success_targets=bool(args.prefer_success_targets),
    )
    x_raw = dataset["x"]
    y_raw = dataset["y"]
    prior_raw = dataset["prior"]
    progress_raw = dataset["progress"]
    binary_raw = dataset["binary"]
    next_state_raw = dataset["next_state"]
    used_rows = dataset["used_rows"]
    train_idx, val_idx = split_indices(len(used_rows), float(args.val_fraction), int(args.seed))
    eval_idx = val_idx if len(val_idx) else train_idx
    train_weights = phase_sample_weights(
        used_rows,
        train_idx,
        phase_sampling=str(args.phase_sampling),
        hard_phases=hard_phases,
        hard_phase_sample_weight=float(args.hard_phase_sample_weight),
    )

    x_mean = x_raw[train_idx].mean(axis=0, keepdims=True)
    x_std = np.where(x_raw[train_idx].std(axis=0, keepdims=True) < 1e-6, 1.0, x_raw[train_idx].std(axis=0, keepdims=True))
    residual_raw = y_raw - prior_raw
    residual_mean = residual_raw[train_idx].mean(axis=0, keepdims=True)
    residual_std = np.where(
        residual_raw[train_idx].std(axis=0, keepdims=True) < 1e-6,
        1.0,
        residual_raw[train_idx].std(axis=0, keepdims=True),
    )
    x_norm = ((x_raw - x_mean) / x_std).astype(np.float32)
    residual_norm = ((residual_raw - residual_mean) / residual_std).astype(np.float32)
    value_raw = value_target(progress_raw, binary_raw)
    phase_residual_l2_caps = compute_phase_residual_l2_caps(
        used_rows=used_rows,
        train_idx=train_idx,
        residual_raw=residual_raw,
        args=args,
    )

    train_ds = TensorDataset(
        torch.from_numpy(x_norm[train_idx].astype(np.float32)),
        torch.from_numpy(residual_norm[train_idx].astype(np.float32)),
        torch.from_numpy(progress_raw[train_idx].astype(np.float32)),
        torch.from_numpy(binary_raw[train_idx].astype(np.float32)),
        torch.from_numpy(value_raw[train_idx].astype(np.float32)),
        torch.from_numpy(next_state_raw[train_idx].astype(np.float32)),
    )
    sampler = None
    shuffle = True
    if train_weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(train_weights, dtype=torch.double),
            num_samples=int(len(train_idx)),
            replacement=True,
        )
        shuffle = False
    loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )
    model = CandidateExecutorNet(
        feature_dim=int(x_norm.shape[1]),
        target_dim=int(y_raw.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        logstd_min=float(args.logstd_min),
        logstd_max=float(args.logstd_max),
        next_state_dim=int(next_state_raw.shape[1]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    bce = torch.nn.BCEWithLogitsLoss()

    write_json(
        output_root / "training_manifest.json",
        {
            "schema": "cosmos3_outcome_oracle_candidate_executor_training_v1",
            "contact_executor_jsonl": str(Path(args.contact_executor_jsonl).resolve()),
            "outcome_jsonl": [str(Path(item).resolve()) for item in args.outcome_jsonl],
            "output_root": str(output_root),
            "num_samples": int(len(used_rows)),
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "feature_dim": int(x_norm.shape[1]),
            "target_dim": int(y_raw.shape[1]),
            "next_state_dim": int(next_state_raw.shape[1]),
            "action_horizon": int(y_raw.shape[1] // 7),
            "oracle_min_improvement": float(args.oracle_min_improvement),
            "hard_phases": sorted(hard_phases),
            "hard_phase_oracle_min_improvement": float(args.hard_phase_oracle_min_improvement),
            "prefer_success_targets": bool(args.prefer_success_targets),
            "phase_sampling": str(args.phase_sampling),
            "hard_phase_sample_weight": float(args.hard_phase_sample_weight),
            "train_targets": summarize_targets(used_rows, train_idx),
            "eval_targets": summarize_targets(used_rows, eval_idx),
            "train_phase_sampling": phase_sampling_summary(used_rows, train_idx, train_weights),
            "selector_phase_residual_l2_caps": dict(sorted(phase_residual_l2_caps.items())),
            "device": str(device),
            "visible_cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "method_boundary": (
                "Outcome-oracle candidate executor smoke. It distills the best "
                "real replayed short chunk per observed state, with DP as the "
                "default target unless another candidate clearly improves real "
                "final error. It is not live method evidence until replayed in "
                "closed loop with video/contact review."
            ),
            "sample_rows": used_rows[:10],
        },
    )

    history: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] | None = None
    best_eval_mean_mse = float("inf")
    best_step = 0
    start = time.time()
    step = 0
    last_save = 0
    stop_reason = "running"
    residual_mean_t = torch.from_numpy(residual_mean.astype(np.float32)).to(device)
    residual_std_t = torch.from_numpy(residual_std.astype(np.float32)).to(device)
    while True:
        for batch in loader:
            step += 1
            x_b, resid_b, progress_b, binary_b, value_b, next_state_b = [
                item.to(device, non_blocking=True) for item in batch
            ]
            mean_norm, logstd, pred_progress, pred_logits, pred_value, pred_next_state = model(x_b, resid_b)
            std = torch.exp(logstd)
            nll = 0.5 * torch.mean(((resid_b - mean_norm) / std) ** 2 + 2.0 * logstd)
            mean_loss = torch.mean((mean_norm - resid_b) ** 2)
            progress_loss = torch.mean((pred_progress - progress_b) ** 2)
            binary_loss = bce(pred_logits, binary_b)
            value_loss = torch.mean((pred_value - value_b) ** 2)
            next_state_loss = torch.mean((pred_next_state - next_state_b) ** 2)
            loss = (
                float(args.nll_loss_weight) * nll
                + float(args.mean_action_loss_weight) * mean_loss
                + float(args.progress_loss_weight) * progress_loss
                + float(args.binary_loss_weight) * binary_loss
                + float(args.value_loss_weight) * value_loss
                + float(args.next_state_loss_weight) * next_state_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step == 1 or step % int(args.eval_every_steps) == 0:
                eval_metrics = candidate_eval(
                    model=model,
                    x_norm=x_norm,
                    y_raw=y_raw,
                    prior_raw=prior_raw,
                    residual_norm=residual_norm,
                    residual_mean=residual_mean,
                    residual_std=residual_std,
                    progress_raw=progress_raw,
                    binary_raw=binary_raw,
                    next_state_raw=next_state_raw,
                    indices=eval_idx,
                    used_rows=used_rows,
                    phase_residual_l2_caps=phase_residual_l2_caps,
                    args=args,
                    device=device,
                )
                train_metrics = candidate_eval(
                    model=model,
                    x_norm=x_norm,
                    y_raw=y_raw,
                    prior_raw=prior_raw,
                    residual_norm=residual_norm,
                    residual_mean=residual_mean,
                    residual_std=residual_std,
                    progress_raw=progress_raw,
                    binary_raw=binary_raw,
                    next_state_raw=next_state_raw,
                    indices=train_idx,
                    used_rows=used_rows,
                    phase_residual_l2_caps=phase_residual_l2_caps,
                    args=args,
                    device=device,
                )
                metrics = {
                    "step": int(step),
                    "elapsed_seconds": float(time.time() - start),
                    "train_batch_loss": float(loss.detach().cpu().item()),
                    "train_batch_nll": float(nll.detach().cpu().item()),
                    "train_batch_mean_residual_mse": float(mean_loss.detach().cpu().item()),
                    "train_batch_progress_mse": float(progress_loss.detach().cpu().item()),
                    "train_batch_binary_bce": float(binary_loss.detach().cpu().item()),
                    "train_batch_value_mse": float(value_loss.detach().cpu().item()),
                    "train_batch_next_state_mse": float(next_state_loss.detach().cpu().item()),
                    "train": train_metrics,
                    "eval": eval_metrics,
                }
                history.append(metrics)
                write_json(output_root / "training_history.json", history)
                mean_mse = float(eval_metrics.get("mean_action_mse", float("inf")))
                if np.isfinite(mean_mse) and mean_mse < best_eval_mean_mse:
                    best_eval_mean_mse = mean_mse
                    best_metrics = metrics
                    best_step = int(step)
                    atomic_torch_save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": opt.state_dict(),
                            "step": int(step),
                            "feature_dim": int(x_norm.shape[1]),
                            "target_dim": int(y_raw.shape[1]),
                            "next_state_dim": int(next_state_raw.shape[1]),
                            "x_mean": x_mean.astype(np.float32),
                            "x_std": x_std.astype(np.float32),
                            "residual_mean": residual_mean.astype(np.float32),
                            "residual_std": residual_std.astype(np.float32),
                            "phase_residual_l2_caps": dict(sorted(phase_residual_l2_caps.items())),
                            "args": vars(args),
                            "latest_metrics": metrics,
                            "best_selection_metric": "eval.mean_action_mse_to_outcome_oracle_target",
                        },
                        output_root / "checkpoint_best_offline.pt",
                    )
                print(json.dumps(jsonable(metrics), sort_keys=True), flush=True)

            if step == 1 or step - last_save >= int(args.save_every_steps):
                atomic_torch_save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "step": int(step),
                        "feature_dim": int(x_norm.shape[1]),
                        "target_dim": int(y_raw.shape[1]),
                        "next_state_dim": int(next_state_raw.shape[1]),
                        "x_mean": x_mean.astype(np.float32),
                        "x_std": x_std.astype(np.float32),
                        "residual_mean": residual_mean.astype(np.float32),
                        "residual_std": residual_std.astype(np.float32),
                        "phase_residual_l2_caps": dict(sorted(phase_residual_l2_caps.items())),
                        "args": vars(args),
                        "latest_metrics": history[-1] if history else {},
                    },
                    output_root / "checkpoint_latest.pt",
                )
                last_save = step

            elapsed = time.time() - start
            if int(args.min_wall_seconds) > 0:
                max_wall = int(args.max_wall_seconds) if int(args.max_wall_seconds) > 0 else int(args.min_wall_seconds)
                if elapsed >= max_wall:
                    stop_reason = "max_wall_seconds"
                elif elapsed >= int(args.min_wall_seconds) and step >= int(args.min_steps):
                    stop_reason = "min_wall_and_min_steps"
            elif step >= int(args.max_steps):
                stop_reason = "max_steps"
            if stop_reason != "running":
                break
        if stop_reason != "running":
            break

    final_metrics = history[-1] if history else {}
    final_path = output_root / "checkpoint_final.pt"
    atomic_torch_save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "step": int(step),
            "feature_dim": int(x_norm.shape[1]),
            "target_dim": int(y_raw.shape[1]),
            "next_state_dim": int(next_state_raw.shape[1]),
            "x_mean": x_mean.astype(np.float32),
            "x_std": x_std.astype(np.float32),
            "residual_mean": residual_mean.astype(np.float32),
            "residual_std": residual_std.astype(np.float32),
            "phase_residual_l2_caps": dict(sorted(phase_residual_l2_caps.items())),
            "args": vars(args),
            "latest_metrics": final_metrics,
        },
        final_path,
    )
    elapsed = float(time.time() - start)
    visible_gpus = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    formal_floor_met = bool(
        visible_gpus >= int(args.formal_min_gpus)
        and int(args.min_wall_seconds) >= 10800
        and elapsed >= int(args.min_wall_seconds)
    )
    best_metrics_for_summary = best_metrics if best_metrics is not None else final_metrics
    summary = {
        "schema": "cosmos3_outcome_oracle_candidate_executor_training_summary_v1",
        "contact_executor_jsonl": str(Path(args.contact_executor_jsonl).resolve()),
        "outcome_jsonl": [str(Path(item).resolve()) for item in args.outcome_jsonl],
        "output_root": str(output_root),
        "num_samples": int(len(used_rows)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "steps": int(step),
        "elapsed_seconds": elapsed,
        "stop_reason": stop_reason,
        "visible_cuda_device_count": visible_gpus,
        "formal_min_gpus": int(args.formal_min_gpus),
        "formal_training_floor_met": formal_floor_met,
        "feature_dim": int(x_norm.shape[1]),
        "target_dim": int(y_raw.shape[1]),
        "next_state_dim": int(next_state_raw.shape[1]),
        "action_horizon": int(y_raw.shape[1] // 7),
        "oracle_min_improvement": float(args.oracle_min_improvement),
        "hard_phases": sorted(hard_phases),
        "hard_phase_oracle_min_improvement": float(args.hard_phase_oracle_min_improvement),
        "prefer_success_targets": bool(args.prefer_success_targets),
        "phase_sampling": str(args.phase_sampling),
        "hard_phase_sample_weight": float(args.hard_phase_sample_weight),
        "train_targets": summarize_targets(used_rows, train_idx),
        "eval_targets": summarize_targets(used_rows, eval_idx),
        "train_phase_sampling": phase_sampling_summary(used_rows, train_idx, train_weights),
        "selector_phase_residual_l2_caps": dict(sorted(phase_residual_l2_caps.items())),
        "final_metrics": final_metrics,
        "best_offline_metrics": best_metrics_for_summary,
        "best_offline_step": int(best_step),
        "best_offline_checkpoint": str(output_root / "checkpoint_best_offline.pt"),
        "ready_for_candidate_replay_smoke": bool(
            (output_root / "checkpoint_best_offline.pt").is_file()
            and float((best_metrics_for_summary.get("eval") or {}).get("mean_action_mse", float("inf")))
            < float((best_metrics_for_summary.get("eval") or {}).get("dp_prior_action_mse", float("inf")))
        ),
        "boundary": (
            "Outcome-oracle candidate executor training only. This is an aligned "
            "candidate-generator repair, not method evidence. It must be replayed "
            "as candidate outcomes and then tested in live receding control before "
            "any success claim."
        ),
    }
    write_json(output_root / "training_summary.json", summary)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True), flush=True)
    _ = residual_mean_t, residual_std_t
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
