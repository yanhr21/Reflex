#!/usr/bin/env python3
"""Train a direct-contact action diffusion model from the clean manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import socket
import time
from typing import Any

import numpy as np


PHASE_NAMES = ["lost_grasp", "far", "lateral_align", "preinsert_aligned", "dp_continuable", "inserted"]


def require_slurm_compute_step() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit(
            "Refusing to train outside a Slurm allocation. "
            f"host={socket.gethostname()}. Use tmux-held allocation + srun."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--actions-npz", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument(
        "--risk-loss-weight",
        type=float,
        default=0.25,
        help="Auxiliary BCE weight for direct-positive versus live-hard-negative action scoring.",
    )
    parser.add_argument("--diffusion-steps", type=int, default=32)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--min-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=500000)
    parser.add_argument("--min-wall-seconds", type=int, default=3660)
    parser.add_argument("--max-wall-seconds", type=int, default=3900)
    parser.add_argument("--eval-every-steps", type=int, default=200)
    parser.add_argument("--save-every-steps", type=int, default=1000)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=20260624)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


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


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    torch.save(payload, tmp)
    os.replace(tmp, path)


def phase_one_hot(value: Any) -> np.ndarray:
    out = np.zeros((len(PHASE_NAMES) + 1,), dtype=np.float32)
    try:
        idx = int(value)
    except Exception:
        idx = -1
    if 0 <= idx < len(PHASE_NAMES):
        out[idx] = 1.0
    else:
        out[-1] = 1.0
    return out


def as_vec3(value: Any) -> np.ndarray:
    arr = np.asarray(value if value is not None else [0.0, 0.0, 0.0], dtype=np.float32).reshape(-1)
    out = np.zeros((3,), dtype=np.float32)
    out[: min(3, arr.shape[0])] = arr[:3]
    return out


def row_features(row: dict[str, Any], horizon: int) -> np.ndarray:
    head = as_vec3(row.get("current_peg_head_at_hole"))
    abs_y = abs(float(head[1]))
    abs_z = abs(float(head[2]))
    yz_l2 = math.sqrt(float(head[1]) ** 2 + float(head[2]) ** 2)
    start = float(row.get("action_start_frame") or row.get("prefix_frame_index") or 0.0)
    offset = float(row.get("offset_before_insert") or 0.0)
    progress = float(row.get("current_contact_progress") or 0.0)
    vals = np.asarray(
        [
            float(head[0]),
            float(head[1]),
            float(head[2]),
            abs_y,
            abs_z,
            abs_y + abs_z,
            yz_l2,
            progress,
            float(bool(row.get("current_grasped", row.get("grasp_preserved_label", False)))),
            float(bool(row.get("current_dp_continuable", False))),
            offset / 96.0,
            start / 300.0,
            (300.0 - start) / 300.0,
            float(row.get("valid_action_steps") or horizon) / float(horizon),
        ],
        dtype=np.float32,
    )
    return np.concatenate([vals, phase_one_hot(row.get("current_phase_id"))]).astype(np.float32)


def load_training_arrays(rows: list[dict[str, Any]], actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    if actions.ndim != 3 or actions.shape[-1] != 7:
        raise RuntimeError(f"expected actions shape (N,H,7), got {actions.shape}")
    horizon = int(actions.shape[1])
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    used_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for row in rows:
        idx = int(row.get("array_index", -1))
        if idx < 0 or idx >= actions.shape[0]:
            skipped["bad_array_index"] += 1
            continue
        if not bool(row.get("direct_contact_positive", False)):
            skipped["not_direct_positive"] += 1
            continue
        if str(row.get("supervision_role")) != "primary_action_positive":
            skipped["not_primary_positive"] += 1
            continue
        features.append(row_features(row, horizon))
        targets.append(actions[idx].reshape(-1).astype(np.float32))
        used_rows.append(row)
    if not features:
        raise RuntimeError("no primary direct-contact positive rows")
    metadata = {
        "num_used_rows": len(used_rows),
        "num_manifest_rows": len(rows),
        "horizon": horizon,
        "action_dim": 7,
        "target_dim": horizon * 7,
        "feature_dim": int(features[0].shape[0]),
        "skipped_counts": dict(sorted(skipped.items())),
        "sample_kind_counts": dict(sorted(Counter(str(row.get("sample_kind")) for row in used_rows).items())),
        "scenario_counts_diagnostic_only": dict(sorted(Counter(str(row.get("scenario")) for row in used_rows).items())),
    }
    return np.stack(features).astype(np.float32), np.stack(targets).astype(np.float32), used_rows, metadata


def load_risk_arrays(rows: list[dict[str, Any]], actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    if actions.ndim != 3 or actions.shape[-1] != 7:
        raise RuntimeError(f"expected actions shape (N,H,7), got {actions.shape}")
    horizon = int(actions.shape[1])
    features: list[np.ndarray] = []
    action_flat: list[np.ndarray] = []
    labels: list[float] = []
    used_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for row in rows:
        idx = int(row.get("array_index", -1))
        if idx < 0 or idx >= actions.shape[0]:
            skipped["bad_array_index"] += 1
            continue
        is_positive = bool(row.get("direct_contact_positive", False)) and str(row.get("supervision_role")) == "primary_action_positive"
        is_hard_negative = str(row.get("sample_kind", "")).endswith("hard_negative")
        if not (is_positive or is_hard_negative):
            skipped["not_risk_supervision"] += 1
            continue
        features.append(row_features(row, horizon))
        action_flat.append(actions[idx].reshape(-1).astype(np.float32))
        labels.append(float(is_positive))
        used_rows.append(row)
    if not features:
        raise RuntimeError("no risk/value supervision rows")
    label_arr = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
    metadata = {
        "num_risk_rows": len(used_rows),
        "risk_positive_rows": int(label_arr.sum()),
        "risk_negative_rows": int(label_arr.shape[0] - label_arr.sum()),
        "risk_sample_kind_counts": dict(sorted(Counter(str(row.get("sample_kind")) for row in used_rows).items())),
        "risk_candidate_family_counts": dict(sorted(Counter(str(row.get("candidate_family")) for row in used_rows).items())),
        "risk_skipped_counts": dict(sorted(skipped.items())),
    }
    return (
        np.stack(features).astype(np.float32),
        np.stack(action_flat).astype(np.float32),
        label_arr,
        used_rows,
        metadata,
    )


def split_by_source(rows: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    sources = np.asarray(sorted({str(row.get("source_uuid") or row.get("source_h5") or i) for i, row in enumerate(rows)}), dtype=object)
    rng.shuffle(sources)
    if sources.size <= 1 or val_fraction <= 0:
        val_sources: set[str] = set()
    else:
        n_val = max(1, int(round(sources.size * float(val_fraction))))
        n_val = min(n_val, sources.size - 1)
        val_sources = set(str(x) for x in sources[:n_val].tolist())
    train_idx: list[int] = []
    val_idx: list[int] = []
    for i, row in enumerate(rows):
        src = str(row.get("source_uuid") or row.get("source_h5") or i)
        if src in val_sources:
            val_idx.append(i)
        else:
            train_idx.append(i)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def split_risk_by_label(labels: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    flat = np.asarray(labels, dtype=np.float32).reshape(-1)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for positive in (False, True):
        idx = np.flatnonzero(flat >= 0.5) if positive else np.flatnonzero(flat < 0.5)
        if idx.size == 0:
            continue
        rng.shuffle(idx)
        if idx.size == 1 or val_fraction <= 0:
            n_val = 0
        else:
            n_val = max(1, int(round(idx.size * float(val_fraction))))
            n_val = min(n_val, idx.size - 1)
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train = np.concatenate([part for part in train_parts if part.size], axis=0)
    val = np.concatenate([part for part in val_parts if part.size], axis=0)
    if train.size == 0:
        train = np.arange(flat.shape[0], dtype=np.int64)
    if val.size == 0:
        val = train[: min(train.size, max(1, train.size // 10))]
    rng.shuffle(train)
    rng.shuffle(val)
    return train.astype(np.int64), val.astype(np.int64)


def stats_1d(arr: np.ndarray) -> dict[str, Any]:
    values = np.asarray(arr, dtype=np.float64).reshape(-1)
    return {
        "n": int(values.size),
        "min": float(values.min()) if values.size else 0.0,
        "mean": float(values.mean()) if values.size else 0.0,
        "median": float(np.median(values)) if values.size else 0.0,
        "p90": float(np.percentile(values, 90)) if values.size else 0.0,
        "max": float(values.max()) if values.size else 0.0,
    }


def diffusion_schedule(steps: int, beta_start: float, beta_end: float, device: Any) -> Any:
    import torch

    step_count = max(2, int(steps))
    betas = torch.linspace(float(beta_start), float(beta_end), step_count, device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def timestep_embedding(t: Any, dim: int) -> Any:
    import torch

    half = int(dim) // 2
    freqs = torch.exp(
        torch.arange(half, device=t.device, dtype=torch.float32)
        * (-math.log(10000.0) / max(1, half - 1))
    )
    angles = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if emb.shape[1] < int(dim):
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=t.device, dtype=emb.dtype)], dim=1)
    return emb


def make_model(input_dim: int, output_dim: int, args: argparse.Namespace) -> Any:
    import torch

    layers: list[Any] = []
    width = int(args.hidden_dim)
    dim = int(input_dim)
    for _ in range(int(args.num_layers)):
        layers.append(torch.nn.Linear(dim, width))
        layers.append(torch.nn.SiLU())
        if float(args.dropout) > 0:
            layers.append(torch.nn.Dropout(float(args.dropout)))
        dim = width
    layers.append(torch.nn.Linear(dim, int(output_dim)))
    return torch.nn.Sequential(*layers)


def eval_model(model: Any, x: Any, y: Any, alphas_cumprod: Any, device: Any, batch_size: int) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.eval()
    losses: list[float] = []
    x0_losses: list[float] = []
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(batch_size)):
            xb = x[start : start + int(batch_size)].to(device)
            yb = y[start : start + int(batch_size)].to(device)
            t = torch.full((xb.shape[0],), int(alphas_cumprod.shape[0]) // 2, device=device, dtype=torch.long)
            noise = torch.randn_like(yb)
            a = alphas_cumprod[t].reshape(-1, 1)
            noisy = torch.sqrt(a) * yb + torch.sqrt(1.0 - a) * noise
            pred = model(torch.cat([xb, noisy, timestep_embedding(t, 64)], dim=1))
            loss = F.mse_loss(pred, noise)
            x0 = (noisy - torch.sqrt(1.0 - a) * pred) / torch.sqrt(a)
            x0_loss = F.mse_loss(x0, yb)
            losses.append(float(loss.item()))
            x0_losses.append(float(x0_loss.item()))
    return {
        "denoise_mse": float(np.mean(losses)) if losses else 0.0,
        "x0_action_mse_mid_t": float(np.mean(x0_losses)) if x0_losses else 0.0,
    }


def eval_risk_model(risk_model: Any, x: Any, y: Any, device: Any, batch_size: int) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    risk_model.eval()
    losses: list[float] = []
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(batch_size)):
            xb = x[start : start + int(batch_size)].to(device)
            yb = y[start : start + int(batch_size)].to(device)
            logits = risk_model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            losses.append(float(loss.item()))
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels.append(yb.detach().cpu().numpy())
    if not probs:
        return {"risk_bce": 0.0, "risk_acc": 0.0, "risk_pos_prob_mean": 0.0, "risk_neg_prob_mean": 0.0}
    prob_np = np.concatenate(probs, axis=0).reshape(-1)
    label_np = np.concatenate(labels, axis=0).reshape(-1)
    pred_np = (prob_np >= 0.5).astype(np.float32)
    pos = prob_np[label_np >= 0.5]
    neg = prob_np[label_np < 0.5]
    return {
        "risk_bce": float(np.mean(losses)),
        "risk_acc": float(np.mean(pred_np == label_np)) if label_np.size else 0.0,
        "risk_pos_prob_mean": float(pos.mean()) if pos.size else 0.0,
        "risk_neg_prob_mean": float(neg.mean()) if neg.size else 0.0,
        "risk_margin_pos_minus_neg": float(pos.mean() - neg.mean()) if pos.size and neg.size else 0.0,
    }


def main() -> int:
    require_slurm_compute_step()
    args = parse_args()
    import torch
    import torch.nn.functional as F

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if bool(args.require_cuda) and not torch.cuda.is_available():
        raise SystemExit("CUDA is required but unavailable")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = read_jsonl(Path(args.manifest_jsonl).resolve())
    arrays = np.load(str(Path(args.actions_npz).resolve()), allow_pickle=False)
    actions_all = np.asarray(arrays["actions"], dtype=np.float32)
    features_np, targets_np, used_rows, meta = load_training_arrays(rows, actions_all)
    risk_features_np, risk_actions_np, risk_labels_np, risk_rows, risk_meta = load_risk_arrays(rows, actions_all)
    train_idx, val_idx = split_by_source(used_rows, float(args.val_fraction), int(args.seed))
    if train_idx.size == 0:
        raise RuntimeError("empty train split")
    if val_idx.size == 0:
        val_idx = train_idx[: min(train_idx.size, max(1, train_idx.size // 10))]
    risk_train_idx, risk_val_idx = split_risk_by_label(risk_labels_np, float(args.val_fraction), int(args.seed) + 17)
    if risk_train_idx.size == 0:
        raise RuntimeError("empty risk train split")
    if risk_val_idx.size == 0:
        risk_val_idx = risk_train_idx[: min(risk_train_idx.size, max(1, risk_train_idx.size // 10))]

    feature_mean = features_np[train_idx].mean(axis=0, keepdims=True)
    feature_std = features_np[train_idx].std(axis=0, keepdims=True) + 1e-6
    target_mean = targets_np[train_idx].mean(axis=0, keepdims=True)
    target_std = targets_np[train_idx].std(axis=0, keepdims=True) + 1e-6
    features_z = (features_np - feature_mean) / feature_std
    targets_z = (targets_np - target_mean) / target_std
    risk_features_z = (risk_features_np - feature_mean) / feature_std
    risk_actions_z = (risk_actions_np - target_mean) / target_std
    risk_input_np = np.concatenate([risk_features_z, risk_actions_z], axis=1).astype(np.float32)

    x_train = torch.from_numpy(features_z[train_idx]).float()
    y_train = torch.from_numpy(targets_z[train_idx]).float()
    x_val = torch.from_numpy(features_z[val_idx]).float()
    y_val = torch.from_numpy(targets_z[val_idx]).float()
    risk_x_train = torch.from_numpy(risk_input_np[risk_train_idx]).float()
    risk_y_train = torch.from_numpy(risk_labels_np[risk_train_idx]).float()
    risk_x_val = torch.from_numpy(risk_input_np[risk_val_idx]).float()
    risk_y_val = torch.from_numpy(risk_labels_np[risk_val_idx]).float()
    risk_pos_train = np.flatnonzero(risk_labels_np[risk_train_idx].reshape(-1) >= 0.5)
    risk_neg_train = np.flatnonzero(risk_labels_np[risk_train_idx].reshape(-1) < 0.5)
    alphas = diffusion_schedule(int(args.diffusion_steps), float(args.beta_start), float(args.beta_end), device)
    model = make_model(x_train.shape[1] + y_train.shape[1] + 64, y_train.shape[1], args).to(device)
    risk_model = make_model(risk_x_train.shape[1], 1, args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    risk_opt = torch.optim.AdamW(risk_model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    metadata = {
        **meta,
        **risk_meta,
        "manifest_jsonl": str(Path(args.manifest_jsonl).resolve()),
        "actions_npz": str(Path(args.actions_npz).resolve()),
        "output_root": str(output_root),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "train_rows": int(train_idx.size),
        "val_rows": int(val_idx.size),
        "risk_train_rows": int(risk_train_idx.size),
        "risk_val_rows": int(risk_val_idx.size),
        "risk_train_positive_rows": int(risk_pos_train.size),
        "risk_train_negative_rows": int(risk_neg_train.size),
        "risk_loss_weight": float(args.risk_loss_weight),
        "feature_schema": [
            "rel_x",
            "rel_y",
            "rel_z",
            "abs_y",
            "abs_z",
            "abs_yz_sum",
            "yz_l2",
            "current_contact_progress",
            "current_grasped",
            "current_dp_continuable",
            "offset_norm96",
            "start_frame_norm300",
            "remaining_episode_norm300",
            "valid_steps_norm_horizon",
        ]
        + [f"phase_{name}" for name in PHASE_NAMES]
        + ["phase_unknown"],
        "causal_feature_boundary": (
            "No scenario label, source id, future insertion frame, end contact "
            "state, or DP96 outcome is used as an input feature."
        ),
        "target_action_stats_abs": stats_1d(np.abs(targets_np)),
        "risk_boundary": (
            "The auxiliary risk head sees positive source insertion chunks and "
            "failed live replay chunks as supervised labels. It is a value/risk "
            "training signal only; it is not a controller input until sampled "
            "actions are replayed from saved live snapshots."
        ),
    }
    write_json(output_root / "training_metadata.json", metadata)

    start_time = time.time()
    best_val = float("inf")
    best_payload: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    step = 0
    stop_reason = "running"
    batch_size = int(args.batch_size)
    while True:
        model.train()
        risk_model.train()
        batch_idx = np.random.choice(train_idx.size, size=min(batch_size, train_idx.size), replace=train_idx.size < batch_size)
        xb = x_train[batch_idx].to(device)
        yb = y_train[batch_idx].to(device)
        t = torch.randint(0, int(alphas.shape[0]), (xb.shape[0],), device=device)
        noise = torch.randn_like(yb)
        a = alphas[t].reshape(-1, 1)
        noisy = torch.sqrt(a) * yb + torch.sqrt(1.0 - a) * noise
        pred = model(torch.cat([xb, noisy, timestep_embedding(t, 64)], dim=1))
        loss = F.mse_loss(pred, noise)
        if risk_pos_train.size and risk_neg_train.size and float(args.risk_loss_weight) > 0:
            half = max(1, min(batch_size // 2, risk_pos_train.size, risk_neg_train.size))
            risk_local_idx = np.concatenate(
                [
                    np.random.choice(risk_pos_train, size=half, replace=risk_pos_train.size < half),
                    np.random.choice(risk_neg_train, size=half, replace=risk_neg_train.size < half),
                ]
            )
        else:
            risk_local_idx = np.random.choice(
                risk_train_idx.size,
                size=min(batch_size, risk_train_idx.size),
                replace=risk_train_idx.size < batch_size,
            )
        rb = risk_x_train[risk_local_idx].to(device)
        ry = risk_y_train[risk_local_idx].to(device)
        risk_logits = risk_model(rb)
        risk_loss = F.binary_cross_entropy_with_logits(risk_logits, ry)

        opt.zero_grad(set_to_none=True)
        risk_opt.zero_grad(set_to_none=True)
        total_loss = loss + float(args.risk_loss_weight) * risk_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(risk_model.parameters(), 10.0)
        opt.step()
        risk_opt.step()
        step += 1

        elapsed = time.time() - start_time
        if step % int(args.eval_every_steps) == 0 or step == 1:
            train_eval = eval_model(model, x_train[: min(2048, x_train.shape[0])], y_train[: min(2048, y_train.shape[0])], alphas, device, batch_size)
            val_eval = eval_model(model, x_val, y_val, alphas, device, batch_size)
            risk_train_eval = eval_risk_model(
                risk_model,
                risk_x_train[: min(4096, risk_x_train.shape[0])],
                risk_y_train[: min(4096, risk_y_train.shape[0])],
                device,
                batch_size,
            )
            risk_val_eval = eval_risk_model(risk_model, risk_x_val, risk_y_val, device, batch_size)
            record = {
                "step": int(step),
                "elapsed_seconds": float(elapsed),
                "train_loss_last": float(loss.item()),
                "risk_loss_last": float(risk_loss.item()),
                "total_loss_last": float(total_loss.item()),
                "train_eval": train_eval,
                "val_eval": val_eval,
                "risk_train_eval": risk_train_eval,
                "risk_val_eval": risk_val_eval,
            }
            history.append(record)
            write_json(output_root / "training_progress.json", {"history": history[-50:], "latest": record})
            if val_eval["x0_action_mse_mid_t"] < best_val:
                best_val = float(val_eval["x0_action_mse_mid_t"])
                best_payload = {
                    "model_state": model.state_dict(),
                    "risk_model_state": risk_model.state_dict(),
                    "args": vars(args),
                    "metadata": metadata,
                    "feature_mean": feature_mean.astype(np.float32),
                    "feature_std": feature_std.astype(np.float32),
                    "target_mean": target_mean.astype(np.float32),
                    "target_std": target_std.astype(np.float32),
                    "best_record": record,
                }
                atomic_torch_save(best_payload, output_root / "checkpoint_best_eval.pt")
        if step % int(args.save_every_steps) == 0:
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "risk_model_state": risk_model.state_dict(),
                    "args": vars(args),
                    "metadata": metadata,
                    "feature_mean": feature_mean.astype(np.float32),
                    "feature_std": feature_std.astype(np.float32),
                    "target_mean": target_mean.astype(np.float32),
                    "target_std": target_std.astype(np.float32),
                    "step": int(step),
                },
                output_root / "checkpoint_latest.pt",
            )

        elapsed = time.time() - start_time
        if elapsed >= float(args.min_wall_seconds) and step >= int(args.min_steps):
            stop_reason = "min_wall_and_min_steps"
            break
        if int(args.max_wall_seconds) > 0 and elapsed >= float(args.max_wall_seconds):
            stop_reason = "max_wall_seconds"
            break
        if step >= int(args.max_steps):
            stop_reason = "max_steps"
            break

    final_eval = eval_model(model, x_val, y_val, alphas, device, batch_size)
    final_risk_eval = eval_risk_model(risk_model, risk_x_val, risk_y_val, device, batch_size)
    if best_payload is None:
        atomic_torch_save(
            {
                "model_state": model.state_dict(),
                "risk_model_state": risk_model.state_dict(),
                "args": vars(args),
                "metadata": metadata,
                "feature_mean": feature_mean.astype(np.float32),
                "feature_std": feature_std.astype(np.float32),
                "target_mean": target_mean.astype(np.float32),
                "target_std": target_std.astype(np.float32),
            },
            output_root / "checkpoint_best_eval.pt",
        )
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "risk_model_state": risk_model.state_dict(),
            "args": vars(args),
            "metadata": metadata,
            "feature_mean": feature_mean.astype(np.float32),
            "feature_std": feature_std.astype(np.float32),
            "target_mean": target_mean.astype(np.float32),
            "target_std": target_std.astype(np.float32),
            "step": int(step),
            "final_eval": final_eval,
            "final_risk_eval": final_risk_eval,
        },
        output_root / "checkpoint_final.pt",
    )
    elapsed = time.time() - start_time
    summary = {
        "schema": "direct_contact_executor_diffusion_training_summary_v1",
        "output_root": str(output_root),
        "stop_reason": stop_reason,
        "elapsed_seconds": float(elapsed),
        "steps": int(step),
        "formal_one_gpu_hour_floor_met": bool(elapsed >= 3600 and str(device).startswith("cuda")),
        "best_val_x0_action_mse_mid_t": float(best_val),
        "final_val_eval": final_eval,
        "final_risk_eval": final_risk_eval,
        "ready_for_saved_snapshot_replay_gate": bool(best_val < 0.5 and elapsed >= 3600),
        "metadata": metadata,
        "history_tail": history[-20:],
        "boundary": (
            "Training result only. It is not method evidence until sampled "
            "actions are replayed from saved dynamic live snapshots and visually reviewed."
        ),
    }
    write_json(output_root / "training_summary.json", summary)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
