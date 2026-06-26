#!/usr/bin/env python3
"""Train a live-outcome-conditioned contact action diffusion generator.

This is the contact-action reset follow-up to the failed deterministic
source-suffix MLP. It trains from real saved live snapshot candidate outcomes:
base causal state + DP prior + contact context condition the model, successful
or DP-continuable candidate chunks supervise action generation, and all
candidate outcomes supervise consequence/value heads.

The trainer is a diagnostic action-generator baseline. It is not live method
evidence until generated chunks are replayed on saved live states and then
validated in full live panels with visual/final-state review.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
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
from train_cosmos3_candidate_outcome_scorer import (  # noqa: E402
    build_base_feature,
    candidate_family,
    load_base_rows,
    load_candidate_actions,
    load_outcome_rows,
    outcome_progress_targets,
    read_jsonl,
    split_by_uuid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-rows-jsonl", required=True)
    parser.add_argument("--outcome-jsonl", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-base-rows", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--diffusion-steps", type=int, default=24)
    parser.add_argument("--diffusion-beta-start", type=float, default=1e-4)
    parser.add_argument("--diffusion-beta-end", type=float, default=2e-2)
    parser.add_argument("--diffusion-loss-weight", type=float, default=1.0)
    parser.add_argument("--positive-continuable-weight", type=float, default=0.35)
    parser.add_argument("--error-loss-weight", type=float, default=0.4)
    parser.add_argument("--state-loss-weight", type=float, default=0.4)
    parser.add_argument("--progress-loss-weight", type=float, default=0.4)
    parser.add_argument("--binary-loss-weight", type=float, default=0.5)
    parser.add_argument("--value-loss-weight", type=float, default=0.4)
    parser.add_argument("--rank-loss-weight", type=float, default=0.25)
    parser.add_argument("--rank-loss-temperature", type=float, default=0.05)
    parser.add_argument("--candidate-family-filter", default="")
    parser.add_argument("--min-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=500000)
    parser.add_argument("--min-wall-seconds", type=int, default=3660)
    parser.add_argument("--max-wall-seconds", type=int, default=3900)
    parser.add_argument("--eval-every-steps", type=int, default=200)
    parser.add_argument("--save-every-steps", type=int, default=1000)
    parser.add_argument("--formal-min-gpus", type=int, default=1)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=20260623)
    return parser.parse_args()


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


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def fixed_width(values: Any, width: int, default: float = 0.0) -> np.ndarray:
    arr = np.asarray(values if values is not None else [], dtype=np.float32).reshape(-1)
    if arr.size < width:
        arr = np.pad(arr, (0, width - arr.size), mode="constant", constant_values=float(default))
    return arr[:width].astype(np.float32)


def truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def continuous_value(error: float, progress: np.ndarray, binary: np.ndarray) -> float:
    return float(
        -0.75 * float(error)
        + 0.35 * float(progress[0])
        + 0.55 * float(progress[1])
        + 1.0 * float(binary[1])
        + 0.7 * float(binary[0])
        + 0.35 * float(binary[2])
        + 0.2 * float(binary[3])
        + 0.45 * float(binary[4])
    )


def make_binary(outcome: dict[str, Any], continuable_proxy: bool) -> np.ndarray:
    handoff_success = truthy(outcome.get("dp_rollout_success")) or truthy(outcome.get("final_success"))
    return np.asarray(
        [
            float(truthy(outcome.get("final_success"))),
            float(handoff_success),
            float(truthy(outcome.get("final_inserted_live_pose"))),
            float(truthy(outcome.get("final_grasped"))),
            float(bool(continuable_proxy)),
        ],
        dtype=np.float32,
    )


def positive_weight(binary: np.ndarray, *, continuable_weight: float) -> float:
    if bool(binary[0] > 0.5 or binary[1] > 0.5):
        return 1.0
    if bool(binary[4] > 0.5):
        return float(continuable_weight)
    return 0.0


def family_filter(text: str) -> set[str] | None:
    values = {item.strip() for item in str(text).split(",") if item.strip()}
    return values or None


def load_dataset(args: argparse.Namespace) -> dict[str, Any]:
    allowed = family_filter(str(args.candidate_family_filter))
    base_rows = load_base_rows(Path(args.base_rows_jsonl).resolve(), int(args.max_base_rows))
    outcome_rows = load_outcome_rows(
        [str(Path(path).resolve()) for path in args.outcome_jsonl],
        dedupe=True,
        allowed_families=allowed,
    )
    base_cache: dict[str, tuple[np.ndarray, np.ndarray, int, dict[str, Any]]] = {}
    features: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    priors: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    errors: list[float] = []
    states: list[np.ndarray] = []
    progresses: list[np.ndarray] = []
    binaries: list[np.ndarray] = []
    values: list[float] = []
    weights: list[float] = []
    metas: list[dict[str, Any]] = []
    missing_base = 0
    skipped_horizon = 0
    for outcome in outcome_rows:
        uuid = str(outcome.get("uuid") or "")
        base_row = base_rows.get(uuid)
        if base_row is None:
            missing_base += 1
            continue
        if uuid not in base_cache:
            base_cache[uuid] = build_base_feature(base_row)
        base_feature, prior_flat, horizon, base_meta = base_cache[uuid]
        if int(outcome.get("horizon") or horizon) != int(horizon):
            skipped_horizon += 1
            continue
        candidate = load_candidate_actions(outcome, horizon)
        residual = candidate - prior_flat
        progress, continuable = outcome_progress_targets(outcome, base_row)
        binary = make_binary(outcome, continuable)
        weight = positive_weight(binary, continuable_weight=float(args.positive_continuable_weight))
        final_state = fixed_width(outcome.get("final_peg_head_at_hole"), 3, 0.0)
        error = float(outcome.get("final_abs_task_error_weighted"))
        value = continuous_value(error, progress, binary)
        features.append(base_feature.astype(np.float32))
        residuals.append(residual.astype(np.float32))
        priors.append(prior_flat.astype(np.float32))
        actions.append(candidate.astype(np.float32))
        errors.append(error)
        states.append(final_state)
        progresses.append(progress.astype(np.float32))
        binaries.append(binary.astype(np.float32))
        values.append(value)
        weights.append(weight)
        candidate_name = str(outcome.get("candidate_name") or "")
        candidate_source = str(outcome.get("candidate_source") or "")
        metas.append(
            {
                **base_meta,
                "candidate_name": candidate_name,
                "candidate_source": candidate_source,
                "candidate_family": candidate_family(candidate_name, candidate_source),
                "candidate_actions_npz": outcome.get("candidate_actions_npz"),
                "outcome_jsonl": outcome.get("_outcome_jsonl"),
                "positive_weight": float(weight),
                "true_final_abs_task_error_weighted": float(error),
                "true_final_peg_head_at_hole": final_state.astype(float).tolist(),
                "true_progress": progress.astype(float).tolist(),
                "true_binary": binary.astype(float).tolist(),
                "true_value": float(value),
            }
        )
    if not features:
        raise RuntimeError("no joined live outcome action rows")
    widths = {
        "feature": {item.shape[0] for item in features},
        "residual": {item.shape[0] for item in residuals},
        "prior": {item.shape[0] for item in priors},
        "action": {item.shape[0] for item in actions},
    }
    if any(len(items) != 1 for items in widths.values()) or widths["residual"] != widths["prior"]:
        raise RuntimeError(f"nonuniform widths: {widths}")
    return {
        "x": np.stack(features).astype(np.float32),
        "residual": np.stack(residuals).astype(np.float32),
        "prior": np.stack(priors).astype(np.float32),
        "action": np.stack(actions).astype(np.float32),
        "error": np.asarray(errors, dtype=np.float32).reshape(-1, 1),
        "state": np.stack(states).astype(np.float32),
        "progress": np.stack(progresses).astype(np.float32),
        "binary": np.stack(binaries).astype(np.float32),
        "value": np.asarray(values, dtype=np.float32).reshape(-1, 1),
        "positive_weight": np.asarray(weights, dtype=np.float32).reshape(-1, 1),
        "meta": metas,
        "num_base_rows": int(len(base_cache)),
        "num_outcome_rows": int(len(outcome_rows)),
        "missing_base_rows": int(missing_base),
        "skipped_horizon_rows": int(skipped_horizon),
    }


def diffusion_schedule(steps: int, beta_start: float, beta_end: float, device: Any) -> tuple[Any, Any, Any]:
    import torch

    step_count = max(2, int(steps))
    betas = torch.linspace(float(beta_start), float(beta_end), step_count, device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


class LiveOutcomeActionDiffusionNet:
    def __new__(cls, feature_dim: int, residual_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> Any:
        import torch

        class _Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layers: list[torch.nn.Module] = []
                in_dim = int(feature_dim)
                for _ in range(int(num_layers)):
                    layers.append(torch.nn.Linear(in_dim, int(hidden_dim)))
                    layers.append(torch.nn.LayerNorm(int(hidden_dim)))
                    layers.append(torch.nn.GELU())
                    if float(dropout) > 0:
                        layers.append(torch.nn.Dropout(float(dropout)))
                    in_dim = int(hidden_dim)
                self.encoder = torch.nn.Sequential(*layers)
                self.denoiser = torch.nn.Sequential(
                    torch.nn.Linear(int(hidden_dim) + int(residual_dim) + 1, int(hidden_dim)),
                    torch.nn.LayerNorm(int(hidden_dim)),
                    torch.nn.GELU(),
                    torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
                    torch.nn.GELU(),
                    torch.nn.Linear(int(hidden_dim), int(residual_dim)),
                )
                self.scorer = torch.nn.Sequential(
                    torch.nn.Linear(int(hidden_dim) + int(residual_dim), int(hidden_dim)),
                    torch.nn.LayerNorm(int(hidden_dim)),
                    torch.nn.GELU(),
                    torch.nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
                    torch.nn.GELU(),
                )
                score_dim = int(hidden_dim) // 2
                self.error_head = torch.nn.Linear(score_dim, 1)
                self.state_head = torch.nn.Linear(score_dim, 3)
                self.progress_head = torch.nn.Linear(score_dim, 2)
                self.binary_head = torch.nn.Linear(score_dim, 5)
                self.value_head = torch.nn.Linear(score_dim, 1)

            def encode(self, x: Any) -> Any:
                return self.encoder(x)

            def denoise(self, z: Any, noisy_resid: Any, t_norm: Any) -> Any:
                if t_norm.ndim == 1:
                    t_norm = t_norm[:, None]
                return self.denoiser(torch.cat([z, noisy_resid, t_norm], dim=-1))

            def score(self, z: Any, resid: Any) -> tuple[Any, Any, Any, Any, Any]:
                h = self.scorer(torch.cat([z, resid], dim=-1))
                return (
                    self.error_head(h),
                    self.state_head(h),
                    self.progress_head(h),
                    self.binary_head(h),
                    self.value_head(h),
                )

        return _Net()


def split_groups(meta: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train, val = split_by_uuid(meta, float(val_fraction), int(seed))
    if val.size == 0:
        indices = np.arange(len(meta), dtype=np.int64)
        return indices, indices[:0]
    return train, val


def normalize(train_idx: np.ndarray, array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = array[train_idx].mean(axis=0, keepdims=True)
    std = array[train_idx].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((array - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def true_score_np(error: np.ndarray, progress: np.ndarray, binary: np.ndarray) -> np.ndarray:
    return (
        -0.75 * error.reshape(-1)
        + 0.35 * progress[:, 0]
        + 0.55 * progress[:, 1]
        + 1.0 * binary[:, 1]
        + 0.7 * binary[:, 0]
        + 0.35 * binary[:, 2]
        + 0.2 * binary[:, 3]
        + 0.45 * binary[:, 4]
    ).astype(np.float32)


def grouped_rank_loss(pred_value: Any, true_value: Any, global_indices: Any, meta: list[dict[str, Any]], temperature: float) -> tuple[Any, int]:
    import torch
    import torch.nn.functional as F

    groups: dict[str, list[int]] = defaultdict(list)
    global_list = [int(i) for i in global_indices.detach().cpu().tolist()]
    for local_i, global_i in enumerate(global_list):
        groups[str(meta[global_i].get("uuid") or "")].append(local_i)
    losses: list[Any] = []
    for local_indices in groups.values():
        if len(local_indices) < 2:
            continue
        idx = torch.as_tensor(local_indices, dtype=torch.long, device=pred_value.device)
        target = torch.softmax(true_value[idx].reshape(-1) / max(float(temperature), 1e-6), dim=0)
        log_prob = torch.log_softmax(pred_value[idx].reshape(-1), dim=0)
        losses.append(-(target * log_prob).sum())
    if not losses:
        return pred_value.sum() * 0.0, 0
    return torch.stack(losses).mean(), len(losses)


def evaluate(
    *,
    model: Any,
    x_norm: np.ndarray,
    residual_norm: np.ndarray,
    error: np.ndarray,
    state: np.ndarray,
    progress: np.ndarray,
    binary: np.ndarray,
    value: np.ndarray,
    positive_weight: np.ndarray,
    indices: np.ndarray,
    meta: list[dict[str, Any]],
    device: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    if indices.size == 0:
        return {}
    model.eval()
    with torch.no_grad():
        idx = np.asarray(indices, dtype=np.int64)
        x = torch.from_numpy(x_norm[idx]).to(device)
        resid = torch.from_numpy(residual_norm[idx]).to(device)
        err = torch.from_numpy(error[idx]).to(device)
        st = torch.from_numpy(state[idx]).to(device)
        prog = torch.from_numpy(progress[idx]).to(device)
        bin_t = torch.from_numpy(binary[idx]).to(device)
        val_t = torch.from_numpy(value[idx]).to(device)
        z = model.encode(x)
        pred_error, pred_state, pred_progress, pred_logits, pred_value = model.score(z, resid)
        pred_binary = (torch.sigmoid(pred_logits) >= 0.5).float()
        mse = torch.mean((pred_value - val_t) ** 2)
        true_score = true_score_np(error[idx], progress[idx], binary[idx])
        groups: dict[str, list[int]] = defaultdict(list)
        for local_i, global_i in enumerate(idx.tolist()):
            groups[str(meta[int(global_i)].get("uuid") or "")].append(local_i)
        dp_success = 0
        selected_success = 0
        oracle_success = 0
        dp_cont = 0
        selected_cont = 0
        oracle_cont = 0
        selected_non_dp = 0
        group_count = 0
        for local_indices in groups.values():
            if not local_indices:
                continue
            group_count += 1
            local = np.asarray(local_indices, dtype=np.int64)
            names = [str(meta[int(idx[i])].get("candidate_name") or "") for i in local]
            dp_local = next((j for j, name in enumerate(names) if name == "dp_prior"), 0)
            pred_local = pred_value.detach().cpu().numpy().reshape(-1)[local]
            true_local = true_score[local]
            best_pred_j = int(np.argmax(pred_local))
            best_true_j = int(np.argmax(true_local))
            dp_i = int(local[dp_local])
            selected_i = int(local[best_pred_j])
            oracle_i = int(local[best_true_j])
            dp_success += int(binary[idx[dp_i], 1] > 0.5 or binary[idx[dp_i], 0] > 0.5)
            selected_success += int(binary[idx[selected_i], 1] > 0.5 or binary[idx[selected_i], 0] > 0.5)
            oracle_success += int(binary[idx[oracle_i], 1] > 0.5 or binary[idx[oracle_i], 0] > 0.5)
            dp_cont += int(binary[idx[dp_i], 4] > 0.5)
            selected_cont += int(binary[idx[selected_i], 4] > 0.5)
            oracle_cont += int(binary[idx[oracle_i], 4] > 0.5)
            selected_non_dp += int(str(meta[int(idx[selected_i])].get("candidate_name") or "") != "dp_prior")
        out = {
            "num_rows": int(indices.size),
            "num_groups": int(group_count),
            "error_mse": float(torch.mean((pred_error - err) ** 2).item()),
            "state_mse": float(torch.mean((pred_state - st) ** 2).item()),
            "progress_mse": float(torch.mean((pred_progress - prog) ** 2).item()),
            "binary_bce": float(F.binary_cross_entropy_with_logits(pred_logits, bin_t).item()),
            "value_mse": float(mse.item()),
            "positive_rows": int((positive_weight[idx] > 0.0).sum()),
            "binary_success_acc": float((pred_binary[:, 1] == bin_t[:, 1]).float().mean().item()),
            "binary_continuable_acc": float((pred_binary[:, 4] == bin_t[:, 4]).float().mean().item()),
            "dp_handoff_success_groups": int(dp_success),
            "selected_handoff_success_groups": int(selected_success),
            "oracle_handoff_success_groups": int(oracle_success),
            "dp_continuable_groups": int(dp_cont),
            "selected_continuable_groups": int(selected_cont),
            "oracle_continuable_groups": int(oracle_cont),
            "selected_non_dp_groups": int(selected_non_dp),
        }
        if group_count > 0:
            out.update(
                {
                    "dp_handoff_success_fraction": float(dp_success / group_count),
                    "selected_handoff_success_fraction": float(selected_success / group_count),
                    "oracle_handoff_success_fraction": float(oracle_success / group_count),
                    "selected_minus_dp_handoff_success_fraction": float((selected_success - dp_success) / group_count),
                    "selected_non_dp_fraction": float(selected_non_dp / group_count),
                }
            )
    model.train()
    return out


def main() -> int:
    args = parse_args()
    require_compute_step()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("require_cuda=true but torch.cuda.is_available() is false")
    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if int(args.min_wall_seconds) > 0 and visible_gpus < int(args.formal_min_gpus):
        raise SystemExit(f"formal training requires visible_gpus >= {args.formal_min_gpus}, got {visible_gpus}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.set_float32_matmul_precision("high")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args)
    meta = dataset["meta"]
    train_idx, val_idx = split_groups(meta, float(args.val_fraction), int(args.seed))
    x_norm, x_mean, x_std = normalize(train_idx, dataset["x"])
    resid_norm, resid_mean, resid_std = normalize(train_idx, dataset["residual"])
    value_norm, value_mean, value_std = normalize(train_idx, dataset["value"])
    true_score = true_score_np(dataset["error"], dataset["progress"], dataset["binary"]).reshape(-1, 1)

    family_counts = Counter(str(item.get("candidate_family") or "") for item in meta)
    positive_rows = int((dataset["positive_weight"] > 0.0).sum())
    hard_positive_rows = int(((dataset["binary"][:, 0] > 0.5) | (dataset["binary"][:, 1] > 0.5)).sum())
    manifest = {
        "schema": "live_outcome_action_diffusion_training_manifest_v1",
        "base_rows_jsonl": str(Path(args.base_rows_jsonl).resolve()),
        "outcome_jsonl": [str(Path(path).resolve()) for path in args.outcome_jsonl],
        "output_root": str(output_root),
        "num_rows": int(dataset["x"].shape[0]),
        "num_base_rows": int(dataset["num_base_rows"]),
        "num_train_rows": int(train_idx.size),
        "num_val_rows": int(val_idx.size),
        "feature_dim": int(dataset["x"].shape[1]),
        "residual_dim": int(dataset["residual"].shape[1]),
        "visible_cuda_device_count": int(visible_gpus),
        "positive_rows": int(positive_rows),
        "hard_positive_rows": int(hard_positive_rows),
        "candidate_family_counts": dict(sorted(family_counts.items())),
        "min_wall_seconds": int(args.min_wall_seconds),
        "max_wall_seconds": int(args.max_wall_seconds),
        "boundary": (
            "Live-outcome contact-action diffusion training. Conditions are "
            "causal base features and contact context from saved live states; "
            "outcome labels supervise generation/scoring. This is not live "
            "method evidence until generated chunks pass saved-snapshot replay "
            "and full live visual/final-state gates."
        ),
    }
    write_json(output_root / "training_manifest.json", manifest)
    if hard_positive_rows <= 0:
        raise RuntimeError("no hard positive action rows for diffusion target")

    train_ds = TensorDataset(
        torch.from_numpy(x_norm[train_idx]),
        torch.from_numpy(resid_norm[train_idx]),
        torch.from_numpy(dataset["error"][train_idx]),
        torch.from_numpy(dataset["state"][train_idx]),
        torch.from_numpy(dataset["progress"][train_idx]),
        torch.from_numpy(dataset["binary"][train_idx]),
        torch.from_numpy(value_norm[train_idx]),
        torch.from_numpy(dataset["positive_weight"][train_idx]),
        torch.from_numpy(true_score[train_idx]),
        torch.from_numpy(train_idx.astype(np.int64)),
    )
    loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    model = LiveOutcomeActionDiffusionNet(
        feature_dim=int(dataset["x"].shape[1]),
        residual_dim=int(dataset["residual"].shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    betas, _alphas, alpha_bars = diffusion_schedule(
        int(args.diffusion_steps), float(args.diffusion_beta_start), float(args.diffusion_beta_end), device
    )
    del betas

    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_step = 0
    start = time.time()
    step = 0
    last_save = 0
    stop_reason = "running"
    value_mean_t = torch.from_numpy(value_mean).to(device)
    value_std_t = torch.from_numpy(value_std).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + 173)

    def eval_now() -> dict[str, Any]:
        return evaluate(
            model=model,
            x_norm=x_norm,
            residual_norm=resid_norm,
            error=dataset["error"],
            state=dataset["state"],
            progress=dataset["progress"],
            binary=dataset["binary"],
            value=value_norm,
            positive_weight=dataset["positive_weight"],
            indices=val_idx,
            meta=meta,
            device=device,
            args=args,
        )

    while stop_reason == "running":
        for xb, rb, errb, stateb, progb, binb, valb, posw, scoreb, global_idx in loader:
            step += 1
            xb = xb.to(device, non_blocking=True)
            rb = rb.to(device, non_blocking=True)
            errb = errb.to(device, non_blocking=True)
            stateb = stateb.to(device, non_blocking=True)
            progb = progb.to(device, non_blocking=True)
            binb = binb.to(device, non_blocking=True)
            valb = valb.to(device, non_blocking=True)
            posw = posw.to(device, non_blocking=True)
            scoreb = scoreb.to(device, non_blocking=True)
            global_idx = global_idx.to(device, non_blocking=True)

            z = model.encode(xb)
            pred_error, pred_state, pred_progress, pred_logits, pred_value = model.score(z, rb)
            scorer_loss = (
                float(args.error_loss_weight) * F.mse_loss(pred_error, errb)
                + float(args.state_loss_weight) * F.mse_loss(pred_state, stateb)
                + float(args.progress_loss_weight) * F.mse_loss(pred_progress, progb)
                + float(args.binary_loss_weight) * F.binary_cross_entropy_with_logits(pred_logits, binb)
                + float(args.value_loss_weight) * F.mse_loss(pred_value, valb)
            )
            rank_loss, rank_groups = grouped_rank_loss(
                pred_value=pred_value,
                true_value=scoreb,
                global_indices=global_idx,
                meta=meta,
                temperature=float(args.rank_loss_temperature),
            )
            scorer_loss = scorer_loss + float(args.rank_loss_weight) * rank_loss

            positive_mask = posw.reshape(-1) > 0.0
            if bool(torch.any(positive_mask)):
                rb_pos = rb[positive_mask]
                z_pos = z[positive_mask]
                weights = posw[positive_mask].reshape(-1)
                t = torch.randint(0, int(alpha_bars.shape[0]), (rb_pos.shape[0],), generator=generator, device=device)
                alpha_bar = alpha_bars[t].reshape(-1, 1)
                noise = torch.randn(rb_pos.shape, generator=generator, device=device)
                noisy = torch.sqrt(alpha_bar) * rb_pos + torch.sqrt(1.0 - alpha_bar) * noise
                t_norm = t.float().reshape(-1, 1) / max(1, int(alpha_bars.shape[0]) - 1)
                pred_noise = model.denoise(z_pos, noisy, t_norm)
                per_row = torch.mean((pred_noise - noise) ** 2, dim=1)
                diffusion_loss = torch.mean(per_row * weights)
            else:
                diffusion_loss = scorer_loss * 0.0
            value_raw = pred_value * value_std_t + value_mean_t
            value_raw_target = valb * value_std_t + value_mean_t
            raw_value_mse = F.mse_loss(value_raw, value_raw_target)
            loss = scorer_loss + float(args.diffusion_loss_weight) * diffusion_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step == 1 or step % int(args.eval_every_steps) == 0:
                metrics = {
                    "step": int(step),
                    "elapsed_seconds": float(time.time() - start),
                    "train_loss": float(loss.detach().cpu().item()),
                    "train_scorer_loss": float(scorer_loss.detach().cpu().item()),
                    "train_diffusion_loss": float(diffusion_loss.detach().cpu().item()),
                    "train_rank_loss": float(rank_loss.detach().cpu().item()),
                    "train_rank_groups": int(rank_groups),
                    "train_value_mse_raw": float(raw_value_mse.detach().cpu().item()),
                }
                metrics.update({f"eval_{k}": v for k, v in eval_now().items()})
                history.append(metrics)
                write_json(output_root / "training_history.json", history)
                candidate_metric = float(metrics.get("eval_value_mse", float("inf")))
                if candidate_metric < best_metric:
                    best_metric = candidate_metric
                    best_step = int(step)
                    atomic_torch_save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": opt.state_dict(),
                            "step": int(step),
                            "feature_dim": int(dataset["x"].shape[1]),
                            "residual_dim": int(dataset["residual"].shape[1]),
                            "x_mean": x_mean,
                            "x_std": x_std,
                            "resid_mean": resid_mean,
                            "resid_std": resid_std,
                            "value_mean": value_mean,
                            "value_std": value_std,
                            "args": vars(args),
                            "manifest": manifest,
                            "best_metrics": metrics,
                        },
                        output_root / "checkpoint_best_eval.pt",
                    )
                print(json.dumps(metrics, sort_keys=True), flush=True)
            if step == 1 or step - last_save >= int(args.save_every_steps):
                atomic_torch_save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "step": int(step),
                        "feature_dim": int(dataset["x"].shape[1]),
                        "residual_dim": int(dataset["residual"].shape[1]),
                        "x_mean": x_mean,
                        "x_std": x_std,
                        "resid_mean": resid_mean,
                        "resid_std": resid_std,
                        "value_mean": value_mean,
                        "value_std": value_std,
                        "args": vars(args),
                        "manifest": manifest,
                        "latest_metrics": history[-1] if history else {},
                    },
                    output_root / "checkpoint_latest.pt",
                )
                last_save = step
            elapsed = time.time() - start
            if elapsed >= int(args.max_wall_seconds):
                stop_reason = "max_wall_seconds"
            elif elapsed >= int(args.min_wall_seconds) and step >= int(args.min_steps):
                stop_reason = "min_wall_and_min_steps"
            elif int(args.min_wall_seconds) <= 0 and step >= int(args.max_steps):
                stop_reason = "max_steps"
            if stop_reason != "running":
                break
            if step >= int(args.max_steps) and int(args.min_wall_seconds) <= 0:
                stop_reason = "max_steps"
                break

    final_metrics = history[-1] if history else {}
    final_checkpoint = output_root / "checkpoint_final.pt"
    atomic_torch_save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "step": int(step),
            "feature_dim": int(dataset["x"].shape[1]),
            "residual_dim": int(dataset["residual"].shape[1]),
            "x_mean": x_mean,
            "x_std": x_std,
            "resid_mean": resid_mean,
            "resid_std": resid_std,
            "value_mean": value_mean,
            "value_std": value_std,
            "args": vars(args),
            "manifest": manifest,
            "latest_metrics": final_metrics,
        },
        final_checkpoint,
    )
    elapsed = float(time.time() - start)
    formal_floor_met = bool(
        visible_gpus >= int(args.formal_min_gpus)
        and int(args.min_wall_seconds) >= 3600
        and elapsed >= int(args.min_wall_seconds)
    )
    selected_improvement = float(final_metrics.get("eval_selected_minus_dp_handoff_success_fraction", -999.0))
    selected_non_dp = float(final_metrics.get("eval_selected_non_dp_fraction", 0.0))
    ready_for_replay_gate = bool(
        final_metrics
        and formal_floor_met
        and selected_improvement > 0.0
        and selected_non_dp > 0.0
    )
    summary = {
        "schema": "live_outcome_action_diffusion_training_summary_v1",
        "output_root": str(output_root),
        "num_rows": int(dataset["x"].shape[0]),
        "num_base_rows": int(dataset["num_base_rows"]),
        "num_train_rows": int(train_idx.size),
        "num_val_rows": int(val_idx.size),
        "visible_cuda_device_count": int(visible_gpus),
        "positive_rows": int(positive_rows),
        "hard_positive_rows": int(hard_positive_rows),
        "steps": int(step),
        "elapsed_seconds": elapsed,
        "stop_reason": stop_reason,
        "final_metrics": final_metrics,
        "best_metric": "eval_value_mse",
        "best_step": int(best_step),
        "best_checkpoint": str(output_root / "checkpoint_best_eval.pt"),
        "formal_one_gpu_hour_floor_met": formal_floor_met,
        "ready_for_saved_snapshot_replay_gate": ready_for_replay_gate,
        "boundary": (
            "Training evidence for a live-outcome-conditioned action diffusion "
            "generator only. It does not prove dynamic task completion until "
            "generated chunks are replayed on saved live failure states and then "
            "validated in full live panels."
        ),
    }
    write_json(output_root / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
