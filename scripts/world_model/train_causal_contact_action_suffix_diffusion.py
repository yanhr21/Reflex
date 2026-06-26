#!/usr/bin/env python3
"""Train a causal contact-action suffix diffusion generator.

This is the repair after the deterministic source-suffix MLP and live-outcome
diffusion negatives. It uses direct inserted source suffixes from the accepted
733 H5 set, but removes non-causal scenario labels and future first-insert
features. The model conditions only on task-frame state that can be produced
from live observation/history plus a requested insertion-offset control token.

The output is still training evidence only. Generated chunks must pass saved
live-snapshot replay and full live visual/final-state gates before any method
claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix-bank-npz", required=True)
    parser.add_argument("--suffix-bank-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--diffusion-steps", type=int, default=32)
    parser.add_argument("--diffusion-beta-start", type=float, default=1e-4)
    parser.add_argument("--diffusion-beta-end", type=float, default=2e-2)
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


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
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def decode_str_array(value: np.ndarray) -> list[str]:
    return [str(item) for item in np.asarray(value).reshape(-1).tolist()]


def one_hot_int(values: np.ndarray, vocab: list[int]) -> np.ndarray:
    index = {int(v): i for i, v in enumerate(vocab)}
    out = np.zeros((values.shape[0], len(vocab)), dtype=np.float32)
    for row_i, value in enumerate(values.reshape(-1).tolist()):
        out[row_i, index[int(value)]] = 1.0
    return out


def split_by_source(source_uuid: list[str], val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    unique = np.asarray(sorted(set(source_uuid)), dtype=object)
    rng.shuffle(unique)
    if unique.size <= 1 or val_fraction <= 0:
        train_sources = set(str(x) for x in unique.tolist())
        val_sources: set[str] = set()
    else:
        n_val = max(1, int(round(unique.size * float(val_fraction))))
        n_val = min(n_val, unique.size - 1)
        val_sources = set(str(x) for x in unique[:n_val].tolist())
        train_sources = set(str(x) for x in unique[n_val:].tolist())
    train_idx = np.asarray([i for i, src in enumerate(source_uuid) if src in train_sources], dtype=np.int64)
    val_idx = np.asarray([i for i, src in enumerate(source_uuid) if src in val_sources], dtype=np.int64)
    return train_idx, val_idx


def stats_1d(arr: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(arr, dtype=np.float32).reshape(-1)
    return {
        "n": int(values.size),
        "min": float(values.min()) if values.size else 0.0,
        "max": float(values.max()) if values.size else 0.0,
        "mean": float(values.mean()) if values.size else 0.0,
        "median": float(np.median(values)) if values.size else 0.0,
        "p10": float(np.percentile(values, 10)) if values.size else 0.0,
        "p90": float(np.percentile(values, 90)) if values.size else 0.0,
    }


def build_features(bank: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    actions_all = np.asarray(bank["actions"], dtype=np.float32)
    if actions_all.ndim != 3 or actions_all.shape[2] != 7:
        raise RuntimeError(f"expected actions shape (N,H,7), got {actions_all.shape}")
    inserted = np.asarray(bank["inserted_within_chunk"], dtype=bool).reshape(-1)
    grasped = np.asarray(bank["grasped_at_start"], dtype=bool).reshape(-1)
    keep = np.flatnonzero(inserted & grasped)
    if keep.size <= 0:
        raise RuntimeError("no inserted+grasped source suffix rows")

    actions = actions_all[keep]
    scenario = [decode_str_array(bank["scenario"])[i] for i in keep.tolist()]
    source_uuid = [decode_str_array(bank["source_uuid"])[i] for i in keep.tolist()]
    start_rel = np.asarray(bank["start_peg_head_at_hole"], dtype=np.float32)[keep]
    end_rel = np.asarray(bank["end_peg_head_at_hole"], dtype=np.float32)[keep]
    offset = np.asarray(bank["offset_before_insert"], dtype=np.int32).reshape(-1)[keep]
    start_frame = np.asarray(bank["start_frame"], dtype=np.float32).reshape(-1, 1)[keep]
    valid_steps = np.asarray(bank["valid_steps"], dtype=np.float32).reshape(-1, 1)[keep]
    grasped_f = grasped[keep].astype(np.float32).reshape(-1, 1)

    offset_values = sorted({int(x) for x in offset.reshape(-1).tolist()})
    offset_onehot = one_hot_int(offset.reshape(-1), offset_values)
    abs_y = np.abs(start_rel[:, 1:2]).astype(np.float32)
    abs_z = np.abs(start_rel[:, 2:3]).astype(np.float32)
    yz_l2 = np.linalg.norm(start_rel[:, 1:3], axis=1, keepdims=True).astype(np.float32)
    abs_yz_sum = (abs_y + abs_z).astype(np.float32)
    remaining_episode = (300.0 - start_frame) / 300.0
    valid_steps_norm = valid_steps / float(actions.shape[1])
    feature = np.concatenate(
        [
            start_rel.astype(np.float32),
            abs_y,
            abs_z,
            abs_yz_sum,
            yz_l2,
            offset.reshape(-1, 1).astype(np.float32) / 96.0,
            start_frame / 300.0,
            remaining_episode.astype(np.float32),
            valid_steps_norm.astype(np.float32),
            grasped_f,
            offset_onehot,
        ],
        axis=1,
    ).astype(np.float32)
    target = actions.reshape(actions.shape[0], -1).astype(np.float32)
    metadata = {
        "num_rows": int(actions.shape[0]),
        "source_bank_rows": int(actions_all.shape[0]),
        "horizon": int(actions.shape[1]),
        "action_dim": int(actions.shape[2]),
        "feature_schema": [
            "start_rel_x",
            "start_rel_y",
            "start_rel_z",
            "abs_start_rel_y",
            "abs_start_rel_z",
            "abs_start_rel_yz_sum",
            "start_rel_yz_l2",
            "requested_offset_before_insert_norm96",
            "start_frame_norm300",
            "remaining_episode_norm300",
            "valid_steps_norm_horizon",
            "grasped_at_start",
        ]
        + [f"requested_offset_onehot_{value}" for value in offset_values],
        "causal_feature_boundary": (
            "No scenario label and no future first_insert_frame are used. "
            "requested_offset_before_insert is a sampler control token, not a "
            "privileged observed sample label."
        ),
        "offset_values": offset_values,
        "source_uuid": source_uuid,
        "scenario_counts_diagnostic_only": dict(sorted(Counter(scenario).items())),
        "source_uuid_count": int(len(set(source_uuid))),
        "offset_counts": {str(k): int(v) for k, v in sorted(Counter(int(x) for x in offset).items())},
        "start_rel_stats": {
            "x": stats_1d(start_rel[:, 0]),
            "y": stats_1d(start_rel[:, 1]),
            "z": stats_1d(start_rel[:, 2]),
            "abs_yz_sum": stats_1d(abs_yz_sum),
        },
        "end_rel_stats": {
            "x": stats_1d(end_rel[:, 0]),
            "y": stats_1d(end_rel[:, 1]),
            "z": stats_1d(end_rel[:, 2]),
            "abs_yz_sum": stats_1d(np.abs(end_rel[:, 1:2]) + np.abs(end_rel[:, 2:3])),
        },
    }
    return feature, target, metadata


def diffusion_schedule(steps: int, beta_start: float, beta_end: float, device: Any) -> Any:
    import torch

    step_count = max(2, int(steps))
    betas = torch.linspace(float(beta_start), float(beta_end), step_count, device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def timestep_embedding(t: Any, dim: int) -> Any:
    import torch

    half = int(dim) // 2
    if half <= 0:
        return t[:, None]
    freqs = torch.exp(
        torch.arange(half, device=t.device, dtype=torch.float32)
        * (-math.log(10000.0) / max(1, half - 1))
    )
    angles = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if emb.shape[1] < int(dim):
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=t.device, dtype=emb.dtype)], dim=1)
    return emb


class CausalSuffixDiffusionNet:
    def __new__(cls, feature_dim: int, target_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> Any:
        import torch

        time_dim = min(256, max(32, int(hidden_dim) // 8))

        class _Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                cond_layers: list[torch.nn.Module] = []
                in_dim = int(feature_dim)
                for _ in range(max(1, int(num_layers) // 2)):
                    cond_layers.append(torch.nn.Linear(in_dim, int(hidden_dim)))
                    cond_layers.append(torch.nn.LayerNorm(int(hidden_dim)))
                    cond_layers.append(torch.nn.SiLU())
                    if dropout > 0:
                        cond_layers.append(torch.nn.Dropout(float(dropout)))
                    in_dim = int(hidden_dim)
                self.cond = torch.nn.Sequential(*cond_layers)
                denoise_layers: list[torch.nn.Module] = []
                denoise_in = int(hidden_dim) + int(target_dim) + int(time_dim)
                for _ in range(int(num_layers)):
                    denoise_layers.append(torch.nn.Linear(denoise_in, int(hidden_dim)))
                    denoise_layers.append(torch.nn.LayerNorm(int(hidden_dim)))
                    denoise_layers.append(torch.nn.SiLU())
                    if dropout > 0:
                        denoise_layers.append(torch.nn.Dropout(float(dropout)))
                    denoise_in = int(hidden_dim)
                denoise_layers.append(torch.nn.Linear(int(hidden_dim), int(target_dim)))
                self.denoiser = torch.nn.Sequential(*denoise_layers)

            def forward(self, x: Any, noisy_action: Any, t_norm: Any) -> Any:
                cond = self.cond(x)
                t_emb = timestep_embedding(t_norm.reshape(-1), time_dim)
                return self.denoiser(torch.cat([cond, noisy_action, t_emb], dim=-1))

        return _Net()


def main() -> int:
    args = parse_args()
    require_compute_step()

    import torch
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

    suffix_bank_npz = Path(args.suffix_bank_npz).resolve()
    suffix_bank_jsonl = Path(args.suffix_bank_jsonl).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(suffix_bank_jsonl)
    with np.load(suffix_bank_npz, allow_pickle=False) as bank:
        x_raw, y_raw, metadata = build_features(bank)
    if len(rows) < int(metadata["num_rows"]):
        raise RuntimeError(f"jsonl rows {len(rows)} < kept bank rows {metadata['num_rows']}")

    train_idx, val_idx = split_by_source(metadata["source_uuid"], float(args.val_fraction), int(args.seed))
    if train_idx.size <= 0 or val_idx.size <= 0:
        raise RuntimeError(f"bad split train={train_idx.size} val={val_idx.size}")

    x_mean = x_raw[train_idx].mean(axis=0, keepdims=True)
    x_std = x_raw[train_idx].std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    y_mean = y_raw[train_idx].mean(axis=0, keepdims=True)
    y_std = y_raw[train_idx].std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)
    x = ((x_raw - x_mean) / x_std).astype(np.float32)
    y = ((y_raw - y_mean) / y_std).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx]))
    loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )
    model = CausalSuffixDiffusionNet(
        feature_dim=int(x.shape[1]),
        target_dim=int(y.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    alpha_bars = diffusion_schedule(
        int(args.diffusion_steps), float(args.diffusion_beta_start), float(args.diffusion_beta_end), device
    )
    x_val = torch.from_numpy(x[val_idx]).to(device)
    y_val = torch.from_numpy(y[val_idx]).to(device)
    y_raw_val = torch.from_numpy(y_raw[val_idx].astype(np.float32)).to(device)
    y_mean_t = torch.from_numpy(y_mean.astype(np.float32)).to(device)
    y_std_t = torch.from_numpy(y_std.astype(np.float32)).to(device)
    val_mean_baseline = torch.from_numpy(np.repeat(y_mean.astype(np.float32), val_idx.size, axis=0)).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + 405)

    manifest = {
        "schema": "causal_contact_action_suffix_diffusion_training_manifest_v1",
        "suffix_bank_npz": str(suffix_bank_npz),
        "suffix_bank_jsonl": str(suffix_bank_jsonl),
        "output_root": str(output_root),
        "num_rows": int(x_raw.shape[0]),
        "num_train_rows": int(train_idx.size),
        "num_val_rows": int(val_idx.size),
        "feature_dim": int(x.shape[1]),
        "target_dim": int(y.shape[1]),
        "action_horizon": int(metadata["horizon"]),
        "device": str(device),
        "visible_cuda_device_count": int(visible_gpus),
        "min_wall_seconds": int(args.min_wall_seconds),
        "max_wall_seconds": int(args.max_wall_seconds),
        "metadata": metadata,
        "boundary": (
            "Causal direct-insertion source-suffix diffusion training. It uses "
            "inserted+grasped source suffixes from the 733 accepted H5 set and "
            "removes scenario and future first-insert conditions. It is not live "
            "method evidence until generated chunks pass saved-snapshot replay "
            "and full 301/300 live visual/final-state gates."
        ),
    }
    write_json(output_root / "training_manifest.json", manifest)

    def eval_now() -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            count = int(y_val.shape[0])
            t = torch.full((count,), max(0, int(alpha_bars.shape[0]) // 2), device=device, dtype=torch.long)
            alpha_bar = alpha_bars[t].reshape(-1, 1)
            noise = torch.randn(y_val.shape, generator=generator, device=device)
            noisy = torch.sqrt(alpha_bar) * y_val + torch.sqrt(1.0 - alpha_bar) * noise
            t_norm = t.float() / max(1, int(alpha_bars.shape[0]) - 1)
            pred_noise = model(x_val, noisy, t_norm)
            denoise_mse = torch.mean((pred_noise - noise) ** 2)
            x0_pred = (noisy - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            pred_raw = x0_pred * y_std_t + y_mean_t
            x0_action_mse = torch.mean((pred_raw - y_raw_val) ** 2)
            mean_action_mse = torch.mean((val_mean_baseline - y_raw_val) ** 2)
            zero_noise_baseline = torch.mean(noise**2)
        model.train()
        return {
            "eval_denoise_mse": float(denoise_mse.item()),
            "eval_zero_noise_baseline_mse": float(zero_noise_baseline.item()),
            "eval_x0_action_mse_mid_t": float(x0_action_mse.item()),
            "eval_mean_action_baseline_mse": float(mean_action_mse.item()),
        }

    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_step = 0
    best_metrics: dict[str, Any] = {}
    start = time.time()
    step = 0
    last_save = 0
    stop_reason = "running"
    while stop_reason == "running":
        for xb, yb in loader:
            step += 1
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            t = torch.randint(0, int(alpha_bars.shape[0]), (yb.shape[0],), generator=generator, device=device)
            alpha_bar = alpha_bars[t].reshape(-1, 1)
            noise = torch.randn(yb.shape, generator=generator, device=device)
            noisy = torch.sqrt(alpha_bar) * yb + torch.sqrt(1.0 - alpha_bar) * noise
            t_norm = t.float() / max(1, int(alpha_bars.shape[0]) - 1)
            pred_noise = model(xb, noisy, t_norm)
            loss = torch.mean((pred_noise - noise) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step == 1 or step % int(args.eval_every_steps) == 0:
                metrics = {
                    "step": int(step),
                    "elapsed_seconds": float(time.time() - start),
                    "train_denoise_mse": float(loss.detach().cpu().item()),
                }
                metrics.update(eval_now())
                history.append(metrics)
                write_json(output_root / "training_history.json", history)
                candidate_metric = float(metrics["eval_x0_action_mse_mid_t"])
                if candidate_metric < best_metric:
                    best_metric = candidate_metric
                    best_step = int(step)
                    best_metrics = dict(metrics)
                    atomic_torch_save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": opt.state_dict(),
                            "step": int(step),
                            "feature_dim": int(x.shape[1]),
                            "target_dim": int(y.shape[1]),
                            "x_mean": x_mean.astype(np.float32),
                            "x_std": x_std.astype(np.float32),
                            "y_mean": y_mean.astype(np.float32),
                            "y_std": y_std.astype(np.float32),
                            "args": vars(args),
                            "manifest": manifest,
                            "best_metric": "eval_x0_action_mse_mid_t",
                            "best_metrics": best_metrics,
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
                        "feature_dim": int(x.shape[1]),
                        "target_dim": int(y.shape[1]),
                        "x_mean": x_mean.astype(np.float32),
                        "x_std": x_std.astype(np.float32),
                        "y_mean": y_mean.astype(np.float32),
                        "y_std": y_std.astype(np.float32),
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
            "feature_dim": int(x.shape[1]),
            "target_dim": int(y.shape[1]),
            "x_mean": x_mean.astype(np.float32),
            "x_std": x_std.astype(np.float32),
            "y_mean": y_mean.astype(np.float32),
            "y_std": y_std.astype(np.float32),
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
    ready_for_replay_gate = bool(
        best_metrics
        and formal_floor_met
        and float(best_metrics["eval_x0_action_mse_mid_t"]) < float(best_metrics["eval_mean_action_baseline_mse"])
        and float(best_metrics["eval_denoise_mse"]) < float(best_metrics["eval_zero_noise_baseline_mse"])
    )
    summary = {
        "schema": "causal_contact_action_suffix_diffusion_training_summary_v1",
        "suffix_bank_npz": str(suffix_bank_npz),
        "output_root": str(output_root),
        "num_rows": int(x_raw.shape[0]),
        "num_train_rows": int(train_idx.size),
        "num_val_rows": int(val_idx.size),
        "visible_cuda_device_count": int(visible_gpus),
        "steps": int(step),
        "elapsed_seconds": elapsed,
        "stop_reason": stop_reason,
        "final_metrics": final_metrics,
        "best_metric": "eval_x0_action_mse_mid_t",
        "best_step": int(best_step),
        "best_metrics": best_metrics,
        "best_checkpoint": str(output_root / "checkpoint_best_eval.pt") if best_metrics else None,
        "formal_one_gpu_hour_floor_met": formal_floor_met,
        "ready_for_saved_snapshot_replay_gate": ready_for_replay_gate,
        "boundary": (
            "Training evidence for a causal direct-insertion source-suffix "
            "diffusion generator only. It does not prove dynamic task completion "
            "until sampled chunks are replayed on saved live failure states and "
            "then validated in full live panels."
        ),
    }
    write_json(output_root / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
