#!/usr/bin/env python3
"""Train a source-suffix contact action generator.

This is the first contact-action reset training target. It learns executable
insertion suffix chunks from the accepted 733 H5 source set's mined successful
suffix bank. The model is not closed-loop method evidence by itself; it is a
contact-action generator baseline that must later be evaluated on saved live
failure states and then in real full-episode panels.
"""

from __future__ import annotations

import argparse
from collections import Counter
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix-bank-npz", required=True)
    parser.add_argument("--suffix-bank-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
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
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    torch.save(payload, tmp)
    os.replace(tmp, path)


def decode_str_array(value: np.ndarray) -> list[str]:
    return [str(item) for item in np.asarray(value).reshape(-1).tolist()]


def one_hot(values: list[str], vocab: list[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(vocab)}
    out = np.zeros((len(values), len(vocab)), dtype=np.float32)
    for row_i, value in enumerate(values):
        out[row_i, index[value]] = 1.0
    return out


def split_by_source(source_uuid: list[str], val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique = np.asarray(sorted(set(source_uuid)), dtype=object)
    rng.shuffle(unique)
    if unique.size <= 1 or val_fraction <= 0:
        train_sources = set(unique.tolist())
        val_sources: set[str] = set()
    else:
        n_val = max(1, int(round(unique.size * float(val_fraction))))
        n_val = min(n_val, unique.size - 1)
        val_sources = set(str(x) for x in unique[:n_val].tolist())
        train_sources = set(str(x) for x in unique[n_val:].tolist())
    train_idx = np.asarray([i for i, src in enumerate(source_uuid) if src in train_sources], dtype=np.int64)
    val_idx = np.asarray([i for i, src in enumerate(source_uuid) if src in val_sources], dtype=np.int64)
    return train_idx, val_idx


def build_features(bank: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    actions = np.asarray(bank["actions"], dtype=np.float32)
    if actions.ndim != 3 or actions.shape[2] != 7:
        raise RuntimeError(f"expected actions shape (N,H,7), got {actions.shape}")
    scenario = decode_str_array(bank["scenario"])
    source_uuid = decode_str_array(bank["source_uuid"])
    start_rel = np.asarray(bank["start_peg_head_at_hole"], dtype=np.float32)
    end_rel = np.asarray(bank["end_peg_head_at_hole"], dtype=np.float32)
    offset = np.asarray(bank["offset_before_insert"], dtype=np.float32).reshape(-1, 1)
    start_frame = np.asarray(bank["start_frame"], dtype=np.float32).reshape(-1, 1)
    first_insert = np.asarray(bank["first_insert_frame"], dtype=np.float32).reshape(-1, 1)
    valid_steps = np.asarray(bank["valid_steps"], dtype=np.float32).reshape(-1, 1)
    if start_rel.shape != (actions.shape[0], 3):
        raise RuntimeError(f"start_rel shape mismatch: {start_rel.shape} vs {actions.shape}")
    scenario_vocab = sorted(set(scenario))
    offset_values = sorted({int(x) for x in offset.reshape(-1).tolist()})
    offset_names = [str(x) for x in offset_values]
    offset_onehot = one_hot([str(int(x)) for x in offset.reshape(-1).tolist()], offset_names)
    scenario_onehot = one_hot(scenario, scenario_vocab)
    abs_yz = (np.abs(start_rel[:, 1:2]) + np.abs(start_rel[:, 2:3])).astype(np.float32)
    feature = np.concatenate(
        [
            start_rel.astype(np.float32),
            abs_yz,
            offset / 96.0,
            start_frame / 300.0,
            first_insert / 300.0,
            valid_steps / actions.shape[1],
            offset_onehot,
            scenario_onehot,
        ],
        axis=1,
    ).astype(np.float32)
    target = actions.reshape(actions.shape[0], -1).astype(np.float32)
    metadata = {
        "num_rows": int(actions.shape[0]),
        "horizon": int(actions.shape[1]),
        "action_dim": int(actions.shape[2]),
        "scenario_vocab": scenario_vocab,
        "offset_values": offset_values,
        "source_uuid": source_uuid,
        "scenario_counts": dict(sorted(Counter(scenario).items())),
        "offset_counts": {str(k): int(v) for k, v in sorted(Counter(int(x) for x in offset.reshape(-1)).items())},
        "start_rel_stats": {
            "x_mean": float(start_rel[:, 0].mean()),
            "x_min": float(start_rel[:, 0].min()),
            "x_max": float(start_rel[:, 0].max()),
            "abs_yz_mean": float(abs_yz.mean()),
            "abs_yz_max": float(abs_yz.max()),
        },
        "end_rel_stats": {
            "x_mean": float(end_rel[:, 0].mean()),
            "x_min": float(end_rel[:, 0].min()),
            "x_max": float(end_rel[:, 0].max()),
        },
    }
    return feature, target, metadata


class SuffixActionNet:
    def __new__(cls, feature_dim: int, target_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> Any:
        import torch

        class _Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layers: list[torch.nn.Module] = []
                in_dim = int(feature_dim)
                for _ in range(int(num_layers)):
                    layers.append(torch.nn.Linear(in_dim, int(hidden_dim)))
                    layers.append(torch.nn.LayerNorm(int(hidden_dim)))
                    layers.append(torch.nn.SiLU())
                    if dropout > 0:
                        layers.append(torch.nn.Dropout(float(dropout)))
                    in_dim = int(hidden_dim)
                self.trunk = torch.nn.Sequential(*layers)
                self.head = torch.nn.Linear(int(hidden_dim), int(target_dim))

            def forward(self, x: Any) -> Any:
                return self.head(self.trunk(x))

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
    if len(rows) != int(metadata["num_rows"]):
        raise RuntimeError(f"jsonl rows {len(rows)} != bank rows {metadata['num_rows']}")
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
    model = SuffixActionNet(
        feature_dim=int(x.shape[1]),
        target_dim=int(y.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    x_eval = torch.from_numpy(x[val_idx]).to(device)
    y_eval = torch.from_numpy(y[val_idx]).to(device)
    x_train_eval = torch.from_numpy(x[train_idx]).to(device)
    y_train_eval = torch.from_numpy(y[train_idx]).to(device)
    y_mean_t = torch.from_numpy(y_mean.astype(np.float32)).to(device)
    y_std_t = torch.from_numpy(y_std.astype(np.float32)).to(device)
    y_raw_eval = torch.from_numpy(y_raw[val_idx].astype(np.float32)).to(device)
    y_raw_train = torch.from_numpy(y_raw[train_idx].astype(np.float32)).to(device)
    train_mean_baseline = torch.from_numpy(np.repeat(y_mean.astype(np.float32), train_idx.size, axis=0)).to(device)
    val_mean_baseline = torch.from_numpy(np.repeat(y_mean.astype(np.float32), val_idx.size, axis=0)).to(device)

    def evaluate(prefix: str, xx: Any, yy_norm: Any, yy_raw: Any, baseline_raw: Any) -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            pred_norm = model(xx)
            pred_raw = pred_norm * y_std_t + y_mean_t
            out = {
                f"{prefix}_norm_mse": float(torch.mean((pred_norm - yy_norm) ** 2).item()),
                f"{prefix}_action_mse": float(torch.mean((pred_raw - yy_raw) ** 2).item()),
                f"{prefix}_mean_action_baseline_mse": float(torch.mean((baseline_raw - yy_raw) ** 2).item()),
            }
        model.train()
        return out

    write_json(
        output_root / "training_manifest.json",
        {
            "schema": "contact_action_suffix_generator_training_manifest_v1",
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
                "First contact-action reset training target. It learns source "
                "insertion suffix actions from accepted 733 H5 data. This is an "
                "action generator baseline, not live method evidence until replay "
                "and full-episode visual/final-state evaluation pass."
            ),
        },
    )

    history: list[dict[str, Any]] = []
    best_eval_action_mse = float("inf")
    best_metrics: dict[str, Any] = {}
    best_step = 0
    start = time.time()
    step = 0
    last_save = 0
    stop_reason = "running"
    while stop_reason == "running":
        for xb, yb in loader:
            step += 1
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 1 or step % int(args.eval_every_steps) == 0:
                metrics = {
                    "step": int(step),
                    "elapsed_seconds": float(time.time() - start),
                    "train_batch_norm_mse": float(loss.detach().cpu().item()),
                }
                metrics.update(evaluate("train", x_train_eval, y_train_eval, y_raw_train, train_mean_baseline))
                metrics.update(evaluate("eval", x_eval, y_eval, y_raw_eval, val_mean_baseline))
                history.append(metrics)
                write_json(output_root / "training_history.json", history)
                current_eval_action_mse = float(metrics["eval_action_mse"])
                if current_eval_action_mse < best_eval_action_mse:
                    best_eval_action_mse = current_eval_action_mse
                    best_metrics = dict(metrics)
                    best_step = int(step)
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
                            "metadata": metadata,
                            "best_metric": "eval_action_mse",
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
                        "metadata": metadata,
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
            "metadata": metadata,
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
        and best_metrics["eval_action_mse"] < best_metrics["eval_mean_action_baseline_mse"]
    )
    summary = {
        "schema": "contact_action_suffix_generator_training_summary_v1",
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
        "best_metric": "eval_action_mse",
        "best_step": int(best_step),
        "best_metrics": best_metrics,
        "best_checkpoint": str(output_root / "checkpoint_best_eval.pt") if best_metrics else None,
        "formal_one_gpu_hour_floor_met": formal_floor_met,
        "ready_for_saved_snapshot_replay_gate": ready_for_replay_gate,
        "boundary": (
            "Training evidence for a source-suffix contact-action generator only. "
            "It does not prove dynamic task completion until generated chunks are "
            "tested on saved live failure states and then in real full-episode panels."
        ),
    }
    write_json(output_root / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
