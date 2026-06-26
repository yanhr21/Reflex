#!/usr/bin/env python3
"""Train the Cosmos task-path executor residual policy."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--executor-jsonl", required=True)
    parser.add_argument("--dp-prior-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--residual-l2-weight", type=float, default=1e-4)
    parser.add_argument("--min-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--min-wall-seconds", type=int, default=10800)
    parser.add_argument("--max-wall-seconds", type=int, default=14400)
    parser.add_argument("--eval-every-steps", type=int, default=200)
    parser.add_argument("--save-every-steps", type=int, default=1000)
    parser.add_argument("--formal-min-gpus", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260615)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_arrays(
    executor_rows: list[dict[str, Any]],
    prior_rows: list[dict[str, Any]],
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    prior_by_uuid = {str(row["uuid"]): row for row in prior_rows}
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    priors: list[np.ndarray] = []
    used_rows: list[dict[str, Any]] = []
    for row in executor_rows:
        if max_samples > 0 and len(features) >= max_samples:
            break
        uuid = str(row["uuid"])
        prior_row = prior_by_uuid.get(uuid)
        if prior_row is None:
            continue
        executor_npz = np.load(str(row["sample_npz"]), allow_pickle=False)
        prior_npz = np.load(str(prior_row["dp_prior_npz"]), allow_pickle=False)
        current = executor_npz["current_state"].astype(np.float32).reshape(-1)
        task_path = executor_npz["task_path"].astype(np.float32)
        teacher = executor_npz["teacher_robot_actions"].astype(np.float32)
        prior = prior_npz["dp_prior_actions"].astype(np.float32)
        horizon = min(int(task_path.shape[0]), int(teacher.shape[0]), int(prior.shape[0]))
        if horizon <= 0:
            continue
        task_path = task_path[:horizon]
        teacher = teacher[:horizon]
        prior = prior[:horizon]
        feature = np.concatenate([current, task_path.reshape(-1), prior.reshape(-1)]).astype(np.float32)
        target = teacher.reshape(-1).astype(np.float32)
        prior_flat = prior.reshape(-1).astype(np.float32)
        features.append(feature)
        targets.append(target)
        priors.append(prior_flat)
        used_rows.append(
            {
                "uuid": uuid,
                "source_uuid": row.get("source_uuid"),
                "scenario": row.get("scenario"),
                "prefix_role": row.get("prefix_role"),
                "horizon": horizon,
                "task_path_source": row.get("task_path_source"),
                "executor_sample_npz": row.get("sample_npz"),
                "dp_prior_npz": prior_row.get("dp_prior_npz"),
            }
        )
    if not features:
        raise RuntimeError("no matched executor/DP-prior samples")
    feature_widths = {item.shape[0] for item in features}
    target_widths = {item.shape[0] for item in targets}
    prior_widths = {item.shape[0] for item in priors}
    if len(feature_widths) != 1 or len(target_widths) != 1 or target_widths != prior_widths:
        raise RuntimeError(f"nonuniform widths: features={feature_widths} targets={target_widths} priors={prior_widths}")
    return np.stack(features), np.stack(targets), np.stack(priors), used_rows


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction))) if n >= 4 else 1
    n_val = min(n_val, max(1, n - 1))
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


class ResidualExecutorDataset:
    def __init__(self, x: np.ndarray, residual: np.ndarray, prior: np.ndarray, target: np.ndarray) -> None:
        self.x = x
        self.residual = residual
        self.prior = prior
        self.target = target

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[Any, Any, Any, Any]:
        return self.x[idx], self.residual[idx], self.prior[idx], self.target[idx]


def build_model(feature_dim: int, target_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> Any:
    import torch

    layers: list[torch.nn.Module] = []
    width = int(hidden_dim)
    in_dim = int(feature_dim)
    for _ in range(int(num_layers)):
        layers.append(torch.nn.Linear(in_dim, width))
        layers.append(torch.nn.LayerNorm(width))
        layers.append(torch.nn.GELU())
        if dropout > 0:
            layers.append(torch.nn.Dropout(float(dropout)))
        in_dim = width
    layers.append(torch.nn.Linear(width, int(target_dim)))
    return torch.nn.Sequential(*layers)


def init_distributed() -> tuple[int, int, int]:
    import torch
    import torch.distributed as dist

    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def cleanup_distributed(world_size: int) -> None:
    if world_size <= 1:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> int:
    args = parse_args()
    require_compute_step()

    import torch
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

    rank, world_size, local_rank = init_distributed()
    is_rank0 = rank == 0
    if int(args.formal_min_gpus) > 1 and world_size < int(args.formal_min_gpus):
        raise SystemExit(f"formal training requires world_size >= {args.formal_min_gpus}, got {world_size}")

    seed = int(args.seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")

    output_root = Path(args.output_root).resolve()
    if is_rank0:
        output_root.mkdir(parents=True, exist_ok=True)

    executor_rows = read_jsonl(Path(args.executor_jsonl).resolve())
    prior_rows = read_jsonl(Path(args.dp_prior_jsonl).resolve())
    x_raw, y_raw, prior_raw, used_rows = load_arrays(executor_rows, prior_rows, int(args.max_samples))
    task_sources = sorted({str(row.get("task_path_source") or "unknown") for row in used_rows})
    if any("gt" in source and "debug" in source for source in task_sources):
        raise SystemExit(f"refusing formal training on GT debug task paths: {task_sources}")

    train_idx, val_idx = split_indices(len(used_rows), float(args.val_fraction), int(args.seed))
    x_mean = x_raw[train_idx].mean(axis=0, keepdims=True)
    x_std = x_raw[train_idx].std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    residual_raw = y_raw - prior_raw
    residual_mean = residual_raw[train_idx].mean(axis=0, keepdims=True)
    residual_std = residual_raw[train_idx].std(axis=0, keepdims=True)
    residual_std = np.where(residual_std < 1e-6, 1.0, residual_std)
    x_norm = ((x_raw - x_mean) / x_std).astype(np.float32)
    residual_norm = ((residual_raw - residual_mean) / residual_std).astype(np.float32)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    train_ds = TensorDataset(
        torch.from_numpy(x_norm[train_idx]),
        torch.from_numpy(residual_norm[train_idx]),
        torch.from_numpy(prior_raw[train_idx].astype(np.float32)),
        torch.from_numpy(y_raw[train_idx].astype(np.float32)),
    )
    val_x = torch.from_numpy(x_norm[val_idx]).to(device)
    val_resid = torch.from_numpy(residual_norm[val_idx]).to(device)
    val_prior = torch.from_numpy(prior_raw[val_idx].astype(np.float32)).to(device)
    val_target = torch.from_numpy(y_raw[val_idx].astype(np.float32)).to(device)

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=int(args.seed))
    loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    model = build_model(x_norm.shape[1], y_raw.shape[1], int(args.hidden_dim), int(args.num_layers), float(args.dropout)).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    residual_mean_t = torch.from_numpy(residual_mean.astype(np.float32)).to(device)
    residual_std_t = torch.from_numpy(residual_std.astype(np.float32)).to(device)
    start = time.time()
    history: list[dict[str, Any]] = []
    step = 0
    last_save_step = 0

    def evaluate() -> dict[str, float]:
        module = model.module if hasattr(model, "module") else model
        module.eval()
        with torch.no_grad():
            pred_resid_norm = module(val_x)
            pred_resid = pred_resid_norm * residual_std_t + residual_mean_t
            pred_action = val_prior + pred_resid
            train_target_space = torch.mean((pred_resid_norm - val_resid) ** 2).item()
            action_mse = torch.mean((pred_action - val_target) ** 2).item()
            baseline_mse = torch.mean((val_prior - val_target) ** 2).item()
            max_abs = torch.max(torch.abs(pred_action - val_target)).item()
        module.train()
        return {
            "val_residual_norm_mse": float(train_target_space),
            "val_action_mse": float(action_mse),
            "val_baseline_dp_prior_mse": float(baseline_mse),
            "val_max_abs_action_error": float(max_abs),
        }

    def save_checkpoint(name: str, metrics: dict[str, Any]) -> None:
        if not is_rank0:
            return
        module = model.module if hasattr(model, "module") else model
        payload = {
            "model_state_dict": module.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "step": int(step),
            "feature_dim": int(x_norm.shape[1]),
            "target_dim": int(y_raw.shape[1]),
            "x_mean": x_mean.astype(np.float32),
            "x_std": x_std.astype(np.float32),
            "residual_mean": residual_mean.astype(np.float32),
            "residual_std": residual_std.astype(np.float32),
            "args": vars(args),
            "metrics": metrics,
        }
        torch.save(payload, output_root / name)

    if is_rank0:
        baseline_train = float(np.mean((prior_raw[train_idx] - y_raw[train_idx]) ** 2))
        baseline_val = float(np.mean((prior_raw[val_idx] - y_raw[val_idx]) ** 2))
        write_json(
            output_root / "training_manifest.json",
            {
                "schema": "cosmos3_executor_residual_training_v1",
                "executor_jsonl": str(Path(args.executor_jsonl).resolve()),
                "dp_prior_jsonl": str(Path(args.dp_prior_jsonl).resolve()),
                "output_root": str(output_root),
                "num_samples": int(len(used_rows)),
                "num_train": int(len(train_idx)),
                "num_val": int(len(val_idx)),
                "feature_dim": int(x_norm.shape[1]),
                "target_dim": int(y_raw.shape[1]),
                "task_path_sources": task_sources,
                "world_size": int(world_size),
                "formal_min_gpus": int(args.formal_min_gpus),
                "min_wall_seconds": int(args.min_wall_seconds),
                "baseline_train_dp_prior_mse": baseline_train,
                "baseline_val_dp_prior_mse": baseline_val,
                "resource_boundary": "tmux-held Slurm allocation; torchrun/DDP; no sbatch.",
                "method_boundary": "Residual executor uses causal Cosmos-predicted task path plus frozen DP prior; no GT future task path conditioning.",
            },
        )

    stop_reason = "running"
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for batch in loader:
            step += 1
            x_b, resid_b, prior_b, target_b = [item.to(device, non_blocking=True) for item in batch]
            pred_resid_norm = model(x_b)
            pred_resid = pred_resid_norm * residual_std_t + residual_mean_t
            pred_action = prior_b + pred_resid
            residual_loss = torch.mean((pred_resid_norm - resid_b) ** 2)
            action_loss = torch.mean((pred_action - target_b) ** 2)
            residual_reg = torch.mean(pred_resid**2)
            loss = residual_loss + action_loss + float(args.residual_l2_weight) * residual_reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            elapsed = time.time() - start
            should_eval = step == 1 or step % int(args.eval_every_steps) == 0
            if should_eval:
                metrics = {
                    "step": int(step),
                    "elapsed_seconds": float(elapsed),
                    "train_loss": float(loss.detach().cpu().item()),
                    "train_action_mse": float(action_loss.detach().cpu().item()),
                    "train_residual_norm_mse": float(residual_loss.detach().cpu().item()),
                }
                if is_rank0:
                    metrics.update(evaluate())
                    history.append(metrics)
                    write_json(output_root / "training_history.json", history)
                    print(json.dumps(metrics, sort_keys=True), flush=True)
                if world_size > 1:
                    import torch.distributed as dist

                    dist.barrier()
            if step == 1 or step - last_save_step >= int(args.save_every_steps):
                metrics = history[-1] if history else {"step": int(step), "elapsed_seconds": float(elapsed)}
                save_checkpoint("checkpoint_latest.pt", metrics)
                if world_size > 1:
                    import torch.distributed as dist

                    dist.barrier()
                last_save_step = step

            elapsed = time.time() - start
            if elapsed >= int(args.max_wall_seconds):
                stop_reason = "max_wall_seconds"
                break
            if elapsed >= int(args.min_wall_seconds) and step >= int(args.min_steps):
                stop_reason = "min_wall_and_min_steps"
                break
            if int(args.min_wall_seconds) <= 0 and step >= int(args.max_steps):
                stop_reason = "max_steps"
                break
        epoch += 1
        if stop_reason != "running":
            break

    elapsed = time.time() - start
    final_metrics = evaluate() if is_rank0 else {}
    if is_rank0:
        save_checkpoint("checkpoint_final.pt", final_metrics | {"step": int(step), "elapsed_seconds": float(elapsed)})
        summary = {
            "schema": "cosmos3_executor_residual_training_summary_v1",
            "executor_jsonl": str(Path(args.executor_jsonl).resolve()),
            "dp_prior_jsonl": str(Path(args.dp_prior_jsonl).resolve()),
            "output_root": str(output_root),
            "num_samples": int(len(used_rows)),
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "world_size": int(world_size),
            "steps": int(step),
            "elapsed_seconds": float(elapsed),
            "stop_reason": stop_reason,
            "task_path_sources": task_sources,
            "final_metrics": final_metrics,
            "ready_for_closed_loop_eval": bool(final_metrics and final_metrics["val_action_mse"] < final_metrics["val_baseline_dp_prior_mse"]),
            "requested_min_wall_seconds": int(args.min_wall_seconds),
            "formal_training_floor_met": bool(
                world_size >= int(args.formal_min_gpus)
                and int(args.min_wall_seconds) >= 10800
                and elapsed >= int(args.min_wall_seconds)
            ),
            "boundary": "Training evidence only. Method success still requires closed-loop real-state eval and video review.",
        }
        write_json(output_root / "training_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    cleanup_distributed(world_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
