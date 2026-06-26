#!/usr/bin/env python3
"""Train a contact/progress-conditioned Cosmos executor debug model."""

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
    parser.add_argument("--contact-executor-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--residual-l2-weight", type=float, default=1e-4)
    parser.add_argument("--progress-loss-weight", type=float, default=0.2)
    parser.add_argument("--continuability-loss-weight", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--min-steps", type=int, default=0)
    parser.add_argument("--min-wall-seconds", type=int, default=0)
    parser.add_argument("--max-wall-seconds", type=int, default=0)
    parser.add_argument("--formal-min-gpus", type=int, default=1)
    parser.add_argument("--eval-every-steps", type=int, default=10)
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=20260615)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def phase_one_hot(phase_id: int, width: int = 6) -> np.ndarray:
    out = np.zeros((width,), dtype=np.float32)
    if 0 <= int(phase_id) < width:
        out[int(phase_id)] = 1.0
    return out


def contact_targets_for_horizon(row: dict[str, Any], horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.load(str(row["contact_label_npz"]), allow_pickle=False)
    start = int(row.get("action_start_step") or row.get("prefix_frame_index") or 0)
    end_frame = min(start + int(horizon), int(labels["contact_progress"].shape[0]) - 1)
    future_slice = slice(start + 1, end_frame + 1)
    current_progress = float(labels["contact_progress"][start])
    end_progress = float(labels["contact_progress"][end_frame])
    current_phase_id = int(labels["phase_id"][start])
    current_inserted = float(bool(labels["inserted"][start]))
    current_grasped = float(bool(labels["grasped"][start]))
    current_dp = float(bool(labels["dp_continuable"][start]))
    peg_head = labels["peg_head_at_hole"][start].astype(np.float32).reshape(3)
    current_context = np.concatenate(
        [
            phase_one_hot(current_phase_id),
            np.asarray(
                [
                    current_progress,
                    current_grasped,
                    current_inserted,
                    current_dp,
                    float(start) / 300.0,
                    float(end_frame) / 300.0,
                ],
                dtype=np.float32,
            ),
            peg_head,
        ]
    ).astype(np.float32)
    progress_target = np.asarray([end_progress, end_progress - current_progress], dtype=np.float32)
    if start + 1 <= end_frame:
        future_inserted = float(bool(np.any(labels["inserted"][future_slice])))
        future_dp = float(bool(np.any(labels["dp_continuable"][future_slice])))
    else:
        future_inserted = 0.0
        future_dp = 0.0
    binary_target = np.asarray([future_inserted, future_dp], dtype=np.float32)
    return current_context, progress_target, binary_target


def load_arrays(
    rows: list[dict[str, Any]],
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    priors: list[np.ndarray] = []
    progress_targets: list[np.ndarray] = []
    binary_targets: list[np.ndarray] = []
    used_rows: list[dict[str, Any]] = []
    for row in rows:
        if max_samples > 0 and len(features) >= max_samples:
            break
        task_source = str(row.get("task_path_source") or "")
        if "gt" in task_source and "debug" in task_source:
            raise SystemExit(f"refusing GT debug task path source: {task_source}")
        executor_npz = np.load(str(row["executor_sample_npz"]), allow_pickle=False)
        prior_npz = np.load(str(row["dp_prior_npz"]), allow_pickle=False)
        current = executor_npz["current_state"].astype(np.float32).reshape(-1)
        task_path = executor_npz["task_path"].astype(np.float32)
        teacher = executor_npz["teacher_robot_actions"].astype(np.float32)
        prior = prior_npz["dp_prior_actions"].astype(np.float32)
        horizon = min(int(task_path.shape[0]), int(teacher.shape[0]), int(prior.shape[0]))
        if horizon <= 0:
            continue
        contact_context, progress_target, binary_target = contact_targets_for_horizon(row, horizon)
        task_path = task_path[:horizon]
        teacher = teacher[:horizon]
        prior = prior[:horizon]
        feature = np.concatenate(
            [current, task_path.reshape(-1), prior.reshape(-1), contact_context]
        ).astype(np.float32)
        features.append(feature)
        targets.append(teacher.reshape(-1).astype(np.float32))
        priors.append(prior.reshape(-1).astype(np.float32))
        progress_targets.append(progress_target)
        binary_targets.append(binary_target)
        used_rows.append(
            {
                "uuid": row.get("uuid"),
                "source_uuid": row.get("source_uuid"),
                "scenario": row.get("scenario"),
                "prefix_role": row.get("prefix_role"),
                "current_phase": row.get("current_phase"),
                "horizon": int(horizon),
                "prefix_frame_index": row.get("prefix_frame_index"),
                "action_start_step": row.get("action_start_step"),
                "action_end_step": row.get("action_end_step"),
                "contact_label_npz": row.get("contact_label_npz"),
                "task_path_source": task_source,
            }
        )
    if not features:
        raise RuntimeError("no usable contact executor samples")
    widths = {
        "feature": {item.shape[0] for item in features},
        "target": {item.shape[0] for item in targets},
        "prior": {item.shape[0] for item in priors},
        "progress": {item.shape[0] for item in progress_targets},
        "binary": {item.shape[0] for item in binary_targets},
    }
    if any(len(value) != 1 for value in widths.values()) or widths["target"] != widths["prior"]:
        raise RuntimeError(f"nonuniform sample widths: {widths}")
    return (
        np.stack(features),
        np.stack(targets),
        np.stack(priors),
        np.stack(progress_targets),
        np.stack(binary_targets),
        used_rows,
    )


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    if n <= 1 or val_fraction <= 0:
        return np.sort(perm), np.sort(perm[:0])
    n_val = max(1, int(round(n * val_fraction)))
    n_val = min(n_val, max(1, n - 1))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


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


def distributed_barrier(world_size: int) -> None:
    if world_size <= 1:
        return
    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", "0"))])
        else:
            dist.barrier()


def synchronize_stop_reason(local_reason: str, world_size: int, device: Any) -> str:
    if world_size <= 1:
        return local_reason
    import torch
    import torch.distributed as dist

    reason_to_code = {
        "running": 0,
        "min_wall_and_min_steps": 1,
        "max_wall_seconds": 2,
        "max_steps": 3,
    }
    code_to_reason = {value: key for key, value in reason_to_code.items()}
    code = reason_to_code.get(local_reason, 0)
    tensor = torch.tensor([int(code)], dtype=torch.int32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return code_to_reason[int(tensor.item())]


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    torch.save(payload, tmp)
    os.replace(tmp, path)


class ContactExecutorNet:
    def __new__(
        cls,
        feature_dim: int,
        target_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> Any:
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
                    if dropout > 0:
                        layers.append(torch.nn.Dropout(float(dropout)))
                    in_dim = int(hidden_dim)
                self.trunk = torch.nn.Sequential(*layers)
                self.action_head = torch.nn.Linear(int(hidden_dim), int(target_dim))
                self.progress_head = torch.nn.Linear(int(hidden_dim), 2)
                self.binary_head = torch.nn.Linear(int(hidden_dim), 2)

            def forward(self, x: Any) -> tuple[Any, Any, Any]:
                z = self.trunk(x)
                return self.action_head(z), self.progress_head(z), self.binary_head(z)

        return _Net()


def main() -> int:
    args = parse_args()
    require_compute_step()

    import torch
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

    rank, world_size, local_rank = init_distributed()
    is_rank0 = rank == 0
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("require_cuda=true but torch.cuda.is_available() is false")
    if int(args.min_wall_seconds) > 0 and world_size < int(args.formal_min_gpus):
        raise SystemExit(f"formal training requires world_size >= {args.formal_min_gpus}, got {world_size}")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    random.seed(int(args.seed) + rank)
    np.random.seed(int(args.seed) + rank)
    torch.manual_seed(int(args.seed) + rank)
    torch.set_float32_matmul_precision("high")

    output_root = Path(args.output_root).resolve()
    if is_rank0:
        output_root.mkdir(parents=True, exist_ok=True)
    distributed_barrier(world_size)
    rows = read_jsonl(Path(args.contact_executor_jsonl).resolve())
    x_raw, y_raw, prior_raw, progress_raw, binary_raw, used_rows = load_arrays(rows, int(args.max_samples))

    train_idx, val_idx = split_indices(len(used_rows), float(args.val_fraction), int(args.seed))
    eval_idx = val_idx if len(val_idx) else train_idx
    x_mean = x_raw[train_idx].mean(axis=0, keepdims=True)
    x_std = x_raw[train_idx].std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    residual_raw = y_raw - prior_raw
    residual_mean = residual_raw[train_idx].mean(axis=0, keepdims=True)
    residual_std = residual_raw[train_idx].std(axis=0, keepdims=True)
    residual_std = np.where(residual_std < 1e-6, 1.0, residual_std)
    x_norm = ((x_raw - x_mean) / x_std).astype(np.float32)
    residual_norm = ((residual_raw - residual_mean) / residual_std).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(x_norm[train_idx]),
        torch.from_numpy(residual_norm[train_idx]),
        torch.from_numpy(prior_raw[train_idx].astype(np.float32)),
        torch.from_numpy(y_raw[train_idx].astype(np.float32)),
        torch.from_numpy(progress_raw[train_idx].astype(np.float32)),
        torch.from_numpy(binary_raw[train_idx].astype(np.float32)),
    )
    sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=int(args.seed))
        if world_size > 1
        else None
    )
    loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )

    model = ContactExecutorNet(
        feature_dim=int(x_norm.shape[1]),
        target_dim=int(y_raw.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
    ).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    bce = torch.nn.BCEWithLogitsLoss()
    residual_mean_t = torch.from_numpy(residual_mean.astype(np.float32)).to(device)
    residual_std_t = torch.from_numpy(residual_std.astype(np.float32)).to(device)

    eval_tensors = {
        "x": torch.from_numpy(x_norm[eval_idx]).to(device),
        "resid": torch.from_numpy(residual_norm[eval_idx]).to(device),
        "prior": torch.from_numpy(prior_raw[eval_idx].astype(np.float32)).to(device),
        "target": torch.from_numpy(y_raw[eval_idx].astype(np.float32)).to(device),
        "progress": torch.from_numpy(progress_raw[eval_idx].astype(np.float32)).to(device),
        "binary": torch.from_numpy(binary_raw[eval_idx].astype(np.float32)).to(device),
    }
    train_eval_tensors = {
        "x": torch.from_numpy(x_norm[train_idx]).to(device),
        "resid": torch.from_numpy(residual_norm[train_idx]).to(device),
        "prior": torch.from_numpy(prior_raw[train_idx].astype(np.float32)).to(device),
        "target": torch.from_numpy(y_raw[train_idx].astype(np.float32)).to(device),
        "progress": torch.from_numpy(progress_raw[train_idx].astype(np.float32)).to(device),
        "binary": torch.from_numpy(binary_raw[train_idx].astype(np.float32)).to(device),
    }

    def evaluate(prefix: str, tensors: dict[str, Any]) -> dict[str, float]:
        module = model.module if hasattr(model, "module") else model
        model.eval()
        with torch.no_grad():
            pred_resid_norm, pred_progress, pred_logits = module(tensors["x"])
            pred_resid = pred_resid_norm * residual_std_t + residual_mean_t
            pred_action = tensors["prior"] + pred_resid
            probs = torch.sigmoid(pred_logits)
            pred_binary = probs >= 0.5
            target_binary = tensors["binary"] >= 0.5
            out = {
                f"{prefix}_residual_norm_mse": float(torch.mean((pred_resid_norm - tensors["resid"]) ** 2).item()),
                f"{prefix}_action_mse": float(torch.mean((pred_action - tensors["target"]) ** 2).item()),
                f"{prefix}_baseline_dp_prior_mse": float(torch.mean((tensors["prior"] - tensors["target"]) ** 2).item()),
                f"{prefix}_progress_mse": float(torch.mean((pred_progress - tensors["progress"]) ** 2).item()),
                f"{prefix}_binary_bce": float(bce(pred_logits, tensors["binary"]).item()),
                f"{prefix}_inserted_acc": float((pred_binary[:, 0] == target_binary[:, 0]).float().mean().item()),
                f"{prefix}_dp_continuable_acc": float((pred_binary[:, 1] == target_binary[:, 1]).float().mean().item()),
            }
        model.train()
        return out

    if is_rank0:
        write_json(
            output_root / "training_manifest.json",
            {
                "schema": "cosmos3_contact_executor_training_v1",
                "contact_executor_jsonl": str(Path(args.contact_executor_jsonl).resolve()),
                "output_root": str(output_root),
                "num_samples": int(len(used_rows)),
                "num_train": int(len(train_idx)),
                "num_val": int(len(val_idx)),
                "feature_dim": int(x_norm.shape[1]),
                "target_dim": int(y_raw.shape[1]),
                "action_horizon": int(y_raw.shape[1] // 7),
                "device": str(device),
                "world_size": int(world_size),
                "formal_min_gpus": int(args.formal_min_gpus),
                "min_wall_seconds": int(args.min_wall_seconds),
                "max_wall_seconds": int(args.max_wall_seconds),
                "method_boundary": (
                    "Contact/progress labels supervise action-relevant features and "
                    "offline scoring. Future labels are targets only; live execution "
                    "must condition on causal current state and Cosmos-predicted task/contact paths."
                ),
                "debug_boundary": (
                    "This trainer is still deterministic. It is the first contact/progress "
                    "executor gate, not final diffusion/candidate executor evidence."
                ),
                "sample_rows": used_rows[:10],
            },
        )

    history: list[dict[str, Any]] = []
    start = time.time()
    step = 0
    last_save = 0
    stop_reason = "running"
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            step += 1
            x_b, resid_b, prior_b, target_b, progress_b, binary_b = [item.to(device, non_blocking=True) for item in batch]
            pred_resid_norm, pred_progress, pred_logits = model(x_b)
            pred_resid = pred_resid_norm * residual_std_t + residual_mean_t
            pred_action = prior_b + pred_resid
            residual_loss = torch.mean((pred_resid_norm - resid_b) ** 2)
            action_loss = torch.mean((pred_action - target_b) ** 2)
            progress_loss = torch.mean((pred_progress - progress_b) ** 2)
            binary_loss = bce(pred_logits, binary_b)
            residual_reg = torch.mean(pred_resid**2)
            loss = (
                residual_loss
                + action_loss
                + float(args.progress_loss_weight) * progress_loss
                + float(args.continuability_loss_weight) * binary_loss
                + float(args.residual_l2_weight) * residual_reg
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 1 or step % int(args.eval_every_steps) == 0:
                metrics = {
                    "step": int(step),
                    "elapsed_seconds": float(time.time() - start),
                    "train_batch_loss": float(loss.detach().cpu().item()),
                    "train_batch_action_mse": float(action_loss.detach().cpu().item()),
                    "train_batch_progress_mse": float(progress_loss.detach().cpu().item()),
                    "train_batch_binary_bce": float(binary_loss.detach().cpu().item()),
                }
                if is_rank0:
                    metrics.update(evaluate("train", train_eval_tensors))
                    metrics.update(evaluate("eval", eval_tensors))
                    history.append(metrics)
                    write_json(output_root / "training_history.json", history)
                    print(json.dumps(metrics, sort_keys=True), flush=True)
                distributed_barrier(world_size)
            if step == 1 or step - last_save >= int(args.save_every_steps):
                if is_rank0:
                    module = model.module if hasattr(model, "module") else model
                    atomic_torch_save(
                        {
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
                            "latest_metrics": history[-1] if history else {},
                        },
                        output_root / "checkpoint_latest.pt",
                    )
                distributed_barrier(world_size)
                last_save = step
            elapsed = time.time() - start
            local_stop_reason = "running"
            if int(args.min_wall_seconds) > 0:
                max_wall = int(args.max_wall_seconds) if int(args.max_wall_seconds) > 0 else int(args.min_wall_seconds)
                if elapsed >= max_wall:
                    local_stop_reason = "max_wall_seconds"
                elif elapsed >= int(args.min_wall_seconds) and step >= int(args.min_steps):
                    local_stop_reason = "min_wall_and_min_steps"
            elif step >= int(args.max_steps):
                local_stop_reason = "max_steps"
            stop_reason = synchronize_stop_reason(local_stop_reason, world_size, device)
            if stop_reason != "running":
                break
        epoch += 1
        if stop_reason != "running":
            break

    final_metrics = history[-1] if history else {}
    if is_rank0:
        module = model.module if hasattr(model, "module") else model
        final_checkpoint_path = output_root / "checkpoint_final.pt"
        final_payload = {
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
            "latest_metrics": final_metrics,
        }
        atomic_torch_save(final_payload, final_checkpoint_path)
        try:
            torch.load(final_checkpoint_path, map_location="cpu")
        except Exception as exc:
            write_json(
                output_root / "training_final_checkpoint_error.json",
                {
                    "schema": "cosmos3_contact_executor_final_checkpoint_error_v1",
                    "checkpoint": str(final_checkpoint_path),
                    "error": repr(exc),
                    "step": int(step),
                    "stop_reason": stop_reason,
                },
            )
            raise
        elapsed = float(time.time() - start)
        formal_floor_met = bool(
            world_size >= int(args.formal_min_gpus)
            and int(args.min_wall_seconds) >= 10800
            and elapsed >= int(args.min_wall_seconds)
        )
        ready_for_debug = bool(
            final_metrics
            and final_metrics["train_action_mse"] < final_metrics["train_baseline_dp_prior_mse"]
            and final_metrics["train_progress_mse"] < 0.05
        )
        ready_for_formal_eval = bool(
            formal_floor_met
            and final_metrics
            and final_metrics["eval_action_mse"] <= final_metrics["eval_baseline_dp_prior_mse"]
            and final_metrics["eval_progress_mse"] < 0.05
            and final_metrics["eval_dp_continuable_acc"] >= 0.75
            and final_metrics["eval_inserted_acc"] >= 0.75
        )
        summary = {
            "schema": "cosmos3_contact_executor_training_summary_v1",
            "contact_executor_jsonl": str(Path(args.contact_executor_jsonl).resolve()),
            "output_root": str(output_root),
            "num_samples": int(len(used_rows)),
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "world_size": int(world_size),
            "steps": int(step),
            "elapsed_seconds": elapsed,
            "stop_reason": stop_reason,
            "final_metrics": final_metrics,
            "ready_for_debug_gate": ready_for_debug,
            "ready_for_formal_eval": ready_for_formal_eval,
            "formal_training_floor_met": formal_floor_met,
            "boundary": (
                "Contact/progress executor training evidence only. Method success "
                "requires closed-loop real-state metrics and inspected videos."
            ),
        }
        write_json(output_root / "training_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    distributed_barrier(world_size)
    cleanup_distributed(world_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
