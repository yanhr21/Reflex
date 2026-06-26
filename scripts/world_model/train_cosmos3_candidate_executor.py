#!/usr/bin/env python3
"""Train a stochastic candidate executor with contact/progress value scoring."""

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
from train_cosmos3_contact_executor import load_arrays, read_jsonl, split_indices  # noqa: E402

OFFLINE_GATE_THRESHOLDS = {
    "max_teacher_progress_mse": 0.05,
    "max_teacher_value_mse": 0.25,
    "min_teacher_inserted_acc": 0.75,
    "min_teacher_dp_continuable_acc": 0.75,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-executor-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--nll-loss-weight", type=float, default=1.0)
    parser.add_argument("--mean-action-loss-weight", type=float, default=0.2)
    parser.add_argument("--progress-loss-weight", type=float, default=0.5)
    parser.add_argument("--binary-loss-weight", type=float, default=0.5)
    parser.add_argument("--value-loss-weight", type=float, default=0.5)
    parser.add_argument("--next-state-loss-weight", type=float, default=0.5)
    parser.add_argument("--candidate-rank-loss-weight", type=float, default=0.35)
    parser.add_argument("--candidate-rank-random-count", type=int, default=4)
    parser.add_argument("--candidate-rank-diffusion-count", type=int, default=0)
    parser.add_argument("--candidate-rank-temperature", type=float, default=1.0)
    parser.add_argument("--logstd-min", type=float, default=-4.5)
    parser.add_argument("--logstd-max", type=float, default=1.0)
    parser.add_argument("--generator-type", choices=("gaussian", "diffusion"), default="gaussian")
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--diffusion-beta-start", type=float, default=1e-4)
    parser.add_argument("--diffusion-beta-end", type=float, default=2e-2)
    parser.add_argument("--diffusion-loss-weight", type=float, default=1.0)
    parser.add_argument("--candidate-samples", type=int, default=24)
    parser.add_argument("--candidate-temps", default="0.5,1.0,1.5")
    parser.add_argument("--candidate-scales", default="0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--score-inserted-weight", type=float, default=0.6)
    parser.add_argument("--score-dp-continuable-weight", type=float, default=0.3)
    parser.add_argument("--score-value-weight", type=float, default=0.4)
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
    parser.add_argument("--dp-fallback-score-margin", type=float, default=0.25)
    parser.add_argument("--selector-residual-l2-cap-quantile", type=float, default=0.9)
    parser.add_argument("--selector-residual-l2-cap-min", type=float, default=1e-4)
    parser.add_argument("--selector-residual-l2-cap-max", type=float, default=0.02)
    parser.add_argument("--selector-residual-l2-cap-multiplier", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--min-steps", type=int, default=0)
    parser.add_argument("--min-wall-seconds", type=int, default=0)
    parser.add_argument("--max-wall-seconds", type=int, default=0)
    parser.add_argument("--formal-min-gpus", type=int, default=2)
    parser.add_argument("--eval-every-steps", type=int, default=50)
    parser.add_argument("--save-every-steps", type=int, default=1000)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=20260615)
    return parser.parse_args()


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
    tensor = torch.tensor([reason_to_code.get(local_reason, 0)], dtype=torch.int32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return code_to_reason[int(tensor.item())]


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    return values or [1.0]


def parse_scale_list(text: str) -> list[float]:
    values = []
    for value in parse_float_list(text):
        if value < 0:
            raise ValueError(f"candidate scales must be non-negative, got {value}")
        if value == 0.0:
            continue
        values.append(float(value))
    return values or [0.05, 0.1, 0.2]


def value_target(progress: np.ndarray, binary: np.ndarray) -> np.ndarray:
    delta = progress[:, 1]
    inserted = binary[:, 0]
    dp_cont = binary[:, 1]
    return (delta + 0.6 * inserted + 0.3 * dp_cont).astype(np.float32).reshape(-1, 1)


def fixed_width_vector(text: str, width: int, default: float = 0.0) -> np.ndarray:
    values = parse_float_list(str(text)) if str(text).strip() else []
    if not values:
        values = [float(default)] * int(width)
    if len(values) < int(width):
        values.extend([float(values[-1])] * (int(width) - len(values)))
    return np.asarray(values[: int(width)], dtype=np.float32)


def future_task_state_targets(used_rows: list[dict[str, Any]]) -> np.ndarray:
    targets: list[np.ndarray] = []
    for row in used_rows:
        label_path = row.get("contact_label_npz")
        if not label_path:
            raise RuntimeError(f"missing contact_label_npz for row {row.get('uuid')}")
        labels = np.load(str(label_path), allow_pickle=False)
        peg_head = labels["peg_head_at_hole"].astype(np.float32)
        start = int(row.get("action_start_step") or row.get("prefix_frame_index") or 0)
        horizon = int(row.get("horizon") or 0)
        end_frame = min(max(0, start + horizon), int(peg_head.shape[0]) - 1)
        targets.append(peg_head[end_frame].reshape(3).astype(np.float32))
    if not targets:
        raise RuntimeError("no future task-state targets")
    return np.stack(targets).astype(np.float32)


def compute_phase_residual_l2_caps(
    *,
    used_rows: list[dict[str, Any]],
    train_idx: np.ndarray,
    residual_raw: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float]:
    residual_l2 = np.mean(np.asarray(residual_raw, dtype=np.float32) ** 2, axis=1)
    quantile = float(args.selector_residual_l2_cap_quantile)
    multiplier = float(args.selector_residual_l2_cap_multiplier)
    cap_min = float(args.selector_residual_l2_cap_min)
    cap_max = float(args.selector_residual_l2_cap_max)

    def cap(values: np.ndarray) -> float:
        if values.size == 0 or quantile <= 0:
            raw = cap_max
        else:
            raw = float(np.quantile(values.astype(np.float64), min(max(quantile, 0.0), 1.0))) * multiplier
        return float(np.clip(raw, cap_min, cap_max))

    train_idx = np.asarray(train_idx, dtype=np.int64)
    caps: dict[str, float] = {"__global__": cap(residual_l2[train_idx])}
    phases = sorted({str(used_rows[int(i)].get("current_phase") or "unknown") for i in train_idx})
    for phase in phases:
        phase_indices = [int(i) for i in train_idx if str(used_rows[int(i)].get("current_phase") or "unknown") == phase]
        caps[phase] = cap(residual_l2[np.asarray(phase_indices, dtype=np.int64)])
    return caps


def offline_gate_from_eval(eval_metrics: dict[str, Any]) -> bool:
    source_counts = eval_metrics.get("candidate_source_counts") if isinstance(eval_metrics, dict) else {}
    source_counts = source_counts if isinstance(source_counts, dict) else {}
    non_dp_selected = 0
    for key, value in source_counts.items():
        if str(key) == "dp_prior":
            continue
        try:
            non_dp_selected += int(value)
        except (TypeError, ValueError):
            pass
    try:
        teacher_progress_mse = float(eval_metrics.get("teacher_progress_mse", float("inf")))
        teacher_value_mse = float(eval_metrics.get("teacher_value_mse", float("inf")))
        teacher_inserted_acc = float(eval_metrics.get("teacher_inserted_acc", 0.0))
        teacher_dp_continuable_acc = float(eval_metrics.get("teacher_dp_continuable_acc", 0.0))
        selected_action_mse = float(eval_metrics.get("selected_action_mse", float("inf")))
        dp_prior_action_mse = float(eval_metrics.get("dp_prior_action_mse", float("inf")))
    except (TypeError, ValueError):
        return False
    return bool(
        eval_metrics
        and np.isfinite(teacher_progress_mse)
        and np.isfinite(teacher_value_mse)
        and np.isfinite(teacher_inserted_acc)
        and np.isfinite(teacher_dp_continuable_acc)
        and np.isfinite(selected_action_mse)
        and np.isfinite(dp_prior_action_mse)
        and teacher_progress_mse <= OFFLINE_GATE_THRESHOLDS["max_teacher_progress_mse"]
        and teacher_value_mse <= OFFLINE_GATE_THRESHOLDS["max_teacher_value_mse"]
        and teacher_inserted_acc >= OFFLINE_GATE_THRESHOLDS["min_teacher_inserted_acc"]
        and teacher_dp_continuable_acc >= OFFLINE_GATE_THRESHOLDS["min_teacher_dp_continuable_acc"]
        and selected_action_mse <= dp_prior_action_mse
        and non_dp_selected > 0
    )


def diffusion_schedule(
    *,
    steps: int,
    beta_start: float,
    beta_end: float,
    device: Any,
) -> tuple[Any, Any, Any]:
    import torch

    step_count = max(2, int(steps))
    betas = torch.linspace(float(beta_start), float(beta_end), step_count, device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


class CandidateExecutorNet:
    def __new__(
        cls,
        feature_dim: int,
        target_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        logstd_min: float,
        logstd_max: float,
        next_state_dim: int = 0,
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
                self.encoder = torch.nn.Sequential(*layers)
                self.mean_head = torch.nn.Linear(int(hidden_dim), int(target_dim))
                self.logstd_head = torch.nn.Linear(int(hidden_dim), int(target_dim))
                self.scorer = torch.nn.Sequential(
                    torch.nn.Linear(int(hidden_dim) + int(target_dim), int(hidden_dim)),
                    torch.nn.LayerNorm(int(hidden_dim)),
                    torch.nn.GELU(),
                    torch.nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
                    torch.nn.GELU(),
                )
                score_dim = int(hidden_dim) // 2
                self.progress_head = torch.nn.Linear(score_dim, 2)
                self.binary_head = torch.nn.Linear(score_dim, 2)
                self.value_head = torch.nn.Linear(score_dim, 1)
                self.next_state_head = (
                    torch.nn.Linear(score_dim, int(next_state_dim))
                    if int(next_state_dim) > 0
                    else None
                )
                self.logstd_min = float(logstd_min)
                self.logstd_max = float(logstd_max)

            def encode(self, x: Any) -> Any:
                return self.encoder(x)

            def distribution(self, z: Any) -> tuple[Any, Any]:
                mean = self.mean_head(z)
                raw = self.logstd_head(z)
                logstd = self.logstd_min + (self.logstd_max - self.logstd_min) * torch.sigmoid(raw)
                return mean, logstd

            def score_candidate(self, z: Any, candidate_resid_norm: Any) -> tuple[Any, Any, Any, Any]:
                h = self.scorer(torch.cat([z, candidate_resid_norm], dim=-1))
                next_state = self.next_state_head(h) if self.next_state_head is not None else None
                return self.progress_head(h), self.binary_head(h), self.value_head(h), next_state

            def forward(self, x: Any, candidate_resid_norm: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
                z = self.encode(x)
                mean, logstd = self.distribution(z)
                progress, binary, value, next_state = self.score_candidate(z, candidate_resid_norm)
                return mean, logstd, progress, binary, value, next_state

        return _Net()


class DiffusionCandidateExecutorNet:
    def __new__(
        cls,
        feature_dim: int,
        target_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        logstd_min: float,
        logstd_max: float,
        next_state_dim: int = 0,
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
                self.encoder = torch.nn.Sequential(*layers)
                self.mean_head = torch.nn.Linear(int(hidden_dim), int(target_dim))
                self.logstd_head = torch.nn.Linear(int(hidden_dim), int(target_dim))
                self.denoiser = torch.nn.Sequential(
                    torch.nn.Linear(int(hidden_dim) + int(target_dim) + 1, int(hidden_dim)),
                    torch.nn.LayerNorm(int(hidden_dim)),
                    torch.nn.GELU(),
                    torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
                    torch.nn.GELU(),
                    torch.nn.Linear(int(hidden_dim), int(target_dim)),
                )
                self.scorer = torch.nn.Sequential(
                    torch.nn.Linear(int(hidden_dim) + int(target_dim), int(hidden_dim)),
                    torch.nn.LayerNorm(int(hidden_dim)),
                    torch.nn.GELU(),
                    torch.nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
                    torch.nn.GELU(),
                )
                score_dim = int(hidden_dim) // 2
                self.progress_head = torch.nn.Linear(score_dim, 2)
                self.binary_head = torch.nn.Linear(score_dim, 2)
                self.value_head = torch.nn.Linear(score_dim, 1)
                self.next_state_head = (
                    torch.nn.Linear(score_dim, int(next_state_dim))
                    if int(next_state_dim) > 0
                    else None
                )
                self.logstd_min = float(logstd_min)
                self.logstd_max = float(logstd_max)

            def encode(self, x: Any) -> Any:
                return self.encoder(x)

            def distribution(self, z: Any) -> tuple[Any, Any]:
                mean = self.mean_head(z)
                raw = self.logstd_head(z)
                logstd = self.logstd_min + (self.logstd_max - self.logstd_min) * torch.sigmoid(raw)
                return mean, logstd

            def denoise(self, z: Any, noisy_resid_norm: Any, t_norm: Any) -> Any:
                if t_norm.ndim == 1:
                    t_norm = t_norm[:, None]
                return self.denoiser(torch.cat([z, noisy_resid_norm, t_norm], dim=-1))

            def score_candidate(self, z: Any, candidate_resid_norm: Any) -> tuple[Any, Any, Any, Any]:
                h = self.scorer(torch.cat([z, candidate_resid_norm], dim=-1))
                next_state = self.next_state_head(h) if self.next_state_head is not None else None
                return self.progress_head(h), self.binary_head(h), self.value_head(h), next_state

            def forward(self, x: Any, candidate_resid_norm: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
                z = self.encode(x)
                mean, logstd = self.distribution(z)
                progress, binary, value, next_state = self.score_candidate(z, candidate_resid_norm)
                return mean, logstd, progress, binary, value, next_state

        return _Net()


def diffusion_candidate_samples(
    *,
    module: Any,
    z: Any,
    sample_count: int,
    steps: int,
    beta_start: float,
    beta_end: float,
    generator: Any,
    device: Any,
) -> list[tuple[str, Any]]:
    import torch

    if int(sample_count) <= 0:
        return []
    betas, alphas, alpha_bars = diffusion_schedule(
        steps=int(steps),
        beta_start=float(beta_start),
        beta_end=float(beta_end),
        device=device,
    )
    batch = int(z.shape[0])
    target_dim = int(module.mean_head.out_features)
    samples: list[tuple[str, Any]] = []
    def randn(shape: tuple[int, ...]) -> Any:
        if generator is None:
            return torch.randn(shape, device=device)
        return torch.randn(shape, generator=generator, device=device)

    for sample_idx in range(int(sample_count)):
        x_t = randn((batch, target_dim))
        for step_idx in reversed(range(max(2, int(steps)))):
            t_norm = torch.full((batch, 1), float(step_idx) / max(1, int(steps) - 1), device=device)
            pred_noise = module.denoise(z, x_t, t_norm)
            beta_t = betas[step_idx]
            alpha_t = alphas[step_idx]
            alpha_bar_t = alpha_bars[step_idx]
            coef = beta_t / torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-6))
            mean = (x_t - coef * pred_noise) / torch.sqrt(torch.clamp(alpha_t, min=1e-6))
            if step_idx > 0:
                noise = randn(tuple(x_t.shape))
                x_t = mean + torch.sqrt(beta_t) * noise
            else:
                x_t = mean
        samples.append((f"diffusion_{sample_idx}", x_t))
    return samples


def candidate_eval(
    *,
    model: Any,
    x_norm: np.ndarray,
    y_raw: np.ndarray,
    prior_raw: np.ndarray,
    residual_norm: np.ndarray,
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
    progress_raw: np.ndarray,
    binary_raw: np.ndarray,
    next_state_raw: np.ndarray,
    indices: np.ndarray,
    used_rows: list[dict[str, Any]],
    phase_residual_l2_caps: dict[str, float],
    args: argparse.Namespace,
    device: Any,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    module = model.module if hasattr(model, "module") else model
    module.eval()
    temps = parse_float_list(str(args.candidate_temps))
    scale_values = parse_scale_list(str(args.candidate_scales))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + 991)
    with torch.no_grad():
        x = torch.from_numpy(x_norm[indices]).to(device)
        z = module.encode(x)
        mean_norm, logstd = module.distribution(z)
        std = torch.exp(logstd)
        target = torch.from_numpy(y_raw[indices]).to(device)
        prior = torch.from_numpy(prior_raw[indices]).to(device)
        resid_target = torch.from_numpy(residual_norm[indices]).to(device)
        progress = torch.from_numpy(progress_raw[indices]).to(device)
        binary = torch.from_numpy(binary_raw[indices]).to(device)
        next_state = torch.from_numpy(next_state_raw[indices]).to(device)
        next_state_axis = torch.from_numpy(
            fixed_width_vector(str(args.score_next_state_axis_weights), 3, 1.0)
        ).to(device).reshape(1, 3)
        next_state_target = torch.from_numpy(
            fixed_width_vector(str(args.score_next_state_target), 3, 0.0)
        ).to(device).reshape(1, 3)
        raw_mean = torch.from_numpy(residual_mean.astype(np.float32)).to(device)
        raw_std = torch.from_numpy(residual_std.astype(np.float32)).to(device)
        zero_resid_norm = (torch.zeros_like(mean_norm) - raw_mean) / raw_std
        cap_values = []
        default_cap = float(phase_residual_l2_caps.get("__global__", float(args.selector_residual_l2_cap_max)))
        for global_idx in indices:
            phase = str(used_rows[int(global_idx)].get("current_phase") or "unknown")
            cap_values.append(float(phase_residual_l2_caps.get(phase, default_cap)))
        cap_t = torch.tensor(cap_values, dtype=torch.float32, device=device).reshape(-1, 1)

        def score_candidates(candidates: list[tuple[str, Any]]) -> tuple[Any, list[str], Any, int, int]:
            scores: list[Any] = []
            raw_resids: list[Any] = []
            names: list[str] = []
            over_cap_count = 0
            for name, cand_norm in candidates:
                pred_progress, pred_logits, pred_value, pred_next_state = module.score_candidate(z, cand_norm)
                probs = torch.sigmoid(pred_logits)
                raw_resid = cand_norm * raw_std + raw_mean
                penalty = torch.mean(raw_resid**2, dim=1, keepdim=True)
                if pred_next_state is None:
                    next_state_penalty = torch.zeros_like(penalty)
                else:
                    next_state_penalty = torch.mean(
                        ((pred_next_state - next_state_target) * next_state_axis) ** 2,
                        dim=1,
                        keepdim=True,
                    )
                logprob = -0.5 * torch.mean(((cand_norm - mean_norm) / std) ** 2 + 2.0 * logstd, dim=1, keepdim=True)
                score = (
                    pred_progress[:, 1:2]
                    + float(args.score_inserted_weight) * probs[:, 0:1]
                    + float(args.score_dp_continuable_weight) * probs[:, 1:2]
                    + float(args.score_value_weight) * pred_value
                    + float(args.score_logprob_weight) * logprob
                    - float(args.score_residual_l2_penalty) * penalty
                    - float(args.score_next_state_weight) * next_state_penalty
                )
                if name == "mean":
                    score = score - float(args.score_mean_source_penalty)
                if name.startswith("scale_"):
                    score = score - float(args.score_scale_source_penalty)
                    try:
                        scale_value = float(name.split("_", 1)[1])
                    except ValueError:
                        scale_value = 0.0
                    if scale_value >= 0.5:
                        score = score - float(args.score_large_scale_source_penalty)
                if name.startswith("sample_") or name.startswith("diffusion_"):
                    score = score - float(args.score_stochastic_source_penalty)
                if name != "dp_prior":
                    over_cap = penalty > cap_t
                    over_cap_count += int(over_cap.detach().sum().cpu().item())
                    score = torch.where(over_cap, torch.full_like(score, -1.0e9), score)
                scores.append(score)
                raw_resids.append(raw_resid)
                names.append(name)
            score_t = torch.cat(scores, dim=1)
            raw_resid_t = torch.stack(raw_resids, dim=1)
            best_idx = torch.argmax(score_t, dim=1)
            fallback_phases = {item.strip() for item in str(args.dp_fallback_phases).split(",") if item.strip()}
            fallback_all = "all" in fallback_phases or "*" in fallback_phases
            fallback_count = 0
            if fallback_phases:
                best_cpu = best_idx.detach().cpu().numpy()
                score_cpu = score_t.detach().cpu().numpy()
                for local_i, global_idx in enumerate(indices):
                    phase = str(used_rows[int(global_idx)].get("current_phase") or "")
                    if not fallback_all and phase not in fallback_phases:
                        continue
                    chosen = int(best_cpu[local_i])
                    if chosen == 0:
                        continue
                    dp_score = float(score_cpu[local_i, 0])
                    chosen_score = float(score_cpu[local_i, chosen])
                    if chosen_score < dp_score + float(args.dp_fallback_score_margin):
                        best_idx[local_i] = 0
                        fallback_count += 1
            selected = raw_resid_t[torch.arange(raw_resid_t.shape[0], device=device), best_idx]
            return selected, names, best_idx, fallback_count, over_cap_count

        candidates: list[tuple[str, Any]] = [("dp_prior", zero_resid_norm), ("mean", mean_norm)]
        for scale in scale_values:
            raw = float(scale) * (mean_norm * raw_std + raw_mean)
            candidates.append((f"scale_{scale:g}", (raw - raw_mean) / raw_std))
        if str(getattr(args, "generator_type", "gaussian")) == "diffusion":
            candidates.extend(
                diffusion_candidate_samples(
                    module=module,
                    z=z,
                    sample_count=int(args.candidate_samples),
                    steps=int(args.diffusion_steps),
                    beta_start=float(args.diffusion_beta_start),
                    beta_end=float(args.diffusion_beta_end),
                    generator=generator,
                    device=device,
                )
            )
        elif int(args.candidate_samples) > 0:
            samples_per_temp = max(1, int(args.candidate_samples) // max(1, len(temps)))
            for temp in temps:
                for sample_idx in range(samples_per_temp):
                    noise = torch.randn(mean_norm.shape, generator=generator, device=device)
                    candidates.append((f"sample_t{temp:g}_{sample_idx}", mean_norm + float(temp) * std * noise))

        selected_resid, names, best_idx, fallback_count, over_cap_count = score_candidates(candidates)
        selected_action = prior + selected_resid
        mean_action = prior + mean_norm * raw_std + raw_mean
        dp_action = prior
        pred_progress, pred_logits, pred_value, pred_next_state = module.score_candidate(z, resid_target)
        pred_binary = torch.sigmoid(pred_logits) >= 0.5
        target_binary = binary >= 0.5
        value = torch.from_numpy(value_target(progress_raw[indices], binary_raw[indices])).to(device)
        teacher_next_state_mse = (
            float(torch.mean((pred_next_state - next_state) ** 2).item())
            if pred_next_state is not None
            else None
        )

        source_counts: dict[str, int] = {}
        best_cpu = best_idx.detach().cpu().numpy()
        for item in best_cpu:
            source_counts[names[int(item)]] = source_counts.get(names[int(item)], 0) + 1
        non_dp_selected_count = sum(count for name, count in source_counts.items() if name != "dp_prior")
        diffusion_selected_count = sum(
            count for name, count in source_counts.items() if str(name).startswith("diffusion_")
        )

        out = {
            "num_rows": int(len(indices)),
            "dp_prior_action_mse": float(torch.mean((dp_action - target) ** 2).item()),
            "mean_action_mse": float(torch.mean((mean_action - target) ** 2).item()),
            "selected_action_mse": float(torch.mean((selected_action - target) ** 2).item()),
            "selected_minus_dp_prior_mse": float(
                torch.mean((selected_action - target) ** 2).item() - torch.mean((dp_action - target) ** 2).item()
            ),
            "teacher_progress_mse": float(torch.mean((pred_progress - progress) ** 2).item()),
            "teacher_value_mse": float(torch.mean((pred_value - value) ** 2).item()),
            "teacher_next_state_mse": teacher_next_state_mse,
            "teacher_inserted_acc": float((pred_binary[:, 0] == target_binary[:, 0]).float().mean().item()),
            "teacher_dp_continuable_acc": float((pred_binary[:, 1] == target_binary[:, 1]).float().mean().item()),
            "candidate_source_counts": dict(sorted(source_counts.items())),
            "selected_non_dp_candidate_count": int(non_dp_selected_count),
            "selected_non_dp_candidate_fraction": float(non_dp_selected_count / max(1, len(indices))),
            "selected_diffusion_candidate_count": int(diffusion_selected_count),
            "dp_fallback_count": int(fallback_count),
            "dp_fallback_phases": sorted(
                [item.strip() for item in str(args.dp_fallback_phases).split(",") if item.strip()]
            ),
            "dp_fallback_score_margin": float(args.dp_fallback_score_margin),
            "num_candidate_sources": int(len(names)),
            "generator_type": str(args.generator_type),
            "diffusion_steps": int(args.diffusion_steps) if str(args.generator_type) == "diffusion" else 0,
            "candidate_scales": [float(item) for item in scale_values],
            "selector_residual_l2_cap_quantile": float(args.selector_residual_l2_cap_quantile),
            "selector_residual_l2_cap_min": float(args.selector_residual_l2_cap_min),
            "selector_residual_l2_cap_max": float(args.selector_residual_l2_cap_max),
            "selector_residual_l2_cap_multiplier": float(args.selector_residual_l2_cap_multiplier),
            "score_mean_source_penalty": float(args.score_mean_source_penalty),
            "score_scale_source_penalty": float(args.score_scale_source_penalty),
            "score_large_scale_source_penalty": float(args.score_large_scale_source_penalty),
            "score_stochastic_source_penalty": float(args.score_stochastic_source_penalty),
            "score_next_state_weight": float(args.score_next_state_weight),
            "score_next_state_axis_weights": fixed_width_vector(
                str(args.score_next_state_axis_weights), 3, 1.0
            ).astype(float).tolist(),
            "score_next_state_target": fixed_width_vector(
                str(args.score_next_state_target), 3, 0.0
            ).astype(float).tolist(),
            "selector_over_cap_candidate_count": int(over_cap_count),
            "selector_phase_residual_l2_caps": dict(sorted(phase_residual_l2_caps.items())),
            "nll_residual_mse": float(F.mse_loss(mean_norm, resid_target).item()),
            "boundary": (
                "Offline candidate-selection diagnostic. Selected candidates are "
                "not live-controller evidence until executed in the simulator "
                "with video/final-state review."
            ),
        }
    module.train()
    return out


def candidate_rank_loss_for_batch(
    *,
    module: Any,
    z: Any,
    mean_norm: Any,
    logstd: Any,
    resid_target_norm: Any,
    raw_mean: Any,
    raw_std: Any,
    next_state_target: Any,
    next_state_axis: Any,
    args: argparse.Namespace,
) -> Any:
    import torch
    import torch.nn.functional as F

    scale_values = parse_scale_list(str(args.candidate_scales))
    std = torch.exp(logstd)
    zero_resid_norm = (torch.zeros_like(mean_norm) - raw_mean) / raw_std
    mean_detached = mean_norm.detach()
    candidates: list[tuple[str, Any]] = [("dp_prior", zero_resid_norm), ("mean", mean_detached)]
    mean_raw = mean_detached * raw_std + raw_mean
    for scale in scale_values:
        raw = float(scale) * mean_raw
        candidates.append((f"scale_{scale:g}", (raw - raw_mean) / raw_std))
    random_count = max(0, int(args.candidate_rank_random_count))
    for sample_idx in range(random_count):
        noise = torch.randn_like(mean_detached)
        candidates.append((f"rank_random_{sample_idx}", mean_detached + std.detach() * noise))
    diffusion_rank_count = max(0, int(getattr(args, "candidate_rank_diffusion_count", 0)))
    if str(getattr(args, "generator_type", "gaussian")) == "diffusion" and diffusion_rank_count > 0:
        with torch.no_grad():
            diffusion_candidates = diffusion_candidate_samples(
                module=module,
                z=z.detach(),
                sample_count=diffusion_rank_count,
                steps=int(args.diffusion_steps),
                beta_start=float(args.diffusion_beta_start),
                beta_end=float(args.diffusion_beta_end),
                generator=None,
                device=z.device,
            )
        for name, cand_norm in diffusion_candidates:
            candidates.append((f"rank_{name}", cand_norm.detach()))

    score_columns: list[Any] = []
    raw_resids: list[Any] = []
    for name, cand_norm in candidates:
        pred_progress, pred_logits, pred_value, pred_next_state = module.score_candidate(z, cand_norm)
        probs = torch.sigmoid(pred_logits)
        raw_resid = cand_norm * raw_std + raw_mean
        penalty = torch.mean(raw_resid**2, dim=1, keepdim=True)
        if pred_next_state is None:
            next_state_penalty = torch.zeros_like(penalty)
        else:
            next_state_penalty = torch.mean(
                ((pred_next_state - next_state_target) * next_state_axis) ** 2,
                dim=1,
                keepdim=True,
            )
        logprob = -0.5 * torch.mean(((cand_norm - mean_norm.detach()) / std.detach()) ** 2 + 2.0 * logstd.detach(), dim=1, keepdim=True)
        score = (
            pred_progress[:, 1:2]
            + float(args.score_inserted_weight) * probs[:, 0:1]
            + float(args.score_dp_continuable_weight) * probs[:, 1:2]
            + float(args.score_value_weight) * pred_value
            + float(args.score_logprob_weight) * logprob
            - float(args.score_residual_l2_penalty) * penalty
            - float(args.score_next_state_weight) * next_state_penalty
        )
        if name == "mean":
            score = score - float(args.score_mean_source_penalty)
        if name.startswith("scale_"):
            score = score - float(args.score_scale_source_penalty)
            try:
                scale_value = float(name.split("_", 1)[1])
            except ValueError:
                scale_value = 0.0
            if scale_value >= 0.5:
                score = score - float(args.score_large_scale_source_penalty)
        if name.startswith("rank_random_") or name.startswith("rank_diffusion_"):
            score = score - float(args.score_stochastic_source_penalty)
        score_columns.append(score)
        raw_resids.append(raw_resid)

    score_t = torch.cat(score_columns, dim=1)
    raw_resid_t = torch.stack(raw_resids, dim=1)
    target_raw = resid_target_norm * raw_std + raw_mean
    oracle_mse = torch.mean((raw_resid_t - target_raw[:, None, :]) ** 2, dim=2)
    oracle_best = torch.argmin(oracle_mse.detach(), dim=1)
    temperature = max(1e-6, float(args.candidate_rank_temperature))
    return F.cross_entropy(score_t / temperature, oracle_best)


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
    next_state_raw = future_task_state_targets(used_rows)
    next_state_axis_t = torch.from_numpy(
        fixed_width_vector(str(args.score_next_state_axis_weights), 3, 1.0)
    ).to(device).reshape(1, 3)
    next_state_target_t = torch.from_numpy(
        fixed_width_vector(str(args.score_next_state_target), 3, 0.0)
    ).to(device).reshape(1, 3)
    phase_residual_l2_caps = compute_phase_residual_l2_caps(
        used_rows=used_rows,
        train_idx=train_idx,
        residual_raw=residual_raw,
        args=args,
    )
    residual_mean_t = torch.from_numpy(residual_mean.astype(np.float32)).to(device)
    residual_std_t = torch.from_numpy(residual_std.astype(np.float32)).to(device)

    train_ds = TensorDataset(
        torch.from_numpy(x_norm[train_idx]),
        torch.from_numpy(residual_norm[train_idx].astype(np.float32)),
        torch.from_numpy(progress_raw[train_idx].astype(np.float32)),
        torch.from_numpy(binary_raw[train_idx].astype(np.float32)),
        torch.from_numpy(value_raw[train_idx].astype(np.float32)),
        torch.from_numpy(next_state_raw[train_idx].astype(np.float32)),
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

    model_cls = DiffusionCandidateExecutorNet if str(args.generator_type) == "diffusion" else CandidateExecutorNet
    model = model_cls(
        feature_dim=int(x_norm.shape[1]),
        target_dim=int(y_raw.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        logstd_min=float(args.logstd_min),
        logstd_max=float(args.logstd_max),
        next_state_dim=int(next_state_raw.shape[1]),
    ).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    bce = torch.nn.BCEWithLogitsLoss()

    if is_rank0:
        write_json(
            output_root / "training_manifest.json",
            {
                "schema": "cosmos3_candidate_executor_training_v1",
                "contact_executor_jsonl": str(Path(args.contact_executor_jsonl).resolve()),
                "output_root": str(output_root),
                "num_samples": int(len(used_rows)),
                "num_train": int(len(train_idx)),
                "num_val": int(len(val_idx)),
                "feature_dim": int(x_norm.shape[1]),
                "target_dim": int(y_raw.shape[1]),
                "next_state_dim": int(next_state_raw.shape[1]),
                "action_horizon": int(y_raw.shape[1] // 7),
                "selector_phase_residual_l2_caps": dict(sorted(phase_residual_l2_caps.items())),
                "candidate_scales": parse_scale_list(str(args.candidate_scales)),
                "generator_type": str(args.generator_type),
                "diffusion_steps": int(args.diffusion_steps),
                "diffusion_beta_start": float(args.diffusion_beta_start),
                "diffusion_beta_end": float(args.diffusion_beta_end),
                "candidate_rank_loss_weight": float(args.candidate_rank_loss_weight),
                "candidate_rank_random_count": int(args.candidate_rank_random_count),
                "candidate_rank_diffusion_count": int(args.candidate_rank_diffusion_count),
                "candidate_rank_temperature": float(args.candidate_rank_temperature),
                "next_state_loss_weight": float(args.next_state_loss_weight),
                "score_next_state_weight": float(args.score_next_state_weight),
                "score_next_state_axis_weights": fixed_width_vector(
                    str(args.score_next_state_axis_weights), 3, 1.0
                ).astype(float).tolist(),
                "score_next_state_target": fixed_width_vector(
                    str(args.score_next_state_target), 3, 0.0
                ).astype(float).tolist(),
                "device": str(device),
                "world_size": int(world_size),
                "formal_min_gpus": int(args.formal_min_gpus),
                "min_wall_seconds": int(args.min_wall_seconds),
                "max_wall_seconds": int(args.max_wall_seconds),
                "offline_gate_thresholds": dict(OFFLINE_GATE_THRESHOLDS),
                "method_boundary": (
                    "Candidate executor path: Cosmos-predicted task/contact path "
                    "conditions a stochastic residual generator; an action-conditioned "
                    "progress/contact/value scorer selects a short action chunk. "
                    "Live evidence still requires real simulator execution and video review."
                ),
                "sample_rows": used_rows[:10],
            },
        )

    history: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] | None = None
    best_selected_action_mse = float("inf")
    best_step = 0
    best_gate_metrics: dict[str, Any] | None = None
    best_gate_selected_action_mse = float("inf")
    best_gate_step = 0
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
            x_b, resid_b, progress_b, binary_b, value_b, next_state_b = [
                item.to(device, non_blocking=True) for item in batch
            ]
            mean_norm, logstd, pred_progress, pred_logits, pred_value, pred_next_state = model(x_b, resid_b)
            std = torch.exp(logstd)
            nll = 0.5 * torch.mean(((resid_b - mean_norm) / std) ** 2 + 2.0 * logstd)
            mean_loss = torch.mean((mean_norm - resid_b) ** 2)
            diffusion_loss = torch.zeros((), dtype=resid_b.dtype, device=device)
            if str(args.generator_type) == "diffusion":
                module = model.module if hasattr(model, "module") else model
                z_diff = module.encode(x_b)
                _, _, alpha_bars = diffusion_schedule(
                    steps=int(args.diffusion_steps),
                    beta_start=float(args.diffusion_beta_start),
                    beta_end=float(args.diffusion_beta_end),
                    device=device,
                )
                step_count = max(2, int(args.diffusion_steps))
                t_idx = torch.randint(0, step_count, (resid_b.shape[0],), device=device)
                noise = torch.randn_like(resid_b)
                alpha_bar_t = alpha_bars[t_idx].reshape(-1, 1)
                noisy = torch.sqrt(alpha_bar_t) * resid_b + torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-6)) * noise
                t_norm = t_idx.to(dtype=resid_b.dtype).reshape(-1, 1) / float(max(1, step_count - 1))
                pred_noise = module.denoise(z_diff, noisy, t_norm)
                diffusion_loss = torch.mean((pred_noise - noise) ** 2)
            progress_loss = torch.mean((pred_progress - progress_b) ** 2)
            binary_loss = bce(pred_logits, binary_b)
            value_loss = torch.mean((pred_value - value_b) ** 2)
            next_state_loss = (
                torch.mean((pred_next_state - next_state_b) ** 2)
                if pred_next_state is not None
                else torch.zeros((), dtype=resid_b.dtype, device=device)
            )
            module = model.module if hasattr(model, "module") else model
            z_rank = module.encode(x_b)
            rank_loss = candidate_rank_loss_for_batch(
                module=module,
                z=z_rank,
                mean_norm=mean_norm,
                logstd=logstd,
                resid_target_norm=resid_b,
                raw_mean=residual_mean_t,
                raw_std=residual_std_t,
                next_state_target=next_state_target_t,
                next_state_axis=next_state_axis_t,
                args=args,
            )
            loss = (
                float(args.nll_loss_weight) * nll
                + float(args.mean_action_loss_weight) * mean_loss
                + float(args.diffusion_loss_weight) * diffusion_loss
                + float(args.progress_loss_weight) * progress_loss
                + float(args.binary_loss_weight) * binary_loss
                + float(args.value_loss_weight) * value_loss
                + float(args.next_state_loss_weight) * next_state_loss
                + float(args.candidate_rank_loss_weight) * rank_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step == 1 or step % int(args.eval_every_steps) == 0:
                if is_rank0:
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
                        "train_batch_diffusion_noise_mse": float(diffusion_loss.detach().cpu().item()),
                        "train_batch_progress_mse": float(progress_loss.detach().cpu().item()),
                        "train_batch_binary_bce": float(binary_loss.detach().cpu().item()),
                        "train_batch_value_mse": float(value_loss.detach().cpu().item()),
                        "train_batch_next_state_mse": float(next_state_loss.detach().cpu().item()),
                        "train_batch_candidate_rank_ce": float(rank_loss.detach().cpu().item()),
                        "train": train_metrics,
                        "eval": eval_metrics,
                    }
                    history.append(metrics)
                    write_json(output_root / "training_history.json", history)
                    eval_metrics = dict(metrics.get("eval") or {})
                    selected_mse = float(eval_metrics.get("selected_action_mse", float("inf")))
                    if np.isfinite(selected_mse) and selected_mse < best_selected_action_mse:
                        best_selected_action_mse = selected_mse
                        best_metrics = metrics
                        best_step = int(step)
                        module = model.module if hasattr(model, "module") else model
                        atomic_torch_save(
                            {
                                "model_state_dict": module.state_dict(),
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
                                "best_selection_metric": "eval.selected_action_mse",
                            },
                            output_root / "checkpoint_best_offline.pt",
                        )
                    if offline_gate_from_eval(eval_metrics) and np.isfinite(selected_mse):
                        if selected_mse < best_gate_selected_action_mse:
                            best_gate_selected_action_mse = selected_mse
                            best_gate_metrics = metrics
                            best_gate_step = int(step)
                            module = model.module if hasattr(model, "module") else model
                            atomic_torch_save(
                                {
                                    "model_state_dict": module.state_dict(),
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
                                    "best_selection_metric": "eval.selected_action_mse_after_offline_gate",
                                },
                                output_root / "checkpoint_best_gate.pt",
                            )
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
        final_path = output_root / "checkpoint_final.pt"
        atomic_torch_save(
            {
                "model_state_dict": module.state_dict(),
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
        torch.load(final_path, map_location="cpu", weights_only=False)
        best_path = output_root / "checkpoint_best_offline.pt"
        if not best_path.is_file():
            best_metrics = final_metrics
            best_step = int(step)
            atomic_torch_save(
                {
                    "model_state_dict": module.state_dict(),
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
                    "best_selection_metric": "eval.selected_action_mse",
                },
                best_path,
            )
        torch.load(best_path, map_location="cpu", weights_only=False)
        best_gate_path = output_root / "checkpoint_best_gate.pt"
        if best_gate_path.is_file():
            torch.load(best_gate_path, map_location="cpu", weights_only=False)
        elapsed = float(time.time() - start)
        formal_floor_met = bool(
            world_size >= int(args.formal_min_gpus)
            and int(args.min_wall_seconds) >= 10800
            and elapsed >= int(args.min_wall_seconds)
        )
        eval_metrics = dict(final_metrics.get("eval") or {})
        ready_for_offline_gate = offline_gate_from_eval(eval_metrics)
        best_metrics_for_summary = best_metrics if best_metrics is not None else final_metrics
        best_eval_metrics = dict((best_metrics_for_summary or {}).get("eval") or {})
        best_ready_for_offline_gate = offline_gate_from_eval(best_eval_metrics)
        best_gate_eval_metrics = dict((best_gate_metrics or {}).get("eval") or {})
        best_gate_ready_for_offline_gate = offline_gate_from_eval(best_gate_eval_metrics)
        formal_live_eval_checkpoint = None
        formal_live_eval_metrics_source = None
        if formal_floor_met and ready_for_offline_gate:
            formal_live_eval_checkpoint = str(final_path)
            formal_live_eval_metrics_source = "final"
        elif formal_floor_met and best_gate_ready_for_offline_gate and best_gate_path.is_file():
            formal_live_eval_checkpoint = str(best_gate_path)
            formal_live_eval_metrics_source = "best_gate"
        summary = {
            "schema": "cosmos3_candidate_executor_training_summary_v1",
            "contact_executor_jsonl": str(Path(args.contact_executor_jsonl).resolve()),
            "output_root": str(output_root),
            "num_samples": int(len(used_rows)),
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "world_size": int(world_size),
            "steps": int(step),
            "elapsed_seconds": elapsed,
            "stop_reason": stop_reason,
            "feature_dim": int(x_norm.shape[1]),
            "target_dim": int(y_raw.shape[1]),
            "next_state_dim": int(next_state_raw.shape[1]),
            "action_horizon": int(y_raw.shape[1] // 7),
            "final_metrics": final_metrics,
            "best_offline_metrics": best_metrics_for_summary,
            "best_offline_step": int(best_step),
            "best_offline_checkpoint": str(best_path),
            "best_offline_selection_metric": "eval.selected_action_mse",
            "best_ready_for_offline_gate": best_ready_for_offline_gate,
            "best_gate_metrics": best_gate_metrics,
            "best_gate_step": int(best_gate_step),
            "best_gate_checkpoint": str(best_gate_path) if best_gate_path.is_file() else None,
            "best_gate_selection_metric": "eval.selected_action_mse_after_offline_gate",
            "best_gate_ready_for_offline_gate": best_gate_ready_for_offline_gate,
            "ready_for_offline_gate": ready_for_offline_gate,
            "ready_for_formal_live_eval": bool(formal_live_eval_checkpoint),
            "formal_live_eval_checkpoint": formal_live_eval_checkpoint,
            "formal_live_eval_metrics_source": formal_live_eval_metrics_source,
            "formal_training_floor_met": formal_floor_met,
            "offline_gate_thresholds": dict(OFFLINE_GATE_THRESHOLDS),
            "generator_type": str(args.generator_type),
            "diffusion_steps": int(args.diffusion_steps) if str(args.generator_type) == "diffusion" else 0,
            "candidate_samples": int(args.candidate_samples),
            "candidate_scales": parse_scale_list(str(args.candidate_scales)),
            "candidate_rank_loss_weight": float(args.candidate_rank_loss_weight),
            "candidate_rank_random_count": int(args.candidate_rank_random_count),
            "candidate_rank_diffusion_count": int(args.candidate_rank_diffusion_count),
            "candidate_rank_temperature": float(args.candidate_rank_temperature),
            "next_state_loss_weight": float(args.next_state_loss_weight),
            "score_next_state_weight": float(args.score_next_state_weight),
            "score_next_state_axis_weights": fixed_width_vector(
                str(args.score_next_state_axis_weights), 3, 1.0
            ).astype(float).tolist(),
            "score_next_state_target": fixed_width_vector(
                str(args.score_next_state_target), 3, 0.0
            ).astype(float).tolist(),
            "boundary": (
                "Candidate executor training evidence only. Live method evidence "
                "requires held-out offline gate, then real closed-loop final-state "
                "metrics plus inspected videos/contact sheets. The formal run must "
                "meet the GPU and wall-clock floor; after that floor is met, live "
                "eval may use the final checkpoint if it passes the gate or the "
                "best held-out gate-passing checkpoint from the same formal run."
            ),
        }
        write_json(output_root / "training_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    distributed_barrier(world_size)
    cleanup_distributed(world_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
