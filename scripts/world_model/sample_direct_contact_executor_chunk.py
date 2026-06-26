#!/usr/bin/env python3
"""Sample direct-contact executor chunks for saved-snapshot replay."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
from typing import Any

import numpy as np


PHASE_NAMES = ["lost_grasp", "far", "lateral_align", "preinsert_aligned", "dp_continuable", "inserted"]


def require_slurm_compute_step() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit(
            "Refusing to sample outside a Slurm allocation. "
            f"host={socket.gethostname()}. Use tmux-held allocation + srun."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--replay-label-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--requested-offsets", default="24,16,8")
    parser.add_argument("--samples-per-offset", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampling-mode", choices=("x0_mid", "ddim"), default="x0_mid")
    parser.add_argument("--force-query-grasped", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260624)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def parse_ints(text: str) -> list[int]:
    out: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            value = int(item)
            if value not in out:
                out.append(value)
    return out


def phase_one_hot(value: int) -> np.ndarray:
    out = np.zeros((len(PHASE_NAMES) + 1,), dtype=np.float32)
    if 0 <= int(value) < len(PHASE_NAMES):
        out[int(value)] = 1.0
    else:
        out[-1] = 1.0
    return out


def infer_phase(head: np.ndarray, grasped: bool) -> int:
    if not bool(grasped):
        return 0
    x, y, z = [float(v) for v in head[:3]]
    lateral = math.sqrt(y * y + z * z)
    if x >= -0.015 and abs(y) <= 0.012 and abs(z) <= 0.008:
        return 5
    if -0.134 <= x <= 0.04 and abs(y) <= 0.025 and abs(z) <= 0.025:
        return 4
    if x >= -0.25 and abs(y) <= 0.05 and abs(z) <= 0.025:
        return 3
    if lateral <= 0.05:
        return 2
    return 1


def contact_progress_proxy(head: np.ndarray, grasped: bool) -> float:
    x, y, z = [float(v) for v in head[:3]]
    lateral = math.sqrt(y * y + z * z)
    insertion = max(0.0, min(1.0, (x - (-0.25)) / max((-0.015) - (-0.25), 1e-6)))
    lateral_score = max(0.0, min(1.0, 1.0 - lateral / 0.05))
    return float(0.45 * insertion + 0.45 * lateral_score + 0.10 * float(bool(grasped)))


def row_features(head: np.ndarray, *, prefix_frame: int, offset: int, grasped: bool, horizon: int) -> np.ndarray:
    phase_id = infer_phase(head, grasped)
    abs_y = abs(float(head[1]))
    abs_z = abs(float(head[2]))
    yz_l2 = math.sqrt(float(head[1]) ** 2 + float(head[2]) ** 2)
    vals = np.asarray(
        [
            float(head[0]),
            float(head[1]),
            float(head[2]),
            abs_y,
            abs_z,
            abs_y + abs_z,
            yz_l2,
            contact_progress_proxy(head, grasped),
            float(bool(grasped)),
            float(phase_id == 4),
            float(offset) / 96.0,
            float(prefix_frame) / 300.0,
            (300.0 - float(prefix_frame)) / 300.0,
            float(horizon) / float(horizon),
        ],
        dtype=np.float32,
    )
    return np.concatenate([vals, phase_one_hot(phase_id)]).astype(np.float32)


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


def make_model(input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> Any:
    import torch

    layers: list[Any] = []
    dim = int(input_dim)
    for _ in range(int(num_layers)):
        layers.append(torch.nn.Linear(dim, int(hidden_dim)))
        layers.append(torch.nn.SiLU())
        if float(dropout) > 0:
            layers.append(torch.nn.Dropout(float(dropout)))
        dim = int(hidden_dim)
    layers.append(torch.nn.Linear(dim, int(output_dim)))
    return torch.nn.Sequential(*layers)


def sample_action(
    model: Any,
    feature_z: np.ndarray,
    ckpt: dict[str, Any],
    temperature: float,
    seed: int,
    *,
    sampling_mode: str,
    clip_action: bool,
) -> np.ndarray:
    import torch

    args = ckpt["args"]
    metadata = ckpt["metadata"]
    horizon = int(metadata["horizon"])
    target_dim = int(metadata["target_dim"])
    steps = max(2, int(args.get("diffusion_steps", 32)))
    beta_start = float(args.get("beta_start", args.get("diffusion_beta_start", 1e-4)))
    beta_end = float(args.get("beta_end", args.get("diffusion_beta_end", 2e-2)))
    device = next(model.parameters()).device
    torch.manual_seed(int(seed))
    betas = torch.linspace(beta_start, beta_end, steps, device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    feat = torch.from_numpy(feature_z.reshape(1, -1)).float().to(device)
    if str(sampling_mode) == "x0_mid":
        t_i = int(steps) // 2
        t = torch.full((1,), int(t_i), device=device, dtype=torch.long)
        x = torch.randn((1, target_dim), device=device, dtype=torch.float32) * float(temperature)
        pred_noise = model(torch.cat([feat, x, timestep_embedding(t, 64)], dim=1))
        a_bar = alphas_cumprod[t_i]
        x0 = (x - torch.sqrt(1.0 - a_bar) * pred_noise) / torch.sqrt(a_bar)
        x = x0
    else:
        x = torch.randn((1, target_dim), device=device, dtype=torch.float32) * float(temperature)
        for t_i in reversed(range(steps)):
            t = torch.full((1,), int(t_i), device=device, dtype=torch.long)
            pred_noise = model(torch.cat([feat, x, timestep_embedding(t, 64)], dim=1))
            a_bar = alphas_cumprod[t_i]
            x0 = (x - torch.sqrt(1.0 - a_bar) * pred_noise) / torch.sqrt(a_bar)
            if t_i > 0:
                prev_a_bar = alphas_cumprod[t_i - 1]
                x = torch.sqrt(prev_a_bar) * x0 + torch.sqrt(1.0 - prev_a_bar) * pred_noise
            else:
                x = x0
    target_mean = np.asarray(ckpt["target_mean"], dtype=np.float32).reshape(1, -1)
    target_std = np.asarray(ckpt["target_std"], dtype=np.float32).reshape(1, -1)
    arr = x.detach().cpu().numpy() * target_std + target_mean
    action = arr.reshape(horizon, 7).astype(np.float32)
    if bool(clip_action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
    return action


def main() -> int:
    require_slurm_compute_step()
    args = parse_args()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for direct-contact sampling")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint).resolve()
    label_path = Path(args.replay_label_json).resolve()
    ckpt = torch.load(str(ckpt_path), map_location="cuda")
    metadata = ckpt["metadata"]
    model_args = ckpt["args"]
    feature_dim = int(metadata["feature_dim"])
    target_dim = int(metadata["target_dim"])
    model = make_model(
        feature_dim + target_dim + 64,
        target_dim,
        int(model_args.get("hidden_dim", 2048)),
        int(model_args.get("num_layers", 5)),
        float(model_args.get("dropout", 0.0)),
    ).cuda()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    label = read_json(label_path)
    head = np.asarray(label.get("before_peg_head_at_hole"), dtype=np.float32).reshape(3)
    prefix_frame = int(label.get("prefix_frame_index") or (label.get("candidate_meta") or {}).get("snapshot_attrs", {}).get("prefix_frame_index") or 0)
    before_gate = label.get("before_continuability_gate") if isinstance(label.get("before_continuability_gate"), dict) else {}
    checks = before_gate.get("checks") if isinstance(before_gate.get("checks"), dict) else {}
    grasped = bool(checks.get("grasped", False))
    if bool(args.force_query_grasped):
        grasped = True
    horizon = int(metadata["horizon"])
    feature_mean = np.asarray(ckpt["feature_mean"], dtype=np.float32).reshape(1, -1)
    feature_std = np.asarray(ckpt["feature_std"], dtype=np.float32).reshape(1, -1)
    rows: list[dict[str, Any]] = []
    for offset in parse_ints(str(args.requested_offsets)):
        for sample_i in range(max(1, int(args.samples_per_offset))):
            feature = row_features(head, prefix_frame=prefix_frame, offset=int(offset), grasped=grasped, horizon=horizon)
            feature_z = ((feature.reshape(1, -1) - feature_mean) / feature_std).astype(np.float32)
            action = sample_action(
                model,
                feature_z.reshape(-1),
                ckpt,
                float(args.temperature),
                int(args.seed) + 1000 * int(offset) + int(sample_i),
                sampling_mode=str(args.sampling_mode),
                clip_action=bool(args.clip_action),
            )
            name = f"direct_contact_diffusion_o{int(offset)}_s{int(sample_i)}"
            chunk = {
                "schema": "direct_contact_executor_action_chunk_v1",
                "ok": bool(np.isfinite(action).all()),
                "candidate_name": name,
                "checkpoint": str(ckpt_path),
                "source_replay_label_json": str(label_path),
                "prefix_frame_index": int(prefix_frame),
                "chunk_start": int(prefix_frame),
                "chunk_end_exclusive": int(prefix_frame + horizon),
                "chunk_steps": int(horizon),
                "robot_action_dim": 7,
                "requested_offset_before_insert": int(offset),
                "sample_index": int(sample_i),
                "temperature": float(args.temperature),
                "sampling_mode": str(args.sampling_mode),
                "clip_action": bool(args.clip_action),
                "query_peg_head_at_hole": head.astype(float).tolist(),
                "query_grasped": bool(grasped),
                "denormalized_robot_action_chunk": action.astype(float).tolist(),
                "denormalized_robot_action_stats": {
                    "finite": bool(np.isfinite(action).all()),
                    "min": float(np.min(action)),
                    "max": float(np.max(action)),
                    "max_abs": float(np.max(np.abs(action))),
                    "mean_abs": float(np.mean(np.abs(action))),
                    "num_values": int(action.size),
                },
                "boundary": (
                    "Direct-contact executor sampled chunk for saved-snapshot replay. "
                    "It uses only current replay-label state as query context; replay "
                    "from simulator snapshot is required before any claim."
                ),
            }
            path = output_root / f"{name}.action_chunk.json"
            write_json(path, chunk)
            rows.append({"candidate_name": name, "action_chunk_json": str(path), "stats": chunk["denormalized_robot_action_stats"]})
    summary = {
        "schema": "direct_contact_executor_sampling_summary_v1",
        "checkpoint": str(ckpt_path),
        "source_replay_label_json": str(label_path),
        "output_root": str(output_root),
        "prefix_frame_index": int(prefix_frame),
        "query_peg_head_at_hole": head.astype(float).tolist(),
        "query_grasped": bool(grasped),
        "sampling_mode": str(args.sampling_mode),
        "clip_action": bool(args.clip_action),
        "num_chunks": len(rows),
        "chunks": rows,
        "boundary": "Sampling only. These chunks need saved-snapshot replay.",
    }
    write_json(output_root / "direct_contact_executor_sampling_summary.json", summary)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
