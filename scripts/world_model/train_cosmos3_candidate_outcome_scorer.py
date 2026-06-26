#!/usr/bin/env python3
"""Train an action-conditioned scorer from real candidate rollout outcomes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_cosmos3_live_receding_loop import require_compute_step  # noqa: E402
from train_cosmos3_contact_executor import contact_targets_for_horizon, read_jsonl, split_indices  # noqa: E402

CANDIDATE_DESCRIPTOR_NAMES = (
    "is_dp_prior",
    "is_teacher",
    "is_teacher_scale",
    "is_model_mean",
    "is_model_scale",
    "model_scale_value",
    "is_model_sample",
    "model_sample_temperature",
    "model_sample_index_norm64",
    "is_model_diffusion",
    "model_diffusion_index_norm15",
    "is_retrieval_success_residual",
    "retrieval_rank_norm8",
    "retrieval_residual_scale",
    "source_legacy_teacher_scale",
    "source_checkpoint_model_or_model_generated",
    "source_retrieval_success_residual",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-executor-jsonl", required=True)
    parser.add_argument("--outcome-jsonl", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-base-rows", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--error-loss-weight", type=float, default=1.0)
    parser.add_argument("--state-loss-weight", type=float, default=0.5)
    parser.add_argument("--progress-loss-weight", type=float, default=0.25)
    parser.add_argument("--binary-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--binary-positive-weights",
        default="",
        help=(
            "Optional comma-separated BCE positive weights for the binary "
            "heads. This is useful when DP-handoff positives are rare in live "
            "replay labels."
        ),
    )
    parser.add_argument("--rank-loss-weight", type=float, default=0.0)
    parser.add_argument("--rank-loss-temperature", type=float, default=0.05)
    parser.add_argument("--grouped-rank-batches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--min-steps", type=int, default=0)
    parser.add_argument("--min-wall-seconds", type=int, default=0)
    parser.add_argument("--max-wall-seconds", type=int, default=0)
    parser.add_argument("--formal-min-gpus", type=int, default=1)
    parser.add_argument("--eval-every-steps", type=int, default=20)
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument("--score-success-weight", type=float, default=0.5)
    parser.add_argument("--score-handoff-success-weight", type=float, default=0.0)
    parser.add_argument("--score-inserted-weight", type=float, default=0.25)
    parser.add_argument("--score-grasped-weight", type=float, default=0.1)
    parser.add_argument("--score-progress-weight", type=float, default=0.25)
    parser.add_argument("--score-progress-delta-weight", type=float, default=0.15)
    parser.add_argument("--score-continuable-weight", type=float, default=0.25)
    parser.add_argument(
        "--score-state-abs-axis-weights",
        default="0,0,0",
        help=(
            "Optional x/y/z weights for penalizing predicted final "
            "peg-head-in-hole error inside the selector score. This is a "
            "general insertion-manifold calibration term, not an eval gate."
        ),
    )
    parser.add_argument("--score-state-target", default="0,0,0")
    parser.add_argument(
        "--allowed-candidate-families",
        default="",
        help=(
            "Optional comma-separated family filter applied before training. "
            "Use this to train on deployment-available candidates only, for "
            "example dp_prior,checkpoint_model,model_generated,model_mean,"
            "model_scale,model_sample,model_diffusion."
        ),
    )
    parser.add_argument("--min-selected-error-improvement", type=float, default=0.005)
    parser.add_argument("--min-selected-progress-delta-improvement", type=float, default=0.0)
    parser.add_argument("--min-selected-handoff-success-improvement", type=float, default=0.0)
    parser.add_argument("--max-selected-error-degradation-for-handoff-gate", type=float, default=0.0)
    parser.add_argument("--min-non-dp-selected-fraction", type=float, default=0.1)
    parser.add_argument("--min-eval-groups-for-gate", type=int, default=16)
    parser.add_argument("--allow-handoff-only-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dedupe-candidate-name-per-row", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=20260617)
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


def fixed_width_vector(text: Any, width: int, default: float = 0.0) -> np.ndarray:
    values: list[float] = []
    if isinstance(text, (list, tuple, np.ndarray)):
        values = [float(item) for item in list(text)]
    else:
        for item in str(text).split(","):
            item = item.strip()
            if item:
                values.append(float(item))
    if len(values) < int(width):
        values.extend([float(default)] * (int(width) - len(values)))
    return np.asarray(values[: int(width)], dtype=np.float32)


def optional_positive_weights(text: Any, width: int) -> np.ndarray | None:
    if not str(text or "").strip():
        return None
    weights = fixed_width_vector(text, width, 1.0)
    if np.any(weights <= 0.0):
        raise ValueError(f"binary positive weights must be positive, got {weights.tolist()}")
    return weights.astype(np.float32)


def load_base_rows(path: Path, max_rows: int) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if max_rows > 0 and len(out) >= max_rows:
            break
        uuid = str(row.get("uuid") or "")
        if uuid:
            out[uuid] = row
    if not out:
        raise RuntimeError(f"no usable base rows in {path}")
    return out


def parse_candidate_family_set(text: str | None) -> set[str] | None:
    if text is None:
        return None
    families = {item.strip() for item in str(text).split(",") if item.strip()}
    return families or None


def load_outcome_rows(
    paths: list[str],
    *,
    dedupe: bool,
    allowed_families: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for text in paths:
        path = Path(text).resolve()
        for row in read_jsonl(path):
            if row.get("schema") != "cosmos3_candidate_outcome_label_v1":
                continue
            if allowed_families is not None:
                family = candidate_family(str(row.get("candidate_name") or ""), str(row.get("candidate_source") or ""))
                if family not in allowed_families:
                    continue
            key = (str(row.get("uuid") or ""), str(row.get("candidate_name") or ""))
            if dedupe and key in seen:
                continue
            seen.add(key)
            copied = dict(row)
            copied["_outcome_jsonl"] = str(path)
            rows.append(copied)
    if not rows:
        raise RuntimeError("no successful candidate outcome rows")
    return rows


def build_base_feature(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    task_source = str(row.get("task_path_source") or "")
    if "gt" in task_source and "debug" in task_source:
        raise RuntimeError(f"refusing GT debug task path source: {task_source}")
    executor_npz = np.load(str(row["executor_sample_npz"]), allow_pickle=False)
    prior_npz = np.load(str(row["dp_prior_npz"]), allow_pickle=False)
    current = executor_npz["current_state"].astype(np.float32).reshape(-1)
    task_path = executor_npz["task_path"].astype(np.float32)
    teacher = executor_npz["teacher_robot_actions"].astype(np.float32)[:, :7]
    prior = prior_npz["dp_prior_actions"].astype(np.float32)[:, :7]
    horizon = min(int(task_path.shape[0]), int(teacher.shape[0]), int(prior.shape[0]))
    if horizon <= 0:
        raise RuntimeError(f"empty horizon for {row.get('uuid')}")
    contact_context, _progress_target, _binary_target = contact_targets_for_horizon(row, horizon)
    task_path = task_path[:horizon]
    prior = prior[:horizon]
    feature = np.concatenate([current, task_path.reshape(-1), prior.reshape(-1), contact_context]).astype(np.float32)
    meta = {
        "uuid": row.get("uuid"),
        "source_uuid": row.get("source_uuid"),
        "scenario": row.get("scenario"),
        "prefix_role": row.get("prefix_role"),
        "current_phase": row.get("current_phase"),
        "prefix_frame_index": row.get("prefix_frame_index"),
        "horizon": int(horizon),
        "task_path_source": task_source,
    }
    return feature, prior.reshape(-1).astype(np.float32), int(horizon), meta


def load_candidate_actions(row: dict[str, Any], horizon: int) -> np.ndarray:
    arr = np.load(str(row["candidate_actions_npz"]), allow_pickle=False)["actions"].astype(np.float32)[:, :7]
    if arr.shape[0] < horizon:
        raise RuntimeError(f"candidate horizon {arr.shape[0]} < expected {horizon} for {row.get('uuid')}")
    return arr[:horizon].reshape(-1).astype(np.float32)


def contact_progress_proxy(rel: np.ndarray, *, grasped: bool, inserted: bool) -> float:
    rel = np.asarray(rel, dtype=np.float32).reshape(-1)
    if rel.size < 3:
        rel = np.pad(rel, (0, 3 - rel.size), mode="constant")
    if bool(inserted):
        return 1.0
    x, y, z = float(rel[0]), float(rel[1]), float(rel[2])
    lateral = float(np.linalg.norm([y, z]))
    insertion_progress = float(np.clip((x - -0.25) / max(-0.015 - -0.25, 1e-6), 0.0, 1.0))
    lateral_progress = float(np.clip(1.0 - lateral / 0.05, 0.0, 1.0))
    hold_progress = 1.0 if bool(grasped) else 0.0
    return float(np.clip(0.45 * insertion_progress + 0.45 * lateral_progress + 0.10 * hold_progress, 0.0, 1.0))


def outcome_progress_targets(outcome: dict[str, Any], base_row: dict[str, Any]) -> tuple[np.ndarray, bool]:
    rel = np.asarray(outcome.get("final_peg_head_at_hole") or [0.0, 0.0, 0.0], dtype=np.float32).reshape(-1)[:3]
    final_progress = outcome.get("final_contact_progress_proxy")
    if final_progress is None:
        final_progress = contact_progress_proxy(
            rel,
            grasped=bool(outcome.get("final_grasped", False)),
            inserted=bool(outcome.get("final_inserted_live_pose", False)),
        )
    current_progress = float(base_row.get("current_contact_progress") or 0.0)
    delta = outcome.get("final_contact_progress_delta_proxy")
    if delta is None:
        delta = float(final_progress) - current_progress
    if "dp_rollout_continuable_proxy" in outcome:
        continuable = bool(
            outcome.get("dp_rollout_continuable_proxy", False)
            or outcome.get("final_success", False)
            or outcome.get("final_inserted_live_pose", False)
        )
    else:
        continuable = bool(
            outcome.get("final_continuable_proxy", False)
            or outcome.get("final_success", False)
            or outcome.get("final_inserted_live_pose", False)
            or (
                bool(outcome.get("final_grasped", False))
                and bool(outcome.get("final_contact_stable_proxy", True))
                and float(final_progress) >= 0.75
            )
        )
    return np.asarray([float(final_progress), float(delta)], dtype=np.float32), continuable


def candidate_descriptor_base_name(name: str) -> str:
    return re.sub(r"^short\d+_", "", str(name))


def is_dp_prior_candidate(name: str) -> bool:
    return candidate_descriptor_base_name(name) in {"dp_prior", "model_dp_prior"}


def candidate_family(name: str, source: str) -> str:
    base = candidate_descriptor_base_name(name)
    source = str(source or "")
    if is_dp_prior_candidate(base):
        return "dp_prior"
    if base == "teacher" or base.startswith("scale_"):
        return "legacy_teacher_scale" if source == "legacy_teacher_scale" else "teacher_scale"
    if base == "model_mean":
        return "model_mean"
    if base.startswith("model_scale_"):
        return "model_scale"
    if base.startswith("model_sample_"):
        return "model_sample"
    if base.startswith("model_diffusion_"):
        return "model_diffusion"
    if base.startswith("retrieval_resid_") or source == "retrieval_success_residual":
        return "retrieval_success_residual"
    return source or "other"


def candidate_descriptor(name: str, source: str) -> np.ndarray:
    name = candidate_descriptor_base_name(name)
    is_model_scale = name.startswith("model_scale_")
    is_model_sample = name.startswith("model_sample_")
    is_model_diffusion = name.startswith("model_diffusion_")
    is_retrieval = name.startswith("retrieval_resid_")
    model_scale_value = 0.0
    model_sample_temp = 0.0
    model_sample_index = -1.0
    diffusion_index = -1.0
    retrieval_rank = -1.0
    retrieval_scale = 0.0
    if is_model_scale:
        try:
            model_scale_value = float(name.rsplit("_", 1)[-1])
        except ValueError:
            model_scale_value = 0.0
    if is_model_sample:
        match = re.search(r"model_sample_t(?P<temp>[-+0-9.eE]+)_(?P<idx>\d+)$", name)
        if match:
            model_sample_temp = float(match.group("temp"))
            model_sample_index = float(match.group("idx"))
    if is_model_diffusion:
        try:
            diffusion_index = float(name.rsplit("_", 1)[-1])
        except ValueError:
            diffusion_index = -1.0
    if is_retrieval:
        match = re.search(r"retrieval_resid(?:_r(?P<rank>\d+)|_(?P<oldrank>\d+))(?:_s(?P<scale>[0-9p]+))?", name)
        if match:
            rank_text = match.group("rank") or match.group("oldrank")
            retrieval_rank = float(rank_text)
            scale_text = match.group("scale")
            retrieval_scale = float(scale_text.replace("p", ".")) if scale_text else 1.0
    is_teacher_scale = name.startswith("scale_")
    teacher_scale_value = 0.0
    if is_teacher_scale:
        try:
            teacher_scale_value = float(name.rsplit("_", 1)[-1])
        except ValueError:
            teacher_scale_value = 0.0
    values = [
        float(is_dp_prior_candidate(name)),
        float(name == "teacher"),
        float(is_teacher_scale),
        float(name == "model_mean"),
        float(is_model_scale),
        float(model_scale_value),
        float(is_model_sample),
        float(model_sample_temp),
        float(model_sample_index / 64.0 if model_sample_index >= 0.0 else 0.0),
        float(is_model_diffusion),
        float(diffusion_index / 15.0 if diffusion_index >= 0.0 else 0.0),
        float(is_retrieval),
        float(retrieval_rank / 8.0 if retrieval_rank >= 0.0 else 0.0),
        float(retrieval_scale),
        float(source == "legacy_teacher_scale"),
        float(source in {"checkpoint_model", "model_generated"}),
        float(source == "retrieval_success_residual"),
    ]
    if len(values) != len(CANDIDATE_DESCRIPTOR_NAMES):
        raise AssertionError("candidate descriptor schema length mismatch")
    return np.asarray(values, dtype=np.float32)


def load_dataset(
    *,
    base_rows: dict[str, dict[str, Any]],
    outcome_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    base_cache: dict[str, tuple[np.ndarray, np.ndarray, int, dict[str, Any]]] = {}
    features: list[np.ndarray] = []
    errors: list[float] = []
    states: list[np.ndarray] = []
    progresses: list[np.ndarray] = []
    binaries: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    for outcome in outcome_rows:
        uuid = str(outcome.get("uuid") or "")
        if uuid not in base_rows:
            continue
        if uuid not in base_cache:
            base_cache[uuid] = build_base_feature(base_rows[uuid])
        base_feature, prior_flat, horizon, base_meta = base_cache[uuid]
        if int(outcome.get("horizon") or horizon) != horizon:
            continue
        candidate = load_candidate_actions(outcome, horizon)
        residual = candidate - prior_flat
        candidate_name = str(outcome.get("candidate_name") or "")
        candidate_source = str(outcome.get("candidate_source") or "")
        descriptor = candidate_descriptor(candidate_name, candidate_source)
        feature = np.concatenate([base_feature, candidate, residual, descriptor]).astype(np.float32)
        final_state = np.asarray(outcome["final_peg_head_at_hole"], dtype=np.float32).reshape(3)
        progress_target, continuable_proxy = outcome_progress_targets(outcome, base_rows[uuid])
        handoff_success = bool(outcome.get("dp_rollout_success", False) or outcome.get("final_success", False))
        binary = np.asarray(
            [
                float(bool(outcome.get("final_success", False))),
                float(handoff_success),
                float(bool(outcome.get("final_inserted_live_pose", False))),
                float(bool(outcome.get("final_grasped", False))),
                float(continuable_proxy),
            ],
            dtype=np.float32,
        )
        features.append(feature)
        errors.append(float(outcome["final_abs_task_error_weighted"]))
        states.append(final_state)
        progresses.append(progress_target)
        binaries.append(binary)
        metas.append(
            {
                **base_meta,
                "candidate_name": candidate_name,
                "candidate_source": candidate_source,
                "candidate_descriptor": descriptor.astype(float).tolist(),
                "outcome_jsonl": outcome.get("_outcome_jsonl"),
                "candidate_actions_npz": outcome.get("candidate_actions_npz"),
                "true_final_abs_task_error_weighted": float(outcome["final_abs_task_error_weighted"]),
                "true_final_peg_head_at_hole": final_state.astype(float).tolist(),
                "true_final_success": bool(outcome.get("final_success", False)),
                "true_final_inserted_live_pose": bool(outcome.get("final_inserted_live_pose", False)),
                "true_final_grasped": bool(outcome.get("final_grasped", False)),
                "true_final_contact_stable_proxy": bool(outcome.get("final_contact_stable_proxy", False)),
                "true_dp_rollout_continuable_proxy": (
                    bool(outcome.get("dp_rollout_continuable_proxy"))
                    if "dp_rollout_continuable_proxy" in outcome
                    else None
                ),
                "true_dp_rollout_success": (
                    bool(outcome.get("dp_rollout_success")) if "dp_rollout_success" in outcome else None
                ),
                "true_final_contact_progress_proxy": float(progress_target[0]),
                "true_final_contact_progress_delta_proxy": float(progress_target[1]),
                "true_final_continuable_proxy": bool(continuable_proxy),
            }
        )
    if not features:
        raise RuntimeError("no joined outcome scorer samples")
    widths = {item.shape[0] for item in features}
    if len(widths) != 1:
        raise RuntimeError(f"nonuniform feature widths: {sorted(widths)}")
    return {
        "x": np.stack(features).astype(np.float32),
        "error": np.asarray(errors, dtype=np.float32).reshape(-1, 1),
        "state": np.stack(states).astype(np.float32),
        "progress": np.stack(progresses).astype(np.float32),
        "binary": np.stack(binaries).astype(np.float32),
        "meta": metas,
        "num_base_rows": int(len(base_cache)),
    }


def split_by_uuid(meta: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    uuids = sorted({str(item.get("uuid") or "") for item in meta})
    if len(uuids) <= 1 or val_fraction <= 0:
        indices = np.arange(len(meta), dtype=np.int64)
        return indices, indices[:0]
    uuid_train, uuid_val = split_indices(len(uuids), val_fraction, seed)
    val_set = {uuids[int(i)] for i in uuid_val}
    train = [idx for idx, item in enumerate(meta) if str(item.get("uuid") or "") not in val_set]
    val = [idx for idx, item in enumerate(meta) if str(item.get("uuid") or "") in val_set]
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


class OutcomeScorerNet:
    def __new__(cls, feature_dim: int, hidden_dim: int, num_layers: int, dropout: float, binary_dim: int = 5) -> Any:
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
                self.error_head = torch.nn.Linear(int(hidden_dim), 1)
                self.state_head = torch.nn.Linear(int(hidden_dim), 3)
                self.progress_head = torch.nn.Linear(int(hidden_dim), 2)
                self.binary_head = torch.nn.Linear(int(hidden_dim), int(binary_dim))

            def forward(self, x: Any) -> tuple[Any, Any, Any, Any]:
                h = self.encoder(x)
                return self.error_head(h), self.state_head(h), self.progress_head(h), self.binary_head(h)

        return _Net()


def candidate_score(
    pred_error: np.ndarray,
    pred_state: np.ndarray | None,
    pred_progress: np.ndarray,
    pred_binary: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    progress = np.asarray(pred_progress, dtype=np.float32)
    if progress.ndim != 2 or progress.shape[1] < 2:
        progress = np.zeros((np.asarray(pred_error).reshape(-1).shape[0], 2), dtype=np.float32)
    handoff_success = pred_binary[:, 1] if pred_binary.shape[1] > 4 else 0.0
    inserted_idx = 2 if pred_binary.shape[1] > 4 else 1
    grasped_idx = 3 if pred_binary.shape[1] > 4 else 2
    continuable_idx = 4 if pred_binary.shape[1] > 4 else 3
    continuable = pred_binary[:, continuable_idx] if pred_binary.shape[1] > continuable_idx else 0.0
    state_penalty = np.zeros((np.asarray(pred_error).reshape(-1).shape[0],), dtype=np.float32)
    if pred_state is not None:
        state = np.asarray(pred_state, dtype=np.float32)
        if state.ndim == 2 and state.shape[1] >= 3:
            weights = fixed_width_vector(getattr(args, "score_state_abs_axis_weights", "0,0,0"), 3, 0.0)
            target = fixed_width_vector(getattr(args, "score_state_target", "0,0,0"), 3, 0.0)
            state_penalty = np.sum(np.abs(state[:, :3] - target.reshape(1, 3)) * weights.reshape(1, 3), axis=1)
    return (
        -pred_error.reshape(-1)
        + float(getattr(args, "score_success_weight", 0.5)) * pred_binary[:, 0]
        + float(getattr(args, "score_handoff_success_weight", 0.0)) * handoff_success
        + float(getattr(args, "score_inserted_weight", 0.25)) * pred_binary[:, inserted_idx]
        + float(getattr(args, "score_grasped_weight", 0.1)) * pred_binary[:, grasped_idx]
        + float(getattr(args, "score_progress_weight", 0.0)) * progress[:, 0]
        + float(getattr(args, "score_progress_delta_weight", 0.0)) * progress[:, 1]
        + float(getattr(args, "score_continuable_weight", 0.0)) * continuable
        - state_penalty
    )


def candidate_score_torch(
    pred_error_raw: Any,
    pred_state: Any | None,
    pred_progress: Any,
    pred_logits: Any,
    args: argparse.Namespace,
) -> Any:
    import torch

    pred_binary = torch.sigmoid(pred_logits)
    handoff_success = pred_binary[:, 1] if pred_binary.shape[1] > 4 else pred_binary.new_zeros((pred_binary.shape[0],))
    inserted_idx = 2 if pred_binary.shape[1] > 4 else 1
    grasped_idx = 3 if pred_binary.shape[1] > 4 else 2
    continuable_idx = 4 if pred_binary.shape[1] > 4 else 3
    continuable = (
        pred_binary[:, continuable_idx]
        if pred_binary.shape[1] > continuable_idx
        else pred_binary.new_zeros((pred_binary.shape[0],))
    )
    state_penalty = pred_error_raw.new_zeros((pred_error_raw.reshape(-1).shape[0],))
    if pred_state is not None:
        weights_np = fixed_width_vector(getattr(args, "score_state_abs_axis_weights", "0,0,0"), 3, 0.0)
        if np.any(weights_np != 0.0):
            target_np = fixed_width_vector(getattr(args, "score_state_target", "0,0,0"), 3, 0.0)
            weights = torch.as_tensor(weights_np, device=pred_state.device, dtype=pred_state.dtype).reshape(1, 3)
            target = torch.as_tensor(target_np, device=pred_state.device, dtype=pred_state.dtype).reshape(1, 3)
            state_penalty = torch.sum(torch.abs(pred_state[:, :3] - target) * weights, dim=1)
    return (
        -pred_error_raw.reshape(-1)
        + float(getattr(args, "score_success_weight", 0.5)) * pred_binary[:, 0]
        + float(getattr(args, "score_handoff_success_weight", 0.0)) * handoff_success
        + float(getattr(args, "score_inserted_weight", 0.25)) * pred_binary[:, inserted_idx]
        + float(getattr(args, "score_grasped_weight", 0.1)) * pred_binary[:, grasped_idx]
        + float(getattr(args, "score_progress_weight", 0.0)) * pred_progress[:, 0]
        + float(getattr(args, "score_progress_delta_weight", 0.0)) * pred_progress[:, 1]
        + float(getattr(args, "score_continuable_weight", 0.0)) * continuable
        - state_penalty
    )


def grouped_candidate_rank_loss(
    *,
    pred_score: Any,
    true_score: Any,
    global_indices: Any,
    meta: list[dict[str, Any]],
    temperature: float,
) -> tuple[Any, int]:
    import torch
    import torch.nn.functional as F

    groups: dict[str, list[int]] = defaultdict(list)
    for local_i, global_i in enumerate(global_indices.detach().cpu().tolist()):
        uuid = str(meta[int(global_i)].get("uuid") or "")
        if uuid:
            groups[uuid].append(int(local_i))
    losses: list[Any] = []
    temp = max(float(temperature), 1e-6)
    for local_list in groups.values():
        if len(local_list) < 2:
            continue
        group_idx = torch.as_tensor(local_list, dtype=torch.long, device=pred_score.device)
        target_pos = int(torch.argmax(true_score[group_idx]).detach().cpu().item())
        logits = pred_score[group_idx].reshape(1, -1) / temp
        target = torch.as_tensor([target_pos], dtype=torch.long, device=pred_score.device)
        losses.append(F.cross_entropy(logits, target))
    if not losses:
        return pred_score.new_zeros(()), 0
    return torch.stack(losses).mean(), len(losses)


def meta_handoff_success(item: dict[str, Any]) -> bool:
    return bool(item.get("true_dp_rollout_success", False) or item.get("true_final_success", False))


def meta_handoff_continuable(item: dict[str, Any]) -> bool:
    return bool(
        item.get("true_dp_rollout_continuable_proxy", False)
        or item.get("true_final_continuable_proxy", False)
        or item.get("true_final_success", False)
    )


def evaluate(
    *,
    model: Any,
    x_norm: np.ndarray,
    error: np.ndarray,
    error_mean: np.ndarray,
    error_std: np.ndarray,
    state: np.ndarray,
    progress: np.ndarray,
    binary: np.ndarray,
    meta: list[dict[str, Any]],
    indices: np.ndarray,
    args: argparse.Namespace,
    device: Any,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    if len(indices) == 0:
        return {"num_rows": 0, "num_groups": 0}
    module = model.module if hasattr(model, "module") else model
    module.eval()
    with torch.no_grad():
        x = torch.from_numpy(x_norm[indices]).to(device)
        pred_error_norm, pred_state, pred_progress, pred_logits = module(x)
        pred_error = pred_error_norm.detach().cpu().numpy() * error_std + error_mean
        pred_state_np = pred_state.detach().cpu().numpy()
        pred_progress_np = pred_progress.detach().cpu().numpy()
        pred_binary = torch.sigmoid(pred_logits).detach().cpu().numpy()
        target_error = error[indices]
        target_state = state[indices]
        target_progress = progress[indices]
        target_binary = binary[indices]
        error_mse = float(np.mean((pred_error - target_error) ** 2))
        state_mse = float(np.mean((pred_state_np - target_state) ** 2))
        progress_mse = float(np.mean((pred_progress_np - target_progress) ** 2))
        binary_bce = float(F.binary_cross_entropy_with_logits(pred_logits, torch.from_numpy(target_binary).to(device)).item())

    groups: dict[str, list[int]] = defaultdict(list)
    for local_i, global_i in enumerate(indices):
        groups[str(meta[int(global_i)].get("uuid") or "")].append(local_i)

    selected_errors: list[float] = []
    dp_errors: list[float] = []
    oracle_errors: list[float] = []
    selected_progress_deltas: list[float] = []
    dp_progress_deltas: list[float] = []
    oracle_progress_deltas: list[float] = []
    selected_handoff_successes: list[float] = []
    dp_handoff_successes: list[float] = []
    oracle_handoff_successes: list[float] = []
    handoff_oracle_successes: list[float] = []
    selected_handoff_continuables: list[float] = []
    dp_handoff_continuables: list[float] = []
    selected_names: list[str] = []
    oracle_names: list[str] = []
    handoff_oracle_names: list[str] = []
    top1_match = 0
    top1_handoff_match = 0
    group_count = 0
    scores = candidate_score(pred_error, pred_state_np, pred_progress_np, pred_binary, args)
    for _uuid, local_list in groups.items():
        if not local_list:
            continue
        true_vals = target_error[local_list, 0]
        pred_scores = scores[local_list]
        best_local = local_list[int(np.argmax(pred_scores))]
        oracle_local = local_list[int(np.argmin(true_vals))]
        handoff_candidates = [
            local_i
            for local_i in local_list
            if meta_handoff_success(meta[int(indices[local_i])])
        ]
        handoff_oracle_local = (
            min(handoff_candidates, key=lambda local_i: float(target_error[local_i, 0]))
            if handoff_candidates
            else None
        )
        dp_local = None
        for local_i in local_list:
            global_i = int(indices[local_i])
            if is_dp_prior_candidate(str(meta[global_i].get("candidate_name") or "")):
                dp_local = local_i
                break
        if dp_local is None:
            continue
        group_count += 1
        selected_errors.append(float(target_error[best_local, 0]))
        dp_errors.append(float(target_error[dp_local, 0]))
        oracle_errors.append(float(target_error[oracle_local, 0]))
        selected_progress_deltas.append(float(target_progress[best_local, 1]))
        dp_progress_deltas.append(float(target_progress[dp_local, 1]))
        oracle_progress_deltas.append(float(target_progress[oracle_local, 1]))
        selected_meta = meta[int(indices[best_local])]
        dp_meta = meta[int(indices[dp_local])]
        oracle_meta = meta[int(indices[oracle_local])]
        selected_handoff_successes.append(float(meta_handoff_success(selected_meta)))
        dp_handoff_successes.append(float(meta_handoff_success(dp_meta)))
        oracle_handoff_successes.append(float(meta_handoff_success(oracle_meta)))
        handoff_oracle_successes.append(float(handoff_oracle_local is not None))
        selected_handoff_continuables.append(float(meta_handoff_continuable(selected_meta)))
        dp_handoff_continuables.append(float(meta_handoff_continuable(dp_meta)))
        selected_names.append(str(meta[int(indices[best_local])].get("candidate_name") or ""))
        oracle_names.append(str(meta[int(indices[oracle_local])].get("candidate_name") or ""))
        handoff_oracle_names.append(
            str(meta[int(indices[handoff_oracle_local])].get("candidate_name") or "")
            if handoff_oracle_local is not None
            else "none"
        )
        if best_local == oracle_local:
            top1_match += 1
        if handoff_oracle_local is not None and best_local == handoff_oracle_local:
            top1_handoff_match += 1

    selected = np.asarray(selected_errors, dtype=np.float32)
    dp = np.asarray(dp_errors, dtype=np.float32)
    oracle = np.asarray(oracle_errors, dtype=np.float32)
    selected_progress_delta = np.asarray(selected_progress_deltas, dtype=np.float32)
    dp_progress_delta = np.asarray(dp_progress_deltas, dtype=np.float32)
    oracle_progress_delta = np.asarray(oracle_progress_deltas, dtype=np.float32)
    selected_handoff_success = np.asarray(selected_handoff_successes, dtype=np.float32)
    dp_handoff_success = np.asarray(dp_handoff_successes, dtype=np.float32)
    oracle_handoff_success = np.asarray(oracle_handoff_successes, dtype=np.float32)
    handoff_oracle_success = np.asarray(handoff_oracle_successes, dtype=np.float32)
    selected_handoff_continuable = np.asarray(selected_handoff_continuables, dtype=np.float32)
    dp_handoff_continuable = np.asarray(dp_handoff_continuables, dtype=np.float32)
    non_dp_selected = sum(1 for name in selected_names if not is_dp_prior_candidate(name))
    out = {
        "num_rows": int(len(indices)),
        "num_groups": int(group_count),
        "error_mse": error_mse,
        "state_mse": state_mse,
        "progress_mse": progress_mse,
        "binary_bce": binary_bce,
        "selected_true_weighted_error_mean": float(selected.mean()) if selected.size else None,
        "dp_prior_true_weighted_error_mean": float(dp.mean()) if dp.size else None,
        "oracle_true_weighted_error_mean": float(oracle.mean()) if oracle.size else None,
        "selected_minus_dp_prior_weighted_error_mean": float((selected - dp).mean()) if selected.size else None,
        "oracle_minus_dp_prior_weighted_error_mean": float((oracle - dp).mean()) if oracle.size else None,
        "selected_true_contact_progress_delta_mean": (
            float(selected_progress_delta.mean()) if selected_progress_delta.size else None
        ),
        "dp_prior_true_contact_progress_delta_mean": float(dp_progress_delta.mean()) if dp_progress_delta.size else None,
        "oracle_true_contact_progress_delta_mean": (
            float(oracle_progress_delta.mean()) if oracle_progress_delta.size else None
        ),
        "selected_minus_dp_prior_contact_progress_delta_mean": (
            float((selected_progress_delta - dp_progress_delta).mean()) if selected_progress_delta.size else None
        ),
        "selected_handoff_success_count": int(selected_handoff_success.sum()) if selected_handoff_success.size else 0,
        "dp_prior_handoff_success_count": int(dp_handoff_success.sum()) if dp_handoff_success.size else 0,
        "oracle_handoff_success_count": int(oracle_handoff_success.sum()) if oracle_handoff_success.size else 0,
        "handoff_oracle_success_count": int(handoff_oracle_success.sum()) if handoff_oracle_success.size else 0,
        "selected_handoff_success_fraction": (
            float(selected_handoff_success.mean()) if selected_handoff_success.size else None
        ),
        "dp_prior_handoff_success_fraction": (
            float(dp_handoff_success.mean()) if dp_handoff_success.size else None
        ),
        "selected_minus_dp_prior_handoff_success_fraction": (
            float((selected_handoff_success - dp_handoff_success).mean())
            if selected_handoff_success.size
            else None
        ),
        "selected_handoff_continuable_fraction": (
            float(selected_handoff_continuable.mean()) if selected_handoff_continuable.size else None
        ),
        "dp_prior_handoff_continuable_fraction": (
            float(dp_handoff_continuable.mean()) if dp_handoff_continuable.size else None
        ),
        "top1_oracle_match_fraction": float(top1_match / max(1, group_count)),
        "top1_handoff_oracle_match_fraction": float(top1_handoff_match / max(1, group_count)),
        "selected_non_dp_candidate_count": int(non_dp_selected),
        "selected_non_dp_candidate_fraction": float(non_dp_selected / max(1, group_count)),
        "selected_candidate_counts": dict(sorted(Counter(selected_names).items())),
        "oracle_candidate_counts": dict(sorted(Counter(oracle_names).items())),
        "handoff_oracle_candidate_counts": dict(sorted(Counter(handoff_oracle_names).items())),
    }
    module.train()
    return out


def offline_gate_from_eval(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    try:
        delta = float(metrics.get("selected_minus_dp_prior_weighted_error_mean"))
        progress_delta = float(metrics.get("selected_minus_dp_prior_contact_progress_delta_mean"))
        handoff_delta = float(metrics.get("selected_minus_dp_prior_handoff_success_fraction"))
        non_dp = float(metrics.get("selected_non_dp_candidate_fraction"))
        groups = int(metrics.get("num_groups", 0))
    except (TypeError, ValueError):
        return False
    common_ok = bool(groups >= int(args.min_eval_groups_for_gate) and non_dp >= float(args.min_non_dp_selected_fraction))
    continuous_progress_ok = bool(
        np.isfinite(delta)
        and np.isfinite(progress_delta)
        and np.isfinite(handoff_delta)
        and handoff_delta > float(args.min_selected_handoff_success_improvement)
        and delta <= -float(args.min_selected_error_improvement)
        and progress_delta >= float(args.min_selected_progress_delta_improvement)
    )
    handoff_success_ok = bool(
        np.isfinite(handoff_delta)
        and handoff_delta > float(args.min_selected_handoff_success_improvement)
    )
    safe_handoff_success_ok = bool(
        handoff_success_ok
        and np.isfinite(delta)
        and np.isfinite(progress_delta)
        and delta <= float(args.max_selected_error_degradation_for_handoff_gate)
        and progress_delta >= float(args.min_selected_progress_delta_improvement)
    )
    if bool(getattr(args, "allow_handoff_only_gate", False)):
        handoff_branch_ok = handoff_success_ok
    else:
        handoff_branch_ok = safe_handoff_success_ok
    return bool(
        common_ok
        and (continuous_progress_ok or handoff_branch_ok)
    )


def offline_gate_rank_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    try:
        handoff_delta = float(metrics.get("selected_minus_dp_prior_handoff_success_fraction"))
    except (TypeError, ValueError):
        handoff_delta = float("-inf")
    try:
        error_delta = float(metrics.get("selected_minus_dp_prior_weighted_error_mean"))
    except (TypeError, ValueError):
        error_delta = float("inf")
    try:
        progress_delta = float(metrics.get("selected_minus_dp_prior_contact_progress_delta_mean"))
    except (TypeError, ValueError):
        progress_delta = float("-inf")
    return handoff_delta, -error_delta, progress_delta


def main() -> int:
    args = parse_args()
    require_compute_step()

    import torch
    from torch.utils.data import DataLoader, TensorDataset

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
    allowed_candidate_families = parse_candidate_family_set(args.allowed_candidate_families)
    outcome_rows = load_outcome_rows(
        list(args.outcome_jsonl),
        dedupe=bool(args.dedupe_candidate_name_per_row),
        allowed_families=allowed_candidate_families,
    )
    dataset = load_dataset(base_rows=base_rows, outcome_rows=outcome_rows)
    x = dataset["x"]
    error = dataset["error"]
    state = dataset["state"]
    progress = dataset["progress"]
    binary = dataset["binary"]
    meta = dataset["meta"]
    train_idx, val_idx = split_by_uuid(meta, float(args.val_fraction), int(args.seed))
    eval_idx = val_idx if len(val_idx) else train_idx
    x_mean = x[train_idx].mean(axis=0, keepdims=True)
    x_std = np.where(x[train_idx].std(axis=0, keepdims=True) < 1e-6, 1.0, x[train_idx].std(axis=0, keepdims=True))
    error_mean = error[train_idx].mean(axis=0, keepdims=True)
    error_std = np.where(error[train_idx].std(axis=0, keepdims=True) < 1e-6, 1.0, error[train_idx].std(axis=0, keepdims=True))
    x_norm = ((x - x_mean) / x_std).astype(np.float32)
    error_norm = ((error - error_mean) / error_std).astype(np.float32)

    use_grouped_rank_batches = bool(args.grouped_rank_batches) and float(args.rank_loss_weight) > 0.0
    train_order_idx = train_idx
    if use_grouped_rank_batches:
        train_order_idx = np.asarray(
            sorted(
                (int(i) for i in train_idx),
                key=lambda i: (
                    str(meta[i].get("uuid") or ""),
                    str(meta[i].get("candidate_name") or ""),
                ),
            ),
            dtype=np.int64,
        )
    train_ds = TensorDataset(
        torch.from_numpy(x_norm[train_order_idx]),
        torch.from_numpy(error_norm[train_order_idx]),
        torch.from_numpy(error[train_order_idx]),
        torch.from_numpy(state[train_order_idx]),
        torch.from_numpy(progress[train_order_idx]),
        torch.from_numpy(binary[train_order_idx]),
        torch.from_numpy(train_order_idx.astype(np.int64)),
    )
    loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=not use_grouped_rank_batches,
        num_workers=0,
        drop_last=False,
    )
    model = OutcomeScorerNet(
        feature_dim=int(x.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        binary_dim=int(binary.shape[1]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    binary_pos_weight_np = optional_positive_weights(args.binary_positive_weights, int(binary.shape[1]))
    binary_pos_weight_t = (
        torch.as_tensor(binary_pos_weight_np, device=device, dtype=torch.float32)
        if binary_pos_weight_np is not None
        else None
    )
    error_mean_t = torch.from_numpy(error_mean.astype(np.float32)).to(device)
    error_std_t = torch.from_numpy(error_std.astype(np.float32)).to(device)

    write_json(
        output_root / "training_manifest.json",
        {
            "schema": "cosmos3_candidate_outcome_scorer_training_v1",
            "contact_executor_jsonl": str(Path(args.contact_executor_jsonl).resolve()),
            "outcome_jsonl": [str(Path(item).resolve()) for item in args.outcome_jsonl],
            "output_root": str(output_root),
            "num_joined_candidate_rows": int(len(meta)),
            "num_base_rows": int(dataset["num_base_rows"]),
            "num_train_candidate_rows": int(len(train_idx)),
            "num_val_candidate_rows": int(len(val_idx)),
            "num_train_groups": int(len({str(meta[int(i)].get("uuid") or "") for i in train_idx})),
            "num_val_groups": int(len({str(meta[int(i)].get("uuid") or "") for i in val_idx})),
            "candidate_family_filter": {
                "allowed_candidate_families": sorted(allowed_candidate_families) if allowed_candidate_families else None,
                "family_counts": dict(
                    sorted(
                        Counter(
                            candidate_family(
                                str(item.get("candidate_name") or ""),
                                str(item.get("candidate_source") or ""),
                            )
                            for item in meta
                        ).items()
                    )
                ),
            },
            "feature_dim": int(x.shape[1]),
            "binary_dim": int(binary.shape[1]),
            "candidate_descriptor_names": list(CANDIDATE_DESCRIPTOR_NAMES),
            "device": str(device),
            "visible_cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "formal_min_gpus": int(args.formal_min_gpus),
            "min_wall_seconds": int(args.min_wall_seconds),
            "max_wall_seconds": int(args.max_wall_seconds),
            "rank_loss": {
                "weight": float(args.rank_loss_weight),
                "temperature": float(args.rank_loss_temperature),
                "grouped_rank_batches": bool(args.grouped_rank_batches),
                "active_grouped_rank_batches": bool(use_grouped_rank_batches),
                "reason": (
                    "Within each live-state uuid, rank candidates by real "
                    "simulator composite value: low final error plus final "
                    "success, h96 DP-handoff success, insertion, grasp "
                    "preservation, contact progress, and continuability. "
                    "This keeps the rank objective aligned with the live "
                    "selector score."
                ),
            },
            "prediction_heads": {
                "error": ["final_abs_task_error_weighted"],
                "state": ["final_peg_head_at_hole_x", "final_peg_head_at_hole_y", "final_peg_head_at_hole_z"],
                "progress": ["final_contact_progress_proxy", "final_contact_progress_delta_proxy"],
                "binary": [
                    "final_success",
                    "dp_rollout_success_or_final_success",
                    "final_inserted_live_pose",
                    "final_grasped",
                    "dp_rollout_continuable_proxy_or_final_continuable_proxy",
                ],
            },
            "score_weights": {
                "success": float(args.score_success_weight),
                "handoff_success": float(args.score_handoff_success_weight),
                "inserted": float(args.score_inserted_weight),
                "grasped": float(args.score_grasped_weight),
                "progress": float(args.score_progress_weight),
                "progress_delta": float(args.score_progress_delta_weight),
                "continuable": float(args.score_continuable_weight),
                "state_abs_axis_weights": fixed_width_vector(args.score_state_abs_axis_weights, 3, 0.0).astype(float).tolist(),
                "state_target": fixed_width_vector(args.score_state_target, 3, 0.0).astype(float).tolist(),
            },
            "loss_weights": {
                "error": float(args.error_loss_weight),
                "state": float(args.state_loss_weight),
                "progress": float(args.progress_loss_weight),
                "binary": float(args.binary_loss_weight),
                "rank": float(args.rank_loss_weight),
                "binary_positive_weights": (
                    binary_pos_weight_np.astype(float).tolist() if binary_pos_weight_np is not None else None
                ),
            },
            "offline_gate": {
                "min_selected_error_improvement": float(args.min_selected_error_improvement),
                "min_selected_progress_delta_improvement": float(args.min_selected_progress_delta_improvement),
                "min_selected_handoff_success_improvement": float(args.min_selected_handoff_success_improvement),
                "max_selected_error_degradation_for_handoff_gate": float(
                    args.max_selected_error_degradation_for_handoff_gate
                ),
                "min_non_dp_selected_fraction": float(args.min_non_dp_selected_fraction),
                "min_eval_groups_for_gate": int(args.min_eval_groups_for_gate),
                "allow_handoff_only_gate": bool(args.allow_handoff_only_gate),
            },
            "sample_meta": meta[:10],
            "method_boundary": (
                "This scorer is trained only from real simulator outcomes of "
                "candidate short action chunks. It may rank candidate actions "
                "offline; live evidence still requires executing selected chunks "
                "in the real closed-loop rollout with final-state and video review."
            ),
        },
    )

    history: list[dict[str, Any]] = []
    best_gate_metrics: dict[str, Any] | None = None
    best_gate_step = 0
    best_eval_delta = float("inf")
    best_eval_metrics: dict[str, Any] | None = None
    start = time.time()
    step = 0
    last_save = 0
    stop_reason = "running"
    while True:
        for batch in loader:
            step += 1
            x_b, error_b, error_raw_b, state_b, progress_b, binary_b, global_idx_b = [
                item.to(device, non_blocking=True) for item in batch
            ]
            pred_error, pred_state, pred_progress, pred_logits = model(x_b)
            error_loss = torch.mean((pred_error - error_b) ** 2)
            state_loss = torch.mean((pred_state - state_b) ** 2)
            progress_loss = torch.mean((pred_progress - progress_b) ** 2)
            binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pred_logits,
                binary_b,
                pos_weight=binary_pos_weight_t,
            )
            rank_loss = pred_error.new_zeros(())
            rank_group_count = 0
            if float(args.rank_loss_weight) > 0.0:
                pred_error_raw = pred_error * error_std_t + error_mean_t
                pred_score = candidate_score_torch(pred_error_raw, pred_state, pred_progress, pred_logits, args)
                true_state_penalty = error_raw_b.new_zeros((error_raw_b.reshape(-1).shape[0],))
                weights_np = fixed_width_vector(args.score_state_abs_axis_weights, 3, 0.0)
                if np.any(weights_np != 0.0):
                    target_np = fixed_width_vector(args.score_state_target, 3, 0.0)
                    weights = torch.as_tensor(weights_np, device=state_b.device, dtype=state_b.dtype).reshape(1, 3)
                    target = torch.as_tensor(target_np, device=state_b.device, dtype=state_b.dtype).reshape(1, 3)
                    true_state_penalty = torch.sum(torch.abs(state_b[:, :3] - target) * weights, dim=1)
                true_score = (
                    -error_raw_b.reshape(-1)
                    + float(args.score_success_weight) * binary_b[:, 0]
                    + float(args.score_handoff_success_weight) * (
                        binary_b[:, 1] if binary_b.shape[1] > 4 else binary_b.new_zeros((binary_b.shape[0],))
                    )
                    + float(args.score_inserted_weight) * binary_b[:, 2 if binary_b.shape[1] > 4 else 1]
                    + float(args.score_grasped_weight) * binary_b[:, 3 if binary_b.shape[1] > 4 else 2]
                    + float(args.score_progress_weight) * progress_b[:, 0]
                    + float(args.score_progress_delta_weight) * progress_b[:, 1]
                    + float(args.score_continuable_weight) * binary_b[:, 4 if binary_b.shape[1] > 4 else 3]
                    - true_state_penalty
                )
                rank_loss, rank_group_count = grouped_candidate_rank_loss(
                    pred_score=pred_score,
                    true_score=true_score,
                    global_indices=global_idx_b,
                    meta=meta,
                    temperature=float(args.rank_loss_temperature),
                )
            loss = (
                float(args.error_loss_weight) * error_loss
                + float(args.state_loss_weight) * state_loss
                + float(args.progress_loss_weight) * progress_loss
                + float(args.binary_loss_weight) * binary_loss
                + float(args.rank_loss_weight) * rank_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step == 1 or step % int(args.eval_every_steps) == 0:
                eval_metrics = evaluate(
                    model=model,
                    x_norm=x_norm,
                    error=error,
                    error_mean=error_mean,
                    error_std=error_std,
                    state=state,
                    progress=progress,
                    binary=binary,
                    meta=meta,
                    indices=eval_idx,
                    args=args,
                    device=device,
                )
                train_metrics = evaluate(
                    model=model,
                    x_norm=x_norm,
                    error=error,
                    error_mean=error_mean,
                    error_std=error_std,
                    state=state,
                    progress=progress,
                    binary=binary,
                    meta=meta,
                    indices=train_idx,
                    args=args,
                    device=device,
                )
                metrics = {
                    "step": int(step),
                    "elapsed_seconds": float(time.time() - start),
                    "train_batch_loss": float(loss.detach().cpu().item()),
                    "train_batch_error_mse_norm": float(error_loss.detach().cpu().item()),
                    "train_batch_state_mse": float(state_loss.detach().cpu().item()),
                    "train_batch_progress_mse": float(progress_loss.detach().cpu().item()),
                    "train_batch_binary_bce": float(binary_loss.detach().cpu().item()),
                    "train_batch_rank_ce": float(rank_loss.detach().cpu().item()),
                    "train_batch_rank_groups": int(rank_group_count),
                    "train": train_metrics,
                    "eval": eval_metrics,
                }
                history.append(metrics)
                write_json(output_root / "training_history.json", history)
                delta = float(eval_metrics.get("selected_minus_dp_prior_weighted_error_mean", float("inf")))
                if np.isfinite(delta) and delta < best_eval_delta:
                    best_eval_delta = delta
                    best_eval_metrics = metrics
                    atomic_torch_save(
                        {
                            "model_state_dict": model.state_dict(),
                            "step": int(step),
                            "feature_dim": int(x.shape[1]),
                            "x_mean": x_mean.astype(np.float32),
                            "x_std": x_std.astype(np.float32),
                            "error_mean": error_mean.astype(np.float32),
                            "error_std": error_std.astype(np.float32),
                            "candidate_descriptor_names": list(CANDIDATE_DESCRIPTOR_NAMES),
                            "binary_dim": int(binary.shape[1]),
                            "args": vars(args),
                            "latest_metrics": metrics,
                            "best_selection_metric": "eval.selected_minus_dp_prior_weighted_error_mean",
                        },
                        output_root / "checkpoint_best_offline.pt",
                    )
                if offline_gate_from_eval(eval_metrics, args):
                    if best_gate_metrics is None or offline_gate_rank_key(eval_metrics) > offline_gate_rank_key(
                        best_gate_metrics["eval"]
                    ):
                        best_gate_metrics = metrics
                        best_gate_step = int(step)
                        atomic_torch_save(
                            {
                                "model_state_dict": model.state_dict(),
                                "step": int(step),
                                "feature_dim": int(x.shape[1]),
                                "x_mean": x_mean.astype(np.float32),
                                "x_std": x_std.astype(np.float32),
                                "error_mean": error_mean.astype(np.float32),
                                "error_std": error_std.astype(np.float32),
                                "candidate_descriptor_names": list(CANDIDATE_DESCRIPTOR_NAMES),
                                "binary_dim": int(binary.shape[1]),
                                "args": vars(args),
                                "latest_metrics": metrics,
                                "best_selection_metric": "eval.gate_passing_delta",
                            },
                            output_root / "checkpoint_best_gate.pt",
                        )
                print(json.dumps(jsonable(metrics), sort_keys=True), flush=True)

            if step == 1 or step - last_save >= int(args.save_every_steps):
                atomic_torch_save(
                    {
                        "model_state_dict": model.state_dict(),
                        "step": int(step),
                        "feature_dim": int(x.shape[1]),
                        "x_mean": x_mean.astype(np.float32),
                        "x_std": x_std.astype(np.float32),
                        "error_mean": error_mean.astype(np.float32),
                        "error_std": error_std.astype(np.float32),
                        "candidate_descriptor_names": list(CANDIDATE_DESCRIPTOR_NAMES),
                        "binary_dim": int(binary.shape[1]),
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
            "step": int(step),
            "feature_dim": int(x.shape[1]),
            "x_mean": x_mean.astype(np.float32),
            "x_std": x_std.astype(np.float32),
            "error_mean": error_mean.astype(np.float32),
            "error_std": error_std.astype(np.float32),
            "candidate_descriptor_names": list(CANDIDATE_DESCRIPTOR_NAMES),
            "binary_dim": int(binary.shape[1]),
            "args": vars(args),
            "latest_metrics": final_metrics,
        },
        final_path,
    )
    elapsed = float(time.time() - start)
    visible_gpus = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    formal_floor_met = bool(
        visible_gpus >= int(args.formal_min_gpus)
        and int(args.min_wall_seconds) >= 3600
        and elapsed >= int(args.min_wall_seconds)
    )
    final_eval = dict(final_metrics.get("eval") or {})
    final_gate = offline_gate_from_eval(final_eval, args)
    best_gate_eval = dict((best_gate_metrics or {}).get("eval") or {})
    best_gate_ready = bool(best_gate_metrics is not None and offline_gate_from_eval(best_gate_eval, args))
    ready_for_formal_live_eval = bool(formal_floor_met and best_gate_ready)
    summary = {
        "schema": "cosmos3_candidate_outcome_scorer_training_summary_v1",
        "output_root": str(output_root),
        "contact_executor_jsonl": str(Path(args.contact_executor_jsonl).resolve()),
        "outcome_jsonl": [str(Path(item).resolve()) for item in args.outcome_jsonl],
        "num_joined_candidate_rows": int(len(meta)),
        "num_base_rows": int(dataset["num_base_rows"]),
        "num_train_candidate_rows": int(len(train_idx)),
        "num_val_candidate_rows": int(len(val_idx)),
        "candidate_family_filter": {
            "allowed_candidate_families": sorted(allowed_candidate_families) if allowed_candidate_families else None,
            "family_counts": dict(
                sorted(
                    Counter(
                        candidate_family(
                            str(item.get("candidate_name") or ""),
                            str(item.get("candidate_source") or ""),
                        )
                        for item in meta
                    ).items()
                )
            ),
        },
        "feature_dim": int(x.shape[1]),
        "binary_dim": int(binary.shape[1]),
        "candidate_descriptor_names": list(CANDIDATE_DESCRIPTOR_NAMES),
        "steps": int(step),
        "elapsed_seconds": elapsed,
        "stop_reason": stop_reason,
        "visible_cuda_device_count": visible_gpus,
        "formal_min_gpus": int(args.formal_min_gpus),
        "formal_training_floor_met": formal_floor_met,
        "rank_loss": {
            "weight": float(args.rank_loss_weight),
            "temperature": float(args.rank_loss_temperature),
            "grouped_rank_batches": bool(args.grouped_rank_batches),
            "active_grouped_rank_batches": bool(use_grouped_rank_batches),
        },
        "prediction_heads": {
            "error": ["final_abs_task_error_weighted"],
            "state": ["final_peg_head_at_hole_x", "final_peg_head_at_hole_y", "final_peg_head_at_hole_z"],
            "progress": ["final_contact_progress_proxy", "final_contact_progress_delta_proxy"],
            "binary": [
                "final_success",
                "dp_rollout_success_or_final_success",
                "final_inserted_live_pose",
                "final_grasped",
                "dp_rollout_continuable_proxy_or_final_continuable_proxy",
            ],
        },
            "score_weights": {
                "success": float(args.score_success_weight),
                "handoff_success": float(args.score_handoff_success_weight),
                "inserted": float(args.score_inserted_weight),
                "grasped": float(args.score_grasped_weight),
                "progress": float(args.score_progress_weight),
                "progress_delta": float(args.score_progress_delta_weight),
                "continuable": float(args.score_continuable_weight),
                "state_abs_axis_weights": fixed_width_vector(args.score_state_abs_axis_weights, 3, 0.0).astype(float).tolist(),
                "state_target": fixed_width_vector(args.score_state_target, 3, 0.0).astype(float).tolist(),
            },
        "loss_weights": {
            "error": float(args.error_loss_weight),
            "state": float(args.state_loss_weight),
            "progress": float(args.progress_loss_weight),
            "binary": float(args.binary_loss_weight),
            "rank": float(args.rank_loss_weight),
            "binary_positive_weights": (
                binary_pos_weight_np.astype(float).tolist() if binary_pos_weight_np is not None else None
            ),
        },
        "final_metrics": final_metrics,
        "final_ready_for_offline_gate": final_gate,
        "best_offline_metrics": best_eval_metrics,
        "best_offline_checkpoint": str(output_root / "checkpoint_best_offline.pt"),
        "best_gate_metrics": best_gate_metrics,
        "best_gate_step": int(best_gate_step),
        "best_gate_checkpoint": str(output_root / "checkpoint_best_gate.pt") if (output_root / "checkpoint_best_gate.pt").is_file() else None,
        "best_gate_ready_for_offline_gate": best_gate_ready,
        "ready_for_formal_live_eval": ready_for_formal_live_eval,
        "formal_live_eval_checkpoint": str(output_root / "checkpoint_best_gate.pt") if ready_for_formal_live_eval else None,
        "offline_gate": {
            "min_selected_error_improvement": float(args.min_selected_error_improvement),
            "min_selected_progress_delta_improvement": float(args.min_selected_progress_delta_improvement),
            "min_selected_handoff_success_improvement": float(args.min_selected_handoff_success_improvement),
            "max_selected_error_degradation_for_handoff_gate": float(
                args.max_selected_error_degradation_for_handoff_gate
            ),
            "min_non_dp_selected_fraction": float(args.min_non_dp_selected_fraction),
            "min_eval_groups_for_gate": int(args.min_eval_groups_for_gate),
            "allow_handoff_only_gate": bool(args.allow_handoff_only_gate),
        },
        "boundary": (
            "Outcome-scorer training only. Passing offline ranking is not live "
            "method evidence. A checkpoint may only drive live control after "
            "the formal GPU/time floor and offline improvement over DP prior; "
            "live evidence still requires real rollout final-state metrics and "
            "video/contact review."
        ),
    }
    write_json(output_root / "training_summary.json", summary)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
