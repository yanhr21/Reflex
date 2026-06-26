#!/usr/bin/env python3
"""Replay candidate executor chunks and record real simulator outcomes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_cosmos3_live_receding_loop import (  # noqa: E402
    apply_external_target_pose,
    _diffusion_candidate_samples,
    live_pose_row,
    load_candidate_executor_checkpoint,
    require_compute_step,
    read_state_obs,
)
from run_cosmos3_receding_closed_loop import (  # noqa: E402
    _action_space_bounds,
    _get_base_env,
    _import_live_control_stack,
    _load_dp_agent,
    _live_eval,
    _make_live_env,
    _parse_seed_from_text,
    _prepare_step_action,
    jsonable,
)
from train_cosmos3_contact_executor import contact_targets_for_horizon  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-executor-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--dp-manifest",
        default=str(ROOT / "experiments/dp_peg1000/run_90201/manifest.json"),
    )
    parser.add_argument(
        "--dp-checkpoint",
        default=str(ROOT / "experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_at_end.pt"),
    )
    parser.add_argument(
        "--dp-state-key",
        choices=("ema_agent", "agent"),
        default="ema_agent",
    )
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--exec-horizon", type=int, default=8)
    parser.add_argument(
        "--dp-rollout-continuability-horizon",
        type=int,
        default=0,
        help=(
            "Optional label-only frozen-DP rollout after each candidate chunk. "
            "Zero preserves the old single-frame continuability proxy."
        ),
    )
    parser.add_argument("--dp-rollout-continuability-min-stable-steps", type=int, default=4)
    parser.add_argument("--contact-stable-min-rel-x", type=float, default=-0.06)
    parser.add_argument("--contact-stable-max-rel-x", type=float, default=0.03)
    parser.add_argument("--contact-stable-max-abs-y", type=float, default=0.018)
    parser.add_argument("--contact-stable-max-abs-z", type=float, default=0.012)
    parser.add_argument("--candidate-scales", default="0.05,0.1,0.2,0.5,1.0")
    parser.add_argument(
        "--candidate-short-prefix-steps",
        default="",
        help=(
            "Optional comma-separated executable prefix lengths. For each "
            "non-DP candidate, export an extra fixed-width candidate whose "
            "first N actions come from that candidate and whose suffix comes "
            "from the DP prior, but only execute the first N actions before "
            "the DP-rollout label. This matches the live short-chunk "
            "reobserve diagnostic."
        ),
    )
    parser.add_argument("--candidate-executor-checkpoint", default="")
    parser.add_argument(
        "--model-candidate-samples",
        type=int,
        default=-1,
        help="Override checkpoint candidate_samples when exporting model-generated candidates.",
    )
    parser.add_argument(
        "--model-candidate-scales",
        default="",
        help="Override checkpoint candidate scales for model-generated candidates.",
    )
    parser.add_argument("--model-candidate-temps", default="")
    parser.add_argument("--include-legacy-teacher-scale-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-retrieval-residual-candidates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--retrieval-source-jsonl", default="")
    parser.add_argument("--retrieval-k", type=int, default=4)
    parser.add_argument(
        "--retrieval-positive-fields",
        default="future_inserted_within_chunk,future_dp_continuable_within_chunk",
    )
    parser.add_argument("--retrieval-phase-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retrieval-residual-scales", default="1.0")
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--robot-action-dim", type=int, default=7)
    parser.add_argument("--clip-live-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--external-target-mode",
        choices=("source_env_state", "none"),
        default="source_env_state",
    )
    parser.add_argument("--seed", type=int, default=20260617)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
    os.replace(tmp, path)


def parse_scales(text: str) -> list[float]:
    out: list[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0:
            raise ValueError(f"candidate scale must be non-negative: {value}")
        out.append(value)
    return out or [0.05, 0.1, 0.2, 0.5, 1.0]


def parse_positive_ints(text: str) -> list[int]:
    out: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"expected positive integer, got {value}")
        if value not in out:
            out.append(value)
    return out


def is_dp_prior_candidate_name(name: str) -> bool:
    base = re.sub(r"^short\d+_", "", str(name))
    return base in {"dp_prior", "model_dp_prior"}


def contact_progress_proxy(
    rel: np.ndarray,
    *,
    grasped: bool,
    inserted: bool,
) -> float:
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


def contact_stable_proxy(
    rel: np.ndarray,
    *,
    grasped: bool,
    inserted: bool,
    args: argparse.Namespace,
) -> bool:
    rel = np.asarray(rel, dtype=np.float32).reshape(-1)
    if rel.size < 3:
        rel = np.pad(rel, (0, 3 - rel.size), mode="constant")
    if bool(inserted):
        return True
    x, y, z = float(rel[0]), float(rel[1]), float(rel[2])
    return bool(
        bool(grasped)
        and x >= float(args.contact_stable_min_rel_x)
        and x <= float(args.contact_stable_max_rel_x)
        and abs(y) <= float(args.contact_stable_max_abs_y)
        and abs(z) <= float(args.contact_stable_max_abs_z)
    )


def truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def retrieval_vector(row: dict[str, Any]) -> np.ndarray:
    rel = np.asarray(row.get("current_peg_head_at_hole") or [0.0, 0.0, 0.0], dtype=np.float32).reshape(-1)
    if rel.size < 3:
        rel = np.pad(rel, (0, 3 - rel.size), mode="constant")
    progress = float(row.get("current_contact_progress") or 0.0)
    phase_id = float(row.get("current_phase_id") or 0.0)
    return np.asarray([rel[0], 2.0 * rel[1], 4.0 * rel[2], progress, 0.05 * phase_id], dtype=np.float32)


def build_retrieval_bank(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    positive_fields = [item.strip() for item in str(args.retrieval_positive_fields).split(",") if item.strip()]
    bank: list[dict[str, Any]] = []
    for row in rows:
        if positive_fields and not any(truthy(row.get(field)) for field in positive_fields):
            continue
        try:
            executor_npz = np.load(str(row["executor_sample_npz"]), allow_pickle=False)
            prior_npz = np.load(str(row["dp_prior_npz"]), allow_pickle=False)
            teacher = np.asarray(executor_npz["teacher_robot_actions"], dtype=np.float32)[:, :7]
            prior = np.asarray(prior_npz["dp_prior_actions"], dtype=np.float32)[:, :7]
            horizon = min(int(args.exec_horizon), int(teacher.shape[0]), int(prior.shape[0]))
            if horizon <= 0:
                continue
            bank.append(
                {
                    "uuid": str(row.get("uuid") or ""),
                    "source_uuid": str(row.get("source_uuid") or ""),
                    "scenario": row.get("scenario"),
                    "current_phase": str(row.get("current_phase") or "unknown"),
                    "feature": retrieval_vector(row),
                    "residual": (teacher[:horizon] - prior[:horizon]).astype(np.float32),
                    "horizon": int(horizon),
                    "positive_fields": {field: truthy(row.get(field)) for field in positive_fields},
                }
            )
        except Exception:
            continue
    return bank


def load_env_states(source_h5: Path, trajectory_utils: Any) -> list[dict[str, Any]]:
    import h5py

    with h5py.File(source_h5, "r") as h5:
        traj_name = "traj_0" if "traj_0" in h5 else next(iter(h5.keys()))
        return trajectory_utils.dict_to_list_of_dicts(h5[traj_name]["env_states"])


def array_stats(arr: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    return {
        "finite": bool(np.isfinite(flat).all()),
        "num_values": int(flat.size),
        "min": float(np.min(flat)) if flat.size else None,
        "max": float(np.max(flat)) if flat.size else None,
        "mean_abs": float(np.mean(np.abs(flat))) if flat.size else None,
        "max_abs": float(np.max(np.abs(flat))) if flat.size else None,
    }


def candidate_actions(row: dict[str, Any], args: argparse.Namespace) -> tuple[list[tuple[str, np.ndarray]], int]:
    executor_npz = np.load(str(row["executor_sample_npz"]), allow_pickle=False)
    prior_npz = np.load(str(row["dp_prior_npz"]), allow_pickle=False)
    teacher = np.asarray(executor_npz["teacher_robot_actions"], dtype=np.float32)[:, :7]
    prior = np.asarray(prior_npz["dp_prior_actions"], dtype=np.float32)[:, :7]
    horizon = min(int(args.exec_horizon), int(teacher.shape[0]), int(prior.shape[0]))
    if horizon <= 0:
        raise ValueError("empty teacher/prior horizon")
    teacher = teacher[:horizon]
    prior = prior[:horizon]
    residual = teacher - prior
    candidates: list[tuple[str, np.ndarray]] = [
        ("dp_prior", prior),
        ("teacher", teacher),
    ]
    for scale in parse_scales(str(args.candidate_scales)):
        candidates.append((f"scale_{scale:g}", prior + float(scale) * residual))
    return candidates, horizon


def generated_candidate_actions(
    row: dict[str, Any],
    args: argparse.Namespace,
    bundle: dict[str, Any],
) -> tuple[list[tuple[str, np.ndarray, dict[str, Any]]], int]:
    import torch

    executor_npz = np.load(str(row["executor_sample_npz"]), allow_pickle=False)
    prior_npz = np.load(str(row["dp_prior_npz"]), allow_pickle=False)
    current = np.asarray(executor_npz["current_state"], dtype=np.float32).reshape(-1)
    task_path = np.asarray(executor_npz["task_path"], dtype=np.float32)
    prior = np.asarray(prior_npz["dp_prior_actions"], dtype=np.float32)[:, :7]
    teacher = np.asarray(executor_npz["teacher_robot_actions"], dtype=np.float32)[:, :7]
    horizon = min(int(args.exec_horizon), int(bundle["horizon"]), int(task_path.shape[0]), int(prior.shape[0]), int(teacher.shape[0]))
    if horizon <= 0:
        raise ValueError("empty model-generated candidate horizon")
    contact_context, _progress_target, _binary_target = contact_targets_for_horizon(row, horizon)
    task_path = task_path[:horizon]
    prior = prior[:horizon]
    prior_flat = prior.reshape(-1).astype(np.float32)
    feature = np.concatenate([current, task_path.reshape(-1), prior_flat, contact_context]).astype(np.float32)
    if feature.shape[0] != int(bundle["feature_dim"]):
        raise RuntimeError(f"model candidate feature width {feature.shape[0]} != checkpoint feature_dim {bundle['feature_dim']}")

    device = bundle["device"]
    model = bundle["model"]
    x = ((feature - np.asarray(bundle["x_mean"], dtype=np.float32)) / np.asarray(bundle["x_std"], dtype=np.float32)).astype(np.float32)
    raw_mean = torch.as_tensor(bundle["residual_mean"], device=device, dtype=torch.float32).reshape(1, -1)
    raw_std = torch.as_tensor(bundle["residual_std"], device=device, dtype=torch.float32).reshape(1, -1)
    phase = str(row.get("current_phase") or "unknown")
    phase_caps = dict(bundle.get("phase_residual_l2_caps") or {})
    cap_value = float(
        phase_caps.get(
            phase,
            phase_caps.get("__global__", float(bundle.get("selector_residual_l2_cap_max", 0.02))),
        )
    )
    with torch.no_grad():
        z = model.encode(torch.as_tensor(x[None], device=device, dtype=torch.float32))
        mean_norm, logstd = model.distribution(z)
        std = torch.exp(logstd)
        zero_resid_norm = (torch.zeros_like(mean_norm) - raw_mean) / raw_std
        candidates: list[tuple[str, Any]] = [("model_dp_prior", zero_resid_norm), ("model_mean", mean_norm)]
        scale_values = (
            parse_scales(str(args.model_candidate_scales))
            if str(args.model_candidate_scales).strip()
            else list(bundle.get("candidate_scales") or [0.05, 0.1, 0.2])
        )
        for scale in scale_values:
            raw = float(scale) * (mean_norm * raw_std + raw_mean)
            candidates.append((f"model_scale_{scale:g}", (raw - raw_mean) / raw_std))
        generator = torch.Generator(device=device)
        generator.manual_seed(int(bundle.get("seed", 20260615)) + int(row.get("prefix_frame_index") or 0) * 1009)
        sample_count = (
            int(args.model_candidate_samples)
            if int(args.model_candidate_samples) >= 0
            else int(bundle.get("candidate_samples", 0))
        )
        if str(bundle.get("generator_type", "gaussian")) == "diffusion":
            for name, cand_norm in _diffusion_candidate_samples(
                model=model,
                z=z,
                sample_count=sample_count,
                steps=int(bundle.get("diffusion_steps", 16)),
                beta_start=float(bundle.get("diffusion_beta_start", 1e-4)),
                beta_end=float(bundle.get("diffusion_beta_end", 2e-2)),
                generator=generator,
                device=device,
            ):
                candidates.append((f"model_{name}", cand_norm))
        elif sample_count > 0:
            temps = (
                parse_scales(str(args.model_candidate_temps))
                if str(args.model_candidate_temps).strip()
                else list(bundle.get("candidate_temps") or [1.0])
            )
            samples_per_temp = max(1, sample_count // max(1, len(temps)))
            for temp in temps:
                for sample_idx in range(samples_per_temp):
                    noise = torch.randn(mean_norm.shape, generator=generator, device=device)
                    candidates.append((f"model_sample_t{float(temp):g}_{sample_idx}", mean_norm + float(temp) * std * noise))

        out: list[tuple[str, np.ndarray, dict[str, Any]]] = []
        for name, cand_norm in candidates:
            raw_resid = cand_norm * raw_std + raw_mean
            penalty = float(torch.mean(raw_resid**2, dim=1).detach().cpu().numpy()[0])
            actions = (torch.as_tensor(prior_flat, device=device, dtype=torch.float32).reshape(1, -1) + raw_resid)
            actions_np = actions.detach().cpu().numpy().reshape(horizon, 7).astype(np.float32)
            out.append(
                (
                    name,
                    actions_np,
                    {
                        "candidate_source": "checkpoint_model",
                        "selector_residual_l2_penalty_input": penalty,
                        "selector_residual_l2_cap": cap_value,
                        "selector_over_cap": bool(name != "model_dp_prior" and penalty > cap_value),
                    },
                )
            )
    return out, horizon


def retrieval_residual_candidate_actions(
    row: dict[str, Any],
    args: argparse.Namespace,
    bank: list[dict[str, Any]],
) -> tuple[list[tuple[str, np.ndarray, dict[str, Any]]], int]:
    if not bank or int(args.retrieval_k) <= 0:
        return [], 0
    prior_npz = np.load(str(row["dp_prior_npz"]), allow_pickle=False)
    prior = np.asarray(prior_npz["dp_prior_actions"], dtype=np.float32)[:, :7]
    query_phase = str(row.get("current_phase") or "unknown")
    query_uuid = str(row.get("uuid") or "")
    query_feature = retrieval_vector(row)
    same_phase_available = any(
        item["current_phase"] == query_phase and item["uuid"] != query_uuid
        for item in bank
    )

    scored: list[tuple[float, dict[str, Any]]] = []
    for item in bank:
        if item["uuid"] == query_uuid:
            continue
        if bool(args.retrieval_phase_match) and same_phase_available and item["current_phase"] != query_phase:
            continue
        dist = float(np.linalg.norm(query_feature - np.asarray(item["feature"], dtype=np.float32)))
        scored.append((dist, item))
    scored.sort(key=lambda pair: pair[0])
    if not scored:
        return [], 0

    out: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    horizons: list[int] = []
    residual_scales = parse_scales(str(args.retrieval_residual_scales))
    for rank, (dist, item) in enumerate(scored[: int(args.retrieval_k)]):
        horizon = min(int(args.exec_horizon), int(prior.shape[0]), int(item["horizon"]))
        if horizon <= 0:
            continue
        horizons.append(horizon)
        residual = np.asarray(item["residual"], dtype=np.float32)[:horizon]
        for scale in residual_scales:
            actions = prior[:horizon] + float(scale) * residual
            scale_tag = f"{float(scale):g}".replace(".", "p")
            out.append(
                (
                    f"retrieval_resid_r{rank}_s{scale_tag}",
                    actions.astype(np.float32),
                    {
                        "candidate_source": "retrieval_success_residual",
                        "retrieval_rank": int(rank),
                        "retrieval_residual_scale": float(scale),
                        "retrieval_distance": float(dist),
                        "retrieval_uuid": item["uuid"],
                        "retrieval_source_uuid": item["source_uuid"],
                        "retrieval_scenario": item["scenario"],
                        "retrieval_phase": item["current_phase"],
                        "retrieval_positive_fields": item["positive_fields"],
                    },
                )
            )
    return out, min(horizons) if horizons else 0


def replay_candidate(
    *,
    env: Any,
    base_env: Any,
    stack: dict[str, Any],
    env_states: list[dict[str, Any]],
    prefix_frame: int,
    actions: np.ndarray,
    execute_actions: np.ndarray | None = None,
    low: Any,
    high: Any,
    args: argparse.Namespace,
    dp_agent: Any | None = None,
    dp_args: Any | None = None,
) -> dict[str, Any]:
    base_env.set_state_dict(env_states[int(prefix_frame)])
    start_live = live_pose_row(base_env, stack, None)
    previous_hole_xyz = np.asarray(start_live["hole_xyz"], dtype=np.float32).copy()
    start_state_obs = read_state_obs(base_env, stack)
    dp_obs_history: list[np.ndarray] = [start_state_obs.copy(), start_state_obs.copy()]
    start_eval = _live_eval(base_env)
    step_records: list[dict[str, Any]] = []
    last_live = start_live
    replay_actions = np.asarray(actions if execute_actions is None else execute_actions, dtype=np.float32)
    for local_i, action in enumerate(replay_actions):
        step_action, action_record = _prepare_step_action(
            action,
            low,
            high,
            bool(args.clip_live_actions),
        )
        _obs, reward, terminated, truncated, _info = env.step(step_action)
        source_frame = min(int(prefix_frame) + local_i + 1, len(env_states) - 1)
        external_target = apply_external_target_pose(
            base_env=base_env,
            stack=stack,
            env_states=env_states,
            source_frame=source_frame,
            args=args,
        )
        current_state_obs = read_state_obs(base_env, stack)
        dp_obs_history.append(current_state_obs.copy())
        last_live = live_pose_row(base_env, stack, previous_hole_xyz)
        previous_hole_xyz = np.asarray(last_live["hole_xyz"], dtype=np.float32).copy()
        step_eval = _live_eval(base_env)
        step_records.append(
            {
                "local_step": int(local_i),
                "source_frame": int(source_frame),
                "action": action_record,
                "external_target": external_target,
                "reward": jsonable(reward),
                "terminated": jsonable(terminated),
                "truncated": jsonable(truncated),
                "eval": step_eval,
                "peg_head_at_hole": np.asarray(last_live["peg_head_at_hole"], dtype=np.float32)[:3].astype(float).tolist(),
                "grasped": bool(last_live.get("grasped", False)),
                "inserted": bool(last_live.get("inserted", False)),
            }
        )
        if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
            break
    final_eval = _live_eval(base_env)
    rel = np.asarray(last_live["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
    final_grasped = bool(last_live.get("grasped", False))
    final_inserted = bool(last_live.get("inserted", False))
    final_contact_stable = contact_stable_proxy(
        rel,
        grasped=final_grasped,
        inserted=final_inserted,
        args=args,
    )
    outcome = {
        "start_eval": start_eval,
        "final_eval": final_eval,
        "final_peg_head_at_hole": rel.astype(float).tolist(),
        "final_abs_task_error_weighted": float(abs(rel[0]) + 2.0 * abs(rel[1]) + 4.0 * abs(rel[2])),
        "final_lateral_error_yz": float(np.linalg.norm(rel[1:3])),
        "final_grasped": final_grasped,
        "final_inserted_live_pose": final_inserted,
        "final_contact_stable_proxy": bool(final_contact_stable),
        "final_success": bool(final_eval.get("success", False)),
        "executed_steps": int(len(step_records)),
        "step_records": step_records,
    }
    if int(args.dp_rollout_continuability_horizon) > 0:
        if dp_agent is None or dp_args is None:
            raise RuntimeError("dp_rollout_continuability_horizon > 0 requires a loaded DP agent")
        outcome["dp_rollout_continuability"] = run_dp_rollout_continuability_label(
            env=env,
            base_env=base_env,
            stack=stack,
            env_states=env_states,
            start_frame=int(prefix_frame) + int(len(step_records)),
            previous_hole_xyz=previous_hole_xyz,
            dp_obs_history=dp_obs_history,
            dp_agent=dp_agent,
            dp_args=dp_args,
            low=low,
            high=high,
            args=args,
        )
        dp_label = outcome["dp_rollout_continuability"]
        outcome["dp_rollout_continuable_proxy"] = bool(dp_label.get("continuable", False))
        outcome["dp_rollout_success"] = bool(dp_label.get("success", False))
        outcome["dp_rollout_final_contact_stable_proxy"] = bool(dp_label.get("final_contact_stable", False))
        outcome["dp_rollout_stable_step_count"] = int(dp_label.get("stable_step_count", 0))
    return outcome


def run_dp_rollout_continuability_label(
    *,
    env: Any,
    base_env: Any,
    stack: dict[str, Any],
    env_states: list[dict[str, Any]],
    start_frame: int,
    previous_hole_xyz: np.ndarray,
    dp_obs_history: list[np.ndarray],
    dp_agent: Any,
    dp_args: Any,
    low: Any,
    high: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    torch = stack["torch"]
    horizon = max(0, int(args.dp_rollout_continuability_horizon))
    current_frame = int(start_frame)
    requested_horizon = int(horizon)
    stable_steps = 0
    step_records: list[dict[str, Any]] = []
    last_live = live_pose_row(base_env, stack, previous_hole_xyz)
    dp_device = next(dp_agent.parameters()).device
    dp_call_index = 0
    stop_reason = "horizon_exhausted"
    while len(step_records) < requested_horizon and current_frame < min(int(args.max_episode_steps), len(env_states) - 1):
        obs_seq = np.stack(dp_obs_history[-2:], axis=0)[None].astype(np.float32)
        obs_tensor = torch.as_tensor(obs_seq, device=dp_device, dtype=torch.float32)
        with torch.no_grad():
            action_seq = dp_agent.get_action(obs_tensor)
        if getattr(dp_args, "sim_backend", "physx_cpu") == "physx_cpu" or hasattr(action_seq, "detach"):
            action_seq_np = action_seq.detach().cpu().numpy()
        else:
            action_seq_np = np.asarray(action_seq)
        act_horizon = int(action_seq_np.shape[1])
        if act_horizon <= 0:
            stop_reason = "empty_dp_action_sequence"
            break
        for chunk_local_i in range(act_horizon):
            if len(step_records) >= requested_horizon or current_frame >= min(int(args.max_episode_steps), len(env_states) - 1):
                break
            step_action, action_record = _prepare_step_action(
                action_seq_np[:, chunk_local_i],
                low,
                high,
                bool(args.clip_live_actions),
            )
            _obs, reward, terminated, truncated, _info = env.step(step_action)
            external_target = apply_external_target_pose(
                base_env=base_env,
                stack=stack,
                env_states=env_states,
                source_frame=current_frame + 1,
                args=args,
            )
            current_state_obs = read_state_obs(base_env, stack)
            dp_obs_history.append(current_state_obs.copy())
            last_live = live_pose_row(base_env, stack, previous_hole_xyz)
            previous_hole_xyz = np.asarray(last_live["hole_xyz"], dtype=np.float32).copy()
            current_frame += 1
            step_eval = _live_eval(base_env)
            rel = np.asarray(last_live["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
            stable = contact_stable_proxy(
                rel,
                grasped=bool(last_live.get("grasped", False)),
                inserted=bool(last_live.get("inserted", False)),
                args=args,
            )
            if stable:
                stable_steps += 1
            step_records.append(
                {
                    "local_step": int(len(step_records)),
                    "global_action_index": int(current_frame - 1),
                    "dp_call_index": int(dp_call_index),
                    "chunk_local_step": int(chunk_local_i),
                    "action": action_record,
                    "external_target": external_target,
                    "reward": jsonable(reward),
                    "terminated": jsonable(terminated),
                    "truncated": jsonable(truncated),
                    "live_eval": step_eval,
                    "peg_head_at_hole": rel.astype(float).tolist(),
                    "grasped": bool(last_live.get("grasped", False)),
                    "inserted": bool(last_live.get("inserted", False)),
                    "contact_stable_proxy": bool(stable),
                }
            )
            if bool(step_eval.get("success", False)):
                stop_reason = "success"
                break
            if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                stop_reason = "terminated_or_truncated"
                break
        if stop_reason != "horizon_exhausted":
            break
        dp_call_index += 1
    final_eval = _live_eval(base_env)
    final_rel = np.asarray(last_live["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
    final_contact_stable = contact_stable_proxy(
        final_rel,
        grasped=bool(last_live.get("grasped", False)),
        inserted=bool(last_live.get("inserted", False)),
        args=args,
    )
    success = bool(final_eval.get("success", False) or last_live.get("inserted", False))
    min_stable = max(1, int(args.dp_rollout_continuability_min_stable_steps))
    continuable = bool(success or (stable_steps >= min_stable and final_contact_stable))
    return {
        "schema": "cosmos3_candidate_dp_rollout_continuability_label_v1",
        "requested_horizon": requested_horizon,
        "executed_steps": int(len(step_records)),
        "min_stable_steps": int(min_stable),
        "stable_step_count": int(stable_steps),
        "success": bool(success),
        "continuable": bool(continuable),
        "stop_reason": stop_reason,
        "final_eval": final_eval,
        "final_peg_head_at_hole": final_rel.astype(float).tolist(),
        "final_grasped": bool(last_live.get("grasped", False)),
        "final_inserted_live_pose": bool(last_live.get("inserted", False)),
        "final_contact_stable": bool(final_contact_stable),
        "step_records": step_records,
        "boundary": (
            "Label-only short frozen-DP rollout after a candidate chunk. It "
            "tests whether the candidate produced a state the real DP can "
            "keep stable or finish from. This is training/evaluation label "
            "generation, not a privileged controller input."
        ),
    }


def main() -> int:
    args = parse_args()
    require_compute_step()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(args.contact_executor_jsonl).resolve())
    if not rows:
        raise SystemExit("empty contact executor jsonl")

    stack = _import_live_control_stack(ROOT)
    env = _make_live_env(stack, Path(args.dp_manifest).resolve(), args)
    base_env = _get_base_env(env)
    low, high = _action_space_bounds(env, int(args.robot_action_dim))
    dp_agent = None
    dp_args = None
    if int(args.dp_rollout_continuability_horizon) > 0:
        torch = stack["torch"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dp_agent, dp_args = _load_dp_agent(env, stack, args, device)
    candidate_executor_bundle = None
    if str(args.candidate_executor_checkpoint).strip():
        candidate_executor_bundle = load_candidate_executor_checkpoint(
            Path(args.candidate_executor_checkpoint).resolve(),
            int(args.robot_action_dim),
        )
    retrieval_bank = None
    if bool(args.include_retrieval_residual_candidates):
        retrieval_source = Path(args.retrieval_source_jsonl or args.contact_executor_jsonl).resolve()
        retrieval_bank = build_retrieval_bank(read_jsonl(retrieval_source), args)
        write_json(
            output_root / "retrieval_bank_summary.json",
            {
                "schema": "cosmos3_candidate_retrieval_bank_summary_v1",
                "retrieval_source_jsonl": str(retrieval_source),
                "retrieval_k": int(args.retrieval_k),
                "retrieval_positive_fields": [
                    item.strip()
                    for item in str(args.retrieval_positive_fields).split(",")
                    if item.strip()
                ],
                "retrieval_phase_match": bool(args.retrieval_phase_match),
                "retrieval_residual_scales": parse_scales(str(args.retrieval_residual_scales)),
                "num_bank_rows": int(len(retrieval_bank)),
                "phase_counts": dict(sorted(Counter(item["current_phase"] for item in retrieval_bank).items())),
                "boundary": (
                    "Retrieval candidates reuse residual chunks from successful or DP-continuable "
                    "phase/contact neighbors as an action-prior source. This is a candidate-distribution "
                    "debug input, not controller evidence by itself."
                ),
            },
        )

    records: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    source_cache: dict[str, list[dict[str, Any]]] = {}
    action_dir = output_root / "candidate_actions"

    try:
        for row_idx, row in enumerate(rows):
            if int(args.max_samples) > 0 and row_idx >= int(args.max_samples):
                break
            try:
                source_h5 = Path(str(row["source_h5"])).resolve()
                source_key = str(source_h5)
                if source_key not in source_cache:
                    source_cache[source_key] = load_env_states(source_h5, stack["trajectory_utils"])
                env_states = source_cache[source_key]
                prefix_frame = int(row.get("prefix_frame_index", -1))
                if prefix_frame < 0 or prefix_frame >= len(env_states):
                    raise ValueError(f"bad prefix_frame_index={prefix_frame}")
                reset_seed = _parse_seed_from_text(" ".join([str(row.get("source_uuid", "")), source_h5.name]))
                env.reset(seed=int(reset_seed if reset_seed is not None else args.seed))
                candidates_ext: list[tuple[str, np.ndarray, dict[str, Any]]] = []
                horizon_values: list[int] = []
                if bool(args.include_legacy_teacher_scale_candidates):
                    candidates, horizon = candidate_actions(row, args)
                    horizon_values.append(int(horizon))
                    candidates_ext.extend(
                        (cand_name, actions, {"candidate_source": "legacy_teacher_scale"})
                        for cand_name, actions in candidates
                    )
                if candidate_executor_bundle is not None:
                    generated, horizon = generated_candidate_actions(row, args, candidate_executor_bundle)
                    horizon_values.append(int(horizon))
                    candidates_ext.extend(generated)
                if retrieval_bank is not None:
                    retrieved, horizon = retrieval_residual_candidate_actions(row, args, retrieval_bank)
                    if retrieved:
                        horizon_values.append(int(horizon))
                        candidates_ext.extend(retrieved)
                if not candidates_ext:
                    raise ValueError("no candidates requested")
                horizon = min(horizon_values)
                prior_actions = np.asarray(
                    np.load(str(row["dp_prior_npz"]), allow_pickle=False)["dp_prior_actions"],
                    dtype=np.float32,
                )[:horizon, : int(args.robot_action_dim)]
                short_prefix_steps = [
                    step
                    for step in parse_positive_ints(str(args.candidate_short_prefix_steps))
                    if 0 < step < int(horizon)
                ]
                replay_candidates: list[tuple[str, np.ndarray, np.ndarray | None, dict[str, Any]]] = []
                for cand_name, actions, candidate_meta in candidates_ext:
                    actions = np.asarray(actions, dtype=np.float32)[:horizon]
                    base_meta = {
                        **candidate_meta,
                        "candidate_score_horizon": int(horizon),
                        "candidate_execute_steps": int(horizon),
                        "candidate_short_prefix_steps": 0,
                        "candidate_prefix_base_name": cand_name,
                        "candidate_action_feature_source": "full_candidate",
                    }
                    replay_candidates.append((cand_name, actions, None, base_meta))
                    if short_prefix_steps and not is_dp_prior_candidate_name(cand_name):
                        for prefix_steps in short_prefix_steps:
                            score_actions = prior_actions.copy()
                            score_actions[:prefix_steps] = actions[:prefix_steps]
                            short_meta = {
                                **candidate_meta,
                                "candidate_score_horizon": int(horizon),
                                "candidate_execute_steps": int(prefix_steps),
                                "candidate_short_prefix_steps": int(prefix_steps),
                                "candidate_prefix_base_name": cand_name,
                                "candidate_action_feature_source": "prefix_candidate_suffix_dp_prior",
                            }
                            replay_candidates.append(
                                (
                                    f"short{int(prefix_steps)}_{cand_name}",
                                    score_actions.astype(np.float32),
                                    score_actions[:prefix_steps].astype(np.float32),
                                    short_meta,
                                )
                            )

                for cand_name, actions, execute_actions, candidate_meta in replay_candidates:
                    action_rel = action_dir / f"{row_idx:06d}_{row['uuid']}__{cand_name}.npz"
                    action_path = output_root / action_rel
                    action_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(action_path, actions=np.asarray(actions, dtype=np.float32))
                    outcome = replay_candidate(
                        env=env,
                        base_env=base_env,
                        stack=stack,
                        env_states=env_states,
                        prefix_frame=prefix_frame,
                        actions=actions,
                        execute_actions=execute_actions,
                        low=low,
                        high=high,
                        args=args,
                        dp_agent=dp_agent,
                        dp_args=dp_args,
                    )
                    final_progress = contact_progress_proxy(
                        np.asarray(outcome["final_peg_head_at_hole"], dtype=np.float32),
                        grasped=bool(outcome.get("final_grasped", False)),
                        inserted=bool(outcome.get("final_inserted_live_pose", False)),
                    )
                    current_progress = float(row.get("current_contact_progress") or 0.0)
                    outcome["final_contact_progress_proxy"] = float(final_progress)
                    outcome["final_contact_progress_delta_proxy"] = float(final_progress - current_progress)
                    outcome["final_continuable_proxy"] = bool(
                        outcome.get("dp_rollout_continuable_proxy", False)
                        or outcome.get("final_success", False)
                        or outcome.get("final_inserted_live_pose", False)
                        or (
                            int(args.dp_rollout_continuability_horizon) <= 0
                            and bool(outcome.get("final_grasped", False))
                            and bool(outcome.get("final_contact_stable_proxy", False))
                            and final_progress >= 0.75
                        )
                    )
                    records.append(
                        {
                            "schema": "cosmos3_candidate_outcome_label_v1",
                            "uuid": row.get("uuid"),
                            "source_uuid": row.get("source_uuid"),
                            "source_h5": str(source_h5),
                            "scenario": row.get("scenario"),
                            "split": row.get("split"),
                            "prefix_role": row.get("prefix_role"),
                            "current_phase": row.get("current_phase"),
                            "prefix_frame_index": int(prefix_frame),
                            "horizon": int(horizon),
                            "candidate_name": cand_name,
                            **candidate_meta,
                            "candidate_actions_npz": str(action_path),
                            "candidate_actions_npz_rel": str(action_rel),
                            "candidate_action_stats": array_stats(actions),
                            **outcome,
                            "boundary": (
                                "Real simulator outcome label for one candidate short action chunk. "
                                "This may supervise an action-conditioned scorer, but it is not a "
                                "controller input and not method evidence by itself."
                            ),
                        }
                    )
            except Exception as exc:
                failures[type(exc).__name__] += 1
                records.append(
                    {
                        "schema": "cosmos3_candidate_outcome_label_failure_v1",
                        "uuid": row.get("uuid"),
                        "row_index": int(row_idx),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    finally:
        if hasattr(env, "close"):
            env.close()

    write_jsonl(output_root / "candidate_outcome_labels.jsonl", records)
    success_records = [r for r in records if r.get("schema") == "cosmos3_candidate_outcome_label_v1"]
    summary = {
        "schema": "cosmos3_candidate_outcome_export_summary_v1",
        "contact_executor_jsonl": str(Path(args.contact_executor_jsonl).resolve()),
        "output_root": str(output_root),
        "max_samples": int(args.max_samples),
        "exec_horizon": int(args.exec_horizon),
        "dp_rollout_continuability_horizon": int(args.dp_rollout_continuability_horizon),
        "dp_rollout_continuability_min_stable_steps": int(args.dp_rollout_continuability_min_stable_steps),
        "contact_stable_thresholds": {
            "min_rel_x": float(args.contact_stable_min_rel_x),
            "max_rel_x": float(args.contact_stable_max_rel_x),
            "max_abs_y": float(args.contact_stable_max_abs_y),
            "max_abs_z": float(args.contact_stable_max_abs_z),
        },
        "candidate_scales": parse_scales(str(args.candidate_scales)),
        "candidate_short_prefix_steps": parse_positive_ints(str(args.candidate_short_prefix_steps)),
        "candidate_executor_checkpoint": (
            str(Path(args.candidate_executor_checkpoint).resolve())
            if str(args.candidate_executor_checkpoint).strip()
            else None
        ),
        "model_candidate_samples": int(args.model_candidate_samples),
        "model_candidate_scales": (
            parse_scales(str(args.model_candidate_scales))
            if str(args.model_candidate_scales).strip()
            else None
        ),
        "include_legacy_teacher_scale_candidates": bool(args.include_legacy_teacher_scale_candidates),
        "include_retrieval_residual_candidates": bool(args.include_retrieval_residual_candidates),
        "retrieval_source_jsonl": (
            str(Path(args.retrieval_source_jsonl or args.contact_executor_jsonl).resolve())
            if bool(args.include_retrieval_residual_candidates)
            else None
        ),
        "retrieval_k": int(args.retrieval_k),
        "retrieval_positive_fields": str(args.retrieval_positive_fields),
        "retrieval_phase_match": bool(args.retrieval_phase_match),
        "retrieval_residual_scales": parse_scales(str(args.retrieval_residual_scales)),
        "num_records": int(len(records)),
        "num_successful_outcome_records": int(len(success_records)),
        "num_failed_rows": int(sum(failures.values())),
        "failure_counts": dict(sorted(failures.items())),
        "final_contact_stable_count": int(
            sum(1 for r in success_records if bool(r.get("final_contact_stable_proxy", False)))
        ),
        "final_continuable_proxy_count": int(
            sum(1 for r in success_records if bool(r.get("final_continuable_proxy", False)))
        ),
        "dp_rollout_continuable_proxy_count": int(
            sum(1 for r in success_records if bool(r.get("dp_rollout_continuable_proxy", False)))
        ),
        "dp_rollout_success_count": int(
            sum(1 for r in success_records if bool(r.get("dp_rollout_success", False)))
        ),
        "success_by_candidate": dict(
            sorted(
                Counter(
                    str(r.get("candidate_name"))
                    for r in success_records
                    if bool(r.get("final_success", False))
                ).items()
            )
        ),
        "short_prefix_record_count": int(
            sum(int(r.get("candidate_short_prefix_steps") or 0) > 0 for r in success_records)
        ),
        "short_prefix_success_count": int(
            sum(
                int(r.get("candidate_short_prefix_steps") or 0) > 0
                and bool(r.get("final_success", False))
                for r in success_records
            )
        ),
        "short_prefix_dp_rollout_continuable_count": int(
            sum(
                int(r.get("candidate_short_prefix_steps") or 0) > 0
                and bool(r.get("dp_rollout_continuable_proxy", False))
                for r in success_records
            )
        ),
        "ready_for_outcome_scorer_smoke": bool(success_records and not failures),
        "boundary": (
            "Outcome export only. It restores saved source prefix states and "
            "executes short candidate chunks in the real simulator so the next "
            "scorer can learn candidate-conditioned physical outcomes instead "
            "of ungrounded readout penalties."
        ),
    }
    write_json(output_root / "candidate_outcome_export_summary.json", summary)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True), flush=True)
    return 0 if summary["ready_for_outcome_scorer_smoke"] else 64


if __name__ == "__main__":
    raise SystemExit(main())
