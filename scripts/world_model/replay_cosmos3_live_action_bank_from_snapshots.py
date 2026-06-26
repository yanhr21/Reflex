#!/usr/bin/env python3
"""Replay saved live action-bank candidates from live simulator snapshots.

This is a label/calibration tool. It does not feed snapshot state to the
controller and does not produce method evidence by itself.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_cosmos3_candidate_outcome_labels import (  # noqa: E402
    contact_progress_proxy,
    contact_stable_proxy,
    run_dp_rollout_continuability_label,
)
from run_cosmos3_live_receding_loop import (  # noqa: E402
    apply_external_target_pose,
    continuability_gate,
    fill_live_history_row,
    live_pose_row,
    read_state_obs,
    require_compute_step,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--dp-manifest",
        default=str(ROOT / "experiments/dp_peg1000/run_90201/manifest.json"),
    )
    parser.add_argument(
        "--dp-checkpoint",
        default=str(ROOT / "experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_at_end.pt"),
    )
    parser.add_argument("--dp-state-key", choices=("ema_agent", "agent"), default="ema_agent")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--max-iter-dirs", type=int, default=8)
    parser.add_argument("--max-candidates-per-iter", type=int, default=8)
    parser.add_argument(
        "--scenario-regex",
        default="",
        help="Optional regex over the sample summary scenario, applied before --max-samples.",
    )
    parser.add_argument(
        "--sample-name-regex",
        default="",
        help="Optional regex over the sample summary sample_name, applied before --max-samples.",
    )
    parser.add_argument(
        "--iteration-indices",
        default="",
        help="Optional comma-separated replay-bank iteration ids to keep.",
    )
    parser.add_argument(
        "--candidate-name-regex",
        default=r"^(dp_prior|mean|scale_0\.5|scale_1|short8_mean|short12_mean|short16_mean)$",
    )
    parser.add_argument(
        "--all-candidates",
        action="store_true",
        help="Replay every candidate in each saved bank, ignoring regex and index filters.",
    )
    parser.add_argument("--candidate-indices", default="")
    parser.add_argument(
        "--candidate-filter-tsv",
        default="",
        help=(
            "Optional TSV/whitespace file. Rows may be either "
            "'iteration candidate_index' or "
            "'scenario iteration candidate_index'. When set, only listed "
            "candidates are replayed for each matching scenario/iteration."
        ),
    )
    parser.add_argument("--candidate-shard-index", type=int, default=0)
    parser.add_argument("--candidate-shard-count", type=int, default=1)
    parser.add_argument("--include-selected", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--source-insertion-suffix-bank",
        default="",
        help="Optional NPZ from build_cosmos3_source_insertion_suffix_bank.py.",
    )
    parser.add_argument(
        "--source-suffix-k",
        type=int,
        default=0,
        help="Number of nearest source insertion suffixes to append per live snapshot.",
    )
    parser.add_argument(
        "--source-suffix-blends",
        default="1.0",
        help="Comma-separated blends: (1-blend)*live_dp_prior + blend*source_suffix.",
    )
    parser.add_argument(
        "--source-suffix-execute-steps",
        type=int,
        default=32,
        help="Executed prefix for source-suffix candidates. Zero executes the available suffix horizon.",
    )
    parser.add_argument(
        "--source-suffix-offsets",
        default="",
        help="Optional comma-separated allowed offsets-before-insertion from the suffix bank.",
    )
    parser.add_argument(
        "--source-suffix-scenario-match",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Diagnostic-only exact scenario filter for suffix retrieval. Default is off to avoid scenario branching.",
    )
    parser.add_argument(
        "--suffix-generator-checkpoint",
        default="",
        help="Optional checkpoint from train_contact_action_suffix_generator.py.",
    )
    parser.add_argument(
        "--suffix-generator-offsets",
        default="64,48,32,24,16,8,0",
        help="Offsets-before-insertion to condition the learned suffix generator on.",
    )
    parser.add_argument("--suffix-generator-execute-steps", type=int, default=8)
    parser.add_argument(
        "--causal-suffix-diffusion-checkpoint",
        default="",
        help="Optional checkpoint from train_causal_contact_action_suffix_diffusion.py.",
    )
    parser.add_argument(
        "--causal-suffix-diffusion-offsets",
        default="64,48,32,24,16,8",
        help="Requested offset-before-insertion control tokens to sample.",
    )
    parser.add_argument("--causal-suffix-diffusion-samples-per-offset", type=int, default=2)
    parser.add_argument("--causal-suffix-diffusion-execute-steps", type=int, default=8)
    parser.add_argument("--causal-suffix-diffusion-temperature", type=float, default=1.0)
    parser.add_argument("--source-suffix-query-x-weight", type=float, default=1.0)
    parser.add_argument("--source-suffix-query-y-weight", type=float, default=2.0)
    parser.add_argument("--source-suffix-query-z-weight", type=float, default=4.0)
    parser.add_argument("--external-target-mode", choices=("source_env_state", "none"), default="source_env_state")
    parser.add_argument("--robot-action-dim", type=int, default=7)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--clip-live-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-step-records",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep per-step replay records in the JSONL labels. Disable for large calibration sweeps.",
    )
    parser.add_argument("--dp-rollout-continuability-horizon", type=int, default=0)
    parser.add_argument("--dp-rollout-continuability-min-stable-steps", type=int, default=4)
    parser.add_argument("--contact-stable-min-rel-x", type=float, default=-0.06)
    parser.add_argument("--contact-stable-max-rel-x", type=float, default=0.03)
    parser.add_argument("--contact-stable-max-abs-y", type=float, default=0.018)
    parser.add_argument("--contact-stable-max-abs-z", type=float, default=0.012)
    parser.add_argument("--progress-every-labels", type=int, default=50)
    parser.add_argument(
        "--persist-replayed-action-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write generated/synthetic candidate action chunks next to the "
            "outcome labels so later manifests are not forced to recover them "
            "from an original candidate_action_bank.npz."
        ),
    )
    parser.add_argument(
        "--persist-saved-bank-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also persist original saved-bank candidates. Default persists only synthetic/generated candidates.",
    )
    parser.add_argument("--seed", type=int, default=20260622)
    return parser.parse_args()


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_ints(text: str) -> list[int]:
    out: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value not in out:
            out.append(value)
    return out


def parse_floats(text: str) -> list[float]:
    out: list[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"source suffix blend must be in [0, 1], got {value}")
        if value not in out:
            out.append(value)
    return out or [1.0]


def load_candidate_filter_tsv(path_text: str) -> dict[tuple[str | None, int], set[int]] | None:
    if not str(path_text).strip():
        return None
    path = Path(path_text).resolve()
    out: dict[tuple[str | None, int], set[int]] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        scenario: str | None = None
        try:
            iteration = int(parts[0])
            candidate_index = int(parts[1])
        except ValueError:
            if len(parts) < 3:
                continue
            try:
                scenario = str(parts[0])
                iteration = int(parts[1])
                candidate_index = int(parts[2])
            except ValueError:
                continue
        out.setdefault((scenario, iteration), set()).add(candidate_index)
    if not out:
        raise ValueError(f"candidate filter TSV has no usable rows: {path}")
    return out


def load_source_suffix_bank(path_text: str) -> dict[str, np.ndarray] | None:
    if not str(path_text).strip():
        return None
    path = Path(path_text).resolve()
    payload = np.load(str(path), allow_pickle=False)
    bank = {key: payload[key] for key in payload.files}
    if "actions" not in bank or "start_peg_head_at_hole" not in bank:
        raise ValueError(f"source suffix bank missing required arrays: {path}")
    bank["_bank_path"] = np.asarray([str(path)], dtype="<U512")
    return bank


def load_suffix_generator(path_text: str) -> dict[str, Any] | None:
    if not str(path_text).strip():
        return None
    import torch
    from train_contact_action_suffix_generator import SuffixActionNet

    path = Path(path_text).resolve()
    payload = torch.load(path, map_location="cpu")
    ckpt_args = payload.get("args") or {}
    metadata = payload.get("metadata") or {}
    model = SuffixActionNet(
        feature_dim=int(payload["feature_dim"]),
        target_dim=int(payload["target_dim"]),
        hidden_dim=int(ckpt_args.get("hidden_dim", 4096)),
        num_layers=int(ckpt_args.get("num_layers", 5)),
        dropout=0.0,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return {
        "path": str(path),
        "payload": payload,
        "model": model,
        "metadata": metadata,
        "x_mean": np.asarray(payload["x_mean"], dtype=np.float32),
        "x_std": np.asarray(payload["x_std"], dtype=np.float32),
        "y_mean": np.asarray(payload["y_mean"], dtype=np.float32),
        "y_std": np.asarray(payload["y_std"], dtype=np.float32),
    }


def load_causal_suffix_diffusion(path_text: str) -> dict[str, Any] | None:
    if not str(path_text).strip():
        return None
    import torch
    from train_causal_contact_action_suffix_diffusion import CausalSuffixDiffusionNet

    path = Path(path_text).resolve()
    payload = torch.load(path, map_location="cpu")
    ckpt_args = payload.get("args") or {}
    manifest = payload.get("manifest") or {}
    metadata = manifest.get("metadata") or {}
    model = CausalSuffixDiffusionNet(
        feature_dim=int(payload["feature_dim"]),
        target_dim=int(payload["target_dim"]),
        hidden_dim=int(ckpt_args.get("hidden_dim", 2048)),
        num_layers=int(ckpt_args.get("num_layers", 5)),
        dropout=float(ckpt_args.get("dropout", 0.05)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return {
        "path": str(path),
        "payload": payload,
        "model": model,
        "manifest": manifest,
        "metadata": metadata,
        "x_mean": np.asarray(payload["x_mean"], dtype=np.float32),
        "x_std": np.asarray(payload["x_std"], dtype=np.float32),
        "y_mean": np.asarray(payload["y_mean"], dtype=np.float32),
        "y_std": np.asarray(payload["y_std"], dtype=np.float32),
    }


def load_env_states(source_h5: Path, trajectory_utils: Any) -> list[dict[str, Any]]:
    import h5py

    with h5py.File(source_h5, "r") as h5:
        traj_name = "traj_0" if "traj_0" in h5 else next(iter(h5.keys()))
        return trajectory_utils.dict_to_list_of_dicts(h5[traj_name]["env_states"])


def _read_h5_value(obj: Any) -> Any:
    import h5py

    if isinstance(obj, h5py.Group):
        return {key: _read_h5_value(value) for key, value in obj.items()}
    arr = np.asarray(obj)
    # Snapshots are written from common.batch(get_state_dict()), producing
    # (1, 1, N), while ManiSkill set_state_dict expects the usual (1, N).
    while arr.ndim >= 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def load_snapshot_state(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import h5py

    with h5py.File(path, "r") as h5:
        attrs = {key: jsonable(value) for key, value in h5.attrs.items()}
        state = _read_h5_value(h5["state"])
    return state, attrs


def load_history(path: Path) -> np.ndarray:
    payload = read_json(path)
    action = np.asarray(payload["action"], dtype=np.float32)
    if action.ndim != 2:
        raise ValueError(f"history action/state must be rank-2, got {action.shape}")
    return action


def sample_summary_paths(panel_root: Path) -> list[Path]:
    paths = list(panel_root.rglob("live_receding_loop_summary.json"))
    paths.extend(panel_root.rglob("live_receding_loop_partial_summary.json"))
    best_by_dir: dict[Path, Path] = {}
    for path in sorted(paths):
        current = best_by_dir.get(path.parent)
        if current is None or path.name == "live_receding_loop_summary.json":
            best_by_dir[path.parent] = path
    return sorted(best_by_dir.values())


def iter_dirs_for_summary(summary_path: Path, max_iter_dirs: int) -> list[Path]:
    dirs: list[Path] = []
    for bank in sorted(summary_path.parent.glob("iter_*_prefix_f*/candidate_action_bank.npz")):
        iter_dir = bank.parent
        if not (iter_dir / "live_state_before_controller.h5").exists():
            continue
        if not (iter_dir / "live_history_raw_action_state.json").exists():
            continue
        dirs.append(iter_dir)
        if int(max_iter_dirs) > 0 and len(dirs) >= int(max_iter_dirs):
            break
    return dirs


def select_candidate_indices(bank: dict[str, np.ndarray], args: argparse.Namespace) -> list[int]:
    names = [str(item) for item in np.asarray(bank["candidate_names"]).tolist()]
    selected_mask = np.asarray(bank.get("candidate_selected_mask", np.zeros(len(names), dtype=bool)), dtype=bool)
    if bool(args.all_candidates):
        deduped = list(range(len(names)))
    else:
        requested = parse_ints(str(args.candidate_indices))
        regex = re.compile(str(args.candidate_name_regex)) if str(args.candidate_name_regex).strip() else None
        chosen: list[int] = []
        if bool(args.include_selected):
            chosen.extend(int(i) for i, flag in enumerate(selected_mask) if bool(flag))
        chosen.extend(i for i in requested if 0 <= i < len(names))
        if regex is not None:
            chosen.extend(i for i, name in enumerate(names) if regex.search(name))
        deduped = []
        for idx in chosen:
            if idx not in deduped:
                deduped.append(idx)
    deduped.sort(key=lambda idx: (0 if bool(selected_mask[idx]) else 1, idx))
    shard_count = max(1, int(args.candidate_shard_count))
    shard_index = int(args.candidate_shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"candidate shard index {shard_index} outside [0,{shard_count})")
    if shard_count > 1:
        deduped = [idx for idx in deduped if int(idx) % shard_count == shard_index]
    if int(args.max_candidates_per_iter) > 0:
        deduped = deduped[: int(args.max_candidates_per_iter)]
    return deduped


def rel_metrics(rel: np.ndarray) -> dict[str, float]:
    rel = np.asarray(rel, dtype=np.float32).reshape(-1)[:3]
    if rel.size < 3:
        rel = np.pad(rel, (0, 3 - rel.size), mode="constant")
    return {
        "abs_x": float(abs(rel[0])),
        "abs_y": float(abs(rel[1])),
        "abs_z": float(abs(rel[2])),
        "abs_yz_sum": float(abs(rel[1]) + abs(rel[2])),
        "yz_l2": float(np.linalg.norm(rel[1:3])),
        "weighted_abs_xyz": float(abs(rel[0]) + 2.0 * abs(rel[1]) + 4.0 * abs(rel[2])),
    }


def safe_slug(text: Any, max_len: int = 96) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._")
    return (slug or "item")[: int(max_len)]


def action_chunk_stats(actions: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.size == 0:
        return {"shape": list(arr.shape), "size": 0}
    return {
        "shape": [int(x) for x in arr.shape],
        "mean_abs": float(np.mean(np.abs(arr))),
        "max_abs": float(np.max(np.abs(arr))),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def persist_action_chunk_json(
    *,
    output_root: Path,
    sample_name: Any,
    scenario: Any,
    iteration: int,
    prefix_frame: int,
    replay_item: dict[str, Any],
) -> Path:
    actions = np.asarray(replay_item["actions"], dtype=np.float32)
    candidate_name = str(replay_item["candidate_name"])
    candidate_index = int(replay_item["candidate_index"])
    sample_dir = safe_slug(sample_name or scenario or "sample")
    path = (
        output_root
        / "persisted_action_chunks"
        / sample_dir
        / f"iter_{int(iteration):03d}_f{int(prefix_frame):03d}_cand_{candidate_index:05d}_{safe_slug(candidate_name)}.action_chunk.json"
    )
    payload = {
        "schema": "persisted_live_replay_action_chunk_v1",
        "candidate_name": candidate_name,
        "candidate_index": candidate_index,
        "candidate_selected_by_live_scorer": bool(replay_item.get("selected", False)),
        "candidate_meta": dict(replay_item.get("meta") or {}),
        "scenario": scenario,
        "sample_name": sample_name,
        "iteration": int(iteration),
        "prefix_frame_index": int(prefix_frame),
        "execute_steps_requested": int(replay_item.get("execute_steps", actions.shape[0])),
        "action_horizon_available": int(actions.shape[0]),
        "robot_action_dim": int(actions.shape[1]) if actions.ndim == 2 else None,
        "denormalized_robot_action_chunk": actions.astype(float).tolist(),
        "denormalized_robot_action_stats": action_chunk_stats(actions),
        "boundary": (
            "Persisted copy of the replayed candidate action chunk. It is "
            "training/evaluation data provenance only; live controllers must "
            "still condition only on causal observations/history."
        ),
    }
    write_json(path, payload)
    return path


def source_suffix_candidates(
    *,
    source_bank: dict[str, np.ndarray] | None,
    scenario: str,
    query_rel: np.ndarray,
    dp_prior_actions: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if source_bank is None or int(args.source_suffix_k) <= 0:
        return []
    source_actions_all = np.asarray(source_bank["actions"], dtype=np.float32)
    if source_actions_all.ndim != 3:
        raise ValueError(f"source suffix actions must be rank-3, got {source_actions_all.shape}")
    start_rel = np.asarray(source_bank["start_peg_head_at_hole"], dtype=np.float32)
    scenarios = [str(item) for item in np.asarray(source_bank.get("scenario", np.asarray([], dtype="<U1"))).tolist()]
    offsets = np.asarray(source_bank.get("offset_before_insert", np.full((source_actions_all.shape[0],), -1)), dtype=np.int32)
    valid_steps = np.asarray(source_bank.get("valid_steps", np.full((source_actions_all.shape[0],), source_actions_all.shape[1])), dtype=np.int32)
    allowed_offsets = set(parse_ints(str(args.source_suffix_offsets)))
    query = np.asarray(query_rel, dtype=np.float32).reshape(-1)[:3]
    if query.size < 3:
        query = np.pad(query, (0, 3 - query.size), mode="constant")
    weights = np.asarray(
        [
            float(args.source_suffix_query_x_weight),
            float(args.source_suffix_query_y_weight),
            float(args.source_suffix_query_z_weight),
        ],
        dtype=np.float32,
    )
    scenario_available = bool(scenarios) and any(item == str(scenario) for item in scenarios)
    scored: list[tuple[float, int]] = []
    for idx in range(int(source_actions_all.shape[0])):
        if allowed_offsets and int(offsets[idx]) not in allowed_offsets:
            continue
        if bool(args.source_suffix_scenario_match) and scenario_available and scenarios[idx] != str(scenario):
            continue
        rel = np.asarray(start_rel[idx, :3], dtype=np.float32)
        dist = float(np.linalg.norm((rel - query) * weights))
        scored.append((dist, idx))
    scored.sort(key=lambda item: item[0])

    dp_prior = np.asarray(dp_prior_actions, dtype=np.float32)[:, : int(args.robot_action_dim)]
    blends = parse_floats(str(args.source_suffix_blends))
    out: list[dict[str, Any]] = []
    for rank, (dist, source_idx) in enumerate(scored[: int(args.source_suffix_k)]):
        source_actions = np.asarray(source_actions_all[source_idx], dtype=np.float32)[:, : int(args.robot_action_dim)]
        horizon = min(int(dp_prior.shape[0]), int(source_actions.shape[0]))
        if horizon <= 0:
            continue
        for blend in blends:
            actions = (1.0 - float(blend)) * dp_prior[:horizon] + float(blend) * source_actions[:horizon]
            execute_steps = int(args.source_suffix_execute_steps)
            if execute_steps <= 0:
                execute_steps = min(horizon, int(valid_steps[source_idx]))
            execute_steps = max(1, min(int(execute_steps), int(horizon), int(valid_steps[source_idx])))
            blend_tag = f"{float(blend):g}".replace(".", "p")
            name = f"source_suffix_r{rank}_o{int(offsets[source_idx])}_b{blend_tag}"
            out.append(
                {
                    "candidate_name": name,
                    "actions": actions.astype(np.float32),
                    "execute_steps": int(execute_steps),
                    "selected": False,
                    "meta": {
                        "candidate_source": "source_insertion_suffix",
                        "source_suffix_bank_npz": str(np.asarray(source_bank["_bank_path"]).reshape(-1)[0]),
                        "source_suffix_rank": int(rank),
                        "source_suffix_distance": float(dist),
                        "source_suffix_index": int(source_idx),
                        "source_suffix_blend": float(blend),
                        "source_suffix_offset_before_insert": int(offsets[source_idx]),
                        "source_suffix_valid_steps": int(valid_steps[source_idx]),
                        "source_suffix_scenario": scenarios[source_idx] if scenarios else None,
                        "source_suffix_source_h5": (
                            str(np.asarray(source_bank["source_h5"]).tolist()[source_idx])
                            if "source_h5" in source_bank
                            else None
                        ),
                        "source_suffix_start_frame": (
                            int(np.asarray(source_bank["start_frame"], dtype=np.int32)[source_idx])
                            if "start_frame" in source_bank
                            else None
                        ),
                        "source_suffix_first_insert_frame": (
                            int(np.asarray(source_bank["first_insert_frame"], dtype=np.int32)[source_idx])
                            if "first_insert_frame" in source_bank
                            else None
                        ),
                        "source_suffix_start_peg_head_at_hole": start_rel[source_idx, :3].astype(float).tolist(),
                        "source_suffix_query_peg_head_at_hole": query.astype(float).tolist(),
                    },
                }
            )
    return out


def suffix_generator_candidates(
    *,
    suffix_generator: dict[str, Any] | None,
    scenario: str,
    query_rel: np.ndarray,
    prefix_frame: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if suffix_generator is None:
        return []
    import torch

    metadata = suffix_generator.get("metadata") or {}
    scenario_vocab = [str(x) for x in metadata.get("scenario_vocab", [])]
    offset_values = [int(x) for x in metadata.get("offset_values", [])]
    horizon = int(metadata.get("horizon", 96))
    action_dim = int(metadata.get("action_dim", int(args.robot_action_dim)))
    if not scenario_vocab or not offset_values or horizon <= 0 or action_dim <= 0:
        raise ValueError("suffix generator checkpoint metadata is incomplete")

    scenario_name = str(scenario)
    if scenario_name not in scenario_vocab:
        scenario_name = "none" if "none" in scenario_vocab else scenario_vocab[0]
    scenario_onehot = np.zeros((len(scenario_vocab),), dtype=np.float32)
    scenario_onehot[scenario_vocab.index(scenario_name)] = 1.0

    query = np.asarray(query_rel, dtype=np.float32).reshape(-1)[:3]
    if query.size < 3:
        query = np.pad(query, (0, 3 - query.size), mode="constant")
    requested_offsets = [offset for offset in parse_ints(str(args.suffix_generator_offsets)) if offset in set(offset_values)]
    if not requested_offsets:
        requested_offsets = offset_values

    out: list[dict[str, Any]] = []
    model = suffix_generator["model"]
    x_mean = np.asarray(suffix_generator["x_mean"], dtype=np.float32)
    x_std = np.asarray(suffix_generator["x_std"], dtype=np.float32)
    y_mean = np.asarray(suffix_generator["y_mean"], dtype=np.float32)
    y_std = np.asarray(suffix_generator["y_std"], dtype=np.float32)
    for offset in requested_offsets:
        offset_onehot = np.zeros((len(offset_values),), dtype=np.float32)
        offset_onehot[offset_values.index(int(offset))] = 1.0
        abs_yz = np.asarray([abs(float(query[1])) + abs(float(query[2]))], dtype=np.float32)
        first_insert_est = min(300, int(prefix_frame) + int(offset))
        feature = np.concatenate(
            [
                query.astype(np.float32),
                abs_yz,
                np.asarray([float(offset) / 96.0], dtype=np.float32),
                np.asarray([float(prefix_frame) / 300.0], dtype=np.float32),
                np.asarray([float(first_insert_est) / 300.0], dtype=np.float32),
                np.asarray([1.0], dtype=np.float32),
                offset_onehot,
                scenario_onehot,
            ],
            axis=0,
        ).reshape(1, -1)
        if feature.shape != x_mean.shape:
            raise ValueError(f"suffix generator feature shape {feature.shape} != expected {x_mean.shape}")
        with torch.no_grad():
            x = torch.from_numpy(((feature - x_mean) / x_std).astype(np.float32))
            pred_norm = model(x).cpu().numpy()
        pred = pred_norm * y_std + y_mean
        actions = pred.reshape(horizon, action_dim)[:, : int(args.robot_action_dim)].astype(np.float32)
        execute_steps = int(args.suffix_generator_execute_steps)
        if execute_steps <= 0:
            execute_steps = horizon
        execute_steps = max(1, min(int(execute_steps), int(actions.shape[0])))
        out.append(
            {
                "candidate_name": f"suffix_generator_o{int(offset)}",
                "actions": actions,
                "execute_steps": int(execute_steps),
                "selected": False,
                "meta": {
                    "candidate_source": "suffix_action_generator",
                    "suffix_generator_checkpoint": str(suffix_generator["path"]),
                    "suffix_generator_condition_offset_before_insert": int(offset),
                    "suffix_generator_condition_scenario": scenario_name,
                    "suffix_generator_query_peg_head_at_hole": query.astype(float).tolist(),
                    "suffix_generator_prefix_frame": int(prefix_frame),
                    "suffix_generator_estimated_first_insert_frame": int(first_insert_est),
                },
            }
        )
    return out


def causal_suffix_diffusion_feature(
    *,
    query_rel: np.ndarray,
    offset: int,
    prefix_frame: int,
    grasped: bool,
    offset_values: list[int],
) -> np.ndarray:
    query = np.asarray(query_rel, dtype=np.float32).reshape(-1)[:3]
    if query.size < 3:
        query = np.pad(query, (0, 3 - query.size), mode="constant")
    offset_onehot = np.zeros((len(offset_values),), dtype=np.float32)
    if int(offset) not in set(int(x) for x in offset_values):
        raise ValueError(f"offset {offset} not in diffusion checkpoint offset values {offset_values}")
    offset_onehot[[int(x) for x in offset_values].index(int(offset))] = 1.0
    abs_y = abs(float(query[1]))
    abs_z = abs(float(query[2]))
    prefix = max(0, min(int(prefix_frame), 300))
    values = np.concatenate(
        [
            query.astype(np.float32),
            np.asarray([abs_y], dtype=np.float32),
            np.asarray([abs_z], dtype=np.float32),
            np.asarray([abs_y + abs_z], dtype=np.float32),
            np.asarray([float(np.linalg.norm(query[1:3]))], dtype=np.float32),
            np.asarray([float(offset) / 96.0], dtype=np.float32),
            np.asarray([float(prefix) / 300.0], dtype=np.float32),
            np.asarray([(300.0 - float(prefix)) / 300.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            np.asarray([1.0 if bool(grasped) else 0.0], dtype=np.float32),
            offset_onehot,
        ],
        axis=0,
    )
    return values.reshape(1, -1).astype(np.float32)


def sample_causal_suffix_diffusion_action(
    *,
    diffusion: dict[str, Any],
    feature: np.ndarray,
    sample_seed: int,
    temperature: float,
) -> np.ndarray:
    import torch
    from train_causal_contact_action_suffix_diffusion import diffusion_schedule

    payload = diffusion["payload"]
    args_payload = payload.get("args") or {}
    metadata = (payload.get("manifest") or {}).get("metadata") or {}
    horizon = int(metadata.get("horizon", 96))
    action_dim = int(metadata.get("action_dim", int(payload["target_dim"]) // max(1, horizon)))
    x_mean = np.asarray(diffusion["x_mean"], dtype=np.float32)
    x_std = np.asarray(diffusion["x_std"], dtype=np.float32)
    y_mean = np.asarray(diffusion["y_mean"], dtype=np.float32)
    y_std = np.asarray(diffusion["y_std"], dtype=np.float32)
    if feature.shape != x_mean.shape:
        raise ValueError(f"causal suffix diffusion feature shape {feature.shape} != expected {x_mean.shape}")
    steps = int(args_payload.get("diffusion_steps", 32))
    beta_start = float(args_payload.get("diffusion_beta_start", 1e-4))
    beta_end = float(args_payload.get("diffusion_beta_end", 2e-2))
    model = diffusion["model"]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(sample_seed))
    x = torch.from_numpy(((feature - x_mean) / x_std).astype(np.float32))
    y = torch.randn((1, int(payload["target_dim"])), generator=generator, dtype=torch.float32) * float(temperature)
    alpha_bars = diffusion_schedule(steps, beta_start, beta_end, torch.device("cpu"))
    with torch.no_grad():
        for t in range(int(alpha_bars.shape[0]) - 1, -1, -1):
            t_tensor = torch.full((1,), int(t), dtype=torch.long)
            t_norm = t_tensor.float() / max(1, int(alpha_bars.shape[0]) - 1)
            alpha_bar = alpha_bars[t].reshape(1, 1)
            pred_noise = model(x, y, t_norm)
            x0 = (y - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            if t > 0:
                prev_alpha_bar = alpha_bars[t - 1].reshape(1, 1)
                y = torch.sqrt(prev_alpha_bar) * x0 + torch.sqrt(1.0 - prev_alpha_bar) * pred_noise
            else:
                y = x0
    y_np = y.cpu().numpy() * y_std + y_mean
    return y_np.reshape(horizon, action_dim).astype(np.float32)


def causal_suffix_diffusion_candidates(
    *,
    diffusion: dict[str, Any] | None,
    query_rel: np.ndarray,
    query_grasped: bool,
    prefix_frame: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if diffusion is None:
        return []
    metadata = diffusion.get("metadata") or {}
    offset_values = [int(x) for x in metadata.get("offset_values", [])]
    if not offset_values:
        raise ValueError("causal suffix diffusion checkpoint missing offset_values metadata")
    requested_offsets = [offset for offset in parse_ints(str(args.causal_suffix_diffusion_offsets)) if offset in set(offset_values)]
    if not requested_offsets:
        requested_offsets = offset_values
    sample_count = max(1, int(args.causal_suffix_diffusion_samples_per_offset))
    execute_steps = max(1, int(args.causal_suffix_diffusion_execute_steps))
    out: list[dict[str, Any]] = []
    query = np.asarray(query_rel, dtype=np.float32).reshape(-1)[:3]
    if query.size < 3:
        query = np.pad(query, (0, 3 - query.size), mode="constant")
    for offset in requested_offsets:
        feature = causal_suffix_diffusion_feature(
            query_rel=query,
            offset=int(offset),
            prefix_frame=int(prefix_frame),
            grasped=bool(query_grasped),
            offset_values=offset_values,
        )
        for sample_i in range(sample_count):
            seed = int(args.seed) + int(offset) * 1009 + int(prefix_frame) * 17 + int(sample_i)
            actions = sample_causal_suffix_diffusion_action(
                diffusion=diffusion,
                feature=feature,
                sample_seed=seed,
                temperature=float(args.causal_suffix_diffusion_temperature),
            )[:, : int(args.robot_action_dim)]
            out.append(
                {
                    "candidate_name": f"causal_suffix_diffusion_o{int(offset)}_s{int(sample_i)}",
                    "actions": actions.astype(np.float32),
                    "execute_steps": min(execute_steps, int(actions.shape[0])),
                    "selected": False,
                    "meta": {
                        "candidate_source": "causal_suffix_diffusion",
                        "causal_suffix_diffusion_checkpoint": str(diffusion["path"]),
                        "causal_suffix_diffusion_condition_offset_before_insert": int(offset),
                        "causal_suffix_diffusion_sample_index": int(sample_i),
                        "causal_suffix_diffusion_temperature": float(args.causal_suffix_diffusion_temperature),
                        "causal_suffix_diffusion_query_peg_head_at_hole": query.astype(float).tolist(),
                        "causal_suffix_diffusion_query_grasped": bool(query_grasped),
                        "causal_suffix_diffusion_prefix_frame": int(prefix_frame),
                    },
                }
            )
    return out


def thresholds_from_summary(args: argparse.Namespace, summary: dict[str, Any]) -> argparse.Namespace:
    out = SimpleNamespace(**vars(args))
    thresholds = summary.get("continuability_thresholds") or {}
    out.continuability_min_rel_x = float(thresholds.get("min_rel_x", -0.08))
    out.continuability_max_rel_x = float(thresholds.get("max_rel_x", 0.04))
    out.continuability_max_abs_y = float(thresholds.get("max_abs_y", 0.025))
    out.continuability_max_abs_z = float(thresholds.get("max_abs_z", 0.025))
    out.continuability_max_hole_speed = float(thresholds.get("max_hole_speed", 0.01))
    return out


def replay_bank_candidate(
    *,
    env: Any,
    base_env: Any,
    stack: dict[str, Any],
    snapshot_state: dict[str, Any],
    env_states: list[dict[str, Any]],
    history_template: np.ndarray,
    prefix_frame: int,
    candidate_name: str,
    candidate_index: int,
    selected: bool,
    actions_full: np.ndarray,
    execute_steps: int,
    candidate_meta: dict[str, Any] | None,
    low: Any,
    high: Any,
    args: argparse.Namespace,
    dp_agent: Any | None,
    dp_args: Any | None,
) -> dict[str, Any]:
    base_env.set_state_dict(snapshot_state)
    history = np.asarray(history_template, dtype=np.float32).copy()
    start_live = live_pose_row(base_env, stack, None)
    previous_hole_xyz = np.asarray(start_live["hole_xyz"], dtype=np.float32).copy()
    before_eval = _live_eval(base_env)
    before_rel = np.asarray(start_live["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
    before_gate = continuability_gate(start_live, history, int(prefix_frame), args)
    start_state_obs = read_state_obs(base_env, stack)
    dp_obs_history: list[np.ndarray] = [start_state_obs.copy(), start_state_obs.copy()]

    max_steps = min(
        int(execute_steps),
        int(actions_full.shape[0]),
        max(0, int(args.max_episode_steps) - int(prefix_frame)),
        max(0, len(env_states) - 1 - int(prefix_frame)),
    )
    step_records: list[dict[str, Any]] = []
    last_live = start_live
    stop_reason = "candidate_horizon_exhausted"
    for local_i, action in enumerate(np.asarray(actions_full[:max_steps], dtype=np.float32)):
        step_action, action_record = _prepare_step_action(action, low, high, bool(args.clip_live_actions))
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
        global_action_index = int(prefix_frame) + int(local_i)
        if 0 <= global_action_index < history.shape[0]:
            fill_live_history_row(history, global_action_index, np.asarray(action_record["executed"]), last_live)
        step_eval = _live_eval(base_env)
        step_rel = np.asarray(last_live["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
        step_records.append(
            {
                "local_step": int(local_i),
                "global_action_index": int(global_action_index),
                "source_frame": int(source_frame),
                "action": action_record,
                "external_target": external_target,
                "reward": jsonable(reward),
                "terminated": jsonable(terminated),
                "truncated": jsonable(truncated),
                "live_eval": step_eval,
                "peg_head_at_hole": step_rel.astype(float).tolist(),
                "rel_metrics": rel_metrics(step_rel),
                "grasped": bool(last_live.get("grasped", False)),
                "inserted": bool(last_live.get("inserted", False)),
            }
        )
        if bool(step_eval.get("success", False)):
            stop_reason = "success"
            break
        if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
            stop_reason = "terminated_or_truncated"
            break

    after_eval = _live_eval(base_env)
    after_rel = np.asarray(last_live["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
    after_prefix_frame = min(int(prefix_frame) + len(step_records), int(args.max_episode_steps))
    after_gate = continuability_gate(last_live, history, after_prefix_frame, args)
    after_grasped = bool(last_live.get("grasped", False))
    after_inserted = bool(last_live.get("inserted", False))
    after_contact_stable = contact_stable_proxy(
        after_rel,
        grasped=after_grasped,
        inserted=after_inserted,
        args=args,
    )
    before_m = rel_metrics(before_rel)
    after_m = rel_metrics(after_rel)

    out: dict[str, Any] = {
        "schema": "cosmos3_live_snapshot_action_bank_outcome_label_v1",
        "candidate_name": candidate_name,
        "candidate_index": int(candidate_index),
        "candidate_selected_by_live_scorer": bool(selected),
        "prefix_frame_index": int(prefix_frame),
        "execute_steps_requested": int(execute_steps),
        "execute_steps_actual": int(len(step_records)),
        "stop_reason": stop_reason,
        "before_eval": before_eval,
        "after_eval": after_eval,
        "before_peg_head_at_hole": before_rel.astype(float).tolist(),
        "after_peg_head_at_hole": after_rel.astype(float).tolist(),
        "before_rel_metrics": before_m,
        "after_rel_metrics": after_m,
        "delta_abs_yz_sum": float(after_m["abs_yz_sum"] - before_m["abs_yz_sum"]),
        "delta_yz_l2": float(after_m["yz_l2"] - before_m["yz_l2"]),
        "before_continuability_gate": before_gate,
        "after_continuability_gate": after_gate,
        "after_grasped": after_grasped,
        "after_inserted_live_pose": after_inserted,
        "after_contact_progress_proxy": contact_progress_proxy(
            after_rel,
            grasped=after_grasped,
            inserted=after_inserted,
        ),
        "after_contact_stable_proxy": bool(after_contact_stable),
        "after_success": bool(after_eval.get("success", False)),
        "boundary": (
            "Label-only replay from a real live simulator snapshot. The replay "
            "answers what each saved candidate action chunk actually does from "
            "the same live state; it is not a privileged controller input and "
            "is not method success evidence."
        ),
    }
    if candidate_meta:
        out["candidate_meta"] = candidate_meta
    if bool(args.save_step_records):
        out["step_records"] = step_records
    if int(args.dp_rollout_continuability_horizon) > 0:
        if dp_agent is None or dp_args is None:
            raise RuntimeError("DP rollout label requested without a loaded DP agent")
        out["dp_rollout_continuability"] = run_dp_rollout_continuability_label(
            env=env,
            base_env=base_env,
            stack=stack,
            env_states=env_states,
            start_frame=after_prefix_frame,
            previous_hole_xyz=previous_hole_xyz,
            dp_obs_history=dp_obs_history,
            dp_agent=dp_agent,
            dp_args=dp_args,
            low=low,
            high=high,
            args=args,
        )
    return out


def main() -> int:
    args = parse_args()
    require_compute_step()
    panel_root = Path(args.panel_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

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

    source_cache: dict[str, list[dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    summary_paths = sample_summary_paths(panel_root)
    scenario_regex = re.compile(str(args.scenario_regex)) if str(args.scenario_regex).strip() else None
    sample_name_regex = re.compile(str(args.sample_name_regex)) if str(args.sample_name_regex).strip() else None
    if scenario_regex is not None or sample_name_regex is not None:
        filtered_summary_paths: list[Path] = []
        for summary_path in summary_paths:
            sample_summary = read_json(summary_path)
            scenario_ok = scenario_regex is None or bool(scenario_regex.search(str(sample_summary.get("scenario", ""))))
            sample_name_ok = sample_name_regex is None or bool(
                sample_name_regex.search(str(sample_summary.get("sample_name", "")))
            )
            if scenario_ok and sample_name_ok:
                filtered_summary_paths.append(summary_path)
        summary_paths = filtered_summary_paths
    if int(args.max_samples) > 0:
        summary_paths = summary_paths[: int(args.max_samples)]
    iteration_filter = set(parse_ints(str(args.iteration_indices)))
    candidate_filter = load_candidate_filter_tsv(str(args.candidate_filter_tsv))
    source_suffix_bank = load_source_suffix_bank(str(args.source_insertion_suffix_bank))
    suffix_generator = load_suffix_generator(str(args.suffix_generator_checkpoint))
    causal_suffix_diffusion = load_causal_suffix_diffusion(str(args.causal_suffix_diffusion_checkpoint))

    for summary_path in summary_paths:
        sample_summary = read_json(summary_path)
        source_h5 = Path(str(sample_summary["source_h5"])).resolve()
        source_key = str(source_h5)
        if source_key not in source_cache:
            source_cache[source_key] = load_env_states(source_h5, stack["trajectory_utils"])
        env_states = source_cache[source_key]
        threshold_args = thresholds_from_summary(args, sample_summary)
        reset_seed = _parse_seed_from_text(" ".join([source_h5.name, str(sample_summary.get("sample_name", ""))]))
        iter_dirs = iter_dirs_for_summary(summary_path, int(args.max_iter_dirs))
        for iter_dir in iter_dirs:
            bank_path = iter_dir / "candidate_action_bank.npz"
            snapshot_path = iter_dir / "live_state_before_controller.h5"
            history_path = iter_dir / "live_history_raw_action_state.json"
            try:
                bank_npz = np.load(str(bank_path), allow_pickle=False)
                bank = {key: bank_npz[key] for key in bank_npz.files}
                names = [str(item) for item in np.asarray(bank["candidate_names"]).tolist()]
                actions = np.asarray(bank["candidate_full_actions"], dtype=np.float32)
                execute_steps = np.asarray(bank["candidate_execute_steps"], dtype=np.int32)
                selected_mask = np.asarray(bank.get("candidate_selected_mask", np.zeros(len(names), dtype=bool)), dtype=bool)
                prefix_frame = int(np.asarray(bank.get("prefix_frame_index", [-1])).reshape(-1)[0])
                iteration = int(np.asarray(bank.get("iteration", [-1])).reshape(-1)[0])
                if iteration_filter and int(iteration) not in iteration_filter:
                    continue
                if prefix_frame < 0:
                    raise ValueError("candidate bank missing prefix_frame_index")
                snapshot_state, snapshot_attrs = load_snapshot_state(snapshot_path)
                history = load_history(history_path)
                candidate_indices = select_candidate_indices(bank, args)
                if not candidate_indices:
                    has_synthetic = bool(
                        (source_suffix_bank is not None and int(args.source_suffix_k) > 0)
                        or suffix_generator is not None
                        or causal_suffix_diffusion is not None
                    )
                    if not has_synthetic:
                        raise ValueError("no candidate indices selected")
                replay_items: list[dict[str, Any]] = []
                for candidate_index in candidate_indices:
                    replay_items.append(
                        {
                            "candidate_index": int(candidate_index),
                            "candidate_name": names[candidate_index],
                            "selected": bool(selected_mask[candidate_index]),
                            "actions": actions[candidate_index, :, : int(args.robot_action_dim)],
                            "execute_steps": int(execute_steps[candidate_index]),
                            "meta": {"candidate_source": "saved_live_action_bank"},
                        }
                    )
                if source_suffix_bank is not None and int(args.source_suffix_k) > 0:
                    env.reset(seed=int(reset_seed if reset_seed is not None else args.seed))
                    base_env.set_state_dict(snapshot_state)
                    query_live = live_pose_row(base_env, stack, None)
                    query_rel = np.asarray(query_live["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
                    dp_prior_actions = np.asarray(bank["dp_prior_actions"], dtype=np.float32)[:, : int(args.robot_action_dim)]
                    source_items = source_suffix_candidates(
                        source_bank=source_suffix_bank,
                        scenario=str(sample_summary.get("scenario", "")),
                        query_rel=query_rel,
                        dp_prior_actions=dp_prior_actions,
                        args=args,
                    )
                    synthetic_base = len(names)
                    shard_count = max(1, int(args.candidate_shard_count))
                    shard_index = int(args.candidate_shard_index)
                    for local_idx, item in enumerate(source_items):
                        synthetic_index = int(synthetic_base + local_idx)
                        if shard_count > 1 and synthetic_index % shard_count != shard_index:
                            continue
                        replay_items.append(
                            {
                                "candidate_index": synthetic_index,
                                "candidate_name": item["candidate_name"],
                                "selected": bool(item.get("selected", False)),
                                "actions": np.asarray(item["actions"], dtype=np.float32),
                                "execute_steps": int(item["execute_steps"]),
                                "meta": dict(item.get("meta") or {}),
                            }
                        )
                if suffix_generator is not None:
                    env.reset(seed=int(reset_seed if reset_seed is not None else args.seed))
                    base_env.set_state_dict(snapshot_state)
                    query_live = live_pose_row(base_env, stack, None)
                    query_rel = np.asarray(query_live["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
                    generated_items = suffix_generator_candidates(
                        suffix_generator=suffix_generator,
                        scenario=str(sample_summary.get("scenario", "")),
                        query_rel=query_rel,
                        prefix_frame=int(prefix_frame),
                        args=args,
                    )
                    synthetic_base = len(names) + len(replay_items)
                    shard_count = max(1, int(args.candidate_shard_count))
                    shard_index = int(args.candidate_shard_index)
                    for local_idx, item in enumerate(generated_items):
                        synthetic_index = int(synthetic_base + local_idx)
                        if shard_count > 1 and synthetic_index % shard_count != shard_index:
                            continue
                        replay_items.append(
                            {
                                "candidate_index": synthetic_index,
                                "candidate_name": item["candidate_name"],
                                "selected": bool(item.get("selected", False)),
                                "actions": np.asarray(item["actions"], dtype=np.float32),
                                "execute_steps": int(item["execute_steps"]),
                                "meta": dict(item.get("meta") or {}),
                            }
                        )
                if causal_suffix_diffusion is not None:
                    env.reset(seed=int(reset_seed if reset_seed is not None else args.seed))
                    base_env.set_state_dict(snapshot_state)
                    query_live = live_pose_row(base_env, stack, None)
                    query_rel = np.asarray(query_live["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
                    generated_items = causal_suffix_diffusion_candidates(
                        diffusion=causal_suffix_diffusion,
                        query_rel=query_rel,
                        query_grasped=bool(query_live.get("grasped", False)),
                        prefix_frame=int(prefix_frame),
                        args=args,
                    )
                    # Keep generated causal-suffix candidate ids stable across
                    # all-candidate and selected replay. Converted outcome rows
                    # store these ids, so adding a saved dp_prior replay item
                    # must not shift the generated candidate namespace.
                    synthetic_base = len(names)
                    shard_count = max(1, int(args.candidate_shard_count))
                    shard_index = int(args.candidate_shard_index)
                    for local_idx, item in enumerate(generated_items):
                        synthetic_index = int(synthetic_base + local_idx)
                        if shard_count > 1 and synthetic_index % shard_count != shard_index:
                            continue
                        replay_items.append(
                            {
                                "candidate_index": synthetic_index,
                                "candidate_name": item["candidate_name"],
                                "selected": bool(item.get("selected", False)),
                                "actions": np.asarray(item["actions"], dtype=np.float32),
                                "execute_steps": int(item["execute_steps"]),
                                "meta": dict(item.get("meta") or {}),
                            }
                        )
                if candidate_filter is not None:
                    scenario_key = str(sample_summary.get("scenario", ""))
                    allowed = set(candidate_filter.get((None, int(iteration)), set()))
                    allowed.update(candidate_filter.get((scenario_key, int(iteration)), set()))
                    replay_items = [
                        item for item in replay_items if int(item["candidate_index"]) in allowed
                    ]
                    if not replay_items:
                        continue
                for replay_item in replay_items:
                    env.reset(seed=int(reset_seed if reset_seed is not None else args.seed))
                    replay_item_meta = dict(replay_item.get("meta") or {})
                    candidate_source = str(replay_item_meta.get("candidate_source") or "")
                    should_persist_actions = bool(args.persist_replayed_action_chunks) and (
                        bool(args.persist_saved_bank_actions) or candidate_source != "saved_live_action_bank"
                    )
                    persisted_action_chunk_json = None
                    if should_persist_actions:
                        persisted_action_chunk_json = persist_action_chunk_json(
                            output_root=output_root,
                            sample_name=sample_summary.get("sample_name"),
                            scenario=sample_summary.get("scenario"),
                            iteration=int(iteration),
                            prefix_frame=int(prefix_frame),
                            replay_item=replay_item,
                        )
                    label = replay_bank_candidate(
                        env=env,
                        base_env=base_env,
                        stack=stack,
                        snapshot_state=snapshot_state,
                        env_states=env_states,
                        history_template=history,
                        prefix_frame=prefix_frame,
                        candidate_name=str(replay_item["candidate_name"]),
                        candidate_index=int(replay_item["candidate_index"]),
                        selected=bool(replay_item["selected"]),
                        actions_full=np.asarray(replay_item["actions"], dtype=np.float32),
                        execute_steps=int(replay_item["execute_steps"]),
                        candidate_meta=dict(replay_item.get("meta") or {}),
                        low=low,
                        high=high,
                        args=threshold_args,
                        dp_agent=dp_agent,
                        dp_args=dp_args,
                    )
                    if persisted_action_chunk_json is not None:
                        label["persisted_action_chunk_json"] = str(persisted_action_chunk_json)
                    label.update(
                        {
                            "panel_root": str(panel_root),
                            "sample_summary": str(summary_path),
                            "sample_name": sample_summary.get("sample_name"),
                            "scenario": sample_summary.get("scenario"),
                            "source_h5": str(source_h5),
                            "iter_dir": str(iter_dir),
                            "iteration": int(iteration),
                            "candidate_action_bank_npz": str(bank_path),
                            "live_state_snapshot_before_controller": str(snapshot_path),
                            "snapshot_attrs": snapshot_attrs,
                            "history_action_state": str(history_path),
                        }
                    )
                    records.append(label)
                    if int(args.progress_every_labels) > 0 and len(records) % int(args.progress_every_labels) == 0:
                        print(
                            "live_action_bank_replay_progress "
                            f"records={len(records)} "
                            f"valid={sum(1 for item in records if item.get('schema') == 'cosmos3_live_snapshot_action_bank_outcome_label_v1')} "
                            f"sample={sample_summary.get('sample_name')} "
                            f"iteration={iteration} "
                            f"candidate_index={replay_item['candidate_index']}",
                            flush=True,
                        )
            except Exception as exc:
                failures[type(exc).__name__] += 1
                records.append(
                    {
                        "schema": "cosmos3_live_snapshot_action_bank_outcome_label_failure_v1",
                        "sample_summary": str(summary_path),
                        "iter_dir": str(iter_dir),
                        "candidate_action_bank_npz": str(bank_path),
                        "live_state_snapshot_before_controller": str(snapshot_path),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    output_jsonl = output_root / "live_snapshot_action_bank_outcome_labels.jsonl"
    write_jsonl(output_jsonl, records)
    valid = [row for row in records if row.get("schema") == "cosmos3_live_snapshot_action_bank_outcome_label_v1"]
    dp_rollout_rows = [
        row for row in valid if isinstance(row.get("dp_rollout_continuability"), dict)
    ]
    summary = {
        "schema": "cosmos3_live_snapshot_action_bank_outcome_summary_v1",
        "panel_root": str(panel_root),
        "output_jsonl": str(output_jsonl),
        "sample_summaries_seen": len(summary_paths),
        "candidate_filter_tsv": str(Path(args.candidate_filter_tsv).resolve())
        if str(args.candidate_filter_tsv).strip()
        else None,
        "source_insertion_suffix_bank": str(Path(args.source_insertion_suffix_bank).resolve())
        if str(args.source_insertion_suffix_bank).strip()
        else None,
        "suffix_generator_checkpoint": str(Path(args.suffix_generator_checkpoint).resolve())
        if str(args.suffix_generator_checkpoint).strip()
        else None,
        "suffix_generator_offsets": parse_ints(str(args.suffix_generator_offsets))
        if str(args.suffix_generator_checkpoint).strip()
        else [],
        "suffix_generator_execute_steps": int(args.suffix_generator_execute_steps),
        "causal_suffix_diffusion_checkpoint": str(Path(args.causal_suffix_diffusion_checkpoint).resolve())
        if str(args.causal_suffix_diffusion_checkpoint).strip()
        else None,
        "causal_suffix_diffusion_offsets": parse_ints(str(args.causal_suffix_diffusion_offsets))
        if str(args.causal_suffix_diffusion_checkpoint).strip()
        else [],
        "causal_suffix_diffusion_samples_per_offset": int(args.causal_suffix_diffusion_samples_per_offset),
        "causal_suffix_diffusion_execute_steps": int(args.causal_suffix_diffusion_execute_steps),
        "causal_suffix_diffusion_temperature": float(args.causal_suffix_diffusion_temperature),
        "source_suffix_k": int(args.source_suffix_k),
        "source_suffix_blends": parse_floats(str(args.source_suffix_blends))
        if str(args.source_insertion_suffix_bank).strip()
        else [],
        "source_suffix_execute_steps": int(args.source_suffix_execute_steps),
        "records": len(records),
        "valid_records": len(valid),
        "failure_counts": dict(sorted(failures.items())),
        "candidate_name_counts": dict(sorted(Counter(str(row.get("candidate_name")) for row in valid).items())),
        "selected_records": int(sum(1 for row in valid if row.get("candidate_selected_by_live_scorer"))),
        "source_suffix_records": int(
            sum(
                1
                for row in valid
                if (row.get("candidate_meta") or {}).get("candidate_source") == "source_insertion_suffix"
            )
        ),
        "suffix_generator_records": int(
            sum(
                1
                for row in valid
                if (row.get("candidate_meta") or {}).get("candidate_source") == "suffix_action_generator"
            )
        ),
        "causal_suffix_diffusion_records": int(
            sum(
                1
                for row in valid
                if (row.get("candidate_meta") or {}).get("candidate_source") == "causal_suffix_diffusion"
            )
        ),
        "after_success_count": int(sum(1 for row in valid if row.get("after_success"))),
        "after_gate_ok_count": int(sum(1 for row in valid if (row.get("after_continuability_gate") or {}).get("ok"))),
        "dp_rollout_label_count": int(len(dp_rollout_rows)),
        "dp_rollout_success_count": int(
            sum(1 for row in dp_rollout_rows if (row.get("dp_rollout_continuability") or {}).get("success"))
        ),
        "dp_rollout_continuable_count": int(
            sum(1 for row in dp_rollout_rows if (row.get("dp_rollout_continuability") or {}).get("continuable"))
        ),
        "dp_rollout_final_contact_stable_count": int(
            sum(
                1
                for row in dp_rollout_rows
                if (row.get("dp_rollout_continuability") or {}).get("final_contact_stable")
            )
        ),
        "worsened_abs_yz_sum_count": int(sum(1 for row in valid if float(row.get("delta_abs_yz_sum", 0.0)) > 0.0)),
        "improved_abs_yz_sum_count": int(sum(1 for row in valid if float(row.get("delta_abs_yz_sum", 0.0)) < 0.0)),
        "boundary": (
            "This is live-snapshot replay labeling for action-consequence "
            "calibration. It should be used to train/audit a live outcome "
            "scorer, not reported as closed-loop method success."
        ),
    }
    write_json(output_root / "live_snapshot_action_bank_outcome_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
