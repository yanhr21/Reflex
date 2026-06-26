#!/usr/bin/env python3
"""Convert live-snapshot replay labels into outcome-scorer training rows.

The converter preserves the live selector feature contract: current live state,
Cosmos task path, frozen-DP prior, live contact context, candidate action, and
candidate residual. It reconstructs the base feature once per live snapshot and
writes the same JSONL/NPZ layout consumed by
train_cosmos3_candidate_outcome_scorer.py.
"""

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

from export_cosmos3_candidate_outcome_labels import contact_progress_proxy  # noqa: E402
from replay_cosmos3_live_action_bank_from_snapshots import (  # noqa: E402
    causal_suffix_diffusion_feature,
    load_causal_suffix_diffusion,
    load_snapshot_state,
    load_history,
    read_json,
    sample_causal_suffix_diffusion_action,
)
from run_cosmos3_live_receding_loop import (  # noqa: E402
    contact_context_from_live,
    current_executor_state_from_live,
    live_pose_row,
    require_compute_step,
    task_path_from_cosmos_prediction,
)
from run_cosmos3_receding_closed_loop import (  # noqa: E402
    _get_base_env,
    _import_live_control_stack,
    _make_live_env,
    _parse_seed_from_text,
    jsonable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-snapshot-jsonl", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--condition-root", required=True)
    parser.add_argument(
        "--dp-manifest",
        default=str(ROOT / "experiments/dp_peg1000/run_90201/manifest.json"),
    )
    parser.add_argument("--robot-action-dim", type=int, default=7)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument(
        "--snapshot-namespace",
        default="",
        help=(
            "Optional stable namespace for grouping rows from multiple replay "
            "JSONLs that share the same live snapshot states. Leave empty to "
            "derive it from each input JSONL parent directory."
        ),
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


def iter_jsonl_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for text in inputs:
        path = Path(text).resolve()
        if path.is_dir():
            paths.extend(sorted(path.rglob("live_snapshot_action_bank_outcome_labels.jsonl")))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(path)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def read_live_labels(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                if row.get("schema") == "cosmos3_live_snapshot_action_bank_outcome_label_v1":
                    copied = dict(row)
                    copied["_live_snapshot_jsonl"] = str(path)
                    rows.append(copied)
    if not rows:
        raise RuntimeError("no live snapshot outcome labels found")
    return rows


def sanitize(text: Any, limit: int = 120) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    return out[:limit].strip("_") or "item"


def phase_name_to_id(name: str) -> int:
    mapping = {
        "no_grasp": 0,
        "far": 1,
        "lateral_align": 2,
        "preinsert_aligned": 3,
        "dp_continuable": 4,
        "inserted": 5,
    }
    return mapping.get(str(name), 1)


def threshold_args_from_gate(args: argparse.Namespace, gate: dict[str, Any]) -> argparse.Namespace:
    thresholds = gate.get("thresholds") or {}
    return argparse.Namespace(
        continuability_min_rel_x=float(thresholds.get("min_rel_x", -0.08)),
        continuability_max_rel_x=float(thresholds.get("max_rel_x", 0.04)),
        continuability_max_abs_y=float(thresholds.get("max_abs_y", 0.025)),
        continuability_max_abs_z=float(thresholds.get("max_abs_z", 0.025)),
        continuability_max_hole_speed=float(thresholds.get("max_hole_speed", 0.01)),
        max_episode_steps=int(args.max_episode_steps),
    )


def make_contact_label_npz(
    path: Path,
    *,
    context: dict[str, Any],
    prefix_frame: int,
    max_episode_steps: int,
) -> None:
    rows = max(int(max_episode_steps) + 1, int(prefix_frame) + 2)
    progress = np.full((rows,), float(context.get("contact_progress", 0.0)), dtype=np.float32)
    phase_id = np.full(
        (rows,),
        int(context.get("phase_id", phase_name_to_id(str(context.get("phase_name", "far"))))),
        dtype=np.int32,
    )
    grasped = np.full((rows,), bool(context.get("grasped", False)), dtype=bool)
    inserted = np.full((rows,), bool(context.get("inserted", False)), dtype=bool)
    dp_continuable = np.full((rows,), bool(context.get("dp_continuable", False)), dtype=bool)
    rel = np.asarray(context.get("peg_head_at_hole", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)[:3]
    if rel.size < 3:
        rel = np.pad(rel, (0, 3 - rel.size), mode="constant")
    peg_head = np.repeat(rel.reshape(1, 3), rows, axis=0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        contact_progress=progress,
        phase_id=phase_id,
        grasped=grasped,
        inserted=inserted,
        dp_continuable=dp_continuable,
        peg_head_at_hole=peg_head,
    )


def source_suffix_live_name(meta: dict[str, Any], fallback: str) -> str:
    if "source_suffix_rank" not in meta:
        return fallback
    rank = int(meta.get("source_suffix_rank", 0))
    offset = int(meta.get("source_suffix_offset_before_insert", -1))
    blend = float(meta.get("source_suffix_blend", 1.0))
    blend_tag = f"{blend:g}".replace(".", "p")
    return f"retrieval_resid_srcsuffix_r{rank}_s{blend_tag}_o{offset}"


def snapshot_namespace(row: dict[str, Any], override: str = "") -> str:
    """Keep same sample/iteration from different live panels as separate states."""
    if str(override).strip():
        return sanitize(str(override), 40)
    path_text = str(row.get("_live_snapshot_jsonl") or "")
    if path_text:
        parent = Path(path_text).resolve().parent.name
        if parent:
            return sanitize(parent, 40)
    return "live"


def load_source_bank(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(str(path), allow_pickle=False)
    return {key: payload[key] for key in payload.files}


def reconstruct_candidate_actions(
    *,
    row: dict[str, Any],
    bank: dict[str, np.ndarray],
    dp_prior: np.ndarray,
    robot_action_dim: int,
    causal_diffusion_cache: dict[str, dict[str, Any]],
    seed: int,
) -> tuple[str, str, np.ndarray, dict[str, Any]]:
    candidate_meta = dict(row.get("candidate_meta") or {})
    name = str(row.get("candidate_name") or "")
    horizon = int(dp_prior.shape[0])
    if candidate_meta.get("candidate_source") == "causal_suffix_diffusion" or name.startswith("causal_suffix_diffusion_"):
        ckpt = Path(str(candidate_meta.get("causal_suffix_diffusion_checkpoint") or "")).resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"causal suffix diffusion checkpoint missing for {name}: {ckpt}")
        cache_key = str(ckpt)
        if cache_key not in causal_diffusion_cache:
            loaded = load_causal_suffix_diffusion(cache_key)
            if loaded is None:
                raise RuntimeError(f"failed to load causal suffix diffusion checkpoint {ckpt}")
            causal_diffusion_cache[cache_key] = loaded
        diffusion = causal_diffusion_cache[cache_key]
        metadata = diffusion.get("metadata") or {}
        offset_values = [int(x) for x in metadata.get("offset_values", [])]
        if not offset_values:
            raise ValueError(f"causal suffix diffusion checkpoint lacks offset_values: {ckpt}")
        offset = int(candidate_meta.get("causal_suffix_diffusion_condition_offset_before_insert"))
        sample_i = int(candidate_meta.get("causal_suffix_diffusion_sample_index", 0))
        prefix_frame = int(candidate_meta.get("causal_suffix_diffusion_prefix_frame", row.get("prefix_frame_index", 0)))
        query_rel = np.asarray(
            candidate_meta.get("causal_suffix_diffusion_query_peg_head_at_hole")
            or row.get("before_peg_head_at_hole")
            or [0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        query_grasped = bool(candidate_meta.get("causal_suffix_diffusion_query_grasped", False))
        temperature = float(candidate_meta.get("causal_suffix_diffusion_temperature", 1.0))
        feature = causal_suffix_diffusion_feature(
            query_rel=query_rel,
            offset=offset,
            prefix_frame=prefix_frame,
            grasped=query_grasped,
            offset_values=offset_values,
        )
        sample_seed = int(seed) + int(offset) * 1009 + int(prefix_frame) * 17 + int(sample_i)
        generated = sample_causal_suffix_diffusion_action(
            diffusion=diffusion,
            feature=feature,
            sample_seed=sample_seed,
            temperature=temperature,
        )[:, :robot_action_dim]
        local = min(horizon, int(generated.shape[0]))
        if local <= 0:
            raise RuntimeError(f"empty causal suffix diffusion action for {name}")
        execute_steps = int(row.get("execute_steps_actual") or row.get("execute_steps_requested") or local)
        execute_steps = max(1, min(int(execute_steps), local, horizon))
        actions = np.asarray(dp_prior[:horizon, :robot_action_dim], dtype=np.float32).copy()
        actions[:execute_steps, :robot_action_dim] = generated[:execute_steps, :robot_action_dim]
        candidate_meta.update(
            {
                "candidate_action_feature_mode": "executed_generated_steps_then_dp_prior_fill",
                "candidate_action_reconstructed_execute_steps": int(execute_steps),
                "candidate_action_reconstructed_generated_horizon": int(generated.shape[0]),
                "candidate_action_reconstructed_sample_seed": int(sample_seed),
            }
        )
        return name, "causal_suffix_diffusion", actions.astype(np.float32), candidate_meta

    if "source_suffix_index" in candidate_meta:
        source_bank_path = Path(str(candidate_meta["source_suffix_bank_npz"])).resolve()
        source_bank = load_source_bank(source_bank_path)
        source_idx = int(candidate_meta["source_suffix_index"])
        blend = float(candidate_meta.get("source_suffix_blend", 1.0))
        source_actions = np.asarray(source_bank["actions"], dtype=np.float32)[source_idx, :, :robot_action_dim]
        local = min(horizon, int(source_actions.shape[0]))
        if local <= 0:
            raise RuntimeError(f"empty source suffix action for {name}")
        actions = (1.0 - blend) * dp_prior[:local, :robot_action_dim] + blend * source_actions[:local]
        if local < horizon:
            pad = np.repeat(actions[-1:, :], horizon - local, axis=0)
            actions = np.concatenate([actions, pad], axis=0)
        live_name = source_suffix_live_name(candidate_meta, name)
        return live_name, "retrieval_success_residual", actions.astype(np.float32), candidate_meta

    idx = int(row.get("candidate_index", -1))
    names = [str(item) for item in np.asarray(bank["candidate_names"]).tolist()]
    if idx < 0 or idx >= len(names):
        raise IndexError(f"candidate_index {idx} outside saved bank with {len(names)} names")
    actions = np.asarray(bank["candidate_full_actions"], dtype=np.float32)[idx, :horizon, :robot_action_dim]
    return names[idx], "checkpoint_model", actions.astype(np.float32), candidate_meta


def dp_rollout_summary(row: dict[str, Any]) -> dict[str, Any]:
    label = row.get("dp_rollout_continuability")
    if not isinstance(label, dict):
        return {
            "dp_rollout_continuable_proxy": False,
            "dp_rollout_success": False,
            "dp_rollout_final_contact_stable_proxy": False,
            "dp_rollout_stable_step_count": 0,
        }
    final_eval = label.get("final_eval") or {}
    return {
        "dp_rollout_continuable_proxy": bool(label.get("continuable", False)),
        "dp_rollout_success": bool(label.get("success", False) or final_eval.get("success", False)),
        "dp_rollout_final_contact_stable_proxy": bool(label.get("final_contact_stable", False)),
        "dp_rollout_stable_step_count": int(label.get("stable_step_count", 0)),
    }


def make_base_snapshot(
    *,
    iter_dir: Path,
    exemplar: dict[str, Any],
    env: Any,
    base_env: Any,
    stack: dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    bank_npz = np.load(str(iter_dir / "candidate_action_bank.npz"), allow_pickle=False)
    bank = {key: bank_npz[key] for key in bank_npz.files}
    dp_prior = np.asarray(bank["dp_prior_actions"], dtype=np.float32)[:, : int(args.robot_action_dim)]
    horizon = int(dp_prior.shape[0])
    prefix_frame = int(np.asarray(bank.get("prefix_frame_index", [-1])).reshape(-1)[0])
    iteration = int(np.asarray(bank.get("iteration", [-1])).reshape(-1)[0])
    if prefix_frame < 0:
        raise ValueError(f"{iter_dir} missing prefix_frame_index")

    source_h5 = Path(str(exemplar["source_h5"])).resolve()
    reset_seed = _parse_seed_from_text(" ".join([source_h5.name, str(exemplar.get("sample_name", ""))]))
    env.reset(seed=int(reset_seed if reset_seed is not None else args.seed))
    snapshot_state, _snapshot_attrs = load_snapshot_state(iter_dir / "live_state_before_controller.h5")
    base_env.set_state_dict(snapshot_state)
    live = live_pose_row(base_env, stack, None)
    history = load_history(iter_dir / "live_history_raw_action_state.json")
    threshold_args = threshold_args_from_gate(args, dict(exemplar.get("before_continuability_gate") or {}))
    live_context = contact_context_from_live(
        live=live,
        history=history,
        prefix_frame=prefix_frame,
        horizon=horizon,
        args=threshold_args,
    )
    current_state = current_executor_state_from_live(base_env, stack, live)

    chunk = read_json(iter_dir / "candidate_executor_action_chunk.json")
    action_chunk_path = Path(str(chunk.get("raw_cosmos_action_chunk_json") or ""))
    if not action_chunk_path.is_file():
        action_chunk_path = iter_dir / "cosmos_live_prefix" / "live_prefix_action_chunk.json"
    action_chunk = read_json(action_chunk_path)
    sample_output_json = Path(str(action_chunk.get("sample_output_json") or chunk.get("cosmos_task_path", {}).get("sample_output_json"))).resolve()
    task_result = task_path_from_cosmos_prediction(
        sample_output_json=sample_output_json,
        condition_root=Path(args.condition_root).resolve(),
        prefix_frame=prefix_frame,
        horizon=horizon,
    )

    panel_namespace = snapshot_namespace(exemplar, str(getattr(args, "snapshot_namespace", "") or ""))
    uuid = sanitize(
        f"{panel_namespace}__{exemplar.get('sample_name')}__iter{iteration:02d}__f{prefix_frame:03d}",
        220,
    )
    base_dir = output_root / "base_features"
    executor_npz = base_dir / f"{uuid}__executor_sample.npz"
    dp_npz = base_dir / f"{uuid}__dp_prior.npz"
    contact_npz = base_dir / f"{uuid}__contact_labels.npz"
    executor_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        executor_npz,
        current_state=current_state.astype(np.float32),
        task_path=np.asarray(task_result["task_path"], dtype=np.float32),
        teacher_robot_actions=dp_prior.astype(np.float32),
    )
    np.savez_compressed(dp_npz, dp_prior_actions=dp_prior.astype(np.float32))
    make_contact_label_npz(
        contact_npz,
        context=live_context,
        prefix_frame=prefix_frame,
        max_episode_steps=int(args.max_episode_steps),
    )
    return {
        "uuid": uuid,
        "snapshot_namespace": panel_namespace,
        "source_uuid": sanitize(Path(str(source_h5)).stem, 180),
        "source_h5": str(source_h5),
        "sample_name": exemplar.get("sample_name"),
        "scenario": exemplar.get("scenario"),
        "split": "live_snapshot_replay",
        "prefix_role": "target_motion_observed",
        "current_phase": str(live_context.get("phase_name")),
        "prefix_frame_index": int(prefix_frame),
        "action_start_step": int(prefix_frame),
        "horizon": int(horizon),
        "current_contact_progress": float(live_context.get("contact_progress", 0.0)),
        "task_path_source": task_result.get("source"),
        "executor_sample_npz": str(executor_npz),
        "dp_prior_npz": str(dp_npz),
        "contact_label_npz": str(contact_npz),
        "_bank": bank,
        "_dp_prior": dp_prior,
    }


def main() -> int:
    args = parse_args()
    require_compute_step()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    input_paths = iter_jsonl_paths(list(args.live_snapshot_jsonl))
    live_rows = read_live_labels(input_paths)

    stack = _import_live_control_stack(ROOT)
    env = _make_live_env(stack, Path(args.dp_manifest).resolve(), args)
    base_env = _get_base_env(env)

    base_cache: dict[Path, dict[str, Any]] = {}
    base_rows_public: dict[str, dict[str, Any]] = {}
    outcome_rows: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    action_dir = output_root / "candidate_actions"
    causal_diffusion_cache: dict[str, dict[str, Any]] = {}

    try:
        for row_index, row in enumerate(live_rows):
            try:
                iter_dir = Path(str(row["iter_dir"])).resolve()
                if iter_dir not in base_cache:
                    base_cache[iter_dir] = make_base_snapshot(
                        iter_dir=iter_dir,
                        exemplar=row,
                        env=env,
                        base_env=base_env,
                        stack=stack,
                        args=args,
                        output_root=output_root,
                    )
                    public = {k: v for k, v in base_cache[iter_dir].items() if not k.startswith("_")}
                    base_rows_public[public["uuid"]] = public
                base = base_cache[iter_dir]
                candidate_name, candidate_source, actions, candidate_meta = reconstruct_candidate_actions(
                    row=row,
                    bank=base["_bank"],
                    dp_prior=base["_dp_prior"],
                    robot_action_dim=int(args.robot_action_dim),
                    causal_diffusion_cache=causal_diffusion_cache,
                    seed=int(args.seed),
                )
                action_path = action_dir / f"{row_index:06d}_{base['uuid']}__{sanitize(candidate_name, 80)}.npz"
                action_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(action_path, actions=actions.astype(np.float32))

                rel = np.asarray(row.get("after_peg_head_at_hole", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)[:3]
                if rel.size < 3:
                    rel = np.pad(rel, (0, 3 - rel.size), mode="constant")
                progress = row.get("after_contact_progress_proxy")
                if progress is None:
                    progress = contact_progress_proxy(
                        rel,
                        grasped=bool(row.get("after_grasped", False)),
                        inserted=bool(row.get("after_inserted_live_pose", False)),
                    )
                current_progress = float(base.get("current_contact_progress", 0.0))
                dp_summary = dp_rollout_summary(row)
                final_continuable = bool(
                    dp_summary["dp_rollout_continuable_proxy"]
                    or row.get("after_success", False)
                    or row.get("after_inserted_live_pose", False)
                    or row.get("after_contact_stable_proxy", False)
                )
                outcome_rows.append(
                    {
                        "schema": "cosmos3_candidate_outcome_label_v1",
                        "uuid": base["uuid"],
                        "source_uuid": base["source_uuid"],
                        "source_h5": base["source_h5"],
                        "scenario": base.get("scenario"),
                        "split": "live_snapshot_replay",
                        "prefix_role": "target_motion_observed",
                        "current_phase": base.get("current_phase"),
                        "prefix_frame_index": int(base["prefix_frame_index"]),
                        "horizon": int(base["horizon"]),
                        "candidate_name": candidate_name,
                        "candidate_source": candidate_source,
                        "candidate_actions_npz": str(action_path),
                        "candidate_action_stats": {
                            "num_values": int(actions.size),
                            "mean_abs": float(np.mean(np.abs(actions))) if actions.size else None,
                            "max_abs": float(np.max(np.abs(actions))) if actions.size else None,
                        },
                        "candidate_meta": candidate_meta,
                        "live_snapshot_jsonl": row.get("_live_snapshot_jsonl"),
                        "live_snapshot_iter_dir": str(iter_dir),
                        "live_snapshot_candidate_index": row.get("candidate_index"),
                        "final_peg_head_at_hole": rel.astype(float).tolist(),
                        "final_abs_task_error_weighted": float(abs(rel[0]) + 2.0 * abs(rel[1]) + 4.0 * abs(rel[2])),
                        "final_lateral_error_yz": float(np.linalg.norm(rel[1:3])),
                        "final_grasped": bool(row.get("after_grasped", False)),
                        "final_inserted_live_pose": bool(row.get("after_inserted_live_pose", False)),
                        "final_contact_stable_proxy": bool(row.get("after_contact_stable_proxy", False)),
                        "final_success": bool(row.get("after_success", False)),
                        "executed_steps": int(row.get("execute_steps_actual", 0)),
                        "final_contact_progress_proxy": float(progress),
                        "final_contact_progress_delta_proxy": float(progress) - current_progress,
                        "final_continuable_proxy": final_continuable,
                        **dp_summary,
                        "boundary": (
                            "Converted from real live-snapshot candidate replay. "
                            "The label supervises scorer calibration only; live "
                            "method evidence still requires closed-loop execution "
                            "and video/contact review."
                        ),
                    }
                )
            except Exception as exc:
                failures[type(exc).__name__] += 1
                outcome_rows.append(
                    {
                        "schema": "cosmos3_candidate_outcome_label_failure_v1",
                        "row_index": int(row_index),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    finally:
        if hasattr(env, "close"):
            env.close()

    base_rows = list(base_rows_public.values())
    write_jsonl(output_root / "live_snapshot_base_rows.jsonl", base_rows)
    write_jsonl(output_root / "candidate_outcome_labels.jsonl", outcome_rows)
    valid = [row for row in outcome_rows if row.get("schema") == "cosmos3_candidate_outcome_label_v1"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in valid:
        grouped.setdefault(str(row.get("uuid") or ""), []).append(row)
    groups_with_dp = sum(1 for rows in grouped.values() if any(str(r.get("candidate_name")) == "dp_prior" for r in rows))
    groups_with_source_success = sum(
        1
        for rows in grouped.values()
        if any(
            str(r.get("candidate_name", "")).startswith("retrieval_resid_srcsuffix_")
            and bool(r.get("dp_rollout_success", False))
            for r in rows
        )
    )
    groups_with_causal_success = sum(
        1
        for rows in grouped.values()
        if any(
            str(r.get("candidate_source", "")) == "causal_suffix_diffusion"
            and bool(r.get("dp_rollout_success", False))
            for r in rows
        )
    )
    summary = {
        "schema": "cosmos3_live_snapshot_to_outcome_scorer_conversion_v1",
        "input_jsonl": [str(path) for path in input_paths],
        "output_root": str(output_root),
        "base_rows_jsonl": str(output_root / "live_snapshot_base_rows.jsonl"),
        "candidate_outcome_jsonl": str(output_root / "candidate_outcome_labels.jsonl"),
        "valid_rows": int(len(valid)),
        "failure_rows": int(len(outcome_rows) - len(valid)),
        "failure_types": dict(sorted(failures.items())),
        "base_groups": int(len(base_rows)),
        "groups_with_dp_prior": int(groups_with_dp),
        "groups_with_source_suffix_dp96_success": int(groups_with_source_success),
        "groups_with_causal_suffix_diffusion_dp96_success": int(groups_with_causal_success),
        "candidate_source_counts": dict(sorted(Counter(str(row.get("candidate_source") or "") for row in valid).items())),
        "candidate_family_counts": dict(
            sorted(
                Counter(
                    "source_suffix"
                    if str(row.get("candidate_name", "")).startswith("retrieval_resid_srcsuffix_")
                    else str(row.get("candidate_name", "")).split("_", 1)[0]
                    for row in valid
                ).items()
            )
        ),
        "dp_rollout_success_count": int(sum(1 for row in valid if row.get("dp_rollout_success"))),
        "dp_rollout_continuable_count": int(sum(1 for row in valid if row.get("dp_rollout_continuable_proxy"))),
        "boundary": (
            "Dataset conversion only. It joins live replay labels with the "
            "same feature contract used by live scorer selection."
        ),
    }
    write_json(output_root / "conversion_summary.json", summary)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True), flush=True)
    if not valid:
        return 2
    if groups_with_dp <= 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
