#!/usr/bin/env python3
"""Audit the diffusion candidate-executor chain before formal launch.

It reads manifests, JSONL rows, and small NPZ metadata through the same loader
used by the formal trainer. Under the current project execution rules it must
be launched inside a tmux-held Slurm allocation, not directly on the login
node.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_cosmos3_executor_training_dataset import CURRENT_STATE_NAMES, TASK_PATH_NAMES  # noqa: E402
from train_cosmos3_contact_executor import load_arrays, read_jsonl  # noqa: E402


DEFAULT_CONTACT_EXECUTOR_JSONL = (
    "experiments/world_model_task_rebinding/cosmos3/"
    "contact_executor_dataset_iter1500_train512_scale005_repair_20260615/"
    "contact_executor_dataset_file.jsonl"
)
DEFAULT_SFT_ROOT = (
    "experiments/world_model_task_rebinding/cosmos3/"
    "sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_"
    "fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299"
)
DEFAULT_EVAL_ROOT = (
    "experiments/world_model_task_rebinding/cosmos3/"
    "sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_"
    "fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/"
    "eval_full_episode_wam_iter_000001500_formal_after_3h_abs4gpu_retry2"
)
DEFAULT_CONDITION_ROOT = (
    "experiments/world_model_task_rebinding/cosmos3/"
    "full_episode_wam_conditions_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_"
    "20260614_130050"
)
DEFAULT_SOURCE_H5_ROOT = (
    "experiments/world_model_task_rebinding/cosmos3/"
    "fix3_v7_dp_user_override_sft_source_20260612_733/canonical_h5"
)
DEFAULT_DP_MANIFEST = "experiments/dp_peg1000/run_90201/manifest.json"
DEFAULT_DP_CHECKPOINT = "experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_at_end.pt"
DEFAULT_CONTINUABILITY_STATS = (
    "experiments/world_model_task_rebinding/cosmos3/"
    "dp_static_continuability_stats_20260613/dp_static_continuability_stats.json"
)
DEFAULT_OUTPUT_ROOT = (
    "experiments/world_model_task_rebinding/cosmos3/"
    "candidate_executor_train_20260616_formal_2gpu_alloc131564_diffusion_rankcal_finalgate"
)
SFT_RUN_REL = Path("outputs/cosmos3/sft/vision_sft_droid_policy_full1000_rgb_300step_wam")
LIVE_CONTACT_CONTEXT_WIDTH = 15
ROBOT_ACTION_DIM = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--contact-executor-jsonl", default=DEFAULT_CONTACT_EXECUTOR_JSONL)
    parser.add_argument("--sft-root", default=DEFAULT_SFT_ROOT)
    parser.add_argument("--checkpoint-iter", type=int, default=1500)
    parser.add_argument("--eval-root", default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--condition-root", default=DEFAULT_CONDITION_ROOT)
    parser.add_argument("--source-h5-root", default=DEFAULT_SOURCE_H5_ROOT)
    parser.add_argument("--dp-manifest", default=DEFAULT_DP_MANIFEST)
    parser.add_argument("--dp-checkpoint", default=DEFAULT_DP_CHECKPOINT)
    parser.add_argument("--continuability-stats-json", default=DEFAULT_CONTINUABILITY_STATS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--planned-generator-type", default="diffusion")
    parser.add_argument("--planned-candidate-samples", type=int, default=8)
    parser.add_argument("--planned-rank-diffusion-count", type=int, default=1)
    parser.add_argument("--planned-nproc-per-node", type=int, default=2)
    parser.add_argument("--planned-min-wall-seconds", type=int, default=10800)
    parser.add_argument("--ready-blocker-class", default="slurm_resource_pending")
    parser.add_argument("--sample-npz-count", type=int, default=3)
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def path_status(path: Path, *, kind: str = "file", required: bool = True) -> dict[str, Any]:
    exists = path.is_dir() if kind == "dir" else path.is_file()
    nonempty = True
    if exists and kind == "file":
        nonempty = path.stat().st_size > 0
    ok = bool(exists and nonempty)
    return {
        "path": str(path),
        "kind": kind,
        "required": bool(required),
        "exists": bool(exists),
        "nonempty": bool(nonempty),
        "ok": ok if required else True,
    }


def count_missing_rows(rows: list[dict[str, Any]], root: Path, key: str) -> tuple[int, list[str]]:
    missing: list[str] = []
    for row in rows:
        value = row.get(key)
        if not value or not resolve(root, str(value)).exists():
            missing.append(str(row.get("uuid") or row.get("source_uuid") or "<unknown>"))
    return len(missing), missing[:10]


def npz_shape_summary(rows: list[dict[str, Any]], root: Path, sample_count: int) -> list[dict[str, Any]]:
    if not rows or sample_count <= 0:
        return []
    indices = sorted({0, len(rows) // 2, len(rows) - 1})[:sample_count]
    samples: list[dict[str, Any]] = []
    for index in indices:
        row = rows[index]
        item: dict[str, Any] = {
            "row_index": int(index),
            "uuid": row.get("uuid"),
            "prefix_role": row.get("prefix_role"),
            "current_phase": row.get("current_phase"),
        }
        try:
            executor_npz = np.load(resolve(root, str(row["executor_sample_npz"])), allow_pickle=False)
            item["executor_sample_shapes"] = {
                name: list(executor_npz[name].shape) for name in executor_npz.files
            }
        except Exception as exc:  # pragma: no cover - audit report path
            item["executor_sample_error"] = repr(exc)
        try:
            prior_npz = np.load(resolve(root, str(row["dp_prior_npz"])), allow_pickle=False)
            item["dp_prior_shapes"] = {name: list(prior_npz[name].shape) for name in prior_npz.files}
        except Exception as exc:  # pragma: no cover - audit report path
            item["dp_prior_error"] = repr(exc)
        try:
            labels = np.load(resolve(root, str(row["contact_label_npz"])), allow_pickle=False)
            keep = ("contact_progress", "phase_id", "inserted", "grasped", "dp_continuable")
            item["contact_label_shapes"] = {
                name: list(labels[name].shape) for name in keep if name in labels.files
            }
        except Exception as exc:  # pragma: no cover - audit report path
            item["contact_label_error"] = repr(exc)
        samples.append(item)
    return samples


def audit_rows(root: Path, jsonl_path: Path, sample_npz_count: int) -> dict[str, Any]:
    rows = read_jsonl(jsonl_path)
    task_sources = Counter(str(row.get("task_path_source") or "") for row in rows)
    prefix_roles = Counter(str(row.get("prefix_role") or "") for row in rows)
    phases = Counter(str(row.get("current_phase") or "") for row in rows)
    scenarios = Counter(str(row.get("scenario") or "") for row in rows)
    debug_rows = [
        str(row.get("uuid") or row.get("source_uuid") or "<unknown>")
        for row in rows
        if "gt" in str(row.get("task_path_source") or "")
        and "debug" in str(row.get("task_path_source") or "")
    ]
    missing_by_key: dict[str, Any] = {}
    for key in ("executor_sample_npz", "dp_prior_npz", "contact_label_npz", "source_h5"):
        count, examples = count_missing_rows(rows, root, key)
        missing_by_key[key] = {"count": int(count), "examples": examples}

    load_error = None
    shapes: dict[str, Any] = {}
    feature_contract: dict[str, Any] = {}
    try:
        x_raw, y_raw, prior_raw, progress_raw, binary_raw, used_rows = load_arrays(rows, 0)
        target_dim = int(y_raw.shape[1])
        horizon = int(target_dim // ROBOT_ACTION_DIM) if target_dim % ROBOT_ACTION_DIM == 0 else -1
        expected_feature_dim = (
            len(CURRENT_STATE_NAMES)
            + horizon * len(TASK_PATH_NAMES)
            + horizon * ROBOT_ACTION_DIM
            + LIVE_CONTACT_CONTEXT_WIDTH
        )
        shapes = {
            "x": list(x_raw.shape),
            "y": list(y_raw.shape),
            "prior": list(prior_raw.shape),
            "progress": list(progress_raw.shape),
            "binary": list(binary_raw.shape),
            "used_rows": int(len(used_rows)),
        }
        feature_contract = {
            "current_state_width": len(CURRENT_STATE_NAMES),
            "task_path_width": len(TASK_PATH_NAMES),
            "robot_action_dim": ROBOT_ACTION_DIM,
            "live_contact_context_width": LIVE_CONTACT_CONTEXT_WIDTH,
            "target_dim": target_dim,
            "horizon": horizon,
            "expected_feature_dim": expected_feature_dim,
            "actual_feature_dim": int(x_raw.shape[1]),
            "actual_prior_dim": int(prior_raw.shape[1]),
            "ok": bool(
                horizon > 0
                and int(x_raw.shape[1]) == expected_feature_dim
                and int(prior_raw.shape[1]) == target_dim
                and int(progress_raw.shape[1]) == 2
                and int(binary_raw.shape[1]) == 2
            ),
        }
    except BaseException as exc:  # noqa: BLE001 - audit must record trainer refusal
        load_error = repr(exc)

    return {
        "path": str(jsonl_path),
        "rows": int(len(rows)),
        "task_path_sources": dict(sorted(task_sources.items())),
        "prefix_roles": dict(sorted(prefix_roles.items())),
        "current_phases": dict(sorted(phases.items())),
        "scenarios": dict(sorted(scenarios.items())),
        "future_inserted_true": int(sum(bool(row.get("future_inserted_within_chunk")) for row in rows)),
        "future_dp_continuable_true": int(sum(bool(row.get("future_dp_continuable_within_chunk")) for row in rows)),
        "gt_debug_row_count": int(len(debug_rows)),
        "gt_debug_examples": debug_rows[:10],
        "missing_by_key": missing_by_key,
        "load_arrays_error": load_error,
        "load_arrays_shapes": shapes,
        "feature_contract": feature_contract,
        "sample_npz_shapes": npz_shape_summary(rows, root, sample_npz_count),
    }


def audit_static_dependencies(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    sft_root = resolve(root, args.sft_root)
    sft_run = sft_root / SFT_RUN_REL
    ckpt_dir = sft_run / "checkpoints" / f"iter_{int(args.checkpoint_iter):09d}"
    required = {
        "sft_root": path_status(sft_root, kind="dir"),
        "sft_run_dir": path_status(sft_run, kind="dir"),
        "sft_checkpoint_dir": path_status(ckpt_dir, kind="dir"),
        "eval_root": path_status(resolve(root, args.eval_root), kind="dir"),
        "condition_root": path_status(resolve(root, args.condition_root), kind="dir"),
        "source_h5_root": path_status(resolve(root, args.source_h5_root), kind="dir"),
        "dp_manifest": path_status(resolve(root, args.dp_manifest), kind="file"),
        "dp_checkpoint": path_status(resolve(root, args.dp_checkpoint), kind="file"),
        "continuability_stats_json": path_status(resolve(root, args.continuability_stats_json), kind="file"),
    }
    optional = {
        "sft_config_yaml": path_status(sft_run / "config.yaml", kind="file", required=False),
    }
    missing_required = [name for name, item in required.items() if not item["ok"]]
    return {
        "required": required,
        "optional": optional,
        "missing_required": missing_required,
        "ok": not missing_required,
    }


def planned_config(args: argparse.Namespace) -> dict[str, Any]:
    ok = (
        args.planned_generator_type == "diffusion"
        and int(args.planned_candidate_samples) > 0
        and int(args.planned_rank_diffusion_count) > 0
        and int(args.planned_nproc_per_node) >= 1
        and int(args.planned_min_wall_seconds) >= 10800
    )
    return {
        "generator_type": str(args.planned_generator_type),
        "candidate_samples": int(args.planned_candidate_samples),
        "candidate_rank_diffusion_count": int(args.planned_rank_diffusion_count),
        "nproc_per_node": int(args.planned_nproc_per_node),
        "minimum_allowed_nproc_per_node": 1,
        "min_wall_seconds": int(args.planned_min_wall_seconds),
        "minimum_allowed_min_wall_seconds": 10800,
        "ok": bool(ok),
    }


def render_md(payload: dict[str, Any]) -> str:
    rows = payload["training_input"]
    deps = payload["static_dependencies"]
    contract = rows.get("feature_contract") or {}
    blocker_class = str(payload.get("blocker_class", ""))
    if payload.get("ready_for_allocation_launch") and blocker_class == "ready_for_chain_launch":
        interpretation = (
            "This audit ran inside the active Slurm allocation and cleared the "
            "row/path/feature-contract gate. The chain may proceed to CUDA "
            "canary, smoke, formal training, or live eval according to the "
            "watcher state."
        )
    else:
        interpretation = (
            "If this audit is ready but the chain has not started training, the "
            "current blocker is Slurm allocation availability, not known bad "
            "training rows, missing Cosmos/DP assets, or a feature-width "
            "mismatch."
        )
    lines = [
        "# Candidate Executor Diffusion Prelaunch Audit",
        "",
        f"Date: {payload['date']}",
        "",
        "## Verdict",
        "",
        f"- ready_for_allocation_launch: `{payload['ready_for_allocation_launch']}`",
        f"- blocker_class: `{payload['blocker_class']}`",
        "- method evidence: `false` (this is a prelaunch audit only)",
        "",
        "## Training Input",
        "",
        f"- jsonl: `{rows['path']}`",
        f"- rows: `{rows['rows']}`",
        f"- task path sources: `{rows['task_path_sources']}`",
        f"- gt debug rows: `{rows['gt_debug_row_count']}`",
        f"- missing paths: `{ {key: value['count'] for key, value in rows['missing_by_key'].items()} }`",
        f"- load_arrays shapes: `{rows['load_arrays_shapes']}`",
        f"- prefix roles: `{rows['prefix_roles']}`",
        f"- current phases: `{rows['current_phases']}`",
        "",
        "## Feature Contract",
        "",
        f"- expected feature dim: `{contract.get('expected_feature_dim')}`",
        f"- actual feature dim: `{contract.get('actual_feature_dim')}`",
        f"- target dim: `{contract.get('target_dim')}`",
        f"- horizon: `{contract.get('horizon')}`",
        f"- ok: `{contract.get('ok')}`",
        "",
        "The feature layout is current real state + Cosmos predicted task path + frozen-DP prior chunk + causal live contact context.",
        "",
        "## Static Live Dependencies",
        "",
        f"- missing required: `{deps['missing_required']}`",
        f"- ok: `{deps['ok']}`",
        "",
        "## Planned Formal Run",
        "",
        f"- config: `{payload['planned_formal_config']}`",
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = resolve(Path.cwd(), args.root)
    output_root = resolve(root, args.output_root)
    output_json = resolve(root, args.output_json) if args.output_json else output_root / "prelaunch_audit.json"
    output_md = resolve(root, args.output_md) if args.output_md else output_root / "prelaunch_audit.md"
    jsonl_path = resolve(root, args.contact_executor_jsonl)

    training_input = audit_rows(root, jsonl_path, int(args.sample_npz_count))
    static_dependencies = audit_static_dependencies(args, root)
    config = planned_config(args)

    missing_counts_ok = all(
        int(item.get("count", 1)) == 0 for item in training_input["missing_by_key"].values()
    )
    training_input_ok = bool(
        training_input["rows"] > 0
        and training_input["gt_debug_row_count"] == 0
        and missing_counts_ok
        and training_input["load_arrays_error"] is None
        and (training_input.get("feature_contract") or {}).get("ok") is True
    )
    ready = bool(training_input_ok and static_dependencies["ok"] and config["ok"])
    payload = {
        "schema": "cosmos3_candidate_executor_diffusion_prelaunch_audit_v1",
        "date": os.popen("date -Is").read().strip(),
        "root": str(root),
        "training_input_ok": training_input_ok,
        "static_dependencies_ok": bool(static_dependencies["ok"]),
        "planned_formal_config_ok": bool(config["ok"]),
        "ready_for_allocation_launch": ready,
        "blocker_class": str(args.ready_blocker_class) if ready else "prelaunch_input_or_path_failure",
        "training_input": training_input,
        "static_dependencies": static_dependencies,
        "planned_formal_config": config,
        "boundary": (
            "Prelaunch audit for the direct diffusion candidate-executor chain. "
            "It proves only that the rows, paths, and feature contract are ready "
            "for the tmux-held allocation watcher. It is not training, live "
            "rollout, video, or method evidence."
        ),
    }
    write_json(output_json, payload)
    write_text(output_md, render_md(payload))
    print(json.dumps({"ready_for_allocation_launch": ready, "output_json": str(output_json), "output_md": str(output_md)}, sort_keys=True))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
