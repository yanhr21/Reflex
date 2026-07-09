#!/usr/bin/env python3
"""Inspect real A-D dataset, DP, and Cosmos interfaces for joint training.

This is a diagnostic interface check. It does not train a model, generate data,
or claim method evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any


PRODUCTION_INDEXES = {
    "b_dynamic_production": {
        "dataset_class": "B_dynamic_rgb_observation",
        "target_train": 900,
        "target_val": 100,
    },
    "c_frozen_dp_production": {
        "dataset_class": "C_frozen_dp_dynamic_failure",
        "target_train": 450,
        "target_val": 50,
    },
    "d_future_teacher_production": {
        "dataset_class": "D_future_frame_cooperation_teacher",
        "target_train": 450,
        "target_val": 50,
    },
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _read_jsonl_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def _path_status(path: Path) -> dict[str, Any]:
    exists = path.exists()
    out: dict[str, Any] = {"path": path, "exists": exists}
    if exists:
        out["is_file"] = path.is_file()
        out["is_dir"] = path.is_dir()
        if path.is_file():
            out["size_bytes"] = path.stat().st_size
    return out


def _load_trace_summary(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": path, "exists": path.is_file()}
    if not path.is_file():
        out["ok"] = False
        out["reason"] = "missing_trace"
        return out
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    out["top_level_keys"] = sorted(payload.keys())
    for key in ("action_rows", "motion_rows", "task_rows", "quality_by_episode", "quality_by_rollout"):
        value = payload.get(key)
        if isinstance(value, list):
            out[f"{key}_count"] = len(value)
            if value:
                first = value[0]
                out[f"{key}_first_keys"] = sorted(first.keys()) if isinstance(first, dict) else []
    action_rows = payload.get("action_rows")
    if isinstance(action_rows, list) and action_rows:
        first_action = action_rows[0].get("action") if isinstance(action_rows[0], dict) else None
        out["first_action_dim"] = len(first_action) if isinstance(first_action, list) else None
        out["first_action_preview"] = first_action[:7] if isinstance(first_action, list) else None
        out["has_success_field"] = isinstance(action_rows[0], dict) and "success" in action_rows[0]
        out["has_positive_policy_data_allowed"] = (
            isinstance(action_rows[0], dict) and "positive_policy_data_allowed" in action_rows[0]
        )
    motion_rows = payload.get("motion_rows")
    if isinstance(motion_rows, list) and motion_rows:
        first_motion = motion_rows[0] if isinstance(motion_rows[0], dict) else {}
        out["first_motion_step"] = first_motion.get("step")
        out["first_motion_keys"] = sorted(first_motion.keys()) if isinstance(first_motion, dict) else []
    out["ok"] = bool(out.get("action_rows_count", 0))
    return out


def _inspect_sample(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "sample_id": row.get("sample_id"),
        "split": row.get("split"),
        "dataset_class": row.get("dataset_class"),
        "allowed_losses": row.get("allowed_losses"),
        "positive_dp_bc_allowed": row.get("positive_dp_bc_allowed"),
        "teacher_evidence_allowed": row.get("teacher_evidence_allowed"),
        "method_evidence_allowed": row.get("method_evidence_allowed"),
    }
    for field in ("video", "trace", "summary", "manifest"):
        value = row.get(field)
        if value:
            out[field] = _path_status(Path(value))
    if row.get("trace"):
        out["trace_schema"] = _load_trace_summary(Path(row["trace"]))
    return out


def inspect_project(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    registry = Path(args.registry).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    result: dict[str, Any] = {
        "mode": "project",
        "status": "started",
        "diagnostic_only": True,
        "method_evidence_allowed": False,
        "runtime": {
            "python": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
        },
        "root": root,
        "registry": registry,
    }

    a_static = registry / "a_static"
    result["a_static"] = {
        "h5": _path_status(a_static / "official_state_pd_ee_delta_pose.h5"),
        "json": _path_status(a_static / "official_state_pd_ee_delta_pose.json"),
    }

    indexes: dict[str, Any] = {}
    failures: list[str] = []
    for name, meta in PRODUCTION_INDEXES.items():
        index_dir = registry / name
        train_jsonl = index_dir / "train_samples.jsonl"
        val_jsonl = index_dir / "val_samples.jsonl"
        train_count = _line_count(train_jsonl)
        val_count = _line_count(val_jsonl)
        stage_out: dict[str, Any] = {
            "index_dir": _path_status(index_dir),
            "manifest": _path_status(index_dir / "index_manifest.txt"),
            "train_jsonl": _path_status(train_jsonl),
            "val_jsonl": _path_status(val_jsonl),
            "train_count": train_count,
            "val_count": val_count,
            "target_train": meta["target_train"],
            "target_val": meta["target_val"],
        }
        if train_count < int(meta["target_train"]):
            failures.append(f"{name}: train_count {train_count} < {meta['target_train']}")
        if val_count < int(meta["target_val"]):
            failures.append(f"{name}: val_count {val_count} < {meta['target_val']}")
        sample_rows = _read_jsonl_rows(train_jsonl, args.samples_per_index) if train_jsonl.is_file() else []
        stage_out["sample_count_inspected"] = len(sample_rows)
        stage_out["samples"] = [_inspect_sample(row) for row in sample_rows]
        for row in sample_rows:
            if row.get("dataset_class") != meta["dataset_class"]:
                failures.append(f"{name}: bad dataset_class {row.get('dataset_class')}")
            if row.get("positive_dp_bc_allowed") != "false":
                failures.append(f"{name}: positive_dp_bc_allowed is not false")
        indexes[name] = stage_out
    result["production_indexes"] = indexes

    ckpt_path = Path(args.dp_checkpoint).resolve()
    dp_out: dict[str, Any] = {"path": ckpt_path, "exists": ckpt_path.is_file()}
    if ckpt_path.is_file():
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        dp_out["top_level_keys"] = sorted(checkpoint.keys())
        saved_args = checkpoint.get("args", {})
        dp_out["args_keys"] = sorted(saved_args.keys()) if isinstance(saved_args, dict) else []
        if isinstance(saved_args, dict):
            for key in (
                "env_id",
                "control_mode",
                "obs_mode",
                "obs_horizon",
                "pred_horizon",
                "action_horizon",
                "max_episode_steps",
            ):
                if key in saved_args:
                    dp_out[key] = saved_args[key]
        for state_key in ("agent", "ema_agent"):
            state = checkpoint.get(state_key)
            if isinstance(state, dict):
                dp_out[f"{state_key}_tensor_count"] = len(state)
                first_keys = list(state.keys())[:10]
                dp_out[f"{state_key}_first_keys"] = first_keys
                dp_out[f"{state_key}_total_numel"] = int(
                    sum(value.numel() for value in state.values() if hasattr(value, "numel"))
                )
    else:
        failures.append(f"missing DP checkpoint: {ckpt_path}")
    result["dp_checkpoint"] = dp_out

    result["status"] = "ok" if not failures else "failed"
    result["failure_count"] = len(failures)
    result["failures"] = failures
    out_path = output_dir / "project_interface_summary.json"
    out_path.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return result


def inspect_cosmos(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cosmos_root = Path(args.cosmos_root).resolve()
    latest_file = cosmos_root / "checkpoints" / "latest_checkpoint.txt"
    latest_name = latest_file.read_text(encoding="utf-8").strip() if latest_file.is_file() else ""
    latest_dir = cosmos_root / "checkpoints" / latest_name if latest_name else cosmos_root / "checkpoints"

    import torch

    checks: dict[str, Any] = {
        "mode": "cosmos",
        "status": "started",
        "diagnostic_only": True,
        "method_evidence_allowed": False,
        "runtime": {
            "python": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
        },
        "root": root,
        "cosmos_root": cosmos_root,
        "config_yaml": _path_status(cosmos_root / "config.yaml"),
        "config_pkl": _path_status(cosmos_root / "config.pkl"),
        "normalization_stats": _path_status(cosmos_root / "normalization_stats.json"),
        "latest_checkpoint_txt": _path_status(latest_file),
        "latest_checkpoint_name": latest_name,
        "latest_checkpoint_dir": _path_status(latest_dir),
        "latest_model_metadata": _path_status(latest_dir / "model" / ".metadata"),
        "local_tokenizer_dir": _path_status(Path(args.cosmos_local_tokenizer_dir).resolve()),
        "wan_vae": _path_status(Path(args.wan_vae_path).resolve()),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "cosmos_framework_spec_found": importlib.util.find_spec("cosmos_framework") is not None,
    }
    failures: list[str] = []
    for label in (
        "config_yaml",
        "normalization_stats",
        "latest_checkpoint_txt",
        "latest_checkpoint_dir",
        "latest_model_metadata",
        "local_tokenizer_dir",
        "wan_vae",
    ):
        if not checks[label]["exists"]:
            failures.append(f"missing {label}: {checks[label]['path']}")
    if not checks["cosmos_framework_spec_found"]:
        failures.append("cosmos_framework import spec not found")

    checks["status"] = "ok" if not failures else "failed"
    checks["failure_count"] = len(failures)
    checks["failures"] = failures
    out_path = output_dir / "cosmos_interface_summary.json"
    out_path.write_text(json.dumps(_jsonable(checks), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_jsonable(checks), indent=2, sort_keys=True))
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("project", "cosmos"), required=True)
    parser.add_argument("--root", default="/public/home/yanhongru/ICLR2027/Reflex")
    parser.add_argument("--registry", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dp-checkpoint",
        default="/public/home/yanhongru/ICLR2027/Reflex/experiments/maniskill/dp_checkpoint/run_90201/checkpoints/best_eval_success_at_end.pt",
    )
    parser.add_argument(
        "--cosmos-root",
        default="/public/home/yanhongru/ICLR2027/Reflex/experiments/maniskill/cosmos3_checkpoint/vision_sft_droid_policy_full1000_rgb_300step_wam",
    )
    parser.add_argument(
        "--cosmos-local-tokenizer-dir",
        default="/public/home/yanhongru/ICLR2027/Reflex/checkpoints/cosmos3/Cosmos3-Nano",
    )
    parser.add_argument(
        "--wan-vae-path",
        default="/public/home/yanhongru/ICLR2027/Reflex/checkpoints/cosmos3/wan22_vae/Wan2.2_VAE.pth",
    )
    parser.add_argument("--samples-per-index", type=int, default=2)
    args = parser.parse_args()
    if args.registry is None:
        args.registry = str(Path(args.root) / "experiments/maniskill/data/active")
    return args


def main() -> int:
    args = parse_args()
    if args.mode == "project":
        result = inspect_project(args)
    else:
        result = inspect_cosmos(args)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
