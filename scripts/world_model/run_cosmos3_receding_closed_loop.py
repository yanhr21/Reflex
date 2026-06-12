#!/usr/bin/env python3
"""Preflight entry point for Cosmos3 receding closed-loop control.

The live controller is intentionally guarded: this script must run inside a
Slurm compute step and refuses to touch the simulator unless the selected
Cosmos3 eval root passes the strict artifact/readout/visual gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--condition-root", required=True)
    parser.add_argument("--dp-checkpoint", required=True)
    parser.add_argument("--dp-manifest", required=True)
    parser.add_argument(
        "--dp-state-key",
        choices=("ema_agent", "agent"),
        default="ema_agent",
        help="State-dict key to require for the frozen static DP policy.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--readout-subdir", default="task_state_readout_v7_733")
    parser.add_argument(
        "--visual-review-status",
        choices=("missing", "pass", "fail"),
        default="missing",
    )
    parser.add_argument("--visual-review-note", default="")
    parser.add_argument("--expected-video-frames", type=int, default=301)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--expected-action-dim", type=int, default=32)
    parser.add_argument("--robot-action-dim", type=int, default=7)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--action-exec-horizon", type=int, default=8)
    parser.add_argument("--action-preview-sample-index", type=int, default=0)
    parser.add_argument("--mode", choices=("preflight", "smoke"), default="preflight")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require_compute_step() -> None:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    step_id = os.environ.get("SLURM_STEP_ID", "")
    if not job_id or not step_id or step_id == "extern":
        raise SystemExit(
            "refusing_login_node_execution=true\n"
            "reason=Run this entry point only inside a compute-node srun step."
        )
    ntasks = int(os.environ.get("SLURM_NTASKS", "1") or "1")
    procid = int(os.environ.get("SLURM_PROCID", "0") or "0")
    if ntasks != 1 or procid != 0:
        raise SystemExit(
            "refusing_multi_task_execution=true\n"
            "reason=Closed-loop control must run as exactly one Slurm task."
        )


def assert_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")


def assert_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"missing {label}: {path}")


def check_dp_manifest(dp_manifest: Path, args: argparse.Namespace) -> dict[str, Any]:
    data = read_json(dp_manifest)
    dp_args = data.get("args") or {}
    expected = {
        "env_id": "PegInsertionSide-v1",
        "control_mode": "pd_ee_delta_pose",
        "obs_horizon": 2,
        "act_horizon": 8,
        "max_episode_steps": args.max_episode_steps,
    }
    mismatches = {
        key: {"expected": value, "actual": dp_args.get(key)}
        for key, value in expected.items()
        if dp_args.get(key) != value
    }
    if args.action_exec_horizon > int(dp_args.get("act_horizon", 0) or 0):
        mismatches["action_exec_horizon"] = {
            "expected": f"<= {dp_args.get('act_horizon')}",
            "actual": args.action_exec_horizon,
        }
    return {
        "path": str(dp_manifest),
        "expected": expected,
        "mismatches": mismatches,
        "ok": not mismatches,
    }


def check_dp_checkpoint(dp_checkpoint: Path, dp_manifest: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Validate the frozen DP artifact without constructing the simulator.

    This keeps the closed-loop entrypoint tied to the real ManiSkill state-DP
    checkpoint format used by scripts/training/eval_dp_state.py, while avoiding
    any env construction before the Cosmos visual/readout gate passes.
    """
    import torch

    manifest = read_json(dp_manifest)
    manifest_args = manifest.get("args") or {}
    try:
        ckpt = torch.load(dp_checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(dp_checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict):
        return {
            "path": str(dp_checkpoint),
            "ok": False,
            "error": f"checkpoint_load_returned_{type(ckpt).__name__}",
        }

    required_keys = ["args", "agent", args.dp_state_key, "iteration"]
    missing_keys = [key for key in required_keys if key not in ckpt]
    ckpt_args = ckpt.get("args") if isinstance(ckpt.get("args"), dict) else {}
    expected_args = {
        "env_id": "PegInsertionSide-v1",
        "control_mode": "pd_ee_delta_pose",
        "obs_horizon": 2,
        "act_horizon": 8,
        "max_episode_steps": args.max_episode_steps,
        "pred_horizon": 16,
    }
    arg_mismatches = {
        key: {"expected": value, "actual": ckpt_args.get(key)}
        for key, value in expected_args.items()
        if ckpt_args.get(key) != value
    }
    manifest_arg_mismatches = {
        key: {
            "manifest": manifest_args.get(key),
            "checkpoint": ckpt_args.get(key),
        }
        for key in expected_args
        if manifest_args.get(key) != ckpt_args.get(key)
    }
    state_value = ckpt.get(args.dp_state_key)
    state_nonempty = isinstance(state_value, dict) and bool(state_value)
    sample_tensors: list[dict[str, Any]] = []
    if isinstance(state_value, dict):
        for name, value in state_value.items():
            if hasattr(value, "shape"):
                sample_tensors.append(
                    {
                        "name": str(name),
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                )
            if len(sample_tensors) >= 3:
                break

    return {
        "path": str(dp_checkpoint),
        "state_key": args.dp_state_key,
        "required_keys": required_keys,
        "present_keys": sorted(str(key) for key in ckpt.keys()),
        "missing_keys": missing_keys,
        "iteration": ckpt.get("iteration"),
        "world_size": ckpt.get("world_size"),
        "global_batch_size": ckpt.get("global_batch_size"),
        "per_gpu_batch_size": ckpt.get("per_gpu_batch_size"),
        "expected_args": expected_args,
        "arg_mismatches": arg_mismatches,
        "manifest_arg_mismatches": manifest_arg_mismatches,
        "state_nonempty": state_nonempty,
        "sample_tensors": sample_tensors,
        "ok": (
            not missing_keys
            and not arg_mismatches
            and not manifest_arg_mismatches
            and state_nonempty
        ),
        "boundary": (
            "Checkpoint structure check only. It does not construct ManiSkill "
            "envs, run DP inference, or prove controller success."
        ),
    }


def check_condition_root(condition_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = condition_root / "manifest.json"
    stats_path = condition_root / "normalization_stats.json"
    assert_file(manifest_path, "condition manifest")
    assert_file(stats_path, "normalization stats")
    manifest = read_json(manifest_path)
    stats = read_json(stats_path)
    checks = {
        "num_video_frames": manifest.get("num_video_frames") == args.expected_video_frames,
        "num_action_steps": manifest.get("num_action_steps") == args.expected_action_steps,
        "raw_action_dim": manifest.get("raw_action_dim") == args.expected_action_dim,
        "normalization_type": stats.get("type") == "zscore",
        "normalization_raw_action_dim": stats.get("raw_action_dim") == args.expected_action_dim,
        "normalization_mean_len": len(stats.get("mean") or []) == args.expected_action_dim,
        "normalization_std_len": len(stats.get("std") or []) == args.expected_action_dim,
        "normalization_vector_names_len": len(stats.get("vector_names") or []) == args.expected_action_dim,
        "robot_action_dim": 0 < args.robot_action_dim <= args.expected_action_dim,
    }
    return {
        "path": str(condition_root),
        "manifest_path": str(manifest_path),
        "normalization_stats_path": str(stats_path),
        "checks": checks,
        "ok": all(checks.values()),
    }


def run_gate(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    gate_json = output_root / "closed_loop_gate_verdict.json"
    script = Path(__file__).with_name("check_cosmos3_closed_loop_gate.py")
    cmd = [
        sys.executable,
        str(script),
        "--eval-root",
        str(Path(args.eval_root).resolve()),
        "--readout-subdir",
        args.readout_subdir,
        "--visual-review-status",
        args.visual_review_status,
        "--visual-review-note",
        args.visual_review_note,
        "--output-json",
        str(gate_json),
        "--allow-nonpassing-exit-zero",
    ]
    subprocess.run(cmd, check=True)
    return read_json(gate_json)


def resolve_eval_source_context(eval_root: Path, sample_name: str) -> dict[str, Any]:
    """Recover source trajectory metadata for a generated eval sample.

    The strict eval artifact focuses on generated/reference outputs. The live
    smoke path needs the causal source trajectory too, so recover it through the
    eval input manifest and the original condition JSONL row.
    """
    context: dict[str, Any] = {
        "sample_name": sample_name,
        "eval_input_manifest": str(eval_root / "eval_input_manifest.json"),
        "manifest_sample_found": False,
        "condition_row_found": False,
        "source_h5": None,
    }
    manifest_path = eval_root / "eval_input_manifest.json"
    if not manifest_path.is_file():
        context["missing"] = "eval_input_manifest"
        return context

    manifest = read_json(manifest_path)
    samples = manifest.get("samples") or []
    manifest_sample = next((s for s in samples if s.get("name") == sample_name), None)
    if not isinstance(manifest_sample, dict):
        context["missing"] = "sample_in_eval_input_manifest"
        return context

    context["manifest_sample_found"] = True
    for key in [
        "source_row_uuid",
        "source_uuid",
        "source_jsonl",
        "scenario",
        "prefix_role",
        "prefix_frame_index",
        "condition_prefix_frames",
        "reference_video_path",
        "reference_action_path",
        "state_target_path",
        "task_label_path",
    ]:
        if key in manifest_sample:
            context[key] = manifest_sample[key]

    source_jsonl = manifest_sample.get("source_jsonl")
    source_row_uuid = manifest_sample.get("source_row_uuid")
    if not source_jsonl or not source_row_uuid:
        context["missing"] = "source_jsonl_or_source_row_uuid"
        return context

    source_path = Path(str(source_jsonl))
    if not source_path.is_file():
        context["missing"] = "source_jsonl_file"
        return context

    with source_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("uuid") != source_row_uuid:
                continue
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            context["condition_row_found"] = True
            context["source_h5"] = row.get("source_h5") or metadata.get("source_h5")
            context["condition_row"] = {
                "uuid": row.get("uuid"),
                "source_uuid": row.get("source_uuid"),
                "split": row.get("split"),
                "scenario": row.get("scenario") or metadata.get("scenario"),
                "prefix_role": row.get("prefix_role"),
                "prefix_frame_index": row.get("prefix_frame_index"),
                "condition_prefix_frames": row.get("condition_prefix_frames"),
                "rgb_video": row.get("rgb_video"),
                "action_path": row.get("action_path"),
                "state_target_path": row.get("state_target_path"),
                "task_label_path": row.get("task_label_path"),
                "task_summary_path": row.get("task_summary_path"),
                "num_rgb_frames": row.get("num_rgb_frames"),
                "num_action_steps": row.get("num_action_steps"),
                "sidecar_target_mode": row.get("sidecar_target_mode"),
            }
            break

    return context


def _to_float_matrix(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise ValueError("predicted action is not a non-empty list")
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list):
            raise ValueError("predicted action row is not a list")
        matrix.append([float(x) for x in row])
    return matrix


def _column_stats(matrix: list[list[float]]) -> dict[str, Any]:
    flat = [x for row in matrix for x in row]
    finite = all(math.isfinite(x) for x in flat)
    if not flat:
        return {"finite": finite, "num_values": 0}
    return {
        "finite": finite,
        "num_values": len(flat),
        "min": min(flat),
        "max": max(flat),
        "mean_abs": sum(abs(x) for x in flat) / float(len(flat)),
        "max_abs": max(abs(x) for x in flat),
    }


def build_action_chunk_preview(eval_root: Path, condition_root: Path, args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    inspection = read_json(eval_root / "eval_artifact_inspection.json")
    samples = inspection.get("samples") or []
    if not samples:
        raise ValueError("eval artifact inspection has no samples")
    sample_index = max(0, min(args.action_preview_sample_index, len(samples) - 1))
    sample = samples[sample_index]
    sample_name = sample["name"]
    source_context = resolve_eval_source_context(eval_root, sample_name)
    sample_outputs = read_json(eval_root / "inference" / sample_name / "sample_outputs.json")
    outputs = sample_outputs.get("outputs") or []
    content = (outputs[0].get("content") if outputs else None) or {}
    action = _to_float_matrix(content.get("action"))
    stats = read_json(condition_root / "normalization_stats.json")
    mean = [float(x) for x in stats["mean"]]
    std = [float(x) for x in stats["std"]]
    vector_names = list(stats.get("vector_names") or [])

    shape_ok = (
        len(action) == args.expected_action_steps
        and all(len(row) == args.expected_action_dim for row in action)
    )
    future_start = int(
        sample.get("action_metrics", {}).get(
            "future_start_action_index",
            sample.get("action_metrics", {}).get("prefix_end_action_index_exclusive", sample.get("prefix_frame_index", 0)),
        )
    )
    future_start = max(0, min(future_start, args.expected_action_steps))
    chunk_end = min(args.expected_action_steps, future_start + args.action_exec_horizon)

    normalized_chunk = [
        row[: args.robot_action_dim]
        for row in action[future_start:chunk_end]
    ]
    denormalized_chunk = []
    for row in normalized_chunk:
        denormalized_chunk.append(
            [
                row[i] * std[i] + mean[i]
                for i in range(args.robot_action_dim)
            ]
        )

    preview = {
        "sample_index": sample_index,
        "sample_name": sample_name,
        "source_context": source_context,
        "prefix_role": sample.get("prefix_role"),
        "scenario": sample.get("scenario"),
        "prefix_frame_index": sample.get("prefix_frame_index"),
        "future_start_action_index": future_start,
        "action_exec_horizon": args.action_exec_horizon,
        "chunk_start": future_start,
        "chunk_end_exclusive": chunk_end,
        "chunk_steps": chunk_end - future_start,
        "expected_action_shape": [args.expected_action_steps, args.expected_action_dim],
        "predicted_action_shape": [len(action), len(action[0]) if action else 0],
        "shape_ok": shape_ok,
        "robot_action_dim": args.robot_action_dim,
        "robot_action_vector_names": vector_names[: args.robot_action_dim],
        "normalized_robot_action_stats": _column_stats(normalized_chunk),
        "denormalized_robot_action_stats": _column_stats(denormalized_chunk),
        "denormalized_robot_action_chunk": denormalized_chunk,
        "boundary": (
            "Action chunk preview only. These actions are not executed and are "
            "not controller evidence; live execution remains gated by visual "
            "review and simulator rollout."
        ),
    }
    preview["ok"] = bool(
        shape_ok
        and preview["chunk_steps"] > 0
        and preview["normalized_robot_action_stats"].get("finite")
        and preview["denormalized_robot_action_stats"].get("finite")
    )
    write_manifest(output_root / "candidate_action_chunk_preview.json", preview)
    return preview


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    require_compute_step()

    eval_root = Path(args.eval_root).resolve()
    checkpoint_path = Path(args.checkpoint_path).resolve()
    condition_root = Path(args.condition_root).resolve()
    dp_checkpoint = Path(args.dp_checkpoint).resolve()
    dp_manifest = Path(args.dp_manifest).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    assert_dir(eval_root, "eval root")
    assert_dir(checkpoint_path, "Cosmos checkpoint")
    assert_dir(condition_root, "condition root")
    assert_file(dp_checkpoint, "DP checkpoint")
    assert_file(dp_manifest, "DP manifest")

    dp_contract = check_dp_manifest(dp_manifest, args)
    dp_checkpoint_contract = check_dp_checkpoint(dp_checkpoint, dp_manifest, args)
    condition_contract = check_condition_root(condition_root, args)
    action_chunk_preview = build_action_chunk_preview(eval_root, condition_root, args, output_root)
    gate = run_gate(args, output_root)

    manifest: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "step_id": os.environ.get("SLURM_STEP_ID"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        },
        "mode": args.mode,
        "eval_root": str(eval_root),
        "checkpoint_path": str(checkpoint_path),
        "condition_root": str(condition_root),
        "dp_checkpoint": str(dp_checkpoint),
        "dp_manifest": str(dp_manifest),
        "dp_state_key": args.dp_state_key,
        "output_root": str(output_root),
        "max_episode_steps": args.max_episode_steps,
        "action_exec_horizon": args.action_exec_horizon,
        "action_preview_sample_index": args.action_preview_sample_index,
        "expected_video_frames": args.expected_video_frames,
        "expected_action_steps": args.expected_action_steps,
        "expected_action_dim": args.expected_action_dim,
        "robot_action_dim": args.robot_action_dim,
        "dp_contract": dp_contract,
        "dp_checkpoint_contract": dp_checkpoint_contract,
        "condition_contract": condition_contract,
        "action_chunk_preview": action_chunk_preview,
        "gate": gate,
        "boundary": (
            "Preflight only unless the gate passes. Live simulator success must "
            "come from real env.step rollouts plus video review; generated RGB "
            "or readout-only success is not controller evidence."
        ),
    }
    write_manifest(output_root / "closed_loop_preflight_manifest.json", manifest)

    failures: list[str] = []
    if not dp_contract["ok"]:
        failures.append("dp_contract_failed")
    if not dp_checkpoint_contract["ok"]:
        failures.append("dp_checkpoint_contract_failed")
    if not condition_contract["ok"]:
        failures.append("condition_contract_failed")
    if not action_chunk_preview["ok"]:
        failures.append("action_chunk_preview_failed")
    if not gate.get("closed_loop_allowed"):
        failures.append("closed_loop_gate_blocked")
    if args.mode == "smoke" and not action_chunk_preview.get("source_context", {}).get("source_h5"):
        failures.append("live_source_h5_missing")

    if failures:
        print(json.dumps({"closed_loop_preflight_ok": False, "failures": failures}, indent=2))
        return 40

    if args.mode == "preflight":
        print(json.dumps({"closed_loop_preflight_ok": True, "mode": "preflight"}, indent=2))
        return 0

    print(
        json.dumps(
            {
                "closed_loop_preflight_ok": True,
                "live_smoke_started": False,
                "reason": "live_receding_smoke_not_implemented_in_this_guarded_entrypoint_yet",
            },
            indent=2,
        )
    )
    return 51


if __name__ == "__main__":
    raise SystemExit(main())
