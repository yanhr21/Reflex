#!/usr/bin/env python3
"""Preflight entry point for Cosmos3 receding closed-loop control.

The live controller is intentionally guarded: this script must run inside a
Slurm compute step and refuses to touch the simulator unless the selected
Cosmos3 eval root passes the strict artifact/readout/visual gate.
"""

from __future__ import annotations

import argparse
import json
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
    condition_contract = check_condition_root(condition_root, args)
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
        "output_root": str(output_root),
        "max_episode_steps": args.max_episode_steps,
        "action_exec_horizon": args.action_exec_horizon,
        "expected_video_frames": args.expected_video_frames,
        "expected_action_steps": args.expected_action_steps,
        "expected_action_dim": args.expected_action_dim,
        "robot_action_dim": args.robot_action_dim,
        "dp_contract": dp_contract,
        "condition_contract": condition_contract,
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
    if not condition_contract["ok"]:
        failures.append("condition_contract_failed")
    if not gate.get("closed_loop_allowed"):
        failures.append("closed_loop_gate_blocked")

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
