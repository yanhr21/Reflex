#!/usr/bin/env python3
"""Summarize the current Cosmos3 SFT and closed-loop evidence state.

This script is intentionally read-only with respect to experiments. It parses
existing logs, checkpoints, gate files, and live closed-loop summaries, then
writes a compact JSON/Markdown status artifact for monitoring.
"""

from __future__ import annotations

import argparse
import json
import re
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ITER_RE = re.compile(r"iter_(\d+)$")
TRAIN_ITER_RE = re.compile(r"Iteration\s+(\d+):")
VALIDATION_RE = re.compile(r"Validation loss \(iteration (\d+)\):\s*([0-9.eE+-]+)")

PROCESS_PATTERN = (
    r"run_cosmos3_live_receding_panel|run_cosmos3_live_receding_loop|"
    r"cosmos_framework.scripts.inference|torchrun.*cosmos|torchrun.*sft|"
    r"cosmos_framework.*train|train_cosmos"
)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - status tooling should report parse errors.
        return {"_parse_error": str(exc), "_path": str(path)}


def iso_mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).astimezone().isoformat()


def iter_number(path: Path) -> int | None:
    match = ITER_RE.search(path.name)
    return int(match.group(1)) if match else None


def run_command(args: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(args, check=False, text=True, capture_output=True)
    except FileNotFoundError as exc:
        return {"available": False, "error": str(exc), "stdout": "", "stderr": "", "returncode": None}
    return {
        "available": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def summarize_slurm(job_id: str | None) -> dict[str, Any]:
    if not job_id:
        return {"job_id": None, "checked": False}
    queue = run_command(["squeue", "-j", job_id])
    steps = run_command(["squeue", "--steps", "-j", job_id])
    extern_only = False
    if steps.get("available") and steps.get("returncode") == 0:
        step_lines = [line for line in steps.get("stdout", "").splitlines() if line.strip()]
        body = step_lines[1:] if step_lines else []
        extern_only = bool(body) and all(".extern" in line for line in body)
    return {
        "job_id": job_id,
        "checked": True,
        "queue": queue,
        "steps": steps,
        "extern_only": extern_only,
    }


def summarize_processes() -> dict[str, Any]:
    result = run_command(["ps", "-eo", "pid=,args="])
    pattern = re.compile(PROCESS_PATTERN)
    current_pid = os.getpid()
    matches = []
    for line in result.get("stdout", "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            pid = -1
        args = parts[1] if len(parts) > 1 else ""
        if pid == current_pid:
            continue
        if not pattern.search(args):
            continue
        # Avoid the classic pgrep self-match and our read-only status tools.
        if "pgrep -af" in args or "summarize_cosmos3_sft_closed_loop_status.py" in args:
            continue
        matches.append(stripped)
    return {
        "pattern": PROCESS_PATTERN,
        "returncode": result.get("returncode"),
        "source": "ps",
        "matches": matches,
        "active_matching_process_count": len(matches),
    }


def summarize_training(sft_root: Path) -> dict[str, Any]:
    checkpoint_paths = sorted(
        [p for p in (sft_root / "outputs" / "cosmos3" / "sft").glob("*/checkpoints/iter_*") if p.is_dir()],
        key=lambda p: (iter_number(p) or -1, str(p)),
    )
    checkpoints = [{"name": p.name, "path": str(p), "iteration": iter_number(p)} for p in checkpoint_paths]

    log_path = sft_root / "sft_train.log"
    log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    train_iterations = [int(m.group(1)) for m in TRAIN_ITER_RE.finditer(log_text)]
    validation_from_log = [
        {"iteration": int(m.group(1)), "val_loss": float(m.group(2))}
        for m in VALIDATION_RE.finditer(log_text)
    ]
    val_loss_summary = read_json(sft_root / "val_loss_summary.json")

    completed_path = sft_root / "sft_completed"
    failed_path = sft_root / "sft_failed"
    completed_text = completed_path.read_text(errors="replace").strip() if completed_path.is_file() else None
    completed_mtime = completed_path.stat().st_mtime if completed_path.exists() else None
    log_mtime = log_path.stat().st_mtime if log_path.exists() else None

    return {
        "sft_root": str(sft_root),
        "exists": sft_root.is_dir(),
        "checkpoints": checkpoints,
        "latest_checkpoint": checkpoints[-1] if checkpoints else None,
        "sft_completed": {
            "exists": completed_path.is_file(),
            "path": str(completed_path),
            "text": completed_text,
            "mtime": iso_mtime(completed_path),
            "stale_vs_log": bool(completed_mtime and log_mtime and completed_mtime < log_mtime),
        },
        "sft_failed": {
            "exists": failed_path.is_file(),
            "path": str(failed_path),
            "mtime": iso_mtime(failed_path),
        },
        "sft_train_log": {
            "exists": log_path.is_file(),
            "path": str(log_path),
            "mtime": iso_mtime(log_path),
            "latest_visible_iteration": max(train_iterations) if train_iterations else None,
            "num_iteration_lines": len(train_iterations),
        },
        "validation_from_log": validation_from_log,
        "val_loss_summary": val_loss_summary,
    }


def summarize_generated_eval(sft_root: Path) -> dict[str, Any]:
    eval_roots = []
    for path in sorted(sft_root.glob("eval_full_episode_wam_iter_*")):
        if not path.is_dir():
            continue
        match = re.search(r"iter_(\d+)", path.name)
        iteration = int(match.group(1)) if match else None
        visual_gate = read_json(path / "closed_loop_gate_visual_review.json")
        pre_gate = read_json(path / "closed_loop_gate_pre_visual.json")
        readout = read_json(path / "task_state_readout_v7_733" / "readout_eval_summary.json")
        gate = visual_gate or pre_gate
        eval_roots.append(
            {
                "name": path.name,
                "path": str(path),
                "iteration": iteration,
                "has_visual_gate": visual_gate is not None,
                "has_pre_gate": pre_gate is not None,
                "closed_loop_allowed": gate.get("closed_loop_allowed") if isinstance(gate, dict) else None,
                "visual_review_status": gate.get("visual_review_status") if isinstance(gate, dict) else None,
                "gate_reasons": gate.get("reasons") if isinstance(gate, dict) else None,
                "gate_metrics": gate.get("metrics") if isinstance(gate, dict) else None,
                "readout_aggregate": readout.get("aggregate") if isinstance(readout, dict) else None,
            }
        )
    eval_roots.sort(key=lambda row: (row["iteration"] if row["iteration"] is not None else -1, row["name"]))
    return {
        "eval_roots": eval_roots,
        "latest_eval": eval_roots[-1] if eval_roots else None,
    }


def summarize_live_runs(sft_root: Path, max_runs: int) -> dict[str, Any]:
    rows = []
    for summary_path in sft_root.glob("live_receding*/live_receding_panel_summary.json"):
        summary = read_json(summary_path)
        if not isinstance(summary, dict):
            continue
        samples = summary.get("samples") or []
        rows.append(
            {
                "kind": "panel",
                "name": summary_path.parent.name,
                "path": str(summary_path.parent),
                "summary_path": str(summary_path),
                "mtime": iso_mtime(summary_path),
                "completed_samples": summary.get("completed_samples"),
                "final_success_count": summary.get("final_success_count"),
                "requested_samples": summary.get("requested_samples"),
                "method_evidence_allowed": summary.get("method_evidence_allowed"),
                "sample_results": [
                    {
                        "sample_index": sample.get("sample_index"),
                        "scenario": sample.get("scenario"),
                        "completed_iterations": sample.get("completed_iterations"),
                        "final_success": sample.get("final_success"),
                        "final_eval": sample.get("final_eval"),
                        "final_prefix_frame_index": sample.get("final_prefix_frame_index"),
                        "dp_handoff_executed_steps": sample.get("dp_handoff_executed_steps"),
                    }
                    for sample in samples[:10]
                    if isinstance(sample, dict)
                ],
            }
        )
    for summary_path in sft_root.glob("live_receding*/**/live_receding_loop_summary.json"):
        summary = read_json(summary_path)
        if not isinstance(summary, dict):
            continue
        iterations = summary.get("iterations") or []
        final_eval = summary.get("final_eval") if isinstance(summary.get("final_eval"), dict) else {}
        rows.append(
            {
                "kind": "loop",
                "name": summary_path.parent.name,
                "path": str(summary_path.parent),
                "summary_path": str(summary_path),
                "mtime": iso_mtime(summary_path),
                "scenario": summary.get("scenario"),
                "completed_iterations": len(iterations),
                "final_success": final_eval.get("success"),
                "final_eval": final_eval,
                "final_prefix_frame_index": summary.get("final_prefix_frame_index"),
                "dp_handoff_executed_steps": summary.get("dp_handoff_executed_steps"),
            }
        )
    rows.sort(key=lambda row: row.get("mtime") or "", reverse=True)
    selected = rows[:max_runs]
    return {
        "num_live_summary_files": len(rows),
        "recent_runs": selected,
        "any_success_in_recent_runs": any(row.get("final_success") is True for row in selected),
    }


def write_outputs(report: dict[str, Any], output_json: Path | None, output_md: Path | None) -> None:
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        training = report["training"]
        latest_ckpt = training.get("latest_checkpoint") or {}
        latest_eval = (report.get("generated_eval") or {}).get("latest_eval") or {}
        lines = [
            "# Cosmos3 SFT Closed-Loop Status",
            "",
            f"- generated_at: `{report['generated_at']}`",
            f"- sft_root: `{report['sft_root']}`",
            f"- active_matching_process_count: `{report['processes']['active_matching_process_count']}`",
            f"- slurm_extern_only: `{report['slurm'].get('extern_only')}`",
            f"- latest_checkpoint: `{latest_ckpt.get('name')}`",
            f"- latest_visible_train_iteration: `{training['sft_train_log'].get('latest_visible_iteration')}`",
            f"- sft_completed_stale_vs_log: `{training['sft_completed'].get('stale_vs_log')}`",
            f"- latest_val_loss: `{(training.get('val_loss_summary') or {}).get('latest_val_loss')}`",
            f"- latest_eval_root: `{latest_eval.get('name')}`",
            f"- latest_eval_closed_loop_allowed: `{latest_eval.get('closed_loop_allowed')}`",
            f"- latest_eval_visual_review_status: `{latest_eval.get('visual_review_status')}`",
            f"- recent_live_any_success: `{report['live_closed_loop']['any_success_in_recent_runs']}`",
            "",
            "## Boundary",
            "",
            report["boundary"],
            "",
            "## Next Allowed Work",
            "",
            report["next_allowed_work"],
            "",
        ]
        output_md.write_text("\n".join(lines))


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    sft_root = Path(args.sft_root).resolve()
    training = summarize_training(sft_root)
    generated_eval = summarize_generated_eval(sft_root)
    live_closed_loop = summarize_live_runs(sft_root, args.max_live_runs)
    processes = summarize_processes()
    slurm = summarize_slurm(args.slurm_job_id)
    active_process_count = processes["active_matching_process_count"]
    latest_eval = generated_eval.get("latest_eval") or {}
    blocked_by_gate = latest_eval.get("closed_loop_allowed") is False
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "sft_root": str(sft_root),
        "slurm": slurm,
        "processes": processes,
        "training": training,
        "generated_eval": generated_eval,
        "live_closed_loop": live_closed_loop,
        "status_flags": {
            "training_or_eval_process_active": active_process_count > 0,
            "latest_generated_eval_controller_blocked": blocked_by_gate,
            "current_condition_root_needs_clean_dense_repair": True,
            "safe_to_resume_current_condition_sft_without_user_approval": False,
            "safe_to_launch_broad_panel_from_current_checkpoint": False,
        },
        "boundary": (
            "Corrected live closed-loop evidence is negative for the current "
            "checkpoint/condition root. Do not treat validation loss or generated "
            "readout metrics as controller success, and do not continue SFT from "
            "the current condition root as if more iterations alone are the fix."
        ),
        "next_allowed_work": (
            "After explicit user approval, run the clean-role/dense-receding "
            "condition preflight inside a compute allocation with RUN_SFT=false. "
            "Training requires a matching clean_dense_preflight_summary.json with "
            "ready_for_overfit=true and another explicit approval."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-root", required=True)
    parser.add_argument("--slurm-job-id", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--max-live-runs", type=int, default=20)
    args = parser.parse_args()
    report = build_report(args)
    write_outputs(report, args.output_json, args.output_md)
    print(
        json.dumps(
            {
                "active_matching_process_count": report["processes"]["active_matching_process_count"],
                "latest_checkpoint": (report["training"].get("latest_checkpoint") or {}).get("name"),
                "latest_eval_closed_loop_allowed": (
                    (report.get("generated_eval") or {}).get("latest_eval") or {}
                ).get("closed_loop_allowed"),
                "recent_live_any_success": report["live_closed_loop"]["any_success_in_recent_runs"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
