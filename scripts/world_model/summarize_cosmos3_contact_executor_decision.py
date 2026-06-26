#!/usr/bin/env python3
"""Write a plain final/waiting decision summary for contact-executor training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument(
        "--launcher",
        default="scripts/slurm/run_cosmos3_contact_executor_live_panel_after_gate_in_allocation.sh",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def fmt_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "missing"
    try:
        return f"{float(value):.{digits}g}"
    except Exception:
        return str(value)


def latest_history_metrics(root: Path) -> dict[str, Any]:
    summary = read_json(root / "training_history_summary.json")
    if isinstance(summary, dict) and isinstance(summary.get("latest"), dict):
        latest = dict(summary["latest"])
        latest["num_points"] = summary.get("num_points")
        latest["num_points_action_worse_than_prior"] = summary.get("num_points_action_worse_than_prior")
        latest["all_points_action_worse_than_prior"] = summary.get("all_points_action_worse_than_prior")
        latest["latest_ratio"] = summary.get("latest_ratio")
        latest["best_ratio"] = summary.get("best_ratio")
        return latest

    history = read_json(root / "training_history.json")
    if isinstance(history, dict):
        history = history.get("history") or history.get("records")
    if not isinstance(history, list) or not history:
        return {}
    latest = dict(history[-1])
    prior = latest.get("eval_baseline_dp_prior_mse")
    action = latest.get("eval_action_mse")
    if prior:
        latest["latest_ratio"] = float(action) / float(prior)
    latest["num_points"] = len(history)
    return latest


def checkpoint_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "load_ok": False, "error": "missing"}
    status: dict[str, Any] = {"exists": True, "size_bytes": path.stat().st_size}
    try:
        import torch

        payload = torch.load(path, map_location="cpu")
        status.update(
            {
                "load_ok": True,
                "step": payload.get("step") if isinstance(payload, dict) else None,
            }
        )
    except Exception as exc:
        status.update({"load_ok": False, "error": repr(exc)})
    return status


def console_failure(root: Path) -> dict[str, Any]:
    path = root / "formal_train_console.log"
    if not path.is_file():
        return {"exists": False, "failed": False, "matches": []}
    needles = (
        "ChildFailedError",
        "ProcessGroupNCCL",
        "Watchdog caught collective operation timeout",
        "SIGABRT",
        "Exited with exit code 1",
    )
    matches: list[str] = []
    for line in path.read_text(errors="replace").splitlines()[-240:]:
        if any(needle in line for needle in needles):
            matches.append(line.strip())
    return {"exists": True, "failed": bool(matches), "matches": matches[-12:]}


def choose_status(
    *,
    root: Path,
    summary: dict[str, Any] | None,
    gate: dict[str, Any] | None,
    final_checkpoint: dict[str, Any],
    console: dict[str, Any],
) -> tuple[str, str, list[str]]:
    if summary is None and final_checkpoint.get("exists") and not final_checkpoint.get("load_ok"):
        blockers = ["training_summary.json", "invalid_checkpoint_final"]
        if console.get("failed"):
            blockers.append("training_process_failed")
        return (
            "failed_invalid_final_checkpoint_stop_for_user",
            "Stop and report the failed formal run. Do not run live eval or substitute checkpoint_latest.pt.",
            blockers,
        )
    if summary is None and console.get("failed"):
        return (
            "failed_training_process_stop_for_user",
            "Stop and report the failed formal run. Do not run live eval from checkpoint_latest.pt.",
            ["training_summary.json", "training_process_failed"],
        )
    missing = []
    for filename in ("training_summary.json", "checkpoint_final.pt"):
        if not (root / filename).exists():
            missing.append(filename)
    if missing:
        return (
            "waiting_for_formal_floor_or_final_files",
            "Keep the held allocation and wait. Do not run live eval from checkpoint_latest.pt.",
            missing,
        )
    if gate is None:
        return (
            "waiting_for_formal_gate",
            "Run or wait for the formal gate checker before any live eval.",
            ["formal_live_eval_gate.json"],
        )
    if gate.get("live_eval_allowed") is True:
        return (
            "gate_open_live_eval_allowed",
            "Launch the guarded contact-executor live panel, save videos/contact sheets, and inspect them before any claim.",
            [],
        )
    reasons = [str(item) for item in gate.get("failure_reasons", [])]
    if not reasons:
        reasons = ["formal_gate_false_without_reason"]
    return (
        "gate_closed_stop_for_user",
        "Stop and report the final offline blocker to the user. Do not run closed-loop videos from this checkpoint.",
        reasons,
    )


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    latest = payload.get("latest_history") or {}
    gate = payload.get("gate") or {}
    lines = [
        "# Contact Executor Formal Decision Summary",
        "",
        f"- status: `{payload['status']}`",
        f"- next action: {payload['next_action']}",
        f"- training root: `{payload['training_root']}`",
        f"- gate live eval allowed: `{gate.get('live_eval_allowed', 'missing')}`",
        f"- blockers: `{', '.join(payload.get('blockers') or []) or 'none'}`",
        "",
        "## Latest Training Trend",
        "",
        f"- step: `{latest.get('step', 'missing')}`",
        f"- elapsed seconds: `{fmt_float(latest.get('elapsed_seconds'))}`",
        f"- eval action MSE: `{fmt_float(latest.get('eval_action_mse'))}`",
        f"- DP-prior eval MSE: `{fmt_float(latest.get('eval_baseline_dp_prior_mse'))}`",
        f"- action MSE ratio to DP prior: `{fmt_float(latest.get('latest_ratio'))}`",
        f"- eval progress MSE: `{fmt_float(latest.get('eval_progress_mse'))}`",
        f"- inserted accuracy: `{fmt_float(latest.get('eval_inserted_acc'))}`",
        f"- DP-continuable accuracy: `{fmt_float(latest.get('eval_dp_continuable_acc'))}`",
        f"- eval points worse than DP prior: `{latest.get('num_points_action_worse_than_prior', 'missing')}/{latest.get('num_points', 'missing')}`",
        "",
        "## Boundary",
        "",
        payload["boundary"],
    ]
    if payload.get("guarded_launcher_command"):
        lines += [
            "",
            "## Guarded Launcher",
            "",
            "Run only inside a compute-node Slurm step after confirming the gate is open:",
            "",
            "```bash",
            payload["guarded_launcher_command"],
            "```",
        ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    root = Path(args.training_root).resolve()
    summary = read_json(root / "training_summary.json")
    gate = read_json(root / "formal_live_eval_gate.json")
    groups = read_json(root / "post_training_group_metrics.json")
    latest = latest_history_metrics(root)
    final_checkpoint = checkpoint_status(root / "checkpoint_final.pt")
    console = console_failure(root)
    status, next_action, blockers = choose_status(
        root=root,
        summary=summary,
        gate=gate,
        final_checkpoint=final_checkpoint,
        console=console,
    )

    launcher_cmd = ""
    if status == "gate_open_live_eval_allowed":
        launcher_cmd = f"bash {args.launcher}"

    payload: dict[str, Any] = {
        "schema": "cosmos3_contact_executor_formal_decision_summary_v1",
        "training_root": str(root),
        "status": status,
        "next_action": next_action,
        "blockers": blockers,
        "summary_exists": summary is not None,
        "checkpoint_final_exists": (root / "checkpoint_final.pt").is_file(),
        "checkpoint_final_status": final_checkpoint,
        "console_failure": console,
        "gate_exists": gate is not None,
        "gate": gate or {},
        "group_metrics_exists": groups is not None,
        "latest_history": latest,
        "guarded_launcher_command": launcher_cmd,
        "boundary": (
            "This is a read-only decision summary. It does not relax the formal gate, "
            "does not launch live eval, and does not prove method success. Major "
            "success still requires final-state metrics plus inspected video/contact sheets."
        ),
    }
    output_json = Path(args.output_json).resolve() if args.output_json else root / "formal_decision_summary.json"
    output_md = Path(args.output_md).resolve() if args.output_md else root / "formal_decision_summary.md"
    write_json(output_json, payload)
    write_markdown(output_md, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status in {"waiting_for_formal_floor_or_final_files", "waiting_for_formal_gate", "gate_open_live_eval_allowed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
