#!/usr/bin/env python3
"""Cancel dead gated-video branches after no-video success gates finish.

This is Slurm hygiene only. It does not change controller metrics, success
criteria, or video evidence requirements.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any


TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REQUEUED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}


@dataclass
class JobState:
    job_id: str
    state: str
    exit_code: str = ""
    reason: str = ""
    source: str = ""


@dataclass
class Branch:
    label: str
    gate_job: str
    video_job: str
    inspection_job: str


def _run(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _base_state(state: str) -> str:
    return state.split()[0].split("+")[0].strip()


def _query_job(job_id: str) -> JobState:
    sacct = _run(
        [
            "sacct",
            "-n",
            "-P",
            "-j",
            job_id,
            "--format=JobID,State,ExitCode,Reason%80",
        ]
    )
    if sacct.returncode == 0:
        for raw_line in sacct.stdout.splitlines():
            parts = raw_line.split("|")
            if len(parts) < 4:
                continue
            if parts[0] == job_id:
                return JobState(
                    job_id=job_id,
                    state=parts[1],
                    exit_code=parts[2],
                    reason=parts[3],
                    source="sacct",
                )

    squeue = _run(["squeue", "-h", "-j", job_id, "-o", "%i|%T|%R"])
    if squeue.returncode == 0:
        for raw_line in squeue.stdout.splitlines():
            parts = raw_line.split("|", maxsplit=2)
            if len(parts) < 3:
                continue
            if parts[0] == job_id:
                return JobState(job_id=job_id, state=parts[1], reason=parts[2], source="squeue")

    return JobState(job_id=job_id, state="UNKNOWN", source="missing")


def _load_branches(path: Path) -> list[Branch]:
    branches: list[Branch] = []
    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"{path}:{line_no} expected '<label> <gate_job> <video_job> <inspection_job>'")
        branches.append(Branch(*parts))
    if not branches:
        raise ValueError(f"{path} contains no branches")
    return branches


def _should_cancel_after_gate(gate: JobState) -> tuple[bool, str]:
    base = _base_state(gate.state)
    if base == "COMPLETED" and gate.exit_code in {"0:0", "0"}:
        return False, "gate_passed"
    if base in TERMINAL_STATES:
        return True, "gate_did_not_pass"
    return False, "gate_not_terminal"


def _maybe_cancel(job: JobState, *, dry_run: bool) -> dict[str, Any]:
    base = _base_state(job.state)
    record: dict[str, Any] = {
        "job_id": job.job_id,
        "state": job.state,
        "reason": job.reason,
        "source": job.source,
        "cancel_attempted": False,
        "cancel_returncode": None,
        "cancel_stdout": "",
        "cancel_stderr": "",
    }
    if base not in {"PENDING", "CONFIGURING", "SUSPENDED"}:
        record["cancel_reason"] = "not_pending"
        return record
    record["cancel_attempted"] = True
    record["cancel_reason"] = "dead_gated_video_branch"
    if dry_run:
        record["cancel_returncode"] = 0
        record["cancel_stdout"] = "dry_run"
        return record
    result = _run(["scancel", job.job_id])
    record["cancel_returncode"] = result.returncode
    record["cancel_stdout"] = result.stdout
    record["cancel_stderr"] = result.stderr
    return record


def _write_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Gated Video Branch Cleanup",
        "",
        "This is Slurm dependency hygiene. It does not change the controller metric, "
        "success gate, or direct video/contact-sheet review requirement.",
        "",
        f"- branches file: `{report['branches_file']}`",
        f"- dry run: `{report['dry_run']}`",
        f"- branches: `{len(report['branches'])}`",
        "",
        "| label | gate | gate state | action | video cancel | inspection cancel |",
        "|---|---:|---|---|---:|---:|",
    ]
    for branch in report["branches"]:
        video_cancel = branch["video"].get("cancel_attempted", False)
        inspection_cancel = branch["inspection"].get("cancel_attempted", False)
        lines.append(
            "| {label} | {gate_job} | {gate_state} | {action} | {video_cancel} | {inspection_cancel} |".format(
                label=branch["label"],
                gate_job=branch["gate"]["job_id"],
                gate_state=branch["gate"]["state"],
                action=branch["action"],
                video_cancel=video_cancel,
                inspection_cancel=inspection_cancel,
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches-file", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    branches_path = Path(args.branches_file)
    rows: list[dict[str, Any]] = []
    for branch in _load_branches(branches_path):
        gate = _query_job(branch.gate_job)
        should_cancel, action = _should_cancel_after_gate(gate)
        video = _query_job(branch.video_job)
        inspection = _query_job(branch.inspection_job)
        video_record = _maybe_cancel(video, dry_run=args.dry_run) if should_cancel else {
            "job_id": video.job_id,
            "state": video.state,
            "reason": video.reason,
            "source": video.source,
            "cancel_attempted": False,
            "cancel_reason": action,
        }
        inspection_record = _maybe_cancel(inspection, dry_run=args.dry_run) if should_cancel else {
            "job_id": inspection.job_id,
            "state": inspection.state,
            "reason": inspection.reason,
            "source": inspection.source,
            "cancel_attempted": False,
            "cancel_reason": action,
        }
        rows.append(
            {
                "label": branch.label,
                "action": action,
                "gate": vars(gate),
                "video": video_record,
                "inspection": inspection_record,
            }
        )

    report = {
        "branches_file": str(branches_path),
        "dry_run": bool(args.dry_run),
        "branches": rows,
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_md(report, output_md)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
