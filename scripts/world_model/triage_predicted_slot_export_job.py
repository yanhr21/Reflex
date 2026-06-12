#!/usr/bin/env python3
"""Triage RGB-D predicted-slot export without changing strict export gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
from typing import Any


PATTERNS: dict[str, tuple[str, ...]] = {
    "import_or_environment_error": (
        "ModuleNotFoundError",
        "ImportError",
        "No module named",
        "undefined symbol",
        "numpy._typing",
    ),
    "hdf5_lock_error": (
        "No locks available",
        "HDF5_USE_FILE_LOCKING",
        "unable to lock file",
    ),
    "memory_or_cuda_oom": (
        "CUDA out of memory",
        "Out of memory",
        "oom-kill",
        "Cannot allocate memory",
    ),
    "rgbd_input_or_file_count_error": (
        "Refusing RGB-D slot export",
        "MIN_RGBD_FILES",
        "EXPECTED_RGBD_FILES",
        "Set RGBD_ROOT",
        "No RGB-D slot samples",
        ".rgbd.h5",
    ),
    "slot_ensemble_or_checkpoint_error": (
        "SLOT_ENSEMBLE_DIR",
        "inspection.json",
        "best_model.pt",
        "model.pt",
        "Missing",
        "No such file or directory",
        "not compliant",
    ),
    "allocation_or_srun_error": (
        "srun: error",
        "slurmstepd:",
        "Unable to create step",
        "Job credential expired",
    ),
    "export_runtime_error": (
        "Traceback",
        "RuntimeError",
        "ValueError",
        "AssertionError",
    ),
}


@dataclass
class Args:
    job_id: str
    export_dir: str
    stdout: str | None = None
    stderr: str | None = None
    output_json: str | None = None
    output_md: str | None = None
    tail_lines: int = 120


def _run(cmd: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=20)
    except Exception as exc:  # pragma: no cover
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def _tail(path: Path, lines: int) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "tail": ""}
    text = path.read_text(errors="replace")
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "tail": "\n".join(text.splitlines()[-lines:]),
    }


def _parse_scontrol(stdout: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key in ("JobState", "Reason", "RunTime", "TimeLimit", "StartTime", "NodeList"):
        match = re.search(rf"\b{key}=([^ \n]+)", stdout)
        if match:
            fields[key] = match.group(1)
    return fields


def _pattern_hits(text: str) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    lowered = text.lower()
    for category, patterns in PATTERNS.items():
        found = [pattern for pattern in patterns if pattern.lower() in lowered]
        if found:
            hits[category] = found
    return hits


def _artifact_report(export_dir: Path) -> dict[str, Any]:
    if not export_dir.exists():
        return {
            "export_dir_exists": False,
            "slurm_manifest_exists": False,
            "export_report_exists": False,
            "predicted_slot_paths_file_exists": False,
            "num_h5_files": 0,
            "h5_files": [],
        }
    h5_files = sorted(str(path) for path in export_dir.rglob("*.h5"))
    return {
        "export_dir_exists": True,
        "slurm_manifest_exists": (export_dir / "slurm_manifest.txt").exists(),
        "export_report_exists": (export_dir / "export_report.json").exists(),
        "predicted_slot_paths_file_exists": (export_dir / "predicted_slot_h5s.txt").exists(),
        "num_h5_files": len(h5_files),
        "h5_files": h5_files[:20],
    }


def _failure_classes(
    slurm_fields: dict[str, str],
    logs: list[dict[str, Any]],
    artifacts: dict[str, Any],
) -> list[str]:
    classes: list[str] = []
    state = slurm_fields.get("JobState")
    reason = slurm_fields.get("Reason")
    if state == "PENDING":
        classes.append("scheduling_pending")
        if reason:
            classes.append(f"pending_reason_{reason.lower()}")
    if state in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
        classes.append(f"slurm_{state.lower()}")
    combined_logs = "\n".join(str(item.get("tail", "")) for item in logs)
    classes.extend(_pattern_hits(combined_logs).keys())
    if not artifacts.get("export_dir_exists"):
        classes.append("no_export_artifacts_yet")
    elif not artifacts.get("slurm_manifest_exists"):
        classes.append("missing_export_slurm_manifest")
    if artifacts.get("export_dir_exists") and artifacts.get("num_h5_files", 0) == 0:
        classes.append("missing_predicted_slot_h5_files")
    return sorted(set(classes))


def _recommended_next_action(classes: list[str]) -> str:
    class_set = set(classes)
    if "scheduling_pending" in class_set:
        return "wait_on_current_path_then_recheck_or_probe_only_if_same_objective_export_shape_may_start_earlier"
    if "slot_ensemble_or_checkpoint_error" in class_set:
        return "inspect_strict_rgbd_slot_training_and_inspection_outputs_do_not_use_oracle_slots"
    if "rgbd_input_or_file_count_error" in class_set:
        return "inspect_exact96_rgbd_root_and_file_list_do_not_relax_expected_count"
    if "hdf5_lock_error" in class_set:
        return "verify_hdf5_file_locking_false_then_retry_same_export_contract"
    if "memory_or_cuda_oom" in class_set:
        return "inspect_export_memory_failure_then_retry_same_rgbd_export_contract_with_supported_resource_fix"
    if "allocation_or_srun_error" in class_set:
        return "inspect_slurm_step_failure_before_resubmitting_same_export_contract"
    if "missing_predicted_slot_h5_files" in class_set:
        return "inspect_export_logs_and_manifest_before_interpreting_predicted_slot_quality"
    return "run_predicted_slot_export_inspection_and_record_quality_diagnostics_before_claiming_any_evidence"


def _write_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# RGB-D Predicted-Slot Export Triage",
        "",
        f"- job: `{report['job_id']}`",
        f"- export dir: `{report['export_dir']}`",
        f"- job state: `{report['slurm_fields'].get('JobState')}`",
        f"- pending reason: `{report['slurm_fields'].get('Reason')}`",
        f"- failure classes: `{', '.join(report['failure_classes']) or 'none'}`",
        f"- recommended next action: `{report['recommended_next_action']}`",
        "",
        "## Boundary",
        "",
        report["boundary"],
        "",
        "## Artifacts",
        "",
        f"- export dir exists: `{report['artifacts']['export_dir_exists']}`",
        f"- slurm manifest exists: `{report['artifacts']['slurm_manifest_exists']}`",
        f"- export report exists: `{report['artifacts']['export_report_exists']}`",
        f"- predicted-slot H5 files: `{report['artifacts']['num_h5_files']}`",
        "",
        "## Logs",
        "",
    ]
    for log in report["logs"]:
        lines.append(f"- `{log['path']}` exists `{log['exists']}` bytes `{log.get('bytes')}`")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--stdout")
    parser.add_argument("--stderr")
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    parser.add_argument("--tail-lines", type=int, default=120)
    ns = parser.parse_args()
    args = Args(**vars(ns))

    export_dir = Path(args.export_dir)
    logs = []
    if args.stdout:
        logs.append(_tail(Path(args.stdout), args.tail_lines))
    if args.stderr:
        logs.append(_tail(Path(args.stderr), args.tail_lines))
    scontrol = _run(["scontrol", "show", "job", str(args.job_id)])
    sacct = _run(
        [
            "sacct",
            "-j",
            str(args.job_id),
            "--format=JobID,JobName%28,State,Elapsed,Start,End,NodeList%24,ExitCode",
            "-P",
        ]
    )
    artifacts = _artifact_report(export_dir)
    slurm_fields = _parse_scontrol(scontrol.get("stdout", ""))
    classes = _failure_classes(slurm_fields, logs, artifacts)
    report = {
        "args": asdict(args),
        "job_id": str(args.job_id),
        "export_dir": str(export_dir),
        "timestamp": datetime.now().astimezone().isoformat(),
        "boundary": (
            "Diagnostic only. This triage classifies predicted-slot export Slurm/log/artifact "
            "failure modes. It does not change exact96 RGB-D input gates, predicted-slot "
            "structural handoff checks, advisory quality diagnostics, RGB-D-derived world-model "
            "requirements, or controller/video gates."
        ),
        "slurm_fields": slurm_fields,
        "scontrol": scontrol,
        "sacct": sacct,
        "logs": logs,
        "log_pattern_hits": _pattern_hits("\n".join(str(item.get("tail", "")) for item in logs)),
        "artifacts": artifacts,
        "failure_classes": classes,
        "recommended_next_action": _recommended_next_action(classes),
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True))
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        _write_md(report, Path(args.output_md))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
