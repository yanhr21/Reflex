#!/usr/bin/env python3
"""Triage an ensemble-training Slurm job without approving method evidence.

This read-only tool classifies scheduling, log, and artifact failure modes for
training jobs such as RGB-D slot extraction or RGB-D-derived world models. It
does not replace the strict inspection scripts and does not change evaluation
gates.
"""

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
        "numpy._typing",
        "No module named",
        "undefined symbol",
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
        "std::bad_alloc",
        "Cannot allocate memory",
    ),
    "rgbd_slot_input_or_file_count_error": (
        "No RGB-D slot samples",
        "Missing observations/",
        "Missing slots/",
        "Refusing RGB-D slot training",
        "MIN_RGBD_FILES",
        "EXPECTED_RGBD_FILES",
    ),
    "predicted_slot_input_or_file_count_error": (
        "Refusing RGB-D-derived world-model training",
        "MIN_PREDICTED_SLOT_FILES",
        "EXPECTED_PREDICTED_SLOT_FILES",
        "Need at least two trajectory files",
        "missing slots group",
        "is not an RGB-D-predicted slot export",
        "missing prediction uncertainty",
        "missing slots attr",
        "missing oracle_slots",
    ),
    "allocation_or_srun_error": (
        "srun: error",
        "slurmstepd:",
        "Unable to create step",
        "Job credential expired",
    ),
    "undersized_training_refusal": (
        "Refusing undersized training",
        "need >=MIN_TRAIN_GPUS",
        "need at least 1 H200",
        "need >=10800",
        "Refusing sparse training allocation",
    ),
}


@dataclass
class Args:
    job_id: str
    ensemble_dir: str
    job_kind: str = "ensemble_training"
    stdout: str | None = None
    stderr: str | None = None
    output_json: str | None = None
    output_md: str | None = None
    tail_lines: int = 100


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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"_json_decode_error": True}


def _load_kv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _member_elapsed(member_manifest: Path) -> float | None:
    if not member_manifest.exists():
        return None
    start: datetime | None = None
    complete: datetime | None = None
    for line in member_manifest.read_text(errors="replace").splitlines():
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            try:
                if key == "start":
                    start = datetime.fromisoformat(value)
                elif key == "complete":
                    complete = datetime.fromisoformat(value)
            except ValueError:
                continue
    if start is None or complete is None:
        return None
    return float((complete - start).total_seconds())


def _member_reports(ensemble_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for member_dir in sorted(ensemble_dir.glob("member_*")):
        if not member_dir.is_dir():
            continue
        metrics = _load_json(member_dir / "metrics.json")
        manifest = _load_json(member_dir / "manifest.json")
        dataset = manifest.get("dataset", {})
        args = manifest.get("args", {})
        reports.append(
            {
                "member": member_dir.name,
                "path": str(member_dir),
                "has_manifest_json": (member_dir / "manifest.json").exists(),
                "has_metrics_json": (member_dir / "metrics.json").exists(),
                "has_model_pt": (member_dir / "model.pt").exists(),
                "has_member_manifest": (member_dir / "member_manifest.txt").exists(),
                "complete": (member_dir / "metrics.json").exists() and (member_dir / "model.pt").exists(),
                "elapsed_from_member_manifest_seconds": _member_elapsed(member_dir / "member_manifest.txt"),
                "metrics_total_elapsed_seconds": metrics.get("total_elapsed_seconds"),
                "best_epoch": metrics.get("best_epoch"),
                "best_score": metrics.get("best_score"),
                "seed": args.get("seed"),
                "input_representation": dataset.get("input_representation"),
                "input_modality": dataset.get("input_modality"),
                "world_model_input_group": dataset.get("world_model_input_group"),
                "oracle_slots_read": dataset.get("oracle_slots_read"),
                "num_samples": dataset.get("num_samples"),
                "feature_shape": dataset.get("feature_shape"),
                "image_shape": dataset.get("image_shape"),
            }
        )
    return reports


def _parse_scontrol(stdout: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key in ("JobState", "Reason", "RunTime", "TimeLimit", "StartTime", "NodeList", "ReqNodeList"):
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


def _failure_classes(
    slurm_fields: dict[str, str],
    logs: list[dict[str, Any]],
    members: list[dict[str, Any]],
    ensemble_dir: Path,
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
    if not ensemble_dir.exists():
        classes.append("no_ensemble_artifacts_yet")
    elif not (ensemble_dir / "slurm_manifest.txt").exists():
        classes.append("missing_slurm_manifest")
    if members:
        complete = sum(1 for member in members if member.get("complete"))
        if complete < 4:
            classes.append("incomplete_ensemble_members")
    return sorted(set(classes))


def _recommended_next_action(classes: list[str], job_kind: str) -> str:
    class_set = set(classes)
    if "scheduling_pending" in class_set:
        return "wait_on_current_path_then_recheck_or_probe_only_if_a_legal_same_objective_shape_may_start_earlier"
    if "import_or_environment_error" in class_set:
        return f"inspect_import_trace_fix_environment_then_rerun_same_{job_kind}_contract"
    if "hdf5_lock_error" in class_set:
        return "verify_hdf5_file_locking_false_in_submitted_snapshot_then_retry_same_training_contract"
    if "memory_or_cuda_oom" in class_set:
        return "inspect_memory_failure_then_retry_same_rgbd_training_contract_with_supported_loading_or_memory_fix"
    if "rgbd_slot_input_or_file_count_error" in class_set:
        return "inspect_rgbd_root_and_exact_expected_file_list_do_not_relax_expected_count"
    if "predicted_slot_input_or_file_count_error" in class_set:
        return "inspect_rgbd_predicted_slot_export_and_exact_expected_file_list_do_not_use_oracle_slots_as_replacement"
    if "undersized_training_refusal" in class_set:
        return "fix_submission_shape_to_one_node_at_least_one_h200_min_train_seconds_10800"
    if "incomplete_ensemble_members" in class_set:
        return "inspect_member_logs_and_manifests_before_interpreting_training_quality"
    if "no_ensemble_artifacts_yet" in class_set:
        return "no_training_artifacts_to_interpret_keep_waiting_or_probe_same_objective_replacement"
    return "run_strict_inspection_or_inspect_logs_before_claiming_any_evidence"


def _write_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Ensemble Training Triage",
        "",
        f"- job: `{report['job_id']}`",
        f"- kind: `{report['job_kind']}`",
        f"- ensemble: `{report['ensemble_dir']}`",
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
        f"- ensemble exists: `{report['ensemble_exists']}`",
        f"- slurm manifest exists: `{report['slurm_manifest_exists']}`",
        f"- member dirs: `{report['num_member_dirs']}`",
        f"- complete members: `{report['num_complete_members']}`",
        "",
        "## Logs",
        "",
    ]
    for log in report["logs"]:
        lines.append(f"- `{log['path']}` exists `{log['exists']}` bytes `{log.get('bytes')}`")
    lines.extend(
        [
            "",
            "## Members",
            "",
            "| member | complete | metrics | model | elapsed s | input representation | samples |",
            "|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for member in report["members"]:
        elapsed = member.get("metrics_total_elapsed_seconds") or member.get("elapsed_from_member_manifest_seconds")
        lines.append(
            "| {member} | {complete} | {metrics} | {model} | {elapsed} | `{inp}` | {samples} |".format(
                member=member["member"],
                complete=member["complete"],
                metrics=member["has_metrics_json"],
                model=member["has_model_pt"],
                elapsed="n/a" if elapsed is None else f"{float(elapsed):.1f}",
                inp=member.get("input_representation") or member.get("input_modality"),
                samples=member.get("num_samples"),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--ensemble-dir", required=True)
    parser.add_argument("--job-kind", default="ensemble_training")
    parser.add_argument("--stdout")
    parser.add_argument("--stderr")
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    parser.add_argument("--tail-lines", type=int, default=100)
    ns = parser.parse_args()
    args = Args(**vars(ns))

    ensemble_dir = Path(args.ensemble_dir)
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
    members = _member_reports(ensemble_dir)
    slurm_fields = _parse_scontrol(scontrol.get("stdout", ""))
    classes = _failure_classes(slurm_fields, logs, members, ensemble_dir)
    report = {
        "args": asdict(args),
        "job_id": str(args.job_id),
        "job_kind": args.job_kind,
        "ensemble_dir": str(ensemble_dir),
        "timestamp": datetime.now().astimezone().isoformat(),
        "boundary": (
            "Diagnostic only. This triage classifies scheduling/log/artifact failure modes. "
            "It does not change RGB-D slot quality thresholds, exact expected-count gates, "
            "the >=1 H200/>=3h training floor, "
            "RGB-D-derived world-model requirements, controller metrics, or video evidence rules."
        ),
        "slurm_fields": slurm_fields,
        "scontrol": scontrol,
        "sacct": sacct,
        "logs": logs,
        "log_pattern_hits": _pattern_hits("\n".join(str(item.get("tail", "")) for item in logs)),
        "ensemble_exists": ensemble_dir.exists(),
        "slurm_manifest_exists": (ensemble_dir / "slurm_manifest.txt").exists(),
        "slurm_manifest": _load_kv(ensemble_dir / "slurm_manifest.txt"),
        "members": members,
        "num_member_dirs": len(members),
        "num_complete_members": sum(1 for member in members if member.get("complete")),
        "failure_classes": classes,
        "recommended_next_action": _recommended_next_action(classes, args.job_kind),
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
