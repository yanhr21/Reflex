#!/usr/bin/env python3
"""Inspect object-state world-model ensemble training outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from object_slot_dataset import BASE_FEATURE_NAMES


@dataclass
class Args:
    ensemble_dir: str
    output_json: str | None = None
    output_md: str | None = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return list(value)
    return value


def _member_dirs(ensemble_dir: Path) -> list[Path]:
    return sorted([path for path in ensemble_dir.glob("member_*") if path.is_dir()])


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_key_value_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _requested_gpu_count(manifest: dict[str, str]) -> int | None:
    requested_gres = manifest.get("requested_gres", "")
    if not requested_gres.startswith("gpu:"):
        return None
    try:
        return int(requested_gres.rsplit(":", 1)[-1])
    except ValueError:
        return None


def _manifest_int(manifest: dict[str, str], key: str) -> int | None:
    value = manifest.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _last_elapsed(metrics: dict[str, Any]) -> float | None:
    if metrics.get("total_elapsed_seconds") is not None:
        return float(metrics["total_elapsed_seconds"])
    history = metrics.get("history", [])
    if not history:
        return None
    return float(history[-1].get("elapsed_seconds", 0.0))


def _member_manifest_elapsed(path: Path) -> float | None:
    if not path.exists():
        return None
    start: datetime | None = None
    complete: datetime | None = None
    for line in path.read_text().splitlines():
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key == "start":
                start = datetime.fromisoformat(value)
            elif key == "complete":
                complete = datetime.fromisoformat(value)
    if start is None or complete is None:
        return None
    return float((complete - start).total_seconds())


def _member_elapsed(metrics: dict[str, Any], member_dir: Path) -> float | None:
    values = [
        value
        for value in (
            _last_elapsed(metrics),
            _member_manifest_elapsed(member_dir / "member_manifest.txt"),
        )
        if value is not None
    ]
    return float(max(values)) if values else None


def _horizon_rows(metrics: dict[str, Any], split: str = "val") -> list[dict[str, Any]]:
    return list(metrics.get(split, {}).get("by_horizon", []))


def _aggregate_members(member_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not member_reports:
        return []
    horizons = [row["horizon"] for row in member_reports[0]["val_by_horizon"]]
    rows = []
    for h_i, horizon in enumerate(horizons):
        keys = [
            "hole_delta_rmse_m",
            "static_hole_delta_rmse_m",
            "cv_hole_delta_rmse_m",
            "peg_head_hole_rmse_m",
            "binary_accuracy",
            "mean_hole_uncertainty_m",
            "hole_error_uncertainty_pearson",
        ]
        row: dict[str, Any] = {"horizon": int(horizon)}
        for key in keys:
            values = []
            for member in member_reports:
                value = member["val_by_horizon"][h_i].get(key)
                if value is not None:
                    values.append(float(value))
            row[f"{key}_mean"] = float(np.mean(values)) if values else None
            row[f"{key}_std"] = float(np.std(values)) if values else None
        rows.append(row)
    return rows


def _rgbd_aux_feature_evidence(member: dict[str, Any]) -> bool:
    feature_names = member.get("feature_names")
    dataset_feature_names = member.get("dataset_feature_names")
    aux_names = member.get("dataset_rgbd_aux_feature_names")
    if not isinstance(feature_names, list) or not isinstance(dataset_feature_names, list):
        return False
    if feature_names != dataset_feature_names:
        return False
    if not isinstance(aux_names, list) or not aux_names:
        return False
    feature_set = set(str(name) for name in feature_names)
    if not set(BASE_FEATURE_NAMES).issubset(feature_set):
        return False
    aux_set = set(str(name) for name in aux_names)
    if not aux_set.issubset(feature_set):
        return False
    prefixes = ("rgbd_cont_std_", "rgbd_bin_prob_", "rgbd_bin_std_")
    return all(any(str(name).startswith(prefix) for name in aux_names) for prefix in prefixes)


def _write_md(report: dict[str, Any], path: Path):
    lines = [
        "# World-Model Ensemble Inspection",
        "",
        f"- ensemble: `{report['ensemble_dir']}`",
        f"- Slurm job: `{report['slurm_job_id']}`",
        f"- Slurm node: `{report['slurm_node_list']}`",
        f"- requested GPUs: `{report['requested_gpu_count']}`",
        f"- expected GPU type: `{report['expected_gpu_type']}`",
        f"- input representation: `{report['input_representation']}`",
        f"- world-model input group: `{report['world_model_input_group']}`",
        f"- oracle slots not used: `{report['oracle_slots_not_used']}`",
        f"- min predicted slot files: `{report['min_predicted_slot_files']}`",
        f"- expected predicted slot files: `{report['expected_predicted_slot_files']}`",
        f"- predicted slot file count: `{report['predicted_slot_file_count']}`",
        f"- exact expected predicted-slot input evidence: `{report['exact_expected_predicted_slot_input_evidence']}`",
        f"- member RGB-D predicted-slot input evidence: `{report['member_rgbd_predicted_slot_input_evidence']}`",
        f"- member RGB-D aux feature evidence: `{report['member_rgbd_aux_feature_input_evidence']}`",
        f"- RGB-D predicted-slot input evidence: `{report['rgbd_predicted_slot_input_evidence']}`",
        f"- members found: `{report['num_members']}`",
        f"- required complete members: `{report['required_complete_members']}`",
        f"- complete members: `{report['num_complete_members']}`",
        f"- min elapsed seconds: `{report['min_elapsed_seconds']}`",
        f"- compliant 3h training: `{report['compliant_3h_training']}`",
        f"- compliant H200 request: `{report['compliant_h200_request']}`",
        f"- legacy compliant 4xH200 request: `{report['compliant_4x_h200_request']}`",
        f"- compliant training evidence: `{report['compliant_training_evidence']}`",
        f"- RGB-D-derived training evidence: `{report['rgbd_derived_training_evidence']}`",
        "",
        "## Validation Aggregate",
        "",
        "| horizon | learned RMSE mean | static RMSE mean | CV RMSE mean | peg-head RMSE mean | binary acc mean | uncertainty mean | unc-error corr mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["val_aggregate_by_horizon"]:
        def fmt(value):
            return "n/a" if value is None else f"{value:.5f}"

        lines.append(
            "| {horizon} | {learned} | {static} | {cv} | {peg} | {acc} | {unc} | {corr} |".format(
                horizon=row["horizon"],
                learned=fmt(row["hole_delta_rmse_m_mean"]),
                static=fmt(row["static_hole_delta_rmse_m_mean"]),
                cv=fmt(row["cv_hole_delta_rmse_m_mean"]),
                peg=fmt(row["peg_head_hole_rmse_m_mean"]),
                acc=fmt(row["binary_accuracy_mean"]),
                unc=fmt(row["mean_hole_uncertainty_m_mean"]),
                corr=fmt(row["hole_error_uncertainty_pearson_mean"]),
            )
        )
    lines.extend(["", "## Members", "", "| member | complete | elapsed s | best epoch | best val score | RGB-D slot input | RGB-D aux features | oracle read |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for member in report["members"]:
        lines.append(
            "| {member} | {complete} | {elapsed} | {best_epoch} | {best_val_score} | {rgbd_input} | {rgbd_aux} | {oracle_read} |".format(
                member=member["member"],
                complete=member["complete"],
                elapsed="n/a" if member["elapsed_seconds"] is None else f"{member['elapsed_seconds']:.1f}",
                best_epoch=member.get("best_epoch", "n/a"),
                best_val_score=(
                    "n/a"
                    if member.get("best_val_score") is None
                    else f"{member['best_val_score']:.5f}"
                ),
                rgbd_input=member.get("dataset_rgbd_predicted_slot_input_evidence"),
                rgbd_aux=member.get("dataset_rgbd_aux_feature_evidence"),
                oracle_read=member.get("dataset_oracle_slots_read"),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    args = tyro.cli(Args)
    ensemble_dir = Path(args.ensemble_dir)
    slurm_manifest = _load_key_value_manifest(ensemble_dir / "slurm_manifest.txt")
    requested_gpu_count = _requested_gpu_count(slurm_manifest)
    expected_gpu_type = slurm_manifest.get("cluster_gpu_type_expected")
    input_representation = slurm_manifest.get("input_representation")
    world_model_input_group = slurm_manifest.get("world_model_input_group")
    oracle_slots_not_used = slurm_manifest.get("oracle_slots_not_used")
    min_predicted_slot_files = _manifest_int(slurm_manifest, "min_predicted_slot_files")
    expected_predicted_slot_files = _manifest_int(slurm_manifest, "expected_predicted_slot_files")
    predicted_slot_file_count = _manifest_int(slurm_manifest, "predicted_slot_file_count")
    manifest_ntasks = _manifest_int(slurm_manifest, "ntasks")
    min_train_gpus = _manifest_int(slurm_manifest, "min_train_gpus") or 1
    member_reports = []
    for member_dir in _member_dirs(ensemble_dir):
        metrics_path = member_dir / "metrics.json"
        manifest_path = member_dir / "manifest.json"
        complete = metrics_path.exists() and (member_dir / "model.pt").exists()
        metrics = _load_json(metrics_path) if metrics_path.exists() else {}
        manifest = _load_json(manifest_path) if manifest_path.exists() else {}
        dataset_meta = manifest.get("dataset_meta", {})
        feature_names = manifest.get("feature_names")
        dataset_feature_names = dataset_meta.get("feature_names")
        rgbd_aux_feature_names = dataset_meta.get("rgbd_aux_feature_names")
        rgbd_aux_feature_count = (
            len(rgbd_aux_feature_names)
            if isinstance(rgbd_aux_feature_names, list)
            else 0
        )
        member_report = {
            "member": member_dir.name,
            "path": str(member_dir),
            "complete": complete,
            "elapsed_seconds": _member_elapsed(metrics, member_dir),
            "metrics_elapsed_seconds": _last_elapsed(metrics),
            "member_manifest_elapsed_seconds": _member_manifest_elapsed(
                member_dir / "member_manifest.txt"
            ),
            "best_epoch": metrics.get("best_epoch"),
            "best_val_score": metrics.get("best_val_score"),
            "val_by_horizon": _horizon_rows(metrics, "val"),
            "train_by_horizon": _horizon_rows(metrics, "train"),
            "seed": manifest.get("args", {}).get("seed"),
            "slurm_job_id": manifest.get("env", {}).get("SLURM_JOB_ID"),
            "feature_names": feature_names,
            "dataset_feature_names": dataset_feature_names,
            "dataset_rgbd_aux_feature_names": rgbd_aux_feature_names,
            "dataset_rgbd_aux_feature_count": rgbd_aux_feature_count,
            "dataset_input_representation": dataset_meta.get("input_representation"),
            "dataset_world_model_input_group": dataset_meta.get("world_model_input_group"),
            "dataset_oracle_slots_read": dataset_meta.get("oracle_slots_read"),
            "dataset_rgbd_predicted_slot_input_evidence": dataset_meta.get(
                "rgbd_predicted_slot_input_evidence"
            ),
        }
        member_report["dataset_rgbd_aux_feature_evidence"] = _rgbd_aux_feature_evidence(member_report)
        member_reports.append(
            member_report
        )
    elapsed_values = [
        item["elapsed_seconds"] for item in member_reports if item["elapsed_seconds"] is not None
    ]
    complete_members = int(sum(item["complete"] for item in member_reports))
    required_complete_members = max(1, manifest_ntasks or requested_gpu_count or min_train_gpus)
    required_members_complete = bool(
        len(member_reports) >= required_complete_members
        and complete_members >= required_complete_members
        and all(item["complete"] for item in member_reports)
    )
    compliant_3h_training = bool(
        required_members_complete
        and elapsed_values
        and min(elapsed_values) >= 10800
    )
    compliant_h200_request = bool(
        requested_gpu_count is not None
        and requested_gpu_count >= min_train_gpus
        and expected_gpu_type == "NVIDIAH200"
    )
    compliant_4h200_request = bool(
        requested_gpu_count is not None
        and requested_gpu_count >= 4
        and expected_gpu_type == "NVIDIAH200"
    )
    member_rgbd_predicted_slot_input = bool(
        len(member_reports) >= required_complete_members
        and all(
            item.get("dataset_input_representation") == "rgbd_predicted_slots"
            and item.get("dataset_world_model_input_group") == "slots"
            and item.get("dataset_oracle_slots_read") is False
            and item.get("dataset_rgbd_predicted_slot_input_evidence") is True
            for item in member_reports
        )
    )
    member_rgbd_aux_feature_input = bool(
        len(member_reports) >= required_complete_members
        and all(item.get("dataset_rgbd_aux_feature_evidence") is True for item in member_reports)
    )
    rgbd_predicted_slot_input = bool(
        input_representation == "rgbd_predicted_slots"
        and world_model_input_group == "slots"
        and oracle_slots_not_used == "true"
        and min_predicted_slot_files is not None
        and min_predicted_slot_files >= 96
        and expected_predicted_slot_files is not None
        and expected_predicted_slot_files >= 96
        and predicted_slot_file_count is not None
        and predicted_slot_file_count >= 96
        and predicted_slot_file_count == expected_predicted_slot_files
        and member_rgbd_predicted_slot_input
        and member_rgbd_aux_feature_input
    )
    exact_expected_predicted_slot_input = bool(
        expected_predicted_slot_files is not None
        and predicted_slot_file_count is not None
        and predicted_slot_file_count == expected_predicted_slot_files
    )
    report = {
        "args": asdict(args),
        "ensemble_dir": str(ensemble_dir),
        "slurm_manifest_path": str(ensemble_dir / "slurm_manifest.txt"),
        "slurm_job_id": slurm_manifest.get("job_id"),
        "slurm_node_list": slurm_manifest.get("node_list"),
        "requested_gres": slurm_manifest.get("requested_gres"),
        "requested_gpu_count": requested_gpu_count,
        "expected_gpu_type": expected_gpu_type,
        "manifest_ntasks": manifest_ntasks,
        "min_train_gpus": min_train_gpus,
        "input_representation": input_representation,
        "world_model_input_group": world_model_input_group,
        "oracle_slots_not_used": oracle_slots_not_used,
        "min_predicted_slot_files": min_predicted_slot_files,
        "expected_predicted_slot_files": expected_predicted_slot_files,
        "predicted_slot_file_count": predicted_slot_file_count,
        "exact_expected_predicted_slot_input_evidence": exact_expected_predicted_slot_input,
        "member_rgbd_predicted_slot_input_evidence": member_rgbd_predicted_slot_input,
        "member_rgbd_aux_feature_input_evidence": member_rgbd_aux_feature_input,
        "rgbd_predicted_slot_input_evidence": rgbd_predicted_slot_input,
        "num_members": len(member_reports),
        "required_complete_members": required_complete_members,
        "num_complete_members": complete_members,
        "min_elapsed_seconds": float(min(elapsed_values)) if elapsed_values else None,
        "compliant_3h_training": compliant_3h_training,
        "compliant_h200_request": compliant_h200_request,
        "compliant_4h200_request": compliant_4h200_request,
        "compliant_4x_h200_request": compliant_4h200_request,
        "compliant_training_evidence": bool(compliant_3h_training and compliant_h200_request),
        "rgbd_derived_training_evidence": bool(
            compliant_3h_training and compliant_h200_request and rgbd_predicted_slot_input
        ),
        "evidence_boundary": "rgbd_derived_training_evidence requires RGB-D-predicted slots, oracle_slots_not_used=true, member manifests proving oracle_slots_read=false, RGB-D uncertainty/probability auxiliary features in the world-model input, exact expected input files, at least 1 H200 GPU, and >=3h training. If a job requests multiple GPU tasks, the corresponding member directories must all complete.",
        "members": member_reports,
        "val_aggregate_by_horizon": _aggregate_members(
            [item for item in member_reports if item["complete"] and item["val_by_horizon"]]
        ),
    }
    if args.output_json is not None:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    if args.output_md is not None:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        _write_md(report, Path(args.output_md))
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
