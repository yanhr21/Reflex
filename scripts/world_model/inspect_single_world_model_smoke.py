#!/usr/bin/env python3
"""Inspect a single RGB-D-derived world-model smoke output."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from object_slot_dataset import BASE_FEATURE_NAMES


@dataclass
class Args:
    model_dir: str
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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _last_elapsed(metrics: dict[str, Any]) -> float | None:
    if metrics.get("total_elapsed_seconds") is not None:
        return float(metrics["total_elapsed_seconds"])
    history = metrics.get("history", [])
    if not history:
        return None
    return float(history[-1].get("elapsed_seconds", 0.0))


def _rgbd_aux_feature_evidence(dataset_meta: dict[str, Any]) -> bool:
    feature_names = dataset_meta.get("feature_names")
    aux_names = dataset_meta.get("rgbd_aux_feature_names")
    if not isinstance(feature_names, list) or not isinstance(aux_names, list) or not aux_names:
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
        "# RGB-D-Derived World-Model Smoke Inspection",
        "",
        f"- model dir: `{report['model_dir']}`",
        f"- complete: `{report['complete']}`",
        f"- smoke RGB-D chain valid: `{report['smoke_rgbd_chain_valid']}`",
        f"- training evidence: `{report['training_evidence']}`",
        f"- input representation: `{report['dataset_input_representation']}`",
        f"- world-model input group: `{report['dataset_world_model_input_group']}`",
        f"- oracle slots read: `{report['dataset_oracle_slots_read']}`",
        f"- RGB-D predicted-slot input evidence: `{report['dataset_rgbd_predicted_slot_input_evidence']}`",
        f"- RGB-D aux feature evidence: `{report['dataset_rgbd_aux_feature_evidence']}`",
        f"- elapsed seconds: `{report['elapsed_seconds']}`",
        f"- best epoch: `{report['best_epoch']}`",
        f"- best val score: `{report['best_val_score']}`",
        "",
        "This is a smoke check only. It is not >=1xH200/>=3h training evidence and cannot support a method claim.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    args = tyro.cli(Args)
    model_dir = Path(args.model_dir)
    manifest = _load_json(model_dir / "manifest.json")
    metrics = _load_json(model_dir / "metrics.json")
    dataset_meta = manifest.get("dataset_meta", {})
    complete = bool(
        (model_dir / "model.pt").exists()
        and (model_dir / "best_model.pt").exists()
        and (model_dir / "metrics.json").exists()
        and (model_dir / "manifest.json").exists()
    )
    rgbd_aux_evidence = _rgbd_aux_feature_evidence(dataset_meta)
    smoke_valid = bool(
        complete
        and dataset_meta.get("input_representation") == "rgbd_predicted_slots"
        and dataset_meta.get("world_model_input_group") == "slots"
        and dataset_meta.get("oracle_slots_read") is False
        and dataset_meta.get("rgbd_predicted_slot_input_evidence") is True
        and rgbd_aux_evidence
    )
    report = {
        "args": asdict(args),
        "model_dir": str(model_dir),
        "complete": complete,
        "smoke_rgbd_chain_valid": smoke_valid,
        "training_evidence": False,
        "evidence_boundary": (
            "Smoke validates only the RGB-D predicted-slot to world-model code path. "
            "It is not method evidence, not full data evidence, and not >=1xH200/>=3h training evidence."
        ),
        "manifest_path": str(model_dir / "manifest.json"),
        "metrics_path": str(model_dir / "metrics.json"),
        "dataset_input_representation": dataset_meta.get("input_representation"),
        "dataset_world_model_input_group": dataset_meta.get("world_model_input_group"),
        "dataset_oracle_slots_read": dataset_meta.get("oracle_slots_read"),
        "dataset_rgbd_predicted_slot_input_evidence": dataset_meta.get(
            "rgbd_predicted_slot_input_evidence"
        ),
        "dataset_rgbd_aux_feature_names": dataset_meta.get("rgbd_aux_feature_names"),
        "dataset_rgbd_aux_feature_count": len(dataset_meta.get("rgbd_aux_feature_names", [])),
        "dataset_rgbd_aux_feature_evidence": rgbd_aux_evidence,
        "num_samples": dataset_meta.get("num_samples"),
        "num_paths": dataset_meta.get("num_paths"),
        "feature_dim": dataset_meta.get("feature_dim"),
        "target_cont_dim": dataset_meta.get("target_cont_dim"),
        "target_bin_dim": dataset_meta.get("target_bin_dim"),
        "elapsed_seconds": _last_elapsed(metrics),
        "best_epoch": metrics.get("best_epoch"),
        "best_val_score": metrics.get("best_val_score"),
        "val_by_horizon": metrics.get("val", {}).get("by_horizon", []),
        "train_by_horizon": metrics.get("train", {}).get("by_horizon", []),
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
