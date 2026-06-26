#!/usr/bin/env python3
"""Repair LeRobot parquet HuggingFace feature metadata for current datasets.

The approved 733 OpenPI/LeRobot conversion wrote valid Arrow fixed-size-list
columns, but the parquet footer's HuggingFace feature metadata used the legacy
feature type name "List". The installed `datasets` version expects "Sequence".
This script changes only that footer metadata for vector columns such as
`state` and `actions`; it does not alter the model path, weights, rows, images,
or numeric values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import pyarrow.parquet as pq
import tyro


DEFAULT_DATASET_DIR = (
    "experiments/world_model_task_rebinding/openpi/lerobot_home/"
    "yanhongru/maniskill_peg733_openpi_libero"
)


@dataclass(frozen=True)
class Args:
    dataset_dir: str = DEFAULT_DATASET_DIR
    output_manifest: str | None = None
    expected_parquet_files: int = 733
    allow_login_node: bool = False
    dry_run: bool = False


def _refuse_login_node(args: Args) -> None:
    if args.allow_login_node:
        return
    step_id = os.environ.get("SLURM_STEP_ID", "")
    if not os.environ.get("SLURM_JOB_ID") or step_id in {"", "extern"}:
        raise SystemExit(
            "refusing_login_node_execution=true\n"
            "reason=Run LeRobot parquet metadata repair only inside a compute-node srun step."
        )


def _repair_one(path: Path, dry_run: bool) -> dict[str, Any]:
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata or {})
    raw_hf = metadata.get(b"huggingface")
    if raw_hf is None:
        return {"path": str(path), "changed": False, "reason": "missing_huggingface_metadata"}

    hf = json.loads(raw_hf)
    features = hf.get("info", {}).get("features", {})
    changed_keys: list[str] = []
    for key, value in features.items():
        if isinstance(value, dict) and value.get("_type") == "List":
            value["_type"] = "Sequence"
            changed_keys.append(key)

    if not changed_keys:
        return {"path": str(path), "changed": False, "reason": "already_compatible"}

    if not dry_run:
        metadata[b"huggingface"] = json.dumps(hf, separators=(",", ":"), sort_keys=True).encode()
        repaired = table.replace_schema_metadata(metadata)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            pq.write_table(repaired, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return {"path": str(path), "changed": True, "changed_keys": changed_keys}


def main(args: Args) -> None:
    _refuse_login_node(args)
    dataset_dir = Path(args.dataset_dir).resolve()
    parquet_files = sorted((dataset_dir / "data").glob("chunk-*/*.parquet"))
    if args.expected_parquet_files and len(parquet_files) != args.expected_parquet_files:
        raise RuntimeError(
            f"expected {args.expected_parquet_files} parquet files under {dataset_dir / 'data'}, "
            f"found {len(parquet_files)}"
        )

    records = [_repair_one(path, args.dry_run) for path in parquet_files]
    changed = [r for r in records if r.get("changed")]
    manifest = {
        "schema": "openpi_lerobot_parquet_hf_sequence_metadata_repair_v1",
        "dataset_dir": str(dataset_dir),
        "parquet_file_count": len(parquet_files),
        "changed_count": len(changed),
        "dry_run": args.dry_run,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "boundary": (
            "Parquet footer HuggingFace feature metadata repair only: legacy _type=List "
            "is rewritten to _type=Sequence for vector columns. Numeric data, image data, "
            "OpenPI model code, weight loader, and checkpoint paths are unchanged."
        ),
        "args": asdict(args),
        "records": records,
    }
    output_manifest = (
        Path(args.output_manifest).resolve()
        if args.output_manifest
        else dataset_dir / "maniskill_peg733_hf_sequence_metadata_repair_manifest.json"
    )
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in ["schema", "parquet_file_count", "changed_count", "dry_run"]}, indent=2))
    print(f"manifest={output_manifest}")


if __name__ == "__main__":
    tyro.cli(main)
