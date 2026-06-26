#!/usr/bin/env python3
"""Stage-by-stage OpenPI LeRobot dataset debug inside a Slurm compute step."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", default="/public/home/yanhongru/ICLR2027/openpi")
    parser.add_argument("--config-name", default="pi05_maniskill_peg733_contact_suffix16_object17_clean_20260626")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def require_compute_step() -> None:
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_STEP_ID") in (None, "extern"):
        raise SystemExit("refusing_login_node_execution: run inside a Slurm compute step")


def append_event(path: Path, stage: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "time": time.time(),
        "stage": stage,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        **payload,
    }
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(row, sort_keys=True), flush=True)


def summarize_tree(tree: Any) -> dict[str, Any]:
    leaves: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                visit(f"{prefix}.{field.name}" if prefix else field.name, getattr(value, field.name))
            return
        if isinstance(value, dict):
            for key, item in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), item)
            return
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is not None:
            leaves[prefix] = {"shape": list(shape), "dtype": str(dtype)}
        else:
            leaves[prefix] = {"type": type(value).__name__, "repr": repr(value)[:200]}

    visit("", tree)
    return leaves


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> int:
    args = parse_args()
    require_compute_step()
    events = Path(args.output_jsonl).resolve()
    summary_path = Path(args.output_json).resolve()
    t0 = time.time()
    summary: dict[str, Any] = {
        "schema": "openpi_lerobot_staged_debug_v1",
        "config_name": args.config_name,
        "index": int(args.index),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "openpi_root": str(Path(args.openpi_root).resolve()),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
    }

    try:
        append_event(events, "start", python=sys.executable, argv=sys.argv)
        openpi_root = Path(args.openpi_root).resolve()
        sys.path.insert(0, str(openpi_root / "src"))
        append_event(events, "sys_path_ready", openpi_src=str(openpi_root / "src"))

        t = time.time()
        from openpi.training import config as _config  # noqa: PLC0415

        append_event(events, "import_config_done", seconds=time.time() - t)

        t = time.time()
        from openpi.training import data_loader as _data_loader  # noqa: PLC0415

        append_event(events, "import_data_loader_done", seconds=time.time() - t)

        t = time.time()
        config = _config.get_config(str(args.config_name))
        config = dataclasses.replace(config, num_workers=int(args.num_workers), batch_size=int(args.batch_size))
        append_event(
            events,
            "get_config_done",
            seconds=time.time() - t,
            repo_id=getattr(config.data, "repo_id", None),
            model_action_horizon=int(config.model.action_horizon),
        )

        t = time.time()
        data_config = config.data.create(config.assets_dirs, config.model)
        append_event(
            events,
            "data_config_create_done",
            seconds=time.time() - t,
            repo_id=getattr(data_config, "repo_id", None),
        )

        t = time.time()
        dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
        append_event(
            events,
            "create_torch_dataset_done",
            seconds=time.time() - t,
            dataset_type=type(dataset).__name__,
            dataset_repr=repr(dataset)[:500],
        )

        t = time.time()
        length = len(dataset)
        append_event(events, "len_dataset_done", seconds=time.time() - t, length=int(length))

        t = time.time()
        item = dataset[int(args.index)]
        item_summary = summarize_tree(item)
        append_event(events, "getitem_done", seconds=time.time() - t, key_count=len(item_summary))

        summary.update(
            {
                "ok": True,
                "total_seconds": time.time() - t0,
                "dataset_len": int(length),
                "item_summary": item_summary,
            }
        )
        write_summary(summary_path, summary)
        append_event(events, "summary_written", output_json=str(summary_path))
        return 0
    except Exception as exc:  # noqa: BLE001
        summary.update(
            {
                "ok": False,
                "total_seconds": time.time() - t0,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_summary(summary_path, summary)
        append_event(events, "exception", error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
