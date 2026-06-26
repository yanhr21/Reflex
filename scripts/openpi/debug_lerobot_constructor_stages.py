#!/usr/bin/env python3
"""Localize official LeRobot dataset construction hangs inside a Slurm step."""

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


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def summarize_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"type": type(item).__name__, "repr": repr(item)[:500]}
    summary: dict[str, Any] = {}
    for key, value in item.items():
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is not None:
            summary[str(key)] = {"shape": list(shape), "dtype": str(dtype)}
        else:
            summary[str(key)] = {"type": type(value).__name__, "repr": repr(value)[:200]}
    return summary


def main() -> int:
    args = parse_args()
    require_compute_step()
    events = Path(args.output_jsonl).resolve()
    summary_path = Path(args.output_json).resolve()
    t0 = time.time()
    summary: dict[str, Any] = {
        "schema": "lerobot_constructor_stages_v1",
        "config_name": str(args.config_name),
        "index": int(args.index),
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

        append_event(events, "import_openpi_config_done", seconds=time.time() - t)

        t = time.time()
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: PLC0415

        append_event(
            events,
            "import_lerobot_dataset_done",
            seconds=time.time() - t,
            module_file=str(getattr(lerobot_dataset, "__file__", "")),
        )

        t = time.time()
        config = _config.get_config(str(args.config_name))
        data_config = config.data.create(config.assets_dirs, config.model)
        repo_id = str(data_config.repo_id)
        action_keys = list(data_config.action_sequence_keys)
        append_event(
            events,
            "config_ready",
            seconds=time.time() - t,
            repo_id=repo_id,
            action_horizon=int(config.model.action_horizon),
            action_sequence_keys=action_keys,
        )

        t = time.time()
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        append_event(
            events,
            "metadata_done",
            seconds=time.time() - t,
            fps=float(dataset_meta.fps),
            tasks_count=len(getattr(dataset_meta, "tasks", {})),
            root=str(getattr(dataset_meta, "root", "")),
        )

        t = time.time()
        dataset_plain = lerobot_dataset.LeRobotDataset(repo_id)
        append_event(
            events,
            "dataset_plain_done",
            seconds=time.time() - t,
            dataset_type=type(dataset_plain).__name__,
            dataset_repr=repr(dataset_plain)[:500],
        )

        t = time.time()
        plain_len = len(dataset_plain)
        append_event(events, "dataset_plain_len_done", seconds=time.time() - t, length=int(plain_len))

        t = time.time()
        plain_item = dataset_plain[int(args.index)]
        append_event(
            events,
            "dataset_plain_getitem_done",
            seconds=time.time() - t,
            item_summary=summarize_item(plain_item),
        )

        delta_timestamps = {
            key: [timestep / dataset_meta.fps for timestep in range(int(config.model.action_horizon))]
            for key in action_keys
        }
        append_event(events, "delta_timestamps_ready", delta_timestamps=delta_timestamps)

        t = time.time()
        dataset_delta = lerobot_dataset.LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
        append_event(
            events,
            "dataset_delta_done",
            seconds=time.time() - t,
            dataset_type=type(dataset_delta).__name__,
            dataset_repr=repr(dataset_delta)[:500],
        )

        t = time.time()
        delta_len = len(dataset_delta)
        append_event(events, "dataset_delta_len_done", seconds=time.time() - t, length=int(delta_len))

        t = time.time()
        delta_item = dataset_delta[int(args.index)]
        append_event(
            events,
            "dataset_delta_getitem_done",
            seconds=time.time() - t,
            item_summary=summarize_item(delta_item),
        )

        summary.update(
            {
                "ok": True,
                "total_seconds": time.time() - t0,
                "repo_id": repo_id,
                "plain_len": int(plain_len),
                "delta_len": int(delta_len),
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
