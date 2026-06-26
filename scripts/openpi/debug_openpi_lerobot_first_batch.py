#!/usr/bin/env python3
"""Debug OpenPI LeRobot first-batch loading inside a Slurm compute step."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import jax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", default="/public/home/yanhongru/ICLR2027/openpi")
    parser.add_argument("--config-name", default="pi05_maniskill_peg733_contact_suffix16_object17")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--mode", choices=("raw_item", "transformed_item", "first_batch"), default="first_batch")
    parser.add_argument("--framework", choices=("jax", "pytorch"), default="jax")
    parser.add_argument("--shuffle", action="store_true")
    return parser.parse_args()


def require_compute_step() -> None:
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_STEP_ID") in (None, "extern"):
        raise SystemExit("refusing_login_node_execution: run inside a Slurm compute step")


def summarize_tree(tree: Any) -> Any:
    leaves = {}

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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> int:
    args = parse_args()
    require_compute_step()
    openpi_root = Path(args.openpi_root).resolve()
    os.chdir(openpi_root)
    sys.path.insert(0, str(openpi_root / "src"))

    from openpi.training import config as _config  # noqa: PLC0415
    from openpi.training import data_loader as _data_loader  # noqa: PLC0415
    from openpi.training import sharding  # noqa: PLC0415

    config = _config.get_config(str(args.config_name))
    config = dataclasses.replace(config, num_workers=int(args.num_workers), batch_size=int(args.batch_size))

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    payload: dict[str, Any] = {
        "schema": "openpi_lerobot_first_batch_debug_v1",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "config_name": config.name,
        "repo_id": getattr(config.data, "repo_id", None),
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "index": int(args.index),
        "mode": str(args.mode),
        "framework": str(args.framework),
        "shuffle": bool(args.shuffle),
        "openpi_root": str(openpi_root),
    }

    t0 = time.time()
    data_config = config.data.create(config.assets_dirs, config.model)
    if args.mode == "raw_item":
        dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
        payload["create_dataset_seconds"] = time.time() - t0
        t1 = time.time()
        item = dataset[int(args.index)]
        payload["item_seconds"] = time.time() - t1
        payload["item"] = summarize_tree(item)
    elif args.mode == "transformed_item":
        dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
        dataset = _data_loader.transform_dataset(dataset, data_config)
        payload["create_dataset_seconds"] = time.time() - t0
        t1 = time.time()
        item = dataset[int(args.index)]
        payload["item_seconds"] = time.time() - t1
        payload["item"] = summarize_tree(item)
    else:
        loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding if args.framework == "jax" else None,
            shuffle=bool(args.shuffle),
            framework=str(args.framework),
        )
        payload["create_data_loader_seconds"] = time.time() - t0
        t1 = time.time()
        batch = next(iter(loader))
        payload["first_batch_seconds"] = time.time() - t1
        observation, actions = batch
        payload["observation"] = summarize_tree(observation)
        payload["actions"] = summarize_tree(actions)
    payload["total_seconds"] = time.time() - t0
    write_json(Path(args.output_json).resolve(), payload)
    print(json.dumps({"ok": True, "output_json": str(Path(args.output_json).resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
