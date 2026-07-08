#!/usr/bin/env python3
"""Extract a denormalized robot-action chunk from Cosmos3 policy output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-output-json", required=True)
    parser.add_argument("--normalization-stats", required=True)
    parser.add_argument("--prefix-frame-index", type=int, required=True)
    parser.add_argument(
        "--action-row-offset",
        type=int,
        default=0,
        help=(
            "Diagnostic/action-label alignment offset added to prefix_frame_index "
            "before slicing the predicted action rows. Default 0 preserves the "
            "original extractor behavior."
        ),
    )
    parser.add_argument("--action-exec-horizon", type=int, default=8)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--expected-action-dim", type=int, default=32)
    parser.add_argument("--robot-action-dim", type=int, default=7)
    parser.add_argument("--allow-padded-action-dim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def float_matrix(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise ValueError("action output is not a non-empty list")
    out: list[list[float]] = []
    for row in value:
        if not isinstance(row, list):
            raise ValueError("action output row is not a list")
        out.append([float(x) for x in row])
    return out


def stats(values: list[list[float]]) -> dict[str, Any]:
    flat = [x for row in values for x in row]
    return {
        "finite": all(math.isfinite(x) for x in flat),
        "num_values": len(flat),
        "min": min(flat) if flat else None,
        "max": max(flat) if flat else None,
        "mean_abs": sum(abs(x) for x in flat) / float(len(flat)) if flat else None,
        "max_abs": max(abs(x) for x in flat) if flat else None,
    }


def main() -> int:
    args = parse_args()
    sample_output_path = Path(args.sample_output_json).resolve()
    stats_path = Path(args.normalization_stats).resolve()
    output_path = Path(args.output_json).resolve()

    payload = read_json(sample_output_path)
    outputs = payload.get("outputs") or []
    content = (outputs[0].get("content") if outputs else None) or {}
    action = float_matrix(content.get("action"))
    row_dims = [len(row) for row in action]
    action_dim_ok = all(
        dim == args.expected_action_dim
        or (args.allow_padded_action_dim and dim > args.expected_action_dim)
        for dim in row_dims
    )
    if len(action) != args.expected_action_steps or not action_dim_ok:
        raise ValueError(
            f"predicted action shape {[len(action), sorted(set(row_dims))]} "
            f"is incompatible with [{args.expected_action_steps}, {args.expected_action_dim}]"
        )
    action = [row[: args.expected_action_dim] for row in action]

    norm = read_json(stats_path)
    mean = [float(x) for x in norm["mean"]]
    std = [float(x) for x in norm["std"]]
    vector_names = list(norm.get("vector_names") or [])
    if len(mean) != args.expected_action_dim or len(std) != args.expected_action_dim:
        raise ValueError("normalization stats dimension mismatch")

    raw_start = int(args.prefix_frame_index) + int(args.action_row_offset)
    start = max(0, min(raw_start, args.expected_action_steps))
    end = min(args.expected_action_steps, start + max(1, int(args.action_exec_horizon)))
    normalized_robot = [row[: args.robot_action_dim] for row in action[start:end]]
    denormalized_robot = [
        [row[i] * std[i] + mean[i] for i in range(args.robot_action_dim)]
        for row in normalized_robot
    ]
    result = {
        "sample_output_json": str(sample_output_path),
        "normalization_stats": str(stats_path),
        "prefix_frame_index": int(args.prefix_frame_index),
        "action_row_offset": int(args.action_row_offset),
        "raw_chunk_start": int(raw_start),
        "chunk_start": start,
        "chunk_end_exclusive": end,
        "chunk_steps": end - start,
        "expected_action_shape": [args.expected_action_steps, args.expected_action_dim],
        "robot_action_dim": args.robot_action_dim,
        "robot_action_vector_names": vector_names[: args.robot_action_dim],
        "normalized_robot_action_stats": stats(normalized_robot),
        "denormalized_robot_action_stats": stats(denormalized_robot),
        "denormalized_robot_action_chunk": denormalized_robot,
        "boundary": (
            "Robot-action extraction only. Columns after robot_action_dim are "
            "task-state sidecars for diagnostics/scoring and are not simulator "
            "state. Execute only a short prefix, then reobserve and rerun Cosmos."
        ),
    }
    result["ok"] = bool(
        result["chunk_steps"] > 0
        and result["normalized_robot_action_stats"]["finite"]
        and result["denormalized_robot_action_stats"]["finite"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": result["ok"], "output_json": str(output_path)}, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
