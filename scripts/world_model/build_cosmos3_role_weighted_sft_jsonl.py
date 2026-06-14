#!/usr/bin/env python3
"""Build a role-weighted Cosmos3 SFT JSONL without changing source episodes.

The active 300-step contract forbids sliced 128/129-frame samples. This utility
only repeats full-episode JSONL rows so underrepresented causal roles are sampled
more often by the SFT dataloader.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_weights(spec: str) -> dict[str, int]:
    weights: dict[str, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"role weight must be ROLE=N, got {item!r}")
        role, value = item.split("=", 1)
        role = role.strip()
        weight = int(value)
        if not role or weight < 1:
            raise argparse.ArgumentTypeError(f"invalid role weight {item!r}")
        weights[role] = weight
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--role-weights",
        default=(
            "target_pre_motion=2,target_motion_observed=1,target_post_motion=2,"
            "insert_resume=2,peg_recovery=3,static_monitor=3,static_late_monitor=3"
        ),
    )
    parser.add_argument("--late-rebind-weight", type=int, default=1)
    parser.add_argument(
        "--late-rebind-roles",
        default="target_motion_observed,target_post_motion,insert_resume",
        help="Comma-separated controller-facing roles or physical modes eligible for late-rebind upweighting.",
    )
    parser.add_argument("--late-rebind-min-abs-x", type=float, default=0.05)
    parser.add_argument("--late-rebind-min-abs-y", type=float, default=0.01)
    parser.add_argument("--late-rebind-min-abs-z", type=float, default=0.004)
    parser.add_argument("--manifest-out", type=Path, default=None)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty input JSONL: {path}")
    return rows


def row_role(row: dict[str, Any]) -> str:
    if row.get("prefix_role"):
        return str(row["prefix_role"])
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        payload = metadata.get("prefix_causal_state")
        if isinstance(payload, dict) and payload.get("prefix_role"):
            return str(payload["prefix_role"])
    return "unknown"


def prefix_payload(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    payload = metadata.get("prefix_causal_state")
    return payload if isinstance(payload, dict) else {}


def parse_csv_set(spec: str) -> set[str]:
    return {item.strip() for item in spec.split(",") if item.strip()}


def is_late_rebind_candidate(row: dict[str, Any], args: argparse.Namespace, roles: set[str]) -> bool:
    payload = prefix_payload(row)
    role = row_role(row)
    mode = str(row.get("physical_mode") or payload.get("mode", "unknown"))
    if roles and role not in roles and mode not in roles:
        return False
    if not bool(payload.get("target_motion_observed", False)):
        return False
    if not bool(payload.get("grasped", False)):
        return False
    if bool(payload.get("inserted", False)):
        return False
    rel = payload.get("peg_head_at_hole_xyz")
    if not isinstance(rel, list) or len(rel) < 3:
        return False
    abs_x, abs_y, abs_z = abs(float(rel[0])), abs(float(rel[1])), abs(float(rel[2]))
    return (
        abs_x > float(args.late_rebind_min_abs_x)
        or abs_y > float(args.late_rebind_min_abs_y)
        or abs_z > float(args.late_rebind_min_abs_z)
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    if int(args.late_rebind_weight) < 1:
        raise ValueError("--late-rebind-weight must be >= 1")
    weights = parse_weights(args.role_weights)
    late_rebind_roles = parse_csv_set(args.late_rebind_roles)
    rows = read_rows(args.input_jsonl)
    weighted: list[dict[str, Any]] = []
    role_counts: dict[str, int] = {}
    weighted_role_counts: dict[str, int] = {}
    late_rebind_counts: dict[str, int] = {}
    late_rebind_weighted_counts: dict[str, int] = {}
    for row in rows:
        role = row_role(row)
        role_counts[role] = role_counts.get(role, 0) + 1
        repeat = int(weights.get(role, 1))
        late_rebind = is_late_rebind_candidate(row, args, late_rebind_roles)
        if late_rebind:
            repeat *= int(args.late_rebind_weight)
            late_rebind_counts[role] = late_rebind_counts.get(role, 0) + 1
        weighted.extend([row] * repeat)
        weighted_role_counts[role] = weighted_role_counts.get(role, 0) + repeat
        if late_rebind:
            late_rebind_weighted_counts[role] = late_rebind_weighted_counts.get(role, 0) + repeat

    write_jsonl(args.output_jsonl, weighted)
    report = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "role_weights": weights,
        "late_rebind_weight": int(args.late_rebind_weight),
        "late_rebind_roles": sorted(late_rebind_roles),
        "late_rebind_thresholds": {
            "min_abs_x": float(args.late_rebind_min_abs_x),
            "min_abs_y": float(args.late_rebind_min_abs_y),
            "min_abs_z": float(args.late_rebind_min_abs_z),
        },
        "num_input_rows": len(rows),
        "num_output_rows": len(weighted),
        "input_role_counts": role_counts,
        "weighted_role_counts": weighted_role_counts,
        "late_rebind_input_counts": late_rebind_counts,
        "late_rebind_weighted_counts": late_rebind_weighted_counts,
        "contract": (
            "Rows are repeated full-episode 301/300 samples only. This does not "
            "slice, truncate, render, relabel, or regenerate source data."
        ),
    }
    if args.manifest_out is not None:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
