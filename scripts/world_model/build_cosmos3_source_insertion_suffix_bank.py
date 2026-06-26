#!/usr/bin/env python3
"""Build a bank of successful source insertion action suffixes.

This is a diagnostic candidate-source artifact. It mines successful H5
trajectories for action chunks immediately before insertion so live snapshot
replay can test whether a stronger contact/insertion action source has
physical headroom. It is not method evidence by itself.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_cosmos3_receding_closed_loop import jsonable  # noqa: E402


DEFAULT_SOURCE_ROOT = (
    ROOT
    / "experiments/world_model_task_rebinding/cosmos3/"
    "fix3_v7_dp_user_override_sft_source_20260612_733"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5-paths-file",
        default=str(DEFAULT_SOURCE_ROOT / "fix3_h5_paths_canonical.txt"),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--start-offsets-before-insert", default="96,64,48,32,24,16,8,0")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--scenario-regex", default="")
    parser.add_argument("--require-success-at-end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-grasped-at-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-start-abs-yz-sum", type=float, default=0.08)
    parser.add_argument("--expected-actions", type=int, default=300)
    parser.add_argument("--expected-frames", type=int, default=301)
    return parser.parse_args()


def parse_ints(text: str) -> list[int]:
    out: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0:
            raise ValueError(f"offset must be non-negative: {value}")
        if value not in out:
            out.append(value)
    return out


def read_paths(path: Path) -> list[Path]:
    return [Path(line.strip()).resolve() for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
    os.replace(tmp, path)


def first_true(values: np.ndarray) -> int:
    idx = np.flatnonzero(np.asarray(values).reshape(-1).astype(bool))
    return int(idx[0]) if idx.size else -1


def scenario_from_path(path: Path) -> str:
    match = re.match(r"^(?P<scenario>.+)_seed\d+", path.name)
    if match:
        return str(match.group("scenario"))
    return path.stem


def read_h5_row(path: Path, args: argparse.Namespace, offsets: list[int]) -> tuple[list[dict[str, Any]], str | None]:
    import h5py

    rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as h5:
        traj_names = sorted(key for key in h5.keys() if key.startswith("traj_"))
        if len(traj_names) != 1:
            return rows, f"expected_one_traj_got_{len(traj_names)}"
        group = h5[traj_names[0]]
        if "actions" not in group or "slots" not in group:
            return rows, "missing_actions_or_slots"
        slots = group["slots"]
        required = ("peg_head_at_hole", "inserted", "grasped")
        missing = [name for name in required if name not in slots]
        if missing:
            return rows, "missing_slots_" + "_".join(missing)
        actions = np.asarray(group["actions"], dtype=np.float32)
        peg_head = np.asarray(slots["peg_head_at_hole"], dtype=np.float32)
        inserted = np.asarray(slots["inserted"], dtype=bool).reshape(-1)
        grasped = np.asarray(slots["grasped"], dtype=bool).reshape(-1)
        if actions.shape != (int(args.expected_actions), 7):
            return rows, f"bad_actions_shape_{tuple(actions.shape)}"
        if int(peg_head.shape[0]) != int(args.expected_frames):
            return rows, f"bad_peg_head_frames_{peg_head.shape[0]}"
        if int(inserted.shape[0]) != int(args.expected_frames):
            return rows, f"bad_inserted_frames_{inserted.shape[0]}"
        if int(grasped.shape[0]) != int(args.expected_frames):
            return rows, f"bad_grasped_frames_{grasped.shape[0]}"
        first_insert = first_true(inserted)
        if first_insert < 0:
            return rows, "never_inserted"
        if bool(args.require_success_at_end) and not bool(inserted[-1]):
            return rows, "not_inserted_at_end"
        scenario = scenario_from_path(path)
        source_uuid = path.stem
        for offset in offsets:
            start = max(0, int(first_insert) - int(offset))
            if start >= int(actions.shape[0]):
                continue
            if bool(args.require_grasped_at_start) and not bool(grasped[start]):
                continue
            valid = min(int(args.horizon), int(actions.shape[0]) - int(start))
            if valid <= 0:
                continue
            chunk = np.asarray(actions[start : start + valid], dtype=np.float32)
            if valid < int(args.horizon):
                pad = np.repeat(chunk[-1:, :], int(args.horizon) - valid, axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            start_rel = np.asarray(peg_head[start, :3], dtype=np.float32)
            end_frame = min(int(start) + int(valid), int(peg_head.shape[0]) - 1)
            end_rel = np.asarray(peg_head[end_frame, :3], dtype=np.float32)
            start_abs_yz = float(abs(start_rel[1]) + abs(start_rel[2]))
            if start_abs_yz > float(args.max_start_abs_yz_sum):
                continue
            rows.append(
                {
                    "source_h5": str(path),
                    "source_uuid": source_uuid,
                    "scenario": scenario,
                    "trajectory": traj_names[0],
                    "first_insert_frame": int(first_insert),
                    "start_frame": int(start),
                    "offset_before_insert": int(offset),
                    "valid_steps": int(valid),
                    "end_frame": int(end_frame),
                    "start_peg_head_at_hole": start_rel.astype(float).tolist(),
                    "end_peg_head_at_hole": end_rel.astype(float).tolist(),
                    "start_abs_yz_sum": start_abs_yz,
                    "inserted_within_chunk": bool(np.asarray(inserted[start : end_frame + 1]).any()),
                    "grasped_at_start": bool(grasped[start]),
                    "actions": chunk.astype(np.float32),
                }
            )
    return rows, None


def summarize_axis(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    offsets = parse_ints(str(args.start_offsets_before_insert))
    paths = read_paths(Path(args.h5_paths_file).resolve())
    scenario_re = re.compile(str(args.scenario_regex)) if str(args.scenario_regex).strip() else None

    rows: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    seen = 0
    for path in paths:
        if scenario_re is not None and not scenario_re.search(scenario_from_path(path)):
            continue
        if int(args.max_rows) > 0 and seen >= int(args.max_rows):
            break
        seen += 1
        try:
            path_rows, failure = read_h5_row(path, args, offsets)
            if failure is not None:
                failures[failure] += 1
            rows.extend(path_rows)
        except Exception as exc:
            failures[type(exc).__name__] += 1

    if not rows:
        raise RuntimeError("no insertion suffix rows were built")

    actions = np.stack([np.asarray(row.pop("actions"), dtype=np.float32) for row in rows]).astype(np.float32)
    start_rel = np.asarray([row["start_peg_head_at_hole"] for row in rows], dtype=np.float32)
    end_rel = np.asarray([row["end_peg_head_at_hole"] for row in rows], dtype=np.float32)
    bank_npz = output_root / "source_insertion_suffix_bank.npz"
    np.savez_compressed(
        bank_npz,
        schema=np.asarray(["cosmos3_source_insertion_suffix_bank_v1"]),
        actions=actions,
        source_h5=np.asarray([row["source_h5"] for row in rows], dtype="<U512"),
        source_uuid=np.asarray([row["source_uuid"] for row in rows], dtype="<U256"),
        scenario=np.asarray([row["scenario"] for row in rows], dtype="<U128"),
        first_insert_frame=np.asarray([row["first_insert_frame"] for row in rows], dtype=np.int32),
        start_frame=np.asarray([row["start_frame"] for row in rows], dtype=np.int32),
        offset_before_insert=np.asarray([row["offset_before_insert"] for row in rows], dtype=np.int32),
        valid_steps=np.asarray([row["valid_steps"] for row in rows], dtype=np.int32),
        end_frame=np.asarray([row["end_frame"] for row in rows], dtype=np.int32),
        start_peg_head_at_hole=start_rel,
        end_peg_head_at_hole=end_rel,
        inserted_within_chunk=np.asarray([row["inserted_within_chunk"] for row in rows], dtype=bool),
        grasped_at_start=np.asarray([row["grasped_at_start"] for row in rows], dtype=bool),
        horizon=np.asarray([int(args.horizon)], dtype=np.int32),
        robot_action_dim=np.asarray([7], dtype=np.int32),
    )
    write_jsonl(output_root / "source_insertion_suffix_bank.jsonl", rows)

    scenario_counts = Counter(str(row["scenario"]) for row in rows)
    offset_counts = Counter(int(row["offset_before_insert"]) for row in rows)
    summary = {
        "schema": "cosmos3_source_insertion_suffix_bank_summary_v1",
        "h5_paths_file": str(Path(args.h5_paths_file).resolve()),
        "output_root": str(output_root),
        "bank_npz": str(bank_npz),
        "source_h5_seen": int(seen),
        "suffix_rows": int(len(rows)),
        "horizon": int(args.horizon),
        "offsets_before_insert": offsets,
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "offset_counts": {str(k): int(v) for k, v in sorted(offset_counts.items())},
        "failure_counts": dict(sorted(failures.items())),
        "start_rel_stats": {
            "x": summarize_axis(start_rel[:, 0]),
            "y": summarize_axis(start_rel[:, 1]),
            "z": summarize_axis(start_rel[:, 2]),
            "abs_yz_sum": summarize_axis(np.abs(start_rel[:, 1]) + np.abs(start_rel[:, 2])),
        },
        "end_rel_stats": {
            "x": summarize_axis(end_rel[:, 0]),
            "y": summarize_axis(end_rel[:, 1]),
            "z": summarize_axis(end_rel[:, 2]),
            "abs_yz_sum": summarize_axis(np.abs(end_rel[:, 1]) + np.abs(end_rel[:, 2])),
        },
        "boundary": (
            "This bank mines successful source insertion suffixes for candidate-source "
            "diagnostics. It is not closed-loop method evidence and must be replayed "
            "from real live snapshots before any controller use."
        ),
    }
    write_json(output_root / "source_insertion_suffix_bank_summary.json", summary)
    md = [
        "# Source Insertion Suffix Bank",
        "",
        f"source_h5_seen={seen}",
        f"suffix_rows={len(rows)}",
        f"horizon={int(args.horizon)}",
        "",
        "Boundary: diagnostic candidate-source artifact, not method evidence.",
    ]
    (output_root / "source_insertion_suffix_bank_summary.md").write_text("\n".join(md) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
