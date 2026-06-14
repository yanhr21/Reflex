#!/usr/bin/env python3
"""Audit whether Cosmos3 SFT rows match live receding-controller queries.

This is a read-only diagnostic. It does not render, train, or rewrite the
condition root. The goal is to make the closed-loop failure analysis
reproducible: role/mode drift, sparse prefix masks, and weak coverage of
late receding recovery chunks should be visible before launching another SFT.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--chunk-horizon", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument(
        "--prefix-role-source",
        choices=("sampled", "physical_mode", "unknown"),
        default="unknown",
        help="If physical_mode, nonzero role/mode mismatch is a hard audit failure.",
    )
    parser.add_argument("--require-no-condition-mask-errors", action="store_true")
    parser.add_argument("--min-late-rebind-candidates", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def percentile_stats(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "min": None, "p25": None, "p50": None, "p75": None, "p90": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def load_stats(condition_root: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    path = condition_root / "normalization_stats.json"
    if not path.exists():
        return None, None
    stats = json.loads(path.read_text())
    if stats.get("type") != "zscore":
        return None, None
    mean = np.asarray(stats.get("mean", []), dtype=np.float32)
    std = np.asarray(stats.get("std", []), dtype=np.float32)
    if mean.shape != std.shape or mean.size == 0:
        return None, None
    std = np.where(np.abs(std) < 1e-8, 1.0, std)
    return mean, std


def denormalize_action(action: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    if mean is None or std is None:
        return action
    if action.shape[-1] > mean.size:
        raise ValueError(f"action dim {action.shape[-1]} exceeds stats dim {mean.size}")
    return action * std[: action.shape[-1]] + mean[: action.shape[-1]]


def group_stats(items: list[dict[str, Any]], key: str, value: str) -> dict[str, dict[str, float | int | None]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for item in items:
        groups[str(item[key])].append(float(item[value]))
    return {name: percentile_stats(vals) for name, vals in sorted(groups.items())}


def main() -> None:
    args = parse_args()
    root = args.condition_root.resolve()
    rows: list[dict[str, Any]] = []
    for split in ("train", "val"):
        for row in read_jsonl(root / split / "video_action_dataset_file.jsonl"):
            row["_split"] = split
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no train/val JSONL rows found under {root}")

    mean, std = load_stats(root)
    role_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    role_mode_counts: Counter[str] = Counter()
    prefix_by_role: dict[str, list[int]] = defaultdict(list)
    prefix_by_mode: dict[str, list[int]] = defaultdict(list)
    mismatch_examples: list[dict[str, Any]] = []
    condition_mask_errors: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    late_rebind_count_by_role: Counter[str] = Counter()

    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        payload = metadata.get("prefix_causal_state") if isinstance(metadata.get("prefix_causal_state"), dict) else {}
        role = str(row.get("prefix_role") or payload.get("prefix_role") or "unknown")
        mode = str(payload.get("mode") or "unknown")
        split = str(row.get("_split", "unknown"))
        scenario = str(metadata.get("scenario", "unknown"))
        prefix = int(row.get("prefix_frame_index", payload.get("prefix_frame_index", -1)))
        role_counts[role] += 1
        mode_counts[mode] += 1
        split_counts[split] += 1
        scenario_counts[scenario] += 1
        role_mode_counts[f"{role}|{mode}"] += 1
        prefix_by_role[role].append(prefix)
        prefix_by_mode[mode].append(prefix)

        if role != mode and len(mismatch_examples) < int(args.top_k):
            mismatch_examples.append(
                {
                    "uuid": row.get("uuid"),
                    "split": split,
                    "scenario": scenario,
                    "prefix_frame_index": prefix,
                    "prefix_role": role,
                    "mode": mode,
                    "peg_head_at_hole_xyz": payload.get("peg_head_at_hole_xyz"),
                    "target_motion_observed": payload.get("target_motion_observed"),
                }
            )

        expected_condition = list(range(max(prefix, 0)))
        got_condition = row.get("condition_frame_indexes_action", [])
        if got_condition != expected_condition and len(condition_mask_errors) < int(args.top_k):
            condition_mask_errors.append(
                {
                    "uuid": row.get("uuid"),
                    "prefix_frame_index": prefix,
                    "expected_len": len(expected_condition),
                    "got_len": len(got_condition) if isinstance(got_condition, list) else None,
                }
            )

        action_path = Path(row["action_path"])
        action = np.load(action_path)
        raw = denormalize_action(np.asarray(action, dtype=np.float32), mean, std)
        start = max(0, min(prefix, raw.shape[0]))
        end = max(start, min(start + int(args.chunk_horizon), raw.shape[0]))
        chunk = raw[start:end, :7]
        mean_abs_xyz = np.mean(np.abs(chunk[:, :3]), axis=0) if chunk.size else np.zeros(3, dtype=np.float32)
        mean_abs_7d = float(np.mean(np.abs(chunk))) if chunk.size else 0.0
        rel = np.asarray(payload.get("peg_head_at_hole_xyz", [0.0, 0.0, 0.0]), dtype=np.float32)[:3]
        late_rebind_candidate = bool(
            payload.get("target_motion_observed")
            and payload.get("grasped")
            and not payload.get("inserted")
            and (abs(float(rel[0])) > 0.05 or abs(float(rel[1])) > 0.01 or abs(float(rel[2])) > 0.004)
        )
        if late_rebind_candidate:
            late_rebind_count_by_role[role] += 1
        chunk_rows.append(
            {
                "uuid": row.get("uuid"),
                "split": split,
                "scenario": scenario,
                "prefix_role": role,
                "mode": mode,
                "prefix_frame_index": prefix,
                "future_chunk_len": int(end - start),
                "teacher_mean_abs_action_xyz": [float(x) for x in mean_abs_xyz.tolist()],
                "teacher_mean_abs_action_7d": mean_abs_7d,
                "abs_rel_x": abs(float(rel[0])),
                "abs_rel_y": abs(float(rel[1])),
                "abs_rel_z": abs(float(rel[2])),
                "late_rebind_candidate": late_rebind_candidate,
            }
        )

    mismatch_count = sum(count for key, count in role_mode_counts.items() if key.split("|", 1)[0] != key.split("|", 1)[1])
    top_teacher_chunks = sorted(chunk_rows, key=lambda item: item["teacher_mean_abs_action_7d"], reverse=True)[
        : int(args.top_k)
    ]

    hard_failures: list[str] = []
    if args.prefix_role_source == "physical_mode" and int(mismatch_count) != 0:
        hard_failures.append(f"role_mode_mismatch_count={int(mismatch_count)} under physical_mode export")
    if args.require_no_condition_mask_errors and condition_mask_errors:
        hard_failures.append(f"condition_mask_error_examples={len(condition_mask_errors)}")
    late_rebind_total = int(sum(late_rebind_count_by_role.values()))
    if late_rebind_total < int(args.min_late_rebind_candidates):
        hard_failures.append(
            f"late_rebind_candidate_total={late_rebind_total} < min_late_rebind_candidates={args.min_late_rebind_candidates}"
        )

    report = {
        "condition_root": str(root),
        "prefix_role_source": args.prefix_role_source,
        "num_rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "role_mode_counts": dict(sorted(role_mode_counts.items())),
        "role_mode_mismatch_count": int(mismatch_count),
        "role_mode_mismatch_fraction": float(mismatch_count / max(len(rows), 1)),
        "mismatch_examples": mismatch_examples,
        "prefix_stats_by_role": {k: percentile_stats(v) for k, v in sorted(prefix_by_role.items())},
        "prefix_stats_by_mode": {k: percentile_stats(v) for k, v in sorted(prefix_by_mode.items())},
        "future_chunk_teacher_abs_action_7d_by_role": group_stats(chunk_rows, "prefix_role", "teacher_mean_abs_action_7d"),
        "future_chunk_teacher_abs_action_7d_by_mode": group_stats(chunk_rows, "mode", "teacher_mean_abs_action_7d"),
        "late_rebind_candidate_count_by_role": dict(sorted(late_rebind_count_by_role.items())),
        "late_rebind_candidate_total": late_rebind_total,
        "condition_mask_error_examples": condition_mask_errors,
        "hard_failures": hard_failures,
        "strict_ok": not hard_failures,
        "top_teacher_action_chunks": top_teacher_chunks,
        "interpretation": {
            "role_mode_mismatch": (
                "High mismatch means conditioning text uses an intended sampled role that often differs "
                "from the current physical mode seen by the live receding controller."
            ),
            "late_rebind_candidate": (
                "Rows where target motion has been observed, peg is grasped, insertion is not done, "
                "and peg-head-to-hole error is outside a small local band. These rows are the closest "
                "existing source-distribution proxy for the failed live recovery state."
            ),
            "condition_mask_errors": (
                "For joint-policy history-action rows, conditioned action indexes should be exactly "
                "range(prefix_frame_index); any examples here need code review before SFT."
            ),
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    print(text)
    if hard_failures:
        raise SystemExit("receding training distribution audit failed: " + "; ".join(hard_failures))


if __name__ == "__main__":
    main()
