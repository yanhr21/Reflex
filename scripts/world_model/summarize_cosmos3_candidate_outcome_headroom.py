#!/usr/bin/env python3
"""Summarize real candidate-outcome headroom by state, phase, and scenario."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_cosmos3_live_receding_loop import require_compute_step  # noqa: E402
from train_cosmos3_contact_executor import split_indices  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome-jsonl", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--meaningful-improvement", type=float, default=0.01)
    parser.add_argument("--large-improvement", type=float, default=0.03)
    parser.add_argument("--max-hard-examples", type=int, default=25)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260617)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_jsonl(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for text in paths:
        path = Path(text).resolve()
        with path.open("r") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("schema") != "cosmos3_candidate_outcome_label_v1":
                    continue
                copied = dict(row)
                copied["_outcome_jsonl"] = str(path)
                copied["_line_no"] = int(line_no)
                rows.append(copied)
    if not rows:
        raise RuntimeError("no candidate outcome rows found")
    return rows


def is_dp_prior_candidate(name: str) -> bool:
    return name in {"dp_prior", "model_dp_prior"}


def candidate_family(name: str, source: str) -> str:
    if is_dp_prior_candidate(name):
        return "dp_prior"
    if name == "teacher" or name.startswith("scale_"):
        return "teacher_scale"
    if name == "model_mean":
        return "model_mean"
    if name.startswith("model_scale_"):
        return "model_scale"
    if name.startswith("model_diffusion_"):
        return "model_diffusion"
    if name.startswith("retrieval_resid_") or source == "retrieval_success_residual":
        return "retrieval_success_residual"
    return source or "other"


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def peg_vec(row: dict[str, Any], key: str) -> np.ndarray:
    value = row.get(key)
    if value is None and key == "start_peg_head_at_hole":
        value = (row.get("start_eval") or {}).get("peg_head_pos_at_hole")
    if value is None and key == "final_peg_head_at_hole":
        value = row.get("final_peg_head_at_hole")
    arr = np.asarray(value or [np.nan, np.nan, np.nan], dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)
    return arr[:3]


def start_error_bin(start_vec: np.ndarray) -> str:
    if not np.all(np.isfinite(start_vec)):
        return "unknown"
    l2 = float(np.linalg.norm(start_vec))
    if l2 < 0.025:
        return "near_<2.5cm"
    if l2 < 0.075:
        return "mid_2.5_7.5cm"
    if l2 < 0.15:
        return "far_7.5_15cm"
    return "very_far_>=15cm"


def summarize_group(rows: list[dict[str, Any]], meaningful: float, large: float) -> dict[str, Any] | None:
    dp_rows = [row for row in rows if is_dp_prior_candidate(str(row.get("candidate_name") or ""))]
    if not dp_rows:
        return None
    dp = min(dp_rows, key=lambda row: safe_float(row.get("final_abs_task_error_weighted")))
    oracle = min(rows, key=lambda row: safe_float(row.get("final_abs_task_error_weighted")))
    dp_error = safe_float(dp.get("final_abs_task_error_weighted"))
    oracle_error = safe_float(oracle.get("final_abs_task_error_weighted"))
    delta = oracle_error - dp_error
    success_rows = [row for row in rows if bool(row.get("final_success", False))]
    dp_rollout_success_rows = [
        row
        for row in rows
        if bool(row.get("dp_rollout_success", False)) or bool(row.get("final_success", False))
    ]
    dp_rollout_continuable_rows = [
        row
        for row in rows
        if bool(row.get("dp_rollout_continuable_proxy", False))
        or bool(row.get("final_continuable_proxy", False))
        or bool(row.get("final_success", False))
    ]
    start_vec = peg_vec(dp, "start_peg_head_at_hole")
    out = {
        "uuid": str(dp.get("uuid") or ""),
        "scenario": str(dp.get("scenario") or ""),
        "current_phase": str(dp.get("current_phase") or ""),
        "prefix_role": str(dp.get("prefix_role") or ""),
        "prefix_frame_index": dp.get("prefix_frame_index"),
        "start_peg_head_at_hole": start_vec.astype(float).tolist(),
        "start_error_l2": float(np.linalg.norm(start_vec)) if np.all(np.isfinite(start_vec)) else None,
        "start_error_bin": start_error_bin(start_vec),
        "dp_candidate_name": str(dp.get("candidate_name") or ""),
        "dp_error": dp_error,
        "dp_success": bool(dp.get("final_success", False)),
        "dp_handoff_success": bool(dp.get("dp_rollout_success", False) or dp.get("final_success", False)),
        "dp_handoff_continuable": bool(
            dp.get("dp_rollout_continuable_proxy", False)
            or dp.get("final_continuable_proxy", False)
            or dp.get("final_success", False)
        ),
        "oracle_candidate_name": str(oracle.get("candidate_name") or ""),
        "oracle_candidate_family": candidate_family(
            str(oracle.get("candidate_name") or ""),
            str(oracle.get("candidate_source") or ""),
        ),
        "oracle_error": oracle_error,
        "oracle_success": bool(oracle.get("final_success", False)),
        "oracle_minus_dp_error": delta,
        "meaningful_improvement": bool(delta <= -float(meaningful)),
        "large_improvement": bool(delta <= -float(large)),
        "num_candidates": int(len(rows)),
        "num_success_candidates": int(len(success_rows)),
        "num_handoff_success_candidates": int(len(dp_rollout_success_rows)),
        "num_handoff_continuable_candidates": int(len(dp_rollout_continuable_rows)),
        "handoff_oracle_success": bool(dp_rollout_success_rows),
        "handoff_oracle_continuable": bool(dp_rollout_continuable_rows),
        "handoff_oracle_candidate_name": (
            str(min(dp_rollout_success_rows, key=lambda row: safe_float(row.get("final_abs_task_error_weighted"))).get("candidate_name") or "")
            if dp_rollout_success_rows
            else None
        ),
        "handoff_oracle_candidate_family": (
            candidate_family(
                str(min(dp_rollout_success_rows, key=lambda row: safe_float(row.get("final_abs_task_error_weighted"))).get("candidate_name") or ""),
                str(min(dp_rollout_success_rows, key=lambda row: safe_float(row.get("final_abs_task_error_weighted"))).get("candidate_source") or ""),
            )
            if dp_rollout_success_rows
            else None
        ),
        "success_candidate_families": dict(
            sorted(
                Counter(
                    candidate_family(str(row.get("candidate_name") or ""), str(row.get("candidate_source") or ""))
                    for row in success_rows
                ).items()
            )
        ),
        "handoff_success_candidate_families": dict(
            sorted(
                Counter(
                    candidate_family(str(row.get("candidate_name") or ""), str(row.get("candidate_source") or ""))
                    for row in dp_rollout_success_rows
                ).items()
            )
        ),
    }
    return out


def aggregate_groups(groups: list[dict[str, Any]]) -> dict[str, Any]:
    if not groups:
        return {"num_groups": 0}
    dp_errors = np.asarray([item["dp_error"] for item in groups], dtype=np.float32)
    oracle_errors = np.asarray([item["oracle_error"] for item in groups], dtype=np.float32)
    deltas = np.asarray([item["oracle_minus_dp_error"] for item in groups], dtype=np.float32)
    success_candidate_family_counts: Counter[str] = Counter()
    success_candidate_family_group_counts: Counter[str] = Counter()
    handoff_success_candidate_family_counts: Counter[str] = Counter()
    handoff_success_candidate_family_group_counts: Counter[str] = Counter()
    for item in groups:
        for family, count in dict(item.get("success_candidate_families") or {}).items():
            success_candidate_family_counts[str(family)] += int(count)
            if int(count) > 0:
                success_candidate_family_group_counts[str(family)] += 1
        for family, count in dict(item.get("handoff_success_candidate_families") or {}).items():
            handoff_success_candidate_family_counts[str(family)] += int(count)
            if int(count) > 0:
                handoff_success_candidate_family_group_counts[str(family)] += 1
    return {
        "num_groups": int(len(groups)),
        "dp_success_count": int(sum(1 for item in groups if item["dp_success"])),
        "dp_handoff_success_count": int(sum(1 for item in groups if item["dp_handoff_success"])),
        "dp_handoff_continuable_count": int(sum(1 for item in groups if item["dp_handoff_continuable"])),
        "oracle_success_count": int(sum(1 for item in groups if item["oracle_success"])),
        "handoff_oracle_success_count": int(sum(1 for item in groups if item["handoff_oracle_success"])),
        "handoff_oracle_continuable_count": int(sum(1 for item in groups if item["handoff_oracle_continuable"])),
        "any_success_candidate_count": int(sum(1 for item in groups if item["num_success_candidates"] > 0)),
        "any_handoff_success_candidate_count": int(
            sum(1 for item in groups if item["num_handoff_success_candidates"] > 0)
        ),
        "any_handoff_continuable_candidate_count": int(
            sum(1 for item in groups if item["num_handoff_continuable_candidates"] > 0)
        ),
        "meaningful_improvement_count": int(sum(1 for item in groups if item["meaningful_improvement"])),
        "large_improvement_count": int(sum(1 for item in groups if item["large_improvement"])),
        "mean_dp_error": float(dp_errors.mean()),
        "mean_oracle_error": float(oracle_errors.mean()),
        "mean_oracle_minus_dp_error": float(deltas.mean()),
        "median_oracle_minus_dp_error": float(np.median(deltas)),
        "oracle_candidate_family_counts": dict(sorted(Counter(item["oracle_candidate_family"] for item in groups).items())),
        "oracle_candidate_counts": dict(sorted(Counter(item["oracle_candidate_name"] for item in groups).items())),
        "handoff_oracle_candidate_family_counts": dict(
            sorted(Counter(str(item.get("handoff_oracle_candidate_family") or "none") for item in groups).items())
        ),
        "handoff_oracle_candidate_counts": dict(
            sorted(Counter(str(item.get("handoff_oracle_candidate_name") or "none") for item in groups).items())
        ),
        "success_candidate_family_counts": dict(sorted(success_candidate_family_counts.items())),
        "success_candidate_family_group_counts": dict(sorted(success_candidate_family_group_counts.items())),
        "handoff_success_candidate_family_counts": dict(sorted(handoff_success_candidate_family_counts.items())),
        "handoff_success_candidate_family_group_counts": dict(
            sorted(handoff_success_candidate_family_group_counts.items())
        ),
    }


def grouped_breakdown(group_summaries: list[dict[str, Any]], key: str) -> dict[str, Any]:
    bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in group_summaries:
        bucket[str(item.get(key) or "unknown")].append(item)
    return {name: aggregate_groups(items) for name, items in sorted(bucket.items())}


def split_breakdown(group_summaries: list[dict[str, Any]], val_fraction: float, seed: int) -> dict[str, Any]:
    if val_fraction <= 0.0 or len(group_summaries) <= 1:
        return {}
    by_uuid = {str(item["uuid"]): item for item in group_summaries}
    uuids = sorted(by_uuid)
    train_idx, val_idx = split_indices(len(uuids), float(val_fraction), int(seed))
    train_set = {uuids[int(i)] for i in train_idx}
    val_set = {uuids[int(i)] for i in val_idx}
    return {
        "train": aggregate_groups([by_uuid[uuid] for uuid in uuids if uuid in train_set]),
        "val": aggregate_groups([by_uuid[uuid] for uuid in uuids if uuid in val_set]),
        "seed": int(seed),
        "val_fraction": float(val_fraction),
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    lines = [
        "# Candidate Outcome Headroom Summary",
        "",
        "This is a replay diagnostic, not live method evidence. It asks whether",
        "the current candidate set contains actions that beat the DP prior when",
        "executed in the simulator from the same state.",
        "",
        "## Overall",
        "",
        f"- groups with DP baseline: `{overall['num_groups']}`",
        f"- DP successes: `{overall['dp_success_count']}`",
        f"- DP handoff successes: `{overall['dp_handoff_success_count']}`",
        f"- oracle successes: `{overall['oracle_success_count']}`",
        f"- handoff-oracle successes: `{overall['handoff_oracle_success_count']}`",
        f"- any successful candidate: `{overall['any_success_candidate_count']}`",
        f"- any handoff-success candidate: `{overall['any_handoff_success_candidate_count']}`",
        f"- meaningful improvements: `{overall['meaningful_improvement_count']}`",
        f"- large improvements: `{overall['large_improvement_count']}`",
        f"- mean DP error: `{overall['mean_dp_error']:.6f}`",
        f"- mean oracle error: `{overall['mean_oracle_error']:.6f}`",
        f"- mean oracle-minus-DP: `{overall['mean_oracle_minus_dp_error']:.6f}`",
        f"- oracle candidate families: `{overall.get('oracle_candidate_family_counts', {})}`",
        f"- handoff-oracle candidate families: `{overall.get('handoff_oracle_candidate_family_counts', {})}`",
        f"- success candidate family groups: `{overall.get('success_candidate_family_group_counts', {})}`",
        f"- handoff-success candidate family groups: `{overall.get('handoff_success_candidate_family_group_counts', {})}`",
        "",
        "## Plain Readout",
        "",
    ]
    if overall["oracle_success_count"] <= overall["dp_success_count"]:
        lines.append("Oracle candidate success does not exceed DP success overall.")
    else:
        lines.append("Oracle candidates add successful chunks beyond DP on this replay.")
    if overall["mean_oracle_minus_dp_error"] < 0:
        lines.append("There is average headroom, but it must be selectable on held-out states.")
    else:
        lines.append("There is no average headroom over DP in this candidate set.")
    if summary.get("by_split"):
        lines.extend(["", "## By Train/Val Split", ""])
        split_info = summary["by_split"]
        for name in ["train", "val"]:
            item = split_info[name]
            lines.append(
                f"- `{name}`: groups `{item['num_groups']}`, DP success "
                f"`{item['dp_success_count']}`, DP handoff success "
                f"`{item.get('dp_handoff_success_count', 0)}`, oracle success "
                f"`{item['oracle_success_count']}`, handoff-oracle success "
                f"`{item.get('handoff_oracle_success_count', 0)}`, mean delta "
                f"`{item['mean_oracle_minus_dp_error']:.6f}`, oracle families "
                f"`{item.get('oracle_candidate_family_counts', {})}`"
            )
    lines.extend(["", "## By Phase", ""])
    for name, item in summary["by_current_phase"].items():
        lines.append(
            f"- `{name}`: groups `{item['num_groups']}`, DP success "
            f"`{item['dp_success_count']}`, DP handoff success "
            f"`{item.get('dp_handoff_success_count', 0)}`, oracle success "
            f"`{item['oracle_success_count']}`, handoff-oracle success "
            f"`{item.get('handoff_oracle_success_count', 0)}`, mean delta "
            f"`{item['mean_oracle_minus_dp_error']:.6f}`, oracle families "
            f"`{item.get('oracle_candidate_family_counts', {})}`"
        )
    lines.extend(["", "## By Scenario", ""])
    for name, item in summary["by_scenario"].items():
        lines.append(
            f"- `{name}`: groups `{item['num_groups']}`, DP success "
            f"`{item['dp_success_count']}`, DP handoff success "
            f"`{item.get('dp_handoff_success_count', 0)}`, oracle success "
            f"`{item['oracle_success_count']}`, handoff-oracle success "
            f"`{item.get('handoff_oracle_success_count', 0)}`, mean delta "
            f"`{item['mean_oracle_minus_dp_error']:.6f}`, oracle families "
            f"`{item.get('oracle_candidate_family_counts', {})}`"
        )
    lines.extend(["", "## Hard Examples", ""])
    for item in summary["hard_examples"]:
        lines.append(
            f"- `{item['uuid']}` phase `{item['current_phase']}` scenario "
            f"`{item['scenario']}`: DP `{item['dp_error']:.6f}`, oracle "
            f"`{item['oracle_error']:.6f}`, delta "
            f"`{item['oracle_minus_dp_error']:.6f}`, oracle "
            f"`{item['oracle_candidate_name']}`"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text("\n".join(lines).rstrip() + "\n")
    os.replace(tmp, path)


def main() -> int:
    args = parse_args()
    require_compute_step()
    rows = read_jsonl(list(args.outcome_jsonl))
    by_uuid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        uuid = str(row.get("uuid") or "")
        if uuid:
            by_uuid[uuid].append(row)
    group_summaries = [
        item
        for item in (
            summarize_group(rows_for_uuid, float(args.meaningful_improvement), float(args.large_improvement))
            for rows_for_uuid in by_uuid.values()
        )
        if item is not None
    ]
    hard_examples = sorted(
        [
            item
            for item in group_summaries
            if not item["oracle_success"] and not item["meaningful_improvement"]
        ],
        key=lambda item: (float(item["dp_error"]), float(item["oracle_error"])),
        reverse=True,
    )[: int(args.max_hard_examples)]
    summary = {
        "schema": "cosmos3_candidate_outcome_headroom_summary_v1",
        "outcome_jsonl": [str(Path(item).resolve()) for item in args.outcome_jsonl],
        "output_root": str(Path(args.output_root).resolve()),
        "num_candidate_rows": int(len(rows)),
        "num_uuid_groups": int(len(by_uuid)),
        "meaningful_improvement": float(args.meaningful_improvement),
        "large_improvement": float(args.large_improvement),
        "overall": aggregate_groups(group_summaries),
        "by_scenario": grouped_breakdown(group_summaries, "scenario"),
        "by_current_phase": grouped_breakdown(group_summaries, "current_phase"),
        "by_prefix_role": grouped_breakdown(group_summaries, "prefix_role"),
        "by_start_error_bin": grouped_breakdown(group_summaries, "start_error_bin"),
        "by_split": split_breakdown(group_summaries, float(args.val_fraction), int(args.seed)),
        "hard_examples": hard_examples,
        "boundary": (
            "Replay-only candidate headroom diagnostic. It does not change "
            "controller behavior and is not live method evidence."
        ),
    }
    output_root = Path(args.output_root).resolve()
    write_json(output_root / "candidate_outcome_headroom_summary.json", summary)
    write_markdown(output_root / "candidate_outcome_headroom_summary.md", summary)
    print(json.dumps(jsonable(summary["overall"]), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
