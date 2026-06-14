#!/usr/bin/env python3
"""Audit whether live closed-loop Cosmos queries are covered by SFT rows.

This is a read-only diagnostic. It compares the real live-receding query
states recorded in closed-loop summaries against the condition rows used for
Cosmos3 SFT. It does not run simulation, rendering, inference, or training.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PREFIX_TOLERANCE = 16
DEFAULT_REL_L2_TOLERANCE = 0.05
DEFAULT_REL_YZ_TOLERANCE = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument(
        "--live-summary",
        type=Path,
        action="append",
        required=True,
        help=(
            "A live_receding_panel_summary.json or a single "
            "live_receding_loop_summary.json. Repeatable."
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--prefix-tolerance", type=int, default=DEFAULT_PREFIX_TOLERANCE)
    parser.add_argument("--rel-l2-tolerance", type=float, default=DEFAULT_REL_L2_TOLERANCE)
    parser.add_argument("--rel-yz-tolerance", type=float, default=DEFAULT_REL_YZ_TOLERANCE)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def as_rel(value: Any) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    arr = np.asarray(value[:3], dtype=np.float32)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        return None
    return arr


def training_rows(condition_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "val"):
        for row in read_jsonl(condition_root / split / "video_action_dataset_file.jsonl"):
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            payload = metadata.get("prefix_causal_state") if isinstance(metadata.get("prefix_causal_state"), dict) else {}
            rel = as_rel(payload.get("peg_head_at_hole_xyz"))
            prefix = int(row.get("prefix_frame_index", payload.get("prefix_frame_index", -1)))
            prefix_role = str(row.get("prefix_role") or payload.get("prefix_role") or "unknown")
            physical_mode = str(row.get("physical_mode") or payload.get("mode") or "unknown")
            rows.append(
                {
                    "uuid": row.get("uuid"),
                    "split": split,
                    "scenario": metadata.get("scenario"),
                    "prefix_frame_index": prefix,
                    "prefix_role": prefix_role,
                    "physical_mode": physical_mode,
                    "role_mode_match": prefix_role == physical_mode,
                    "rel": rel,
                    "rel_list": rel.astype(float).tolist() if rel is not None else None,
                    "target_motion_observed": bool(payload.get("target_motion_observed")),
                    "grasped": bool(payload.get("grasped")),
                    "inserted": bool(payload.get("inserted")),
                    "condition_action_len": (
                        len(row.get("condition_frame_indexes_action"))
                        if isinstance(row.get("condition_frame_indexes_action"), list)
                        else None
                    ),
                }
            )
    return rows


def load_loop_summary(path: Path, sample_hint: dict[str, Any] | None = None) -> dict[str, Any]:
    if path.is_file():
        return read_json(path)
    if sample_hint and isinstance(sample_hint.get("summary_path"), str):
        summary_path = Path(sample_hint["summary_path"])
        if summary_path.is_file():
            return read_json(summary_path)
    raise FileNotFoundError(path)


def iter_live_summaries(paths: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        data = read_json(path)
        if isinstance(data, dict) and isinstance(data.get("iterations"), list):
            out.append((path, data))
            continue
        if isinstance(data, dict) and isinstance(data.get("samples"), list):
            for sample in data["samples"]:
                summary_path_text = sample.get("summary_path")
                if isinstance(summary_path_text, str) and Path(summary_path_text).is_file():
                    summary_path = Path(summary_path_text)
                else:
                    scenario = str(sample.get("scenario") or "")
                    sample_index = sample.get("sample_index")
                    guessed = path.parent / f"sample_{int(sample_index):02d}_{scenario}" / "live_receding_loop_summary.json"
                    summary_path = guessed
                out.append((summary_path, load_loop_summary(summary_path, sample)))
            continue
        raise ValueError(f"{path} is neither a loop summary nor a panel summary")
    return out


def query_rows(summary_path: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    final_eval = summary.get("final_eval") if isinstance(summary.get("final_eval"), dict) else {}
    prefix_selection = summary.get("prefix_selection") if isinstance(summary.get("prefix_selection"), dict) else {}
    for iteration in summary.get("iterations") or []:
        if not isinstance(iteration, dict):
            continue
        if iteration.get("controller_step_type") != "cosmos_rebind_short_chunk":
            continue
        pre_gate = (
            iteration.get("pre_controller_continuability_gate")
            if isinstance(iteration.get("pre_controller_continuability_gate"), dict)
            else {}
        )
        rel = as_rel(pre_gate.get("peg_head_at_hole"))
        if rel is None:
            before_eval = iteration.get("before_eval") if isinstance(iteration.get("before_eval"), dict) else {}
            rel = as_rel(before_eval.get("peg_head_pos_at_hole"))
        if rel is None:
            continue
        role_info = iteration.get("prefix_role_info") if isinstance(iteration.get("prefix_role_info"), dict) else {}
        rows.append(
            {
                "summary_path": str(summary_path),
                "sample_name": summary.get("sample_name"),
                "scenario": summary.get("scenario"),
                "sample_final_success": bool(final_eval.get("success", False)),
                "target_motion_trigger_frame": prefix_selection.get("detected_frame_index"),
                "iteration": int(iteration.get("iteration", -1)),
                "prefix_frame_index": int(iteration.get("prefix_frame_index", -1)),
                "prefix_role": str(iteration.get("prefix_role") or role_info.get("role") or "unknown"),
                "prefix_role_source": role_info.get("source"),
                "role_target_delta": role_info.get("target_delta"),
                "role_hole_speed": role_info.get("hole_speed"),
                "pre_gate_ok": bool(pre_gate.get("ok", False)),
                "pre_gate_checks": pre_gate.get("checks"),
                "rel": rel,
                "rel_list": rel.astype(float).tolist(),
                "chunk_start": iteration.get("chunk_start"),
                "chunk_end_exclusive": iteration.get("chunk_end_exclusive"),
                "chunk_steps": iteration.get("chunk_steps"),
                "after_eval": iteration.get("after_eval"),
                "post_cosmos_continuability_gate": iteration.get("post_cosmos_continuability_gate"),
            }
        )
    return rows


def nearest(query: dict[str, Any], candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    qrel = query["rel"]
    qprefix = int(query["prefix_frame_index"])
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in candidates:
        rel = row.get("rel")
        if rel is None:
            continue
        prefix_dist = abs(int(row["prefix_frame_index"]) - qprefix)
        rel_delta = np.asarray(rel, dtype=np.float32) - qrel
        rel_l2 = float(np.linalg.norm(rel_delta))
        yz_l2 = float(np.linalg.norm(rel_delta[1:3]))
        score = rel_l2 + (prefix_dist / 100.0)
        scored.append((score, {**row, "rel_l2": rel_l2, "rel_yz_l2": yz_l2, "prefix_abs_delta": prefix_dist}))
    scored.sort(key=lambda item: item[0])
    out: list[dict[str, Any]] = []
    for _, row in scored[: max(1, int(top_k))]:
        cleaned = dict(row)
        cleaned.pop("rel", None)
        out.append(cleaned)
    return out


def close_count(
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    prefix_tolerance: int,
    rel_l2_tolerance: float,
    rel_yz_tolerance: float,
) -> int:
    count = 0
    qrel = query["rel"]
    qprefix = int(query["prefix_frame_index"])
    for row in candidates:
        rel = row.get("rel")
        if rel is None:
            continue
        if abs(int(row["prefix_frame_index"]) - qprefix) > int(prefix_tolerance):
            continue
        diff = np.asarray(rel, dtype=np.float32) - qrel
        if float(np.linalg.norm(diff)) <= float(rel_l2_tolerance) and float(np.linalg.norm(diff[1:3])) <= float(rel_yz_tolerance):
            count += 1
    return count


def audit(args: argparse.Namespace) -> dict[str, Any]:
    condition_root = args.condition_root.resolve()
    train = training_rows(condition_root)
    live: list[dict[str, Any]] = []
    for summary_path, summary in iter_live_summaries([path.resolve() for path in args.live_summary]):
        live.extend(query_rows(summary_path, summary))

    role_counts = Counter(row["prefix_role"] for row in train)
    physical_counts = Counter(row["physical_mode"] for row in train)
    role_mode_counts = Counter(f"{row['prefix_role']}|{row['physical_mode']}" for row in train)
    role_mode_mismatch_count = sum(1 for row in train if not row["role_mode_match"])
    live_role_counts = Counter(row["prefix_role"] for row in live)

    query_reports: list[dict[str, Any]] = []
    undercovered = 0
    for query in live:
        role = query["prefix_role"]
        same_role = [row for row in train if row["prefix_role"] == role]
        same_physical = [row for row in train if row["physical_mode"] == role]
        consistent = [row for row in train if row["prefix_role"] == role and row["physical_mode"] == role]
        close_consistent = close_count(
            query,
            consistent,
            prefix_tolerance=int(args.prefix_tolerance),
            rel_l2_tolerance=float(args.rel_l2_tolerance),
            rel_yz_tolerance=float(args.rel_yz_tolerance),
        )
        nearest_consistent = nearest(query, consistent, int(args.top_k))
        nearest_same_role = nearest(query, same_role, int(args.top_k))
        nearest_all = nearest(query, train, int(args.top_k))
        nearest_rel_l2 = nearest_consistent[0]["rel_l2"] if nearest_consistent else None
        undercovered_query = close_consistent == 0
        if undercovered_query:
            undercovered += 1
        clean = dict(query)
        clean.pop("rel", None)
        query_reports.append(
            {
                **clean,
                "same_prefix_role_train_count": len(same_role),
                "same_physical_mode_train_count": len(same_physical),
                "role_mode_consistent_train_count": len(consistent),
                "close_role_mode_consistent_count": close_consistent,
                "undercovered_by_strict_local_criterion": undercovered_query,
                "nearest_role_mode_consistent_rel_l2": nearest_rel_l2,
                "nearest_role_mode_consistent": nearest_consistent,
                "nearest_same_prefix_role": nearest_same_role,
                "nearest_any_training_row": nearest_all,
            }
        )

    return {
        "boundary": (
            "Read-only coverage audit. It compares recorded live Cosmos query "
            "states against SFT condition rows; it does not judge task success "
            "and does not execute or train anything."
        ),
        "condition_root": str(condition_root),
        "live_summary_paths": [str(path.resolve()) for path in args.live_summary],
        "train_row_count": len(train),
        "live_cosmos_query_count": len(live),
        "train_prefix_role_counts": dict(sorted(role_counts.items())),
        "train_physical_mode_counts": dict(sorted(physical_counts.items())),
        "train_role_mode_counts": dict(sorted(role_mode_counts.items())),
        "train_role_mode_mismatch_count": int(role_mode_mismatch_count),
        "train_role_mode_mismatch_fraction": float(role_mode_mismatch_count / max(len(train), 1)),
        "live_query_role_counts": dict(sorted(live_role_counts.items())),
        "strict_local_coverage": {
            "prefix_tolerance": int(args.prefix_tolerance),
            "rel_l2_tolerance": float(args.rel_l2_tolerance),
            "rel_yz_tolerance": float(args.rel_yz_tolerance),
            "undercovered_query_count": int(undercovered),
            "undercovered_query_fraction": float(undercovered / max(len(live), 1)),
        },
        "queries": query_reports,
        "interpretation": {
            "role_mode_consistent": (
                "Rows whose text prefix_role and physical_mode both match the "
                "live inferred role. These are the cleanest SFT neighbors for "
                "the corrected live-receding controller."
            ),
            "undercovered_by_strict_local_criterion": (
                "No role/mode-consistent training row within the configured "
                "prefix and peg-head-at-hole local tolerances. This is not a "
                "hard method blocker by itself, but it identifies live queries "
                "that the old SFT distribution did not directly cover."
            ),
        },
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    cov = report["strict_local_coverage"]
    lines = [
        "# Cosmos3 Live Query Training Coverage Audit",
        "",
        f"- train rows: `{report['train_row_count']}`",
        f"- live Cosmos queries: `{report['live_cosmos_query_count']}`",
        f"- train role/mode mismatch: `{report['train_role_mode_mismatch_count']}` "
        f"(`{report['train_role_mode_mismatch_fraction']:.4f}`)",
        f"- undercovered live queries: `{cov['undercovered_query_count']}` "
        f"(`{cov['undercovered_query_fraction']:.4f}`)",
        f"- prefix tolerance: `{cov['prefix_tolerance']}`",
        f"- rel L2 tolerance: `{cov['rel_l2_tolerance']}`",
        f"- rel yz tolerance: `{cov['rel_yz_tolerance']}`",
        "",
        "## Live Query Roles",
        "",
        *(f"- `{key}`: `{value}`" for key, value in report["live_query_role_counts"].items()),
        "",
        "## Training Prefix Roles",
        "",
        *(f"- `{key}`: `{value}`" for key, value in report["train_prefix_role_counts"].items()),
        "",
        "## Undercovered Queries",
        "",
    ]
    for query in report["queries"]:
        if not query["undercovered_by_strict_local_criterion"]:
            continue
        lines.append(
            "- "
            f"`{query['scenario']}` iter `{query['iteration']}` "
            f"frame `{query['prefix_frame_index']}` role `{query['prefix_role']}` "
            f"rel `{query['rel_list']}` nearest_consistent_l2 "
            f"`{query['nearest_role_mode_consistent_rel_l2']}`"
        )
    if not any(query["undercovered_by_strict_local_criterion"] for query in report["queries"]):
        lines.append("- none")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    report = audit(args)
    write_json(args.output_json.resolve(), report)
    if args.output_md:
        write_markdown(args.output_md.resolve(), report)
    print(
        json.dumps(
            {
                "train_row_count": report["train_row_count"],
                "live_cosmos_query_count": report["live_cosmos_query_count"],
                "undercovered_query_count": report["strict_local_coverage"]["undercovered_query_count"],
                "train_role_mode_mismatch_count": report["train_role_mode_mismatch_count"],
                "output_json": str(args.output_json.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
