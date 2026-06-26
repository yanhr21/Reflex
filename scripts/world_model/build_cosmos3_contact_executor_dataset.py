#!/usr/bin/env python3
"""Join causal executor rows with DP priors and contact/progress labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executor-jsonl", required=True)
    parser.add_argument("--dp-prior-jsonl", required=True)
    parser.add_argument("--contact-label-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def index_by_uuid(rows: list[dict[str, Any]], *, name: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        uuid = str(row.get("uuid", ""))
        if not uuid:
            continue
        if uuid in out:
            duplicates.append(uuid)
            continue
        out[uuid] = row
    if duplicates:
        raise RuntimeError(f"{name} has duplicate uuids, first duplicates: {duplicates[:5]}")
    return out


def load_contact_labels(root: Path) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        path = root / split / "contact_progress_row_labels.jsonl"
        if path.is_file():
            rows.extend(read_jsonl(path))
    return index_by_uuid(rows, name="contact labels")


def main() -> int:
    args = parse_args()
    executor_jsonl = Path(args.executor_jsonl).resolve()
    dp_prior_jsonl = Path(args.dp_prior_jsonl).resolve()
    contact_label_root = Path(args.contact_label_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    executor_rows = read_jsonl(executor_jsonl)
    if int(args.max_rows) > 0:
        executor_rows = executor_rows[: int(args.max_rows)]
    dp_rows = index_by_uuid(read_jsonl(dp_prior_jsonl), name="dp prior")
    label_rows = load_contact_labels(contact_label_root)

    joined: list[dict[str, Any]] = []
    missing: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    future_insert_count = 0
    future_dp_count = 0

    for row in executor_rows:
        uuid = str(row.get("uuid", ""))
        dp = dp_rows.get(uuid)
        label = label_rows.get(uuid)
        if dp is None:
            missing["dp_prior"] += 1
            continue
        if label is None:
            missing["contact_label"] += 1
            continue
        scenario = str(row.get("scenario") or label.get("scenario") or "unknown")
        phase = str(label.get("current_phase", "unknown"))
        scenario_counts[scenario] += 1
        phase_counts[phase] += 1
        role_counts[str(row.get("prefix_role", "unknown"))] += 1
        future_insert_count += int(bool(label.get("future_inserted_within_chunk")))
        future_dp_count += int(bool(label.get("future_dp_continuable_within_chunk")))
        joined.append(
            {
                "uuid": uuid,
                "split": row.get("split"),
                "source_uuid": row.get("source_uuid"),
                "source_h5": row.get("source_h5"),
                "scenario": scenario,
                "prefix_role": row.get("prefix_role"),
                "prefix_frame_index": row.get("prefix_frame_index"),
                "action_start_step": row.get("action_start_step"),
                "action_end_step": row.get("action_end_step"),
                "executor_sample_npz": row.get("sample_npz"),
                "dp_prior_npz": dp.get("dp_prior_npz"),
                "contact_label_npz": str(contact_label_root / str(label.get("contact_label_npz_rel"))),
                "task_path_source": row.get("task_path_source"),
                "current_phase": phase,
                "current_phase_id": label.get("current_phase_id"),
                "current_contact_progress": label.get("current_contact_progress"),
                "chunk_end_contact_progress": label.get("chunk_end_contact_progress"),
                "chunk_contact_progress_delta": label.get("chunk_contact_progress_delta"),
                "future_inserted_within_chunk": bool(label.get("future_inserted_within_chunk")),
                "future_dp_continuable_within_chunk": bool(label.get("future_dp_continuable_within_chunk")),
                "current_dp_continuable": bool(label.get("current_dp_continuable")),
                "current_grasped": bool(label.get("current_grasped")),
                "current_inserted": bool(label.get("current_inserted")),
                "current_peg_head_at_hole": label.get("current_peg_head_at_hole"),
                "boundary": (
                    "Joined manifest for contact/progress-conditioned executor training. "
                    "Contact labels are supervision/scoring targets and cannot be used as "
                    "future privileged controller inputs during live evaluation."
                ),
            }
        )

    write_jsonl(output_root / "contact_executor_dataset_file.jsonl", joined)
    summary = {
        "schema": "cosmos3_contact_executor_dataset_join_v1",
        "executor_jsonl": str(executor_jsonl),
        "dp_prior_jsonl": str(dp_prior_jsonl),
        "contact_label_root": str(contact_label_root),
        "output_root": str(output_root),
        "num_executor_rows": len(executor_rows),
        "num_joined_rows": len(joined),
        "missing_counts": dict(sorted(missing.items())),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "prefix_role_counts": dict(sorted(role_counts.items())),
        "current_phase_counts": dict(sorted(phase_counts.items())),
        "future_inserted_within_chunk_count": int(future_insert_count),
        "future_dp_continuable_within_chunk_count": int(future_dp_count),
        "ready_for_contact_executor_training": bool(joined and len(joined) == len(executor_rows) and not missing),
        "boundary": (
            "Dataset join only. Formal training still needs a trainer with "
            "progress/contact losses and closed-loop video/final-state gates."
        ),
    }
    write_json(output_root / "contact_executor_dataset_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ready_for_contact_executor_training"] else 64


if __name__ == "__main__":
    raise SystemExit(main())
