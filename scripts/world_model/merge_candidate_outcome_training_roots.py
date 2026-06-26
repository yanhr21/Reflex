#!/usr/bin/env python3
"""Merge base rows and candidate outcome labels for consequence/value training."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_cosmos3_live_receding_loop import require_compute_step  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-jsonl", action="append", required=True)
    parser.add_argument("--outcome-jsonl", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
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


def main() -> int:
    args = parse_args()
    require_compute_step()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base_by_uuid: dict[str, dict[str, Any]] = {}
    base_source_counts: Counter[str] = Counter()
    for text in args.base_jsonl:
        path = Path(text).resolve()
        for row in read_jsonl(path):
            uuid = str(row.get("uuid") or "")
            if not uuid:
                continue
            base_by_uuid[uuid] = row
            base_source_counts[str(path)] += 1

    outcome_rows: list[dict[str, Any]] = []
    outcome_source_counts: Counter[str] = Counter()
    for text in args.outcome_jsonl:
        path = Path(text).resolve()
        for row in read_jsonl(path):
            if row.get("schema") != "cosmos3_candidate_outcome_label_v1":
                continue
            outcome_rows.append(row)
            outcome_source_counts[str(path)] += 1

    known = set(base_by_uuid)
    joined = [row for row in outcome_rows if str(row.get("uuid") or "") in known]
    missing = [row for row in outcome_rows if str(row.get("uuid") or "") not in known]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in joined:
        groups.setdefault(str(row.get("uuid") or ""), []).append(row)
    groups_with_dp = sum(
        1 for rows in groups.values() if any(str(row.get("candidate_name") or "") == "dp_prior" for row in rows)
    )
    groups_with_causal = sum(
        1 for rows in groups.values() if any(str(row.get("candidate_source") or "") == "causal_suffix_diffusion" for row in rows)
    )
    groups_with_causal_handoff = sum(
        1
        for rows in groups.values()
        if any(
            str(row.get("candidate_source") or "") == "causal_suffix_diffusion"
            and bool(row.get("dp_rollout_success", False))
            for row in rows
        )
    )
    groups_with_dp_handoff = sum(
        1
        for rows in groups.values()
        if any(str(row.get("candidate_name") or "") == "dp_prior" and bool(row.get("dp_rollout_success", False)) for row in rows)
    )

    base_rows = sorted(base_by_uuid.values(), key=lambda row: str(row.get("uuid") or ""))
    write_jsonl(output_root / "live_snapshot_base_rows.jsonl", base_rows)
    write_jsonl(output_root / "candidate_outcome_labels.jsonl", joined)
    summary = {
        "schema": "candidate_outcome_training_roots_merge_v1",
        "output_root": str(output_root),
        "base_rows_jsonl": str(output_root / "live_snapshot_base_rows.jsonl"),
        "candidate_outcome_jsonl": str(output_root / "candidate_outcome_labels.jsonl"),
        "base_rows": int(len(base_rows)),
        "input_outcome_rows": int(len(outcome_rows)),
        "joined_outcome_rows": int(len(joined)),
        "missing_base_rows": int(len(missing)),
        "groups": int(len(groups)),
        "groups_with_dp_prior": int(groups_with_dp),
        "groups_with_causal_suffix_diffusion": int(groups_with_causal),
        "groups_with_causal_suffix_diffusion_dp96_success": int(groups_with_causal_handoff),
        "groups_with_dp_prior_dp96_success": int(groups_with_dp_handoff),
        "candidate_source_counts": dict(sorted(Counter(str(row.get("candidate_source") or "") for row in joined).items())),
        "candidate_name_counts_top20": dict(Counter(str(row.get("candidate_name") or "") for row in joined).most_common(20)),
        "base_source_counts": dict(sorted(base_source_counts.items())),
        "outcome_source_counts": dict(sorted(outcome_source_counts.items())),
        "boundary": (
            "Merged dataset for offline consequence/value training. It is not "
            "live controller evidence; live evidence still requires closed-loop "
            "execution and video/final-state review."
        ),
    }
    write_json(output_root / "merge_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not joined:
        return 2
    if groups_with_dp <= 0:
        return 3
    if groups_with_causal <= 0:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
