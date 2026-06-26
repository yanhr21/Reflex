#!/usr/bin/env python3
"""Select executor rows in the same order as a Cosmos eval input manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executor-jsonl", required=True)
    parser.add_argument("--eval-input-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def main() -> int:
    args = parse_args()
    executor_jsonl = Path(args.executor_jsonl).resolve()
    manifest_path = Path(args.eval_input_manifest).resolve()
    output_root = Path(args.output_root).resolve()
    split = str(args.split)

    executor_rows = read_jsonl(executor_jsonl)
    rows_by_uuid = {str(row.get("uuid")): row for row in executor_rows}
    manifest = read_json(manifest_path)
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for sample in manifest.get("samples", []):
        uuid = str(sample.get("source_row_uuid") or "")
        row = rows_by_uuid.get(uuid)
        if row is None:
            missing.append(uuid)
            continue
        out = dict(row)
        for key in ("scenario", "source_uuid", "prefix_role", "prefix_frame_index"):
            if out.get(key) in (None, "") and sample.get(key) not in (None, ""):
                out[key] = sample.get(key)
        out["selected_from_eval_input_manifest"] = str(manifest_path)
        out["cosmos_eval_sample_name"] = sample.get("name")
        selected.append(out)

    output_jsonl = output_root / split / "executor_dataset_file.jsonl"
    write_jsonl(output_jsonl, selected)
    role_counts = Counter(str(row.get("prefix_role")) for row in selected)
    scenario_counts = Counter(str(row.get("scenario")) for row in selected)
    summary = {
        "schema": "cosmos3_selected_executor_rows_from_eval_manifest_v1",
        "executor_jsonl": str(executor_jsonl),
        "eval_input_manifest": str(manifest_path),
        "output_root": str(output_root),
        "output_jsonl": str(output_jsonl),
        "split": split,
        "num_manifest_samples": len(manifest.get("samples", [])),
        "num_selected_rows": len(selected),
        "missing_count": len(missing),
        "missing_preview": missing[:20],
        "role_counts": dict(sorted(role_counts.items())),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "strict_selection_ok": len(missing) == 0 and len(selected) == len(manifest.get("samples", [])),
    }
    write_json(output_root / "selected_executor_rows_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["strict_selection_ok"] else 65


if __name__ == "__main__":
    raise SystemExit(main())
