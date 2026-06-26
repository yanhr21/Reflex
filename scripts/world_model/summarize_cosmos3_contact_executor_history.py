#!/usr/bin/env python3
"""Summarize contact-executor training history for user-facing blocker reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    root = Path(args.training_root).resolve()
    history_path = root / "training_history.json"
    history = json.loads(history_path.read_text())
    if not history:
        raise SystemExit(f"empty history: {history_path}")

    def get(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
        value = row.get(key, default)
        return float(value)

    ratios = []
    for row in history:
        prior = get(row, "eval_baseline_dp_prior_mse")
        action = get(row, "eval_action_mse")
        ratios.append(action / prior if prior > 0 else float("inf"))
    best_action_idx = min(range(len(history)), key=lambda i: get(history[i], "eval_action_mse"))
    best_ratio_idx = min(range(len(history)), key=lambda i: ratios[i])
    latest = history[-1]
    over_prior = [row for row, ratio in zip(history, ratios) if ratio > 1.0]
    payload = {
        "schema": "cosmos3_contact_executor_history_summary_v1",
        "training_root": str(root),
        "history_path": str(history_path),
        "num_points": int(len(history)),
        "first": history[0],
        "latest": latest,
        "best_eval_action_mse": history[best_action_idx],
        "best_eval_action_mse_ratio_to_prior": ratios[best_action_idx],
        "best_ratio_point": history[best_ratio_idx],
        "best_ratio": ratios[best_ratio_idx],
        "latest_ratio": ratios[-1],
        "num_points_action_worse_than_prior": int(len(over_prior)),
        "all_points_action_worse_than_prior": bool(len(over_prior) == len(history)),
        "boundary": (
            "History trend only. Final decision still comes from the formal "
            "post-floor summary and formal_live_eval_gate.json."
        ),
    }
    output = Path(args.output_json).resolve() if args.output_json else root / "training_history_summary.json"
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
