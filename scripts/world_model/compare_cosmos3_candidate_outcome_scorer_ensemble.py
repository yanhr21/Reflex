#!/usr/bin/env python3
"""Compare simple candidate outcome scorer ensembles on saved DP-rollout labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from eval_cosmos3_candidate_outcome_scorer_margins import evaluate_margin, predict
from run_cosmos3_live_receding_loop import require_compute_step
from train_cosmos3_candidate_outcome_scorer import (
    CANDIDATE_DESCRIPTOR_NAMES,
    candidate_score,
    is_dp_prior_candidate,
    load_base_rows,
    load_dataset,
    load_outcome_rows,
    split_by_uuid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-executor-jsonl", required=True)
    parser.add_argument("--outcome-jsonl", action="append", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Name/path pair, for example h2048=/path/checkpoint_best_gate.pt",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--margins", default="0,0.025,0.05,0.1,0.2,0.3,0.5")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def parse_margins(text: str) -> list[float]:
    out: list[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    return out or [0.0]


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


def parse_checkpoint_items(items: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"checkpoint must be name=path, got {item!r}")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"empty checkpoint name in {item!r}")
        out.append((name, Path(path).resolve()))
    return out


def score_delta_by_group(scores: np.ndarray, meta: list[dict[str, Any]]) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.float32)
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(meta):
        groups[str(item.get("uuid") or "")].append(int(idx))
    for indices in groups.values():
        dp_idx = None
        for idx in indices:
            if is_dp_prior_candidate(str(meta[idx].get("candidate_name") or "")):
                dp_idx = int(idx)
                break
        if dp_idx is None:
            continue
        out[indices] = scores[indices] - float(scores[dp_idx])
        out[dp_idx] = 0.0
    return out


def best_by_handoff_then_error(items: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        items,
        key=lambda item: (
            int(item.get("selected_handoff_success_count", 0)),
            -float(item.get("selected_minus_dp_prior_weighted_error_mean") or 1e9),
            float(item.get("selected_minus_dp_prior_contact_progress_delta_mean") or -1e9),
            -float(item.get("margin") or 0.0),
        ),
    )


def main() -> int:
    args = parse_args()
    require_compute_step()

    import torch

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("require_cuda=true but CUDA is unavailable")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint_items = parse_checkpoint_items(args.checkpoint)
    base_rows = load_base_rows(Path(args.contact_executor_jsonl).resolve(), 0)
    outcome_rows = load_outcome_rows(list(args.outcome_jsonl), dedupe=True, allowed_families=None)
    dataset = load_dataset(base_rows=base_rows, outcome_rows=outcome_rows)
    meta = dataset["meta"]
    train_idx, val_idx = split_by_uuid(meta, float(args.val_fraction), int(args.seed))
    eval_idx = val_idx if len(val_idx) else train_idx

    all_scores: dict[str, np.ndarray] = {}
    score_deltas: dict[str, np.ndarray] = {}
    checkpoints_summary: dict[str, Any] = {}
    for name, path in checkpoint_items:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        descriptor_names = list(checkpoint.get("candidate_descriptor_names") or [])
        if descriptor_names and descriptor_names != list(CANDIDATE_DESCRIPTOR_NAMES):
            raise RuntimeError(
                f"{name} candidate descriptor mismatch: {descriptor_names} != {list(CANDIDATE_DESCRIPTOR_NAMES)}"
            )
        scorer_args = SimpleNamespace(**dict(checkpoint.get("args") or {}))
        predictions = predict(checkpoint=checkpoint, dataset=dataset, scorer_args=scorer_args, device=device)
        scores = candidate_score(
            predictions["pred_error"],
            predictions["pred_state"],
            predictions["pred_progress"],
            predictions["pred_binary"],
            scorer_args,
        ).astype(np.float32)
        all_scores[name] = scores
        score_deltas[name] = score_delta_by_group(scores, meta)
        checkpoints_summary[name] = {
            "checkpoint": str(path),
            "feature_dim": int(checkpoint.get("feature_dim", 0)),
            "binary_dim": int(checkpoint.get("binary_dim", 0)),
            "score_weights": {
                key: getattr(scorer_args, key)
                for key in [
                    "score_success_weight",
                    "score_handoff_success_weight",
                    "score_inserted_weight",
                    "score_grasped_weight",
                    "score_progress_weight",
                    "score_progress_delta_weight",
                    "score_continuable_weight",
                    "score_state_abs_axis_weights",
                    "score_state_target",
                ]
                if hasattr(scorer_args, key)
            },
        }

    names = [name for name, _path in checkpoint_items]
    delta_stack = np.stack([score_deltas[name] for name in names], axis=0)
    score_stack = np.stack([all_scores[name] for name in names], axis=0)
    strategies: dict[str, np.ndarray] = {}
    for name in names:
        strategies[f"single_{name}"] = all_scores[name]
    strategies["mean_raw_score"] = score_stack.mean(axis=0).astype(np.float32)
    strategies["mean_delta_vs_dp"] = delta_stack.mean(axis=0).astype(np.float32)
    strategies["max_delta_vs_dp"] = delta_stack.max(axis=0).astype(np.float32)
    strategies["min_delta_vs_dp"] = delta_stack.min(axis=0).astype(np.float32)

    margins = parse_margins(args.margins)
    rows: list[dict[str, Any]] = []
    for strategy_name, strategy_scores in strategies.items():
        eval_results = [
            evaluate_margin(
                margin=margin,
                indices=eval_idx,
                meta=meta,
                true_error=dataset["error"],
                true_progress=dataset["progress"],
                pred_scores=strategy_scores,
            )
            for margin in margins
        ]
        train_results = [
            evaluate_margin(
                margin=margin,
                indices=train_idx,
                meta=meta,
                true_error=dataset["error"],
                true_progress=dataset["progress"],
                pred_scores=strategy_scores,
            )
            for margin in margins
        ]
        rows.append(
            {
                "strategy": strategy_name,
                "best_eval_by_handoff_then_error": best_by_handoff_then_error(eval_results),
                "eval": eval_results,
                "train": train_results,
            }
        )

    payload = {
        "schema": "cosmos3_candidate_outcome_scorer_ensemble_compare_v1",
        "contact_executor_jsonl": str(Path(args.contact_executor_jsonl).resolve()),
        "outcome_jsonl": [str(Path(item).resolve()) for item in args.outcome_jsonl],
        "checkpoint_order": names,
        "checkpoints": checkpoints_summary,
        "num_base_rows": int(dataset["num_base_rows"]),
        "num_candidate_rows": int(len(meta)),
        "num_train_rows": int(len(train_idx)),
        "num_eval_rows": int(len(eval_idx)),
        "val_fraction": float(args.val_fraction),
        "seed": int(args.seed),
        "margins": margins,
        "rows": rows,
        "boundary": (
            "Offline saved-label ensemble diagnostic. It uses only real "
            "candidate DP-rollout labels and does not change live evaluation."
        ),
    }

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "ensemble_compare_summary.json").write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Candidate Outcome Scorer Ensemble Compare",
        "",
        "| strategy | margin | selected/DP/oracle handoff | handoff delta | error delta | progress delta | non-DP | top1 handoff oracle |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        best = row["best_eval_by_handoff_then_error"]
        lines.append(
            "| {strategy} | {margin:g} | {selected_handoff_success_count}/{dp_prior_handoff_success_count}/{handoff_oracle_success_count} | {selected_minus_dp_prior_handoff_success_fraction:.6f} | {selected_minus_dp_prior_weighted_error_mean:.6f} | {selected_minus_dp_prior_contact_progress_delta_mean:.6f} | {selected_non_dp_candidate_fraction:.6f} | {top1_handoff_oracle_match_fraction:.6f} |".format(
                strategy=row["strategy"],
                **best,
            )
        )
    lines.append("")
    lines.append("Boundary: offline saved-label diagnostic only; not live controller evidence.")
    (output_root / "ensemble_compare_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(jsonable(payload), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
