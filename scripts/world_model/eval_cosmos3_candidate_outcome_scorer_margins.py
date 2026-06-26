#!/usr/bin/env python3
"""Evaluate conservative DP-default margins for a candidate outcome scorer."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from run_cosmos3_live_receding_loop import require_compute_step
from train_cosmos3_candidate_outcome_scorer import (
    CANDIDATE_DESCRIPTOR_NAMES,
    OutcomeScorerNet,
    candidate_score,
    fixed_width_vector,
    is_dp_prior_candidate,
    load_base_rows,
    load_dataset,
    load_outcome_rows,
    meta_handoff_continuable,
    meta_handoff_success,
    split_by_uuid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-executor-jsonl", required=True)
    parser.add_argument("--outcome-jsonl", action="append", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-base-rows", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--margins", default="0,0.005,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.3,0.5,1.0")
    parser.add_argument("--min-selected-error-improvement", type=float, default=0.005)
    parser.add_argument("--min-selected-progress-delta-improvement", type=float, default=0.0)
    parser.add_argument("--min-selected-handoff-success-improvement", type=float, default=0.0)
    parser.add_argument("--max-selected-error-degradation-for-handoff-gate", type=float, default=0.0)
    parser.add_argument("--min-non-dp-selected-fraction", type=float, default=0.1)
    parser.add_argument("--allowed-candidate-families", default=None)
    parser.add_argument("--allow-handoff-only-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--score-state-abs-axis-weights", default=None)
    parser.add_argument("--score-state-target", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def parse_margins(text: str) -> list[float]:
    out: list[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    return out or [0.0]


def predict(
    *,
    checkpoint: dict[str, Any],
    dataset: dict[str, Any],
    scorer_args: argparse.Namespace,
    device: Any,
) -> dict[str, np.ndarray]:
    import torch

    x = dataset["x"]
    x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
    x_std = np.asarray(checkpoint["x_std"], dtype=np.float32)
    error_mean = np.asarray(checkpoint["error_mean"], dtype=np.float32)
    error_std = np.asarray(checkpoint["error_std"], dtype=np.float32)
    x_norm = ((x - x_mean) / x_std).astype(np.float32)
    state_dict = checkpoint["model_state_dict"]
    binary_dim = int(
        checkpoint.get("binary_dim")
        or state_dict.get("binary_head.weight", np.empty((4, 0))).shape[0]
    )

    model = OutcomeScorerNet(
        feature_dim=int(checkpoint["feature_dim"]),
        hidden_dim=int(getattr(scorer_args, "hidden_dim")),
        num_layers=int(getattr(scorer_args, "num_layers")),
        dropout=float(getattr(scorer_args, "dropout")),
        binary_dim=binary_dim,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    pred_error_parts: list[np.ndarray] = []
    pred_state_parts: list[np.ndarray] = []
    pred_progress_parts: list[np.ndarray] = []
    pred_binary_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x_norm.shape[0], 8192):
            xb = torch.from_numpy(x_norm[start : start + 8192]).to(device)
            pred_error_norm, pred_state, pred_progress, pred_logits = model(xb)
            pred_error = pred_error_norm.detach().cpu().numpy() * error_std + error_mean
            pred_binary = torch.sigmoid(pred_logits).detach().cpu().numpy()
            pred_error_parts.append(pred_error.astype(np.float32))
            pred_state_parts.append(pred_state.detach().cpu().numpy().astype(np.float32))
            pred_progress_parts.append(pred_progress.detach().cpu().numpy().astype(np.float32))
            pred_binary_parts.append(pred_binary.astype(np.float32))
    return {
        "pred_error": np.concatenate(pred_error_parts, axis=0),
        "pred_state": np.concatenate(pred_state_parts, axis=0),
        "pred_progress": np.concatenate(pred_progress_parts, axis=0),
        "pred_binary": np.concatenate(pred_binary_parts, axis=0),
    }


def evaluate_margin(
    *,
    margin: float,
    indices: np.ndarray,
    meta: list[dict[str, Any]],
    true_error: np.ndarray,
    true_progress: np.ndarray,
    pred_scores: np.ndarray,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for global_i in indices:
        groups[str(meta[int(global_i)].get("uuid") or "")].append(int(global_i))

    selected_errors: list[float] = []
    dp_errors: list[float] = []
    oracle_errors: list[float] = []
    selected_progress_deltas: list[float] = []
    dp_progress_deltas: list[float] = []
    oracle_progress_deltas: list[float] = []
    selected_handoff_successes: list[float] = []
    dp_handoff_successes: list[float] = []
    oracle_handoff_successes: list[float] = []
    handoff_oracle_successes: list[float] = []
    selected_handoff_continuables: list[float] = []
    dp_handoff_continuables: list[float] = []
    selected_names: list[str] = []
    oracle_names: list[str] = []
    handoff_oracle_names: list[str] = []
    switched = 0
    improved_switches = 0
    harmful_switches = 0
    top1_match = 0
    top1_handoff_match = 0
    for global_list in groups.values():
        if not global_list:
            continue
        dp_global = None
        non_dp_globals: list[int] = []
        for global_i in global_list:
            name = str(meta[global_i].get("candidate_name") or "")
            if dp_global is None and is_dp_prior_candidate(name):
                dp_global = global_i
            elif not is_dp_prior_candidate(name):
                non_dp_globals.append(global_i)
        if dp_global is None:
            continue
        oracle_global = min(global_list, key=lambda i: float(true_error[i, 0]))
        handoff_candidates = [
            global_i for global_i in global_list if meta_handoff_success(meta[int(global_i)])
        ]
        handoff_oracle_global = (
            min(handoff_candidates, key=lambda i: float(true_error[i, 0]))
            if handoff_candidates
            else None
        )
        selected_global = dp_global
        if non_dp_globals:
            best_non_dp = max(non_dp_globals, key=lambda i: float(pred_scores[i]))
            if float(pred_scores[best_non_dp]) >= float(pred_scores[dp_global]) + float(margin):
                selected_global = best_non_dp
        selected_error = float(true_error[selected_global, 0])
        dp_error = float(true_error[dp_global, 0])
        oracle_error = float(true_error[oracle_global, 0])
        selected_errors.append(selected_error)
        dp_errors.append(dp_error)
        oracle_errors.append(oracle_error)
        selected_progress_deltas.append(float(true_progress[selected_global, 1]))
        dp_progress_deltas.append(float(true_progress[dp_global, 1]))
        oracle_progress_deltas.append(float(true_progress[oracle_global, 1]))
        selected_handoff_successes.append(float(meta_handoff_success(meta[selected_global])))
        dp_handoff_successes.append(float(meta_handoff_success(meta[dp_global])))
        oracle_handoff_successes.append(float(meta_handoff_success(meta[oracle_global])))
        handoff_oracle_successes.append(float(handoff_oracle_global is not None))
        selected_handoff_continuables.append(float(meta_handoff_continuable(meta[selected_global])))
        dp_handoff_continuables.append(float(meta_handoff_continuable(meta[dp_global])))
        selected_name = str(meta[selected_global].get("candidate_name") or "")
        oracle_name = str(meta[oracle_global].get("candidate_name") or "")
        selected_names.append(selected_name)
        oracle_names.append(oracle_name)
        handoff_oracle_names.append(
            str(meta[handoff_oracle_global].get("candidate_name") or "")
            if handoff_oracle_global is not None
            else "none"
        )
        if selected_global == oracle_global:
            top1_match += 1
        if handoff_oracle_global is not None and selected_global == handoff_oracle_global:
            top1_handoff_match += 1
        if selected_global != dp_global:
            switched += 1
            if selected_error < dp_error:
                improved_switches += 1
            elif selected_error > dp_error:
                harmful_switches += 1

    selected = np.asarray(selected_errors, dtype=np.float32)
    dp = np.asarray(dp_errors, dtype=np.float32)
    oracle = np.asarray(oracle_errors, dtype=np.float32)
    selected_progress_delta = np.asarray(selected_progress_deltas, dtype=np.float32)
    dp_progress_delta = np.asarray(dp_progress_deltas, dtype=np.float32)
    oracle_progress_delta = np.asarray(oracle_progress_deltas, dtype=np.float32)
    group_count = int(selected.size)
    selected_delta = selected - dp if group_count else np.asarray([], dtype=np.float32)
    selected_progress_minus_dp = (
        selected_progress_delta - dp_progress_delta if group_count else np.asarray([], dtype=np.float32)
    )
    selected_handoff_success = np.asarray(selected_handoff_successes, dtype=np.float32)
    dp_handoff_success = np.asarray(dp_handoff_successes, dtype=np.float32)
    oracle_handoff_success = np.asarray(oracle_handoff_successes, dtype=np.float32)
    handoff_oracle_success = np.asarray(handoff_oracle_successes, dtype=np.float32)
    selected_handoff_continuable = np.asarray(selected_handoff_continuables, dtype=np.float32)
    dp_handoff_continuable = np.asarray(dp_handoff_continuables, dtype=np.float32)
    return {
        "margin": float(margin),
        "num_groups": group_count,
        "selected_true_weighted_error_mean": float(selected.mean()) if group_count else None,
        "dp_prior_true_weighted_error_mean": float(dp.mean()) if group_count else None,
        "oracle_true_weighted_error_mean": float(oracle.mean()) if group_count else None,
        "selected_minus_dp_prior_weighted_error_mean": float(selected_delta.mean()) if group_count else None,
        "oracle_minus_dp_prior_weighted_error_mean": float((oracle - dp).mean()) if group_count else None,
        "selected_true_contact_progress_delta_mean": (
            float(selected_progress_delta.mean()) if group_count else None
        ),
        "dp_prior_true_contact_progress_delta_mean": float(dp_progress_delta.mean()) if group_count else None,
        "oracle_true_contact_progress_delta_mean": float(oracle_progress_delta.mean()) if group_count else None,
        "selected_minus_dp_prior_contact_progress_delta_mean": (
            float(selected_progress_minus_dp.mean()) if group_count else None
        ),
        "selected_handoff_success_count": int(selected_handoff_success.sum()) if group_count else 0,
        "dp_prior_handoff_success_count": int(dp_handoff_success.sum()) if group_count else 0,
        "oracle_handoff_success_count": int(oracle_handoff_success.sum()) if group_count else 0,
        "handoff_oracle_success_count": int(handoff_oracle_success.sum()) if group_count else 0,
        "selected_handoff_success_fraction": (
            float(selected_handoff_success.mean()) if group_count else None
        ),
        "dp_prior_handoff_success_fraction": (
            float(dp_handoff_success.mean()) if group_count else None
        ),
        "selected_minus_dp_prior_handoff_success_fraction": (
            float((selected_handoff_success - dp_handoff_success).mean()) if group_count else None
        ),
        "selected_handoff_continuable_fraction": (
            float(selected_handoff_continuable.mean()) if group_count else None
        ),
        "dp_prior_handoff_continuable_fraction": (
            float(dp_handoff_continuable.mean()) if group_count else None
        ),
        "top1_oracle_match_fraction": float(top1_match / max(1, group_count)),
        "top1_handoff_oracle_match_fraction": float(top1_handoff_match / max(1, group_count)),
        "selected_non_dp_candidate_count": int(switched),
        "selected_non_dp_candidate_fraction": float(switched / max(1, group_count)),
        "improved_switch_count": int(improved_switches),
        "harmful_switch_count": int(harmful_switches),
        "unchanged_or_equal_switch_count": int(switched - improved_switches - harmful_switches),
        "selected_candidate_counts": dict(sorted(Counter(selected_names).items())),
        "oracle_candidate_counts": dict(sorted(Counter(oracle_names).items())),
        "handoff_oracle_candidate_counts": dict(sorted(Counter(handoff_oracle_names).items())),
    }


def gate(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    delta = metrics.get("selected_minus_dp_prior_weighted_error_mean")
    progress_delta = metrics.get("selected_minus_dp_prior_contact_progress_delta_mean")
    handoff_delta = metrics.get("selected_minus_dp_prior_handoff_success_fraction")
    non_dp = metrics.get("selected_non_dp_candidate_fraction")
    if non_dp is None:
        return False
    continuous_progress_ok = bool(
        delta is not None
        and progress_delta is not None
        and handoff_delta is not None
        and float(handoff_delta) > float(args.min_selected_handoff_success_improvement)
        and float(delta) <= -float(args.min_selected_error_improvement)
        and float(progress_delta) >= float(args.min_selected_progress_delta_improvement)
    )
    handoff_success_ok = bool(
        handoff_delta is not None
        and float(handoff_delta) > float(args.min_selected_handoff_success_improvement)
    )
    safe_handoff_success_ok = bool(
        handoff_success_ok
        and delta is not None
        and progress_delta is not None
        and float(delta) <= float(args.max_selected_error_degradation_for_handoff_gate)
        and float(progress_delta) >= float(args.min_selected_progress_delta_improvement)
    )
    handoff_branch_ok = handoff_success_ok if bool(args.allow_handoff_only_gate) else safe_handoff_success_ok
    return bool(float(non_dp) >= float(args.min_non_dp_selected_fraction) and (continuous_progress_ok or handoff_branch_ok))


def main() -> int:
    args = parse_args()
    require_compute_step()

    import torch

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("require_cuda=true but CUDA is unavailable")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(Path(args.checkpoint).resolve(), map_location=device, weights_only=False)
    descriptor_names = list(checkpoint.get("candidate_descriptor_names") or [])
    if descriptor_names and descriptor_names != list(CANDIDATE_DESCRIPTOR_NAMES):
        raise SystemExit(
            "candidate_descriptor_schema_mismatch=true\n"
            f"checkpoint_descriptor_names={descriptor_names}\n"
            f"current_descriptor_names={list(CANDIDATE_DESCRIPTOR_NAMES)}"
        )
    saved_args = dict(checkpoint.get("args") or {})
    if args.score_state_abs_axis_weights is not None:
        saved_args["score_state_abs_axis_weights"] = args.score_state_abs_axis_weights
    if args.score_state_target is not None:
        saved_args["score_state_target"] = args.score_state_target
    scorer_args = SimpleNamespace(**saved_args)

    base_rows = load_base_rows(Path(args.contact_executor_jsonl).resolve(), int(args.max_base_rows))
    allowed_family_text = args.allowed_candidate_families
    if allowed_family_text is None:
        allowed_family_text = saved_args.get("allowed_candidate_families")
    from train_cosmos3_candidate_outcome_scorer import parse_candidate_family_set

    allowed_candidate_families = parse_candidate_family_set(allowed_family_text)
    outcome_rows = load_outcome_rows(
        list(args.outcome_jsonl),
        dedupe=bool(saved_args.get("dedupe_candidate_name_per_row", True)),
        allowed_families=allowed_candidate_families,
    )
    dataset = load_dataset(base_rows=base_rows, outcome_rows=outcome_rows)
    meta = dataset["meta"]
    train_idx, val_idx = split_by_uuid(meta, float(args.val_fraction), int(args.seed))
    eval_idx = val_idx if len(val_idx) else train_idx
    predictions = predict(checkpoint=checkpoint, dataset=dataset, scorer_args=scorer_args, device=device)
    pred_scores = candidate_score(
        predictions["pred_error"],
        predictions["pred_state"],
        predictions["pred_progress"],
        predictions["pred_binary"],
        scorer_args,
    )

    margins = parse_margins(args.margins)
    train_results = [
        evaluate_margin(
            margin=m,
            indices=train_idx,
            meta=meta,
            true_error=dataset["error"],
            true_progress=dataset["progress"],
            pred_scores=pred_scores,
        )
        for m in margins
    ]
    eval_results = [
        evaluate_margin(
            margin=m,
            indices=eval_idx,
            meta=meta,
            true_error=dataset["error"],
            true_progress=dataset["progress"],
            pred_scores=pred_scores,
        )
        for m in margins
    ]
    def _selection_delta(item: dict[str, Any]) -> float:
        value = item.get("selected_minus_dp_prior_weighted_error_mean")
        return float(value) if value is not None else float("inf")

    best_eval = min(eval_results, key=_selection_delta)
    gate_eval = [item for item in eval_results if gate(item, args)]
    summary = {
        "schema": "cosmos3_candidate_outcome_scorer_margin_eval_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "contact_executor_jsonl": str(Path(args.contact_executor_jsonl).resolve()),
        "outcome_jsonl": [str(Path(item).resolve()) for item in args.outcome_jsonl],
        "output_root": str(Path(args.output_root).resolve()),
        "num_base_rows": int(dataset["num_base_rows"]),
        "num_candidate_rows": int(len(meta)),
        "num_train_rows": int(len(train_idx)),
        "num_eval_rows": int(len(eval_idx)),
        "candidate_family_filter": {
            "allowed_candidate_families": (
                sorted(allowed_candidate_families) if allowed_candidate_families else None
            ),
        },
        "device": str(device),
        "candidate_descriptor_names": descriptor_names,
        "margins": margins,
        "score_weights": {
            "success": float(getattr(scorer_args, "score_success_weight", 0.5)),
            "handoff_success": float(getattr(scorer_args, "score_handoff_success_weight", 0.0)),
            "inserted": float(getattr(scorer_args, "score_inserted_weight", 0.25)),
            "grasped": float(getattr(scorer_args, "score_grasped_weight", 0.1)),
            "progress": float(getattr(scorer_args, "score_progress_weight", 0.0)),
            "progress_delta": float(getattr(scorer_args, "score_progress_delta_weight", 0.0)),
            "continuable": float(getattr(scorer_args, "score_continuable_weight", 0.0)),
            "state_abs_axis_weights": fixed_width_vector(
                getattr(scorer_args, "score_state_abs_axis_weights", "0,0,0"), 3, 0.0
            ).astype(float).tolist(),
            "state_target": fixed_width_vector(
                getattr(scorer_args, "score_state_target", "0,0,0"), 3, 0.0
            ).astype(float).tolist(),
        },
        "offline_gate": {
            "min_selected_error_improvement": float(args.min_selected_error_improvement),
            "min_selected_progress_delta_improvement": float(args.min_selected_progress_delta_improvement),
            "min_selected_handoff_success_improvement": float(args.min_selected_handoff_success_improvement),
            "max_selected_error_degradation_for_handoff_gate": float(
                args.max_selected_error_degradation_for_handoff_gate
            ),
            "min_non_dp_selected_fraction": float(args.min_non_dp_selected_fraction),
            "allow_handoff_only_gate": bool(args.allow_handoff_only_gate),
        },
        "train": train_results,
        "eval": eval_results,
        "best_eval_margin": best_eval,
        "gate_passing_eval_margins": gate_eval,
        "ready_for_conservative_offline_gate": bool(gate_eval),
        "boundary": (
            "Offline conservative selection diagnostic only. A margin means the "
            "controller keeps the frozen DP prior unless the learned scorer "
            "predicts a non-DP candidate is better by that score gap. Passing "
            "this diagnostic still requires formal training and live rollout "
            "with video/contact review before method evidence."
        ),
    }
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "margin_eval_summary.json", summary)
    lines = [
        "# Conservative Margin Eval",
        "",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- eval groups: `{best_eval['num_groups']}`",
        f"- best eval margin: `{best_eval['margin']}`",
        f"- best eval selected-minus-DP: `{best_eval['selected_minus_dp_prior_weighted_error_mean']:.6f}`",
        f"- best eval selected-minus-DP progress delta: `{best_eval['selected_minus_dp_prior_contact_progress_delta_mean']:.6f}`",
        f"- best eval non-DP fraction: `{best_eval['selected_non_dp_candidate_fraction']:.6f}`",
        f"- gate passing margins: `{len(gate_eval)}`",
        "",
        "## Eval Margins",
    ]
    for item in eval_results:
        lines.append(
            "- margin `{margin:g}`: delta `{delta:.6f}`, progress-delta `{progress_delta:.6f}`, non-DP `{non_dp:.3f}`, "
            "improved `{improved}`, harmful `{harmful}`".format(
                margin=float(item["margin"]),
                delta=float(item["selected_minus_dp_prior_weighted_error_mean"]),
                progress_delta=float(item["selected_minus_dp_prior_contact_progress_delta_mean"]),
                non_dp=float(item["selected_non_dp_candidate_fraction"]),
                improved=int(item["improved_switch_count"]),
                harmful=int(item["harmful_switch_count"]),
            )
        )
    (output_root / "margin_eval_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
