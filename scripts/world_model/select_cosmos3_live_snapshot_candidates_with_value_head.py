#!/usr/bin/env python3
"""Select saved-snapshot candidates with the trained outcome value head.

This writes a replay filter TSV. It is a diagnostic bridge: the resulting
filter must still be replayed from the simulator snapshots before it counts as
evidence about action execution.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_cosmos3_live_receding_loop import require_compute_step  # noqa: E402
from train_cosmos3_candidate_outcome_scorer import (  # noqa: E402
    CANDIDATE_DESCRIPTOR_NAMES,
    OutcomeScorerNet,
    build_base_feature,
    candidate_descriptor,
    candidate_family,
    candidate_score,
    fixed_width_vector,
    is_dp_prior_candidate,
    load_base_rows,
    load_candidate_actions,
    load_outcome_rows,
    parse_candidate_family_set,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-executor-jsonl", required=True)
    parser.add_argument("--outcome-jsonl", action="append", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--max-base-rows", type=int, default=0)
    parser.add_argument("--allowed-candidate-families", default="dp_prior,causal_suffix_diffusion")
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
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def iter_index_from_row(row: dict[str, Any]) -> int:
    iter_dir = str(row.get("live_snapshot_iter_dir") or "")
    match = re.search(r"iter_(\d+)_prefix", iter_dir)
    if match:
        return int(match.group(1))
    uuid = str(row.get("uuid") or "")
    match = re.search(r"__iter(\d+)__", uuid)
    if match:
        return int(match.group(1))
    value = row.get("iteration")
    if value is not None:
        return int(value)
    raise ValueError(f"cannot infer iteration for uuid={uuid}")


def build_selection_dataset(
    *,
    base_rows: dict[str, dict[str, Any]],
    outcome_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    base_cache: dict[str, tuple[np.ndarray, np.ndarray, int, dict[str, Any]]] = {}
    features: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    for row in outcome_rows:
        uuid = str(row.get("uuid") or "")
        if not uuid or uuid not in base_rows:
            continue
        if uuid not in base_cache:
            base_cache[uuid] = build_base_feature(base_rows[uuid])
        base_feature, prior_flat, horizon, _base_meta = base_cache[uuid]
        if int(row.get("horizon") or horizon) != horizon:
            continue
        candidate = load_candidate_actions(row, horizon)
        residual = candidate - prior_flat
        candidate_name = str(row.get("candidate_name") or "")
        candidate_source = str(row.get("candidate_source") or "")
        descriptor = candidate_descriptor(candidate_name, candidate_source)
        feature = np.concatenate([base_feature, candidate, residual, descriptor]).astype(np.float32)
        meta = {
            "uuid": uuid,
            "scenario": str(row.get("scenario") or ""),
            "iteration": iter_index_from_row(row),
            "candidate_index": int(row.get("live_snapshot_candidate_index")),
            "candidate_name": candidate_name,
            "candidate_source": candidate_source,
            "candidate_family": candidate_family(candidate_name, candidate_source),
            "true_dp_rollout_success": bool(row.get("dp_rollout_success", False)),
            "true_dp_rollout_continuable_proxy": bool(row.get("dp_rollout_continuable_proxy", False)),
            "true_final_success": bool(row.get("final_success", False)),
            "true_final_abs_task_error_weighted": float(row.get("final_abs_task_error_weighted", 0.0)),
            "true_final_contact_progress_delta_proxy": float(row.get("final_contact_progress_delta_proxy", 0.0)),
            "live_snapshot_iter_dir": row.get("live_snapshot_iter_dir"),
            "source_h5": row.get("source_h5"),
        }
        features.append(feature)
        metas.append(meta)
    if not features:
        raise RuntimeError("no selectable outcome rows after joining base rows")
    widths = {int(item.shape[0]) for item in features}
    if len(widths) != 1:
        raise RuntimeError(f"nonuniform feature widths: {sorted(widths)}")
    return {"x": np.stack(features).astype(np.float32), "meta": metas}


def predict_scores(
    *,
    checkpoint: dict[str, Any],
    x: np.ndarray,
    scorer_args: argparse.Namespace,
    device: Any,
) -> dict[str, np.ndarray]:
    import torch

    x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
    x_std = np.asarray(checkpoint["x_std"], dtype=np.float32)
    error_mean = np.asarray(checkpoint["error_mean"], dtype=np.float32)
    error_std = np.asarray(checkpoint["error_std"], dtype=np.float32)
    x_norm = ((x - x_mean) / x_std).astype(np.float32)
    state_dict = checkpoint["model_state_dict"]
    binary_dim = int(checkpoint.get("binary_dim") or state_dict["binary_head.weight"].shape[0])
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
    pred_error = np.concatenate(pred_error_parts, axis=0)
    pred_state = np.concatenate(pred_state_parts, axis=0)
    pred_progress = np.concatenate(pred_progress_parts, axis=0)
    pred_binary = np.concatenate(pred_binary_parts, axis=0)
    score = candidate_score(pred_error, pred_state, pred_progress, pred_binary, scorer_args)
    return {
        "score": score.astype(np.float32),
        "pred_error": pred_error.astype(np.float32),
        "pred_state": pred_state.astype(np.float32),
        "pred_progress": pred_progress.astype(np.float32),
        "pred_binary": pred_binary.astype(np.float32),
    }


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

    allowed = parse_candidate_family_set(args.allowed_candidate_families)
    base_rows = load_base_rows(Path(args.contact_executor_jsonl).resolve(), int(args.max_base_rows))
    outcome_rows = load_outcome_rows(
        list(args.outcome_jsonl),
        dedupe=True,
        allowed_families=allowed,
    )
    dataset = build_selection_dataset(base_rows=base_rows, outcome_rows=outcome_rows)
    predictions = predict_scores(
        checkpoint=checkpoint,
        x=dataset["x"],
        scorer_args=scorer_args,
        device=device,
    )
    meta = dataset["meta"]
    scores = predictions["score"]
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(meta):
        groups[str(item["uuid"])].append(int(idx))

    selected: list[dict[str, Any]] = []
    for uuid, indices in sorted(groups.items()):
        dp_idx = None
        non_dp: list[int] = []
        for idx in indices:
            if is_dp_prior_candidate(str(meta[idx]["candidate_name"])):
                if dp_idx is None:
                    dp_idx = idx
            else:
                non_dp.append(idx)
        if dp_idx is None:
            continue
        chosen = dp_idx
        best_non_dp = None
        if non_dp:
            best_non_dp = max(non_dp, key=lambda i: float(scores[i]))
            if float(scores[best_non_dp]) >= float(scores[dp_idx]) + float(args.margin):
                chosen = best_non_dp
        item = dict(meta[chosen])
        item.update(
            {
                "uuid": uuid,
                "selected_score": float(scores[chosen]),
                "dp_score": float(scores[dp_idx]),
                "best_non_dp_score": float(scores[best_non_dp]) if best_non_dp is not None else None,
                "margin": float(args.margin),
                "selected_by_value_head": bool(chosen != dp_idx),
                "pred_error": float(predictions["pred_error"][chosen, 0]),
                "pred_state": predictions["pred_state"][chosen].astype(float).tolist(),
                "pred_progress": predictions["pred_progress"][chosen].astype(float).tolist(),
                "pred_binary": predictions["pred_binary"][chosen].astype(float).tolist(),
            }
        )
        selected.append(item)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tsv_path = output_root / "selected_candidate_filter.tsv"
    with tsv_path.open("w") as f:
        f.write("# scenario iteration candidate_index candidate_name uuid\n")
        for item in selected:
            f.write(
                "{scenario}\t{iteration}\t{candidate_index}\t{candidate_name}\t{uuid}\n".format(
                    scenario=item["scenario"],
                    iteration=int(item["iteration"]),
                    candidate_index=int(item["candidate_index"]),
                    candidate_name=item["candidate_name"],
                    uuid=item["uuid"],
                )
            )
    summary = {
        "schema": "cosmos3_live_snapshot_value_head_selected_candidates_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "contact_executor_jsonl": str(Path(args.contact_executor_jsonl).resolve()),
        "outcome_jsonl": [str(Path(item).resolve()) for item in args.outcome_jsonl],
        "output_root": str(output_root),
        "selected_candidate_filter_tsv": str(tsv_path),
        "device": str(device),
        "margin": float(args.margin),
        "allowed_candidate_families": sorted(allowed) if allowed else None,
        "candidate_descriptor_names": descriptor_names,
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
        "input_candidate_rows": int(len(meta)),
        "selected_groups": int(len(selected)),
        "selected_non_dp_count": int(sum(1 for item in selected if item["selected_by_value_head"])),
        "selected_dp_count": int(sum(1 for item in selected if not item["selected_by_value_head"])),
        "selected_candidate_counts": dict(sorted(Counter(str(item["candidate_name"]) for item in selected).items())),
        "selected_family_counts": dict(sorted(Counter(str(item["candidate_family"]) for item in selected).items())),
        "selected_true_dp_rollout_success_count": int(
            sum(1 for item in selected if item["true_dp_rollout_success"])
        ),
        "selected_true_dp_rollout_continuable_count": int(
            sum(1 for item in selected if item["true_dp_rollout_continuable_proxy"])
        ),
        "selected_true_final_success_count": int(sum(1 for item in selected if item["true_final_success"])),
        "selected_rows": selected,
        "boundary": (
            "Value-head selection over already converted saved-snapshot replay "
            "labels. This only writes the replay filter; the selected chunks "
            "must be replayed from simulator snapshots before execution "
            "evidence can be claimed."
        ),
    }
    write_json(output_root / "selected_candidate_summary.json", summary)
    print(json.dumps(jsonable(summary), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
