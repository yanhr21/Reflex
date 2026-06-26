#!/usr/bin/env python3
"""Audit h96 selector blockers before launching more live evaluation."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome-jsonl", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scorer-summary", action="append", default=[])
    parser.add_argument("--scorer-history", action="append", default=[])
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument(
        "--live-candidate-families",
        default="dp_prior,checkpoint_model,model_generated,model_mean,model_scale,model_diffusion",
        help=(
            "Comma-separated candidate families treated as available in the "
            "current live loop. Teacher and retrieval residual families are "
            "excluded by default because they are replay/oracle-side helpers."
        ),
    )
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--allow-login-inspection", action="store_true")
    return parser.parse_args()


def require_compute_step(allow_login: bool) -> None:
    if allow_login:
        return
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_STEP_ID") in {None, "extern"}:
        raise SystemExit(
            "refusing_login_node_execution=true\n"
            "reason=Run this audit inside a compute-node srun step, or pass "
            "--allow-login-inspection for a small read-only inspection."
        )


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


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


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        uuid = str(row.get("uuid") or "")
        if uuid:
            groups[uuid].append(row)
    return dict(groups)


def row_handoff_success(row: dict[str, Any]) -> bool:
    return bool(row.get("dp_rollout_success", False) or row.get("final_success", False))


def summarize_headroom(
    groups: dict[str, list[dict[str, Any]]],
    *,
    allowed_families: set[str] | None,
    max_examples: int,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for uuid, local in sorted(groups.items()):
        dp_rows = [row for row in local if is_dp_prior_candidate(str(row.get("candidate_name") or ""))]
        if not dp_rows:
            continue
        dp = min(dp_rows, key=lambda row: safe_float(row.get("final_abs_task_error_weighted")))
        candidates = list(local)
        if allowed_families is not None:
            candidates = [
                row
                for row in local
                if candidate_family(str(row.get("candidate_name") or ""), str(row.get("candidate_source") or ""))
                in allowed_families
            ]
            if dp not in candidates:
                candidates.append(dp)
        oracle = min(candidates, key=lambda row: safe_float(row.get("final_abs_task_error_weighted")))
        handoff_rows = [row for row in candidates if row_handoff_success(row)]
        handoff_oracle = (
            min(handoff_rows, key=lambda row: safe_float(row.get("final_abs_task_error_weighted")))
            if handoff_rows
            else None
        )
        handoff_family = (
            candidate_family(
                str(handoff_oracle.get("candidate_name") or ""),
                str(handoff_oracle.get("candidate_source") or ""),
            )
            if handoff_oracle is not None
            else "none"
        )
        oracle_family = candidate_family(
            str(oracle.get("candidate_name") or ""),
            str(oracle.get("candidate_source") or ""),
        )
        summaries.append(
            {
                "uuid": uuid,
                "source_uuid": str(dp.get("source_uuid") or ""),
                "scenario": str(dp.get("scenario") or ""),
                "current_phase": str(dp.get("current_phase") or ""),
                "num_candidates": int(len(candidates)),
                "dp_handoff_success": row_handoff_success(dp),
                "dp_error": safe_float(dp.get("final_abs_task_error_weighted")),
                "oracle_success": bool(oracle.get("final_success", False)),
                "oracle_family": oracle_family,
                "oracle_candidate_name": str(oracle.get("candidate_name") or ""),
                "oracle_error": safe_float(oracle.get("final_abs_task_error_weighted")),
                "oracle_minus_dp_error": safe_float(oracle.get("final_abs_task_error_weighted"))
                - safe_float(dp.get("final_abs_task_error_weighted")),
                "handoff_oracle_success": handoff_oracle is not None,
                "handoff_oracle_family": handoff_family,
                "handoff_oracle_candidate_name": (
                    str(handoff_oracle.get("candidate_name") or "") if handoff_oracle is not None else None
                ),
            }
        )
    dp_handoff = sum(1 for item in summaries if item["dp_handoff_success"])
    oracle_success = sum(1 for item in summaries if item["oracle_success"])
    handoff_oracle_success = sum(1 for item in summaries if item["handoff_oracle_success"])
    deltas = np.asarray([item["oracle_minus_dp_error"] for item in summaries], dtype=np.float32)
    return {
        "num_groups": int(len(summaries)),
        "dp_handoff_success_count": int(dp_handoff),
        "oracle_final_success_count": int(oracle_success),
        "handoff_oracle_success_count": int(handoff_oracle_success),
        "mean_oracle_minus_dp_error": float(deltas.mean()) if deltas.size else None,
        "oracle_family_counts": dict(sorted(Counter(item["oracle_family"] for item in summaries).items())),
        "handoff_oracle_family_counts": dict(
            sorted(Counter(item["handoff_oracle_family"] for item in summaries).items())
        ),
        "examples": summaries[: int(max_examples)],
    }


def split_uuids(uuids: list[str], val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(int(seed))
    uuids = sorted(uuids)
    if len(uuids) <= 1 or val_fraction <= 0:
        return set(uuids), set()
    perm = rng.permutation(len(uuids))
    n_val = max(1, int(round(len(uuids) * float(val_fraction))))
    n_val = min(n_val, max(1, len(uuids) - 1))
    val = {uuids[int(i)] for i in perm[:n_val]}
    train = {uuid for uuid in uuids if uuid not in val}
    return train, val


def key_for(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row.get(field) or "") for field in fields)


def split_leakage_audit(groups: dict[str, list[dict[str, Any]]], val_fraction: float, seed: int) -> dict[str, Any]:
    train_uuids, val_uuids = split_uuids(list(groups), val_fraction, seed)
    representative = {uuid: rows[0] for uuid, rows in groups.items() if rows}
    audits: dict[str, Any] = {}
    for fields in [
        ("source_uuid",),
        ("scenario",),
        ("current_phase",),
        ("source_uuid", "scenario"),
        ("source_uuid", "current_phase"),
        ("scenario", "current_phase"),
        ("source_uuid", "scenario", "current_phase"),
    ]:
        train_keys: dict[tuple[str, ...], list[str]] = defaultdict(list)
        val_keys: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for uuid in train_uuids:
            train_keys[key_for(representative[uuid], fields)].append(uuid)
        for uuid in val_uuids:
            val_keys[key_for(representative[uuid], fields)].append(uuid)
        overlap = sorted(set(train_keys) & set(val_keys))
        audits["+".join(fields)] = {
            "overlap_key_count": int(len(overlap)),
            "overlap_group_count_train": int(sum(len(train_keys[key]) for key in overlap)),
            "overlap_group_count_val": int(sum(len(val_keys[key]) for key in overlap)),
            "examples": [
                {
                    "key": list(key),
                    "train_uuids": train_keys[key][:5],
                    "val_uuids": val_keys[key][:5],
                }
                for key in overlap[:10]
            ],
        }
    return {
        "num_train_groups": int(len(train_uuids)),
        "num_val_groups": int(len(val_uuids)),
        "val_fraction": float(val_fraction),
        "seed": int(seed),
        "overlap_audits": audits,
    }


def load_metric_source(path_text: str) -> dict[str, Any]:
    path = Path(path_text).resolve()
    payload = read_json(path)
    if "best_gate_metrics" in payload:
        metrics = payload.get("best_gate_metrics") or payload.get("best_offline_metrics") or payload.get("final_metrics")
        return {"path": str(path), "kind": "training_summary", "metrics": metrics}
    if isinstance(payload, list):
        if not payload:
            return {"path": str(path), "kind": "training_history", "metrics": {}}
        best = max(
            payload,
            key=lambda item: float(
                ((item.get("eval") or {}).get("selected_minus_dp_prior_handoff_success_fraction"))
                if ((item.get("eval") or {}).get("selected_minus_dp_prior_handoff_success_fraction")) is not None
                else -999.0
            ),
        )
        return {"path": str(path), "kind": "training_history_best_handoff", "metrics": best}
    return {"path": str(path), "kind": "unknown", "metrics": payload}


def scorer_family_audit(paths: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for text in paths:
        source = load_metric_source(text)
        metrics = source.get("metrics") or {}
        eval_metrics = metrics.get("eval") if isinstance(metrics, dict) and "eval" in metrics else metrics
        selected = dict((eval_metrics or {}).get("selected_candidate_counts") or {})
        handoff_oracle = dict((eval_metrics or {}).get("handoff_oracle_candidate_counts") or {})
        oracle = dict((eval_metrics or {}).get("oracle_candidate_counts") or {})
        out.append(
            {
                "path": source["path"],
                "kind": source["kind"],
                "selected_handoff_success_count": (eval_metrics or {}).get("selected_handoff_success_count"),
                "dp_prior_handoff_success_count": (eval_metrics or {}).get("dp_prior_handoff_success_count"),
                "selected_minus_dp_prior_handoff_success_fraction": (eval_metrics or {}).get(
                    "selected_minus_dp_prior_handoff_success_fraction"
                ),
                "selected_minus_dp_prior_weighted_error_mean": (eval_metrics or {}).get(
                    "selected_minus_dp_prior_weighted_error_mean"
                ),
                "selected_candidate_counts": selected,
                "handoff_oracle_candidate_counts": handoff_oracle,
                "oracle_candidate_counts": oracle,
                "selected_not_in_handoff_oracle": {
                    name: int(count)
                    for name, count in selected.items()
                    if name not in handoff_oracle
                },
            }
        )
    return out


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Cosmos3 h96 Selector Blocker Audit",
        "",
        "## Live-Candidate-Only Headroom",
        "",
        f"- all groups handoff oracle: `{summary['all_candidates_headroom']['handoff_oracle_success_count']}/{summary['all_candidates_headroom']['num_groups']}`",
        f"- live-family handoff oracle: `{summary['live_candidate_headroom']['handoff_oracle_success_count']}/{summary['live_candidate_headroom']['num_groups']}`",
        f"- live families: `{summary['live_candidate_families']}`",
        "",
        "## Split Leakage",
        "",
        f"- train groups: `{summary['split_leakage']['num_train_groups']}`",
        f"- val groups: `{summary['split_leakage']['num_val_groups']}`",
    ]
    for name, item in summary["split_leakage"]["overlap_audits"].items():
        lines.append(f"- `{name}` overlap keys: `{item['overlap_key_count']}`")
    lines.extend(["", "## Scorer Family Audits", ""])
    for item in summary["scorer_family_audits"]:
        lines.append(
            f"- `{item['kind']}` `{item['path']}`: selected handoff "
            f"`{item.get('selected_handoff_success_count')}` vs DP "
            f"`{item.get('dp_prior_handoff_success_count')}`, delta "
            f"`{item.get('selected_minus_dp_prior_handoff_success_fraction')}`"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    require_compute_step(bool(args.allow_login_inspection))
    rows = read_jsonl(list(args.outcome_jsonl))
    groups = group_rows(rows)
    live_families = {item.strip() for item in str(args.live_candidate_families).split(",") if item.strip()}
    audit_paths = list(args.scorer_summary) + list(args.scorer_history)
    summary = {
        "schema": "cosmos3_h96_selector_blocker_audit_v1",
        "outcome_jsonl": [str(Path(item).resolve()) for item in args.outcome_jsonl],
        "num_rows": int(len(rows)),
        "num_uuid_groups": int(len(groups)),
        "live_candidate_families": sorted(live_families),
        "all_candidates_headroom": summarize_headroom(groups, allowed_families=None, max_examples=int(args.max_examples)),
        "live_candidate_headroom": summarize_headroom(
            groups,
            allowed_families=live_families,
            max_examples=int(args.max_examples),
        ),
        "split_leakage": split_leakage_audit(groups, float(args.val_fraction), int(args.seed)),
        "scorer_family_audits": scorer_family_audit(audit_paths),
        "boundary": (
            "Offline audit only. This does not prove live controller success. "
            "It decides whether the next fix belongs to candidate generation, "
            "selector generalization, or split/gate hygiene."
        ),
    }
    output_root = Path(args.output_root).resolve()
    write_json(output_root / "selector_blocker_audit_summary.json", summary)
    write_markdown(output_root / "selector_blocker_audit_summary.md", summary)
    print(json.dumps(jsonable(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
