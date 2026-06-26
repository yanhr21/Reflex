#!/usr/bin/env python3
"""Build executor rows that use Cosmos-predicted task paths.

This replaces the debug GT task path in the first executor dataset with task
state predicted by a Cosmos full-episode WAM eval run. It does not run Cosmos;
it consumes already-generated `sample_outputs.json` files that passed strict
artifact inspection.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from build_cosmos3_executor_training_dataset import TASK_PATH_NAMES


WAM_TO_EXECUTOR_TASK_PATH = {
    "hole_pose_x": "task_hole_x",
    "hole_pose_y": "task_hole_y",
    "hole_pose_z": "task_hole_z",
    "peg_head_at_hole_x": "task_peg_head_hole_x",
    "peg_head_at_hole_y": "task_peg_head_hole_y",
    "peg_head_at_hole_z": "task_peg_head_hole_z",
    "tcp_pose_x": "task_tcp_x",
    "tcp_pose_y": "task_tcp_y",
    "tcp_pose_z": "task_tcp_z",
    "hole_velocity_step_x": "task_hole_velocity_x",
    "hole_velocity_step_y": "task_hole_velocity_y",
    "hole_velocity_step_z": "task_hole_velocity_z",
    "grasped": "task_grasped",
    "inserted": "task_inserted",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executor-jsonl", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--condition-root", default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--require-strict-eval-ok", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


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


def action_array_from_sample_output(path: Path) -> np.ndarray:
    payload = read_json(path)
    outputs = payload.get("outputs") or []
    if not outputs:
        raise RuntimeError(f"{path} has no outputs")
    content = outputs[0].get("content") or {}
    if "action" not in content:
        raise RuntimeError(f"{path} has no action output")
    arr = np.asarray(content["action"], dtype=np.float32)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape != (300, 32):
        raise RuntimeError(f"{path} action shape {arr.shape} != (300, 32)")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{path} action has non-finite values")
    return arr


def denormalize(pred: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(1, -1)
    if mean.shape != (1, pred.shape[1]) or std.shape != (1, pred.shape[1]):
        raise RuntimeError(f"normalization shape mismatch: {mean.shape} {std.shape} for {pred.shape}")
    return pred * std + mean


def eval_prediction_index(eval_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_path = eval_root / "eval_input_manifest.json"
    inspection_path = eval_root / "eval_artifact_inspection.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing eval input manifest: {manifest_path}")
    if not inspection_path.is_file():
        raise FileNotFoundError(f"missing eval artifact inspection: {inspection_path}")
    manifest = read_json(manifest_path)
    inspection = read_json(inspection_path)
    inspection_by_name = {str(row["name"]): row for row in inspection.get("samples", [])}
    by_uuid: dict[str, dict[str, Any]] = {}
    for sample in manifest.get("samples", []):
        uuid = str(sample.get("source_row_uuid") or "")
        name = str(sample.get("name") or "")
        if not uuid or not name:
            continue
        by_uuid[uuid] = {
            "name": name,
            "sample": sample,
            "inspection": inspection_by_name.get(name, {}),
            "sample_outputs_json": str(eval_root / "inference" / name / "sample_outputs.json"),
        }
    return by_uuid, {
        "manifest": manifest,
        "inspection": inspection,
        "strict_eval_artifacts_ok": inspection.get("strict_eval_artifacts_ok"),
        "num_eval_samples": inspection.get("num_samples"),
        "strict_failures": inspection.get("strict_failures"),
    }


def copy_with_predicted_path(
    *,
    row: dict[str, Any],
    pred_info: dict[str, Any],
    stats: dict[str, Any],
    vector_names: list[str],
    output_root: Path,
    output_split: str,
    sample_index: int,
) -> tuple[dict[str, Any] | None, str | None]:
    original_npz = Path(str(row["sample_npz"]))
    if not original_npz.is_file():
        return None, "missing_executor_npz"
    sample_outputs = Path(str(pred_info["sample_outputs_json"]))
    if not sample_outputs.is_file():
        return None, "missing_sample_outputs_json"
    inspection = pred_info.get("inspection") or {}
    if inspection.get("strict_sample_ok") is not True:
        return None, "strict_sample_not_ok"

    prefix_frame = int(row.get("prefix_frame_index", -1))
    chunk_size = int(row.get("chunk_size", 0))
    if prefix_frame < 0 or chunk_size <= 0:
        return None, "bad_prefix_or_chunk"
    if prefix_frame + chunk_size > 300:
        return None, "chunk_exceeds_predicted_horizon"

    pred_norm = action_array_from_sample_output(sample_outputs)
    pred_raw = denormalize(pred_norm, stats)
    by_name = {name: idx for idx, name in enumerate(vector_names)}
    pred_cols: list[int] = []
    missing: list[str] = []
    for exec_name in TASK_PATH_NAMES:
        wam_name = WAM_TO_EXECUTOR_TASK_PATH[exec_name]
        if wam_name not in by_name:
            missing.append(wam_name)
        else:
            pred_cols.append(by_name[wam_name])
    if missing:
        raise RuntimeError(f"normalization stats missing vector names: {missing}")

    # Action row i is aligned to state/frame i+1 in the WAM exporter.
    task_path = pred_raw[prefix_frame : prefix_frame + chunk_size, pred_cols].astype(np.float32)
    if task_path.shape != (chunk_size, len(TASK_PATH_NAMES)):
        return None, "bad_predicted_task_path_shape"

    src = np.load(original_npz, allow_pickle=False)
    sample_rel = Path("samples") / output_split / f"{sample_index:06d}_{row['uuid']}.npz"
    sample_path = output_root / sample_rel
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sample_path,
        current_state=src["current_state"].astype(np.float32),
        current_state_names=src["current_state_names"],
        task_path=task_path,
        task_path_names=np.asarray(TASK_PATH_NAMES),
        teacher_robot_actions=src["teacher_robot_actions"].astype(np.float32),
        dp_prior_actions=src["dp_prior_actions"].astype(np.float32),
        prefix_frame=src["prefix_frame"],
        action_start_step=src["action_start_step"],
        action_end_step=src["action_end_step"],
    )

    sample_meta = pred_info["sample"]
    out = dict(row)
    out.update(
        {
            "sample_npz": str(sample_path),
            "sample_npz_rel": str(sample_rel),
            "task_path_source": "cosmos_predicted_action_sidecar",
            "cosmos_eval_sample_name": pred_info["name"],
            "cosmos_sample_outputs_json": str(sample_outputs),
            "cosmos_eval_root": str(output_root.parent if False else Path(sample_outputs).parents[2]),
            "cosmos_reference_action_path": sample_meta.get("reference_action_path"),
            "cosmos_reference_video_path": sample_meta.get("reference_video_path"),
            "cosmos_strict_sample_ok": True,
            "ready_for_debug_overfit": True,
            "ready_for_formal_executor_training": False,
            "formal_blockers": ["missing_dp_prior_actions"],
            "boundary": (
                "Task path comes from causal Cosmos WAM prediction sidecar, not "
                "from future GT state targets. Formal training still needs DP-prior "
                "chunks at training scale and downstream closed-loop video/final-state gates."
            ),
        }
    )

    gt = src["task_path"].astype(np.float32)
    if gt.shape == task_path.shape and gt.size:
        diff = task_path - gt
        out["predicted_task_path_vs_gt_debug"] = {
            "rmse": float(np.sqrt(np.mean(diff * diff))),
            "mae": float(np.mean(np.abs(diff))),
            "max_abs": float(np.max(np.abs(diff))),
            "boundary": "Diagnostic only; GT task path is not used as executor input in this row.",
        }
    return out, None


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Executor Predicted Task-Path Dataset",
        "",
        f"eval_root: `{summary['eval_root']}`",
        f"executor_jsonl: `{summary['executor_jsonl']}`",
        "",
        "## Result",
        "",
        f"- samples_written: `{summary['samples_written_total']}`",
        f"- ready_for_debug_overfit: `{summary['ready_for_debug_overfit']}`",
        f"- ready_for_formal_executor_training: `{summary['ready_for_formal_executor_training']}`",
        "",
        "## Counts",
        "",
    ]
    for key, value in summary["samples_written_by_role"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Excluded", ""])
    for key, value in summary["excluded_reason_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This replaces GT debug task paths with Cosmos-predicted WAM sidecar",
            "task paths. It still needs DP-prior chunks joined at the selected",
            "training scale before full executor training.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    executor_jsonl = Path(args.executor_jsonl).resolve()
    eval_root = Path(args.eval_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    pred_by_uuid, eval_summary = eval_prediction_index(eval_root)
    if args.require_strict_eval_ok and eval_summary.get("strict_eval_artifacts_ok") is not True:
        raise SystemExit(f"strict eval artifacts are not OK: {eval_summary}")

    condition_root = Path(args.condition_root).resolve() if args.condition_root else Path(
        eval_summary["manifest"]["condition_root"]
    ).resolve()
    stats = read_json(condition_root / "normalization_stats.json")
    vector_names = [str(name) for name in stats["vector_names"]]

    rows = read_jsonl(executor_jsonl)
    out_rows: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    rmses: list[float] = []
    output_split = executor_jsonl.parent.name

    for row in rows:
        if args.max_samples > 0 and len(out_rows) >= args.max_samples:
            excluded["max_samples_reached"] += 1
            continue
        uuid = str(row.get("uuid") or "")
        pred_info = pred_by_uuid.get(uuid)
        if pred_info is None:
            excluded["no_matching_cosmos_prediction"] += 1
            continue
        out, reason = copy_with_predicted_path(
            row=row,
            pred_info=pred_info,
            stats=stats,
            vector_names=vector_names,
            output_root=output_root,
            output_split=output_split,
            sample_index=len(out_rows),
        )
        if out is None:
            excluded[str(reason)] += 1
            continue
        out_rows.append(out)
        role_counts[str(out.get("prefix_role"))] += 1
        split_counts[str(out.get("split") or output_split)] += 1
        diag = out.get("predicted_task_path_vs_gt_debug") or {}
        if "rmse" in diag:
            rmses.append(float(diag["rmse"]))

    write_jsonl(output_root / output_split / "executor_dataset_file.jsonl", out_rows)
    summary = {
        "schema": "cosmos3_executor_predicted_task_path_dataset_v1",
        "executor_jsonl": str(executor_jsonl),
        "eval_root": str(eval_root),
        "condition_root": str(condition_root),
        "output_root": str(output_root),
        "output_split": output_split,
        "eval_strict_artifacts_ok": eval_summary.get("strict_eval_artifacts_ok"),
        "eval_num_samples": eval_summary.get("num_eval_samples"),
        "samples_written_total": len(out_rows),
        "samples_written_by_role": dict(sorted(role_counts.items())),
        "samples_written_by_split": dict(sorted(split_counts.items())),
        "excluded_reason_counts": dict(sorted(excluded.items())),
        "ready_for_debug_overfit": len(out_rows) >= 2,
        "ready_for_formal_executor_training": False,
        "formal_training_blocker": "DP-prior chunks must be exported/joined for these predicted-task-path rows before full executor training.",
        "task_path_source": "cosmos_predicted_action_sidecar",
        "gt_debug_rmse_mean": float(np.mean(rmses)) if rmses else None,
        "gt_debug_rmse_max": float(np.max(rmses)) if rmses else None,
        "boundary": (
            "Rows use causal Cosmos-predicted WAM sidecar task paths. GT state "
            "targets are used only for diagnostic error numbers, not as executor input."
        ),
    }
    write_json(output_root / "predicted_task_path_dataset_summary.json", summary)
    write_markdown(output_root / "predicted_task_path_dataset_summary.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if out_rows else 67


if __name__ == "__main__":
    raise SystemExit(main())
