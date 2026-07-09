#!/usr/bin/env python3
"""Inspect a real A/B/C/D joint overfit batch before training.

The inspection validates data windows, action shapes, video readability, and
loss-role permissions. It does not train a model or claim method evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any


def require_slurm_unless_allowed(allow_login: bool) -> None:
    if allow_login:
        return
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_STEP_ID") == "extern":
        print(
            "refusing_login_node_execution=true\n"
            "reason=Run this batch inspector inside a compute-node srun step.",
            file=sys.stderr,
        )
        raise SystemExit(30)


def jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None
    if isinstance(value, Path):
        return str(value)
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def load_trace(path: Path, cache: dict[Path, dict[str, Any]]) -> dict[str, Any]:
    path = path.resolve()
    if path not in cache:
        cache[path] = load_json(path)
    return cache[path]


def trace_entity_rows(trace: dict[str, Any], entity_key: str, entity_id: int) -> list[dict[str, Any]]:
    return [
        row
        for row in trace.get("action_rows", [])
        if isinstance(row, dict) and int(row.get(entity_key, -1)) == int(entity_id)
    ]


def numeric_action_array(rows: list[dict[str, Any]], start: int, length: int):
    import numpy as np

    chunk = rows[int(start) : int(start) + int(length)]
    actions = []
    for row in chunk:
        action = row.get("action")
        if not isinstance(action, list):
            raise ValueError("missing_action_list")
        actions.append(action[:7])
    return np.asarray(actions, dtype=np.float32)


def find_h5_group(h5, episode_id: int, local_episode_id: int | None = None):
    for key in (f"traj_{episode_id}", str(episode_id)):
        if key in h5:
            return key, h5[key]
    if local_episode_id is not None:
        for key in (f"traj_{local_episode_id}", str(local_episode_id)):
            if key in h5:
                return key, h5[key]
    raise KeyError(f"missing_h5_group_for_episode_{episode_id}")


def inspect_static_actions(row: dict[str, Any]) -> dict[str, Any]:
    import h5py
    import numpy as np

    static_h5 = Path(str(row["static_h5"]))
    render_h5 = Path(str(row.get("render_h5") or ""))
    episode_id = int(row["episode_id"])
    local_episode_id = int(row.get("local_episode_id", episode_id))
    window_start = int(row["window_start"])
    action_length = int(row["action_length"])
    failures: list[str] = []
    action_report: dict[str, Any] = {
        "static_h5": static_h5,
        "render_h5": render_h5,
        "episode_id": episode_id,
        "local_episode_id": local_episode_id,
    }
    source_path = static_h5
    try:
        with h5py.File(static_h5, "r") as h5:
            group_key, group = find_h5_group(h5, episode_id)
            actions = np.asarray(group["actions"], dtype=np.float32)
            action_report["h5_source"] = "static_h5"
            action_report["h5_group"] = group_key
    except Exception as exc:
        if not render_h5.is_file():
            raise
        failures.append(f"static_h5_fallback:{exc}")
        source_path = render_h5
        with h5py.File(render_h5, "r") as h5:
            group_key, group = find_h5_group(h5, episode_id, local_episode_id)
            actions = np.asarray(group["actions"], dtype=np.float32)
            action_report["h5_source"] = "render_h5"
            action_report["h5_group"] = group_key
    chunk = actions[window_start : window_start + action_length, :7]
    action_report.update(
        {
            "source_path": source_path,
            "action_shape": list(actions.shape),
            "chunk_shape": list(chunk.shape),
            "chunk_abs_max": float(np.abs(chunk).max()) if chunk.size else None,
            "fallback_warnings": failures,
        }
    )
    return action_report


def inspect_dynamic_actions(row: dict[str, Any], trace_cache: dict[Path, dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    trace = load_trace(Path(str(row["trace"])), trace_cache)
    rows = trace_entity_rows(trace, str(row["entity_key"]), int(row["entity_id"]))
    action_length = int(row["action_length"])
    window_start = int(row["window_start"])
    actions = numeric_action_array(rows, window_start, action_length)
    return {
        "trace": row["trace"],
        "entity_key": row["entity_key"],
        "entity_id": int(row["entity_id"]),
        "action_row_count": len(rows),
        "window_start": window_start,
        "action_length": action_length,
        "chunk_shape": list(actions.shape),
        "chunk_abs_max": float(np.abs(actions).max()) if actions.size else None,
        "success_once_from_trace": any(parse_bool(item.get("success", False)) for item in rows),
        "success_at_end_from_trace": parse_bool(rows[-1].get("success", False)) if rows else False,
    }


def inspect_video(path: Path, root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(root / "scripts/world_model"))
    from video_contract_utils import inspect_video_file

    return inspect_video_file(path)


def inspect_permissions(row: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    source_class = row.get("source_class")
    if parse_bool(row.get("method_evidence_allowed", False)):
        failures.append("method_evidence_allowed_true")
    if source_class in {"B", "C", "D"} and parse_bool(row.get("positive_dp_bc_allowed", False)):
        failures.append("dynamic_positive_dp_bc_allowed_true")
    if source_class == "C" and parse_bool(row.get("success_once", False)):
        if row.get("c_success_is_valid_outcome_label") is not True:
            failures.append("c_success_not_marked_valid_outcome_label")
    if source_class == "D" and not parse_bool(row.get("teacher_evidence_allowed", False)):
        failures.append("d_teacher_evidence_not_true")
    for key in ("state_intervention", "snap_or_teleport", "target_assisted"):
        if parse_bool(summary.get(key, False)):
            failures.append(f"{key}_true")
    return failures


def inspect_row(row: dict[str, Any], root: Path, trace_cache: dict[Path, dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    summary = load_json(Path(str(row["summary"]))) if row.get("summary") else {}
    video_report = inspect_video(Path(str(row["video"])), root)
    if not video_report.get("exists"):
        failures.append("missing_video")
    if video_report.get("error"):
        failures.append(f"video_decode_error:{video_report.get('error')}")
    if source_class := row.get("source_class"):
        if source_class == "A":
            action_report = inspect_static_actions(row)
        else:
            action_report = inspect_dynamic_actions(row, trace_cache)
    else:
        failures.append("missing_source_class")
        action_report = {}
    chunk_shape = action_report.get("chunk_shape")
    if not (
        isinstance(chunk_shape, list)
        and len(chunk_shape) == 2
        and int(chunk_shape[0]) == int(row.get("action_length", -1))
        and int(chunk_shape[1]) == 7
    ):
        failures.append(f"bad_action_chunk_shape:{chunk_shape}")
    failures.extend(inspect_permissions(row, summary))
    return {
        "sample_id": row.get("sample_id"),
        "source_class": row.get("source_class"),
        "dataset_class": row.get("dataset_class"),
        "success_once": row.get("success_once"),
        "success_at_end": row.get("success_at_end"),
        "allowed_losses": row.get("allowed_losses"),
        "dp_action_supervision_allowed": row.get("dp_action_supervision_allowed"),
        "cosmos_future_supervision_allowed": row.get("cosmos_future_supervision_allowed"),
        "teacher_adapter_action_allowed": row.get("teacher_adapter_action_allowed"),
        "outcome_label_allowed": row.get("outcome_label_allowed"),
        "video": video_report,
        "action": action_report,
        "permission_failures": failures,
        "ok": not failures,
    }


def write_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Joint DP/Cosmos Batch Inspection",
        "",
        f"- status: `{report['status']}`",
        f"- sample count: `{report['sample_count']}`",
        f"- failure count: `{report['failure_count']}`",
        f"- class counts: `{json.dumps(report['class_counts'], sort_keys=True)}`",
        f"- success counts: `{json.dumps(report['success_once_counts'], sort_keys=True)}`",
        "",
        "| sample | class | success_once | action chunk | decoded frames | ok |",
        "|---|---|---:|---|---:|---:|",
    ]
    for item in report["samples"]:
        lines.append(
            "| `{sample}` | `{cls}` | {success} | `{shape}` | {frames} | {ok} |".format(
                sample=item.get("sample_id"),
                cls=item.get("source_class"),
                success=item.get("success_once"),
                shape=item.get("action", {}).get("chunk_shape"),
                frames=item.get("video", {}).get("decoded_frame_count"),
                ok=item.get("ok"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/public/home/yanhongru/ICLR2027/Reflex")
    parser.add_argument(
        "--dataset-dir",
        default="/public/home/yanhongru/ICLR2027/Reflex/experiments/maniskill/runs/02_joint_training/joint_overfit_dataset/overfit01",
    )
    parser.add_argument("--samples-jsonl", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--allow-login", action="store_true")
    args = parser.parse_args()

    require_slurm_unless_allowed(args.allow_login)
    root = Path(args.root).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    samples_path = Path(args.samples_jsonl).resolve() if args.samples_jsonl else dataset_dir / "joint_overfit_samples.jsonl"
    output_json = Path(args.output_json).resolve() if args.output_json else dataset_dir / "batch_inspection.json"
    output_md = Path(args.output_md).resolve() if args.output_md else dataset_dir / "batch_inspection.md"

    rows = load_jsonl(samples_path)
    trace_cache: dict[Path, dict[str, Any]] = {}
    sample_reports = [inspect_row(row, root, trace_cache) for row in rows]
    failures = [
        f"{item.get('sample_id')}:{';'.join(item.get('permission_failures', []))}"
        for item in sample_reports
        if not item.get("ok")
    ]
    class_counts = Counter(str(row.get("source_class")) for row in rows)
    success_counts = Counter(str(row.get("source_class")) for row in rows if parse_bool(row.get("success_once", False)))
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase": "02_joint_training",
        "stage": "joint_overfit_batch_inspect",
        "status": "ok" if not failures else "failed",
        "root": root,
        "dataset_dir": dataset_dir,
        "samples_jsonl": samples_path,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "sample_count": len(rows),
        "class_counts": dict(class_counts),
        "success_once_counts": dict(success_counts),
        "failure_count": len(failures),
        "failures": failures,
        "method_evidence_allowed": False,
        "training_started": False,
        "data_generation_started": False,
        "uses_toy_model": False,
        "batch_inspect_ready": not failures,
        "samples": sample_reports,
    }
    output_json.write_text(json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(jsonable(report), output_md)
    print(json.dumps(jsonable(report), indent=2, sort_keys=True))
    if failures:
        raise SystemExit(73)


if __name__ == "__main__":
    main()
