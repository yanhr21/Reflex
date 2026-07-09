#!/usr/bin/env python3
"""Build a real A/B/C/D manifest for joint DP/Cosmos training checks.

This script only selects and records real rows/windows. It does not train a
model, generate new data, or claim method evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from typing import Any


PRODUCTION_SPECS = {
    "B": {
        "index": "b_dynamic_production",
        "dataset_class": "B_dynamic_rgb_observation",
        "entity_key": "episode",
        "id_pattern": re.compile(r"_episode_(\d+)$"),
        "dp_action_supervision_allowed": False,
        "cosmos_future_supervision_allowed": True,
        "teacher_adapter_action_allowed": False,
        "outcome_label_allowed": False,
    },
    "C": {
        "index": "c_frozen_dp_production",
        "dataset_class": "C_frozen_dp_dynamic_failure",
        "entity_key": "rollout",
        "id_pattern": re.compile(r"_rollout_(\d+)$"),
        "dp_action_supervision_allowed": False,
        "cosmos_future_supervision_allowed": False,
        "teacher_adapter_action_allowed": False,
        "outcome_label_allowed": True,
    },
    "D": {
        "index": "d_future_teacher_production",
        "dataset_class": "D_future_frame_cooperation_teacher",
        "entity_key": "episode",
        "id_pattern": re.compile(r"_episode_(\d+)$"),
        "dp_action_supervision_allowed": False,
        "cosmos_future_supervision_allowed": False,
        "teacher_adapter_action_allowed": True,
        "outcome_label_allowed": False,
    },
}


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def require_slurm_unless_allowed(allow_login: bool) -> None:
    if allow_login:
        return
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_STEP_ID") == "extern":
        print(
            "refusing_login_node_execution=true\n"
            "reason=Run this builder inside a compute-node srun step.",
            file=sys.stderr,
        )
        raise SystemExit(30)


def run_guard(root: Path, command: list[str], output_path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output_path.write_text(proc.stdout, encoding="utf-8")
    return {"command": command, "returncode": proc.returncode, "output": output_path}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def split_losses(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [item for item in str(value).split(",") if item]


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_trace(path: Path, cache: dict[Path, dict[str, Any]]) -> dict[str, Any]:
    path = path.resolve()
    if path not in cache:
        cache[path] = load_json(path)
    return cache[path]


def entity_rows(trace: dict[str, Any], entity_key: str, entity_id: int) -> list[dict[str, Any]]:
    return [
        row
        for row in trace.get("action_rows", [])
        if isinstance(row, dict) and int(row.get(entity_key, -1)) == int(entity_id)
    ]


def quality_row(trace: dict[str, Any], quality_key: str, entity_key: str, entity_id: int) -> dict[str, Any]:
    for row in trace.get(quality_key, []):
        if isinstance(row, dict) and int(row.get(entity_key, -1)) == int(entity_id):
            return row
    return {}


def action_dim(rows: list[dict[str, Any]]) -> int | None:
    for row in rows:
        action = row.get("action")
        if isinstance(action, list):
            return len(action)
    return None


def first_success_step(rows: list[dict[str, Any]]) -> int | None:
    for row in rows:
        if parse_bool(row.get("success", False)):
            for key in ("step", "env_step", "action_idx"):
                if key in row:
                    return int(row[key])
            return 0
    return None


def choose_window_start(preferred: int | None, row_count: int, obs_horizon: int, pred_horizon: int) -> int:
    max_start = max(0, int(row_count) - int(pred_horizon))
    if preferred is None:
        return 0
    return max(0, min(int(preferred), max_start))


def collect_static_rows(args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    registry = root / "experiments/maniskill/data/active"
    static_h5 = registry / "a_static/official_state_pd_ee_delta_pose.h5"
    static_json = registry / "a_static/official_state_pd_ee_delta_pose.json"
    static_meta = load_json(static_json)
    episodes = {
        int(item["episode_id"]): item
        for item in static_meta.get("episodes", [])
        if isinstance(item, dict) and "episode_id" in item
    }
    shard_root = root / "experiments/maniskill/runs/01_dataset/static_rgb"
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(shard_root.glob("full_s*/summary.json")):
        summary = load_json(summary_path)
        shard_dir = summary_path.parent
        episode_start = int(summary.get("episode_start", 0))
        videos = sorted(
            shard_dir.glob("*.mp4"),
            key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
        )
        for video in videos:
            if not video.stem.isdigit():
                continue
            local_episode_id = int(video.stem)
            episode_id = episode_start + local_episode_id
            episode_meta = episodes.get(episode_id, {})
            if args.static_success_only and not parse_bool(episode_meta.get("success", False)):
                continue
            elapsed_steps = int(episode_meta.get("elapsed_steps") or args.max_episode_steps)
            action_length = min(int(args.pred_horizon), max(0, elapsed_steps))
            if action_length <= 0:
                continue
            rows.append(
                {
                    "sample_id": f"a_static_ep_{episode_id:06d}",
                    "dataset_class": "A_static_expert",
                    "dataset_role": "static_expert_state_action_rgb",
                    "source_class": "A",
                    "split": "train",
                    "video": video,
                    "trace": None,
                    "summary": summary_path,
                    "manifest": shard_dir / "manifest.txt",
                    "static_h5": static_h5,
                    "static_json": static_json,
                    "render_h5": shard_dir / "trajectory.h5",
                    "render_json": shard_dir / "trajectory.json",
                    "episode_id": episode_id,
                    "local_episode_id": local_episode_id,
                    "window_start": 0,
                    "obs_horizon": int(args.obs_horizon),
                    "pred_horizon": int(args.pred_horizon),
                    "action_length": action_length,
                    "success_once": parse_bool(episode_meta.get("success", False)),
                    "success_at_end": parse_bool(episode_meta.get("success", False)),
                    "outcome_success_label": parse_bool(episode_meta.get("success", False)),
                    "allowed_losses": ["protected_dp_bc", "cosmos_static_future"],
                    "disallowed_losses": [
                        "positive_dynamic_policy_data",
                        "final_method_evidence",
                    ],
                    "method_evidence_allowed": False,
                    "teacher_evidence_allowed": False,
                    "positive_dp_bc_allowed": True,
                    "dp_action_supervision_allowed": True,
                    "cosmos_future_supervision_allowed": True,
                    "teacher_adapter_action_allowed": False,
                    "outcome_label_allowed": False,
                    "controller_action_contract": "pd_ee_delta_pose",
                    "diagnostic_only": True,
                }
            )
            if len(rows) >= int(args.a_count):
                return rows
    return rows


def select_dynamic_rows_for_class(
    args: argparse.Namespace,
    root: Path,
    class_key: str,
    count: int,
    trace_cache: dict[Path, dict[str, Any]],
) -> list[dict[str, Any]]:
    spec = PRODUCTION_SPECS[class_key]
    index_path = root / "experiments/maniskill/data/active" / spec["index"] / "train_samples.jsonl"
    candidates = load_jsonl(index_path)
    selected: list[dict[str, Any]] = []
    selected_success: list[dict[str, Any]] = []
    selected_failure: list[dict[str, Any]] = []
    quality_key = "quality_by_rollout" if spec["entity_key"] == "rollout" else "quality_by_episode"

    for row in candidates:
        sample_id = str(row.get("sample_id", ""))
        match = spec["id_pattern"].search(sample_id)
        if not match:
            continue
        entity_id = int(match.group(1))
        if row.get("dataset_class") != spec["dataset_class"]:
            continue
        trace_path = Path(str(row["trace"]))
        trace = load_trace(trace_path, trace_cache)
        rows = entity_rows(trace, str(spec["entity_key"]), entity_id)
        if not rows:
            continue
        qrow = quality_row(trace, quality_key, str(spec["entity_key"]), entity_id)
        first_motion = qrow.get("first_motion_step")
        if first_motion is None:
            first_motion = qrow.get("motion_trigger_step")
        window_start = choose_window_start(
            int(first_motion) if first_motion is not None else None,
            len(rows),
            int(args.obs_horizon),
            int(args.pred_horizon),
        )
        success_once = any(parse_bool(item.get("success", False)) for item in rows)
        item = {
            "sample_id": sample_id,
            "dataset_class": spec["dataset_class"],
            "dataset_role": row.get("dataset_role"),
            "source_class": class_key,
            "split": row.get("split", "train"),
            "video": Path(str(row["video"])),
            "trace": trace_path,
            "summary": Path(str(row["summary"])),
            "manifest": Path(str(row["manifest"])),
            "entity_key": spec["entity_key"],
            "entity_id": entity_id,
            "window_start": window_start,
            "obs_horizon": int(args.obs_horizon),
            "pred_horizon": int(args.pred_horizon),
            "action_length": min(int(args.pred_horizon), len(rows) - window_start),
            "action_row_count": len(rows),
            "action_dim": action_dim(rows),
            "first_motion_step": first_motion,
            "first_success_step": first_success_step(rows),
            "success_once": success_once,
            "success_at_end": parse_bool(rows[-1].get("success", False)),
            "outcome_success_label": success_once,
            "quality_gate_passed": parse_bool(qrow.get("quality_gate_passed", True)),
            "accepted_by_quality_gate": parse_bool(qrow.get("accepted", True)),
            "allowed_losses": split_losses(row.get("allowed_losses")),
            "disallowed_losses": split_losses(row.get("disallowed_losses")),
            "method_evidence_allowed": parse_bool(row.get("method_evidence_allowed", False)),
            "teacher_evidence_allowed": parse_bool(row.get("teacher_evidence_allowed", False)),
            "positive_dp_bc_allowed": parse_bool(row.get("positive_dp_bc_allowed", False)),
            "dp_action_supervision_allowed": bool(spec["dp_action_supervision_allowed"]),
            "cosmos_future_supervision_allowed": bool(spec["cosmos_future_supervision_allowed"]),
            "teacher_adapter_action_allowed": bool(spec["teacher_adapter_action_allowed"]),
            "outcome_label_allowed": bool(spec["outcome_label_allowed"]),
            "controller_action_contract": "pd_ee_delta_pose",
            "diagnostic_only": True,
            "c_success_is_valid_outcome_label": class_key == "C" and success_once,
        }
        if class_key == "C":
            if success_once:
                selected_success.append(item)
            else:
                selected_failure.append(item)
            if len(selected_success) >= max(1, count // 2) and len(selected_failure) >= count - max(1, count // 2):
                break
        else:
            selected.append(item)
            if len(selected) >= count:
                break

    if class_key == "C":
        needed_success = min(max(1, count // 2), len(selected_success))
        needed_failure = min(count - needed_success, len(selected_failure))
        selected = selected_success[:needed_success] + selected_failure[:needed_failure]
        if len(selected) < count:
            selected.extend((selected_success[needed_success:] + selected_failure[needed_failure:])[: count - len(selected)])
    return selected[:count]


def validate_selected_rows(rows: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    if not rows:
        return ["no_rows_selected"]
    for row in rows:
        source_class = row.get("source_class")
        if source_class in {"B", "C", "D"} and row.get("positive_dp_bc_allowed") is not False:
            failures.append(f"{row.get('sample_id')}: dynamic positive_dp_bc_allowed_not_false")
        if source_class == "C" and row.get("success_once"):
            if row.get("c_success_is_valid_outcome_label") is not True:
                failures.append(f"{row.get('sample_id')}: c_success_not_marked_valid_outcome")
        if source_class == "D" and row.get("teacher_evidence_allowed") is not True:
            failures.append(f"{row.get('sample_id')}: d_teacher_evidence_not_true")
        if int(row.get("action_length") or 0) <= 0:
            failures.append(f"{row.get('sample_id')}: non_positive_action_length")
        for key in ("video", "summary", "manifest"):
            value = row.get(key)
            if value and not Path(value).exists():
                failures.append(f"{row.get('sample_id')}: missing_{key}")
    return failures


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), sort_keys=True) + "\n")


def write_manifest(path: Path, report: dict[str, Any]) -> None:
    lines = []
    for key, value in report.items():
        if isinstance(value, (dict, list)):
            lines.append(f"{key}={json.dumps(jsonable(value), sort_keys=True)}")
        else:
            lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/public/home/yanhongru/ICLR2027/Reflex")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to experiments/maniskill/runs/02_joint_training/joint_overfit_dataset/overfit01",
    )
    parser.add_argument("--interface-run-dir", default=None)
    parser.add_argument("--a-count", type=int, default=4)
    parser.add_argument("--b-count", type=int, default=4)
    parser.add_argument("--c-count", type=int, default=4)
    parser.add_argument("--d-count", type=int, default=4)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--pred-horizon", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--stage-name", default="joint_overfit_dataset")
    parser.add_argument("--static-success-only", action="store_true", default=True)
    parser.add_argument("--allow-login", action="store_true")
    args = parser.parse_args()

    require_slurm_unless_allowed(args.allow_login)
    root = Path(args.root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else root / "experiments/maniskill/runs/02_joint_training/joint_overfit_dataset/overfit01"
    )
    case_root = root / "experiments/maniskill/runs/02_joint_training"
    try:
        output_dir.relative_to(case_root)
    except ValueError:
        print("refusing_output_dir_outside_02_joint_training=true", file=sys.stderr)
        print(f"output_dir={output_dir}", file=sys.stderr)
        raise SystemExit(31)
    if output_dir.exists() and any(output_dir.iterdir()):
        print("refusing_existing_nonempty_output_dir=true", file=sys.stderr)
        print(f"output_dir={output_dir}", file=sys.stderr)
        raise SystemExit(32)
    output_dir.mkdir(parents=True, exist_ok=True)

    guard_dir = output_dir / "guards"
    guard_dir.mkdir(parents=True, exist_ok=True)
    dataset_guard = run_guard(
        root,
        [str(root / "scripts/world_model/require_dataset_training_inputs_ready.sh"), "joint_overfit_abcd"],
        guard_dir / "joint_overfit_abcd_gate.txt",
    )
    interface_run_dir = (
        Path(args.interface_run_dir).resolve()
        if args.interface_run_dir
        else root / "experiments/maniskill/runs/02_joint_training/interface_inspect/inspect02"
    )
    interface_guard = run_guard(
        root,
        [
            "bash",
            "-lc",
            "RUN_DIR=\"$1\" \"$0\"",
            str(root / "scripts/world_model/require_joint_interface_inspect_ready.sh"),
            str(interface_run_dir),
        ],
        guard_dir / "joint_interface_inspect_gate.txt",
    )
    if dataset_guard["returncode"] != 0 or interface_guard["returncode"] != 0:
        report = {
            "joint_dataset_built": False,
            "reason": "required_gate_failed",
            "dataset_gate_returncode": dataset_guard["returncode"],
            "interface_gate_returncode": interface_guard["returncode"],
        }
        write_manifest(output_dir / "manifest.txt", report)
        print(json.dumps(jsonable(report), indent=2, sort_keys=True))
        raise SystemExit(70)

    trace_cache: dict[Path, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    rows.extend(collect_static_rows(args, root))
    rows.extend(select_dynamic_rows_for_class(args, root, "B", int(args.b_count), trace_cache))
    rows.extend(select_dynamic_rows_for_class(args, root, "C", int(args.c_count), trace_cache))
    rows.extend(select_dynamic_rows_for_class(args, root, "D", int(args.d_count), trace_cache))

    failures = validate_selected_rows(rows)
    class_counts = Counter(str(row.get("source_class")) for row in rows)
    success_counts = Counter(
        str(row.get("source_class")) for row in rows if parse_bool(row.get("success_once", False))
    )
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase": "02_joint_training",
        "stage": str(args.stage_name),
        "output_dir": output_dir,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "interface_run_dir": interface_run_dir,
        "dataset_gate": dataset_guard["output"],
        "interface_gate": interface_guard["output"],
        "obs_horizon": int(args.obs_horizon),
        "pred_horizon": int(args.pred_horizon),
        "requested_counts": {
            "A": int(args.a_count),
            "B": int(args.b_count),
            "C": int(args.c_count),
            "D": int(args.d_count),
        },
        "class_counts": dict(class_counts),
        "success_once_counts": dict(success_counts),
        "sample_count": len(rows),
        "method_evidence_allowed": False,
        "training_started": False,
        "data_generation_started": False,
        "uses_toy_model": False,
        "controller_action_contract": "pd_ee_delta_pose",
        "c_success_policy": "valid_outcome_label_not_positive_dp_bc",
        "failure_count": len(failures),
        "failures": failures,
        "joint_dataset_built": not failures,
    }
    write_jsonl(output_dir / "joint_overfit_samples.jsonl", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_manifest(output_dir / "manifest.txt", manifest)
    print(json.dumps(jsonable(manifest), indent=2, sort_keys=True))
    if failures:
        raise SystemExit(71)


if __name__ == "__main__":
    main()
