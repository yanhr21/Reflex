#!/usr/bin/env python3
"""Build strict full-episode Cosmos3 WAM validation inference inputs.

This prepares a small, auditable validation subset for post-SFT generation.
It does not run inference and is not model evidence by itself.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


ROLE_PRIORITY = [
    "target_pre_motion",
    "target_motion_observed",
    "target_post_motion",
    "insert_resume",
    "peg_recovery",
    "static_monitor",
    "static_late_monitor",
]

DESIRED_ROLE_SCENARIO_PAIRS = [
    # Current fix3 v7 scenario names. Keep these first so the default panel
    # covers the active data instead of falling back to older candidate names.
    ("target_pre_motion", "hole_late_move_stop"),
    ("target_motion_observed", "hole_late_constant"),
    ("target_post_motion", "hole_late_reverse"),
    ("insert_resume", "hole_late_fast_shift"),
    ("target_pre_motion", "hole_late_sine"),
    ("target_motion_observed", "hole_late_continuous_insert"),
    ("target_post_motion", "hole_late_continuous_insert"),
    ("peg_recovery", "peg_drop"),
    ("static_monitor", "none"),
    ("static_late_monitor", "none"),
    ("insert_resume", "none"),
    # Historical scenario names from earlier fix3 candidates. Keep them after
    # the active names for old roots, but do not let them dominate v7 eval.
    ("target_pre_motion", "hole_continuous_insert_large"),
    ("target_motion_observed", "hole_continuous_insert_large"),
    ("target_post_motion", "hole_reverse_large"),
    ("insert_resume", "hole_move_stop_large"),
    ("target_pre_motion", "hole_move_stop_large"),
    ("target_motion_observed", "hole_constant_large"),
    ("target_post_motion", "hole_sine_large"),
    ("insert_resume", "hole_late_shift_large"),
    ("peg_recovery", "peg_drop"),
    ("peg_recovery", "peg_disturb"),
    ("static_monitor", "none"),
    ("static_late_monitor", "none"),
    ("target_pre_motion", "hole_move_stop"),
    ("target_motion_observed", "hole_move_stop"),
    ("target_post_motion", "hole_reverse"),
    ("insert_resume", "hole_move_stop"),
    ("target_pre_motion", "hole_constant"),
    ("insert_resume", "none"),
]

ACTIVE_V7_SCENARIOS = [
    "hole_late_move_stop",
    "hole_late_constant",
    "hole_late_reverse",
    "hole_late_sine",
    "hole_late_continuous_insert",
    "hole_late_fast_shift",
    "none",
    "peg_drop",
    "peg_disturb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--expected-video-frames", type=int, default=301)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--expected-action-dim", type=int, default=32)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value[:180].strip("._") or "sample"


def row_caption(row: dict[str, Any]) -> str:
    windows = row.get("t2w_windows") or []
    if windows and isinstance(windows[0], dict) and windows[0].get("caption"):
        return str(windows[0]["caption"])
    return str(row.get("prompt") or row.get("ai_caption") or "")


def row_scenario(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("scenario") or metadata.get("scenario") or "unknown")


def row_source_uuid(row: dict[str, Any]) -> str:
    return str(row.get("source_uuid") or row.get("source_sample_id") or row.get("uuid") or "unknown")


def row_unique_key(row: dict[str, Any]) -> str:
    return str(
        row.get("uuid")
        or "|".join(
            [
                row_scenario(row),
                row_source_uuid(row),
                str(row.get("prefix_role") or "unknown"),
                str(row.get("prefix_frame_index") or "unknown"),
            ]
        )
    )


def row_source_key(row: dict[str, Any]) -> tuple[str, str]:
    return (row_scenario(row), row_source_uuid(row))


def validate_row(
    row: dict[str, Any],
    *,
    expected_video_frames: int,
    expected_action_steps: int,
    expected_action_dim: int,
) -> list[str]:
    failures: list[str] = []
    if int(row.get("num_video_frames", -1)) != expected_video_frames:
        failures.append(f"num_video_frames:{row.get('num_video_frames')}!={expected_video_frames}")
    if int(row.get("action_chunk_size", row.get("num_action_steps", -1))) != expected_action_steps:
        failures.append(
            f"action_chunk_size:{row.get('action_chunk_size', row.get('num_action_steps'))}!={expected_action_steps}"
        )
    if int(row.get("raw_action_dim", -1)) != expected_action_dim:
        failures.append(f"raw_action_dim:{row.get('raw_action_dim')}!={expected_action_dim}")

    vision_path = Path(str(row.get("vision_path", "")))
    if not vision_path.is_file():
        failures.append(f"missing_vision_path:{vision_path}")

    action_path = Path(str(row.get("action_path", "")))
    if not action_path.is_file():
        failures.append(f"missing_action_path:{action_path}")
    else:
        try:
            arr = np.load(action_path, allow_pickle=False) if action_path.suffix == ".npy" else np.asarray(
                json.loads(action_path.read_text()).get("action"), dtype=np.float32
            )
            if list(arr.shape) != [expected_action_steps, expected_action_dim]:
                failures.append(f"action_shape:{list(arr.shape)}!=[{expected_action_steps},{expected_action_dim}]")
            if not np.isfinite(arr).all():
                failures.append("action_nonfinite")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"action_load_error:{exc!r}")

    windows = row.get("t2w_windows") or []
    if not windows:
        failures.append("missing_t2w_window")
    else:
        window = windows[0]
        if int(window.get("start_frame", -1)) != 0:
            failures.append(f"t2w_start:{window.get('start_frame')}!=0")
        if int(window.get("end_frame", -1)) != expected_video_frames - 1:
            failures.append(f"t2w_end:{window.get('end_frame')}!={expected_video_frames - 1}")
        if int(window.get("temporal_interval", -1)) != 1:
            failures.append(f"t2w_interval:{window.get('temporal_interval')}!=1")

    cond_vision = row.get("condition_frame_indexes_vision") or []
    if not isinstance(cond_vision, list) or not cond_vision:
        failures.append("missing_condition_frame_indexes_vision")
    if cond_vision and max(int(x) for x in cond_vision) > 75:
        failures.append(f"condition_vision_latent_index_too_large:{max(int(x) for x in cond_vision)}")

    cond_action = row.get("condition_frame_indexes_action") or []
    if cond_action and max(int(x) for x in cond_action) >= expected_action_steps:
        failures.append(f"condition_action_index_too_large:{max(int(x) for x in cond_action)}")

    return failures


def choose_rows(rows: list[dict[str, Any]], num_samples: int) -> list[dict[str, Any]]:
    if len(rows) <= num_samples:
        return rows[:num_samples]

    selected: list[dict[str, Any]] = []
    used_row_keys: set[str] = set()
    used_source_keys: set[tuple[str, str]] = set()

    def add_first(candidates: list[dict[str, Any]]) -> None:
        for prefer_new_source in (True, False):
            for row in candidates:
                row_key = row_unique_key(row)
                if row_key in used_row_keys:
                    continue
                source_key = row_source_key(row)
                if prefer_new_source and source_key in used_source_keys and len(candidates) > 1:
                    continue
                selected.append(row)
                used_row_keys.add(row_key)
                used_source_keys.add(source_key)
                return

    for role, scenario in DESIRED_ROLE_SCENARIO_PAIRS:
        candidates = [
            row for row in rows if row.get("prefix_role") == role and row_scenario(row) == scenario
        ]
        add_first(candidates)
        if len(selected) >= num_samples:
            return selected[:num_samples]

    for role in ROLE_PRIORITY:
        candidates = [row for row in rows if row.get("prefix_role") == role]
        add_first(candidates)
        if len(selected) >= num_samples:
            return selected[:num_samples]

    for row in rows:
        row_key = row_unique_key(row)
        if row_key in used_row_keys:
            continue
        source_key = row_source_key(row)
        if source_key in used_source_keys and len(selected) + 1 < num_samples:
            continue
        selected.append(row)
        used_row_keys.add(row_key)
        used_source_keys.add(source_key)
        if len(selected) >= num_samples:
            break
    return selected


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    condition_root = Path(args.condition_root)
    output_root = Path(args.output_root)
    source_jsonl = condition_root / args.split / "video_action_dataset_file.jsonl"
    if not source_jsonl.is_file():
        raise SystemExit(f"missing split jsonl: {source_jsonl}")

    rows = read_jsonl(source_jsonl)
    candidates = []
    rejected = []
    for idx, row in enumerate(rows):
        failures = validate_row(
            row,
            expected_video_frames=args.expected_video_frames,
            expected_action_steps=args.expected_action_steps,
            expected_action_dim=args.expected_action_dim,
        )
        if failures:
            rejected.append({"idx": idx, "uuid": row.get("uuid"), "failures": failures})
        else:
            candidates.append(row)
    if len(candidates) < args.num_samples:
        raise SystemExit(f"not enough strict candidates: {len(candidates)} < {args.num_samples}")

    selected = choose_rows(candidates, args.num_samples)
    input_dir = output_root / "inputs"
    action_dir = input_dir / "actions"
    input_jsonl = input_dir / f"{args.split}_full_episode_wam_policy_samples.jsonl"
    manifest_samples = []
    input_rows = []

    for sample_i, row in enumerate(selected):
        uuid = str(row["uuid"])
        scenario = row_scenario(row)
        source_uuid = row_source_uuid(row)
        name = sanitize_name(f"{sample_i:02d}_{row.get('prefix_role')}_{scenario}_{uuid}")
        source_action_path = Path(str(row["action_path"]))
        action = np.load(source_action_path, allow_pickle=False)
        action_json_path = action_dir / f"{name}.json"
        write_json(action_json_path, {"action": action.astype(float).tolist()})

        cond_action = [int(x) for x in row.get("condition_frame_indexes_action", [])]
        cond_vision = [int(x) for x in row.get("condition_frame_indexes_vision", [])]
        sample = {
            "name": name,
            "model_mode": "policy",
            "prompt": row_caption(row),
            "vision_path": row["vision_path"],
            "action_path": str(action_json_path),
            "domain_name": row.get("domain_name", "maniskill_peg_insertion"),
            "view_point": "third_person_view",
            "fps": args.fps,
            "num_frames": args.expected_video_frames,
            "image_size": args.image_size,
            "aspect_ratio": "1,1",
            "action_chunk_size": args.expected_action_steps,
            "raw_action_dim": args.expected_action_dim,
            "condition_frame_indexes_vision": cond_vision,
            "condition_frame_indexes_action": cond_action,
            "num_steps": 30,
            "guidance": 1.0,
            "seed": args.seed + sample_i,
            "extra": {
                "condition_root": str(condition_root),
                "source_jsonl": str(source_jsonl),
                "source_row_uuid": uuid,
                "source_uuid": source_uuid,
                "scenario": scenario,
                "prefix_role": row.get("prefix_role"),
                "prefix_frame_index": row.get("prefix_frame_index"),
                "condition_prefix_frames": row.get("condition_prefix_frames"),
                "reference_video_path": row.get("vision_path"),
                "reference_action_path": str(source_action_path),
                "reference_action_json_path": str(action_json_path),
                "task_label_path": row.get("task_label_path"),
                "task_state_target_path": row.get("task_state_target_path"),
                "state_target_path": row.get("state_target_path"),
                "expected_video_frames": args.expected_video_frames,
                "expected_action_steps": args.expected_action_steps,
                "expected_action_dim": args.expected_action_dim,
                "target_object": row.get("target_object"),
                "tool_object": row.get("tool_object"),
                "actor": row.get("actor"),
                "evidence_boundary": "Generated sample must still pass post-inference length/action/video/readout inspection.",
            },
        }
        input_rows.append(sample)
        manifest_samples.append(sample["extra"] | {"name": name, "input_action_json": str(action_json_path)})

    input_jsonl.parent.mkdir(parents=True, exist_ok=True)
    input_jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in input_rows) + "\n")

    role_counts: dict[str, int] = {}
    scenario_counts: dict[str, int] = {}
    for row in selected:
        role_counts[str(row.get("prefix_role"))] = role_counts.get(str(row.get("prefix_role")), 0) + 1
        scenario = row_scenario(row)
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
    candidate_role_counts: dict[str, int] = {}
    candidate_scenario_counts: dict[str, int] = {}
    for row in candidates:
        role = str(row.get("prefix_role"))
        scenario = row_scenario(row)
        candidate_role_counts[role] = candidate_role_counts.get(role, 0) + 1
        candidate_scenario_counts[scenario] = candidate_scenario_counts.get(scenario, 0) + 1
    candidate_scenarios = set(candidate_scenario_counts)
    selected_scenarios = set(scenario_counts)
    active_missing_from_candidates = [
        scenario for scenario in ACTIVE_V7_SCENARIOS if scenario not in candidate_scenarios
    ]
    active_missing_from_selection = [
        scenario
        for scenario in ACTIVE_V7_SCENARIOS
        if scenario in candidate_scenarios and scenario not in selected_scenarios
    ]

    manifest = {
        "boundary": "Eval input manifest only; not model evidence until generated outputs pass inspection.",
        "strict_eval_input_ok": True,
        "condition_root": str(condition_root),
        "source_jsonl": str(source_jsonl),
        "input_jsonl": str(input_jsonl),
        "output_root": str(output_root),
        "expected_video_frames": args.expected_video_frames,
        "expected_action_steps": args.expected_action_steps,
        "expected_action_dim": args.expected_action_dim,
        "num_source_rows": len(rows),
        "num_rejected_rows": len(rejected),
        "num_selected_samples": len(selected),
        "role_counts": role_counts,
        "scenario_counts": scenario_counts,
        "candidate_role_counts": dict(sorted(candidate_role_counts.items())),
        "candidate_scenario_counts": dict(sorted(candidate_scenario_counts.items())),
        "active_v7_scenarios": ACTIVE_V7_SCENARIOS,
        "active_v7_scenarios_missing_from_candidates": active_missing_from_candidates,
        "active_v7_scenarios_missing_from_selection": active_missing_from_selection,
        "coverage_boundary": (
            "Default eval panels are diagnostic. Missing scenarios do not relax "
            "closed-loop gates; they identify coverage gaps that need a larger "
            "or split-specific panel before method claims."
        ),
        "samples": manifest_samples,
        "rejected_preview": rejected[:20],
    }
    write_json(output_root / "eval_input_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
