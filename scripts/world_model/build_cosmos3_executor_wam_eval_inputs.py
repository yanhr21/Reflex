#!/usr/bin/env python3
"""Build Cosmos WAM inference inputs for executor rows."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from build_cosmos3_full_episode_wam_eval_inputs import (
    row_caption,
    sanitize_name,
    validate_row,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-root", required=True)
    parser.add_argument("--executor-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--include-roles", default="target_motion_observed,target_post_motion,insert_resume,peg_recovery")
    parser.add_argument("--expected-video-frames", type=int, default=301)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--expected-action-dim", type=int, default=32)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=9100)
    parser.add_argument(
        "--selection-policy",
        choices=("role_scenario_round_robin", "first"),
        default="role_scenario_round_robin",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def infer_split(executor_jsonl: Path, requested: str) -> str:
    if requested:
        return requested
    parent = executor_jsonl.parent.name
    if parent in {"train", "val"}:
        return parent
    return "train"


def scenario_from_uuid(uuid: str) -> str:
    if "_seed" in uuid:
        return uuid.split("_seed", 1)[0]
    return "unknown"


def source_uuid_from_row_uuid(uuid: str) -> str:
    match = re.match(r"(.+?\\.fix3_traj_0)__", uuid)
    if match:
        return match.group(1)
    if "__" in uuid:
        return uuid.split("__", 1)[0]
    return uuid


def select_candidates(candidates: list[dict[str, Any]], max_samples: int, policy: str) -> list[dict[str, Any]]:
    if max_samples == 0:
        return []
    if policy == "first":
        return candidates[:max_samples] if max_samples > 0 else candidates
    buckets: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for candidate in candidates:
        buckets[(str(candidate["role"]), str(candidate["scenario"]))].append(candidate)
    keys = sorted(buckets)
    selected: list[dict[str, Any]] = []
    seen_source: set[str] = set()
    while keys and (max_samples < 0 or len(selected) < max_samples):
        progressed = False
        for key in list(keys):
            bucket = buckets[key]
            chosen: dict[str, Any] | None = None
            skipped_same_source: list[dict[str, Any]] = []
            while bucket:
                item = bucket.popleft()
                if str(item["source_uuid"]) not in seen_source:
                    chosen = item
                    break
                skipped_same_source.append(item)
            if chosen is None and skipped_same_source:
                chosen = skipped_same_source.pop(0)
            for item in reversed(skipped_same_source):
                bucket.appendleft(item)
            if chosen is not None:
                selected.append(chosen)
                seen_source.add(str(chosen["source_uuid"]))
                progressed = True
                if max_samples > 0 and len(selected) >= max_samples:
                    break
            if not bucket:
                keys.remove(key)
        if not progressed:
            break
    return selected


def main() -> int:
    args = parse_args()
    condition_root = Path(args.condition_root).resolve()
    executor_jsonl = Path(args.executor_jsonl).resolve()
    output_root = Path(args.output_root).resolve()
    split = infer_split(executor_jsonl, args.split)
    source_jsonl = condition_root / split / "video_action_dataset_file.jsonl"
    if not source_jsonl.is_file():
        raise SystemExit(f"missing condition split JSONL: {source_jsonl}")
    if not executor_jsonl.is_file():
        raise SystemExit(f"missing executor JSONL: {executor_jsonl}")

    include_roles = {item.strip() for item in args.include_roles.split(",") if item.strip()}
    condition_rows = read_jsonl(source_jsonl)
    condition_by_uuid = {str(row.get("uuid")): row for row in condition_rows}
    executor_rows = read_jsonl(executor_jsonl)
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for exec_idx, executor_row in enumerate(executor_rows):
        uuid = str(executor_row.get("uuid") or "")
        role = str(executor_row.get("prefix_role") or "")
        if include_roles and role not in include_roles:
            rejected.append({"idx": exec_idx, "uuid": uuid, "reason": "role_not_selected"})
            continue
        row = condition_by_uuid.get(uuid)
        if row is None:
            rejected.append({"idx": exec_idx, "uuid": uuid, "reason": "missing_condition_row"})
            continue
        failures = validate_row(
            row,
            expected_video_frames=int(args.expected_video_frames),
            expected_action_steps=int(args.expected_action_steps),
            expected_action_dim=int(args.expected_action_dim),
        )
        if failures:
            rejected.append({"idx": exec_idx, "uuid": uuid, "reason": "condition_row_invalid", "failures": failures})
            continue
        candidates.append(
            {
                "exec_idx": exec_idx,
                "executor_row": executor_row,
                "row": row,
                "uuid": uuid,
                "role": role,
                "scenario": str(row.get("scenario") or scenario_from_uuid(uuid)),
                "source_uuid": str(row.get("source_uuid") or source_uuid_from_row_uuid(uuid)),
            }
        )

    selected_candidates = select_candidates(candidates, int(args.max_samples), str(args.selection_policy))
    input_rows: list[dict[str, Any]] = []
    manifest_samples: list[dict[str, Any]] = []
    role_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    input_dir = output_root / "inputs"
    action_dir = input_dir / "actions"
    action_dir.mkdir(parents=True, exist_ok=True)

    for selected in selected_candidates:
        executor_row = selected["executor_row"]
        row = selected["row"]
        uuid = str(selected["uuid"])
        role = str(selected["role"])
        source_action_path = Path(str(row["action_path"]))
        action = np.load(source_action_path, allow_pickle=False)
        scenario = str(selected["scenario"])
        source_uuid = str(selected["source_uuid"])
        name = sanitize_name(f"{len(input_rows):03d}_{role}_{scenario}_{uuid}")
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
            "fps": int(args.fps),
            "num_frames": int(args.expected_video_frames),
            "image_size": int(args.image_size),
            "aspect_ratio": "1,1",
            "action_chunk_size": int(args.expected_action_steps),
            "raw_action_dim": int(args.expected_action_dim),
            "condition_frame_indexes_vision": cond_vision,
            "condition_frame_indexes_action": cond_action,
            "num_steps": 30,
            "guidance": 1.0,
            "seed": int(args.seed) + len(input_rows),
            "extra": {
                "condition_root": str(condition_root),
                "executor_jsonl": str(executor_jsonl),
                "source_jsonl": str(source_jsonl),
                "source_row_uuid": uuid,
                "source_uuid": source_uuid,
                "scenario": scenario,
                "prefix_role": role,
                "prefix_frame_index": row.get("prefix_frame_index"),
                "condition_prefix_frames": row.get("condition_prefix_frames"),
                "reference_video_path": row.get("vision_path"),
                "reference_action_path": str(source_action_path),
                "reference_action_json_path": str(action_json_path),
                "state_target_path": row.get("state_target_path"),
                "executor_sample_npz": executor_row.get("sample_npz"),
                "expected_video_frames": int(args.expected_video_frames),
                "expected_action_steps": int(args.expected_action_steps),
                "expected_action_dim": int(args.expected_action_dim),
                "evidence_boundary": (
                    "Executor-targeted Cosmos prediction input. Generated "
                    "sidecars may be used as causal task-path inputs only "
                    "after strict inspection passes."
                ),
            },
        }
        input_rows.append(sample)
        manifest_samples.append(sample["extra"] | {"name": name, "input_action_json": str(action_json_path)})
        role_counts[role] += 1
        scenario_counts[scenario] += 1

    if not input_rows:
        raise SystemExit("no executor WAM eval inputs selected")

    input_jsonl = input_dir / f"{split}_executor_wam_policy_samples.jsonl"
    input_jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in input_rows) + "\n")
    manifest = {
        "schema": "cosmos3_executor_wam_eval_inputs_v1",
        "strict_eval_input_ok": True,
        "condition_root": str(condition_root),
        "executor_jsonl": str(executor_jsonl),
        "source_jsonl": str(source_jsonl),
        "input_jsonl": str(input_jsonl),
        "output_root": str(output_root),
        "split": split,
        "num_executor_rows": len(executor_rows),
        "num_candidate_samples": len(candidates),
        "num_selected_samples": len(input_rows),
        "max_samples": int(args.max_samples),
        "selection_policy": str(args.selection_policy),
        "include_roles": sorted(include_roles),
        "role_counts": dict(sorted(role_counts.items())),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "samples": manifest_samples,
        "rejected_preview": rejected[:50],
        "boundary": (
            "This is an input manifest only. It becomes usable for executor "
            "training only after Cosmos inference and strict generated-artifact "
            "inspection pass."
        ),
    }
    write_json(output_root / "eval_input_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
