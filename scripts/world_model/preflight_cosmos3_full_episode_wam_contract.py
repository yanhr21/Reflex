#!/usr/bin/env python3
"""Strict preflight for full-episode Cosmos3 WAM condition exports."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty jsonl: {path}")
    return rows


def _video_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


def _resolve(path_value: str, jsonl_path: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = jsonl_path.parent / path
    return path


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_action(path: Path) -> Any:
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    payload = _load_json(path)
    return payload.get("action") if isinstance(payload, dict) else payload


def _action_shape(path: Path) -> tuple[int, int]:
    action = _load_action(path)
    if isinstance(action, np.ndarray):
        if action.ndim != 2:
            return (int(action.shape[0]) if action.ndim >= 1 else 0, -1)
        return int(action.shape[0]), int(action.shape[1])
    if not action:
        return 0, -1
    return len(action), len(action[0])


def _expected_latent_prefix(prefix_frames: int, temporal_compression_factor: int) -> list[int]:
    if temporal_compression_factor <= 0:
        raise ValueError("temporal_compression_factor must be positive")
    latent_prefix = 1 + (int(prefix_frames) - 1) // int(temporal_compression_factor)
    return list(range(max(1, latent_prefix)))


def _check_rows(rows: list[dict[str, Any]], jsonl_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    source_uuids = set()
    scenarios = Counter()
    prefix_roles = Counter()
    prefixes = []
    failures = []
    video_frame_cache: dict[Path, int] = {}
    action_shape_cache: dict[Path, tuple[int, int]] = {}
    state_cache: dict[Path, tuple[int, int, int]] = {}
    summary_cache: dict[Path, dict[str, Any]] = {}
    for row in rows:
        uuid = str(row.get("uuid", ""))
        source_uuid = str(row.get("source_uuid", ""))
        source_uuids.add(source_uuid)
        scenarios[str((row.get("metadata") or {}).get("scenario", row.get("scenario", "unknown")))] += 1
        prefix_roles[str(row.get("prefix_role", "unknown"))] += 1

        def fail(reason: str) -> None:
            failures.append({"uuid": uuid, "reason": reason})

        if "chunk" in uuid.lower() or "chunked" in uuid.lower():
            fail("uuid contains chunk/chunked")
        if int(row.get("num_video_frames", args.expected_video_frames)) not in {args.expected_video_frames}:
            fail(f"num_video_frames mismatch {row.get('num_video_frames')}")
        if int(row.get("num_action_steps", args.expected_action_steps)) not in {args.expected_action_steps}:
            fail(f"num_action_steps mismatch {row.get('num_action_steps')}")
        if int(row.get("action_chunk_size", -1)) != args.expected_action_steps:
            fail(f"action_chunk_size mismatch {row.get('action_chunk_size')}")
        prefix_frames = int(row.get("condition_prefix_frames", 0))
        cond_vision = row.get("condition_frame_indexes_vision")
        if not isinstance(cond_vision, list) or not cond_vision:
            fail("missing condition_frame_indexes_vision prefix mask")
        else:
            cond_vision_int = [int(item) for item in cond_vision]
            expected_cond_vision = _expected_latent_prefix(prefix_frames, args.temporal_compression_factor)
            if cond_vision_int != expected_cond_vision:
                fail(
                    "condition_frame_indexes_vision is not exact causal latent prefix "
                    f"0..{len(expected_cond_vision) - 1} for raw prefix_frames={prefix_frames}"
                )
            expected_latent_frames = 1 + (args.expected_video_frames - 1) // args.temporal_compression_factor
            if any(item >= expected_latent_frames for item in cond_vision_int):
                fail("condition_frame_indexes_vision exceeds full episode latent length")
        if str(row.get("target_object")) != "hole" or str(row.get("tool_object")) != "peg":
            fail("missing target/tool role binding")
        if any("depth" in str(key).lower() for key in row.keys()):
            fail("depth key found in active RGB-only row")

        windows = row.get("t2w_windows") or []
        if len(windows) != 1:
            fail(f"expected one full t2w window, got {len(windows)}")
        else:
            window = windows[0]
            if int(window.get("start_frame", -1)) != 0 or int(window.get("end_frame", -1)) != args.expected_video_frames - 1:
                fail(f"t2w window is not full episode: {window}")
            caption = str(window.get("caption", ""))
            for forbidden in ("target_final_xyz", "target final xyz", "final target pose", "target_final"):
                if forbidden in caption:
                    fail(f"caption leaks forbidden future field {forbidden}")

        video_path = _resolve(str(row["vision_path"]), jsonl_path)
        if not video_path.exists():
            fail(f"missing video {video_path}")
        else:
            if video_path not in video_frame_cache:
                video_frame_cache[video_path] = _video_frames(video_path)
            frames = video_frame_cache[video_path]
            if frames != args.expected_video_frames:
                fail(f"video frames {frames} != {args.expected_video_frames}")

        action_path = _resolve(str(row["action_path"]), jsonl_path)
        if not action_path.exists():
            fail(f"missing action {action_path}")
        else:
            if action_path not in action_shape_cache:
                action_shape_cache[action_path] = _action_shape(action_path)
            action_len, action_dim = action_shape_cache[action_path]
            if action_len != args.expected_action_steps:
                fail(f"action length {action_len} != {args.expected_action_steps}")
            elif action_dim != args.expected_action_dim:
                fail(f"action dim {action_dim} != {args.expected_action_dim}")

        state_path = _resolve(str(row.get("state_target_path") or row.get("task_state_target_path")), jsonl_path)
        if not state_path.exists():
            fail(f"missing state target {state_path}")
        else:
            if state_path not in state_cache:
                payload = _load_json(state_path)
                states = payload.get("states", [])
                names = payload.get("state_vector_names", [])
                state_cache[state_path] = (
                    len(states),
                    len(states[0]) if states else 0,
                    len(names),
                )
            state_len, state_dim, state_name_count = state_cache[state_path]
            if state_len != args.expected_video_frames:
                fail(f"state length {state_len} != {args.expected_video_frames}")
            if state_len and state_dim != state_name_count:
                fail("state dim/name mismatch")

        label_path = _resolve(str(row.get("task_label_path") or row.get("task_switch_label_path")), jsonl_path)
        if not label_path.exists():
            fail(f"missing task label {label_path}")
        else:
            payload = _load_json(label_path)
            summary = payload.get("summary")
            if summary is None and payload.get("summary_path"):
                summary_path = _resolve(str(payload["summary_path"]), jsonl_path)
                if not summary_path.exists():
                    fail(f"missing task summary {summary_path}")
                    summary = {}
                else:
                    if summary_path not in summary_cache:
                        summary_payload = _load_json(summary_path)
                        summary_cache[summary_path] = summary_payload.get("summary", {})
                    summary = summary_cache[summary_path]
            if summary is None:
                summary = {}
            labels = summary.get("frame_labels", [])
            if len(labels) != args.expected_video_frames:
                fail(f"frame label length {len(labels)} != {args.expected_video_frames}")
            if summary.get("target_object") != "hole" or summary.get("tool_object") != "peg":
                fail("label role binding mismatch")

        prefixes.append(int(row.get("condition_prefix_frames", -1)))

    return {
        "rows": len(rows),
        "unique_source_episodes": len(source_uuids),
        "scenario_counts": dict(sorted(scenarios.items())),
        "prefix_role_counts": dict(sorted(prefix_roles.items())),
        "prefix_frame_count_min": min(prefixes) if prefixes else None,
        "prefix_frame_count_max": max(prefixes) if prefixes else None,
        "failures": failures[:100],
        "num_failures": len(failures),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-root", required=True)
    parser.add_argument("--expected-source-episodes", type=int, default=1000)
    parser.add_argument("--expected-video-frames", type=int, default=301)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--expected-action-dim", type=int, default=32)
    parser.add_argument("--temporal-compression-factor", type=int, default=4)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    root = Path(args.condition_root)
    train_jsonl = root / "train" / "video_action_dataset_file.jsonl"
    val_jsonl = root / "val" / "video_action_dataset_file.jsonl"
    train = _read_jsonl(train_jsonl)
    val = _read_jsonl(val_jsonl)
    train_report = _check_rows(train, train_jsonl, args)
    val_report = _check_rows(val, val_jsonl, args)

    all_source = {str(row.get("source_uuid", "")) for row in train + val}
    failures = []
    if len(all_source) != args.expected_source_episodes:
        failures.append(f"unique_source_episodes {len(all_source)} != {args.expected_source_episodes}")
    for name, report in (("train", train_report), ("val", val_report)):
        if report["num_failures"]:
            failures.append(f"{name} row failures={report['num_failures']}")

    manifest = json.loads((root / "manifest.json").read_text())
    if int(manifest.get("num_video_frames", -1)) != args.expected_video_frames:
        failures.append("manifest num_video_frames mismatch")
    if int(manifest.get("num_action_steps", -1)) != args.expected_action_steps:
        failures.append("manifest num_action_steps mismatch")

    report = {
        "condition_root": str(root),
        "strict_alignment_ok": not failures,
        "failures": failures,
        "train": train_report,
        "val": val_report,
        "manifest_contract": manifest.get("contract"),
        "boundary": "Full-episode preflight only; not training or method success evidence.",
    }
    output_json = Path(args.output_json) if args.output_json else root / "full_episode_wam_preflight.json"
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output_md = Path(args.output_md) if args.output_md else root / "full_episode_wam_preflight.md"
    lines = [
        "# Cosmos3 Full-Episode WAM Preflight",
        "",
        f"- condition root: `{root}`",
        f"- strict alignment ok: `{report['strict_alignment_ok']}`",
        f"- failures: `{len(failures)}`",
        f"- train rows: `{train_report['rows']}`",
        f"- val rows: `{val_report['rows']}`",
        f"- unique source episodes: `{len(all_source)}`",
        "",
        "## Prefix Roles",
        "",
    ]
    all_prefix_roles = Counter(train_report["prefix_role_counts"])
    all_prefix_roles.update(val_report["prefix_role_counts"])
    for key, value in sorted(all_prefix_roles.items()):
        lines.append(f"- `{key}`: `{value}`")
    if failures:
        lines += ["", "## Failures", ""]
        lines += [f"- {failure}" for failure in failures]
    output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"strict_alignment_ok": report["strict_alignment_ok"], "output_json": str(output_json)}, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
