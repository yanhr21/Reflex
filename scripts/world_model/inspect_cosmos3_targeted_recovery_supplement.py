#!/usr/bin/env python3
"""Inspect the targeted recovery supplement before visual approval.

This is a post-generation structural gate for the 2026-06-14 targeted
hard-teacher proposal. It does not approve merging and it does not start SFT.
It only checks whether the generated H5/RGB package is coherent enough for
direct review-sheet inspection.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as handle:
        for line_i, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_i}: invalid JSONL") from exc
    return rows


def _parse_quota_map(raw: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"quota item must be scenario=count, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = int(value)
    return dict(sorted(out.items()))


def _scenario_from_path(path: Path) -> str:
    stem = path.stem
    for scenario in (
        "hole_late_continuous_insert",
        "hole_late_fast_shift",
        "hole_late_constant",
        "hole_late_reverse",
        "hole_late_move_stop",
        "hole_late_sine",
        "peg_disturb",
        "peg_drop",
        "none",
    ):
        if stem.startswith(scenario + "_"):
            return scenario
    return "unknown"


def _load_summary(group: h5py.Group) -> dict[str, Any]:
    raw = group.attrs.get("source_summary_json") or group.attrs.get("summary_json")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not raw:
        return {}
    return json.loads(str(raw))


def _trajectory_group(h5: h5py.File) -> h5py.Group:
    names = sorted([key for key in h5.keys() if key.startswith("traj_")])
    if len(names) != 1:
        raise ValueError(f"expected one traj_* group, found {names}")
    return h5[names[0]]


def _approval_marker(path: Path) -> tuple[bool, str]:
    if not str(path):
        return False, ""
    if not path.exists():
        return False, "missing"
    text = path.read_text().strip()
    if not text:
        return False, "empty"
    try:
        obj = json.loads(text)
        approved = bool(obj.get("visual_review_approved") or obj.get("approved"))
        return approved, "json"
    except json.JSONDecodeError:
        approved = "visual_review_approved=true" in text or "approved=true" in text
        return approved, "text"


def _check_h5_paths(
    paths: list[Path],
    *,
    expected_frames: int,
    expected_actions: int,
    expected_source_kind: str,
    require_regrasp_scenarios: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    failures: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    scenario_counts: Counter[str] = Counter()
    for path in paths:
        if not path.exists():
            failures.append({"path": str(path), "failure": "missing_h5"})
            continue
        scenario_from_path = _scenario_from_path(path)
        try:
            with h5py.File(path, "r") as h5:
                group = _trajectory_group(h5)
                summary = _load_summary(group)
                scenario = str(summary.get("scenario") or scenario_from_path)
                scenario_counts[scenario] += 1
                actions = group.get("actions")
                slots = group.get("slots")
                source_frames = group.get("source_frame_indices")
                if actions is None or int(actions.shape[0]) != expected_actions:
                    failures.append(
                        {
                            "path": str(path),
                            "failure": "bad_action_length",
                            "shape": list(actions.shape) if actions is not None else None,
                        }
                    )
                if source_frames is None or int(source_frames.shape[0]) != expected_frames:
                    failures.append(
                        {
                            "path": str(path),
                            "failure": "bad_source_frame_length",
                            "shape": list(source_frames.shape) if source_frames is not None else None,
                        }
                    )
                if slots is None:
                    failures.append({"path": str(path), "failure": "missing_slots"})
                else:
                    for key in ("hole_pose", "peg_pose", "tcp_pose", "inserted", "grasped"):
                        arr = slots.get(key)
                        if arr is None or int(arr.shape[0]) != expected_frames:
                            failures.append(
                                {
                                    "path": str(path),
                                    "failure": f"bad_slot_{key}_length",
                                    "shape": list(arr.shape) if arr is not None else None,
                                }
                            )
                    inserted = slots.get("inserted")
                    if inserted is not None and len(inserted) and not bool(np.asarray(inserted)[-1]):
                        failures.append({"path": str(path), "failure": "inserted_false_at_end"})
                if expected_source_kind and str(summary.get("source_kind")) != expected_source_kind:
                    failures.append(
                        {
                            "path": str(path),
                            "failure": "wrong_source_kind",
                            "source_kind": summary.get("source_kind"),
                        }
                    )
                if scenario in require_regrasp_scenarios:
                    phases = set(map(str, summary.get("teacher_phases") or []))
                    if not bool(summary.get("post_motion_release_regrasp_done")):
                        failures.append({"path": str(path), "failure": "missing_regrasp_summary_flag"})
                    if "post_motion_release_regrasp" not in phases:
                        failures.append({"path": str(path), "failure": "missing_regrasp_teacher_phase"})
                records.append(
                    {
                        "path": str(path),
                        "scenario": scenario,
                        "source_kind": summary.get("source_kind"),
                        "teacher_phases": summary.get("teacher_phases") or [],
                        "trigger_step": summary.get("trigger_step"),
                        "first_insert_step": summary.get("first_insert_step"),
                        "post_motion_release_regrasp_done": summary.get("post_motion_release_regrasp_done"),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - report all corrupt samples.
            failures.append({"path": str(path), "failure": "h5_read_error", "error": repr(exc)})
    return failures, records, dict(sorted(scenario_counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-root", required=True)
    parser.add_argument("--rgb-root", required=True)
    parser.add_argument("--run-root", default="")
    parser.add_argument("--gap-manifest", default="")
    parser.add_argument("--expected-count", type=int, default=112)
    parser.add_argument(
        "--expected-scenario-quotas",
        default="hole_late_sine=40,hole_late_constant=24,hole_late_continuous_insert=24,hole_late_fast_shift=24",
    )
    parser.add_argument("--expected-source-kind", default="hard_dynamic_teacher_targeted_recovery_gap_20260614")
    parser.add_argument("--expected-width", type=int, default=512)
    parser.add_argument("--expected-height", type=int, default=512)
    parser.add_argument("--expected-fps", type=float, default=30.0)
    parser.add_argument("--expected-frames", type=int, default=301)
    parser.add_argument("--expected-actions", type=int, default=300)
    parser.add_argument("--min-review-sheets", type=int, default=10)
    parser.add_argument("--require-regrasp-scenarios", default="hole_late_fast_shift")
    parser.add_argument("--visual-approval-file", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    h5_root = Path(args.h5_root)
    rgb_root = Path(args.rgb_root)
    run_root = Path(args.run_root) if args.run_root else rgb_root
    quotas = _parse_quota_map(args.expected_scenario_quotas)
    required_regrasp = {item.strip() for item in args.require_regrasp_scenarios.split(",") if item.strip()}

    h5_manifest_path = h5_root / "manifest.json"
    h5_audit_path = h5_root / "source_audit.json"
    rgb_manifest_path = rgb_root / "manifest.json"
    paths_file = h5_root / "fix3_h5_paths.txt"
    h5_manifest = _read_json(h5_manifest_path)
    h5_audit = _read_json(h5_audit_path)
    rgb_manifest = _read_json(rgb_manifest_path)
    gap_manifest = _read_json(Path(args.gap_manifest)) if args.gap_manifest else {}

    paths = []
    if paths_file.exists():
        paths = [Path(line.strip()) for line in paths_file.read_text().splitlines() if line.strip()]
    h5_failures, h5_records, h5_counts_from_files = _check_h5_paths(
        paths,
        expected_frames=int(args.expected_frames),
        expected_actions=int(args.expected_actions),
        expected_source_kind=str(args.expected_source_kind),
        require_regrasp_scenarios=required_regrasp,
    )

    train_rows = _read_jsonl(rgb_root / "train" / "video_dataset_file.jsonl")
    val_rows = _read_jsonl(rgb_root / "val" / "video_dataset_file.jsonl")
    all_rows = [("train", row) for row in train_rows] + [("val", row) for row in val_rows]
    jsonl_counts: Counter[str] = Counter()
    jsonl_failures: list[dict[str, Any]] = []
    for split, row in all_rows:
        metadata = row.get("metadata") or {}
        scenario = str(metadata.get("scenario", "unknown"))
        jsonl_counts[scenario] += 1
        video_path = rgb_root / split / str(row.get("vision_path", ""))
        caption = ""
        windows = row.get("t2w_windows") or []
        if windows:
            caption = str(windows[0].get("caption", ""))
        if not video_path.exists():
            jsonl_failures.append({"uuid": row.get("uuid"), "failure": "missing_video", "path": str(video_path)})
        if int(row.get("width", -1)) != int(args.expected_width) or int(row.get("height", -1)) != int(args.expected_height):
            jsonl_failures.append({"uuid": row.get("uuid"), "failure": "bad_record_size"})
        if abs(float(metadata.get("fps", -1.0)) - float(args.expected_fps)) > 1e-6:
            jsonl_failures.append({"uuid": row.get("uuid"), "failure": "bad_record_fps"})
        if metadata.get("conditioning_policy") != "video_prefix_not_single_image_i2v":
            jsonl_failures.append({"uuid": row.get("uuid"), "failure": "bad_conditioning_policy"})
        if "Robot and object state condition:" not in caption:
            jsonl_failures.append({"uuid": row.get("uuid"), "failure": "missing_state_caption"})
        if not (metadata.get("task_state_condition") or {}):
            jsonl_failures.append({"uuid": row.get("uuid"), "failure": "missing_task_state_metadata"})

    videos = rgb_manifest.get("videos") or []
    video_failures: list[dict[str, Any]] = []
    for item in videos:
        video_path = Path(str(item.get("video", "")))
        input_h5 = Path(str(item.get("input_h5", "")))
        if not video_path.exists():
            video_failures.append({"sample_id": item.get("sample_id"), "failure": "manifest_video_missing", "path": str(video_path)})
        if not input_h5.exists():
            video_failures.append({"sample_id": item.get("sample_id"), "failure": "manifest_input_h5_missing", "path": str(input_h5)})
        if int(item.get("num_video_frames", -1)) != int(args.expected_frames):
            video_failures.append({"sample_id": item.get("sample_id"), "failure": "bad_manifest_frame_count"})
        if int(item.get("num_source_states", -1)) != int(args.expected_frames):
            video_failures.append({"sample_id": item.get("sample_id"), "failure": "bad_manifest_source_state_count"})
        if abs(float(item.get("fps", -1.0)) - float(args.expected_fps)) > 1e-6:
            video_failures.append({"sample_id": item.get("sample_id"), "failure": "bad_manifest_fps"})

    final_mp4s = sorted(path for path in rgb_root.rglob("*.mp4") if not path.name.endswith(".tmp.mp4"))
    tmp_mp4s = sorted(rgb_root.rglob("*.tmp.mp4"))
    review_sheets = sorted((rgb_root / "review_sheets").glob("*.png"))
    h5_audit_failures = h5_audit.get("failures") if isinstance(h5_audit, dict) else None
    manifest_audit_failures = (h5_manifest.get("audit") or {}).get("failures") if isinstance(h5_manifest, dict) else None
    visual_approval_file = Path(args.visual_approval_file) if args.visual_approval_file else Path("")
    visual_approved, visual_approval_format = _approval_marker(visual_approval_file) if args.visual_approval_file else (False, "")

    checks = {
        "h5_manifest_exists": h5_manifest_path.exists(),
        "h5_audit_exists": h5_audit_path.exists(),
        "h5_paths_file_exists": paths_file.exists(),
        "h5_manifest_count_matches": int(h5_manifest.get("num_records", -1)) == int(args.expected_count),
        "h5_paths_count_matches": len(paths) == int(args.expected_count),
        "h5_manifest_source_kind_matches": str(h5_manifest.get("source_kind", "")) == str(args.expected_source_kind),
        "h5_manifest_quotas_match": dict(sorted((h5_manifest.get("scenario_counts") or {}).items())) == quotas,
        "h5_audit_count_matches": int(h5_audit.get("num_paths", -1)) == int(args.expected_count),
        "h5_audit_quotas_match": dict(sorted((h5_audit.get("counts") or {}).items())) == quotas,
        "h5_audit_has_no_failures": h5_audit_failures == [],
        "h5_manifest_audit_has_no_failures": manifest_audit_failures == [],
        "h5_file_scan_has_no_failures": not h5_failures,
        "h5_file_scan_quotas_match": h5_counts_from_files == quotas,
        "rgb_manifest_exists": rgb_manifest_path.exists(),
        "rgb_manifest_count_matches": int(rgb_manifest.get("num_videos", -1)) == int(args.expected_count),
        "rgb_manifest_quotas_match": dict(sorted((rgb_manifest.get("scenario_counts") or {}).items())) == quotas,
        "rgb_manifest_camera_size_matches": int((rgb_manifest.get("camera") or {}).get("width", -1)) == int(args.expected_width)
        and int((rgb_manifest.get("camera") or {}).get("height", -1)) == int(args.expected_height),
        "rgb_manifest_fps_matches": int((rgb_manifest.get("args") or {}).get("fps", -1)) == int(args.expected_fps),
        "rgb_jsonl_count_matches": len(all_rows) == int(args.expected_count),
        "rgb_jsonl_quotas_match": dict(sorted(jsonl_counts.items())) == quotas,
        "rgb_jsonl_has_no_failures": not jsonl_failures,
        "rgb_manifest_videos_have_no_failures": not video_failures,
        "rgb_final_mp4_count_matches": len(final_mp4s) == int(args.expected_count),
        "rgb_tmp_mp4_count_zero": not tmp_mp4s,
        "review_sheet_count_sufficient": len(review_sheets) >= int(args.min_review_sheets),
    }
    structural_ok = all(checks.values())
    ready_for_visual_review = bool(structural_ok)
    ready_for_merge = bool(structural_ok and visual_approved)
    report = {
        "schema": "cosmos3_targeted_recovery_supplement_inspection_v1",
        "h5_root": str(h5_root),
        "rgb_root": str(rgb_root),
        "run_root": str(run_root),
        "gap_manifest": str(args.gap_manifest) if args.gap_manifest else "",
        "gap_summary": {
            "undercovered_query_count": gap_manifest.get("undercovered_query_count"),
            "by_role": gap_manifest.get("by_role"),
            "by_scenario": gap_manifest.get("by_scenario"),
            "plain_interpretation": gap_manifest.get("plain_interpretation"),
        },
        "expected_count": int(args.expected_count),
        "expected_scenario_quotas": quotas,
        "h5_counts_from_files": h5_counts_from_files,
        "rgb_jsonl_counts": dict(sorted(jsonl_counts.items())),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "num_final_mp4s": len(final_mp4s),
        "num_tmp_mp4s": len(tmp_mp4s),
        "num_review_sheets": len(review_sheets),
        "review_sheet_paths": [str(path) for path in review_sheets[: max(0, int(args.min_review_sheets))]],
        "checks": checks,
        "structural_ok": structural_ok,
        "ready_for_visual_review": ready_for_visual_review,
        "visual_approval": {
            "file": str(visual_approval_file) if args.visual_approval_file else "",
            "approved": visual_approved,
            "format": visual_approval_format,
        },
        "ready_for_merge": ready_for_merge,
        "failures": {
            "h5_file_failures": h5_failures[:50],
            "jsonl_failures": jsonl_failures[:50],
            "video_failures": video_failures[:50],
            "tmp_mp4s": [str(path) for path in tmp_mp4s[:20]],
        },
        "sample_h5_records": h5_records[:10],
        "boundary": (
            "This inspection only checks whether the targeted supplement is structurally ready "
            "for direct visual review. Merge/export/SFT still require visual approval and a "
            "separate strict merged preflight."
        ),
    }

    output_json = Path(args.output_json) if args.output_json else run_root / "targeted_recovery_supplement_inspection.json"
    output_md = Path(args.output_md) if args.output_md else run_root / "targeted_recovery_supplement_inspection.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n")

    lines = [
        "# Targeted Recovery Supplement Inspection",
        "",
        f"- h5 root: `{h5_root}`",
        f"- rgb root: `{rgb_root}`",
        f"- expected rows: `{args.expected_count}`",
        f"- structural ok: `{structural_ok}`",
        f"- ready for visual review: `{ready_for_visual_review}`",
        f"- ready for merge: `{ready_for_merge}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Review Sheets", ""])
    for path in report["review_sheet_paths"]:
        lines.append(f"- `{path}`")
    output_md.write_text("\n".join(lines) + "\n")

    print(
        json.dumps(
            {
                "inspection": str(output_json),
                "structural_ok": structural_ok,
                "ready_for_visual_review": ready_for_visual_review,
                "ready_for_merge": ready_for_merge,
            },
            sort_keys=True,
        )
    )
    if not structural_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
