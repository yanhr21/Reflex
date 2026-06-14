#!/usr/bin/env python3
"""Objective-level gate for the Cosmos3 full-episode closed-loop interface.

This is a read-only checker for the user-facing closed-loop requirements:
full 300-action/301-frame videos, causal target-motion detector, explicit
controller annotation, no separate static branch, Cosmos usage on moving
targets, same-source pure-DP comparison, and hard-case action/rebind evidence.
It does not run simulation, training, rendering, or inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from video_contract_utils import inspect_video_file


EXPECTED_ACTION_STEPS = 300
EXPECTED_VIDEO_FRAMES = 301


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-cosmos-panel-summary", required=True)
    parser.add_argument("--val-pure-dp-panel-summary", required=True)
    parser.add_argument("--hard-cosmos-panel-summary", required=True)
    parser.add_argument("--hard-action-rebind-analysis", required=True)
    parser.add_argument("--static-no-motion-summary", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    parser.add_argument(
        "--min-moving-wm-active-frames",
        type=int,
        default=8,
        help=(
            "Minimum WM_ACTIVE frames required for moving-target samples. "
            "The default enforces that Cosmos executes at least one short "
            "action chunk, not just a token one-frame activation."
        ),
    )
    parser.add_argument(
        "--min-hard-case-success-fraction",
        type=float,
        default=0.5,
        help=(
            "Minimum Cosmos success fraction on the panel restricted to full "
            "pure-DP failures. This explicitly encodes the requirement that "
            "Cosmos help on most large-motion hard cases; the stricter "
            "hard_case_not_broadly_reliable all-success failure is kept too."
        ),
    )
    parser.add_argument(
        "--skip-video-scan",
        action="store_true",
        help=(
            "Skip direct mp4 frame scanning. Default scans raw/annotated videos "
            "to prove the actual files match the 301-frame evidence contract."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def sample_summary(sample: dict[str, Any]) -> dict[str, Any]:
    summary_path = sample.get("summary_path")
    if isinstance(summary_path, str) and summary_path:
        path = Path(summary_path)
        if path.is_file():
            return read_json(path)
    return sample


def path_exists(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).is_file()


def inspect_video(path_text: Any, *, scan: bool) -> dict[str, Any]:
    out = {
        "path": path_text if isinstance(path_text, str) else None,
        "exists": path_exists(path_text),
        "scan_enabled": bool(scan),
        "decoder": None,
        "decoder_errors": [],
        "scanned_frame_count": None,
        "decoded_frame_count": None,
        "fps": None,
        "duration_seconds": None,
        "error": None,
    }
    if not out["exists"] or not scan:
        return out
    report = inspect_video_file(Path(path_text))
    out.update(
        {
            "decoder": report.get("decoder"),
            "decoder_errors": report.get("decoder_errors") or [],
            "scanned_frame_count": report.get("decoded_frame_count"),
            "decoded_frame_count": report.get("decoded_frame_count"),
            "fps": report.get("fps"),
            "duration_seconds": report.get("duration_seconds"),
            "error": report.get("error"),
        }
    )
    return out


def video_contract_failures(prefix: str, inspection: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if not inspection.get("exists"):
        failures.append(f"missing_{prefix}_video")
        return failures
    if inspection.get("scan_enabled"):
        if inspection.get("error"):
            failures.append(f"{prefix}_video_scan_error")
        elif int(inspection.get("scanned_frame_count") or -1) != EXPECTED_VIDEO_FRAMES:
            failures.append(f"{prefix}_video_frame_count_not_301")
    return failures


def timeline_counts(timeline: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(timeline, list):
        return counts
    for meta in timeline:
        if not isinstance(meta, dict):
            continue
        controller = str(meta.get("controller"))
        counts[controller] = counts.get(controller, 0) + 1
    return counts


def normalized_controller_counts(raw_counts: Any) -> tuple[dict[str, int], bool]:
    if not isinstance(raw_counts, dict) or not raw_counts:
        return {}, False
    counts: dict[str, int] = {}
    try:
        for key, value in raw_counts.items():
            counts[str(key)] = int(value)
    except (TypeError, ValueError):
        return {}, False
    return counts, True


def check_controller_timeline(
    detail: dict[str, Any],
    *,
    require_wm_on_trigger: bool,
    detected_frame_index: Any,
    min_moving_wm_active_frames: int,
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    timeline = detail.get("controller_timeline")
    annotated = detail.get("annotated_video_summary")
    annotated_timeline = annotated.get("timeline") if isinstance(annotated, dict) else None
    expected_counts, expected_counts_valid = normalized_controller_counts(detail.get("controller_frame_counts"))
    annotated_summary_counts, annotated_summary_counts_valid = normalized_controller_counts(
        annotated.get("controller_frame_counts") if isinstance(annotated, dict) else None
    )
    expected_counts_sum = sum(expected_counts.values()) if expected_counts_valid else None
    annotated_summary_counts_sum = (
        sum(annotated_summary_counts.values()) if annotated_summary_counts_valid else None
    )

    timeline_count = len(timeline) if isinstance(timeline, list) else None
    annotated_timeline_count = len(annotated_timeline) if isinstance(annotated_timeline, list) else None
    timeline_controller_counts = timeline_counts(timeline)
    annotated_controller_counts = timeline_counts(annotated_timeline)

    if not expected_counts_valid:
        failures.append("missing_or_invalid_controller_frame_counts")
    elif expected_counts_sum != EXPECTED_VIDEO_FRAMES:
        failures.append("controller_frame_counts_sum_not_301")
    if not isinstance(timeline, list):
        failures.append("missing_controller_timeline")
    elif timeline_count != EXPECTED_VIDEO_FRAMES:
        failures.append("controller_timeline_length_not_301")
    if not isinstance(annotated, dict):
        failures.append("missing_annotated_video_summary")
    else:
        if not annotated_summary_counts_valid:
            failures.append("missing_or_invalid_annotated_controller_frame_counts")
        elif annotated_summary_counts_sum != EXPECTED_VIDEO_FRAMES:
            failures.append("annotated_controller_frame_counts_sum_not_301")
        elif expected_counts_valid and annotated_summary_counts != expected_counts:
            failures.append("annotated_summary_controller_counts_mismatch")
        if int(annotated.get("frame_count") or -1) != EXPECTED_VIDEO_FRAMES:
            failures.append("annotated_video_summary_frame_count_not_301")
        if annotated_timeline_count != EXPECTED_VIDEO_FRAMES:
            failures.append("annotated_video_summary_timeline_length_not_301")

    if expected_counts_valid and timeline_controller_counts != expected_counts:
        failures.append("controller_timeline_counts_mismatch")
    if expected_counts_valid and annotated_controller_counts != expected_counts:
        failures.append("annotated_timeline_counts_mismatch")

    wm_frames = 0
    detected_frames = 0
    bad_wm_frames = 0
    bad_pretrigger_detected = 0
    for meta in annotated_timeline if isinstance(annotated_timeline, list) else []:
        if not isinstance(meta, dict):
            continue
        if bool(meta.get("wm_active")) or meta.get("controller") == "WM_ACTIVE":
            wm_frames += 1
            if meta.get("controller") != "WM_ACTIVE" or not bool(meta.get("wm_active")):
                bad_wm_frames += 1
        if bool(meta.get("target_motion_detected")):
            detected_frames += 1
            if isinstance(detected_frame_index, int) and int(meta.get("frame_index", -1)) < detected_frame_index:
                bad_pretrigger_detected += 1

    if require_wm_on_trigger and wm_frames < int(min_moving_wm_active_frames):
        failures.append("annotated_timeline_missing_required_wm_active")
    if require_wm_on_trigger and detected_frames <= 0:
        failures.append("annotated_timeline_missing_target_motion_detected")
    if not require_wm_on_trigger and wm_frames != 0:
        failures.append("static_annotated_timeline_has_wm_active")
    if not require_wm_on_trigger and detected_frames != 0:
        failures.append("static_annotated_timeline_has_target_motion_detected")
    if bad_wm_frames:
        failures.append("annotated_timeline_inconsistent_wm_frames")
    if bad_pretrigger_detected:
        failures.append("annotated_timeline_detected_before_trigger")

    return failures, {
        "controller_timeline_length": timeline_count,
        "annotated_timeline_length": annotated_timeline_count,
        "expected_controller_counts": expected_counts,
        "expected_controller_count_sum": expected_counts_sum,
        "annotated_summary_controller_counts": annotated_summary_counts,
        "annotated_summary_controller_count_sum": annotated_summary_counts_sum,
        "controller_timeline_counts": timeline_controller_counts,
        "annotated_timeline_counts": annotated_controller_counts,
        "annotated_wm_active_frames": wm_frames,
        "annotated_target_motion_detected_frames": detected_frames,
    }


def check_full_episode_sample(
    sample: dict[str, Any],
    *,
    require_wm_on_trigger: bool,
    scan_videos: bool,
    min_moving_wm_active_frames: int,
) -> dict[str, Any]:
    detail = sample_summary(sample)
    counts = detail.get("controller_frame_counts") or sample.get("controller_frame_counts") or {}
    prefix_selection = detail.get("prefix_selection") or sample.get("prefix_selection") or {}
    triggered = bool(prefix_selection.get("triggered", counts.get("WM_ACTIVE", 0) > 0))
    mode = prefix_selection.get("mode")
    failures: list[str] = []

    if not bool(detail.get("full_episode_length_ok", sample.get("full_episode_length_ok", False))):
        failures.append("full_episode_length_ok_false")
    if int(detail.get("final_prefix_frame_index", sample.get("final_prefix_frame_index", -1)) or -1) != EXPECTED_ACTION_STEPS:
        failures.append("final_prefix_frame_index_not_300")
    if int(detail.get("final_observed_frames", sample.get("final_observed_frames", -1)) or -1) != EXPECTED_VIDEO_FRAMES:
        failures.append("final_observed_frames_not_301")
    if mode == "manual":
        failures.append("manual_prefix_selection_used")
    if require_wm_on_trigger and mode != "target_motion_onset":
        failures.append("moving_sample_not_causal_target_motion_onset")
    if require_wm_on_trigger and not triggered:
        failures.append("moving_sample_detector_not_triggered")
    if require_wm_on_trigger and not isinstance(prefix_selection.get("detected_frame_index"), int):
        failures.append("moving_sample_missing_detected_frame_index")
    if require_wm_on_trigger and not isinstance(prefix_selection.get("first_streak_frame_index"), int):
        failures.append("moving_sample_missing_first_streak_frame_index")
    wm_active_frames = int(counts.get("WM_ACTIVE", 0) or 0)
    if require_wm_on_trigger and wm_active_frames < int(min_moving_wm_active_frames):
        failures.append(f"moving_sample_wm_active_frames_lt_{int(min_moving_wm_active_frames)}")
    timeline_failures, timeline_report = check_controller_timeline(
        detail,
        require_wm_on_trigger=require_wm_on_trigger,
        detected_frame_index=prefix_selection.get("detected_frame_index"),
        min_moving_wm_active_frames=min_moving_wm_active_frames,
    )
    failures.extend(timeline_failures)
    raw_video = inspect_video(detail.get("final_observed_video") or sample.get("final_observed_video"), scan=scan_videos)
    annotated_video = inspect_video(
        detail.get("final_observed_annotated_video") or sample.get("final_observed_annotated_video"),
        scan=scan_videos,
    )
    failures.extend(video_contract_failures("raw", raw_video))
    failures.extend(video_contract_failures("annotated", annotated_video))

    return {
        "sample_index": sample.get("sample_index", detail.get("sample_index")),
        "scenario": sample.get("scenario", detail.get("scenario")),
        "summary_path": sample.get("summary_path"),
        "ok": not failures,
        "failures": failures,
        "full_episode_length_ok": detail.get("full_episode_length_ok", sample.get("full_episode_length_ok")),
        "final_prefix_frame_index": detail.get("final_prefix_frame_index", sample.get("final_prefix_frame_index")),
        "final_observed_frames": detail.get("final_observed_frames", sample.get("final_observed_frames")),
        "prefix_selection_mode": mode,
        "prefix_selection_triggered": triggered,
        "detected_frame_index": prefix_selection.get("detected_frame_index"),
        "first_streak_frame_index": prefix_selection.get("first_streak_frame_index"),
        "controller_frame_counts": counts,
        "wm_active_frames": wm_active_frames,
        "timeline_report": timeline_report,
        "final_success": (detail.get("final_eval") or {}).get("success", sample.get("final_success")),
        "raw_video_inspection": raw_video,
        "annotated_video_inspection": annotated_video,
    }


def check_static_sample(path: Path | None, *, scan_videos: bool) -> dict[str, Any] | None:
    if path is None:
        return None
    detail = read_json(path)
    counts = detail.get("controller_frame_counts") or {}
    prefix_selection = detail.get("prefix_selection") or {}
    failures: list[str] = []
    if not bool(detail.get("full_episode_length_ok")):
        failures.append("full_episode_length_ok_false")
    if int(detail.get("final_prefix_frame_index", -1) or -1) != EXPECTED_ACTION_STEPS:
        failures.append("final_prefix_frame_index_not_300")
    if int(detail.get("final_observed_frames", -1) or -1) != EXPECTED_VIDEO_FRAMES:
        failures.append("final_observed_frames_not_301")
    if bool(prefix_selection.get("triggered")):
        failures.append("static_sample_detector_triggered")
    if prefix_selection.get("mode") not in {
        "target_motion_detector_never_triggered",
        "target_motion_detector_never_triggered_after_terminal_completion",
    }:
        failures.append("static_sample_not_causal_detector_never_triggered")
    if prefix_selection.get("pretrigger_control_mode") != "frozen_dp_until_target_motion":
        failures.append("static_sample_not_same_frozen_dp_until_target_motion_controller")
    if prefix_selection.get("detected_frame_index") is not None:
        failures.append("static_sample_has_detected_frame_index")
    if int(counts.get("WM_ACTIVE", 0) or 0) != 0:
        failures.append("static_sample_used_wm")
    if int(detail.get("wm_active_frame_count", 0) or 0) != 0:
        failures.append("static_sample_wm_active_frame_count_nonzero")
    timeline_failures, timeline_report = check_controller_timeline(
        detail,
        require_wm_on_trigger=False,
        detected_frame_index=prefix_selection.get("detected_frame_index"),
        min_moving_wm_active_frames=0,
    )
    failures.extend(timeline_failures)
    raw_video = inspect_video(detail.get("final_observed_video"), scan=scan_videos)
    annotated_video = inspect_video(detail.get("final_observed_annotated_video"), scan=scan_videos)
    failures.extend(video_contract_failures("raw", raw_video))
    failures.extend(video_contract_failures("annotated", annotated_video))
    return {
        "summary_path": str(path),
        "ok": not failures,
        "failures": failures,
        "scenario": detail.get("scenario"),
        "prefix_selection_mode": prefix_selection.get("mode"),
        "pretrigger_control_mode": prefix_selection.get("pretrigger_control_mode"),
        "prefix_selection_triggered": bool(prefix_selection.get("triggered")),
        "detected_frame_index": prefix_selection.get("detected_frame_index"),
        "controller_frame_counts": counts,
        "wm_active_frame_count": detail.get("wm_active_frame_count"),
        "dp_active_frame_count": detail.get("dp_active_frame_count"),
        "timeline_report": timeline_report,
        "final_success": (detail.get("final_eval") or {}).get("success"),
        "raw_video_inspection": raw_video,
        "annotated_video_inspection": annotated_video,
    }


def check_pure_dp_panel(path: Path, *, scan_videos: bool) -> dict[str, Any]:
    data = read_json(path)
    samples = data.get("samples") or []
    failures: list[str] = []
    rows = []
    for sample in samples:
        counts = sample.get("controller_frame_counts") or {}
        sample_failures: list[str] = []
        if not bool(sample.get("full_episode_length_ok")):
            sample_failures.append("full_episode_length_ok_false")
        if int(counts.get("PURE_DP", 0) or 0) != EXPECTED_ACTION_STEPS:
            sample_failures.append("pure_dp_count_not_300")
        if int(counts.get("WM_ACTIVE", 0) or 0) != 0:
            sample_failures.append("pure_dp_has_wm_active")
        raw_video = inspect_video(sample.get("final_observed_video"), scan=scan_videos)
        annotated_video = inspect_video(sample.get("final_observed_annotated_video"), scan=scan_videos)
        sample_failures.extend(video_contract_failures("raw", raw_video))
        sample_failures.extend(video_contract_failures("annotated", annotated_video))
        rows.append(
            {
                "sample_index": sample.get("sample_index"),
                "scenario": sample.get("scenario"),
                "ok": not sample_failures,
                "failures": sample_failures,
                "final_success": sample.get("final_success"),
                "controller_frame_counts": counts,
                "raw_video_inspection": raw_video,
                "annotated_video_inspection": annotated_video,
            }
        )
        failures.extend(sample_failures)
    return {
        "summary_path": str(path),
        "ok": not failures,
        "sample_count": len(samples),
        "final_success_count": data.get("pure_dp_final_success_count"),
        "samples": rows,
    }


def rollup_video_inspections(*groups: Any) -> dict[str, Any]:
    inspections: list[dict[str, Any]] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if {
                "exists",
                "scan_enabled",
                "scanned_frame_count",
            }.issubset(value.keys()):
                inspections.append(value)
            else:
                for item in value.values():
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    for group in groups:
        collect(group)
    scanned = [item for item in inspections if item.get("scan_enabled")]
    frame_counts = [
        int(item["scanned_frame_count"])
        for item in scanned
        if item.get("scanned_frame_count") is not None
    ]
    durations = [
        float(item["duration_seconds"])
        for item in scanned
        if item.get("duration_seconds") is not None
    ]
    return {
        "video_file_count": len(inspections),
        "scanned_video_file_count": len(scanned),
        "scan_error_count": sum(1 for item in scanned if item.get("error")),
        "all_scanned_videos_have_301_frames": bool(frame_counts) and all(
            count == EXPECTED_VIDEO_FRAMES for count in frame_counts
        ),
        "min_scanned_frame_count": min(frame_counts) if frame_counts else None,
        "max_scanned_frame_count": max(frame_counts) if frame_counts else None,
        "min_duration_seconds": min(durations) if durations else None,
        "max_duration_seconds": max(durations) if durations else None,
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= float(args.min_hard_case_success_fraction) <= 1.0:
        raise SystemExit("--min-hard-case-success-fraction must be in [0, 1]")
    val_cosmos = read_json(Path(args.val_cosmos_panel_summary))
    hard_cosmos = read_json(Path(args.hard_cosmos_panel_summary))
    hard_action = read_json(Path(args.hard_action_rebind_analysis))

    val_samples = [
        check_full_episode_sample(
            sample,
            require_wm_on_trigger=True,
            scan_videos=not args.skip_video_scan,
            min_moving_wm_active_frames=int(args.min_moving_wm_active_frames),
        )
        for sample in (val_cosmos.get("samples") or [])
    ]
    hard_samples = [
        check_full_episode_sample(
            sample,
            require_wm_on_trigger=True,
            scan_videos=not args.skip_video_scan,
            min_moving_wm_active_frames=int(args.min_moving_wm_active_frames),
        )
        for sample in (hard_cosmos.get("samples") or [])
    ]
    pure_dp = check_pure_dp_panel(Path(args.val_pure_dp_panel_summary), scan_videos=not args.skip_video_scan)
    static = (
        check_static_sample(Path(args.static_no_motion_summary), scan_videos=not args.skip_video_scan)
        if args.static_no_motion_summary
        else None
    )

    contract_failures: list[str] = []
    for group_name, panel in (("val_cosmos", val_cosmos), ("hard_cosmos", hard_cosmos)):
        if panel.get("panel_full_episode_contract_ok") is False:
            contract_failures.append(f"{group_name}_panel_full_episode_contract_false")
    for group_name, rows in (("val_cosmos", val_samples), ("hard_cosmos", hard_samples)):
        bad = [row for row in rows if not row["ok"]]
        if bad:
            contract_failures.append(f"{group_name}_sample_contract_failures:{len(bad)}")
    if not pure_dp["ok"]:
        contract_failures.append("pure_dp_contract_failures")
    if read_json(Path(args.val_pure_dp_panel_summary)).get("panel_full_episode_contract_ok") is False:
        contract_failures.append("pure_dp_panel_full_episode_contract_false")
    if static is not None and not static["ok"]:
        contract_failures.append("static_no_motion_contract_failures")
    if int(hard_action.get("sample_count", 0) or 0) <= 0:
        contract_failures.append("missing_hard_action_rebind_analysis_samples")

    val_success = int(val_cosmos.get("final_success_count", 0) or 0)
    val_total = int(val_cosmos.get("completed_samples", len(val_samples)) or len(val_samples))
    val_dp_success = int(pure_dp.get("final_success_count", 0) or 0)
    hard_success = int(hard_cosmos.get("final_success_count", 0) or 0)
    hard_total = int(hard_cosmos.get("completed_samples", len(hard_samples)) or len(hard_samples))
    hard_dp_matched = int(hard_action.get("pure_dp_final_success_count_on_matched", -1))
    video_scan_rollup = rollup_video_inspections(val_samples, hard_samples, pure_dp, static)

    method_effectiveness_failures: list[str] = []
    if val_success < val_dp_success:
        method_effectiveness_failures.append(
            f"val_cosmos_underperforms_same_source_pure_dp:{val_success}/{val_total}<pure_dp:{val_dp_success}/{pure_dp['sample_count']}"
        )
    if hard_success <= 0:
        method_effectiveness_failures.append("hard_cases_no_cosmos_success_on_pure_dp_failures")
    hard_success_fraction = (float(hard_success) / float(hard_total)) if hard_total > 0 else 0.0
    if hard_total <= 0 or hard_success_fraction < float(args.min_hard_case_success_fraction):
        method_effectiveness_failures.append(
            "hard_case_success_fraction_below_minimum:"
            f"{hard_success}/{hard_total}<min_fraction:{float(args.min_hard_case_success_fraction):.3g}"
        )
    if hard_success < hard_total:
        method_effectiveness_failures.append(f"hard_case_not_broadly_reliable:{hard_success}/{hard_total}")
    if hard_dp_matched != 0:
        method_effectiveness_failures.append(
            f"hard_matched_pure_dp_success_count_unexpected:{hard_dp_matched}"
        )

    verdict = {
        "boundary": (
            "Read-only objective gate. Passing the implementation contract means "
            "the closed-loop artifacts satisfy the user's full-length/causal/"
            "annotation/comparison requirements. It does not mean the method is "
            "effective unless method_effectiveness_ok is also true."
        ),
        "implementation_contract_ok": not contract_failures,
        "contract_failures": contract_failures,
        "method_effectiveness_ok": not method_effectiveness_failures,
        "method_effectiveness_failures": method_effectiveness_failures,
        "video_scan_enabled": not args.skip_video_scan,
        "min_moving_wm_active_frames": int(args.min_moving_wm_active_frames),
        "min_hard_case_success_fraction": float(args.min_hard_case_success_fraction),
        "hard_case_success_fraction": hard_success_fraction,
        "video_scan_rollup": video_scan_rollup,
        "next_action_verdict": (
            "do_not_continue_old_checkpoint_as_broad_method_evidence; proceed_to_clean_dense_repair_preflight_when_user_approved"
            if method_effectiveness_failures
            else "current_checkpoint_has_method_evidence_for_this_gate"
        ),
        "val_cosmos": {
            "summary_path": str(Path(args.val_cosmos_panel_summary).resolve()),
            "completed_samples": val_total,
            "final_success_count": val_success,
            "samples": val_samples,
        },
        "val_pure_dp": pure_dp,
        "static_no_motion": static,
        "hard_cosmos_on_pure_dp_failures": {
            "summary_path": str(Path(args.hard_cosmos_panel_summary).resolve()),
            "completed_samples": hard_total,
            "final_success_count": hard_success,
            "samples": hard_samples,
        },
        "hard_action_rebind_analysis": {
            "summary_path": str(Path(args.hard_action_rebind_analysis).resolve()),
            "sample_count": hard_action.get("sample_count"),
            "cosmos_final_success_count": hard_action.get("cosmos_final_success_count"),
            "pure_dp_final_success_count_on_matched": hard_action.get("pure_dp_final_success_count_on_matched"),
        },
    }

    out_json = Path(args.output_json).resolve()
    write_json(out_json, verdict)
    if args.output_md:
        md = [
            "# Cosmos3 Closed-Loop Objective Gate",
            "",
            f"- implementation_contract_ok: `{verdict['implementation_contract_ok']}`",
            f"- method_effectiveness_ok: `{verdict['method_effectiveness_ok']}`",
            f"- next_action_verdict: `{verdict['next_action_verdict']}`",
            f"- val Cosmos: `{val_success}/{val_total}`",
            f"- val pure DP: `{val_dp_success}/{pure_dp['sample_count']}`",
            f"- hard Cosmos on pure-DP failures: `{hard_success}/{hard_total}`",
            f"- hard matched pure-DP success: `{hard_dp_matched}/{hard_action.get('sample_count')}`",
            f"- video scan enabled: `{not args.skip_video_scan}`",
            f"- min moving WM_ACTIVE frames: `{int(args.min_moving_wm_active_frames)}`",
            f"- min hard-case success fraction: `{float(args.min_hard_case_success_fraction)}`",
            f"- hard-case success fraction: `{hard_success_fraction}`",
            f"- scanned video files: `{video_scan_rollup['scanned_video_file_count']}`",
            f"- all scanned videos have 301 frames: `{video_scan_rollup['all_scanned_videos_have_301_frames']}`",
            f"- scanned frame count range: `{video_scan_rollup['min_scanned_frame_count']}..{video_scan_rollup['max_scanned_frame_count']}`",
            f"- scanned duration range seconds: `{video_scan_rollup['min_duration_seconds']}..{video_scan_rollup['max_duration_seconds']}`",
            "",
            "## Contract Failures",
            "",
            *(f"- `{item}`" for item in contract_failures),
            "",
            "## Method Effectiveness Failures",
            "",
            *(f"- `{item}`" for item in method_effectiveness_failures),
            "",
        ]
        Path(args.output_md).resolve().write_text("\n".join(md) + "\n")
    print(json.dumps({
        "implementation_contract_ok": verdict["implementation_contract_ok"],
        "method_effectiveness_ok": verdict["method_effectiveness_ok"],
        "next_action_verdict": verdict["next_action_verdict"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
