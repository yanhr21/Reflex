#!/usr/bin/env python3
"""Requirement-level audit for the Cosmos3 iter2700 closed-loop objective.

This read-only reducer maps the user's concrete closed-loop requirements to
current evidence. It intentionally separates implementation-contract evidence
from method-effectiveness evidence, so a full-length causal rollout cannot be
mistaken for proof that Cosmos improves dynamic task completion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective-gate-json", required=True)
    parser.add_argument("--video-length-audit-json", required=True)
    parser.add_argument("--failure-modes-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def root_by_label(video_audit: dict[str, Any], label: str) -> dict[str, Any]:
    for root in video_audit.get("roots") or []:
        if root.get("label") == label:
            return root
    return {}


def dynamic_samples(gate: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    out.extend(((gate.get("val_cosmos") or {}).get("samples") or []))
    out.extend(((gate.get("hard_cosmos_on_pure_dp_failures") or {}).get("samples") or []))
    return out


def requirement(
    *,
    req_id: str,
    text: str,
    status: str,
    evidence: dict[str, Any],
    conclusion: str,
) -> dict[str, Any]:
    if status not in {"passed", "failed", "partial", "missing"}:
        raise ValueError(f"invalid requirement status {status!r}")
    return {
        "id": req_id,
        "requirement": text,
        "status": status,
        "evidence": evidence,
        "conclusion": conclusion,
    }


def main() -> None:
    args = parse_args()
    gate = read_json(Path(args.objective_gate_json))
    video_audit = read_json(Path(args.video_length_audit_json))
    failure = read_json(Path(args.failure_modes_json))
    samples = dynamic_samples(gate)
    static = gate.get("static_no_motion") or {}
    video_rollup = gate.get("video_scan_rollup") or {}

    old_root = root_by_label(video_audit, "old_iter2100")
    current_roots = [
        root
        for root in video_audit.get("roots") or []
        if str(root.get("label", "")).startswith("iter2700_")
    ]
    current_video_count = sum(int(root.get("video_count", 0) or 0) for root in current_roots)
    current_all_301 = bool(current_roots) and all(
        bool(root.get("all_videos_match_expected_frames")) for root in current_roots
    )

    moving_detector_ok = bool(samples) and all(
        sample.get("prefix_selection_mode") == "target_motion_onset"
        and sample.get("prefix_selection_triggered") is True
        and isinstance(sample.get("detected_frame_index"), int)
        and isinstance(sample.get("first_streak_frame_index"), int)
        for sample in samples
    )
    moving_wm_ok = bool(samples) and all(int(sample.get("wm_active_frames", 0) or 0) >= 8 for sample in samples)
    moving_annotation_ok = bool(samples) and all(
        int(((sample.get("timeline_report") or {}).get("annotated_wm_active_frames") or 0)) >= 8
        and int(((sample.get("timeline_report") or {}).get("annotated_timeline_length") or -1)) == 301
        for sample in samples
    )
    static_detector_ok = (
        bool(static)
        and bool(static.get("ok"))
        and not bool(static.get("prefix_selection_triggered"))
        and static.get("pretrigger_control_mode") == "frozen_dp_until_target_motion"
    )
    static_no_wm_ok = static_detector_ok and int(static.get("wm_active_frame_count", 0) or 0) == 0

    val_cosmos = (gate.get("val_cosmos") or {})
    val_pure = (gate.get("val_pure_dp") or {})
    hard_cosmos = (gate.get("hard_cosmos_on_pure_dp_failures") or {})
    hard_action = (gate.get("hard_action_rebind_analysis") or {})
    val_cosmos_success = int(val_cosmos.get("final_success_count", 0) or 0)
    val_total = int(val_cosmos.get("completed_samples", 0) or 0)
    val_pure_success = int(val_pure.get("final_success_count", 0) or 0)
    val_pure_total = int(val_pure.get("sample_count", 0) or 0)
    hard_success = int(hard_cosmos.get("final_success_count", 0) or 0)
    hard_total = int(hard_cosmos.get("completed_samples", 0) or 0)
    hard_success_fraction = (float(hard_success) / float(hard_total)) if hard_total > 0 else 0.0
    hard_dp_matched = int(hard_action.get("pure_dp_final_success_count_on_matched", -1))

    requirements = [
        requirement(
            req_id="reject_old_short_iter2100",
            text="The user-flagged old iter2100 videos must be treated as incomplete and not as current evidence.",
            status=(
                "passed"
                if old_root
                and not bool(old_root.get("all_videos_match_expected_frames"))
                and int(old_root.get("max_frame_count") or 9999) < 301
                else "failed"
            ),
            evidence={
                "root": old_root.get("root"),
                "video_count": old_root.get("video_count"),
                "frame_count_range": [old_root.get("min_frame_count"), old_root.get("max_frame_count")],
                "duration_seconds_range": [
                    old_root.get("min_duration_seconds"),
                    old_root.get("max_duration_seconds"),
                ],
            },
            conclusion="Old iter2100 is correctly rejected as short-video negative evidence.",
        ),
        requirement(
            req_id="current_full_300_action_301_frame_videos",
            text="Current closed-loop evidence must run the full 300 actions / 301 frames, about 10 seconds at 30 fps.",
            status="passed" if current_all_301 and bool(gate.get("implementation_contract_ok")) else "failed",
            evidence={
                "video_scan_rollup": video_rollup,
                "video_length_audit_current_video_count": current_video_count,
                "current_roots_all_301": current_all_301,
            },
            conclusion="Current iter2700 artifacts satisfy the full-length video contract.",
        ),
        requirement(
            req_id="causal_target_motion_detection",
            text="The controller must not be told a manifest/manual target-motion frame; it must causally detect target motion.",
            status="passed" if moving_detector_ok else "failed",
            evidence={
                "dynamic_sample_count": len(samples),
                "bad_samples": [
                    {
                        "sample_index": sample.get("sample_index"),
                        "scenario": sample.get("scenario"),
                        "prefix_selection_mode": sample.get("prefix_selection_mode"),
                        "detected_frame_index": sample.get("detected_frame_index"),
                        "first_streak_frame_index": sample.get("first_streak_frame_index"),
                    }
                    for sample in samples
                    if not (
                        sample.get("prefix_selection_mode") == "target_motion_onset"
                        and sample.get("prefix_selection_triggered") is True
                        and isinstance(sample.get("detected_frame_index"), int)
                        and isinstance(sample.get("first_streak_frame_index"), int)
                    )
                ],
            },
            conclusion="Current moving samples use causal target-motion detector provenance.",
        ),
        requirement(
            req_id="cosmos_takeover_after_motion",
            text="For moving-target cases, Cosmos must actually take over for action chunks after target motion is detected.",
            status="passed" if moving_wm_ok else "failed",
            evidence={
                "min_required_wm_active_frames": gate.get("min_moving_wm_active_frames"),
                "wm_active_frames": [
                    {
                        "sample_index": sample.get("sample_index"),
                        "scenario": sample.get("scenario"),
                        "wm_active_frames": sample.get("wm_active_frames"),
                        "controller_frame_counts": sample.get("controller_frame_counts"),
                    }
                    for sample in samples
                ],
            },
            conclusion="Cosmos is not a no-op in the current moving-target implementation artifacts.",
        ),
        requirement(
            req_id="explicit_takeover_annotation",
            text="Videos must explicitly mark when Cosmos is active and when DP is active.",
            status="passed" if moving_annotation_ok else "failed",
            evidence={
                "annotated_wm_active_frames": [
                    {
                        "sample_index": sample.get("sample_index"),
                        "scenario": sample.get("scenario"),
                        "annotated_wm_active_frames": (sample.get("timeline_report") or {}).get("annotated_wm_active_frames"),
                        "annotated_timeline_length": (sample.get("timeline_report") or {}).get("annotated_timeline_length"),
                    }
                    for sample in samples
                ],
            },
            conclusion="Current annotated timelines expose WM_ACTIVE and DP modes for moving samples.",
        ),
        requirement(
            req_id="static_no_motion_same_detector_dp_only",
            text=(
                "A no-motion witness must use the same frozen-DP-until-target-motion controller and detector; "
                "if the detector never fires, the controller should remain DP-only."
            ),
            status="passed" if static_no_wm_ok else "failed",
            evidence={
                "no_motion_witness_summary_path": static.get("summary_path"),
                "prefix_selection_mode": static.get("prefix_selection_mode"),
                "pretrigger_control_mode": static.get("pretrigger_control_mode"),
                "prefix_selection_triggered": static.get("prefix_selection_triggered"),
                "controller_frame_counts": static.get("controller_frame_counts"),
                "wm_active_frame_count": static.get("wm_active_frame_count"),
                "final_success": static.get("final_success"),
            },
            conclusion=(
                "Controller-selection behavior is correct for the no-motion witness only if it is produced by "
                "the same detector/controller path; final DP task success remains a separate performance issue."
            ),
        ),
        requirement(
            req_id="dp_handoff_available_but_not_proven_reliable",
            text="After Cosmos brings the peg near the moved hole, frozen DP may resume through a real-state continuability gate.",
            status="partial",
            evidence={
                "dynamic_samples_with_dp_handoff": [
                    {
                        "sample_index": sample.get("sample_index"),
                        "scenario": sample.get("scenario"),
                        "final_success": sample.get("final_success"),
                        "controller_frame_counts": sample.get("controller_frame_counts"),
                    }
                    for sample in samples
                    if int((sample.get("controller_frame_counts") or {}).get("DP_HANDOFF", 0) or 0) > 0
                ],
                "failure_mode_gate_blocks": failure.get("hard_failure_gate_fail_totals"),
            },
            conclusion=(
                "The handoff mechanism exists and can succeed on some samples, but current hard failures show "
                "it is not reliable enough for method evidence."
            ),
        ),
        requirement(
            req_id="method_effectiveness_against_pure_dp",
            text="Cosmos closed-loop should be useful on large-motion dynamic cases, not merely run; compare against full pure DP.",
            status="failed" if not bool(gate.get("method_effectiveness_ok")) else "passed",
            evidence={
                "val_cosmos": f"{val_cosmos_success}/{val_total}",
                "val_pure_dp": f"{val_pure_success}/{val_pure_total}",
                "hard_cosmos_on_pure_dp_failures": f"{hard_success}/{hard_total}",
                "hard_case_success_fraction": hard_success_fraction,
                "min_hard_case_success_fraction": gate.get("min_hard_case_success_fraction"),
                "hard_matched_pure_dp_success": f"{hard_dp_matched}/{hard_action.get('sample_count')}",
                "method_effectiveness_failures": gate.get("method_effectiveness_failures") or [],
                "primary_current_failure": failure.get("primary_current_failure"),
            },
            conclusion=(
                "Current iter2700 is not successful method evidence: val underperforms pure DP and hard "
                "pure-DP failures are only rescued on 1/6 samples."
            ),
        ),
    ]

    status_counts: dict[str, int] = {}
    for item in requirements:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
    current_goal_achieved = bool(requirements) and all(item["status"] == "passed" for item in requirements)
    report = {
        "boundary": (
            "Read-only requirement-level audit of the user's closed-loop objective. "
            "This does not run simulation, rendering, training, or inference."
        ),
        "objective_gate_json": str(Path(args.objective_gate_json).resolve()),
        "video_length_audit_json": str(Path(args.video_length_audit_json).resolve()),
        "failure_modes_json": str(Path(args.failure_modes_json).resolve()),
        "current_goal_achieved": current_goal_achieved,
        "status_counts": status_counts,
        "requirements": requirements,
        "next_action": (
            "Do not mark the objective complete. Continue with the clean-role/dense-receding repair "
            "path and approved overfit/full SFT before claiming Cosmos method effectiveness."
            if not current_goal_achieved
            else "All audited requirements are passed."
        ),
    }
    write_json(Path(args.output_json).resolve(), report)
    if args.output_md:
        lines = [
            "# Cosmos3 Closed-Loop Requirement Audit",
            "",
            f"- current_goal_achieved: `{current_goal_achieved}`",
            f"- status_counts: `{status_counts}`",
            "",
            "## Requirements",
            "",
        ]
        for item in requirements:
            lines.extend(
                [
                    f"### {item['id']}",
                    "",
                    f"- status: `{item['status']}`",
                    f"- requirement: {item['requirement']}",
                    f"- conclusion: {item['conclusion']}",
                    "",
                ]
            )
        lines.extend(["## Next Action", "", report["next_action"], ""])
        Path(args.output_md).resolve().write_text("\n".join(lines) + "\n")
    print(json.dumps({"current_goal_achieved": current_goal_achieved, "status_counts": status_counts}, sort_keys=True))


if __name__ == "__main__":
    main()
