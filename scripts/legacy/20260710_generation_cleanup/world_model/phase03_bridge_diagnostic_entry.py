#!/usr/bin/env python3
"""Create the bridge diagnostic package.

This script is intentionally conservative. It does not execute a controller,
does not run Oracle, and does not edit simulator state. It collects the valid
Phase 01 static DP evidence, Phase 02 RGB Cosmos evidence, and Phase 03 Oracle
action-interface diagnostics into the controller-facing tables Phase 04 needs
before any physical execution claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def require_slurm_step() -> None:
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise SystemExit("refusing_login_node_execution=true; run inside a compute-node srun step")
    if os.environ.get("SLURM_STEP_ID") == "extern":
        raise SystemExit("refusing_extern_step=true; run inside an active compute-node srun step")


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def find_success_trace(phase01_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = read_json(phase01_run / "trace_metrics.json")
    successful = []
    for episode in metrics.get("episodes", []):
        if episode.get("success_at_end") or episode.get("success_once"):
            successful.append(int(episode["episode"]))
    if not successful:
        raise SystemExit(f"no_successful_phase01_episode_found={phase01_run}")
    episode_idx = successful[0]
    trace = read_json(phase01_run / f"episode_{episode_idx:03d}_action_trace.json")
    return metrics, trace.get("steps", [])


def tail_actions(steps: list[dict[str, Any]], n: int) -> list[list[float]]:
    actions = [row["action"] for row in steps if "action" in row]
    return actions[-n:]


def action_summary(actions: list[list[float]]) -> dict[str, Any]:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.size == 0:
        return {"count": 0, "shape": [0, 0]}
    return {
        "count": int(arr.shape[0]),
        "shape": list(arr.shape),
        "min": arr.min(axis=0).astype(float).tolist(),
        "max": arr.max(axis=0).astype(float).tolist(),
        "mean": arr.mean(axis=0).astype(float).tolist(),
        "std": arr.std(axis=0).astype(float).tolist(),
        "abs_max": np.abs(arr).max(axis=0).astype(float).tolist(),
    }


def make_candidate_chunks(success_tail: list[list[float]], global_abs_max: list[float]) -> list[dict[str, Any]]:
    arr = np.asarray(success_tail, dtype=np.float32)
    if arr.size == 0:
        median_tail = np.zeros(7, dtype=np.float32)
    else:
        median_tail = np.median(arr, axis=0)
    limits = np.asarray(global_abs_max if global_abs_max else [1.0] * 7, dtype=np.float32)
    limits = np.maximum(limits, 1e-6)

    gripper_tail = float(median_tail[-1]) if median_tail.shape[0] >= 7 else 0.0
    push = np.zeros(7, dtype=np.float32)
    push[0] = -0.10 * limits[0]
    push[1] = 0.10 * limits[1]
    push[2] = 0.0
    push[3:6] = 0.0
    push[6] = gripper_tail

    retreat = np.zeros(7, dtype=np.float32)
    retreat[0] = 0.10 * limits[0]
    retreat[1] = -0.10 * limits[1]
    retreat[6] = gripper_tail

    hold = np.zeros(7, dtype=np.float32)
    hold[6] = gripper_tail

    return [
        {
            "candidate_id": "dp_success_tail_reference",
            "action_contract": "pd_ee_delta_pose",
            "chunk": success_tail,
            "execution_allowed": False,
            "reason": "Reference only; must be regenerated from live DP observations before execution.",
        },
        {
            "candidate_id": "bounded_insertion_axis_push_template",
            "action_contract": "pd_ee_delta_pose",
            "chunk": [push.astype(float).tolist()],
            "execution_allowed": False,
            "reason": "Template only; no trusted RGB task frame or insertion axis is available yet.",
        },
        {
            "candidate_id": "bounded_retreat_reapproach_template",
            "action_contract": "pd_ee_delta_pose",
            "chunk": [retreat.astype(float).tolist(), push.astype(float).tolist()],
            "execution_allowed": False,
            "reason": "Template only; requires live contact/jam diagnosis and trusted task frame.",
        },
        {
            "candidate_id": "hold_reobserve",
            "action_contract": "pd_ee_delta_pose",
            "chunk": [hold.astype(float).tolist()],
            "execution_allowed": True,
            "reason": "Safe diagnostic candidate; no insertion claim.",
        },
    ]


def summarize_phase02(phase02_run: Path) -> dict[str, Any]:
    videos = sorted(phase02_run.glob("cosmos_outputs/*/vision.mp4"))
    frames = sorted((phase02_run / "cosmos_review_frames").glob("*.png"))
    charts = sorted(phase02_run.glob("state_audit_rgb/*/task_state_chart.csv"))
    review = read_text(phase02_run / "reports" / "visual_review.md")
    classification = read_text(phase02_run / "classification.txt")
    return {
        "phase02_run": str(phase02_run),
        "classification": classification.strip().splitlines(),
        "visual_review_path": str(phase02_run / "reports" / "visual_review.md"),
        "cosmos_video_count": len(videos),
        "review_frame_count": len(frames),
        "diagnostic_chart_count": len(charts),
        "visual_review_mentions_artifacts": "artifact" in review.lower(),
        "charts_are_diagnostic_state_audit": "diagnostic_simulator_state_audit" in review,
        "cosmos_videos": [str(path) for path in videos],
        "review_frames_manifest": str(phase02_run / "cosmos_review_frames" / "frames_manifest.jsonl"),
    }


def summarize_rgb_extractor(extractor_run: Path | None) -> dict[str, Any]:
    if extractor_run is None:
        return {
            "provided": False,
            "ruling": "missing_rgb_extractor_run",
            "reliable": False,
            "reason": "No RGB extractor run was provided to the bridge entry.",
        }
    chart = extractor_run / "rgb_task_state_chart.csv"
    report = extractor_run / "rgb_extractor_report.md"
    counts: dict[str, int] = {}
    total = 0
    if chart.exists():
        with chart.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                status = row.get("extraction_status", "")
                counts[status] = counts.get(status, 0) + 1
    reliable = total > 0 and counts.get("rgb_extracted", 0) == total
    return {
        "provided": True,
        "run": str(extractor_run),
        "chart": str(chart),
        "report": str(report),
        "frame_count": total,
        "status_counts": counts,
        "reliable": reliable,
        "ruling": "rgb_extractor_reliable" if reliable else "rgb_extractor_not_reliable",
    }


def summarize_action_diagnostic(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    comparison = payload.get("cosmos_dynamic_vs_teacher") or {}
    pred_stats = comparison.get("pred_stats") or {}
    teacher_stats = comparison.get("teacher_stats") or {}
    return {
        "path": str(path),
        "run_dir": payload.get("run_dir"),
        "classification": payload.get("classification"),
        "simulator_success_metric": payload.get("simulator_success_metric"),
        "target_motion_complete_before_finisher": payload.get("target_motion_complete_before_finisher"),
        "cosmos_dynamic_action_count": payload.get("cosmos_dynamic_action_count"),
        "finisher_row_count": payload.get("finisher_row_count"),
        "valid_count": comparison.get("valid_count"),
        "rmse_7d": comparison.get("rmse_7d"),
        "rmse_xyz": comparison.get("rmse_xyz"),
        "l2_delta": comparison.get("l2_delta"),
        "mean_xyz_sign_agreement": comparison.get("mean_xyz_sign_agreement"),
        "pred_mean_abs_xyz": (pred_stats.get("mean_abs") or [])[:3],
        "teacher_mean_abs_xyz": (teacher_stats.get("mean_abs") or [])[:3],
    }


def summarize_action_diagnostics(paths: list[Path]) -> dict[str, Any]:
    rows = [summarize_action_diagnostic(path) for path in paths if path.is_file()]
    reverse_failures = [
        row
        for row in rows
        if "h5_reverse" in str(row.get("run_dir"))
        and row.get("simulator_success_metric") is False
        and (row.get("l2_delta") is not None and float(row["l2_delta"]) > 0.0)
    ]
    accepted_continuous = [
        row
        for row in rows
        if "h5_continuous_insert" in str(row.get("run_dir"))
        and row.get("simulator_success_metric") is True
        and (row.get("l2_delta") is not None and float(row["l2_delta"]) <= 0.0)
    ]
    return {
        "provided": bool(rows),
        "diagnostic_count": len(rows),
        "diagnostics": rows,
        "reverse_cosmos_action_trusted": not reverse_failures,
        "accepted_continuous_cosmos_action_reference_count": len(accepted_continuous),
        "reverse_failure_count": len(reverse_failures),
        "ruling": "reverse_cosmos_action_not_trusted" if reverse_failures else "no_reverse_action_failure_loaded",
        "reason": (
            "Loaded reverse action diagnostics show Cosmos dynamic actions increase peg-head distance "
            "and/or drift from source-H5 teacher signs/magnitudes."
            if reverse_failures
            else "No reverse action diagnostic failure was loaded into the bridge gate."
        ),
    }


def read_chart_sources(chart_paths: list[Path]) -> dict[str, list[str]]:
    sources: dict[str, list[str]] = {}
    for chart in chart_paths:
        seen = set()
        with chart.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = row.get("source", "")
                if src:
                    seen.add(src)
        sources[chart.parent.name] = sorted(seen)
    return sources


def annotate_review_frames(phase02_run: Path, out_dir: Path, limit: int) -> list[str]:
    src_dir = phase02_run / "cosmos_review_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for idx, frame_path in enumerate(sorted(src_dir.glob("*.png"))[:limit]):
        image = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        text = "Phase03: extractor not trusted"
        draw.rectangle((4, 4, min(image.width - 4, 4 + 7 * len(text)), 24), fill=(0, 0, 0))
        draw.text((8, 8), text, fill=(255, 255, 255))
        out_path = out_dir / frame_path.name
        image.save(out_path)
        written.append(str(out_path))
    return written


def write_candidate_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate_id",
                "action_contract",
                "chunk_len",
                "execution_allowed",
                "selected_by_trust_gate",
                "rejection_or_selection_reason",
            ],
        )
        writer.writeheader()
        for candidate in candidates:
            selected = candidate["candidate_id"] == "hold_reobserve"
            writer.writerow(
                {
                    "candidate_id": candidate["candidate_id"],
                    "action_contract": candidate["action_contract"],
                    "chunk_len": len(candidate["chunk"]),
                    "execution_allowed": candidate["execution_allowed"],
                    "selected_by_trust_gate": selected,
                    "rejection_or_selection_reason": (
                        "Selected only for reobserve/no-op diagnostic."
                        if selected
                        else candidate["reason"]
                    ),
                }
            )


def write_report(path: Path, manifest: dict[str, Any], trust_gate: dict[str, Any]) -> None:
    lines = [
        "# Phase 04 Bridge Diagnostic Entry",
        "",
        f"Run: `{manifest['run_id']}`",
        "",
        "Status: bridge inputs prepared; no controller execution.",
        "",
        "Ruling:",
        "",
        f"- trust_cosmos: `{trust_gate['trust_cosmos']}`",
        f"- execute_chunk_len: `{trust_gate['execute_chunk_len']}`",
        f"- handoff_mode: `{trust_gate['handoff_mode']}`",
        f"- method_evidence_allowed: `{manifest['method_evidence_allowed']}`",
        "",
        "Reason:",
        "",
        "- Phase 01 provides valid static DP physical success evidence.",
        "- Phase 02 provides nonblank RGB Cosmos videos for six scenarios.",
        "- Phase 02 visual review found gripper/peg/contact artifacts in mid/final frames.",
        "- Phase 03 action-interface diagnostics reject blind reverse Cosmos action execution.",
        "- No deployed RGB task-state extractor has passed yet.",
        "- Therefore insertion candidates are not allowed to execute; the only selected candidate is hold/reobserve.",
        "",
        "Forbidden paths:",
        "",
        "- no Oracle",
        "- no `set_pose`",
        "- no `set_state` or `set_state_dict`",
        "- no source-state restore",
        "- no saved-state replay",
        "- no future simulator labels as method evidence",
        "",
        "Next required work:",
        "",
        "1. implement a real RGB task-state extractor or classify why RGB extraction fails;",
        "2. draw extractor overlays on live/Cosmos RGB frames;",
        "3. rerun this bridge entry to allow candidate scoring only after extractor confidence exists;",
        "4. keep Oracle disabled until task-state extraction, candidate scoring, and trust gate outputs exist.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase01-run", required=True)
    parser.add_argument("--phase02-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", default="04_integration")
    parser.add_argument("--rgb-extractor-run", default="")
    parser.add_argument("--oracle-action-diagnostic", action="append", default=[])
    parser.add_argument("--tail-actions", type=int, default=8)
    parser.add_argument("--overlay-limit", type=int, default=256)
    args = parser.parse_args()

    require_slurm_step()

    phase01_run = Path(args.phase01_run)
    phase02_run = Path(args.phase02_run)
    extractor_run = Path(args.rgb_extractor_run) if args.rgb_extractor_run else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics, success_steps = find_success_trace(phase01_run)
    success_tail = tail_actions(success_steps, args.tail_actions)
    global_abs_max = metrics.get("action_stats", {}).get("abs_max", [])
    candidates = make_candidate_chunks(success_tail, global_abs_max)
    phase02_summary = summarize_phase02(phase02_run)
    extractor_summary = summarize_rgb_extractor(extractor_run)
    action_diagnostics = summarize_action_diagnostics([Path(path) for path in args.oracle_action_diagnostic])
    chart_sources = read_chart_sources(sorted(phase02_run.glob("state_audit_rgb/*/task_state_chart.csv")))
    overlays = annotate_review_frames(phase02_run, out_dir / "cosmos_manual_review_overlays", args.overlay_limit)

    trust_cosmos = bool(extractor_summary["reliable"]) and not bool(
        phase02_summary["visual_review_mentions_artifacts"]
    ) and bool(action_diagnostics["reverse_cosmos_action_trusted"])
    trust_reasons = [
        "Phase02 visual review found generated peg/gripper/contact artifacts.",
        "Diagnostic simulator-state audit charts are not method evidence.",
        "Phase03 Oracle diagnostics are upper-bound references only; they are not method evidence.",
    ]
    if action_diagnostics["provided"] and not action_diagnostics["reverse_cosmos_action_trusted"]:
        trust_reasons.insert(
            0,
            (
                "Reverse Phase03 action diagnostics reject direct Cosmos action execution: "
                f"{action_diagnostics['reverse_failure_count']} loaded reverse failures."
            ),
        )
    elif not action_diagnostics["provided"]:
        trust_reasons.insert(0, "No Phase03 action-interface diagnostic was loaded.")
    if not extractor_summary["provided"]:
        trust_reasons.insert(1, "No deployed RGB task-state extractor run was provided.")
    elif not extractor_summary["reliable"]:
        trust_reasons.insert(
            1,
            f"RGB extractor is not reliable: {extractor_summary['status_counts']}.",
        )
    else:
        trust_reasons.insert(1, "RGB extractor passed its narrow frame-level diagnostic.")

    trust_gate = {
        "trust_cosmos": trust_cosmos,
        "execute_chunk_len": 0,
        "handoff_mode": "hold_reobserve_only",
        "selected_candidate_id": "hold_reobserve",
        "method_evidence_allowed": False,
        "reasons": trust_reasons,
    }

    manifest = {
        "phase": str(args.phase),
        "run_id": args.run_id,
        "output_dir": str(out_dir),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "node": os.uname().nodename,
        "phase01_run": str(phase01_run),
        "phase02_run": str(phase02_run),
        "evidence_type": "offline_bridge_diagnostic_entry_no_controller_execution",
        "controller_execution_used": False,
        "oracle_used": False,
        "method_evidence_allowed": False,
        "forbidden_state_intervention_used": False,
        "action_contract": "pd_ee_delta_pose",
        "phase01_success_tail_action_summary": action_summary(success_tail),
        "phase02_summary": phase02_summary,
        "phase03_action_diagnostics": action_diagnostics,
        "rgb_extractor_summary": extractor_summary,
        "chart_sources": chart_sources,
        "manual_review_overlays": overlays,
        "outputs": {
            "manifest": str(out_dir / "manifest.json"),
            "candidate_chunks": str(out_dir / "candidate_chunks.json"),
            "candidate_table": str(out_dir / "candidate_table.csv"),
            "trust_gate": str(out_dir / "trust_gate.json"),
            "report": str(out_dir / "bridge_entry_report.md"),
        },
    }

    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "candidate_chunks.json", candidates)
    write_candidate_csv(out_dir / "candidate_table.csv", candidates)
    write_json(out_dir / "trust_gate.json", trust_gate)
    write_report(out_dir / "bridge_entry_report.md", manifest, trust_gate)

    print(
        json.dumps(
            {
                "phase04_status": "bridge_entry_created_no_controller_execution",
                "run_id": args.run_id,
                "trust_cosmos": trust_gate["trust_cosmos"],
                "selected_candidate_id": trust_gate["selected_candidate_id"],
                "output_dir": str(out_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
