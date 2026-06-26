#!/usr/bin/env python3
"""Run a small panel through the real live-receding Cosmos3 interface.

This is the panel wrapper for the planned controller-facing loop:

1. restore a source prefix,
2. build a causal observed-prefix input,
3. run Cosmos for one short robot-action chunk,
4. execute only that chunk in the live simulator,
5. replay only the source target actor motion when the source trajectory used
   external target motion,
6. reobserve real state/video before the next Cosmos call,
7. allow frozen DP only after the live continuability gate passes.

It intentionally does not call the older one-shot
``run_cosmos3_receding_closed_loop.py`` diagnostic.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from panel_contract_utils import panel_contract_rows, panel_full_episode_contract_ok  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--source-h5-root", required=True)
    parser.add_argument("--condition-root", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--dp-manifest", required=True)
    parser.add_argument("--dp-checkpoint", default="")
    parser.add_argument("--dp-state-key", choices=("ema_agent", "agent"), default="ema_agent")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sample-indices", default="")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument(
        "--prefix-role-mode",
        choices=("auto", "manifest"),
        default="auto",
        help="Use live observed-history role inference by default; manifest roles are diagnostic.",
    )
    parser.add_argument(
        "--prefix-start-mode",
        choices=("target_motion_onset", "manifest"),
        default="target_motion_onset",
        help=(
            "target_motion_onset causally detects when the target begins moving. "
            "manifest reuses hand-picked eval prefix frames and is diagnostic only."
        ),
    )
    parser.add_argument(
        "--pretrigger-control-mode",
        choices=("frozen_dp_until_target_motion", "source_restore"),
        default="frozen_dp_until_target_motion",
        help=(
            "How to produce the observed prefix before the first Cosmos call. "
            "source_restore is diagnostic only."
        ),
    )
    parser.add_argument("--min-dynamic-prefix-frame", type=int, default=8)
    parser.add_argument("--target-motion-consecutive-frames", type=int, default=2)
    parser.add_argument("--max-receding-iterations", type=int, default=40)
    parser.add_argument("--action-exec-horizon", type=int, default=8)
    parser.add_argument("--dp-handoff-horizon", type=int, default=32)
    parser.add_argument("--dp-handoff-chunk-horizon", type=int, default=0)
    parser.add_argument("--cosmos-step-handoff-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continuability-stats-json", default="")
    parser.add_argument("--continuability-stats-horizon", type=int, default=32)
    parser.add_argument("--external-target-mode", choices=("source_env_state", "none"), default="source_env_state")
    parser.add_argument("--run-cosmos-inference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--controller-action-source",
        choices=("cosmos_robot_action", "residual_executor", "contact_executor", "candidate_executor"),
        default="cosmos_robot_action",
    )
    parser.add_argument("--executor-checkpoint", default="")
    parser.add_argument("--candidate-outcome-scorer-checkpoint", default="")
    parser.add_argument("--candidate-outcome-scorer-dp-margin", type=float, default=0.0)
    parser.add_argument("--candidate-outcome-scorer-min-progress-delta", type=float, default=-1.0e9)
    parser.add_argument("--candidate-outcome-scorer-min-continuable-prob", type=float, default=0.0)
    parser.add_argument("--candidate-outcome-scorer-min-inserted-prob", type=float, default=0.0)
    parser.add_argument("--candidate-outcome-scorer-score-state-abs-axis-weights", default="")
    parser.add_argument("--candidate-outcome-scorer-score-state-target", default="")
    parser.add_argument("--candidate-executor-short-prefix-steps", default="")
    parser.add_argument("--source-insertion-suffix-bank", default="")
    parser.add_argument("--source-suffix-k", type=int, default=0)
    parser.add_argument("--source-suffix-blends", default="1.0")
    parser.add_argument("--source-suffix-execute-steps", type=int, default=32)
    parser.add_argument("--source-suffix-offsets", default="")
    parser.add_argument("--source-suffix-scenario-match", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--source-suffix-query-x-weight", type=float, default=1.0)
    parser.add_argument("--source-suffix-query-y-weight", type=float, default=2.0)
    parser.add_argument("--source-suffix-query-z-weight", type=float, default=4.0)
    parser.add_argument("--source-suffix-max-distance", type=float, default=-1.0)
    parser.add_argument("--source-suffix-ignore-residual-cap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--executor-residual-scale", type=float, default=1.0)
    parser.add_argument("--clip-live-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-live-state-snapshots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-candidate-action-bank", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--live-progress-interval", type=int, default=0)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--expected-video-frames", type=int, default=301)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--expected-action-dim", type=int, default=32)
    parser.add_argument("--robot-action-dim", type=int, default=7)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def require_compute_step() -> None:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    step_id = os.environ.get("SLURM_STEP_ID", "")
    if not job_id or not step_id or step_id == "extern":
        raise SystemExit(
            "refusing_login_node_execution=true\n"
            "reason=Run live receding panel only inside a compute-node srun step."
        )
    ntasks = int(os.environ.get("SLURM_NTASKS", "1") or "1")
    procid = int(os.environ.get("SLURM_PROCID", "0") or "0")
    if ntasks != 1 or procid != 0:
        raise SystemExit(
            "refusing_multi_task_execution=true\n"
            "reason=Live receding panel must be serialized as exactly one Slurm task."
        )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def slug(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return out.strip("_") or "sample"


def parse_indices(value: str, max_samples: int, available: int) -> list[int]:
    if value.strip():
        indices: list[int] = []
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            idx = int(part)
            if idx < 0 or idx >= available:
                raise ValueError(f"sample index {idx} outside eval manifest range 0..{available - 1}")
            indices.append(idx)
        return indices
    return list(range(min(max(0, int(max_samples)), available)))


def source_uuid_from_sample(sample: dict[str, Any]) -> str:
    for key in ("source_uuid", "uuid", "sample_uuid"):
        value = sample.get(key)
        if isinstance(value, str) and value:
            return value
    extra = sample.get("extra")
    if isinstance(extra, dict):
        for key in ("source_uuid", "uuid", "sample_uuid"):
            value = extra.get(key)
            if isinstance(value, str) and value:
                return value
    raise KeyError(f"sample is missing source uuid: keys={sorted(sample.keys())}")


def source_h5_from_uuid(source_h5_root: Path, source_uuid: str) -> Path:
    base = source_uuid
    suffix = "_traj_0"
    if base.endswith(suffix):
        base = base[: -len(suffix)]
    if not base.endswith(".fix3"):
        raise ValueError(f"source uuid does not map to a fix3 source directory: {source_uuid}")
    h5_name = base.replace(".fix3", "") + ".h5"
    path = source_h5_root / base / h5_name
    if not path.is_file():
        raise FileNotFoundError(f"missing source H5 for {source_uuid}: {path}")
    return path


def sample_name(idx: int, sample: dict[str, Any], source_uuid: str) -> str:
    for key in ("sample_name", "name"):
        value = sample.get(key)
        if isinstance(value, str) and value:
            return value
    scenario = str(sample.get("scenario") or "scenario")
    short_uuid = source_uuid
    if short_uuid.endswith("_traj_0"):
        short_uuid = short_uuid[: -len("_traj_0")]
    return f"eval{idx:02d}_{scenario}_{short_uuid}"


def subprocess_cmd(args: argparse.Namespace, sample: dict[str, Any], sample_idx: int, sample_dir: Path) -> list[str]:
    source_uuid = source_uuid_from_sample(sample)
    source_h5 = source_h5_from_uuid(Path(args.source_h5_root).resolve(), source_uuid)
    prefix_frame = int(sample.get("prefix_frame_index"))
    scenario = str(sample.get("scenario") or "live_dynamic")
    manifest_role = str(sample.get("prefix_role") or "auto")
    prefix_role = "auto" if args.prefix_role_mode == "auto" else manifest_role
    name = sample_name(sample_idx, sample, source_uuid)

    cmd = [
        sys.executable,
        str(ROOT / "scripts/world_model/run_cosmos3_live_receding_loop.py"),
        "--source-h5",
        str(source_h5),
        "--condition-root",
        str(Path(args.condition_root).resolve()),
        "--checkpoint-path",
        str(Path(args.checkpoint_path).resolve()),
        "--config-file",
        str(Path(args.config_file).resolve()),
        "--dp-manifest",
        str(Path(args.dp_manifest).resolve()),
        "--dp-state-key",
        args.dp_state_key,
        "--output-root",
        str(sample_dir),
        "--prefix-frame-index",
        str(prefix_frame),
        "--prefix-start-mode",
        "manual" if args.prefix_start_mode == "manifest" else "target_motion_onset",
        "--pretrigger-control-mode",
        args.pretrigger_control_mode,
        "--min-dynamic-prefix-frame",
        str(args.min_dynamic_prefix_frame),
        "--target-motion-consecutive-frames",
        str(args.target_motion_consecutive_frames),
        "--scenario",
        scenario,
        "--prefix-role",
        prefix_role,
        "--sample-name",
        slug(name),
        "--max-receding-iterations",
        str(args.max_receding_iterations),
        "--action-exec-horizon",
        str(args.action_exec_horizon),
        "--dp-handoff-horizon",
        str(args.dp_handoff_horizon),
        "--dp-handoff-chunk-horizon",
        str(args.dp_handoff_chunk_horizon),
        "--cosmos-step-handoff-gate" if args.cosmos_step_handoff_gate else "--no-cosmos-step-handoff-gate",
        "--external-target-mode",
        args.external_target_mode,
        "--video-fps",
        str(args.video_fps),
        "--expected-video-frames",
        str(args.expected_video_frames),
        "--expected-action-steps",
        str(args.expected_action_steps),
        "--expected-action-dim",
        str(args.expected_action_dim),
        "--robot-action-dim",
        str(args.robot_action_dim),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--live-progress-interval",
        str(args.live_progress_interval),
        "--controller-action-source",
        args.controller_action_source,
    ]
    if args.executor_checkpoint:
        cmd.extend(["--executor-checkpoint", str(Path(args.executor_checkpoint).resolve())])
    if args.candidate_outcome_scorer_checkpoint:
        cmd.extend(
            [
                "--candidate-outcome-scorer-checkpoint",
                str(Path(args.candidate_outcome_scorer_checkpoint).resolve()),
                "--candidate-outcome-scorer-dp-margin",
                str(args.candidate_outcome_scorer_dp_margin),
                "--candidate-outcome-scorer-min-progress-delta",
                str(args.candidate_outcome_scorer_min_progress_delta),
                "--candidate-outcome-scorer-min-continuable-prob",
                str(args.candidate_outcome_scorer_min_continuable_prob),
                "--candidate-outcome-scorer-min-inserted-prob",
                str(args.candidate_outcome_scorer_min_inserted_prob),
            ]
        )
        if args.candidate_outcome_scorer_score_state_abs_axis_weights:
            cmd.extend(
                [
                    "--candidate-outcome-scorer-score-state-abs-axis-weights",
                    args.candidate_outcome_scorer_score_state_abs_axis_weights,
                ]
            )
        if args.candidate_outcome_scorer_score_state_target:
            cmd.extend(
                [
                    "--candidate-outcome-scorer-score-state-target",
                    args.candidate_outcome_scorer_score_state_target,
                ]
            )
    if args.candidate_executor_short_prefix_steps:
        cmd.extend(
            [
                "--candidate-executor-short-prefix-steps",
                args.candidate_executor_short_prefix_steps,
            ]
        )
    if args.source_insertion_suffix_bank and int(args.source_suffix_k) > 0:
        cmd.extend(
            [
                "--source-insertion-suffix-bank",
                str(Path(args.source_insertion_suffix_bank).resolve()),
                "--source-suffix-k",
                str(args.source_suffix_k),
                "--source-suffix-blends",
                args.source_suffix_blends,
                "--source-suffix-execute-steps",
                str(args.source_suffix_execute_steps),
                "--source-suffix-query-x-weight",
                str(args.source_suffix_query_x_weight),
                "--source-suffix-query-y-weight",
                str(args.source_suffix_query_y_weight),
                "--source-suffix-query-z-weight",
                str(args.source_suffix_query_z_weight),
                "--source-suffix-max-distance",
                str(args.source_suffix_max_distance),
                "--source-suffix-scenario-match"
                if args.source_suffix_scenario_match
                else "--no-source-suffix-scenario-match",
                "--source-suffix-ignore-residual-cap"
                if args.source_suffix_ignore_residual_cap
                else "--no-source-suffix-ignore-residual-cap",
            ]
        )
        if args.source_suffix_offsets:
            cmd.extend(["--source-suffix-offsets", args.source_suffix_offsets])
    cmd.extend(["--executor-residual-scale", str(args.executor_residual_scale)])
    if args.dp_checkpoint:
        cmd.extend(["--dp-checkpoint", str(Path(args.dp_checkpoint).resolve())])
    if args.continuability_stats_json:
        cmd.extend(
            [
                "--continuability-stats-json",
                str(Path(args.continuability_stats_json).resolve()),
                "--continuability-stats-horizon",
                str(args.continuability_stats_horizon),
            ]
        )
    cmd.append("--run-cosmos-inference" if args.run_cosmos_inference else "--no-run-cosmos-inference")
    cmd.append("--clip-live-actions" if args.clip_live_actions else "--no-clip-live-actions")
    cmd.append(
        "--save-live-state-snapshots"
        if args.save_live_state_snapshots
        else "--no-save-live-state-snapshots"
    )
    cmd.append(
        "--save-candidate-action-bank"
        if args.save_candidate_action_bank
        else "--no-save-candidate-action-bank"
    )
    return cmd


def summarize_sample(sample_dir: Path, sample_idx: int, sample: dict[str, Any]) -> dict[str, Any]:
    summary_path = sample_dir / "live_receding_loop_summary.json"
    source_uuid = None
    try:
        source_uuid = source_uuid_from_sample(sample)
    except Exception:
        pass
    out: dict[str, Any] = {
        "sample_index": sample_idx,
        "scenario": sample.get("scenario"),
        "manifest_prefix_frame_index": sample.get("prefix_frame_index"),
        "manifest_prefix_role": sample.get("prefix_role"),
        "source_uuid": source_uuid,
        "sample_dir": str(sample_dir),
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "ok": summary_path.exists(),
    }
    if not summary_path.exists():
        return out
    data = read_json(summary_path)
    iterations = data.get("iterations") or []
    roles = [it.get("prefix_role") for it in iterations]
    gates: list[dict[str, Any]] = []
    for it in iterations:
        for key in (
            "pre_controller_continuability_gate",
            "post_cosmos_continuability_gate",
            "after_dp_handoff_continuability_gate",
        ):
            gate = it.get(key)
            if isinstance(gate, dict):
                gates.append(gate)
    dp_steps = 0
    external_frames: list[int] = []
    for it in iterations:
        dp_steps += len(it.get("dp_handoff_steps") or [])
        for step in it.get("executed_steps") or []:
            target = step.get("external_target") if isinstance(step, dict) else None
            if isinstance(target, dict) and target.get("applied"):
                external_frames.append(int(target.get("source_frame")))
        for step in it.get("dp_handoff_steps") or []:
            target = step.get("external_target") if isinstance(step, dict) else None
            if isinstance(target, dict) and target.get("applied"):
                external_frames.append(int(target.get("source_frame")))
    final_eval = data.get("final_eval") or {}
    out.update(
        {
            "ok": True,
            "sample_name": data.get("sample_name"),
            "final_eval": final_eval,
            "final_success": bool(final_eval.get("success", False)),
            "final_peg_head_pos_at_hole": final_eval.get("peg_head_pos_at_hole"),
            "final_prefix_frame_index": data.get("final_prefix_frame_index"),
            "final_observed_frames": data.get("final_observed_frames"),
            "final_observed_video": data.get("final_observed_video"),
            "final_observed_annotated_video": data.get("final_observed_annotated_video"),
            "final_observed_video_inspection": data.get("final_observed_video_inspection"),
            "final_observed_annotated_video_inspection": data.get("final_observed_annotated_video_inspection"),
            "video_file_contract_ok": data.get("video_file_contract_ok"),
            "completed_iterations": data.get("completed_iterations"),
            "full_episode_length_ok": data.get("full_episode_length_ok"),
            "controller_frame_counts": data.get("controller_frame_counts"),
            "wm_active_frame_count": data.get("wm_active_frame_count"),
            "dp_active_frame_count": data.get("dp_active_frame_count"),
            "observed_prefix_roles": roles,
            "continuability_gate_ok_count": sum(1 for gate in gates if gate.get("ok")),
            "continuability_gate_count": len(gates),
            "dp_handoff_executed_steps": dp_steps,
            "external_target_mode": data.get("external_target_mode"),
            "external_target_source_frame_min": min(external_frames) if external_frames else None,
            "external_target_source_frame_max": max(external_frames) if external_frames else None,
            "method_evidence_allowed": False,
        }
    )
    return out


def load_video_frames(path: Path, max_frames: int = 4) -> list[Any]:
    import imageio.v2 as imageio

    frames: list[Any] = []
    reader = imageio.get_reader(path)
    try:
        try:
            total = int(reader.count_frames())
        except Exception:
            total = -1
        if total and total > 0:
            wanted = sorted(set(int(round(x)) for x in [0, (total - 1) / 3, 2 * (total - 1) / 3, total - 1]))
            for idx in wanted[:max_frames]:
                frames.append(reader.get_data(idx))
        else:
            all_frames = [frame for frame in reader]
            if all_frames:
                total = len(all_frames)
                wanted = sorted(set(int(round(x)) for x in [0, (total - 1) / 3, 2 * (total - 1) / 3, total - 1]))
                frames = [all_frames[idx] for idx in wanted[:max_frames]]
    finally:
        reader.close()
    return frames


def write_contact_sheet(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    cell_w = 220
    cell_h = 220
    label_h = 46
    row_gap = 8
    selected = [
        row
        for row in rows
        if row.get("final_observed_annotated_video") or row.get("final_observed_video")
    ]
    if not selected:
        return {"ok": False, "reason": "no videos available"}
    frames_by_row: list[tuple[dict[str, Any], list[Any]]] = []
    max_cols = 0
    for row in selected:
        video_path = row.get("final_observed_annotated_video") or row.get("final_observed_video")
        frames = load_video_frames(Path(str(video_path)))
        if frames:
            frames_by_row.append((row, frames))
            max_cols = max(max_cols, len(frames))
    if not frames_by_row:
        return {"ok": False, "reason": "videos had no readable frames"}

    canvas = Image.new(
        "RGB",
        (max_cols * cell_w, len(frames_by_row) * (cell_h + label_h + row_gap)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row_i, (row, frames) in enumerate(frames_by_row):
        y0 = row_i * (cell_h + label_h + row_gap)
        label = (
            f"{row.get('sample_index')} {row.get('scenario')} "
            f"success={row.get('final_success')} f={row.get('final_prefix_frame_index')} "
            f"frames={row.get('final_observed_frames')} wm={row.get('wm_active_frame_count')} "
            f"dp={row.get('dp_handoff_executed_steps')}"
        )
        draw.text((4, y0 + 4), label[:160], fill=(0, 0, 0))
        for col, frame in enumerate(frames):
            img = Image.fromarray(frame).convert("RGB")
            img.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            x = col * cell_w + (cell_w - img.width) // 2
            y = y0 + label_h + (cell_h - img.height) // 2
            canvas.paste(img, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return {"ok": True, "contact_sheet": str(path), "rows": len(frames_by_row), "columns": max_cols}


def main() -> int:
    args = parse_args()
    require_compute_step()
    if int(args.expected_video_frames) != 301 or int(args.expected_action_steps) != 300:
        raise SystemExit("active contract requires 301 video frames and 300 action steps")
    if int(args.dp_handoff_horizon) > 0 and not args.dp_checkpoint:
        raise SystemExit("--dp-checkpoint is required when --dp-handoff-horizon > 0")
    if args.controller_action_source in {"residual_executor", "contact_executor", "candidate_executor"} and not args.executor_checkpoint:
        raise SystemExit(f"--executor-checkpoint is required for {args.controller_action_source} panel eval")

    eval_root = Path(args.eval_root).resolve()
    manifest_path = eval_root / "eval_input_manifest.json"
    manifest = read_json(manifest_path)
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"eval manifest has no samples: {manifest_path}")
    indices = parse_indices(args.sample_indices, args.max_samples, len(samples))
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    panel_manifest = {
        "boundary": (
            "Corrected full-300 live-receding diagnostic panel. One causal "
            "target-motion detector controls whether the unified controller "
            "keeps frozen DP active or enters Cosmos WM-active short-chunk "
            "rebind mode; frozen DP can resume only after a real C_pi gate. "
            "This small panel is still not full method evidence without "
            "scenario-diverse scale and direct video review."
        ),
        "method_evidence_allowed": False,
        "eval_root": str(eval_root),
        "eval_manifest": str(manifest_path),
        "source_h5_root": str(Path(args.source_h5_root).resolve()),
        "condition_root": str(Path(args.condition_root).resolve()),
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "config_file": str(Path(args.config_file).resolve()),
        "dp_manifest": str(Path(args.dp_manifest).resolve()),
        "dp_checkpoint": str(Path(args.dp_checkpoint).resolve()) if args.dp_checkpoint else None,
        "selected_sample_indices": indices,
        "prefix_role_mode": args.prefix_role_mode,
        "prefix_start_mode": args.prefix_start_mode,
        "pretrigger_control_mode": args.pretrigger_control_mode,
        "min_dynamic_prefix_frame": int(args.min_dynamic_prefix_frame),
        "target_motion_consecutive_frames": int(args.target_motion_consecutive_frames),
        "external_target_mode": args.external_target_mode,
        "run_cosmos_inference": bool(args.run_cosmos_inference),
        "controller_action_source": args.controller_action_source,
        "executor_checkpoint": str(Path(args.executor_checkpoint).resolve()) if args.executor_checkpoint else None,
        "candidate_outcome_scorer_checkpoint": (
            str(Path(args.candidate_outcome_scorer_checkpoint).resolve())
            if args.candidate_outcome_scorer_checkpoint
            else None
        ),
        "candidate_outcome_scorer_dp_margin": float(args.candidate_outcome_scorer_dp_margin),
        "candidate_outcome_scorer_min_progress_delta": float(args.candidate_outcome_scorer_min_progress_delta),
        "candidate_outcome_scorer_min_continuable_prob": float(args.candidate_outcome_scorer_min_continuable_prob),
        "candidate_outcome_scorer_min_inserted_prob": float(args.candidate_outcome_scorer_min_inserted_prob),
        "candidate_executor_short_prefix_steps": args.candidate_executor_short_prefix_steps or None,
        "source_insertion_suffix_bank": (
            str(Path(args.source_insertion_suffix_bank).resolve())
            if args.source_insertion_suffix_bank
            else None
        ),
        "source_suffix_k": int(args.source_suffix_k),
        "source_suffix_blends": args.source_suffix_blends,
        "source_suffix_execute_steps": int(args.source_suffix_execute_steps),
        "source_suffix_offsets": args.source_suffix_offsets or None,
        "source_suffix_scenario_match": bool(args.source_suffix_scenario_match),
        "source_suffix_max_distance": float(args.source_suffix_max_distance),
        "source_suffix_ignore_residual_cap": bool(args.source_suffix_ignore_residual_cap),
        "executor_residual_scale": float(args.executor_residual_scale),
        "max_receding_iterations": int(args.max_receding_iterations),
        "action_exec_horizon": int(args.action_exec_horizon),
        "dp_handoff_horizon": int(args.dp_handoff_horizon),
        "dp_handoff_chunk_horizon": int(args.dp_handoff_chunk_horizon),
        "cosmos_step_handoff_gate": bool(args.cosmos_step_handoff_gate),
        "save_live_state_snapshots": bool(args.save_live_state_snapshots),
        "live_progress_interval": int(args.live_progress_interval),
        "continuability_stats_json": str(Path(args.continuability_stats_json).resolve()) if args.continuability_stats_json else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
    }
    write_json(output_root / "live_receding_panel_manifest.json", panel_manifest)

    sample_results: list[dict[str, Any]] = []
    failed = False
    for sample_idx in indices:
        sample = samples[sample_idx]
        source_uuid = source_uuid_from_sample(sample)
        scenario = str(sample.get("scenario") or "live_dynamic")
        sample_dir = output_root / f"sample_{sample_idx:02d}_{slug(scenario)}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            sample_dir / "selected_sample_manifest.json",
            {
                "sample_index": sample_idx,
                "source_uuid": source_uuid,
                "source_h5": str(source_h5_from_uuid(Path(args.source_h5_root).resolve(), source_uuid)),
                "sample": sample,
            },
        )
        cmd = subprocess_cmd(args, sample, sample_idx, sample_dir)
        (sample_dir / "live_receding_command.txt").write_text(" ".join(cmd) + "\n")
        stdout_path = sample_dir / "live_receding_stdout.log"
        with stdout_path.open("w") as stdout:
            code = subprocess.run(cmd, cwd=str(ROOT), stdout=stdout, stderr=subprocess.STDOUT).returncode
        result = summarize_sample(sample_dir, sample_idx, sample)
        result["exit_code"] = code
        result["stdout_log"] = str(stdout_path)
        if code != 0:
            failed = True
            result["ok"] = False
        sample_results.append(result)
        write_json(sample_dir / "sample_panel_result.json", result)
        if code != 0 and not args.continue_on_error:
            break

    successes = sum(1 for row in sample_results if row.get("final_success"))
    completed = sum(1 for row in sample_results if row.get("ok"))
    contact = write_contact_sheet(sample_results, output_root / "live_receding_panel_contact_sheet.png")
    contract_rows = panel_contract_rows(
        sample_results,
        expected_frames=int(args.expected_video_frames),
        expected_actions=int(args.expected_action_steps),
    )
    sample_contract_failures_rows = [row for row in contract_rows if row["contract_failures"]]
    panel_contract_ok = panel_full_episode_contract_ok(
        contract_rows,
        expected_samples=len(indices),
    )
    summary = {
        **panel_manifest,
        "completed_samples": completed,
        "requested_samples": len(indices),
        "final_success_count": successes,
        "failed_process_count": sum(1 for row in sample_results if row.get("exit_code") not in (0, None)),
        "panel_full_episode_contract_ok": panel_contract_ok,
        "sample_contract_failures": sample_contract_failures_rows,
        "samples": sample_results,
        "contact_sheet": contact,
        "visual_review_status": "needs_direct_agent_or_user_review",
        "method_evidence_allowed": False,
    }
    write_json(output_root / "live_receding_panel_summary.json", summary)
    failed = bool(failed or not panel_contract_ok)
    print(json.dumps({"summary": str(output_root / "live_receding_panel_summary.json"), "failed": failed}, sort_keys=True))
    return 50 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
