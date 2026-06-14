#!/usr/bin/env python3
"""Receding Cosmos3 live-loop scaffold for PegInsertionSide.

This script is compute-node only. In dry-run mode it restores a source prefix,
builds the causal prefix-only video and live WAM history needed for one Cosmos
reobservation call, and stops. With ``--run-cosmos-inference`` it can call the
single-prefix wrapper, execute the returned short robot-action chunk, append
real simulator observations to the history, and repeat.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_cosmos3_maniskill_full_episode_wam_conditions import (  # noqa: E402
    FULL_EPISODE_VECTOR_NAMES,
    _load_episode_arrays,
    _raw_vectors_from_arrays,
)
from run_cosmos3_receding_closed_loop import (  # noqa: E402
    _action_space_bounds,
    _get_base_env,
    _import_live_control_stack,
    _load_dp_agent,
    _live_eval,
    _make_live_env,
    _parse_seed_from_text,
    _prepare_step_action,
    _render_frame,
    jsonable,
)
from video_contract_utils import inspect_video_file, video_inspections_match_contract  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-h5", required=True)
    parser.add_argument("--initial-video", default=None)
    parser.add_argument(
        "--prefix-frame-source",
        choices=("render_env_states", "initial_video"),
        default="render_env_states",
        help=(
            "How to build the initial observed prefix. render_env_states replays "
            "source env_states through the live renderer; initial_video reads a "
            "provided prefix-only/reference video."
        ),
    )
    parser.add_argument("--condition-root", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--dp-manifest", required=True)
    parser.add_argument("--dp-checkpoint", default="")
    parser.add_argument(
        "--dp-state-key",
        choices=("ema_agent", "agent"),
        default="ema_agent",
        help="Frozen DP state key used only after the live continuability gate passes.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--prefix-frame-index",
        type=int,
        default=-1,
        help=(
            "Manual diagnostic start frame. Active controller eval should use "
            "--prefix-start-mode=target_motion_onset instead of hand-picking this."
        ),
    )
    parser.add_argument(
        "--prefix-start-mode",
        choices=("manual", "target_motion_onset"),
        default="manual",
        help=(
            "How to choose the first Cosmos prefix. target_motion_onset scans "
            "observed target poses causally and starts only when target motion "
            "is detected; manual is diagnostic/backward-compatible."
        ),
    )
    parser.add_argument("--min-dynamic-prefix-frame", type=int, default=8)
    parser.add_argument("--target-motion-consecutive-frames", type=int, default=2)
    parser.add_argument("--scenario", default="live_dynamic")
    parser.add_argument(
        "--prefix-role",
        default="auto",
        help=(
            "Observed-prefix role. Use auto for live receding evaluation so "
            "the role is recomputed from real history at each reobservation; "
            "fixed values are diagnostic overrides."
        ),
    )
    parser.add_argument("--sample-name", default="live_receding")
    parser.add_argument("--max-receding-iterations", type=int, default=1)
    parser.add_argument("--action-exec-horizon", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--expected-video-frames", type=int, default=301)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--expected-action-dim", type=int, default=32)
    parser.add_argument("--robot-action-dim", type=int, default=7)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument(
        "--full-episode-rollout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When true, keep executing the unified detector/controller loop "
            "until the 300-action episode horizon unless the simulator "
            "terminates/truncates. Success is recorded but does not shorten "
            "the demo video."
        ),
    )
    parser.add_argument(
        "--annotate-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write an annotated rollout video with per-frame controller mode "
            "and target-motion detector state. Raw video is always written."
        ),
    )
    parser.add_argument(
        "--pretrigger-control-mode",
        choices=("frozen_dp_until_target_motion", "source_restore"),
        default="frozen_dp_until_target_motion",
        help=(
            "How to build the observed prefix before the first Cosmos call. "
            "frozen_dp_until_target_motion runs the frozen static DP in the "
            "live env from source frame 0 while replaying only external target "
            "motion, then triggers Cosmos from observed target motion. "
            "source_restore directly restores the source prefix and is "
            "diagnostic only."
        ),
    )
    parser.add_argument(
        "--external-target-mode",
        choices=("source_env_state", "none"),
        default="source_env_state",
        help=(
            "How the exogenous moving target is advanced during live eval. "
            "source_env_state replays only the source H5 target actor pose after "
            "each live robot action, preserving the dynamic task while leaving "
            "robot/peg state live."
        ),
    )
    parser.add_argument("--run-cosmos-inference", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cosmos-wrapper", default="scripts/slurm/run_cosmos3_live_prefix_policy_inference_in_allocation.sh")
    parser.add_argument("--clip-live-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dp-handoff-horizon",
        type=int,
        default=0,
        help=(
            "Optional frozen-DP continuation horizon after a real live state "
            "passes the conservative continuability gate. Zero disables DP."
        ),
    )
    parser.add_argument(
        "--cosmos-step-handoff-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After every executed Cosmos action step, evaluate the real-state "
            "C_pi gate. If it passes, stop the current Cosmos chunk early so "
            "the next observe-decide iteration can immediately hand off a "
            "short chunk to frozen DP. This does not relax C_pi and does not "
            "use generated state as authority."
        ),
    )
    parser.add_argument("--continuability-min-rel-x", type=float, default=-0.08)
    parser.add_argument("--continuability-max-rel-x", type=float, default=0.04)
    parser.add_argument("--continuability-max-abs-y", type=float, default=0.025)
    parser.add_argument("--continuability-max-abs-z", type=float, default=0.025)
    parser.add_argument("--continuability-max-hole-speed", type=float, default=0.01)
    parser.add_argument(
        "--continuability-stats-json",
        default="",
        help=(
            "Optional static-DP success-manifold statistics JSON. When set, "
            "the live DP handoff gate derives distance thresholds from the "
            "specified within-N-steps profile instead of ad-hoc CLI defaults."
        ),
    )
    parser.add_argument(
        "--continuability-stats-horizon",
        type=int,
        default=32,
        help="Use within_<horizon>_steps_to_first_success from the stats JSON.",
    )
    parser.add_argument(
        "--continuability-stats-x-lower-quantile",
        type=float,
        default=0.01,
        help="Lower x quantile used as the far-before-hole DP handoff bound.",
    )
    parser.add_argument(
        "--continuability-stats-abs-quantile",
        type=float,
        default=0.95,
        help="Absolute y/z quantile used as lateral/vertical DP handoff bounds.",
    )
    parser.add_argument(
        "--continuability-stats-set-x-upper",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, also derive max_rel_x from the static-DP x quantile. "
            "By default the CLI safety cap is preserved because closer-than-demo "
            "states can still be safe handoff candidates."
        ),
    )
    parser.add_argument(
        "--continuability-stats-x-upper-quantile",
        type=float,
        default=1.0,
        help="Upper x quantile used only with --continuability-stats-set-x-upper.",
    )
    parser.add_argument("--target-motion-delta-threshold", type=float, default=0.002)
    parser.add_argument("--target-motion-speed-threshold", type=float, default=0.001)
    return parser.parse_args()


def require_compute_step() -> None:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    step_id = os.environ.get("SLURM_STEP_ID", "")
    if not job_id or not step_id or step_id == "extern":
        raise SystemExit(
            "refusing_login_node_execution=true\n"
            "reason=Run live receding loop only inside a compute-node srun step."
        )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, sort_keys=True) + "\n")


def write_action_json(path: Path, raw: np.ndarray) -> None:
    write_json(path, {"action": raw.astype(float).tolist()})


def _lookup_quantile(quantiles: dict[str, Any], requested: float) -> tuple[float, str]:
    if not quantiles:
        raise ValueError("empty quantile table")
    target = float(requested)
    best_key = min(quantiles.keys(), key=lambda key: abs(float(key) - target))
    if abs(float(best_key) - target) > 1e-9:
        raise ValueError(f"quantile {requested} not found; available={sorted(quantiles.keys(), key=float)}")
    value = float(quantiles[best_key])
    if not np.isfinite(value):
        raise ValueError(f"quantile {best_key} is not finite: {value}")
    return value, str(best_key)


def apply_continuability_stats(args: argparse.Namespace) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "source": "cli_thresholds",
        "thresholds": {
            "min_rel_x": float(args.continuability_min_rel_x),
            "max_rel_x": float(args.continuability_max_rel_x),
            "max_abs_y": float(args.continuability_max_abs_y),
            "max_abs_z": float(args.continuability_max_abs_z),
            "max_hole_speed": float(args.continuability_max_hole_speed),
        },
        "profile": getattr(args, "continuability_profile", None),
    }
    if not args.continuability_stats_json:
        return profile

    stats_path = Path(args.continuability_stats_json).resolve()
    data = json.loads(stats_path.read_text())
    horizon = int(args.continuability_stats_horizon)
    key = f"within_{horizon}_steps_to_first_success"
    if key not in data:
        available = sorted(k for k in data if k.startswith("within_"))
        raise ValueError(f"missing continuability stats profile {key}; available={available}")
    stats = data[key]
    min_rel_x, x_lower_key = _lookup_quantile(
        stats.get("x_quantiles", {}),
        float(args.continuability_stats_x_lower_quantile),
    )
    max_abs_y, y_abs_key = _lookup_quantile(
        stats.get("y_abs_quantiles", {}),
        float(args.continuability_stats_abs_quantile),
    )
    max_abs_z, z_abs_key = _lookup_quantile(
        stats.get("z_abs_quantiles", {}),
        float(args.continuability_stats_abs_quantile),
    )
    max_rel_x = float(args.continuability_max_rel_x)
    x_upper_key = None
    if bool(args.continuability_stats_set_x_upper):
        max_rel_x, x_upper_key = _lookup_quantile(
            stats.get("x_quantiles", {}),
            float(args.continuability_stats_x_upper_quantile),
        )

    args.continuability_min_rel_x = min_rel_x
    args.continuability_max_rel_x = max_rel_x
    args.continuability_max_abs_y = max_abs_y
    args.continuability_max_abs_z = max_abs_z

    profile = {
        "source": "static_dp_success_manifold_stats",
        "stats_json": str(stats_path),
        "source_h5": data.get("source_h5"),
        "num_trajectories": data.get("num_trajectories"),
        "success_trajectories": data.get("success_trajectories"),
        "profile_key": key,
        "profile_n": stats.get("n"),
        "quantiles": {
            "x_lower": x_lower_key,
            "abs_y": y_abs_key,
            "abs_z": z_abs_key,
            "x_upper": x_upper_key,
        },
        "x_upper_source": "stats" if x_upper_key is not None else "cli_safety_cap",
        "thresholds": {
            "min_rel_x": float(args.continuability_min_rel_x),
            "max_rel_x": float(args.continuability_max_rel_x),
            "max_abs_y": float(args.continuability_max_abs_y),
            "max_abs_z": float(args.continuability_max_abs_z),
            "max_hole_speed": float(args.continuability_max_hole_speed),
        },
        "boundary": (
            "This profile is derived from states on successful frozen-static-DP "
            "trajectories that are within the requested remaining step horizon "
            "of first simulator success. It is a data-calibrated handoff gate, "
            "not a learned C_pi model and not success evidence by itself."
        ),
    }
    return profile


def target_actor_position_from_env_state(env_state: dict[str, Any]) -> np.ndarray:
    actors = env_state.get("actors") if isinstance(env_state, dict) else None
    if not isinstance(actors, dict) or "box_with_hole" not in actors:
        raise KeyError("source env_state is missing actors/box_with_hole")
    actor_state = np.asarray(actors["box_with_hole"], dtype=np.float32).reshape(-1, 13)[0]
    return actor_state[:3].astype(np.float32)


def select_initial_prefix_frame(env_states: list[Any], args: argparse.Namespace) -> dict[str, Any]:
    max_frame = min(len(env_states), int(args.expected_video_frames)) - 2
    if max_frame < 0:
        raise ValueError(f"not enough env states for prefix selection: {len(env_states)}")
    if args.prefix_start_mode == "manual":
        if int(args.prefix_frame_index) < 0:
            raise ValueError("--prefix-frame-index is required when --prefix-start-mode=manual")
        prefix_frame = max(0, min(int(args.prefix_frame_index), max_frame))
        return {
            "mode": "manual",
            "prefix_frame_index": prefix_frame,
            "requested_prefix_frame_index": int(args.prefix_frame_index),
            "method_evidence_allowed": False,
            "boundary": (
                "Manual prefix selection is diagnostic only. It must not be "
                "reported as dynamic trigger/controller evidence."
            ),
        }

    initial = target_actor_position_from_env_state(env_states[0])
    min_frame = max(1, int(args.min_dynamic_prefix_frame))
    consecutive_required = max(1, int(args.target_motion_consecutive_frames))
    consecutive = 0
    first_streak_frame: int | None = None
    last_pos = initial
    records: list[dict[str, Any]] = []
    for frame in range(1, max_frame + 1):
        pos = target_actor_position_from_env_state(env_states[frame])
        delta = float(np.linalg.norm(pos - initial))
        speed = float(np.linalg.norm(pos - last_pos))
        moving = (
            frame >= min_frame
            and (
                delta >= float(args.target_motion_delta_threshold)
                or speed >= float(args.target_motion_speed_threshold)
            )
        )
        if moving:
            if consecutive == 0:
                first_streak_frame = frame
            consecutive += 1
        else:
            consecutive = 0
            first_streak_frame = None
        if moving or frame in {1, min_frame, max_frame}:
            records.append(
                {
                    "frame": frame,
                    "target_delta": delta,
                    "target_speed": speed,
                    "moving": moving,
                    "consecutive_moving": consecutive,
                }
            )
        if consecutive >= consecutive_required:
            return {
                "mode": "target_motion_onset",
                "prefix_frame_index": frame,
                "detected_frame_index": frame,
                "first_streak_frame_index": first_streak_frame,
                "triggered": True,
                "thresholds": {
                    "min_dynamic_prefix_frame": min_frame,
                    "target_motion_delta": float(args.target_motion_delta_threshold),
                    "target_motion_speed": float(args.target_motion_speed_threshold),
                    "consecutive_frames": consecutive_required,
                },
                "causal_boundary": (
                    "The start frame is the frame where the consecutive-motion "
                    "rule becomes observable from past/current target poses. "
                    "It does not back-date the prefix to the first moving frame."
                ),
                "records_tail": records[-12:],
            }
        last_pos = pos
    raise ValueError(
        "target_motion_onset trigger never fired; do not silently fall back to "
        "a hand-picked prefix for method evidence"
    )


def target_motion_update(
    *,
    frame: int,
    pos: np.ndarray,
    initial_pos: np.ndarray,
    previous_pos: np.ndarray,
    consecutive: int,
    first_streak_frame: int | None,
    args: argparse.Namespace,
) -> tuple[bool, int, int | None, dict[str, Any]]:
    delta = float(np.linalg.norm(pos - initial_pos))
    speed = float(np.linalg.norm(pos - previous_pos))
    moving = (
        frame >= max(1, int(args.min_dynamic_prefix_frame))
        and (
            delta >= float(args.target_motion_delta_threshold)
            or speed >= float(args.target_motion_speed_threshold)
        )
    )
    if moving:
        if consecutive == 0:
            first_streak_frame = int(frame)
        consecutive += 1
    else:
        consecutive = 0
        first_streak_frame = None
    triggered = bool(consecutive >= max(1, int(args.target_motion_consecutive_frames)))
    record = {
        "frame": int(frame),
        "target_delta": delta,
        "target_speed": speed,
        "moving": bool(moving),
        "consecutive_moving": int(consecutive),
        "triggered": triggered,
        "first_streak_frame_index": first_streak_frame,
    }
    return triggered, consecutive, first_streak_frame, record


def future_target_motion_scan(
    *,
    env_states: list[Any],
    start_frame: int,
    initial_pos: np.ndarray,
    previous_pos: np.ndarray,
    consecutive: int,
    first_streak_frame: int | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    max_frame = min(int(args.expected_action_steps), len(env_states) - 1)
    scan_records: list[dict[str, Any]] = []
    prev = np.asarray(previous_pos, dtype=np.float32).copy()
    consecutive_i = int(consecutive)
    first_streak_i = first_streak_frame
    max_delta = 0.0
    max_speed = 0.0
    for frame in range(int(start_frame) + 1, max_frame + 1):
        pos = target_actor_position_from_env_state(env_states[frame])
        triggered, consecutive_i, first_streak_i, record = target_motion_update(
            frame=frame,
            pos=pos,
            initial_pos=initial_pos,
            previous_pos=prev,
            consecutive=consecutive_i,
            first_streak_frame=first_streak_i,
            args=args,
        )
        prev = pos
        max_delta = max(max_delta, float(record["target_delta"]))
        max_speed = max(max_speed, float(record["target_speed"]))
        if record["moving"] or triggered:
            scan_records.append(record)
        if triggered:
            return {
                "would_trigger": True,
                "trigger_frame": int(frame),
                "first_streak_frame_index": first_streak_i,
                "records_tail": scan_records[-12:],
                "max_delta": max_delta,
                "max_speed": max_speed,
            }
    return {
        "would_trigger": False,
        "trigger_frame": None,
        "first_streak_frame_index": None,
        "records_tail": scan_records[-12:],
        "max_delta": max_delta,
        "max_speed": max_speed,
    }


def read_initial_frames(video_path: Path, keep: int) -> list[Any]:
    import imageio.v2 as imageio

    frames: list[Any] = []
    reader = imageio.get_reader(video_path)
    try:
        for frame in reader:
            if len(frames) >= keep:
                break
            frames.append(np.asarray(frame))
    finally:
        reader.close()
    if len(frames) != keep:
        raise ValueError(f"initial video yielded {len(frames)} frames, expected {keep}")
    return frames


def render_prefix_from_env_states(env: Any, base_env: Any, env_states: list[Any], prefix_frame: int) -> list[Any]:
    frames: list[Any] = []
    for frame_idx in range(prefix_frame + 1):
        base_env.set_state_dict(env_states[frame_idx])
        frames.append(_render_frame(env))
    base_env.set_state_dict(env_states[prefix_frame])
    return frames


def write_video(path: Path, frames: list[Any], fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, [np.asarray(frame) for frame in frames], fps=max(1, int(fps)))


def controller_timeline_from_summary(summary: dict[str, Any], frame_count: int) -> list[dict[str, Any]]:
    prefix_selection = summary.get("prefix_selection") or {}
    trigger_frame = prefix_selection.get("detected_frame_index")
    try:
        trigger_frame_int = int(trigger_frame) if trigger_frame is not None else None
    except Exception:
        trigger_frame_int = None
    initial_prefix = summary.get("initial_prefix_frame_index")
    try:
        initial_prefix_int = int(initial_prefix) if initial_prefix is not None else 0
    except Exception:
        initial_prefix_int = 0
    triggered = bool(prefix_selection.get("triggered", False))
    timeline: list[dict[str, Any]] = []
    for frame_idx in range(frame_count):
        if frame_idx == 0:
            controller = "INIT_OBS"
        elif triggered and frame_idx <= initial_prefix_int:
            controller = "DP_SCAN_TARGET"
        elif not triggered:
            controller = "DP_SCAN_TARGET"
        else:
            controller = "UNASSIGNED"
        timeline.append(
            {
                "frame_index": int(frame_idx),
                "controller": controller,
                "target_motion_detected": bool(trigger_frame_int is not None and frame_idx >= trigger_frame_int),
                "target_motion_trigger_frame": trigger_frame_int,
                "wm_active": False,
                "dp_active": controller.startswith("DP"),
                "prefix_role": None,
                "iteration": None,
            }
        )

    for iteration in summary.get("iterations") or []:
        iter_idx = iteration.get("iteration")
        role = iteration.get("prefix_role")
        for step in iteration.get("executed_steps") or []:
            frame_idx = int(step.get("global_action_index", -1)) + 1
            if 0 <= frame_idx < frame_count:
                timeline[frame_idx].update(
                    {
                        "controller": "WM_ACTIVE",
                        "target_motion_detected": True if trigger_frame_int is not None else timeline[frame_idx]["target_motion_detected"],
                        "wm_active": True,
                        "dp_active": False,
                        "prefix_role": role,
                        "iteration": iter_idx,
                    }
                )
        for step in iteration.get("dp_handoff_steps") or []:
            frame_idx = int(step.get("global_action_index", -1)) + 1
            if 0 <= frame_idx < frame_count:
                timeline[frame_idx].update(
                    {
                        "controller": "DP_HANDOFF",
                        "target_motion_detected": True if trigger_frame_int is not None else timeline[frame_idx]["target_motion_detected"],
                        "wm_active": False,
                        "dp_active": True,
                        "prefix_role": role,
                        "iteration": iter_idx,
                    }
                )
    return timeline


def write_annotated_video(path: Path, frames: list[Any], fps: int, summary: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    timeline = controller_timeline_from_summary(summary, len(frames))
    annotated: list[np.ndarray] = []
    for frame, meta in zip(frames, timeline):
        img = Image.fromarray(np.asarray(frame)).convert("RGB")
        draw = ImageDraw.Draw(img)
        controller = str(meta.get("controller"))
        trigger = meta.get("target_motion_trigger_frame")
        role = meta.get("prefix_role") or "-"
        iteration = meta.get("iteration")
        lines = [
            f"frame {meta['frame_index']:03d}/300  controller={controller}",
            f"target_motion_detected={bool(meta.get('target_motion_detected'))}  trigger={trigger if trigger is not None else 'none'}",
            f"wm_active={bool(meta.get('wm_active'))}  dp_active={bool(meta.get('dp_active'))}  iter={iteration if iteration is not None else '-'}  role={role}",
        ]
        text_w = max(draw.textlength(line) for line in lines) + 12
        text_h = 18 * len(lines) + 8
        draw.rectangle((6, 6, 6 + int(text_w), 6 + text_h), fill=(255, 255, 255), outline=(0, 0, 0))
        for i, line in enumerate(lines):
            draw.text((12, 11 + i * 18), line, fill=(0, 0, 0))
        annotated.append(np.asarray(img))
    write_video(path, annotated, fps)
    video_inspection = inspect_video_file(path)
    counts: dict[str, int] = {}
    for meta in timeline:
        controller = str(meta.get("controller"))
        counts[controller] = counts.get(controller, 0) + 1
    return {
        "annotated_video": str(path),
        "frame_count": len(frames),
        "video_inspection": video_inspection,
        "controller_frame_counts": counts,
        "wm_active_frame_count": sum(1 for meta in timeline if meta.get("wm_active")),
        "dp_active_frame_count": sum(1 for meta in timeline if meta.get("dp_active")),
        "target_motion_detected_frame_count": sum(1 for meta in timeline if meta.get("target_motion_detected")),
        "timeline": timeline,
    }


def initialize_history_from_source(source_h5: Path, prefix_frame: int, args: argparse.Namespace) -> np.ndarray:
    export_args = SimpleNamespace(
        total_video_frames=args.expected_video_frames,
        total_action_steps=args.expected_action_steps,
    )
    arrays = _load_episode_arrays(source_h5, export_args)
    raw = _raw_vectors_from_arrays(
        arrays,
        args.expected_video_frames,
        prefix_frame + 1,
        "future_aligned_state",
    )
    if raw.shape != (args.expected_action_steps, args.expected_action_dim):
        raise ValueError(f"raw source history shape {raw.shape} is invalid")
    history = np.zeros_like(raw)
    history[:prefix_frame] = raw[:prefix_frame]
    return history


def empty_history(args: argparse.Namespace) -> np.ndarray:
    history = np.zeros((args.expected_action_steps, args.expected_action_dim), dtype=np.float32)
    denom = max(1, args.expected_action_steps - 1)
    history[:, args.expected_action_dim - 2] = np.arange(args.expected_action_steps, dtype=np.float32) / float(denom)
    return history


def live_pose_row(base_env: Any, stack: dict[str, Any], previous_hole_xyz: np.ndarray | None) -> dict[str, Any]:
    common = stack["common"]
    tcp_pose = common.to_numpy(base_env.agent.tcp.pose.raw_pose)[0].astype(np.float32)
    peg_pose = common.to_numpy(base_env.peg.pose.raw_pose)[0].astype(np.float32)
    hole_pose = common.to_numpy(base_env.box_hole_pose.raw_pose)[0].astype(np.float32)
    peg_head_at_hole = common.to_numpy((base_env.box_hole_pose.inv() * base_env.peg_head_pose).p)[0].astype(np.float32)
    hole_xyz = hole_pose[:3].astype(np.float32)
    if previous_hole_xyz is None:
        hole_velocity = np.zeros((3,), dtype=np.float32)
    else:
        hole_velocity = (hole_xyz - previous_hole_xyz.astype(np.float32)).astype(np.float32)
    grasped = bool(common.to_numpy(base_env.agent.is_grasping(base_env.peg, max_angle=20))[0])
    inserted = bool(_live_eval(base_env)["success"])
    return {
        "tcp_pose": tcp_pose,
        "peg_pose": peg_pose,
        "hole_pose": hole_pose,
        "peg_head_at_hole": peg_head_at_hole,
        "hole_velocity": hole_velocity,
        "grasped": grasped,
        "inserted": inserted,
        "hole_xyz": hole_xyz,
    }


def read_state_obs(base_env: Any, stack: dict[str, Any]) -> np.ndarray:
    common = stack["common"]
    obs = common.to_numpy(base_env.get_obs())
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim == 2:
        return obs[0].astype(np.float32)
    if obs.ndim == 1:
        return obs.astype(np.float32)
    raise RuntimeError(f"unexpected state obs shape {obs.shape}")


def apply_external_target_pose(
    *,
    base_env: Any,
    stack: dict[str, Any],
    env_states: list[Any],
    source_frame: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.external_target_mode == "none":
        return {"applied": False, "mode": "none", "source_frame": int(source_frame)}

    frame = int(source_frame)
    if frame < 0 or frame >= len(env_states):
        raise IndexError(f"external target source frame {frame} outside env state length {len(env_states)}")
    actors = env_states[frame].get("actors") if isinstance(env_states[frame], dict) else None
    if not isinstance(actors, dict) or "box_with_hole" not in actors:
        raise KeyError("source env_state is missing actors/box_with_hole for external target replay")

    actor_state = np.asarray(actors["box_with_hole"], dtype=np.float32).reshape(-1, 13)[0]
    position = actor_state[:3]
    quat = actor_state[3:7]
    torch = stack["torch"]
    from mani_skill.utils.structs import Pose

    p_t = torch.as_tensor(position, device=base_env.device, dtype=base_env.box.pose.p.dtype).view(1, 3)
    q_t = torch.as_tensor(quat, device=base_env.device, dtype=base_env.box.pose.q.dtype).view(1, 4)
    base_env.box.set_pose(Pose.create_from_pq(p_t, q_t))
    return {
        "applied": True,
        "mode": "source_env_state",
        "source_frame": frame,
        "actor": "box_with_hole",
        "target_actor_state_pq": actor_state[:7].astype(float).tolist(),
    }


def fill_live_history_row(
    history: np.ndarray,
    step: int,
    robot_action: np.ndarray,
    live: dict[str, Any],
) -> None:
    denom = max(1, history.shape[0] - 1)
    row = np.zeros((history.shape[1],), dtype=np.float32)
    row[0:7] = np.asarray(robot_action, dtype=np.float32).reshape(-1)[:7]
    row[7:10] = np.asarray(live["tcp_pose"], dtype=np.float32)[:3]
    row[10:13] = np.asarray(live["peg_pose"], dtype=np.float32)[:3]
    row[13:16] = np.asarray(live["hole_pose"], dtype=np.float32)[:3]
    row[16:19] = np.asarray(live["peg_head_at_hole"], dtype=np.float32)[:3]
    row[19:22] = np.asarray(live["hole_velocity"], dtype=np.float32)[:3]
    row[22] = float(bool(live["grasped"]))
    row[23] = float(bool(live["inserted"]))
    row[30] = float(step) / float(denom)
    history[step] = row


def latest_hole_speed(history: np.ndarray, prefix_frame: int) -> float:
    if prefix_frame <= 0 or prefix_frame > history.shape[0]:
        return 0.0
    velocity = np.asarray(history[prefix_frame - 1, 19:22], dtype=np.float32)
    if not np.isfinite(velocity).all():
        return float("inf")
    return float(np.linalg.norm(velocity))


def infer_prefix_role(
    *,
    history: np.ndarray,
    prefix_frame: int,
    scenario: str,
    live: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    end = max(0, min(int(prefix_frame), history.shape[0]))
    role = str(args.prefix_role)
    if role != "auto":
        return {
            "role": role,
            "source": "explicit_diagnostic_override",
            "target_delta": None,
            "hole_speed": None,
            "grasped": bool(live.get("grasped", False)),
        }

    if end <= 0:
        role = "target_pre_motion"
        return {
            "role": role,
            "source": "empty_history_no_target_motion_observed",
            "target_delta": 0.0,
            "hole_speed": 0.0,
            "grasped": bool(live.get("grasped", False)),
        }

    grasp_history = np.asarray(history[:end, 22], dtype=np.float32) if end > 0 else np.asarray([], dtype=np.float32)
    ever_grasped = bool(grasp_history.size and float(np.max(grasp_history)) > 0.5)
    recent_grasped = bool(grasp_history.size and float(np.max(grasp_history[max(0, end - 8) : end])) > 0.5)
    if "peg_drop" in scenario or "peg_disturb" in scenario:
        return {
            "role": "peg_recovery",
            "source": "scenario_peg_perturbation",
            "target_delta": None,
            "hole_speed": latest_hole_speed(history, end),
            "grasped": bool(live.get("grasped", False)),
            "ever_grasped": ever_grasped,
        }
    if ever_grasped and not recent_grasped and not bool(live.get("grasped", False)):
        return {
            "role": "peg_recovery",
            "source": "stable_lost_grasp_after_prior_grasp",
            "target_delta": None,
            "hole_speed": latest_hole_speed(history, end),
            "grasped": False,
            "ever_grasped": True,
            "recent_grasped": False,
        }

    hole = np.asarray(history[:end, 13:16], dtype=np.float32)
    hole_delta = np.linalg.norm(hole - hole[0:1], axis=1) if hole.size else np.asarray([0.0], dtype=np.float32)
    target_delta = float(np.max(hole_delta)) if hole_delta.size else 0.0
    hole_speed = latest_hole_speed(history, end)
    moved = target_delta >= float(args.target_motion_delta_threshold)
    moving_now = hole_speed >= float(args.target_motion_speed_threshold)
    if moving_now:
        role = "target_motion_observed"
        source = "observed_target_velocity"
    elif moved:
        role = "target_post_motion"
        source = "observed_target_motion_settled"
    else:
        role = "target_pre_motion"
        source = "no_target_motion_observed_yet"
    return {
        "role": role,
        "source": source,
        "target_delta": target_delta,
        "hole_speed": hole_speed,
        "grasped": bool(live.get("grasped", False)),
        "ever_grasped": ever_grasped,
        "recent_grasped": recent_grasped,
        "thresholds": {
            "target_motion_delta": float(args.target_motion_delta_threshold),
            "target_motion_speed": float(args.target_motion_speed_threshold),
        },
    }


def continuability_gate(live: dict[str, Any], history: np.ndarray, prefix_frame: int, args: argparse.Namespace) -> dict[str, Any]:
    rel = np.asarray(live["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
    hole_speed = latest_hole_speed(history, prefix_frame)
    checks = {
        "grasped": bool(live.get("grasped", False)),
        "rel_x_min": bool(rel[0] >= float(args.continuability_min_rel_x)),
        "rel_x_max": bool(rel[0] <= float(args.continuability_max_rel_x)),
        "rel_y_abs": bool(abs(float(rel[1])) <= float(args.continuability_max_abs_y)),
        "rel_z_abs": bool(abs(float(rel[2])) <= float(args.continuability_max_abs_z)),
        "hole_speed": bool(hole_speed <= float(args.continuability_max_hole_speed)),
    }
    return {
        "boundary": (
            "Conservative diagnostic C_pi gate from real live state only. "
            "It permits frozen-DP handoff only when the peg is still grasped, "
            "the peg head is close to the current hole frame, and recent "
            "target motion is slow. Passing this gate is not method success "
            "by itself; final live simulator success plus video review remain "
            "the authority."
        ),
        "ok": bool(all(checks.values())),
        "checks": checks,
        "peg_head_at_hole": rel.astype(float).tolist(),
        "hole_speed": hole_speed,
        "thresholds": {
            "min_rel_x": float(args.continuability_min_rel_x),
            "max_rel_x": float(args.continuability_max_rel_x),
            "max_abs_y": float(args.continuability_max_abs_y),
            "max_abs_z": float(args.continuability_max_abs_z),
            "max_hole_speed": float(args.continuability_max_hole_speed),
        },
    }


def write_partial_summary(output_root: Path, summary: dict[str, Any], base_env: Any, prefix_frame: int, observed_frames: list[Any]) -> None:
    partial = dict(summary)
    partial["partial"] = True
    partial["partial_eval"] = _live_eval(base_env)
    partial["partial_prefix_frame_index"] = int(prefix_frame)
    partial["partial_observed_frames"] = len(observed_frames)
    write_json(output_root / "live_receding_loop_partial_summary.json", partial)


def run_prefix_inference(
    *,
    args: argparse.Namespace,
    iter_dir: Path,
    prefix_video: Path,
    history_path: Path,
    prefix_frame: int,
    prefix_role: str,
    iteration: int,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "ROOT": str(ROOT),
            "CONDITION_ROOT": str(Path(args.condition_root).resolve()),
            "CHECKPOINT_PATH": str(Path(args.checkpoint_path).resolve()),
            "CONFIG_FILE": str(Path(args.config_file).resolve()),
            "OUTPUT_ROOT": str(iter_dir / "cosmos_live_prefix"),
            "PREFIX_VIDEO": str(prefix_video),
            "PREFIX_FRAME_INDEX": str(prefix_frame),
            "HISTORY_ACTION_PATH": str(history_path),
            "SAMPLE_NAME": f"{args.sample_name}_iter{iteration:02d}",
            "SCENARIO": args.scenario,
            "PREFIX_ROLE": prefix_role,
            "ACTION_EXEC_HORIZON": str(args.action_exec_horizon),
            "RUN_INFERENCE": "true",
            # The outer shell may carry SOURCE_H5 for the live-loop source
            # trajectory. Per-reobservation Cosmos calls must consume the live
            # history file instead of silently falling back to source rows.
            "SOURCE_H5": "",
        }
    )
    cmd = ["bash", str((ROOT / args.cosmos_wrapper).resolve())]
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    chunk_path = iter_dir / "cosmos_live_prefix" / "live_prefix_action_chunk.json"
    return json.loads(chunk_path.read_text())


def build_source_restore_prefix(
    *,
    env: Any,
    base_env: Any,
    stack: dict[str, Any],
    source_h5: Path,
    env_states: list[Any],
    args: argparse.Namespace,
) -> tuple[int, list[Any], np.ndarray, list[np.ndarray], np.ndarray, dict[str, Any], Any | None, Any | None]:
    prefix_selection = select_initial_prefix_frame(env_states, args)
    prefix_frame = int(prefix_selection["prefix_frame_index"])
    if prefix_frame >= len(env_states):
        raise ValueError(f"prefix_frame {prefix_frame} outside env state length {len(env_states)}")
    base_env.set_state_dict(env_states[prefix_frame])
    if args.prefix_frame_source == "render_env_states":
        observed_frames = render_prefix_from_env_states(env, base_env, env_states, prefix_frame)
    else:
        initial_video = Path(args.initial_video).resolve() if args.initial_video else None
        if initial_video is None:
            raise SystemExit("--initial-video is required when --prefix-frame-source=initial_video")
        observed_frames = read_initial_frames(initial_video, prefix_frame + 1)
    history = initialize_history_from_source(source_h5, prefix_frame, args)
    current_state_obs = read_state_obs(base_env, stack)
    dp_obs_history: list[np.ndarray] = [current_state_obs.copy(), current_state_obs.copy()]
    previous_hole_xyz = live_pose_row(base_env, stack, None)["hole_xyz"]
    prefix_selection = {
        **prefix_selection,
        "pretrigger_control_mode": "source_restore",
        "method_evidence_allowed": False,
        "boundary": (
            "This prefix was restored from source env states rather than "
            "generated by live DP execution. It is diagnostic only."
        ),
    }
    return prefix_frame, observed_frames, history, dp_obs_history, previous_hole_xyz, prefix_selection, None, None


def run_live_dp_until_trigger(
    *,
    env: Any,
    base_env: Any,
    stack: dict[str, Any],
    env_states: list[Any],
    args: argparse.Namespace,
    low: np.ndarray,
    high: np.ndarray,
) -> tuple[int, list[Any], np.ndarray, list[np.ndarray], np.ndarray, dict[str, Any], Any, Any]:
    if not args.dp_checkpoint:
        raise SystemExit("--dp-checkpoint is required for frozen_dp_until_target_motion pretrigger control")
    if args.prefix_frame_source != "render_env_states":
        raise SystemExit("frozen_dp_until_target_motion requires --prefix-frame-source=render_env_states")
    if args.prefix_start_mode != "target_motion_onset":
        raise SystemExit("frozen_dp_until_target_motion currently requires --prefix-start-mode=target_motion_onset")

    torch = stack["torch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp_agent, dp_args = _load_dp_agent(env, stack, args, device)
    dp_device = next(dp_agent.parameters()).device

    base_env.set_state_dict(env_states[0])
    observed_frames: list[Any] = [_render_frame(env)]
    history = empty_history(args)
    current_state_obs = read_state_obs(base_env, stack)
    dp_obs_history: list[np.ndarray] = [current_state_obs.copy(), current_state_obs.copy()]
    initial_live = live_pose_row(base_env, stack, None)
    initial_hole_xyz = initial_live["hole_xyz"].copy()
    previous_hole_xyz = initial_hole_xyz.copy()
    consecutive = 0
    first_streak_frame: int | None = None
    trigger_records: list[dict[str, Any]] = []
    pretrigger_steps: list[dict[str, Any]] = []
    prefix_frame = 0
    dp_call_index = 0

    while prefix_frame < min(args.expected_action_steps, len(env_states) - 1):
        obs_seq = np.stack(dp_obs_history[-2:], axis=0)[None].astype(np.float32)
        obs_tensor = torch.as_tensor(obs_seq, device=dp_device, dtype=torch.float32)
        with torch.no_grad():
            action_seq = dp_agent.get_action(obs_tensor)
        if getattr(dp_args, "sim_backend", "physx_cpu") == "physx_cpu":
            action_seq_np = action_seq.detach().cpu().numpy()
        else:
            action_seq_np = action_seq
        act_horizon = int(action_seq_np.shape[1])
        if act_horizon <= 0:
            raise RuntimeError("frozen DP returned empty pretrigger action sequence")
        for chunk_local_i in range(act_horizon):
            if prefix_frame >= min(args.expected_action_steps, len(env_states) - 1):
                break
            step_action, action_record = _prepare_step_action(
                action_seq_np[:, chunk_local_i],
                low,
                high,
                bool(args.clip_live_actions),
            )
            obs, reward, terminated, truncated, info = env.step(step_action)
            external_target = apply_external_target_pose(
                base_env=base_env,
                stack=stack,
                env_states=env_states,
                source_frame=prefix_frame + 1,
                args=args,
            )
            current_state_obs = read_state_obs(base_env, stack)
            dp_obs_history.append(current_state_obs.copy())
            live = live_pose_row(base_env, stack, previous_hole_xyz)
            fill_live_history_row(history, prefix_frame, np.asarray(action_record["executed"]), live)
            observed_frames.append(_render_frame(env))
            prefix_frame += 1
            triggered, consecutive, first_streak_frame, trigger_record = target_motion_update(
                frame=prefix_frame,
                pos=live["hole_xyz"],
                initial_pos=initial_hole_xyz,
                previous_pos=previous_hole_xyz,
                consecutive=consecutive,
                first_streak_frame=first_streak_frame,
                args=args,
            )
            previous_hole_xyz = live["hole_xyz"]
            if trigger_record["moving"] or prefix_frame in {1, int(args.min_dynamic_prefix_frame)}:
                trigger_records.append(trigger_record)
            pretrigger_steps.append(
                {
                    "global_action_index": int(prefix_frame - 1),
                    "dp_call_index": int(dp_call_index),
                    "chunk_local_step": int(chunk_local_i),
                    "action": action_record,
                    "external_target": external_target,
                    "reward": jsonable(reward),
                    "terminated": jsonable(terminated),
                    "truncated": jsonable(truncated),
                    "live_eval": _live_eval(base_env),
                    "target_motion": trigger_record,
                }
            )
            if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                terminal_record = {
                    "frame_index": int(prefix_frame),
                    "global_action_index": int(prefix_frame - 1),
                    "terminated": jsonable(terminated),
                    "truncated": jsonable(truncated),
                    "live_eval": _live_eval(base_env),
                    "target_motion_at_terminal": trigger_record,
                }
                future_motion = future_target_motion_scan(
                    env_states=env_states,
                    start_frame=prefix_frame,
                    initial_pos=initial_hole_xyz,
                    previous_pos=previous_hole_xyz,
                    consecutive=consecutive,
                    first_streak_frame=first_streak_frame,
                    args=args,
                )
                if (not bool(args.full_episode_rollout)) or bool(future_motion["would_trigger"]):
                    raise RuntimeError(
                        "pretrigger frozen DP rollout terminated before target-motion trigger: "
                        + json.dumps(
                            {
                                "terminal_record": terminal_record,
                                "future_target_motion_scan": future_motion,
                                "full_episode_rollout": bool(args.full_episode_rollout),
                            },
                            sort_keys=True,
                        )
                    )

                zero_action = np.zeros((7,), dtype=np.float32)
                terminal_padding_steps: list[dict[str, Any]] = []
                while prefix_frame < min(args.expected_action_steps, len(env_states) - 1):
                    pad_global_action_index = int(prefix_frame)
                    external_target = apply_external_target_pose(
                        base_env=base_env,
                        stack=stack,
                        env_states=env_states,
                        source_frame=prefix_frame + 1,
                        args=args,
                    )
                    current_state_obs = read_state_obs(base_env, stack)
                    dp_obs_history.append(current_state_obs.copy())
                    live = live_pose_row(base_env, stack, previous_hole_xyz)
                    fill_live_history_row(history, prefix_frame, zero_action, live)
                    observed_frames.append(_render_frame(env))
                    prefix_frame += 1
                    triggered_during_padding, consecutive, first_streak_frame, pad_motion_record = target_motion_update(
                        frame=prefix_frame,
                        pos=live["hole_xyz"],
                        initial_pos=initial_hole_xyz,
                        previous_pos=previous_hole_xyz,
                        consecutive=consecutive,
                        first_streak_frame=first_streak_frame,
                        args=args,
                    )
                    previous_hole_xyz = live["hole_xyz"]
                    if pad_motion_record["moving"] or prefix_frame in {1, int(args.min_dynamic_prefix_frame)}:
                        trigger_records.append(pad_motion_record)
                    terminal_padding_steps.append(
                        {
                            "global_action_index": pad_global_action_index,
                            "dp_call_index": int(dp_call_index),
                            "chunk_local_step": None,
                            "terminal_padding": True,
                            "action": {
                                "raw": zero_action.astype(float).tolist(),
                                "executed": zero_action.astype(float).tolist(),
                                "clipped": False,
                                "within_action_space": True,
                                "max_action_space_violation": 0.0,
                            },
                            "external_target": external_target,
                            "reward": None,
                            "terminated": terminal_record["terminated"],
                            "truncated": terminal_record["truncated"],
                            "live_eval": _live_eval(base_env),
                            "target_motion": pad_motion_record,
                        }
                    )
                    if triggered_during_padding:
                        raise RuntimeError(
                            "target motion triggered during terminal padding despite future no-motion scan"
                        )

                pretrigger_steps.extend(terminal_padding_steps)
                return (
                    int(prefix_frame),
                    observed_frames,
                    history,
                    dp_obs_history,
                    previous_hole_xyz,
                    {
                        "mode": "target_motion_detector_never_triggered_after_terminal_completion",
                        "pretrigger_control_mode": "frozen_dp_until_target_motion",
                        "prefix_frame_index": int(prefix_frame),
                        "detected_frame_index": None,
                        "first_streak_frame_index": None,
                        "triggered": False,
                        "wm_triggered": False,
                        "thresholds": {
                            "min_dynamic_prefix_frame": int(args.min_dynamic_prefix_frame),
                            "target_motion_delta": float(args.target_motion_delta_threshold),
                            "target_motion_speed": float(args.target_motion_speed_threshold),
                            "consecutive_frames": int(args.target_motion_consecutive_frames),
                        },
                        "causal_boundary": (
                            "The same causal target-motion detector was used for the "
                            "entire rollout. Frozen DP reached a terminal state before "
                            "any target motion was observed, and a scan of the remaining "
                            "source target poses showed the detector would still never "
                            "fire. The final live state was therefore held to the "
                            "300-action/301-frame evidence contract without entering a "
                            "separate static-sample branch or invoking Cosmos."
                        ),
                        "pretrigger_dp_steps": len(pretrigger_steps),
                        "terminal_before_target_motion": terminal_record,
                        "terminal_padding_steps": len(terminal_padding_steps),
                        "future_target_motion_scan_after_terminal": future_motion,
                        "records_tail": trigger_records[-12:],
                    },
                    dp_agent,
                    dp_args,
                )
            if triggered:
                prefix_selection = {
                    "mode": "target_motion_onset",
                    "pretrigger_control_mode": "frozen_dp_until_target_motion",
                    "prefix_frame_index": int(prefix_frame),
                    "detected_frame_index": int(prefix_frame),
                    "first_streak_frame_index": first_streak_frame,
                    "triggered": True,
                    "wm_triggered": True,
                    "thresholds": {
                        "min_dynamic_prefix_frame": int(args.min_dynamic_prefix_frame),
                        "target_motion_delta": float(args.target_motion_delta_threshold),
                        "target_motion_speed": float(args.target_motion_speed_threshold),
                        "consecutive_frames": int(args.target_motion_consecutive_frames),
                    },
                    "causal_boundary": (
                        "The prefix was produced by live frozen-DP execution "
                        "from source frame 0 while only the target actor pose "
                        "was externally replayed from the source trajectory. "
                        "Cosmos starts at the first frame where target motion "
                        "is observable under the consecutive-frame rule."
                    ),
                    "pretrigger_dp_steps": len(pretrigger_steps),
                    "records_tail": trigger_records[-12:],
                }
                return (
                    int(prefix_frame),
                    observed_frames,
                    history,
                    dp_obs_history,
                    previous_hole_xyz,
                    prefix_selection,
                    dp_agent,
                    dp_args,
                )
        dp_call_index += 1
    return (
        int(prefix_frame),
        observed_frames,
        history,
        dp_obs_history,
        previous_hole_xyz,
        {
            "mode": "target_motion_detector_never_triggered",
            "pretrigger_control_mode": "frozen_dp_until_target_motion",
            "prefix_frame_index": int(prefix_frame),
            "detected_frame_index": None,
            "first_streak_frame_index": None,
            "triggered": False,
            "wm_triggered": False,
            "thresholds": {
                "min_dynamic_prefix_frame": int(args.min_dynamic_prefix_frame),
                "target_motion_delta": float(args.target_motion_delta_threshold),
                "target_motion_speed": float(args.target_motion_speed_threshold),
                "consecutive_frames": int(args.target_motion_consecutive_frames),
            },
            "causal_boundary": (
                "The same causal target-motion detector was used for the "
                "entire rollout. It never fired before the 300-action horizon, "
                "so the unified controller never entered WM-active mode and "
                "frozen DP produced the full episode. This is not a separate "
                "static-sample branch."
            ),
            "pretrigger_dp_steps": len(pretrigger_steps),
            "records_tail": trigger_records[-12:],
        },
        dp_agent,
        dp_args,
    )


def main() -> int:
    args = parse_args()
    require_compute_step()
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    if args.expected_video_frames != 301 or args.expected_action_steps != 300:
        raise SystemExit("active contract requires 301 video frames and 300 action steps")
    if args.expected_action_dim != len(FULL_EPISODE_VECTOR_NAMES):
        raise SystemExit("expected action dim does not match WAM vector names")
    try:
        args.continuability_profile = apply_continuability_stats(args)
    except Exception as exc:
        raise SystemExit(f"invalid continuability stats profile: {exc}") from exc

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_h5 = Path(args.source_h5).resolve()
    initial_video = Path(args.initial_video).resolve() if args.initial_video else None
    if args.prefix_frame_source == "initial_video" and initial_video is None:
        raise SystemExit("--initial-video is required when --prefix-frame-source=initial_video")
    source_uuid_text = " ".join([source_h5.name, args.sample_name, args.scenario])
    reset_seed = _parse_seed_from_text(source_uuid_text) or 0

    stack = _import_live_control_stack(ROOT)
    trajectory_utils = stack["trajectory_utils"]
    env = _make_live_env(stack, Path(args.dp_manifest).resolve(), args)
    summary: dict[str, Any] = {
        "boundary": (
            "Live receding loop scaffold. Dry-run builds causal prefix video "
            "and WAM history only. With Cosmos inference enabled, every short "
            "chunk must be followed by real re-observation before the next call."
        ),
        "evidence_boundary": (
            "Single-sample or short-run outputs are interface diagnostics. "
            "Method evidence still requires scenario-diverse live receding "
            "rollouts, continuability-gated DP handoff, real final-state "
            "success metrics, and direct video review."
        ),
        "source_h5": str(source_h5),
        "initial_video": str(initial_video) if initial_video else None,
        "condition_root": str(Path(args.condition_root).resolve()),
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "config_file": str(Path(args.config_file).resolve()),
        "dp_manifest": str(Path(args.dp_manifest).resolve()),
        "sample_name": args.sample_name,
        "scenario": args.scenario,
        "prefix_role_request": args.prefix_role,
        "prefix_start_mode": args.prefix_start_mode,
        "pretrigger_control_mode": args.pretrigger_control_mode,
        "manual_prefix_frame_index": int(args.prefix_frame_index),
        "prefix_frame_source": args.prefix_frame_source,
        "initial_prefix_frame_index": None,
        "max_receding_iterations": int(args.max_receding_iterations),
        "action_exec_horizon": int(args.action_exec_horizon),
        "cosmos_step_handoff_gate": bool(args.cosmos_step_handoff_gate),
        "full_episode_rollout": bool(args.full_episode_rollout),
        "annotate_video": bool(args.annotate_video),
        "max_episode_steps": int(args.max_episode_steps),
        "expected_video_frames": int(args.expected_video_frames),
        "expected_action_steps": int(args.expected_action_steps),
        "expected_action_dim": int(args.expected_action_dim),
        "robot_action_dim": int(args.robot_action_dim),
        "clip_live_actions": bool(args.clip_live_actions),
        "external_target_mode": args.external_target_mode,
        "dp_checkpoint": str(Path(args.dp_checkpoint).resolve()) if args.dp_checkpoint else None,
        "dp_state_key": args.dp_state_key,
        "dp_handoff_horizon": int(args.dp_handoff_horizon),
        "continuability_thresholds": {
            "min_rel_x": float(args.continuability_min_rel_x),
            "max_rel_x": float(args.continuability_max_rel_x),
            "max_abs_y": float(args.continuability_max_abs_y),
            "max_abs_z": float(args.continuability_max_abs_z),
            "max_hole_speed": float(args.continuability_max_hole_speed),
        },
        "continuability_profile": args.continuability_profile,
        "target_motion_role_thresholds": {
            "delta": float(args.target_motion_delta_threshold),
            "speed": float(args.target_motion_speed_threshold),
        },
        "run_cosmos_inference": bool(args.run_cosmos_inference),
        "iterations": [],
    }
    if int(args.dp_handoff_horizon) > 0 and not args.dp_checkpoint:
        raise SystemExit("--dp-checkpoint is required when --dp-handoff-horizon > 0")
    try:
        import h5py

        with h5py.File(source_h5, "r") as h5:
            traj_name = "traj_0" if "traj_0" in h5 else next(iter(h5.keys()))
            env_states = trajectory_utils.dict_to_list_of_dicts(h5[traj_name]["env_states"])
        obs, _ = env.reset(seed=reset_seed)
        base_env = _get_base_env(env)
        low, high = _action_space_bounds(env, args.robot_action_dim)
        torch = stack["torch"]
        if args.pretrigger_control_mode == "frozen_dp_until_target_motion":
            (
                prefix_frame,
                observed_frames,
                history,
                dp_obs_history,
                previous_hole_xyz,
                prefix_selection,
                dp_agent,
                dp_args,
            ) = run_live_dp_until_trigger(
                env=env,
                base_env=base_env,
                stack=stack,
                env_states=env_states,
                args=args,
                low=low,
                high=high,
            )
        else:
            (
                prefix_frame,
                observed_frames,
                history,
                dp_obs_history,
                previous_hole_xyz,
                prefix_selection,
                dp_agent,
                dp_args,
            ) = build_source_restore_prefix(
                env=env,
                base_env=base_env,
                stack=stack,
                source_h5=source_h5,
                env_states=env_states,
                args=args,
            )
        summary["prefix_selection"] = prefix_selection
        summary["initial_prefix_frame_index"] = prefix_frame
        if prefix_frame >= len(env_states):
            raise ValueError(f"prefix_frame {prefix_frame} outside env state length {len(env_states)}")

        iteration = 0
        while prefix_frame < args.expected_action_steps and iteration < max(1, int(args.max_receding_iterations)):
            iter_dir = output_root / f"iter_{iteration:02d}_prefix_f{prefix_frame:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            prefix_video = iter_dir / "observed_prefix.mp4"
            history_path = iter_dir / "live_history_raw_action_state.json"
            last_live = live_pose_row(base_env, stack, previous_hole_xyz)
            prefix_role_info = infer_prefix_role(
                history=history,
                prefix_frame=prefix_frame,
                scenario=args.scenario,
                live=last_live,
                args=args,
            )
            write_video(prefix_video, observed_frames, args.video_fps)
            write_action_json(history_path, history)

            iter_record: dict[str, Any] = {
                "iteration": iteration,
                "prefix_frame_index": prefix_frame,
                "prefix_role": prefix_role_info["role"],
                "prefix_role_info": prefix_role_info,
                "prefix_video": str(prefix_video),
                "history_action_state": str(history_path),
                "observed_prefix_frames": len(observed_frames),
                "prefix_frame_source": args.prefix_frame_source,
                "external_target_mode": args.external_target_mode,
                "before_eval": _live_eval(base_env),
            }
            if not args.run_cosmos_inference:
                iter_record["dry_run_stop"] = "cosmos_inference_disabled"
                summary["iterations"].append(iter_record)
                write_partial_summary(output_root, summary, base_env, prefix_frame, observed_frames)
                break

            pre_gate = continuability_gate(last_live, history, prefix_frame, args)
            iter_record["pre_controller_continuability_gate"] = pre_gate
            if bool(pre_gate["ok"]) and int(args.dp_handoff_horizon) > 0:
                iter_record["controller_step_type"] = "frozen_dp_short_chunk"
                iter_record["dp_handoff_steps"] = []
                dp_steps_this_iteration = min(
                    max(1, int(args.action_exec_horizon)),
                    max(1, int(args.dp_handoff_horizon)),
                )
                iter_record["dp_handoff_requested_horizon"] = int(args.dp_handoff_horizon)
                iter_record["dp_handoff_chunk_horizon"] = int(dp_steps_this_iteration)
                iter_record["dp_boundary"] = (
                    "Frozen DP is executed only as a short reobserved chunk "
                    "after the real live state passes C_pi. This is not a "
                    "blind long takeover; after this chunk the loop either "
                    "stops on real success/termination or refreshes the prefix "
                    "and chooses DP/Cosmos again."
                )
                if dp_agent is None:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    dp_agent, dp_args = _load_dp_agent(env, stack, args, device)
                assert dp_args is not None
                dp_device = next(dp_agent.parameters()).device
                dp_call_index = 0
                while len(iter_record["dp_handoff_steps"]) < dp_steps_this_iteration and prefix_frame < args.expected_action_steps:
                    obs_seq = np.stack(dp_obs_history[-2:], axis=0)[None].astype(np.float32)
                    obs_tensor = torch.as_tensor(obs_seq, device=dp_device, dtype=torch.float32)
                    with torch.no_grad():
                        action_seq = dp_agent.get_action(obs_tensor)
                    if getattr(dp_args, "sim_backend", "physx_cpu") == "physx_cpu":
                        action_seq_np = action_seq.detach().cpu().numpy()
                    else:
                        action_seq_np = action_seq
                    act_horizon = int(action_seq_np.shape[1])
                    if act_horizon <= 0:
                        iter_record["dp_handoff_stop_reason"] = "empty_dp_action_sequence"
                        break
                    for chunk_local_i in range(min(dp_steps_this_iteration - len(iter_record["dp_handoff_steps"]), act_horizon)):
                        if prefix_frame >= args.expected_action_steps:
                            break
                        step_action, action_record = _prepare_step_action(
                            action_seq_np[:, chunk_local_i],
                            low,
                            high,
                            bool(args.clip_live_actions),
                        )
                        obs, reward, terminated, truncated, info = env.step(step_action)
                        external_target = apply_external_target_pose(
                            base_env=base_env,
                            stack=stack,
                            env_states=env_states,
                            source_frame=prefix_frame + 1,
                            args=args,
                        )
                        current_state_obs = read_state_obs(base_env, stack)
                        dp_obs_history.append(current_state_obs.copy())
                        live = live_pose_row(base_env, stack, previous_hole_xyz)
                        previous_hole_xyz = live["hole_xyz"]
                        last_live = live
                        fill_live_history_row(history, prefix_frame, np.asarray(action_record["executed"]), live)
                        observed_frames.append(_render_frame(env))
                        prefix_frame += 1
                        iter_record["dp_handoff_steps"].append(
                            {
                                "local_step": len(iter_record["dp_handoff_steps"]),
                                "global_action_index": prefix_frame - 1,
                                "dp_call_index": dp_call_index,
                                "chunk_local_step": chunk_local_i,
                                "action": action_record,
                                "external_target": external_target,
                                "reward": jsonable(reward),
                                "terminated": jsonable(terminated),
                                "truncated": jsonable(truncated),
                                "live_eval": _live_eval(base_env),
                            }
                        )
                        if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                            iter_record["terminated_or_truncated"] = True
                            iter_record["stop_reason"] = "terminated_or_truncated_after_dp_short_chunk"
                            break
                        if _live_eval(base_env).get("success"):
                            iter_record["live_success_observed_during_dp_short_chunk"] = True
                            if bool(args.full_episode_rollout):
                                continue
                            iter_record["dp_handoff_stop_reason"] = "live_success_after_dp_short_chunk"
                            iter_record["stop_reason"] = "live_success_after_dp_short_chunk"
                            break
                    if iter_record.get("terminated_or_truncated") or iter_record.get("dp_handoff_stop_reason"):
                        break
                    dp_call_index += 1
                iter_record["after_dp_handoff_eval"] = _live_eval(base_env)
                iter_record["dp_handoff_executed"] = bool(iter_record["dp_handoff_steps"])
                iter_record["after_dp_handoff_continuability_gate"] = continuability_gate(last_live, history, prefix_frame, args)
                summary["iterations"].append(iter_record)
                write_partial_summary(output_root, summary, base_env, prefix_frame, observed_frames)
                if (
                    iter_record.get("terminated_or_truncated")
                    or prefix_frame >= args.expected_action_steps
                ):
                    break
                iteration += 1
                continue

            iter_record["controller_step_type"] = "cosmos_rebind_short_chunk"
            chunk = run_prefix_inference(
                args=args,
                iter_dir=iter_dir,
                prefix_video=prefix_video,
                history_path=history_path,
                prefix_frame=prefix_frame,
                prefix_role=str(prefix_role_info["role"]),
                iteration=iteration,
            )
            if not bool(chunk.get("ok", False)):
                raise RuntimeError(f"Cosmos action chunk extraction failed: {chunk}")
            iter_record["action_chunk_json"] = str(iter_dir / "cosmos_live_prefix" / "live_prefix_action_chunk.json")
            iter_record["sample_output_json"] = chunk.get("sample_output_json")
            iter_record["chunk_start"] = chunk["chunk_start"]
            iter_record["chunk_end_exclusive"] = chunk["chunk_end_exclusive"]
            iter_record["chunk_steps"] = chunk.get("chunk_steps")
            iter_record["normalized_robot_action_stats"] = chunk.get("normalized_robot_action_stats")
            iter_record["denormalized_robot_action_stats"] = chunk.get("denormalized_robot_action_stats")
            actions = chunk.get("denormalized_robot_action_chunk") or []
            iter_record["executed_steps"] = []
            iter_record["step_continuability_gates"] = []
            for local_i, action in enumerate(actions):
                if prefix_frame >= args.expected_action_steps:
                    break
                step_action, action_record = _prepare_step_action(action, low, high, bool(args.clip_live_actions))
                obs, reward, terminated, truncated, info = env.step(step_action)
                external_target = apply_external_target_pose(
                    base_env=base_env,
                    stack=stack,
                    env_states=env_states,
                    source_frame=prefix_frame + 1,
                    args=args,
                )
                current_state_obs = read_state_obs(base_env, stack)
                dp_obs_history.append(current_state_obs.copy())
                live = live_pose_row(base_env, stack, previous_hole_xyz)
                previous_hole_xyz = live["hole_xyz"]
                last_live = live
                fill_live_history_row(history, prefix_frame, np.asarray(action_record["executed"]), live)
                observed_frames.append(_render_frame(env))
                prefix_frame += 1
                step_record = {
                    "local_step": local_i,
                    "global_action_index": prefix_frame - 1,
                    "action": action_record,
                    "external_target": external_target,
                    "reward": jsonable(reward),
                    "terminated": jsonable(terminated),
                    "truncated": jsonable(truncated),
                    "live_eval": _live_eval(base_env),
                }
                if bool(args.cosmos_step_handoff_gate) and int(args.dp_handoff_horizon) > 0:
                    step_gate = continuability_gate(last_live, history, prefix_frame, args)
                    step_record["continuability_gate_after_step"] = step_gate
                    iter_record["step_continuability_gates"].append(
                        {
                            "local_step": local_i,
                            "global_action_index": prefix_frame - 1,
                            "gate": step_gate,
                        }
                    )
                iter_record["executed_steps"].append(step_record)
                if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                    iter_record["terminated_or_truncated"] = True
                    break
                if (
                    bool(args.cosmos_step_handoff_gate)
                    and int(args.dp_handoff_horizon) > 0
                    and bool(step_record.get("continuability_gate_after_step", {}).get("ok", False))
                ):
                    iter_record["cosmos_chunk_stop_reason"] = "step_level_continuability_gate_ok"
                    iter_record["step_level_handoff_ready"] = True
                    break
            iter_record["after_eval"] = _live_eval(base_env)
            if iter_record["after_eval"].get("success"):
                iter_record["live_success_observed_after_cosmos_chunk"] = True
                if not bool(args.full_episode_rollout):
                    iter_record["stop_reason"] = "live_success_after_cosmos_chunk"
                    summary["iterations"].append(iter_record)
                    write_partial_summary(output_root, summary, base_env, prefix_frame, observed_frames)
                    break
            if iter_record.get("stop_reason") == "live_success_after_cosmos_chunk":
                summary["iterations"].append(iter_record)
                write_partial_summary(output_root, summary, base_env, prefix_frame, observed_frames)
                break
            if iter_record.get("terminated_or_truncated"):
                iter_record["stop_reason"] = "terminated_or_truncated_after_cosmos_chunk"
                summary["iterations"].append(iter_record)
                write_partial_summary(output_root, summary, base_env, prefix_frame, observed_frames)
                break

            iter_record["dp_handoff_executed"] = False
            iter_record["post_cosmos_continuability_gate"] = continuability_gate(last_live, history, prefix_frame, args)
            summary["iterations"].append(iter_record)
            write_partial_summary(output_root, summary, base_env, prefix_frame, observed_frames)
            if iter_record.get("terminated_or_truncated") or prefix_frame >= args.expected_action_steps:
                break
            iteration += 1

        final_observed_video = output_root / "live_observed_rollout.mp4"
        write_video(final_observed_video, observed_frames, args.video_fps)
        summary["final_observed_video"] = str(final_observed_video)
        summary["final_observed_video_inspection"] = inspect_video_file(final_observed_video)
        summary["final_observed_frames"] = len(observed_frames)
        summary["final_eval"] = _live_eval(base_env)
        summary["final_prefix_frame_index"] = prefix_frame
        summary["completed_iterations"] = len(summary["iterations"])
        summary["full_episode_length_ok"] = bool(
            prefix_frame >= args.expected_action_steps
            and len(observed_frames) == args.expected_video_frames
        )
        summary["unified_detector_controller_boundary"] = (
            "Controller selection is made by one causal target-motion detector "
            "over observed hole poses. Before that detector fires, frozen DP "
            "runs. After it fires, Cosmos3 may generate short rebind chunks, "
            "and frozen DP can resume only through the same real-state C_pi "
            "gate. If the detector never fires, DP runs the full episode by "
            "the same rule; static samples are not handled by a separate "
            "scenario branch."
        )
        summary["controller_timeline"] = controller_timeline_from_summary(summary, len(observed_frames))
        summary["controller_frame_counts"] = {}
        for meta in summary["controller_timeline"]:
            controller = str(meta.get("controller"))
            summary["controller_frame_counts"][controller] = summary["controller_frame_counts"].get(controller, 0) + 1
        summary["wm_active_frame_count"] = sum(1 for meta in summary["controller_timeline"] if meta.get("wm_active"))
        summary["dp_active_frame_count"] = sum(1 for meta in summary["controller_timeline"] if meta.get("dp_active"))
        if bool(args.annotate_video):
            annotated_video = output_root / "live_observed_rollout_annotated.mp4"
            annotation_summary = write_annotated_video(
                annotated_video,
                observed_frames,
                args.video_fps,
                summary,
            )
            summary["annotated_video_summary"] = annotation_summary
            summary["final_observed_annotated_video"] = str(annotated_video)
            summary["final_observed_annotated_video_inspection"] = annotation_summary.get("video_inspection")
        inspections = [summary.get("final_observed_video_inspection")]
        if bool(args.annotate_video):
            inspections.append(summary.get("final_observed_annotated_video_inspection"))
        summary["video_file_contract_ok"] = video_inspections_match_contract(
            inspections,
            expected_video_frames=int(args.expected_video_frames),
            expected_inspection_count=2 if bool(args.annotate_video) else 1,
        )
        write_json(output_root / "live_receding_loop_summary.json", summary)
        print(json.dumps({"summary": str(output_root / "live_receding_loop_summary.json")}, sort_keys=True))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
