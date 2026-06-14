#!/usr/bin/env python3
"""Full-episode frozen-DP baseline under the same dynamic target replay.

This is the direct control comparison for Cosmos3 live receding evaluation:
the robot executes the frozen static DP for all 300 actions while the target
actor is advanced from the source H5 env-state stream. The target-motion
detector is recorded for evidence only and never changes the controller.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_cosmos3_live_receding_loop import (  # noqa: E402
    apply_external_target_pose,
    empty_history,
    fill_live_history_row,
    jsonable,
    live_pose_row,
    read_state_obs,
    require_compute_step,
    target_motion_update,
    write_json,
    write_video,
)
from video_contract_utils import inspect_video_file, video_inspections_match_contract  # noqa: E402
from run_cosmos3_receding_closed_loop import (  # noqa: E402
    _action_space_bounds,
    _get_base_env,
    _import_live_control_stack,
    _live_eval,
    _load_dp_agent,
    _make_live_env,
    _parse_seed_from_text,
    _prepare_step_action,
    _render_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-h5", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dp-manifest", required=True)
    parser.add_argument("--dp-checkpoint", required=True)
    parser.add_argument("--dp-state-key", choices=("ema_agent", "agent"), default="ema_agent")
    parser.add_argument("--sample-name", default="pure_dp_full_episode")
    parser.add_argument("--scenario", default="unknown")
    parser.add_argument("--expected-video-frames", type=int, default=301)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--expected-action-dim", type=int, default=32)
    parser.add_argument("--robot-action-dim", type=int, default=7)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--clip-live-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--external-target-mode",
        choices=("source_env_state", "none"),
        default="source_env_state",
    )
    parser.add_argument("--min-dynamic-prefix-frame", type=int, default=8)
    parser.add_argument("--target-motion-consecutive-frames", type=int, default=2)
    parser.add_argument("--target-motion-delta-threshold", type=float, default=0.002)
    parser.add_argument("--target-motion-speed-threshold", type=float, default=0.001)
    parser.add_argument("--annotate-video", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def write_annotated_video(path: Path, frames: list[Any], fps: int, timeline: list[dict[str, Any]]) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    annotated: list[np.ndarray] = []
    for frame, meta in zip(frames, timeline):
        img = Image.fromarray(np.asarray(frame)).convert("RGB")
        draw = ImageDraw.Draw(img)
        trigger = meta.get("target_motion_trigger_frame")
        lines = [
            f"frame {meta['frame_index']:03d}/300  controller=PURE_DP",
            f"target_motion_detected={bool(meta.get('target_motion_detected'))}  trigger={trigger if trigger is not None else 'none'}",
            "wm_active=False  dp_active=True  policy=frozen_static_dp",
        ]
        text_w = max(draw.textlength(line) for line in lines) + 12
        text_h = 18 * len(lines) + 8
        draw.rectangle((6, 6, 6 + int(text_w), 6 + text_h), fill=(255, 255, 255), outline=(0, 0, 0))
        for i, line in enumerate(lines):
            draw.text((12, 11 + i * 18), line, fill=(0, 0, 0))
        annotated.append(np.asarray(img))
    write_video(path, annotated, fps)
    video_inspection = inspect_video_file(path)
    counts = {"PURE_DP": max(0, len(frames) - 1), "INIT_OBS": 1 if frames else 0}
    return {
        "annotated_video": str(path),
        "frame_count": len(frames),
        "video_inspection": video_inspection,
        "controller_frame_counts": counts,
        "wm_active_frame_count": 0,
        "dp_active_frame_count": max(0, len(frames) - 1),
        "target_motion_detected_frame_count": sum(1 for meta in timeline if meta.get("target_motion_detected")),
        "timeline": timeline,
    }


def build_timeline(frame_count: int, trigger_frame: int | None) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for frame_idx in range(frame_count):
        timeline.append(
            {
                "frame_index": int(frame_idx),
                "controller": "INIT_OBS" if frame_idx == 0 else "PURE_DP",
                "target_motion_detected": bool(trigger_frame is not None and frame_idx >= trigger_frame),
                "target_motion_trigger_frame": trigger_frame,
                "wm_active": False,
                "dp_active": frame_idx > 0,
                "policy": "frozen_static_dp" if frame_idx > 0 else None,
            }
        )
    return timeline


def main() -> int:
    args = parse_args()
    require_compute_step()
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    if args.expected_video_frames != args.expected_action_steps + 1:
        raise SystemExit("expected_video_frames must equal expected_action_steps + 1")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_h5 = Path(args.source_h5).resolve()
    source_uuid_text = " ".join([source_h5.name, args.sample_name, args.scenario])
    reset_seed = _parse_seed_from_text(source_uuid_text) or 0

    stack = _import_live_control_stack(ROOT)
    trajectory_utils = stack["trajectory_utils"]
    torch = stack["torch"]

    import h5py

    env = _make_live_env(stack, Path(args.dp_manifest).resolve(), args)
    try:
        with h5py.File(source_h5, "r") as h5:
            traj_name = "traj_0" if "traj_0" in h5 else next(iter(h5.keys()))
            env_states = trajectory_utils.dict_to_list_of_dicts(h5[traj_name]["env_states"])
        if len(env_states) < args.expected_video_frames:
            raise ValueError(f"source has {len(env_states)} env states, expected at least {args.expected_video_frames}")

        env.reset(seed=reset_seed)
        base_env = _get_base_env(env)
        base_env.set_state_dict(env_states[0])
        low, high = _action_space_bounds(env, args.robot_action_dim)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dp_agent, dp_args = _load_dp_agent(env, stack, args, device)
        dp_device = next(dp_agent.parameters()).device

        observed_frames: list[Any] = [_render_frame(env)]
        history = empty_history(args)
        current_state_obs = read_state_obs(base_env, stack)
        dp_obs_history: list[np.ndarray] = [current_state_obs.copy(), current_state_obs.copy()]
        initial_live = live_pose_row(base_env, stack, None)
        initial_target_pos = initial_live["hole_xyz"].copy()
        previous_hole_xyz = initial_live["hole_xyz"].copy()
        previous_target_pos = initial_target_pos.copy()
        consecutive = 0
        first_streak_frame: int | None = None
        trigger_frame: int | None = None
        trigger_first_streak_frame: int | None = None
        target_motion_records: list[dict[str, Any]] = []
        executed_steps: list[dict[str, Any]] = []
        prefix_frame = 0
        dp_call_index = 0

        while prefix_frame < args.expected_action_steps:
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
                raise RuntimeError("frozen DP returned empty action sequence")
            for chunk_local_i in range(act_horizon):
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
                fill_live_history_row(history, prefix_frame, np.asarray(action_record["executed"]), live)
                observed_frames.append(_render_frame(env))
                prefix_frame += 1

                target_pos = live["hole_xyz"]
                triggered, consecutive, first_streak_frame, motion_record = target_motion_update(
                    frame=prefix_frame,
                    pos=target_pos,
                    initial_pos=initial_target_pos,
                    previous_pos=previous_target_pos,
                    consecutive=consecutive,
                    first_streak_frame=first_streak_frame,
                    args=args,
                )
                previous_target_pos = np.asarray(target_pos, dtype=np.float32).copy()
                previous_hole_xyz = np.asarray(live["hole_xyz"], dtype=np.float32).copy()
                if motion_record["moving"] or prefix_frame in {1, int(args.min_dynamic_prefix_frame)}:
                    target_motion_records.append(motion_record)
                if triggered and trigger_frame is None:
                    trigger_frame = int(prefix_frame)
                    trigger_first_streak_frame = first_streak_frame

                executed_steps.append(
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
                        "target_motion": motion_record,
                    }
                )
            dp_call_index += 1

        video_path = output_root / "pure_dp_observed_rollout.mp4"
        write_video(video_path, observed_frames, args.video_fps)
        video_inspection = inspect_video_file(video_path)
        timeline = build_timeline(len(observed_frames), trigger_frame)
        final_eval = _live_eval(base_env)
        summary: dict[str, Any] = {
            "boundary": (
                "Full-episode pure frozen-DP baseline. Target motion is replayed "
                "from the same source H5 as the Cosmos closed-loop eval, but the "
                "detector is reporting-only and never switches controller mode."
            ),
            "method_comparison_role": (
                "Use this only as the all-DP baseline for the same dynamic source "
                "trajectory. It is not a world-model result."
            ),
            "source_h5": str(source_h5),
            "dp_manifest": str(Path(args.dp_manifest).resolve()),
            "dp_checkpoint": str(Path(args.dp_checkpoint).resolve()),
            "dp_state_key": args.dp_state_key,
            "sample_name": args.sample_name,
            "scenario": args.scenario,
            "external_target_mode": args.external_target_mode,
            "expected_action_steps": int(args.expected_action_steps),
            "expected_video_frames": int(args.expected_video_frames),
            "final_prefix_frame_index": int(prefix_frame),
            "final_observed_frames": len(observed_frames),
            "full_episode_length_ok": bool(prefix_frame == args.expected_action_steps and len(observed_frames) == args.expected_video_frames),
            "target_motion_detector": {
                "triggered": trigger_frame is not None,
                "detected_frame_index": trigger_frame,
                "first_streak_frame_index": trigger_first_streak_frame,
                "thresholds": {
                    "min_dynamic_prefix_frame": int(args.min_dynamic_prefix_frame),
                    "target_motion_delta": float(args.target_motion_delta_threshold),
                    "target_motion_speed": float(args.target_motion_speed_threshold),
                    "consecutive_frames": int(args.target_motion_consecutive_frames),
                },
                "records_tail": target_motion_records[-12:],
            },
            "controller_timeline": timeline,
            "controller_frame_counts": {"INIT_OBS": 1, "PURE_DP": int(args.expected_action_steps)},
            "wm_active_frame_count": 0,
            "dp_active_frame_count": int(args.expected_action_steps),
            "executed_steps": executed_steps,
            "final_observed_video": str(video_path),
            "final_observed_video_inspection": video_inspection,
            "final_eval": final_eval,
        }
        if bool(args.annotate_video):
            annotated_path = output_root / "pure_dp_observed_rollout_annotated.mp4"
            summary["annotated_video_summary"] = write_annotated_video(
                annotated_path,
                observed_frames,
                args.video_fps,
                timeline,
            )
            summary["final_observed_annotated_video"] = str(annotated_path)
            summary["final_observed_annotated_video_inspection"] = summary["annotated_video_summary"].get("video_inspection")
        inspections = [summary.get("final_observed_video_inspection")]
        if bool(args.annotate_video):
            inspections.append(summary.get("final_observed_annotated_video_inspection"))
        summary["video_file_contract_ok"] = video_inspections_match_contract(
            inspections,
            expected_video_frames=int(args.expected_video_frames),
            expected_inspection_count=2 if bool(args.annotate_video) else 1,
        )
        write_json(output_root / "pure_dp_full_episode_summary.json", summary)
        print(json.dumps({"summary": str(output_root / "pure_dp_full_episode_summary.json")}, sort_keys=True))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
