#!/usr/bin/env python3
"""Replay recorded live-loop actions without rendering and save state snapshots."""

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
    write_live_state_snapshot,
)
from run_cosmos3_receding_closed_loop import (  # noqa: E402
    _action_space_bounds,
    _get_base_env,
    _import_live_control_stack,
    _live_eval,
    _make_live_env,
    _parse_seed_from_text,
    _prepare_step_action,
    jsonable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dp-manifest", required=True)
    parser.add_argument("--expected-video-frames", type=int, default=301)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--robot-action-dim", type=int, default=7)
    parser.add_argument("--clip-live-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--snapshot-all-query-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-interval", type=int, default=25)
    return parser.parse_args()


def require_compute_step() -> None:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    step_id = os.environ.get("SLURM_STEP_ID", "")
    if not job_id or not step_id or step_id == "extern":
        raise SystemExit(
            "refusing_login_node_execution=true\n"
            "reason=Replay live history snapshots only inside a compute-node srun step."
        )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, sort_keys=True) + "\n")


def emit(event: str, **payload: Any) -> None:
    print(json.dumps(jsonable({"event": event, **payload}), sort_keys=True), flush=True)


def load_episode_env_states(source_h5: Path, trajectory_utils: Any) -> list[Any]:
    import h5py

    with h5py.File(source_h5, "r") as h5:
        traj_name = "traj_0" if "traj_0" in h5 else next(iter(h5.keys()))
        env_states = trajectory_utils.dict_to_list_of_dicts(h5[traj_name]["env_states"])
    return env_states


def extract_recorded_actions(summary: dict[str, Any], expected_steps: int, robot_dim: int) -> dict[str, Any]:
    iterations = summary.get("iterations") or []
    if not iterations:
        raise ValueError("summary has no iterations")
    last_iter = iterations[-1]
    history_path = Path(str(last_iter.get("history_action_state") or ""))
    if not history_path.is_file():
        raise FileNotFoundError(f"missing last history_action_state: {history_path}")
    history = np.asarray(read_json(history_path).get("action"), dtype=np.float32)
    if history.shape[0] != expected_steps or history.shape[1] < robot_dim:
        raise ValueError(f"unexpected history shape {history.shape}, expected ({expected_steps}, >= {robot_dim})")

    actions = np.zeros((expected_steps, robot_dim), dtype=np.float32)
    action_source = ["missing"] * expected_steps
    last_prefix = int(last_iter.get("prefix_frame_index", 0))
    fill_until = max(0, min(last_prefix, expected_steps))
    actions[:fill_until] = history[:fill_until, :robot_dim]
    for idx in range(fill_until):
        action_source[idx] = "history_before_last_iteration"

    for iteration in iterations:
        for key in ("dp_handoff_steps", "executed_steps"):
            for step in iteration.get(key) or []:
                global_idx = int(step.get("global_action_index", -1))
                if global_idx < 0 or global_idx >= expected_steps:
                    continue
                action = step.get("action") or {}
                executed = action.get("executed")
                if executed is None:
                    continue
                arr = np.asarray(executed, dtype=np.float32).reshape(-1)
                if arr.shape[0] < robot_dim:
                    raise ValueError(f"recorded action at {global_idx} has dim {arr.shape[0]}, expected {robot_dim}")
                actions[global_idx] = arr[:robot_dim]
                action_source[global_idx] = f"summary_{key}"

    missing = [idx for idx, source in enumerate(action_source) if source == "missing"]
    return {
        "actions": actions,
        "action_source": action_source,
        "missing_action_indices": missing,
        "last_history_path": str(history_path),
        "last_prefix_frame_index": last_prefix,
    }


def query_frames_from_summary(summary: dict[str, Any], expected_steps: int) -> list[int]:
    frames = {
        int(iteration.get("prefix_frame_index"))
        for iteration in (summary.get("iterations") or [])
        if iteration.get("prefix_frame_index") is not None
    }
    frames.add(int(summary.get("final_prefix_frame_index") or expected_steps))
    return sorted(frame for frame in frames if 0 <= frame <= expected_steps)


def main() -> int:
    args = parse_args()
    require_compute_step()
    if args.expected_video_frames != 301 or args.expected_action_steps != 300:
        raise SystemExit("active contract requires 301 video frames and 300 action steps")

    loop_summary_path = Path(args.loop_summary).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary = read_json(loop_summary_path)
    source_h5 = Path(str(summary.get("source_h5") or "")).resolve()
    if not source_h5.is_file():
        raise FileNotFoundError(f"summary source_h5 does not exist: {source_h5}")
    sample_name = str(summary.get("sample_name") or loop_summary_path.parent.name)
    scenario = str(summary.get("scenario") or "live_dynamic")

    action_info = extract_recorded_actions(
        summary,
        expected_steps=int(args.expected_action_steps),
        robot_dim=int(args.robot_action_dim),
    )
    if action_info["missing_action_indices"]:
        raise ValueError(f"missing recorded actions: {action_info['missing_action_indices'][:20]}")

    emit(
        "replay_snapshot_start",
        loop_summary=str(loop_summary_path),
        source_h5=str(source_h5),
        output_root=str(output_root),
        sample_name=sample_name,
        scenario=scenario,
    )

    stack = _import_live_control_stack(ROOT)
    trajectory_utils = stack["trajectory_utils"]
    env_states = load_episode_env_states(source_h5, trajectory_utils)
    if len(env_states) < int(args.expected_video_frames):
        raise ValueError(f"source has {len(env_states)} env_states, expected {args.expected_video_frames}")

    env_args = argparse.Namespace(
        max_episode_steps=int(args.expected_action_steps),
        robot_action_dim=int(args.robot_action_dim),
    )
    env = _make_live_env(stack, Path(args.dp_manifest).resolve(), env_args)
    replay_summary: dict[str, Any] = {
        "boundary": (
            "Replay of recorded live-loop robot actions without rendering or "
            "Cosmos inference. The output snapshots are restore candidates for "
            "failed-state recovery data construction, not method evidence."
        ),
        "loop_summary": str(loop_summary_path),
        "source_h5": str(source_h5),
        "sample_name": sample_name,
        "scenario": scenario,
        "expected_action_steps": int(args.expected_action_steps),
        "expected_video_frames": int(args.expected_video_frames),
        "old_final_eval": summary.get("final_eval"),
        "action_source_counts": {
            source: action_info["action_source"].count(source)
            for source in sorted(set(action_info["action_source"]))
        },
        "last_history_path": action_info["last_history_path"],
        "snapshots": [],
        "frame_evals": [],
    }
    try:
        reset_seed = _parse_seed_from_text(" ".join([source_h5.name, sample_name, scenario])) or 0
        obs, _ = env.reset(seed=reset_seed)
        base_env = _get_base_env(env)
        base_env.set_state_dict(env_states[0])
        low, high = _action_space_bounds(env, int(args.robot_action_dim))

        query_frames = query_frames_from_summary(summary, int(args.expected_action_steps))
        query_set = set(query_frames)
        actions = action_info["actions"]
        replay_summary["query_frames"] = query_frames

        for frame in range(0, int(args.expected_action_steps) + 1):
            if frame in query_set:
                live_eval = _live_eval(base_env)
                snapshot_path = output_root / f"frame_{frame:03d}_live_state.h5"
                write_live_state_snapshot(
                    path=snapshot_path,
                    base_env=base_env,
                    stack=stack,
                    prefix_frame=frame,
                    iteration=len(replay_summary["snapshots"]),
                    label=f"replayed_frame_{frame:03d}",
                )
                replay_summary["snapshots"].append(
                    {
                        "frame": frame,
                        "path": str(snapshot_path),
                        "live_eval": live_eval,
                    }
                )
                replay_summary["frame_evals"].append({"frame": frame, "live_eval": live_eval})
                emit("replay_snapshot_written", frame=frame, live_eval=live_eval, path=str(snapshot_path))
            if frame >= int(args.expected_action_steps):
                break

            step_action, action_record = _prepare_step_action(
                actions[frame],
                low,
                high,
                bool(args.clip_live_actions),
            )
            obs, reward, terminated, truncated, info = env.step(step_action)
            external_target = apply_external_target_pose(
                base_env=base_env,
                stack=stack,
                env_states=env_states,
                source_frame=frame + 1,
                args=argparse.Namespace(external_target_mode="source_env_state"),
            )
            if int(args.progress_interval) > 0 and ((frame + 1) % int(args.progress_interval) == 0):
                emit(
                    "replay_step",
                    frame=frame + 1,
                    live_eval=_live_eval(base_env),
                    external_target=external_target,
                )
            if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                replay_summary["terminated_or_truncated"] = {
                    "frame": frame + 1,
                    "terminated": jsonable(terminated),
                    "truncated": jsonable(truncated),
                }
                break

        final_eval = _live_eval(base_env)
        replay_summary["final_eval"] = final_eval
        old_rel = np.asarray((summary.get("final_eval") or {}).get("peg_head_pos_at_hole") or [], dtype=np.float32)
        new_rel = np.asarray(final_eval.get("peg_head_pos_at_hole") or [], dtype=np.float32)
        if old_rel.shape == new_rel.shape and old_rel.size:
            replay_summary["final_peg_head_pos_at_hole_l2_vs_old"] = float(np.linalg.norm(new_rel - old_rel))
        write_json(output_root / "replay_live_history_state_snapshots_summary.json", replay_summary)
        emit("replay_snapshot_complete", final_eval=final_eval, summary=str(output_root / "replay_live_history_state_snapshots_summary.json"))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
