#!/usr/bin/env python3
"""Replay one Cosmos Policy-DROID action chunk from a saved live snapshot."""

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

from replay_cosmos3_live_action_bank_from_snapshots import (  # noqa: E402
    load_env_states,
    load_history,
    load_snapshot_state,
    replay_bank_candidate,
)
from run_cosmos3_live_receding_loop import require_compute_step  # noqa: E402
from run_cosmos3_receding_closed_loop import (  # noqa: E402
    _action_space_bounds,
    _get_base_env,
    _import_live_control_stack,
    _load_dp_agent,
    _make_live_env,
    _parse_seed_from_text,
    jsonable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-chunk-json", required=True)
    parser.add_argument("--snapshot-state-h5", required=True)
    parser.add_argument("--history-action-state-json", required=True)
    parser.add_argument("--source-h5", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--dp-manifest",
        default=str(ROOT / "experiments/dp_peg1000/run_90201/manifest.json"),
    )
    parser.add_argument(
        "--dp-checkpoint",
        default=str(ROOT / "experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_at_end.pt"),
    )
    parser.add_argument("--dp-state-key", choices=("ema_agent", "agent"), default="ema_agent")
    parser.add_argument("--candidate-name", default="cosmos_policy_droid_live_snapshot_10step")
    parser.add_argument("--execute-steps", type=int, default=8)
    parser.add_argument("--robot-action-dim", type=int, default=7)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--expected-video-frames", type=int, default=301)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--expected-action-dim", type=int, default=32)
    parser.add_argument("--clip-live-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-step-records", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--external-target-mode", choices=("source_env_state", "none"), default="source_env_state")
    parser.add_argument("--dp-rollout-continuability-horizon", type=int, default=96)
    parser.add_argument("--dp-rollout-continuability-min-stable-steps", type=int, default=4)
    parser.add_argument("--contact-stable-min-rel-x", type=float, default=-0.06)
    parser.add_argument("--contact-stable-max-rel-x", type=float, default=0.03)
    parser.add_argument("--contact-stable-max-abs-y", type=float, default=0.018)
    parser.add_argument("--contact-stable-max-abs-z", type=float, default=0.012)
    parser.add_argument("--continuability-min-rel-x", type=float, default=-0.08)
    parser.add_argument("--continuability-max-rel-x", type=float, default=0.04)
    parser.add_argument("--continuability-max-abs-y", type=float, default=0.025)
    parser.add_argument("--continuability-max-abs-z", type=float, default=0.025)
    parser.add_argument("--continuability-max-hole-speed", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260624)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> int:
    args = parse_args()
    require_compute_step()

    action_path = Path(args.action_chunk_json).resolve()
    snapshot_path = Path(args.snapshot_state_h5).resolve()
    history_path = Path(args.history_action_state_json).resolve()
    source_h5 = Path(args.source_h5).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    chunk = read_json(action_path)
    prefix_frame = int(chunk["prefix_frame_index"])
    actions = np.asarray(chunk["denormalized_robot_action_chunk"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < int(args.robot_action_dim):
        raise ValueError(f"invalid action chunk shape {actions.shape}")
    actions = actions[:, : int(args.robot_action_dim)]
    execute_steps = min(int(args.execute_steps), int(actions.shape[0]))
    if execute_steps <= 0:
        raise ValueError("no action steps to execute")

    stack = _import_live_control_stack(ROOT)
    env = _make_live_env(stack, Path(args.dp_manifest).resolve(), args)
    base_env = _get_base_env(env)
    low, high = _action_space_bounds(env, int(args.robot_action_dim))
    env_states = load_env_states(source_h5, stack["trajectory_utils"])
    if prefix_frame < 0 or prefix_frame >= len(env_states):
        raise IndexError(f"prefix_frame {prefix_frame} outside env state length {len(env_states)}")
    snapshot_state, snapshot_attrs = load_snapshot_state(snapshot_path)
    history = load_history(history_path)

    reset_seed = _parse_seed_from_text(" ".join([source_h5.name, str(chunk.get("sample_output_json", ""))]))
    env.reset(seed=int(reset_seed if reset_seed is not None else args.seed))

    dp_agent = None
    dp_args = None
    if int(args.dp_rollout_continuability_horizon) > 0:
        torch = stack["torch"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dp_agent, dp_args = _load_dp_agent(env, stack, args, device)

    label = replay_bank_candidate(
        env=env,
        base_env=base_env,
        stack=stack,
        snapshot_state=snapshot_state,
        env_states=env_states,
        history_template=history,
        prefix_frame=prefix_frame,
        candidate_name=str(args.candidate_name),
        candidate_index=0,
        selected=True,
        actions_full=actions,
        execute_steps=execute_steps,
        candidate_meta={
            "candidate_source": "cosmos_policy_droid_live_prefix_snapshot",
            "action_chunk_json": str(action_path),
            "sample_output_json": str(chunk.get("sample_output_json", "")),
            "snapshot_state_h5": str(snapshot_path),
            "history_action_state_json": str(history_path),
            "source_h5": str(source_h5),
            "snapshot_attrs": snapshot_attrs,
        },
        low=low,
        high=high,
        args=args,
        dp_agent=dp_agent,
        dp_args=dp_args,
    )
    label.update(
        {
            "schema": "policy_droid_live_snapshot_action_replay_v1",
            "source_h5": str(source_h5),
            "snapshot_state_h5": str(snapshot_path),
            "history_action_state_json": str(history_path),
            "action_chunk_json": str(action_path),
            "sample_output_json": str(chunk.get("sample_output_json", "")),
            "execute_steps_requested": int(args.execute_steps),
            "action_chunk_stats": {
                "shape": list(actions.shape),
                "finite": bool(np.isfinite(actions).all()),
                "min": float(np.min(actions)),
                "max": float(np.max(actions)),
                "mean_abs": float(np.mean(np.abs(actions))),
            },
            "boundary": (
                "Diagnostic replay of one Cosmos Policy-DROID action chunk "
                "from a saved dynamic live simulator snapshot. This checks "
                "physical effect and DP handoff from the real takeover state; "
                "it is still not full receding controller evidence until the "
                "action is generated from that same live prefix and reviewed "
                "with video."
            ),
        }
    )
    write_json(output_root / "policy_droid_snapshot_action_replay_label.json", label)
    summary = {
        "label_path": str(output_root / "policy_droid_snapshot_action_replay_label.json"),
        "source_h5": str(source_h5),
        "snapshot_state_h5": str(snapshot_path),
        "history_action_state_json": str(history_path),
        "action_chunk_json": str(action_path),
        "prefix_frame_index": int(prefix_frame),
        "execute_steps_actual": int(label.get("execute_steps_actual", 0)),
        "after_success": bool(label.get("after_success", False)),
        "after_continuability_gate": label.get("after_continuability_gate"),
        "after_contact_stable_proxy": bool(label.get("after_contact_stable_proxy", False)),
        "after_grasped": bool(label.get("after_grasped", False)),
        "after_inserted_live_pose": bool(label.get("after_inserted_live_pose", False)),
        "before_rel_metrics": label.get("before_rel_metrics"),
        "after_rel_metrics": label.get("after_rel_metrics"),
        "delta_abs_yz_sum": label.get("delta_abs_yz_sum"),
        "delta_yz_l2": label.get("delta_yz_l2"),
        "dp_rollout_continuability": label.get("dp_rollout_continuability"),
        "boundary": label["boundary"],
    }
    write_json(output_root / "policy_droid_snapshot_action_replay_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
