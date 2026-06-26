#!/usr/bin/env python3
"""Export frozen-DP prior action chunks for executor dataset rows."""

from __future__ import annotations

import argparse
from collections import Counter
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

from run_cosmos3_live_receding_loop import apply_external_target_pose, read_state_obs, require_compute_step  # noqa: E402
from run_cosmos3_receding_closed_loop import (  # noqa: E402
    _action_space_bounds,
    _get_base_env,
    _import_live_control_stack,
    _load_dp_agent,
    _make_live_env,
    _parse_seed_from_text,
    _prepare_step_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executor-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dp-manifest", default=str(ROOT / "experiments/dp_peg1000/run_90201/manifest.json"))
    parser.add_argument(
        "--dp-checkpoint",
        default=str(ROOT / "experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_at_end.pt"),
    )
    parser.add_argument("--dp-state-key", choices=("ema_agent", "agent"), default="ema_agent")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--robot-action-dim", type=int, default=7)
    parser.add_argument("--clip-live-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rollout-horizon", type=int, default=0)
    parser.add_argument(
        "--external-target-mode",
        choices=("source_env_state", "none"),
        default="source_env_state",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def load_env_states(source_h5: Path, trajectory_utils: Any) -> list[dict[str, Any]]:
    import h5py

    with h5py.File(source_h5, "r") as h5:
        traj_name = "traj_0" if "traj_0" in h5 else next(iter(h5.keys()))
        return trajectory_utils.dict_to_list_of_dicts(h5[traj_name]["env_states"])


def export_one(
    *,
    row: dict[str, Any],
    sample_index: int,
    output_root: Path,
    env: Any,
    stack: dict[str, Any],
    dp_agent: Any,
    dp_device: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, str | None]:
    source_h5_value = row.get("source_h5")
    if not source_h5_value:
        return None, "missing_source_h5"
    source_h5 = Path(str(source_h5_value)).resolve()
    if not source_h5.is_file():
        return None, "source_h5_not_found"
    prefix_frame = int(row.get("prefix_frame_index", -1))
    if prefix_frame < 0:
        return None, "missing_prefix_frame"

    env_states = load_env_states(source_h5, stack["trajectory_utils"])
    if len(env_states) < 2:
        return None, "too_few_env_states"
    prefix_idx = max(0, min(prefix_frame, len(env_states) - 1))
    prev_idx = max(0, prefix_idx - 1)
    reset_seed = _parse_seed_from_text(" ".join([str(row.get("source_uuid", "")), source_h5.name])) or 0

    env.reset(seed=reset_seed)
    base_env = _get_base_env(env)
    base_env.set_state_dict(env_states[prev_idx])
    prev_obs = read_state_obs(base_env, stack)
    base_env.set_state_dict(env_states[prefix_idx])
    current_obs = read_state_obs(base_env, stack)

    torch = stack["torch"]
    low, high = _action_space_bounds(env, int(args.robot_action_dim))

    executed: list[list[float]] = []
    action_records: list[dict[str, Any]] = []
    if int(args.rollout_horizon) > 0:
        target_horizon = min(
            int(args.rollout_horizon),
            max(0, int(args.max_episode_steps) - int(prefix_frame)),
            max(0, len(env_states) - 1 - int(prefix_frame)),
        )
        if target_horizon <= 0:
            return None, "empty_rollout_horizon"
        obs_history = [prev_obs, current_obs]
        cached_actions = np.zeros((0, int(args.robot_action_dim)), dtype=np.float32)
        cached_cursor = 0
        for local_idx in range(target_horizon):
            if cached_cursor >= int(cached_actions.shape[0]):
                obs_seq = np.stack(obs_history[-2:], axis=0)[None].astype(np.float32)
                obs_tensor = torch.as_tensor(obs_seq, device=dp_device, dtype=torch.float32)
                with torch.no_grad():
                    action_seq = dp_agent.get_action(obs_tensor)
                action_seq_np = action_seq.detach().cpu().numpy()
                if action_seq_np.ndim != 3 or action_seq_np.shape[0] != 1:
                    return None, f"invalid_dp_action_shape:{tuple(action_seq_np.shape)}"
                cached_actions = action_seq_np[0].astype(np.float32)
                cached_cursor = 0
                if cached_actions.shape[0] <= 0:
                    return None, "empty_dp_action_sequence"
            step_action, action_record = _prepare_step_action(
                cached_actions[cached_cursor],
                low,
                high,
                bool(args.clip_live_actions),
            )
            cached_cursor += 1
            _obs, _reward, terminated, truncated, _info = env.step(step_action)
            source_frame = min(int(prefix_frame) + local_idx + 1, len(env_states) - 1)
            external_target = apply_external_target_pose(
                base_env=base_env,
                stack=stack,
                env_states=env_states,
                source_frame=source_frame,
                args=args,
            )
            next_obs = read_state_obs(base_env, stack)
            obs_history = [obs_history[-1], next_obs]
            executed.append([float(x) for x in action_record["executed"]])
            action_records.append(
                {
                    **action_record,
                    "local_step": int(local_idx),
                    "source_frame": int(source_frame),
                    "external_target": jsonable(external_target),
                    "terminated": jsonable(terminated),
                    "truncated": jsonable(truncated),
                }
            )
            if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                break
    else:
        obs_seq = np.stack([prev_obs, current_obs], axis=0)[None].astype(np.float32)
        obs_tensor = torch.as_tensor(obs_seq, device=dp_device, dtype=torch.float32)
        with torch.no_grad():
            action_seq = dp_agent.get_action(obs_tensor)
        action_seq_np = action_seq.detach().cpu().numpy()
        if action_seq_np.ndim != 3 or action_seq_np.shape[0] != 1:
            return None, f"invalid_dp_action_shape:{tuple(action_seq_np.shape)}"
        action_seq_np = action_seq_np[0]
        for local_idx in range(action_seq_np.shape[0]):
            _, action_record = _prepare_step_action(
                action_seq_np[local_idx],
                low,
                high,
                bool(args.clip_live_actions),
            )
            executed.append([float(x) for x in action_record["executed"]])
            action_records.append(action_record)
    if not executed:
        return None, "empty_dp_action_sequence"

    prior = np.asarray(executed, dtype=np.float32)
    sample_rel = Path("dp_prior_chunks") / f"{sample_index:06d}_{row['uuid']}.npz"
    sample_path = output_root / sample_rel
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sample_path,
        dp_prior_actions=prior,
        prefix_frame=np.asarray([prefix_frame], dtype=np.int32),
        env_state_history_frames=np.asarray([prev_idx, prefix_idx], dtype=np.int32),
    )

    return (
        {
            "uuid": row["uuid"],
            "source_uuid": row.get("source_uuid"),
            "scenario": row.get("scenario"),
            "prefix_role": row.get("prefix_role"),
            "split": row.get("split"),
            "source_h5": str(source_h5),
            "executor_sample_npz": row.get("sample_npz"),
            "dp_prior_npz": str(sample_path),
            "dp_prior_npz_rel": str(sample_rel),
            "prefix_frame_index": prefix_frame,
            "env_state_history_frames": [prev_idx, prefix_idx],
            "dp_prior_shape": list(prior.shape),
            "dp_prior_horizon": int(prior.shape[0]),
            "rollout_horizon_request": int(args.rollout_horizon),
            "external_target_mode": str(args.external_target_mode),
            "robot_action_dim": int(prior.shape[1]),
            "action_records": action_records,
            "dp_checkpoint": str(Path(args.dp_checkpoint).resolve()),
            "dp_state_key": args.dp_state_key,
            "boundary": (
                "Frozen static DP proposal only. This is a prior input for "
                "the executor, not a claim that DP solves the dynamic task."
            ),
        },
        None,
    )


def main() -> int:
    args = parse_args()
    require_compute_step()

    executor_jsonl = Path(args.executor_jsonl).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(executor_jsonl)
    if not rows:
        raise SystemExit(f"empty executor jsonl: {executor_jsonl}")

    stack = _import_live_control_stack(ROOT)
    torch = stack["torch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = _make_live_env(stack, Path(args.dp_manifest).resolve(), args)
    records: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    try:
        dp_agent, dp_args = _load_dp_agent(env, stack, args, device)
        dp_device = next(dp_agent.parameters()).device
        for row in rows:
            if args.max_samples > 0 and len(records) >= args.max_samples:
                break
            record, reason = export_one(
                row=row,
                sample_index=len(records),
                output_root=output_root,
                env=env,
                stack=stack,
                dp_agent=dp_agent,
                dp_device=dp_device,
                args=args,
            )
            if record is None:
                failures[str(reason)] += 1
                continue
            records.append(record)
    finally:
        env.close()

    write_jsonl(output_root / "dp_prior_dataset_file.jsonl", records)
    summary = {
        "schema": "cosmos3_executor_dp_prior_chunks_v1",
        "executor_jsonl": str(executor_jsonl),
        "output_root": str(output_root),
        "num_records": len(records),
        "failure_counts": dict(sorted(failures.items())),
        "dp_checkpoint": str(Path(args.dp_checkpoint).resolve()),
        "dp_manifest": str(Path(args.dp_manifest).resolve()),
        "dp_state_key": args.dp_state_key,
        "rollout_horizon": int(args.rollout_horizon),
        "external_target_mode": str(args.external_target_mode),
        "device": str(device),
        "ready_for_debug_executor_overfit": len(records) >= 2,
        "ready_for_formal_executor_training": False,
        "formal_training_blocker": (
            "DP prior is present for these rows, but formal executor training "
            "still needs causal Cosmos-predicted task paths and visual/closed-loop gates."
        ),
    }
    write_json(output_root / "dp_prior_summary.json", summary)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    return 0 if records else 65


if __name__ == "__main__":
    raise SystemExit(main())
