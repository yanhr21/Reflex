#!/usr/bin/env python3
"""Run OpenPI pi0.5 inference from saved live snapshots and replay actions.

This is a compute-node-only evaluation bridge:

1. restore a saved dynamic live simulator snapshot,
2. render the current RGB observation and build the same 8D qpos state used by
   the 733 LeRobot conversion,
3. call official OpenPI ``policy_config.create_trained_policy`` on a preserved
   pi0.5 checkpoint,
4. replay the predicted short action chunk through the existing saved-snapshot
   evaluator.

The script does not introduce a custom policy model or intermediate network.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
WORLD_MODEL_DIR = ROOT / "scripts" / "world_model"
OPENPI_ROOT = Path(os.environ.get("OPENPI_ROOT", "/public/home/yanhongru/ICLR2027/openpi")).resolve()
for path in (str(WORLD_MODEL_DIR), str(OPENPI_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from replay_cosmos3_live_action_bank_from_snapshots import (  # noqa: E402
    iter_dirs_for_summary,
    load_env_states,
    load_history,
    load_snapshot_state,
    replay_bank_candidate,
    sample_summary_paths,
)
from run_cosmos3_live_receding_loop import require_compute_step  # noqa: E402
from run_cosmos3_receding_closed_loop import (  # noqa: E402
    _action_space_bounds,
    _get_base_env,
    _import_live_control_stack,
    _load_dp_agent,
    _make_live_env,
    _parse_seed_from_text,
    _render_frame,
    jsonable,
    read_json,
)


DEFAULT_PANEL_ROOT = (
    ROOT
    / "experiments/world_model_task_rebinding/cosmos3/"
    "sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/"
    "live_receding_candidate_executor_iter1500_seed20260725_h8192_safegate1_source_suffix_offsets64_48_32_24_panel0134_20260623_alloc146658"
)
DEFAULT_CHECKPOINT = (
    ROOT
    / "experiments/world_model_task_rebinding/openpi/checkpoints_local_preserved/pi05_maniskill_peg733/"
    "pi05_peg733_1gpu1h_20260625_after_nfs_enolck_localckpt_1600_skipnorm_alloc150773/1599"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "experiments/world_model_task_rebinding/openpi/"
    "openpi_pi05_snapshot_replay_20260625_manual"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-root", default=str(DEFAULT_PANEL_ROOT))
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--config-name", default="pi05_maniskill_peg733")
    parser.add_argument("--default-prompt", default="insert the peg into the current target hole")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-iter-dirs", type=int, default=1)
    parser.add_argument("--iteration-indices", default="")
    parser.add_argument("--scenario-regex", default="")
    parser.add_argument("--sample-name-regex", default="")
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
    parser.add_argument("--dp-manifest", default=str(ROOT / "experiments/dp_peg1000/run_90201/manifest.json"))
    parser.add_argument(
        "--dp-checkpoint",
        default=str(ROOT / "experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_at_end.pt"),
    )
    parser.add_argument("--dp-state-key", choices=("ema_agent", "agent"), default="ema_agent")
    parser.add_argument("--contact-stable-min-rel-x", type=float, default=-0.06)
    parser.add_argument("--contact-stable-max-rel-x", type=float, default=0.03)
    parser.add_argument("--contact-stable-max-abs-y", type=float, default=0.018)
    parser.add_argument("--contact-stable-max-abs-z", type=float, default=0.012)
    parser.add_argument("--continuability-min-rel-x", type=float, default=-0.08)
    parser.add_argument("--continuability-max-rel-x", type=float, default=0.04)
    parser.add_argument("--continuability-max-abs-y", type=float, default=0.025)
    parser.add_argument("--continuability-max-abs-z", type=float, default=0.025)
    parser.add_argument("--continuability-max-hole-speed", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260625)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
    os.replace(tmp, path)


def parse_ints(text: str) -> set[int]:
    out: set[int] = set()
    for item in str(text).split(","):
        item = item.strip()
        if item:
            out.add(int(item))
    return out


def openpi_state_from_live_qpos(base_env: Any, stack: dict[str, Any]) -> np.ndarray:
    common = stack["common"]
    qpos = common.to_numpy(base_env.agent.robot.get_qpos())[0].astype(np.float32).reshape(-1)
    state = np.zeros((8,), dtype=np.float32)
    state[: min(7, qpos.shape[0])] = qpos[:7]
    if qpos.shape[0] >= 9:
        state[7] = float(np.mean(qpos[7:9]))
    elif qpos.shape[0] >= 8:
        state[7] = float(qpos[7])
    return state


def make_openpi_observation(env: Any, base_env: Any, stack: dict[str, Any], prompt: str) -> dict[str, Any]:
    image = _render_frame(env)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    image = np.ascontiguousarray(image.astype(np.uint8))
    return {
        "observation/state": openpi_state_from_live_qpos(base_env, stack),
        "observation/image": image,
        "observation/wrist_image": image.copy(),
        "prompt": prompt,
    }


def load_openpi_policy(args: argparse.Namespace) -> Any:
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    train_config = openpi_config.get_config(str(args.config_name))
    return policy_config.create_trained_policy(
        train_config,
        Path(args.checkpoint_dir).resolve(),
        default_prompt=str(args.default_prompt),
    )


def filtered_summary_paths(args: argparse.Namespace) -> list[Path]:
    panel_root = Path(args.panel_root).resolve()
    paths = sample_summary_paths(panel_root)
    scenario_regex = re.compile(str(args.scenario_regex)) if str(args.scenario_regex).strip() else None
    sample_name_regex = re.compile(str(args.sample_name_regex)) if str(args.sample_name_regex).strip() else None
    if scenario_regex is None and sample_name_regex is None:
        return paths[: int(args.max_samples)] if int(args.max_samples) > 0 else paths

    out: list[Path] = []
    for path in paths:
        summary = read_json(path)
        scenario_ok = scenario_regex is None or bool(scenario_regex.search(str(summary.get("scenario", ""))))
        sample_ok = sample_name_regex is None or bool(sample_name_regex.search(str(summary.get("sample_name", ""))))
        if scenario_ok and sample_ok:
            out.append(path)
    return out[: int(args.max_samples)] if int(args.max_samples) > 0 else out


def main() -> int:
    args = parse_args()
    require_compute_step()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / "run_manifest.json",
        {
            "schema": "openpi_pi05_snapshot_replay_manifest_v1",
            "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
            "config_name": str(args.config_name),
            "panel_root": str(Path(args.panel_root).resolve()),
            "output_root": str(output_root),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "host": os.uname().nodename,
            "boundary": (
                "Official OpenPI pi0.5 checkpoint inference plus existing saved-snapshot "
                "ManiSkill replay. No custom action model, VAE, MLP, or scorer-only "
                "selection is introduced."
            ),
            "args": vars(args),
        },
    )

    stack = _import_live_control_stack(ROOT)
    env = _make_live_env(stack, Path(args.dp_manifest).resolve(), args)
    base_env = _get_base_env(env)
    low, high = _action_space_bounds(env, int(args.robot_action_dim))

    dp_agent = None
    dp_args = None
    if int(args.dp_rollout_continuability_horizon) > 0:
        torch = stack["torch"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dp_agent, dp_args = _load_dp_agent(env, stack, args, device)

    policy = load_openpi_policy(args)
    source_cache: dict[str, list[dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    iteration_filter = parse_ints(str(args.iteration_indices))

    for summary_path in filtered_summary_paths(args):
        sample_summary = read_json(summary_path)
        source_h5 = Path(str(sample_summary["source_h5"])).resolve()
        source_key = str(source_h5)
        if source_key not in source_cache:
            source_cache[source_key] = load_env_states(source_h5, stack["trajectory_utils"])
        env_states = source_cache[source_key]
        reset_seed = _parse_seed_from_text(" ".join([source_h5.name, str(sample_summary.get("sample_name", ""))]))
        iter_dirs = iter_dirs_for_summary(summary_path, int(args.max_iter_dirs))
        for iter_dir in iter_dirs:
            try:
                bank_npz = np.load(str(iter_dir / "candidate_action_bank.npz"), allow_pickle=False)
                bank = {key: bank_npz[key] for key in bank_npz.files}
                prefix_frame = int(np.asarray(bank.get("prefix_frame_index", [-1])).reshape(-1)[0])
                iteration = int(np.asarray(bank.get("iteration", [-1])).reshape(-1)[0])
                if iteration_filter and iteration not in iteration_filter:
                    continue
                if prefix_frame < 0:
                    raise ValueError("candidate bank missing prefix_frame_index")

                snapshot_path = iter_dir / "live_state_before_controller.h5"
                history_path = iter_dir / "live_history_raw_action_state.json"
                snapshot_state, snapshot_attrs = load_snapshot_state(snapshot_path)
                history = load_history(history_path)
                env.reset(seed=int(reset_seed if reset_seed is not None else args.seed))
                base_env.set_state_dict(snapshot_state)

                prompt = f"{args.default_prompt}; scenario {sample_summary.get('scenario', 'unknown')}"
                obs = make_openpi_observation(env, base_env, stack, prompt)
                result = policy.infer(obs)
                actions = np.asarray(result["actions"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] < int(args.robot_action_dim):
                    raise ValueError(f"OpenPI returned invalid action shape {actions.shape}")
                actions = actions[:, : int(args.robot_action_dim)]
                execute_steps = min(int(args.execute_steps), int(actions.shape[0]))
                action_record = {
                    "schema": "openpi_pi05_snapshot_action_chunk_v1",
                    "source_h5": str(source_h5),
                    "summary_path": str(summary_path),
                    "iter_dir": str(iter_dir),
                    "snapshot_state_h5": str(snapshot_path),
                    "history_action_state_json": str(history_path),
                    "prefix_frame_index": int(prefix_frame),
                    "iteration": int(iteration),
                    "prompt": prompt,
                    "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
                    "config_name": str(args.config_name),
                    "policy_timing": result.get("policy_timing"),
                    "openpi_observation": {
                        "image_shape": list(obs["observation/image"].shape),
                        "state_shape": list(obs["observation/state"].shape),
                        "state": np.asarray(obs["observation/state"], dtype=np.float32).astype(float).tolist(),
                    },
                    "denormalized_robot_action_chunk": actions.astype(float).tolist(),
                    "action_shape": list(actions.shape),
                    "execute_steps": int(execute_steps),
                }
                action_path = output_root / "action_chunks" / f"iter_{iteration:03d}_f{prefix_frame:03d}_openpi_pi05.action_chunk.json"
                write_json(action_path, action_record)

                label = replay_bank_candidate(
                    env=env,
                    base_env=base_env,
                    stack=stack,
                    snapshot_state=snapshot_state,
                    env_states=env_states,
                    history_template=history,
                    prefix_frame=prefix_frame,
                    candidate_name="openpi_pi05_finetuned_733",
                    candidate_index=0,
                    selected=True,
                    actions_full=actions,
                    execute_steps=execute_steps,
                    candidate_meta={
                        "candidate_source": "openpi_pi05_finetuned_733",
                        "action_chunk_json": str(action_path),
                        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
                        "snapshot_attrs": snapshot_attrs,
                        "sample_summary": {
                            "scenario": sample_summary.get("scenario"),
                            "sample_name": sample_summary.get("sample_name"),
                            "source_h5": str(source_h5),
                        },
                    },
                    low=low,
                    high=high,
                    args=args,
                    dp_agent=dp_agent,
                    dp_args=dp_args,
                )
                label.update(
                    {
                        "schema": "openpi_pi05_live_snapshot_replay_label_v1",
                        "action_chunk_json": str(action_path),
                        "summary_path": str(summary_path),
                        "iter_dir": str(iter_dir),
                        "source_h5": str(source_h5),
                        "snapshot_state_h5": str(snapshot_path),
                        "history_action_state_json": str(history_path),
                        "boundary": (
                            "OpenPI pi0.5 finetuned checkpoint generated the action chunk "
                            "from a causal restored live snapshot observation; the existing "
                            "replay evaluator then measured physical effect. This is "
                            "saved-snapshot evaluation, not full live receding evidence."
                        ),
                    }
                )
                records.append(label)
            except Exception as exc:  # noqa: BLE001
                failures[type(exc).__name__] += 1
                records.append(
                    {
                        "schema": "openpi_pi05_live_snapshot_replay_failure_v1",
                        "summary_path": str(summary_path),
                        "iter_dir": str(iter_dir),
                        "failure_type": type(exc).__name__,
                        "failure": str(exc),
                    }
                )

    write_jsonl(output_root / "openpi_pi05_snapshot_replay_labels.jsonl", records)
    valid = [row for row in records if row.get("schema") == "openpi_pi05_live_snapshot_replay_label_v1"]
    summary = {
        "schema": "openpi_pi05_snapshot_replay_summary_v1",
        "output_root": str(output_root),
        "records": int(len(records)),
        "valid_records": int(len(valid)),
        "failures": int(len(records) - len(valid)),
        "failure_types": dict(sorted(failures.items())),
        "after_success_count": int(sum(1 for row in valid if row.get("after_success"))),
        "after_inserted_live_pose_count": int(sum(1 for row in valid if row.get("after_inserted_live_pose"))),
        "after_contact_stable_proxy_count": int(sum(1 for row in valid if row.get("after_contact_stable_proxy"))),
        "after_grasped_count": int(sum(1 for row in valid if row.get("after_grasped"))),
        "dp96_success_count": int(
            sum(1 for row in valid if bool((row.get("dp_rollout_continuability") or {}).get("success")))
        ),
        "dp96_continuable_count": int(
            sum(1 for row in valid if bool((row.get("dp_rollout_continuability") or {}).get("continuable")))
        ),
        "boundary": (
            "Counts summarize saved-snapshot replay only. They can falsify or support "
            "OpenPI action quality, but video/contact-sheet review is still required "
            "before claiming dynamic insertion progress."
        ),
    }
    write_json(output_root / "openpi_pi05_snapshot_replay_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
