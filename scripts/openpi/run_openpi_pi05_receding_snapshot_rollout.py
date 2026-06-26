#!/usr/bin/env python3
"""Run split-env OpenPI pi0.5 receding control from saved live snapshots.

The project Python process owns ManiSkill/SAPIEN state and execution. Each
OpenPI query is delegated to the official OpenPI environment by calling the
existing prepared-observation inference helper in a subprocess.
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

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
WORLD_MODEL_DIR = ROOT / "scripts" / "world_model"
OPENPI_ROOT = Path(os.environ.get("OPENPI_ROOT", "/public/home/yanhongru/ICLR2027/openpi")).resolve()
for path in (str(WORLD_MODEL_DIR), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from replay_cosmos3_live_action_bank_from_snapshots import (  # noqa: E402
    _prepare_step_action,
    apply_external_target_pose,
    contact_progress_proxy,
    contact_stable_proxy,
    iter_dirs_for_summary,
    load_env_states,
    load_history,
    load_snapshot_state,
    rel_metrics,
    sample_summary_paths,
)
from run_cosmos3_live_receding_loop import (  # noqa: E402
    fill_live_history_row,
    live_pose_row,
    require_compute_step,
)
from run_cosmos3_receding_closed_loop import (  # noqa: E402
    _action_space_bounds,
    _get_base_env,
    _import_live_control_stack,
    _live_eval,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-root", default=str(DEFAULT_PANEL_ROOT))
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--default-prompt", default="insert the peg into the current target hole")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-iter-dirs", type=int, default=1)
    parser.add_argument("--iteration-indices", default="")
    parser.add_argument("--scenario-regex", default="")
    parser.add_argument("--sample-name-regex", default="")
    parser.add_argument("--max-receding-queries", type=int, default=3)
    parser.add_argument("--execute-steps-per-query", type=int, default=4)
    parser.add_argument("--robot-action-dim", type=int, default=7)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--expected-video-frames", type=int, default=301)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--expected-action-dim", type=int, default=32)
    parser.add_argument("--clip-live-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--external-target-mode", choices=("source_env_state", "none"), default="source_env_state")
    parser.add_argument("--image-source", choices=("observed_prefix_static", "live_render"), default="observed_prefix_static")
    parser.add_argument("--openpi-python", default=os.environ.get("OPENPI_PYTHON", ""))
    parser.add_argument("--uv-run-python", default=os.environ.get("UV_RUN_PYTHON", ""))
    parser.add_argument("--uv-python-platform", default=os.environ.get("UV_PYTHON_PLATFORM", ""))
    parser.add_argument("--contact-stable-min-rel-x", type=float, default=-0.06)
    parser.add_argument("--contact-stable-max-rel-x", type=float, default=0.03)
    parser.add_argument("--contact-stable-max-abs-y", type=float, default=0.018)
    parser.add_argument("--contact-stable-max-abs-z", type=float, default=0.012)
    parser.add_argument("--dp-manifest", default=str(ROOT / "experiments/dp_peg1000/run_90201/manifest.json"))
    parser.add_argument("--seed", type=int, default=20260626)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def parse_ints(text: str) -> set[int]:
    return {int(x.strip()) for x in str(text).split(",") if x.strip()}


def read_last_video_frame(path: Path) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open observed prefix video: {path}")
    last = None
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            last = frame_bgr[..., ::-1].copy()
    finally:
        cap.release()
    if last is None:
        raise RuntimeError(f"no frames decoded from observed prefix video: {path}")
    return np.ascontiguousarray(last.astype(np.uint8))


def object17_from_live(live: dict[str, Any]) -> np.ndarray:
    parts = [
        np.asarray(live["tcp_pose"], dtype=np.float32).reshape(-1)[:3],
        np.asarray(live["peg_pose"], dtype=np.float32).reshape(-1)[:3],
        np.asarray(live["hole_pose"], dtype=np.float32).reshape(-1)[:3],
        np.asarray(live["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3],
        np.asarray(live["hole_velocity"], dtype=np.float32).reshape(-1)[:3],
        np.asarray([float(bool(live["grasped"])), float(bool(live["inserted"]))], dtype=np.float32),
    ]
    state = np.concatenate(parts, axis=0).astype(np.float32)
    if state.shape != (17,):
        raise RuntimeError(f"object17 state has invalid shape {state.shape}")
    return state


def filtered_summary_paths(args: argparse.Namespace) -> list[Path]:
    paths = sample_summary_paths(Path(args.panel_root).resolve())
    scenario_regex = re.compile(str(args.scenario_regex)) if str(args.scenario_regex).strip() else None
    sample_name_regex = re.compile(str(args.sample_name_regex)) if str(args.sample_name_regex).strip() else None
    out: list[Path] = []
    for path in paths:
        summary = read_json(path)
        scenario_ok = scenario_regex is None or bool(scenario_regex.search(str(summary.get("scenario", ""))))
        sample_ok = sample_name_regex is None or bool(sample_name_regex.search(str(summary.get("sample_name", ""))))
        if scenario_ok and sample_ok:
            out.append(path)
    return out[: int(args.max_samples)] if int(args.max_samples) > 0 else out


def openpi_command(args: argparse.Namespace) -> list[str]:
    if str(args.openpi_python).strip():
        return [str(Path(args.openpi_python).resolve())]
    cmd = ["uv", "run", "--frozen"]
    if str(args.uv_run_python).strip():
        cmd.extend(["--python", str(args.uv_run_python)])
    if str(args.uv_python_platform).strip():
        cmd.extend(["--python-platform", str(args.uv_python_platform)])
    return cmd


def infer_one_action_chunk(
    *,
    args: argparse.Namespace,
    query_dir: Path,
    observation_npz: Path,
    source_h5: Path,
    summary_path: Path,
    iter_dir: Path,
    snapshot_path: Path,
    history_path: Path,
    current_frame: int,
    receding_index: int,
    prompt: str,
    state: np.ndarray,
    image: np.ndarray,
) -> Path:
    manifest = {
        "schema": "openpi_pi05_receding_prepared_manifest_v1",
        "rows": [
            {
                "schema": "openpi_pi05_receding_prepared_observation_v1",
                "observation_npz": str(observation_npz),
                "source_h5": str(source_h5),
                "summary_path": str(summary_path),
                "iter_dir": str(iter_dir),
                "snapshot_state_h5": str(snapshot_path),
                "history_action_state_json": str(history_path),
                "prefix_frame_index": int(current_frame),
                "iteration": int(receding_index),
                "prompt": prompt,
                "image_shape": list(image.shape),
                "state": state.astype(float).tolist(),
                "state_mode": "object_state17_live",
                "observation_source": str(args.image_source),
            }
        ],
    }
    manifest_path = query_dir / "prepared_observations_manifest.json"
    write_json(manifest_path, manifest)
    cmd = openpi_command(args) + [
        str(ROOT / "scripts/openpi/infer_openpi_pi05_from_prepared_observations.py"),
        "--prepared-manifest",
        str(manifest_path),
        "--config-name",
        str(args.config_name),
        "--checkpoint-dir",
        str(Path(args.checkpoint_dir).resolve()),
        "--output-root",
        str(query_dir),
        "--execute-steps",
        str(args.execute_steps_per_query),
    ]
    env = os.environ.copy()
    env["OPENPI_ROOT"] = str(OPENPI_ROOT)
    with (query_dir / "openpi_infer.log").open("w") as log:
        subprocess.run(cmd, cwd=str(OPENPI_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    paths = sorted((query_dir / "action_chunks").glob("*.action_chunk.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one action chunk under {query_dir}, found {len(paths)}")
    return paths[0]


def main() -> int:
    args = parse_args()
    require_compute_step()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / "run_manifest.json",
        {
            "schema": "openpi_pi05_receding_snapshot_rollout_manifest_v1",
            "config_name": str(args.config_name),
            "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
            "panel_root": str(Path(args.panel_root).resolve()),
            "output_root": str(output_root),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "host": os.uname().nodename,
            "args": vars(args),
            "boundary": (
                "Receding diagnostic using official OpenPI pi0.5 inference in a subprocess and "
                "ManiSkill live execution in the project environment. It refreshes privileged "
                "object17 simulator state after each short OpenPI chunk; this is an upper-bound "
                "diagnostic, not final RGB-derived method evidence."
            ),
        },
    )

    stack = _import_live_control_stack(ROOT)
    env = _make_live_env(stack, Path(args.dp_manifest).resolve(), args)
    base_env = _get_base_env(env)
    low, high = _action_space_bounds(env, int(args.robot_action_dim))
    iteration_filter = parse_ints(str(args.iteration_indices))
    source_cache: dict[str, list[dict[str, Any]]] = {}
    labels: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for summary_path in filtered_summary_paths(args):
        sample_summary = read_json(summary_path)
        source_h5 = Path(str(sample_summary["source_h5"])).resolve()
        source_key = str(source_h5)
        if source_key not in source_cache:
            source_cache[source_key] = load_env_states(source_h5, stack["trajectory_utils"])
        env_states = source_cache[source_key]
        reset_seed = _parse_seed_from_text(" ".join([source_h5.name, str(sample_summary.get("sample_name", ""))]))
        for iter_dir in iter_dirs_for_summary(summary_path, int(args.max_iter_dirs)):
            try:
                bank_npz = np.load(str(iter_dir / "candidate_action_bank.npz"), allow_pickle=False)
                bank = {key: bank_npz[key] for key in bank_npz.files}
                prefix_frame = int(np.asarray(bank.get("prefix_frame_index", [-1])).reshape(-1)[0])
                iteration = int(np.asarray(bank.get("iteration", [-1])).reshape(-1)[0])
                if iteration_filter and iteration not in iteration_filter:
                    continue
                if prefix_frame < 0:
                    raise RuntimeError(f"{iter_dir} missing prefix_frame_index")
                snapshot_path = iter_dir / "live_state_before_controller.h5"
                history_path = iter_dir / "live_history_raw_action_state.json"
                snapshot_state, snapshot_attrs = load_snapshot_state(snapshot_path)
                history = load_history(history_path)
                env.reset(seed=int(reset_seed if reset_seed is not None else args.seed))
                base_env.set_state_dict(snapshot_state)
                static_image = read_last_video_frame(iter_dir / "observed_prefix.mp4")
                current_frame = int(prefix_frame)
                previous_hole_xyz = None
                start_live = live_pose_row(base_env, stack, previous_hole_xyz)
                before_eval = _live_eval(base_env)
                previous_hole_xyz = np.asarray(start_live["hole_xyz"], dtype=np.float32).copy()
                before_rel = np.asarray(start_live["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
                step_records: list[dict[str, Any]] = []
                query_records: list[dict[str, Any]] = []
                stop_reason = "max_receding_queries_exhausted"
                prompt = f"{args.default_prompt}; scenario {sample_summary.get('scenario', 'unknown')}"
                last_live = start_live

                for receding_i in range(int(args.max_receding_queries)):
                    live = live_pose_row(base_env, stack, previous_hole_xyz)
                    previous_hole_xyz = np.asarray(live["hole_xyz"], dtype=np.float32).copy()
                    state = object17_from_live(live)
                    image = _render_frame(env) if args.image_source == "live_render" else static_image
                    if image.shape[-1] == 4:
                        image = image[..., :3]
                    image = np.ascontiguousarray(image.astype(np.uint8))
                    query_dir = output_root / "queries" / f"{summary_path.parent.name}_iter{iteration:03d}_q{receding_i:02d}_f{current_frame:03d}"
                    query_dir.mkdir(parents=True, exist_ok=True)
                    obs_npz = query_dir / "observation.npz"
                    np.savez_compressed(obs_npz, image=image, wrist_image=image.copy(), state=state)
                    action_path = infer_one_action_chunk(
                        args=args,
                        query_dir=query_dir,
                        observation_npz=obs_npz,
                        source_h5=source_h5,
                        summary_path=summary_path,
                        iter_dir=iter_dir,
                        snapshot_path=snapshot_path,
                        history_path=history_path,
                        current_frame=current_frame,
                        receding_index=receding_i,
                        prompt=prompt,
                        state=state,
                        image=image,
                    )
                    action_payload = json.loads(action_path.read_text())
                    actions = np.asarray(action_payload["denormalized_robot_action_chunk"], dtype=np.float32)
                    actions = actions[:, : int(args.robot_action_dim)]
                    execute_steps = min(int(args.execute_steps_per_query), int(actions.shape[0]))
                    query_records.append(
                        {
                            "receding_index": int(receding_i),
                            "current_frame_before_query": int(current_frame),
                            "action_chunk_json": str(action_path),
                            "state": state.astype(float).tolist(),
                            "live": {
                                "peg_head_at_hole": np.asarray(live["peg_head_at_hole"], dtype=np.float32)
                                .reshape(-1)[:3]
                                .astype(float)
                                .tolist(),
                                "grasped": bool(live["grasped"]),
                                "inserted": bool(live["inserted"]),
                            },
                            "execute_steps": int(execute_steps),
                        }
                    )
                    for local_i, action in enumerate(actions[:execute_steps]):
                        if current_frame >= int(args.max_episode_steps) or current_frame >= len(env_states) - 1:
                            stop_reason = "episode_horizon_exhausted"
                            break
                        step_action, action_record = _prepare_step_action(action, low, high, bool(args.clip_live_actions))
                        _obs, reward, terminated, truncated, _info = env.step(step_action)
                        source_frame = min(current_frame + 1, len(env_states) - 1)
                        external_target = apply_external_target_pose(
                            base_env=base_env,
                            stack=stack,
                            env_states=env_states,
                            source_frame=source_frame,
                            args=args,
                        )
                        last_live = live_pose_row(base_env, stack, previous_hole_xyz)
                        previous_hole_xyz = np.asarray(last_live["hole_xyz"], dtype=np.float32).copy()
                        if 0 <= current_frame < history.shape[0]:
                            fill_live_history_row(history, current_frame, np.asarray(action_record["executed"]), last_live)
                        step_eval = _live_eval(base_env)
                        step_rel = np.asarray(last_live["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
                        step_records.append(
                            {
                                "receding_index": int(receding_i),
                                "chunk_local_step": int(local_i),
                                "local_step": int(len(step_records)),
                                "global_action_index": int(current_frame),
                                "source_frame": int(source_frame),
                                "action": action_record,
                                "external_target": external_target,
                                "reward": jsonable(reward),
                                "terminated": jsonable(terminated),
                                "truncated": jsonable(truncated),
                                "live_eval": step_eval,
                                "peg_head_at_hole": step_rel.astype(float).tolist(),
                                "rel_metrics": rel_metrics(step_rel),
                                "grasped": bool(last_live["grasped"]),
                                "inserted": bool(last_live["inserted"]),
                            }
                        )
                        current_frame += 1
                        if bool(step_eval.get("success", False)):
                            stop_reason = "success"
                            break
                        if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                            stop_reason = "terminated_or_truncated"
                            break
                    if stop_reason != "max_receding_queries_exhausted":
                        break

                after_eval = _live_eval(base_env)
                after_rel = np.asarray(last_live["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
                after_grasped = bool(last_live.get("grasped", False))
                after_inserted = bool(last_live.get("inserted", False))
                before_m = rel_metrics(before_rel)
                after_m = rel_metrics(after_rel)
                label = {
                    "schema": "openpi_pi05_receding_snapshot_rollout_label_v1",
                    "source_h5": str(source_h5),
                    "summary_path": str(summary_path),
                    "iter_dir": str(iter_dir),
                    "snapshot_state_h5": str(snapshot_path),
                    "history_action_state_json": str(history_path),
                    "snapshot_attrs": snapshot_attrs,
                    "prefix_frame_index": int(prefix_frame),
                    "final_frame_index": int(current_frame),
                    "query_count": int(len(query_records)),
                    "execute_steps_actual": int(len(step_records)),
                    "stop_reason": stop_reason,
                    "before_eval": before_eval,
                    "after_eval": after_eval,
                    "before_peg_head_at_hole": before_rel.astype(float).tolist(),
                    "after_peg_head_at_hole": after_rel.astype(float).tolist(),
                    "before_rel_metrics": before_m,
                    "after_rel_metrics": after_m,
                    "delta_abs_yz_sum": float(after_m["abs_yz_sum"] - before_m["abs_yz_sum"]),
                    "delta_yz_l2": float(after_m["yz_l2"] - before_m["yz_l2"]),
                    "after_grasped": after_grasped,
                    "after_inserted_live_pose": after_inserted,
                    "after_contact_progress_proxy": contact_progress_proxy(
                        after_rel, grasped=after_grasped, inserted=after_inserted
                    ),
                    "after_contact_stable_proxy": bool(
                        contact_stable_proxy(after_rel, grasped=after_grasped, inserted=after_inserted, args=args)
                    ),
                    "after_success": bool(after_eval.get("success", False)),
                    "query_records": query_records,
                    "step_records": step_records,
                    "boundary": (
                        "Official OpenPI pi0.5 receding diagnostic with refreshed simulator object17 state. "
                        "This tests whether short-query receding execution fixes the open-loop chunk failure. "
                        "It is privileged-state upper-bound evidence, not final RGB-derived method evidence."
                    ),
                }
                label_path = output_root / "labels" / f"{summary_path.parent.name}_iter{iteration:03d}_receding_label.json"
                write_json(label_path, label)
                labels.append({**label, "label_path": str(label_path)})
            except Exception as exc:  # noqa: BLE001
                failure = {
                    "schema": "openpi_pi05_receding_snapshot_rollout_failure_v1",
                    "summary_path": str(summary_path),
                    "iter_dir": str(iter_dir),
                    "failure_type": type(exc).__name__,
                    "failure": str(exc),
                }
                failures.append(failure)

    summary = {
        "schema": "openpi_pi05_receding_snapshot_rollout_summary_v1",
        "output_root": str(output_root),
        "label_count": len(labels),
        "failure_count": len(failures),
        "after_success_count": sum(bool(x.get("after_success")) for x in labels),
        "after_inserted_live_pose_count": sum(bool(x.get("after_inserted_live_pose")) for x in labels),
        "after_contact_stable_proxy_count": sum(bool(x.get("after_contact_stable_proxy")) for x in labels),
        "after_grasped_count": sum(bool(x.get("after_grasped")) for x in labels),
        "labels": [
            {
                "label_path": x.get("label_path"),
                "source_h5": x.get("source_h5"),
                "prefix_frame_index": x.get("prefix_frame_index"),
                "final_frame_index": x.get("final_frame_index"),
                "query_count": x.get("query_count"),
                "execute_steps_actual": x.get("execute_steps_actual"),
                "stop_reason": x.get("stop_reason"),
                "after_success": x.get("after_success"),
                "after_inserted_live_pose": x.get("after_inserted_live_pose"),
                "after_contact_stable_proxy": x.get("after_contact_stable_proxy"),
                "after_grasped": x.get("after_grasped"),
                "delta_abs_yz_sum": x.get("delta_abs_yz_sum"),
                "after_peg_head_at_hole": x.get("after_peg_head_at_hole"),
            }
            for x in labels
        ],
        "failures": failures,
        "boundary": (
            "Receding OpenPI diagnostic; counts exclude DP handoff and use refreshed privileged object17 state."
        ),
    }
    write_json(output_root / "openpi_pi05_receding_snapshot_rollout_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
