#!/usr/bin/env python3
"""Prepare RGB/state observations from saved live snapshots for OpenPI.

Default mode intentionally avoids live rendering. Existing panel artifacts
already contain the causal observed prefix video and raw live state history, so
we can construct the OpenPI observation without touching SAPIEN/Vulkan.
"""

from __future__ import annotations

import argparse
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
if str(WORLD_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(WORLD_MODEL_DIR))

from replay_cosmos3_live_action_bank_from_snapshots import (  # noqa: E402
    iter_dirs_for_summary,
    load_history,
    load_snapshot_state,
    sample_summary_paths,
)
from run_cosmos3_live_receding_loop import require_compute_step  # noqa: E402
from run_cosmos3_receding_closed_loop import (  # noqa: E402
    _get_base_env,
    _import_live_control_stack,
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
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--default-prompt", default="insert the peg into the current target hole")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-iter-dirs", type=int, default=1)
    parser.add_argument("--iteration-indices", default="")
    parser.add_argument("--scenario-regex", default="")
    parser.add_argument("--sample-name-regex", default="")
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--dp-manifest", default=str(ROOT / "experiments/dp_peg1000/run_90201/manifest.json"))
    parser.add_argument(
        "--observation-source",
        choices=("observed_prefix", "live_render"),
        default="observed_prefix",
        help="Default uses saved observed_prefix.mp4 plus live_history qpos, avoiding SAPIEN render startup.",
    )
    parser.add_argument(
        "--state-mode",
        choices=("qpos8", "qpos8_rel3_history", "object_state17"),
        default="qpos8",
        help=(
            "OpenPI observation/state layout. qpos8_rel3_history appends the current "
            "causal peg_head_at_hole rel3 from saved live history columns 27:30. "
            "object_state17 uses the causal object/task columns from saved live history."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260625)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def parse_ints(text: str) -> set[int]:
    return {int(x.strip()) for x in str(text).split(",") if x.strip()}


def openpi_state_from_qpos(qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    state = np.zeros((8,), dtype=np.float32)
    state[: min(7, qpos.shape[0])] = qpos[:7]
    if qpos.shape[0] >= 9:
        state[7] = float(np.mean(qpos[7:9]))
    elif qpos.shape[0] >= 8:
        state[7] = float(qpos[7])
    return state


def openpi_state_from_live_qpos(base_env: Any, stack: dict[str, Any]) -> np.ndarray:
    common = stack["common"]
    qpos = common.to_numpy(base_env.agent.robot.get_qpos())[0].astype(np.float32).reshape(-1)
    return openpi_state_from_qpos(qpos)


def openpi_state_from_history(history: np.ndarray, prefix_frame: int, state_mode: str) -> np.ndarray:
    if history.ndim != 2 or history.shape[1] < 18:
        raise RuntimeError(f"history must be [T, >=18] to recover qpos, got {history.shape}")
    row_index = max(0, min(int(prefix_frame), history.shape[0] - 1))
    # current_executor_state_from_live stores qpos in columns 9:18:
    # tcp3, peg3, hole3, qpos9, qvel9, rel3, holevel3, flags2.
    qpos_state = openpi_state_from_qpos(history[row_index, 9:18])
    if state_mode == "qpos8":
        return qpos_state
    if state_mode == "qpos8_rel3_history":
        if history.shape[1] < 30:
            raise RuntimeError(f"history must be [T, >=30] to recover rel3 columns 27:30, got {history.shape}")
        rel3 = np.asarray(history[row_index, 27:30], dtype=np.float32).reshape(3)
        return np.concatenate([qpos_state, rel3], axis=0).astype(np.float32)
    if state_mode == "object_state17":
        if history.shape[1] >= 35:
            parts = [
                history[row_index, 0:3],  # tcp_pose3
                history[row_index, 3:6],  # peg_pose3
                history[row_index, 6:9],  # hole_pose3
                history[row_index, 27:30],  # peg_head_at_hole3
                history[row_index, 30:33],  # hole_velocity_step3
                history[row_index, 33:35],  # grasped, inserted
            ]
        elif history.shape[1] >= 24:
            # Older live-receding panels store action first, then the same
            # causal task fields used by the object17 OpenPI training data:
            # action7, tcp3, peg3, hole3, peg_head_at_hole3, hole_velocity3,
            # grasped, inserted, plus optional diagnostics/progress columns.
            parts = [
                history[row_index, 7:10],  # tcp_pose3
                history[row_index, 10:13],  # peg_pose3
                history[row_index, 13:16],  # hole_pose3
                history[row_index, 16:19],  # peg_head_at_hole3
                history[row_index, 19:22],  # hole_velocity_step3
                history[row_index, 22:24],  # grasped, inserted
            ]
        else:
            raise RuntimeError(f"history must be [T, >=24] to recover object_state17, got {history.shape}")
        return np.concatenate([np.asarray(part, dtype=np.float32).reshape(-1) for part in parts], axis=0).reshape(17)
    raise ValueError(f"unsupported state_mode={state_mode!r}")


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


def main() -> int:
    args = parse_args()
    require_compute_step()
    output_root = Path(args.output_root).resolve()
    obs_dir = output_root / "prepared_observations"
    obs_dir.mkdir(parents=True, exist_ok=True)

    stack = None
    env = None
    base_env = None
    if args.observation_source == "live_render":
        stack = _import_live_control_stack(ROOT)
        env = _make_live_env(stack, Path(args.dp_manifest).resolve(), args)
        base_env = _get_base_env(env)
    iteration_filter = parse_ints(str(args.iteration_indices))

    rows: list[dict[str, Any]] = []
    for summary_path in filtered_summary_paths(args):
        sample_summary = read_json(summary_path)
        source_h5 = Path(str(sample_summary["source_h5"])).resolve()
        reset_seed = _parse_seed_from_text(" ".join([source_h5.name, str(sample_summary.get("sample_name", ""))]))
        for iter_dir in iter_dirs_for_summary(summary_path, int(args.max_iter_dirs)):
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
            snapshot_attrs: dict[str, Any] = {}
            if args.observation_source == "live_render":
                if args.state_mode != "qpos8":
                    raise RuntimeError("live_render currently supports only --state-mode qpos8")
                if stack is None or env is None or base_env is None:
                    raise RuntimeError("live_render requested without initialized environment")
                snapshot_state, snapshot_attrs = load_snapshot_state(snapshot_path)
                env.reset(seed=int(reset_seed if reset_seed is not None else args.seed))
                base_env.set_state_dict(snapshot_state)
                image = _render_frame(env)
                state = openpi_state_from_live_qpos(base_env, stack)
            else:
                image = read_last_video_frame(iter_dir / "observed_prefix.mp4")
                history = load_history(history_path)
                state = openpi_state_from_history(history, prefix_frame, str(args.state_mode))
            if image.shape[-1] == 4:
                image = image[..., :3]
            if image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)
            image = np.ascontiguousarray(image.astype(np.uint8))
            stem = f"iter_{iteration:03d}_f{prefix_frame:03d}_{len(rows):04d}"
            npz_path = obs_dir / f"{stem}.npz"
            np.savez_compressed(npz_path, image=image, wrist_image=image.copy(), state=state)
            prompt = f"{args.default_prompt}; scenario {sample_summary.get('scenario', 'unknown')}"
            row = {
                "schema": "openpi_pi05_prepared_snapshot_observation_v1",
                "observation_npz": str(npz_path),
                "source_h5": str(source_h5),
                "summary_path": str(summary_path),
                "iter_dir": str(iter_dir),
                "snapshot_state_h5": str(snapshot_path),
                "history_action_state_json": str(history_path),
                "prefix_frame_index": int(prefix_frame),
                "iteration": int(iteration),
                "prompt": prompt,
                "image_shape": list(image.shape),
                "state": state.astype(float).tolist(),
                "state_mode": str(args.state_mode),
                "observation_source": str(args.observation_source),
                "state_source": (
                    {
                        "qpos8": "live_history_raw_action_state_qpos_9_18",
                        "qpos8_rel3_history": "live_history_raw_action_state_qpos_9_18_rel3_27_30",
                        "object_state17": "live_history_raw_action_state_tcp_0_3_peg_3_6_hole_6_9_rel3_27_30_holevel_30_33_flags_33_35",
                    }[str(args.state_mode)]
                )
                if args.observation_source == "observed_prefix"
                else "restored_live_env_qpos",
                "image_source": str(iter_dir / "observed_prefix.mp4")
                if args.observation_source == "observed_prefix"
                else "restored_live_env_render",
                "snapshot_attrs": snapshot_attrs,
            }
            write_json(obs_dir / f"{stem}.json", row)
            rows.append(row)

    manifest = {
        "schema": "openpi_pi05_prepared_snapshot_observations_manifest_v1",
        "output_root": str(output_root),
        "count": len(rows),
        "rows": rows,
    }
    write_json(output_root / "prepared_observations_manifest.json", manifest)
    print(json.dumps({"prepared_count": len(rows), "manifest": str(output_root / "prepared_observations_manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
