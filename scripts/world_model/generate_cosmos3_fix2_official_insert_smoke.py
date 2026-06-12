#!/usr/bin/env python3
"""Reproduce the fix2/official PegInsertionSide insertion protocol as smoke data.

This is not a fix3 dynamic dataset generator. It first verifies the baseline
physical insertion path that the later fix3 data must preserve: official
motion-planning grasp, refined pre-insert alignment, and final +0.05 m insert.
No target motion enlargement and no constrained peg projection are applied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import sapien
import torch
import tyro

import mani_skill.envs  # noqa: F401
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.utils import common
from mani_skill.utils.wrappers.record import RecordEpisode

from scripts.world_model import generate_cosmos3_fix3_late_trigger_dynamic_experts as fix3


@dataclass
class Args:
    output_root: str = "experiments/world_model_task_rebinding/cosmos3/fix2_official_insert_repro_smoke6_20260611"
    paths_file: str = ""
    trajectory_name: str = "fix2_official_insert_raw_pd_joint_pos"
    env_id: str = "PegInsertionSide-v1"
    obs_mode: str = "state"
    control_mode: str = "pd_joint_pos"
    sim_backend: str = "physx_cpu"
    render_backend: str = "none"
    reward_mode: str = "dense"
    render_mode: str = "none"
    num_demos: int = 6
    max_attempts: int = 60
    seed: int = 1002000
    total_video_frames: int = 301
    total_action_steps: int = 300
    joint_vel_limits: float = 0.75
    joint_acc_limits: float = 0.75
    val_fraction: float = 0.1
    overwrite: bool = False


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _last_success(res: Any) -> bool:
    if res == -1 or not res:
        return False
    try:
        return bool(res[-1]["success"].item())
    except Exception:  # noqa: BLE001
        return False


def _official_insert_episode(env: RecordEpisode, args: Args) -> dict[str, Any] | None:
    base_env = env.unwrapped
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=base_env.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=float(args.joint_vel_limits),
        joint_acc_limits=float(args.joint_acc_limits),
    )
    try:
        episode_start_step = int(env._elapsed_record_steps)
        initial_box_p, initial_box_q = fix3._to_np_pose(base_env.box.pose)
        obb = get_actor_obb(base_env.peg)
        approaching = np.array([0, 0, -1])
        target_closing = base_env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        peg_init_pose = base_env.peg.pose
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=0.025,
        )
        grasp_pose = base_env.agent.build_grasp_pose(
            approaching,
            grasp_info["closing"],
            grasp_info["center"],
        )
        grasp_offset = sapien.Pose([-max(0.05, base_env.peg_half_sizes[0, 0].item() / 2 + 0.01), 0, 0])
        grasp_pose = grasp_pose * grasp_offset

        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        if planner.move_to_pose_with_screw(reach_pose) == -1:
            return None
        if planner.move_to_pose_with_screw(grasp_pose) == -1:
            return None
        planner.close_gripper()

        insert_pose = base_env.goal_pose * peg_init_pose.inv() * grasp_pose
        insert_offset_m = float(-0.01 - base_env.peg_half_sizes[0, 0].item())
        insert_offset = sapien.Pose([insert_offset_m, 0, 0])
        pre_insert_pose = insert_pose * insert_offset
        if planner.move_to_pose_with_screw(pre_insert_pose) == -1:
            return None
        for _ in range(3):
            delta_pose = base_env.goal_pose * insert_offset * base_env.peg.pose.inv()
            pre_insert_pose = delta_pose * pre_insert_pose
            if planner.move_to_pose_with_screw(pre_insert_pose) == -1:
                return None

        final_insert_start_step = int(env._elapsed_record_steps) - int(episode_start_step)
        res = planner.move_to_pose_with_screw(insert_pose * sapien.Pose([0.05, 0, 0]))
        if not _last_success(res):
            return None
        final_box_p, final_box_q = fix3._to_np_pose(base_env.box.pose)
        raw_steps = int(env._elapsed_record_steps) - int(episode_start_step)
        return {
            "scenario": "none",
            "trigger_step": int(args.total_action_steps),
            "move_steps": 0,
            "raw_action_steps": raw_steps,
            "initial_box_xyz": initial_box_p.astype(float).tolist(),
            "final_box_xyz": final_box_p.astype(float).tolist(),
            "initial_box_quat": initial_box_q.astype(float).tolist(),
            "final_box_quat": final_box_q.astype(float).tolist(),
            "target_motion_norm_m": float(np.linalg.norm(final_box_p[:3] - initial_box_p[:3])),
            "motion_path_xyz": [initial_box_p.astype(float).tolist(), final_box_p.astype(float).tolist()],
            "final_insert_start_step": int(final_insert_start_step),
            "insert_offset_m": float(insert_offset_m),
            "final_insert_offset_m": 0.05,
            "insert_step_m": 0.05,
            "preinsert_refine_steps": 3,
            "final_insert_refine_steps": 0,
            "final_insert_settle_steps": 0,
            "trigger_reason": "static fix2/official insertion reproduction; no target switch",
            "target_motion_policy": "target stays static; official solver performs grasp, refined pre-insert alignment, and final +0.05 m insertion",
            "success": True,
        }
    finally:
        planner.close()


def _make_standardize_args(args: Args) -> fix3.Args:
    return fix3.Args(
        output_root=args.output_root,
        paths_file=args.paths_file,
        trajectory_name=args.trajectory_name,
        env_id=args.env_id,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        reward_mode=args.reward_mode,
        render_mode=args.render_mode,
        num_demos=args.num_demos,
        max_attempts=args.max_attempts,
        seed=args.seed,
        total_video_frames=args.total_video_frames,
        total_action_steps=args.total_action_steps,
        max_raw_action_steps=args.total_action_steps,
        min_trigger_step=0,
        max_trigger_step=args.total_action_steps,
        strict_axis_cos_min=0.985,
        strict_centerline_yz_max_m=0.01,
        joint_vel_limits=args.joint_vel_limits,
        joint_acc_limits=args.joint_acc_limits,
        val_fraction=args.val_fraction,
        overwrite=args.overwrite,
        constrained_insert_projection=False,
    )


def _copy_raw_trajectory_for_render(
    *,
    raw_group: h5py.Group,
    out_h5: Path,
    sample_id: str,
    record: dict[str, Any],
    args: Args,
) -> dict[str, Any]:
    def pad_first_axis(arr: np.ndarray, target_len: int) -> np.ndarray:
        arr = np.asarray(arr)
        if target_len <= 0 or arr.ndim == 0 or arr.shape[0] == target_len:
            return arr
        if arr.shape[0] > target_len:
            raise ValueError(f"cannot copy {sample_id}: array length {arr.shape[0]} exceeds target {target_len}")
        if arr.shape[0] == 0:
            raise ValueError(f"cannot pad empty array for {sample_id}")
        pad = np.repeat(arr[-1:], target_len - arr.shape[0], axis=0)
        return np.concatenate([arr, pad], axis=0)

    def copy_item(src: h5py.Group | h5py.Dataset, dst_parent: h5py.Group, name: str, rel_path: str) -> None:
        if isinstance(src, h5py.Group):
            child = dst_parent.create_group(name)
            for key, value in src.attrs.items():
                child.attrs[key] = value
            for child_name, child_src in src.items():
                copy_item(child_src, child, child_name, f"{rel_path}/{child_name}" if rel_path else child_name)
            return
        arr = np.asarray(src)
        target_len = 0
        if rel_path == "actions":
            target_len = int(args.total_action_steps)
        elif rel_path == "obs" or rel_path.startswith("env_states/"):
            target_len = int(args.total_video_frames)
        arr = pad_first_axis(arr, target_len)
        dst_parent.create_dataset(name, data=arr, compression="gzip", compression_opts=1)

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    if out_h5.exists():
        out_h5.unlink()
    raw_steps = int(raw_group["actions"].shape[0]) if "actions" in raw_group else int(record.get("raw_action_steps", -1))
    raw_state_count = 0
    if "env_states" in raw_group:
        for section in ("actors", "articulations"):
            section_group = raw_group["env_states"].get(section)
            if section_group:
                for dataset in section_group.values():
                    raw_state_count = int(dataset.shape[0])
                    break
            if raw_state_count:
                break
    summary = {
        **record,
        "sample_id": sample_id,
        "raw_num_source_states": raw_state_count,
        "num_source_states": int(args.total_video_frames),
        "num_video_frames": int(args.total_video_frames),
        "num_action_steps": int(args.total_action_steps),
        "raw_action_steps": raw_steps,
        "official_solver_success": bool(record.get("success", False)),
        "standardization_policy": (
            "raw official fix2 trajectory split for rendering only; env_states "
            "and obs are padded to 301 frames, actions to 300 steps; no strict "
            "v8 centerline gate and no constrained peg projection"
        ),
        "success_at_end": bool(record.get("success", False)),
        "inserted_end": bool(record.get("success", False)),
    }
    with h5py.File(out_h5, "w") as h5:
        group = h5.create_group("traj_0")
        group.attrs["summary_json"] = json.dumps(_jsonable(summary), sort_keys=True)
        group.attrs["source_summary_json"] = json.dumps(_jsonable(summary), sort_keys=True)
        for key, value in raw_group.attrs.items():
            group.attrs[key] = value
        for key in raw_group.keys():
            copy_item(raw_group[key], group, key, key)
    return {
        "sample_id": sample_id,
        "scenario": "none",
        "path": str(out_h5),
        "split": "val" if fix3._stable_val(sample_id, args.val_fraction) else "train",
        "raw_action_steps": raw_steps,
        "raw_num_source_states": raw_state_count,
        "num_source_states": int(args.total_video_frames),
        "num_video_frames": int(args.total_video_frames),
        "num_action_steps": int(args.total_action_steps),
        "official_solver_success": bool(record.get("success", False)),
        "target_motion_norm_m": float(record.get("target_motion_norm_m", 0.0)),
    }


def main() -> None:
    args = tyro.cli(Args)
    if args.total_video_frames != 301 or args.total_action_steps != 300:
        raise ValueError("fix2 smoke must keep the 301-frame / 300-action contract")
    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing non-empty output_root without --overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    raw_dir = output_root / "raw_pd_joint_pos"
    review_dir = output_root / "h5"
    raw_dir.mkdir(parents=True, exist_ok=True)

    render_mode = None if str(args.render_mode).lower() in {"", "none", "null"} else args.render_mode
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        render_mode=render_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
    )
    env = RecordEpisode(
        env,
        output_dir=str(raw_dir),
        trajectory_name=args.trajectory_name,
        save_video=False,
        save_trajectory=True,
        save_on_reset=False,
        record_reward=False,
        source_type="fix2_official_insert_reproduction",
        source_desc="Official ManiSkill PegInsertionSide insertion protocol reproduction; no target motion and no projection.",
        video_fps=30,
    )
    raw_h5_path = Path(env._h5_file.filename)
    accepted: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    try:
        for attempt_i in range(int(args.max_attempts)):
            if len(accepted) >= int(args.num_demos):
                break
            episode_seed = int(args.seed + attempt_i)
            env.reset(seed=episode_seed)
            fix3._replace_record_initial_state(env)
            result = _official_insert_episode(env, args)
            attempt_record = {
                "attempt": attempt_i,
                "episode_seed": episode_seed,
                "accepted": result is not None,
            }
            if result is not None:
                env.flush_trajectory()
                traj_id = f"traj_{env._episode_id}"
                with h5py.File(raw_h5_path, "r") as raw_h5:
                    raw_steps = int(raw_h5[traj_id]["actions"].shape[0])
                if raw_steps > int(args.total_action_steps):
                    result = None
                    attempt_record.update({"accepted": False, "reject_reason": "raw_steps_exceed_300", "raw_steps": raw_steps})
                else:
                    result = {
                        **result,
                        "attempt": attempt_i,
                        "episode_seed": episode_seed,
                        "trajectory": traj_id,
                        "raw_action_steps": raw_steps,
                    }
                    with h5py.File(raw_h5_path, "a") as raw_h5:
                        raw_h5[traj_id].attrs["fix2_official_repro_source_json"] = json.dumps(
                            _jsonable(result),
                            sort_keys=True,
                        )
                    accepted.append(result)
                    attempt_record.update(result)
            else:
                env.flush_trajectory(save=False)
            attempts.append(attempt_record)
            print(
                json.dumps(
                    {
                        "event": "fix2_official_repro_attempt",
                        "attempt": attempt_i,
                        "accepted": len(accepted),
                        "success": result is not None,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        env.close()

    if len(accepted) < int(args.num_demos):
        raise RuntimeError(f"only accepted {len(accepted)} / {args.num_demos} demos")

    records: list[dict[str, Any]] = []
    with h5py.File(raw_h5_path, "r") as raw_h5:
        for idx, record in enumerate(accepted):
            sample_id = f"none_fix2_official_seed{record['episode_seed']:06d}_idx{idx:04d}"
            out_h5 = review_dir / f"{sample_id}.fix2" / f"{sample_id}.h5"
            records.append(
                _copy_raw_trajectory_for_render(
                    raw_group=raw_h5[record["trajectory"]],
                    out_h5=out_h5,
                    sample_id=sample_id,
                    record=record,
                    args=args,
                )
            )

    paths_file = Path(args.paths_file) if args.paths_file else output_root / "fix2_h5_paths.txt"
    paths_file.write_text("\n".join(record["path"] for record in records) + "\n")
    manifest = {
        "args": _jsonable(asdict(args)),
        "boundary": (
            "Fix2/official insertion reproduction smoke only. This verifies the "
            "baseline physical insertion protocol before any fix3 target-motion "
            "enlargement. It is not dynamic fix3 data and not SFT evidence."
        ),
        "raw_h5_path": str(raw_h5_path),
        "raw_json_path": str(raw_h5_path.with_suffix(".json")),
        "output_root": str(output_root),
        "paths_file": str(paths_file),
        "num_records": len(records),
        "accepted": accepted,
        "attempts": attempts,
        "records": records,
    }
    (output_root / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "event": "fix2_official_repro_manifest_written",
                "manifest": str(output_root / "manifest.json"),
                "paths_file": str(paths_file),
                "num_records": len(records),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
