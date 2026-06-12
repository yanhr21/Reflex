#!/usr/bin/env python3
"""Generate successful fix3 final-target expert sources with motion planning.

This script creates the source trajectories that the fix3 dynamic postprocessor
is allowed to consume. It samples new, widened final target-hole poses, solves
PegInsertionSide at those poses with the ManiSkill motion-planning expert, and
records only successful insertions. It does not create dynamic target motion by
itself; that is the next step.
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
from mani_skill.utils.structs import Pose
from mani_skill.utils.wrappers.record import RecordEpisode


@dataclass
class Args:
    output_dir: str
    trajectory_name: str = "fix3_final_pose_expert_pd_joint_pos"
    env_id: str = "PegInsertionSide-v1"
    obs_mode: str = "none"
    control_mode: str = "pd_joint_pos"
    sim_backend: str = "physx_cpu"
    render_backend: str = "none"
    reward_mode: str = "dense"
    render_mode: str = "none"
    num_demos: int = 12
    max_attempts: int = 200
    seed: int = 910000
    final_x_min: float = -0.20
    final_x_max: float = 0.20
    final_y_min: float = 0.08
    final_y_max: float = 0.54
    min_final_shift_m: float = 0.12
    max_final_shift_m: float = 0.36
    joint_vel_limits: float = 0.75
    joint_acc_limits: float = 0.75
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


def _replace_record_initial_state(wrapper: RecordEpisode) -> None:
    """Replace RecordEpisode's reset snapshot after manual target-pose edit."""

    if wrapper._trajectory_buffer is None:
        return
    state = common.to_numpy(common.batch(wrapper.base_env.get_state_dict()))
    obs = common.to_numpy(common.batch(wrapper.base_env.get_obs()))

    def recursive_replace(dst: Any, src: Any) -> None:
        if isinstance(dst, np.ndarray):
            dst[-1] = src[-1]
            return
        for key in dst.keys():
            recursive_replace(dst[key], src[key])

    recursive_replace(wrapper._trajectory_buffer.state, state)
    recursive_replace(wrapper._trajectory_buffer.observation, obs)


def _sample_final_pose(base_env, rng: np.random.Generator, args: Args) -> dict[str, Any]:
    old_p = common.to_numpy(base_env.box.pose.p)[0].astype(np.float32)
    old_q = common.to_numpy(base_env.box.pose.q)[0].astype(np.float32)
    for _ in range(200):
        new_xy = np.asarray(
            [
                rng.uniform(args.final_x_min, args.final_x_max),
                rng.uniform(args.final_y_min, args.final_y_max),
            ],
            dtype=np.float32,
        )
        shift = float(np.linalg.norm(new_xy - old_p[:2]))
        if args.min_final_shift_m <= shift <= args.max_final_shift_m:
            break
    else:
        raise RuntimeError("failed to sample a final target pose within shift bounds")

    new_p = old_p.copy()
    new_p[:2] = new_xy
    p_t = torch.as_tensor(new_p, device=base_env.device, dtype=base_env.box.pose.p.dtype).view(1, 3)
    q_t = torch.as_tensor(old_q, device=base_env.device, dtype=base_env.box.pose.q.dtype).view(1, 4)
    base_env.box.set_pose(Pose.create_from_pq(p_t, q_t))
    return {
        "old_box_xyz": old_p.tolist(),
        "new_box_xyz": new_p.tolist(),
        "box_shift_xy_m": shift,
        "box_quat": old_q.tolist(),
    }


def _solve_current_final_pose(env: RecordEpisode, args: Args) -> Any:
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
    finger_length = 0.025
    try:
        obb = get_actor_obb(base_env.peg)
        approaching = np.array([0, 0, -1])
        target_closing = base_env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        peg_init_pose = base_env.peg.pose

        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=finger_length,
        )
        grasp_pose = base_env.agent.build_grasp_pose(
            approaching,
            grasp_info["closing"],
            grasp_info["center"],
        )
        offset = sapien.Pose([-max(0.05, base_env.peg_half_sizes[0, 0].item() / 2 + 0.01), 0, 0])
        grasp_pose = grasp_pose * offset

        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        res = planner.move_to_pose_with_screw(reach_pose)
        if res == -1:
            return -1
        res = planner.move_to_pose_with_screw(grasp_pose)
        if res == -1:
            return -1
        planner.close_gripper()

        insert_pose = base_env.goal_pose * peg_init_pose.inv() * grasp_pose
        insert_offset = sapien.Pose([-0.01 - base_env.peg_half_sizes[0, 0].item(), 0, 0])
        pre_insert_pose = insert_pose * insert_offset
        res = planner.move_to_pose_with_screw(pre_insert_pose)
        if res == -1:
            return -1
        for _ in range(3):
            delta_pose = base_env.goal_pose * insert_offset * base_env.peg.pose.inv()
            pre_insert_pose = delta_pose * pre_insert_pose
            res = planner.move_to_pose_with_screw(pre_insert_pose)
            if res == -1:
                return -1
        return planner.move_to_pose_with_screw(insert_pose * sapien.Pose([0.05, 0, 0]))
    finally:
        planner.close()


def _last_success(res: Any) -> bool:
    if res == -1 or not res:
        return False
    try:
        return bool(res[-1]["success"].item())
    except Exception:  # noqa: BLE001
        return False


def _set_group_attr(h5_path: Path, traj_id: str, key: str, value: Any) -> None:
    with h5py.File(h5_path, "a") as h5:
        h5[traj_id].attrs[key] = json.dumps(_jsonable(value), sort_keys=True)


def main() -> None:
    args = tyro.cli(Args)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing non-empty output_dir without --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

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
        output_dir=str(output_dir),
        trajectory_name=args.trajectory_name,
        save_video=False,
        save_trajectory=True,
        save_on_reset=False,
        record_reward=False,
        source_type="fix3_final_pose_motionplanning",
        source_desc="Motion-planning expert solved at widened/new final target poses for fix3.",
        video_fps=30,
    )
    h5_path = Path(env._h5_file.filename)
    rng = np.random.default_rng(args.seed)
    accepted: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []

    try:
        for attempt_i in range(int(args.max_attempts)):
            if len(accepted) >= int(args.num_demos):
                break
            episode_seed = int(args.seed + attempt_i)
            env.reset(seed=episode_seed)
            pose_record = _sample_final_pose(env.unwrapped, rng, args)
            _replace_record_initial_state(env)
            res = _solve_current_final_pose(env, args)
            success = _last_success(res)
            record = {
                "attempt": attempt_i,
                "episode_seed": episode_seed,
                "success": bool(success),
                **pose_record,
            }
            attempts.append(record)
            if success:
                env.flush_trajectory()
                traj_id = f"traj_{env._episode_id}"
                record["trajectory"] = traj_id
                accepted.append(record)
                _set_group_attr(h5_path, traj_id, "fix3_final_pose_source_json", record)
            else:
                env.flush_trajectory(save=False)
            print(
                json.dumps(
                    {
                        "event": "fix3_final_pose_attempt",
                        "attempt": attempt_i,
                        "accepted": len(accepted),
                        "success": bool(success),
                        "box_shift_xy_m": pose_record["box_shift_xy_m"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        env.close()

    if len(accepted) < int(args.num_demos):
        raise RuntimeError(f"only accepted {len(accepted)} / {args.num_demos} demos")

    manifest = {
        "args": _jsonable(asdict(args)),
        "boundary": (
            "Successful final-target source data for fix3. Dynamic target motion "
            "has not been overlaid yet. This is not SFT or controller evidence."
        ),
        "h5_path": str(h5_path),
        "json_path": str(h5_path.with_suffix(".json")),
        "num_accepted": len(accepted),
        "num_attempts": len(attempts),
        "accepted": accepted,
        "attempts": attempts,
    }
    manifest_path = output_dir / "fix3_final_pose_source_manifest.json"
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "fix3_final_pose_manifest_written", "manifest": str(manifest_path), "h5_path": str(h5_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
