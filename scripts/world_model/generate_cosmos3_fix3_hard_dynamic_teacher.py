#!/usr/bin/env python3
"""Generate hard fix3 supplement demos with a scripted/oracle teacher.

This is the 2026-06-12 supplement path for the rows still missing after the
retained v7 DP-success rows. It does not discard existing accepted H5 files.

The key semantic difference from the v7 DP-success generator is that positive
rows are not defined by the frozen/static DP already solving the dynamic task.
The teacher deliberately constructs physical successful trajectories: grasp and
pre-align with a motion planner, inject a hard late target/peg disturbance, and
then replan/recover to insert in the changed world. The generated rows still
keep the 301-frame / 300-action contract and remain blocked on render/user
approval before SFT.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import sapien
import torch
import tyro
from transforms3d.quaternions import qinverse, qmult, quat2axangle

import mani_skill.envs  # noqa: F401
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.utils import common
from mani_skill.utils.structs import Pose
from mani_skill.utils.wrappers.record import RecordEpisode


LAYOUT = {
    "qpos": slice(0, 9),
    "qvel": slice(9, 18),
    "tcp_pose": slice(18, 25),
    "peg_pose": slice(25, 32),
    "peg_half_size": slice(32, 35),
    "hole_pose": slice(35, 42),
    "hole_radius": slice(42, 43),
}

SCENARIOS = (
    "hole_late_move_stop",
    "hole_late_constant",
    "hole_late_reverse",
    "hole_late_sine",
    "hole_late_continuous_insert",
    "hole_late_fast_shift",
    "none",
    "peg_drop",
    "peg_disturb",
)

HOLE_MOTION_SCENARIOS = {
    "hole_late_move_stop",
    "hole_late_constant",
    "hole_late_reverse",
    "hole_late_sine",
    "hole_late_continuous_insert",
    "hole_late_fast_shift",
}
PEG_PERTURB_SCENARIOS = {"peg_drop", "peg_disturb"}


@dataclass
class Args:
    output_root: str = (
        "experiments/world_model_task_rebinding/cosmos3/"
        "fix3_hard_dynamic_teacher_supplement_smoke_20260612"
    )
    paths_file: str = ""
    trajectory_name: str = "fix3_hard_dynamic_teacher_raw_pd_joint_pos"
    env_id: str = "PegInsertionSide-v1"
    obs_mode: str = "state"
    control_mode: str = "pd_joint_pos"
    sim_backend: str = "physx_cpu"
    render_backend: str = "none"
    reward_mode: str = "dense"
    render_mode: str = "none"
    num_demos: int = 9
    max_attempts: int = 300
    seed: int = 30000000
    total_video_frames: int = 301
    total_action_steps: int = 300
    motion_min_m: float = 0.22
    motion_max_m: float = 0.30
    min_abs_y_motion_m: float = 0.06
    final_x_min: float = -0.075
    final_x_max: float = 0.075
    final_y_min: float = 0.15
    final_y_max: float = 0.45
    move_steps_min: int = 24
    move_steps_max: int = 44
    continuous_move_steps_min: int = 48
    continuous_move_steps_max: int = 76
    max_raw_action_steps: int = 300
    min_trigger_step: int = 45
    max_trigger_step: int = 230
    preinsert_yz_max_m: float = 0.01
    initial_preinsert_line_yz_max_m: float = 0.0
    final_insert_offset_m: float = 0.05
    insert_step_m: float = 0.02
    initial_preinsert_retreat_extra_m: float = 0.0
    strict_insert_x_min: float = -0.015
    strict_insert_x_max: float = 0.055
    strict_yz_radius_fraction: float = 1.0
    strict_axis_cos_min: float = 0.995
    strict_centerline_yz_max_m: float = 0.004
    strict_clearance_slack_m: float = 0.0
    anti_self_insert_final_yz_min_m: float = 0.055
    anti_self_insert_radius_multiplier: float = 2.4
    anti_self_insert_exempt_scenarios: str = ""
    wall_x_margin_m: float = 0.002
    wall_outer_margin_m: float = 0.002
    peg_line_samples: int = 9
    preinsert_refine_steps: int = 5
    final_insert_refine_steps: int = 4
    final_insert_settle_steps: int = 0
    static_insert_controller: str = "refined_goal_pose"
    static_tcp_servo_step_m: float = 0.005
    static_tcp_servo_max_steps: int = 80
    static_tcp_servo_feedback_gain: float = 1.0
    static_tcp_servo_max_yz_correction_m: float = 0.006
    static_tcp_servo_wall_yz_abort_m: float = 0.012
    static_tcp_servo_close_steps: int = 2
    static_tcp_servo_final_settle_passes: int = 4
    constrained_insert_projection: bool = False
    max_constrained_insert_raw_line_yz_m: float = 0.025
    trigger_peg_lift_min_m: float = 0.012
    trigger_finger_width_max: float = 0.026
    trigger_peg_tcp_dist_max: float = 0.120
    peg_disturb_delta_xyz: tuple[float, float, float] = (0.0, -0.055, 0.025)
    peg_drop_delta_y_m: float = -0.055
    post_motion_release_regrasp_scenarios: str = ""
    post_motion_release_regrasp_delta_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    post_motion_release_regrasp_hold_steps: int = 4
    source_kind: str = "hard_dynamic_teacher"
    scenario_sequence: str = ""
    scenario_quotas: str = ""
    scenario_seed_bases: str = ""
    accepted_index_offset: int = 0
    save_reject_log_limit: int = 2000
    reject_log_every: int = 50
    joint_vel_limits: float = 0.90
    joint_acc_limits: float = 0.90
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


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q = q / norm
    if q[0] < 0:
        q = -q
    return q


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quat(q)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (_quat_to_matrix(q) @ np.asarray(v, dtype=np.float64)).astype(np.float32)


def _inv_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (_quat_to_matrix(q).T @ np.asarray(v, dtype=np.float64)).astype(np.float32)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _scenario_subset(text: str) -> set[str]:
    if not text.strip():
        return set()
    out = {item.strip() for item in text.split(",") if item.strip()}
    unknown = sorted(out - set(SCENARIOS))
    if unknown:
        raise ValueError(f"unknown scenario subset entries: {unknown}")
    return out


def _compact_axis_angle(q: np.ndarray) -> np.ndarray:
    axis, theta = quat2axangle(_normalize_quat(q))
    if theta > np.pi:
        theta -= 2.0 * np.pi
    return np.asarray(axis, dtype=np.float32) * np.float32(theta)


def _replace_record_initial_state(wrapper: RecordEpisode) -> None:
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


def _to_np_pose(pose) -> tuple[np.ndarray, np.ndarray]:
    p = common.to_numpy(pose.p)[0].astype(np.float32)
    q = common.to_numpy(pose.q)[0].astype(np.float32)
    return p, q


def _set_box_pose(base_env, p: np.ndarray, q: np.ndarray) -> None:
    p_t = torch.as_tensor(p, device=base_env.device, dtype=base_env.box.pose.p.dtype).view(1, 3)
    q_t = torch.as_tensor(q, device=base_env.device, dtype=base_env.box.pose.q.dtype).view(1, 4)
    base_env.box.set_pose(Pose.create_from_pq(p_t, q_t))


def _set_peg_pose(base_env, p: np.ndarray, q: np.ndarray) -> None:
    p_t = torch.as_tensor(p, device=base_env.device, dtype=base_env.peg.pose.p.dtype).view(1, 3)
    q_t = torch.as_tensor(q, device=base_env.device, dtype=base_env.peg.pose.q.dtype).view(1, 4)
    base_env.peg.set_pose(Pose.create_from_pq(p_t, q_t))


def _to_np_actor_pose(actor) -> tuple[np.ndarray, np.ndarray]:
    p = common.to_numpy(actor.pose.p)[0].astype(np.float32)
    q = common.to_numpy(actor.pose.q)[0].astype(np.float32)
    return p, q


def _live_peg_head_at_hole(base_env) -> np.ndarray:
    info = base_env.evaluate()
    return common.to_numpy(info["peg_head_pos_at_hole"])[0].astype(np.float32)


def _live_success(base_env) -> bool:
    info = base_env.evaluate()
    return bool(common.to_numpy(info["success"])[0])


def _live_radius(base_env) -> float:
    return float(common.to_numpy(base_env.box_hole_radii)[0])


def _strict_live_inserted(base_env, args: Args) -> tuple[bool, np.ndarray, float, dict[str, Any]]:
    head = _live_peg_head_at_hole(base_env)
    radius = _live_radius(base_env)
    line_info = _live_peg_line_gate_info(base_env, args)
    ok = (
        bool(_live_success(base_env))
        and bool(line_info["centerline_yz_max_m"] <= line_info["centerline_yz_limit_m"])
        and bool(line_info["peg_axis_cos_hole_x"] >= float(args.strict_axis_cos_min))
        and not bool(line_info["wall_collision_risk"])
    )
    return ok, head, radius, line_info


def _goal_rel_xyz(base_env, pose) -> np.ndarray:
    return common.to_numpy((base_env.goal_pose.inv() * pose).p)[0].astype(np.float32)


def _hole_rel_xyz(base_env, pose) -> np.ndarray:
    return common.to_numpy((base_env.box_hole_pose.inv() * pose).p)[0].astype(np.float32)


def _peg_axis_at_hole(base_env, args: Args) -> tuple[dict[str, Any], float]:
    half_length = float(common.to_numpy(base_env.peg_half_sizes)[0, 0])
    head = _hole_rel_xyz(base_env, base_env.peg_head_pose)
    center = _hole_rel_xyz(base_env, base_env.peg.pose)
    tail = _hole_rel_xyz(base_env, base_env.peg.pose * sapien.Pose([-half_length, 0, 0]))
    axis = head - tail
    axis_norm = max(float(np.linalg.norm(axis)), 1e-6)
    axis_unit = axis / axis_norm
    radius = _live_radius(base_env)
    yz_limit = float(radius * args.strict_yz_radius_fraction)
    head_yz = float(np.linalg.norm(head[1:]))
    center_yz = float(np.linalg.norm(center[1:]))
    tail_yz = float(np.linalg.norm(tail[1:]))
    info = {
        "peg_head_hole_xyz": head.astype(float).tolist(),
        "peg_center_hole_xyz": center.astype(float).tolist(),
        "peg_tail_hole_xyz": tail.astype(float).tolist(),
        "peg_axis_unit_hole": axis_unit.astype(float).tolist(),
        "peg_axis_cos_hole_x": float(axis_unit[0]),
        "peg_head_hole_yz_m": head_yz,
        "peg_center_hole_yz_m": center_yz,
        "peg_tail_hole_yz_m": tail_yz,
        "hole_radius": radius,
        "axis_yz_limit_m": yz_limit,
        "strict_axis_cos_min": float(args.strict_axis_cos_min),
    }
    return info, yz_limit


def _live_peg_line_points_at_hole(base_env, num_points: int) -> np.ndarray:
    half_length = float(common.to_numpy(base_env.peg_half_sizes)[0, 0])
    xs = np.linspace(-half_length, half_length, max(3, int(num_points)), dtype=np.float32)
    points = []
    for x in xs:
        points.append(_hole_rel_xyz(base_env, base_env.peg.pose * sapien.Pose([float(x), 0, 0])))
    return np.stack(points, axis=0).astype(np.float32)


def _centerline_yz_limit(peg_radius: float, hole_radius: float, args: Args) -> float:
    clearance = max(float(hole_radius) - float(peg_radius), 0.0)
    return float(min(float(args.strict_centerline_yz_max_m), clearance + float(args.strict_clearance_slack_m)))


def _peg_line_gate_info(points_hole: np.ndarray, peg_half_length: float, peg_radius: float, hole_radius: float, args: Args) -> dict[str, Any]:
    points = np.asarray(points_hole, dtype=np.float32)
    yz_norm = np.linalg.norm(points[:, 1:], axis=1)
    axis = points[-1] - points[0]
    axis_cos = float(axis[0] / max(float(np.linalg.norm(axis)), 1e-6))
    centerline_limit = _centerline_yz_limit(float(peg_radius), float(hole_radius), args)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    depth = float(peg_half_length)
    within_x = np.abs(x) <= depth + float(args.wall_x_margin_m)
    within_outer_yz = (
        (np.abs(y) <= depth + float(peg_radius) + float(args.wall_outer_margin_m))
        & (np.abs(z) <= depth + float(peg_radius) + float(args.wall_outer_margin_m))
    )
    outside_hole_channel = (np.abs(y) > centerline_limit) | (np.abs(z) > centerline_limit)
    wall_collision = bool(np.any(within_x & within_outer_yz & outside_hole_channel))
    return {
        "peg_line_points_hole": points.astype(float).tolist(),
        "centerline_yz_max_m": float(yz_norm.max()) if yz_norm.size else 0.0,
        "centerline_yz_limit_m": float(centerline_limit),
        "peg_axis_cos_hole_x": axis_cos,
        "wall_collision_risk": wall_collision,
        "peg_radius_m": float(peg_radius),
        "hole_radius_m": float(hole_radius),
        "block_depth_half_m": float(depth),
    }


def _live_peg_line_gate_info(base_env, args: Args) -> dict[str, Any]:
    points = _live_peg_line_points_at_hole(base_env, int(args.peg_line_samples))
    peg_half_length = float(common.to_numpy(base_env.peg_half_sizes)[0, 0])
    peg_radius = float(common.to_numpy(base_env.peg_half_sizes)[0, 1])
    hole_radius = _live_radius(base_env)
    return _peg_line_gate_info(points, peg_half_length, peg_radius, hole_radius, args)


def _line_center_x_yz(line_info: dict[str, Any]) -> tuple[float, np.ndarray]:
    points = np.asarray(line_info["peg_line_points_hole"], dtype=np.float32)
    center = points.mean(axis=0)
    return float(center[0]), center[1:3].astype(np.float32)


def _live_wall_collision_risk(base_env, args: Args) -> tuple[bool, dict[str, Any]]:
    info = _live_peg_line_gate_info(base_env, args)
    return bool(info["wall_collision_risk"]), info


def _preinsert_ready(
    base_env,
    args: Args,
    *,
    line_yz_max_override_m: float | None = None,
) -> tuple[bool, dict[str, Any]]:
    peg_head_goal = _goal_rel_xyz(base_env, base_env.peg_head_pose)
    peg_goal = _goal_rel_xyz(base_env, base_env.peg.pose)
    head_yz = float(np.linalg.norm(peg_head_goal[1:]))
    peg_yz = float(np.linalg.norm(peg_goal[1:]))
    line_info = _live_peg_line_gate_info(base_env, args)
    line_yz_limit_m = (
        float(line_yz_max_override_m)
        if line_yz_max_override_m is not None and float(line_yz_max_override_m) > 0.0
        else float(line_info["centerline_yz_limit_m"])
    )
    ok = (
        head_yz <= float(args.preinsert_yz_max_m)
        and peg_yz <= float(args.preinsert_yz_max_m)
        and float(line_info["centerline_yz_max_m"]) <= line_yz_limit_m
        and not bool(line_info["wall_collision_risk"])
        and line_info["peg_axis_cos_hole_x"] >= float(args.strict_axis_cos_min)
    )
    axis_info, _ = _peg_axis_at_hole(base_env, args)
    return ok, {
        "peg_head_goal_xyz": peg_head_goal.astype(float).tolist(),
        "peg_goal_xyz": peg_goal.astype(float).tolist(),
        "peg_head_goal_yz_m": head_yz,
        "peg_goal_yz_m": peg_yz,
        "preinsert_yz_max_m": float(args.preinsert_yz_max_m),
        "preinsert_line_yz_limit_m": float(line_yz_limit_m),
        **line_info,
        **axis_info,
    }


def _move_to_refined_preinsert(
    *,
    base_env,
    planner: PandaArmMotionPlanningSolver,
    insert_pose: sapien.Pose,
    insert_offset: sapien.Pose,
    stage_offset_m: float | None = None,
    refine_steps: int = 3,
) -> sapien.Pose | None:
    pre_insert_pose = insert_pose * insert_offset
    if planner.move_to_pose_with_screw(pre_insert_pose) == -1:
        return None
    for _ in range(max(1, int(refine_steps))):
        delta_pose = base_env.goal_pose * insert_offset * base_env.peg.pose.inv()
        pre_insert_pose = delta_pose * pre_insert_pose
        if planner.move_to_pose_with_screw(pre_insert_pose) == -1:
            return None
    if stage_offset_m is not None:
        pre_insert_pose = insert_pose * sapien.Pose([float(stage_offset_m), 0, 0])
        if planner.move_to_pose_with_screw(pre_insert_pose) == -1:
            return None
    return pre_insert_pose


def _incremental_insert(
    *,
    base_env,
    planner: PandaArmMotionPlanningSolver,
    insert_pose: sapien.Pose,
    insert_offset_m: float,
    final_offset_m: float,
    step_m: float,
) -> bool:
    total = float(final_offset_m) - float(insert_offset_m)
    if total <= 0:
        return False
    steps = max(1, int(np.ceil(total / max(float(step_m), 1e-4))))
    for i in range(steps):
        alpha = float(i + 1) / float(steps)
        x = float(insert_offset_m) + alpha * total
        if planner.move_to_pose_with_screw(insert_pose * sapien.Pose([x, 0, 0])) == -1:
            return False
    return True


def _move_to_refined_final_insert(
    *,
    base_env,
    planner: PandaArmMotionPlanningSolver,
    insert_pose: sapien.Pose,
    insert_offset_m: float,
    final_offset_m: float,
    step_m: float,
    refine_steps: int,
    args: Args,
) -> tuple[bool, dict[str, Any]]:
    total = float(final_offset_m) - float(insert_offset_m)
    if total <= 0:
        return False, {
            "reason": "non_positive_insert_distance",
            "insert_offset_m": float(insert_offset_m),
            "final_offset_m": float(final_offset_m),
        }

    num_steps = max(1, int(np.ceil(total / max(float(step_m), 1e-4))))
    num_refines = max(1, int(refine_steps))
    sweep_yz_max = 0.0
    sweep_limit_min = float("inf")
    raw_wall_collision_risk_any = False
    final_line_info: dict[str, Any] | None = None

    for step_idx in range(num_steps):
        alpha = float(step_idx + 1) / float(num_steps)
        target_x = float(insert_offset_m) + alpha * total
        nominal_tcp_pose = insert_pose * sapien.Pose([target_x, 0, 0])
        desired_peg_pose = base_env.goal_pose * sapien.Pose([target_x, 0, 0])
        target_tcp_pose = desired_peg_pose * base_env.peg.pose.inv() * nominal_tcp_pose
        step_line_info: dict[str, Any] | None = None

        for refine_idx in range(num_refines):
            try:
                move_result = planner.move_to_pose_with_screw(
                    target_tcp_pose,
                    refine_steps=max(0, int(args.final_insert_settle_steps)),
                )
            except RuntimeError as exc:
                return False, {
                    "reason": "planner_final_insert_exception",
                    "error": str(exc),
                    "insert_step_idx": int(step_idx),
                    "insert_num_steps": int(num_steps),
                    "refine_idx": int(refine_idx),
                    "target_insert_x_m": float(target_x),
                }
            if move_result == -1:
                return False, {
                    "reason": "planner_final_insert_refine_failed",
                    "insert_step_idx": int(step_idx),
                    "insert_num_steps": int(num_steps),
                    "refine_idx": int(refine_idx),
                    "target_insert_x_m": float(target_x),
                }

            line_info = _live_peg_line_gate_info(base_env, args)
            final_line_info = line_info
            step_line_info = line_info
            sweep_yz_max = max(sweep_yz_max, float(line_info["centerline_yz_max_m"]))
            sweep_limit_min = min(sweep_limit_min, float(line_info["centerline_yz_limit_m"]))
            if bool(line_info["wall_collision_risk"]):
                raw_wall_collision_risk_any = True

            delta_pose = desired_peg_pose * base_env.peg.pose.inv()
            target_tcp_pose = delta_pose * target_tcp_pose
        if step_line_info is not None and bool(step_line_info["wall_collision_risk"]):
            if (
                not bool(args.constrained_insert_projection)
                or float(step_line_info["centerline_yz_max_m"]) > float(args.max_constrained_insert_raw_line_yz_m)
            ):
                return False, {
                    "reason": "final_insert_wall_collision_risk",
                    "insert_step_idx": int(step_idx),
                    "insert_num_steps": int(num_steps),
                    "refine_idx": int(num_refines - 1),
                    "target_insert_x_m": float(target_x),
                    **step_line_info,
                }

    if final_line_info is None:
        final_line_info = _live_peg_line_gate_info(base_env, args)
    return True, {
        "insert_num_steps": int(num_steps),
        "insert_refine_steps": int(num_refines),
        "insert_settle_steps": int(max(0, int(args.final_insert_settle_steps))),
        "sweep_centerline_yz_max_m": float(sweep_yz_max),
        "sweep_centerline_yz_limit_min_m": float(sweep_limit_min),
        "raw_wall_collision_risk_any": bool(raw_wall_collision_risk_any),
        "final_line_gate": final_line_info,
    }


def _move_to_static_tcp_servo_final_insert(
    *,
    base_env,
    planner: PandaArmMotionPlanningSolver,
    env: RecordEpisode,
    episode_start_step: int,
    args: Args,
) -> tuple[bool, dict[str, Any]]:
    """Advance the live TCP by small hole-frame deltas for static teacher data.

    This is a candidate-teacher controller, not a method policy. It changes the
    physical insertion construction after retry1 showed that the absolute
    refined-goal insertion path can hit centerline and wall-risk gates even
    when the simulator briefly reports live success.
    """

    start_line = _live_peg_line_gate_info(base_env, args)
    start_x, _ = _line_center_x_yz(start_line)
    final_x = float(args.final_insert_offset_m)
    max_raw_steps = int(args.max_raw_action_steps)
    goal_mat = base_env.goal_pose.to_transformation_matrix()[0, :3, :3].cpu().numpy().astype(np.float32)
    goal_x = goal_mat[:, 0]
    goal_y = goal_mat[:, 1]
    goal_z = goal_mat[:, 2]

    def raw_steps_now() -> int:
        return int(env._elapsed_record_steps) - int(episode_start_step)

    def budget_failure(reason: str, step_idx: int | None, records: list[dict[str, Any]]) -> dict[str, Any] | None:
        raw_steps = raw_steps_now()
        if raw_steps <= max_raw_steps:
            return None
        return {
            "controller": "static_tcp_incremental_servo",
            "reason": reason,
            "raw_steps": int(raw_steps),
            "max_raw_action_steps": int(max_raw_steps),
            "insert_step_idx": None if step_idx is None else int(step_idx),
            "records": records,
        }

    if final_x <= start_x:
        strict_ok, head, radius, final_line = _strict_live_inserted(base_env, args)
        return strict_ok, {
            "controller": "static_tcp_incremental_servo",
            "reason": "already_at_or_past_final_x" if not strict_ok else "already_strict_inserted",
            "start_center_x_m": float(start_x),
            "final_insert_offset_m": float(final_x),
            "live_success": bool(_live_success(base_env)),
            "peg_head_at_hole": head.astype(float).tolist(),
            "hole_radius": float(radius),
            "start_line_gate": start_line,
            "final_line_gate": final_line,
        }

    step_m = max(float(args.static_tcp_servo_step_m), 1e-4)
    natural_steps = int(np.ceil((final_x - start_x) / step_m))
    num_steps = max(1, min(int(args.static_tcp_servo_max_steps), natural_steps))
    records: list[dict[str, Any]] = []

    def run_incremental_pose(step_idx: int, phase: str) -> tuple[bool, dict[str, Any] | None]:
        line_before = _live_peg_line_gate_info(base_env, args)
        line_x, line_center_yz = _line_center_x_yz(line_before)
        remaining_x = max(0.0, final_x - line_x)
        dx = min(step_m, remaining_x)
        yz_correction = -float(args.static_tcp_servo_feedback_gain) * line_center_yz
        yz_norm = float(np.linalg.norm(yz_correction))
        max_corr = max(float(args.static_tcp_servo_max_yz_correction_m), 1e-6)
        if yz_norm > max_corr:
            yz_correction = yz_correction * (max_corr / yz_norm)

        if int(args.static_tcp_servo_close_steps) > 0:
            planner.close_gripper(t=int(args.static_tcp_servo_close_steps))
            budget = budget_failure("static_tcp_servo_raw_steps_exceed_limit_after_close", step_idx, records)
            if budget is not None:
                return False, budget

        tcp_p, tcp_q = _to_np_pose(base_env.agent.tcp.pose)
        correction_world = (
            float(dx) * goal_x
            + float(yz_correction[0]) * goal_y
            + float(yz_correction[1]) * goal_z
        )
        target_tcp_pose = sapien.Pose(tcp_p + correction_world.astype(np.float32), tcp_q)
        try:
            result = planner.move_to_pose_with_screw(
                target_tcp_pose,
                refine_steps=max(0, int(args.final_insert_settle_steps)),
            )
        except RuntimeError as exc:
            return False, {
                "controller": "static_tcp_incremental_servo",
                "reason": "static_tcp_servo_planner_exception",
                "error": str(exc),
                "insert_step_idx": int(step_idx),
                "records": records,
            }

        budget = budget_failure("static_tcp_servo_raw_steps_exceed_limit_after_move", step_idx, records)
        if budget is not None:
            return False, budget
        if int(args.static_tcp_servo_close_steps) > 0:
            planner.close_gripper(t=int(args.static_tcp_servo_close_steps))
            budget = budget_failure("static_tcp_servo_raw_steps_exceed_limit_after_post_move_close", step_idx, records)
            if budget is not None:
                return False, budget

        line_after = _live_peg_line_gate_info(base_env, args)
        head_after = _live_peg_head_at_hole(base_env)
        record = {
            "phase": str(phase),
            "insert_step_idx": int(step_idx),
            "insert_num_steps": int(num_steps),
            "line_center_x_before_m": float(line_x),
            "remaining_insert_x_m": float(remaining_x),
            "dx_m": float(dx),
            "yz_correction_m": yz_correction.astype(float).tolist(),
            "raw_steps_after_step": int(raw_steps_now()),
            "planner_result": _jsonable(result),
            "peg_head_at_hole": head_after.astype(float).tolist(),
            "line_gate_before": line_before,
            "line_gate_after": line_after,
        }
        records.append(record)
        if result == -1:
            return False, {
                "controller": "static_tcp_incremental_servo",
                "reason": "static_tcp_servo_planner_step_failed",
                "insert_step_idx": int(step_idx),
                "records": records,
            }
        if bool(line_after["wall_collision_risk"]) and (
            float(line_after["centerline_yz_max_m"]) > float(args.static_tcp_servo_wall_yz_abort_m)
        ):
            return False, {
                "controller": "static_tcp_incremental_servo",
                "reason": "static_tcp_servo_wall_collision_risk",
                "insert_step_idx": int(step_idx),
                "records": records,
            }
        return True, None

    for step_idx in range(num_steps):
        line_now = _live_peg_line_gate_info(base_env, args)
        line_x, _ = _line_center_x_yz(line_now)
        if line_x >= final_x:
            break
        ok, failure = run_incremental_pose(step_idx, "advance")
        if not ok:
            return False, failure or {
                "controller": "static_tcp_incremental_servo",
                "reason": "static_tcp_servo_unknown_advance_failure",
                "insert_step_idx": int(step_idx),
                "records": records,
            }

    for settle_idx in range(max(0, int(args.static_tcp_servo_final_settle_passes))):
        ok, failure = run_incremental_pose(num_steps + settle_idx, "final_yz_settle")
        if not ok:
            return False, failure or {
                "controller": "static_tcp_incremental_servo",
                "reason": "static_tcp_servo_unknown_final_settle_failure",
                "insert_step_idx": int(num_steps + settle_idx),
                "records": records,
            }

    strict_ok, head, radius, final_line = _strict_live_inserted(base_env, args)
    if not strict_ok:
        return False, {
            "controller": "static_tcp_incremental_servo",
            "reason": "static_tcp_servo_strict_insert_failed",
            "peg_head_at_hole": head.astype(float).tolist(),
            "hole_radius": float(radius),
            "live_success": bool(_live_success(base_env)),
            "raw_steps_after_insert": int(raw_steps_now()),
            "max_raw_action_steps": int(max_raw_steps),
            "records": records,
            "final_line_gate": final_line,
        }
    return True, {
        "controller": "static_tcp_incremental_servo",
        "start_center_x_m": float(start_x),
        "final_insert_offset_m": float(final_x),
        "static_tcp_servo_step_m": float(args.static_tcp_servo_step_m),
        "static_tcp_servo_max_steps": int(args.static_tcp_servo_max_steps),
        "static_tcp_servo_close_steps": int(args.static_tcp_servo_close_steps),
        "static_tcp_servo_feedback_gain": float(args.static_tcp_servo_feedback_gain),
        "static_tcp_servo_max_yz_correction_m": float(args.static_tcp_servo_max_yz_correction_m),
        "static_tcp_servo_wall_yz_abort_m": float(args.static_tcp_servo_wall_yz_abort_m),
        "static_tcp_servo_final_settle_passes": int(args.static_tcp_servo_final_settle_passes),
        "raw_steps_after_insert": int(raw_steps_now()),
        "max_raw_action_steps": int(max_raw_steps),
        "records": records,
        "strict_peg_head_at_hole": head.astype(float).tolist(),
        "strict_final_line_gate": final_line,
        "hole_radius": float(radius),
    }


def _reject(reject_reason: str, **details: Any) -> None:
    details = dict(details)
    if "reason" in details:
        details["inner_reason"] = details.pop("reason")
    print(
        json.dumps(
            _jsonable({"event": "fix3_late_trigger_reject", "reason": reject_reason, **details}),
            sort_keys=True,
        ),
        flush=True,
    )
    return None


def _sample_final_pose(
    *,
    base_env,
    rng: np.random.Generator,
    initial_p: np.ndarray,
    initial_q: np.ndarray,
    peg_head_xy: np.ndarray,
    scenario: str,
    args: Args,
) -> tuple[np.ndarray, np.ndarray, float]:
    del peg_head_xy
    for _ in range(5000):
        final_xy = np.asarray(
            [
                rng.uniform(args.final_x_min, args.final_x_max),
                rng.uniform(args.final_y_min, args.final_y_max),
            ],
            dtype=np.float32,
        )
        delta = final_xy - initial_p[:2].astype(np.float32)
        mag = float(np.linalg.norm(delta))
        if mag < float(args.motion_min_m) or mag > float(args.motion_max_m):
            continue
        if abs(float(delta[1])) < float(args.min_abs_y_motion_m):
            continue
        if not (args.final_x_min <= final_xy[0] <= args.final_x_max):
            continue
        if not (args.final_y_min <= final_xy[1] <= args.final_y_max):
            continue
        final_p = initial_p.copy()
        final_p[:2] = final_xy
        return final_p.astype(np.float32), initial_q.astype(np.float32), mag
    raise RuntimeError("failed to sample a large late target final pose inside domain")


def _target_motion_path(
    *,
    start_p: np.ndarray,
    final_p: np.ndarray,
    scenario: str,
    move_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if move_steps <= 0:
        return final_p.reshape(1, 3).astype(np.float32)
    s = np.linspace(0.0, 1.0, move_steps, dtype=np.float32)
    if scenario in {"hole_late_move_stop", "hole_late_fast_shift"}:
        prog = _smoothstep(s)
    elif scenario == "hole_late_constant":
        prog = s
    elif scenario == "hole_late_reverse":
        prog = np.where(s < 0.55, 1.22 * _smoothstep(s / 0.55), 1.22 - 0.22 * _smoothstep((s - 0.55) / 0.45))
    elif scenario == "hole_late_sine":
        prog = _smoothstep(s)
    elif scenario == "hole_late_continuous_insert":
        prog = s
    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    delta = (final_p - start_p).astype(np.float32)
    path = start_p[None, :] + prog[:, None] * delta[None, :]
    if scenario == "hole_late_sine":
        norm = max(float(np.linalg.norm(delta[:2])), 1e-6)
        orth = np.asarray([-delta[1] / norm, delta[0] / norm, 0.0], dtype=np.float32)
        amp = float(rng.uniform(0.015, 0.035))
        path += (amp * np.sin(np.pi * s) * (1.0 - s))[:, None] * orth[None, :]
    path[-1] = final_p
    return path.astype(np.float32)


def _move_target_fast(
    *,
    env: RecordEpisode,
    episode_start_step: int,
    final_p: np.ndarray,
    final_q: np.ndarray,
    scenario: str,
    rng: np.random.Generator,
    args: Args,
) -> tuple[int, int, np.ndarray, dict[str, Any]]:
    base_env = env.unwrapped
    start_p, _ = _to_np_pose(base_env.box.pose)
    if scenario == "hole_late_continuous_insert":
        move_steps = int(rng.integers(args.continuous_move_steps_min, args.continuous_move_steps_max + 1))
    elif scenario == "hole_late_fast_shift":
        move_steps = int(rng.integers(max(6, args.move_steps_min - 4), max(8, args.move_steps_min + 4)))
    else:
        move_steps = int(rng.integers(args.move_steps_min, args.move_steps_max + 1))
    path = _target_motion_path(start_p=start_p, final_p=final_p, scenario=scenario, move_steps=move_steps, rng=rng)
    trigger_step = int(env._elapsed_record_steps) - int(episode_start_step)
    qpos = common.to_numpy(base_env.agent.robot.get_qpos())[0, :7]
    action = np.hstack([qpos, -1.0]).astype(np.float32)
    sweep_yz_max = 0.0
    sweep_limit_min = float("inf")
    for p in path:
        _set_box_pose(base_env, p, final_q)
        env.step(action)
        wall_risk, line_info = _live_wall_collision_risk(base_env, args)
        sweep_yz_max = max(sweep_yz_max, float(line_info["centerline_yz_max_m"]))
        sweep_limit_min = min(sweep_limit_min, float(line_info["centerline_yz_limit_m"]))
        if wall_risk:
            raise RuntimeError(
                json.dumps(
                    _jsonable(
                        {
                            "reason": "target_motion_swept_wall_collision_risk",
                            "trigger_step": trigger_step,
                            "move_steps": move_steps,
                            **line_info,
                        }
                    ),
                    sort_keys=True,
                )
            )
    return trigger_step, move_steps, path, {
        "sweep_centerline_yz_max_m": float(sweep_yz_max),
        "sweep_centerline_yz_limit_min_m": float(sweep_limit_min),
    }


def _make_planner(env: RecordEpisode, args: Args) -> PandaArmMotionPlanningSolver:
    base_env = env.unwrapped
    return PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=base_env.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=float(args.joint_vel_limits),
        joint_acc_limits=float(args.joint_acc_limits),
    )


def _build_current_grasp_pose(base_env) -> tuple[sapien.Pose, sapien.Pose]:
    obb = get_actor_obb(base_env.peg)
    approaching = np.array([0, 0, -1])
    target_closing = base_env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
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
    return grasp_pose, base_env.peg.pose


def _grasp_current_peg(base_env, planner: PandaArmMotionPlanningSolver) -> tuple[sapien.Pose, sapien.Pose] | None:
    grasp_pose, peg_pose_at_grasp = _build_current_grasp_pose(base_env)
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    if planner.move_to_pose_with_screw(reach_pose) == -1:
        return None
    if planner.move_to_pose_with_screw(grasp_pose) == -1:
        return None
    planner.close_gripper()
    return grasp_pose, peg_pose_at_grasp


def _insert_current_goal(
    *,
    base_env,
    planner: PandaArmMotionPlanningSolver,
    env: RecordEpisode,
    episode_start_step: int,
    grasp_pose: sapien.Pose,
    peg_pose_at_grasp: sapien.Pose,
    args: Args,
) -> tuple[bool, dict[str, Any]]:
    insert_pose = base_env.goal_pose * peg_pose_at_grasp.inv() * grasp_pose
    insert_offset_m = float(-0.01 - base_env.peg_half_sizes[0, 0].item())
    insert_offset = sapien.Pose([insert_offset_m, 0, 0])
    pre_insert_pose = _move_to_refined_preinsert(
        base_env=base_env,
        planner=planner,
        insert_pose=insert_pose,
        insert_offset=insert_offset,
        refine_steps=int(args.preinsert_refine_steps),
    )
    if pre_insert_pose is None:
        return False, {"reason": "planner_preinsert_failed", "insert_offset_m": insert_offset_m}
    preinsert_ok, preinsert_info = _preinsert_ready(base_env, args)
    if not preinsert_ok:
        return False, {"reason": "preinsert_gate_failed", **preinsert_info, "insert_offset_m": insert_offset_m}
    if str(args.static_insert_controller) == "refined_goal_pose":
        insert_ok, insert_sweep_info = _move_to_refined_final_insert(
            base_env=base_env,
            planner=planner,
            insert_pose=insert_pose,
            insert_offset_m=insert_offset_m,
            final_offset_m=float(args.final_insert_offset_m),
            step_m=float(args.insert_step_m),
            refine_steps=int(args.final_insert_refine_steps),
            args=args,
        )
    elif str(args.static_insert_controller) == "tcp_incremental_servo":
        insert_ok, insert_sweep_info = _move_to_static_tcp_servo_final_insert(
            base_env=base_env,
            planner=planner,
            env=env,
            episode_start_step=episode_start_step,
            args=args,
        )
    else:
        raise ValueError(f"unknown static_insert_controller: {args.static_insert_controller!r}")
    if not insert_ok:
        return False, {"reason": "planner_final_insert_failed", **insert_sweep_info, "insert_offset_m": insert_offset_m}
    strict_ok, head, radius, final_line_info = _strict_live_inserted(base_env, args)
    if not strict_ok:
        return False, {
            "reason": "strict_insert_failed",
            "peg_head_at_hole": head,
            "hole_radius": radius,
            "live_success": _live_success(base_env),
            **final_line_info,
            "insert_offset_m": insert_offset_m,
        }
    return True, {
        "insert_offset_m": insert_offset_m,
        "final_insert_offset_m": float(args.final_insert_offset_m),
        "insert_step_m": float(args.insert_step_m),
        "preinsert": preinsert_info,
        "final_insert_sweep": insert_sweep_info,
        "strict_peg_head_at_hole": head.astype(float).tolist(),
        "strict_final_line_gate": final_line_info,
        "hole_radius": float(radius),
    }


def _hold_action(base_env, gripper: float) -> np.ndarray:
    qpos = common.to_numpy(base_env.agent.robot.get_qpos())[0, :7]
    return np.hstack([qpos, float(gripper)]).astype(np.float32)


def _run_static_teacher_episode(env: RecordEpisode, rng: np.random.Generator, args: Args) -> dict[str, Any] | None:
    del rng
    base_env = env.unwrapped
    planner = _make_planner(env, args)
    try:
        episode_start_step = int(env._elapsed_record_steps)
        initial_box_p, initial_box_q = _to_np_pose(base_env.box.pose)
        grasp_result = _grasp_current_peg(base_env, planner)
        if grasp_result is None:
            return _reject("planner_static_grasp_failed", scenario="none")
        grasp_pose, peg_pose_at_grasp = grasp_result
        insert_start = int(env._elapsed_record_steps) - int(episode_start_step)
        insert_ok, insert_info = _insert_current_goal(
            base_env=base_env,
            planner=planner,
            env=env,
            episode_start_step=episode_start_step,
            grasp_pose=grasp_pose,
            peg_pose_at_grasp=peg_pose_at_grasp,
            args=args,
        )
        if not insert_ok:
            return _reject("planner_static_insert_failed", scenario="none", **insert_info)
        final_box_p, final_box_q = _to_np_pose(base_env.box.pose)
        raw_steps = int(env._elapsed_record_steps) - int(episode_start_step)
        if raw_steps > int(args.max_raw_action_steps):
            return _reject("raw_steps_exceed_limit", scenario="none", raw_steps=raw_steps)
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
            "final_insert_start_step": int(insert_start),
            "source_kind": str(args.source_kind),
            "teacher_phases": [
                "motion_planner_grasp",
                f"motion_planner_static_insert_{args.static_insert_controller}",
            ],
            "baseline_dp_expected": "static control sample; not a hard dynamic supplement row",
            "success": True,
            **insert_info,
        }
    finally:
        planner.close()


def _run_peg_recovery_teacher_episode(
    env: RecordEpisode,
    scenario: str,
    rng: np.random.Generator,
    args: Args,
) -> dict[str, Any] | None:
    del rng
    base_env = env.unwrapped
    planner = _make_planner(env, args)
    try:
        episode_start_step = int(env._elapsed_record_steps)
        initial_box_p, initial_box_q = _to_np_pose(base_env.box.pose)
        initial_peg_p, initial_peg_q = _to_np_actor_pose(base_env.peg)
        grasp_result = _grasp_current_peg(base_env, planner)
        if grasp_result is None:
            return _reject("planner_peg_initial_grasp_failed", scenario=scenario)
        grasp_pose, peg_pose_at_grasp = grasp_result
        preinsert_pose = _move_to_refined_preinsert(
            base_env=base_env,
            planner=planner,
            insert_pose=base_env.goal_pose * peg_pose_at_grasp.inv() * grasp_pose,
            insert_offset=sapien.Pose([float(-0.01 - base_env.peg_half_sizes[0, 0].item()), 0, 0]),
            refine_steps=int(args.preinsert_refine_steps),
        )
        if preinsert_pose is None:
            return _reject("planner_peg_initial_preinsert_failed", scenario=scenario)
        preinsert_ok, preinsert_info = _preinsert_ready(base_env, args)
        if not preinsert_ok:
            return _reject("peg_initial_preinsert_gate_failed", scenario=scenario, **preinsert_info)

        trigger_step = int(env._elapsed_record_steps) - int(episode_start_step)
        peg_before_p, peg_before_q = _to_np_actor_pose(base_env.peg)
        planner.open_gripper(t=4)
        if scenario == "peg_drop":
            new_peg_p = peg_before_p.copy()
            new_peg_p[1] += float(args.peg_drop_delta_y_m)
            new_peg_p[2] = float(initial_peg_p[2])
        elif scenario == "peg_disturb":
            new_peg_p = peg_before_p + np.asarray(args.peg_disturb_delta_xyz, dtype=np.float32)
        else:
            raise ValueError(f"unexpected peg recovery scenario {scenario!r}")
        _set_peg_pose(base_env, new_peg_p, peg_before_q)
        env.step(_hold_action(base_env, PandaArmMotionPlanningSolver.OPEN))
        peg_delta = (new_peg_p - peg_before_p).astype(np.float32)

        grasp_result = _grasp_current_peg(base_env, planner)
        if grasp_result is None:
            return _reject(
                "planner_peg_regrasp_failed",
                scenario=scenario,
                trigger_step=trigger_step,
                peg_delta=peg_delta,
            )
        grasp_pose, peg_pose_at_grasp = grasp_result
        final_insert_start_step = int(env._elapsed_record_steps) - int(episode_start_step)
        insert_ok, insert_info = _insert_current_goal(
            base_env=base_env,
            planner=planner,
            env=env,
            episode_start_step=episode_start_step,
            grasp_pose=grasp_pose,
            peg_pose_at_grasp=peg_pose_at_grasp,
            args=args,
        )
        if not insert_ok:
            return _reject(
                "planner_peg_recovery_insert_failed",
                scenario=scenario,
                trigger_step=trigger_step,
                peg_delta=peg_delta,
                **insert_info,
            )
        final_box_p, final_box_q = _to_np_pose(base_env.box.pose)
        raw_steps = int(env._elapsed_record_steps) - int(episode_start_step)
        if raw_steps > int(args.max_raw_action_steps):
            return _reject("raw_steps_exceed_limit", scenario=scenario, raw_steps=raw_steps)
        return {
            "scenario": scenario,
            "trigger_step": int(trigger_step),
            "move_steps": 1,
            "raw_action_steps": raw_steps,
            "initial_box_xyz": initial_box_p.astype(float).tolist(),
            "final_box_xyz": final_box_p.astype(float).tolist(),
            "initial_box_quat": initial_box_q.astype(float).tolist(),
            "final_box_quat": final_box_q.astype(float).tolist(),
            "target_motion_norm_m": float(np.linalg.norm(final_box_p[:3] - initial_box_p[:3])),
            "motion_path_xyz": [initial_box_p.astype(float).tolist(), final_box_p.astype(float).tolist()],
            "peg_delta_applied_step": int(trigger_step),
            "peg_delta_applied_xyz": peg_delta.astype(float).tolist(),
            "final_insert_start_step": int(final_insert_start_step),
            "source_kind": str(args.source_kind),
            "teacher_phases": [
                "motion_planner_grasp",
                "late_peg_disturbance_or_drop",
                "motion_planner_regrasp",
                "motion_planner_insert",
            ],
            "baseline_dp_expected": "hard peg recovery supplement; original DP must be measured separately on matching perturbations",
            "success": True,
            **insert_info,
        }
    finally:
        planner.close()


def _run_late_trigger_episode(env: RecordEpisode, scenario: str, rng: np.random.Generator, args: Args) -> dict[str, Any] | None:
    base_env = env.unwrapped
    planner = _make_planner(env, args)
    try:
        episode_start_step = int(env._elapsed_record_steps)
        initial_box_p, initial_box_q = _to_np_pose(base_env.box.pose)
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
        offset = sapien.Pose([-max(0.05, base_env.peg_half_sizes[0, 0].item() / 2 + 0.01), 0, 0])
        grasp_pose = grasp_pose * offset

        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        if planner.move_to_pose_with_screw(reach_pose) == -1:
            return _reject("planner_reach_grasp_failed", scenario=scenario)
        if planner.move_to_pose_with_screw(grasp_pose) == -1:
            return _reject("planner_grasp_pose_failed", scenario=scenario)
        planner.close_gripper()

        initial_insert_pose = base_env.goal_pose * peg_init_pose.inv() * grasp_pose
        insert_offset_m = float(-0.01 - base_env.peg_half_sizes[0, 0].item())
        insert_offset = sapien.Pose([insert_offset_m, 0, 0])
        initial_preinsert_stage_offset_m = None
        if float(args.initial_preinsert_retreat_extra_m) > 0.0:
            initial_preinsert_stage_offset_m = insert_offset_m - float(args.initial_preinsert_retreat_extra_m)
        initial_pre_insert_pose = _move_to_refined_preinsert(
            base_env=base_env,
            planner=planner,
            insert_pose=initial_insert_pose,
            insert_offset=insert_offset,
            stage_offset_m=initial_preinsert_stage_offset_m,
            refine_steps=int(args.preinsert_refine_steps),
        )
        if initial_pre_insert_pose is None:
            return _reject("planner_initial_preinsert_failed", scenario=scenario)
        initial_preinsert_line_yz_max_m = (
            float(args.initial_preinsert_line_yz_max_m)
            if float(args.initial_preinsert_line_yz_max_m) > 0.0
            else None
        )
        initial_preinsert_ok, initial_preinsert_info = _preinsert_ready(
            base_env,
            args,
            line_yz_max_override_m=initial_preinsert_line_yz_max_m,
        )
        if not initial_preinsert_ok:
            return _reject(
                "initial_preinsert_gate_failed",
                scenario=scenario,
                **initial_preinsert_info,
            )

        trigger_step = int(env._elapsed_record_steps) - int(episode_start_step)
        if trigger_step < args.min_trigger_step or trigger_step > args.max_trigger_step:
            return _reject(
                "trigger_step_out_of_range",
                scenario=scenario,
                trigger_step=trigger_step,
                absolute_step=int(env._elapsed_record_steps),
                episode_start_step=episode_start_step,
            )
        peg_head_world = common.to_numpy(base_env.peg_head_pose.p)[0].astype(np.float32)
        try:
            final_box_p, final_box_q, motion_norm = _sample_final_pose(
                base_env=base_env,
                rng=rng,
                initial_p=initial_box_p,
                initial_q=initial_box_q,
                peg_head_xy=peg_head_world[:2],
                scenario=scenario,
                args=args,
            )
        except RuntimeError as exc:
            return _reject(
                "sample_final_pose_failed",
                scenario=scenario,
                error=str(exc),
                initial_box_xyz=initial_box_p,
                peg_head_xy=peg_head_world[:2],
            )
        try:
            trigger_step, move_steps, motion_path, sweep_info = _move_target_fast(
                env=env,
                episode_start_step=episode_start_step,
                final_p=final_box_p,
                final_q=final_box_q,
                scenario=scenario,
                rng=rng,
                args=args,
            )
        except RuntimeError as exc:
            try:
                details = json.loads(str(exc))
            except json.JSONDecodeError:
                details = {"error": str(exc)}
            details.pop("reason", None)
            return _reject(
                "target_motion_swept_wall_collision_risk",
                scenario=scenario,
                **details,
            )
        anti_line_info = _live_peg_line_gate_info(base_env, args)
        anti_threshold = max(
            float(args.anti_self_insert_final_yz_min_m),
            float(args.anti_self_insert_radius_multiplier) * float(anti_line_info["hole_radius_m"]),
        )
        anti_exempt = scenario in _scenario_subset(str(args.anti_self_insert_exempt_scenarios))
        if (not anti_exempt) and float(anti_line_info["centerline_yz_max_m"]) < anti_threshold:
            return _reject(
                "target_self_insert_after_motion_gate_failed",
                scenario=scenario,
                trigger_step=trigger_step,
                move_steps=move_steps,
                target_motion_norm_m=float(motion_norm),
                anti_self_insert_yz_at_motion_end=float(anti_line_info["centerline_yz_max_m"]),
                anti_self_insert_yz_threshold_m=float(anti_threshold),
                **anti_line_info,
            )

        post_motion_regrasp_done = False
        post_motion_regrasp_delta = np.asarray(args.post_motion_release_regrasp_delta_xyz, dtype=np.float32)
        peg_pose_for_final_insert = peg_init_pose
        if scenario in _scenario_subset(str(args.post_motion_release_regrasp_scenarios)):
            peg_before_release_p, peg_before_release_q = _to_np_actor_pose(base_env.peg)
            planner.open_gripper(t=4)
            if float(np.linalg.norm(post_motion_regrasp_delta)) > 0.0:
                _set_peg_pose(base_env, peg_before_release_p + post_motion_regrasp_delta, peg_before_release_q)
            for _ in range(max(1, int(args.post_motion_release_regrasp_hold_steps))):
                env.step(_hold_action(base_env, PandaArmMotionPlanningSolver.OPEN))
            regrasp_result = _grasp_current_peg(base_env, planner)
            if regrasp_result is None:
                return _reject(
                    "planner_post_motion_regrasp_failed",
                    scenario=scenario,
                    trigger_step=trigger_step,
                    move_steps=move_steps,
                    post_motion_regrasp_delta=post_motion_regrasp_delta,
                    post_motion_regrasp_hold_steps=int(args.post_motion_release_regrasp_hold_steps),
                )
            grasp_pose, peg_pose_for_final_insert = regrasp_result
            post_motion_regrasp_done = True

        final_insert_pose = base_env.goal_pose * peg_pose_for_final_insert.inv() * grasp_pose
        final_pre_insert_pose = _move_to_refined_preinsert(
            base_env=base_env,
            planner=planner,
            insert_pose=final_insert_pose,
            insert_offset=insert_offset,
            refine_steps=int(args.preinsert_refine_steps),
        )
        if final_pre_insert_pose is None:
            return _reject(
                "planner_final_preinsert_failed",
                scenario=scenario,
                trigger_step=trigger_step,
                move_steps=move_steps,
            )
        final_preinsert_ok, final_preinsert_info = _preinsert_ready(base_env, args)
        if not final_preinsert_ok:
            return _reject(
                "final_preinsert_gate_failed",
                scenario=scenario,
                trigger_step=trigger_step,
                move_steps=move_steps,
                **final_preinsert_info,
            )
        final_insert_start_step = int(env._elapsed_record_steps) - int(episode_start_step)
        insert_ok, insert_sweep_info = _move_to_refined_final_insert(
            base_env=base_env,
            planner=planner,
            insert_pose=final_insert_pose,
            insert_offset_m=insert_offset_m,
            final_offset_m=float(args.final_insert_offset_m),
            step_m=float(args.insert_step_m),
            refine_steps=int(args.final_insert_refine_steps),
            args=args,
        )
        if not insert_ok:
            insert_reject_info = dict(insert_sweep_info)
            insert_reject_info.pop("reason", None)
            return _reject(
                "planner_final_insert_failed",
                scenario=scenario,
                trigger_step=trigger_step,
                move_steps=move_steps,
                **insert_reject_info,
            )
        strict_ok, head, radius, final_line_info = _strict_live_inserted(base_env, args)
        constrained_projection_required = bool(
            args.constrained_insert_projection
            and (not strict_ok or bool(insert_sweep_info.get("raw_wall_collision_risk_any", False)))
        )
        if constrained_projection_required and not bool(_live_success(base_env)):
            return _reject(
                "constrained_insert_projection_raw_final_not_official_success",
                scenario=scenario,
                trigger_step=trigger_step,
                move_steps=move_steps,
                peg_head_at_hole=head,
                hole_radius=radius,
                live_success=_live_success(base_env),
                **final_line_info,
            )
        if constrained_projection_required and (
            float(final_line_info["centerline_yz_max_m"]) > float(args.max_constrained_insert_raw_line_yz_m)
        ):
            return _reject(
                "constrained_insert_projection_raw_final_too_far",
                scenario=scenario,
                trigger_step=trigger_step,
                move_steps=move_steps,
                peg_head_at_hole=head,
                hole_radius=radius,
                live_success=_live_success(base_env),
                **final_line_info,
            )
        if (not strict_ok) and not constrained_projection_required:
            return _reject(
                "strict_insert_failed",
                scenario=scenario,
                trigger_step=trigger_step,
                move_steps=move_steps,
                peg_head_at_hole=head,
                hole_radius=radius,
                live_success=_live_success(base_env),
                **final_line_info,
            )
        raw_steps = int(env._elapsed_record_steps) - int(episode_start_step)
        if raw_steps > int(args.max_raw_action_steps):
            return _reject(
                "raw_steps_exceed_limit",
                scenario=scenario,
                trigger_step=trigger_step,
                move_steps=move_steps,
                raw_steps=raw_steps,
                max_raw_action_steps=args.max_raw_action_steps,
            )
        return {
            "scenario": scenario,
            "trigger_step": int(trigger_step),
            "move_steps": int(move_steps),
            "raw_action_steps": raw_steps,
            "initial_box_xyz": initial_box_p.astype(float).tolist(),
            "final_box_xyz": final_box_p.astype(float).tolist(),
            "target_motion_norm_m": float(motion_norm),
            "motion_path_xyz": motion_path.astype(float).tolist(),
            "initial_preinsert": initial_preinsert_info,
            "final_preinsert": final_preinsert_info,
            "target_motion_sweep": sweep_info,
            "final_insert_sweep": insert_sweep_info,
            "final_insert_start_step": int(final_insert_start_step),
            "constrained_insert_projection_required": bool(constrained_projection_required),
            "constrained_insert_projection_max_raw_line_yz_m": float(args.max_constrained_insert_raw_line_yz_m),
            "insert_offset_m": float(insert_offset_m),
            "initial_preinsert_stage_offset_m": (
                float(initial_preinsert_stage_offset_m)
                if initial_preinsert_stage_offset_m is not None
                else None
            ),
            "initial_preinsert_retreat_extra_m": float(args.initial_preinsert_retreat_extra_m),
            "initial_preinsert_line_yz_max_m": float(args.initial_preinsert_line_yz_max_m),
            "final_insert_offset_m": float(args.final_insert_offset_m),
            "insert_step_m": float(args.insert_step_m),
            "preinsert_refine_steps": int(args.preinsert_refine_steps),
            "final_insert_refine_steps": int(args.final_insert_refine_steps),
            "final_insert_settle_steps": int(args.final_insert_settle_steps),
            "post_motion_release_regrasp_done": bool(post_motion_regrasp_done),
            "post_motion_release_regrasp_delta_xyz": post_motion_regrasp_delta.astype(float).tolist(),
            "post_motion_release_regrasp_hold_steps": int(args.post_motion_release_regrasp_hold_steps),
            "anti_self_insert_exempt": bool(anti_exempt),
            "anti_self_insert_exempt_scenarios": str(args.anti_self_insert_exempt_scenarios),
            "strict_peg_head_at_hole": head.astype(float).tolist(),
            "strict_final_line_gate": final_line_info,
            "hole_radius": float(radius),
            "source_kind": str(args.source_kind),
            "teacher_phases": [
                "motion_planner_grasp",
                "motion_planner_preinsert_at_initial_hole",
                "late_target_motion",
                *(
                    ["post_motion_release_regrasp"]
                    if post_motion_regrasp_done
                    else []
                ),
                "motion_planner_replan_to_moved_hole",
                "motion_planner_insert",
            ],
            "baseline_dp_expected": (
                "hard moving-hole supplement; original/static DP must be measured "
                "separately on the matching target-motion family"
            ),
            "success": True,
        }
    finally:
        planner.close()


def _pad_first_axis(arr: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape[0] > target_len:
        raise ValueError(f"cannot pad array with len {arr.shape[0]} to shorter target {target_len}")
    if arr.shape[0] == target_len:
        return arr
    pad = np.repeat(arr[-1:], target_len - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def _write_dataset(group: h5py.Group, name: str, value: np.ndarray) -> None:
    group.create_dataset(name, data=np.asarray(value), compression="gzip", compression_opts=1)


def _write_dict_group(group: h5py.Group, data: dict[str, Any]) -> None:
    for key, value in data.items():
        if isinstance(value, dict):
            child = group.create_group(str(key))
            _write_dict_group(child, value)
        else:
            _write_dataset(group, str(key), np.asarray(value))


def _apply_constrained_insert_projection(
    *,
    obs: np.ndarray,
    env_states: dict[str, Any],
    record: dict[str, Any],
    args: Args,
) -> dict[str, Any]:
    if not bool(record.get("constrained_insert_projection_required", False)):
        return {"applied": False}
    if not bool(args.constrained_insert_projection):
        raise RuntimeError("record requires constrained insert projection, but args disabled it")
    if os.environ.get("ALLOW_REJECTED_CONSTRAINED_INSERT_DIAGNOSTIC") != "true":
        raise RuntimeError(
            "constrained_insert_projection was rejected as non-physical diagnostic output; "
            "set ALLOW_REJECTED_CONSTRAINED_INSERT_DIAGNOSTIC=true only for archived diagnostics"
        )

    total_frames = int(obs.shape[0])
    start = int(np.clip(int(record.get("final_insert_start_step", total_frames - 1)), 0, total_frames - 1))
    denom = max(1, total_frames - 1 - start)
    insert_offset_m = float(record["insert_offset_m"])
    final_offset_m = float(record["final_insert_offset_m"])
    peg_actor = env_states.get("actors", {}).get("peg")

    for t in range(start, total_frames):
        alpha = float(t - start) / float(denom)
        alpha = float(_smoothstep(np.asarray([alpha], dtype=np.float32))[0])
        offset_m = insert_offset_m + alpha * (final_offset_m - insert_offset_m)
        half = float(obs[t, LAYOUT["peg_half_size"]][0])
        hole_pose = obs[t, LAYOUT["hole_pose"]].astype(np.float32)
        hole_p = hole_pose[:3]
        hole_q = _normalize_quat(hole_pose[3:7])
        peg_center_hole = np.asarray([-half + offset_m, 0.0, 0.0], dtype=np.float32)
        peg_p = hole_p + _rotate(hole_q, peg_center_hole)
        peg_q = hole_q.astype(np.float32)
        obs[t, LAYOUT["peg_pose"]] = np.concatenate([peg_p.astype(np.float32), peg_q], axis=0)
        if isinstance(peg_actor, np.ndarray) and peg_actor.shape[0] > t and peg_actor.shape[1] >= 7:
            peg_actor[t, :3] = peg_p.astype(np.float32)
            peg_actor[t, 3:7] = peg_q
            if peg_actor.shape[1] > 7:
                peg_actor[t, 7:] = 0.0

    return {
        "applied": True,
        "start_step": int(start),
        "end_step": int(total_frames - 1),
        "insert_offset_m": float(insert_offset_m),
        "final_insert_offset_m": float(final_offset_m),
        "policy": "rejected diagnostic only: project peg actor/slot onto the final hole axis during the constrained insert segment",
    }


def _stack_obs(obs: np.ndarray) -> np.ndarray:
    prev = np.concatenate([obs[:1], obs[:-1]], axis=0)
    return np.stack([prev, obs], axis=1).astype(np.float32)


def _peg_head_at_hole(peg_pose: np.ndarray, hole_pose: np.ndarray, peg_half_length: np.ndarray) -> np.ndarray:
    out = np.zeros((peg_pose.shape[0], 3), dtype=np.float32)
    for i in range(peg_pose.shape[0]):
        peg_q = _normalize_quat(peg_pose[i, 3:7])
        hole_q = _normalize_quat(hole_pose[i, 3:7])
        head_world = peg_pose[i, :3] + _rotate(peg_q, np.asarray([peg_half_length[i], 0.0, 0.0], dtype=np.float32))
        out[i] = _inv_rotate(hole_q, head_world - hole_pose[i, :3])
    return out


def _peg_axis_at_hole_np(
    peg_pose: np.ndarray,
    hole_pose: np.ndarray,
    peg_half_length: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(peg_pose.shape[0])
    head = np.zeros((n, 3), dtype=np.float32)
    center = np.zeros((n, 3), dtype=np.float32)
    tail = np.zeros((n, 3), dtype=np.float32)
    axis_cos = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        peg_q = _normalize_quat(peg_pose[i, 3:7])
        hole_q = _normalize_quat(hole_pose[i, 3:7])
        half = float(peg_half_length[i])
        head_world = peg_pose[i, :3] + _rotate(peg_q, np.asarray([half, 0.0, 0.0], dtype=np.float32))
        center_world = peg_pose[i, :3]
        tail_world = peg_pose[i, :3] + _rotate(peg_q, np.asarray([-half, 0.0, 0.0], dtype=np.float32))
        head[i] = _inv_rotate(hole_q, head_world - hole_pose[i, :3])
        center[i] = _inv_rotate(hole_q, center_world - hole_pose[i, :3])
        tail[i] = _inv_rotate(hole_q, tail_world - hole_pose[i, :3])
        axis = head[i] - tail[i]
        axis_cos[i] = float(axis[0] / max(float(np.linalg.norm(axis)), 1e-6))
    return head, center, tail, axis_cos


def _peg_line_samples_at_hole_np(
    peg_pose: np.ndarray,
    hole_pose: np.ndarray,
    peg_half_length: np.ndarray,
    num_points: int,
) -> np.ndarray:
    n = int(peg_pose.shape[0])
    samples = max(3, int(num_points))
    out = np.zeros((n, samples, 3), dtype=np.float32)
    for i in range(n):
        peg_q = _normalize_quat(peg_pose[i, 3:7])
        hole_q = _normalize_quat(hole_pose[i, 3:7])
        half = float(peg_half_length[i])
        for j, x in enumerate(np.linspace(-half, half, samples, dtype=np.float32)):
            world = peg_pose[i, :3] + _rotate(peg_q, np.asarray([x, 0.0, 0.0], dtype=np.float32))
            out[i, j] = _inv_rotate(hole_q, world - hole_pose[i, :3])
    return out


def _peg_line_gate_info_np(
    points_hole: np.ndarray,
    peg_half_length: np.ndarray,
    peg_radius: np.ndarray,
    hole_radius: np.ndarray,
    args: Args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_hole, dtype=np.float32)
    yz_norm = np.linalg.norm(points[:, :, 1:], axis=2)
    centerline_yz_max = yz_norm.max(axis=1)
    axis = points[:, -1] - points[:, 0]
    axis_cos = axis[:, 0] / np.maximum(np.linalg.norm(axis, axis=1), 1e-6)
    clearance = np.maximum(np.asarray(hole_radius, dtype=np.float32) - np.asarray(peg_radius, dtype=np.float32), 0.0)
    centerline_limit = np.minimum(
        float(args.strict_centerline_yz_max_m),
        clearance + float(args.strict_clearance_slack_m),
    ).astype(np.float32)
    depth = np.asarray(peg_half_length, dtype=np.float32)[:, None]
    peg_r = np.asarray(peg_radius, dtype=np.float32)[:, None]
    limit = centerline_limit[:, None]
    x = points[:, :, 0]
    y = points[:, :, 1]
    z = points[:, :, 2]
    within_x = np.abs(x) <= depth + float(args.wall_x_margin_m)
    within_outer_yz = (
        (np.abs(y) <= depth + peg_r + float(args.wall_outer_margin_m))
        & (np.abs(z) <= depth + peg_r + float(args.wall_outer_margin_m))
    )
    outside_hole_channel = (np.abs(y) > limit) | (np.abs(z) > limit)
    wall_risk = np.any(within_x & within_outer_yz & outside_hole_channel, axis=1)
    return centerline_yz_max.astype(np.float32), centerline_limit.astype(np.float32), axis_cos.astype(np.float32), wall_risk.astype(bool)


def _strict_inserted_from_line(
    head_at_hole: np.ndarray,
    line_yz_max: np.ndarray,
    line_yz_limit: np.ndarray,
    axis_cos: np.ndarray,
    wall_risk: np.ndarray,
    args: Args,
) -> np.ndarray:
    return (
        (head_at_hole[:, 0] >= float(args.strict_insert_x_min))
        & (head_at_hole[:, 0] <= float(args.strict_insert_x_max))
        & (line_yz_max <= line_yz_limit)
        & (axis_cos >= float(args.strict_axis_cos_min))
        & (~wall_risk)
    ).astype(bool)


def _strict_inserted_from_head(head_at_hole: np.ndarray, radius: np.ndarray, args: Args) -> np.ndarray:
    radius = np.asarray(radius, dtype=np.float32).reshape(-1)
    yz = radius * float(args.strict_yz_radius_fraction)
    return (
        (head_at_hole[:, 0] >= float(args.strict_insert_x_min))
        & (head_at_hole[:, 0] <= float(args.strict_insert_x_max))
        & (np.abs(head_at_hole[:, 1]) <= yz)
        & (np.abs(head_at_hole[:, 2]) <= yz)
    ).astype(bool)


def _grasped_from_obs(obs: np.ndarray) -> np.ndarray:
    fingers = obs[:, 7:9].mean(axis=1)
    peg_tcp = np.linalg.norm(obs[:, LAYOUT["peg_pose"]][:, :3] - obs[:, LAYOUT["tcp_pose"]][:, :3], axis=1)
    return ((fingers < 0.026) & (peg_tcp < 0.16)).astype(bool)


def _robust_held_from_obs(obs: np.ndarray, initial_peg_z: float, args: Args) -> np.ndarray:
    fingers = obs[:, 7:9].mean(axis=1)
    peg_xyz = obs[:, LAYOUT["peg_pose"]][:, :3]
    tcp_xyz = obs[:, LAYOUT["tcp_pose"]][:, :3]
    peg_tcp = np.linalg.norm(peg_xyz - tcp_xyz, axis=1)
    lifted = peg_xyz[:, 2] >= float(initial_peg_z + args.trigger_peg_lift_min_m)
    return (
        (fingers <= float(args.trigger_finger_width_max))
        & (peg_tcp <= float(args.trigger_peg_tcp_dist_max))
        & lifted
    ).astype(bool)


def _ee_delta_action_labels(obs: np.ndarray, raw_actions: np.ndarray, raw_steps: int, total_steps: int) -> np.ndarray:
    out = np.zeros((total_steps, 7), dtype=np.float32)
    tcp = obs[:, LAYOUT["tcp_pose"]].astype(np.float32)
    last_gripper = -1.0
    if raw_actions.size and raw_actions.shape[1] > 0:
        last_gripper = float(np.clip(raw_actions[min(raw_steps - 1, raw_actions.shape[0] - 1), -1], -1.0, 1.0))
    for i in range(total_steps):
        gripper = last_gripper
        if i < raw_steps and i < raw_actions.shape[0]:
            gripper = float(np.clip(raw_actions[i, -1], -1.0, 1.0))
        if i + 1 < obs.shape[0] and i < raw_steps:
            delta_pos = np.clip((tcp[i + 1, :3] - tcp[i, :3]) / 0.1, -1.0, 1.0)
            q_delta = qmult(_normalize_quat(tcp[i + 1, 3:7]), qinverse(_normalize_quat(tcp[i, 3:7])))
            delta_rot = np.clip(_compact_axis_angle(q_delta) / 0.1, -1.0, 1.0)
            out[i, :3] = delta_pos
            out[i, 3:6] = delta_rot
        out[i, 6] = gripper
    return out


def _stable_val(sample_id: str, val_fraction: float) -> bool:
    if val_fraction <= 0:
        return False
    if val_fraction >= 1:
        return True
    import hashlib

    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 10000) < int(round(val_fraction * 10000))


def _parse_sequence(text: str) -> list[str]:
    if not text.strip():
        return list(SCENARIOS)
    out = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(out) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"unknown scenarios in scenario_sequence: {unknown}")
    return out


def _parse_int_map(text: str, *, allowed: set[str], flag_name: str) -> dict[str, int]:
    out: dict[str, int] = {}
    if not text.strip():
        return out
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"{flag_name} entry must be SCENARIO=INT, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in allowed:
            raise ValueError(f"unknown scenario in {flag_name}: {key!r}")
        out[key] = int(value)
    return out


def _standardize_one(
    *,
    raw_group: h5py.Group,
    out_h5: Path,
    sample_id: str,
    record: dict[str, Any],
    args: Args,
) -> dict[str, Any]:
    raw_obs = np.asarray(raw_group["obs"], dtype=np.float32)
    raw_actions = np.asarray(raw_group["actions"], dtype=np.float32)
    raw_steps = int(raw_actions.shape[0])
    if raw_steps > int(args.total_action_steps):
        raise RuntimeError(f"{sample_id} raw steps {raw_steps} exceed {args.total_action_steps}")
    obs = _pad_first_axis(raw_obs, int(args.total_video_frames)).astype(np.float32)
    action_labels = _ee_delta_action_labels(obs, raw_actions, raw_steps, int(args.total_action_steps))

    env_states: dict[str, Any] = {"actors": {}, "articulations": {}}
    for section in ("actors", "articulations"):
        for name, dataset in raw_group["env_states"][section].items():
            arr = np.asarray(dataset, dtype=np.float32)
            env_states[section][name] = _pad_first_axis(arr, int(args.total_video_frames)).astype(np.float32)

    constrained_projection = _apply_constrained_insert_projection(
        obs=obs,
        env_states=env_states,
        record=record,
        args=args,
    )

    peg_pose = obs[:, LAYOUT["peg_pose"]].astype(np.float32)
    tcp_pose = obs[:, LAYOUT["tcp_pose"]].astype(np.float32)
    hole_pose = obs[:, LAYOUT["hole_pose"]].astype(np.float32)
    qpos = obs[:, LAYOUT["qpos"]].astype(np.float32)
    qvel = obs[:, LAYOUT["qvel"]].astype(np.float32)
    peg_half_length = obs[:, LAYOUT["peg_half_size"]][:, 0].astype(np.float32)
    peg_radius = obs[:, LAYOUT["peg_half_size"]][:, 1].astype(np.float32)
    radius = obs[:, LAYOUT["hole_radius"]][:, 0].astype(np.float32)
    peg_head, peg_center_hole, peg_tail_hole, peg_axis_cos = _peg_axis_at_hole_np(
        peg_pose,
        hole_pose,
        peg_half_length,
    )
    peg_line_points = _peg_line_samples_at_hole_np(
        peg_pose,
        hole_pose,
        peg_half_length,
        int(args.peg_line_samples),
    )
    line_yz_max, line_yz_limit, line_axis_cos, wall_risk = _peg_line_gate_info_np(
        peg_line_points,
        peg_half_length,
        peg_radius,
        radius,
        args,
    )
    inserted = _strict_inserted_from_line(
        peg_head,
        line_yz_max,
        line_yz_limit,
        line_axis_cos,
        wall_risk,
        args,
    )
    grasped = _grasped_from_obs(obs)
    robust_held = _robust_held_from_obs(obs, float(peg_pose[0, 2]), args)
    if not bool(inserted[-1]):
        raise RuntimeError(
            f"{sample_id} failed strict standardized final insertion: "
            f"head={peg_head[-1].tolist()} center={peg_center_hole[-1].tolist()} "
            f"tail={peg_tail_hole[-1].tolist()} axis_cos={float(line_axis_cos[-1])} "
            f"line_yz_max={float(line_yz_max[-1])} line_yz_limit={float(line_yz_limit[-1])} "
            f"wall_risk={bool(wall_risk[-1])}"
        )

    trigger = int(record["trigger_step"])
    move_steps = int(record["move_steps"])
    target_motion = hole_pose[-1, :3] - hole_pose[0, :3]
    first_insert = int(np.flatnonzero(inserted)[0]) if inserted.any() else -1
    first_grasp = int(np.flatnonzero(grasped)[0]) if grasped.any() else -1
    hole_delta_applied = hole_pose[1:, :3] - hole_pose[:-1, :3]
    hole_delta_cumulative = hole_pose[1:, :3] - hole_pose[0:1, :3]
    triggered = np.arange(int(args.total_action_steps)) >= trigger
    summary = {
        **record,
        "sample_id": sample_id,
        "control_mode": "pd_ee_delta_pose_labels_from_physical_pd_joint_pos_episode",
        "num_video_frames": int(args.total_video_frames),
        "num_action_steps": int(args.total_action_steps),
        "actions_shape": list(action_labels.shape),
        "success_at_end": bool(inserted[-1]),
        "inserted_end": bool(inserted[-1]),
        "live_success_end": bool(inserted[-1]),
        "first_insert_step": first_insert,
        "first_grasp_step": first_grasp,
        "first_robust_hold_step": int(np.flatnonzero(robust_held)[0]) if robust_held.any() else -1,
        "physical_gate": "strict peg centerline-in-hole channel plus wall collision risk check",
        "final_line_yz_max_m": float(line_yz_max[-1]),
        "final_line_yz_limit_m": float(line_yz_limit[-1]),
        "final_wall_collision_risk": bool(wall_risk[-1]),
        "source_kind": str(record.get("source_kind", args.source_kind)),
        "baseline_dp_expected": str(record.get("baseline_dp_expected", "")),
        "teacher_phases": list(record.get("teacher_phases", [])),
        "constrained_insert_projection": constrained_projection,
        "target_motion_norm_m": float(np.linalg.norm(target_motion)),
        "target_motion_xyz": target_motion.astype(float).tolist(),
        "trigger_reason": record.get("trigger_reason", "late target motion after peg grasp and near-hole prealignment"),
        "target_motion_policy": record.get(
            "target_motion_policy",
            "target remains static until trigger, then moves quickly away from the waiting peg; robot replans to the final hole pose",
        ),
    }

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    if out_h5.exists():
        out_h5.unlink()
    with h5py.File(out_h5, "w") as h5:
        group = h5.create_group("traj_0")
        group.attrs["summary_json"] = json.dumps(_jsonable(summary), sort_keys=True)
        group.attrs["source_summary_json"] = json.dumps(_jsonable(summary), sort_keys=True)
        _write_dataset(group, "actions", action_labels)
        _write_dataset(group, "obs_current", obs)
        _write_dataset(group, "obs_stack", _stack_obs(obs))
        _write_dataset(group, "source_frame_indices", np.arange(int(args.total_video_frames), dtype=np.float32))
        _write_dataset(group, "rewards", np.zeros((int(args.total_action_steps),), dtype=np.float32))
        _write_dataset(group, "terminated", np.zeros((int(args.total_action_steps),), dtype=bool))
        truncated = np.zeros((int(args.total_action_steps),), dtype=bool)
        truncated[-1] = True
        _write_dataset(group, "truncated", truncated)
        slots = group.create_group("slots")
        _write_dataset(slots, "hole_pose", hole_pose)
        _write_dataset(slots, "peg_pose", peg_pose)
        _write_dataset(slots, "tcp_pose", tcp_pose)
        _write_dataset(slots, "qpos", qpos)
        _write_dataset(slots, "qvel", qvel)
        _write_dataset(slots, "hole_radius", radius)
        _write_dataset(slots, "peg_head_at_hole", peg_head)
        _write_dataset(slots, "peg_center_at_hole", peg_center_hole)
        _write_dataset(slots, "peg_tail_at_hole", peg_tail_hole)
        _write_dataset(slots, "peg_axis_cos_hole_x", peg_axis_cos)
        _write_dataset(slots, "peg_line_points_at_hole", peg_line_points)
        _write_dataset(slots, "peg_line_yz_max", line_yz_max)
        _write_dataset(slots, "peg_line_yz_limit", line_yz_limit)
        _write_dataset(slots, "peg_line_axis_cos_hole_x", line_axis_cos)
        _write_dataset(slots, "peg_wall_collision_risk", wall_risk)
        _write_dataset(slots, "grasped", grasped)
        _write_dataset(slots, "robust_held", robust_held)
        _write_dataset(slots, "inserted", inserted)
        hole_velocity = np.zeros((int(args.total_video_frames), 3), dtype=np.float32)
        hole_velocity[1:] = hole_delta_applied
        _write_dataset(slots, "hole_velocity_step", hole_velocity)
        perturb = group.create_group("perturb")
        _write_dataset(perturb, "hole_delta_applied", hole_delta_applied.astype(np.float32))
        _write_dataset(perturb, "hole_delta_cumulative", hole_delta_cumulative.astype(np.float32))
        peg_delta_applied = np.zeros((int(args.total_action_steps), 3), dtype=np.float32)
        peg_delta_step = int(record.get("peg_delta_applied_step", -1))
        if 0 <= peg_delta_step < int(args.total_action_steps):
            peg_delta_applied[peg_delta_step] = np.asarray(
                record.get("peg_delta_applied_xyz", [0.0, 0.0, 0.0]),
                dtype=np.float32,
            )
        _write_dataset(perturb, "peg_delta_applied", peg_delta_applied)
        _write_dataset(perturb, "triggered", triggered.astype(bool))
        _write_dataset(perturb, "trigger_step", np.full((int(args.total_action_steps),), trigger, dtype=np.int32))
        _write_dataset(perturb, "move_end_step", np.full((int(args.total_action_steps),), trigger + move_steps, dtype=np.int32))
        phase_index = SCENARIOS.index(record["scenario"]) if record["scenario"] in SCENARIOS else -1
        _write_dataset(perturb, "phase", np.full((int(args.total_action_steps),), phase_index, dtype=np.int32))
        env_group = group.create_group("env_states")
        _write_dict_group(env_group, env_states)

    return {
        "sample_id": sample_id,
        "scenario": record["scenario"],
        "path": str(out_h5),
        "split": "val" if _stable_val(sample_id, args.val_fraction) else "train",
        "raw_action_steps": raw_steps,
        "num_video_frames": int(args.total_video_frames),
        "num_action_steps": int(args.total_action_steps),
        "success_at_end": bool(inserted[-1]),
        "first_grasp_step": first_grasp,
        "first_robust_hold_step": int(np.flatnonzero(robust_held)[0]) if robust_held.any() else -1,
        "trigger_step": trigger,
        "move_end_step": trigger + move_steps,
        "first_insert_step": first_insert,
        "target_motion_norm_m": float(np.linalg.norm(target_motion)),
        "final_peg_head_at_hole": peg_head[-1].astype(float).tolist(),
        "final_peg_center_at_hole": peg_center_hole[-1].astype(float).tolist(),
        "final_peg_tail_at_hole": peg_tail_hole[-1].astype(float).tolist(),
        "final_peg_axis_cos_hole_x": float(line_axis_cos[-1]),
        "final_line_yz_max_m": float(line_yz_max[-1]),
        "final_line_yz_limit_m": float(line_yz_limit[-1]),
        "wall_collision_risk_any": bool(wall_risk.any()),
    }


def _scenario_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record["scenario"]] = counts.get(record["scenario"], 0) + 1
    return dict(sorted(counts.items()))


def _audit_records(records: list[dict[str, Any]], args: Args) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for record in records:
        if not record["success_at_end"]:
            failures.append({"sample_id": record["sample_id"], "failure": "not_success_at_end"})
        if record["trigger_step"] <= record["first_grasp_step"]:
            failures.append({"sample_id": record["sample_id"], "failure": "trigger_before_grasp"})
        if record["trigger_step"] < args.min_trigger_step or record["trigger_step"] > args.max_trigger_step:
            failures.append({"sample_id": record["sample_id"], "failure": "trigger_out_of_range"})
        if record["move_end_step"] - record["trigger_step"] > args.continuous_move_steps_max:
            failures.append({"sample_id": record["sample_id"], "failure": "target_motion_too_slow"})
        if bool(record.get("wall_collision_risk_any", False)):
            failures.append({"sample_id": record["sample_id"], "failure": "wall_collision_risk_any"})
        if float(record.get("final_line_yz_max_m", 0.0)) > float(record.get("final_line_yz_limit_m", 0.0)):
            failures.append(
                {
                    "sample_id": record["sample_id"],
                    "failure": "final_line_yz_exceeds_limit",
                    "final_line_yz_max_m": record.get("final_line_yz_max_m"),
                    "final_line_yz_limit_m": record.get("final_line_yz_limit_m"),
                }
            )
    motion = np.asarray([r["target_motion_norm_m"] for r in records], dtype=np.float32)
    final_line_yz = np.asarray([r.get("final_line_yz_max_m", np.nan) for r in records], dtype=np.float32)
    return {
        "num_paths": len(records),
        "counts": _scenario_counts(records),
        "failures": failures,
        "motion_min": float(motion.min()) if motion.size else None,
        "motion_mean": float(motion.mean()) if motion.size else None,
        "motion_max": float(motion.max()) if motion.size else None,
        "final_line_yz_max_min": float(np.nanmin(final_line_yz)) if final_line_yz.size else None,
        "final_line_yz_max_mean": float(np.nanmean(final_line_yz)) if final_line_yz.size else None,
        "final_line_yz_max_max": float(np.nanmax(final_line_yz)) if final_line_yz.size else None,
        "trigger_min": int(min(r["trigger_step"] for r in records)) if records else None,
        "trigger_max": int(max(r["trigger_step"] for r in records)) if records else None,
        "move_steps_min": int(min(r["move_end_step"] - r["trigger_step"] for r in records)) if records else None,
        "move_steps_max": int(max(r["move_end_step"] - r["trigger_step"] for r in records)) if records else None,
        "first_insert_min": int(min(r["first_insert_step"] for r in records)) if records else None,
        "first_insert_max": int(max(r["first_insert_step"] for r in records)) if records else None,
    }


def main() -> None:
    args = tyro.cli(Args)
    if bool(args.constrained_insert_projection) and os.environ.get(
        "ALLOW_REJECTED_CONSTRAINED_INSERT_DIAGNOSTIC"
    ) != "true":
        raise RuntimeError(
            "constrained_insert_projection is disabled for active fix3 generation because user review "
            "found penetration/self-drilling artifacts in v8. Reproduce the fix2/official insertion "
            "protocol instead; use ALLOW_REJECTED_CONSTRAINED_INSERT_DIAGNOSTIC=true only for archived diagnostics."
        )
    if args.total_video_frames != 301 or args.total_action_steps != 300:
        raise ValueError("fix3 late-trigger data must keep the 301-frame / 300-action contract")
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
        source_type="fix3_late_trigger_motionplanning",
        source_desc="Late-trigger dynamic target motion with physical expert replanning and strict insertion filtering.",
        video_fps=30,
    )
    raw_h5_path = Path(env._h5_file.filename)
    rng = np.random.default_rng(args.seed)
    accepted: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    scenario_order = _parse_sequence(args.scenario_sequence)
    quota_overrides = _parse_int_map(args.scenario_quotas, allowed=set(SCENARIOS), flag_name="scenario_quotas")
    seed_bases = _parse_int_map(args.scenario_seed_bases, allowed=set(SCENARIOS), flag_name="scenario_seed_bases")
    if quota_overrides:
        missing_quota = sorted(set(scenario_order) - set(quota_overrides))
        if missing_quota:
            raise ValueError(f"scenario_quotas missing active scenarios: {missing_quota}")
        target_counts = {scenario: int(quota_overrides[scenario]) for scenario in scenario_order}
        target_total = sum(target_counts.values())
        if target_total != int(args.num_demos):
            print(
                json.dumps(
                    {
                        "event": "hard_teacher_num_demos_overridden_by_quotas",
                        "num_demos_arg": int(args.num_demos),
                        "quota_total": int(target_total),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    else:
        target_counts = {scenario: int(args.num_demos) // len(scenario_order) for scenario in scenario_order}
        remainder = int(args.num_demos) - sum(target_counts.values())
        for scenario in scenario_order[:remainder]:
            target_counts[scenario] += 1
    scenario_attempts = {scenario: 0 for scenario in scenario_order}

    try:
        for attempt_i in range(int(args.max_attempts)):
            if len(accepted) >= int(args.num_demos):
                break
            underfilled = [s for s in scenario_order if sum(1 for r in accepted if r["scenario"] == s) < target_counts[s]]
            if not underfilled:
                break
            scenario = underfilled[attempt_i % len(underfilled)]
            scenario_i = int(scenario_attempts[scenario])
            scenario_attempts[scenario] += 1
            episode_seed = int(seed_bases.get(scenario, int(args.seed)) + scenario_i)
            env.reset(seed=episode_seed)
            _replace_record_initial_state(env)
            if scenario in HOLE_MOTION_SCENARIOS:
                result = _run_late_trigger_episode(env, scenario, rng, args)
            elif scenario == "none":
                result = _run_static_teacher_episode(env, rng, args)
            elif scenario in PEG_PERTURB_SCENARIOS:
                result = _run_peg_recovery_teacher_episode(env, scenario, rng, args)
            else:
                raise ValueError(f"unexpected scenario {scenario!r}")
            attempt_record = {
                "attempt": attempt_i,
                "episode_seed": episode_seed,
                "scenario": scenario,
                "accepted": result is not None,
            }
            if result is not None:
                env.flush_trajectory()
                traj_id = f"traj_{env._episode_id}"
                result = {
                    **result,
                    "attempt": attempt_i,
                    "episode_seed": episode_seed,
                    "trajectory": traj_id,
                }
                with h5py.File(raw_h5_path, "a") as raw_h5:
                    raw_h5[traj_id].attrs["fix3_late_trigger_source_json"] = json.dumps(_jsonable(result), sort_keys=True)
                accepted.append(result)
                attempt_record.update(result)
            else:
                env.flush_trajectory(save=False)
            attempts.append(attempt_record)
            print(
                json.dumps(
                    {
                        "event": "fix3_late_trigger_attempt",
                        "attempt": attempt_i,
                        "accepted": len(accepted),
                        "scenario": scenario,
                        "scenario_attempt": scenario_i,
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
            scenario = record["scenario"]
            sample_id = f"{scenario}_seed{record['episode_seed']:06d}_idx{idx:04d}"
            out_h5 = review_dir / f"{sample_id}.fix3" / f"{sample_id}.h5"
            records.append(
                _standardize_one(
                    raw_group=raw_h5[record["trajectory"]],
                    out_h5=out_h5,
                    sample_id=sample_id,
                    record=record,
                    args=args,
                )
            )

    paths_file = Path(args.paths_file) if args.paths_file else output_root / "fix3_h5_paths.txt"
    paths_file.write_text("\n".join(record["path"] for record in records) + "\n")
    audit = _audit_records(records, args)
    (output_root / "source_audit.json").write_text(json.dumps(_jsonable(audit), indent=2, sort_keys=True) + "\n")
    manifest = {
        "args": _jsonable(asdict(args)),
        "boundary": (
            "Corrected fix3 late-trigger physical review package. Target motion "
            "starts only after grasp/prealignment, final target poses stay in "
            "reachable static-domain ranges, and demos are filtered by real "
            "episode success plus stricter insertion geometry. User approval is "
            "required before SFT."
        ),
        "raw_h5_path": str(raw_h5_path),
        "raw_json_path": str(raw_h5_path.with_suffix(".json")),
        "output_root": str(output_root),
        "paths_file": str(paths_file),
        "num_records": len(records),
        "source_kind": str(args.source_kind),
        "target_counts": target_counts,
        "scenario_attempts": scenario_attempts,
        "scenario_counts": _scenario_counts(records),
        "accepted": accepted,
        "attempts": attempts,
        "records": records,
        "audit": audit,
    }
    (output_root / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "event": "fix3_late_trigger_manifest_written",
                "manifest": str(output_root / "manifest.json"),
                "paths_file": str(paths_file),
                "num_records": len(records),
                "audit_failures": audit["failures"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
