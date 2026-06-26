#!/usr/bin/env python3
"""Generate recovery-teacher rows from real failed live-loop state snapshots.

This is not the broad hard-teacher generator. It starts from simulator states
replayed from failed closed-loop artifacts, then tries to regrasp/replan/insert
from that real failed geometry.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import gymnasium as gym
import h5py
import numpy as np
import sapien

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import mani_skill.envs  # noqa: F401,E402
from mani_skill.examples.motionplanning.base_motionplanner.utils import (  # noqa: E402
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill.utils.wrappers.record import RecordEpisode  # noqa: E402

from generate_cosmos3_fix3_hard_dynamic_teacher import (  # noqa: E402
    Args as TeacherArgs,
    _grasp_current_peg,
    _hold_action,
    _insert_current_goal,
    _jsonable,
    _live_peg_head_at_hole,
    _live_peg_line_gate_info,
    _live_success,
    _make_planner,
    _replace_record_initial_state,
    _standardize_one,
    _to_np_pose,
    PandaArmMotionPlanningSolver,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--replay-summary", action="append", required=True)
    parser.add_argument("--num-demos", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--seed", type=int, default=59000000)
    parser.add_argument("--min-frame", type=int, default=200)
    parser.add_argument("--preferred-frames", default="296,288,280,272,264,256,248,240,232,224,216,208,200")
    parser.add_argument("--source-kind", default="failed_state_recovery_teacher_20260614")
    parser.add_argument("--trajectory-name", default="failed_state_recovery_teacher_raw_pd_joint_pos")
    parser.add_argument("--env-id", default="PegInsertionSide-v1")
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--render-backend", default="none")
    parser.add_argument("--render-mode", default="none")
    parser.add_argument("--reward-mode", default="dense")
    parser.add_argument("--total-video-frames", type=int, default=301)
    parser.add_argument("--total-action-steps", type=int, default=300)
    parser.add_argument("--max-raw-action-steps", type=int, default=300)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--joint-vel-limits", type=float, default=0.90)
    parser.add_argument("--joint-acc-limits", type=float, default=0.90)
    parser.add_argument("--preinsert-yz-max-m", type=float, default=0.012)
    parser.add_argument("--final-insert-offset-m", type=float, default=0.05)
    parser.add_argument("--insert-step-m", type=float, default=0.02)
    parser.add_argument("--preinsert-refine-steps", type=int, default=5)
    parser.add_argument("--final-insert-refine-steps", type=int, default=4)
    parser.add_argument("--final-insert-settle-steps", type=int, default=0)
    parser.add_argument("--use-existing-grasp-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--planned-regrasp-mode",
        choices=("current_tcp", "goal_y", "world_y", "world_x"),
        default="current_tcp",
    )
    parser.add_argument("--stage-before-insert", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stage-offsets-m", default="-0.22,-0.18,-0.14,-0.11")
    parser.add_argument("--execute-all-stage-offsets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-regrasp-after-stage", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--release-regrasp-mode",
        choices=("current_tcp", "goal_y", "world_y", "world_x"),
        default="current_tcp",
    )
    parser.add_argument("--release-regrasp-hold-steps", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_compute_step() -> None:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    step_id = os.environ.get("SLURM_STEP_ID", "")
    if not job_id or not step_id or step_id == "extern":
        raise SystemExit(
            "refusing_login_node_execution=true\n"
            "reason=Failed-state recovery teacher must run inside a compute-node srun step."
        )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n")


def emit(event: str, **payload: Any) -> None:
    print(json.dumps(_jsonable({"event": event, **payload}), sort_keys=True), flush=True)


def _read_h5_payload(group: h5py.Group | h5py.Dataset) -> Any:
    if isinstance(group, h5py.Dataset):
        arr = np.asarray(group)
        if arr.ndim >= 1 and arr.shape[0] == 1:
            arr = arr[0]
        return arr
    return {key: _read_h5_payload(value) for key, value in group.items()}


def load_snapshot_state(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        return _read_h5_payload(h5["state"])


def preferred_frame_order(raw: str) -> list[int]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def parse_float_list(raw: str) -> list[float]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    return out


def collect_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    preferred = preferred_frame_order(args.preferred_frames)
    rank = {frame: i for i, frame in enumerate(preferred)}
    candidates: list[dict[str, Any]] = []
    for summary_path_text in args.replay_summary:
        summary_path = Path(summary_path_text).resolve()
        summary = read_json(summary_path)
        old_final = summary.get("old_final_eval") or {}
        if bool(old_final.get("success")):
            emit("skip_replay_success_case", summary=str(summary_path), old_final_eval=old_final)
            continue
        scenario = str(summary.get("scenario") or "unknown")
        sample_name = str(summary.get("sample_name") or summary_path.parent.name)
        for snapshot in summary.get("snapshots") or []:
            frame = int(snapshot.get("frame", -1))
            if frame < int(args.min_frame):
                continue
            live_eval = snapshot.get("live_eval") or {}
            if bool(live_eval.get("success")):
                continue
            path = Path(str(snapshot.get("path") or "")).resolve()
            if not path.is_file():
                continue
            candidates.append(
                {
                    "summary_path": str(summary_path),
                    "snapshot_path": str(path),
                    "scenario": scenario,
                    "sample_name": sample_name,
                    "frame": frame,
                    "old_final_eval": old_final,
                    "snapshot_live_eval": live_eval,
                    "rank": rank.get(frame, len(rank) + max(0, 300 - frame)),
                }
            )
    candidates.sort(key=lambda item: (item["rank"], item["sample_name"], -int(item["frame"])))
    return candidates


def build_teacher_args(args: argparse.Namespace) -> TeacherArgs:
    return TeacherArgs(
        output_root=str(args.output_root),
        trajectory_name=str(args.trajectory_name),
        env_id=str(args.env_id),
        obs_mode=str(args.obs_mode),
        control_mode=str(args.control_mode),
        sim_backend=str(args.sim_backend),
        render_backend=str(args.render_backend),
        render_mode=str(args.render_mode),
        reward_mode=str(args.reward_mode),
        num_demos=int(args.num_demos),
        max_attempts=int(args.max_attempts),
        seed=int(args.seed),
        total_video_frames=int(args.total_video_frames),
        total_action_steps=int(args.total_action_steps),
        max_raw_action_steps=int(args.max_raw_action_steps),
        min_trigger_step=0,
        max_trigger_step=int(args.total_action_steps),
        preinsert_yz_max_m=float(args.preinsert_yz_max_m),
        final_insert_offset_m=float(args.final_insert_offset_m),
        insert_step_m=float(args.insert_step_m),
        preinsert_refine_steps=int(args.preinsert_refine_steps),
        final_insert_refine_steps=int(args.final_insert_refine_steps),
        final_insert_settle_steps=int(args.final_insert_settle_steps),
        source_kind=str(args.source_kind),
        joint_vel_limits=float(args.joint_vel_limits),
        joint_acc_limits=float(args.joint_acc_limits),
        val_fraction=float(args.val_fraction),
        overwrite=bool(args.overwrite),
    )


def tensor_bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    return bool(np.asarray(arr).reshape(-1)[0])


def current_grasp_transform(base_env: Any, planner: Any) -> tuple[sapien.Pose, sapien.Pose, dict[str, Any]] | None:
    try:
        grasping_before = tensor_bool(base_env.agent.is_grasping(base_env.peg))
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic branch.
        return None
    if not grasping_before:
        return None
    planner.close_gripper(t=2)
    try:
        grasping_after = tensor_bool(base_env.agent.is_grasping(base_env.peg))
    except Exception:
        grasping_after = grasping_before
    tcp_p, tcp_q = _to_np_pose(base_env.agent.tcp.pose)
    peg_p, peg_q = _to_np_pose(base_env.peg.pose)
    return (
        sapien.Pose(tcp_p, tcp_q),
        sapien.Pose(peg_p, peg_q),
        {
            "grasp_source": "existing_live_grasp",
            "agent_is_grasping_before_close": bool(grasping_before),
            "agent_is_grasping_after_close": bool(grasping_after),
        },
    )


def stage_grasped_peg_before_insert(
    *,
    base_env: Any,
    planner: Any,
    teacher_args: TeacherArgs,
    grasp_pose: sapien.Pose,
    peg_pose_at_grasp: sapien.Pose,
    stage_offsets_m: list[float],
    execute_all_stage_offsets: bool,
) -> tuple[bool, list[dict[str, Any]]]:
    """Move the grasped peg to a safe pose in front of the current moved hole.

    The failed live states often leave the peg skewed near the hole/wall. A
    direct insertion plan then fails before it can create a clean preinsert
    state. This stage first aligns the grasped peg to the current goal frame
    at a negative x offset, preserving the actual grasp transform.
    """

    records: list[dict[str, Any]] = []
    any_success = False
    for offset_m in stage_offsets_m:
        desired_peg_pose = base_env.goal_pose * sapien.Pose([float(offset_m), 0.0, 0.0])
        stage_tcp_pose = desired_peg_pose * peg_pose_at_grasp.inv() * grasp_pose
        result = planner.move_to_pose_with_screw(stage_tcp_pose)
        head = _live_peg_head_at_hole(base_env)
        line_info = _live_peg_line_gate_info(base_env, teacher_args)
        record = {
            "stage_offset_m": float(offset_m),
            "planner_result": _jsonable(result),
            "peg_head_at_hole": head.astype(float).tolist(),
            "line_gate": line_info,
        }
        records.append(record)
        if result != -1:
            any_success = True
            if not bool(execute_all_stage_offsets):
                return True, records
        elif any_success:
            return True, records
    return any_success, records


def pose_axis_world(pose: Any, axis_idx: int) -> np.ndarray:
    mat = pose.to_transformation_matrix()
    if hasattr(mat, "detach"):
        arr = mat.detach().cpu().numpy()
    elif hasattr(mat, "cpu"):
        arr = mat.cpu().numpy()
    else:
        arr = np.asarray(mat)
    return np.asarray(arr[0, :3, int(axis_idx)], dtype=np.float32)


def target_closing_vector(base_env: Any, mode: str) -> np.ndarray:
    if mode == "current_tcp":
        return pose_axis_world(base_env.agent.tcp.pose, 1)
    if mode == "goal_y":
        return pose_axis_world(base_env.goal_pose, 1)
    if mode == "world_y":
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    if mode == "world_x":
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    raise ValueError(f"unknown release_regrasp_mode {mode!r}")


def task_axis_grasp_current_peg(
    base_env: Any,
    planner: Any,
    *,
    release_regrasp_mode: str,
) -> tuple[sapien.Pose, sapien.Pose, dict[str, Any]] | None:
    """Regrasp with an explicit task-frame closing axis instead of live TCP history."""

    approaching = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    target_closing = target_closing_vector(base_env, str(release_regrasp_mode))
    obb = get_actor_obb(base_env.peg)
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
    return (
        grasp_pose,
        base_env.peg.pose,
        {
            "grasp_source": f"task_axis_regrasp_{release_regrasp_mode}",
            "regrasp_mode": str(release_regrasp_mode),
            "target_closing_world": target_closing.astype(float).tolist(),
            "selected_closing_world": np.asarray(grasp_info["closing"], dtype=np.float32).astype(float).tolist(),
            "selected_approaching_world": approaching.astype(float).tolist(),
        },
    )


def release_and_regrasp_after_stage(
    *,
    env: RecordEpisode,
    base_env: Any,
    planner: Any,
    teacher_args: TeacherArgs,
    release_regrasp_mode: str,
    hold_steps: int,
) -> tuple[sapien.Pose, sapien.Pose, dict[str, Any]] | None:
    """Drop the bad live grasp at the staged pose and establish a fresh grasp."""

    before_head = _live_peg_head_at_hole(base_env)
    before_line = _live_peg_line_gate_info(base_env, teacher_args)
    planner.open_gripper(t=4)
    for _ in range(max(0, int(hold_steps))):
        env.step(_hold_action(base_env, PandaArmMotionPlanningSolver.OPEN))
    if str(release_regrasp_mode) == "current_tcp":
        base_result = _grasp_current_peg(base_env, planner)
        if base_result is None:
            return None
        grasp_pose, peg_pose_at_grasp = base_result
        grasp_meta = {
            "grasp_source": "release_regrasp_after_stage_current_tcp",
            "release_regrasp_mode": "current_tcp",
            "regrasp_mode": "current_tcp",
            "target_closing_world": target_closing_vector(base_env, "current_tcp").astype(float).tolist(),
        }
    else:
        grasp_result = task_axis_grasp_current_peg(
            base_env,
            planner,
            release_regrasp_mode=str(release_regrasp_mode),
        )
        if grasp_result is None:
            return None
        grasp_pose, peg_pose_at_grasp, grasp_meta = grasp_result
    after_head = _live_peg_head_at_hole(base_env)
    after_line = _live_peg_line_gate_info(base_env, teacher_args)
    return (
        grasp_pose,
        peg_pose_at_grasp,
        {
            "release_regrasp_hold_steps": int(hold_steps),
            "before_release_head_at_hole": before_head.astype(float).tolist(),
            "after_regrasp_head_at_hole": after_head.astype(float).tolist(),
            "before_release_line_gate": before_line,
            "after_regrasp_line_gate": after_line,
            **grasp_meta,
        },
    )


def run_recovery_attempt(
    *,
    env: RecordEpisode,
    candidate: dict[str, Any],
    teacher_args: TeacherArgs,
    recovery_args: argparse.Namespace,
    attempt_i: int,
) -> dict[str, Any] | None:
    base_env = env.unwrapped
    planner = _make_planner(env, teacher_args)
    try:
        episode_start_step = int(env._elapsed_record_steps)
        snapshot_state = load_snapshot_state(Path(candidate["snapshot_path"]))
        base_env.set_state_dict(snapshot_state)
        _replace_record_initial_state(env)

        initial_box_p, initial_box_q = _to_np_pose(base_env.box.pose)
        initial_peg_p, initial_peg_q = _to_np_pose(base_env.peg.pose)
        initial_head = _live_peg_head_at_hole(base_env)
        initial_success = _live_success(base_env)
        if bool(initial_success):
            return None

        grasp_meta: dict[str, Any] = {}
        grasp_result = None
        if bool(recovery_args.use_existing_grasp_first):
            grasp_result = current_grasp_transform(base_env, planner)
        if grasp_result is None:
            if str(recovery_args.planned_regrasp_mode) == "current_tcp":
                grasp_result = _grasp_current_peg(base_env, planner)
                grasp_meta = {
                    "grasp_source": "planned_regrasp_current_peg",
                    "planned_regrasp_mode": "current_tcp",
                }
            else:
                grasp_result = task_axis_grasp_current_peg(
                    base_env,
                    planner,
                    release_regrasp_mode=str(recovery_args.planned_regrasp_mode),
                )
                if grasp_result is not None:
                    grasp_pose, peg_pose_at_grasp, grasp_meta = grasp_result
                    grasp_meta = {
                        **grasp_meta,
                        "grasp_source": f"planned_task_axis_regrasp_{recovery_args.planned_regrasp_mode}",
                        "planned_regrasp_mode": str(recovery_args.planned_regrasp_mode),
                    }
                    grasp_result = (grasp_pose, peg_pose_at_grasp)
        else:
            grasp_pose, peg_pose_at_grasp, grasp_meta = grasp_result
            grasp_result = (grasp_pose, peg_pose_at_grasp)
        if grasp_result is None:
            emit(
                "failed_state_recovery_reject",
                reason="planner_regrasp_current_peg_failed",
                attempt=attempt_i,
                candidate=candidate,
            )
            return None
        grasp_pose, peg_pose_at_grasp = grasp_result
        stage_records: list[dict[str, Any]] = []
        stage_offsets = parse_float_list(str(recovery_args.stage_offsets_m))
        if bool(recovery_args.stage_before_insert):
            stage_ok, stage_records = stage_grasped_peg_before_insert(
                base_env=base_env,
                planner=planner,
                teacher_args=teacher_args,
                grasp_pose=grasp_pose,
                peg_pose_at_grasp=peg_pose_at_grasp,
                stage_offsets_m=stage_offsets,
                execute_all_stage_offsets=bool(recovery_args.execute_all_stage_offsets),
            )
            if not stage_ok:
                emit(
                    "failed_state_recovery_reject",
                    reason="planner_stage_before_insert_failed",
                    attempt=attempt_i,
                    candidate=candidate,
                    stage_offsets_m=stage_offsets,
                    stage_records=stage_records,
                )
                return None
        release_regrasp_meta: dict[str, Any] = {}
        if bool(recovery_args.release_regrasp_after_stage):
            release_grasp_result = release_and_regrasp_after_stage(
                env=env,
                base_env=base_env,
                planner=planner,
                teacher_args=teacher_args,
                release_regrasp_mode=str(recovery_args.release_regrasp_mode),
                hold_steps=int(recovery_args.release_regrasp_hold_steps),
            )
            if release_grasp_result is None:
                emit(
                    "failed_state_recovery_reject",
                    reason="planner_release_regrasp_after_stage_failed",
                    attempt=attempt_i,
                    candidate=candidate,
                    stage_records=stage_records,
                )
                return None
            grasp_pose, peg_pose_at_grasp, release_regrasp_meta = release_grasp_result
            grasp_meta = release_regrasp_meta

        insert_start = int(env._elapsed_record_steps) - int(episode_start_step)
        insert_ok, insert_info = _insert_current_goal(
            base_env=base_env,
            planner=planner,
            grasp_pose=grasp_pose,
            peg_pose_at_grasp=peg_pose_at_grasp,
            args=teacher_args,
        )
        if not insert_ok:
            emit(
                "failed_state_recovery_reject",
                reason="planner_recovery_insert_failed",
                attempt=attempt_i,
                candidate=candidate,
                stage_records=stage_records,
                insert_info=insert_info,
            )
            return None

        raw_steps = int(env._elapsed_record_steps) - int(episode_start_step)
        if raw_steps > int(teacher_args.max_raw_action_steps):
            emit(
                "failed_state_recovery_reject",
                reason="raw_steps_exceed_limit",
                raw_steps=raw_steps,
                episode_start_step=episode_start_step,
                absolute_elapsed_record_steps=int(env._elapsed_record_steps),
                attempt=attempt_i,
                candidate=candidate,
            )
            return None

        final_box_p, final_box_q = _to_np_pose(base_env.box.pose)
        final_peg_p, final_peg_q = _to_np_pose(base_env.peg.pose)
        final_head = _live_peg_head_at_hole(base_env)
        return {
            "scenario": str(candidate["scenario"]),
            "trigger_step": 0,
            "move_steps": 0,
            "raw_action_steps": raw_steps,
            "initial_box_xyz": initial_box_p.astype(float).tolist(),
            "final_box_xyz": final_box_p.astype(float).tolist(),
            "initial_box_quat": initial_box_q.astype(float).tolist(),
            "final_box_quat": final_box_q.astype(float).tolist(),
            "initial_peg_xyz": initial_peg_p.astype(float).tolist(),
            "final_peg_xyz": final_peg_p.astype(float).tolist(),
            "initial_peg_quat": initial_peg_q.astype(float).tolist(),
            "final_peg_quat": final_peg_q.astype(float).tolist(),
            "target_motion_norm_m": 0.0,
            "motion_path_xyz": [initial_box_p.astype(float).tolist(), final_box_p.astype(float).tolist()],
            "final_insert_start_step": int(insert_start),
            "source_kind": str(teacher_args.source_kind),
            "teacher_phases": [
                "restore_failed_live_state",
                str(grasp_meta.get("grasp_source", "unknown_grasp_source")),
                "motion_planner_stage_grasped_peg_before_insert",
                *(
                    ["motion_planner_release_regrasp_after_stage"]
                    if bool(recovery_args.release_regrasp_after_stage)
                    else []
                ),
                "motion_planner_replan_to_current_hole",
                "motion_planner_insert",
            ],
            "trigger_reason": "row starts from a real failed post-motion live-loop state",
            "target_motion_policy": (
                "The target has already moved before frame 0. perturb.triggered is true from step 0 "
                "so full-episode WAM export treats later prefixes as post-motion recovery context."
            ),
            "baseline_dp_expected": (
                "failed-state recovery supplement; original iter2700 closed loop already failed on "
                "the source live state recorded in replay_summary"
            ),
            "failed_state_snapshot": {
                **candidate,
                "initial_peg_head_at_hole": initial_head.astype(float).tolist(),
                "final_peg_head_at_hole": final_head.astype(float).tolist(),
                "grasp_meta": grasp_meta,
                "planned_regrasp_mode": str(recovery_args.planned_regrasp_mode),
                "stage_before_insert": bool(recovery_args.stage_before_insert),
                "stage_offsets_m": stage_offsets,
                "execute_all_stage_offsets": bool(recovery_args.execute_all_stage_offsets),
                "stage_records": stage_records,
                "release_regrasp_after_stage": bool(recovery_args.release_regrasp_after_stage),
                "release_regrasp_meta": release_regrasp_meta,
            },
            "success": True,
        }
    finally:
        planner.close()


def custom_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    for record in records:
        if not bool(record.get("success_at_end")):
            failures.append({"sample_id": record.get("sample_id"), "failure": "not_success_at_end"})
        if bool(record.get("wall_collision_risk_any")):
            failures.append({"sample_id": record.get("sample_id"), "failure": "wall_collision_risk_any"})
        if float(record.get("final_line_yz_max_m", 0.0)) > float(record.get("final_line_yz_limit_m", 0.0)):
            failures.append(
                {
                    "sample_id": record.get("sample_id"),
                    "failure": "final_line_yz_exceeds_limit",
                    "final_line_yz_max_m": record.get("final_line_yz_max_m"),
                    "final_line_yz_limit_m": record.get("final_line_yz_limit_m"),
                }
            )
    return {
        "num_paths": len(records),
        "failures": failures,
        "ready_for_render_review": len(records) > 0 and not failures,
        "boundary": (
            "Failed-state recovery rows start from post-motion live failed states. "
            "They need rendered visual review before merge/export/SFT."
        ),
    }


def main() -> int:
    args = parse_args()
    require_compute_step()
    if args.total_video_frames != 301 or args.total_action_steps != 300:
        raise SystemExit("active contract requires 301 video frames and 300 action steps")
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing non-empty output_root without --overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    raw_dir = output_root / "raw_pd_joint_pos"
    h5_root = output_root / "h5"
    raw_dir.mkdir(parents=True, exist_ok=True)
    h5_root.mkdir(parents=True, exist_ok=True)

    teacher_args = build_teacher_args(args)
    candidates = collect_candidates(args)
    if not candidates:
        raise RuntimeError("no failed-state snapshot candidates found")
    write_json(output_root / "candidate_snapshots.json", candidates)
    emit("failed_state_recovery_candidates", count=len(candidates), output_root=str(output_root))

    render_mode = None if str(args.render_mode).lower() in {"", "none", "null"} else str(args.render_mode)
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
        source_type="failed_state_recovery_teacher",
        source_desc="Recovery teacher starts from replayed failed live-loop simulator states.",
        video_fps=30,
    )
    raw_h5_path = Path(env._h5_file.filename)
    accepted: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    try:
        for attempt_i, candidate in enumerate(candidates[: int(args.max_attempts)]):
            if len(accepted) >= int(args.num_demos):
                break
            env.reset(seed=int(args.seed) + attempt_i)
            result = run_recovery_attempt(
                env=env,
                candidate=candidate,
                teacher_args=teacher_args,
                recovery_args=args,
                attempt_i=attempt_i,
            )
            attempt = {
                "attempt": attempt_i,
                "accepted": result is not None,
                "candidate": candidate,
            }
            if result is not None:
                env.flush_trajectory()
                traj_id = f"traj_{env._episode_id}"
                result = {**result, "attempt": attempt_i, "trajectory": traj_id}
                with h5py.File(raw_h5_path, "a") as raw_h5:
                    raw_h5[traj_id].attrs["failed_state_recovery_source_json"] = json.dumps(
                        _jsonable(result),
                        sort_keys=True,
                    )
                accepted.append(result)
                attempt.update(result)
            else:
                env.flush_trajectory(save=False)
            attempts.append(attempt)
            emit(
                "failed_state_recovery_attempt",
                attempt=attempt_i,
                accepted=len(accepted),
                success=result is not None,
                sample=candidate["sample_name"],
                frame=candidate["frame"],
            )
    finally:
        env.close()

    if len(accepted) < int(args.num_demos):
        write_json(
            output_root / "failed_state_recovery_partial_manifest.json",
            {
                "args": vars(args),
                "teacher_args": asdict(teacher_args),
                "accepted": accepted,
                "attempts": attempts,
                "raw_h5_path": str(raw_h5_path),
            },
        )
        raise RuntimeError(f"only accepted {len(accepted)} / {args.num_demos} recovery demos")

    records: list[dict[str, Any]] = []
    with h5py.File(raw_h5_path, "r") as raw_h5:
        for idx, record in enumerate(accepted):
            scenario = record["scenario"]
            frame = int((record.get("failed_state_snapshot") or {}).get("frame", -1))
            sample_id = f"{scenario}_failed_state_recovery_seed{int(args.seed) + idx:06d}_f{frame:03d}_idx{idx:04d}"
            out_h5 = h5_root / f"{sample_id}.fix3" / f"{sample_id}.h5"
            records.append(
                _standardize_one(
                    raw_group=raw_h5[record["trajectory"]],
                    out_h5=out_h5,
                    sample_id=sample_id,
                    record=record,
                    args=teacher_args,
                )
            )

    paths_file = output_root / "fix3_h5_paths.txt"
    paths_file.write_text("\n".join(record["path"] for record in records) + "\n")
    audit = custom_audit(records)
    write_json(output_root / "source_audit.json", audit)
    manifest = {
        "args": vars(args),
        "teacher_args": asdict(teacher_args),
        "boundary": (
            "Failed-state recovery supplement candidate. These rows start from "
            "real failed live-loop simulator states and are not approved for "
            "merge/export/SFT until RGB review confirms real regrasp/hold/insertion."
        ),
        "raw_h5_path": str(raw_h5_path),
        "output_root": str(output_root),
        "paths_file": str(paths_file),
        "num_records": len(records),
        "source_kind": str(args.source_kind),
        "accepted": accepted,
        "attempts": attempts,
        "records": records,
        "audit": audit,
    }
    write_json(output_root / "manifest.json", manifest)
    emit("failed_state_recovery_manifest_written", manifest=str(output_root / "manifest.json"), records=len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
