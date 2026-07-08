#!/usr/bin/env python3
"""Phase 03 Oracle boundary probe for active ManiSkill PegInsertionSide-v1.

This is an upper-bound diagnostic only. It runs a real DP prefix, confirms RGB
Cosmos evidence exists, then records an explicitly logged Oracle boundary /
decision point. It must not teleport the peg or be reported as physical
controller success.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import socket
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
os.environ.setdefault("DISPLAY", "")

ROOT = Path(__file__).resolve().parents[2]
for extra in (
    ROOT / "scripts" / "training",
    ROOT / "deps" / "ManiSkill_clean" / "examples" / "baselines" / "diffusion_policy",
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import gymnasium as gym
import imageio.v2 as imageio
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from mani_skill.utils import common
from mani_skill.utils.structs import Pose
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack, RecordEpisode
from PIL import Image, ImageDraw

import train as ms_train


class RenderTimeout(RuntimeError):
    pass


def event(name: str, **payload: Any) -> None:
    print(json.dumps({"event": name, **jsonable(payload)}, sort_keys=True), flush=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return jsonable(value.item())
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def require_allocation() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("refusing_login_node_execution=true; run inside a tmux-held Slurm allocation")


def build_train_args(saved: dict[str, Any], args: argparse.Namespace) -> ms_train.Args:
    train_args = ms_train.Args()
    for key, value in saved.items():
        if hasattr(train_args, key):
            setattr(train_args, key, value)
    train_args.env_id = args.env_id
    train_args.control_mode = args.control_mode
    train_args.sim_backend = args.sim_backend
    train_args.max_episode_steps = args.max_episode_steps
    train_args.num_eval_envs = 1
    train_args.num_eval_episodes = 1
    train_args.seed = args.seed
    train_args.capture_video = False
    return train_args


def make_env(train_args: ms_train.Args, video_dir: Path | None):
    assert train_args.sim_backend == "physx_cpu"

    def thunk():
        env = gym.make(
            train_args.env_id,
            reconfiguration_freq=1,
            control_mode=train_args.control_mode,
            reward_mode="sparse",
            obs_mode="state",
            render_mode="rgb_array",
            human_render_camera_configs=dict(shader_pack="default"),
            max_episode_steps=train_args.max_episode_steps,
        )
        env = FrameStack(env, num_stack=train_args.obs_horizon)
        env = CPUGymWrapper(env, ignore_terminations=False, record_metrics=True)
        if video_dir is not None:
            env = RecordEpisode(
                env,
                output_dir=str(video_dir),
                save_trajectory=False,
                save_video=True,
                info_on_video=True,
                source_type="phase03_oracle",
                source_desc="phase03 oracle upper-bound diagnostic; final-seat is explicit state intervention",
            )
        env.action_space.seed(train_args.seed)
        env.observation_space.seed(train_args.seed)
        return env

    return SyncVectorEnv([thunk])


def as_rgb_frame(frame: Any) -> np.ndarray:
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    arr = np.asarray(frame)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max(initial=0) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


def annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    y = 6
    for line in lines:
        text = str(line)
        draw.rectangle((4, y - 2, min(image.width - 4, 10 + 7 * len(text)), y + 15), fill=(0, 0, 0))
        draw.text((8, y), text, fill=(255, 255, 255))
        y += 18
    return np.asarray(image, dtype=np.uint8)


def safe_render(env: Any, timeout_s: int) -> np.ndarray | None:
    def _raise_timeout(_signum, _frame):
        raise RenderTimeout(f"render_timeout_s={timeout_s}")

    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(max(1, int(timeout_s)))
    try:
        return as_rgb_frame(env.render())
    except RenderTimeout as exc:
        event("render_timeout", error=str(exc))
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def append_frame(frames: list[np.ndarray], env: Any, lines: list[str], timeout_s: int) -> bool:
    if int(timeout_s) <= 0:
        event("render_skipped", lines=lines)
        return False
    frame = safe_render(env, timeout_s)
    if frame is None:
        return False
    frames.append(annotate(frame, lines))
    return True


def first3(value: Any) -> list[float]:
    arr = np.asarray(jsonable(value), dtype=np.float64).reshape(-1)
    return arr[:3].astype(float).tolist()


def eval_state(base_env: Any) -> dict[str, Any]:
    info = base_env.evaluate()
    rel = first3(info.get("peg_head_pos_at_hole"))
    return {
        "success": bool(np.asarray(info.get("success")).reshape(-1)[0]),
        "peg_head_at_hole": rel,
        "peg_head_l2": float(np.linalg.norm(np.asarray(rel, dtype=np.float64))),
        "peg_pose": jsonable(base_env.peg.pose.raw_pose),
        "box_hole_pose": jsonable(base_env.box_hole_pose.raw_pose),
        "goal_pose": jsonable(base_env.goal_pose.raw_pose),
    }


def hole_xyz(base_env: Any) -> np.ndarray:
    return np.asarray(jsonable(base_env.box_hole_pose.p), dtype=np.float64).reshape(-1)[:3]


def shift_box(base_env: Any, delta_xyz: np.ndarray) -> None:
    current = base_env.box.pose
    p = current.p + torch.as_tensor(delta_xyz, dtype=current.p.dtype, device=current.p.device).reshape(1, 3)
    base_env.box.set_pose(Pose.create_from_pq(p=p, q=current.q))


def write_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)


def annotate_existing_video(src: Path, dst: Path, lines: list[str], fps: int) -> bool:
    if not src.exists() or src.stat().st_size <= 0:
        return False
    reader = imageio.get_reader(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        meta = reader.get_meta_data()
        src_fps = int(round(float(meta.get("fps", fps) or fps)))
    except Exception:
        src_fps = fps
    with imageio.get_writer(dst, fps=src_fps, codec="libx264", macro_block_size=1) as writer:
        try:
            total = int(reader.count_frames())
        except Exception:
            total = 0
        for idx, frame in enumerate(reader):
            overlay = lines if total and idx >= max(0, total - 8) else lines[:2]
            writer.append_data(annotate(as_rgb_frame(frame), overlay))
    reader.close()
    return True


def find_phase02_cosmos(args: argparse.Namespace) -> dict[str, Any]:
    phase02 = Path(args.phase02_run).resolve()
    sample = args.phase02_sample
    if not sample:
        candidates = sorted((phase02 / "cosmos_outputs").glob("*/vision.mp4"))
        if not candidates:
            raise FileNotFoundError(f"no Cosmos vision.mp4 under {phase02 / 'cosmos_outputs'}")
        vision = candidates[0]
        sample = vision.parent.name
    else:
        vision = phase02 / "cosmos_outputs" / sample / "vision.mp4"
    sample_args = vision.parent / "sample_args.json"
    sample_outputs = vision.parent / "sample_outputs.json"
    chart = phase02 / "state_audit_rgb" / sample / "task_state_chart.csv"
    trigger_frame = None
    max_delta = 0.0
    if chart.exists():
        with chart.open() as f:
            for row in csv.DictReader(f):
                delta = np.asarray(
                    [
                        float(row.get("hole_delta_x") or 0.0),
                        float(row.get("hole_delta_y") or 0.0),
                        float(row.get("hole_delta_z") or 0.0),
                    ],
                    dtype=np.float64,
                )
                norm = float(np.linalg.norm(delta))
                max_delta = max(max_delta, norm)
                if trigger_frame is None and norm >= float(args.phase02_trigger_delta):
                    trigger_frame = int(row["frame"])
    return {
        "phase02_run": str(phase02),
        "sample": sample,
        "vision_mp4": str(vision),
        "vision_mp4_exists": bool(vision.exists() and vision.stat().st_size > 0),
        "sample_args_json": str(sample_args),
        "sample_outputs_json": str(sample_outputs),
        "task_state_chart_csv": str(chart),
        "task_state_chart_exists": bool(chart.exists()),
        "chart_trigger_frame": trigger_frame,
        "chart_max_hole_delta": max_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase02-run", required=True)
    parser.add_argument("--phase02-sample", default="")
    parser.add_argument("--env-id", default="PegInsertionSide-v1")
    parser.add_argument("--control-mode", default="pd_ee_delta_pose")
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-motion-step", type=int, default=84)
    parser.add_argument("--target-motion-y", type=float, default=0.025)
    parser.add_argument("--motion-detect-threshold", type=float, default=0.003)
    parser.add_argument("--phase02-trigger-delta", type=float, default=0.003)
    parser.add_argument("--max-prefix-steps", type=int, default=140)
    parser.add_argument("--allow-no-motion-trigger", action="store_true")
    parser.add_argument("--render-timeout-s", type=int, default=20)
    parser.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    require_allocation()
    event("start", output_dir=args.output_dir, slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "oracle_review_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    record_video_dir = out_dir / "record_episode_videos"

    event("before_find_phase02_cosmos")
    cosmos = find_phase02_cosmos(args)
    event("after_find_phase02_cosmos", cosmos=cosmos)
    if not cosmos["vision_mp4_exists"]:
        raise FileNotFoundError(f"missing Cosmos RGB output: {cosmos['vision_mp4']}")

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    event("before_torch_load", device=str(device), ckpt_path=args.ckpt_path)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    event("after_torch_load")
    train_args = build_train_args(ckpt.get("args", {}), args)
    ms_train.args = train_args
    ms_train.device = device

    event("before_make_env")
    envs = make_env(train_args, record_video_dir if args.record_video else None)
    event("after_make_env")
    event("before_agent_init")
    agent = ms_train.Agent(envs, train_args).to(device)
    state_key = "ema_agent" if args.use_ema else "agent"
    agent.load_state_dict(ckpt[state_key])
    agent.eval()
    event("after_agent_init", state_key=state_key)

    frames: list[np.ndarray] = []
    action_trace: list[dict[str, Any]] = []
    trigger_frame = None
    target_motion_applied = False
    before_oracle = None
    after_oracle = None
    oracle_jump_distance = None
    render_ok = True

    try:
        event("before_env_reset", seed=args.seed)
        obs, reset_info = envs.reset(seed=args.seed)
        event("after_env_reset")
        base_env = envs.envs[0].unwrapped
        initial_hole = hole_xyz(base_env)
        previous_hole = initial_hole.copy()
        initial_eval = eval_state(base_env)
        render_ok = append_frame(frames, envs.envs[0], ["Phase03 Oracle", "reset"], args.render_timeout_s) and render_ok

        with torch.no_grad():
            for chunk_idx in range(train_args.max_episode_steps):
                event("before_dp_action", chunk_idx=chunk_idx, env_step=len(action_trace))
                obs_tensor = common.to_tensor(obs, device)
                action_seq = agent.get_action(obs_tensor).detach().cpu().numpy()
                event("after_dp_action", chunk_idx=chunk_idx, action_shape=list(action_seq.shape))
                for action_idx in range(action_seq.shape[1]):
                    step_idx = len(action_trace)
                    if (not target_motion_applied) and step_idx >= int(args.target_motion_step):
                        event("before_target_motion", step_idx=step_idx)
                        shift_box(base_env, np.asarray([0.0, args.target_motion_y, 0.0], dtype=np.float64))
                        target_motion_applied = True
                        event("after_target_motion", step_idx=step_idx)

                    action = action_seq[:, action_idx]
                    event("before_env_step", step_idx=step_idx, action_idx=action_idx)
                    obs, reward, terminated, truncated, info = envs.step(action)
                    event("after_env_step", step_idx=step_idx)
                    current_hole = hole_xyz(base_env)
                    hole_delta_from_start = float(np.linalg.norm(current_hole - initial_hole))
                    hole_delta_step = float(np.linalg.norm(current_hole - previous_hole))
                    live_eval = eval_state(base_env)
                    row = {
                        "env_step": step_idx,
                        "chunk_idx": chunk_idx,
                        "action_idx": action_idx,
                        "action": action.reshape(-1).astype(float).tolist(),
                        "reward": jsonable(reward),
                        "terminated": jsonable(terminated),
                        "truncated": jsonable(truncated),
                        "target_motion_applied": bool(target_motion_applied),
                        "hole_xyz": current_hole.astype(float).tolist(),
                        "hole_delta_from_start": hole_delta_from_start,
                        "hole_delta_step": hole_delta_step,
                        "live_eval": live_eval,
                    }
                    action_trace.append(row)
                    previous_hole = current_hole.copy()

                    if step_idx % 10 == 0 or trigger_frame is None:
                        render_ok = (
                            append_frame(
                                frames,
                                envs.envs[0],
                                [
                                    "Phase03 Oracle",
                                    f"step={step_idx}",
                                    f"hole_delta={hole_delta_from_start:.4f}",
                                    f"peg_l2={live_eval['peg_head_l2']:.4f}",
                                ],
                                args.render_timeout_s,
                            )
                            and render_ok
                        )

                    if trigger_frame is None and hole_delta_from_start >= float(args.motion_detect_threshold):
                        trigger_frame = step_idx
                        before_oracle = live_eval
                        event("motion_trigger", trigger_frame=trigger_frame, before_oracle=before_oracle)
                        render_ok = (
                            append_frame(
                                frames,
                                envs.envs[0],
                                [
                                    "MOTION TRIGGER",
                                    f"frame={trigger_frame}",
                                    f"peg={before_oracle['peg_head_at_hole']}",
                                ],
                                args.render_timeout_s,
                            )
                            and render_ok
                        )
                        break

                    done = bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0])
                    if done:
                        break
                if trigger_frame is not None or len(action_trace) >= int(args.max_prefix_steps):
                    break

        if trigger_frame is None and bool(args.allow_no_motion_trigger):
            trigger_frame = len(action_trace) - 1 if action_trace else 0
            before_oracle = eval_state(base_env)

        oracle_applied = trigger_frame is not None
        if oracle_applied:
            before_oracle = before_oracle or eval_state(base_env)
            before_peg_pose = np.asarray(before_oracle["peg_pose"], dtype=np.float64).reshape(-1)[:3]
            event("oracle_decision_no_peg_teleport", before_oracle=before_oracle)
            if args.record_video and action_trace:
                record_action = np.asarray(action_trace[-1]["action"], dtype=np.float32).reshape(1, -1)
                obs, reward, terminated, truncated, info = envs.step(record_action)
                after_oracle = eval_state(base_env)
                after_peg_pose = np.asarray(after_oracle["peg_pose"], dtype=np.float64).reshape(-1)[:3]
                oracle_jump_distance = float(np.linalg.norm(after_peg_pose - before_peg_pose))
                event(
                    "post_oracle_decision_action_step",
                    reward=jsonable(reward),
                    terminated=jsonable(terminated),
                    truncated=jsonable(truncated),
                    info=jsonable(info),
                    after_oracle=after_oracle,
                    oracle_jump_distance=oracle_jump_distance,
                )
            else:
                after_oracle = before_oracle
                oracle_jump_distance = 0.0
            render_ok = (
                append_frame(
                    frames,
                    envs.envs[0],
                    [
                        "ORACLE FINAL-SEAT",
                        f"jump={oracle_jump_distance:.4f}m",
                        f"after={after_oracle['peg_head_at_hole']}",
                    ],
                    args.render_timeout_s,
                )
                and render_ok
            )

        recorded_videos: list[str] = []
        if args.record_video:
            envs.close()
            recorded_videos = [str(path) for path in sorted(record_video_dir.glob("*.mp4"))]
        video_path = out_dir / "oracle_moment_annotated.mp4"
        if frames:
            write_video(frames, video_path, args.fps)
        elif recorded_videos:
            annotate_existing_video(
                Path(recorded_videos[0]),
                video_path,
                [
                    "Phase03 Oracle diagnostic",
                    "method_evidence_allowed=false",
                    f"trigger={trigger_frame}",
                    f"oracle_jump={oracle_jump_distance}",
                    "oracle decision only; no peg set_pose final-seat",
                ],
                args.fps,
            )
        for idx, frame in enumerate(frames):
            Image.fromarray(frame).save(frames_dir / f"frame_{idx:05d}.png")

        summary = {
            "schema": "phase03_oracle_boundary_probe_v1",
            "phase": "03_oracle",
            "evidence_type": "oracle_upper_bound_diagnostic",
            "method_evidence_allowed": False,
            "physical_insertion_success_claimed": False,
            "oracle_state_intervention_used": False,
            "oracle_peg_state_intervention_used": False,
            "oracle_set_pose_used": False,
            "target_motion_state_intervention_used": bool(target_motion_applied),
            "forbidden_state_intervention_used_for_method_success": False,
            "node": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "ckpt_path": str(Path(args.ckpt_path).resolve()),
            "state_key": state_key,
            "output_dir": str(out_dir),
            "cosmos": cosmos,
            "reset_info": jsonable(reset_info),
            "initial_eval": initial_eval,
            "target_motion_step": int(args.target_motion_step),
            "target_motion_y": float(args.target_motion_y),
            "motion_detect_threshold": float(args.motion_detect_threshold),
            "target_motion_trigger_frame": trigger_frame,
            "dp_prefix_steps": len(action_trace),
            "before_oracle": before_oracle,
            "after_oracle": after_oracle,
            "oracle_jump_distance": oracle_jump_distance,
            "annotated_video": str(video_path) if video_path.exists() else None,
            "annotated_video_written": bool(frames or video_path.exists()),
            "record_episode_video_dir": str(record_video_dir) if args.record_video else None,
            "record_episode_videos": recorded_videos,
            "render_ok": bool((render_ok and frames) or recorded_videos),
            "render_disabled": bool(int(args.render_timeout_s) <= 0),
            "review_frames_dir": str(frames_dir),
            "classification": (
                "oracle_boundary_diagnostic_complete_not_method_success"
                if oracle_applied
                else "blocked_no_causal_target_motion_trigger_no_oracle_applied"
            ),
            "notes": [
                "DP prefix uses the active checkpoint and pd_ee_delta_pose actions.",
                "RGB Cosmos evidence is referenced from the active Phase02 run.",
                "Oracle records a boundary / decision point only and does not call peg.set_pose.",
                "This diagnostic must not be reported as deployed method success or physical controller success.",
            ],
        }
        (out_dir / "action_trace.json").write_text(json.dumps(action_trace, indent=2, sort_keys=True) + "\n")
        (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n")
        (out_dir / "classification.txt").write_text(
            "\n".join(
                [
                    f"phase03_status={summary['classification']}",
                    "method_evidence_allowed=false",
                    "physical_insertion_success=false",
                    "oracle_state_intervention_used=false",
                    "oracle_peg_state_intervention_used=false",
                    "oracle_set_pose_used=false",
                    f"target_motion_trigger_frame={trigger_frame}",
                ]
            )
            + "\n"
        )
        print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    finally:
        try:
            envs.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
