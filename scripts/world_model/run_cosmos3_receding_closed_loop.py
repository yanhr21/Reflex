#!/usr/bin/env python3
"""Preflight entry point for Cosmos3 receding closed-loop control.

The live controller is intentionally guarded: this script must run inside a
Slurm compute step and refuses to touch the simulator unless the selected
Cosmos3 eval root passes the strict artifact/readout/visual gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--condition-root", required=True)
    parser.add_argument("--dp-checkpoint", required=True)
    parser.add_argument("--dp-manifest", required=True)
    parser.add_argument(
        "--dp-state-key",
        choices=("ema_agent", "agent"),
        default="ema_agent",
        help="State-dict key to require for the frozen static DP policy.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--readout-subdir", default="task_state_readout_v7_733")
    parser.add_argument(
        "--visual-review-status",
        choices=("missing", "pass", "fail"),
        default="missing",
    )
    parser.add_argument("--visual-review-note", default="")
    parser.add_argument("--expected-video-frames", type=int, default=301)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--expected-action-dim", type=int, default=32)
    parser.add_argument("--robot-action-dim", type=int, default=7)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--action-exec-horizon", type=int, default=8)
    parser.add_argument(
        "--dp-resume-horizon",
        type=int,
        default=8,
        help="After the Cosmos action chunk, execute up to this many frozen-DP actions for handoff smoke.",
    )
    parser.add_argument("--action-preview-sample-index", type=int, default=0)
    parser.add_argument(
        "--capture-live-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture a short before/chunk/resume RGB video in smoke mode.",
    )
    parser.add_argument("--live-video-fps", type=int, default=10)
    parser.add_argument(
        "--clip-live-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip actions to the ManiSkill action space before env.step, while recording the pre-clip violation.",
    )
    parser.add_argument("--mode", choices=("preflight", "smoke"), default="preflight")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def jsonable(value: Any) -> Any:
    try:
        import numpy as np
        import torch
    except Exception:  # pragma: no cover - best effort for optional deps
        np = None  # type: ignore[assignment]
        torch = None  # type: ignore[assignment]

    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def require_compute_step() -> None:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    step_id = os.environ.get("SLURM_STEP_ID", "")
    if not job_id or not step_id or step_id == "extern":
        raise SystemExit(
            "refusing_login_node_execution=true\n"
            "reason=Run this entry point only inside a compute-node srun step."
        )
    ntasks = int(os.environ.get("SLURM_NTASKS", "1") or "1")
    procid = int(os.environ.get("SLURM_PROCID", "0") or "0")
    if ntasks != 1 or procid != 0:
        raise SystemExit(
            "refusing_multi_task_execution=true\n"
            "reason=Closed-loop control must run as exactly one Slurm task."
        )


def assert_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")


def assert_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"missing {label}: {path}")


def check_dp_manifest(dp_manifest: Path, args: argparse.Namespace) -> dict[str, Any]:
    data = read_json(dp_manifest)
    dp_args = data.get("args") or {}
    expected = {
        "env_id": "PegInsertionSide-v1",
        "control_mode": "pd_ee_delta_pose",
        "obs_horizon": 2,
        "act_horizon": 8,
        "max_episode_steps": args.max_episode_steps,
    }
    mismatches = {
        key: {"expected": value, "actual": dp_args.get(key)}
        for key, value in expected.items()
        if dp_args.get(key) != value
    }
    if args.action_exec_horizon > int(dp_args.get("act_horizon", 0) or 0):
        mismatches["action_exec_horizon"] = {
            "expected": f"<= {dp_args.get('act_horizon')}",
            "actual": args.action_exec_horizon,
        }
    return {
        "path": str(dp_manifest),
        "expected": expected,
        "mismatches": mismatches,
        "ok": not mismatches,
    }


def check_dp_checkpoint(dp_checkpoint: Path, dp_manifest: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Validate the frozen DP artifact without constructing the simulator.

    This keeps the closed-loop entrypoint tied to the real ManiSkill state-DP
    checkpoint format used by scripts/training/eval_dp_state.py, while avoiding
    any env construction before the Cosmos visual/readout gate passes.
    """
    import torch

    manifest = read_json(dp_manifest)
    manifest_args = manifest.get("args") or {}
    try:
        ckpt = torch.load(dp_checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(dp_checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict):
        return {
            "path": str(dp_checkpoint),
            "ok": False,
            "error": f"checkpoint_load_returned_{type(ckpt).__name__}",
        }

    required_keys = ["args", "agent", args.dp_state_key, "iteration"]
    missing_keys = [key for key in required_keys if key not in ckpt]
    ckpt_args = ckpt.get("args") if isinstance(ckpt.get("args"), dict) else {}
    expected_args = {
        "env_id": "PegInsertionSide-v1",
        "control_mode": "pd_ee_delta_pose",
        "obs_horizon": 2,
        "act_horizon": 8,
        "max_episode_steps": args.max_episode_steps,
        "pred_horizon": 16,
    }
    arg_mismatches = {
        key: {"expected": value, "actual": ckpt_args.get(key)}
        for key, value in expected_args.items()
        if ckpt_args.get(key) != value
    }
    manifest_arg_mismatches = {
        key: {
            "manifest": manifest_args.get(key),
            "checkpoint": ckpt_args.get(key),
        }
        for key in expected_args
        if manifest_args.get(key) != ckpt_args.get(key)
    }
    state_value = ckpt.get(args.dp_state_key)
    state_nonempty = isinstance(state_value, dict) and bool(state_value)
    sample_tensors: list[dict[str, Any]] = []
    if isinstance(state_value, dict):
        for name, value in state_value.items():
            if hasattr(value, "shape"):
                sample_tensors.append(
                    {
                        "name": str(name),
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                )
            if len(sample_tensors) >= 3:
                break

    return {
        "path": str(dp_checkpoint),
        "state_key": args.dp_state_key,
        "required_keys": required_keys,
        "present_keys": sorted(str(key) for key in ckpt.keys()),
        "missing_keys": missing_keys,
        "iteration": ckpt.get("iteration"),
        "world_size": ckpt.get("world_size"),
        "global_batch_size": ckpt.get("global_batch_size"),
        "per_gpu_batch_size": ckpt.get("per_gpu_batch_size"),
        "expected_args": expected_args,
        "arg_mismatches": arg_mismatches,
        "manifest_arg_mismatches": manifest_arg_mismatches,
        "state_nonempty": state_nonempty,
        "sample_tensors": sample_tensors,
        "ok": (
            not missing_keys
            and not arg_mismatches
            and not manifest_arg_mismatches
            and state_nonempty
        ),
        "boundary": (
            "Checkpoint structure check only. It does not construct ManiSkill "
            "envs, run DP inference, or prove controller success."
        ),
    }


def check_condition_root(condition_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = condition_root / "manifest.json"
    stats_path = condition_root / "normalization_stats.json"
    assert_file(manifest_path, "condition manifest")
    assert_file(stats_path, "normalization stats")
    manifest = read_json(manifest_path)
    stats = read_json(stats_path)
    checks = {
        "num_video_frames": manifest.get("num_video_frames") == args.expected_video_frames,
        "num_action_steps": manifest.get("num_action_steps") == args.expected_action_steps,
        "raw_action_dim": manifest.get("raw_action_dim") == args.expected_action_dim,
        "normalization_type": stats.get("type") == "zscore",
        "normalization_raw_action_dim": stats.get("raw_action_dim") == args.expected_action_dim,
        "normalization_mean_len": len(stats.get("mean") or []) == args.expected_action_dim,
        "normalization_std_len": len(stats.get("std") or []) == args.expected_action_dim,
        "normalization_vector_names_len": len(stats.get("vector_names") or []) == args.expected_action_dim,
        "robot_action_dim": 0 < args.robot_action_dim <= args.expected_action_dim,
    }
    return {
        "path": str(condition_root),
        "manifest_path": str(manifest_path),
        "normalization_stats_path": str(stats_path),
        "checks": checks,
        "ok": all(checks.values()),
    }


def run_gate(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    gate_json = output_root / "closed_loop_gate_verdict.json"
    script = Path(__file__).with_name("check_cosmos3_closed_loop_gate.py")
    cmd = [
        sys.executable,
        str(script),
        "--eval-root",
        str(Path(args.eval_root).resolve()),
        "--readout-subdir",
        args.readout_subdir,
        "--visual-review-status",
        args.visual_review_status,
        "--visual-review-note",
        args.visual_review_note,
        "--output-json",
        str(gate_json),
        "--allow-nonpassing-exit-zero",
    ]
    subprocess.run(cmd, check=True)
    return read_json(gate_json)


def resolve_eval_source_context(eval_root: Path, sample_name: str) -> dict[str, Any]:
    """Recover source trajectory metadata for a generated eval sample.

    The strict eval artifact focuses on generated/reference outputs. The live
    smoke path needs the causal source trajectory too, so recover it through the
    eval input manifest and the original condition JSONL row.
    """
    context: dict[str, Any] = {
        "sample_name": sample_name,
        "eval_input_manifest": str(eval_root / "eval_input_manifest.json"),
        "manifest_sample_found": False,
        "condition_row_found": False,
        "source_h5": None,
    }
    manifest_path = eval_root / "eval_input_manifest.json"
    if not manifest_path.is_file():
        context["missing"] = "eval_input_manifest"
        return context

    manifest = read_json(manifest_path)
    samples = manifest.get("samples") or []
    manifest_sample = next((s for s in samples if s.get("name") == sample_name), None)
    if not isinstance(manifest_sample, dict):
        context["missing"] = "sample_in_eval_input_manifest"
        return context

    context["manifest_sample_found"] = True
    for key in [
        "source_row_uuid",
        "source_uuid",
        "source_jsonl",
        "scenario",
        "prefix_role",
        "prefix_frame_index",
        "condition_prefix_frames",
        "reference_video_path",
        "reference_action_path",
        "state_target_path",
        "task_label_path",
    ]:
        if key in manifest_sample:
            context[key] = manifest_sample[key]

    source_jsonl = manifest_sample.get("source_jsonl")
    source_row_uuid = manifest_sample.get("source_row_uuid")
    if not source_jsonl or not source_row_uuid:
        context["missing"] = "source_jsonl_or_source_row_uuid"
        return context

    source_path = Path(str(source_jsonl))
    if not source_path.is_file():
        context["missing"] = "source_jsonl_file"
        return context

    with source_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("uuid") != source_row_uuid:
                continue
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            context["condition_row_found"] = True
            context["source_h5"] = row.get("source_h5") or metadata.get("source_h5")
            context["condition_row"] = {
                "uuid": row.get("uuid"),
                "source_uuid": row.get("source_uuid"),
                "split": row.get("split"),
                "scenario": row.get("scenario") or metadata.get("scenario"),
                "prefix_role": row.get("prefix_role"),
                "prefix_frame_index": row.get("prefix_frame_index"),
                "condition_prefix_frames": row.get("condition_prefix_frames"),
                "rgb_video": row.get("rgb_video"),
                "action_path": row.get("action_path"),
                "state_target_path": row.get("state_target_path"),
                "task_label_path": row.get("task_label_path"),
                "task_summary_path": row.get("task_summary_path"),
                "num_rgb_frames": row.get("num_rgb_frames"),
                "num_action_steps": row.get("num_action_steps"),
                "sidecar_target_mode": row.get("sidecar_target_mode"),
            }
            break

    return context


def check_source_h5_contract(source_context: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    source_h5_value = source_context.get("source_h5")
    result: dict[str, Any] = {
        "source_h5": source_h5_value,
        "ok": False,
        "checks": {},
        "boundary": (
            "Source-H5 structure check only. It does not restore simulator "
            "state or execute actions."
        ),
    }
    if not source_h5_value:
        result["error"] = "missing_source_h5"
        return result

    source_h5 = Path(str(source_h5_value))
    if not source_h5.is_file():
        result["error"] = "source_h5_not_found"
        return result

    try:
        import h5py
    except Exception as exc:
        result["error"] = "h5py_import_failed"
        result["exception"] = repr(exc)
        return result

    source_uuid = str(source_context.get("source_uuid") or "")
    traj_name = "traj_0"
    if "_traj_" in source_uuid:
        traj_index = source_uuid.rsplit("_traj_", 1)[1].split("_", 1)[0]
        if traj_index.isdigit():
            traj_name = f"traj_{traj_index}"

    with h5py.File(source_h5, "r") as h5:
        if traj_name not in h5:
            traj_name = "traj_0" if "traj_0" in h5 else ""
        result["traj_name"] = traj_name
        if not traj_name:
            result["error"] = "missing_traj_group"
            return result
        group = h5[traj_name]

        actions = group.get("actions")
        env_states = group.get("env_states")
        slots = group.get("slots")
        checks = {
            "actions_shape": bool(
                actions is not None
                and tuple(actions.shape) == (args.expected_action_steps, args.robot_action_dim)
            ),
            "env_states_present": env_states is not None,
            "env_state_peg_frames": bool(
                env_states is not None
                and "actors" in env_states
                and "peg" in env_states["actors"]
                and env_states["actors"]["peg"].shape[0] == args.expected_video_frames
            ),
            "env_state_hole_frames": bool(
                env_states is not None
                and "actors" in env_states
                and "box_with_hole" in env_states["actors"]
                and env_states["actors"]["box_with_hole"].shape[0] == args.expected_video_frames
            ),
            "env_state_robot_frames": bool(
                env_states is not None
                and "articulations" in env_states
                and "panda_wristcam" in env_states["articulations"]
                and env_states["articulations"]["panda_wristcam"].shape[0] == args.expected_video_frames
            ),
            "slots_present": slots is not None,
            "slots_tcp_frames": bool(
                slots is not None
                and "tcp_pose" in slots
                and slots["tcp_pose"].shape[0] == args.expected_video_frames
            ),
            "slots_peg_frames": bool(
                slots is not None
                and "peg_pose" in slots
                and slots["peg_pose"].shape[0] == args.expected_video_frames
            ),
            "slots_hole_frames": bool(
                slots is not None
                and "hole_pose" in slots
                and slots["hole_pose"].shape[0] == args.expected_video_frames
            ),
        }
        result["checks"] = checks
        result["shapes"] = {
            "actions": list(actions.shape) if actions is not None else None,
            "env_states/actors/peg": (
                list(env_states["actors"]["peg"].shape)
                if env_states is not None and "actors" in env_states and "peg" in env_states["actors"]
                else None
            ),
            "env_states/actors/box_with_hole": (
                list(env_states["actors"]["box_with_hole"].shape)
                if env_states is not None and "actors" in env_states and "box_with_hole" in env_states["actors"]
                else None
            ),
            "env_states/articulations/panda_wristcam": (
                list(env_states["articulations"]["panda_wristcam"].shape)
                if env_states is not None
                and "articulations" in env_states
                and "panda_wristcam" in env_states["articulations"]
                else None
            ),
        }
        result["ok"] = all(checks.values())
    return result


def _to_float_matrix(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise ValueError("predicted action is not a non-empty list")
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list):
            raise ValueError("predicted action row is not a list")
        matrix.append([float(x) for x in row])
    return matrix


def _column_stats(matrix: list[list[float]]) -> dict[str, Any]:
    flat = [x for row in matrix for x in row]
    finite = all(math.isfinite(x) for x in flat)
    if not flat:
        return {"finite": finite, "num_values": 0}
    return {
        "finite": finite,
        "num_values": len(flat),
        "min": min(flat),
        "max": max(flat),
        "mean_abs": sum(abs(x) for x in flat) / float(len(flat)),
        "max_abs": max(abs(x) for x in flat),
    }


def _parse_seed_from_text(text: str) -> int | None:
    match = re.search(r"_seed(\d+)", text)
    if match is None:
        match = re.search(r"seed(\d+)", text)
    return int(match.group(1)) if match is not None else None


def _get_base_env(env: Any) -> Any:
    if hasattr(env, "envs") and env.envs:
        candidate = env.envs[0]
        if hasattr(candidate, "base_env"):
            return candidate.base_env
    if hasattr(env, "base_env"):
        return env.base_env
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    return env


def _live_eval(base_env: Any) -> dict[str, Any]:
    from mani_skill.utils import common

    info = base_env.evaluate()
    return {
        "success": bool(common.to_numpy(info["success"])[0]),
        "peg_head_pos_at_hole": common.to_numpy(info["peg_head_pos_at_hole"])[0].astype(float).tolist(),
    }


def _render_frame(env: Any) -> Any:
    import numpy as np

    frame = env.render()
    while isinstance(frame, (list, tuple)) and frame:
        frame = frame[0]
    if isinstance(frame, dict):
        for key in ("rgb", "color", "Color", "image", "human"):
            if key in frame:
                frame = frame[key]
                break
        else:
            frame = next(iter(frame.values()))
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    arr = np.asarray(frame)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 2, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3 or arr.shape[-1] not in (1, 2, 3, 4):
        raise ValueError(f"Unsupported live render frame shape for video: {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _action_space_bounds(env: Any, robot_action_dim: int) -> tuple[Any, Any]:
    import numpy as np

    space = getattr(env, "single_action_space", None) or getattr(env, "action_space", None)
    if space is None or not hasattr(space, "low") or not hasattr(space, "high"):
        return None, None
    low = np.asarray(space.low, dtype=np.float32).reshape(-1)[:robot_action_dim]
    high = np.asarray(space.high, dtype=np.float32).reshape(-1)[:robot_action_dim]
    if low.shape[0] != robot_action_dim or high.shape[0] != robot_action_dim:
        return None, None
    return low, high


def _prepare_step_action(action: Any, low: Any, high: Any, clip: bool) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    raw = np.asarray(action, dtype=np.float32).reshape(-1)
    clipped = raw.copy()
    violation = 0.0
    within = True
    if low is not None and high is not None:
        below = np.maximum(low - raw, 0.0)
        above = np.maximum(raw - high, 0.0)
        violation = float(max(float(below.max(initial=0.0)), float(above.max(initial=0.0))))
        within = violation <= 1e-6
        if clip:
            clipped = np.clip(raw, low, high)
    return clipped[None, :], {
        "raw": raw.astype(float).tolist(),
        "executed": clipped.astype(float).tolist(),
        "within_action_space": bool(within),
        "max_action_space_violation": violation,
        "clipped": bool(clip and not within),
    }


def _import_live_control_stack(root: Path) -> dict[str, Any]:
    dp_root = root / "deps/ManiSkill_clean/examples/baselines/diffusion_policy"
    training_root = root / "scripts/training"
    for path in (str(dp_root), str(training_root), str(root / "deps/ManiSkill_clean")):
        if path not in sys.path:
            sys.path.insert(0, path)

    import mani_skill.envs  # noqa: F401
    from diffusion_policy.make_env import make_eval_envs
    from mani_skill.trajectory import utils as trajectory_utils
    from mani_skill.utils import common
    import torch
    import train as ms_train

    return {
        "make_eval_envs": make_eval_envs,
        "trajectory_utils": trajectory_utils,
        "common": common,
        "torch": torch,
        "ms_train": ms_train,
    }


def _make_live_env(stack: dict[str, Any], dp_manifest: Path, args: argparse.Namespace) -> Any:
    manifest = read_json(dp_manifest)
    dp_args = manifest.get("args") or {}
    env_kwargs = dict(
        control_mode=dp_args.get("control_mode", "pd_ee_delta_pose"),
        reward_mode="sparse",
        obs_mode="state",
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default"),
        max_episode_steps=args.max_episode_steps,
    )
    return stack["make_eval_envs"](
        dp_args.get("env_id", "PegInsertionSide-v1"),
        1,
        dp_args.get("sim_backend", "physx_cpu"),
        env_kwargs,
        dict(obs_horizon=int(dp_args.get("obs_horizon", 2))),
        video_dir=None,
    )


def _load_dp_agent(env: Any, stack: dict[str, Any], args: argparse.Namespace, device: Any) -> tuple[Any, Any]:
    torch = stack["torch"]
    ms_train = stack["ms_train"]
    ckpt = torch.load(args.dp_checkpoint, map_location=device)
    train_args = dict(ckpt.get("args") or {})
    train_args.update(
        {
            "env_id": train_args.get("env_id", "PegInsertionSide-v1"),
            "control_mode": train_args.get("control_mode", "pd_ee_delta_pose"),
            "sim_backend": train_args.get("sim_backend", "physx_cpu"),
            "obs_horizon": int(train_args.get("obs_horizon", 2)),
            "act_horizon": int(train_args.get("act_horizon", 8)),
            "pred_horizon": int(train_args.get("pred_horizon", 16)),
            "diffusion_step_embed_dim": int(train_args.get("diffusion_step_embed_dim", 64)),
            "unet_dims": list(train_args.get("unet_dims", [64, 128, 256])),
            "n_groups": int(train_args.get("n_groups", 8)),
        }
    )
    dp_args = SimpleNamespace(**train_args)
    ms_train.args = dp_args
    ms_train.device = device
    agent = ms_train.Agent(env, dp_args).to(device)
    state = ckpt.get(args.dp_state_key)
    if state is None:
        raise KeyError(f"checkpoint {args.dp_checkpoint} missing state key {args.dp_state_key!r}")
    agent.load_state_dict(state)
    agent.eval()
    return agent, dp_args


def run_live_smoke(
    *,
    output_root: Path,
    action_chunk_preview: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run a gated short live smoke.

    This restores the real simulator to the selected source prefix frame and
    executes only a short predicted robot-action chunk, then optionally lets the
    frozen DP act for one short horizon. It does not write object states into
    the simulator after restore and does not execute a 300-step open loop.
    """
    import h5py
    import imageio.v2 as imageio
    import numpy as np

    root = Path(__file__).resolve().parents[2]
    stack = _import_live_control_stack(root)
    torch = stack["torch"]
    common = stack["common"]
    trajectory_utils = stack["trajectory_utils"]

    source_context = action_chunk_preview.get("source_context") or {}
    source_h5_value = source_context.get("source_h5")
    if not source_h5_value:
        raise ValueError("live smoke requires source_h5 in action chunk preview")
    source_h5 = Path(str(source_h5_value))
    source_h5_contract = action_chunk_preview.get("source_h5_contract") or {}
    traj_name = str(source_h5_contract.get("traj_name") or "traj_0")
    chunk_start = int(action_chunk_preview["chunk_start"])
    chunk_end = int(action_chunk_preview["chunk_end_exclusive"])
    cosmos_chunk = action_chunk_preview.get("denormalized_robot_action_chunk") or []
    if not cosmos_chunk:
        raise ValueError("live smoke requires a non-empty denormalized robot-action chunk")

    seed_text = " ".join(
        str(x)
        for x in [
            source_context.get("source_uuid"),
            source_context.get("source_row_uuid"),
            action_chunk_preview.get("sample_name"),
        ]
        if x
    )
    reset_seed = _parse_seed_from_text(seed_text) or 0

    env = _make_live_env(stack, Path(args.dp_manifest), args)
    frames: list[Any] = []
    cosmos_steps: list[dict[str, Any]] = []
    dp_steps: list[dict[str, Any]] = []
    video_path = output_root / "live_smoke_short_chunk.mp4"
    try:
        with h5py.File(source_h5, "r") as h5:
            if traj_name not in h5:
                traj_name = "traj_0" if "traj_0" in h5 else next(iter(h5.keys()))
            group = h5[traj_name]
            env_states = trajectory_utils.dict_to_list_of_dicts(group["env_states"])
            source_slots = group.get("slots")
            reset_frame = max(0, min(chunk_start, len(env_states) - 1))
            reference_end_frame = max(0, min(chunk_end, len(env_states) - 1))
            reference_summary = {}
            if source_slots is not None:
                reference_summary = {
                    "reference_end_frame": reference_end_frame,
                    "reference_inserted": bool(source_slots["inserted"][reference_end_frame])
                    if "inserted" in source_slots
                    else None,
                    "reference_peg_head_at_hole": source_slots["peg_head_at_hole"][reference_end_frame].astype(float).tolist()
                    if "peg_head_at_hole" in source_slots
                    else None,
                    "reference_hole_pose": source_slots["hole_pose"][reference_end_frame].astype(float).tolist()
                    if "hole_pose" in source_slots
                    else None,
                    "reference_peg_pose": source_slots["peg_pose"][reference_end_frame].astype(float).tolist()
                    if "peg_pose" in source_slots
                    else None,
                    "reference_tcp_pose": source_slots["tcp_pose"][reference_end_frame].astype(float).tolist()
                    if "tcp_pose" in source_slots
                    else None,
                }

        obs, _ = env.reset(seed=reset_seed)
        base_env = _get_base_env(env)
        base_env.set_state_dict(env_states[reset_frame])
        before_eval = _live_eval(base_env)
        if args.capture_live_video:
            frames.append(_render_frame(env))

        low, high = _action_space_bounds(env, args.robot_action_dim)
        for local_i, action in enumerate(cosmos_chunk):
            step_action, action_record = _prepare_step_action(
                action,
                low,
                high,
                bool(args.clip_live_actions),
            )
            obs, reward, terminated, truncated, info = env.step(step_action)
            live = _live_eval(base_env)
            cosmos_steps.append(
                {
                    "local_step": local_i,
                    "action_index": chunk_start + local_i,
                    "reward": jsonable(reward),
                    "terminated": jsonable(terminated),
                    "truncated": jsonable(truncated),
                    "action": action_record,
                    "live_eval": live,
                }
            )
            if args.capture_live_video:
                frames.append(_render_frame(env))
            if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                break

        after_cosmos_eval = _live_eval(base_env)

        dp_resume_horizon = max(0, int(args.dp_resume_horizon))
        dp_agent_loaded = False
        if dp_resume_horizon > 0:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            agent, dp_args = _load_dp_agent(env, stack, args, device)
            dp_agent_loaded = True
            dp_call_index = 0
            while len(dp_steps) < dp_resume_horizon:
                obs_tensor = common.to_tensor(obs, device)
                with torch.no_grad():
                    action_seq = agent.get_action(obs_tensor)
                if getattr(dp_args, "sim_backend", "physx_cpu") == "physx_cpu":
                    action_seq_np = action_seq.detach().cpu().numpy()
                else:
                    action_seq_np = action_seq
                act_horizon = int(action_seq_np.shape[1])
                if act_horizon <= 0:
                    break
                for chunk_local_i in range(min(dp_resume_horizon - len(dp_steps), act_horizon)):
                    step_action = action_seq_np[:, chunk_local_i]
                    obs, reward, terminated, truncated, info = env.step(step_action)
                    live = _live_eval(base_env)
                    dp_steps.append(
                        {
                            "local_step": len(dp_steps),
                            "dp_call_index": dp_call_index,
                            "chunk_local_step": chunk_local_i,
                            "reward": jsonable(reward),
                            "terminated": jsonable(terminated),
                            "truncated": jsonable(truncated),
                            "live_eval": live,
                        }
                    )
                    if args.capture_live_video:
                        frames.append(_render_frame(env))
                    if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                        break
                if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                    break
                dp_call_index += 1

        final_eval = _live_eval(base_env)
        video_written = False
        if args.capture_live_video and frames:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(video_path, frames, fps=max(1, int(args.live_video_fps)))
            video_written = True

        report = {
            "ok": True,
            "source_h5": str(source_h5),
            "traj_name": traj_name,
            "reset_seed": int(reset_seed),
            "reset_frame": int(reset_frame),
            "chunk_start": int(chunk_start),
            "chunk_end_exclusive": int(chunk_end),
            "cosmos_chunk_steps_requested": len(cosmos_chunk),
            "cosmos_chunk_steps_executed": len(cosmos_steps),
            "dp_resume_horizon_requested": dp_resume_horizon,
            "dp_resume_steps_executed": len(dp_steps),
            "dp_agent_loaded": dp_agent_loaded,
            "before_eval": before_eval,
            "after_cosmos_eval": after_cosmos_eval,
            "final_eval": final_eval,
            "cosmos_steps": cosmos_steps,
            "dp_steps": dp_steps,
            "reference_summary": reference_summary,
            "action_space_low": None if low is None else np.asarray(low, dtype=float).tolist(),
            "action_space_high": None if high is None else np.asarray(high, dtype=float).tolist(),
            "clip_live_actions": bool(args.clip_live_actions),
            "video_path": str(video_path) if video_written else None,
            "boundary": (
                "Gated short live smoke. It starts from a restored source prefix "
                "state, executes only a short predicted chunk plus optional DP "
                "resume, and must be judged by live simulator metrics and video."
            ),
        }
        write_manifest(output_root / "live_smoke_result.json", jsonable(report))
        return report
    finally:
        env.close()


def build_action_chunk_preview(eval_root: Path, condition_root: Path, args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    inspection = read_json(eval_root / "eval_artifact_inspection.json")
    samples = inspection.get("samples") or []
    if not samples:
        raise ValueError("eval artifact inspection has no samples")
    sample_index = max(0, min(args.action_preview_sample_index, len(samples) - 1))
    sample = samples[sample_index]
    sample_name = sample["name"]
    source_context = resolve_eval_source_context(eval_root, sample_name)
    source_h5_contract = check_source_h5_contract(source_context, args)
    sample_outputs = read_json(eval_root / "inference" / sample_name / "sample_outputs.json")
    outputs = sample_outputs.get("outputs") or []
    content = (outputs[0].get("content") if outputs else None) or {}
    action = _to_float_matrix(content.get("action"))
    stats = read_json(condition_root / "normalization_stats.json")
    mean = [float(x) for x in stats["mean"]]
    std = [float(x) for x in stats["std"]]
    vector_names = list(stats.get("vector_names") or [])

    shape_ok = (
        len(action) == args.expected_action_steps
        and all(len(row) == args.expected_action_dim for row in action)
    )
    future_start = int(
        sample.get("action_metrics", {}).get(
            "future_start_action_index",
            sample.get("action_metrics", {}).get("prefix_end_action_index_exclusive", sample.get("prefix_frame_index", 0)),
        )
    )
    future_start = max(0, min(future_start, args.expected_action_steps))
    chunk_end = min(args.expected_action_steps, future_start + args.action_exec_horizon)

    normalized_chunk = [
        row[: args.robot_action_dim]
        for row in action[future_start:chunk_end]
    ]
    denormalized_chunk = []
    for row in normalized_chunk:
        denormalized_chunk.append(
            [
                row[i] * std[i] + mean[i]
                for i in range(args.robot_action_dim)
            ]
        )

    preview = {
        "sample_index": sample_index,
        "sample_name": sample_name,
        "source_context": source_context,
        "source_h5_contract": source_h5_contract,
        "prefix_role": sample.get("prefix_role"),
        "scenario": sample.get("scenario"),
        "prefix_frame_index": sample.get("prefix_frame_index"),
        "future_start_action_index": future_start,
        "action_exec_horizon": args.action_exec_horizon,
        "chunk_start": future_start,
        "chunk_end_exclusive": chunk_end,
        "chunk_steps": chunk_end - future_start,
        "expected_action_shape": [args.expected_action_steps, args.expected_action_dim],
        "predicted_action_shape": [len(action), len(action[0]) if action else 0],
        "shape_ok": shape_ok,
        "robot_action_dim": args.robot_action_dim,
        "robot_action_vector_names": vector_names[: args.robot_action_dim],
        "normalized_robot_action_stats": _column_stats(normalized_chunk),
        "denormalized_robot_action_stats": _column_stats(denormalized_chunk),
        "denormalized_robot_action_chunk": denormalized_chunk,
        "boundary": (
            "Action chunk preview only. These actions are not executed and are "
            "not controller evidence; live execution remains gated by visual "
            "review and simulator rollout."
        ),
    }
    preview["ok"] = bool(
        shape_ok
        and preview["chunk_steps"] > 0
        and preview["normalized_robot_action_stats"].get("finite")
        and preview["denormalized_robot_action_stats"].get("finite")
    )
    write_manifest(output_root / "candidate_action_chunk_preview.json", preview)
    return preview


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    require_compute_step()

    eval_root = Path(args.eval_root).resolve()
    checkpoint_path = Path(args.checkpoint_path).resolve()
    condition_root = Path(args.condition_root).resolve()
    dp_checkpoint = Path(args.dp_checkpoint).resolve()
    dp_manifest = Path(args.dp_manifest).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    assert_dir(eval_root, "eval root")
    assert_dir(checkpoint_path, "Cosmos checkpoint")
    assert_dir(condition_root, "condition root")
    assert_file(dp_checkpoint, "DP checkpoint")
    assert_file(dp_manifest, "DP manifest")

    dp_contract = check_dp_manifest(dp_manifest, args)
    dp_checkpoint_contract = check_dp_checkpoint(dp_checkpoint, dp_manifest, args)
    condition_contract = check_condition_root(condition_root, args)
    action_chunk_preview = build_action_chunk_preview(eval_root, condition_root, args, output_root)
    gate = run_gate(args, output_root)

    manifest: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "step_id": os.environ.get("SLURM_STEP_ID"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        },
        "mode": args.mode,
        "eval_root": str(eval_root),
        "checkpoint_path": str(checkpoint_path),
        "condition_root": str(condition_root),
        "dp_checkpoint": str(dp_checkpoint),
        "dp_manifest": str(dp_manifest),
        "dp_state_key": args.dp_state_key,
        "output_root": str(output_root),
        "max_episode_steps": args.max_episode_steps,
        "action_exec_horizon": args.action_exec_horizon,
        "dp_resume_horizon": args.dp_resume_horizon,
        "capture_live_video": args.capture_live_video,
        "live_video_fps": args.live_video_fps,
        "clip_live_actions": args.clip_live_actions,
        "action_preview_sample_index": args.action_preview_sample_index,
        "expected_video_frames": args.expected_video_frames,
        "expected_action_steps": args.expected_action_steps,
        "expected_action_dim": args.expected_action_dim,
        "robot_action_dim": args.robot_action_dim,
        "dp_contract": dp_contract,
        "dp_checkpoint_contract": dp_checkpoint_contract,
        "condition_contract": condition_contract,
        "action_chunk_preview": action_chunk_preview,
        "gate": gate,
        "boundary": (
            "Preflight only unless the gate passes. Live simulator success must "
            "come from real env.step rollouts plus video review; generated RGB "
            "or readout-only success is not controller evidence."
        ),
    }
    write_manifest(output_root / "closed_loop_preflight_manifest.json", manifest)

    failures: list[str] = []
    if not dp_contract["ok"]:
        failures.append("dp_contract_failed")
    if not dp_checkpoint_contract["ok"]:
        failures.append("dp_checkpoint_contract_failed")
    if not condition_contract["ok"]:
        failures.append("condition_contract_failed")
    if not action_chunk_preview["ok"]:
        failures.append("action_chunk_preview_failed")
    if not gate.get("closed_loop_allowed"):
        failures.append("closed_loop_gate_blocked")
    if args.mode == "smoke" and not action_chunk_preview.get("source_context", {}).get("source_h5"):
        failures.append("live_source_h5_missing")
    if args.mode == "smoke" and not action_chunk_preview.get("source_h5_contract", {}).get("ok"):
        failures.append("source_h5_contract_failed")

    if failures:
        print(json.dumps({"closed_loop_preflight_ok": False, "failures": failures}, indent=2))
        return 40

    if args.mode == "preflight":
        print(json.dumps({"closed_loop_preflight_ok": True, "mode": "preflight"}, indent=2))
        return 0

    live_smoke = run_live_smoke(
        output_root=output_root,
        action_chunk_preview=action_chunk_preview,
        args=args,
    )
    manifest["live_smoke"] = live_smoke
    write_manifest(output_root / "closed_loop_preflight_manifest.json", jsonable(manifest))
    print(
        json.dumps(
            {
                "closed_loop_preflight_ok": True,
                "live_smoke_started": True,
                "live_smoke_ok": bool(live_smoke.get("ok")),
                "live_smoke_result": str(output_root / "live_smoke_result.json"),
                "video_path": live_smoke.get("video_path"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if live_smoke.get("ok") else 52


if __name__ == "__main__":
    raise SystemExit(main())
