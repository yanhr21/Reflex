#!/usr/bin/env python3
"""Generate fix3 large-motion demos by copying the original DP rollout protocol.

This script is a copy-on-write replacement for the missing 2026-06-03 dynamic
rollout generator. It does not modify the original full1000 dataset, source H5
tree, or generation environment. The defaults come from the
`source_manifest_json` stored in the original 2026-06-06 full1000 RGB H5s.

The key contract is physical acceptance, not postprocessing: the target block
is moved inside the live ManiSkill episode after the DP-controlled peg reaches
the original pre-insertion alignment trigger. A candidate is accepted only if
the live simulator reports final success and the exported slot state is
inserted at the final frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import torch
import tyro

REPO_ROOT = Path(__file__).resolve().parents[2]
DP_ROOT = REPO_ROOT / "deps/ManiSkill_clean/examples/baselines/diffusion_policy"
TRAINING_ROOT = REPO_ROOT / "scripts/training"
sys.path.insert(0, str(DP_ROOT))
sys.path.insert(0, str(TRAINING_ROOT))

import mani_skill.envs  # noqa: E402,F401
from diffusion_policy.make_env import make_eval_envs  # noqa: E402
from mani_skill.utils import common  # noqa: E402
from mani_skill.utils.structs import Pose  # noqa: E402

import train as ms_train  # noqa: E402


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
PEG_PERTURB_SCENARIOS = {"peg_disturb", "peg_drop"}

SCENARIO_PRIORITY_SEEDS = {
    # These are the v7 complete-nine smoke seeds whose rendered 301-frame
    # videos passed all-frame inspection. Do not keep older rejected fast/self-
    # insert seeds here just because they accepted numerically.
    "hole_late_move_stop": (
        1080064,
    ),
    "hole_late_constant": (
        1050118,
    ),
    "hole_late_reverse": (
        1040017,
    ),
    "hole_late_sine": (
        1050127,
    ),
    "hole_late_continuous_insert": (
        1040042,
    ),
    "hole_late_fast_shift": (
        1100041,
    ),
    "none": (
        700107,
    ),
    "peg_drop": (
        705095,
    ),
    "peg_disturb": (
        1051032,
    ),
}


@dataclass
class Args:
    output_root: str = (
        "experiments/world_model_task_rebinding/cosmos3/"
        "fix3_original_protocol_large_motion_dp_v7_complete9_slow_no_self_insert_smoke_20260611"
    )
    paths_file: str = ""
    ckpt_path: str = (
        "experiments/dp_peg1000/run_90201/checkpoints/"
        "best_eval_success_at_end.pt"
    )
    state_key: str = "ema_agent"
    env_id: str = "PegInsertionSide-v1"
    obs_mode: str = "state"
    control_mode: str = "pd_ee_delta_pose"
    sim_backend: str = "physx_cpu"
    reward_mode: str = "sparse"
    render_mode: str = "rgb_array"
    num_demos: int = 9
    max_attempts: int = 240
    seed: int = 1011000
    total_video_frames: int = 301
    total_action_steps: int = 300
    val_fraction: float = 0.1
    overwrite: bool = False
    motion_min_m: float = 0.22
    motion_max_m: float = 0.30
    min_abs_y_motion_m: float = 0.18
    final_x_min: float = -0.075
    final_x_max: float = 0.075
    final_y_min: float = 0.15
    final_y_max: float = 0.45
    min_trigger_step: int = 45
    fallback_trigger_step: int = 90
    force_fallback_step: int = -1
    trigger_x_min: float = -0.14
    trigger_x_max: float = 0.05
    trigger_yz_min: float = 0.0
    trigger_yz_threshold: float = 0.035
    fallback_trigger_yz_threshold: float = 0.060
    trigger_finger_width_max: float = 0.026
    trigger_peg_tcp_dist_max: float = 0.120
    trigger_peg_lift_min_m: float = 0.012
    trigger_grasp_hold_steps: int = 8
    min_trigger_after_grasp_steps: int = 8
    move_stop_steps_min: int = 16
    move_stop_steps_max: int = 32
    constant_steps_min: int = 28
    constant_steps_max: int = 44
    continuous_steps_min: int = 42
    continuous_steps_max: int = 70
    sine_steps_min: int = 28
    sine_steps_max: int = 48
    fast_shift_steps_min: int = 8
    fast_shift_steps_max: int = 16
    reverse_overshoot: float = 1.22
    reverse_after_steps_min: int = 10
    reverse_after_steps_max: int = 18
    reverse_return_steps_min: int = 18
    reverse_return_steps_max: int = 30
    sine_lateral_max_m: float = 0.060
    anti_self_insert_final_yz_min_m: float = 0.055
    anti_self_insert_radius_multiplier: float = 2.4
    anti_self_insert_path_probe_frames: int = 16
    anti_self_insert_min_probe_delta_m: float = 0.080
    resample_final_hole_at_trigger: bool = False
    final_hole_trigger_resample_attempts: int = 128
    self_insert_motion_yz_max_m: float = 0.015
    self_insert_min_target_motion_m: float = 0.080
    self_insert_min_tcp_rebind_m: float = 0.040
    min_insert_after_target_motion_end_steps: int = -1
    target_motion_grasp_guard_steps: int = 4
    peg_disturb_delta_xyz: tuple[float, float, float] = (0.0, -0.04, 0.02)
    peg_drop_delta_y_m: float = -0.04
    accept_requires_live_success: bool = True
    accept_requires_slot_inserted: bool = True
    scenario_sequence: str = ""
    scenario_quotas: str = ""
    scenario_seed_bases: str = ""
    scenario_trigger_yz_min: str = ""
    scenario_min_trigger_step: str = ""
    scenario_fallback_trigger_step: str = ""
    scenario_force_fallback_step: str = ""
    scenario_min_insert_after_target_motion_end_steps: str = ""
    reset_candidate_rng: bool = False
    policy_rng_seed_base: int = -1
    scenario_policy_rng_seed_bases: str = ""
    use_priority_seeds: bool = True
    non_priority_seed_offset: int = 0
    balanced_scenario_quotas: bool = True
    accepted_index_offset: int = 0
    save_reject_log_limit: int = 500
    reject_log_every: int = 10


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


def _smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.clip(x, 0.0, 1.0)
    return arr * arr * (3.0 - 2.0 * arr)


def _seed_candidate_rng(seed: int) -> None:
    seed32 = int(seed) % (2**32 - 1)
    random.seed(seed32)
    np.random.seed(seed32)
    torch.manual_seed(seed32)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed32)


def _read_state_obs(base_env) -> np.ndarray:
    obs = common.to_numpy(base_env.get_obs())
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim == 2:
        return obs[0].astype(np.float32)
    if obs.ndim == 1:
        return obs.astype(np.float32)
    raise RuntimeError(f"unexpected state obs shape {obs.shape}")


def _read_env_state(base_env) -> dict[str, dict[str, np.ndarray]]:
    state = common.to_numpy(base_env.get_state_dict())
    out: dict[str, dict[str, np.ndarray]] = {"actors": {}, "articulations": {}}
    for section in ("actors", "articulations"):
        for key, value in state[section].items():
            out[section][key] = np.asarray(value, dtype=np.float32).copy()
    return out


def _live_eval(base_env) -> dict[str, Any]:
    info = base_env.evaluate()
    return {
        "success": bool(common.to_numpy(info["success"])[0]),
        "peg_head_at_hole": common.to_numpy(info["peg_head_pos_at_hole"])[0].astype(np.float32),
    }


def _set_box_pose(base_env, position: np.ndarray, quat: np.ndarray) -> None:
    p_t = torch.as_tensor(position, device=base_env.device, dtype=base_env.box.pose.p.dtype).view(1, 3)
    q_t = torch.as_tensor(quat, device=base_env.device, dtype=base_env.box.pose.q.dtype).view(1, 4)
    base_env.box.set_pose(Pose.create_from_pq(p_t, q_t))


def _set_peg_pose(base_env, position: np.ndarray, quat: np.ndarray) -> None:
    p_t = torch.as_tensor(position, device=base_env.device, dtype=base_env.peg.pose.p.dtype).view(1, 3)
    q_t = torch.as_tensor(quat, device=base_env.device, dtype=base_env.peg.pose.q.dtype).view(1, 4)
    base_env.peg.set_pose(Pose.create_from_pq(p_t, q_t))


def _box_pose_np(base_env) -> tuple[np.ndarray, np.ndarray]:
    p = common.to_numpy(base_env.box.pose.p)[0].astype(np.float32)
    q = common.to_numpy(base_env.box.pose.q)[0].astype(np.float32)
    return p, q


def _peg_pose_np(base_env) -> tuple[np.ndarray, np.ndarray]:
    p = common.to_numpy(base_env.peg.pose.p)[0].astype(np.float32)
    q = common.to_numpy(base_env.peg.pose.q)[0].astype(np.float32)
    return p, q


def _counterfactual_hole_alignment_for_hole_xyz(
    base_env,
    *,
    initial_hole_xyz: np.ndarray,
    initial_box_xyz: np.ndarray,
    target_hole_xyz: np.ndarray,
) -> np.ndarray:
    """Measure current peg-head alignment if the target were placed at a probe pose."""

    current_box_xyz, current_box_quat = _box_pose_np(base_env)
    final_box_xyz = np.asarray(initial_box_xyz, dtype=np.float32) + (
        np.asarray(target_hole_xyz, dtype=np.float32) - np.asarray(initial_hole_xyz, dtype=np.float32)
    )
    _set_box_pose(base_env, final_box_xyz, current_box_quat)
    try:
        return _live_eval(base_env)["peg_head_at_hole"].astype(np.float32)
    finally:
        _set_box_pose(base_env, current_box_xyz, current_box_quat)


def _counterfactual_final_hole_alignment(
    base_env,
    *,
    initial_hole_xyz: np.ndarray,
    initial_box_xyz: np.ndarray,
    final_hole_xy: np.ndarray,
) -> np.ndarray:
    """Measure current peg-head alignment if the target were placed at its final pose."""

    final_hole_xyz = np.asarray(initial_hole_xyz, dtype=np.float32).copy()
    final_hole_xyz[:2] = np.asarray(final_hole_xy, dtype=np.float32)
    return _counterfactual_hole_alignment_for_hole_xyz(
        base_env,
        initial_hole_xyz=initial_hole_xyz,
        initial_box_xyz=initial_box_xyz,
        target_hole_xyz=final_hole_xyz,
    )


def _counterfactual_min_path_alignment_yz(
    base_env,
    *,
    scenario: str,
    trigger_step: int,
    total_video_frames: int,
    initial_hole_xyz: np.ndarray,
    initial_box_xyz: np.ndarray,
    final_hole_xy: np.ndarray,
    rng_values: dict[str, float],
    num_probes: int,
    min_probe_delta_m: float,
) -> tuple[float, int, np.ndarray]:
    probe_count = max(2, int(num_probes))
    frames = np.unique(
        np.linspace(
            int(trigger_step) + 1,
            int(total_video_frames) - 1,
            num=probe_count,
            dtype=np.int32,
        )
    )
    min_yz = float("inf")
    min_frame = int(frames[0])
    min_head: np.ndarray | None = None
    for frame in frames:
        target_hole_xyz = _target_position(
            scenario,
            int(frame),
            int(trigger_step),
            int(total_video_frames),
            np.asarray(initial_hole_xyz, dtype=np.float32),
            np.asarray(final_hole_xy, dtype=np.float32),
            rng_values,
        )
        if float(np.linalg.norm(target_hole_xyz[:2] - np.asarray(initial_hole_xyz, dtype=np.float32)[:2])) < float(
            min_probe_delta_m
        ):
            continue
        head = _counterfactual_hole_alignment_for_hole_xyz(
            base_env,
            initial_hole_xyz=initial_hole_xyz,
            initial_box_xyz=initial_box_xyz,
            target_hole_xyz=target_hole_xyz,
        )
        yz = float(np.linalg.norm(head[1:3]))
        if yz < min_yz:
            min_yz = yz
            min_frame = int(frame)
            min_head = head.astype(np.float32)
    if min_head is None:
        final_hole_xyz = np.asarray(initial_hole_xyz, dtype=np.float32).copy()
        final_hole_xyz[:2] = np.asarray(final_hole_xy, dtype=np.float32)
        min_head = _counterfactual_hole_alignment_for_hole_xyz(
            base_env,
            initial_hole_xyz=initial_hole_xyz,
            initial_box_xyz=initial_box_xyz,
            target_hole_xyz=final_hole_xyz,
        )
        min_yz = float(np.linalg.norm(min_head[1:3]))
        min_frame = int(total_video_frames) - 1
    return min_yz, min_frame, min_head


def _anti_self_insert_gate_passes(
    *,
    counterfactual_yz: float,
    hole_radius: float,
    args: Args,
) -> tuple[bool, float]:
    threshold = max(
        float(args.anti_self_insert_final_yz_min_m),
        float(args.anti_self_insert_radius_multiplier) * float(hole_radius),
    )
    return float(counterfactual_yz) >= threshold, threshold


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


def _inserted_from_head(head_at_hole: np.ndarray, radius: np.ndarray) -> np.ndarray:
    radius = np.asarray(radius, dtype=np.float32).reshape(-1)
    return (
        (head_at_hole[:, 0] >= -0.015)
        & (np.abs(head_at_hole[:, 1]) <= radius)
        & (np.abs(head_at_hole[:, 2]) <= radius)
    ).astype(bool)


def _stack_env_states(frames: list[dict[str, dict[str, np.ndarray]]]) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {"actors": {}, "articulations": {}}
    for section in ("actors", "articulations"):
        keys = frames[0][section].keys()
        for key in keys:
            out[section][key] = np.stack([frame[section][key] for frame in frames], axis=0).astype(np.float32)
    return out


def _write_dataset(group: h5py.Group, name: str, value: np.ndarray) -> None:
    group.create_dataset(name, data=np.asarray(value), compression="gzip", compression_opts=1)


def _write_dict_group(group: h5py.Group, data: dict[str, Any]) -> None:
    for key, value in data.items():
        if isinstance(value, dict):
            child = group.create_group(str(key))
            _write_dict_group(child, value)
        else:
            _write_dataset(group, str(key), np.asarray(value))


def _stack_obs(obs: np.ndarray) -> np.ndarray:
    prev = np.concatenate([obs[:1], obs[:-1]], axis=0)
    return np.stack([prev, obs], axis=1).astype(np.float32)


def _stable_val(sample_id: str, val_fraction: float) -> bool:
    if val_fraction <= 0:
        return False
    if val_fraction >= 1:
        return True
    import hashlib

    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 10000) < int(round(val_fraction * 10000))


def _attempt_seed_for_scenario(
    *,
    scenario: str,
    base_seed: int,
    global_attempt: int,
    scenario_attempt: int,
    scenario_seed_bases: dict[str, int],
    use_priority_seeds: bool,
    non_priority_seed_offset: int,
) -> int:
    priority = SCENARIO_PRIORITY_SEEDS.get(scenario, ()) if use_priority_seeds else ()
    if use_priority_seeds and scenario_attempt < len(priority):
        return int(priority[scenario_attempt])
    if scenario in scenario_seed_bases:
        base = int(scenario_seed_bases[scenario])
        if not priority:
            return int(base + int(non_priority_seed_offset) + scenario_attempt)
        priority_set = {int(seed) for seed in priority}
        remaining = int(scenario_attempt) - len(priority)
        candidate = base + int(non_priority_seed_offset)
        while True:
            if candidate not in priority_set:
                if remaining <= 0:
                    return int(candidate)
                remaining -= 1
            candidate += 1
    return int(base_seed + global_attempt)


def _scenario_sequence(args: Args) -> tuple[str, ...]:
    if not args.scenario_sequence.strip():
        return SCENARIOS
    scenarios = tuple(item.strip() for item in args.scenario_sequence.split(",") if item.strip())
    unknown = [item for item in scenarios if item not in SCENARIOS]
    if unknown:
        raise ValueError(f"unknown scenarios in --scenario-sequence: {unknown}")
    if not scenarios:
        raise ValueError("--scenario-sequence did not contain any valid scenarios")
    return scenarios


def _parse_scenario_seed_bases(text: str) -> dict[str, int]:
    text = text.strip()
    if not text:
        return {}
    if text.startswith("{"):
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise ValueError("--scenario-seed-bases JSON must be an object")
        items = raw.items()
    else:
        pairs = [item.strip() for item in text.split(",") if item.strip()]
        items = []
        for pair in pairs:
            if "=" not in pair:
                raise ValueError(
                    "--scenario-seed-bases entries must be SCENARIO=BASE, "
                    f"got {pair!r}"
                )
            key, value = pair.split("=", 1)
            items.append((key.strip(), value.strip()))
    out: dict[str, int] = {}
    for key, value in items:
        if key not in SCENARIOS:
            raise ValueError(f"unknown scenario in --scenario-seed-bases: {key!r}")
        out[str(key)] = int(value)
    return out


def _parse_scenario_value_map(text: str, *, flag_name: str, value_type: type) -> dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    if text.startswith("{"):
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise ValueError(f"--{flag_name} JSON must be an object")
        items = raw.items()
    else:
        pairs = [item.strip() for item in text.split(",") if item.strip()]
        items = []
        for pair in pairs:
            if "=" not in pair:
                raise ValueError(
                    f"--{flag_name} entries must be SCENARIO=VALUE, got {pair!r}"
                )
            key, value = pair.split("=", 1)
            items.append((key.strip(), value.strip()))
    out: dict[str, Any] = {}
    for key, value in items:
        if key not in SCENARIOS:
            raise ValueError(f"unknown scenario in --{flag_name}: {key!r}")
        out[str(key)] = value_type(value)
    return out


def _scenario_args(
    args: Args,
    scenario: str,
    *,
    trigger_yz_min_by_scenario: dict[str, float],
    min_trigger_step_by_scenario: dict[str, int],
    fallback_trigger_step_by_scenario: dict[str, int],
    force_fallback_step_by_scenario: dict[str, int],
    min_insert_after_motion_by_scenario: dict[str, int],
) -> Args:
    updates: dict[str, Any] = {}
    if scenario in trigger_yz_min_by_scenario:
        updates["trigger_yz_min"] = float(trigger_yz_min_by_scenario[scenario])
    if scenario in min_trigger_step_by_scenario:
        updates["min_trigger_step"] = int(min_trigger_step_by_scenario[scenario])
    if scenario in fallback_trigger_step_by_scenario:
        updates["fallback_trigger_step"] = int(fallback_trigger_step_by_scenario[scenario])
    if scenario in force_fallback_step_by_scenario:
        updates["force_fallback_step"] = int(force_fallback_step_by_scenario[scenario])
    if scenario in min_insert_after_motion_by_scenario:
        updates["min_insert_after_target_motion_end_steps"] = int(
            min_insert_after_motion_by_scenario[scenario]
        )
    if not updates:
        return args
    return replace(args, **updates)


def _scenario_quotas(
    num_demos: int,
    scenarios: tuple[str, ...],
    balanced: bool,
    explicit_quotas: dict[str, int] | None = None,
) -> dict[str, int]:
    if num_demos < 1:
        raise ValueError("--num-demos must be positive")
    if explicit_quotas:
        unknown = [scenario for scenario in explicit_quotas if scenario not in scenarios]
        if unknown:
            raise ValueError(f"--scenario-quotas contains scenarios outside --scenario-sequence: {unknown}")
        missing = [scenario for scenario in scenarios if scenario not in explicit_quotas]
        if missing:
            raise ValueError(f"--scenario-quotas missing scenarios: {missing}")
        total = sum(int(value) for value in explicit_quotas.values())
        if total != int(num_demos):
            raise ValueError(f"--scenario-quotas sum {total} must equal --num-demos {num_demos}")
        if any(int(value) < 1 for value in explicit_quotas.values()):
            raise ValueError("--scenario-quotas values must be positive")
        return {scenario: int(explicit_quotas[scenario]) for scenario in scenarios}
    if not balanced:
        return {scenario: int(num_demos) for scenario in scenarios}
    base, remainder = divmod(int(num_demos), len(scenarios))
    return {
        scenario: int(base + (1 if idx < remainder else 0))
        for idx, scenario in enumerate(scenarios)
    }


def _next_scenario_with_open_quota(
    *,
    scenarios: tuple[str, ...],
    scenario_accept_counts: dict[str, int],
    scenario_quotas: dict[str, int],
    cursor: int,
) -> tuple[str | None, int]:
    for offset in range(len(scenarios)):
        idx = (int(cursor) + offset) % len(scenarios)
        scenario = scenarios[idx]
        if int(scenario_accept_counts.get(scenario, 0)) < int(scenario_quotas.get(scenario, 0)):
            return scenario, (idx + 1) % len(scenarios)
    return None, int(cursor)


def _sample_final_hole_xy(initial_xy: np.ndarray, rng: np.random.Generator, args: Args) -> np.ndarray | None:
    for _ in range(200):
        final = np.asarray(
            [
                rng.uniform(args.final_x_min, args.final_x_max),
                rng.uniform(args.final_y_min, args.final_y_max),
            ],
            dtype=np.float32,
        )
        delta = final - initial_xy.astype(np.float32)
        norm = float(np.linalg.norm(delta))
        if norm < float(args.motion_min_m) or norm > float(args.motion_max_m):
            continue
        if abs(float(delta[1])) < float(args.min_abs_y_motion_m):
            continue
        return final
    return None


def _target_progress(scenario: str, frame: int, trigger: int, total_frames: int, rng_values: dict[str, float]) -> tuple[float, float]:
    if frame <= trigger:
        return 0.0, 0.0
    t = float(frame - trigger)
    remaining = max(1.0, float(total_frames - 1 - trigger))
    if scenario in {"hole_move_stop", "hole_late_move_stop"}:
        steps = max(1.0, rng_values["move_steps"])
        return float(_smoothstep(t / steps)), 0.0
    if scenario == "hole_late_fast_shift":
        steps = max(1.0, rng_values["move_steps"])
        return float(_smoothstep(t / steps)), 0.0
    if scenario in {"hole_constant", "hole_late_constant"}:
        steps = max(1.0, rng_values["constant_steps"])
        return float(np.clip(t / steps, 0.0, 1.0)), 0.0
    if scenario == "hole_late_continuous_insert":
        steps = max(1.0, rng_values["continuous_steps"])
        return float(np.clip(t / steps, 0.0, 1.0)), 0.0
    if scenario in {"hole_reverse", "hole_late_reverse"}:
        first = max(1.0, float(args_reverse_after_steps(rng_values)))
        if t <= first:
            return float(rng_values["reverse_overshoot"] * _smoothstep(t / first)), 0.0
        return_steps = max(1.0, float(rng_values["reverse_return_steps"]))
        return float(
            rng_values["reverse_overshoot"]
            - (rng_values["reverse_overshoot"] - 1.0) * _smoothstep((t - first) / return_steps)
        ), 0.0
    if scenario == "hole_late_sine":
        steps = max(1.0, rng_values["sine_steps"])
        progress = float(_smoothstep(t / steps))
        lateral = float(rng_values["sine_lateral"] * np.sin(np.pi * 2.0 * progress) * (1.0 - progress))
        return progress, lateral
    raise ValueError(f"unknown scenario {scenario!r}")


def args_reverse_after_steps(rng_values: dict[str, float]) -> float:
    return float(rng_values["reverse_after_steps"])


def _target_position(
    scenario: str,
    frame: int,
    trigger: int,
    total_frames: int,
    initial_xyz: np.ndarray,
    final_xy: np.ndarray,
    rng_values: dict[str, float],
) -> np.ndarray:
    delta_xy = final_xy.astype(np.float32) - initial_xyz[:2].astype(np.float32)
    norm = max(float(np.linalg.norm(delta_xy)), 1e-8)
    unit = delta_xy / norm
    orth = np.asarray([-unit[1], unit[0]], dtype=np.float32)
    progress, lateral = _target_progress(scenario, frame, trigger, total_frames, rng_values)
    xy = initial_xyz[:2].astype(np.float32) + np.asarray(progress * delta_xy + lateral * orth, dtype=np.float32)
    if frame >= total_frames - 1:
        xy = final_xy.astype(np.float32)
    out = initial_xyz.astype(np.float32).copy()
    out[:2] = xy
    return out


def _scenario_rng_values(scenario: str, rng: np.random.Generator, args: Args) -> dict[str, float]:
    return {
        "move_steps": float(
            rng.integers(
                args.fast_shift_steps_min if scenario == "hole_late_fast_shift" else args.move_stop_steps_min,
                (args.fast_shift_steps_max if scenario == "hole_late_fast_shift" else args.move_stop_steps_max) + 1,
            )
        ),
        "constant_steps": float(rng.integers(args.constant_steps_min, args.constant_steps_max + 1)),
        "continuous_steps": float(rng.integers(args.continuous_steps_min, args.continuous_steps_max + 1)),
        "reverse_overshoot": float(args.reverse_overshoot),
        "reverse_after_steps": float(rng.integers(args.reverse_after_steps_min, args.reverse_after_steps_max + 1)),
        "reverse_return_steps": float(rng.integers(args.reverse_return_steps_min, args.reverse_return_steps_max + 1)),
        "sine_steps": float(rng.integers(args.sine_steps_min, args.sine_steps_max + 1)),
        "sine_lateral": float(rng.uniform(-args.sine_lateral_max_m, args.sine_lateral_max_m)),
    }


def _make_env(args: Args):
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        obs_mode=args.obs_mode,
        render_mode=args.render_mode,
        human_render_camera_configs=dict(shader_pack="default"),
        max_episode_steps=args.total_action_steps,
    )
    return make_eval_envs(
        args.env_id,
        1,
        args.sim_backend,
        env_kwargs,
        dict(obs_horizon=2),
        video_dir=None,
    )


def _load_agent(env, args: Args, device: torch.device):
    ckpt = torch.load(args.ckpt_path, map_location=device)
    train_args = dict(ckpt.get("args") or {})
    train_args.update(
        {
            "env_id": args.env_id,
            "control_mode": args.control_mode,
            "sim_backend": args.sim_backend,
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
    state = ckpt.get(args.state_key)
    if state is None:
        raise KeyError(f"checkpoint {args.ckpt_path} missing state key {args.state_key!r}")
    agent.load_state_dict(state)
    agent.eval()
    return agent, dp_args, ckpt


def _run_candidate(
    *,
    env,
    agent,
    dp_args,
    device: torch.device,
    scenario: str,
    attempt_seed: int,
    policy_seed: int | None,
    accepted_index: int,
    args: Args,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if bool(args.reset_candidate_rng):
        _seed_candidate_rng(attempt_seed if policy_seed is None else int(policy_seed))
    rng = np.random.default_rng(attempt_seed)
    obs_stack, _ = env.reset(seed=attempt_seed)
    cpu_env = env.envs[0]
    base_env = cpu_env.base_env

    initial_obs = _read_state_obs(base_env)
    initial_hole = initial_obs[LAYOUT["hole_pose"]].astype(np.float32)
    initial_peg_z = float(initial_obs[LAYOUT["peg_pose"]][2])
    initial_box_xyz, initial_box_quat = _box_pose_np(base_env)
    final_xy = _sample_final_hole_xy(initial_hole[:2], rng, args) if scenario in HOLE_MOTION_SCENARIOS else initial_hole[:2]
    if final_xy is None and scenario in HOLE_MOTION_SCENARIOS:
        return None, {
            "reason": "no_large_final_target_in_bounds",
            "scenario": scenario,
            "attempt_seed": int(attempt_seed),
            "initial_hole_xy": initial_hole[:2].astype(float).tolist(),
        }
    rng_values = _scenario_rng_values(scenario, rng, args)

    obs_frames = [initial_obs.copy()]
    state_frames = [_read_env_state(base_env)]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    policy_query_step: list[bool] = []
    live_heads = [_live_eval(base_env)["peg_head_at_hole"]]
    live_successes = [_live_eval(base_env)["success"]]

    prev_obs = initial_obs.copy()
    cur_obs = initial_obs.copy()
    pending_actions: list[np.ndarray] = []
    triggered = False
    trigger_step = -1
    trigger_reason = ""
    action_step_triggered = np.zeros((args.total_action_steps,), dtype=bool)
    peg_delta_applied = np.zeros((args.total_action_steps, 3), dtype=np.float32)
    robust_hold_count = 0
    first_robust_hold_step = -1
    first_simple_grasp_step = -1
    counterfactual_final_peg_head_at_hole_at_trigger: np.ndarray | None = None
    counterfactual_final_yz_at_trigger: float | None = None
    counterfactual_min_path_peg_head_at_hole_at_trigger: np.ndarray | None = None
    counterfactual_min_path_yz_at_trigger: float | None = None
    counterfactual_min_path_frame_at_trigger: int | None = None
    counterfactual_final_yz_threshold: float | None = None
    final_xy_resampled_at_trigger = False
    final_xy_resample_count = 0

    for step in range(int(args.total_action_steps)):
        live = _live_eval(base_env)
        head = live["peg_head_at_hole"]
        yz = float(np.linalg.norm(head[1:3]))
        current_obs_for_trigger = _read_state_obs(base_env)
        robust_held_now = bool(
            _robust_held_from_obs(current_obs_for_trigger[None, :], initial_peg_z, args)[0]
        )
        simple_grasp_now = bool(_grasped_from_obs(current_obs_for_trigger[None, :])[0])
        if simple_grasp_now and first_simple_grasp_step < 0:
            first_simple_grasp_step = step
        if robust_held_now:
            robust_hold_count += 1
            if first_robust_hold_step < 0:
                first_robust_hold_step = step
        else:
            robust_hold_count = 0

        held_long_enough = robust_hold_count >= int(args.trigger_grasp_hold_steps)
        after_grasp_delay = (
            first_robust_hold_step >= 0
            and step >= first_robust_hold_step + int(args.min_trigger_after_grasp_steps)
        )
        after_simple_grasp_delay = (
            first_simple_grasp_step >= 0
            and step >= first_simple_grasp_step + int(args.min_trigger_after_grasp_steps)
        )
        strict_preinsert = (
            (step >= int(args.min_trigger_step))
            and held_long_enough
            and after_grasp_delay
            and (float(args.trigger_x_min) <= float(head[0]) <= float(args.trigger_x_max))
            and (yz >= float(args.trigger_yz_min))
            and (yz <= float(args.trigger_yz_threshold))
        )
        fallback_preinsert = (
            (step >= int(args.fallback_trigger_step))
            and held_long_enough
            and after_grasp_delay
            and (float(args.trigger_x_min) <= float(head[0]) <= float(args.trigger_x_max))
            and (yz >= float(args.trigger_yz_min))
            and (yz <= float(args.fallback_trigger_yz_threshold))
        )
        forced_fallback = (
            (int(args.force_fallback_step) >= 0)
            and (step >= int(args.force_fallback_step))
            and after_simple_grasp_delay
        )
        trigger_candidate_reason = ""
        force_fallback_enabled = int(args.force_fallback_step) >= 0
        if scenario != "none" and not triggered and force_fallback_enabled and forced_fallback:
            trigger_candidate_reason = "held_force_fallback_step"
        elif scenario != "none" and not triggered and not force_fallback_enabled and strict_preinsert:
            trigger_candidate_reason = "held_preinsert_geometry"
        elif scenario != "none" and not triggered and not force_fallback_enabled and fallback_preinsert:
            trigger_candidate_reason = "held_fallback_preinsert_geometry"
        if trigger_candidate_reason:
            if scenario in HOLE_MOTION_SCENARIOS:
                counterfactual_final_peg_head_at_hole_at_trigger = _counterfactual_final_hole_alignment(
                    base_env,
                    initial_hole_xyz=initial_hole[:3],
                    initial_box_xyz=initial_box_xyz,
                    final_hole_xy=final_xy,
                )
                counterfactual_final_yz_at_trigger = float(
                    np.linalg.norm(counterfactual_final_peg_head_at_hole_at_trigger[1:3])
                )
                (
                    counterfactual_min_path_yz_at_trigger,
                    counterfactual_min_path_frame_at_trigger,
                    counterfactual_min_path_peg_head_at_hole_at_trigger,
                ) = _counterfactual_min_path_alignment_yz(
                    base_env,
                    scenario=scenario,
                    trigger_step=step,
                    total_video_frames=int(args.total_video_frames),
                    initial_hole_xyz=initial_hole[:3],
                    initial_box_xyz=initial_box_xyz,
                    final_hole_xy=final_xy,
                    rng_values=rng_values,
                    num_probes=int(args.anti_self_insert_path_probe_frames),
                    min_probe_delta_m=float(args.anti_self_insert_min_probe_delta_m),
                )
                current_radius = float(current_obs_for_trigger[LAYOUT["hole_radius"]][0])
                anti_yz = min(
                    float(counterfactual_final_yz_at_trigger),
                    float(counterfactual_min_path_yz_at_trigger),
                )
                anti_pass, anti_threshold = _anti_self_insert_gate_passes(
                    counterfactual_yz=anti_yz,
                    hole_radius=current_radius,
                    args=args,
                )
                counterfactual_final_yz_threshold = anti_threshold
                if not anti_pass and bool(args.resample_final_hole_at_trigger):
                    for resample_idx in range(int(args.final_hole_trigger_resample_attempts)):
                        candidate_final_xy = _sample_final_hole_xy(initial_hole[:2], rng, args)
                        if candidate_final_xy is None:
                            continue
                        candidate_final_head = _counterfactual_final_hole_alignment(
                            base_env,
                            initial_hole_xyz=initial_hole[:3],
                            initial_box_xyz=initial_box_xyz,
                            final_hole_xy=candidate_final_xy,
                        )
                        candidate_final_yz = float(np.linalg.norm(candidate_final_head[1:3]))
                        (
                            candidate_min_path_yz,
                            candidate_min_path_frame,
                            candidate_min_path_head,
                        ) = _counterfactual_min_path_alignment_yz(
                            base_env,
                            scenario=scenario,
                            trigger_step=step,
                            total_video_frames=int(args.total_video_frames),
                            initial_hole_xyz=initial_hole[:3],
                            initial_box_xyz=initial_box_xyz,
                            final_hole_xy=candidate_final_xy,
                            rng_values=rng_values,
                            num_probes=int(args.anti_self_insert_path_probe_frames),
                            min_probe_delta_m=float(args.anti_self_insert_min_probe_delta_m),
                        )
                        candidate_anti_yz = min(float(candidate_final_yz), float(candidate_min_path_yz))
                        candidate_pass, candidate_threshold = _anti_self_insert_gate_passes(
                            counterfactual_yz=candidate_anti_yz,
                            hole_radius=current_radius,
                            args=args,
                        )
                        final_xy_resample_count = int(resample_idx) + 1
                        if not candidate_pass:
                            continue
                        final_xy = candidate_final_xy.astype(np.float32)
                        counterfactual_final_peg_head_at_hole_at_trigger = candidate_final_head.astype(np.float32)
                        counterfactual_final_yz_at_trigger = float(candidate_final_yz)
                        counterfactual_min_path_yz_at_trigger = float(candidate_min_path_yz)
                        counterfactual_min_path_frame_at_trigger = int(candidate_min_path_frame)
                        counterfactual_min_path_peg_head_at_hole_at_trigger = candidate_min_path_head.astype(np.float32)
                        counterfactual_final_yz_threshold = float(candidate_threshold)
                        anti_pass = True
                        final_xy_resampled_at_trigger = True
                        break
                if not anti_pass:
                    return None, {
                        "reason": "counterfactual_final_target_self_insert_gate_failed",
                        "scenario": scenario,
                        "attempt_seed": int(attempt_seed),
                        "step": int(step),
                        "trigger_candidate_reason": trigger_candidate_reason,
                        "current_peg_head_at_current_hole": head.astype(float).tolist(),
                        "current_yz_at_current_hole": yz,
                        "counterfactual_final_peg_head_at_hole": (
                            counterfactual_final_peg_head_at_hole_at_trigger.astype(float).tolist()
                        ),
                        "counterfactual_final_yz_at_trigger": float(counterfactual_final_yz_at_trigger),
                        "counterfactual_min_path_peg_head_at_hole": (
                            counterfactual_min_path_peg_head_at_hole_at_trigger.astype(float).tolist()
                        ),
                        "counterfactual_min_path_yz_at_trigger": float(counterfactual_min_path_yz_at_trigger),
                        "counterfactual_min_path_frame_at_trigger": int(counterfactual_min_path_frame_at_trigger),
                        "counterfactual_gate_yz_at_trigger": float(anti_yz),
                        "counterfactual_final_yz_threshold": float(anti_threshold),
                        "hole_radius": current_radius,
                        "initial_hole_xy": initial_hole[:2].astype(float).tolist(),
                        "candidate_final_hole_xy": np.asarray(final_xy, dtype=np.float32).astype(float).tolist(),
                    }
            triggered = True
            trigger_step = step
            trigger_reason = trigger_candidate_reason

        query_now = not pending_actions
        if query_now:
            obs_seq = np.stack([prev_obs, cur_obs], axis=0)[None].astype(np.float32)
            obs_tensor = torch.as_tensor(obs_seq, device=device, dtype=torch.float32)
            with torch.no_grad():
                action_seq = agent.get_action(obs_tensor).detach().cpu().numpy()[0]
            pending_actions = [action.astype(np.float32) for action in action_seq]
        action = pending_actions.pop(0)
        next_obs, reward, terminated_arr, truncated_arr, info = env.step(action[None, :])

        if triggered and scenario in HOLE_MOTION_SCENARIOS:
            frame = step + 1
            target_hole_xyz = _target_position(
                scenario,
                frame,
                trigger_step,
                int(args.total_video_frames),
                initial_hole[:3],
                final_xy,
                rng_values,
            )
            box_xyz = initial_box_xyz + (target_hole_xyz - initial_hole[:3])
            _set_box_pose(base_env, box_xyz, initial_box_quat)
            action_step_triggered[step] = True
        elif triggered and scenario in PEG_PERTURB_SCENARIOS:
            action_step_triggered[step] = True
            if step == trigger_step:
                peg_xyz, peg_quat = _peg_pose_np(base_env)
                if scenario == "peg_disturb":
                    delta = np.asarray(args.peg_disturb_delta_xyz, dtype=np.float32)
                    new_peg_xyz = peg_xyz + delta
                elif scenario == "peg_drop":
                    new_peg_xyz = peg_xyz.copy()
                    new_peg_xyz[1] += float(args.peg_drop_delta_y_m)
                    new_peg_xyz[2] = float(initial_peg_z)
                    delta = new_peg_xyz - peg_xyz
                else:
                    raise ValueError(f"unexpected peg perturb scenario {scenario!r}")
                _set_peg_pose(base_env, new_peg_xyz, peg_quat)
                peg_delta_applied[step] = delta.astype(np.float32)

        new_obs = _read_state_obs(base_env)
        obs_frames.append(new_obs.copy())
        state_frames.append(_read_env_state(base_env))
        live_after = _live_eval(base_env)
        live_heads.append(live_after["peg_head_at_hole"])
        live_successes.append(live_after["success"])

        actions.append(action.astype(np.float32))
        rewards.append(float(np.asarray(reward).reshape(-1)[0]))
        terminated.append(bool(np.asarray(terminated_arr).reshape(-1)[0]))
        truncated.append(bool(np.asarray(truncated_arr).reshape(-1)[0]))
        policy_query_step.append(bool(query_now))
        prev_obs, cur_obs = cur_obs, new_obs

    obs = np.asarray(obs_frames, dtype=np.float32)
    action_arr = np.asarray(actions, dtype=np.float32)
    if obs.shape[0] != int(args.total_video_frames) or action_arr.shape[0] != int(args.total_action_steps):
        raise RuntimeError(f"bad episode lengths obs={obs.shape} actions={action_arr.shape}")

    hole_pose = obs[:, LAYOUT["hole_pose"]].astype(np.float32)
    peg_pose = obs[:, LAYOUT["peg_pose"]].astype(np.float32)
    tcp_pose = obs[:, LAYOUT["tcp_pose"]].astype(np.float32)
    qpos = obs[:, LAYOUT["qpos"]].astype(np.float32)
    qvel = obs[:, LAYOUT["qvel"]].astype(np.float32)
    radius = obs[:, LAYOUT["hole_radius"]][:, 0].astype(np.float32)
    peg_head = np.asarray(live_heads, dtype=np.float32)
    inserted = _inserted_from_head(peg_head, radius)
    grasped = _grasped_from_obs(obs)
    robust_held = _robust_held_from_obs(obs, initial_peg_z, args)
    live_success_end = bool(live_successes[-1])
    slot_inserted_end = bool(inserted[-1])
    if scenario != "none" and not triggered:
        return None, {
            "reason": "never_triggered_after_robust_hold_and_preinsert",
            "scenario": scenario,
            "attempt_seed": int(attempt_seed),
            "first_robust_hold_step": int(first_robust_hold_step),
            "live_success_end": live_success_end,
            "slot_inserted_end": slot_inserted_end,
            "target_motion_norm_m": float(np.linalg.norm(hole_pose[-1, :3] - hole_pose[0, :3])),
        }
    if (args.accept_requires_live_success and not live_success_end) or (
        args.accept_requires_slot_inserted and not slot_inserted_end
    ):
        return None, {
            "reason": "final_insert_failed",
            "scenario": scenario,
            "attempt_seed": int(attempt_seed),
            "triggered": bool(triggered),
            "trigger_step": int(trigger_step),
            "trigger_reason": trigger_reason,
            "live_success_end": live_success_end,
            "slot_inserted_end": slot_inserted_end,
            "final_peg_head_at_hole": peg_head[-1].astype(float).tolist(),
            "target_motion_norm_m": float(np.linalg.norm(hole_pose[-1, :3] - hole_pose[0, :3])),
        }

    first_insert = int(np.flatnonzero(inserted)[0]) if inserted.any() else -1
    first_grasp = int(np.flatnonzero(grasped)[0]) if grasped.any() else -1
    first_robust_hold = int(np.flatnonzero(robust_held)[0]) if robust_held.any() else -1
    motion_indices = np.flatnonzero(np.linalg.norm(hole_pose[:, :3] - hole_pose[0:1, :3], axis=1) > 1e-5)
    motion_step_indices = np.flatnonzero(np.linalg.norm(np.diff(hole_pose[:, :3], axis=0), axis=1) > 1e-5)
    first_motion = int(motion_indices[0]) if motion_indices.size else -1
    last_motion_frame = int(motion_step_indices[-1] + 1) if motion_step_indices.size else -1
    peg_perturb_indices = np.flatnonzero(np.linalg.norm(peg_delta_applied, axis=1) > 1e-8)
    first_peg_perturb_step = int(peg_perturb_indices[0]) if peg_perturb_indices.size else -1
    first_event_step = first_motion if scenario in HOLE_MOTION_SCENARIOS else first_peg_perturb_step
    if scenario != "none":
        if str(trigger_reason) == "held_force_fallback_step" and int(args.force_fallback_step) >= 0:
            event_after_valid_grasp = (
                first_grasp >= 0
                and first_event_step >= first_grasp + int(args.min_trigger_after_grasp_steps)
            )
            if not event_after_valid_grasp:
                return None, {
                    "reason": "forced_fallback_perturbation_before_grasp_delay",
                    "scenario": scenario,
                    "attempt_seed": int(attempt_seed),
                    "trigger_step": int(trigger_step),
                    "first_motion_step": int(first_motion),
                    "first_peg_perturb_step": int(first_peg_perturb_step),
                    "first_grasp_step": int(first_grasp),
                    "first_robust_hold_step": int(first_robust_hold),
                }
        elif first_robust_hold < 0 or first_event_step < first_robust_hold + int(args.min_trigger_after_grasp_steps):
            return None, {
                "reason": "perturbation_before_stable_peg_hold",
                "scenario": scenario,
                "attempt_seed": int(attempt_seed),
                "trigger_step": int(trigger_step),
                "first_motion_step": int(first_motion),
                "first_peg_perturb_step": int(first_peg_perturb_step),
                "first_grasp_step": int(first_grasp),
                "first_robust_hold_step": int(first_robust_hold),
            }
    if scenario in HOLE_MOTION_SCENARIOS:
        if first_motion >= 0 and first_insert >= first_motion and first_insert <= last_motion_frame:
            window_end = min(int(obs.shape[0]), int(first_insert) + 1)
            tcp_motion_before_insert = 0.0
            peg_motion_before_insert = 0.0
            target_motion_before_insert = 0.0
            min_head_yz_before_insert = float("inf")
            if window_end > int(first_motion):
                motion_slice = slice(int(first_motion), window_end)
                tcp_motion_before_insert = float(
                    np.linalg.norm(
                        tcp_pose[motion_slice, :3] - tcp_pose[int(first_motion) : int(first_motion) + 1, :3],
                        axis=1,
                    ).max()
                )
                peg_motion_before_insert = float(
                    np.linalg.norm(
                        peg_pose[motion_slice, :3] - peg_pose[int(first_motion) : int(first_motion) + 1, :3],
                        axis=1,
                    ).max()
                )
                target_motion_before_insert = float(
                    np.linalg.norm(
                        hole_pose[motion_slice, :3] - hole_pose[int(first_motion) : int(first_motion) + 1, :3],
                        axis=1,
                    ).max()
                )
                min_head_yz_before_insert = float(np.linalg.norm(peg_head[motion_slice, 1:3], axis=1).min())
            if (
                target_motion_before_insert >= float(args.self_insert_min_target_motion_m)
                and tcp_motion_before_insert < float(args.self_insert_min_tcp_rebind_m)
                and min_head_yz_before_insert <= float(args.self_insert_motion_yz_max_m)
            ):
                return None, {
                    "reason": "target_self_insert_without_robot_rebind_motion",
                    "scenario": scenario,
                    "attempt_seed": int(attempt_seed),
                    "trigger_step": int(trigger_step),
                    "first_motion_step": int(first_motion),
                    "last_motion_frame": int(last_motion_frame),
                    "first_insert_step": int(first_insert),
                    "target_motion_before_insert_m": float(target_motion_before_insert),
                    "tcp_motion_before_insert_m": float(tcp_motion_before_insert),
                    "peg_motion_before_insert_m": float(peg_motion_before_insert),
                    "min_head_yz_before_insert_m": float(min_head_yz_before_insert),
                    "self_insert_min_target_motion_m": float(args.self_insert_min_target_motion_m),
                    "self_insert_min_tcp_rebind_m": float(args.self_insert_min_tcp_rebind_m),
                    "self_insert_motion_yz_max_m": float(args.self_insert_motion_yz_max_m),
                    "target_motion_norm_m": float(np.linalg.norm(hole_pose[-1, :3] - hole_pose[0, :3])),
                }
        min_insert_step = int(last_motion_frame) + int(args.min_insert_after_target_motion_end_steps)
        if (
            int(args.min_insert_after_target_motion_end_steps) >= 0
            and first_insert >= 0
            and first_insert <= min_insert_step
        ):
            return None, {
                "reason": "inserted_during_or_too_soon_after_target_motion",
                "scenario": scenario,
                "attempt_seed": int(attempt_seed),
                "trigger_step": int(trigger_step),
                "first_motion_step": int(first_motion),
                "last_motion_frame": int(last_motion_frame),
                "first_insert_step": int(first_insert),
                "required_first_insert_after_step": int(min_insert_step),
                "target_motion_norm_m": float(np.linalg.norm(hole_pose[-1, :3] - hole_pose[0, :3])),
            }
        if first_motion >= 0 and last_motion_frame >= first_motion:
            guard_end = min(
                int(obs.shape[0]) - 1,
                int(last_motion_frame) + int(args.target_motion_grasp_guard_steps),
            )
            grasp_window = grasped[int(first_motion) : guard_end + 1]
            if grasp_window.size and not bool(np.all(grasp_window)):
                lost_rel = int(np.flatnonzero(~grasp_window)[0])
                lost_frame = int(first_motion) + lost_rel
                return None, {
                    "reason": "lost_grasp_during_target_motion_window",
                    "scenario": scenario,
                    "attempt_seed": int(attempt_seed),
                    "trigger_step": int(trigger_step),
                    "first_motion_step": int(first_motion),
                    "last_motion_frame": int(last_motion_frame),
                    "grasp_guard_end_frame": int(guard_end),
                    "first_lost_grasp_frame": int(lost_frame),
                    "first_insert_step": int(first_insert),
                }
    target_motion = hole_pose[-1, :3] - hole_pose[0, :3]
    target_motion_norm = float(np.linalg.norm(target_motion))
    if scenario in HOLE_MOTION_SCENARIOS and target_motion_norm < float(args.motion_min_m):
        return None, {
            "reason": "actual_target_motion_too_small",
            "scenario": scenario,
            "attempt_seed": int(attempt_seed),
            "trigger_step": int(trigger_step),
            "first_motion_step": int(first_motion),
            "target_motion_norm_m": target_motion_norm,
            "motion_min_m": float(args.motion_min_m),
        }
    policy_seed_tag = "" if policy_seed is None else f"_pseed{int(policy_seed):06d}"
    sample_id = f"{scenario}_seed{attempt_seed:06d}{policy_seed_tag}_idx{accepted_index:04d}"
    summary = {
        "sample_id": sample_id,
        "scenario": scenario,
        "seed": int(attempt_seed),
        "policy_seed": None if policy_seed is None else int(policy_seed),
        "steps": int(args.total_action_steps),
        "triggered": bool(triggered),
        "trigger_step": int(trigger_step),
        "trigger_reason": trigger_reason,
        "first_grasp_step": first_grasp,
        "first_robust_hold_step": first_robust_hold,
        "first_target_motion_step": first_motion,
        "last_target_motion_frame": last_motion_frame,
        "first_peg_perturb_step": first_peg_perturb_step,
        "first_insert_step": first_insert,
        "success_once": bool(inserted.any()),
        "success_at_end": bool(live_success_end and slot_inserted_end),
        "inserted_end": bool(slot_inserted_end),
        "live_success_end": live_success_end,
        "final_peg_head_at_hole": peg_head[-1].astype(float).tolist(),
        "target_motion_norm_m": target_motion_norm,
        "target_motion_xyz": target_motion.astype(float).tolist(),
        "target_start_xyz": hole_pose[0, :3].astype(float).tolist(),
        "target_final_xyz": hole_pose[-1, :3].astype(float).tolist(),
        "target_motion_policy": (
            "copy-on-write original DP dynamic rollout protocol: target remains static until "
            "the peg is stably held/lifted and reaches preinsert geometry. Original top-level "
            "classes are preserved: no perturbation for none, large fast live box motion for "
            "moving-hole classes, and one-shot live peg perturb/drop for peg classes. No robot "
            "state projection is applied. Moving-hole candidates are rejected if the final "
            "target pose or sampled target path is already aligned with the current peg at "
            "the trigger, because that would let the target self-insert by moving into the "
            "peg instead of requiring rebinding. They are also rejected if the gripper loses "
            "the peg while the target is moving, because that indicates target-driven collision "
            "rather than robot-held predictive insertion"
        ),
        "original_protocol_provenance": {
            "dp_ckpt_path": args.ckpt_path,
            "state_key": args.state_key,
            "candidate_seed_resets_action_rng": bool(args.reset_candidate_rng),
            "policy_rng_seed": None if policy_seed is None else int(policy_seed),
            "source_manifest_defaults": {
                "trigger_mode": "preinsert",
                "trigger_yz_min": float(args.trigger_yz_min),
                "trigger_yz_threshold": 0.035,
                "trigger_x_min": -0.14,
                "trigger_x_max": 0.05,
                "fallback_trigger_step": 90,
                "fallback_requires_robust_hold_and_relaxed_preinsert": True,
                "max_episode_steps": 300,
            },
        },
        "large_motion_rebuild": {
            "motion_min_m": float(args.motion_min_m),
            "motion_max_m": float(args.motion_max_m),
            "final_xy_bounds": [args.final_x_min, args.final_x_max, args.final_y_min, args.final_y_max],
            "scenario_rng_values": rng_values,
            "peg_disturb_delta_xyz": list(args.peg_disturb_delta_xyz),
            "peg_drop_delta_y_m": float(args.peg_drop_delta_y_m),
            "anti_self_insert_final_yz_min_m": float(args.anti_self_insert_final_yz_min_m),
            "anti_self_insert_radius_multiplier": float(args.anti_self_insert_radius_multiplier),
            "anti_self_insert_min_probe_delta_m": float(args.anti_self_insert_min_probe_delta_m),
            "resample_final_hole_at_trigger": bool(args.resample_final_hole_at_trigger),
            "final_hole_trigger_resample_attempts": int(args.final_hole_trigger_resample_attempts),
            "final_xy_resampled_at_trigger": bool(final_xy_resampled_at_trigger),
            "final_xy_resample_count": int(final_xy_resample_count),
            "min_insert_after_target_motion_end_steps": int(args.min_insert_after_target_motion_end_steps),
            "target_motion_grasp_guard_steps": int(args.target_motion_grasp_guard_steps),
            "self_insert_motion_yz_max_m": float(args.self_insert_motion_yz_max_m),
            "self_insert_min_target_motion_m": float(args.self_insert_min_target_motion_m),
            "self_insert_min_tcp_rebind_m": float(args.self_insert_min_tcp_rebind_m),
            "counterfactual_final_peg_head_at_hole_at_trigger": (
                None
                if counterfactual_final_peg_head_at_hole_at_trigger is None
                else counterfactual_final_peg_head_at_hole_at_trigger.astype(float).tolist()
            ),
            "counterfactual_final_yz_at_trigger": counterfactual_final_yz_at_trigger,
            "counterfactual_min_path_peg_head_at_hole_at_trigger": (
                None
                if counterfactual_min_path_peg_head_at_hole_at_trigger is None
                else counterfactual_min_path_peg_head_at_hole_at_trigger.astype(float).tolist()
            ),
            "counterfactual_min_path_yz_at_trigger": counterfactual_min_path_yz_at_trigger,
            "counterfactual_min_path_frame_at_trigger": counterfactual_min_path_frame_at_trigger,
            "counterfactual_final_yz_threshold": counterfactual_final_yz_threshold,
        },
    }
    return {
        "sample_id": sample_id,
        "scenario": scenario,
        "seed": int(attempt_seed),
        "policy_seed": None if policy_seed is None else int(policy_seed),
        "obs": obs,
        "actions": action_arr,
        "env_states": _stack_env_states(state_frames),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminated": np.asarray(terminated, dtype=bool),
        "truncated": np.asarray(truncated, dtype=bool),
        "policy_query_step": np.asarray(policy_query_step, dtype=bool),
        "slots": {
            "hole_pose": hole_pose,
            "peg_pose": peg_pose,
            "tcp_pose": tcp_pose,
            "qpos": qpos,
            "qvel": qvel,
            "hole_radius": radius,
            "peg_head_at_hole": peg_head,
            "hole_velocity_step": np.vstack([np.zeros((1, 3), dtype=np.float32), np.diff(hole_pose[:, :3], axis=0)]),
            "grasped": grasped,
            "robust_held": robust_held,
            "inserted": inserted,
        },
        "perturb": {
            "hole_delta_applied": (hole_pose[1:, :3] - hole_pose[:-1, :3]).astype(np.float32),
            "hole_delta_cumulative": (hole_pose[1:, :3] - hole_pose[0:1, :3]).astype(np.float32),
            "peg_delta_applied": peg_delta_applied,
            "triggered": action_step_triggered.astype(bool),
            "trigger_step": np.full((int(args.total_action_steps),), int(trigger_step), dtype=np.int32),
            "phase": np.full((int(args.total_action_steps),), SCENARIOS.index(scenario), dtype=np.int32),
        },
        "summary": summary,
    }, {}


def _write_record(record: dict[str, Any], out_h5: Path) -> dict[str, Any]:
    if out_h5.exists():
        out_h5.unlink()
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as h5:
        group = h5.create_group("traj_0")
        group.attrs["seed"] = int(record["seed"])
        if record.get("policy_seed") is not None:
            group.attrs["policy_seed"] = int(record["policy_seed"])
        group.attrs["summary_json"] = json.dumps(_jsonable(record["summary"]), sort_keys=True)
        group.attrs["source_summary_json"] = json.dumps(_jsonable(record["summary"]), sort_keys=True)
        _write_dataset(group, "actions", record["actions"])
        _write_dataset(group, "obs_current", record["obs"])
        _write_dataset(group, "obs_stack", _stack_obs(record["obs"]))
        _write_dataset(group, "source_frame_indices", np.arange(record["obs"].shape[0], dtype=np.int64))
        _write_dataset(group, "rewards", record["rewards"])
        _write_dataset(group, "terminated", record["terminated"])
        _write_dataset(group, "truncated", record["truncated"])
        _write_dataset(group, "policy_query_step", record["policy_query_step"])
        slots = group.create_group("slots")
        for key, value in record["slots"].items():
            _write_dataset(slots, key, value)
        perturb = group.create_group("perturb")
        for key, value in record["perturb"].items():
            _write_dataset(perturb, key, value)
        env_group = group.create_group("env_states")
        _write_dict_group(env_group, record["env_states"])
    return {
        "sample_id": record["sample_id"],
        "scenario": record["scenario"],
        "seed": int(record["seed"]),
        "policy_seed": None if record.get("policy_seed") is None else int(record["policy_seed"]),
        "path": str(out_h5),
        "split": "val" if _stable_val(record["sample_id"], 0.1) else "train",
        "success_at_end": bool(record["summary"]["success_at_end"]),
        "first_grasp_step": int(record["summary"]["first_grasp_step"]),
        "first_robust_hold_step": int(record["summary"]["first_robust_hold_step"]),
        "first_target_motion_step": int(record["summary"]["first_target_motion_step"]),
        "last_target_motion_frame": int(record["summary"]["last_target_motion_frame"]),
        "first_insert_step": int(record["summary"]["first_insert_step"]),
        "trigger_step": int(record["summary"]["trigger_step"]),
        "trigger_reason": str(record["summary"]["trigger_reason"]),
        "target_motion_norm_m": float(record["summary"]["target_motion_norm_m"]),
    }


def _validate_args(args: Args) -> None:
    if args.total_video_frames != args.total_action_steps + 1:
        raise ValueError("expected 301 frames and 300 actions")
    forbidden = "sft_dataset_full1000_maniskill_default_regen_20260606_0055"
    if forbidden in str(Path(args.output_root)):
        raise ValueError("refusing to write into the original full1000 dataset root")
    if args.motion_min_m <= 0 or args.motion_max_m < args.motion_min_m:
        raise ValueError("invalid motion range")
    if args.trigger_grasp_hold_steps < 1:
        raise ValueError("trigger_grasp_hold_steps must be positive")
    if args.min_trigger_after_grasp_steps < 0:
        raise ValueError("min_trigger_after_grasp_steps must be non-negative")
    if args.trigger_yz_min < 0:
        raise ValueError("trigger_yz_min must be non-negative")
    if args.trigger_yz_min >= args.fallback_trigger_yz_threshold:
        raise ValueError("trigger_yz_min must be smaller than fallback_trigger_yz_threshold")
    if args.force_fallback_step < -1:
        raise ValueError("force_fallback_step must be -1 or non-negative")
    if args.constant_steps_min < 1 or args.constant_steps_max < args.constant_steps_min:
        raise ValueError("invalid constant step range")
    if args.continuous_steps_min < 1 or args.continuous_steps_max < args.continuous_steps_min:
        raise ValueError("invalid continuous step range")
    if args.sine_steps_min < 1 or args.sine_steps_max < args.sine_steps_min:
        raise ValueError("invalid sine step range")
    if args.reverse_after_steps_min < 1 or args.reverse_after_steps_max < args.reverse_after_steps_min:
        raise ValueError("invalid reverse after-step range")
    if args.reverse_return_steps_min < 1 or args.reverse_return_steps_max < args.reverse_return_steps_min:
        raise ValueError("invalid reverse return-step range")
    if args.fast_shift_steps_min < 1 or args.fast_shift_steps_max < args.fast_shift_steps_min:
        raise ValueError("invalid fast-shift step range")
    if args.move_stop_steps_min < 1 or args.move_stop_steps_max < args.move_stop_steps_min:
        raise ValueError("invalid move-stop step range")
    if args.anti_self_insert_path_probe_frames < 2:
        raise ValueError("anti_self_insert_path_probe_frames must be at least 2")
    if args.anti_self_insert_min_probe_delta_m < 0:
        raise ValueError("anti_self_insert_min_probe_delta_m must be non-negative")
    if args.final_hole_trigger_resample_attempts < 0:
        raise ValueError("final_hole_trigger_resample_attempts must be non-negative")
    if args.self_insert_motion_yz_max_m < 0:
        raise ValueError("self_insert_motion_yz_max_m must be non-negative")
    if args.self_insert_min_target_motion_m < 0:
        raise ValueError("self_insert_min_target_motion_m must be non-negative")
    if args.self_insert_min_tcp_rebind_m < 0:
        raise ValueError("self_insert_min_tcp_rebind_m must be non-negative")
    if args.min_insert_after_target_motion_end_steps < -1:
        raise ValueError("min_insert_after_target_motion_end_steps must be -1 or non-negative")
    if args.target_motion_grasp_guard_steps < 0:
        raise ValueError("target_motion_grasp_guard_steps must be non-negative")
    if args.accepted_index_offset < 0:
        raise ValueError("accepted_index_offset must be non-negative")
    if args.non_priority_seed_offset < 0:
        raise ValueError("non_priority_seed_offset must be non-negative")
    if args.policy_rng_seed_base < -1:
        raise ValueError("policy_rng_seed_base must be -1 or non-negative")
    scenarios = _scenario_sequence(args)
    explicit_quotas = _parse_scenario_value_map(
        args.scenario_quotas,
        flag_name="scenario-quotas",
        value_type=int,
    )
    _scenario_quotas(
        int(args.num_demos),
        scenarios,
        bool(args.balanced_scenario_quotas),
        explicit_quotas=explicit_quotas,
    )
    _parse_scenario_seed_bases(args.scenario_seed_bases)
    _parse_scenario_value_map(
        args.scenario_trigger_yz_min,
        flag_name="scenario-trigger-yz-min",
        value_type=float,
    )
    _parse_scenario_value_map(
        args.scenario_min_trigger_step,
        flag_name="scenario-min-trigger-step",
        value_type=int,
    )
    _parse_scenario_value_map(
        args.scenario_fallback_trigger_step,
        flag_name="scenario-fallback-trigger-step",
        value_type=int,
    )
    scenario_force_fallback_step = _parse_scenario_value_map(
        args.scenario_force_fallback_step,
        flag_name="scenario-force-fallback-step",
        value_type=int,
    )
    for scenario, value in scenario_force_fallback_step.items():
        if int(value) < -1:
            raise ValueError(
                f"--scenario-force-fallback-step for {scenario} must be -1 or non-negative"
            )
    _parse_scenario_value_map(
        args.scenario_min_insert_after_target_motion_end_steps,
        flag_name="scenario-min-insert-after-target-motion-end-steps",
        value_type=int,
    )
    scenario_policy_seed_bases = _parse_scenario_value_map(
        args.scenario_policy_rng_seed_bases,
        flag_name="scenario-policy-rng-seed-bases",
        value_type=int,
    )
    for scenario, value in scenario_policy_seed_bases.items():
        if int(value) < 0:
            raise ValueError(
                f"--scenario-policy-rng-seed-bases for {scenario} must be non-negative"
            )


def main() -> None:
    args = tyro.cli(Args)
    _validate_args(args)
    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing non-empty output root without --overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    paths_file = Path(args.paths_file) if args.paths_file else output_root / "fix3_h5_paths.txt"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = _make_env(args)
    agent, dp_args, ckpt = _load_agent(env, args, device)
    scenarios = _scenario_sequence(args)
    explicit_quotas = _parse_scenario_value_map(
        args.scenario_quotas,
        flag_name="scenario-quotas",
        value_type=int,
    )
    scenario_seed_bases = _parse_scenario_seed_bases(args.scenario_seed_bases)
    trigger_yz_min_by_scenario = _parse_scenario_value_map(
        args.scenario_trigger_yz_min,
        flag_name="scenario-trigger-yz-min",
        value_type=float,
    )
    min_trigger_step_by_scenario = _parse_scenario_value_map(
        args.scenario_min_trigger_step,
        flag_name="scenario-min-trigger-step",
        value_type=int,
    )
    fallback_trigger_step_by_scenario = _parse_scenario_value_map(
        args.scenario_fallback_trigger_step,
        flag_name="scenario-fallback-trigger-step",
        value_type=int,
    )
    force_fallback_step_by_scenario = _parse_scenario_value_map(
        args.scenario_force_fallback_step,
        flag_name="scenario-force-fallback-step",
        value_type=int,
    )
    min_insert_after_motion_by_scenario = _parse_scenario_value_map(
        args.scenario_min_insert_after_target_motion_end_steps,
        flag_name="scenario-min-insert-after-target-motion-end-steps",
        value_type=int,
    )
    policy_rng_seed_bases_by_scenario = _parse_scenario_value_map(
        args.scenario_policy_rng_seed_bases,
        flag_name="scenario-policy-rng-seed-bases",
        value_type=int,
    )
    scenario_quotas = _scenario_quotas(
        int(args.num_demos),
        scenarios,
        bool(args.balanced_scenario_quotas),
        explicit_quotas=explicit_quotas,
    )

    accepted: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    scenario_attempt_counts: dict[str, int] = {scenario: 0 for scenario in SCENARIOS}
    scenario_accept_counts: dict[str, int] = {scenario: 0 for scenario in SCENARIOS}
    scenario_cursor = 0
    attempt = 0
    try:
        while len(accepted) < int(args.num_demos) and attempt < int(args.max_attempts):
            scenario, scenario_cursor = _next_scenario_with_open_quota(
                scenarios=scenarios,
                scenario_accept_counts=scenario_accept_counts,
                scenario_quotas=scenario_quotas,
                cursor=scenario_cursor,
            )
            if scenario is None:
                break
            scenario_attempt = int(scenario_attempt_counts[scenario])
            attempt_seed = _attempt_seed_for_scenario(
                scenario=scenario,
                base_seed=int(args.seed),
                global_attempt=int(attempt),
                scenario_attempt=scenario_attempt,
                scenario_seed_bases=scenario_seed_bases,
                use_priority_seeds=bool(args.use_priority_seeds),
                non_priority_seed_offset=int(args.non_priority_seed_offset),
            )
            policy_seed: int | None = None
            if scenario in policy_rng_seed_bases_by_scenario:
                policy_seed = int(policy_rng_seed_bases_by_scenario[scenario]) + scenario_attempt
            elif int(args.policy_rng_seed_base) >= 0:
                policy_seed = int(args.policy_rng_seed_base) + int(attempt)
            candidate_args = _scenario_args(
                args,
                scenario,
                trigger_yz_min_by_scenario=trigger_yz_min_by_scenario,
                min_trigger_step_by_scenario=min_trigger_step_by_scenario,
                fallback_trigger_step_by_scenario=fallback_trigger_step_by_scenario,
                force_fallback_step_by_scenario=force_fallback_step_by_scenario,
                min_insert_after_motion_by_scenario=min_insert_after_motion_by_scenario,
            )
            scenario_attempt_counts[scenario] = scenario_attempt + 1
            record, reject = _run_candidate(
                env=env,
                agent=agent,
                dp_args=dp_args,
                device=device,
                scenario=scenario,
                attempt_seed=attempt_seed,
                policy_seed=policy_seed,
                accepted_index=int(args.accepted_index_offset) + len(accepted),
                args=candidate_args,
            )
            if record is None:
                if len(rejects) < int(args.save_reject_log_limit):
                    rejects.append(reject)
                scenario_reject_count = int(scenario_attempt_counts[scenario])
                if (
                    int(args.reject_log_every) > 0
                    and (scenario_reject_count <= 3 or scenario_reject_count % int(args.reject_log_every) == 0)
                ):
                    print(
                        json.dumps(
                            {
                                "event": "fix3_original_protocol_large_motion_reject_progress",
                                "accepted": len(accepted),
                                "attempt": int(attempt),
                                "scenario": scenario,
                                "attempt_seed": int(attempt_seed),
                                "policy_seed": None if policy_seed is None else int(policy_seed),
                                "scenario_attempt": scenario_reject_count,
                                "scenario_accepted": int(scenario_accept_counts[scenario]),
                                "scenario_quota": int(scenario_quotas.get(scenario, 0)),
                                "reason": reject.get("reason", "unknown"),
                                "reject": _jsonable(reject),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                attempt += 1
                continue
            sample_dir = output_root / f"{record['sample_id']}.fix3"
            out_h5 = sample_dir / f"{record['sample_id']}.h5"
            accepted.append(_write_record(record, out_h5))
            scenario_accept_counts[scenario] = int(scenario_accept_counts[scenario]) + 1
            paths_file.write_text("\n".join(item["path"] for item in accepted) + "\n")
            partial_manifest = {
                "args": _jsonable(asdict(args)),
                "status": "partial",
                "num_records": len(accepted),
                "num_attempts": int(attempt + 1),
                "paths_file": str(paths_file),
                "scenario_counts": {
                    scenario_name: sum(1 for item in accepted if item["scenario"] == scenario_name)
                    for scenario_name in SCENARIOS
                },
                "scenario_sequence": list(scenarios),
                "scenario_quotas": dict(scenario_quotas),
                "scenario_accept_counts": dict(scenario_accept_counts),
                "accepted_index_offset": int(args.accepted_index_offset),
                "scenario_attempt_counts": dict(scenario_attempt_counts),
                "scenario_seed_bases": dict(scenario_seed_bases),
                "scenario_overrides": {
                    "trigger_yz_min": dict(trigger_yz_min_by_scenario),
                    "min_trigger_step": dict(min_trigger_step_by_scenario),
                    "fallback_trigger_step": dict(fallback_trigger_step_by_scenario),
                    "min_insert_after_target_motion_end_steps": dict(min_insert_after_motion_by_scenario),
                },
                "records": accepted,
                "rejects": rejects,
            }
            (output_root / "partial_manifest.json").write_text(
                json.dumps(_jsonable(partial_manifest), indent=2, sort_keys=True) + "\n"
            )
            print(
                json.dumps(
                    {
                        "event": "fix3_original_protocol_large_motion_accept",
                        "accepted": len(accepted),
                        "attempt": attempt,
                        "scenario": scenario,
                        "scenario_accepted": int(scenario_accept_counts[scenario]),
                        "scenario_quota": int(scenario_quotas.get(scenario, 0)),
                        "path": str(out_h5),
                        "target_motion_norm_m": accepted[-1]["target_motion_norm_m"],
                        "trigger_step": accepted[-1]["trigger_step"],
                        "first_insert_step": accepted[-1]["first_insert_step"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            attempt += 1
    finally:
        env.close()

    if len(accepted) < int(args.num_demos):
        raise RuntimeError(
            f"accepted only {len(accepted)}/{args.num_demos} after {args.max_attempts} attempts; "
            f"see rejects in {output_root}"
        )

    paths_file.write_text("\n".join(item["path"] for item in accepted) + "\n")
    motion = np.asarray([item["target_motion_norm_m"] for item in accepted], dtype=np.float32)
    manifest = {
        "args": _jsonable(asdict(args)),
        "boundary": (
            "Fix3 copy-on-write large-motion data generated from the original DP dynamic "
            "rollout protocol provenance. Original full1000 data/code/env are read-only; "
            "accepted samples require live final success and slot insertion."
        ),
        "num_records": len(accepted),
        "num_attempts": int(attempt),
        "output_root": str(output_root),
        "paths_file": str(paths_file),
        "scenario_sequence": list(scenarios),
        "scenario_quotas": dict(scenario_quotas),
        "scenario_accept_counts": dict(scenario_accept_counts),
        "accepted_index_offset": int(args.accepted_index_offset),
        "scenario_counts": {scenario: sum(1 for item in accepted if item["scenario"] == scenario) for scenario in SCENARIOS},
        "scenario_attempt_counts": dict(scenario_attempt_counts),
        "scenario_seed_bases": dict(scenario_seed_bases),
        "scenario_overrides": {
            "trigger_yz_min": dict(trigger_yz_min_by_scenario),
            "min_trigger_step": dict(min_trigger_step_by_scenario),
            "fallback_trigger_step": dict(fallback_trigger_step_by_scenario),
            "min_insert_after_target_motion_end_steps": dict(min_insert_after_motion_by_scenario),
        },
        "use_priority_seeds": bool(args.use_priority_seeds),
        "balanced_scenario_quotas": bool(args.balanced_scenario_quotas),
        "target_motion_norm_m": {
            "min": float(motion.min()) if motion.size else None,
            "mean": float(motion.mean()) if motion.size else None,
            "max": float(motion.max()) if motion.size else None,
        },
        "records": accepted,
        "rejects": rejects,
        "dp_checkpoint": {
            "path": args.ckpt_path,
            "state_key": args.state_key,
            "iteration": ckpt.get("iteration"),
            "args": _jsonable(vars(dp_args)),
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "fix3_original_protocol_large_motion_done", "manifest": str(output_root / "manifest.json"), "num_records": len(accepted)}, sort_keys=True))


if __name__ == "__main__":
    main()
