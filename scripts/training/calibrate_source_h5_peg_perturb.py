#!/usr/bin/env python3
"""Physical peg-perturb calibration for approved source-H5 keys.

This diagnostic runs the original DP prefix until the source-H5 live gate, then
applies peg forces through physics only. It does not run Cosmos and does not
produce Oracle success evidence.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from mani_skill.utils import common

from eval_dp_oracle_full_pipeline import (
    apply_physical_peg_perturb,
    build_train_args,
    eval_state,
    install_forbidden_peg_state_guard,
    jsonable,
    load_source_h5_protocol,
    make_env,
    peg_perturb_force_vector,
    require_allocation,
    source_protocol_is_peg_perturb,
    source_target_motion_live_gate,
    write_json,
)


@dataclass
class Args:
    ckpt_path: str
    source_h5_path: str
    output_dir: str
    source_key: str = ""
    env_id: str = "PegInsertionSide-v1"
    control_mode: str = "pd_ee_delta_pose"
    sim_backend: str = "physx_cpu"
    max_episode_steps: int = 220
    seed: int = 2
    source_h5_require_live_motion_gate: bool = True
    source_h5_gate_x_margin: float = 0.03
    source_h5_gate_yz_margin: float = 0.015
    source_h5_peg_perturb_mode: str = "force"
    source_h5_peg_force_scale: float = 25.0
    source_h5_peg_force_steps: int = 8
    force_scales_csv: str = "25,50,100,200,400"
    force_steps_csv: str = "8,16,32"
    settle_steps: int = 6
    use_ema: bool = True
    cuda: bool = True


def parse_float_csv(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def norm3(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64).reshape(3)))


def cosine3(a: np.ndarray, b: np.ndarray) -> float | None:
    an = norm3(a)
    bn = norm3(b)
    if an <= 1.0e-12 or bn <= 1.0e-12:
        return None
    return float(np.dot(a.reshape(3), b.reshape(3)) / (an * bn))


def run_one_trial(
    args: Args,
    *,
    train_args: Any,
    agent: Any,
    device: torch.device,
    source_protocol: dict[str, Any],
    force_scale: float,
    force_steps: int,
) -> dict[str, Any]:
    trial_args = Args(**vars(args))
    trial_args.source_h5_peg_force_scale = float(force_scale)
    trial_args.source_h5_peg_force_steps = int(force_steps)
    envs = make_env(train_args)
    trace: list[dict[str, Any]] = []
    gate_rejections: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            obs, _info = envs.reset(seed=int(source_protocol["seed"]))
            base_env = envs.envs[0].unwrapped
            guard = install_forbidden_peg_state_guard(base_env)
            if not guard["ok"]:
                return {
                    "ok": False,
                    "classification": "blocked_peg_state_guard_install_failed",
                    "peg_state_guard": guard,
                    "force_scale": force_scale,
                    "force_steps": force_steps,
                }
            target_step = int(source_protocol["first_peg_perturb_step"])
            if target_step < 0:
                target_step = int(source_protocol["trigger_step"])
            force_xyz = peg_perturb_force_vector(trial_args, source_protocol)
            for _chunk_idx in range(int(train_args.max_episode_steps)):
                obs_tensor = common.to_tensor(obs, device)
                dp_actions = agent.get_action(obs_tensor).detach().cpu().numpy()
                for action_vec in dp_actions[0]:
                    step_idx = len(trace)
                    state_before_action = eval_state(base_env)
                    if step_idx >= target_step:
                        gate_report = source_target_motion_live_gate(trial_args, source_protocol, state_before_action)
                        if gate_report["ok"]:
                            zero_action = np.zeros_like(np.asarray(action_vec, dtype=np.float32))
                            state_before_perturb = eval_state(base_env)
                            per_step: list[dict[str, Any]] = []
                            obs_after = obs
                            reward = terminated = truncated = info = None
                            for force_idx in range(max(1, int(force_steps))):
                                apply_physical_peg_perturb(base_env, force_xyz)
                                obs_after, reward, terminated, truncated, info = envs.step(zero_action.reshape(1, -1))
                                state_now = eval_state(base_env)
                                per_step.append(
                                    {
                                        "force_step": force_idx,
                                        "peg_xyz": state_now["peg_xyz"],
                                        "peg_head_at_hole": state_now["peg_head_at_hole"],
                                        "peg_head_l2": state_now["peg_head_l2"],
                                        "success": state_now["success"],
                                    }
                                )
                            for settle_idx in range(max(0, int(args.settle_steps))):
                                obs_after, reward, terminated, truncated, info = envs.step(zero_action.reshape(1, -1))
                                state_now = eval_state(base_env)
                                per_step.append(
                                    {
                                        "settle_step": settle_idx,
                                        "peg_xyz": state_now["peg_xyz"],
                                        "peg_head_at_hole": state_now["peg_head_at_hole"],
                                        "peg_head_l2": state_now["peg_head_l2"],
                                        "success": state_now["success"],
                                    }
                                )
                            state_after = eval_state(base_env)
                            observed_delta = (
                                np.asarray(state_after["peg_xyz"], dtype=np.float64)
                                - np.asarray(state_before_perturb["peg_xyz"], dtype=np.float64)
                            )
                            expected_delta = np.asarray(
                                source_protocol.get("peg_delta_sum_xyz") or source_protocol.get("peg_delta_first_xyz"),
                                dtype=np.float64,
                            ).reshape(3)
                            return {
                                "ok": True,
                                "classification": "peg_perturb_calibration_trial_complete",
                                "method_evidence_allowed": False,
                                "physical_insertion_success": False,
                                "source_key": source_protocol.get("sample_id"),
                                "trigger_step": step_idx,
                                "force_scale": float(force_scale),
                                "force_steps": int(force_steps),
                                "settle_steps": int(args.settle_steps),
                                "force_xyz": force_xyz.astype(float).tolist(),
                                "expected_delta_xyz": expected_delta.astype(float).tolist(),
                                "observed_delta_xyz": observed_delta.astype(float).tolist(),
                                "observed_delta_fraction": (
                                    norm3(observed_delta) / norm3(expected_delta)
                                    if norm3(expected_delta) > 1.0e-12
                                    else None
                                ),
                                "observed_delta_cosine": cosine3(expected_delta, observed_delta),
                                "state_before_perturb": state_before_perturb,
                                "state_after_perturb": state_after,
                                "per_step": per_step,
                                "reward": jsonable(reward),
                                "terminated": jsonable(terminated),
                                "truncated": jsonable(truncated),
                            }
                        if len(gate_rejections) < 64:
                            gate_rejections.append({"env_step": step_idx, "gate": gate_report})
                    obs, reward, terminated, truncated, info = envs.step(np.asarray(action_vec, dtype=np.float32).reshape(1, -1))
                    state = eval_state(base_env)
                    trace.append(
                        {
                            "env_step": step_idx,
                            "stage": "dp_static_prefix",
                            "action": np.asarray(action_vec, dtype=float).reshape(-1).tolist(),
                            "live_eval": state,
                            "reward": jsonable(reward),
                            "terminated": jsonable(terminated),
                            "truncated": jsonable(truncated),
                        }
                    )
            return {
                "ok": False,
                "classification": "source_h5_motion_gate_never_reached",
                "force_scale": float(force_scale),
                "force_steps": int(force_steps),
                "gate_rejections": gate_rejections,
                "trace_len": len(trace),
            }
    finally:
        envs.close()


def main() -> int:
    require_allocation()
    args = tyro.cli(Args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_protocol = load_source_h5_protocol(args.source_h5_path)
    if args.source_key and args.source_key != source_protocol.get("sample_id"):
        raise RuntimeError(f"source_key_mismatch requested={args.source_key} h5={source_protocol.get('sample_id')}")
    if not source_protocol_is_peg_perturb(source_protocol):
        raise RuntimeError(f"source_key_is_not_peg_perturb scenario={source_protocol.get('scenario')}")
    args.seed = int(source_protocol["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    ms_train = importlib.import_module("train")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    train_args = build_train_args(ckpt.get("args", {}), args, ms_train)
    ms_train.args = train_args
    ms_train.device = device
    envs_for_agent = make_env(train_args)
    try:
        agent = ms_train.Agent(envs_for_agent, train_args).to(device)
        state_key = "ema_agent" if args.use_ema else "agent"
        agent.load_state_dict(ckpt[state_key])
        agent.eval()
    finally:
        envs_for_agent.close()

    trials = []
    for force_scale in parse_float_csv(args.force_scales_csv):
        for force_steps in parse_int_csv(args.force_steps_csv):
            trial = run_one_trial(
                args,
                train_args=train_args,
                agent=agent,
                device=device,
                source_protocol=source_protocol,
                force_scale=force_scale,
                force_steps=force_steps,
            )
            trials.append(trial)

    expected = np.asarray(source_protocol.get("peg_delta_sum_xyz"), dtype=np.float64).reshape(3)
    viable = [
        trial
        for trial in trials
        if trial.get("ok")
        and trial.get("observed_delta_fraction") is not None
        and float(trial["observed_delta_fraction"]) >= 0.2
        and trial.get("observed_delta_cosine") is not None
        and float(trial["observed_delta_cosine"]) >= 0.5
    ]
    best = None
    if viable:
        best = min(
            viable,
            key=lambda trial: float(np.linalg.norm(np.asarray(trial["observed_delta_xyz"], dtype=np.float64) - expected)),
        )
    report = {
        "schema": "phase03_peg_perturb_force_calibration_v1",
        "classification": "peg_perturb_force_calibration_complete",
        "method_evidence_allowed": False,
        "physical_insertion_success": False,
        "source_h5_path": str(Path(args.source_h5_path).resolve()),
        "source_key": source_protocol.get("sample_id"),
        "source_protocol": source_protocol,
        "force_scales": parse_float_csv(args.force_scales_csv),
        "force_steps": parse_int_csv(args.force_steps_csv),
        "settle_steps": int(args.settle_steps),
        "trials": trials,
        "best_viable_trial": best,
    }
    write_json(output_dir / "calibration.json", report)
    (output_dir / "classification.txt").write_text(
        "phase03_status=peg_perturb_force_calibration_complete\n"
        "method_evidence_allowed=false\n"
        "physical_insertion_success=false\n"
        f"source_key={source_protocol.get('sample_id')}\n"
        f"best_viable_force_scale={None if best is None else best.get('force_scale')}\n"
        f"best_viable_force_steps={None if best is None else best.get('force_steps')}\n"
    )
    print(json.dumps(jsonable(report), indent=2, sort_keys=True))
    return 0 if best is not None else 3


if __name__ == "__main__":
    raise SystemExit(main())
