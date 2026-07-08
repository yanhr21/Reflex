#!/usr/bin/env python3
"""Active dynamic-scene adapter for dataset B/C/D/E collection.

This module is source-audited before any dynamic dataset runner is allowed to
launch. It defines the only active path for commanded target / hole motion:
continuous kinematic target commands plus an explicit motion_trace. If the
runtime SAPIEN object does not expose a kinematic-target command, callers must
fail the smoke instead of falling back to direct pose writes.

The adapter does not collect data by itself. It is imported by future
in-allocation runners after `require_dataset_runtime_context.sh` has accepted a
Slurm compute step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Literal

import numpy as np


DatasetClass = Literal[
    "B_dynamic_rgb_observation",
    "C_frozen_dp_dynamic_failure",
    "D_future_frame_cooperation_teacher",
    "E_cosmos_predicted_cooperation",
]

MotionFamily = Literal[
    "constant_lr",
    "constant_fb",
    "reverse",
    "move_stop",
    "sine",
    "continuous",
    "peg_disturb",
]


@dataclass(frozen=True)
class AdapterManifestFields:
    dataset_class: DatasetClass
    allowed_losses: tuple[str, ...]
    disallowed_losses: tuple[str, ...]
    rgb_required: bool = True
    video_required: bool = True
    render_required: bool = True
    state_intervention: bool = False
    snap_or_teleport: bool = False
    target_motion_trace_required: bool = True
    hole_motion_trace_required: bool = True
    motion_trace_schema: str = "step,scenario,target_pose,target_delta,command_kind"


@dataclass(frozen=True)
class DynamicMotionSpec:
    scenario: MotionFamily
    start_step: int
    duration_steps: int
    delta_xyz: tuple[float, float, float]
    reverse_fraction: float = 0.5
    sine_cycles: float = 0.5
    min_step_delta_m: float = 0.0
    max_step_delta_m: float = 0.015

    def validate(self) -> None:
        if self.start_step < 0:
            raise ValueError("start_step must be nonnegative")
        if self.duration_steps <= 0:
            raise ValueError("duration_steps must be positive")
        if self.max_step_delta_m <= 0:
            raise ValueError("max_step_delta_m must be positive")
        if self.reverse_fraction <= 0.0 or self.reverse_fraction >= 1.0:
            raise ValueError("reverse_fraction must be inside (0, 1)")


@dataclass(frozen=True)
class MotionCommand:
    step: int
    scenario: MotionFamily
    command_kind: str
    alpha: float
    target_p: tuple[float, float, float]
    target_q: tuple[float, float, float, float]
    target_delta_xyz: tuple[float, float, float]
    instantaneous_delta_m: float

    def to_trace_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["motion_trace"] = True
        row["target_motion_trace"] = True
        row["hole_motion_trace"] = True
        row["state_intervention"] = False
        row["snap_or_teleport"] = False
        return row


LOSS_RULES: dict[DatasetClass, AdapterManifestFields] = {
    "B_dynamic_rgb_observation": AdapterManifestFields(
        dataset_class="B_dynamic_rgb_observation",
        allowed_losses=(
            "cosmos_dynamic_future",
            "target_frame_readout",
            "trajectory_consistency",
            "uncertainty",
        ),
        disallowed_losses=("positive_dp_bc_from_failed_actions", "final_method_evidence"),
    ),
    "C_frozen_dp_dynamic_failure": AdapterManifestFields(
        dataset_class="C_frozen_dp_dynamic_failure",
        allowed_losses=(
            "negative_classification",
            "discrepancy",
            "infeasible_no_progress",
            "contrastive",
        ),
        disallowed_losses=("positive_dp_bc_from_failed_actions", "target_assisted_success"),
    ),
    "D_future_frame_cooperation_teacher": AdapterManifestFields(
        dataset_class="D_future_frame_cooperation_teacher",
        allowed_losses=(
            "adapter_residual",
            "moving_frame_conditioning",
            "phase_timing",
            "relative_velocity_at_contact",
        ),
        disallowed_losses=("deployed_method_success_claim", "hidden_future_controller"),
    ),
    "E_cosmos_predicted_cooperation": AdapterManifestFields(
        dataset_class="E_cosmos_predicted_cooperation",
        allowed_losses=(
            "adapter_robustness",
            "uncertainty_conditioned_control",
            "live_method_evaluation",
        ),
        disallowed_losses=("hidden_ground_truth_future", "target_assisted_success"),
    ),
}


def _smoothstep(x: float) -> float:
    x = min(max(float(x), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _scenario_alpha(scenario: MotionFamily, local_step: int, spec: DynamicMotionSpec) -> float:
    denom = max(float(spec.duration_steps - 1), 1.0)
    t = min(max(float(local_step) / denom, 0.0), 1.0)
    if scenario in ("constant_lr", "constant_fb", "continuous"):
        return t
    if scenario == "move_stop":
        return _smoothstep(min(t / 0.65, 1.0))
    if scenario == "reverse":
        split = spec.reverse_fraction
        if t <= split:
            return t / split
        return 1.0 - 0.5 * ((t - split) / (1.0 - split))
    if scenario == "sine":
        return 0.5 - 0.5 * math.cos(2.0 * math.pi * spec.sine_cycles * t)
    if scenario == "peg_disturb":
        return _smoothstep(t)
    raise ValueError(f"unknown motion scenario: {scenario}")


def _as_tuple3(value: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _as_tuple4(value: np.ndarray) -> tuple[float, float, float, float]:
    arr = np.asarray(value, dtype=np.float64).reshape(4)
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def build_motion_command(
    *,
    step: int,
    spec: DynamicMotionSpec,
    initial_p: np.ndarray,
    initial_q: np.ndarray,
    previous_p: np.ndarray | None = None,
) -> MotionCommand | None:
    spec.validate()
    if step < spec.start_step:
        return None
    local_step = step - spec.start_step
    if local_step >= spec.duration_steps:
        local_step = spec.duration_steps - 1
    alpha = _scenario_alpha(spec.scenario, local_step, spec)
    delta = np.asarray(spec.delta_xyz, dtype=np.float64) * float(alpha)
    target_p = np.asarray(initial_p, dtype=np.float64).reshape(3) + delta
    target_q = np.asarray(initial_q, dtype=np.float64).reshape(4)
    if previous_p is None:
        instantaneous = float(np.linalg.norm(delta))
    else:
        instantaneous = float(np.linalg.norm(target_p - np.asarray(previous_p, dtype=np.float64).reshape(3)))
    if instantaneous > spec.max_step_delta_m + 1e-9:
        raise ValueError(
            f"motion step too large: {instantaneous:.6f}m > {spec.max_step_delta_m:.6f}m"
        )
    if instantaneous < spec.min_step_delta_m and local_step > 0:
        command_kind = "kinematic_target_hold"
    else:
        command_kind = "kinematic_target_motion"
    return MotionCommand(
        step=int(step),
        scenario=spec.scenario,
        command_kind=command_kind,
        alpha=float(alpha),
        target_p=_as_tuple3(target_p),
        target_q=_as_tuple4(target_q),
        target_delta_xyz=_as_tuple3(delta),
        instantaneous_delta_m=instantaneous,
    )


def _iter_kinematic_bodies(actor: Any) -> list[Any]:
    bodies = getattr(actor, "_bodies", None)
    if bodies is None:
        bodies = [actor]
    return list(bodies)


def _resolve_kinematic_target_writer(body: Any) -> Callable[[Any], None]:
    writer = getattr(body, "set_kinematic_target", None)
    if callable(writer):
        return writer
    if hasattr(type(body), "kinematic_target"):
        return lambda pose: setattr(body, "kinematic_target", pose)
    raise RuntimeError(
        "runtime body does not expose a kinematic target command; "
        "refuse dynamic collection instead of falling back to direct pose writes"
    )


def command_kinematic_target(actor: Any, sapien_pose: Any) -> None:
    for body in _iter_kinematic_bodies(actor):
        _resolve_kinematic_target_writer(body)(sapien_pose)


def command_target_from_motion(actor: Any, sapien_pose_factory: Callable[..., Any], command: MotionCommand) -> dict[str, Any]:
    pose = sapien_pose_factory(command.target_p, command.target_q)
    command_kinematic_target(actor, pose)
    return command.to_trace_row()


def manifest_fields(dataset_class: DatasetClass) -> dict[str, Any]:
    return asdict(LOSS_RULES[dataset_class])


def validate_trace_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "target_motion_trace_valid": False,
            "reason": "empty_motion_trace",
            "state_intervention": False,
            "snap_or_teleport": False,
        }
    has_intervention = any(bool(row.get("state_intervention", False)) for row in rows)
    has_snap = any(bool(row.get("snap_or_teleport", False)) for row in rows)
    max_step = max(float(row.get("instantaneous_delta_m", 0.0)) for row in rows)
    return {
        "target_motion_trace_valid": not has_intervention and not has_snap,
        "row_count": len(rows),
        "max_instantaneous_delta_m": max_step,
        "state_intervention": has_intervention,
        "snap_or_teleport": has_snap,
    }


def main() -> None:
    print("active_dynamic_peg_adapter=true")
    print("supports=kinematic_target_command")
    print("requires=compute_node_runtime_validation")
    print("motion_trace_schema=step,scenario,target_pose,target_delta,command_kind")
    print("rgb_video_render_required=true")


if __name__ == "__main__":
    main()
