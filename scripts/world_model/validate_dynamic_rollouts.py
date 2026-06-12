#!/usr/bin/env python3
"""Validate dynamic task-rebinding rollout H5 files.

This script is intentionally lightweight: it only reads metadata and arrays from
recorded H5 files. It does not step ManiSkill or render.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tyro


DYNAMIC_SCENARIOS = {
    "hole_move_stop",
    "hole_constant",
    "hole_reverse",
    "hole_move_stop_large",
    "hole_constant_large",
    "hole_reverse_large",
    "hole_sine_large",
    "hole_continuous_insert_large",
    "hole_late_shift_large",
    "peg_disturb",
    "peg_drop",
}
HOLE_SCENARIOS = {
    "hole_move_stop",
    "hole_constant",
    "hole_reverse",
    "hole_move_stop_large",
    "hole_constant_large",
    "hole_reverse_large",
    "hole_sine_large",
    "hole_continuous_insert_large",
    "hole_late_shift_large",
}
PEG_EVENT_SCENARIOS = {"peg_disturb", "peg_drop"}


@dataclass
class Args:
    paths: list[str]
    output_json: str | None = None
    output_markdown: str | None = None
    large_hole_motion_warn_m: float = 0.5


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _summary(group: h5py.Group) -> dict[str, Any]:
    raw = group.attrs.get("summary_json")
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _arr(group: h5py.Group, name: str) -> np.ndarray:
    return np.asarray(group[name])


def _first_true(values: np.ndarray) -> int:
    flat = np.asarray(values).reshape(-1).astype(bool)
    return int(np.argmax(flat)) if flat.any() else -1


def _validate_group(
    path: Path,
    trajectory: str,
    group: h5py.Group,
    large_hole_motion_warn_m: float,
) -> dict[str, Any]:
    summary = _summary(group)
    scenario = summary.get("scenario", "unknown")
    warnings: list[str] = []

    slots = group["slots"]
    perturb = group["perturb"]
    inserted = _arr(slots, "inserted").reshape(-1).astype(bool)
    grasped = _arr(slots, "grasped").reshape(-1).astype(bool)
    hole = _arr(slots, "hole_pose")[:, :3]
    peg = _arr(slots, "peg_pose")[:, :3]
    tcp = _arr(slots, "tcp_pose")[:, :3]
    trigger_steps = _arr(perturb, "trigger_step").reshape(-1)
    valid_trigger_steps = trigger_steps[trigger_steps >= 0]
    trigger_step = int(valid_trigger_steps[0]) if len(valid_trigger_steps) else -1
    first_insert_step = _first_true(inserted)
    summary_success_once = bool(summary.get("success_once", False))
    summary_success_at_end = bool(summary.get("success_at_end", False))

    success_once_actual = bool(inserted.any())
    success_at_end_actual = bool(inserted[-1]) if len(inserted) else False
    if summary_success_once != success_once_actual:
        warnings.append("summary_success_once_mismatch")
    if summary_success_at_end != success_at_end_actual:
        warnings.append("summary_success_at_end_mismatch")

    if scenario in DYNAMIC_SCENARIOS and trigger_step < 0:
        warnings.append("dynamic_scenario_missing_trigger")
    if trigger_step >= 0 and first_insert_step >= 0 and trigger_step >= first_insert_step:
        warnings.append("trigger_not_before_first_insert")

    hole_motion = hole[-1] - hole[0]
    hole_motion_norm = float(np.linalg.norm(hole_motion))
    max_hole_step = float(np.linalg.norm(np.diff(hole, axis=0), axis=1).max()) if len(hole) > 1 else 0.0
    hole_delta = _arr(perturb, "hole_delta_cumulative")
    hole_delta_final = hole_delta[-1] if len(hole_delta) else np.zeros(3, dtype=np.float32)
    if scenario in HOLE_SCENARIOS and hole_motion_norm < 1e-4:
        warnings.append("hole_scenario_without_hole_motion")
    if scenario not in HOLE_SCENARIOS and hole_motion_norm > 1e-4:
        warnings.append("non_hole_scenario_has_hole_motion")
    if hole_motion_norm > large_hole_motion_warn_m:
        warnings.append("large_hole_motion_check_reachability")

    peg_delta = _arr(perturb, "peg_delta_applied")
    peg_delta_norm = np.linalg.norm(peg_delta, axis=1) if len(peg_delta) else np.zeros((0,))
    peg_event_steps = np.where(peg_delta_norm > 1e-6)[0]
    if scenario in PEG_EVENT_SCENARIOS and len(peg_event_steps) == 0:
        warnings.append("peg_event_missing")
    if scenario not in PEG_EVENT_SCENARIOS and len(peg_event_steps) > 0:
        warnings.append("unexpected_peg_event")

    if "env_states" not in group:
        warnings.append("missing_env_states")

    trigger_window: dict[str, Any] = {}
    if trigger_step >= 0 and trigger_step < len(grasped):
        lo = max(0, trigger_step - 2)
        hi = min(len(grasped), trigger_step + 6)
        trigger_window = {
            "lo": lo,
            "hi": hi,
            "grasped": grasped[lo:hi].astype(int),
            "peg_z": peg[lo:hi, 2],
            "tcp_z": tcp[lo:hi, 2],
        }

    return {
        "path": str(path),
        "trajectory": trajectory,
        "scenario": scenario,
        "seed": int(summary.get("seed", -1)),
        "steps": int(summary.get("steps", len(group["actions"]))),
        "trigger_step": trigger_step,
        "trigger_reason": summary.get("trigger_reason", ""),
        "first_insert_step": first_insert_step,
        "success_once": success_once_actual,
        "success_at_end": success_at_end_actual,
        "event_before_first_insert": bool(
            trigger_step >= 0 and (first_insert_step < 0 or trigger_step < first_insert_step)
        ),
        "hole_start": hole[0],
        "hole_end": hole[-1],
        "hole_motion": hole_motion,
        "hole_motion_norm": hole_motion_norm,
        "max_hole_step": max_hole_step,
        "hole_delta_final": hole_delta_final,
        "peg_event_steps": peg_event_steps.astype(np.int32),
        "peg_delta_first": peg_delta[peg_event_steps[0]] if len(peg_event_steps) else None,
        "has_env_states": "env_states" in group,
        "action_shape": tuple(group["actions"].shape),
        "obs_stack_shape": tuple(group["obs_stack"].shape),
        "trigger_window": trigger_window,
        "warnings": warnings,
    }


def validate_file(path: Path, large_hole_motion_warn_m: float) -> list[dict[str, Any]]:
    with h5py.File(path, "r") as h5:
        traj_names = sorted([k for k in h5.keys() if k.startswith("traj_")])
        if not traj_names:
            return [
                {
                    "path": str(path),
                    "trajectory": "",
                    "scenario": "unknown",
                    "seed": -1,
                    "trigger_step": -1,
                    "first_insert_step": -1,
                    "success_at_end": False,
                    "hole_motion_norm": 0.0,
                    "peg_event_steps": np.asarray([], dtype=np.int32),
                    "warnings": ["no_traj_groups"],
                    "error": "no traj_* groups",
                }
            ]
        return [
            _validate_group(path, traj_name, h5[traj_name], large_hole_motion_warn_m)
            for traj_name in traj_names
        ]


def _write_markdown(results: list[dict[str, Any]], path: Path):
    lines = [
        "# Dynamic Rollout Validation",
        "",
        "| scenario | seed | trigger | first insert | success end | hole motion m | peg event | warnings |",
        "|---|---:|---:|---:|---|---:|---|---|",
    ]
    for item in results:
        peg_event = ",".join(map(str, item["peg_event_steps"])) if len(item["peg_event_steps"]) else ""
        warnings_text = ", ".join(item["warnings"])
        lines.append(
            "| {scenario} | {seed} | {trigger_step} | {first_insert_step} | {success_at_end} | {hole_motion_norm:.4f} | {peg_event} | {warnings_text} |".format(
                **item, peg_event=peg_event, warnings_text=warnings_text
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    args = tyro.cli(Args)
    paths = [Path(p) for p in args.paths]
    results = [
        item
        for path in paths
        for item in validate_file(path, args.large_hole_motion_warn_m)
    ]
    print(json.dumps(_jsonable(results), indent=2, sort_keys=True))
    if args.output_json is not None:
        Path(args.output_json).write_text(json.dumps(_jsonable(results), indent=2, sort_keys=True))
    if args.output_markdown is not None:
        _write_markdown(results, Path(args.output_markdown))


if __name__ == "__main__":
    main()
