#!/usr/bin/env python3
"""Strict read-only audit for merged fix3 source H5 path lists.

This is a structural/metadata gate for the v7 full1000 source set. It does not
step ManiSkill and does not render video. Passing this script is necessary but
not sufficient for approval: rendered videos/framebooks still need direct
inspection before WAM export or Cosmos3 SFT.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_QUOTAS = {
    "hole_late_move_stop": 70,
    "hole_late_constant": 90,
    "hole_late_reverse": 100,
    "hole_late_sine": 90,
    "hole_late_continuous_insert": 120,
    "hole_late_fast_shift": 120,
    "none": 160,
    "peg_drop": 150,
    "peg_disturb": 100,
}

HOLE_MOTION_SCENARIOS = {
    "hole_late_move_stop",
    "hole_late_constant",
    "hole_late_reverse",
    "hole_late_sine",
    "hole_late_continuous_insert",
    "hole_late_fast_shift",
}
PEG_PERTURB_SCENARIOS = {"peg_drop", "peg_disturb"}
SCENARIOS = set(DEFAULT_QUOTAS)
FILENAME_RE = re.compile(
    r"^(?P<scenario>.+)_seed(?P<seed>\d+)(?:_pseed(?P<policy_seed>\d+))?_idx\d+\.h5$"
)


@dataclass
class Args:
    paths_file: Path
    output_root: Path
    quotas: str = ""
    expected_frames: int = 301
    expected_actions: int = 300
    expected_action_dim: int = 7
    min_hole_motion_m: float = 0.22
    static_hole_motion_max_m: float = 0.005
    motion_onset_threshold_m: float = 1e-5
    peg_event_threshold_m: float = 1e-8
    max_failure_examples: int = 80
    fail_on_count_mismatch: bool = True


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _parse_quotas(text: str) -> dict[str, int]:
    if not text.strip():
        return dict(DEFAULT_QUOTAS)
    quotas: dict[str, int] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"quota entry must be SCENARIO=COUNT, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in DEFAULT_QUOTAS:
            raise ValueError(f"unknown scenario in quotas: {key!r}")
        quotas[key] = int(value)
    missing = sorted(set(DEFAULT_QUOTAS) - set(quotas))
    if missing:
        raise ValueError(f"missing scenarios in quotas: {missing}")
    return quotas


def _load_summary(group: h5py.Group) -> dict[str, Any]:
    raw = group.attrs.get("source_summary_json") or group.attrs.get("summary_json") or "{}"
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        data = json.loads(str(raw))
    except json.JSONDecodeError:
        return {"_raw": str(raw)}
    return data if isinstance(data, dict) else {"_raw": data}


def _first_true(mask: np.ndarray) -> int:
    idx = np.flatnonzero(np.asarray(mask).reshape(-1).astype(bool))
    return int(idx[0]) if idx.size else -1


def _finite(name: str, value: np.ndarray, failures: list[str]) -> None:
    if not np.all(np.isfinite(value)):
        failures.append(f"{name}_contains_nonfinite")


def _scenario_from_filename(path: Path) -> tuple[str, int | None, int | None]:
    match = FILENAME_RE.match(path.name)
    if not match:
        return "unknown", None, None
    policy_seed = match.group("policy_seed")
    return (
        match.group("scenario"),
        int(match.group("seed")),
        None if policy_seed is None else int(policy_seed),
    )


def _as_array(group: h5py.Group, key: str, failures: list[str]) -> np.ndarray | None:
    if key not in group:
        failures.append(f"missing_{key}")
        return None
    arr = np.asarray(group[key])
    _finite(key.replace("/", "_"), arr, failures)
    return arr


def _audit_one(path: Path, index: int, args: Args) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    scenario_from_name, seed_from_name, policy_seed_from_name = _scenario_from_filename(path)
    if scenario_from_name not in SCENARIOS:
        failures.append("filename_scenario_unknown")
    if not path.exists():
        return {
            "index": index,
            "path": str(path),
            "scenario": scenario_from_name,
            "seed": seed_from_name,
            "policy_seed": policy_seed_from_name,
            "failures": failures + ["path_missing"],
            "warnings": warnings,
        }

    try:
        with h5py.File(path, "r") as h5:
            traj_names = sorted(key for key in h5.keys() if key.startswith("traj_"))
            if len(traj_names) != 1:
                return {
                    "index": index,
                    "path": str(path),
                    "scenario": scenario_from_name,
                    "seed": seed_from_name,
                    "policy_seed": policy_seed_from_name,
                    "failures": failures + [f"expected_one_traj_group_got_{len(traj_names)}"],
                    "warnings": warnings,
                }
            group = h5[traj_names[0]]
            summary = _load_summary(group)
            scenario = str(summary.get("scenario") or scenario_from_name)
            if scenario != scenario_from_name:
                failures.append("summary_scenario_filename_mismatch")
            if scenario not in SCENARIOS:
                failures.append("summary_scenario_unknown")

            for required_group in ("slots", "perturb", "env_states"):
                if required_group not in group:
                    failures.append(f"missing_{required_group}_group")
            if "slots" not in group or "perturb" not in group:
                return {
                    "index": index,
                    "path": str(path),
                    "trajectory": traj_names[0],
                    "scenario": scenario,
                    "seed": seed_from_name,
                    "policy_seed": policy_seed_from_name,
                    "failures": failures,
                    "warnings": warnings,
                }

            actions = _as_array(group, "actions", failures)
            obs_current = _as_array(group, "obs_current", failures)
            source_frame_indices = _as_array(group, "source_frame_indices", failures)
            slots = group["slots"]
            perturb = group["perturb"]
            hole = _as_array(slots, "hole_pose", failures)
            peg = _as_array(slots, "peg_pose", failures)
            tcp = _as_array(slots, "tcp_pose", failures)
            peg_head = _as_array(slots, "peg_head_at_hole", failures)
            hole_radius = _as_array(slots, "hole_radius", failures)
            inserted = _as_array(slots, "inserted", failures)
            grasped = _as_array(slots, "grasped", failures)
            robust_held = _as_array(slots, "robust_held", failures)
            hole_delta = _as_array(perturb, "hole_delta_applied", failures)
            peg_delta = _as_array(perturb, "peg_delta_applied", failures)
            triggered_arr = _as_array(perturb, "triggered", failures)
            trigger_step_arr = _as_array(perturb, "trigger_step", failures)

            if actions is not None and actions.shape != (
                int(args.expected_actions),
                int(args.expected_action_dim),
            ):
                failures.append(f"actions_shape_{tuple(actions.shape)}")
            if obs_current is not None and int(obs_current.shape[0]) != int(args.expected_frames):
                failures.append(f"obs_current_frames_{obs_current.shape[0]}")
            if source_frame_indices is not None:
                expected_indices = np.arange(int(args.expected_frames), dtype=np.int64)
                if source_frame_indices.shape != expected_indices.shape or not np.array_equal(
                    source_frame_indices.astype(np.int64), expected_indices
                ):
                    failures.append("source_frame_indices_not_0_to_300")

            frame_arrays = {
                "hole_pose": hole,
                "peg_pose": peg,
                "tcp_pose": tcp,
                "peg_head_at_hole": peg_head,
                "hole_radius": hole_radius,
                "inserted": inserted,
                "grasped": grasped,
                "robust_held": robust_held,
            }
            for name, arr in frame_arrays.items():
                if arr is not None and int(arr.shape[0]) != int(args.expected_frames):
                    failures.append(f"{name}_frames_{arr.shape[0]}")

            step_arrays = {
                "hole_delta_applied": hole_delta,
                "peg_delta_applied": peg_delta,
                "triggered": triggered_arr,
                "trigger_step": trigger_step_arr,
            }
            for name, arr in step_arrays.items():
                if arr is not None and int(arr.shape[0]) != int(args.expected_actions):
                    failures.append(f"{name}_steps_{arr.shape[0]}")

            if any(arr is None for arr in (hole, peg, tcp, peg_head, hole_radius, inserted, grasped, robust_held)):
                return {
                    "index": index,
                    "path": str(path),
                    "trajectory": traj_names[0],
                    "scenario": scenario,
                    "seed": seed_from_name,
                    "policy_seed": policy_seed_from_name,
                    "failures": failures,
                    "warnings": warnings,
                }

            inserted_bool = np.asarray(inserted).reshape(-1).astype(bool)
            grasped_bool = np.asarray(grasped).reshape(-1).astype(bool)
            robust_bool = np.asarray(robust_held).reshape(-1).astype(bool)
            final_radius = float(np.asarray(hole_radius).reshape(-1)[-1])
            final_head = np.asarray(peg_head[-1, :3], dtype=np.float64)
            final_inserted_from_head = bool(
                final_head[0] >= -0.015
                and abs(final_head[1]) <= final_radius
                and abs(final_head[2]) <= final_radius
            )

            first_insert = _first_true(inserted_bool)
            first_grasp = _first_true(grasped_bool)
            first_robust = _first_true(robust_bool)
            hole_motion_vec = np.asarray(hole[-1, :3] - hole[0, :3], dtype=np.float64)
            hole_motion_norm = float(np.linalg.norm(hole_motion_vec))
            hole_step_norm = np.linalg.norm(np.diff(hole[:, :3], axis=0), axis=1)
            motion_steps = np.flatnonzero(hole_step_norm > float(args.motion_onset_threshold_m))
            first_motion = int(motion_steps[0] + 1) if motion_steps.size else -1
            last_motion = int(motion_steps[-1] + 1) if motion_steps.size else -1
            peg_step_norm = np.linalg.norm(np.asarray(peg_delta), axis=1) if peg_delta is not None else np.zeros(0)
            peg_event_steps = np.flatnonzero(peg_step_norm > float(args.peg_event_threshold_m))
            first_peg_event = int(peg_event_steps[0]) if peg_event_steps.size else -1
            triggered = bool(np.asarray(triggered_arr).reshape(-1).astype(bool).any()) if triggered_arr is not None else False
            valid_trigger_steps = (
                np.asarray(trigger_step_arr).reshape(-1).astype(np.int64)
                if trigger_step_arr is not None
                else np.asarray([], dtype=np.int64)
            )
            valid_trigger_steps = valid_trigger_steps[valid_trigger_steps >= 0]
            trigger_step = int(valid_trigger_steps[0]) if valid_trigger_steps.size else int(summary.get("trigger_step", -1) or -1)

            if not bool(inserted_bool[-1]):
                failures.append("inserted_end_false")
            if not final_inserted_from_head:
                failures.append("final_peg_head_not_inside_strict_insert_region")
            if not bool(summary.get("success_at_end", False)):
                failures.append("summary_success_at_end_false")
            if not bool(summary.get("inserted_end", False)):
                failures.append("summary_inserted_end_false")
            if not bool(summary.get("live_success_end", False)):
                failures.append("summary_live_success_end_false")
            if first_insert < 0:
                failures.append("never_inserted")
            if first_grasp < 0:
                failures.append("never_grasped")
            if first_robust < 0 and scenario != "none":
                failures.append("dynamic_never_robust_held")
            if summary.get("seed") is not None and seed_from_name is not None and int(summary.get("seed")) != int(seed_from_name):
                failures.append("summary_seed_filename_mismatch")

            summary_final_head = np.asarray(summary.get("final_peg_head_at_hole", []), dtype=np.float64)
            if summary_final_head.size >= 3 and np.linalg.norm(summary_final_head[:3] - final_head) > 1e-4:
                warnings.append("summary_final_head_differs_from_slots")

            if scenario in HOLE_MOTION_SCENARIOS:
                if not triggered or trigger_step < 0:
                    failures.append("hole_motion_missing_trigger")
                if hole_motion_norm < float(args.min_hole_motion_m):
                    failures.append("hole_motion_too_small")
                if first_motion < 0:
                    failures.append("hole_motion_missing")
                if first_grasp >= 0 and first_motion >= 0 and first_motion < first_grasp:
                    failures.append("hole_motion_before_grasp")
                if first_robust >= 0 and first_motion >= 0 and first_motion < first_robust:
                    failures.append("hole_motion_before_robust_hold")
                if first_insert >= 0 and trigger_step >= 0 and first_insert <= trigger_step:
                    failures.append("inserted_before_or_at_trigger")
                if first_insert >= 0 and first_motion >= 0 and first_insert < first_motion:
                    failures.append("inserted_before_target_motion")
                if peg_event_steps.size:
                    failures.append("hole_motion_scenario_has_peg_delta")
            elif scenario == "none":
                if hole_motion_norm > float(args.static_hole_motion_max_m):
                    failures.append("none_has_hole_motion")
                if peg_event_steps.size:
                    failures.append("none_has_peg_delta")
            elif scenario in PEG_PERTURB_SCENARIOS:
                if not triggered or trigger_step < 0:
                    failures.append("peg_scenario_missing_trigger")
                if hole_motion_norm > float(args.static_hole_motion_max_m):
                    failures.append("peg_scenario_has_hole_motion")
                if not peg_event_steps.size:
                    failures.append("peg_scenario_missing_peg_delta")
                if first_grasp >= 0 and first_peg_event >= 0 and first_peg_event < first_grasp:
                    failures.append("peg_event_before_grasp")
                if first_robust >= 0 and first_peg_event >= 0 and first_peg_event < first_robust:
                    failures.append("peg_event_before_robust_hold")

            return {
                "index": index,
                "path": str(path),
                "trajectory": traj_names[0],
                "scenario": scenario,
                "seed": seed_from_name,
                "policy_seed": policy_seed_from_name,
                "num_frames": int(np.asarray(hole).shape[0]),
                "num_actions": int(actions.shape[0]) if actions is not None else -1,
                "action_dim": int(actions.shape[1]) if actions is not None and actions.ndim >= 2 else -1,
                "success_at_end": bool(summary.get("success_at_end", False)),
                "inserted_end": bool(inserted_bool[-1]),
                "final_inserted_from_head": final_inserted_from_head,
                "first_grasp_step": int(first_grasp),
                "first_robust_hold_step": int(first_robust),
                "triggered": bool(triggered),
                "trigger_step": int(trigger_step),
                "first_target_motion_step": int(first_motion),
                "last_target_motion_frame": int(last_motion),
                "first_peg_perturb_step": int(first_peg_event),
                "first_insert_step": int(first_insert),
                "target_motion_norm_m": float(hole_motion_norm),
                "target_motion_xyz": hole_motion_vec.astype(float).tolist(),
                "final_peg_head_at_hole": final_head.astype(float).tolist(),
                "final_hole_radius": final_radius,
                "failures": failures,
                "warnings": warnings,
            }
    except Exception as exc:
        return {
            "index": index,
            "path": str(path),
            "scenario": scenario_from_name,
            "seed": seed_from_name,
            "policy_seed": policy_seed_from_name,
            "failures": failures + ["read_exception"],
            "warnings": warnings,
            "error": repr(exc),
        }


def _summarize(records: list[dict[str, Any]], quotas: dict[str, int], args: Args) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for scenario in quotas:
        rows = [r for r in records if r.get("scenario") == scenario]
        failures = [r for r in rows if r.get("failures")]
        motions = np.asarray([r.get("target_motion_norm_m", np.nan) for r in rows], dtype=np.float64)
        motions = motions[np.isfinite(motions)]
        summary[scenario] = {
            "count": len(rows),
            "quota": int(quotas[scenario]),
            "count_ok": len(rows) == int(quotas[scenario]),
            "num_failed": len(failures),
            "target_motion_norm_m": {
                "min": float(motions.min()) if motions.size else None,
                "mean": float(motions.mean()) if motions.size else None,
                "max": float(motions.max()) if motions.size else None,
            },
            "failure_examples": [
                {
                    "index": r.get("index"),
                    "path": r.get("path"),
                    "seed": r.get("seed"),
                    "failures": r.get("failures"),
                    "target_motion_norm_m": r.get("target_motion_norm_m"),
                    "final_peg_head_at_hole": r.get("final_peg_head_at_hole"),
                }
                for r in failures[: int(args.max_failure_examples)]
            ],
        }
    unknown = sorted({str(r.get("scenario")) for r in records if str(r.get("scenario")) not in quotas})
    if unknown:
        summary["_unknown_scenarios"] = unknown
    return summary


def _write_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Fix3 Merged Source H5 Audit",
        "",
        "Boundary: read-only structural/metadata audit. Passing this file is not video approval and does not permit SFT by itself.",
        "",
        f"strict_ok: `{audit['strict_ok']}`",
        f"num_paths: `{audit['num_paths']}`",
        f"num_records: `{audit['num_records']}`",
        f"num_failed_records: `{audit['num_failed_records']}`",
        "",
        "| scenario | count/quota | failed | target motion min/mean/max m |",
        "|---|---:|---:|---|",
    ]
    for scenario, item in audit["summary_by_scenario"].items():
        if scenario.startswith("_"):
            continue
        motion = item["target_motion_norm_m"]
        lines.append(
            "| {scenario} | {count}/{quota} | {failed} | {mmin}/{mmean}/{mmax} |".format(
                scenario=scenario,
                count=item["count"],
                quota=item["quota"],
                failed=item["num_failed"],
                mmin="na" if motion["min"] is None else f"{motion['min']:.4f}",
                mmean="na" if motion["mean"] is None else f"{motion['mean']:.4f}",
                mmax="na" if motion["max"] is None else f"{motion['max']:.4f}",
            )
        )
    failures = [r for r in audit["records"] if r.get("failures")]
    if failures:
        lines += ["", "## Failure Examples", ""]
        for row in failures[:80]:
            lines.append(
                "- idx {index} `{scenario}` seed={seed} failures={failures} path={path}".format(
                    index=row.get("index"),
                    scenario=row.get("scenario"),
                    seed=row.get("seed"),
                    failures=",".join(row.get("failures") or []),
                    path=row.get("path"),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-file", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--quotas", default="")
    parser.add_argument("--expected-frames", type=int, default=301)
    parser.add_argument("--expected-actions", type=int, default=300)
    parser.add_argument("--expected-action-dim", type=int, default=7)
    parser.add_argument("--min-hole-motion-m", type=float, default=0.22)
    parser.add_argument("--static-hole-motion-max-m", type=float, default=0.005)
    parser.add_argument("--motion-onset-threshold-m", type=float, default=1e-5)
    parser.add_argument("--peg-event-threshold-m", type=float, default=1e-8)
    parser.add_argument("--max-failure-examples", type=int, default=80)
    parser.add_argument("--no-fail-on-count-mismatch", action="store_true")
    ns = parser.parse_args()
    args = Args(
        paths_file=ns.paths_file,
        output_root=ns.output_root,
        quotas=ns.quotas,
        expected_frames=ns.expected_frames,
        expected_actions=ns.expected_actions,
        expected_action_dim=ns.expected_action_dim,
        min_hole_motion_m=ns.min_hole_motion_m,
        static_hole_motion_max_m=ns.static_hole_motion_max_m,
        motion_onset_threshold_m=ns.motion_onset_threshold_m,
        peg_event_threshold_m=ns.peg_event_threshold_m,
        max_failure_examples=ns.max_failure_examples,
        fail_on_count_mismatch=not ns.no_fail_on_count_mismatch,
    )

    quotas = _parse_quotas(args.quotas)
    paths = [Path(line.strip()) for line in args.paths_file.read_text().splitlines() if line.strip()]
    seen_paths: set[str] = set()
    path_duplicates: list[str] = []
    for path in paths:
        key = str(path.resolve())
        if key in seen_paths:
            path_duplicates.append(key)
        seen_paths.add(key)

    records = [_audit_one(path, index, args) for index, path in enumerate(paths)]
    summary = _summarize(records, quotas, args)
    failed_records = [row for row in records if row.get("failures")]
    count_mismatch = {
        scenario: {"count": item["count"], "quota": item["quota"]}
        for scenario, item in summary.items()
        if not scenario.startswith("_") and not item["count_ok"]
    }
    expected_total = int(sum(quotas.values()))
    global_failures: list[str] = []
    if len(paths) != expected_total:
        global_failures.append("path_count_mismatch")
    if path_duplicates:
        global_failures.append("duplicate_paths")
    if count_mismatch and args.fail_on_count_mismatch:
        global_failures.append("scenario_count_mismatch")
    if failed_records:
        global_failures.append("record_failures")

    audit = {
        "schema": "fix3_merged_source_h5_strict_audit_v1",
        "args": _jsonable(asdict(args)),
        "boundary": (
            "Read-only structural/metadata gate for v7 fix3 source H5s. "
            "This does not replace rendered-video/framebook inspection and does not approve SFT."
        ),
        "paths_file": str(args.paths_file),
        "quotas": quotas,
        "expected_total": expected_total,
        "num_paths": len(paths),
        "num_records": len(records),
        "num_failed_records": len(failed_records),
        "duplicate_paths": path_duplicates[: int(args.max_failure_examples)],
        "count_mismatch": count_mismatch,
        "global_failures": global_failures,
        "strict_ok": not global_failures,
        "summary_by_scenario": summary,
        "records": records,
        "stop_for_user_approval_after_visual_review": not global_failures,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "fix3_merged_source_h5_audit.json").write_text(
        json.dumps(_jsonable(audit), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.output_root / "fix3_merged_source_h5_audit.md", audit)
    print(
        json.dumps(
            {
                "event": "fix3_merged_source_h5_audit_done",
                "output_root": str(args.output_root),
                "strict_ok": audit["strict_ok"],
                "num_paths": len(paths),
                "num_failed_records": len(failed_records),
                "global_failures": global_failures,
                "count_mismatch": count_mismatch,
            },
            sort_keys=True,
        )
    )
    if global_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
