#!/usr/bin/env python3
"""Read-only audit for the original 2026-06-06 Cosmos3 full1000 H5 sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tyro


@dataclass
class Args:
    paths_file: str = (
        "experiments/world_model_task_rebinding/cosmos3/"
        "full1000_env_state_source_h5s_regen_20260606.txt"
    )
    output_root: str = (
        "experiments/world_model_task_rebinding/cosmos3/"
        "original_full1000_readonly_audit_20260611"
    )
    align_yz_threshold_m: float = 0.035
    align_x_min_m: float = -0.14
    align_x_max_m: float = 0.05
    motion_onset_threshold_m: float = 1e-4
    active_step_threshold_m: float = 1e-5
    max_failure_examples_per_scenario: int = 10


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


def _load_summary(group: h5py.Group) -> dict[str, Any]:
    raw = group.attrs.get("source_summary_json") or group.attrs.get("summary_json") or "{}"
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        data = json.loads(str(raw))
    except json.JSONDecodeError:
        return {"_raw": str(raw)}
    return data if isinstance(data, dict) else {"_raw": data}


def _scenario_from_path(path: Path) -> str:
    name = path.parent.name.replace(".rgbd", "")
    return name.split("_seed", 1)[0]


def _first_true(mask: np.ndarray) -> int:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else -1


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _audit_one(path: Path, index: int, args: Args) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        traj_names = sorted([key for key in h5.keys() if key.startswith("traj_")])
        if len(traj_names) != 1:
            raise RuntimeError(f"expected exactly one traj group, got {traj_names}")
        group = h5[traj_names[0]]
        summary = _load_summary(group)
        scenario = str(summary.get("scenario") or _scenario_from_path(path))
        slots = group["slots"]
        hole = np.asarray(slots["hole_pose"], dtype=np.float64)
        peg_head = np.asarray(slots["peg_head_at_hole"], dtype=np.float64)
        inserted = np.asarray(slots["inserted"], dtype=bool)
        grasped = np.asarray(slots["grasped"], dtype=bool)
        radius = np.asarray(slots["hole_radius"], dtype=np.float64).reshape(-1)
        actions = np.asarray(group["actions"])

        motion_from_start = hole[:, :3] - hole[0:1, :3]
        motion_norm_each = np.linalg.norm(motion_from_start, axis=1)
        moved = np.flatnonzero(motion_norm_each > args.motion_onset_threshold_m)
        motion_onset = int(moved[0]) if moved.size else -1

        per_step = np.linalg.norm(np.diff(hole[:, :3], axis=0), axis=1)
        active = np.flatnonzero(per_step > args.active_step_threshold_m)
        first_active_motion = int(active[0] + 1) if active.size else -1

        yz = np.linalg.norm(peg_head[:, 1:3], axis=1)
        x_axis = peg_head[:, 0]
        align_mask = (
            (yz <= args.align_yz_threshold_m)
            & (x_axis >= args.align_x_min_m)
            & (x_axis <= args.align_x_max_m)
        )
        first_align = _first_true(align_mask)
        first_grasp = _first_true(grasped)
        first_insert = _first_true(inserted)
        pre_idx = max(0, motion_onset - 1) if motion_onset >= 0 else max(0, min(len(yz) - 1, int(summary.get("trigger_step", 0) or 0)))

        seed = summary.get("seed", -1)
        try:
            seed = int(seed)
        except Exception:
            pass

        return {
            "index": int(index),
            "path": str(path),
            "sample_id": path.parent.name.replace(".rgbd", ""),
            "trajectory": traj_names[0],
            "scenario": scenario,
            "seed": seed,
            "num_frames": int(hole.shape[0]),
            "num_actions": int(actions.shape[0]),
            "inserted_end": bool(inserted[-1]),
            "inserted_once": bool(inserted.any()),
            "summary_success_at_end": bool(summary.get("success_at_end", False)),
            "summary_success_once": bool(summary.get("success_once", False)),
            "triggered": bool(summary.get("triggered", False)),
            "summary_trigger_step": int(summary.get("trigger_step", -1)),
            "trigger_reason": summary.get("trigger_reason", ""),
            "first_grasp_step": int(first_grasp),
            "first_align_step": int(first_align),
            "motion_onset_frame": int(motion_onset),
            "first_active_motion_frame": int(first_active_motion),
            "first_insert_step": int(first_insert),
            "target_motion_norm_m": float(np.linalg.norm(hole[-1, :3] - hole[0, :3])),
            "target_motion_xyz": (hole[-1, :3] - hole[0, :3]).astype(float).tolist(),
            "pre_motion_peg_head_x": _safe_float(x_axis[pre_idx]),
            "pre_motion_peg_head_yz": _safe_float(yz[pre_idx]),
            "motion_after_grasp": bool(motion_onset < 0 or (first_grasp >= 0 and motion_onset >= first_grasp)),
            "motion_after_align": bool(motion_onset < 0 or (first_align >= 0 and motion_onset >= first_align)),
            "final_peg_head_at_hole": peg_head[-1].astype(float).tolist(),
            "final_hole_radius": _safe_float(radius[-1]),
            "source_summary": summary,
        }


def _summarize(records: list[dict[str, Any]], args: Args) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        scenario = str(record.get("scenario", "error"))
        item = grouped.setdefault(
            scenario,
            {
                "total": 0,
                "inserted_end": 0,
                "inserted_once": 0,
                "triggered": 0,
                "motion_after_grasp": 0,
                "motion_after_align": 0,
                "motion_norms": [],
                "onsets": [],
                "fail_examples": [],
            },
        )
        item["total"] += 1
        if record.get("inserted_end"):
            item["inserted_end"] += 1
        elif len(item["fail_examples"]) < args.max_failure_examples_per_scenario:
            item["fail_examples"].append(
                {
                    key: record.get(key)
                    for key in (
                        "index",
                        "sample_id",
                        "path",
                        "target_motion_norm_m",
                        "motion_onset_frame",
                        "first_align_step",
                        "first_grasp_step",
                        "final_peg_head_at_hole",
                    )
                }
            )
        if record.get("inserted_once"):
            item["inserted_once"] += 1
        if record.get("triggered"):
            item["triggered"] += 1
        if record.get("motion_after_grasp"):
            item["motion_after_grasp"] += 1
        if record.get("motion_after_align"):
            item["motion_after_align"] += 1
        if "target_motion_norm_m" in record:
            item["motion_norms"].append(float(record["target_motion_norm_m"]))
        onset = int(record.get("motion_onset_frame", -1))
        if onset >= 0:
            item["onsets"].append(onset)

    out: dict[str, Any] = {}
    for scenario, item in sorted(grouped.items()):
        norms = np.asarray(item.pop("motion_norms"), dtype=np.float64)
        onsets = np.asarray(item.pop("onsets"), dtype=np.float64)
        out[scenario] = {
            **item,
            "target_motion_norm_m": {
                "min": float(norms.min()) if norms.size else 0.0,
                "mean": float(norms.mean()) if norms.size else 0.0,
                "max": float(norms.max()) if norms.size else 0.0,
            },
            "motion_onset_frame": {
                "min": int(onsets.min()) if onsets.size else -1,
                "mean": float(onsets.mean()) if onsets.size else -1.0,
                "max": int(onsets.max()) if onsets.size else -1,
            },
        }
    return out


def _write_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Original Full1000 Read-Only Audit",
        "",
        "Boundary: read-only audit; no source H5, original dataset root, or generation code modified.",
        "",
        "| scenario | total | inserted_end | inserted_once | triggered | motion_after_grasp | motion_after_align | target motion min/mean/max m | onset min/mean/max |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for scenario, item in sorted(audit["summary_by_scenario"].items()):
        motion = item["target_motion_norm_m"]
        onset = item["motion_onset_frame"]
        lines.append(
            "| {scenario} | {total} | {inserted_end} | {inserted_once} | {triggered} | "
            "{motion_after_grasp} | {motion_after_align} | {mmin:.4f}/{mmean:.4f}/{mmax:.4f} | "
            "{omin}/{omean:.1f}/{omax} |".format(
                scenario=scenario,
                total=item["total"],
                inserted_end=item["inserted_end"],
                inserted_once=item["inserted_once"],
                triggered=item["triggered"],
                motion_after_grasp=item["motion_after_grasp"],
                motion_after_align=item["motion_after_align"],
                mmin=motion["min"],
                mmean=motion["mean"],
                mmax=motion["max"],
                omin=onset["min"],
                omean=onset["mean"],
                omax=onset["max"],
            )
        )

    lines += ["", "## Failure Examples", ""]
    for scenario, item in sorted(audit["summary_by_scenario"].items()):
        examples = item.get("fail_examples") or []
        if not examples:
            continue
        lines.append(f"### {scenario}")
        for example in examples[:5]:
            lines.append(
                "- idx {index} `{sample_id}` motion={motion:.4f} onset={onset} align={align} "
                "grasp={grasp} final_head={head}".format(
                    index=example.get("index"),
                    sample_id=example.get("sample_id"),
                    motion=float(example.get("target_motion_norm_m") or 0.0),
                    onset=example.get("motion_onset_frame"),
                    align=example.get("first_align_step"),
                    grasp=example.get("first_grasp_step"),
                    head=example.get("final_peg_head_at_hole"),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = tyro.cli(Args)
    paths_file = Path(args.paths_file)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = [Path(line.strip()) for line in paths_file.read_text().splitlines() if line.strip()]

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        try:
            records.append(_audit_one(path, index, args))
        except Exception as exc:
            failures.append({"index": index, "path": str(path), "error": repr(exc)})

    audit = {
        "args": _jsonable(asdict(args)),
        "boundary": "read-only audit of original 2026-06-06 full1000 H5 sources; no original file modified",
        "paths_file": str(paths_file),
        "num_paths": len(paths),
        "num_records": len(records),
        "num_read_failures": len(failures),
        "read_failures": failures,
        "summary_by_scenario": _summarize(records, args),
        "records": records,
    }
    (output_root / "original_full1000_readonly_audit.json").write_text(
        json.dumps(_jsonable(audit), indent=2, sort_keys=True) + "\n"
    )
    _write_markdown(output_root / "original_full1000_readonly_audit.md", audit)
    print(
        json.dumps(
            {
                "event": "original_full1000_readonly_audit_done",
                "output_root": str(output_root),
                "num_paths": len(paths),
                "num_records": len(records),
                "num_read_failures": len(failures),
                "summary_by_scenario": audit["summary_by_scenario"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
