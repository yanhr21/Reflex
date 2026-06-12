#!/usr/bin/env python3
"""Build a temporal WAM-style action/state dataset from existing artifacts.

This is a structural bridge toward the corrected controller objective. It does
not make a failed rollout positive. Each sample exposes:

- current policy observation history;
- current RGB-D/slot-derived task state;
- Cosmos/readout future object/task trajectory;
- executed action chunk;
- future metric task-state labels from the real simulator rollout.

The resulting H5 is suitable for preflight/audit and for future WAM-style
policy/scorer training once positive teacher rollouts are admitted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


CURRENT_SLOT_NAMES = [
    "hole_pose:0",
    "hole_pose:1",
    "hole_pose:2",
    "hole_pose:3",
    "hole_pose:4",
    "hole_pose:5",
    "hole_pose:6",
    "peg_pose:0",
    "peg_pose:1",
    "peg_pose:2",
    "peg_pose:3",
    "peg_pose:4",
    "peg_pose:5",
    "peg_pose:6",
    "tcp_pose:0",
    "tcp_pose:1",
    "tcp_pose:2",
    "tcp_pose:3",
    "tcp_pose:4",
    "tcp_pose:5",
    "tcp_pose:6",
    "peg_head_at_hole:0",
    "peg_head_at_hole:1",
    "peg_head_at_hole:2",
    "hole_velocity_step:0",
    "hole_velocity_step:1",
    "hole_velocity_step:2",
    "grasped",
    "inserted",
    "hole_radius",
]

FUTURE_STATE_NAMES = [
    "hole_pose:0",
    "hole_pose:1",
    "hole_pose:2",
    "hole_pose:3",
    "hole_pose:4",
    "hole_pose:5",
    "hole_pose:6",
    "peg_pose:0",
    "peg_pose:1",
    "peg_pose:2",
    "peg_pose:3",
    "peg_pose:4",
    "peg_pose:5",
    "peg_pose:6",
    "tcp_pose:0",
    "tcp_pose:1",
    "tcp_pose:2",
    "tcp_pose:3",
    "tcp_pose:4",
    "tcp_pose:5",
    "tcp_pose:6",
    "peg_head_at_hole:0",
    "peg_head_at_hole:1",
    "peg_head_at_hole:2",
    "grasped_or_probability",
    "inserted_or_probability",
    "hole_radius",
]


@dataclass
class Args:
    controller_h5: str
    cosmos_trajectory_json: str
    output_h5: str
    output_json: str
    trajectory: str = "traj_0"
    action_horizon: int = 8
    future_horizon: int = 16
    max_samples: int = 0
    stride: int = 1
    post_trigger_only: bool = True
    allow_failed_rollout_structural_dataset: bool = True


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _trajectory_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("trajectory"), list):
        return [row for row in payload["trajectory"] if isinstance(row, dict)]
    if isinstance(payload.get("predictions"), list):
        return [row for row in payload["predictions"] if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    segments = payload.get("segments")
    if isinstance(segments, list):
        for segment_idx, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue
            for row in _trajectory_rows(segment.get("trajectory", [])):
                row = dict(row)
                row.setdefault("segment_index", segment_idx)
                rows.append(row)
    return rows


def _cosmos_frame_map(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_frame: dict[int, dict[str, Any]] = {}
    for row in rows:
        if "frame" not in row:
            continue
        # Later receding rows replace earlier rows for the same frame, so the
        # map prefers the most recently observed segment for overlapping frames.
        by_frame[int(row["frame"])] = row
    return by_frame


def _vec(value: Any, dim: int) -> np.ndarray:
    arr = np.asarray(value if value is not None else [], dtype=np.float32).reshape(-1)
    out = np.zeros((dim,), dtype=np.float32)
    n = min(dim, arr.size)
    if n:
        out[:n] = arr[:n]
    return out


def _flag(value: Any) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return 0.0
    return float(bool(arr.reshape(-1)[0]))


def _current_slot_state(group: h5py.Group, frame: int) -> np.ndarray:
    slots = group["slots"]
    pieces = [
        _vec(slots["hole_pose"][frame], 7),
        _vec(slots["peg_pose"][frame], 7),
        _vec(slots["tcp_pose"][frame], 7),
        _vec(slots["peg_head_at_hole"][frame], 3),
        _vec(slots["hole_velocity_step"][frame], 3),
        np.asarray(
            [
                _flag(slots["grasped"][frame]),
                _flag(slots["inserted"][frame]),
                float(np.asarray(slots["hole_radius"][frame], dtype=np.float32).reshape(-1)[0]),
            ],
            dtype=np.float32,
        ),
    ]
    return np.concatenate(pieces).astype(np.float32)


def _metric_future_state(group: h5py.Group, frame: int) -> np.ndarray:
    metric = group["metric_slots"]
    pieces = [
        _vec(metric["hole_pose"][frame], 7),
        _vec(metric["peg_pose"][frame], 7),
        _vec(metric["tcp_pose"][frame], 7),
        _vec(metric["peg_head_at_hole"][frame], 3),
        np.asarray(
            [
                _flag(metric["grasped"][frame]),
                _flag(metric["inserted"][frame]),
                float(np.asarray(metric["hole_radius"][frame], dtype=np.float32).reshape(-1)[0]),
            ],
            dtype=np.float32,
        ),
    ]
    return np.concatenate(pieces).astype(np.float32)


def _cosmos_future_state(row: dict[str, Any]) -> np.ndarray:
    pieces = [
        _vec(row.get("hole_pose"), 7),
        _vec(row.get("peg_pose"), 7),
        _vec(row.get("tcp_pose"), 7),
        _vec(row.get("peg_head_at_hole"), 3),
        np.asarray(
            [
                float(row.get("grasped_probability", row.get("grasped", 0.0))),
                float(row.get("inserted_probability", row.get("inserted", 0.0))),
                float(row.get("hole_radius", 0.0)),
            ],
            dtype=np.float32,
        ),
    ]
    return np.concatenate(pieces).astype(np.float32)


def _frame_mask(group: h5py.Group, length: int, post_trigger_only: bool) -> np.ndarray:
    mask = np.ones((length,), dtype=bool)
    if not post_trigger_only:
        return mask
    if "perturb" in group and "triggered" in group["perturb"]:
        triggered = np.asarray(group["perturb/triggered"], dtype=bool).reshape(-1)
        usable = min(length, triggered.shape[0])
        out = np.zeros((length,), dtype=bool)
        out[:usable] = triggered[:usable]
        if out.any():
            return out
    if "perturb" in group and "trigger_step" in group["perturb"]:
        trigger = np.asarray(group["perturb/trigger_step"]).reshape(-1)
        trigger = trigger[trigger >= 0]
        if trigger.size:
            out = np.zeros((length,), dtype=bool)
            out[int(trigger.min()) :] = True
            return out
    return mask


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-h5", required=True)
    parser.add_argument("--cosmos-trajectory-json", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--trajectory", default="traj_0")
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--future-horizon", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--post-trigger-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-failed-rollout-structural-dataset", action=argparse.BooleanOptionalAction, default=True)
    return Args(**vars(parser.parse_args()))


def main() -> None:
    args = parse_args()
    if args.action_horizon <= 0 or args.future_horizon <= 0:
        raise ValueError("horizons must be positive")
    if args.stride <= 0:
        raise ValueError("stride must be positive")

    rows = _trajectory_rows(_read_json(Path(args.cosmos_trajectory_json)))
    frame_map = _cosmos_frame_map(rows)
    if not frame_map:
        raise ValueError("Cosmos trajectory has no frame-indexed rows")

    output_h5 = Path(args.output_h5)
    output_json = Path(args.output_json)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, np.ndarray | int]] = []
    skipped_missing_cosmos = 0
    skipped_post_trigger = 0
    inserted_any = False
    final_inserted = False
    final_grasped = False
    with h5py.File(args.controller_h5, "r") as src:
        if args.trajectory not in src:
            raise KeyError(f"{args.controller_h5} missing {args.trajectory}")
        group = src[args.trajectory]
        required = [
            "actions",
            "policy_obs_frame_stack",
            "policy_obs",
            "slots/hole_pose",
            "metric_slots/hole_pose",
        ]
        missing = [name for name in required if name not in group]
        if missing:
            raise ValueError(f"controller H5 missing required datasets: {missing}")
        actions = np.asarray(group["actions"], dtype=np.float32)
        policy_obs_frame_stack = np.asarray(group["policy_obs_frame_stack"], dtype=np.float32)
        metric_inserted = np.asarray(group["metric_slots/inserted"], dtype=bool).reshape(-1)
        metric_grasped = np.asarray(group["metric_slots/grasped"], dtype=bool).reshape(-1)
        inserted_any = bool(metric_inserted.any())
        final_inserted = bool(metric_inserted[-1])
        final_grasped = bool(metric_grasped[-1])
        if (not inserted_any or not final_inserted) and not args.allow_failed_rollout_structural_dataset:
            raise SystemExit(
                "Controller rollout has no final/any insertion and cannot be exported "
                "unless --allow-failed-rollout-structural-dataset is true."
            )

        length = int(actions.shape[0])
        mask = _frame_mask(group, length, args.post_trigger_only)
        for frame in range(0, length - max(args.action_horizon, args.future_horizon) + 1, args.stride):
            if args.post_trigger_only and not bool(mask[frame]):
                skipped_post_trigger += 1
                continue
            future_frames = list(range(frame, frame + args.future_horizon))
            if any(future_frame not in frame_map for future_frame in future_frames):
                skipped_missing_cosmos += 1
                continue
            sample = {
                "frame": int(frame),
                "policy_obs_history": policy_obs_frame_stack[frame].astype(np.float32),
                "current_slot_state": _current_slot_state(group, frame),
                "cosmos_future_state": np.stack(
                    [_cosmos_future_state(frame_map[future_frame]) for future_frame in future_frames],
                    axis=0,
                ),
                "action_chunk": actions[frame : frame + args.action_horizon].astype(np.float32),
                "future_metric_state": np.stack(
                    [_metric_future_state(group, future_frame) for future_frame in future_frames],
                    axis=0,
                ),
            }
            samples.append(sample)
            if args.max_samples > 0 and len(samples) >= args.max_samples:
                break

    if not samples:
        raise ValueError("No WAM samples could be built from the supplied artifacts")

    with h5py.File(output_h5, "w") as out:
        group = out.create_group("samples")
        for key in (
            "frame",
            "policy_obs_history",
            "current_slot_state",
            "cosmos_future_state",
            "action_chunk",
            "future_metric_state",
        ):
            data = np.asarray([sample[key] for sample in samples])
            group.create_dataset(key, data=data, compression="gzip", compression_opts=4)
        out.attrs["current_slot_names_json"] = json.dumps(CURRENT_SLOT_NAMES)
        out.attrs["future_state_names_json"] = json.dumps(FUTURE_STATE_NAMES)
        out.attrs["source_controller_h5"] = args.controller_h5
        out.attrs["source_cosmos_trajectory_json"] = args.cosmos_trajectory_json
        out.attrs["positive_takeover_teacher_ok"] = False
        out.attrs["method_evidence_allowed"] = False
        out.attrs["boundary"] = (
            "Structural WAM temporal condition dataset. It does not convert a failed "
            "rollout into positive takeover teacher data."
        )

    cosmos_stack = np.asarray([sample["cosmos_future_state"] for sample in samples], dtype=np.float32)
    metric_stack = np.asarray([sample["future_metric_state"] for sample in samples], dtype=np.float32)
    cosmos_hole = cosmos_stack[..., 0:3]
    metric_hole = metric_stack[..., 0:3]
    cosmos_head = cosmos_stack[..., 21:24]
    metric_head = metric_stack[..., 21:24]
    hole_l2 = np.linalg.norm(cosmos_hole - metric_hole, axis=-1)
    head_l2 = np.linalg.norm(cosmos_head - metric_head, axis=-1)
    report = {
        "args": asdict(args),
        "output_h5": str(output_h5),
        "sample_count": len(samples),
        "frame_minmax": [int(samples[0]["frame"]), int(samples[-1]["frame"])],
        "policy_obs_history_shape": list(np.asarray(samples[0]["policy_obs_history"]).shape),
        "current_slot_state_dim": int(np.asarray(samples[0]["current_slot_state"]).shape[-1]),
        "cosmos_future_state_shape": list(np.asarray(samples[0]["cosmos_future_state"]).shape),
        "action_chunk_shape": list(np.asarray(samples[0]["action_chunk"]).shape),
        "future_metric_state_shape": list(np.asarray(samples[0]["future_metric_state"]).shape),
        "skipped_missing_cosmos_future": int(skipped_missing_cosmos),
        "skipped_pre_trigger": int(skipped_post_trigger),
        "source_rollout_metric": {
            "inserted_any": bool(inserted_any),
            "final_inserted": bool(final_inserted),
            "final_grasped": bool(final_grasped),
        },
        "cosmos_future_vs_metric_future": {
            "hole_pos_l2_mean_m": float(np.mean(hole_l2)),
            "hole_pos_l2_max_m": float(np.max(hole_l2)),
            "peg_head_at_hole_l2_mean_m": float(np.mean(head_l2)),
            "peg_head_at_hole_l2_max_m": float(np.max(head_l2)),
            "interpretation": (
                "These values compare Cosmos imagined future task states with the "
                "future states actually reached by the source rollout's executed "
                "actions. Large values mean the action chunk did not realize the "
                "imagined path; that is useful training signal only when the "
                "teacher/action label is admitted positive or explicitly negative."
            ),
        },
        "positive_takeover_teacher_ok": False,
        "method_evidence_allowed": False,
        "boundary": (
            "This export makes the missing WAM temporal interface concrete: future "
            "Cosmos object/task trajectory, action chunks, and future metric labels "
            "are now aligned for audit/pretraining. Because the source rollout fails "
            "insertion, it is not positive takeover distillation data."
        ),
    }
    output_json.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_jsonable(report), sort_keys=True))


if __name__ == "__main__":
    main()
