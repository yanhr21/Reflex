#!/usr/bin/env python3
"""Export causal action/state conditions for Cosmos3 manipulation SFT.

The current Cosmos3 video SFT JSONL only gives the model pixels plus a compact
caption. This exporter prepares the structured conditioning needed for the
foundation-WM/controller path: a video prefix, future candidate action commands,
and robot/object state observed at the prefix boundary.

The exported vectors intentionally do not contain future ground-truth object
poses. Future rows repeat the prefix-observed task state and carry only the
recorded/candidate action command plus time index, so the world model must
predict future visual/task state rather than read it from the condition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
import tyro


VECTOR_NAMES = [
    "action_0",
    "action_1",
    "action_2",
    "action_3",
    "action_4",
    "action_5",
    "action_6",
    "prefix_tcp_x",
    "prefix_tcp_y",
    "prefix_tcp_z",
    "prefix_peg_x",
    "prefix_peg_y",
    "prefix_peg_z",
    "prefix_hole_x",
    "prefix_hole_y",
    "prefix_hole_z",
    "prefix_peg_head_hole_x",
    "prefix_peg_head_hole_y",
    "prefix_peg_head_hole_z",
    "prefix_hole_velocity_x",
    "prefix_hole_velocity_y",
    "prefix_hole_velocity_z",
    "prefix_grasped",
    "prefix_inserted",
    "prefix_hole_delta_cumulative_x",
    "prefix_hole_delta_cumulative_y",
    "prefix_hole_delta_cumulative_z",
    "prefix_peg_delta_applied_x",
    "prefix_peg_delta_applied_y",
    "prefix_peg_delta_applied_z",
    "action_time_fraction",
    "prefix_perturb_triggered",
]

STATE_TARGET_VECTOR_NAMES = (
    ["source_frame_index", "frame_time_fraction"]
    + [f"tcp_pose_{name}" for name in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    + [f"peg_pose_{name}" for name in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    + [f"hole_pose_{name}" for name in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    + [f"qpos_{i}" for i in range(9)]
    + [f"qvel_{i}" for i in range(9)]
    + [f"peg_head_at_hole_{name}" for name in ("x", "y", "z")]
    + [f"hole_velocity_step_{name}" for name in ("x", "y", "z")]
    + ["grasped", "inserted"]
    + [f"hole_delta_cumulative_{name}" for name in ("x", "y", "z")]
    + [f"peg_delta_applied_{name}" for name in ("x", "y", "z")]
    + ["perturb_triggered"]
)


@dataclass
class Args:
    dataset_root: str
    output_root: str
    total_video_frames: int = 301
    condition_prefix_frames: int = 29
    condition_latent_frames: int = 8
    action_condition_mode: str = "forward_dynamics"
    require_video_frames: int = 0
    max_records: int = 0
    normalize: bool = True
    domain_name: str = "maniskill_peg_insertion"
    domain_id: int = 21
    raw_action_dim: int = 32
    write_state_targets: bool = True
    sanitize_future_caption: bool = True
    progress_every: int = 50


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


def _safe_vec(group: h5py.Group | None, key: str, idx: int, dim: int) -> np.ndarray:
    if group is None or key not in group:
        return np.zeros((dim,), dtype=np.float32)
    arr = np.asarray(group[key])
    if arr.shape[0] == 0:
        return np.zeros((dim,), dtype=np.float32)
    idx = min(max(0, idx), arr.shape[0] - 1)
    value = np.asarray(arr[idx], dtype=np.float32).reshape(-1)
    if value.shape[0] < dim:
        out = np.zeros((dim,), dtype=np.float32)
        out[: value.shape[0]] = value
        return out
    return value[:dim].astype(np.float32)


def _safe_bool(group: h5py.Group | None, key: str, idx: int) -> float:
    if group is None or key not in group:
        return 0.0
    arr = np.asarray(group[key])
    if arr.shape[0] == 0:
        return 0.0
    idx = min(max(0, idx), arr.shape[0] - 1)
    return float(bool(arr[idx]))


def _safe_scalar(group: h5py.Group | None, key: str, idx: int, default: float = 0.0) -> float:
    if group is None or key not in group:
        return float(default)
    arr = np.asarray(group[key])
    if arr.shape[0] == 0:
        return float(default)
    idx = min(max(0, idx), arr.shape[0] - 1)
    return float(np.asarray(arr[idx]).reshape(-1)[0])


def _build_raw_vectors(h5_path: Path, total_video_frames: int, condition_prefix_frames: int) -> np.ndarray:
    action_steps = int(total_video_frames) - 1
    if action_steps <= 0:
        raise ValueError(f"total_video_frames must be at least 2, got {total_video_frames}")
    prefix_idx = max(0, min(int(condition_prefix_frames) - 1, int(total_video_frames) - 1))

    with h5py.File(h5_path, "r") as h5:
        traj_names = sorted([k for k in h5.keys() if k.startswith("traj_")], key=lambda x: int(x.split("_", 1)[1]))
        if len(traj_names) != 1:
            raise ValueError(f"{h5_path} expected one trajectory, found {traj_names}")
        group = h5[traj_names[0]]
        actions = np.asarray(group["actions"], dtype=np.float32)
        slots = group.get("slots")
        perturb = group.get("perturb")

        prefix_tcp = _safe_vec(slots, "tcp_pose", prefix_idx, 3)
        prefix_peg = _safe_vec(slots, "peg_pose", prefix_idx, 3)
        prefix_hole = _safe_vec(slots, "hole_pose", prefix_idx, 3)
        prefix_peg_head_hole = _safe_vec(slots, "peg_head_at_hole", prefix_idx, 3)
        prefix_hole_velocity = _safe_vec(slots, "hole_velocity_step", prefix_idx, 3)
        prefix_grasped = _safe_bool(slots, "grasped", prefix_idx)
        prefix_inserted = _safe_bool(slots, "inserted", prefix_idx)

        perturb_idx = min(prefix_idx, max(0, actions.shape[0] - 1))
        prefix_hole_delta = _safe_vec(perturb, "hole_delta_cumulative", perturb_idx, 3)
        prefix_peg_delta = _safe_vec(perturb, "peg_delta_applied", perturb_idx, 3)
        prefix_triggered = _safe_bool(perturb, "triggered", perturb_idx)

        rows = []
        denom = max(1, action_steps - 1)
        for t in range(action_steps):
            action = np.zeros((7,), dtype=np.float32)
            if t < actions.shape[0]:
                src = np.asarray(actions[t], dtype=np.float32).reshape(-1)
                action[: min(7, src.shape[0])] = src[:7]
            row = np.concatenate(
                [
                    action,
                    prefix_tcp,
                    prefix_peg,
                    prefix_hole,
                    prefix_peg_head_hole,
                    prefix_hole_velocity,
                    np.asarray([prefix_grasped, prefix_inserted], dtype=np.float32),
                    prefix_hole_delta,
                    prefix_peg_delta,
                    np.asarray([float(t) / float(denom), prefix_triggered], dtype=np.float32),
                ]
            )
            rows.append(row)

    out = np.stack(rows, axis=0).astype(np.float32)
    if out.shape != (action_steps, len(VECTOR_NAMES)):
        raise RuntimeError(f"unexpected action/state shape {out.shape}")
    return out


def _build_state_target_sequence(h5_path: Path, total_video_frames: int) -> np.ndarray:
    if total_video_frames <= 1:
        raise ValueError(f"total_video_frames must be at least 2, got {total_video_frames}")
    with h5py.File(h5_path, "r") as h5:
        traj_names = sorted([k for k in h5.keys() if k.startswith("traj_")], key=lambda x: int(x.split("_", 1)[1]))
        if len(traj_names) != 1:
            raise ValueError(f"{h5_path} expected one trajectory, found {traj_names}")
        group = h5[traj_names[0]]
        slots = group.get("slots")
        perturb = group.get("perturb")
        if slots is None:
            raise ValueError(f"{h5_path} has no slots group")

        rows = []
        denom = max(1, total_video_frames - 1)
        for frame_idx in range(total_video_frames):
            perturb_idx = min(frame_idx, total_video_frames - 2)
            source_frame = _safe_scalar(group, "source_frame_indices", frame_idx, default=float(frame_idx))
            row = np.concatenate(
                [
                    np.asarray([source_frame, float(frame_idx) / float(denom)], dtype=np.float32),
                    _safe_vec(slots, "tcp_pose", frame_idx, 7),
                    _safe_vec(slots, "peg_pose", frame_idx, 7),
                    _safe_vec(slots, "hole_pose", frame_idx, 7),
                    _safe_vec(slots, "qpos", frame_idx, 9),
                    _safe_vec(slots, "qvel", frame_idx, 9),
                    _safe_vec(slots, "peg_head_at_hole", frame_idx, 3),
                    _safe_vec(slots, "hole_velocity_step", frame_idx, 3),
                    np.asarray(
                        [
                            _safe_bool(slots, "grasped", frame_idx),
                            _safe_bool(slots, "inserted", frame_idx),
                        ],
                        dtype=np.float32,
                    ),
                    _safe_vec(perturb, "hole_delta_cumulative", perturb_idx, 3),
                    _safe_vec(perturb, "peg_delta_applied", perturb_idx, 3),
                    np.asarray([_safe_bool(perturb, "triggered", perturb_idx)], dtype=np.float32),
                ]
            )
            rows.append(row)
    out = np.stack(rows, axis=0).astype(np.float32)
    if out.shape != (total_video_frames, len(STATE_TARGET_VECTOR_NAMES)):
        raise RuntimeError(f"unexpected state target shape {out.shape}")
    return out


def _zscore_stats(arrays: list[np.ndarray]) -> dict[str, Any]:
    all_rows = np.concatenate(arrays, axis=0).astype(np.float64)
    mean = all_rows.mean(axis=0)
    std = all_rows.std(axis=0)
    std = np.maximum(std, 1e-6)
    return {
        "type": "zscore",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "vector_names": VECTOR_NAMES,
        "raw_action_dim": len(VECTOR_NAMES),
    }


def _apply_stats(array: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return ((array - mean) / std).astype(np.float32)


def _vec_dict(names: list[str], values: np.ndarray) -> dict[str, float]:
    return {name: float(values[idx]) for idx, name in enumerate(names)}


def _prefix_condition_payload(raw: np.ndarray, condition_prefix_frames: int) -> dict[str, Any]:
    if raw.ndim != 2 or raw.shape[1] != len(VECTOR_NAMES):
        raise ValueError(f"unexpected raw action/state condition shape {raw.shape}")
    prefix_row = raw[0]
    values = _vec_dict(VECTOR_NAMES, prefix_row)

    def xyz(prefix: str) -> list[float]:
        return [values[f"{prefix}_{axis}"] for axis in ("x", "y", "z")]

    return {
        "condition_prefix_frames": int(condition_prefix_frames),
        "prefix_frame_index": int(max(0, condition_prefix_frames - 1)),
        "tcp_xyz": xyz("prefix_tcp"),
        "peg_xyz": xyz("prefix_peg"),
        "hole_xyz": xyz("prefix_hole"),
        "peg_head_at_hole_xyz": xyz("prefix_peg_head_hole"),
        "hole_velocity_xyz": xyz("prefix_hole_velocity"),
        "hole_delta_cumulative_xyz": xyz("prefix_hole_delta_cumulative"),
        "peg_delta_applied_xyz": xyz("prefix_peg_delta_applied"),
        "grasped": bool(values["prefix_grasped"] > 0.5),
        "inserted": bool(values["prefix_inserted"] > 0.5),
        "perturb_triggered": bool(values["prefix_perturb_triggered"] > 0.5),
        "boundary": (
            "Prefix/current causal task state only. Future ground-truth object, "
            "robot, insertion, and final positions are not included."
        ),
    }


def _fmt_xyz(values: list[float]) -> str:
    return "[" + ", ".join(f"{float(v):.4f}" for v in values) + "]"


def _causal_caption(scenario: str | None, mode: str, payload: dict[str, Any]) -> str:
    scenario_text = {
        "hole_constant": "the target hole may continue moving at approximately constant velocity",
        "hole_reverse": "the target hole may reverse direction during the rollout",
        "hole_move_stop": "the target hole may move and then stop during the rollout",
        "peg_drop": "the peg may require recovery from a drop or regrasp event",
        "peg_disturb": "the peg may be physically disturbed before recovery",
        "none": "the scene may remain static after the observed prefix",
    }.get(str(scenario), "the dynamic scene may change after the observed prefix")
    if mode == "forward_dynamics":
        task = "predict future RGB and task state from the observed video prefix plus the candidate action/state rows"
    else:
        task = "predict future RGB and the robot action chunk from the observed video prefix and causal prefix task state"
    return (
        "ManiSkill default angled-overhead RGB peg insertion rollout. "
        f"From the observed prefix, {scenario_text}; {task}. "
        f"Prefix frame {payload['prefix_frame_index']} state: "
        f"TCP xyz {_fmt_xyz(payload['tcp_xyz'])}; "
        f"peg xyz {_fmt_xyz(payload['peg_xyz'])}; "
        f"hole xyz {_fmt_xyz(payload['hole_xyz'])}; "
        f"peg-head relative-to-hole xyz {_fmt_xyz(payload['peg_head_at_hole_xyz'])}; "
        f"observed hole velocity xyz {_fmt_xyz(payload['hole_velocity_xyz'])}; "
        f"prefix cumulative hole perturbation xyz {_fmt_xyz(payload['hole_delta_cumulative_xyz'])}; "
        f"prefix peg perturbation xyz {_fmt_xyz(payload['peg_delta_applied_xyz'])}; "
        f"grasped={payload['grasped']}; inserted={payload['inserted']}; "
        f"perturb_triggered={payload['perturb_triggered']}. "
        "Do not assume the original layout is restored; predict completion in the changed world."
    )


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main() -> None:
    args = tyro.cli(Args)
    if args.action_condition_mode not in {"forward_dynamics", "joint_policy"}:
        raise ValueError(
            "action_condition_mode must be 'forward_dynamics' or 'joint_policy', "
            f"got {args.action_condition_mode!r}"
        )
    if args.raw_action_dim != len(VECTOR_NAMES):
        raise ValueError(f"raw_action_dim={args.raw_action_dim} but vector layout has {len(VECTOR_NAMES)} dims")
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    videos = list(manifest.get("videos", []))
    if args.max_records > 0:
        videos = videos[: args.max_records]
    if not videos:
        raise ValueError("no videos found in dataset manifest")
    if args.require_video_frames > 0:
        bad = [
            {
                "sample_id": item.get("sample_id"),
                "num_video_frames": item.get("num_video_frames"),
            }
            for item in videos
            if int(item.get("num_video_frames", -1)) != int(args.require_video_frames)
        ]
        if bad:
            raise ValueError(
                f"dataset manifest frame count mismatch: expected {args.require_video_frames}, "
                f"first_bad={bad[:3]}"
            )
    if args.require_video_frames > 0 and args.total_video_frames != args.require_video_frames:
        raise ValueError(
            f"total_video_frames={args.total_video_frames} but require_video_frames={args.require_video_frames}"
        )
    _progress(
        "export_cosmos3_action_conditions_start "
        f"records={len(videos)} mode={args.action_condition_mode} "
        f"frames={args.total_video_frames} write_state_targets={args.write_state_targets} "
        f"output_root={output_root}"
    )
    source_video_records: dict[str, dict[str, Any]] = {}
    for split in ("train", "val"):
        for row in _read_jsonl(dataset_root / split / "video_dataset_file.jsonl"):
            source_video_records[row["uuid"]] = row

    raw_by_sample: dict[str, np.ndarray] = {}
    state_target_by_sample: dict[str, np.ndarray] = {}
    for idx, item in enumerate(videos, start=1):
        h5_path = Path(item["input_h5"])
        raw_by_sample[item["sample_id"]] = _build_raw_vectors(
            h5_path,
            total_video_frames=args.total_video_frames,
            condition_prefix_frames=args.condition_prefix_frames,
        )
        if args.write_state_targets:
            state_target_by_sample[item["sample_id"]] = _build_state_target_sequence(
                h5_path,
                total_video_frames=args.total_video_frames,
            )
        if args.progress_every > 0 and (idx == 1 or idx % args.progress_every == 0 or idx == len(videos)):
            _progress(
                "export_cosmos3_action_conditions_built_raw "
                f"{idx}/{len(videos)} sample_id={item['sample_id']}"
            )

    stats = _zscore_stats(list(raw_by_sample.values())) if args.normalize else None
    _progress(
        "export_cosmos3_action_conditions_stats_ready "
        f"normalize={args.normalize} records={len(raw_by_sample)}"
    )
    action_root = output_root / "actions"
    state_target_root = output_root / "state_targets"
    records: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    sft_records: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    written = []
    for idx, item in enumerate(videos, start=1):
        sample_id = item["sample_id"]
        split = item["split"]
        raw = raw_by_sample[sample_id]
        action = _apply_stats(raw, stats) if stats is not None else raw
        action_steps = args.total_video_frames - 1
        if args.action_condition_mode == "forward_dynamics":
            model_mode = "forward_dynamics"
            condition_frame_indexes_action = list(range(action_steps))
            mode_note = "video_prefix_plus_all_causal_action_state_rows_predict_future_rgb"
            prompt = (
                "Predict the future peg-in-hole manipulation video from the approved "
                "ManiSkill angled-overhead RGB view, using the video prefix, observed "
                "robot/object task state, and candidate robot action commands."
            )
        else:
            model_mode = "policy"
            condition_frame_indexes_action = []
            mode_note = "video_prefix_predicts_future_rgb_and_robot_action_chunk_jointly"
            prompt = (
                "Jointly predict the future peg-in-hole manipulation video and robot "
                "action chunk from the approved ManiSkill angled-overhead RGB view, "
                "using the video prefix and causal robot/object task state."
            )
        action_rel = Path("actions") / split / f"{int(item['index']):04d}_{sample_id}.json"
        action_path = output_root / action_rel
        action_path.parent.mkdir(parents=True, exist_ok=True)
        action_path.write_text(json.dumps(action.tolist()) + "\n")

        state_target_path: Path | None = None
        if args.write_state_targets:
            state_target_rel = Path("state_targets") / split / f"{int(item['index']):04d}_{sample_id}.json"
            state_target_path = output_root / state_target_rel
            state_target_path.parent.mkdir(parents=True, exist_ok=True)
            state_target_payload = {
                "sample_id": sample_id,
                "source_h5": item.get("input_h5"),
                "num_frames": args.total_video_frames,
                "num_action_steps": args.total_video_frames - 1,
                "state_vector_names": STATE_TARGET_VECTOR_NAMES,
                "states": state_target_by_sample[sample_id].tolist(),
                "boundary": (
                    "Future simulator task state is a target/evaluation label only. "
                    "It is not a clean condition for active Cosmos3 controller-facing inference."
                ),
            }
            state_target_path.write_text(json.dumps(state_target_payload, separators=(",", ":")) + "\n")

        video_path = Path(item["video"])
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        prefix_payload = _prefix_condition_payload(raw, args.condition_prefix_frames)
        causal_caption = _causal_caption(
            scenario=item.get("scenario"),
            mode=args.action_condition_mode,
            payload=prefix_payload,
        )
        safe_metadata = {
            "camera": "PegInsertionSide-v1_default_human_render",
            "conditioning_policy": "video_prefix_not_single_image_i2v",
            "fps": int(item.get("fps", 30)),
            "scenario": item.get("scenario"),
            "source": "maniskill_default_human_render_from_env_states",
            "structured_action_state_condition": {
                "domain_name": args.domain_name,
                "domain_id": int(args.domain_id),
                "raw_action_dim": len(VECTOR_NAMES),
                "vector_names": VECTOR_NAMES,
                "normalization": "zscore" if stats is not None else "none",
                "action_condition_mode": args.action_condition_mode,
                "visual_input": "RGB only; depth is not used",
                "condition_prefix_frames": args.condition_prefix_frames,
                "condition_latent_frames": args.condition_latent_frames,
                "causal_boundary": (
                    "Object/task state entries are copied from the prefix boundary "
                    "and repeated through future action rows; future GT object poses "
                    "are not written into action conditions, text captions, or "
                    "controller-facing metadata."
                ),
                "state_target_policy": (
                    "full301 future task-state sequence is written as target/evaluation label only"
                    if args.write_state_targets
                    else "not_written"
                ),
            },
            "causal_task_state_condition": prefix_payload,
            "caption_policy": (
                "sanitized_prefix_only_no_future_ground_truth"
                if args.sanitize_future_caption
                else "source_caption_preserved"
            ),
        }
        record = {
            "name": f"{sample_id}_fd_prefix{args.condition_prefix_frames}_state_action32",
            "uuid": sample_id,
            "model_mode": model_mode,
            "cosmos_model_mode_note": (
                "Requires a custom Cosmos data/inference wrapper that sets "
                "SequencePlan(has_action=True, condition_frame_indexes_vision=[0..7]) "
                "instead of default single-frame I2V."
            ),
            "domain_name": args.domain_name,
            "domain_id": int(args.domain_id),
            "raw_action_dim": len(VECTOR_NAMES),
            "action_chunk_size": args.total_video_frames - 1,
            "condition_prefix_frames": args.condition_prefix_frames,
            "condition_latent_frames": args.condition_latent_frames,
            "condition_frame_indexes_vision": list(range(args.condition_latent_frames)),
            "condition_frame_indexes_action": condition_frame_indexes_action,
            "condition_policy": (
                f"{mode_note}; "
                "no_future_ground_truth_object_pose_condition"
            ),
            "fps": int(item.get("fps", 30)),
            "vision_path": str(video_path),
            "action_path": str(action_path),
            "prompt": prompt,
            "metadata": {**safe_metadata, "source_h5": item.get("input_h5"), "split": split},
        }
        if state_target_path is not None:
            record["state_target_path"] = str(state_target_path)
            record["task_state_target_path"] = str(state_target_path)
            record["state_target_frame_count"] = args.total_video_frames
            record["state_target_vector_names"] = STATE_TARGET_VECTOR_NAMES
        records[split].append(record)
        if sample_id not in source_video_records:
            raise KeyError(f"missing source video JSONL record for {sample_id}")
        sft_record = json.loads(json.dumps(source_video_records[sample_id]))
        sft_record["vision_path"] = str(video_path.resolve())
        sft_record["action_path"] = str(action_path.resolve())
        sft_record["domain_name"] = args.domain_name
        sft_record["domain_id"] = int(args.domain_id)
        sft_record["raw_action_dim"] = len(VECTOR_NAMES)
        sft_record["max_action_dim"] = 64
        sft_record["action_chunk_size"] = args.total_video_frames - 1
        sft_record["model_mode"] = model_mode
        sft_record["condition_frame_indexes_vision"] = list(range(args.condition_latent_frames))
        sft_record["condition_frame_indexes_action"] = condition_frame_indexes_action
        sft_record["condition_prefix_frames"] = args.condition_prefix_frames
        sft_record["action_condition_policy"] = record["condition_policy"]
        if args.sanitize_future_caption:
            sft_record["t2w_windows"] = [
                {
                    "caption": causal_caption,
                    "start_frame": 0,
                    "end_frame": args.total_video_frames - 1,
                    "temporal_interval": 1,
                }
            ]
            sft_record["metadata"] = safe_metadata
        else:
            sft_record.setdefault("metadata", {})
            sft_record["metadata"].update(safe_metadata)
        if state_target_path is not None:
            sft_record["state_target_path"] = str(state_target_path.resolve())
            sft_record["task_state_target_path"] = str(state_target_path.resolve())
            sft_record["state_target_frame_count"] = args.total_video_frames
            sft_record["state_target_vector_names"] = STATE_TARGET_VECTOR_NAMES
        sft_records[split].append(sft_record)
        written.append(record)
        if args.progress_every > 0 and (idx == 1 or idx % args.progress_every == 0 or idx == len(videos)):
            _progress(
                "export_cosmos3_action_conditions_wrote_rows "
                f"{idx}/{len(videos)} sample_id={sample_id}"
            )

    for split in ("train", "val"):
        _write_jsonl(output_root / split / "action_condition_dataset_file.jsonl", records[split])
        _write_jsonl(output_root / split / "video_action_dataset_file.jsonl", sft_records[split])
    if stats is not None:
        (output_root / "normalization_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")

    output_manifest = {
        "args": asdict(args),
        "source_dataset_root": str(dataset_root),
        "num_records": len(written),
        "num_train": len(records["train"]),
        "num_val": len(records["val"]),
        "raw_action_dim": len(VECTOR_NAMES),
        "vector_names": VECTOR_NAMES,
        "state_target_vector_names": STATE_TARGET_VECTOR_NAMES,
        "write_state_targets": bool(args.write_state_targets),
        "sanitize_future_caption": bool(args.sanitize_future_caption),
        "normalization": "zscore" if stats is not None else "none",
        "action_condition_mode": args.action_condition_mode,
        "visual_input": "RGB only; depth is not used",
        "boundary": (
            "Structured Cosmos3 action/state conditions for manipulation. "
            "This is a data-interface artifact only; it is not training or "
            "controller evidence until consumed by a Cosmos action/video-prefix "
            "SFT wrapper and evaluated downstream."
        ),
        "required_sequence_plan": {
            "has_text": True,
            "has_vision": True,
            "has_action": True,
            "condition_frame_indexes_vision": list(range(args.condition_latent_frames)),
            "condition_frame_indexes_action": (
                list(range(args.total_video_frames - 1))
                if args.action_condition_mode == "forward_dynamics"
                else []
            ),
        },
        "state_target_contract": {
            "conditioned_on_future_state": False,
            "num_frames": args.total_video_frames if args.write_state_targets else 0,
            "caption_contains_future_ground_truth": not bool(args.sanitize_future_caption),
            "purpose": (
                "Evaluate full-length target-object, peg, TCP, qpos/qvel, grasp, "
                "insertion, and perturbation trajectories decoded from future RGB/action predictions."
            ),
        },
        "sft_jsonl": {
            "train": str(output_root / "train" / "video_action_dataset_file.jsonl"),
            "val": str(output_root / "val" / "video_action_dataset_file.jsonl"),
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "output_root": str(output_root), "num_records": len(written)}, sort_keys=True))


if __name__ == "__main__":
    main()
