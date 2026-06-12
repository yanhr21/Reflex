#!/usr/bin/env python3
"""Generate fix3 successful dynamic-hole source H5s for Cosmos3 SFT.

This is a postprocessor for successful expert source trajectories whose final
target poses have already been sampled for fix3. It must not be run directly on
the official static replay as method data: doing so would make the dynamic hole
return to the old static expert terminal pose, which is an invalid downgrade.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tyro


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
    "hole_move_stop_large",
    "hole_constant_large",
    "hole_reverse_large",
    "hole_sine_large",
    "hole_continuous_insert_large",
    "hole_late_shift_large",
)


@dataclass
class Args:
    source_h5: str = "data/official_replay/PegInsertionSide-v1/motionplanning/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    source_json: str = "data/official_replay/PegInsertionSide-v1/motionplanning/trajectory.state.pd_ee_delta_pose.physx_cpu.json"
    output_root: str = "experiments/world_model_task_rebinding/cosmos3/fix3_success_large_motion_env_states"
    paths_file: str = ""
    num_demos: int = 1000
    total_video_frames: int = 301
    total_action_steps: int = 300
    seed: int = 20260610
    min_motion_m: float = 0.18
    max_motion_m: float = 0.34
    val_fraction: float = 0.1
    overwrite: bool = False
    allow_static_terminal_bootstrap_diagnostic: bool = False


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


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q = q / norm
    if q[0] < 0:
        q = -q
    return q.astype(np.float32)


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = _normalize_quat(q)
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quat(q).astype(np.float64)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (_quat_to_matrix(q) @ np.asarray(v, dtype=np.float32)).astype(np.float32)


def _inv_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (_quat_to_matrix(q).T @ np.asarray(v, dtype=np.float32)).astype(np.float32)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _resample_indices(source_len: int, target_len: int) -> np.ndarray:
    if source_len <= 0:
        raise ValueError("source_len must be positive")
    return np.linspace(0, source_len - 1, target_len).round().astype(np.int64)


def _episode_success_ids(source_json: Path) -> list[dict[str, Any]]:
    data = json.loads(source_json.read_text())
    episodes = data.get("episodes") or []
    success = [episode for episode in episodes if bool(episode.get("success", False))]
    if not success:
        raise RuntimeError(f"no successful episodes in {source_json}")
    return success


def _sample_start_offset(rng: np.random.Generator, scenario: str, min_motion: float, max_motion: float) -> np.ndarray:
    magnitude = float(rng.uniform(min_motion, max_motion))
    if scenario == "hole_late_shift_large":
        magnitude = float(rng.uniform(max(min_motion, 0.24), max_motion + 0.04))
    angle = float(rng.uniform(-np.pi, np.pi))
    if scenario == "hole_continuous_insert_large":
        # Bias continuous tracking samples toward lateral target motion, where
        # a static DP should be most misled by the current target frame.
        angle = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.35 * np.pi, 0.65 * np.pi))
    offset = np.asarray([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32) * magnitude
    return offset.astype(np.float32)


def _progress_profile(scenario: str, frames: int, trigger: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(frames, dtype=np.float32)
    progress = np.zeros((frames,), dtype=np.float32)
    lateral = np.zeros((frames,), dtype=np.float32)

    if scenario == "hole_move_stop_large":
        settle = int(rng.integers(175, 235))
        progress = _smoothstep((t - trigger) / max(1, settle - trigger)).astype(np.float32)
        progress[t >= settle] = 1.0
    elif scenario == "hole_constant_large":
        end = int(rng.integers(285, frames))
        progress = np.clip((t - trigger) / max(1, end - trigger), 0.0, 1.0).astype(np.float32)
    elif scenario == "hole_reverse_large":
        mid = int(rng.integers(120, 175))
        settle = int(rng.integers(235, 291))
        up = _smoothstep((t - trigger) / max(1, mid - trigger))
        down = _smoothstep((t - mid) / max(1, settle - mid))
        progress = np.where(t <= mid, 1.25 * up, 1.25 - 0.25 * down).astype(np.float32)
        progress[t >= settle] = 1.0
    elif scenario == "hole_sine_large":
        end = int(rng.integers(265, frames))
        base = _smoothstep((t - trigger) / max(1, end - trigger))
        amp = float(rng.uniform(0.06, 0.13))
        phase = float(rng.uniform(0, 2 * np.pi))
        lateral = (amp * np.sin(2.0 * np.pi * base * float(rng.uniform(1.0, 2.0)) + phase) * (1.0 - base)).astype(np.float32)
        progress = base.astype(np.float32)
    elif scenario == "hole_continuous_insert_large":
        progress = np.clip((t - trigger) / max(1, (frames - 1) - trigger), 0.0, 1.0).astype(np.float32)
    elif scenario == "hole_late_shift_large":
        late = int(rng.integers(115, 155))
        settle = int(rng.integers(245, 286))
        progress = _smoothstep((t - late) / max(1, settle - late)).astype(np.float32)
        trigger = late
        progress[t >= settle] = 1.0
    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    progress[0] = 0.0
    progress[-1] = 1.0
    return progress.astype(np.float32), lateral.astype(np.float32)


def _target_hole_path(
    final_hole_pose: np.ndarray,
    scenario: str,
    frames: int,
    rng: np.random.Generator,
    min_motion: float,
    max_motion: float,
) -> tuple[np.ndarray, int]:
    trigger = int(rng.integers(42, 91))
    start_offset = _sample_start_offset(rng, scenario, min_motion, max_motion)
    progress, lateral = _progress_profile(scenario, frames, trigger, rng)
    unit = start_offset / max(float(np.linalg.norm(start_offset[:2])), 1e-8)
    orth = np.asarray([-unit[1], unit[0], 0.0], dtype=np.float32)
    poses = np.repeat(np.asarray(final_hole_pose, dtype=np.float32).reshape(1, 7), frames, axis=0)
    offsets = (1.0 - progress)[:, None] * start_offset[None, :] + lateral[:, None] * orth[None, :]
    offsets[-1] = 0.0
    poses[:, :3] = poses[:, :3] + offsets.astype(np.float32)
    return poses.astype(np.float32), trigger


def _stack_obs(obs: np.ndarray) -> np.ndarray:
    prev = np.concatenate([obs[:1], obs[:-1]], axis=0)
    return np.stack([prev, obs], axis=1).astype(np.float32)


def _state_dataset(group: h5py.Group, name: str, value: np.ndarray) -> None:
    group.create_dataset(name, data=np.asarray(value), compression="gzip", compression_opts=1)


def _write_dict_group(group: h5py.Group, data: dict[str, Any]) -> None:
    for key, value in data.items():
        if isinstance(value, dict):
            child = group.create_group(str(key))
            _write_dict_group(child, value)
        else:
            _state_dataset(group, str(key), np.asarray(value))


def _resampled_env_states(source_group: h5py.Group, state_indices: np.ndarray) -> dict[str, Any]:
    env_states = source_group["env_states"]
    out: dict[str, Any] = {"actors": {}, "articulations": {}}
    for section in ("actors", "articulations"):
        for name, dataset in env_states[section].items():
            arr = np.asarray(dataset, dtype=np.float32)[state_indices]
            if arr.ndim == 2:
                arr = arr[:, None, :]
            out[section][name] = arr.astype(np.float32)
    return out


def _peg_head_at_hole(peg_pose: np.ndarray, hole_pose: np.ndarray, peg_half_length: np.ndarray) -> np.ndarray:
    out = np.zeros((peg_pose.shape[0], 3), dtype=np.float32)
    for i in range(peg_pose.shape[0]):
        peg_q = _normalize_quat(peg_pose[i, 3:7])
        hole_q = _normalize_quat(hole_pose[i, 3:7])
        head_world = peg_pose[i, :3] + _rotate(peg_q, np.asarray([peg_half_length[i], 0.0, 0.0], dtype=np.float32))
        out[i] = _inv_rotate(hole_q, head_world - hole_pose[i, :3])
    return out


def _inserted_from_head(head_at_hole: np.ndarray, radius: np.ndarray) -> np.ndarray:
    radius = np.asarray(radius, dtype=np.float32).reshape(-1)
    x_flag = head_at_hole[:, 0] >= -0.015
    y_flag = np.abs(head_at_hole[:, 1]) <= radius
    z_flag = np.abs(head_at_hole[:, 2]) <= radius
    return (x_flag & y_flag & z_flag).astype(bool)


def _grasped_from_obs(obs: np.ndarray) -> np.ndarray:
    fingers = obs[:, 7:9].mean(axis=1)
    peg_tcp = np.linalg.norm(obs[:, LAYOUT["peg_pose"]][:, :3] - obs[:, LAYOUT["tcp_pose"]][:, :3], axis=1)
    return ((fingers < 0.026) & (peg_tcp < 0.16)).astype(bool)


def _scenario_counts(paths: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in paths:
        counts[item["scenario"]] = counts.get(item["scenario"], 0) + 1
    return dict(sorted(counts.items()))


def _stable_val(sample_id: str, val_fraction: float) -> bool:
    if val_fraction <= 0:
        return False
    if val_fraction >= 1:
        return True
    import hashlib

    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 10000) < int(round(val_fraction * 10000))


def _write_one(
    *,
    out_h5: Path,
    source_group: h5py.Group,
    source_episode: dict[str, Any],
    scenario: str,
    sample_index: int,
    rng: np.random.Generator,
    args: Args,
) -> dict[str, Any]:
    source_obs = np.asarray(source_group["obs"], dtype=np.float32)
    source_actions = np.asarray(source_group["actions"], dtype=np.float32)
    state_indices = _resample_indices(source_obs.shape[0], args.total_video_frames)
    action_indices = _resample_indices(source_actions.shape[0], args.total_action_steps)
    obs = source_obs[state_indices].astype(np.float32, copy=True)
    actions = source_actions[action_indices].astype(np.float32, copy=True)

    final_hole_pose = source_obs[-1, LAYOUT["hole_pose"]].astype(np.float32, copy=True)
    hole_pose, trigger_frame = _target_hole_path(
        final_hole_pose,
        scenario,
        args.total_video_frames,
        rng,
        args.min_motion_m,
        args.max_motion_m,
    )
    obs[:, LAYOUT["hole_pose"]] = hole_pose

    env_states = _resampled_env_states(source_group, state_indices)
    source_box = np.asarray(source_group["env_states"]["actors"]["box_with_hole"], dtype=np.float32)
    if source_box.ndim == 2:
        source_box = source_box[:, None, :]
    final_box_state = source_box[-1:].repeat(args.total_video_frames, axis=0).astype(np.float32)
    box_offset = (hole_pose[:, :3] - final_hole_pose[:3]).astype(np.float32)
    final_box_state[:, :, 0:3] += box_offset[:, None, :]
    final_box_state[:, :, 7:13] = 0.0
    env_states["actors"]["box_with_hole"] = final_box_state

    peg_pose = obs[:, LAYOUT["peg_pose"]].astype(np.float32)
    tcp_pose = obs[:, LAYOUT["tcp_pose"]].astype(np.float32)
    qpos = obs[:, LAYOUT["qpos"]].astype(np.float32)
    qvel = obs[:, LAYOUT["qvel"]].astype(np.float32)
    peg_half_length = obs[:, LAYOUT["peg_half_size"]][:, 0].astype(np.float32)
    radius = obs[:, LAYOUT["hole_radius"]][:, 0].astype(np.float32)
    peg_head = _peg_head_at_hole(peg_pose, hole_pose, peg_half_length)
    inserted = _inserted_from_head(peg_head, radius)
    grasped = _grasped_from_obs(obs)
    if not bool(inserted[-1]):
        raise RuntimeError(
            f"constructed final state is not inserted for sample={sample_index} "
            f"source_traj={source_episode.get('episode_id')} scenario={scenario} "
            f"head={peg_head[-1].tolist()} radius={float(radius[-1])}"
        )

    hole_velocity = np.zeros((args.total_video_frames, 3), dtype=np.float32)
    hole_velocity[1:] = hole_pose[1:, :3] - hole_pose[:-1, :3]
    hole_delta_applied = hole_pose[1:, :3] - hole_pose[:-1, :3]
    hole_delta_cumulative = hole_pose[1:, :3] - hole_pose[0:1, :3]
    triggered = np.arange(args.total_action_steps) >= max(0, trigger_frame - 1)
    trigger_steps = np.full((args.total_action_steps,), int(trigger_frame), dtype=np.int32)

    first_insert = int(np.flatnonzero(inserted)[0]) if inserted.any() else -1
    first_grasp = int(np.flatnonzero(grasped)[0]) if grasped.any() else -1
    final_motion = hole_pose[-1, :3] - hole_pose[0, :3]
    max_step = float(np.linalg.norm(hole_velocity, axis=1).max())
    sample_id = f"{scenario}_seed{args.seed + sample_index:06d}_src{int(source_episode['episode_id']):04d}_traj_0"
    summary = {
        "scenario": scenario,
        "seed": int(args.seed + sample_index),
        "source_episode_id": int(source_episode["episode_id"]),
        "source_episode_seed": int(source_episode.get("episode_seed", -1)),
        "source_control_mode": source_episode.get("control_mode", "pd_ee_delta_pose"),
        "steps": int(args.total_action_steps),
        "success_once": bool(inserted.any()),
        "success_at_end": bool(inserted[-1]),
        "inserted_end": bool(inserted[-1]),
        "first_insert_step": first_insert,
        "first_grasp_step": first_grasp,
        "trigger_step": int(trigger_frame),
        "trigger_reason": "large_kinematic_target_motion_over_successful_static_expert",
        "target_motion_norm_m": float(np.linalg.norm(final_motion)),
        "target_start_xyz": hole_pose[0, :3].astype(float).tolist(),
        "target_final_xyz": hole_pose[-1, :3].astype(float).tolist(),
        "target_motion_xyz": final_motion.astype(float).tolist(),
        "max_hole_step_m": max_step,
        "construction": (
            "fix3 overlays large target-hole motion on a successful official "
            "pd_ee_delta_pose expert trajectory; final target pose matches the "
            "expert insertion frame, so all generated source demos end inserted."
        ),
    }

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    if out_h5.exists():
        out_h5.unlink()
    with h5py.File(out_h5, "w") as h5:
        group = h5.create_group("traj_0")
        group.attrs["summary_json"] = json.dumps(_jsonable(summary), sort_keys=True)
        group.attrs["source_summary_json"] = json.dumps(_jsonable(summary), sort_keys=True)
        _state_dataset(group, "actions", actions)
        _state_dataset(group, "obs_current", obs)
        _state_dataset(group, "obs_stack", _stack_obs(obs))
        _state_dataset(group, "source_frame_indices", state_indices.astype(np.float32))
        _state_dataset(group, "rewards", np.zeros((args.total_action_steps,), dtype=np.float32))
        _state_dataset(group, "terminated", np.zeros((args.total_action_steps,), dtype=bool))
        truncated = np.zeros((args.total_action_steps,), dtype=bool)
        truncated[-1] = True
        _state_dataset(group, "truncated", truncated)
        slots = group.create_group("slots")
        _state_dataset(slots, "hole_pose", hole_pose)
        _state_dataset(slots, "peg_pose", peg_pose)
        _state_dataset(slots, "tcp_pose", tcp_pose)
        _state_dataset(slots, "qpos", qpos)
        _state_dataset(slots, "qvel", qvel)
        _state_dataset(slots, "hole_radius", radius)
        _state_dataset(slots, "peg_head_at_hole", peg_head)
        _state_dataset(slots, "hole_velocity_step", hole_velocity)
        _state_dataset(slots, "grasped", grasped)
        _state_dataset(slots, "inserted", inserted)
        perturb = group.create_group("perturb")
        _state_dataset(perturb, "hole_delta_applied", hole_delta_applied.astype(np.float32))
        _state_dataset(perturb, "hole_delta_cumulative", hole_delta_cumulative.astype(np.float32))
        _state_dataset(perturb, "peg_delta_applied", np.zeros((args.total_action_steps, 3), dtype=np.float32))
        _state_dataset(perturb, "triggered", triggered.astype(bool))
        _state_dataset(perturb, "trigger_step", trigger_steps)
        _state_dataset(perturb, "phase", np.full((args.total_action_steps,), SCENARIOS.index(scenario), dtype=np.int32))
        env_group = group.create_group("env_states")
        _write_dict_group(env_group, env_states)

    return {
        "sample_id": sample_id,
        "scenario": scenario,
        "source_episode_id": int(source_episode["episode_id"]),
        "source_episode_seed": int(source_episode.get("episode_seed", -1)),
        "path": str(out_h5),
        "num_video_frames": int(args.total_video_frames),
        "num_action_steps": int(args.total_action_steps),
        "success_at_end": bool(inserted[-1]),
        "first_insert_step": first_insert,
        "trigger_step": int(trigger_frame),
        "target_motion_norm_m": float(np.linalg.norm(final_motion)),
        "max_hole_step_m": max_step,
        "split": "val" if _stable_val(sample_id, args.val_fraction) else "train",
    }


def main() -> None:
    args = tyro.cli(Args)
    if args.total_video_frames != 301 or args.total_action_steps != 300:
        raise ValueError("fix3 must keep the 301-frame / 300-action full-episode contract")
    if args.min_motion_m <= 0 or args.max_motion_m < args.min_motion_m:
        raise ValueError("invalid target motion range")

    source_h5 = Path(args.source_h5)
    source_json = Path(args.source_json)
    output_root = Path(args.output_root)
    official_static = "official_replay/PegInsertionSide-v1/motionplanning" in str(source_h5)
    if official_static and not args.allow_static_terminal_bootstrap_diagnostic:
        raise RuntimeError(
            "Refusing to build fix3 from the official static replay terminal poses. "
            "That would make the target return to the static expert insertion pose. "
            "First generate successful expert source trajectories with widened/new "
            "fix3 final target poses, then run this postprocessor on that source. "
            "Use --allow-static-terminal-bootstrap-diagnostic only for a clearly "
            "labeled non-method smoke."
        )
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing non-empty output root without --overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    paths_file = Path(args.paths_file) if args.paths_file else output_root / "fix3_h5_paths.txt"

    episodes = _episode_success_ids(source_json)
    rng = np.random.default_rng(args.seed)
    records: list[dict[str, Any]] = []
    with h5py.File(source_h5, "r") as h5:
        for sample_index in range(int(args.num_demos)):
            source_episode = episodes[sample_index % len(episodes)]
            traj_name = f"traj_{int(source_episode['episode_id'])}"
            if traj_name not in h5:
                raise KeyError(f"missing source trajectory {traj_name}")
            scenario = SCENARIOS[sample_index % len(SCENARIOS)]
            sample_dir = output_root / f"{scenario}_seed{args.seed + sample_index:06d}_src{int(source_episode['episode_id']):04d}_idx{sample_index:04d}.fix3"
            out_h5 = sample_dir / f"{scenario}_seed{args.seed + sample_index:06d}_src{int(source_episode['episode_id']):04d}.h5"
            if out_h5.exists() and not args.overwrite:
                raise FileExistsError(out_h5)
            record = _write_one(
                out_h5=out_h5,
                source_group=h5[traj_name],
                source_episode=source_episode,
                scenario=scenario,
                sample_index=sample_index,
                rng=rng,
                args=args,
            )
            records.append(record)
            if sample_index == 0 or (sample_index + 1) % 50 == 0 or sample_index + 1 == args.num_demos:
                print(
                    json.dumps(
                        {
                            "event": "fix3_sample_written",
                            "index": sample_index,
                            "num_demos": int(args.num_demos),
                            "scenario": scenario,
                            "path": str(out_h5),
                            "target_motion_norm_m": record["target_motion_norm_m"],
                            "success_at_end": record["success_at_end"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    paths_file.parent.mkdir(parents=True, exist_ok=True)
    paths_file.write_text("\n".join(record["path"] for record in records) + "\n")
    motion = np.asarray([record["target_motion_norm_m"] for record in records], dtype=np.float32)
    manifest = {
        "args": _jsonable(asdict(args)),
        "boundary": (
            "Fix3 source data repair: every generated trajectory is full length, "
            "ends inserted, and uses large target-hole motion over successful "
            "official pd_ee_delta_pose expert robot/peg states. These are source "
            "SFT demonstrations, not controller evaluation evidence."
        ),
        "num_records": len(records),
        "num_success_at_end": int(sum(1 for record in records if record["success_at_end"])),
        "output_root": str(output_root),
        "paths_file": str(paths_file),
        "scenario_counts": _scenario_counts(records),
        "source_h5": str(source_h5),
        "source_json": str(source_json),
        "source_success_episodes_available": len(episodes),
        "target_motion_norm_m": {
            "min": float(motion.min()) if motion.size else None,
            "p50": float(np.percentile(motion, 50)) if motion.size else None,
            "p90": float(np.percentile(motion, 90)) if motion.size else None,
            "max": float(motion.max()) if motion.size else None,
        },
        "records": records,
    }
    (output_root / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "fix3_manifest_written", "manifest": str(output_root / "manifest.json"), "paths_file": str(paths_file), "num_records": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
