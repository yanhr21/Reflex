#!/usr/bin/env python3
"""Sanitize Cosmos3 action-condition JSONL captions without rebuilding labels.

This reuses an existing full301 action/state-condition export and rewrites only
the SFT-facing JSONL captions/metadata so training text contains prefix/current
causal task state, not future ground-truth endpoint positions. Action and
state-target JSON files remain the already validated files from the source
export and are referenced by absolute path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import tyro


@dataclass
class Args:
    source_root: str
    output_root: str
    condition_prefix_frames: int = 29
    condition_latent_frames: int = 8


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty jsonl: {path}")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


def _state_at(state_path: Path, frame_idx: int) -> tuple[list[str], list[float]]:
    payload = json.loads(state_path.read_text())
    names = payload.get("state_vector_names") or []
    states = payload.get("states") or []
    if not names or not states:
        raise ValueError(f"missing state names/states in {state_path}")
    idx = max(0, min(frame_idx, len(states) - 1))
    row = states[idx]
    if len(row) != len(names):
        raise ValueError(f"state dim/name mismatch in {state_path}: {len(row)} != {len(names)}")
    return names, [float(x) for x in row]


def _get(names: list[str], values: list[float], key: str) -> float:
    return float(values[names.index(key)])


def _xyz(names: list[str], values: list[float], prefix: str) -> list[float]:
    return [_get(names, values, f"{prefix}_{axis}") for axis in ("x", "y", "z")]


def _fmt_xyz(values: list[float]) -> str:
    return "[" + ", ".join(f"{float(v):.4f}" for v in values) + "]"


def _scenario_text(scenario: str | None) -> str:
    return {
        "hole_constant": "the target hole may continue moving at approximately constant velocity",
        "hole_reverse": "the target hole may reverse direction during the rollout",
        "hole_move_stop": "the target hole may move and then stop during the rollout",
        "peg_drop": "the peg may require recovery from a drop or regrasp event",
        "peg_disturb": "the peg may be physically disturbed before recovery",
        "none": "the scene may remain static after the observed prefix",
    }.get(str(scenario), "the dynamic scene may change after the observed prefix")


def _sanitize_row(row: dict[str, Any], args: Args) -> dict[str, Any]:
    out = json.loads(json.dumps(row))
    prefix_idx = int(max(0, args.condition_prefix_frames - 1))
    state_path = Path(str(out.get("state_target_path") or out.get("task_state_target_path")))
    if not state_path.is_absolute():
        state_path = (Path(args.source_root) / state_path).resolve()
    names, values = _state_at(state_path, prefix_idx)

    scenario = (out.get("metadata") or {}).get("scenario")
    model_mode = str(out.get("model_mode") or "policy")
    if model_mode == "forward_dynamics":
        task = "predict future RGB and task state from the observed video prefix plus candidate action/state rows"
    else:
        task = "predict future RGB and the robot action chunk from the observed video prefix and causal prefix task state"

    tcp_xyz = _xyz(names, values, "tcp_pose")
    peg_xyz = _xyz(names, values, "peg_pose")
    hole_xyz = _xyz(names, values, "hole_pose")
    peg_head_at_hole_xyz = _xyz(names, values, "peg_head_at_hole")
    hole_velocity_xyz = _xyz(names, values, "hole_velocity_step")
    hole_delta_xyz = _xyz(names, values, "hole_delta_cumulative")
    peg_delta_xyz = _xyz(names, values, "peg_delta_applied")
    grasped = bool(_get(names, values, "grasped") > 0.5)
    inserted = bool(_get(names, values, "inserted") > 0.5)
    triggered = bool(_get(names, values, "perturb_triggered") > 0.5)

    caption = (
        "ManiSkill default angled-overhead RGB peg insertion rollout. "
        f"From the observed prefix, {_scenario_text(scenario)}; {task}. "
        f"Prefix frame {prefix_idx} state: "
        f"TCP xyz {_fmt_xyz(tcp_xyz)}; "
        f"peg xyz {_fmt_xyz(peg_xyz)}; "
        f"hole xyz {_fmt_xyz(hole_xyz)}; "
        f"peg-head relative-to-hole xyz {_fmt_xyz(peg_head_at_hole_xyz)}; "
        f"observed hole velocity xyz {_fmt_xyz(hole_velocity_xyz)}; "
        f"prefix cumulative hole perturbation xyz {_fmt_xyz(hole_delta_xyz)}; "
        f"prefix peg perturbation xyz {_fmt_xyz(peg_delta_xyz)}; "
        f"grasped={grasped}; inserted={inserted}; perturb_triggered={triggered}. "
        "Do not assume the original layout is restored; predict completion in the changed world."
    )

    out["t2w_windows"] = [
        {
            "caption": caption,
            "start_frame": 0,
            "end_frame": int(out.get("state_target_frame_count", 301)) - 1,
            "temporal_interval": 1,
        }
    ]
    source_metadata = out.get("metadata") or {}
    structured = source_metadata.get("structured_action_state_condition") or {}
    out["metadata"] = {
        "camera": source_metadata.get("camera", "PegInsertionSide-v1_default_human_render"),
        "conditioning_policy": "video_prefix_not_single_image_i2v",
        "fps": int(source_metadata.get("fps", out.get("fps", 30))),
        "scenario": scenario,
        "source": source_metadata.get("source", "maniskill_default_human_render_from_env_states"),
        "structured_action_state_condition": {
            **structured,
            "causal_boundary": (
                "Prefix/current causal task state only. Future ground-truth object, "
                "robot, insertion, and final positions are not included in captions "
                "or controller-facing metadata."
            ),
        },
        "causal_task_state_condition": {
            "condition_prefix_frames": int(args.condition_prefix_frames),
            "prefix_frame_index": prefix_idx,
            "tcp_xyz": tcp_xyz,
            "peg_xyz": peg_xyz,
            "hole_xyz": hole_xyz,
            "peg_head_at_hole_xyz": peg_head_at_hole_xyz,
            "hole_velocity_xyz": hole_velocity_xyz,
            "hole_delta_cumulative_xyz": hole_delta_xyz,
            "peg_delta_applied_xyz": peg_delta_xyz,
            "grasped": grasped,
            "inserted": inserted,
            "perturb_triggered": triggered,
            "boundary": (
                "Prefix/current causal task state only. Future ground-truth object, "
                "robot, insertion, and final positions are not included."
            ),
        },
        "caption_policy": "sanitized_prefix_only_no_future_ground_truth",
    }
    return out


def _validate_no_future_text(rows: list[dict[str, Any]]) -> None:
    forbidden = ("moves from", "] to [", "xyz_end", "source_frame_end", "grasped changes", "inserted changes")
    for row in rows:
        caption = str((row.get("t2w_windows") or [{}])[0].get("caption", ""))
        lowered = caption.lower()
        bad = [pat for pat in forbidden if pat in lowered]
        if bad:
            raise ValueError(f"future-text pattern {bad} remains in {row.get('uuid')}: {caption}")
        if "task_state_condition" in (row.get("metadata") or {}):
            raise ValueError(f"future task_state_condition metadata remains in {row.get('uuid')}")


def main() -> None:
    args = tyro.cli(Args)
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val"):
        rows = [_sanitize_row(row, args) for row in _read_jsonl(source_root / split / "video_action_dataset_file.jsonl")]
        _validate_no_future_text(rows)
        _write_jsonl(output_root / split / "video_action_dataset_file.jsonl", rows)
        all_rows[split] = rows

    source_manifest = json.loads((source_root / "manifest.json").read_text())
    manifest = {
        **source_manifest,
        "source_condition_root": str(source_root.resolve()),
        "num_train": len(all_rows["train"]),
        "num_val": len(all_rows["val"]),
        "num_records": len(all_rows["train"]) + len(all_rows["val"]),
        "sanitize_future_caption": True,
        "caption_contains_future_ground_truth": False,
        "action_and_state_target_files_reused_from_source": True,
        "sft_jsonl": {
            "train": str((output_root / "train" / "video_action_dataset_file.jsonl").resolve()),
            "val": str((output_root / "val" / "video_action_dataset_file.jsonl").resolve()),
        },
        "boundary": (
            "Clean-caption view of the validated full301 action/state condition export. "
            "Action and state-target JSON files are reused; only SFT-facing captions "
            "and metadata are sanitized to remove future ground-truth endpoint text."
        ),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "output_root": str(output_root), "num_records": manifest["num_records"]}))


if __name__ == "__main__":
    main()
