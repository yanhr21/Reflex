#!/usr/bin/env python3
"""Build a clean direct-contact executor manifest.

This manifest is the data bridge for the contact-action reset. Source suffixes
provide direct insertion positives; live replay labels provide hard negatives
and secondary DP handoff labels. Future labels are supervision only, never
controller-facing inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any

import numpy as np


def require_slurm_compute_step() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        host = socket.gethostname()
        raise SystemExit(
            "Refusing to run outside a Slurm allocation. "
            f"host={host}. Use the tmux-held interactive allocation and srun."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-suffix-bank-npz", required=True)
    parser.add_argument("--source-suffix-bank-jsonl", required=True)
    parser.add_argument("--contact-label-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--action-horizon", type=int, default=24)
    parser.add_argument("--hard-negative-jsonl", action="append", default=[])
    parser.add_argument("--policy-droid-label-json", action="append", default=[])
    parser.add_argument("--max-source-positives", type=int, default=0)
    parser.add_argument("--max-hard-negatives", type=int, default=0)
    parser.add_argument("--positive-offsets", default="0,8,16,24")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
    os.replace(tmp, path)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def parse_int_set(text: str) -> set[int]:
    out: set[int] = set()
    for item in str(text).split(","):
        item = item.strip()
        if item:
            out.add(int(item))
    return out


def decode_str_array(value: np.ndarray) -> list[str]:
    return [str(item) for item in np.asarray(value).reshape(-1).tolist()]


def pad_actions(actions: np.ndarray, horizon: int) -> tuple[np.ndarray, int]:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(f"expected action array shape (T,7), got {arr.shape}")
    valid = min(int(horizon), int(arr.shape[0]))
    if valid <= 0:
        raise ValueError("empty action chunk")
    chunk = arr[:valid]
    if valid < int(horizon):
        pad = np.repeat(chunk[-1:, :], int(horizon) - valid, axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk.astype(np.float32), int(valid)


def stats_1d(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def load_contact_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    episode_jsonl = root / "contact_progress_episode_labels.jsonl"
    if not episode_jsonl.is_file():
        return index
    for row in read_jsonl(episode_jsonl):
        source_h5 = str(row.get("source_h5") or "")
        label = str(row.get("contact_label_npz") or "")
        if source_h5 and label:
            index[str(Path(source_h5).resolve())] = Path(label).resolve()
    return index


def contact_context(label_npz: Path | None, start: int, horizon: int) -> dict[str, Any]:
    if label_npz is None or not label_npz.is_file():
        return {
            "contact_label_npz": str(label_npz) if label_npz else "",
            "contact_label_available": False,
        }
    labels = np.load(str(label_npz), allow_pickle=False)
    n = int(labels["peg_head_at_hole"].shape[0])
    start_i = max(0, min(int(start), n - 1))
    end_i = max(0, min(int(start) + int(horizon), n - 1))
    future = slice(start_i + 1, end_i + 1)
    phase_id = int(np.asarray(labels["phase_id"])[start_i])
    phase_names = [str(x) for x in np.asarray(labels["phase_names"]).tolist()]
    phase = phase_names[phase_id] if 0 <= phase_id < len(phase_names) else str(phase_id)
    inserted = np.asarray(labels["inserted"], dtype=bool)
    dp_cont = np.asarray(labels["dp_continuable"], dtype=bool)
    grasped = np.asarray(labels["grasped"], dtype=bool)
    progress = np.asarray(labels["contact_progress"], dtype=np.float32)
    head = np.asarray(labels["peg_head_at_hole"], dtype=np.float32)
    return {
        "contact_label_npz": str(label_npz),
        "contact_label_available": True,
        "current_phase_id": phase_id,
        "current_phase": phase,
        "current_peg_head_at_hole": head[start_i, :3].astype(float).tolist(),
        "end_peg_head_at_hole": head[end_i, :3].astype(float).tolist(),
        "current_contact_progress": float(progress[start_i]),
        "end_contact_progress": float(progress[end_i]),
        "contact_progress_delta": float(progress[end_i] - progress[start_i]),
        "current_grasped": bool(grasped[start_i]),
        "current_inserted": bool(inserted[start_i]),
        "current_dp_continuable": bool(dp_cont[start_i]),
        "future_inserted_within_horizon": bool(inserted[future].any()) if start_i + 1 <= end_i else False,
        "future_dp_continuable_within_horizon": bool(dp_cont[future].any()) if start_i + 1 <= end_i else False,
        "end_inserted": bool(inserted[end_i]),
        "end_dp_continuable": bool(dp_cont[end_i]),
        "end_grasped": bool(grasped[end_i]),
    }


def add_source_positive_rows(
    *,
    args: argparse.Namespace,
    actions_out: list[np.ndarray],
    rows_out: list[dict[str, Any]],
    summary_counts: Counter[str],
) -> None:
    bank_npz = Path(args.source_suffix_bank_npz).resolve()
    bank_jsonl = Path(args.source_suffix_bank_jsonl).resolve()
    contact_root = Path(args.contact_label_root).resolve()
    contact_index = load_contact_index(contact_root)
    bank = np.load(str(bank_npz), allow_pickle=False)
    rows_meta = read_jsonl(bank_jsonl)
    actions = np.asarray(bank["actions"], dtype=np.float32)
    source_h5 = decode_str_array(bank["source_h5"])
    source_uuid = decode_str_array(bank["source_uuid"])
    scenario = decode_str_array(bank["scenario"])
    start_frame = np.asarray(bank["start_frame"], dtype=np.int32).reshape(-1)
    offset = np.asarray(bank["offset_before_insert"], dtype=np.int32).reshape(-1)
    valid_steps = np.asarray(bank["valid_steps"], dtype=np.int32).reshape(-1)
    inserted = np.asarray(bank["inserted_within_chunk"], dtype=bool).reshape(-1)
    grasped = np.asarray(bank["grasped_at_start"], dtype=bool).reshape(-1)
    allowed_offsets = parse_int_set(str(args.positive_offsets))
    horizon = int(args.action_horizon)
    kept = 0
    for i in range(actions.shape[0]):
        if args.max_source_positives > 0 and kept >= int(args.max_source_positives):
            break
        if not bool(inserted[i]) or not bool(grasped[i]):
            summary_counts["source_skip_not_inserted_or_grasped"] += 1
            continue
        if int(offset[i]) not in allowed_offsets or int(offset[i]) > horizon:
            summary_counts["source_skip_offset_outside_direct_horizon"] += 1
            continue
        if int(valid_steps[i]) < horizon:
            summary_counts["source_skip_short_valid_steps"] += 1
            continue
        chunk, valid = pad_actions(actions[i], horizon)
        row_i = len(rows_out)
        actions_out.append(chunk)
        label_path = contact_index.get(str(Path(source_h5[i]).resolve()))
        ctx = contact_context(label_path, int(start_frame[i]), horizon)
        direct_positive = bool(ctx.get("future_inserted_within_horizon", False)) or int(offset[i]) <= horizon
        rows_out.append(
            {
                "schema": "direct_contact_executor_manifest_row_v1",
                "array_index": row_i,
                "split_hint": "source_positive",
                "sample_kind": "source_direct_positive",
                "candidate_family": "source_insertion_suffix",
                "source_h5": source_h5[i],
                "source_uuid": source_uuid[i],
                "scenario": scenario[i],
                "action_start_frame": int(start_frame[i]),
                "action_horizon": horizon,
                "valid_action_steps": valid,
                "offset_before_insert": int(offset[i]),
                "direct_contact_positive": direct_positive,
                "direct_inserted_within_horizon": bool(ctx.get("future_inserted_within_horizon", direct_positive)),
                "direct_contact_stable_or_dp_continuable": bool(ctx.get("future_dp_continuable_within_horizon", direct_positive)),
                "grasp_preserved_label": bool(ctx.get("end_grasped", True)),
                "dp96_success_secondary": None,
                "dp96_continuable_secondary": None,
                "source_bank_row": i,
                "source_bank_jsonl_row": rows_meta[i] if i < len(rows_meta) else {},
                "supervision_role": "primary_action_positive",
                "boundary": (
                    "Source suffix action target. Future contact labels are supervision "
                    "only and must not be used as live controller inputs."
                ),
                **ctx,
            }
        )
        kept += 1
        summary_counts["source_positive_rows"] += 1


def find_candidate_actions_from_bank(path: Path, candidate_index: int, horizon: int) -> tuple[np.ndarray, str]:
    data = np.load(str(path), allow_pickle=True)
    for key in ("candidate_full_actions", "candidate_actions", "actions", "action_bank", "robot_actions"):
        if key not in data:
            continue
        arr = np.asarray(data[key], dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 7 and 0 <= int(candidate_index) < arr.shape[0]:
            chunk, _valid = pad_actions(arr[int(candidate_index)], horizon)
            return chunk, key
    for key in data.files:
        arr = np.asarray(data[key])
        if arr.ndim == 3 and arr.shape[-1] == 7 and 0 <= int(candidate_index) < arr.shape[0]:
            chunk, _valid = pad_actions(arr[int(candidate_index)].astype(np.float32), horizon)
            return chunk, key
    raise KeyError(f"no candidate action array found in {path}")


def find_candidate_actions_from_chunk_json(path: Path, horizon: int) -> tuple[np.ndarray, int, str]:
    payload = read_json(path)
    for key in ("denormalized_robot_action_chunk", "robot_action_chunk", "actions"):
        if key not in payload:
            continue
        chunk, valid = pad_actions(np.asarray(payload[key], dtype=np.float32), horizon)
        return chunk, valid, key
    raise KeyError(f"no action chunk array found in {path}")


def add_replay_label_row(
    *,
    label: dict[str, Any],
    actions_out: list[np.ndarray],
    rows_out: list[dict[str, Any]],
    summary_counts: Counter[str],
    horizon: int,
    source_label_path: str,
    max_hard_negatives: int,
) -> None:
    if max_hard_negatives > 0 and summary_counts["hard_negative_rows"] >= max_hard_negatives:
        return
    after_success = bool(label.get("after_success", False))
    after_inserted = bool(label.get("after_inserted_live_pose", False))
    after_gate = bool((label.get("after_continuability_gate") or {}).get("ok", False))
    after_contact = bool(label.get("after_contact_stable_proxy", False))
    direct_positive = bool(after_success or after_inserted or after_gate or after_contact)
    if direct_positive:
        summary_counts["replay_skip_direct_positive"] += 1
        return
    persisted_chunk_path = Path(str(label.get("persisted_action_chunk_json") or "")).resolve()
    bank_path_text = ""
    action_source = ""
    action_key = ""
    valid_from_chunk: int | None = None
    try:
        if persisted_chunk_path.is_file():
            chunk, valid_from_chunk, action_key = find_candidate_actions_from_chunk_json(persisted_chunk_path, horizon)
            action_source = "persisted_action_chunk_json"
            bank_text = str(label.get("candidate_action_bank_npz") or "")
            bank_path_text = str(Path(bank_text).resolve()) if bank_text else ""
            candidate_index = int(label.get("candidate_index", -1))
            summary_counts["replay_action_loaded_from_persisted_chunk_json"] += 1
        else:
            bank_path = Path(str(label.get("candidate_action_bank_npz") or "")).resolve()
            candidate_index = int(label.get("candidate_index", -1))
            if not bank_path.is_file() or candidate_index < 0:
                summary_counts["replay_skip_missing_bank"] += 1
                return
            chunk, action_key = find_candidate_actions_from_bank(bank_path, candidate_index, horizon)
            action_source = "candidate_action_bank_npz"
            bank_path_text = str(bank_path)
    except Exception:
        meta = label.get("candidate_meta") if isinstance(label.get("candidate_meta"), dict) else {}
        if str(meta.get("candidate_source") or "") == "causal_suffix_diffusion":
            summary_counts["replay_skip_causal_suffix_action_not_persisted"] += 1
        else:
            summary_counts["replay_skip_action_load_failed"] += 1
        return
    row_i = len(rows_out)
    actions_out.append(chunk)
    dp = label.get("dp_rollout_continuability") if isinstance(label.get("dp_rollout_continuability"), dict) else {}
    rows_out.append(
        {
            "schema": "direct_contact_executor_manifest_row_v1",
            "array_index": row_i,
            "split_hint": "live_hard_negative",
            "sample_kind": "live_replay_hard_negative",
            "candidate_family": str((label.get("candidate_meta") or {}).get("candidate_source") or label.get("candidate_name") or "unknown"),
            "candidate_name": label.get("candidate_name"),
            "candidate_index": candidate_index,
            "candidate_action_bank_npz": bank_path_text,
            "persisted_action_chunk_json": str(persisted_chunk_path) if persisted_chunk_path.is_file() else "",
            "candidate_action_source": action_source,
            "candidate_action_array_key": action_key,
            "source_h5": label.get("source_h5"),
            "scenario": label.get("scenario"),
            "sample_name": label.get("sample_name"),
            "prefix_frame_index": label.get("prefix_frame_index"),
            "action_start_frame": label.get("prefix_frame_index"),
            "action_horizon": horizon,
            "valid_action_steps": min(int(label.get("execute_steps_actual") or valid_from_chunk or horizon), horizon),
            "direct_contact_positive": False,
            "direct_inserted_within_horizon": False,
            "direct_contact_stable_or_dp_continuable": after_gate,
            "after_success": after_success,
            "after_inserted_live_pose": after_inserted,
            "after_contact_stable_proxy": after_contact,
            "after_gate_ok": after_gate,
            "before_peg_head_at_hole": label.get("before_peg_head_at_hole"),
            "after_peg_head_at_hole": label.get("after_peg_head_at_hole"),
            "delta_abs_yz_sum": label.get("delta_abs_yz_sum"),
            "delta_yz_l2": label.get("delta_yz_l2"),
            "grasp_preserved_label": bool(label.get("after_grasped", False)),
            "dp96_success_secondary": bool(dp.get("success", False)),
            "dp96_continuable_secondary": bool(dp.get("continuable", False)),
            "dp96_final_contact_stable_secondary": bool(dp.get("final_contact_stable", False)),
            "source_label_path": source_label_path,
            "supervision_role": "hard_negative_value_or_contrastive",
            "boundary": (
                "Failed live replay action. Use as hard negative/value label, not "
                "as a positive imitation target."
            ),
        }
    )
    summary_counts["hard_negative_rows"] += 1


def add_policy_droid_label_row(
    *,
    label_path: Path,
    actions_out: list[np.ndarray],
    rows_out: list[dict[str, Any]],
    summary_counts: Counter[str],
    horizon: int,
) -> None:
    label = read_json(label_path)
    after_success = bool(label.get("after_success", False))
    after_inserted = bool(label.get("after_inserted_live_pose", False))
    after_gate = bool((label.get("after_continuability_gate") or {}).get("ok", False))
    after_contact = bool(label.get("after_contact_stable_proxy", False))
    direct_positive = bool(after_success or after_inserted or after_gate or after_contact)
    if direct_positive:
        summary_counts["policy_droid_skip_direct_positive"] += 1
        return
    chunk_json = Path(str(label.get("action_chunk_json") or "")).resolve()
    if not chunk_json.is_file():
        summary_counts["policy_droid_skip_missing_chunk_json"] += 1
        return
    chunk_payload = read_json(chunk_json)
    try:
        chunk, valid = pad_actions(np.asarray(chunk_payload["denormalized_robot_action_chunk"], dtype=np.float32), horizon)
    except Exception:
        summary_counts["policy_droid_skip_action_load_failed"] += 1
        return
    row_i = len(rows_out)
    actions_out.append(chunk)
    dp = label.get("dp_rollout_continuability") if isinstance(label.get("dp_rollout_continuability"), dict) else {}
    meta = label.get("candidate_meta") if isinstance(label.get("candidate_meta"), dict) else {}
    rows_out.append(
        {
            "schema": "direct_contact_executor_manifest_row_v1",
            "array_index": row_i,
            "split_hint": "live_hard_negative",
            "sample_kind": "policy_droid_live_hard_negative",
            "candidate_family": "cosmos_policy_droid",
            "candidate_name": label.get("candidate_name"),
            "action_chunk_json": str(chunk_json),
            "sample_output_json": meta.get("sample_output_json"),
            "snapshot_state_h5": meta.get("snapshot_state_h5"),
            "history_action_state_json": meta.get("history_action_state_json"),
            "source_h5": meta.get("source_h5"),
            "prefix_frame_index": chunk_payload.get("prefix_frame_index"),
            "action_start_frame": chunk_payload.get("chunk_start"),
            "action_horizon": horizon,
            "valid_action_steps": valid,
            "direct_contact_positive": False,
            "direct_inserted_within_horizon": False,
            "direct_contact_stable_or_dp_continuable": after_gate,
            "after_success": after_success,
            "after_inserted_live_pose": after_inserted,
            "after_contact_stable_proxy": after_contact,
            "after_gate_ok": after_gate,
            "before_peg_head_at_hole": label.get("before_peg_head_at_hole"),
            "after_peg_head_at_hole": label.get("after_peg_head_at_hole"),
            "delta_abs_yz_sum": label.get("delta_abs_yz_sum"),
            "delta_yz_l2": label.get("delta_yz_l2"),
            "grasp_preserved_label": bool(label.get("after_grasped", False)),
            "dp96_success_secondary": bool(dp.get("success", False)),
            "dp96_continuable_secondary": bool(dp.get("continuable", False)),
            "dp96_final_contact_stable_secondary": bool(dp.get("final_contact_stable", False)),
            "source_label_path": str(label_path),
            "supervision_role": "hard_negative_value_or_contrastive",
            "boundary": (
                "Policy-DROID failed live-snapshot replay action. Use as hard "
                "negative/value label, not as a positive imitation target."
            ),
        }
    )
    summary_counts["policy_droid_hard_negative_rows"] += 1


def main() -> int:
    require_slurm_compute_step()
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    horizon = int(args.action_horizon)
    if horizon <= 0:
        raise SystemExit("--action-horizon must be positive")

    actions_out: list[np.ndarray] = []
    rows_out: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    add_source_positive_rows(args=args, actions_out=actions_out, rows_out=rows_out, summary_counts=counts)

    for jsonl in args.hard_negative_jsonl:
        path = Path(jsonl).resolve()
        if not path.is_file():
            counts["hard_negative_jsonl_missing"] += 1
            continue
        for label in read_jsonl(path):
            add_replay_label_row(
                label=label,
                actions_out=actions_out,
                rows_out=rows_out,
                summary_counts=counts,
                horizon=horizon,
                source_label_path=str(path),
                max_hard_negatives=int(args.max_hard_negatives),
            )

    for item in args.policy_droid_label_json:
        path = Path(item).resolve()
        if not path.is_file():
            counts["policy_droid_label_json_missing"] += 1
            continue
        add_policy_droid_label_row(
            label_path=path,
            actions_out=actions_out,
            rows_out=rows_out,
            summary_counts=counts,
            horizon=horizon,
        )

    if not rows_out:
        raise RuntimeError("no direct-contact manifest rows built")
    actions = np.stack(actions_out).astype(np.float32)
    npz_path = output_root / "direct_contact_executor_manifest_arrays.npz"
    np.savez_compressed(
        npz_path,
        schema=np.asarray(["direct_contact_executor_manifest_arrays_v1"]),
        actions=actions,
        action_horizon=np.asarray([horizon], dtype=np.int32),
        robot_action_dim=np.asarray([7], dtype=np.int32),
        direct_contact_positive=np.asarray([bool(row["direct_contact_positive"]) for row in rows_out], dtype=bool),
        direct_inserted_within_horizon=np.asarray([bool(row["direct_inserted_within_horizon"]) for row in rows_out], dtype=bool),
        hard_negative=np.asarray([row["sample_kind"].endswith("hard_negative") for row in rows_out], dtype=bool),
    )
    manifest_jsonl = output_root / "direct_contact_executor_manifest.jsonl"
    for row in rows_out:
        row["actions_npz"] = str(npz_path)
    write_jsonl(manifest_jsonl, rows_out)
    kind_counts = Counter(str(row["sample_kind"]) for row in rows_out)
    family_counts = Counter(str(row.get("candidate_family", "unknown")) for row in rows_out)
    source_positive_rows = [row for row in rows_out if row["sample_kind"] == "source_direct_positive"]
    hard_negative_rows = [row for row in rows_out if row["sample_kind"].endswith("hard_negative")]
    summary = {
        "schema": "direct_contact_executor_manifest_summary_v1",
        "output_root": str(output_root),
        "manifest_jsonl": str(manifest_jsonl),
        "actions_npz": str(npz_path),
        "action_horizon": horizon,
        "num_rows": len(rows_out),
        "num_source_direct_positive_rows": len(source_positive_rows),
        "num_hard_negative_rows": len(hard_negative_rows),
        "sample_kind_counts": dict(sorted(kind_counts.items())),
        "candidate_family_counts": dict(sorted(family_counts.items())),
        "counts": dict(sorted(counts.items())),
        "source_suffix_bank_npz": str(Path(args.source_suffix_bank_npz).resolve()),
        "source_suffix_bank_jsonl": str(Path(args.source_suffix_bank_jsonl).resolve()),
        "contact_label_root": str(Path(args.contact_label_root).resolve()),
        "hard_negative_jsonl": [str(Path(x).resolve()) for x in args.hard_negative_jsonl],
        "policy_droid_label_json": [str(Path(x).resolve()) for x in args.policy_droid_label_json],
        "action_abs_stats": stats_1d(np.abs(actions)),
        "ready_for_direct_contact_executor_training": bool(source_positive_rows and hard_negative_rows),
        "training_boundary": (
            "This is a manifest/data assembly result, not training or method "
            "evidence. Direct positives may train action generation; hard "
            "negatives should train value/risk or contrastive rejection."
        ),
        "causal_boundary": (
            "Rows may include future direct-contact labels only as targets. "
            "A live controller must condition on current/history RGB-derived "
            "state, proprio/action history, and causal task context."
        ),
    }
    write_json(output_root / "direct_contact_executor_manifest_summary.json", summary)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary["ready_for_direct_contact_executor_training"] else 64


if __name__ == "__main__":
    raise SystemExit(main())
