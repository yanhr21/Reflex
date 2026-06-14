#!/usr/bin/env python3
"""Self-test for live-query/training-coverage audit."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from audit_cosmos3_live_query_training_coverage import audit


def write_json(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True) + "\n")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    return path


def training_row(uuid: str, *, role: str, mode: str, prefix: int, rel: list[float]) -> dict:
    return {
        "uuid": uuid,
        "prefix_frame_index": prefix,
        "prefix_role": role,
        "physical_mode": mode,
        "condition_frame_indexes_action": list(range(prefix)),
        "metadata": {
            "scenario": "synthetic",
            "prefix_causal_state": {
                "prefix_frame_index": prefix,
                "prefix_role": role,
                "mode": mode,
                "peg_head_at_hole_xyz": rel,
                "target_motion_observed": role != "target_pre_motion",
                "grasped": True,
                "inserted": False,
            },
        },
    }


def iteration(index: int, *, role: str, prefix: int, rel: list[float]) -> dict:
    return {
        "iteration": index,
        "controller_step_type": "cosmos_rebind_short_chunk",
        "prefix_frame_index": prefix,
        "prefix_role": role,
        "prefix_role_info": {"role": role, "source": "synthetic"},
        "pre_controller_continuability_gate": {
            "ok": False,
            "peg_head_at_hole": rel,
            "checks": {"grasped": True},
        },
        "chunk_start": prefix,
        "chunk_end_exclusive": prefix + 8,
        "chunk_steps": 8,
        "after_eval": {"success": False},
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cosmos3_live_query_coverage_selftest_") as tmp_text:
        tmp = Path(tmp_text)
        condition_root = tmp / "condition_root"
        write_jsonl(
            condition_root / "train" / "video_action_dataset_file.jsonl",
            [
                training_row(
                    "covered",
                    role="target_post_motion",
                    mode="target_post_motion",
                    prefix=100,
                    rel=[-0.10, 0.01, -0.01],
                ),
                training_row(
                    "mismatch_only",
                    role="target_post_motion",
                    mode="target_pre_motion",
                    prefix=200,
                    rel=[-0.30, 0.20, -0.10],
                ),
            ],
        )
        write_jsonl(condition_root / "val" / "video_action_dataset_file.jsonl", [])
        loop_summary = write_json(
            tmp / "loop_summary.json",
            {
                "sample_name": "synthetic_sample",
                "scenario": "synthetic",
                "final_eval": {"success": False},
                "prefix_selection": {"detected_frame_index": 88},
                "iterations": [
                    iteration(0, role="target_post_motion", prefix=104, rel=[-0.102, 0.011, -0.009]),
                    iteration(1, role="target_post_motion", prefix=250, rel=[-0.30, 0.20, -0.10]),
                ],
            },
        )
        args = SimpleNamespace(
            condition_root=condition_root,
            live_summary=[loop_summary],
            output_json=tmp / "out.json",
            output_md=None,
            prefix_tolerance=16,
            rel_l2_tolerance=0.05,
            rel_yz_tolerance=0.03,
            top_k=3,
        )
        report = audit(args)
        if report["train_row_count"] != 2:
            raise AssertionError(report["train_row_count"])
        if report["live_cosmos_query_count"] != 2:
            raise AssertionError(report["live_cosmos_query_count"])
        if report["train_role_mode_mismatch_count"] != 1:
            raise AssertionError(report["train_role_mode_mismatch_count"])
        undercovered = report["strict_local_coverage"]["undercovered_query_count"]
        if undercovered != 1:
            raise AssertionError(f"expected one undercovered query, got {undercovered}")
        rows = report["queries"]
        if rows[0]["undercovered_by_strict_local_criterion"]:
            raise AssertionError("covered query was marked undercovered")
        if not rows[1]["undercovered_by_strict_local_criterion"]:
            raise AssertionError("undercovered query was not detected")

    print("cosmos3_live_query_training_coverage_selftest=passed")


if __name__ == "__main__":
    main()
