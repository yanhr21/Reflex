#!/usr/bin/env python3
"""Lightweight self-test for the Cosmos3 closed-loop objective gate."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from check_cosmos3_closed_loop_objective_gate import (
    check_full_episode_sample,
    check_static_sample,
)


def touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return str(path)


def counts_from_timeline(timeline: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for meta in timeline:
        controller = str(meta["controller"])
        counts[controller] = counts.get(controller, 0) + 1
    return counts


def make_timeline(*, trigger: int | None, wm_active: int = 0, static: bool = False) -> list[dict]:
    timeline: list[dict] = []
    wm_start = 91 if trigger is not None else 301
    wm_end = wm_start + int(wm_active)
    for frame in range(301):
        if frame == 0:
            controller = "INIT_OBS"
        elif static or trigger is None:
            controller = "DP_SCAN_TARGET"
        elif frame < wm_start:
            controller = "DP_SCAN_TARGET"
        elif frame < wm_end:
            controller = "WM_ACTIVE"
        else:
            controller = "DP_HANDOFF"
        timeline.append(
            {
                "controller": controller,
                "dp_active": controller.startswith("DP"),
                "frame_index": frame,
                "iteration": 0 if controller in {"WM_ACTIVE", "DP_HANDOFF"} else None,
                "prefix_role": "target_motion_observed" if controller == "WM_ACTIVE" else None,
                "target_motion_detected": bool(trigger is not None and frame >= trigger),
                "target_motion_trigger_frame": trigger,
                "wm_active": controller == "WM_ACTIVE",
            }
        )
    return timeline


def moving_sample(tmp: Path, *, mode: str = "target_motion_onset", wm_active: int = 8) -> dict:
    trigger = 90 if mode == "target_motion_onset" else None
    timeline = make_timeline(trigger=trigger, wm_active=wm_active)
    counts = counts_from_timeline(timeline)
    return {
        "sample_index": 0,
        "scenario": "hole_late_fast_shift",
        "full_episode_length_ok": True,
        "final_prefix_frame_index": 300,
        "final_observed_frames": 301,
        "controller_frame_counts": counts,
        "prefix_selection": {
            "mode": mode,
            "triggered": mode == "target_motion_onset",
            "wm_triggered": mode == "target_motion_onset",
            "detected_frame_index": trigger,
            "first_streak_frame_index": 89 if trigger is not None else None,
        },
        "controller_timeline": timeline,
        "final_eval": {"success": True},
        "final_observed_video": touch(tmp / "moving_raw.mp4"),
        "final_observed_annotated_video": touch(tmp / "moving_annotated.mp4"),
        "annotated_video_summary": {
            "frame_count": 301,
            "timeline": timeline,
            "controller_frame_counts": counts,
            "wm_active_frame_count": counts.get("WM_ACTIVE", 0),
            "dp_active_frame_count": counts.get("DP_SCAN_TARGET", 0) + counts.get("DP_HANDOFF", 0),
            "target_motion_detected_frame_count": sum(
                1 for meta in timeline if meta["target_motion_detected"]
            ),
        },
    }


def static_summary(tmp: Path, *, triggered: bool = False) -> dict:
    timeline = make_timeline(trigger=10 if triggered else None, wm_active=1 if triggered else 0, static=not triggered)
    counts = counts_from_timeline(timeline)
    return {
        "scenario": "none",
        "full_episode_length_ok": True,
        "final_prefix_frame_index": 300,
        "final_observed_frames": 301,
        "controller_frame_counts": counts,
        "prefix_selection": {
            "mode": (
                "target_motion_onset"
                if triggered
                else "target_motion_detector_never_triggered_after_terminal_completion"
            ),
            "pretrigger_control_mode": "frozen_dp_until_target_motion",
            "triggered": triggered,
            "wm_triggered": triggered,
            "detected_frame_index": 10 if triggered else None,
            "first_streak_frame_index": 9 if triggered else None,
        },
        "controller_timeline": timeline,
        "wm_active_frame_count": 1 if triggered else 0,
        "dp_active_frame_count": 300,
        "final_eval": {"success": False},
        "final_observed_video": touch(tmp / "static_raw.mp4"),
        "final_observed_annotated_video": touch(tmp / "static_annotated.mp4"),
        "annotated_video_summary": {
            "frame_count": 301,
            "timeline": timeline,
            "controller_frame_counts": counts,
            "wm_active_frame_count": counts.get("WM_ACTIVE", 0),
            "dp_active_frame_count": counts.get("DP_SCAN_TARGET", 0) + counts.get("DP_HANDOFF", 0),
            "target_motion_detected_frame_count": sum(
                1 for meta in timeline if meta["target_motion_detected"]
            ),
        },
    }


def assert_contains(row: dict, failure: str) -> None:
    failures = row.get("failures") or []
    if failure not in failures:
        raise AssertionError(f"expected {failure!r} in failures={failures}")


def write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True) + "\n")
    return path


def pure_dp_panel(tmp: Path, *, panel_contract_ok: bool = True) -> dict:
    return {
        "panel_full_episode_contract_ok": panel_contract_ok,
        "completed_samples": 1,
        "pure_dp_final_success_count": 0,
        "samples": [
            {
                "sample_index": 0,
                "scenario": "hole_late_fast_shift",
                "full_episode_length_ok": True,
                "final_success": False,
                "controller_frame_counts": {"PURE_DP": 300, "WM_ACTIVE": 0},
                "final_observed_video": touch(tmp / "pure_dp_raw.mp4"),
                "final_observed_annotated_video": touch(tmp / "pure_dp_annotated.mp4"),
            }
        ],
    }


def cosmos_panel(
    tmp: Path,
    *,
    name: str,
    panel_contract_ok: bool = True,
    completed_samples: int = 1,
    final_success_count: int = 1,
) -> dict:
    samples = [moving_sample(tmp / f"{name}_{idx}") for idx in range(max(1, int(completed_samples)))]
    return {
        "panel_full_episode_contract_ok": panel_contract_ok,
        "completed_samples": int(completed_samples),
        "final_success_count": int(final_success_count),
        "samples": samples,
    }


def run_objective_gate(
    tmp: Path,
    *,
    val_ok: bool = True,
    hard_ok: bool = True,
    pure_dp_ok: bool = True,
    hard_completed_samples: int = 1,
    hard_final_success_count: int = 1,
) -> dict:
    val_panel = write_json(tmp / "val_cosmos_panel.json", cosmos_panel(tmp, name="val", panel_contract_ok=val_ok))
    hard_panel = write_json(
        tmp / "hard_cosmos_panel.json",
        cosmos_panel(
            tmp,
            name="hard",
            panel_contract_ok=hard_ok,
            completed_samples=hard_completed_samples,
            final_success_count=hard_final_success_count,
        ),
    )
    pure_panel = write_json(tmp / "pure_dp_panel.json", pure_dp_panel(tmp, panel_contract_ok=pure_dp_ok))
    hard_action = write_json(
        tmp / "hard_action.json",
        {
            "sample_count": 1,
            "cosmos_final_success_count": 1,
            "pure_dp_final_success_count_on_matched": 0,
        },
    )
    static_path = write_json(tmp / "static_summary.json", static_summary(tmp / "static_e2e"))
    out_json = tmp / "objective_gate_out.json"
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("check_cosmos3_closed_loop_objective_gate.py")),
        "--val-cosmos-panel-summary",
        str(val_panel),
        "--val-pure-dp-panel-summary",
        str(pure_panel),
        "--hard-cosmos-panel-summary",
        str(hard_panel),
        "--hard-action-rebind-analysis",
        str(hard_action),
        "--static-no-motion-summary",
        str(static_path),
        "--output-json",
        str(out_json),
        "--skip-video-scan",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(
            f"objective gate subprocess failed with {proc.returncode}\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )
    return json.loads(out_json.read_text())


def assert_contract_failure(verdict: dict, failure: str) -> None:
    failures = verdict.get("contract_failures") or []
    if verdict.get("implementation_contract_ok") is not False or failure not in failures:
        raise AssertionError(
            f"expected implementation failure {failure!r}; "
            f"ok={verdict.get('implementation_contract_ok')} failures={failures}"
        )


def assert_method_failure(verdict: dict, failure_prefix: str) -> None:
    failures = verdict.get("method_effectiveness_failures") or []
    if verdict.get("method_effectiveness_ok") is not False or not any(
        str(item).startswith(failure_prefix) for item in failures
    ):
        raise AssertionError(
            f"expected method failure prefix {failure_prefix!r}; "
            f"ok={verdict.get('method_effectiveness_ok')} failures={failures}"
        )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cosmos3_objective_gate_selftest_") as tmp_text:
        tmp = Path(tmp_text)

        ok_moving = check_full_episode_sample(
            moving_sample(tmp),
            require_wm_on_trigger=True,
            scan_videos=False,
            min_moving_wm_active_frames=8,
        )
        if not ok_moving["ok"]:
            raise AssertionError(f"valid moving sample failed gate: {ok_moving}")

        low_wm = check_full_episode_sample(
            moving_sample(tmp, wm_active=1),
            require_wm_on_trigger=True,
            scan_videos=False,
            min_moving_wm_active_frames=8,
        )
        assert_contains(low_wm, "moving_sample_wm_active_frames_lt_8")

        manual = check_full_episode_sample(
            moving_sample(tmp, mode="manual"),
            require_wm_on_trigger=True,
            scan_videos=False,
            min_moving_wm_active_frames=8,
        )
        assert_contains(manual, "manual_prefix_selection_used")
        assert_contains(manual, "moving_sample_not_causal_target_motion_onset")
        assert_contains(manual, "moving_sample_missing_detected_frame_index")

        missing_detector = moving_sample(tmp)
        missing_detector["prefix_selection"]["detected_frame_index"] = None
        missing_detector["prefix_selection"]["first_streak_frame_index"] = None
        missing_detector_row = check_full_episode_sample(
            missing_detector,
            require_wm_on_trigger=True,
            scan_videos=False,
            min_moving_wm_active_frames=8,
        )
        assert_contains(missing_detector_row, "moving_sample_missing_detected_frame_index")
        assert_contains(missing_detector_row, "moving_sample_missing_first_streak_frame_index")

        missing_counts = moving_sample(tmp)
        missing_counts.pop("controller_frame_counts")
        missing_counts_row = check_full_episode_sample(
            missing_counts,
            require_wm_on_trigger=True,
            scan_videos=False,
            min_moving_wm_active_frames=8,
        )
        assert_contains(missing_counts_row, "missing_or_invalid_controller_frame_counts")

        short_count_sum = moving_sample(tmp)
        short_count_sum["controller_frame_counts"] = {"INIT_OBS": 1}
        short_count_sum_row = check_full_episode_sample(
            short_count_sum,
            require_wm_on_trigger=True,
            scan_videos=False,
            min_moving_wm_active_frames=8,
        )
        assert_contains(short_count_sum_row, "controller_frame_counts_sum_not_301")

        bad_count_sum = moving_sample(tmp)
        bad_count_sum["controller_frame_counts"] = {"INIT_OBS": 1, "DP_SCAN_TARGET": 300}
        bad_count_sum_row = check_full_episode_sample(
            bad_count_sum,
            require_wm_on_trigger=True,
            scan_videos=False,
            min_moving_wm_active_frames=8,
        )
        assert_contains(bad_count_sum_row, "controller_timeline_counts_mismatch")

        bad_annotated_counts = moving_sample(tmp)
        bad_annotated_counts["annotated_video_summary"]["controller_frame_counts"] = {
            "INIT_OBS": 1,
            "DP_SCAN_TARGET": 300,
        }
        bad_annotated_counts_row = check_full_episode_sample(
            bad_annotated_counts,
            require_wm_on_trigger=True,
            scan_videos=False,
            min_moving_wm_active_frames=8,
        )
        assert_contains(bad_annotated_counts_row, "annotated_summary_controller_counts_mismatch")

        missing_annotated_counts = moving_sample(tmp)
        missing_annotated_counts["annotated_video_summary"].pop("controller_frame_counts")
        missing_annotated_counts_row = check_full_episode_sample(
            missing_annotated_counts,
            require_wm_on_trigger=True,
            scan_videos=False,
            min_moving_wm_active_frames=8,
        )
        assert_contains(
            missing_annotated_counts_row,
            "missing_or_invalid_annotated_controller_frame_counts",
        )

        static_path = tmp / "static_summary.json"
        static_path.write_text(json.dumps(static_summary(tmp), sort_keys=True) + "\n")
        ok_static = check_static_sample(static_path, scan_videos=False)
        if ok_static is None or not ok_static["ok"]:
            raise AssertionError(f"valid static sample failed gate: {ok_static}")

        wrong_pretrigger = static_summary(tmp)
        wrong_pretrigger["prefix_selection"]["pretrigger_control_mode"] = "source_restore"
        wrong_pretrigger_path = tmp / "wrong_pretrigger_static_summary.json"
        wrong_pretrigger_path.write_text(json.dumps(wrong_pretrigger, sort_keys=True) + "\n")
        wrong_pretrigger_static = check_static_sample(wrong_pretrigger_path, scan_videos=False)
        if wrong_pretrigger_static is None:
            raise AssertionError("wrong-pretrigger static sample produced no gate row")
        assert_contains(
            wrong_pretrigger_static,
            "static_sample_not_same_frozen_dp_until_target_motion_controller",
        )

        bad_static_path = tmp / "bad_static_summary.json"
        bad_static_path.write_text(json.dumps(static_summary(tmp, triggered=True), sort_keys=True) + "\n")
        bad_static = check_static_sample(bad_static_path, scan_videos=False)
        if bad_static is None:
            raise AssertionError("bad static sample produced no gate row")
        assert_contains(bad_static, "static_sample_detector_triggered")
        assert_contains(bad_static, "static_sample_not_causal_detector_never_triggered")
        assert_contains(bad_static, "static_sample_has_detected_frame_index")
        assert_contains(bad_static, "static_sample_used_wm")

        e2e_ok = run_objective_gate(tmp / "e2e_ok")
        if not e2e_ok.get("implementation_contract_ok"):
            raise AssertionError(f"valid e2e objective gate failed: {e2e_ok.get('contract_failures')}")
        if not e2e_ok.get("method_effectiveness_ok"):
            raise AssertionError(
                f"valid e2e objective gate unexpectedly failed method effectiveness: "
                f"{e2e_ok.get('method_effectiveness_failures')}"
            )

        assert_contract_failure(
            run_objective_gate(tmp / "e2e_bad_val_panel", val_ok=False),
            "val_cosmos_panel_full_episode_contract_false",
        )
        assert_contract_failure(
            run_objective_gate(tmp / "e2e_bad_hard_panel", hard_ok=False),
            "hard_cosmos_panel_full_episode_contract_false",
        )
        assert_contract_failure(
            run_objective_gate(tmp / "e2e_bad_pure_dp_panel", pure_dp_ok=False),
            "pure_dp_panel_full_episode_contract_false",
        )
        assert_method_failure(
            run_objective_gate(
                tmp / "e2e_hard_success_fraction_low",
                hard_completed_samples=3,
                hard_final_success_count=1,
            ),
            "hard_case_success_fraction_below_minimum:1/3<min_fraction:0.5",
        )

    print("cosmos3_closed_loop_objective_gate_selftest=passed")


if __name__ == "__main__":
    main()
