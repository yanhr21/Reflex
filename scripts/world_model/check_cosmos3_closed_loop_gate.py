#!/usr/bin/env python3
"""Check whether a Cosmos3 WAM eval root may feed closed-loop control.

This is a conservative gate checker. It does not turn numeric diagnostics into
controller success. A checkpoint can pass only when strict generated artifacts,
generated-RGB readout/profile, and explicit visual review all pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--readout-subdir", default="task_state_readout_v7_733")
    parser.add_argument(
        "--visual-review-status",
        choices=("missing", "pass", "fail"),
        default="missing",
        help="Explicit visual review verdict for all required review sheets/videos.",
    )
    parser.add_argument("--visual-review-note", default="")
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--allow-nonpassing-exit-zero",
        action="store_true",
        help="Write the verdict but exit 0 even when closed-loop is blocked.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def main() -> int:
    args = parse_args()
    eval_root = Path(args.eval_root).resolve()
    readout_root = eval_root / args.readout_subdir

    required = {
        "eval_artifact_inspection": eval_root / "eval_artifact_inspection.json",
        "readout_eval_summary": readout_root / "readout_eval_summary.json",
        "readout_failure_profile": readout_root / "readout_failure_profile.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]

    verdict: dict[str, Any] = {
        "eval_root": str(eval_root),
        "readout_root": str(readout_root),
        "required_files": {name: str(path) for name, path in required.items()},
        "missing_required_files": missing,
        "visual_review_status": args.visual_review_status,
        "visual_review_note": args.visual_review_note,
        "checks": {},
        "metrics": {},
        "closed_loop_allowed": False,
        "reasons": [],
        "boundary": (
            "This gate only permits a closed-loop smoke to start. It is not "
            "controller success evidence and does not override live simulator "
            "metrics or visual review."
        ),
    }

    if missing:
        verdict["reasons"].append("missing_required_artifacts")
    else:
        eval_inspect = read_json(required["eval_artifact_inspection"])
        readout_summary = read_json(required["readout_eval_summary"])
        profile = read_json(required["readout_failure_profile"])

        strict_eval_ok = bool(eval_inspect.get("strict_eval_artifacts_ok")) and not eval_inspect.get(
            "strict_failures"
        )
        strict_readout_ok = bool(readout_summary.get("strict_readout_eval_ok")) and not readout_summary.get(
            "strict_failures"
        )
        strict_profile_ok = bool(profile.get("strict_profile_ok")) and not profile.get("failures")
        visual_ok = args.visual_review_status == "pass"

        verdict["checks"] = {
            "strict_generated_artifacts_ok": strict_eval_ok,
            "generated_rgb_readout_ok": strict_readout_ok,
            "generated_rgb_failure_profile_ok": strict_profile_ok,
            "explicit_visual_review_pass": visual_ok,
        }
        verdict["metrics"] = {
            "num_eval_samples": eval_inspect.get("num_samples"),
            "mean_future_video_psnr_db": nested(
                eval_inspect, "aggregate", "mean_future_video_psnr_db"
            ),
            "mean_action_rmse": nested(eval_inspect, "aggregate", "mean_action_rmse"),
            "mean_robot_action_future_rmse": nested(
                eval_inspect, "aggregate", "mean_robot_action_future_rmse"
            ),
            "mean_state_sidecar_future_rmse": nested(
                eval_inspect, "aggregate", "mean_state_sidecar_future_rmse"
            ),
            "mean_final_hole_pos_error_m": nested(
                profile, "aggregate", "mean_final_hole_pos_error_m"
            ),
            "mean_future_hole_rmse_m": nested(
                profile, "aggregate", "mean_future_hole_rmse_m"
            ),
            "mean_future_peg_rmse_m": nested(
                profile, "aggregate", "mean_future_peg_rmse_m"
            ),
            "mean_future_tcp_rmse_m": nested(
                profile, "aggregate", "mean_future_tcp_rmse_m"
            ),
            "mean_future_peg_head_hole_rmse_m": nested(
                profile, "aggregate", "mean_future_peg_head_hole_rmse_m"
            ),
        }

        if not strict_eval_ok:
            verdict["reasons"].append("strict_generated_artifacts_failed")
        if not strict_readout_ok:
            verdict["reasons"].append("generated_rgb_readout_failed")
        if not strict_profile_ok:
            verdict["reasons"].append("generated_rgb_failure_profile_failed")
        if not visual_ok:
            verdict["reasons"].append("explicit_visual_review_not_passed")

        verdict["closed_loop_allowed"] = bool(
            strict_eval_ok and strict_readout_ok and strict_profile_ok and visual_ok
        )

    output = json.dumps(verdict, indent=2, sort_keys=True)
    print(output)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n")

    if verdict["closed_loop_allowed"] or args.allow_nonpassing_exit_zero:
        return 0
    return 1 if missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
