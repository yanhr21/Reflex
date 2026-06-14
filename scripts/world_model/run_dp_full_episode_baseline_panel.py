#!/usr/bin/env python3
"""Run pure frozen-DP full-episode baselines for a Cosmos live-loop panel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from panel_contract_utils import panel_contract_rows, panel_full_episode_contract_ok  # noqa: E402
from run_cosmos3_live_receding_loop import require_compute_step, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-panel-summary",
        default="",
        help="Existing Cosmos live-panel summary. Uses its sample summaries as the comparison source.",
    )
    parser.add_argument(
        "--eval-manifest",
        default="",
        help="Eval-style manifest with samples/source_uuid for pure-DP-only screening.",
    )
    parser.add_argument("--source-h5-root", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dp-manifest", default="")
    parser.add_argument("--dp-checkpoint", default="")
    parser.add_argument("--dp-state-key", choices=("ema_agent", "agent"), default="ema_agent")
    parser.add_argument("--expected-video-frames", type=int, default=301)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--expected-action-dim", type=int, default=32)
    parser.add_argument("--robot-action-dim", type=int, default=7)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--external-target-mode", choices=("source_env_state", "none"), default="source_env_state")
    parser.add_argument("--sample-indices", default="")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--annotate-video", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sample_dir_name(sample: dict[str, Any], scenario: str) -> str:
    index = int(sample.get("sample_index", 0))
    return f"sample_{index:02d}_{scenario}"


def parse_indices(value: str, max_samples: int, available: int) -> list[int]:
    if value.strip():
        out: list[int] = []
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            idx = int(part)
            if idx < 0 or idx >= available:
                raise ValueError(f"sample index {idx} outside range 0..{available - 1}")
            out.append(idx)
        return out
    limit = available if int(max_samples) <= 0 else min(int(max_samples), available)
    return list(range(limit))


def source_uuid_from_sample(sample: dict[str, Any]) -> str:
    for key in ("source_uuid", "uuid", "sample_uuid"):
        value = sample.get(key)
        if isinstance(value, str) and value:
            return value
    extra = sample.get("extra")
    if isinstance(extra, dict):
        for key in ("source_uuid", "uuid", "sample_uuid"):
            value = extra.get(key)
            if isinstance(value, str) and value:
                return value
    raise KeyError(f"sample is missing source uuid: keys={sorted(sample.keys())}")


def source_h5_from_uuid(source_h5_root: Path, source_uuid: str) -> Path:
    base = source_uuid
    suffix = "_traj_0"
    if base.endswith(suffix):
        base = base[: -len(suffix)]
    if not base.endswith(".fix3"):
        raise ValueError(f"source uuid does not map to a fix3 source directory: {source_uuid}")
    h5_name = base.replace(".fix3", "") + ".h5"
    path = source_h5_root / base / h5_name
    if not path.is_file():
        raise FileNotFoundError(f"missing source H5 for {source_uuid}: {path}")
    return path


def load_sample_specs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if bool(args.source_panel_summary):
        source_panel_summary = Path(args.source_panel_summary).resolve()
        panel = read_json(source_panel_summary)
        rows = []
        samples = list(panel.get("samples") or [])
        if args.sample_limit > 0:
            samples = samples[: int(args.sample_limit)]
        for sample in samples:
            cosmos_summary_path = Path(str(sample["summary_path"])).resolve()
            cosmos_summary = read_json(cosmos_summary_path)
            rows.append(
                {
                    "sample": sample,
                    "cosmos_summary_path": cosmos_summary_path,
                    "cosmos_summary": cosmos_summary,
                    "source_h5": Path(str(cosmos_summary["source_h5"])).resolve(),
                    "scenario": str(cosmos_summary.get("scenario") or sample.get("scenario") or "unknown"),
                    "sample_name": str(cosmos_summary.get("sample_name") or sample.get("sample_name") or ""),
                    "external_target_mode": str(panel.get("external_target_mode") or args.external_target_mode),
                }
            )
        return rows, panel

    if bool(args.eval_manifest):
        eval_manifest = Path(args.eval_manifest).resolve()
        source_h5_root = Path(args.source_h5_root).resolve()
        if not source_h5_root.is_dir():
            raise FileNotFoundError(f"missing source_h5_root: {source_h5_root}")
        manifest = read_json(eval_manifest)
        samples = manifest.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"eval manifest has no samples: {eval_manifest}")
        selected = parse_indices(args.sample_indices, args.sample_limit, len(samples))
        rows = []
        for idx in selected:
            sample = dict(samples[idx])
            sample["sample_index"] = idx
            source_uuid = source_uuid_from_sample(sample)
            scenario = str(sample.get("scenario") or "unknown")
            name = str(sample.get("sample_name") or sample.get("name") or f"eval{idx:02d}_{scenario}_{source_uuid}")
            rows.append(
                {
                    "sample": sample,
                    "cosmos_summary_path": None,
                    "cosmos_summary": None,
                    "source_h5": source_h5_from_uuid(source_h5_root, source_uuid),
                    "scenario": scenario,
                    "sample_name": name,
                    "external_target_mode": args.external_target_mode,
                }
            )
        return rows, {"eval_manifest": str(eval_manifest), "source_h5_root": str(source_h5_root)}

    raise SystemExit("one of --source-panel-summary or --eval-manifest is required")


def main() -> int:
    args = parse_args()
    require_compute_step()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    specs, source_panel = load_sample_specs(args)

    dp_manifest = Path(args.dp_manifest or (source_panel or {}).get("dp_manifest") or "").resolve()
    dp_checkpoint = Path(args.dp_checkpoint or (source_panel or {}).get("dp_checkpoint") or "").resolve()
    if not dp_manifest.is_file():
        raise FileNotFoundError(f"missing dp manifest: {dp_manifest}")
    if not dp_checkpoint.is_file():
        raise FileNotFoundError(f"missing dp checkpoint: {dp_checkpoint}")

    results: list[dict[str, Any]] = []
    for spec in specs:
        sample = spec["sample"]
        cosmos_summary_path = spec.get("cosmos_summary_path")
        cosmos_summary = spec.get("cosmos_summary")
        source_h5 = Path(spec["source_h5"]).resolve()
        scenario = str(spec["scenario"])
        sample_name = str(spec.get("sample_name") or sample_dir_name(sample, scenario))
        out_dir = output_root / sample_dir_name(sample, scenario)
        out_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = out_dir / "pure_dp_stdout.log"
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "run_dp_full_episode_baseline.py"),
            "--source-h5",
            str(source_h5),
            "--output-root",
            str(out_dir),
            "--dp-manifest",
            str(dp_manifest),
            "--dp-checkpoint",
            str(dp_checkpoint),
            "--dp-state-key",
            args.dp_state_key,
            "--sample-name",
            sample_name,
            "--scenario",
            scenario,
            "--expected-video-frames",
            str(args.expected_video_frames),
            "--expected-action-steps",
            str(args.expected_action_steps),
            "--expected-action-dim",
            str(args.expected_action_dim),
            "--robot-action-dim",
            str(args.robot_action_dim),
            "--max-episode-steps",
            str(args.max_episode_steps),
            "--video-fps",
            str(args.video_fps),
            "--external-target-mode",
            str(spec.get("external_target_mode") or args.external_target_mode),
            "--min-dynamic-prefix-frame",
            str((source_panel or {}).get("min_dynamic_prefix_frame") or 8),
            "--target-motion-consecutive-frames",
            str((source_panel or {}).get("target_motion_consecutive_frames") or 2),
        ]
        if not bool(args.annotate_video):
            cmd.append("--no-annotate-video")
        (out_dir / "pure_dp_command.txt").write_text(" ".join(cmd) + "\n")
        with stdout_log.open("w") as log:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        summary_path = out_dir / "pure_dp_full_episode_summary.json"
        result: dict[str, Any] = {
            "sample_index": sample.get("sample_index"),
            "scenario": scenario,
            "source_h5": str(source_h5),
            "cosmos_summary_path": str(cosmos_summary_path) if cosmos_summary_path is not None else None,
            "sample_dir": str(out_dir),
            "stdout_log": str(stdout_log),
            "exit_code": proc.returncode,
            "ok": proc.returncode == 0 and summary_path.is_file(),
            "summary_path": str(summary_path) if summary_path.is_file() else None,
        }
        if summary_path.is_file():
            dp_summary = read_json(summary_path)
            result.update(
                {
                    "final_success": bool((dp_summary.get("final_eval") or {}).get("success", False)),
                    "final_eval": dp_summary.get("final_eval"),
                    "final_prefix_frame_index": dp_summary.get("final_prefix_frame_index"),
                    "final_observed_frames": dp_summary.get("final_observed_frames"),
                    "full_episode_length_ok": dp_summary.get("full_episode_length_ok"),
                    "controller_frame_counts": dp_summary.get("controller_frame_counts"),
                    "wm_active_frame_count": dp_summary.get("wm_active_frame_count"),
                    "dp_active_frame_count": dp_summary.get("dp_active_frame_count"),
                    "target_motion_detector": dp_summary.get("target_motion_detector"),
                    "final_observed_video": dp_summary.get("final_observed_video"),
                    "final_observed_annotated_video": dp_summary.get("final_observed_annotated_video"),
                    "final_observed_video_inspection": dp_summary.get("final_observed_video_inspection"),
                    "final_observed_annotated_video_inspection": dp_summary.get("final_observed_annotated_video_inspection"),
                    "video_file_contract_ok": dp_summary.get("video_file_contract_ok"),
                }
            )
            if isinstance(cosmos_summary, dict):
                result.update(
                    {
                        "cosmos_final_success": bool((cosmos_summary.get("final_eval") or {}).get("success", False)),
                        "cosmos_final_eval": cosmos_summary.get("final_eval"),
                        "cosmos_controller_frame_counts": cosmos_summary.get("controller_frame_counts"),
                        "cosmos_wm_active_frame_count": cosmos_summary.get("wm_active_frame_count"),
                    }
                )
        results.append(result)
        write_json(output_root / "pure_dp_panel_partial_summary.json", {"samples": results})
        if proc.returncode != 0:
            break

    completed = [row for row in results if row.get("ok")]
    contract_rows = panel_contract_rows(
        results,
        expected_frames=int(args.expected_video_frames),
        expected_actions=int(args.expected_action_steps),
        require_pure_dp=True,
    )
    sample_contract_failures_rows = [row for row in contract_rows if row["contract_failures"]]
    panel_contract_ok = panel_full_episode_contract_ok(
        contract_rows,
        expected_samples=len(specs),
    )
    summary = {
        "boundary": (
            "Pure-DP comparison panel for the same source H5 trajectories as a "
            "Cosmos live-loop panel or eval-style H5 manifest. The controller "
            "remains frozen DP for all 300 actions; target-motion detection is "
            "reporting-only."
        ),
        "source_panel_summary": str(Path(args.source_panel_summary).resolve()) if args.source_panel_summary else None,
        "eval_manifest": str(Path(args.eval_manifest).resolve()) if args.eval_manifest else None,
        "source_h5_root": str(Path(args.source_h5_root).resolve()) if args.source_h5_root else None,
        "output_root": str(output_root),
        "dp_manifest": str(dp_manifest),
        "dp_checkpoint": str(dp_checkpoint),
        "requested_samples": len(specs),
        "completed_samples": len(completed),
        "failed_process_count": sum(1 for row in results if row.get("exit_code") != 0),
        "pure_dp_final_success_count": sum(1 for row in completed if row.get("final_success")),
        "cosmos_final_success_count_for_completed": sum(1 for row in completed if row.get("cosmos_final_success")),
        "panel_full_episode_contract_ok": panel_contract_ok,
        "sample_contract_failures": sample_contract_failures_rows,
        "samples": results,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
    }
    write_json(output_root / "pure_dp_panel_summary.json", summary)
    print(json.dumps({"summary": str(output_root / "pure_dp_panel_summary.json")}, sort_keys=True))
    return 0 if len(completed) == len(specs) and panel_contract_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
