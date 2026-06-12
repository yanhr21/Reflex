#!/usr/bin/env python3
"""Inspect action-conditioned Cosmos3 reconstruction artifacts.

This is an artifact/readiness inspector. It does not score dynamic task
completion and must not be used as controller evidence by itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument(
        "--require-strict-contract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail when action prediction artifacts violate the sample manifest contract.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def video_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        return {"exists": True, "path": str(path), "load_error": f"missing imageio: {exc}"}
    reader = imageio.get_reader(path)
    try:
        meta = reader.get_meta_data()
        count = 0
        first_shape = None
        for frame in reader:
            if first_shape is None:
                first_shape = list(frame.shape)
            count += 1
    finally:
        reader.close()
    return {
        "exists": True,
        "path": str(path),
        "num_frames": count,
        "fps": float(meta.get("fps", 0.0) or 0.0),
        "first_frame_shape": first_shape,
    }


def safetensor_shapes(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    try:
        import safetensors.torch
    except Exception as exc:
        return {"exists": True, "path": str(path), "load_error": f"missing safetensors: {exc}"}
    try:
        tensors = safetensors.torch.load_file(str(path), device="cpu")
    except Exception as exc:
        return {"exists": True, "path": str(path), "load_error": str(exc)}
    return {
        "exists": True,
        "path": str(path),
        "keys": sorted(tensors),
        "shapes": {key: list(value.shape) for key, value in tensors.items()},
        "dtypes": {key: str(value.dtype) for key, value in tensors.items()},
    }


def find_prediction_video(eval_root: Path, sample_manifest: dict[str, Any]) -> Path | None:
    sample_name = sample_manifest.get("sample_name") or sample_manifest.get("name")
    if isinstance(sample_name, str):
        candidate = eval_root / "inference" / sample_name / "vision.mp4"
        if candidate.exists():
            return candidate
    matches = sorted((eval_root / "inference").glob("*/vision.mp4"))
    return matches[0] if matches else None


def write_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Cosmos3 Action Eval Artifact Inspection",
        "",
        f"- eval root: `{report['eval_root']}`",
        f"- completed marker: `{report['completed_marker_exists']}`",
        f"- prediction video: `{report['prediction_video'].get('path')}`",
        f"- video frames/fps: `{report['prediction_video'].get('num_frames')}` / `{report['prediction_video'].get('fps')}`",
        f"- pre-inference contract ok: "
        f"`{report['pre_inference_length_contract'].get('strict_pre_inference_contract_ok')}`",
        f"- length precheck predicted/expected: "
        f"`{report['length_contract_precheck'].get('predicted_frames')}` / "
        f"`{report['length_contract_precheck'].get('expected_frames')}`",
        f"- all-frame MAE/RMSE/PSNR: `{report['reconstruction_all_summary'].get('mean_mae_rgb01')}` / "
        f"`{report['reconstruction_all_summary'].get('mean_rmse_rgb01')}` / "
        f"`{report['reconstruction_all_summary'].get('mean_psnr_db')}`",
        f"- future MAE/RMSE/PSNR: `{report['reconstruction_future_summary'].get('mean_mae_rgb01')}` / "
        f"`{report['reconstruction_future_summary'].get('mean_rmse_rgb01')}` / "
        f"`{report['reconstruction_future_summary'].get('mean_psnr_db')}`",
        f"- debug tensor keys: `{report['output_safetensors'].get('keys')}`",
        f"- debug tensor shapes: `{report['output_safetensors'].get('shapes')}`",
        f"- strict contract failures: `{report.get('strict_contract_failures')}`",
        "",
        report["boundary"],
        "",
    ]
    path.write_text("\n".join(lines))


def _shape0(value: Any) -> int | None:
    if isinstance(value, list) and value:
        try:
            return int(value[0])
        except (TypeError, ValueError):
            return None
    return None


def _shape1(value: Any) -> int | None:
    if isinstance(value, list) and len(value) > 1:
        try:
            return int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def strict_contract_failures(
    sample_manifest: dict[str, Any],
    prediction_video: dict[str, Any],
    pre_inference_contract: dict[str, Any],
    framework_contract: dict[str, Any],
    length_contract: dict[str, Any],
    all_metrics: dict[str, Any],
    future_metrics: dict[str, Any],
    action_metrics: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    expected_frames = sample_manifest.get("reference_video_frames")
    expected_steps = sample_manifest.get("expected_full_action_chunk_size")
    try:
        expected_frames = int(expected_frames)
    except (TypeError, ValueError):
        expected_frames = None
    try:
        expected_steps = int(expected_steps)
    except (TypeError, ValueError):
        expected_steps = None

    if expected_frames is None:
        failures.append("sample_manifest_missing_reference_video_frames")
    elif prediction_video.get("num_frames") != expected_frames:
        failures.append(
            f"prediction_video_frame_count_mismatch:{prediction_video.get('num_frames')}!={expected_frames}"
        )

    if not pre_inference_contract:
        failures.append("missing_pre_inference_length_contract")
    else:
        if pre_inference_contract.get("strict_pre_inference_contract_ok") is not True:
            failures.append("pre_inference_length_contract_failed")
        if expected_frames is not None and pre_inference_contract.get("sample_num_frames") != expected_frames:
            failures.append(
                "pre_inference_sample_num_frames_mismatch:"
                f"{pre_inference_contract.get('sample_num_frames')}!={expected_frames}"
            )
        if expected_frames is not None and pre_inference_contract.get("reference_video_frames") != expected_frames:
            failures.append(
                "pre_inference_reference_frames_mismatch:"
                f"{pre_inference_contract.get('reference_video_frames')}!={expected_frames}"
            )
        if expected_steps is not None and pre_inference_contract.get("sample_action_chunk_size") != expected_steps:
            failures.append(
                "pre_inference_action_chunk_mismatch:"
                f"{pre_inference_contract.get('sample_action_chunk_size')}!={expected_steps}"
            )
        if expected_steps is not None:
            action_shape = pre_inference_contract.get("action_sidecar_shape") or []
            if _shape0(action_shape) != expected_steps:
                failures.append(f"pre_inference_action_sidecar_steps_mismatch:{_shape0(action_shape)}!={expected_steps}")
        if expected_frames is not None:
            state_shape = pre_inference_contract.get("state_target_shape") or []
            if state_shape and _shape0(state_shape) != expected_frames:
                failures.append(f"pre_inference_state_target_frames_mismatch:{_shape0(state_shape)}!={expected_frames}")

    if not framework_contract:
        failures.append("missing_framework_action_inference_contract")
    else:
        if framework_contract.get("strict_framework_action_contract_ok") is not True:
            failures.append("framework_action_inference_contract_failed")
        if expected_frames is not None and framework_contract.get("resolved_num_frames") != expected_frames:
            failures.append(
                f"framework_resolved_num_frames_mismatch:{framework_contract.get('resolved_num_frames')}!={expected_frames}"
            )
        if expected_frames is not None and framework_contract.get("batch_video_frames") != expected_frames:
            failures.append(
                f"framework_batch_video_frames_mismatch:{framework_contract.get('batch_video_frames')}!={expected_frames}"
            )
        if expected_steps is not None and framework_contract.get("resolved_action_chunk_size") != expected_steps:
            failures.append(
                "framework_resolved_action_chunk_size_mismatch:"
                f"{framework_contract.get('resolved_action_chunk_size')}!={expected_steps}"
            )
        if expected_steps is not None and framework_contract.get("batch_action_steps") != expected_steps:
            failures.append(
                f"framework_batch_action_steps_mismatch:{framework_contract.get('batch_action_steps')}!={expected_steps}"
            )
        seq_plan = framework_contract.get("sequence_plan") or {}
        expected_condition_indexes = sample_manifest.get("condition_frame_indexes_vision") or []
        if seq_plan.get("condition_frame_indexes_vision") != expected_condition_indexes:
            failures.append(
                "framework_sequence_plan_condition_frames_mismatch:"
                f"{seq_plan.get('condition_frame_indexes_vision')}!={expected_condition_indexes}"
            )
        expected_action_condition_indexes = sample_manifest.get("condition_frame_indexes_action") or []
        if seq_plan.get("condition_frame_indexes_action") != expected_action_condition_indexes:
            failures.append(
                "framework_sequence_plan_condition_action_frames_mismatch:"
                f"{seq_plan.get('condition_frame_indexes_action')}!={expected_action_condition_indexes}"
            )
        if (
            sample_manifest.get("sample_action_condition_future_zeroed") is True
            and pre_inference_contract.get("sample_action_condition_future_zero_ok") is not True
        ):
            failures.append("sample_action_condition_future_rows_not_zero")

    if not length_contract:
        failures.append("missing_length_contract_precheck")
    else:
        if length_contract.get("strict_full_length_ok") is not True:
            failures.append("length_contract_precheck_failed")
        if expected_frames is not None and length_contract.get("expected_frames") != expected_frames:
            failures.append(
                f"length_contract_expected_frames_mismatch:{length_contract.get('expected_frames')}!={expected_frames}"
            )
        if expected_frames is not None and length_contract.get("predicted_frames") != expected_frames:
            failures.append(
                f"length_contract_predicted_frames_mismatch:{length_contract.get('predicted_frames')}!={expected_frames}"
            )

    if all_metrics:
        if all_metrics.get("reference_total_frames") != expected_frames:
            failures.append(
                f"all_reference_total_frames_mismatch:{all_metrics.get('reference_total_frames')}!={expected_frames}"
            )
        if all_metrics.get("prediction_total_frames") != expected_frames:
            failures.append(
                f"all_prediction_total_frames_mismatch:{all_metrics.get('prediction_total_frames')}!={expected_frames}"
            )
        if all_metrics.get("length_match_after_start") is not True:
            failures.append("all_length_match_after_start_false")
        if all_metrics.get("no_truncation_required") is not True:
            failures.append("all_no_truncation_required_false")
        compared = (all_metrics.get("summary") or {}).get("num_compared_frames")
        if expected_frames is not None and compared != expected_frames:
            failures.append(f"all_num_compared_frames_mismatch:{compared}!={expected_frames}")
    else:
        failures.append("missing_reconstruction_all_metrics")

    if future_metrics and expected_frames is not None:
        future_start = future_metrics.get("reference_start")
        try:
            expected_future = expected_frames - int(future_start)
        except (TypeError, ValueError):
            expected_future = None
        compared = (future_metrics.get("summary") or {}).get("num_compared_frames")
        if future_metrics.get("length_match_after_start") is not True:
            failures.append("future_length_match_after_start_false")
        if future_metrics.get("no_truncation_required") is not True:
            failures.append("future_no_truncation_required_false")
        if expected_future is None:
            failures.append("future_reference_start_missing")
        elif compared != expected_future:
            failures.append(f"future_num_compared_frames_mismatch:{compared}!={expected_future}")
    else:
        failures.append("missing_reconstruction_future_metrics")

    model_mode = str(sample_manifest.get("model_mode") or "")
    action_required = bool(sample_manifest.get("action_is_prediction_target")) or model_mode in {
        "policy",
        "inverse_dynamics",
    }
    if action_required:
        if sample_manifest.get("action_is_prediction_target") is not True:
            failures.append("action_prediction_target_flag_missing_for_action_predicting_mode")
        if not action_metrics:
            failures.append("missing_action_prediction_metrics")
        else:
            pred_steps = _shape0(action_metrics.get("pred_shape"))
            target_steps = _shape0(action_metrics.get("target_shape"))
            target_dim = _shape1(action_metrics.get("target_shape"))
            compare_dim = action_metrics.get("compare_dim")
            if pred_steps != expected_steps:
                failures.append(f"pred_action_steps_mismatch:{pred_steps}!={expected_steps}")
            if target_steps != expected_steps:
                failures.append(f"target_action_steps_mismatch:{target_steps}!={expected_steps}")
            if target_dim is not None and compare_dim != target_dim:
                failures.append(f"action_compare_dim_mismatch:{compare_dim}!={target_dim}")
            if action_metrics.get("predicted_action_dim_covers_target") is not True:
                failures.append("predicted_action_dim_does_not_cover_target")
            if action_metrics.get("action_prediction_required") is not True:
                failures.append("action_prediction_required_false_for_policy_sample")
            if action_metrics.get("target_motion_onset_action_index") is not None:
                post_motion = action_metrics.get("post_target_motion") or {}
                if not post_motion:
                    failures.append("missing_post_target_motion_action_metrics")
                elif post_motion.get("steps", 0) <= 0:
                    failures.append("empty_post_target_motion_action_window")
    return failures


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    sample_manifest = read_json(eval_root / "inputs" / "sample_forward_action_prefix_manifest.json")
    prediction_video = find_prediction_video(eval_root, sample_manifest)
    prediction_dir = prediction_video.parent if prediction_video is not None else eval_root / "inference"

    all_metrics = read_json(eval_root / "reconstruction_all" / "metrics.json")
    future_metrics = read_json(eval_root / "reconstruction_future" / "metrics.json")
    action_metrics = read_json(eval_root / "action_prediction_metrics.json")
    pre_inference_contract = read_json(eval_root / "pre_inference_length_contract.json")
    framework_contract = read_json(eval_root / "framework_action_inference_contract.json")
    length_contract = read_json(eval_root / "length_contract_precheck.json")
    prediction_video_info = video_info(prediction_video) if prediction_video is not None else {"exists": False}
    failures = strict_contract_failures(
        sample_manifest=sample_manifest,
        prediction_video=prediction_video_info,
        pre_inference_contract=pre_inference_contract,
        framework_contract=framework_contract,
        length_contract=length_contract,
        all_metrics=all_metrics,
        future_metrics=future_metrics,
        action_metrics=action_metrics,
    )
    report = {
        "eval_root": str(eval_root),
        "completed_marker_exists": (eval_root / "post_sft_action_eval_completed").exists(),
        "sample_manifest": sample_manifest,
        "prediction_video": prediction_video_info,
        "sample_outputs_json_exists": (prediction_dir / "sample_outputs.json").exists(),
        "sample_args_json_exists": (prediction_dir / "sample_args.json").exists(),
        "pre_inference_length_contract": pre_inference_contract,
        "framework_action_inference_contract": framework_contract,
        "length_contract_precheck": length_contract,
        "action_prediction_metrics": action_metrics,
        "sample_data_safetensors": safetensor_shapes(prediction_dir / "sample_data.safetensors"),
        "output_safetensors": safetensor_shapes(prediction_dir / "output.safetensors"),
        "reconstruction_all_summary": all_metrics.get("summary", {}),
        "reconstruction_future_summary": future_metrics.get("summary", {}),
        "strict_contract_failures": failures,
        "strict_contract_ok": not failures,
        "sheets": {
            "all": str(eval_root / "reconstruction_all" / "reconstruction_comparison_sheet.png"),
            "future": str(eval_root / "reconstruction_future" / "reconstruction_comparison_sheet.png"),
        },
        "boundary": (
            "This inspection checks that the Cosmos3 WAM diagnostic "
            "produced valid equal-length videos, reconstruction metrics, and, "
            "for policy/inverse-dynamics modes, the required action prediction. "
            "It is not controller evidence or dynamic task-completion evidence."
        ),
    }
    output_json = Path(args.output_json) if args.output_json else eval_root / "artifact_inspection.json"
    output_md = Path(args.output_md) if args.output_md else eval_root / "artifact_inspection.md"
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_md(report, output_md)
    print(json.dumps(report, sort_keys=True))
    if args.require_strict_contract and failures:
        raise SystemExit("strict action-eval artifact contract failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
