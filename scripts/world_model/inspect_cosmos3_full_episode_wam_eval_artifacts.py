#!/usr/bin/env python3
"""Inspect full-episode Cosmos3 WAM validation generations.

This checks generated video/action length contracts, action metrics, basic RGB
reconstruction diagnostics, and writes contact sheets for visual review.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--expected-video-frames", type=int, default=301)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--expected-action-dim", type=int, default=32)
    parser.add_argument("--robot-action-dim", type=int, default=7)
    parser.add_argument("--sample-frames", type=int, default=12)
    parser.add_argument("--thumb-width", type=int, default=256)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_video(path: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    frames: list[np.ndarray] = []
    reader = imageio.get_reader(path)
    try:
        meta = reader.get_meta_data()
        for frame in reader:
            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            if arr.shape[-1] > 3:
                arr = arr[..., :3]
            frames.append(arr.astype(np.uint8, copy=False))
    finally:
        reader.close()
    return frames, {"fps": float(meta.get("fps", 0.0) or 0.0), "num_frames": len(frames)}


def resize_frame(frame: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if frame.shape[0] == h and frame.shape[1] == w:
        return frame[..., :3]
    return np.asarray(Image.fromarray(frame[..., :3]).resize((w, h), Image.BILINEAR), dtype=np.uint8)


def metric_summary(pred: list[np.ndarray], ref: list[np.ndarray], start: int, end: int) -> dict[str, Any]:
    if not pred or not ref or start >= end:
        return {"available": False}
    end = min(end, len(pred), len(ref))
    if start >= end:
        return {"available": False}
    maes = []
    mses = []
    for i in range(start, end):
        p = pred[i][..., :3].astype(np.float32) / 255.0
        r = resize_frame(ref[i], pred[i].shape[:2]).astype(np.float32) / 255.0
        diff = p - r
        maes.append(float(np.mean(np.abs(diff))))
        mses.append(float(np.mean(diff * diff)))
    mse = float(np.mean(mses))
    psnr = float("inf") if mse == 0.0 else float(-10.0 * math.log10(mse))
    return {
        "available": True,
        "start_frame": int(start),
        "end_frame_exclusive": int(end),
        "num_frames": int(end - start),
        "mean_mae_rgb01": float(np.mean(maes)),
        "mean_rmse_rgb01": float(math.sqrt(mse)),
        "mean_psnr_db": psnr,
    }


def action_array_from_sample_output(path: Path) -> np.ndarray | None:
    data = read_json(path)
    outputs = data.get("outputs") or []
    if not outputs:
        return None
    content = outputs[0].get("content") or {}
    if "action" not in content:
        return None
    arr = np.asarray(content["action"], dtype=np.float32)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _rmse_mae_max(diff: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mae": float(np.mean(np.abs(diff))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(diff * diff))),
        f"{prefix}_max_abs": float(np.max(np.abs(diff))),
    }


def action_metrics(pred: np.ndarray | None, ref_path: Path, future_start: int, robot_action_dim: int) -> dict[str, Any]:
    if pred is None:
        return {"available": False, "failure": "missing_predicted_action"}
    ref = np.load(ref_path, allow_pickle=False)
    result: dict[str, Any] = {
        "available": True,
        "pred_shape": list(pred.shape),
        "ref_shape": list(ref.shape),
        "finite": bool(np.isfinite(pred).all()),
    }
    if pred.shape != ref.shape:
        result["shape_match"] = False
        return result
    result["shape_match"] = True
    diff = pred.astype(np.float64) - ref.astype(np.float64)
    result.update(_rmse_mae_max(diff, "all"))
    robot_action_dim = max(0, min(int(robot_action_dim), pred.shape[1]))
    result["robot_action_dim"] = int(robot_action_dim)
    result["state_sidecar_dim"] = int(pred.shape[1] - robot_action_dim)
    if robot_action_dim > 0:
        result.update(_rmse_mae_max(diff[:, :robot_action_dim], "robot_action_all"))
    if robot_action_dim < pred.shape[1]:
        result.update(_rmse_mae_max(diff[:, robot_action_dim:], "state_sidecar_all"))
    future_start = max(0, min(int(future_start), pred.shape[0]))
    result["prefix_end_action_index_exclusive"] = int(future_start)
    if future_start > 0:
        pdiff = diff[:future_start]
        result.update(_rmse_mae_max(pdiff, "prefix"))
        if robot_action_dim > 0:
            result.update(_rmse_mae_max(pdiff[:, :robot_action_dim], "robot_action_prefix"))
        if robot_action_dim < pred.shape[1]:
            result.update(_rmse_mae_max(pdiff[:, robot_action_dim:], "state_sidecar_prefix"))
    if future_start < pred.shape[0]:
        fdiff = diff[future_start:]
        result["future_start_action_index"] = int(future_start)
        result.update(_rmse_mae_max(fdiff, "future"))
        if robot_action_dim > 0:
            result.update(_rmse_mae_max(fdiff[:, :robot_action_dim], "robot_action_future"))
        if robot_action_dim < pred.shape[1]:
            result.update(_rmse_mae_max(fdiff[:, robot_action_dim:], "state_sidecar_future"))
    return result


def draw_contact_sheet(
    pred: list[np.ndarray],
    ref: list[np.ndarray],
    out_path: Path,
    *,
    sample_frames: int,
    thumb_width: int,
) -> None:
    count = min(len(pred), len(ref))
    if count == 0:
        return
    indices = np.linspace(0, count - 1, min(sample_frames, count)).round().astype(np.int64)
    indices = sorted(set(int(x) for x in indices))
    cells = []
    label_h = 22
    for idx in indices:
        pair = []
        for label, frame in (("ref", ref[idx]), ("pred", pred[idx])):
            image = Image.fromarray(resize_frame(frame, pred[idx].shape[:2])).convert("RGB")
            scale = thumb_width / float(image.width)
            image = image.resize((thumb_width, max(1, int(round(image.height * scale)))))
            pair.append((label, idx, image))
        cells.append(pair)
    cell_h = max(img.height for pair in cells for _label, _idx, img in pair) + label_h
    sheet = Image.new("RGB", (2 * thumb_width, len(cells) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for row_i, pair in enumerate(cells):
        for col_i, (label, idx, image) in enumerate(pair):
            x = col_i * thumb_width
            y = row_i * cell_h
            sheet.paste(image, (x, y))
            draw.text((x + 4, y + image.height + 4), f"{label} frame {idx}", fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def inspect_sample(
    eval_root: Path,
    sample: dict[str, Any],
    *,
    expected_video_frames: int,
    expected_action_steps: int,
    expected_action_dim: int,
    robot_action_dim: int,
    sample_frames: int,
    thumb_width: int,
) -> dict[str, Any]:
    name = sample["name"]
    sample_dir = eval_root / "inference" / name
    sample_output_path = sample_dir / "sample_outputs.json"
    pred_video_path = sample_dir / "vision.mp4"
    ref_video_path = Path(str(sample["reference_video_path"]))
    ref_action_path = Path(str(sample["reference_action_path"]))
    failures: list[str] = []

    if not sample_output_path.is_file():
        failures.append("missing_sample_outputs_json")
        sample_output = {}
    else:
        sample_output = read_json(sample_output_path)
        if sample_output.get("status", "success") != "success":
            failures.append(f"sample_status:{sample_output.get('status')}")

    if not pred_video_path.is_file():
        failures.append("missing_prediction_video")
        pred_frames: list[np.ndarray] = []
        pred_info = {"exists": False, "path": str(pred_video_path), "num_frames": 0}
    else:
        pred_frames, pred_info = load_video(pred_video_path)
        pred_info["exists"] = True
        pred_info["path"] = str(pred_video_path)
        if pred_info["num_frames"] != expected_video_frames:
            failures.append(f"pred_video_frames:{pred_info['num_frames']}!={expected_video_frames}")

    if not ref_video_path.is_file():
        failures.append("missing_reference_video")
        ref_frames: list[np.ndarray] = []
        ref_info = {"exists": False, "path": str(ref_video_path), "num_frames": 0}
    else:
        ref_frames, ref_info = load_video(ref_video_path)
        ref_info["exists"] = True
        ref_info["path"] = str(ref_video_path)
        if ref_info["num_frames"] != expected_video_frames:
            failures.append(f"ref_video_frames:{ref_info['num_frames']}!={expected_video_frames}")

    pred_action = action_array_from_sample_output(sample_output_path) if sample_output_path.is_file() else None
    # Action i advances the rollout from video frame i to i+1. If the
    # conditioning prefix ends at frame f, actions [0, f) are history and
    # action f is the first future action that must be generated.
    future_action_start = max(0, int(sample.get("prefix_frame_index") or 0))
    am = action_metrics(pred_action, ref_action_path, future_action_start, robot_action_dim)
    if am.get("pred_shape") != [expected_action_steps, expected_action_dim]:
        failures.append(f"pred_action_shape:{am.get('pred_shape')}!=[{expected_action_steps},{expected_action_dim}]")
    if am.get("finite") is not True:
        failures.append("pred_action_nonfinite_or_missing")
    if am.get("shape_match") is not True:
        failures.append("pred_ref_action_shape_mismatch")

    prefix_frame = int(sample.get("prefix_frame_index") or -1)
    video_all = metric_summary(pred_frames, ref_frames, 0, expected_video_frames)
    video_future = metric_summary(pred_frames, ref_frames, prefix_frame + 1, expected_video_frames)

    review_sheet = eval_root / "review_sheets" / f"{name}_ref_pred_sheet.png"
    if pred_frames and ref_frames:
        draw_contact_sheet(pred_frames, ref_frames, review_sheet, sample_frames=sample_frames, thumb_width=thumb_width)
    else:
        failures.append("missing_contact_sheet_inputs")

    return {
        "name": name,
        "sample_dir": str(sample_dir),
        "prefix_role": sample.get("prefix_role"),
        "scenario": sample.get("scenario"),
        "prefix_frame_index": sample.get("prefix_frame_index"),
        "reference_video_path": str(ref_video_path),
        "prediction_video": pred_info,
        "reference_video": ref_info,
        "sample_outputs_json": str(sample_output_path),
        "action_metrics": am,
        "video_all_metrics": video_all,
        "video_future_metrics": video_future,
        "review_sheet": str(review_sheet) if review_sheet.is_file() else None,
        "failures": failures,
        "strict_sample_ok": not failures,
    }


def write_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Cosmos3 Full-Episode WAM Eval Inspection",
        "",
        f"- eval root: `{report['eval_root']}`",
        f"- strict eval artifacts ok: `{report['strict_eval_artifacts_ok']}`",
        f"- samples inspected: `{report['num_samples']}`",
        f"- strict failures: `{report['strict_failures']}`",
        f"- mean action RMSE: `{report['aggregate'].get('mean_action_rmse')}`",
        f"- mean robot-action prefix RMSE: `{report['aggregate'].get('mean_robot_action_prefix_rmse')}`",
        f"- mean state-sidecar prefix RMSE: `{report['aggregate'].get('mean_state_sidecar_prefix_rmse')}`",
        f"- mean robot-action future RMSE: `{report['aggregate'].get('mean_robot_action_future_rmse')}`",
        f"- mean state-sidecar future RMSE: `{report['aggregate'].get('mean_state_sidecar_future_rmse')}`",
        f"- mean future video PSNR: `{report['aggregate'].get('mean_future_video_psnr_db')}`",
        "",
        report["boundary"],
        "",
        "| sample | role | scenario | video frames | action shape | action RMSE | robot future RMSE | state future RMSE | future PSNR | sheet | ok |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---|---:|",
    ]
    for item in report["samples"]:
        lines.append(
            "| `{name}` | `{role}` | `{scenario}` | {frames} | `{ashape}` | {armse} | {robot_rmse} | {state_rmse} | {psnr} | `{sheet}` | {ok} |".format(
                name=item["name"],
                role=item.get("prefix_role"),
                scenario=item.get("scenario"),
                frames=item["prediction_video"].get("num_frames"),
                ashape=item["action_metrics"].get("pred_shape"),
                armse=item["action_metrics"].get("all_rmse"),
                robot_rmse=item["action_metrics"].get("robot_action_future_rmse"),
                state_rmse=item["action_metrics"].get("state_sidecar_future_rmse"),
                psnr=item["video_future_metrics"].get("mean_psnr_db"),
                sheet=item.get("review_sheet"),
                ok=item.get("strict_sample_ok"),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    manifest = read_json(eval_root / "eval_input_manifest.json")
    samples = manifest.get("samples") or []
    inspected = [
        inspect_sample(
            eval_root,
            sample,
            expected_video_frames=args.expected_video_frames,
            expected_action_steps=args.expected_action_steps,
            expected_action_dim=args.expected_action_dim,
            robot_action_dim=args.robot_action_dim,
            sample_frames=args.sample_frames,
            thumb_width=args.thumb_width,
        )
        for sample in samples
    ]
    action_rmses = [
        float(item["action_metrics"]["all_rmse"])
        for item in inspected
        if item["action_metrics"].get("shape_match") is True and "all_rmse" in item["action_metrics"]
    ]
    robot_action_rmses = [
        float(item["action_metrics"]["robot_action_future_rmse"])
        for item in inspected
        if item["action_metrics"].get("shape_match") is True
        and "robot_action_future_rmse" in item["action_metrics"]
    ]
    robot_action_prefix_rmses = [
        float(item["action_metrics"]["robot_action_prefix_rmse"])
        for item in inspected
        if item["action_metrics"].get("shape_match") is True
        and "robot_action_prefix_rmse" in item["action_metrics"]
    ]
    state_sidecar_rmses = [
        float(item["action_metrics"]["state_sidecar_future_rmse"])
        for item in inspected
        if item["action_metrics"].get("shape_match") is True
        and "state_sidecar_future_rmse" in item["action_metrics"]
    ]
    state_sidecar_prefix_rmses = [
        float(item["action_metrics"]["state_sidecar_prefix_rmse"])
        for item in inspected
        if item["action_metrics"].get("shape_match") is True
        and "state_sidecar_prefix_rmse" in item["action_metrics"]
    ]
    future_psnrs = [
        float(item["video_future_metrics"]["mean_psnr_db"])
        for item in inspected
        if item["video_future_metrics"].get("available") is True
        and math.isfinite(float(item["video_future_metrics"]["mean_psnr_db"]))
    ]
    strict_failures = [f"{item['name']}:{failure}" for item in inspected for failure in item["failures"]]
    report = {
        "boundary": "Post-SFT generation artifact inspection. This is required training/eval evidence, but controller integration still needs readout metrics and visual review acceptance.",
        "eval_root": str(eval_root),
        "input_manifest": str(eval_root / "eval_input_manifest.json"),
        "num_samples": len(inspected),
        "strict_eval_artifacts_ok": not strict_failures,
        "strict_failures": strict_failures,
        "aggregate": {
            "mean_action_rmse": float(np.mean(action_rmses)) if action_rmses else None,
            "mean_robot_action_prefix_rmse": float(np.mean(robot_action_prefix_rmses))
            if robot_action_prefix_rmses
            else None,
            "mean_state_sidecar_prefix_rmse": float(np.mean(state_sidecar_prefix_rmses))
            if state_sidecar_prefix_rmses
            else None,
            "mean_robot_action_future_rmse": float(np.mean(robot_action_rmses)) if robot_action_rmses else None,
            "mean_state_sidecar_future_rmse": float(np.mean(state_sidecar_rmses)) if state_sidecar_rmses else None,
            "mean_future_video_psnr_db": float(np.mean(future_psnrs)) if future_psnrs else None,
        },
        "samples": inspected,
    }
    output_json = Path(args.output_json) if args.output_json else eval_root / "eval_artifact_inspection.json"
    output_md = Path(args.output_md) if args.output_md else eval_root / "eval_artifact_inspection.md"
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_md(report, output_md)
    print(json.dumps({"strict_eval_artifacts_ok": report["strict_eval_artifacts_ok"], "strict_failures": strict_failures}, sort_keys=True))
    if args.strict and strict_failures:
        raise SystemExit("strict eval artifact inspection failed")


if __name__ == "__main__":
    main()
