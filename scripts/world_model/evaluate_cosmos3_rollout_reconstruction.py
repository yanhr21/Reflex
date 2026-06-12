#!/usr/bin/env python3
"""Compare a generated Cosmos3 rollout video against RGB-D reference frames."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

try:
    from skimage.metrics import structural_similarity
except Exception:  # pragma: no cover - optional dependency
    structural_similarity = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-video", required=True)
    parser.add_argument("--prediction-video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-start", type=int, default=0)
    parser.add_argument("--prediction-start", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=33, help="Maximum frames to compare; <=0 means compare all available frames.")
    parser.add_argument("--sheet-frames", type=int, default=8)
    parser.add_argument(
        "--require-equal-length",
        action="store_true",
        default=True,
        help="Fail unless reference and prediction have the same available frame count after start offsets.",
    )
    parser.add_argument(
        "--allow-length-mismatch",
        dest="require_equal_length",
        action="store_false",
        help="Diagnostic-only override: compare overlapping frames even if reference/prediction lengths differ.",
    )
    parser.add_argument(
        "--require-no-truncation",
        action="store_true",
        default=True,
        help="Fail if --max-frames crops an otherwise longer comparison.",
    )
    parser.add_argument(
        "--allow-truncation",
        dest="require_no_truncation",
        action="store_false",
        help="Diagnostic-only override: allow --max-frames to crop a longer comparison.",
    )
    return parser.parse_args()


def video_frame_count(path: Path) -> int | None:
    try:
        reader = imageio.get_reader(path)
        try:
            count = reader.count_frames()
            if count is not None and count >= 0:
                return int(count)
        finally:
            reader.close()
    except Exception:
        return None
    return None


def read_video(path: Path, start: int, max_frames: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    reader = imageio.get_reader(path)
    try:
        for idx, frame in enumerate(reader):
            if idx < start:
                continue
            if max_frames > 0 and len(frames) >= max_frames:
                break
            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            frames.append(arr[..., :3].astype(np.uint8, copy=False))
    finally:
        reader.close()
    return frames


def resize_to(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.uint8)


def metric_row(reference: np.ndarray, prediction: np.ndarray, frame_index: int) -> dict[str, float | int]:
    ref = reference.astype(np.float32) / 255.0
    pred = prediction.astype(np.float32) / 255.0
    diff = pred - ref
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(mse))
    psnr = float("inf") if mse <= 0.0 else float(20.0 * math.log10(1.0 / rmse))
    row: dict[str, float | int] = {
        "frame": frame_index,
        "mse_rgb01": mse,
        "mae_rgb01": mae,
        "rmse_rgb01": rmse,
        "psnr_db": psnr,
    }
    if structural_similarity is not None:
        row["ssim"] = float(structural_similarity(ref, pred, channel_axis=-1, data_range=1.0))
    return row


def summarize(rows: list[dict[str, float | int]]) -> dict[str, float | int | None]:
    if not rows:
        return {"num_compared_frames": 0}
    keys = [key for key in rows[0] if key != "frame"]
    summary: dict[str, float | int | None] = {"num_compared_frames": len(rows)}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[f"mean_{key}"] = float(finite.mean()) if finite.size else None
        summary[f"median_{key}"] = float(np.median(finite)) if finite.size else None
        summary[f"max_{key}"] = float(finite.max()) if finite.size else None
    return summary


def sample_indices(count: int, desired: int) -> list[int]:
    if count <= 0 or desired <= 0:
        return []
    desired = min(count, desired)
    return [int(x) for x in np.unique(np.linspace(0, count - 1, desired).round().astype(np.int64))]


def draw_sheet(
    references: list[np.ndarray],
    predictions: list[np.ndarray],
    rows: list[dict[str, float | int]],
    output_path: Path,
    desired: int,
) -> None:
    indices = sample_indices(len(rows), desired)
    if not indices:
        return
    tile_w = 220
    label_h = 28
    rows_out: list[tuple[str, Image.Image, Image.Image, Image.Image]] = []
    for idx in indices:
        ref = references[idx]
        pred = predictions[idx]
        diff = np.abs(pred.astype(np.int16) - ref.astype(np.int16)).astype(np.uint8)
        ref_img = Image.fromarray(ref).resize((tile_w, max(1, int(round(ref.shape[0] * tile_w / ref.shape[1])))))
        pred_img = Image.fromarray(pred).resize(ref_img.size)
        diff_img = Image.fromarray(diff).resize(ref_img.size)
        label = (
            f"frame {rows[idx]['frame']}  mse={rows[idx]['mse_rgb01']:.5f} "
            f"psnr={rows[idx]['psnr_db']:.2f}"
        )
        rows_out.append((label, ref_img, pred_img, diff_img))

    tile_h = max(img.height for row in rows_out for img in row[1:])
    sheet = Image.new("RGB", (tile_w * 3, len(rows_out) * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_i, (label, ref_img, pred_img, diff_img) in enumerate(rows_out):
        y = row_i * (tile_h + label_h)
        sheet.paste(ref_img, (0, y))
        sheet.paste(pred_img, (tile_w, y))
        sheet.paste(diff_img, (tile_w * 2, y))
        draw.text((4, y + tile_h + 4), "ref | pred | absdiff  " + label, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def write_jsonl(rows: Iterable[dict[str, float | int]], path: Path) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_video = Path(args.reference_video)
    prediction_video = Path(args.prediction_video)

    reference_total_frames = video_frame_count(reference_video)
    prediction_total_frames = video_frame_count(prediction_video)
    reference_available_frames = (
        max(0, reference_total_frames - int(args.reference_start))
        if reference_total_frames is not None
        else None
    )
    prediction_available_frames = (
        max(0, prediction_total_frames - int(args.prediction_start))
        if prediction_total_frames is not None
        else None
    )
    if args.require_equal_length and (
        reference_available_frames is None
        or prediction_available_frames is None
        or reference_available_frames != prediction_available_frames
    ):
        raise RuntimeError(
            "Cosmos reconstruction length contract failed: "
            f"reference_available={reference_available_frames}, "
            f"prediction_available={prediction_available_frames}, "
            f"reference_total={reference_total_frames}, prediction_total={prediction_total_frames}."
        )
    if (
        args.require_no_truncation
        and args.max_frames > 0
        and reference_available_frames is not None
        and prediction_available_frames is not None
        and args.max_frames < min(reference_available_frames, prediction_available_frames)
    ):
        raise RuntimeError(
            "Cosmos reconstruction would be silently truncated: "
            f"max_frames={args.max_frames}, reference_available={reference_available_frames}, "
            f"prediction_available={prediction_available_frames}. Use --max-frames 0 for full-length comparison."
        )

    reference_frames = read_video(reference_video, args.reference_start, args.max_frames)
    prediction_frames = read_video(prediction_video, args.prediction_start, args.max_frames)
    if args.require_equal_length and len(reference_frames) != len(prediction_frames):
        raise RuntimeError(
            "Cosmos reconstruction loaded-frame length mismatch: "
            f"reference_loaded={len(reference_frames)}, prediction_loaded={len(prediction_frames)}, "
            f"reference_start={args.reference_start}, prediction_start={args.prediction_start}, "
            f"max_frames={args.max_frames}."
        )
    count = min(len(reference_frames), len(prediction_frames))
    if count <= 0:
        raise RuntimeError("No overlapping frames to compare")

    pred_size = (prediction_frames[0].shape[1], prediction_frames[0].shape[0])
    references = [resize_to(frame, pred_size) for frame in reference_frames[:count]]
    predictions = [frame[..., :3] for frame in prediction_frames[:count]]
    rows = [metric_row(ref, pred, idx) for idx, (ref, pred) in enumerate(zip(references, predictions))]
    summary = summarize(rows)
    report = {
        "reference_video": str(reference_video),
        "prediction_video": str(prediction_video),
        "reference_start": args.reference_start,
        "prediction_start": args.prediction_start,
        "max_frames": args.max_frames,
        "reference_total_frames": reference_total_frames,
        "prediction_total_frames": prediction_total_frames,
        "reference_available_frames": reference_available_frames,
        "prediction_available_frames": prediction_available_frames,
        "length_match_after_start": reference_available_frames == prediction_available_frames,
        "loaded_frame_length_match": len(reference_frames) == len(prediction_frames),
        "reference_loaded_frames": len(reference_frames),
        "prediction_loaded_frames": len(prediction_frames),
        "strict_length_required": bool(args.require_equal_length),
        "no_truncation_required": bool(args.require_no_truncation),
        "prediction_size": list(pred_size),
        "summary": summary,
        "boundary": (
            "Pixel reconstruction error is an alignment and plausibility diagnostic "
            "for a generated rollout. It is not by itself dynamic task-completion "
            "or controller evidence."
        ),
    }
    metrics_path = output_dir / "metrics.json"
    rows_path = output_dir / "frame_metrics.jsonl"
    sheet_path = output_dir / "reconstruction_comparison_sheet.png"
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_jsonl(rows, rows_path)
    draw_sheet(references, predictions, rows, sheet_path, args.sheet_frames)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
