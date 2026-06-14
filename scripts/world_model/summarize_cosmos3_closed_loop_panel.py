#!/usr/bin/env python3
"""Summarize a Cosmos3 gated live-smoke panel.

This script reads per-sample outputs produced by
run_cosmos3_closed_loop_panel_in_allocation.sh. It does not run simulation or
claim controller success; it packages live simulator metrics and video frames
for human review.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-root", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-visual-review", required=True)
    parser.add_argument("--contact-sheet", required=True)
    parser.add_argument("--dp-resume-horizon", type=int, default=96)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def parse_status(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.is_file():
        return data
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def norm3(value: Any) -> float | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        vals = [float(value[i]) for i in range(3)]
    except Exception:
        return None
    return math.sqrt(sum(v * v for v in vals))


def int_prefix(value: Any) -> int | None:
    if value is None:
        return None
    token = str(value).strip().split(maxsplit=1)[0]
    try:
        return int(token)
    except Exception:
        return None


def sample_row(sample_dir: Path) -> dict[str, Any]:
    status = parse_status(sample_dir / "launch_status.txt")
    result = read_json(sample_dir / "live_smoke_result.json") or {}
    preview = read_json(sample_dir / "candidate_action_chunk_preview.json") or {}
    preflight = read_json(sample_dir / "closed_loop_preflight_manifest.json") or {}

    final_eval = result.get("final_eval") if isinstance(result.get("final_eval"), dict) else {}
    before_eval = result.get("before_eval") if isinstance(result.get("before_eval"), dict) else {}
    after_cosmos_eval = (
        result.get("after_cosmos_eval") if isinstance(result.get("after_cosmos_eval"), dict) else {}
    )
    source_context = preview.get("source_context") if isinstance(preview.get("source_context"), dict) else {}

    final_head = final_eval.get("peg_head_pos_at_hole")
    before_head = before_eval.get("peg_head_pos_at_hole")
    after_cosmos_head = after_cosmos_eval.get("peg_head_pos_at_hole")
    sample_index = (
        int_prefix(status.get("sample_index"))
        if int_prefix(status.get("sample_index")) is not None
        else int_prefix(status.get("sample_start"))
    )
    if sample_index is None:
        sample_index = int(sample_dir.name.rsplit("_", 1)[-1])
    exit_code = int_prefix(status.get("exit_code"))
    return {
        "sample_dir": str(sample_dir),
        "sample_index": sample_index,
        "exit_code": exit_code,
        "sample_name": preview.get("sample_name"),
        "scenario": preview.get("scenario") or source_context.get("scenario"),
        "prefix_role": preview.get("prefix_role") or source_context.get("prefix_role"),
        "prefix_frame_index": preview.get("prefix_frame_index") or source_context.get("prefix_frame_index"),
        "chunk_start": result.get("chunk_start") or preview.get("chunk_start"),
        "chunk_end_exclusive": result.get("chunk_end_exclusive") or preview.get("chunk_end_exclusive"),
        "cosmos_chunk_steps_executed": result.get("cosmos_chunk_steps_executed"),
        "dp_resume_steps_executed": result.get("dp_resume_steps_executed"),
        "before_success": before_eval.get("success"),
        "after_cosmos_success": after_cosmos_eval.get("success"),
        "final_success": final_eval.get("success"),
        "before_peg_head_hole_norm": norm3(before_head),
        "after_cosmos_peg_head_hole_norm": norm3(after_cosmos_head),
        "final_peg_head_hole_norm": norm3(final_head),
        "video_path": result.get("video_path"),
        "source_h5": result.get("source_h5") or source_context.get("source_h5"),
        "preflight_failures": preflight.get("failures"),
        "ok": bool(result.get("ok")) and exit_code == 0,
    }


def load_video_triplet(video_path: Path) -> list[Any] | None:
    try:
        import imageio.v2 as imageio
    except Exception:
        return None
    if not video_path.is_file():
        return None
    try:
        frames = imageio.mimread(video_path)
    except Exception:
        return None
    if not frames:
        return None
    return [frames[0], frames[len(frames) // 2], frames[-1]]


def write_contact_sheet(rows: list[dict[str, Any]], output_path: Path) -> bool:
    try:
        from PIL import Image, ImageDraw
        import numpy as np
    except Exception:
        return False

    triplets: list[tuple[dict[str, Any], list[Any]]] = []
    for row in rows:
        video = row.get("video_path")
        if not video:
            continue
        frames = load_video_triplet(Path(str(video)))
        if frames is not None:
            triplets.append((row, frames))

    if not triplets:
        return False

    cell_w, cell_h = 256, 256
    label_h = 52
    cols = 3
    sheet_w = cols * cell_w
    sheet_h = len(triplets) * (cell_h + label_h)
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    for row_idx, (row, frames) in enumerate(triplets):
        y = row_idx * (cell_h + label_h)
        label = (
            f"{row['sample_index']:02d} {row.get('scenario')} "
            f"{row.get('prefix_role')} success={row.get('final_success')} "
            f"norm={row.get('final_peg_head_hole_norm')}"
        )
        draw.text((4, y + 4), label[:120], fill=(0, 0, 0))
        for col, frame in enumerate(frames):
            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            if arr.ndim == 3 and arr.shape[-1] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                arr = np.clip(arr.astype("float32"), 0, 255).astype("uint8")
            image = Image.fromarray(arr).convert("RGB")
            image.thumbnail((cell_w, cell_h))
            x = col * cell_w + (cell_w - image.width) // 2
            sheet.paste(image, (x, y + label_h))
            draw.text((col * cell_w + 4, y + label_h + 4), ["start", "mid", "final"][col], fill=(255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return True


def main() -> int:
    args = parse_args()
    panel_root = Path(args.panel_root).resolve()
    sample_dirs = sorted(p for p in panel_root.glob("sample_*") if p.is_dir())
    rows = [sample_row(path) for path in sample_dirs]
    rows.sort(key=lambda row: int(row.get("sample_index", 0)))

    successes = [row for row in rows if row.get("final_success") is True]
    failures = [row for row in rows if row.get("final_success") is not True]
    summary = {
        "panel_root": str(panel_root),
        "num_samples": len(rows),
        "num_completed_ok": sum(1 for row in rows if row.get("ok")),
        "num_final_success": len(successes),
        "success_rate": (len(successes) / len(rows)) if rows else None,
        "dp_resume_horizon": args.dp_resume_horizon,
        "controller_interface": "one_shot_cosmos_chunk_then_optional_frozen_dp_resume",
        "method_evidence_allowed": False,
        "missing_method_requirements": [
            "no_online_cosmos_reprediction_after_live_reobservation",
            "no_external_dynamic_target_replay_after_prefix_reset",
            "no_continuability_guard_before_long_frozen_dp_resume",
            "no_future_task_state_reconstruction_fed_to_dp_resume_interface",
        ],
        "rows": rows,
        "boundary": (
            "Diagnostic panel summary from live simulator metrics and videos. "
            "It is not full receding-Cosmos controller evidence."
        ),
    }
    Path(args.output_summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    contact_written = write_contact_sheet(rows, Path(args.contact_sheet))
    visual_review = {
        "status": "needs_agent_review",
        "num_samples": len(rows),
        "num_final_success": len(successes),
        "success_sample_indices": [row["sample_index"] for row in successes],
        "failure_sample_indices": [row["sample_index"] for row in failures],
        "contact_sheet": str(Path(args.contact_sheet)) if contact_written else None,
        "note": (
            "Open the contact sheet or individual videos before making any "
            "dynamic manipulation claim. Metrics alone are not enough."
        ),
    }
    Path(args.output_visual_review).write_text(
        json.dumps(visual_review, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps({"summary": str(args.output_summary), "contact_sheet_written": contact_written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
