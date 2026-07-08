#!/usr/bin/env python3
"""First RGB task-state extractor for Phase 03 diagnostics.

The extractor operates only on RGB frames. It does not read simulator state,
does not execute control, and does not claim success. It produces conservative
bounding boxes, centers, confidence values, and overlays so Phase 03 can decide
whether Cosmos frames are usable as controller-facing evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def require_slurm_step() -> None:
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise SystemExit("refusing_login_node_execution=true; run inside a compute-node srun step")
    if os.environ.get("SLURM_STEP_ID") == "extern":
        raise SystemExit("refusing_extern_step=true; run inside an active compute-node srun step")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def components(mask: np.ndarray, min_area: int) -> list[dict[str, Any]]:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    try:
        import cv2  # type: ignore

        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=4)
        comps: list[dict[str, Any]] = []
        for label in range(1, int(num_labels)):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x0 = int(stats[label, cv2.CC_STAT_LEFT])
            y0 = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            x1 = x0 + width - 1
            y1 = y0 + height - 1
            comps.append(
                {
                    "area": area,
                    "bbox": [x0, y0, x1, y1],
                    "cx": float(centroids[label][0]),
                    "cy": float(centroids[label][1]),
                    "width": width,
                    "height": height,
                }
            )
        return comps
    except Exception:
        pass

    try:
        from scipy import ndimage  # type: ignore

        labels, num_labels = ndimage.label(mask_u8, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        slices = ndimage.find_objects(labels)
        comps = []
        for idx, slc in enumerate(slices, start=1):
            if slc is None:
                continue
            ys, xs = np.nonzero(labels[slc] == idx)
            area = int(ys.size)
            if area < min_area:
                continue
            y0 = int(slc[0].start + ys.min())
            y1 = int(slc[0].start + ys.max())
            x0 = int(slc[1].start + xs.min())
            x1 = int(slc[1].start + xs.max())
            comps.append(
                {
                    "area": area,
                    "bbox": [x0, y0, x1, y1],
                    "cx": float(slc[1].start + xs.mean()),
                    "cy": float(slc[0].start + ys.mean()),
                    "width": int(x1 - x0 + 1),
                    "height": int(y1 - y0 + 1),
                }
            )
        return comps
    except Exception:
        pass

    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[dict[str, Any]] = []
    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        points = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(points) < min_area:
            continue
        pts = np.asarray(points, dtype=np.int32)
        y0 = int(pts[:, 0].min())
        y1 = int(pts[:, 0].max())
        x0 = int(pts[:, 1].min())
        x1 = int(pts[:, 1].max())
        comps.append(
            {
                "area": int(len(points)),
                "bbox": [x0, y0, x1, y1],
                "cx": float((x0 + x1) / 2.0),
                "cy": float((y0 + y1) / 2.0),
                "width": int(x1 - x0 + 1),
                "height": int(y1 - y0 + 1),
            }
        )
    return comps


def choose_hole_component(rgb: np.ndarray) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    arr = rgb.astype(np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    block_mask = (r > 145) & (g > 105) & (b < 150) & ((r - b) > 45) & ((g - b) > 20)
    block_comps = components(block_mask, min_area=80)
    blocks = []
    for comp in block_comps:
        aspect = comp["width"] / max(1, comp["height"])
        if 0.35 <= aspect <= 3.5 and comp["area"] >= 250:
            blocks.append(comp)
    if not blocks:
        return None, block_comps

    block = max(blocks, key=lambda c: c["area"])
    x0, y0, x1, y1 = block["bbox"]
    margin = 4
    rx0 = max(0, x0 - margin)
    ry0 = max(0, y0 - margin)
    rx1 = min(rgb.shape[1] - 1, x1 + margin)
    ry1 = min(rgb.shape[0] - 1, y1 + margin)

    crop = arr[ry0 : ry1 + 1, rx0 : rx1 + 1]
    cr, cg, cb = crop[..., 0], crop[..., 1], crop[..., 2]
    brightness = cr + cg + cb
    # Hole aperture: darker region on/inside the yellow block. The rendered
    # aperture is often a shaded beige patch rather than pure black, so use
    # adaptive darkness relative to the target block instead of a hard black
    # threshold.
    block_crop = block_mask[ry0 : ry1 + 1, rx0 : rx1 + 1]
    block_brightness = brightness[block_crop]
    if block_brightness.size:
        median_block = float(np.median(block_brightness))
        dark_cutoff = median_block - 80.0
    else:
        dark_cutoff = 360.0
    dark_mask = (
        (brightness < dark_cutoff)
        & (cr < 205)
        & (cg < 190)
        & (cb < 165)
        & ((cr - cb) > 10)
    )
    aperture_comps = components(dark_mask, min_area=4)
    apertures = []
    for comp in aperture_comps:
        ax0, ay0, ax1, ay1 = comp["bbox"]
        comp = dict(comp)
        comp["bbox"] = [ax0 + rx0, ay0 + ry0, ax1 + rx0, ay1 + ry0]
        comp["cx"] = float(comp["cx"] + rx0)
        comp["cy"] = float(comp["cy"] + ry0)
        comp["kind"] = "hole_aperture"
        aspect = comp["width"] / max(1, comp["height"])
        if 5 <= comp["area"] <= 1200 and 0.18 <= aspect <= 5.0:
            # Prefer holes near the visual center of the block face, not table
            # shadows along the outside of the block.
            inside_x = (x0 + 2) <= comp["cx"] <= (x1 - 2)
            inside_y = (y0 + 2) <= comp["cy"] <= (y1 - 2)
            if inside_x and inside_y:
                # Reject long dark edges from block/table boundaries.
                edge_margin = min(comp["cx"] - x0, x1 - comp["cx"], comp["cy"] - y0, y1 - comp["cy"])
                comp["edge_margin"] = float(edge_margin)
                apertures.append(comp)
    if apertures:
        return max(apertures, key=lambda c: c["area"] + 8.0 * c.get("edge_margin", 0.0)), block_comps + apertures

    block = dict(block)
    block["kind"] = "hole_block_fallback"
    return block, block_comps


def expanded_bbox(comp: dict[str, Any] | None, margin: int, width: int, height: int) -> list[int] | None:
    if comp is None:
        return None
    x0, y0, x1, y1 = comp["bbox"]
    return [max(0, x0 - margin), max(0, y0 - margin), min(width - 1, x1 + margin), min(height - 1, y1 + margin)]


def point_in_bbox(x: float, y: float, bbox: list[int] | None) -> bool:
    if bbox is None:
        return False
    x0, y0, x1, y1 = bbox
    return x0 <= x <= x1 and y0 <= y <= y1


def bbox_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax0, ay0, ax1, ay1 = a["bbox"]
    bx0, by0, bx1, by1 = b["bbox"]
    dx = max(0, max(bx0 - ax1, ax0 - bx1))
    dy = max(0, max(by0 - ay1, ay0 - by1))
    return float((dx * dx + dy * dy) ** 0.5)


def merge_components(red: dict[str, Any], white: dict[str, Any]) -> dict[str, Any]:
    x0 = min(red["bbox"][0], white["bbox"][0])
    y0 = min(red["bbox"][1], white["bbox"][1])
    x1 = max(red["bbox"][2], white["bbox"][2])
    y1 = max(red["bbox"][3], white["bbox"][3])
    return {
        "area": int(red["area"] + white["area"]),
        "bbox": [x0, y0, x1, y1],
        "cx": float(red["cx"]),
        "cy": float(red["cy"]),
        "width": int(x1 - x0 + 1),
        "height": int(y1 - y0 + 1),
        "kind": "peg_red_white_pair",
        "red_bbox": red["bbox"],
        "white_bbox": white["bbox"],
        "pair_distance": bbox_distance(red, white),
        "red_cx": red["cx"],
        "red_cy": red["cy"],
        "white_cx": white["cx"],
        "white_cy": white["cy"],
    }


def choose_peg_component(
    rgb: np.ndarray,
    hole: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    arr = rgb.astype(np.int16)
    height, width = rgb.shape[:2]
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    red_mask = (r > 130) & ((r - g) > 35) & ((r - b) > 35) & (g < 150) & (b < 150)
    red_comps = components(red_mask, min_area=8)
    red_candidates = []
    hole_exclusion = expanded_bbox(hole, margin=16, width=width, height=height)
    for comp in red_comps:
        area = comp["area"]
        aspect = comp["width"] / max(1, comp["height"])
        if point_in_bbox(comp["cx"], comp["cy"], hole_exclusion):
            continue
        if hole is not None and bbox_distance(comp, hole) < 10.0:
            continue
        # Most valid peg evidence is on the table / lower half of the rendered
        # frame. This rejects small red robot-joint highlights.
        if comp["cy"] < 0.34 * height:
            continue
        if 8 <= area <= 1600 and 0.15 <= aspect <= 6.0:
            comp = dict(comp)
            comp["kind"] = "red_endpoint"
            red_candidates.append(comp)

    white_mask = (
        (r > 145)
        & (g > 145)
        & (b > 135)
        & ((np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])) < 75)
    )
    white_comps = components(white_mask, min_area=16)
    white_candidates = []
    for comp in white_comps:
        area = comp["area"]
        aspect = comp["width"] / max(1, comp["height"])
        if area > 2200:
            continue
        if comp["cy"] < 0.30 * height:
            continue
        if 0.12 <= aspect <= 8.0 and 16 <= area <= 2200:
            comp = dict(comp)
            comp["kind"] = "white_shaft_candidate"
            white_candidates.append(comp)

    pair_candidates = []
    for red in red_candidates:
        for white in white_candidates:
            dist = bbox_distance(red, white)
            center_dist = float(((red["cx"] - white["cx"]) ** 2 + (red["cy"] - white["cy"]) ** 2) ** 0.5)
            if dist > 18 or center_dist > 75:
                continue
            merged = merge_components(red, white)
            if hole is not None and bbox_distance(merged, hole) < 8.0:
                continue
            merged_aspect = merged["width"] / max(1, merged["height"])
            if not (0.18 <= merged_aspect <= 8.0):
                continue
            compact = merged["area"] / max(1.0, float(merged["width"] * merged["height"]))
            hole_above_penalty = 0.0
            hole_distance_penalty = 0.0
            if hole is not None:
                # Robot wrist and gripper highlights often form red/white
                # pairs above the target face. A table-supported peg can be
                # near the hole, but a candidate well above the hole is risky
                # controller evidence and should lose to lower/table evidence.
                hole_above_penalty = max(0.0, float(hole["cy"] - merged["cy"]))
                hole_distance_penalty = min(120.0, float(((merged["cx"] - hole["cx"]) ** 2 + (merged["cy"] - hole["cy"]) ** 2) ** 0.5))
            horizontal_score = max(0.0, min(1.0, abs(red["cx"] - white["cx"]) / max(1.0, center_dist)))
            table_score = merged["cy"] / max(1.0, float(height))
            # Prefer compact red/white peg-like pairs in the lower/table half,
            # with a horizontal red-white relation. Penalize pairs sitting
            # above the hole because those are usually robot/body artifacts in
            # the current camera view.
            merged["pair_score"] = (
                120.0 * compact
                - 2.0 * dist
                - 0.4 * abs(red["cy"] - white["cy"])
                + 80.0 * horizontal_score
                + 120.0 * table_score
                - 2.2 * hole_above_penalty
                - 0.15 * hole_distance_penalty
            )
            pair_candidates.append(merged)
    debug_comps = red_comps + white_comps + red_candidates + white_candidates + pair_candidates
    if pair_candidates:
        return max(pair_candidates, key=lambda c: c["pair_score"]), debug_comps

    # Fallback: keep a red endpoint only as low-confidence evidence.
    if not red_candidates:
        return None, debug_comps

    def score(comp: dict[str, Any]) -> float:
        compact = comp["area"] / max(1.0, float(comp["width"] * comp["height"]))
        return 0.7 * comp["cy"] + 80.0 * compact - 0.02 * comp["area"]

    fallback = max(red_candidates, key=score)
    fallback = dict(fallback)
    fallback["kind"] = "red_endpoint_fallback"
    return fallback, debug_comps


def confidence_for_hole(comp: dict[str, Any] | None, all_comps: list[dict[str, Any]]) -> float:
    if comp is None:
        return 0.0
    if comp.get("kind") == "hole_aperture":
        area_score = min(1.0, comp["area"] / 120.0)
        aspect = comp["width"] / max(1, comp["height"])
        aspect_score = max(0.0, 1.0 - abs(np.log(max(aspect, 1e-6))) / 2.5)
        ambiguity = 1.0 / max(1.0, len([c for c in all_comps if c.get("kind") == "hole_aperture"]))
        return float(max(0.0, min(1.0, 0.55 * area_score + 0.25 * aspect_score + 0.20 * ambiguity)))
    area_score = min(1.0, comp["area"] / 2500.0)
    ambiguity = 1.0 / max(1.0, len([c for c in all_comps if c["area"] > 250]))
    # A block center is only a fallback for visual debugging. It is not a
    # reliable insertion target, so cap confidence below the pass threshold.
    return float(max(0.0, min(0.40, 0.30 * area_score + 0.10 * ambiguity)))


def confidence_for_peg(comp: dict[str, Any] | None, all_comps: list[dict[str, Any]]) -> float:
    if comp is None:
        return 0.0
    if comp.get("kind") == "peg_red_white_pair":
        dist_score = max(0.0, 1.0 - comp.get("pair_distance", 99.0) / 18.0)
        area_score = min(1.0, comp["area"] / 500.0)
        aspect = comp["width"] / max(1, comp["height"])
        aspect_score = max(0.0, 1.0 - abs(np.log(max(aspect, 1e-6))) / 2.8)
        return float(max(0.0, min(1.0, 0.35 * dist_score + 0.35 * area_score + 0.30 * aspect_score)))
    area = comp["area"]
    area_score = min(1.0, area / 250.0)
    if area > 900:
        area_score *= 0.45
    fragments = len([c for c in all_comps if c["area"] > 25])
    ambiguity = 1.0 / max(1.0, fragments)
    # A red endpoint without a matched white shaft is not enough for control.
    return float(max(0.0, min(0.20, 0.30 * area_score + 0.10 * ambiguity)))


def sample_name_from_frame(path: Path) -> str:
    marker = "_vision_frame_"
    return path.name.split(marker, 1)[0] if marker in path.name else path.stem


def frame_index_from_name(path: Path) -> int:
    marker = "_frame_"
    if marker not in path.stem:
        return -1
    return int(path.stem.rsplit(marker, 1)[1])


def draw_box(draw: ImageDraw.ImageDraw, comp: dict[str, Any] | None, color: tuple[int, int, int], label: str) -> None:
    if comp is None:
        return
    x0, y0, x1, y1 = comp["bbox"]
    draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
    draw.rectangle((x0, max(0, y0 - 12), min(x1 + 80, x0 + 7 * len(label)), y0), fill=(0, 0, 0))
    draw.text((x0 + 2, max(0, y0 - 11)), label, fill=color)


def normalize_2d(x: float, y: float) -> tuple[float, float]:
    norm = float((x * x + y * y) ** 0.5)
    if norm < 1e-6:
        return 0.0, 0.0
    return float(x / norm), float(y / norm)


def peg_axis_2d(comp: dict[str, Any] | None) -> tuple[float | str, float | str]:
    if comp is None:
        return "", ""
    if comp.get("kind") == "peg_red_white_pair":
        return normalize_2d(float(comp["white_cx"] - comp["red_cx"]), float(comp["white_cy"] - comp["red_cy"]))
    if comp["width"] >= comp["height"]:
        return 1.0, 0.0
    return 0.0, 1.0


def hole_axis_2d(comp: dict[str, Any] | None) -> tuple[float | str, float | str]:
    if comp is None:
        return "", ""
    # RGB-only diagnostic: the aperture/block major axis is only an image-plane
    # proxy, not a calibrated 3D insertion axis.
    if comp["width"] >= comp["height"]:
        return 1.0, 0.0
    return 0.0, 1.0


def axis_angle_error_deg(
    peg_axis: tuple[float | str, float | str],
    hole_axis: tuple[float | str, float | str],
) -> float | str:
    if "" in peg_axis or "" in hole_axis:
        return ""
    dot = abs(float(peg_axis[0]) * float(hole_axis[0]) + float(peg_axis[1]) * float(hole_axis[1]))
    dot = max(-1.0, min(1.0, dot))
    return float(np.degrees(np.arccos(dot)))


def project_error_onto_hole_axis(
    peg_to_hole_x: float | str,
    peg_to_hole_y: float | str,
    hole_axis: tuple[float | str, float | str],
) -> tuple[float | str, float | str]:
    if peg_to_hole_x == "" or peg_to_hole_y == "" or "" in hole_axis:
        return "", ""
    vx = float(peg_to_hole_x)
    vy = float(peg_to_hole_y)
    hx = float(hole_axis[0])
    hy = float(hole_axis[1])
    axial = vx * hx + vy * hy
    lateral = vx * (-hy) + vy * hx
    return float(lateral), float(axial)


def draw_axis(
    draw: ImageDraw.ImageDraw,
    comp: dict[str, Any] | None,
    axis: tuple[float | str, float | str],
    color: tuple[int, int, int],
) -> None:
    if comp is None or "" in axis:
        return
    cx = float(comp["cx"])
    cy = float(comp["cy"])
    scale = 0.5 * max(float(comp["width"]), float(comp["height"]), 20.0)
    ax = float(axis[0]) * scale
    ay = float(axis[1]) * scale
    draw.line((cx - ax, cy - ay, cx + ax, cy + ay), fill=color, width=2)


def peg_artifact_risk(comp: dict[str, Any] | None, hole: dict[str, Any] | None, height: int) -> float:
    if comp is None:
        return 1.0
    risk = 0.0
    if comp.get("kind") == "red_endpoint_fallback":
        risk += 0.35
    if comp.get("kind") == "peg_red_white_pair" and comp.get("pair_distance", 99.0) > 14.0:
        risk += 0.20
    if comp["cy"] < 0.45 * height:
        risk += 0.25
    if hole is not None:
        if comp["cy"] < float(hole["cy"]) - 12.0:
            risk += 0.30
        if bbox_distance(comp, hole) < 6.0:
            risk += 0.30
    return float(max(0.0, min(1.0, risk)))


def peg_candidate_sort_key(comp: dict[str, Any], hole: dict[str, Any] | None, height: int) -> float:
    if comp.get("kind") == "peg_red_white_pair":
        score = float(comp.get("pair_score", 0.0))
    elif comp.get("kind") == "red_endpoint":
        compact = comp["area"] / max(1.0, float(comp["width"] * comp["height"]))
        score = 20.0 + 80.0 * compact + 0.3 * comp["cy"]
    else:
        return -1e9
    return score - 120.0 * peg_artifact_risk(comp, hole, height)


def top_peg_candidates(
    comps: list[dict[str, Any]],
    hole: dict[str, Any] | None,
    height: int,
    limit: int = 8,
) -> list[dict[str, Any]]:
    candidates = [c for c in comps if c.get("kind") in {"peg_red_white_pair", "red_endpoint"}]
    ranked = sorted(candidates, key=lambda c: peg_candidate_sort_key(c, hole, height), reverse=True)[:limit]
    out = []
    for rank, comp in enumerate(ranked, start=1):
        item = {
            "rank": rank,
            "kind": comp.get("kind", ""),
            "bbox": comp["bbox"],
            "cx": comp["cx"],
            "cy": comp["cy"],
            "area": comp["area"],
            "width": comp["width"],
            "height": comp["height"],
            "artifact_risk": peg_artifact_risk(comp, hole, height),
            "sort_score": peg_candidate_sort_key(comp, hole, height),
        }
        if "red_cx" in comp:
            item["red_cx"] = comp["red_cx"]
            item["red_cy"] = comp["red_cy"]
            item["white_cx"] = comp["white_cx"]
            item["white_cy"] = comp["white_cy"]
        if "pair_score" in comp:
            item["pair_score"] = comp["pair_score"]
        if "pair_distance" in comp:
            item["pair_distance"] = comp["pair_distance"]
        out.append(item)
    return out


def candidate_confidence(candidate: dict[str, Any]) -> float:
    if candidate.get("kind") == "peg_red_white_pair":
        dist_score = max(0.0, 1.0 - float(candidate.get("pair_distance", 99.0)) / 18.0)
        area_score = min(1.0, float(candidate["area"]) / 500.0)
        aspect = float(candidate["width"]) / max(1.0, float(candidate["height"]))
        aspect_score = max(0.0, 1.0 - abs(np.log(max(aspect, 1e-6))) / 2.8)
        return float(max(0.0, min(1.0, 0.35 * dist_score + 0.35 * area_score + 0.30 * aspect_score)))
    area_score = min(1.0, float(candidate["area"]) / 250.0)
    if float(candidate["area"]) > 900.0:
        area_score *= 0.45
    return float(max(0.0, min(0.20, 0.35 * area_score)))


def axis_from_candidate(candidate: dict[str, Any]) -> tuple[float, float]:
    if all(k in candidate for k in ("red_cx", "red_cy", "white_cx", "white_cy")):
        return normalize_2d(
            float(candidate["white_cx"]) - float(candidate["red_cx"]),
            float(candidate["white_cy"]) - float(candidate["red_cy"]),
        )
    if float(candidate["width"]) >= float(candidate["height"]):
        return 1.0, 0.0
    return 0.0, 1.0


def draw_candidate_overlay(
    image: Image.Image,
    candidates: list[dict[str, Any]],
    selected: dict[str, Any] | None,
    out_path: Path,
) -> None:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((4, 4, 250, 24), fill=(0, 0, 0))
    draw.text((8, 8), "Phase03 peg candidates", fill=(255, 255, 255))
    for candidate in candidates:
        x0, y0, x1, y1 = candidate["bbox"]
        color = (80, 220, 255)
        if selected is not None and candidate["bbox"] == selected.get("bbox"):
            color = (255, 40, 40)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
        label = f"#{candidate['rank']} r={candidate['artifact_risk']:.2f}"
        draw.rectangle((x0, max(0, y0 - 12), min(overlay.width - 1, x0 + 7 * len(label)), y0), fill=(0, 0, 0))
        draw.text((x0 + 2, max(0, y0 - 11)), label, fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)


def extract_one(frame_path: Path, overlay_dir: Path, candidate_overlay_dir: Path) -> dict[str, Any]:
    image = Image.open(frame_path).convert("RGB")
    rgb = np.asarray(image)
    hole, hole_comps = choose_hole_component(rgb)
    peg, peg_comps = choose_peg_component(rgb, hole=hole)
    candidates = top_peg_candidates(peg_comps, hole, rgb.shape[0])
    hole_conf = confidence_for_hole(hole, hole_comps)
    peg_conf = confidence_for_peg(peg, peg_comps)
    confidence = min(hole_conf, peg_conf)
    artifact_risk = peg_artifact_risk(peg, hole, rgb.shape[0])

    peg_to_hole_x: float | str = ""
    peg_to_hole_y: float | str = ""
    distance_px: float | str = ""
    if hole is not None and peg is not None:
        peg_to_hole_x = float(hole["cx"] - peg["cx"])
        peg_to_hole_y = float(hole["cy"] - peg["cy"])
        distance_px = float((peg_to_hole_x**2 + peg_to_hole_y**2) ** 0.5)

    peg_axis = peg_axis_2d(peg)
    hole_axis = hole_axis_2d(hole)
    angle_error = axis_angle_error_deg(peg_axis, hole_axis)
    lateral_error, axial_error = project_error_onto_hole_axis(peg_to_hole_x, peg_to_hole_y, hole_axis)

    status = "rgb_extracted"
    if confidence < 0.45:
        status = "rgb_extracted_low_confidence"
    if hole is None or peg is None:
        status = "rgb_extraction_failed"
    if status == "rgb_extracted" and artifact_risk >= 0.55:
        status = "rgb_extracted_low_confidence"
        confidence = min(confidence, 0.44)

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((4, 4, 224, 24), fill=(0, 0, 0))
    draw.text((8, 8), f"RGB extractor conf={confidence:.2f}", fill=(255, 255, 255))
    draw_box(draw, hole, (255, 230, 0), "hole")
    draw_box(draw, peg, (255, 40, 40), "peg")
    draw_axis(draw, hole, hole_axis, (255, 255, 80))
    draw_axis(draw, peg, peg_axis, (255, 120, 120))
    if peg is not None and peg.get("kind") == "peg_red_white_pair":
        draw.rectangle(peg["red_bbox"], outline=(255, 20, 20), width=1)
        draw.rectangle(peg["white_bbox"], outline=(230, 230, 255), width=1)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    out_path = overlay_dir / frame_path.name
    overlay.save(out_path)
    candidate_overlay_path = candidate_overlay_dir / frame_path.name
    draw_candidate_overlay(image, candidates, peg, candidate_overlay_path)
    selected_rank: int | str = ""
    if peg is not None:
        for candidate in candidates:
            if candidate["bbox"] == peg.get("bbox"):
                selected_rank = candidate["rank"]
                break

    return {
        "sample": sample_name_from_frame(frame_path),
        "frame": frame_index_from_name(frame_path),
        "source": "rgb_extracted",
        "frame_path": str(frame_path),
        "overlay_path": str(out_path),
        "candidate_overlay_path": str(candidate_overlay_path),
        "_peg_candidates": candidates,
        "hole_found": hole is not None,
        "hole_cx": "" if hole is None else hole["cx"],
        "hole_cy": "" if hole is None else hole["cy"],
        "hole_bbox": "" if hole is None else json.dumps(hole["bbox"]),
        "hole_kind": "" if hole is None else hole.get("kind", ""),
        "hole_confidence": hole_conf,
        "peg_found": peg is not None,
        "peg_cx": "" if peg is None else peg["cx"],
        "peg_cy": "" if peg is None else peg["cy"],
        "peg_bbox": "" if peg is None else json.dumps(peg["bbox"]),
        "peg_kind": "" if peg is None else peg.get("kind", ""),
        "peg_confidence": peg_conf,
        "peg_artifact_risk": artifact_risk,
        "peg_candidate_count": len(candidates),
        "peg_selected_candidate_rank": selected_rank,
        "peg_to_hole_px_x": peg_to_hole_x,
        "peg_to_hole_px_y": peg_to_hole_y,
        "peg_to_hole_distance_px": distance_px,
        "peg_axis_2d_x": peg_axis[0],
        "peg_axis_2d_y": peg_axis[1],
        "hole_axis_2d_x": hole_axis[0],
        "hole_axis_2d_y": hole_axis[1],
        "axis_angle_error_deg": angle_error,
        "lateral_error_px": lateral_error,
        "axial_error_px": axial_error,
        "near_hole_rgb": bool(distance_px != "" and float(distance_px) < 45.0 and confidence >= 0.45),
        "preinsert_rgb": bool(distance_px != "" and float(distance_px) < 25.0 and confidence >= 0.60),
        "extractor_confidence": confidence,
        "extraction_status": status,
        "notes": "RGB-only extractor; no simulator state and no success claim.",
    }


def write_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample", "frame", "source", "frame_path", "overlay_path", "candidate_overlay_path",
        "hole_found", "hole_cx", "hole_cy", "hole_bbox", "hole_kind", "hole_confidence",
        "peg_found", "peg_cx", "peg_cy", "peg_bbox", "peg_kind", "peg_confidence",
        "peg_artifact_risk", "peg_candidate_count", "peg_selected_candidate_rank",
        "peg_to_hole_px_x", "peg_to_hole_px_y", "peg_to_hole_distance_px",
        "peg_axis_2d_x", "peg_axis_2d_y", "hole_axis_2d_x", "hole_axis_2d_y",
        "axis_angle_error_deg", "lateral_error_px", "axial_error_px",
        "track_selected_rgb", "track_selection_changed", "track_gate_reason",
        "sequence_consistent_rgb", "sequence_gate_reason",
        "near_hole_rgb", "preinsert_rgb", "extractor_confidence",
        "extraction_status", "notes",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_candidate_debug(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            payload = {
                "sample": row["sample"],
                "frame": row["frame"],
                "frame_path": row["frame_path"],
                "candidate_overlay_path": row["candidate_overlay_path"],
                "selected_candidate_rank": row["peg_selected_candidate_rank"],
                "selected_peg_bbox": row["peg_bbox"],
                "selected_peg_kind": row["peg_kind"],
                "selected_peg_artifact_risk": row["peg_artifact_risk"],
                "track_selected_rgb": row.get("track_selected_rgb", ""),
                "track_selection_changed": row.get("track_selection_changed", ""),
                "track_gate_reason": row.get("track_gate_reason", ""),
                "candidates": row.get("_peg_candidates", []),
            }
            f.write(json.dumps(payload, sort_keys=True) + "\n")


def component_from_row(row: dict[str, Any], prefix: str) -> dict[str, Any] | None:
    bbox_value = row.get(f"{prefix}_bbox", "")
    if not bbox_value:
        return None
    bbox = json.loads(str(bbox_value))
    return {
        "bbox": bbox,
        "cx": float(row[f"{prefix}_cx"]),
        "cy": float(row[f"{prefix}_cy"]),
        "width": int(bbox[2] - bbox[0] + 1),
        "height": int(bbox[3] - bbox[1] + 1),
        "kind": row.get(f"{prefix}_kind", ""),
    }


def redraw_final_overlays(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        image = Image.open(row["frame_path"]).convert("RGB")
        hole = component_from_row(row, "hole")
        peg = component_from_row(row, "peg")
        peg_axis = (
            row["peg_axis_2d_x"],
            row["peg_axis_2d_y"],
        )
        hole_axis = (
            row["hole_axis_2d_x"],
            row["hole_axis_2d_y"],
        )

        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((4, 4, 300, 24), fill=(0, 0, 0))
        draw.text(
            (8, 8),
            f"RGB extractor final conf={float(row['extractor_confidence']):.2f}",
            fill=(255, 255, 255),
        )
        draw_box(draw, hole, (255, 230, 0), "hole")
        draw_box(draw, peg, (255, 40, 40), "peg")
        draw_axis(draw, hole, hole_axis, (255, 255, 80))
        draw_axis(draw, peg, peg_axis, (255, 120, 120))
        overlay.save(row["overlay_path"])

        selected = None
        if row.get("peg_bbox"):
            selected_bbox = json.loads(str(row["peg_bbox"]))
            for candidate in row.get("_peg_candidates", []):
                if candidate["bbox"] == selected_bbox:
                    selected = candidate
                    break
        draw_candidate_overlay(image, row.get("_peg_candidates", []), selected, Path(row["candidate_overlay_path"]))


def row_hole_axis(row: dict[str, Any]) -> tuple[float | str, float | str]:
    if row.get("hole_axis_2d_x", "") == "" or row.get("hole_axis_2d_y", "") == "":
        return "", ""
    return float(row["hole_axis_2d_x"]), float(row["hole_axis_2d_y"])


def apply_candidate_to_row(row: dict[str, Any], candidate: dict[str, Any] | None) -> None:
    if candidate is None:
        row["peg_found"] = False
        row["peg_cx"] = ""
        row["peg_cy"] = ""
        row["peg_bbox"] = ""
        row["peg_kind"] = ""
        row["peg_confidence"] = 0.0
        row["peg_artifact_risk"] = 1.0
        row["peg_selected_candidate_rank"] = ""
        row["peg_to_hole_px_x"] = ""
        row["peg_to_hole_px_y"] = ""
        row["peg_to_hole_distance_px"] = ""
        row["peg_axis_2d_x"] = ""
        row["peg_axis_2d_y"] = ""
        row["axis_angle_error_deg"] = ""
        row["lateral_error_px"] = ""
        row["axial_error_px"] = ""
        row["near_hole_rgb"] = False
        row["preinsert_rgb"] = False
        row["extractor_confidence"] = 0.0
        row["extraction_status"] = "rgb_extraction_failed"
        return

    row["peg_found"] = True
    row["peg_cx"] = float(candidate["cx"])
    row["peg_cy"] = float(candidate["cy"])
    row["peg_bbox"] = json.dumps(candidate["bbox"])
    row["peg_kind"] = candidate.get("kind", "")
    row["peg_confidence"] = candidate_confidence(candidate)
    row["peg_artifact_risk"] = float(candidate["artifact_risk"])
    row["peg_selected_candidate_rank"] = candidate["rank"]
    peg_axis = axis_from_candidate(candidate)
    row["peg_axis_2d_x"] = peg_axis[0]
    row["peg_axis_2d_y"] = peg_axis[1]

    peg_to_hole_x: float | str = ""
    peg_to_hole_y: float | str = ""
    distance_px: float | str = ""
    if row.get("hole_found"):
        peg_to_hole_x = float(row["hole_cx"]) - float(candidate["cx"])
        peg_to_hole_y = float(row["hole_cy"]) - float(candidate["cy"])
        distance_px = float((peg_to_hole_x**2 + peg_to_hole_y**2) ** 0.5)
    row["peg_to_hole_px_x"] = peg_to_hole_x
    row["peg_to_hole_px_y"] = peg_to_hole_y
    row["peg_to_hole_distance_px"] = distance_px

    hole_axis = row_hole_axis(row)
    row["axis_angle_error_deg"] = axis_angle_error_deg(peg_axis, hole_axis)
    lateral_error, axial_error = project_error_onto_hole_axis(peg_to_hole_x, peg_to_hole_y, hole_axis)
    row["lateral_error_px"] = lateral_error
    row["axial_error_px"] = axial_error

    confidence = min(float(row["hole_confidence"]), float(row["peg_confidence"]))
    if float(candidate["artifact_risk"]) >= 0.55:
        confidence = min(confidence, 0.44)
    row["extractor_confidence"] = confidence
    row["near_hole_rgb"] = bool(distance_px != "" and float(distance_px) < 45.0 and confidence >= 0.45)
    row["preinsert_rgb"] = bool(distance_px != "" and float(distance_px) < 25.0 and confidence >= 0.60)
    if not row.get("hole_found"):
        row["extraction_status"] = "rgb_extraction_failed"
    elif confidence < 0.45:
        row["extraction_status"] = "rgb_extracted_low_confidence"
    else:
        row["extraction_status"] = "rgb_extracted"


def apply_track_selection(rows: list[dict[str, Any]]) -> None:
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_sample.setdefault(str(row["sample"]), []).append(row)

    for sample_rows in by_sample.values():
        sample_rows.sort(key=lambda r: int(r["frame"]))
        candidates_by_row: list[list[dict[str, Any] | None]] = []
        for row in sample_rows:
            candidates = list(row.get("_peg_candidates", []))
            candidates_by_row.append([None] + candidates[:6])

        dp: list[list[float]] = []
        back: list[list[int]] = []
        for row_idx, candidates in enumerate(candidates_by_row):
            dp_row = []
            back_row = []
            for cand_idx, candidate in enumerate(candidates):
                if candidate is None:
                    unary = 110.0
                else:
                    unary = (
                        70.0 * float(candidate["artifact_risk"])
                        - 0.04 * float(candidate["sort_score"])
                        + 12.0 * max(0, int(candidate["rank"]) - 1)
                    )
                if row_idx == 0:
                    dp_row.append(unary)
                    back_row.append(-1)
                    continue
                best_cost = float("inf")
                best_prev = 0
                for prev_idx, prev_candidate in enumerate(candidates_by_row[row_idx - 1]):
                    transition = 0.0
                    if candidate is None or prev_candidate is None:
                        transition = 35.0
                    else:
                        jump = float(
                            (
                                (float(candidate["cx"]) - float(prev_candidate["cx"])) ** 2
                                + (float(candidate["cy"]) - float(prev_candidate["cy"])) ** 2
                            )
                            ** 0.5
                        )
                        transition = min(90.0, 0.9 * jump)
                    cost = dp[row_idx - 1][prev_idx] + unary + transition
                    if cost < best_cost:
                        best_cost = cost
                        best_prev = prev_idx
                dp_row.append(best_cost)
                back_row.append(best_prev)
            dp.append(dp_row)
            back.append(back_row)

        if not dp:
            continue
        idx = min(range(len(dp[-1])), key=lambda i: dp[-1][i])
        selected_indices = [idx]
        for row_idx in range(len(sample_rows) - 1, 0, -1):
            idx = back[row_idx][idx]
            selected_indices.append(idx)
        selected_indices.reverse()

        changed = False
        for row, candidates, selected_idx in zip(sample_rows, candidates_by_row, selected_indices):
            selected = candidates[selected_idx]
            old_bbox = row.get("peg_bbox", "")
            apply_candidate_to_row(row, selected)
            row["track_selected_rgb"] = selected is not None
            row["track_selection_changed"] = old_bbox != row.get("peg_bbox", "")
            changed = changed or bool(row["track_selection_changed"])
        for row in sample_rows:
            row["track_gate_reason"] = "sequence_track_selection_changed" if changed else ""


def apply_sequence_gate(rows: list[dict[str, Any]]) -> None:
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_sample.setdefault(str(row["sample"]), []).append(row)

    for sample_rows in by_sample.values():
        sample_rows.sort(key=lambda r: int(r["frame"]))
        found = [r for r in sample_rows if r["peg_found"] and r["hole_found"]]
        total = len(sample_rows)
        found_ratio = len(found) / max(1, total)
        jumps = []
        prev = None
        for row in found:
            cur = (float(row["peg_cx"]), float(row["peg_cy"]))
            if prev is not None:
                jumps.append(float(((cur[0] - prev[0]) ** 2 + (cur[1] - prev[1]) ** 2) ** 0.5))
            prev = cur
        max_jump = max(jumps) if jumps else 0.0

        reasons = []
        if found_ratio < 0.60:
            reasons.append(f"peg_or_hole_found_ratio_low:{found_ratio:.2f}")
        if max_jump > 90.0:
            reasons.append(f"peg_track_jump_px:{max_jump:.1f}")

        for row in sample_rows:
            row["sequence_consistent_rgb"] = not reasons
            row["sequence_gate_reason"] = ";".join(reasons)
            if reasons and row["extraction_status"] == "rgb_extracted":
                row["extraction_status"] = "rgb_extracted_low_confidence"
                row["extractor_confidence"] = min(float(row["extractor_confidence"]), 0.44)
                row["near_hole_rgb"] = False
                row["preinsert_rgb"] = False
                row["notes"] = str(row["notes"]) + " Sequence gate downgraded this frame."


def write_report(path: Path, rows: list[dict[str, Any]], run_id: str) -> None:
    total = len(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["extraction_status"]] = counts.get(row["extraction_status"], 0) + 1
    reliable = counts.get("rgb_extracted", 0) == total and total > 0
    lines = [
        "# Phase 03 RGB Task-State Extractor",
        "",
        f"Run: `{run_id}`",
        "",
        "Status: RGB extractor diagnostic complete.",
        "",
        "Counts:",
        "",
        f"- frames: `{total}`",
        f"- rgb_extracted: `{counts.get('rgb_extracted', 0)}`",
        f"- rgb_extracted_low_confidence: `{counts.get('rgb_extracted_low_confidence', 0)}`",
        f"- rgb_extraction_failed: `{counts.get('rgb_extraction_failed', 0)}`",
        "",
        "Protocol:",
        "",
        "- source is RGB frames only;",
        "- no simulator state is read;",
        "- no controller action is executed;",
        "- no Oracle is used;",
        "- no physical insertion success is claimed.",
        "- `peg_candidate_debug.jsonl` and `peg_candidate_overlays/` are diagnostic artifacts only.",
        "",
        "Ruling:",
        "",
        (
            "- extractor passed this narrow frame-level diagnostic; bridge scoring still needs live validation."
            if reliable
            else "- extractor is not reliable enough to unlock insertion candidates or Oracle."
        ),
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-frame-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    require_slurm_step()

    frame_dir = Path(args.input_frame_dir)
    out_dir = Path(args.output_dir)
    overlay_dir = out_dir / "rgb_extractor_overlays"
    candidate_overlay_dir = out_dir / "peg_candidate_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(frame_dir.glob("*.png"))
    rows = [extract_one(frame, overlay_dir, candidate_overlay_dir) for frame in frames]
    apply_track_selection(rows)
    apply_sequence_gate(rows)
    redraw_final_overlays(rows)

    manifest = {
        "phase": "03_integration",
        "run_id": args.run_id,
        "output_dir": str(out_dir),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "node": os.uname().nodename,
        "input_frame_dir": str(frame_dir),
        "evidence_type": "rgb_task_state_extractor_diagnostic_no_controller_execution",
        "source": "rgb_extracted",
        "simulator_state_read": False,
        "controller_execution_used": False,
        "oracle_used": False,
        "method_evidence_allowed": False,
        "forbidden_state_intervention_used": False,
        "candidate_debug": str(out_dir / "peg_candidate_debug.jsonl"),
        "candidate_overlays": str(candidate_overlay_dir),
    }
    write_chart(out_dir / "rgb_task_state_chart.csv", rows)
    write_candidate_debug(out_dir / "peg_candidate_debug.jsonl", rows)
    write_json(out_dir / "manifest.json", manifest)
    write_report(out_dir / "rgb_extractor_report.md", rows, args.run_id)

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["extraction_status"]] = status_counts.get(row["extraction_status"], 0) + 1
    print(
        json.dumps(
            {
                "phase03_status": "rgb_task_state_extractor_complete",
                "run_id": args.run_id,
                "frames": len(rows),
                "status_counts": status_counts,
                "output_dir": str(out_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
