#!/usr/bin/env python3
"""Build contact-state sheets from saved OpenPI pi0.5 replay labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--max-labels", type=int, default=0)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=760)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def find_label_paths(replay_root: Path) -> list[Path]:
    labels = sorted(replay_root.rglob("policy_droid_snapshot_action_replay_label.json"))
    labels.extend(sorted(replay_root.rglob("*_receding_label.json")))
    if labels:
        return labels
    list_path = replay_root / "replay_labels.list"
    if list_path.exists():
        out = [Path(line).resolve() for line in list_path.read_text().splitlines() if line.strip()]
        return [path for path in out if path.exists()]
    return []


def text(img: np.ndarray, value: str, xy: tuple[int, int], scale: float = 0.55, color: tuple[int, int, int] = (30, 30, 30)) -> None:
    cv2.putText(img, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_polyline(
    img: np.ndarray,
    values: list[float],
    *,
    x0: int,
    y0: int,
    w: int,
    h: int,
    lo: float,
    hi: float,
    color: tuple[int, int, int],
) -> None:
    if not values:
        return
    denom = max(1e-6, float(hi - lo))
    pts = []
    n = len(values)
    for i, value in enumerate(values):
        x = x0 + int(round((w - 1) * i / max(1, n - 1)))
        y = y0 + h - 1 - int(round((h - 1) * (float(value) - lo) / denom))
        pts.append((x, max(y0, min(y0 + h - 1, y))))
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 3, color, -1, cv2.LINE_AA)


def draw_axes(img: np.ndarray, x0: int, y0: int, w: int, h: int, lo: float, hi: float) -> None:
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (185, 185, 185), 1)
    for frac in (0.25, 0.5, 0.75):
        y = y0 + int(round(h * frac))
        cv2.line(img, (x0, y), (x0 + w, y), (230, 230, 230), 1)
    text(img, f"{hi:+.3f}", (x0 + 4, y0 + 18), 0.45, (100, 100, 100))
    text(img, f"{lo:+.3f}", (x0 + 4, y0 + h - 6), 0.45, (100, 100, 100))


def build_sheet(label_path: Path, out_path: Path, width: int, height: int) -> dict[str, Any]:
    label = read_json(label_path)
    records = label.get("step_records") or []
    rels = [np.asarray(r.get("peg_head_at_hole", [np.nan, np.nan, np.nan]), dtype=np.float32).reshape(-1)[:3] for r in records]
    xs = [float(r[0]) for r in rels]
    ys = [float(r[1]) for r in rels]
    zs = [float(r[2]) for r in rels]
    yz = [float(abs(y) + abs(z)) for y, z in zip(ys, zs)]
    inserted_steps = [i for i, r in enumerate(records) if bool(r.get("inserted", False))]
    grasped_steps = [i for i, r in enumerate(records) if bool(r.get("grasped", False))]
    success = bool(label.get("after_success", False))
    dp = label.get("dp_rollout_continuability") or {}

    img = np.full((int(height), int(width), 3), 255, dtype=np.uint8)
    title = label_path.parent.name
    text(img, title, (28, 38), 0.8, (0, 0, 0))
    text(
        img,
        f"direct success={int(success)} inserted={int(bool(label.get('after_inserted_live_pose', False)))} "
        f"grasp={int(bool(label.get('after_grasped', False)))} contact={int(bool(label.get('after_contact_stable_proxy', False)))} "
        f"DP96 success={int(bool(dp.get('success', False)))} continuable={int(bool(dp.get('continuable', False)))}",
        (28, 70),
        0.56,
        (30, 30, 30),
    )
    text(
        img,
        f"prefix={label.get('prefix_frame_index')} execute={label.get('execute_steps_actual')} "
        f"delta_abs_yz={float(label.get('delta_abs_yz_sum', 0.0)):+.5f}",
        (28, 98),
        0.52,
        (60, 60, 60),
    )

    plot_x, plot_y, plot_w, plot_h = 70, 145, int(width) - 120, 310
    vals = xs + ys + zs
    finite = [v for v in vals if np.isfinite(v)]
    lo = min(finite + [-0.12])
    hi = max(finite + [0.04])
    pad = max(0.02, 0.1 * (hi - lo))
    lo -= pad
    hi += pad
    draw_axes(img, plot_x, plot_y, plot_w, plot_h, lo, hi)
    draw_polyline(img, xs, x0=plot_x, y0=plot_y, w=plot_w, h=plot_h, lo=lo, hi=hi, color=(200, 50, 50))
    draw_polyline(img, ys, x0=plot_x, y0=plot_y, w=plot_w, h=plot_h, lo=lo, hi=hi, color=(40, 140, 40))
    draw_polyline(img, zs, x0=plot_x, y0=plot_y, w=plot_w, h=plot_h, lo=lo, hi=hi, color=(50, 80, 220))
    text(img, "x", (plot_x + plot_w - 120, plot_y + 24), 0.55, (200, 50, 50))
    text(img, "y", (plot_x + plot_w - 85, plot_y + 24), 0.55, (40, 140, 40))
    text(img, "z", (plot_x + plot_w - 50, plot_y + 24), 0.55, (50, 80, 220))
    text(img, "peg_head_at_hole over executed OpenPI chunk", (plot_x, plot_y - 12), 0.55, (40, 40, 40))

    bar_x, bar_y, bar_w, bar_h = 70, 515, int(width) - 120, 130
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (185, 185, 185), 1)
    n = max(1, len(records))
    for i, r in enumerate(records):
        x1 = bar_x + int(round(bar_w * i / n))
        x2 = bar_x + int(round(bar_w * (i + 1) / n))
        grasp = bool(r.get("grasped", False))
        inserted = bool(r.get("inserted", False))
        yz_i = yz[i] if i < len(yz) else np.nan
        color = (225, 245, 225) if grasp else (245, 225, 225)
        if inserted:
            color = (200, 230, 255)
        cv2.rectangle(img, (x1, bar_y + 1), (max(x1 + 1, x2), bar_y + bar_h - 1), color, -1)
        if np.isfinite(yz_i) and yz_i < 0.03:
            cv2.line(img, (x1, bar_y + bar_h - 18), (max(x1 + 1, x2), bar_y + bar_h - 18), (50, 120, 50), 3)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (185, 185, 185), 1)
    text(img, "green background=grasped; blue=inserted; bottom green mark=abs(y)+abs(z)<0.03", (bar_x, bar_y - 12), 0.5, (40, 40, 40))
    if inserted_steps:
        text(img, f"first inserted step: {inserted_steps[0]}", (bar_x, bar_y + bar_h + 28), 0.55, (0, 80, 120))
    else:
        text(img, "no inserted step during OpenPI chunk", (bar_x, bar_y + bar_h + 28), 0.55, (120, 40, 40))
    text(img, f"grasped steps: {len(grasped_steps)}/{len(records)}", (bar_x + 360, bar_y + bar_h + 28), 0.55, (40, 80, 40))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return {
        "label_path": str(label_path),
        "sheet_path": str(out_path),
        "steps": int(len(records)),
        "direct_success": success,
        "direct_inserted": bool(label.get("after_inserted_live_pose", False)),
        "direct_contact_stable": bool(label.get("after_contact_stable_proxy", False)),
        "direct_grasped": bool(label.get("after_grasped", False)),
        "dp96_success": bool(dp.get("success", False)),
        "dp96_continuable": bool(dp.get("continuable", False)),
        "inserted_steps": inserted_steps,
        "grasped_step_count": int(len(grasped_steps)),
        "final_peg_head_at_hole": label.get("after_peg_head_at_hole"),
        "delta_abs_yz_sum": label.get("delta_abs_yz_sum"),
    }


def main() -> int:
    args = parse_args()
    replay_root = Path(args.replay_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else replay_root / "contact_state_sheets"
    output_root.mkdir(parents=True, exist_ok=True)
    labels = find_label_paths(replay_root)
    if int(args.max_labels) > 0:
        labels = labels[: int(args.max_labels)]
    if not labels:
        raise RuntimeError(f"no replay labels found under {replay_root}")
    rows = []
    for label in labels:
        item_name = label.parent.name
        if label.name.endswith("_receding_label.json"):
            item_name = label.name[: -len("_receding_label.json")]
        out = output_root / item_name / "contact_state_sheet.png"
        rows.append(build_sheet(label, out, int(args.width), int(args.height)))
    manifest = {
        "schema": "openpi_pi05_contact_state_sheets_manifest_v1",
        "replay_root": str(replay_root),
        "output_root": str(output_root),
        "label_count": len(rows),
        "direct_success_count": sum(bool(r["direct_success"]) for r in rows),
        "direct_inserted_count": sum(bool(r["direct_inserted"]) for r in rows),
        "direct_grasped_count": sum(bool(r["direct_grasped"]) for r in rows),
        "dp96_success_count": sum(bool(r["dp96_success"]) for r in rows),
        "boundary": (
            "Contact-state sheets are rendered from authoritative replay label "
            "step records. They are visual diagnostics of state/contact metrics, "
            "not simulator RGB video and not a success claim."
        ),
        "rows": rows,
    }
    write_json(output_root / "contact_state_sheets_manifest.json", manifest)
    print(json.dumps({"sheets": len(rows), "manifest": str(output_root / "contact_state_sheets_manifest.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
