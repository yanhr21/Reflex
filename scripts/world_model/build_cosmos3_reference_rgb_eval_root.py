#!/usr/bin/env python3
"""Build a reference-RGB readout calibration root from a Cosmos3 eval panel.

The output root has the same eval_input_manifest contract as the source eval
root, but each sample's inference/vision.mp4 points to the ground-truth
reference video. This lets the same task-state readout/profile pipeline
separate readout-calibration error from Cosmos3-generated-video error.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-eval-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def replace_symlink(link_path: Path, target_path: Path, overwrite: bool) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite existing path without --overwrite: {link_path}")
        if link_path.is_dir() and not link_path.is_symlink():
            raise IsADirectoryError(f"refusing to replace directory: {link_path}")
        link_path.unlink()
    os.symlink(target_path, link_path)


def main() -> None:
    args = parse_args()
    source_eval_root = Path(args.source_eval_root)
    output_root = Path(args.output_root)
    source_manifest_path = source_eval_root / "eval_input_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"missing source eval manifest: {source_manifest_path}")
    source_manifest = read_json(source_manifest_path)
    samples = list(source_manifest.get("samples") or [])
    if not samples:
        raise ValueError(f"source eval manifest has no samples: {source_manifest_path}")

    failures: list[str] = []
    linked_samples: list[dict[str, Any]] = []
    for sample in samples:
        name = str(sample.get("name") or "")
        if not name:
            failures.append("sample_missing_name")
            continue
        reference_video = Path(str(sample.get("reference_video_path") or ""))
        if not reference_video.is_file():
            failures.append(f"{name}:missing_reference_video:{reference_video}")
            continue
        link_path = output_root / "inference" / name / "vision.mp4"
        replace_symlink(link_path, reference_video, bool(args.overwrite))
        row = dict(sample)
        row["reference_rgb_calibration_video_path"] = str(reference_video)
        row["generated_video_path"] = str(link_path)
        linked_samples.append(row)

    manifest = dict(source_manifest)
    manifest["boundary"] = (
        "Reference-RGB readout calibration root. The videos are ground-truth "
        "reference RGB symlinks, not Cosmos3 generated videos. Use only to "
        "calibrate task-state readout/onset diagnostics."
    )
    manifest["source_eval_root"] = str(source_eval_root)
    manifest["output_root"] = str(output_root)
    manifest["samples"] = linked_samples
    manifest["num_selected_samples"] = len(linked_samples)
    manifest["reference_rgb_calibration"] = True
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "eval_input_manifest.json").write_text(
        json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n"
    )
    report = {
        "boundary": manifest["boundary"],
        "source_eval_root": str(source_eval_root),
        "output_root": str(output_root),
        "num_source_samples": len(samples),
        "num_linked_samples": len(linked_samples),
        "failures": failures,
    }
    (output_root / "reference_rgb_calibration_manifest.json").write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))
    if failures:
        raise SystemExit("reference-RGB calibration root build failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
