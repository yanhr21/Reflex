#!/usr/bin/env python3
"""Preflight Cosmos3 action inference after framework argument resolution.

The wrapper-level full301 contract is not enough by itself: Cosmos action
inference builds its batch from ``action_chunk_size``. This check imports the
same Cosmos inference helpers, resolves the sample overrides, builds the action
batch on CPU, and verifies that the framework-facing video/action shapes still
match the intended full-length ManiSkill contract before GPU inference starts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-json", required=True)
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--expected-frames", type=int, default=301)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--expected-cond-indexes", default="0,1,2,3,4,5,6,7")
    return parser.parse_args()


def _shape(value: Any) -> list[int]:
    return [int(x) for x in getattr(value, "shape", [])]


def main() -> None:
    args = parse_args()
    sample_path = Path(args.sample_json)
    manifest_path = Path(args.sample_manifest)
    sample = json.loads(sample_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    expected_cond = [int(x) for x in args.expected_cond_indexes.split(",") if x != ""]

    from cosmos_framework.inference.action import get_action_sample_data
    from cosmos_framework.inference.args import ModelMode, OmniSampleOverrides

    model_fields = set(OmniSampleOverrides.model_fields)
    override_payload = {key: value for key, value in sample.items() if key in model_fields}
    override_payload.setdefault("output_dir", str(sample_path.parent / "_framework_preflight_output"))
    dummy_model_config = SimpleNamespace(
        action_gen=True,
        sound_gen=False,
        resolution=str(sample.get("resolution") or "256"),
        tokenizer=SimpleNamespace(temporal_compression_factor=4),
        vlm_config=SimpleNamespace(model_name="Qwen/Qwen3-VL-8B-Instruct"),
    )
    sample_args = OmniSampleOverrides.model_validate(override_payload).build_sample(
        model_config=dummy_model_config
    )

    model_mode = ModelMode(str(sample_args.model_mode))
    action_path = Path(str(sample_args.action_path)) if sample_args.action_path is not None else None
    batch = get_action_sample_data(
        SimpleNamespace(input_video_key="video"),
        batch_size=1,
        prompt=str(sample_args.prompt),
        vision_path=Path(str(sample_args.vision_path)),
        model_mode=model_mode,
        action_path=action_path,
        domain_name=str(sample_args.domain_name),
        view_point=str(sample_args.view_point),
        resolution=str(sample_args.image_size),
        action_chunk_size=int(sample_args.action_chunk_size),
        max_action_dim=int(sample.get("max_action_dim") or 64),
        fps=int(sample_args.fps),
        condition_frame_indexes_vision=list(sample_args.condition_frame_indexes_vision),
        condition_frame_indexes_action=list(sample_args.condition_frame_indexes_action),
        device="cpu",
    )

    video_tensor = batch["video"][0][0]
    action_tensor = batch["action"][0][0]
    sequence_plan = batch["sequence_plan"][0]
    resolved_num_frames = int(sample_args.num_frames)
    resolved_action_steps = int(sample_args.action_chunk_size)
    batch_video_frames = int(video_tensor.shape[1])
    batch_action_steps = int(action_tensor.shape[0])
    batch_action_dim = int(action_tensor.shape[1])
    plan_dict = sequence_plan.as_dict()

    failures: list[str] = []
    if int(manifest.get("reference_video_frames", -1)) != args.expected_frames:
        failures.append(
            f"manifest_reference_frames:{manifest.get('reference_video_frames')}!={args.expected_frames}"
        )
    if int(manifest.get("expected_full_action_chunk_size", -1)) != args.expected_action_steps:
        failures.append(
            "manifest_expected_action_steps:"
            f"{manifest.get('expected_full_action_chunk_size')}!={args.expected_action_steps}"
        )
    if resolved_num_frames != args.expected_frames:
        failures.append(f"resolved_num_frames:{resolved_num_frames}!={args.expected_frames}")
    if resolved_action_steps != args.expected_action_steps:
        failures.append(f"resolved_action_chunk_size:{resolved_action_steps}!={args.expected_action_steps}")
    if batch_video_frames != args.expected_frames:
        failures.append(f"action_batch_video_frames:{batch_video_frames}!={args.expected_frames}")
    if batch_action_steps != args.expected_action_steps:
        failures.append(f"action_batch_action_steps:{batch_action_steps}!={args.expected_action_steps}")
    if batch_action_dim < int(sample.get("raw_action_dim") or 32):
        failures.append(f"action_batch_dim:{batch_action_dim}<raw_action_dim:{sample.get('raw_action_dim')}")
    if list(sample_args.condition_frame_indexes_vision) != expected_cond:
        failures.append(
            "resolved_condition_frame_indexes_vision:"
            f"{list(sample_args.condition_frame_indexes_vision)}!={expected_cond}"
        )
    if list(plan_dict.get("condition_frame_indexes_vision") or []) != expected_cond:
        failures.append(
            "sequence_plan_condition_frame_indexes_vision:"
            f"{plan_dict.get('condition_frame_indexes_vision')}!={expected_cond}"
        )
    expected_action_cond = [int(x) for x in sample.get("condition_frame_indexes_action") or []]
    if list(sample_args.condition_frame_indexes_action) != expected_action_cond:
        failures.append(
            "resolved_condition_frame_indexes_action:"
            f"{list(sample_args.condition_frame_indexes_action)}!={expected_action_cond}"
        )
    if list(plan_dict.get("condition_frame_indexes_action") or []) != expected_action_cond:
        failures.append(
            "sequence_plan_condition_frame_indexes_action:"
            f"{plan_dict.get('condition_frame_indexes_action')}!={expected_action_cond}"
        )

    report = {
        "sample_json": str(sample_path),
        "sample_manifest": str(manifest_path),
        "model_mode": str(model_mode.value),
        "expected_frames": int(args.expected_frames),
        "expected_action_steps": int(args.expected_action_steps),
        "resolved_num_frames": resolved_num_frames,
        "resolved_action_chunk_size": resolved_action_steps,
        "batch_video_shape_cthw": _shape(video_tensor),
        "batch_action_shape": _shape(action_tensor),
        "batch_video_frames": batch_video_frames,
        "batch_action_steps": batch_action_steps,
        "sequence_plan": plan_dict,
        "condition_frame_indexes_vision": list(sample_args.condition_frame_indexes_vision),
        "condition_frame_indexes_action": list(sample_args.condition_frame_indexes_action),
        "strict_framework_action_contract_ok": not failures,
        "failures": failures,
        "boundary": (
            "CPU preflight through Cosmos action inference helpers. This catches "
            "framework defaulting or internal action/video length changes before "
            "the expensive GPU inference step."
        ),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if failures:
        raise SystemExit("strict framework action inference contract failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
