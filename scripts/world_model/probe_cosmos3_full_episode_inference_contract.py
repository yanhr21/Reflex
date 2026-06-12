#!/usr/bin/env python3
"""Probe Cosmos3 full-episode WAM inference contract.

This is a lightweight compute-node sanity check. It does not load a model or
run generation; it only sends saved sample_args through the same Cosmos action
inference helper used by evaluation and verifies the framework-facing video,
action, and conditioning masks still match the active 301/300 contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-json", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--expected-frames", type=int, default=301)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--expected-raw-action-dim", type=int, default=32)
    parser.add_argument("--temporal-compression-factor", type=int, default=4)
    return parser.parse_args()


def shape(value: Any) -> list[int]:
    return [int(x) for x in getattr(value, "shape", [])]


def expected_vision_indexes(prefix_frames: int, temporal_compression_factor: int) -> list[int]:
    latent_prefix = 1 + (int(prefix_frames) - 1) // int(temporal_compression_factor)
    return list(range(max(1, latent_prefix)))


def expected_action_indexes(prefix_frame_index: int) -> list[int]:
    return list(range(max(0, int(prefix_frame_index))))


def probe_one(
    sample_path: Path,
    *,
    expected_frames: int,
    expected_action_steps: int,
    expected_raw_action_dim: int,
    temporal_compression_factor: int,
) -> dict[str, Any]:
    sample = json.loads(sample_path.read_text())

    from cosmos_framework.inference.action import get_action_sample_data
    from cosmos_framework.inference.args import ModelMode, OmniSampleOverrides

    model_fields = set(OmniSampleOverrides.model_fields)
    override_payload = {key: value for key, value in sample.items() if key in model_fields}
    override_payload.setdefault("output_dir", str(sample_path.parent / "_contract_probe_output"))
    dummy_model_config = SimpleNamespace(
        action_gen=True,
        sound_gen=False,
        resolution=str(sample.get("resolution") or sample.get("image_size") or "256"),
        tokenizer=SimpleNamespace(temporal_compression_factor=temporal_compression_factor),
        vlm_config=SimpleNamespace(model_name="Qwen/Qwen3-VL-8B-Instruct"),
    )
    sample_args = OmniSampleOverrides.model_validate(override_payload).build_sample(
        model_config=dummy_model_config
    )
    model_mode = ModelMode(str(sample_args.model_mode))

    batch = get_action_sample_data(
        SimpleNamespace(input_video_key="video"),
        batch_size=1,
        prompt=str(sample_args.prompt),
        vision_path=Path(str(sample_args.vision_path)),
        model_mode=model_mode,
        action_path=Path(str(sample_args.action_path)) if sample_args.action_path is not None else None,
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
    plan = batch["sequence_plan"][0].as_dict()
    extra = sample.get("extra") if isinstance(sample.get("extra"), dict) else {}
    prefix_frame = int(extra.get("prefix_frame_index", -1))
    prefix_frames = int(extra.get("condition_prefix_frames", prefix_frame + 1))
    expected_vision = expected_vision_indexes(prefix_frames, temporal_compression_factor)
    expected_action = expected_action_indexes(prefix_frame)

    failures: list[str] = []
    if int(sample_args.num_frames) != expected_frames:
        failures.append(f"resolved_num_frames:{sample_args.num_frames}!={expected_frames}")
    if int(sample_args.action_chunk_size) != expected_action_steps:
        failures.append(f"resolved_action_steps:{sample_args.action_chunk_size}!={expected_action_steps}")
    if int(video_tensor.shape[1]) != expected_frames:
        failures.append(f"batch_video_frames:{int(video_tensor.shape[1])}!={expected_frames}")
    if int(action_tensor.shape[0]) != expected_action_steps:
        failures.append(f"batch_action_steps:{int(action_tensor.shape[0])}!={expected_action_steps}")
    if int(sample_args.raw_action_dim or -1) != expected_raw_action_dim:
        failures.append(f"resolved_raw_action_dim:{sample_args.raw_action_dim}!={expected_raw_action_dim}")
    if int(action_tensor.shape[1]) < expected_raw_action_dim:
        failures.append(f"batch_action_dim:{int(action_tensor.shape[1])}<{expected_raw_action_dim}")
    if list(sample_args.condition_frame_indexes_vision) != expected_vision:
        failures.append(
            "resolved_vision_condition_indexes:"
            f"{list(sample_args.condition_frame_indexes_vision)}!={expected_vision}"
        )
    if list(plan.get("condition_frame_indexes_vision") or []) != expected_vision:
        failures.append(
            "plan_vision_condition_indexes:"
            f"{plan.get('condition_frame_indexes_vision')}!={expected_vision}"
        )
    if list(sample_args.condition_frame_indexes_action) != expected_action:
        failures.append(
            "resolved_action_condition_indexes:"
            f"{list(sample_args.condition_frame_indexes_action)}!={expected_action}"
        )
    if list(plan.get("condition_frame_indexes_action") or []) != expected_action:
        failures.append(
            "plan_action_condition_indexes:"
            f"{plan.get('condition_frame_indexes_action')}!={expected_action}"
        )

    return {
        "sample_json": str(sample_path),
        "name": sample.get("name"),
        "model_mode": str(model_mode.value),
        "prefix_role": extra.get("prefix_role"),
        "prefix_frame_index": prefix_frame,
        "condition_prefix_frames": prefix_frames,
        "resolved_num_frames": int(sample_args.num_frames),
        "resolved_action_chunk_size": int(sample_args.action_chunk_size),
        "batch_video_shape_cthw": shape(video_tensor),
        "batch_action_shape": shape(action_tensor),
        "sequence_plan": plan,
        "expected_condition_frame_indexes_vision": expected_vision,
        "expected_condition_frame_indexes_action": expected_action,
        "strict_contract_ok": not failures,
        "failures": failures,
    }


def main() -> None:
    args = parse_args()
    reports = [
        probe_one(
            Path(path),
            expected_frames=args.expected_frames,
            expected_action_steps=args.expected_action_steps,
            expected_raw_action_dim=args.expected_raw_action_dim,
            temporal_compression_factor=args.temporal_compression_factor,
        )
        for path in args.sample_json
    ]
    failures = [f"{item['name']}:{failure}" for item in reports for failure in item["failures"]]
    output = {
        "strict_full_episode_inference_contract_ok": not failures,
        "failures": failures,
        "samples": reports,
        "boundary": (
            "Framework-facing inference contract probe only. This does not run "
            "Cosmos3 generation and is not world-model evidence."
        ),
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, sort_keys=True))
    if failures:
        raise SystemExit("strict full-episode inference contract failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
