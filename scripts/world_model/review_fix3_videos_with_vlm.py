#!/usr/bin/env python3
"""Review fix3 PegInsertionSide videos with a local video VLM.

The reviewer consumes each MP4 directly. It does not use flattened review
sheets, because the gate is about temporal behavior: target motion timing,
pre-insertion alignment, visible hole entry, and final non-wall insertion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import glob
import json
from pathlib import Path
import re
from typing import Any

import torch
import tyro
from transformers import AutoModelForImageTextToText, AutoProcessor


PROMPT = """You are reviewing a robotics simulation video from ManiSkill PegInsertionSide-v1.

Task: a Panda robot must grasp the red/white peg, aim the red peg head at the visible square hole in the yellow target block, the target block should move only after the peg is already near and aimed at the old/current hole, then the robot should re-align and insert the red peg head into the visible hole. Reject videos where the peg goes into the block side/wall instead of the visible hole.

Pay special attention to the yellow target block and its square hole. Compare the block/hole position against the table grain, grid background, and robot base before and after the near-hole alignment moment. A real target motion may be a fast 0.09-0.17 meter shift over a short interval; do not call it stationary if the yellow block/hole visibly changes position after the peg is already near the old/current hole.

Watch the video temporally. Do not rely on metadata. Return only a JSON object with exactly these keys:
{
  "target_motion_starts_after_peg_near_hole": boolean,
  "peg_aligned_before_target_motion": boolean,
  "target_moves_late_and_visibly": boolean,
  "red_head_enters_visible_hole": boolean,
  "not_wall_or_side_insert": boolean,
  "final_insert_visible_ok": boolean,
  "peg_remains_grasped": boolean,
  "pass": boolean,
  "failure_reason": string,
  "evidence": string
}

Set "target_motion_starts_after_peg_near_hole" true if the first visible target motion happens only after the robot already holds the peg and the red peg head is near/aimed at the old/current hole.
Set "pass" to true only if all seven boolean checks above are true. Be strict: if the red peg head appears offset from the hole, slides into the side face, disappears through the block wall, or the target starts moving before near-hole alignment, mark pass=false."""


@dataclass
class Args:
    video: list[str] | None = None
    video_glob: str = ""
    manifest: str = ""
    output_dir: str = ""
    model: str = "Qwen/Qwen3-VL-4B-Instruct"
    fallback_model: str = "Qwen/Qwen3-VL-2B-Instruct"
    max_new_tokens: int = 512
    video_fps: float = 4.0
    temperature: float = 0.0
    trust_remote_code: bool = True
    local_files_only: bool = True


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _load_videos(args: Args) -> list[Path]:
    paths: list[Path] = []
    if args.video:
        paths.extend(Path(p) for p in args.video)
    if args.video_glob:
        paths.extend(Path(p) for p in sorted(glob.glob(args.video_glob)))
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        for item in manifest.get("videos", []):
            if item.get("video"):
                paths.append(Path(item["video"]))
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    missing = [str(p) for p in deduped if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing videos: {missing[:5]}")
    if not deduped:
        raise ValueError("no videos provided")
    return deduped


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _load_model(model_name: str, args: Args):
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    return model, processor, model_name


def _load_model_with_fallback(args: Args):
    try:
        return _load_model(args.model, args)
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {"event": "vlm_model_fallback", "model": args.model, "fallback": args.fallback_model, "error": str(exc)},
                sort_keys=True,
            ),
            flush=True,
        )
        return _load_model(args.fallback_model, args)


def _review_one(video_path: Path, model, processor, args: Args) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path), "fps": float(args.video_fps), "max_pixels": 512 * 512},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=bool(args.temperature > 0),
            temperature=float(args.temperature) if args.temperature > 0 else None,
        )
    prompt_len = int(inputs["input_ids"].shape[-1])
    output_ids = generated[:, prompt_len:]
    text = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    parsed = _extract_json(text)
    required = [
        "target_motion_starts_after_peg_near_hole",
        "peg_aligned_before_target_motion",
        "target_moves_late_and_visibly",
        "red_head_enters_visible_hole",
        "not_wall_or_side_insert",
        "final_insert_visible_ok",
        "peg_remains_grasped",
    ]
    computed_pass = all(bool(parsed.get(key, False)) for key in required)
    parsed["pass"] = bool(parsed.get("pass", False) and computed_pass)
    return {"video": str(video_path), "raw_response": text, "review": parsed}


def main() -> None:
    args = tyro.cli(Args)
    videos = _load_videos(args)
    output_dir = Path(args.output_dir) if args.output_dir else videos[0].parent.parent / "vlm_video_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    model, processor, loaded_model = _load_model_with_fallback(args)
    results = []
    with (output_dir / "results.jsonl").open("w") as f:
        for idx, video in enumerate(videos):
            result = _review_one(video, model, processor, args)
            result["index"] = idx
            result["model"] = loaded_model
            results.append(result)
            f.write(json.dumps(_jsonable(result), ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            print(
                json.dumps(
                    {
                        "event": "vlm_video_review_done",
                        "index": idx,
                        "video": str(video),
                        "pass": bool(result["review"].get("pass", False)),
                        "failure_reason": result["review"].get("failure_reason", ""),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
    summary = {
        "model": loaded_model,
        "num_videos": len(results),
        "num_pass": sum(1 for r in results if bool(r["review"].get("pass", False))),
        "all_pass": all(bool(r["review"].get("pass", False)) for r in results),
        "results_jsonl": str(output_dir / "results.jsonl"),
    }
    (output_dir / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "vlm_video_review_summary", **summary}, ensure_ascii=False, sort_keys=True), flush=True)
    if not summary["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
