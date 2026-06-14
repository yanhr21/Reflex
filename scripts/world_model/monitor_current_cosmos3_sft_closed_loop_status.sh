#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745}"
SLURM_JOB_ID_TO_CHECK="${SLURM_JOB_ID_TO_CHECK:-127350}"
OUTPUT_JSON="${OUTPUT_JSON:-${ROOT}/docs/world_model_task_rebinding/2026-06-13_cosmos3_sft_closed_loop_status_auto.json}"
OUTPUT_MD="${OUTPUT_MD:-${ROOT}/docs/world_model_task_rebinding/2026-06-13_cosmos3_sft_closed_loop_status_auto.md}"
MAX_LIVE_RUNS="${MAX_LIVE_RUNS:-20}"

cd "${ROOT}"

python3 "${ROOT}/scripts/world_model/summarize_cosmos3_sft_closed_loop_status.py" \
  --sft-root "${SFT_ROOT}" \
  --slurm-job-id "${SLURM_JOB_ID_TO_CHECK}" \
  --output-json "${OUTPUT_JSON}" \
  --output-md "${OUTPUT_MD}" \
  --max-live-runs "${MAX_LIVE_RUNS}"
