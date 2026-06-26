#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this pipeline only inside a compute-node srun step. It sequentially waits for the current clean-dense checkpoint, runs generated eval, readout diagnostics, and live-receding closed-loop panel.
EOF
  exit 30
fi

SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299}"
CHECKPOINT_ITER="${CHECKPOINT_ITER:-300}"
printf -v CHECKPOINT_NAME 'iter_%09d' "${CHECKPOINT_ITER}"
EVAL_ROOT="${EVAL_ROOT:-${SFT_ROOT}/eval_full_episode_wam_${CHECKPOINT_NAME}}"
PIPELINE_LOG_ROOT="${PIPELINE_LOG_ROOT:-${SFT_ROOT}/iter300_eval_readout_live_pipeline}"

cd "${ROOT}"
mkdir -p "${PIPELINE_LOG_ROOT}"
{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "step_id=${SLURM_STEP_ID}"
  echo "node_list=${SLURM_JOB_NODELIST:-}"
  echo "sft_root=${SFT_ROOT}"
  echo "checkpoint_iter=${CHECKPOINT_ITER}"
  echo "eval_root=${EVAL_ROOT}"
  echo "boundary=sequential current-run pipeline; generated eval and readout are diagnostics, live-receding panel still requires video/contact-sheet inspection before any method claim"
} | tee "${PIPELINE_LOG_ROOT}/pipeline_manifest.txt"

SFT_ROOT="${SFT_ROOT}" \
CHECKPOINT_ITER="${CHECKPOINT_ITER}" \
EVAL_ROOT="${EVAL_ROOT}" \
bash "${ROOT}/scripts/slurm/watch_cosmos3_clean_dense_late299_iter300_eval_in_allocation.sh" \
  2>&1 | tee "${PIPELINE_LOG_ROOT}/01_generated_eval.log"

SFT_ROOT="${SFT_ROOT}" \
CHECKPOINT_ITER="${CHECKPOINT_ITER}" \
EVAL_ROOT="${EVAL_ROOT}" \
bash "${ROOT}/scripts/slurm/watch_cosmos3_clean_dense_late299_iter300_readout_in_allocation.sh" \
  2>&1 | tee "${PIPELINE_LOG_ROOT}/02_readout.log"

SFT_ROOT="${SFT_ROOT}" \
CHECKPOINT_ITER="${CHECKPOINT_ITER}" \
EVAL_ROOT="${EVAL_ROOT}" \
bash "${ROOT}/scripts/slurm/watch_cosmos3_clean_dense_late299_iter300_live_receding_panel_in_allocation.sh" \
  2>&1 | tee "${PIPELINE_LOG_ROOT}/03_live_receding_panel.log"

echo "pipeline_complete=$(date -Is)" | tee -a "${PIPELINE_LOG_ROOT}/pipeline_manifest.txt"
