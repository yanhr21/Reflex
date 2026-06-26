#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this master pipeline only inside a compute-node srun step. It runs early iter300 diagnostics, then waits for a formal checkpoint after the 3-hour training floor.
EOF
  exit 30
fi

SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299}"
MASTER_LOG_ROOT="${MASTER_LOG_ROOT:-${SFT_ROOT}/iter300_then_formal_eval_readout_live_pipeline}"
ITER300_CHECKPOINT_ITER="${ITER300_CHECKPOINT_ITER:-300}"

cd "${ROOT}"
mkdir -p "${MASTER_LOG_ROOT}"
{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "step_id=${SLURM_STEP_ID}"
  echo "node_list=${SLURM_JOB_NODELIST:-}"
  echo "sft_root=${SFT_ROOT}"
  echo "iter300_checkpoint_iter=${ITER300_CHECKPOINT_ITER}"
  echo "boundary=master pipeline: iter300 is early diagnostic only; formal stage selects checkpoint saved after the 2-GPU/3-hour floor"
} | tee "${MASTER_LOG_ROOT}/master_pipeline_manifest.txt"

SFT_ROOT="${SFT_ROOT}" \
CHECKPOINT_ITER="${ITER300_CHECKPOINT_ITER}" \
PIPELINE_LOG_ROOT="${SFT_ROOT}/iter300_eval_readout_live_pipeline" \
bash "${ROOT}/scripts/slurm/run_cosmos3_clean_dense_late299_iter300_eval_readout_live_pipeline_in_allocation.sh" \
  2>&1 | tee "${MASTER_LOG_ROOT}/01_iter300_diagnostic_pipeline.log"

SFT_ROOT="${SFT_ROOT}" \
bash "${ROOT}/scripts/slurm/run_cosmos3_clean_dense_late299_formal_eval_readout_live_pipeline_in_allocation.sh" \
  2>&1 | tee "${MASTER_LOG_ROOT}/02_formal_pipeline.log"

echo "master_pipeline_complete=$(date -Is)" | tee -a "${MASTER_LOG_ROOT}/master_pipeline_manifest.txt"
