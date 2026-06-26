#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  echo "refusing_login_node_execution=true"
  echo "reason=Run this diffusion candidate smoke inside a compute-node srun step from a tmux-held allocation."
  exit 2
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/world_model_task_rebinding/cosmos3/candidate_executor_diffusion_smoke_${STAMP}}"

export OUTPUT_ROOT
export GENERATOR_TYPE="${GENERATOR_TYPE:-diffusion}"
export DIFFUSION_STEPS="${DIFFUSION_STEPS:-16}"
export DIFFUSION_BETA_START="${DIFFUSION_BETA_START:-0.0001}"
export DIFFUSION_BETA_END="${DIFFUSION_BETA_END:-0.02}"
export DIFFUSION_LOSS_WEIGHT="${DIFFUSION_LOSS_WEIGHT:-1.0}"
export CANDIDATE_SAMPLES="${CANDIDATE_SAMPLES:-8}"
export CANDIDATE_SCALES="${CANDIDATE_SCALES:-0.05,0.1,0.2}"
export CANDIDATE_TEMPS="${CANDIDATE_TEMPS:-0.5,1.0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export MAX_SAMPLES="${MAX_SAMPLES:-512}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export HIDDEN_DIM="${HIDDEN_DIM:-1024}"
export NUM_LAYERS="${NUM_LAYERS:-4}"
export DROPOUT="${DROPOUT:-0.05}"
export LR="${LR:-0.0002}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.00005}"
export CANDIDATE_RANK_LOSS_WEIGHT="${CANDIDATE_RANK_LOSS_WEIGHT:-0.35}"
export CANDIDATE_RANK_RANDOM_COUNT="${CANDIDATE_RANK_RANDOM_COUNT:-4}"
export CANDIDATE_RANK_DIFFUSION_COUNT="${CANDIDATE_RANK_DIFFUSION_COUNT:-1}"
export CANDIDATE_RANK_TEMPERATURE="${CANDIDATE_RANK_TEMPERATURE:-1.0}"
export MIN_STEPS="${MIN_STEPS:-50}"
export MAX_STEPS="${MAX_STEPS:-100}"
export MIN_WALL_SECONDS="${MIN_WALL_SECONDS:-0}"
export MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-0}"
export EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-25}"
export SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-50}"
export DP_FALLBACK_PHASES="${DP_FALLBACK_PHASES:-all}"
export DP_FALLBACK_SCORE_MARGIN="${DP_FALLBACK_SCORE_MARGIN:-0.0}"
export SCORE_MEAN_SOURCE_PENALTY="${SCORE_MEAN_SOURCE_PENALTY:-5.0}"
export SCORE_LARGE_SCALE_SOURCE_PENALTY="${SCORE_LARGE_SCALE_SOURCE_PENALTY:-0.5}"
export SCORE_STOCHASTIC_SOURCE_PENALTY="${SCORE_STOCHASTIC_SOURCE_PENALTY:-1.0}"
export SELECTOR_RESIDUAL_L2_CAP_MAX="${SELECTOR_RESIDUAL_L2_CAP_MAX:-0.02}"

mkdir -p "${OUTPUT_ROOT}"
cat > "${OUTPUT_ROOT}/diffusion_smoke_manifest.txt" <<EOF
schema=cosmos3_candidate_executor_diffusion_smoke_wrapper_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
output_root=${OUTPUT_ROOT}
generator_type=${GENERATOR_TYPE}
diffusion_steps=${DIFFUSION_STEPS}
candidate_samples=${CANDIDATE_SAMPLES}
candidate_scales=${CANDIDATE_SCALES}
nproc_per_node=${NPROC_PER_NODE}
max_samples=${MAX_SAMPLES}
max_steps=${MAX_STEPS}
candidate_rank_loss_weight=${CANDIDATE_RANK_LOSS_WEIGHT}
candidate_rank_random_count=${CANDIDATE_RANK_RANDOM_COUNT}
candidate_rank_diffusion_count=${CANDIDATE_RANK_DIFFUSION_COUNT}
candidate_rank_temperature=${CANDIDATE_RANK_TEMPERATURE}
boundary=Short compute-node diffusion candidate executor smoke. This is not formal method evidence and must not launch live eval.
EOF

bash scripts/slurm/run_cosmos3_candidate_executor_train_direct_in_allocation.sh
