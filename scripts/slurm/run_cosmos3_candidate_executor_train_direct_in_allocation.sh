#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  echo "refusing_login_node_execution=true"
  echo "reason=Run this wrapper inside a compute-node srun step from a tmux-held allocation."
  exit 2
fi

CONTACT_EXECUTOR_JSONL="${CONTACT_EXECUTOR_JSONL:-experiments/world_model_task_rebinding/cosmos3/contact_executor_dataset_iter1500_train512_scale005_repair_20260615/contact_executor_dataset_file.jsonl}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_direct_${STAMP}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
FORMAL_MIN_GPUS="${FORMAL_MIN_GPUS:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
HIDDEN_DIM="${HIDDEN_DIM:-1024}"
NUM_LAYERS="${NUM_LAYERS:-4}"
DROPOUT="${DROPOUT:-0.05}"
LR="${LR:-0.0002}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00005}"
CANDIDATE_RANK_LOSS_WEIGHT="${CANDIDATE_RANK_LOSS_WEIGHT:-0.35}"
CANDIDATE_RANK_RANDOM_COUNT="${CANDIDATE_RANK_RANDOM_COUNT:-4}"
CANDIDATE_RANK_DIFFUSION_COUNT="${CANDIDATE_RANK_DIFFUSION_COUNT:-0}"
CANDIDATE_RANK_TEMPERATURE="${CANDIDATE_RANK_TEMPERATURE:-1.0}"
NEXT_STATE_LOSS_WEIGHT="${NEXT_STATE_LOSS_WEIGHT:-0.5}"
MIN_STEPS="${MIN_STEPS:-1000}"
MAX_STEPS="${MAX_STEPS:-200000}"
MIN_WALL_SECONDS="${MIN_WALL_SECONDS:-10800}"
MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-12600}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-500}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-20000}"
VAL_FRACTION="${VAL_FRACTION:-0.15}"
CANDIDATE_SAMPLES="${CANDIDATE_SAMPLES:-24}"
CANDIDATE_TEMPS="${CANDIDATE_TEMPS:-0.5,1.0,1.5}"
CANDIDATE_SCALES="${CANDIDATE_SCALES:-0.05,0.1,0.2,0.5,1.0}"
SCORE_INSERTED_WEIGHT="${SCORE_INSERTED_WEIGHT:-0.6}"
SCORE_DP_CONTINUABLE_WEIGHT="${SCORE_DP_CONTINUABLE_WEIGHT:-0.3}"
SCORE_VALUE_WEIGHT="${SCORE_VALUE_WEIGHT:-0.4}"
SCORE_NEXT_STATE_WEIGHT="${SCORE_NEXT_STATE_WEIGHT:-0.8}"
SCORE_NEXT_STATE_AXIS_WEIGHTS="${SCORE_NEXT_STATE_AXIS_WEIGHTS:-1.0,2.0,4.0}"
SCORE_NEXT_STATE_TARGET="${SCORE_NEXT_STATE_TARGET:-0.0,0.0,0.0}"
SCORE_LOGPROB_WEIGHT="${SCORE_LOGPROB_WEIGHT:-0.05}"
SCORE_RESIDUAL_L2_PENALTY="${SCORE_RESIDUAL_L2_PENALTY:-0.02}"
SCORE_MEAN_SOURCE_PENALTY="${SCORE_MEAN_SOURCE_PENALTY:-0.0}"
SCORE_SCALE_SOURCE_PENALTY="${SCORE_SCALE_SOURCE_PENALTY:-0.0}"
SCORE_LARGE_SCALE_SOURCE_PENALTY="${SCORE_LARGE_SCALE_SOURCE_PENALTY:-0.0}"
SCORE_STOCHASTIC_SOURCE_PENALTY="${SCORE_STOCHASTIC_SOURCE_PENALTY:-0.25}"
GENERATOR_TYPE="${GENERATOR_TYPE:-gaussian}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-16}"
DIFFUSION_BETA_START="${DIFFUSION_BETA_START:-0.0001}"
DIFFUSION_BETA_END="${DIFFUSION_BETA_END:-0.02}"
DIFFUSION_LOSS_WEIGHT="${DIFFUSION_LOSS_WEIGHT:-1.0}"
DP_FALLBACK_PHASES="${DP_FALLBACK_PHASES:-all}"
DP_FALLBACK_SCORE_MARGIN="${DP_FALLBACK_SCORE_MARGIN:-0.25}"
SELECTOR_RESIDUAL_L2_CAP_QUANTILE="${SELECTOR_RESIDUAL_L2_CAP_QUANTILE:-0.9}"
SELECTOR_RESIDUAL_L2_CAP_MIN="${SELECTOR_RESIDUAL_L2_CAP_MIN:-0.0001}"
SELECTOR_RESIDUAL_L2_CAP_MAX="${SELECTOR_RESIDUAL_L2_CAP_MAX:-0.02}"
SELECTOR_RESIDUAL_L2_CAP_MULTIPLIER="${SELECTOR_RESIDUAL_L2_CAP_MULTIPLIER:-1.0}"
SEED="${SEED:-20260615}"

mkdir -p "${OUTPUT_ROOT}"
cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=cosmos3_candidate_executor_train_direct_wrapper_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
contact_executor_jsonl=${CONTACT_EXECUTOR_JSONL}
output_root=${OUTPUT_ROOT}
nproc_per_node=${NPROC_PER_NODE}
formal_min_gpus=${FORMAL_MIN_GPUS}
max_samples=${MAX_SAMPLES}
batch_size=${BATCH_SIZE}
hidden_dim=${HIDDEN_DIM}
num_layers=${NUM_LAYERS}
dropout=${DROPOUT}
lr=${LR}
weight_decay=${WEIGHT_DECAY}
candidate_rank_loss_weight=${CANDIDATE_RANK_LOSS_WEIGHT}
candidate_rank_random_count=${CANDIDATE_RANK_RANDOM_COUNT}
candidate_rank_diffusion_count=${CANDIDATE_RANK_DIFFUSION_COUNT}
candidate_rank_temperature=${CANDIDATE_RANK_TEMPERATURE}
next_state_loss_weight=${NEXT_STATE_LOSS_WEIGHT}
min_steps=${MIN_STEPS}
max_steps=${MAX_STEPS}
min_wall_seconds=${MIN_WALL_SECONDS}
max_wall_seconds=${MAX_WALL_SECONDS}
eval_every_steps=${EVAL_EVERY_STEPS}
save_every_steps=${SAVE_EVERY_STEPS}
candidate_samples=${CANDIDATE_SAMPLES}
candidate_temps=${CANDIDATE_TEMPS}
candidate_scales=${CANDIDATE_SCALES}
score_inserted_weight=${SCORE_INSERTED_WEIGHT}
score_dp_continuable_weight=${SCORE_DP_CONTINUABLE_WEIGHT}
score_value_weight=${SCORE_VALUE_WEIGHT}
score_next_state_weight=${SCORE_NEXT_STATE_WEIGHT}
score_next_state_axis_weights=${SCORE_NEXT_STATE_AXIS_WEIGHTS}
score_next_state_target=${SCORE_NEXT_STATE_TARGET}
score_logprob_weight=${SCORE_LOGPROB_WEIGHT}
score_residual_l2_penalty=${SCORE_RESIDUAL_L2_PENALTY}
score_mean_source_penalty=${SCORE_MEAN_SOURCE_PENALTY}
score_scale_source_penalty=${SCORE_SCALE_SOURCE_PENALTY}
score_large_scale_source_penalty=${SCORE_LARGE_SCALE_SOURCE_PENALTY}
score_stochastic_source_penalty=${SCORE_STOCHASTIC_SOURCE_PENALTY}
generator_type=${GENERATOR_TYPE}
diffusion_steps=${DIFFUSION_STEPS}
diffusion_beta_start=${DIFFUSION_BETA_START}
diffusion_beta_end=${DIFFUSION_BETA_END}
diffusion_loss_weight=${DIFFUSION_LOSS_WEIGHT}
dp_fallback_phases=${DP_FALLBACK_PHASES}
dp_fallback_score_margin=${DP_FALLBACK_SCORE_MARGIN}
selector_residual_l2_cap_quantile=${SELECTOR_RESIDUAL_L2_CAP_QUANTILE}
selector_residual_l2_cap_min=${SELECTOR_RESIDUAL_L2_CAP_MIN}
selector_residual_l2_cap_max=${SELECTOR_RESIDUAL_L2_CAP_MAX}
selector_residual_l2_cap_multiplier=${SELECTOR_RESIDUAL_L2_CAP_MULTIPLIER}
val_fraction=${VAL_FRACTION}
seed=${SEED}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}
boundary=Direct candidate-executor training wrapper. It bypasses older wrapper cache ambiguity and passes candidate_scales explicitly to the trainer. It does not launch live eval.
EOF

.venv/bin/python - <<'PY'
import torch
print({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
})
if not torch.cuda.is_available():
    raise SystemExit("require_cuda_failed=true")
PY

.venv/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/world_model/train_cosmos3_candidate_executor.py \
  --contact-executor-jsonl "${CONTACT_EXECUTOR_JSONL}" \
  --output-root "${OUTPUT_ROOT}" \
  --max-samples "${MAX_SAMPLES}" \
  --val-fraction "${VAL_FRACTION}" \
  --batch-size "${BATCH_SIZE}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --num-layers "${NUM_LAYERS}" \
  --dropout "${DROPOUT}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --candidate-rank-loss-weight "${CANDIDATE_RANK_LOSS_WEIGHT}" \
  --candidate-rank-random-count "${CANDIDATE_RANK_RANDOM_COUNT}" \
  --candidate-rank-diffusion-count "${CANDIDATE_RANK_DIFFUSION_COUNT}" \
  --candidate-rank-temperature "${CANDIDATE_RANK_TEMPERATURE}" \
  --next-state-loss-weight "${NEXT_STATE_LOSS_WEIGHT}" \
  --candidate-samples "${CANDIDATE_SAMPLES}" \
  --candidate-temps "${CANDIDATE_TEMPS}" \
  --candidate-scales "${CANDIDATE_SCALES}" \
  --score-inserted-weight "${SCORE_INSERTED_WEIGHT}" \
  --score-dp-continuable-weight "${SCORE_DP_CONTINUABLE_WEIGHT}" \
  --score-value-weight "${SCORE_VALUE_WEIGHT}" \
  --score-next-state-weight "${SCORE_NEXT_STATE_WEIGHT}" \
  --score-next-state-axis-weights "${SCORE_NEXT_STATE_AXIS_WEIGHTS}" \
  --score-next-state-target "${SCORE_NEXT_STATE_TARGET}" \
  --score-logprob-weight "${SCORE_LOGPROB_WEIGHT}" \
  --score-residual-l2-penalty "${SCORE_RESIDUAL_L2_PENALTY}" \
  --score-mean-source-penalty "${SCORE_MEAN_SOURCE_PENALTY}" \
  --score-scale-source-penalty "${SCORE_SCALE_SOURCE_PENALTY}" \
  --score-large-scale-source-penalty "${SCORE_LARGE_SCALE_SOURCE_PENALTY}" \
  --score-stochastic-source-penalty "${SCORE_STOCHASTIC_SOURCE_PENALTY}" \
  --generator-type "${GENERATOR_TYPE}" \
  --diffusion-steps "${DIFFUSION_STEPS}" \
  --diffusion-beta-start "${DIFFUSION_BETA_START}" \
  --diffusion-beta-end "${DIFFUSION_BETA_END}" \
  --diffusion-loss-weight "${DIFFUSION_LOSS_WEIGHT}" \
  --dp-fallback-phases "${DP_FALLBACK_PHASES}" \
  --dp-fallback-score-margin "${DP_FALLBACK_SCORE_MARGIN}" \
  --selector-residual-l2-cap-quantile "${SELECTOR_RESIDUAL_L2_CAP_QUANTILE}" \
  --selector-residual-l2-cap-min "${SELECTOR_RESIDUAL_L2_CAP_MIN}" \
  --selector-residual-l2-cap-max "${SELECTOR_RESIDUAL_L2_CAP_MAX}" \
  --selector-residual-l2-cap-multiplier "${SELECTOR_RESIDUAL_L2_CAP_MULTIPLIER}" \
  --min-steps "${MIN_STEPS}" \
  --max-steps "${MAX_STEPS}" \
  --min-wall-seconds "${MIN_WALL_SECONDS}" \
  --max-wall-seconds "${MAX_WALL_SECONDS}" \
  --formal-min-gpus "${FORMAL_MIN_GPUS}" \
  --eval-every-steps "${EVAL_EVERY_STEPS}" \
  --save-every-steps "${SAVE_EVERY_STEPS}" \
  --require-cuda \
  --seed "${SEED}"
