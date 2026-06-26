#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this value-head trainer inside a compute-node srun step from a tmux-held allocation.
example=srun --jobid=<held_job_id> --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=80G bash scripts/slurm/run_contact_value_head_train_in_allocation.sh
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
TRAINING_ROOT="${TRAINING_ROOT:?Set TRAINING_ROOT to a merged base/outcome label root.}"
BASE_ROWS_JSONL="${BASE_ROWS_JSONL:-${TRAINING_ROOT}/live_snapshot_base_rows.jsonl}"
OUTCOME_JSONL="${OUTCOME_JSONL:-${TRAINING_ROOT}/candidate_outcome_labels.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/contact_value_head_union_plus_panel0134_causal_suffix_1gpu1h_${STAMP}_alloc${SLURM_JOB_ID}}"

BATCH_SIZE="${BATCH_SIZE:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-2048}"
NUM_LAYERS="${NUM_LAYERS:-4}"
DROPOUT="${DROPOUT:-0.05}"
LR="${LR:-0.0002}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00005}"
ERROR_LOSS_WEIGHT="${ERROR_LOSS_WEIGHT:-0.5}"
STATE_LOSS_WEIGHT="${STATE_LOSS_WEIGHT:-0.5}"
PROGRESS_LOSS_WEIGHT="${PROGRESS_LOSS_WEIGHT:-0.5}"
BINARY_LOSS_WEIGHT="${BINARY_LOSS_WEIGHT:-0.8}"
BINARY_POSITIVE_WEIGHTS="${BINARY_POSITIVE_WEIGHTS:-1,3,1,1,2}"
RANK_LOSS_WEIGHT="${RANK_LOSS_WEIGHT:-0.35}"
RANK_LOSS_TEMPERATURE="${RANK_LOSS_TEMPERATURE:-0.05}"
SCORE_SUCCESS_WEIGHT="${SCORE_SUCCESS_WEIGHT:-0.25}"
SCORE_HANDOFF_SUCCESS_WEIGHT="${SCORE_HANDOFF_SUCCESS_WEIGHT:-1.0}"
SCORE_INSERTED_WEIGHT="${SCORE_INSERTED_WEIGHT:-0.15}"
SCORE_GRASPED_WEIGHT="${SCORE_GRASPED_WEIGHT:-0.1}"
SCORE_PROGRESS_WEIGHT="${SCORE_PROGRESS_WEIGHT:-0.35}"
SCORE_PROGRESS_DELTA_WEIGHT="${SCORE_PROGRESS_DELTA_WEIGHT:-0.25}"
SCORE_CONTINUABLE_WEIGHT="${SCORE_CONTINUABLE_WEIGHT:-0.6}"
SCORE_STATE_ABS_AXIS_WEIGHTS="${SCORE_STATE_ABS_AXIS_WEIGHTS:-1,2,4}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
MIN_STEPS="${MIN_STEPS:-1000}"
MAX_STEPS="${MAX_STEPS:-500000}"
MIN_WALL_SECONDS="${MIN_WALL_SECONDS:-3660}"
MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-3900}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-200}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-1000}"
MIN_NON_DP_SELECTED_FRACTION="${MIN_NON_DP_SELECTED_FRACTION:-0.1}"
MIN_SELECTED_HANDOFF_SUCCESS_IMPROVEMENT="${MIN_SELECTED_HANDOFF_SUCCESS_IMPROVEMENT:-0.02}"
SEED="${SEED:-20260623}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

mkdir -p "${OUTPUT_ROOT}"
cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=contact_value_head_train_wrapper_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
training_root=${TRAINING_ROOT}
base_rows_jsonl=${BASE_ROWS_JSONL}
outcome_jsonl=${OUTCOME_JSONL}
output_root=${OUTPUT_ROOT}
batch_size=${BATCH_SIZE}
hidden_dim=${HIDDEN_DIM}
num_layers=${NUM_LAYERS}
dropout=${DROPOUT}
lr=${LR}
weight_decay=${WEIGHT_DECAY}
rank_loss_weight=${RANK_LOSS_WEIGHT}
rank_loss_temperature=${RANK_LOSS_TEMPERATURE}
binary_positive_weights=${BINARY_POSITIVE_WEIGHTS}
min_wall_seconds=${MIN_WALL_SECONDS}
max_wall_seconds=${MAX_WALL_SECONDS}
seed=${SEED}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
resource_boundary=tmux-held interactive Slurm allocation; no sbatch; compute-node step only.
method_boundary=Formal consequence/value-head training from real saved-snapshot rollout labels. It may rank generated contact-action candidates but is not live method evidence until selected chunks pass saved-snapshot and live closed-loop video/final-state gates.
EOF

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/train_cosmos3_candidate_outcome_scorer.py" \
  --contact-executor-jsonl "${BASE_ROWS_JSONL}" \
  --outcome-jsonl "${OUTCOME_JSONL}" \
  --output-root "${OUTPUT_ROOT}" \
  --batch-size "${BATCH_SIZE}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --num-layers "${NUM_LAYERS}" \
  --dropout "${DROPOUT}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --error-loss-weight "${ERROR_LOSS_WEIGHT}" \
  --state-loss-weight "${STATE_LOSS_WEIGHT}" \
  --progress-loss-weight "${PROGRESS_LOSS_WEIGHT}" \
  --binary-loss-weight "${BINARY_LOSS_WEIGHT}" \
  --binary-positive-weights "${BINARY_POSITIVE_WEIGHTS}" \
  --rank-loss-weight "${RANK_LOSS_WEIGHT}" \
  --rank-loss-temperature "${RANK_LOSS_TEMPERATURE}" \
  --score-success-weight "${SCORE_SUCCESS_WEIGHT}" \
  --score-handoff-success-weight "${SCORE_HANDOFF_SUCCESS_WEIGHT}" \
  --score-inserted-weight "${SCORE_INSERTED_WEIGHT}" \
  --score-grasped-weight "${SCORE_GRASPED_WEIGHT}" \
  --score-progress-weight "${SCORE_PROGRESS_WEIGHT}" \
  --score-progress-delta-weight "${SCORE_PROGRESS_DELTA_WEIGHT}" \
  --score-continuable-weight "${SCORE_CONTINUABLE_WEIGHT}" \
  --score-state-abs-axis-weights "${SCORE_STATE_ABS_AXIS_WEIGHTS}" \
  --min-selected-handoff-success-improvement "${MIN_SELECTED_HANDOFF_SUCCESS_IMPROVEMENT}" \
  --min-non-dp-selected-fraction "${MIN_NON_DP_SELECTED_FRACTION}" \
  --val-fraction "${VAL_FRACTION}" \
  --min-steps "${MIN_STEPS}" \
  --max-steps "${MAX_STEPS}" \
  --min-wall-seconds "${MIN_WALL_SECONDS}" \
  --max-wall-seconds "${MAX_WALL_SECONDS}" \
  --formal-min-gpus 1 \
  --eval-every-steps "${EVAL_EVERY_STEPS}" \
  --save-every-steps "${SAVE_EVERY_STEPS}" \
  --require-cuda \
  --seed "${SEED}" \
  2>&1 | tee "${OUTPUT_ROOT}/train.log"
