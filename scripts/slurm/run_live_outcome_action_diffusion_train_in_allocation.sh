#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this trainer inside a compute-node srun step from a tmux-held allocation.
example=srun --jobid=<held_job_id> --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=80G bash scripts/slurm/run_live_outcome_action_diffusion_train_in_allocation.sh
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LIVE_OUTCOME_ROOT="${LIVE_OUTCOME_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_training_source_suffix_union_plus_panel0245_plus0204fixuuid_dp96_20260623_alloc146658}"
BASE_ROWS_JSONL="${BASE_ROWS_JSONL:-${LIVE_OUTCOME_ROOT}/live_snapshot_base_rows.jsonl}"
OUTCOME_JSONL="${OUTCOME_JSONL:-${LIVE_OUTCOME_ROOT}/candidate_outcome_labels.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/live_outcome_action_diffusion_full_live_union_1gpu1h_${STAMP}_alloc${SLURM_JOB_ID}}"

BATCH_SIZE="${BATCH_SIZE:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-2048}"
NUM_LAYERS="${NUM_LAYERS:-5}"
DROPOUT="${DROPOUT:-0.05}"
LR="${LR:-0.0002}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00005}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-24}"
DIFFUSION_BETA_START="${DIFFUSION_BETA_START:-0.0001}"
DIFFUSION_BETA_END="${DIFFUSION_BETA_END:-0.02}"
DIFFUSION_LOSS_WEIGHT="${DIFFUSION_LOSS_WEIGHT:-1.0}"
POSITIVE_CONTINUABLE_WEIGHT="${POSITIVE_CONTINUABLE_WEIGHT:-0.35}"
ERROR_LOSS_WEIGHT="${ERROR_LOSS_WEIGHT:-0.4}"
STATE_LOSS_WEIGHT="${STATE_LOSS_WEIGHT:-0.4}"
PROGRESS_LOSS_WEIGHT="${PROGRESS_LOSS_WEIGHT:-0.4}"
BINARY_LOSS_WEIGHT="${BINARY_LOSS_WEIGHT:-0.5}"
VALUE_LOSS_WEIGHT="${VALUE_LOSS_WEIGHT:-0.4}"
RANK_LOSS_WEIGHT="${RANK_LOSS_WEIGHT:-0.25}"
RANK_LOSS_TEMPERATURE="${RANK_LOSS_TEMPERATURE:-0.05}"
CANDIDATE_FAMILY_FILTER="${CANDIDATE_FAMILY_FILTER:-}"
MIN_STEPS="${MIN_STEPS:-1000}"
MAX_STEPS="${MAX_STEPS:-500000}"
MIN_WALL_SECONDS="${MIN_WALL_SECONDS:-3660}"
MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-3900}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-200}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-1000}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
SEED="${SEED:-20260623}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

mkdir -p "${OUTPUT_ROOT}"
cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=live_outcome_action_diffusion_train_wrapper_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
live_outcome_root=${LIVE_OUTCOME_ROOT}
base_rows_jsonl=${BASE_ROWS_JSONL}
outcome_jsonl=${OUTCOME_JSONL}
output_root=${OUTPUT_ROOT}
batch_size=${BATCH_SIZE}
hidden_dim=${HIDDEN_DIM}
num_layers=${NUM_LAYERS}
dropout=${DROPOUT}
lr=${LR}
weight_decay=${WEIGHT_DECAY}
diffusion_steps=${DIFFUSION_STEPS}
diffusion_beta_start=${DIFFUSION_BETA_START}
diffusion_beta_end=${DIFFUSION_BETA_END}
diffusion_loss_weight=${DIFFUSION_LOSS_WEIGHT}
positive_continuable_weight=${POSITIVE_CONTINUABLE_WEIGHT}
error_loss_weight=${ERROR_LOSS_WEIGHT}
state_loss_weight=${STATE_LOSS_WEIGHT}
progress_loss_weight=${PROGRESS_LOSS_WEIGHT}
binary_loss_weight=${BINARY_LOSS_WEIGHT}
value_loss_weight=${VALUE_LOSS_WEIGHT}
rank_loss_weight=${RANK_LOSS_WEIGHT}
rank_loss_temperature=${RANK_LOSS_TEMPERATURE}
candidate_family_filter=${CANDIDATE_FAMILY_FILTER}
min_steps=${MIN_STEPS}
max_steps=${MAX_STEPS}
min_wall_seconds=${MIN_WALL_SECONDS}
max_wall_seconds=${MAX_WALL_SECONDS}
eval_every_steps=${EVAL_EVERY_STEPS}
save_every_steps=${SAVE_EVERY_STEPS}
val_fraction=${VAL_FRACTION}
seed=${SEED}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
resource_boundary=tmux-held interactive Slurm allocation; no sbatch; compute-node step only.
method_boundary=Live-outcome contact-action diffusion generator. This is trained from real saved live candidate outcomes and is not live method evidence until saved-snapshot replay and full-panel visual/final-state gates pass.
EOF

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/train_live_outcome_action_diffusion.py" \
  --base-rows-jsonl "${BASE_ROWS_JSONL}" \
  --outcome-jsonl "${OUTCOME_JSONL}" \
  --output-root "${OUTPUT_ROOT}" \
  --val-fraction "${VAL_FRACTION}" \
  --batch-size "${BATCH_SIZE}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --num-layers "${NUM_LAYERS}" \
  --dropout "${DROPOUT}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --diffusion-steps "${DIFFUSION_STEPS}" \
  --diffusion-beta-start "${DIFFUSION_BETA_START}" \
  --diffusion-beta-end "${DIFFUSION_BETA_END}" \
  --diffusion-loss-weight "${DIFFUSION_LOSS_WEIGHT}" \
  --positive-continuable-weight "${POSITIVE_CONTINUABLE_WEIGHT}" \
  --error-loss-weight "${ERROR_LOSS_WEIGHT}" \
  --state-loss-weight "${STATE_LOSS_WEIGHT}" \
  --progress-loss-weight "${PROGRESS_LOSS_WEIGHT}" \
  --binary-loss-weight "${BINARY_LOSS_WEIGHT}" \
  --value-loss-weight "${VALUE_LOSS_WEIGHT}" \
  --rank-loss-weight "${RANK_LOSS_WEIGHT}" \
  --rank-loss-temperature "${RANK_LOSS_TEMPERATURE}" \
  --candidate-family-filter "${CANDIDATE_FAMILY_FILTER}" \
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
