#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this trainer inside a compute-node srun step from a tmux-held allocation.
example=srun --jobid=<held_job_id> --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=80G bash scripts/slurm/run_contact_action_suffix_generator_train_in_allocation.sh
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUFFIX_BANK_ROOT="${SUFFIX_BANK_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/source_insertion_suffix_bank_20260622_alloc145920}"
SUFFIX_BANK_NPZ="${SUFFIX_BANK_NPZ:-${SUFFIX_BANK_ROOT}/source_insertion_suffix_bank.npz}"
SUFFIX_BANK_JSONL="${SUFFIX_BANK_JSONL:-${SUFFIX_BANK_ROOT}/source_insertion_suffix_bank.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/contact_action_suffix_generator_full733_1gpu1h_${STAMP}_alloc${SLURM_JOB_ID}}"

BATCH_SIZE="${BATCH_SIZE:-512}"
HIDDEN_DIM="${HIDDEN_DIM:-4096}"
NUM_LAYERS="${NUM_LAYERS:-5}"
DROPOUT="${DROPOUT:-0.05}"
LR="${LR:-0.0002}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00005}"
MIN_STEPS="${MIN_STEPS:-1000}"
MAX_STEPS="${MAX_STEPS:-500000}"
MIN_WALL_SECONDS="${MIN_WALL_SECONDS:-3660}"
MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-3900}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-200}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-1000}"
VAL_FRACTION="${VAL_FRACTION:-0.15}"
SEED="${SEED:-20260623}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

mkdir -p "${OUTPUT_ROOT}"
cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=contact_action_suffix_generator_train_wrapper_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
suffix_bank_npz=${SUFFIX_BANK_NPZ}
suffix_bank_jsonl=${SUFFIX_BANK_JSONL}
output_root=${OUTPUT_ROOT}
batch_size=${BATCH_SIZE}
hidden_dim=${HIDDEN_DIM}
num_layers=${NUM_LAYERS}
dropout=${DROPOUT}
lr=${LR}
weight_decay=${WEIGHT_DECAY}
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
method_boundary=First contact-action reset training target. This learns source insertion suffix actions from the accepted 733 H5 data; it is not live task-completion evidence until saved-snapshot replay and full-panel visual/final-state gates pass.
EOF

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/train_contact_action_suffix_generator.py" \
  --suffix-bank-npz "${SUFFIX_BANK_NPZ}" \
  --suffix-bank-jsonl "${SUFFIX_BANK_JSONL}" \
  --output-root "${OUTPUT_ROOT}" \
  --val-fraction "${VAL_FRACTION}" \
  --batch-size "${BATCH_SIZE}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --num-layers "${NUM_LAYERS}" \
  --dropout "${DROPOUT}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
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
