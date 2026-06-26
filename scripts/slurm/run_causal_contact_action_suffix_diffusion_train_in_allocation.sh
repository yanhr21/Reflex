#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this trainer inside a compute-node srun step from a tmux-held allocation.
example=srun --jobid=<held_job_id> --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=80G bash scripts/slurm/run_causal_contact_action_suffix_diffusion_train_in_allocation.sh
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUFFIX_BANK_ROOT="${SUFFIX_BANK_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/source_insertion_suffix_bank_20260622_alloc145920}"
SUFFIX_BANK_NPZ="${SUFFIX_BANK_NPZ:-${SUFFIX_BANK_ROOT}/source_insertion_suffix_bank.npz}"
SUFFIX_BANK_JSONL="${SUFFIX_BANK_JSONL:-${SUFFIX_BANK_ROOT}/source_insertion_suffix_bank.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/causal_contact_action_suffix_diffusion_full733_1gpu1h_${STAMP}_alloc${SLURM_JOB_ID}}"

BATCH_SIZE="${BATCH_SIZE:-384}"
HIDDEN_DIM="${HIDDEN_DIM:-2048}"
NUM_LAYERS="${NUM_LAYERS:-5}"
DROPOUT="${DROPOUT:-0.05}"
LR="${LR:-0.0002}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00005}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-32}"
DIFFUSION_BETA_START="${DIFFUSION_BETA_START:-0.0001}"
DIFFUSION_BETA_END="${DIFFUSION_BETA_END:-0.02}"
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
schema=causal_contact_action_suffix_diffusion_train_wrapper_v1
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
diffusion_steps=${DIFFUSION_STEPS}
diffusion_beta_start=${DIFFUSION_BETA_START}
diffusion_beta_end=${DIFFUSION_BETA_END}
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
method_boundary=Causal direct-insertion source-suffix diffusion generator. No scenario label or future first-insert condition. Training evidence only until saved-snapshot replay and full live gates pass.
EOF

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/train_causal_contact_action_suffix_diffusion.py" \
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
  --diffusion-steps "${DIFFUSION_STEPS}" \
  --diffusion-beta-start "${DIFFUSION_BETA_START}" \
  --diffusion-beta-end "${DIFFUSION_BETA_END}" \
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
