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
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
HIDDEN_DIM="${HIDDEN_DIM:-2048}"
NUM_LAYERS="${NUM_LAYERS:-5}"
DROPOUT="${DROPOUT:-0.10}"
LR="${LR:-0.0002}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00005}"
PROGRESS_LOSS_WEIGHT="${PROGRESS_LOSS_WEIGHT:-0.3}"
CONTINUABILITY_LOSS_WEIGHT="${CONTINUABILITY_LOSS_WEIGHT:-0.3}"
RESIDUAL_L2_WEIGHT="${RESIDUAL_L2_WEIGHT:-0.0002}"
MIN_STEPS="${MIN_STEPS:-1000}"
MAX_STEPS="${MAX_STEPS:-200000}"
MIN_WALL_SECONDS="${MIN_WALL_SECONDS:-10800}"
MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-12600}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-200}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-1000}"
VAL_FRACTION="${VAL_FRACTION:-0.15}"
SEED="${SEED:-20260615}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

mkdir -p "${OUTPUT_ROOT}"
cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=cosmos3_contact_executor_train_wrapper_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
contact_executor_jsonl=${CONTACT_EXECUTOR_JSONL}
output_root=${OUTPUT_ROOT}
nproc_per_node=${NPROC_PER_NODE}
batch_size=${BATCH_SIZE}
hidden_dim=${HIDDEN_DIM}
num_layers=${NUM_LAYERS}
dropout=${DROPOUT}
lr=${LR}
weight_decay=${WEIGHT_DECAY}
progress_loss_weight=${PROGRESS_LOSS_WEIGHT}
continuability_loss_weight=${CONTINUABILITY_LOSS_WEIGHT}
residual_l2_weight=${RESIDUAL_L2_WEIGHT}
min_steps=${MIN_STEPS}
max_steps=${MAX_STEPS}
min_wall_seconds=${MIN_WALL_SECONDS}
max_wall_seconds=${MAX_WALL_SECONDS}
eval_every_steps=${EVAL_EVERY_STEPS}
save_every_steps=${SAVE_EVERY_STEPS}
val_fraction=${VAL_FRACTION}
seed=${SEED}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
boundary=Formal training wrapper only. Closed-loop evidence still needs final-state metrics and inspected videos.
EOF

.venv/bin/python - <<'PY'
import torch
print({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
})
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit("require_cuda_2gpu_failed=true")
PY

.venv/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/world_model/train_cosmos3_contact_executor.py \
  --contact-executor-jsonl "${CONTACT_EXECUTOR_JSONL}" \
  --output-root "${OUTPUT_ROOT}" \
  --val-fraction "${VAL_FRACTION}" \
  --batch-size "${BATCH_SIZE}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --num-layers "${NUM_LAYERS}" \
  --dropout "${DROPOUT}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --progress-loss-weight "${PROGRESS_LOSS_WEIGHT}" \
  --continuability-loss-weight "${CONTINUABILITY_LOSS_WEIGHT}" \
  --residual-l2-weight "${RESIDUAL_L2_WEIGHT}" \
  --min-steps "${MIN_STEPS}" \
  --max-steps "${MAX_STEPS}" \
  --min-wall-seconds "${MIN_WALL_SECONDS}" \
  --max-wall-seconds "${MAX_WALL_SECONDS}" \
  --formal-min-gpus 2 \
  --eval-every-steps "${EVAL_EVERY_STEPS}" \
  --save-every-steps "${SAVE_EVERY_STEPS}" \
  --require-cuda \
  --seed "${SEED}"
