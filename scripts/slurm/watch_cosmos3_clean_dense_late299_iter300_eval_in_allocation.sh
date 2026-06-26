#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this wrapper only inside a compute-node srun step. It waits for the current clean-dense SFT checkpoint and then launches strict generated-video eval.
EOF
  exit 30
fi

SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299}"
CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_20260614_130050}"
CHECKPOINT_ITER="${CHECKPOINT_ITER:-300}"
printf -v CHECKPOINT_NAME 'iter_%09d' "${CHECKPOINT_ITER}"
EVAL_ROOT="${EVAL_ROOT:-${SFT_ROOT}/eval_full_episode_wam_${CHECKPOINT_NAME}}"
COSMOS_VENV="${COSMOS_VENV:-${ROOT}/.venv_cosmos313}"
N_EVAL_SAMPLES="${N_EVAL_SAMPLES:-10}"
WATCH_TIMEOUT_SECONDS="${WATCH_TIMEOUT_SECONDS:-10800}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
CHECKPOINT_STABLE_SECONDS="${CHECKPOINT_STABLE_SECONDS:-90}"
MASTER_PORT="${MASTER_PORT:-50237}"

cd "${ROOT}"
mkdir -p "${EVAL_ROOT}"
{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "step_id=${SLURM_STEP_ID}"
  echo "node_list=${SLURM_JOB_NODELIST:-}"
  echo "sft_root=${SFT_ROOT}"
  echo "condition_root=${CONDITION_ROOT}"
  echo "checkpoint_iter=${CHECKPOINT_ITER}"
  echo "eval_root=${EVAL_ROOT}"
  echo "n_eval_samples=${N_EVAL_SAMPLES}"
  echo "watch_timeout_seconds=${WATCH_TIMEOUT_SECONDS}"
  echo "checkpoint_stable_seconds=${CHECKPOINT_STABLE_SECONDS}"
  echo "boundary=strict full-episode generated eval; not controller evidence until visual/readout/closed-loop review completes"
} | tee "${EVAL_ROOT}/watch_eval_manifest.txt"

"${COSMOS_VENV}/bin/python" - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit(21)
print(f"device0={torch.cuda.get_device_name(0)}")
PY

SFT_ROOT="${SFT_ROOT}" \
CONDITION_ROOT="${CONDITION_ROOT}" \
CHECKPOINT_ITER="${CHECKPOINT_ITER}" \
EVAL_ROOT="${EVAL_ROOT}" \
N_EVAL_SAMPLES="${N_EVAL_SAMPLES}" \
WATCH_TIMEOUT_SECONDS="${WATCH_TIMEOUT_SECONDS}" \
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS}" \
CHECKPOINT_STABLE_SECONDS="${CHECKPOINT_STABLE_SECONDS}" \
MASTER_PORT="${MASTER_PORT}" \
COSMOS_VENV="${COSMOS_VENV}" \
bash "${ROOT}/scripts/slurm/watch_cosmos3_300f_checkpoint_eval_in_allocation.sh" \
  2>&1 | tee "${EVAL_ROOT}/watch_checkpoint_eval.log"
