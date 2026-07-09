#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
SESSION="${SESSION:-joint_cosmos_sft_overfit01}"
JOB_NAME="${JOB_NAME:-joint_cosmos_sft}"
PARTITION="${PARTITION:-gpu}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-96G}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
RUN_GROUP="${RUN_GROUP:-cosmos_sft_overfit}"
RUN_NAME="${RUN_NAME:-overfit01}"
CONDITION_RUN_DIR="${CONDITION_RUN_DIR:-${ROOT}/experiments/maniskill/runs/02_joint_training/joint_cosmos_condition/overfit05}"
RUNNER="${ROOT}/scripts/slurm/run_joint_cosmos_sft_overfit_in_allocation.sh"
EXCLUDE_NODES="${EXCLUDE_NODES:-server02,server05,server07,server10,server18,server23,server28,server30,server34,server35,server36,server39,server43,server44,server46,server51,server52,server53,server56,server57,server58,server59,server60,server63}"
NODELIST="${NODELIST:-}"
COSMOS_JOB_NAME="${COSMOS_JOB_NAME:-joint_cosmos_sft_${RUN_NAME}}"
MAX_ITER="${MAX_ITER:-5}"
SAVE_ITER="${SAVE_ITER:-5}"
VALIDATION_ITER="${VALIDATION_ITER:-5}"
MAX_VAL_ITER="${MAX_VAL_ITER:-2}"
RUN_VALIDATION="${RUN_VALIDATION:-true}"
RUN_VALIDATION_ON_START="${RUN_VALIDATION_ON_START:-false}"
GRAD_ACCUM_ITER="${GRAD_ACCUM_ITER:-2}"
MODEL_COMPILE_ENABLED="${MODEL_COMPILE_ENABLED:-true}"
ENABLE_LORA="${ENABLE_LORA:-false}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen}"
USE_TORCHRUN="${USE_TORCHRUN:-true}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "refusing_existing_tmux_session=true" >&2
  echo "session=${SESSION}" >&2
  exit 43
fi
if [[ ! -f "${RUNNER}" ]]; then
  echo "refusing_missing_runner=true" >&2
  echo "runner=${RUNNER}" >&2
  exit 44
fi
if [[ ! -s "${CONDITION_RUN_DIR}/condition_preflight.json" ]]; then
  echo "refusing_missing_condition_preflight=true" >&2
  echo "condition_run_dir=${CONDITION_RUN_DIR}" >&2
  exit 45
fi
if ! grep -q '"strict_alignment_ok"[[:space:]]*:[[:space:]]*true' "${CONDITION_RUN_DIR}/condition_preflight.json"; then
  echo "refusing_condition_preflight_not_strict_ok=true" >&2
  echo "condition_preflight=${CONDITION_RUN_DIR}/condition_preflight.json" >&2
  exit 46
fi

RUN_DIR="${ROOT}/experiments/maniskill/runs/02_joint_training/${RUN_GROUP}/${RUN_NAME}"
LOG_DIR="${ROOT}/logs/02_joint_training/${RUN_GROUP}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

if [[ -e "${RUN_DIR}" ]] && [[ -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing_existing_nonempty_run_dir=true" >&2
  echo "run_dir=${RUN_DIR}" >&2
  exit 47
fi

mkdir -p "${LOG_DIR}"

srun_cmd=(
  srun
  --partition="${PARTITION}"
  --job-name="${JOB_NAME}"
  --gres="gpu:${GPUS}"
  --ntasks=1
  --cpus-per-task="${CPUS_PER_TASK}"
  --mem="${MEMORY}"
  --time="${TIME_LIMIT}"
)
if [[ -n "${EXCLUDE_NODES}" ]]; then
  srun_cmd+=(--exclude="${EXCLUDE_NODES}")
fi
if [[ -n "${NODELIST}" ]]; then
  srun_cmd+=(--nodelist="${NODELIST}")
fi
srun_cmd+=(bash "${RUNNER}")

inner=$(cat <<EOF
set -euo pipefail
cd "${ROOT}"
export ROOT="${ROOT}"
export RUN_GROUP="${RUN_GROUP}"
export RUN_NAME="${RUN_NAME}"
export RUN_DIR="${RUN_DIR}"
export LOG_DIR="${LOG_DIR}"
export LOG_FILE="${LOG_FILE}"
export CONDITION_RUN_DIR="${CONDITION_RUN_DIR}"
export COSMOS_JOB_NAME="${COSMOS_JOB_NAME}"
export MAX_ITER="${MAX_ITER}"
export SAVE_ITER="${SAVE_ITER}"
export VALIDATION_ITER="${VALIDATION_ITER}"
export MAX_VAL_ITER="${MAX_VAL_ITER}"
export RUN_VALIDATION="${RUN_VALIDATION}"
export RUN_VALIDATION_ON_START="${RUN_VALIDATION_ON_START}"
export GRAD_ACCUM_ITER="${GRAD_ACCUM_ITER}"
export MODEL_COMPILE_ENABLED="${MODEL_COMPILE_ENABLED}"
export ENABLE_LORA="${ENABLE_LORA}"
export LORA_RANK="${LORA_RANK}"
export LORA_ALPHA="${LORA_ALPHA}"
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES}"
export USE_TORCHRUN="${USE_TORCHRUN}"
export VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json"
export DISPLAY=
export HDF5_USE_FILE_LOCKING=FALSE
echo "launch_timestamp=\$(date -Is)" | tee -a "${LOG_FILE}"
echo "session=${SESSION}" | tee -a "${LOG_FILE}"
echo "job_name=${JOB_NAME}" | tee -a "${LOG_FILE}"
echo "runner=${RUNNER}" | tee -a "${LOG_FILE}"
echo "run_dir=${RUN_DIR}" | tee -a "${LOG_FILE}"
echo "condition_run_dir=${CONDITION_RUN_DIR}" | tee -a "${LOG_FILE}"
echo "launcher=launch_joint_cosmos_sft_overfit_tmux" | tee -a "${LOG_FILE}"
echo "partition=${PARTITION}" | tee -a "${LOG_FILE}"
echo "gpus=${GPUS}" | tee -a "${LOG_FILE}"
echo "cpus_per_task=${CPUS_PER_TASK}" | tee -a "${LOG_FILE}"
echo "memory=${MEMORY}" | tee -a "${LOG_FILE}"
echo "time_limit=${TIME_LIMIT}" | tee -a "${LOG_FILE}"
echo "exclude_nodes=${EXCLUDE_NODES}" | tee -a "${LOG_FILE}"
echo "nodelist=${NODELIST}" | tee -a "${LOG_FILE}"
echo "max_iter=${MAX_ITER}" | tee -a "${LOG_FILE}"
echo "save_iter=${SAVE_ITER}" | tee -a "${LOG_FILE}"
echo "validation_iter=${VALIDATION_ITER}" | tee -a "${LOG_FILE}"
echo "max_val_iter=${MAX_VAL_ITER}" | tee -a "${LOG_FILE}"
echo "run_validation=${RUN_VALIDATION}" | tee -a "${LOG_FILE}"
echo "run_validation_on_start=${RUN_VALIDATION_ON_START}" | tee -a "${LOG_FILE}"
echo "grad_accum_iter=${GRAD_ACCUM_ITER}" | tee -a "${LOG_FILE}"
echo "model_compile_enabled=${MODEL_COMPILE_ENABLED}" | tee -a "${LOG_FILE}"
echo "enable_lora=${ENABLE_LORA}" | tee -a "${LOG_FILE}"
echo "lora_rank=${LORA_RANK}" | tee -a "${LOG_FILE}"
echo "lora_alpha=${LORA_ALPHA}" | tee -a "${LOG_FILE}"
echo "lora_target_modules=${LORA_TARGET_MODULES}" | tee -a "${LOG_FILE}"
echo "use_torchrun=${USE_TORCHRUN}" | tee -a "${LOG_FILE}"
echo "srun_output_start=\$(date -Is)" | tee -a "${LOG_FILE}"
set +e
$(printf '%q ' "${srun_cmd[@]}") 2>&1 | tee -a "${LOG_FILE}"
srun_status=\${PIPESTATUS[0]}
set -e
echo "srun_output_end=\$(date -Is)" | tee -a "${LOG_FILE}"
echo "srun_exit_status=\${srun_status}" | tee -a "${LOG_FILE}"
if [[ "\${srun_status}" -ne 0 ]]; then
  echo "srun_failed=\$(date -Is)" | tee -a "${LOG_FILE}"
  exit "\${srun_status}"
fi
echo "tmux_command_complete=\$(date -Is)" | tee -a "${LOG_FILE}"
EOF
)

tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${inner}")"

echo "launched_tmux_session=${SESSION}"
echo "log_file=${LOG_FILE}"
echo "run_dir=${RUN_DIR}"
echo "condition_run_dir=${CONDITION_RUN_DIR}"
echo "runner=${RUNNER}"
echo "gpus=${GPUS}"
echo "cpus_per_task=${CPUS_PER_TASK}"
echo "memory=${MEMORY}"
echo "time_limit=${TIME_LIMIT}"
