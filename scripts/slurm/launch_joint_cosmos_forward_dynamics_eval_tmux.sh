#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
SESSION="${SESSION:-joint_cosmos_forward_eval01}"
JOB_NAME="${JOB_NAME:-joint_cosmos_forward_eval}"
PARTITION="${PARTITION:-gpu}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-80G}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
RUN_GROUP="${RUN_GROUP:-cosmos_forward_eval}"
RUN_NAME="${RUN_NAME:-eval01}"
CONDITION_RUN_DIR="${CONDITION_RUN_DIR:-${ROOT}/experiments/maniskill/runs/02_joint_training/joint_cosmos_condition/overfit09}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-${ROOT}/experiments/maniskill/runs/02_joint_training/cosmos_sft_overfit/overfit16}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
CONFIG_FILE="${CONFIG_FILE:-}"
RUNNER="${ROOT}/scripts/slurm/run_joint_cosmos_forward_dynamics_eval_in_allocation.sh"
EXCLUDE_NODES="${EXCLUDE_NODES:-server02,server05,server07,server10,server18,server23,server28,server30,server34,server35,server36,server39,server43,server44,server46,server51,server52,server53,server56,server57,server58,server59,server60,server63}"
NODELIST="${NODELIST:-}"
CONDITION_SPLIT="${CONDITION_SPLIT:-val}"
SAMPLE_INDEX="${SAMPLE_INDEX:-1}"
SAMPLE_UUID="${SAMPLE_UUID:-}"
SEED="${SEED:-0}"
NUM_STEPS="${NUM_STEPS:-8}"
GUIDANCE="${GUIDANCE:-1.0}"
SHIFT="${SHIFT:-10.0}"
SIGMA_MAX="${SIGMA_MAX:-80.0}"
RESOLUTION="${RESOLUTION:-480}"
IMAGE_SIZE="${IMAGE_SIZE:-480}"
ACTION_DOMAIN="${ACTION_DOMAIN:-maniskill_peg_insertion}"

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
export TRAIN_RUN_DIR="${TRAIN_RUN_DIR}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH}"
export CONFIG_FILE="${CONFIG_FILE}"
export CONDITION_SPLIT="${CONDITION_SPLIT}"
export SAMPLE_INDEX="${SAMPLE_INDEX}"
export SAMPLE_UUID="${SAMPLE_UUID}"
export SEED="${SEED}"
export NUM_STEPS="${NUM_STEPS}"
export GUIDANCE="${GUIDANCE}"
export SHIFT="${SHIFT}"
export SIGMA_MAX="${SIGMA_MAX}"
export RESOLUTION="${RESOLUTION}"
export IMAGE_SIZE="${IMAGE_SIZE}"
export ACTION_DOMAIN="${ACTION_DOMAIN}"
export VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json"
export DISPLAY=
export HDF5_USE_FILE_LOCKING=FALSE
echo "launch_timestamp=\$(date -Is)" | tee -a "${LOG_FILE}"
echo "session=${SESSION}" | tee -a "${LOG_FILE}"
echo "job_name=${JOB_NAME}" | tee -a "${LOG_FILE}"
echo "runner=${RUNNER}" | tee -a "${LOG_FILE}"
echo "run_dir=${RUN_DIR}" | tee -a "${LOG_FILE}"
echo "condition_run_dir=${CONDITION_RUN_DIR}" | tee -a "${LOG_FILE}"
echo "train_run_dir=${TRAIN_RUN_DIR}" | tee -a "${LOG_FILE}"
echo "checkpoint_path=${CHECKPOINT_PATH}" | tee -a "${LOG_FILE}"
echo "config_file=${CONFIG_FILE}" | tee -a "${LOG_FILE}"
echo "launcher=launch_joint_cosmos_forward_dynamics_eval_tmux" | tee -a "${LOG_FILE}"
echo "partition=${PARTITION}" | tee -a "${LOG_FILE}"
echo "gpus=${GPUS}" | tee -a "${LOG_FILE}"
echo "cpus_per_task=${CPUS_PER_TASK}" | tee -a "${LOG_FILE}"
echo "memory=${MEMORY}" | tee -a "${LOG_FILE}"
echo "time_limit=${TIME_LIMIT}" | tee -a "${LOG_FILE}"
echo "exclude_nodes=${EXCLUDE_NODES}" | tee -a "${LOG_FILE}"
echo "nodelist=${NODELIST}" | tee -a "${LOG_FILE}"
echo "condition_split=${CONDITION_SPLIT}" | tee -a "${LOG_FILE}"
echo "sample_index=${SAMPLE_INDEX}" | tee -a "${LOG_FILE}"
echo "sample_uuid=${SAMPLE_UUID}" | tee -a "${LOG_FILE}"
echo "num_steps=${NUM_STEPS}" | tee -a "${LOG_FILE}"
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
echo "train_run_dir=${TRAIN_RUN_DIR}"
echo "runner=${RUNNER}"
echo "gpus=${GPUS}"
echo "cpus_per_task=${CPUS_PER_TASK}"
echo "memory=${MEMORY}"
echo "time_limit=${TIME_LIMIT}"
