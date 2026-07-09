#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
SESSION="${SESSION:-joint_cosmos_condition01}"
JOB_NAME="${JOB_NAME:-joint_cosmos_cond}"
PARTITION="${PARTITION:-gpu}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
MEMORY="${MEMORY:-8G}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
RUN_GROUP="${RUN_GROUP:-joint_cosmos_condition}"
RUN_NAME="${RUN_NAME:-overfit01}"
JOINT_DATASET_DIR="${JOINT_DATASET_DIR:-${ROOT}/experiments/maniskill/runs/02_joint_training/joint_overfit_dataset/overfit01}"
RUNNER="${ROOT}/scripts/slurm/run_joint_dp_cosmos_sft_condition_in_allocation.sh"
EXCLUDE_NODES="${EXCLUDE_NODES:-server02,server05,server07,server10,server18,server23,server28,server30,server34,server35,server36,server39,server43,server44,server46,server51,server52,server53,server56,server57,server58,server59,server60,server63}"
NODELIST="${NODELIST:-}"
SOURCE_VIDEO_FRAMES="${SOURCE_VIDEO_FRAMES:-300}"
PREFLIGHT_VIDEO_FRAMES="${PREFLIGHT_VIDEO_FRAMES:-${EXPECTED_VIDEO_FRAMES:-297}}"
EXPECTED_VIDEO_FRAMES="${EXPECTED_VIDEO_FRAMES:-297}"
EXPECTED_STEPS="${EXPECTED_STEPS:-296}"
EXPECTED_SOURCE_EPISODES="${EXPECTED_SOURCE_EPISODES:-8}"

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
if [[ ! -s "${JOINT_DATASET_DIR}/batch_inspection.json" ]]; then
  echo "refusing_missing_joint_batch_inspection=true" >&2
  echo "joint_dataset_dir=${JOINT_DATASET_DIR}" >&2
  exit 45
fi

RUN_DIR="${ROOT}/experiments/maniskill/runs/02_joint_training/${RUN_GROUP}/${RUN_NAME}"
LOG_DIR="${ROOT}/logs/02_joint_training/${RUN_GROUP}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

if [[ -e "${RUN_DIR}" ]] && [[ -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing_existing_nonempty_run_dir=true" >&2
  echo "run_dir=${RUN_DIR}" >&2
  exit 46
fi

mkdir -p "${LOG_DIR}"

srun_cmd=(
  srun
  --partition="${PARTITION}"
  --job-name="${JOB_NAME}"
  --ntasks=1
  --cpus-per-task="${CPUS_PER_TASK}"
  --mem="${MEMORY}"
  --time="${TIME_LIMIT}"
)
if [[ "${GPUS}" != "0" ]]; then
  srun_cmd+=(--gres="gpu:${GPUS}")
fi
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
export JOINT_DATASET_DIR="${JOINT_DATASET_DIR}"
export SOURCE_VIDEO_FRAMES="${SOURCE_VIDEO_FRAMES}"
export PREFLIGHT_VIDEO_FRAMES="${PREFLIGHT_VIDEO_FRAMES}"
export EXPECTED_VIDEO_FRAMES="${EXPECTED_VIDEO_FRAMES}"
export EXPECTED_STEPS="${EXPECTED_STEPS}"
export EXPECTED_SOURCE_EPISODES="${EXPECTED_SOURCE_EPISODES}"
export VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json"
export DISPLAY=
export HDF5_USE_FILE_LOCKING=FALSE
echo "launch_timestamp=\$(date -Is)" | tee -a "${LOG_FILE}"
echo "session=${SESSION}" | tee -a "${LOG_FILE}"
echo "job_name=${JOB_NAME}" | tee -a "${LOG_FILE}"
echo "runner=${RUNNER}" | tee -a "${LOG_FILE}"
echo "run_dir=${RUN_DIR}" | tee -a "${LOG_FILE}"
echo "joint_dataset_dir=${JOINT_DATASET_DIR}" | tee -a "${LOG_FILE}"
echo "source_video_frames=${SOURCE_VIDEO_FRAMES}" | tee -a "${LOG_FILE}"
echo "preflight_video_frames=${PREFLIGHT_VIDEO_FRAMES}" | tee -a "${LOG_FILE}"
echo "expected_video_frames=${EXPECTED_VIDEO_FRAMES}" | tee -a "${LOG_FILE}"
echo "expected_steps=${EXPECTED_STEPS}" | tee -a "${LOG_FILE}"
echo "expected_source_episodes=${EXPECTED_SOURCE_EPISODES}" | tee -a "${LOG_FILE}"
echo "launcher=launch_joint_dp_cosmos_sft_condition_tmux" | tee -a "${LOG_FILE}"
echo "partition=${PARTITION}" | tee -a "${LOG_FILE}"
echo "gpus=${GPUS}" | tee -a "${LOG_FILE}"
echo "cpus_per_task=${CPUS_PER_TASK}" | tee -a "${LOG_FILE}"
echo "memory=${MEMORY}" | tee -a "${LOG_FILE}"
echo "time_limit=${TIME_LIMIT}" | tee -a "${LOG_FILE}"
echo "exclude_nodes=${EXCLUDE_NODES}" | tee -a "${LOG_FILE}"
echo "nodelist=${NODELIST}" | tee -a "${LOG_FILE}"
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
echo "joint_dataset_dir=${JOINT_DATASET_DIR}"
echo "runner=${RUNNER}"
echo "gpus=${GPUS}"
echo "time_limit=${TIME_LIMIT}"
