#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
SESSION="${SESSION:-cosmos_startup_diag01}"
JOB_NAME="${JOB_NAME:-cosmos_start_diag}"
PARTITION="${PARTITION:-cpu}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
MEMORY="${MEMORY:-8G}"
TIME_LIMIT="${TIME_LIMIT:-00:20:00}"
RUN_GROUP="${RUN_GROUP:-cosmos_startup_diag}"
RUN_NAME="${RUN_NAME:-import_trace01}"
RUNNER="${ROOT}/scripts/slurm/run_cosmos_startup_import_diag_in_allocation.sh"
CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/maniskill/runs/02_joint_training/joint_cosmos_condition/overfit09/condition_root}"
NODELIST="${NODELIST:-}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"

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
if [[ ! -s "${CONDITION_ROOT}/train/video_dataset_file.jsonl" ]]; then
  echo "refusing_missing_condition_root=true" >&2
  echo "condition_root=${CONDITION_ROOT}" >&2
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
export CONDITION_ROOT="${CONDITION_ROOT}"
export TIMEOUT_SECONDS="${TIMEOUT_SECONDS}"
echo "launch_timestamp=\$(date -Is)" | tee -a "${LOG_FILE}"
echo "session=${SESSION}" | tee -a "${LOG_FILE}"
echo "job_name=${JOB_NAME}" | tee -a "${LOG_FILE}"
echo "runner=${RUNNER}" | tee -a "${LOG_FILE}"
echo "run_dir=${RUN_DIR}" | tee -a "${LOG_FILE}"
echo "condition_root=${CONDITION_ROOT}" | tee -a "${LOG_FILE}"
echo "launcher=launch_cosmos_startup_import_diag_tmux" | tee -a "${LOG_FILE}"
echo "partition=${PARTITION}" | tee -a "${LOG_FILE}"
echo "cpus_per_task=${CPUS_PER_TASK}" | tee -a "${LOG_FILE}"
echo "memory=${MEMORY}" | tee -a "${LOG_FILE}"
echo "time_limit=${TIME_LIMIT}" | tee -a "${LOG_FILE}"
echo "nodelist=${NODELIST}" | tee -a "${LOG_FILE}"
echo "timeout_seconds=${TIMEOUT_SECONDS}" | tee -a "${LOG_FILE}"
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
echo "runner=${RUNNER}"
