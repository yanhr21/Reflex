#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
SESSION="${SESSION:-real_imagination_discrepancy_eval02}"
JOB_NAME="${JOB_NAME:-real_imag_disc}"
PARTITION="${PARTITION:-cpu}"
GPUS="${GPUS:-0}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
MEMORY="${MEMORY:-8G}"
TIME_LIMIT="${TIME_LIMIT:-00:20:00}"
RUN_GROUP="${RUN_GROUP:-real_imagination_discrepancy}"
RUN_NAME="${RUN_NAME:-eval02}"
RUNNER="${ROOT}/scripts/slurm/run_real_imagination_discrepancy_in_allocation.sh"
REFERENCE_VIDEO="${REFERENCE_VIDEO:-${ROOT}/experiments/maniskill/runs/02_joint_training/joint_cosmos_condition/overfit10/condition_root/samples/B_cont_episode_000002/window_rgb.mp4}"
IMAGINED_VIDEO="${IMAGINED_VIDEO:-${ROOT}/experiments/maniskill/runs/02_joint_training/cosmos_forward_eval/eval02/inference_output/forward_B_cont_episode_000002/vision.mp4}"
EXPECTED_FRAMES="${EXPECTED_FRAMES:-93}"
TRUST_MAE_THRESHOLD="${TRUST_MAE_THRESHOLD:-0.08}"
TRUST_PSNR_THRESHOLD="${TRUST_PSNR_THRESHOLD:-18.0}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
NODELIST="${NODELIST:-}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "refusing_existing_tmux_session=true" >&2
  echo "session=${SESSION}" >&2
  exit 43
fi
for required in "${RUNNER}" "${REFERENCE_VIDEO}" "${IMAGINED_VIDEO}"; do
  if [[ ! -e "${required}" ]]; then
    echo "refusing_missing_required_input=true" >&2
    echo "missing=${required}" >&2
    exit 44
  fi
done

RUN_DIR="${ROOT}/experiments/maniskill/runs/03_imagination_policy/${RUN_GROUP}/${RUN_NAME}"
LOG_DIR="${ROOT}/logs/03_imagination_policy/${RUN_GROUP}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

if [[ -e "${RUN_DIR}" ]] && [[ -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing_existing_nonempty_run_dir=true" >&2
  echo "run_dir=${RUN_DIR}" >&2
  exit 45
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
export REFERENCE_VIDEO="${REFERENCE_VIDEO}"
export IMAGINED_VIDEO="${IMAGINED_VIDEO}"
export EXPECTED_FRAMES="${EXPECTED_FRAMES}"
export TRUST_MAE_THRESHOLD="${TRUST_MAE_THRESHOLD}"
export TRUST_PSNR_THRESHOLD="${TRUST_PSNR_THRESHOLD}"
echo "launch_timestamp=\$(date -Is)" | tee -a "${LOG_FILE}"
echo "session=${SESSION}" | tee -a "${LOG_FILE}"
echo "job_name=${JOB_NAME}" | tee -a "${LOG_FILE}"
echo "runner=${RUNNER}" | tee -a "${LOG_FILE}"
echo "run_dir=${RUN_DIR}" | tee -a "${LOG_FILE}"
echo "reference_video=${REFERENCE_VIDEO}" | tee -a "${LOG_FILE}"
echo "imagined_video=${IMAGINED_VIDEO}" | tee -a "${LOG_FILE}"
echo "expected_frames=${EXPECTED_FRAMES}" | tee -a "${LOG_FILE}"
echo "launcher=launch_real_imagination_discrepancy_tmux" | tee -a "${LOG_FILE}"
echo "partition=${PARTITION}" | tee -a "${LOG_FILE}"
echo "gpus=${GPUS}" | tee -a "${LOG_FILE}"
echo "cpus_per_task=${CPUS_PER_TASK}" | tee -a "${LOG_FILE}"
echo "memory=${MEMORY}" | tee -a "${LOG_FILE}"
echo "time_limit=${TIME_LIMIT}" | tee -a "${LOG_FILE}"
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
echo "gpus=${GPUS}"
echo "time_limit=${TIME_LIMIT}"
