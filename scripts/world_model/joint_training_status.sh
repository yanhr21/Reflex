#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
RUN_GROUP="${RUN_GROUP:-interface_inspect}"
RUN_NAME="${RUN_NAME:-inspect01}"
RUN_DIR="${RUN_DIR:-${ROOT}/experiments/maniskill/runs/02_joint_training/${RUN_GROUP}/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${ROOT}/logs/02_joint_training/${RUN_GROUP}/${RUN_NAME}.log}"
SESSION="${SESSION:-joint_interface_inspect01}"
JOB_ID="${JOB_ID:-173059}"

echo "joint_training_status_ok=true"
echo "read_only=true"
echo "submits_slurm=false"
echo "root=${ROOT}"
echo "run_dir=${RUN_DIR}"
echo "log_file=${LOG_FILE}"
echo "session=${SESSION}"
echo "job_id=${JOB_ID}"

exists_report() {
  local label="$1"
  local path="$2"
  echo "${label}=${path}"
  if [[ -e "${path}" ]]; then
    echo "${label}_exists=true"
    if [[ -f "${path}" ]]; then
      echo "${label}_bytes=$(stat -c '%s' "${path}")"
    fi
  else
    echo "${label}_exists=false"
  fi
}

echo "[guards]"
joint_overfit_status_file="$(mktemp)"
full_joint_status_file="$(mktemp)"
interface_status_file="$(mktemp)"
squeue_status_file="$(mktemp)"
cleanup() {
  rm -f "${joint_overfit_status_file}" "${full_joint_status_file}" "${interface_status_file}" "${squeue_status_file}"
}
trap cleanup EXIT

if "${ROOT}/scripts/world_model/require_dataset_training_inputs_ready.sh" joint_overfit_abcd \
  > "${joint_overfit_status_file}" 2>&1; then
  echo "  joint_overfit_abcd_ready=true"
else
  echo "  joint_overfit_abcd_ready=false"
fi
awk '
  /^dataset_training_inputs_ready=/ ||
  /^allowed_scope=/ ||
  /^e_cosmos_predicted_required=/ ||
  /^reason=/ ||
  /^failure_count=/ {
    print "  joint_overfit_abcd_" $0
  }
' "${joint_overfit_status_file}"

if "${ROOT}/scripts/world_model/require_dataset_training_inputs_ready.sh" full_joint \
  > "${full_joint_status_file}" 2>&1; then
  echo "  full_joint_ready=true"
else
  echo "  full_joint_ready=false"
fi
awk '
  /^dataset_training_inputs_ready=/ ||
  /^allowed_scope=/ ||
  /^e_cosmos_predicted_required=/ ||
  /^reason=/ ||
  /^failure_count=/ {
    print "  full_joint_" $0
  }
' "${full_joint_status_file}"

echo "[interface_inspect]"
if RUN_DIR="${RUN_DIR}" "${ROOT}/scripts/world_model/require_joint_interface_inspect_ready.sh" \
  > "${interface_status_file}" 2>&1; then
  echo "  interface_ready=true"
else
  echo "  interface_ready=false"
fi
awk '
  /^joint_interface_inspect_ready=/ ||
  /^allowed_next_step=/ ||
  /^reason=/ ||
  /^failure_count=/ {
    print "  interface_gate_" $0
  }
' "${interface_status_file}"
exists_report "  run_dir" "${RUN_DIR}"
exists_report "  log_file" "${LOG_FILE}"
exists_report "  manifest" "${RUN_DIR}/manifest.txt"
exists_report "  classification" "${RUN_DIR}/classification.txt"
exists_report "  project_summary" "${RUN_DIR}/project_interface_summary.json"
exists_report "  cosmos_summary" "${RUN_DIR}/cosmos_interface_summary.json"
exists_report "  gate_text" "${RUN_DIR}/joint_overfit_abcd_gate.txt"

if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "  tmux_session_exists=true"
else
  echo "  tmux_session_exists=false"
fi

if command -v squeue >/dev/null 2>&1; then
  echo "  squeue_available=true"
  if squeue -j "${JOB_ID}" -h >"${squeue_status_file}" 2>/dev/null && [[ -s "${squeue_status_file}" ]]; then
    echo "  slurm_job_visible=true"
    squeue -h -j "${JOB_ID}" -o '  slurm %.18i %.9P %.30j %.8T %.10M %.9l %.20R'
  else
    echo "  slurm_job_visible=false"
  fi
else
  echo "  squeue_available=false"
fi

if [[ -f "${LOG_FILE}" ]]; then
  echo "  log_tail_begin"
  tail -n 20 "${LOG_FILE}" | sed 's/^/    /'
  echo "  log_tail_end"
fi
