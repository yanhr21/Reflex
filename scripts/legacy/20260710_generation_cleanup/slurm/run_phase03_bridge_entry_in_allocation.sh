#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  echo "refusing_login_node_execution=true; run inside a tmux-held interactive srun step" >&2
  exit 2
fi

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

PHASE="${PHASE:-04_integration}"
RUN_GROUP="${RUN_GROUP:-bridge_entry}"
if [[ -z "${RUN_NAME:-}" ]]; then
  RUN_ROOT="${ROOT}/experiments/maniskill/runs/${PHASE}/${RUN_GROUP}"
  for try_idx in $(seq 1 99); do
    candidate="$(printf 'try%02d' "${try_idx}")"
    if [[ ! -e "${RUN_ROOT}/${candidate}" ]]; then
      RUN_NAME="${candidate}"
      break
    fi
  done
fi
RUN_NAME="${RUN_NAME:?no_available_short_run_name}"
RUN_ID="${RUN_ID:-${RUN_GROUP}/${RUN_NAME}}"
RUN_DIR="${RUN_DIR:-${ROOT}/experiments/maniskill/runs/${PHASE}/${RUN_GROUP}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/${PHASE}/${RUN_GROUP}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"

PHASE01_RUN="${PHASE01_RUN:-${ROOT}/experiments/maniskill/runs/01_dp_static/p01_static_trace3_20260703_003237_162153_server64}"
PHASE02_RUN="${PHASE02_RUN:-${ROOT}/experiments/maniskill/runs/02_cosmos_imagination/p02_cosmos_rgb6_config_20260703_012531_162203_server44}"
RGB_EXTRACTOR_RUN="${RGB_EXTRACTOR_RUN:-}"
ORACLE_ACTION_DIAGNOSTICS="${ORACLE_ACTION_DIAGNOSTICS:-${ROOT}/experiments/maniskill/runs/03_oracle/action_diag/try04/action_interface_diagnostic.json:${ROOT}/experiments/maniskill/runs/03_oracle/action_diag/try05/action_interface_diagnostic.json:${ROOT}/experiments/maniskill/runs/03_oracle/action_diag/try09/action_interface_diagnostic.json:${ROOT}/experiments/maniskill/runs/03_oracle/action_diag/try10/action_interface_diagnostic.json}"
PROJECT_PYTHON="${PROJECT_PYTHON:-${ROOT}/.venv/bin/python}"
OVERLAY_LIMIT="${OVERLAY_LIMIT:-256}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "phase=${PHASE}"
  echo "run_id=${RUN_ID}"
  echo "run_group=${RUN_GROUP}"
  echo "run_name=${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  echo "log_file=${LOG_FILE}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "slurm_step_id=${SLURM_STEP_ID}"
  echo "node=$(hostname -s)"
  echo "phase01_run=${PHASE01_RUN}"
  echo "phase02_run=${PHASE02_RUN}"
  echo "rgb_extractor_run=${RGB_EXTRACTOR_RUN}"
  echo "oracle_action_diagnostics=${ORACLE_ACTION_DIAGNOSTICS}"
  echo "project_python=${PROJECT_PYTHON}"
  echo "overlay_limit=${OVERLAY_LIMIT}"
  echo "controller_execution_used=false"
  echo "oracle_used=false"
  echo "phase03_oracle_used_as_upper_bound_reference_only=true"
  echo "method_evidence_allowed=false"
  echo "forbidden_state_intervention_used=false"
} > "${RUN_DIR}/manifest.txt"

diagnostic_args=()
IFS=':' read -r -a diagnostic_paths <<< "${ORACLE_ACTION_DIAGNOSTICS}"
for diagnostic_path in "${diagnostic_paths[@]}"; do
  if [[ -n "${diagnostic_path}" ]]; then
    diagnostic_args+=(--oracle-action-diagnostic "${diagnostic_path}")
  fi
done

"${PROJECT_PYTHON}" -u scripts/world_model/phase03_bridge_diagnostic_entry.py \
  --phase01-run "${PHASE01_RUN}" \
  --phase02-run "${PHASE02_RUN}" \
  --output-dir "${RUN_DIR}" \
  --run-id "${RUN_ID}" \
  --phase "${PHASE}" \
  --rgb-extractor-run "${RGB_EXTRACTOR_RUN}" \
  --overlay-limit "${OVERLAY_LIMIT}" \
  "${diagnostic_args[@]}" \
  2>&1 | tee "${LOG_FILE}"

echo "phase04_status=bridge_entry_created_no_controller_execution" > "${RUN_DIR}/classification.txt"
echo "method_evidence_allowed=false" >> "${RUN_DIR}/classification.txt"
echo "physical_insertion_success=false" >> "${RUN_DIR}/classification.txt"
echo "oracle_evidence=false" >> "${RUN_DIR}/classification.txt"
