#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  echo "refusing_login_node_execution=true; run inside a tmux-held interactive srun step" >&2
  exit 2
fi

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

PHASE="03_integration"
RUN_GROUP="${RUN_GROUP:-rgb_extractor}"
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

PHASE02_RUN="${PHASE02_RUN:-${ROOT}/experiments/maniskill/runs/02_cosmos_imagination/p02_cosmos_rgb6_config_20260703_012531_162203_server44}"
INPUT_FRAME_DIR="${INPUT_FRAME_DIR:-${PHASE02_RUN}/cosmos_review_frames}"
PROJECT_PYTHON="${PROJECT_PYTHON:-${ROOT}/.venv/bin/python}"

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
  echo "phase02_run=${PHASE02_RUN}"
  echo "input_frame_dir=${INPUT_FRAME_DIR}"
  echo "project_python=${PROJECT_PYTHON}"
  echo "source=rgb_extracted"
  echo "simulator_state_read=false"
  echo "controller_execution_used=false"
  echo "oracle_used=false"
  echo "method_evidence_allowed=false"
  echo "forbidden_state_intervention_used=false"
} > "${RUN_DIR}/manifest.txt"

"${PROJECT_PYTHON}" -u scripts/world_model/phase03_rgb_task_state_extractor.py \
  --input-frame-dir "${INPUT_FRAME_DIR}" \
  --output-dir "${RUN_DIR}" \
  --run-id "${RUN_ID}" \
  2>&1 | tee "${LOG_FILE}"

echo "phase03_status=rgb_task_state_extractor_complete" > "${RUN_DIR}/classification.txt"
echo "method_evidence_allowed=false" >> "${RUN_DIR}/classification.txt"
echo "physical_insertion_success=false" >> "${RUN_DIR}/classification.txt"
echo "oracle_evidence=false" >> "${RUN_DIR}/classification.txt"
