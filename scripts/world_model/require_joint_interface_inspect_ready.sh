#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
RUN_GROUP="${RUN_GROUP:-interface_inspect}"
RUN_NAME="${RUN_NAME:-inspect01}"
RUN_DIR="${RUN_DIR:-${ROOT}/experiments/maniskill/runs/02_joint_training/${RUN_GROUP}/${RUN_NAME}}"

CLASSIFICATION="${RUN_DIR}/classification.txt"
PROJECT_SUMMARY="${RUN_DIR}/project_interface_summary.json"
COSMOS_SUMMARY="${RUN_DIR}/cosmos_interface_summary.json"
MANIFEST="${RUN_DIR}/manifest.txt"

echo "joint_interface_inspect_ready_check=true"
echo "read_only=true"
echo "submits_slurm=false"
echo "run_dir=${RUN_DIR}"
echo "classification=${CLASSIFICATION}"
echo "project_summary=${PROJECT_SUMMARY}"
echo "cosmos_summary=${COSMOS_SUMMARY}"
echo "manifest=${MANIFEST}"

failures=0

require_file() {
  local label="$1"
  local path="$2"
  if [[ -f "${path}" ]]; then
    echo "${label}_exists=true"
    echo "${label}_bytes=$(stat -c '%s' "${path}")"
  else
    echo "${label}_exists=false"
    failures=$((failures + 1))
  fi
}

require_grep() {
  local label="$1"
  local path="$2"
  local pattern="$3"
  if [[ -f "${path}" ]] && grep -qE "${pattern}" "${path}"; then
    echo "${label}=true"
  else
    echo "${label}=false"
    failures=$((failures + 1))
  fi
}

require_file "manifest" "${MANIFEST}"
require_file "classification" "${CLASSIFICATION}"
require_file "project_summary" "${PROJECT_SUMMARY}"
require_file "cosmos_summary" "${COSMOS_SUMMARY}"

require_grep "classification_complete" "${CLASSIFICATION}" '^interface_inspect_status=complete$'
require_grep "project_status_ok" "${PROJECT_SUMMARY}" '"status"[[:space:]]*:[[:space:]]*"ok"'
require_grep "cosmos_status_ok" "${COSMOS_SUMMARY}" '"status"[[:space:]]*:[[:space:]]*"ok"'
require_grep "manifest_method_evidence_false" "${MANIFEST}" '^method_evidence_allowed=false$'
require_grep "manifest_uses_toy_model_false" "${MANIFEST}" '^uses_toy_model=false$'
require_grep "manifest_training_started_false" "${MANIFEST}" '^training_started=false$'
require_grep "manifest_data_generation_started_false" "${MANIFEST}" '^data_generation_started=false$'

if [[ "${failures}" -eq 0 ]]; then
  echo "joint_interface_inspect_ready=true"
  echo "allowed_next_step=build_joint_overfit_dataset_and_batch_inspector"
  echo "failure_count=0"
  exit 0
fi

echo "joint_interface_inspect_ready=false"
echo "reason=interface_inspect_incomplete_or_failed"
echo "failure_count=${failures}"
exit 62
