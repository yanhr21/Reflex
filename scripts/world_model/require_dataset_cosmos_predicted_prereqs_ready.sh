#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
COSMOS_READOUT_SUMMARY="${COSMOS_READOUT_SUMMARY:-${ROOT}/experiments/maniskill/runs/02_joint_training/cosmos_readout/val01/summary.json}"

echo "dataset_cosmos_predicted_prereqs_check=true"
echo "read_only=true"
echo "submits_slurm=false"
echo "cosmos_readout_summary=${COSMOS_READOUT_SUMMARY}"

failures=0

validate_prod() {
  local stage="$1"
  local prefix="$2"
  local status_file
  status_file="$(mktemp)"
  if "${ROOT}/scripts/world_model/validate_dataset_production_run.sh" "${stage}" >"${status_file}" 2>&1; then
    echo "${prefix}_production_valid=true"
  else
    echo "${prefix}_production_valid=false"
    sed "s/^/${prefix}_/" "${status_file}"
    failures=$((failures + 1))
  fi
  rm -f "${status_file}"
}

require_readout_pattern() {
  local label="$1"
  local pattern="$2"
  if [[ -f "${COSMOS_READOUT_SUMMARY}" ]] && grep -qE "${pattern}" "${COSMOS_READOUT_SUMMARY}"; then
    echo "${label}=true"
  else
    echo "${label}=false"
    failures=$((failures + 1))
  fi
}

validate_prod b_dynamic_production b_dynamic
validate_prod d_future_teacher_production d_future_teacher

if [[ -f "${COSMOS_READOUT_SUMMARY}" ]]; then
  echo "cosmos_readout_summary_exists=true"
else
  echo "cosmos_readout_summary_exists=false"
  failures=$((failures + 1))
fi

require_readout_pattern cosmos_readout_status_validation_complete '"status"[[:space:]]*:[[:space:]]*"validation_complete"'
require_readout_pattern cosmos_readout_heldout_b_validation_true '"heldout_b_validation"[[:space:]]*:[[:space:]]*true'
require_readout_pattern cosmos_readout_future_target_frame_ready_true '"future_target_frame_readout_ready"[[:space:]]*:[[:space:]]*true'
require_readout_pattern cosmos_readout_method_evidence_false '"method_evidence_allowed"[[:space:]]*:[[:space:]]*false'

if [[ "${failures}" -ne 0 ]]; then
  echo "dataset_cosmos_predicted_prereqs_ready=false"
  echo "reason=e_prereqs_incomplete"
  echo "failure_count=${failures}"
  exit 70
fi

echo "dataset_cosmos_predicted_prereqs_ready=true"
echo "failure_count=0"
