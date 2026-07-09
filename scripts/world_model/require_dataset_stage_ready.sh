#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
STAGE="${1:-${DATASET_STAGE:-a_static_full}}"
APPROVAL_RUN_GROUP="${APPROVAL_RUN_GROUP:-static_rgb}"
APPROVAL_RUN_NAME="${APPROVAL_RUN_NAME:-smoke05}"

case "${STAGE}" in
  a_static_full|b_dynamic_smoke|c_frozen_dp_smoke|d_future_teacher_smoke|e_cosmos_predicted_smoke)
    ;;
  *)
    echo "dataset_stage_ready=false"
    echo "stage=${STAGE}"
    echo "reason=unknown_stage"
    exit 60
    ;;
esac

gate_file="$(mktemp)"
if ! RUN_GROUP="${APPROVAL_RUN_GROUP}" RUN_NAME="${APPROVAL_RUN_NAME}" \
  "${ROOT}/scripts/world_model/require_dataset_smoke_approved.sh" >"${gate_file}" 2>&1; then
  echo "dataset_stage_ready=false"
  echo "stage=${STAGE}"
  echo "reason=stage1_smoke_review_pending"
  sed 's/^/stage1_gate_/' "${gate_file}"
  rm -f "${gate_file}"
  exit 61
fi
rm -f "${gate_file}"

case "${STAGE}" in
  b_dynamic_smoke|c_frozen_dp_smoke|d_future_teacher_smoke|e_cosmos_predicted_smoke)
    static_full_file="$(mktemp)"
    if ! RUN_GROUP=static_rgb RUN_NAME=full01 "${ROOT}/scripts/world_model/require_dataset_static_full_ready.sh" >"${static_full_file}" 2>&1; then
      echo "dataset_stage_ready=false"
      echo "stage=${STAGE}"
      echo "reason=a_static_full_not_ready"
      sed 's/^/a_static_full_/' "${static_full_file}"
      rm -f "${static_full_file}"
      exit 62
    fi
    rm -f "${static_full_file}"

    dynamic_adapter_file="$(mktemp)"
    if ! "${ROOT}/scripts/world_model/dataset_dynamic_adapter_status.sh" >"${dynamic_adapter_file}" 2>&1; then
      echo "dataset_stage_ready=false"
      echo "stage=${STAGE}"
      echo "reason=dynamic_adapter_not_ready"
      sed 's/^/dynamic_adapter_/' "${dynamic_adapter_file}"
      rm -f "${dynamic_adapter_file}"
      exit 66
    fi
    rm -f "${dynamic_adapter_file}"
    ;;
esac

if [[ "${STAGE}" == "e_cosmos_predicted_smoke" ]]; then
  e_prereq_file="$(mktemp)"
  if ! "${ROOT}/scripts/world_model/require_dataset_cosmos_predicted_prereqs_ready.sh" >"${e_prereq_file}" 2>&1; then
    echo "dataset_stage_ready=false"
    echo "stage=${STAGE}"
    echo "reason=e_prereqs_not_ready"
    sed 's/^/e_prereq_/' "${e_prereq_file}"
    rm -f "${e_prereq_file}"
    exit 67
  fi
  rm -f "${e_prereq_file}"
fi

case "${STAGE}" in
  a_static_full)
    launcher="${ROOT}/scripts/slurm/launch_dataset_static_rgb_full_tmux.sh"
    runner="${ROOT}/scripts/slurm/run_dataset_static_rgb_smoke_in_allocation.sh"
    collector=""
    ;;
  b_dynamic_smoke)
    launcher="${ROOT}/scripts/slurm/launch_dataset_dynamic_rgb_smoke_tmux.sh"
    runner="${ROOT}/scripts/slurm/run_dataset_dynamic_rgb_smoke_in_allocation.sh"
    collector_stage="${STAGE}"
    ;;
  c_frozen_dp_smoke)
    launcher="${ROOT}/scripts/slurm/launch_dataset_frozen_dp_dynamic_smoke_tmux.sh"
    runner="${ROOT}/scripts/slurm/run_dataset_frozen_dp_dynamic_smoke_in_allocation.sh"
    collector_stage="${STAGE}"
    ;;
  d_future_teacher_smoke)
    launcher="${ROOT}/scripts/slurm/launch_dataset_future_frame_teacher_smoke_tmux.sh"
    runner="${ROOT}/scripts/slurm/run_dataset_future_frame_teacher_smoke_in_allocation.sh"
    collector_stage="${STAGE}"
    ;;
  e_cosmos_predicted_smoke)
    launcher="${ROOT}/scripts/slurm/launch_dataset_cosmos_predicted_coop_smoke_tmux.sh"
    runner="${ROOT}/scripts/slurm/run_dataset_cosmos_predicted_coop_smoke_in_allocation.sh"
    collector_stage="${STAGE}"
    ;;
esac

if [[ ! -x "${launcher}" ]]; then
  echo "dataset_stage_ready=false"
  echo "stage=${STAGE}"
  echo "reason=launcher_missing"
  echo "launcher=${launcher}"
  exit 63
fi

if [[ ! -x "${runner}" ]]; then
  echo "dataset_stage_ready=false"
  echo "stage=${STAGE}"
  echo "reason=runner_missing"
  echo "runner=${runner}"
  exit 64
fi

audit_file="$(mktemp)"
if ! "${ROOT}/scripts/world_model/audit_dataset_runner_source.sh" "${runner}" >"${audit_file}" 2>&1; then
  echo "dataset_stage_ready=false"
  echo "stage=${STAGE}"
  echo "reason=runner_source_audit_failed"
  sed 's/^/runner_audit_/' "${audit_file}"
  rm -f "${audit_file}"
  exit 65
fi
rm -f "${audit_file}"

if [[ -n "${collector_stage:-}" ]]; then
  collector_audit_file="$(mktemp)"
  if ! "${ROOT}/scripts/world_model/audit_dataset_collector_source.sh" "${collector_stage}" >"${collector_audit_file}" 2>&1; then
    echo "dataset_stage_ready=false"
    echo "stage=${STAGE}"
    echo "reason=collector_source_audit_failed"
    sed 's/^/collector_audit_/' "${collector_audit_file}"
    rm -f "${collector_audit_file}"
    exit 68
  fi
  rm -f "${collector_audit_file}"
fi

echo "dataset_stage_ready=true"
echo "stage=${STAGE}"
echo "reason=stage1_smoke_approved_and_launcher_runner_collector_audit_passed"
echo "launcher=${launcher}"
echo "runner=${runner}"
