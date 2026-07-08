#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
APPROVAL_RUN_GROUP="${APPROVAL_RUN_GROUP:-static_rgb}"
APPROVAL_RUN_NAME="${APPROVAL_RUN_NAME:-smoke05}"

echo "dataset_next_stage_status_ok=true"
echo "approval_smoke=${APPROVAL_RUN_GROUP}/${APPROVAL_RUN_NAME}"

stage_paths() {
  case "$1" in
    a_static_full)
      echo "${ROOT}/scripts/slurm/launch_dataset_static_rgb_full_tmux.sh"
      echo "${ROOT}/scripts/slurm/run_dataset_static_rgb_smoke_in_allocation.sh"
      echo ""
      ;;
    b_dynamic_smoke)
      echo "${ROOT}/scripts/slurm/launch_dataset_dynamic_rgb_smoke_tmux.sh"
      echo "${ROOT}/scripts/slurm/run_dataset_dynamic_rgb_smoke_in_allocation.sh"
      echo "b_dynamic_smoke"
      ;;
    c_frozen_dp_smoke)
      echo "${ROOT}/scripts/slurm/launch_dataset_frozen_dp_dynamic_smoke_tmux.sh"
      echo "${ROOT}/scripts/slurm/run_dataset_frozen_dp_dynamic_smoke_in_allocation.sh"
      echo "c_frozen_dp_smoke"
      ;;
    d_future_teacher_smoke)
      echo "${ROOT}/scripts/slurm/launch_dataset_future_frame_teacher_smoke_tmux.sh"
      echo "${ROOT}/scripts/slurm/run_dataset_future_frame_teacher_smoke_in_allocation.sh"
      echo "d_future_teacher_smoke"
      ;;
    e_cosmos_predicted_smoke)
      echo "${ROOT}/scripts/slurm/launch_dataset_cosmos_predicted_coop_smoke_tmux.sh"
      echo "${ROOT}/scripts/slurm/run_dataset_cosmos_predicted_coop_smoke_in_allocation.sh"
      echo "e_cosmos_predicted_smoke"
      ;;
  esac
}

for stage in \
  a_static_full \
  b_dynamic_smoke \
  c_frozen_dp_smoke \
  d_future_teacher_smoke \
  e_cosmos_predicted_smoke; do
  echo "[${stage}]"
  mapfile -t paths < <(stage_paths "${stage}")
  launcher="${paths[0]}"
  runner="${paths[1]}"
  collector_stage="${paths[2]:-}"
  echo "  launcher=${launcher}"
  if [[ -x "${launcher}" ]]; then
    echo "  launcher_exists=true"
  else
    echo "  launcher_exists=false"
  fi
  echo "  runner=${runner}"
  if [[ -x "${runner}" ]]; then
    echo "  runner_exists=true"
  else
    echo "  runner_exists=false"
  fi
  if [[ -n "${collector_stage}" ]]; then
    collector_file="$(mktemp)"
    if "${ROOT}/scripts/world_model/audit_dataset_collector_source.sh" "${collector_stage}" >"${collector_file}" 2>&1; then
      sed 's/^/  collector_audit_/' "${collector_file}"
    else
      sed 's/^/  collector_audit_/' "${collector_file}"
    fi
    rm -f "${collector_file}"
  fi
  status_file="$(mktemp)"
  if APPROVAL_RUN_GROUP="${APPROVAL_RUN_GROUP}" APPROVAL_RUN_NAME="${APPROVAL_RUN_NAME}" \
    "${ROOT}/scripts/world_model/require_dataset_stage_ready.sh" "${stage}" >"${status_file}" 2>&1; then
    sed 's/^/  /' "${status_file}"
  else
    sed 's/^/  /' "${status_file}"
  fi
  rm -f "${status_file}"

  if [[ "${stage}" == "e_cosmos_predicted_smoke" ]]; then
    e_prereq_file="$(mktemp)"
    if "${ROOT}/scripts/world_model/require_dataset_cosmos_predicted_prereqs_ready.sh" >"${e_prereq_file}" 2>&1; then
      sed 's/^/  e_prereq_/' "${e_prereq_file}"
    else
      sed 's/^/  e_prereq_/' "${e_prereq_file}"
    fi
    rm -f "${e_prereq_file}"
  fi
done
