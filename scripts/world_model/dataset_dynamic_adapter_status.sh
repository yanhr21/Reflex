#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
ADAPTER="${ADAPTER:-${ROOT}/scripts/world_model/active_dynamic_peg_adapter.py}"

echo "dataset_dynamic_adapter_status_ok=true"
echo "read_only=true"
echo "submits_slurm=false"
echo "adapter=${ADAPTER}"

if [[ ! -f "${ADAPTER}" ]]; then
  echo "dynamic_adapter_ready=false"
  echo "reason=adapter_missing"
  echo "required_for_stages=B_dynamic_rgb_observation,C_frozen_dp_dynamic_failure,D_future_frame_cooperation_teacher,E_cosmos_predicted_cooperation"
  echo "required_rule=no_per_step_pose_edit_or_state_restore"
  echo "allowed_direction=reviewed_dynamic_env_adapter_or_teacher_only_diagnostic"
  exit 20
fi

if [[ ! -x "${ADAPTER}" ]]; then
  echo "dynamic_adapter_ready=false"
  echo "reason=adapter_not_executable"
  exit 21
fi

failures=0

check_forbidden() {
  local label="$1"
  local pattern="$2"
  local tmp
  tmp="$(mktemp)"
  if grep -nE "${pattern}" "${ADAPTER}" >"${tmp}"; then
    echo "forbidden_${label}=true"
    sed 's/^/  /' "${tmp}"
    failures=$((failures + 1))
  else
    echo "forbidden_${label}=false"
  fi
  rm -f "${tmp}"
}

check_required() {
  local label="$1"
  local pattern="$2"
  if grep -nE "${pattern}" "${ADAPTER}" >/dev/null; then
    echo "required_${label}=true"
  else
    echo "required_${label}=false"
    failures=$((failures + 1))
  fi
}

check_forbidden old_world_model_task_rebinding 'experiments/world_model_task_rebinding|world_model_task_rebinding'
check_forbidden legacy_oracle_route 'phase03|oracle|final_seat|geometric_final|geom_seat'
check_forbidden state_edit '(^|[^A-Za-z0-9_])(set_pose|set_state|set_state_dict)([^A-Za-z0-9_]|$)'
check_forbidden saved_state_restore 'saved-state|saved_state|source-state|source_state|restore'
check_forbidden future_label_controller 'future_label.*controller|teacher.*controller|controller.*future_label'
check_forbidden hidden_manual_finisher 'manual.*finisher|hidden.*finisher|hand-selected|hand_selected'

check_required continuous_motion_trace 'target_motion_trace|hole_motion_trace|motion_trace'
check_required kinematic_target_command 'set_kinematic_target|kinematic_target'
check_required rgb_evidence 'rgb|video|render'
check_required manifest_fields 'dataset_class|allowed_losses|disallowed_losses'
check_required no_state_intervention_label 'state_intervention|snap_or_teleport'

if [[ "${failures}" -ne 0 ]]; then
  echo "dynamic_adapter_ready=false"
  echo "reason=adapter_source_audit_failed"
  echo "failure_count=${failures}"
  exit 30
fi

echo "dynamic_adapter_ready=true"
echo "reason=adapter_exists_and_source_audit_passed"
echo "failure_count=0"
