#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
TARGET="${1:?usage: audit_dataset_collector_source.sh <stage-or-collector.py>}"

case "${TARGET}" in
  b_dynamic_smoke|b_dynamic_production)
    COLLECTOR="${ROOT}/scripts/world_model/collect_dynamic_demo_action_smoke.py"
    DATASET_CLASS="B_dynamic_rgb_observation"
    EXPECT_TEACHER="false"
    ;;
  c_frozen_dp_smoke|c_frozen_dp_production)
    COLLECTOR="${ROOT}/scripts/world_model/collect_frozen_dp_dynamic_failure_smoke.py"
    DATASET_CLASS="C_frozen_dp_dynamic_failure"
    EXPECT_TEACHER="false"
    ;;
  d_future_teacher_smoke|d_future_teacher_production)
    COLLECTOR="${ROOT}/scripts/world_model/collect_dynamic_demo_action_smoke.py"
    DATASET_CLASS="D_future_frame_cooperation_teacher"
    EXPECT_TEACHER="true"
    ;;
  e_cosmos_predicted_smoke|e_cosmos_predicted_production)
    COLLECTOR="${ROOT}/scripts/world_model/collect_cosmos_predicted_coop_smoke.py"
    DATASET_CLASS="E_cosmos_predicted_cooperation"
    EXPECT_TEACHER="false"
    ;;
  *)
    COLLECTOR="${TARGET}"
    DATASET_CLASS="${DATASET_CLASS:-unknown}"
    EXPECT_TEACHER="${EXPECT_TEACHER:-unknown}"
    ;;
esac

echo "dataset_collector_source_audit_ok=true"
echo "target=${TARGET}"
echo "collector=${COLLECTOR}"
echo "dataset_class=${DATASET_CLASS}"
echo "expected_teacher_evidence_allowed=${EXPECT_TEACHER}"

if [[ ! -f "${COLLECTOR}" ]]; then
  echo "collector_source_ready=false"
  echo "reason=collector_missing"
  exit 20
fi

failures=0

check_forbidden() {
  local label="$1"
  local pattern="$2"
  local tmp
  tmp="$(mktemp)"
  if grep -nE "${pattern}" "${COLLECTOR}" >"${tmp}"; then
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
  if grep -qE "${pattern}" "${COLLECTOR}"; then
    echo "required_${label}=true"
  else
    echo "required_${label}=false"
    failures=$((failures + 1))
  fi
}

check_forbidden old_world_model_task_rebinding 'experiments/world_model_task_rebinding|world_model_task_rebinding'
check_forbidden legacy_oracle_route 'phase03|oracle|final_seat|geometric_final|geom_seat'
check_forbidden state_edit '(^|[^A-Za-z0-9_])(set_pose|set_state|set_state_dict)([^A-Za-z0-9_]|$)'
check_forbidden saved_state_restore 'saved-state|saved_state|source-state|source_state'
check_forbidden hidden_manual_finisher 'manual.*finisher|hidden.*finisher|hand-selected|hand_selected'

if [[ "${DATASET_CLASS}" != "unknown" ]]; then
  check_required dataset_class "${DATASET_CLASS}"
fi
check_required active_dynamic_adapter 'active_dynamic_peg_adapter'
check_required legal_env_step 'env\.step|envs\.step'
check_required rgb_video_writer 'imageio\.(mimsave|imwrite)'
check_required summary_json 'summary\.json'
check_required trace_json 'trace.*\.json|motion_trace\.json|frozen_dp_trace\.json|future_teacher_trace\.json|cosmos_predicted_trace\.json'
check_required method_evidence_false '"method_evidence_allowed"[[:space:]]*:[[:space:]]*False|method_evidence_allowed.*False'
check_required positive_policy_false '"positive_policy_data_allowed"[[:space:]]*:[[:space:]]*False|positive_policy_data_allowed.*False'
check_required state_intervention_false '"state_intervention"[[:space:]]*:[[:space:]]*False|state_intervention.*False'
check_required snap_or_teleport_false '"snap_or_teleport"[[:space:]]*:[[:space:]]*False|snap_or_teleport.*False'
check_required manifest_loss_fields 'manifest_fields'
check_required trace_validation 'validate_trace_rows'

case "${EXPECT_TEACHER}" in
  true)
    check_required teacher_evidence_true '"teacher_evidence_allowed"[[:space:]]*:[[:space:]]*True|teacher_evidence_allowed.*True'
    ;;
  false)
    check_required teacher_evidence_false '"teacher_evidence_allowed"[[:space:]]*:[[:space:]]*False|teacher_evidence_allowed.*False'
    ;;
esac

if [[ "${failures}" -ne 0 ]]; then
  echo "collector_source_ready=false"
  echo "failure_count=${failures}"
  exit 30
fi

echo "collector_source_ready=true"
echo "failure_count=0"
