#!/usr/bin/env bash
set -euo pipefail

RUNNER="${1:?usage: audit_dataset_runner_source.sh <runner>}"

echo "dataset_runner_source_audit_ok=true"
echo "runner=${RUNNER}"

if [[ ! -f "${RUNNER}" ]]; then
  echo "runner_source_ready=false"
  echo "reason=runner_missing"
  exit 20
fi

if [[ ! -x "${RUNNER}" ]]; then
  echo "runner_source_ready=false"
  echo "reason=runner_not_executable"
  exit 21
fi

failures=0

check_forbidden() {
  local label="$1"
  local pattern="$2"
  if grep -nE "${pattern}" "${RUNNER}" >/tmp/reflex_runner_audit_$$.txt; then
    echo "forbidden_${label}=true"
    sed 's/^/  /' /tmp/reflex_runner_audit_$$.txt
    failures=$((failures + 1))
  else
    echo "forbidden_${label}=false"
  fi
  rm -f /tmp/reflex_runner_audit_$$.txt
}

check_required() {
  local label="$1"
  local pattern="$2"
  if grep -nE "${pattern}" "${RUNNER}" >/tmp/reflex_runner_required_$$.txt; then
    echo "required_${label}=true"
  else
    echo "required_${label}=false"
    failures=$((failures + 1))
  fi
  rm -f /tmp/reflex_runner_required_$$.txt
}

check_forbidden old_world_model_task_rebinding 'experiments/world_model_task_rebinding|world_model_task_rebinding'
check_forbidden legacy_oracle_route 'phase03|oracle|final_seat|geometric_final|geom_seat'
check_forbidden state_edit '(^|[^A-Za-z0-9_])(set_pose|set_state|set_state_dict)([^A-Za-z0-9_]|$)'
check_forbidden saved_state_restore 'saved-state|saved_state|source-state|source_state|restore'
check_forbidden future_label_controller 'future_label.*controller|teacher.*controller|controller.*future_label'
check_forbidden hidden_manual_finisher 'manual.*finisher|hidden.*finisher|hand-selected|hand_selected'
check_forbidden long_default_run_name 'RUN_(GROUP|NAME)="?\$\{RUN_(GROUP|NAME):-[^}]*((p03_)|full_pipeline|[0-9]{8}|server[0-9]+|job[0-9]+)[^}]*\}'

check_required login_node_refusal 'SLURM_JOB_ID|refusing_login_node_execution|require_dataset_runtime_context\.sh'
check_required active_output_layout 'experiments/maniskill/runs/01_dataset|require_dataset_runtime_context\.sh'
check_required active_log_layout 'logs/01_dataset|require_dataset_runtime_context\.sh'
check_required manifest 'manifest\.txt|summary\.json'
check_required review_gate 'human_review_required|large_scale_production_allowed'
check_required render_env 'VK_ICD_FILENAMES|HDF5_USE_FILE_LOCKING|require_dataset_runtime_context\.sh'

if [[ "${failures}" -ne 0 ]]; then
  echo "runner_source_ready=false"
  echo "failure_count=${failures}"
  exit 30
fi

echo "runner_source_ready=true"
echo "failure_count=0"
