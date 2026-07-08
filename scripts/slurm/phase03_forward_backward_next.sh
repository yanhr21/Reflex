#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ "${SKIP_PHASE03_NEXT_COVERAGE_GUARD:-false}" == "true" ]]; then
  echo "refusing_skip_phase03_next_coverage_guard=true" >&2
  echo "reason=Phase03 forward/backward full-run launcher must run the next-coverage gate." >&2
  exit 52
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "refusing_login_node_execution=true" >&2
  echo "reason=Phase03 Oracle coverage runs, readiness, rendering, and rollout must run inside a tmux-held interactive Slurm allocation." >&2
  exit 53
fi

scripts/world_model/require_phase03_next_coverage.sh forward_backward_target_motion

# Coverage launcher for the completion-gate gap
# next_required_coverage_group=forward_backward_target_motion.
# This is a real approved-source-H5 Oracle attempt, not a future-label,
# row-offset, or synthetic-motion diagnostic.
export SOURCE_KEY="${SOURCE_KEY:-hole_late_reverse_seed1040038_idx0004}"
export RUN_NAME="${RUN_NAME:-try21}"
export REQUIRE_SOURCE_H5_PROTOCOL="${REQUIRE_SOURCE_H5_PROTOCOL:-true}"
export TARGET_MOTION_DURING_FINISHER="${TARGET_MOTION_DURING_FINISHER:-true}"
export REQUIRE_TARGET_MOTION_COMPLETE_BEFORE_FINISHER="${REQUIRE_TARGET_MOTION_COMPLETE_BEFORE_FINISHER:-true}"
export FINISHER_CONTROLLER="${FINISHER_CONTROLLER:-diffusion_policy}"
export MAX_PREMOTION_COSMOS_PREDICTIONS="${MAX_PREMOTION_COSMOS_PREDICTIONS:-4}"
export MAX_COSMOS_ROUNDS="${MAX_COSMOS_ROUNDS:-8}"
export COSMOS_ACTION_HORIZON="${COSMOS_ACTION_HORIZON:-8}"
export MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER="${MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER:-4}"
export METHOD_EVIDENCE_ALLOWED="${METHOD_EVIDENCE_ALLOWED:-false}"

is_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

if ! is_nonnegative_integer "${MAX_PREMOTION_COSMOS_PREDICTIONS}" || [[ "${MAX_PREMOTION_COSMOS_PREDICTIONS}" -lt 4 ]]; then
  cat >&2 <<EOF
refusing_insufficient_premotion_cosmos_predictions_for_coverage=true
max_premotion_cosmos_predictions=${MAX_PREMOTION_COSMOS_PREDICTIONS}
reason=Forward/backward completion coverage must keep repeated premotion RGB Cosmos predictions; the active floor is 4.
EOF
  exit 47
fi

if ! is_nonnegative_integer "${MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER}" || [[ "${MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER}" -lt 4 ]]; then
  cat >&2 <<EOF
refusing_insufficient_cosmos_dynamic_actions_before_finisher_for_coverage=true
min_cosmos_dynamic_actions_before_finisher=${MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER}
reason=Forward/backward completion coverage must execute at least four Cosmos-derived dynamic actions before any DP/manual finisher.
EOF
  exit 47
fi

case "${SOURCE_KEY}" in
  hole_late_fast_shift_*) computed_group="h5_fastshift" ;;
  hole_late_reverse_*) computed_group="h5_reverse" ;;
  hole_late_move_stop_*) computed_group="h5_move_stop" ;;
  *) computed_group="" ;;
esac
if [[ -n "${RUN_GROUP:-}" && -n "${computed_group}" && "${RUN_GROUP}" != "${computed_group}" ]]; then
  cat >&2 <<EOF
refusing_mismatched_forward_backward_run_group=true
source_key=${SOURCE_KEY}
run_group=${RUN_GROUP}
expected_run_group=${computed_group}
reason=Coverage artifacts must be stored under the source-key case group.
EOF
  exit 45
fi
if [[ "${ALLOW_EXISTING_PHASE03_RUN_DIR:-false}" != "true" && -n "${computed_group}" ]]; then
  for path in \
    "${ROOT}/experiments/maniskill/runs/03_oracle/${RUN_GROUP:-${computed_group}}/${RUN_NAME}" \
    "${ROOT}/logs/03_oracle/${RUN_GROUP:-${computed_group}}/${RUN_NAME}.log"
  do
    if [[ -e "${path}" ]]; then
      echo "refusing_existing_phase03_forward_backward_artifact=true path=${path}" >&2
      exit 50
    fi
  done
fi

case "${SOURCE_KEY}" in
  hole_late_reverse_*|hole_late_move_stop_*) ;;
  *)
    cat >&2 <<'EOF'
refusing_non_forward_backward_source_key=true
reason=This launcher is only for the Phase 03 forward/backward coverage gap. Use a hole_late_reverse_* or hole_late_move_stop_* approved source key.
EOF
    exit 46
    ;;
esac

if [[ "${REQUIRE_SOURCE_H5_PROTOCOL}" != "true" ]]; then
  cat >&2 <<'EOF'
refusing_non_source_h5_protocol_for_coverage=true
reason=Forward/backward completion coverage must use the approved fix3_733 source-H5 protocol.
EOF
  exit 47
fi

if [[ "${SOURCE_H5_REQUIRE_LIVE_MOTION_GATE:-true}" != "true" ]]; then
  cat >&2 <<'EOF'
refusing_disabled_source_h5_live_motion_gate_for_coverage=true
reason=Coverage must not start target motion before the approved key's live approach gate is satisfied.
EOF
  exit 47
fi

if [[ "${TARGET_MOTION_DURING_FINISHER}" != "true" ]]; then
  cat >&2 <<'EOF'
refusing_frozen_finisher_for_coverage=true
reason=Freezing target/hole motion is an upper-bound diagnostic, not source-H5 dynamic-trajectory coverage.
EOF
  exit 47
fi

if [[ "${REQUIRE_TARGET_MOTION_COMPLETE_BEFORE_FINISHER}" != "true" ]]; then
  cat >&2 <<'EOF'
refusing_incomplete_target_motion_before_finisher_for_coverage=true
reason=Coverage must complete the approved key's target-motion protocol before the DP/manual finisher.
EOF
  exit 47
fi

if [[ "${COSMOS_ACTION_ROW_OFFSET:-0}" != "0" ]]; then
  cat >&2 <<'EOF'
refusing_row_offset_diagnostic_for_coverage=true
reason=Completion coverage cannot use a nonzero Cosmos action-row-offset diagnostic.
EOF
  exit 47
fi

if [[ "${DYNAMIC_CONTROLLER:-cosmos3_policy}" != "cosmos3_policy" ]]; then
  cat >&2 <<'EOF'
refusing_future_label_dynamic_controller_for_coverage=true
reason=Completion coverage must execute real Cosmos-3 policy actions in the dynamic stage.
EOF
  exit 48
fi

if [[ "${METHOD_EVIDENCE_ALLOWED}" != "false" ]]; then
  cat >&2 <<'EOF'
refusing_method_evidence_for_oracle_coverage=true
reason=Phase 03 Oracle is diagnostic upper-bound evidence only and must keep method_evidence_allowed=false.
EOF
  exit 49
fi

if [[ "${SOURCE_H5_TEACHER_SUFFIX_ENABLED:-false}" == "true" || "${SOURCE_H5_TEACHER_DYNAMIC_ENABLED:-false}" == "true" ]]; then
  cat >&2 <<'EOF'
refusing_future_label_teacher_for_coverage=true
reason=Completion coverage cannot use source-H5 future-label teacher dynamic actions or suffixes.
EOF
  exit 49
fi

if [[ "${SOURCE_H5_TEACHER_DYNAMIC_ACTION_START_OFFSET:-0}" != "0" ]]; then
  cat >&2 <<'EOF'
refusing_future_label_teacher_offset_for_coverage=true
reason=Completion coverage cannot use source-H5 teacher temporal offsets.
EOF
  exit 49
fi

if [[ "${COSMOS_ACTION_DIRECTION_GUARD:-none}" != "none" ]]; then
  cat >&2 <<'EOF'
refusing_direction_guard_diagnostic_for_coverage=true
reason=Forward/backward completion coverage must first run raw real Cosmos-3 policy actions, not a diagnostic source-motion guard.
EOF
  exit 49
fi

bash scripts/slurm/phase03_h5_source.sh
