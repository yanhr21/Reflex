#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

SOURCE_KEY="${SOURCE_KEY:?Set SOURCE_KEY to an approved fix3_733 canonical key.}"
SOURCE_H5_PATH="${SOURCE_H5_PATH:-${ROOT}/experiments/maniskill/data/fix3_733/canonical_h5/${SOURCE_KEY}.fix3/${SOURCE_KEY}.h5}"

if [[ ! -f "${SOURCE_H5_PATH}" ]]; then
  echo "source_h5_missing=true path=${SOURCE_H5_PATH}" >&2
  exit 44
fi

case "${SOURCE_KEY}" in
  hole_late_fast_shift_*) default_group="h5_fastshift" ;;
  hole_late_reverse_*) default_group="h5_reverse" ;;
  hole_late_move_stop_*) default_group="h5_move_stop" ;;
  hole_late_continuous_insert_*) default_group="h5_continuous_insert" ;;
  hole_late_constant_*) default_group="h5_constant" ;;
  hole_late_sine_*) default_group="h5_sine" ;;
  peg_drop_*) default_group="peg_drop" ;;
  peg_disturb_*) default_group="peg_disturb" ;;
  *)
    echo "unsupported_source_key_for_short_launcher=true key=${SOURCE_KEY}" >&2
    exit 45
    ;;
esac

if [[ "${SOURCE_KEY}" == hole_late_continuous_insert_* && "${ALLOW_CONTINUOUS_INSERT_WHILE_NEXT_COVERAGE_MISSING:-false}" != "true" ]]; then
  gate_out="$(mktemp)"
  trap 'rm -f "${gate_out}"' EXIT
  set +e
  scripts/world_model/check_phase03_oracle_completion.sh >"${gate_out}"
  gate_status="$?"
  set -e
  next_group="$(awk -F= '$1 == "next_required_coverage_group" { print $2 }' "${gate_out}" | tail -1)"
  overall_complete="$(awk -F= '$1 == "phase03_oracle_overall_complete" { print $2 }' "${gate_out}" | tail -1)"
  if [[ "${overall_complete:-unknown}" != "true" && "${next_group:-}" != "continuous_insert" && "${next_group:-}" != "h5_continuous_insert" ]]; then
    cat >&2 <<EOF
refusing_continuous_insert_while_other_coverage_missing=true
source_key=${SOURCE_KEY}
completion_gate_exit_status=${gate_status}
phase03_oracle_overall_complete=${overall_complete:-unknown}
next_required_coverage_group=${next_group:-unknown}
reason=Continuous-insert samples are accepted only as single-case references. The next useful Phase 03 Oracle coverage must follow the completion gate, currently forward/backward target motion.
override=ALLOW_CONTINUOUS_INSERT_WHILE_NEXT_COVERAGE_MISSING=true
EOF
    exit 48
  fi
fi

if [[ "${ALLOW_NON_NEXT_PHASE03_COVERAGE:-false}" != "true" ]]; then
  gate_out="$(mktemp)"
  trap 'rm -f "${gate_out}"' EXIT
  set +e
  scripts/world_model/check_phase03_oracle_completion.sh >"${gate_out}"
  gate_status="$?"
  set -e
  next_group="$(awk -F= '$1 == "next_required_coverage_group" { print $2 }' "${gate_out}" | tail -1)"
  overall_complete="$(awk -F= '$1 == "phase03_oracle_overall_complete" { print $2 }' "${gate_out}" | tail -1)"
  allowed=false
  case "${next_group:-}" in
    forward_backward_target_motion)
      [[ "${default_group}" == "h5_reverse" || "${default_group}" == "h5_move_stop" ]] && allowed=true
      ;;
    left_right_target_motion)
      [[ "${default_group}" == "h5_fastshift" ]] && allowed=true
      ;;
    peg_or_wooden_stick_disturbance)
      [[ "${default_group}" == "peg_disturb" || "${default_group}" == "peg_drop" ]] && allowed=true
      ;;
    multiple_approved_fix3_733_keys|none|"")
      allowed=true
      ;;
    *)
      allowed=true
      ;;
  esac
  if [[ "${overall_complete:-unknown}" != "true" && "${allowed}" != "true" ]]; then
    cat >&2 <<EOF
refusing_non_next_phase03_coverage=true
source_key=${SOURCE_KEY}
run_group=${default_group}
completion_gate_exit_status=${gate_status}
phase03_oracle_overall_complete=${overall_complete:-unknown}
next_required_coverage_group=${next_group:-unknown}
reason=Phase 03 Oracle must follow the completion gate's next required coverage group instead of launching another case family.
override=ALLOW_NON_NEXT_PHASE03_COVERAGE=true
EOF
    exit 49
  fi
fi

export SOURCE_KEY
export SOURCE_H5_PATH
export REQUIRE_SOURCE_H5_PROTOCOL="${REQUIRE_SOURCE_H5_PROTOCOL:-true}"
export RUN_GROUP="${RUN_GROUP:-${default_group}}"
export SCENARIO_NAME="${SCENARIO_NAME:-source_${SOURCE_KEY}}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-420}"
export NEAR_TARGET_L2="${NEAR_TARGET_L2:-0.16}"
export MAX_FINISHER_STEPS="${MAX_FINISHER_STEPS:-180}"
export FINISHER_CONTROLLER="${FINISHER_CONTROLLER:-diffusion_policy}"
export MAX_PREMOTION_COSMOS_PREDICTIONS="${MAX_PREMOTION_COSMOS_PREDICTIONS:-4}"
export MAX_COSMOS_ROUNDS="${MAX_COSMOS_ROUNDS:-4}"
export COSMOS_ACTION_HORIZON="${COSMOS_ACTION_HORIZON:-8}"
export MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER="${MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER:-4}"
export TARGET_MOTION_DURING_FINISHER="${TARGET_MOTION_DURING_FINISHER:-true}"
export REQUIRE_TARGET_MOTION_COMPLETE_BEFORE_FINISHER="${REQUIRE_TARGET_MOTION_COMPLETE_BEFORE_FINISHER:-false}"
export PIPELINE_TIMEOUT="${PIPELINE_TIMEOUT:-45m}"
export COSMOS_BUILD_TIMEOUT_S="${COSMOS_BUILD_TIMEOUT_S:-180}"
export COSMOS_INFERENCE_TIMEOUT_S="${COSMOS_INFERENCE_TIMEOUT_S:-900}"
export COSMOS_EXTRACT_TIMEOUT_S="${COSMOS_EXTRACT_TIMEOUT_S:-180}"

if [[ "${SOURCE_KEY}" == peg_drop_* || "${SOURCE_KEY}" == peg_disturb_* ]]; then
  export SOURCE_H5_PEG_PERTURB_MODE="${SOURCE_H5_PEG_PERTURB_MODE:-force}"
  export SOURCE_H5_PEG_FORCE_SCALE="${SOURCE_H5_PEG_FORCE_SCALE:-25.0}"
  export SOURCE_H5_PEG_FORCE_STEPS="${SOURCE_H5_PEG_FORCE_STEPS:-8}"
fi

bash scripts/slurm/run_phase03_oracle_full_pipeline_in_allocation.sh
