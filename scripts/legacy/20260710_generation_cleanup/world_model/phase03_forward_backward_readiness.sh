#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

SOURCE_KEY="${SOURCE_KEY:-hole_late_reverse_seed1040038_idx0004}"
RUN_GROUP="${RUN_GROUP:-h5_reverse}"
RUN_NAME="${RUN_NAME:-try21}"
PROBE_GROUP="${PROBE_GROUP:-render_probe}"
PROBE_NAME="${PROBE_NAME:-fwdback21}"
JOB_NAME="${JOB_NAME:-p03_fb21}"
INCLUDE_SCHEDULER_TEST="${INCLUDE_SCHEDULER_TEST:-false}"
PARTITION="${PARTITION:-gpu}"
TIME_LIMIT="${TIME_LIMIT:-01:30:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
EXCLUDE_NODES="${EXCLUDE_NODES:-server02,server21,server27,server28,server30,server39,server53,server57}"
MAX_TEST_ONLY_DELAY_MINUTES="${MAX_TEST_ONLY_DELAY_MINUTES:-120}"
MAX_PREMOTION_COSMOS_PREDICTIONS="${MAX_PREMOTION_COSMOS_PREDICTIONS:-4}"
MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER="${MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER:-4}"

fail() {
  echo "phase03_forward_backward_readiness_ok=false"
  echo "reason=$1"
  exit "${2:-1}"
}

is_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

default_exclude_nodes_from_script() {
  sed -n 's/^EXCLUDE_NODES="${EXCLUDE_NODES:-\(.*\)}"$/\1/p' "$1" | head -1
}

if ! is_nonnegative_integer "${MAX_PREMOTION_COSMOS_PREDICTIONS}" || [[ "${MAX_PREMOTION_COSMOS_PREDICTIONS}" -lt 4 ]]; then
  echo "max_premotion_cosmos_predictions=${MAX_PREMOTION_COSMOS_PREDICTIONS}"
  fail "insufficient_premotion_cosmos_predictions_for_coverage" 47
fi

if ! is_nonnegative_integer "${MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER}" || [[ "${MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER}" -lt 4 ]]; then
  echo "min_cosmos_dynamic_actions_before_finisher=${MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER}"
  fail "insufficient_cosmos_dynamic_actions_before_finisher_for_coverage" 47
fi

case "${SOURCE_KEY}" in
  hole_late_reverse_*) expected_group="h5_reverse" ;;
  hole_late_move_stop_*) expected_group="h5_move_stop" ;;
  *)
    fail "source_key_is_not_forward_backward" 45
    ;;
esac

if [[ "${RUN_GROUP}" != "${expected_group}" ]]; then
  echo "source_key=${SOURCE_KEY}"
  echo "run_group=${RUN_GROUP}"
  echo "expected_run_group=${expected_group}"
  fail "run_group_does_not_match_source_key" 45
fi

source_h5="${ROOT}/experiments/maniskill/data/fix3_733/canonical_h5/${SOURCE_KEY}.fix3/${SOURCE_KEY}.h5"
if [[ ! -f "${source_h5}" ]]; then
  echo "source_key=${SOURCE_KEY}"
  echo "source_h5_path=${source_h5}"
  fail "source_h5_missing" 44
fi

gate_out="$(mktemp)"
trap 'rm -f "${gate_out}"' EXIT

set +e
scripts/world_model/require_phase03_next_coverage.sh forward_backward_target_motion >"${gate_out}"
gate_status="$?"
set -e
cat "${gate_out}"
if [[ "${gate_status}" -ne 0 ]]; then
  fail "next_coverage_gate_failed" "${gate_status}"
fi

bash -n \
  scripts/slurm/phase03_forward_backward_next.sh \
  scripts/slurm/phase03_forward_backward_probe.sh \
  scripts/slurm/launch_phase03_forward_backward_probe_tmux.sh \
  scripts/slurm/phase03_h5_source.sh \
  scripts/slurm/run_phase03_oracle_full_pipeline_in_allocation.sh \
  scripts/world_model/require_phase03_next_coverage.sh \
  scripts/world_model/require_phase03_forward_backward_launch_allowed.sh \
  scripts/world_model/phase03_next_coverage_status.sh \
  scripts/world_model/phase03_static_protocol_scan.sh

scripts/world_model/phase03_static_protocol_scan.sh

launcher_default_exclude="$(
  default_exclude_nodes_from_script scripts/slurm/launch_phase03_forward_backward_probe_tmux.sh
)"
status_default_exclude="$(
  default_exclude_nodes_from_script scripts/world_model/phase03_next_coverage_status.sh
)"
if [[ -z "${launcher_default_exclude}" || -z "${status_default_exclude}" ]]; then
  echo "launcher_default_exclude=${launcher_default_exclude:-missing}"
  echo "status_default_exclude=${status_default_exclude:-missing}"
  fail "exclude_node_default_missing" 51
fi
if [[ "${launcher_default_exclude}" != "${status_default_exclude}" ]]; then
  echo "launcher_default_exclude=${launcher_default_exclude}"
  echo "status_default_exclude=${status_default_exclude}"
  fail "exclude_node_defaults_mismatch" 51
fi
if [[ "${EXCLUDE_NODES}" != "${launcher_default_exclude}" ]]; then
  echo "requested_exclude_nodes=${EXCLUDE_NODES}"
  echo "launcher_default_exclude=${launcher_default_exclude}"
  fail "requested_exclude_nodes_do_not_match_default" 51
fi

for path in \
  "${ROOT}/experiments/maniskill/runs/03_oracle/${PROBE_GROUP}/${PROBE_NAME}" \
  "${ROOT}/experiments/maniskill/runs/03_oracle/${RUN_GROUP}/${RUN_NAME}" \
  "${ROOT}/logs/03_oracle/${PROBE_GROUP}/${PROBE_NAME}.log" \
  "${ROOT}/logs/03_oracle/${RUN_GROUP}/${RUN_NAME}.log"
do
  if [[ -e "${path}" ]]; then
    echo "existing_artifact=${path}"
    fail "target_artifact_already_exists" 50
  fi
done

if command -v squeue >/dev/null 2>&1; then
  existing_jobs="$(
    squeue -h -u "${USER}" -n "${JOB_NAME}" -o "%i %T %R" \
      | awk -v current_job="${SLURM_JOB_ID:-}" '$1 != current_job { print }' \
      || true
  )"
  if [[ -n "${existing_jobs}" ]]; then
    echo "--- existing_${JOB_NAME}_jobs ---"
    echo "${existing_jobs}"
    fail "same_name_slurm_job_already_exists" 42
  fi
fi

echo "phase03_forward_backward_readiness_ok=true"
echo "source_key=${SOURCE_KEY}"
echo "source_h5_path=${source_h5}"
echo "run_group=${RUN_GROUP}"
echo "run_name=${RUN_NAME}"
echo "probe_group=${PROBE_GROUP}"
echo "probe_name=${PROBE_NAME}"
echo "job_name=${JOB_NAME}"
echo "launcher=scripts/slurm/launch_phase03_forward_backward_probe_tmux.sh"
echo "exclude_nodes=${EXCLUDE_NODES}"

if [[ "${INCLUDE_SCHEDULER_TEST}" == "true" ]] && command -v srun >/dev/null 2>&1; then
  echo "--- scheduler_test_only_${JOB_NAME} ---"
  echo "partition=${PARTITION}"
  echo "time_limit=${TIME_LIMIT}"
  echo "cpus_per_task=${CPUS_PER_TASK}"
  echo "memory=${MEMORY}"
  echo "exclude_nodes=${EXCLUDE_NODES}"
  test_only_output="$(
    srun --test-only -p "${PARTITION}" -N1 -n1 --gres=gpu:1 \
      --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY}" \
      -t "${TIME_LIMIT}" --job-name="${JOB_NAME}" \
      --exclude="${EXCLUDE_NODES}" true 2>&1 || true
  )"
  printf '%s\n' "${test_only_output}"
  scheduler_test_job="$(
    printf '%s\n' "${test_only_output}" |
      sed -n 's/.*Job \([0-9][0-9]*\) to start at.*/\1/p' |
      head -1
  )"
  scheduler_estimated_start="$(
    printf '%s\n' "${test_only_output}" |
      sed -n 's/.*to start at \([^ ]*\) using.*/\1/p' |
      head -1
  )"
  scheduler_estimated_node="$(
    printf '%s\n' "${test_only_output}" |
      sed -n 's/.* on nodes \([^ ]*\) in partition.*/\1/p' |
      head -1
  )"
  echo "scheduler_test_job=${scheduler_test_job:-unknown}"
  echo "scheduler_estimated_start=${scheduler_estimated_start:-unknown}"
  echo "scheduler_estimated_node=${scheduler_estimated_node:-unknown}"
  echo "scheduler_max_delay_minutes=${MAX_TEST_ONLY_DELAY_MINUTES}"
  if [[ -n "${scheduler_estimated_start}" ]]; then
    now_epoch="$(date +%s)"
    start_epoch="$(date -d "${scheduler_estimated_start}" +%s 2>/dev/null || true)"
    if [[ -n "${start_epoch}" ]]; then
      delay_seconds="$((start_epoch - now_epoch))"
      max_delay_seconds="$((MAX_TEST_ONLY_DELAY_MINUTES * 60))"
      echo "scheduler_delay_seconds=${delay_seconds}"
      if (( delay_seconds > max_delay_seconds )); then
        echo "scheduler_within_delay_threshold=false"
      else
        echo "scheduler_within_delay_threshold=true"
      fi
    else
      echo "scheduler_delay_seconds=unknown"
      echo "scheduler_within_delay_threshold=unknown"
    fi
  else
    echo "scheduler_delay_seconds=unknown"
    echo "scheduler_within_delay_threshold=unknown"
  fi
fi
