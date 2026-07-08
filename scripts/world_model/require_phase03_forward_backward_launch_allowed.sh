#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

ALLOW_FAR_SCHEDULER_TEST_ONLY="${ALLOW_FAR_SCHEDULER_TEST_ONLY:-false}"
SCHEDULER_TEST_CACHE_SECONDS="${SCHEDULER_TEST_CACHE_SECONDS:-0}"
EXCLUDE_NODES="${EXCLUDE_NODES:-server02,server21,server27,server28,server30,server39,server53,server57}"
MANDATORY_EXCLUDE_NODES="${MANDATORY_EXCLUDE_NODES:-server02,server21,server27,server28,server30,server39,server53,server57}"

if [[ "${SCHEDULER_TEST_CACHE_SECONDS}" != "0" ]]; then
  echo "refusing_active_launch_scheduler_cache=true" >&2
  echo "scheduler_test_cache_seconds=${SCHEDULER_TEST_CACHE_SECONDS}" >&2
  echo "reason=Active launch-required helper must use a fresh scheduler test." >&2
  exit 45
fi

IFS=',' read -r -a mandatory_exclude_array <<<"${MANDATORY_EXCLUDE_NODES}"
for mandatory_node in "${mandatory_exclude_array[@]}"; do
  [[ -z "${mandatory_node}" ]] && continue
  if [[ ",${EXCLUDE_NODES}," != *",${mandatory_node},"* ]]; then
    echo "refusing_missing_mandatory_exclude_node=true" >&2
    echo "missing_node=${mandatory_node}" >&2
    echo "exclude_nodes=${EXCLUDE_NODES}" >&2
    echo "mandatory_exclude_nodes=${MANDATORY_EXCLUDE_NODES}" >&2
    echo "reason=Active forward/backward launch must keep known bad render nodes excluded before scheduler test." >&2
    exit 46
  fi
done

status_output="$(
  INCLUDE_SCHEDULER_TEST=true \
    PARTITION="${PARTITION:-gpu}" \
    JOB_NAME="${JOB_NAME:-p03_fb21}" \
    TIME_LIMIT="${TIME_LIMIT:-01:30:00}" \
    CPUS_PER_TASK="${CPUS_PER_TASK:-4}" \
    MEMORY="${MEMORY:-32G}" \
    EXCLUDE_NODES="${EXCLUDE_NODES}" \
    MANDATORY_EXCLUDE_NODES="${MANDATORY_EXCLUDE_NODES}" \
    IMMEDIATE_SECONDS="${IMMEDIATE_SECONDS:-60}" \
    SCHEDULER_TEST_CACHE_SECONDS="${SCHEDULER_TEST_CACHE_SECONDS}" \
    MAX_TEST_ONLY_DELAY_MINUTES="${MAX_TEST_ONLY_DELAY_MINUTES:-120}" \
    scripts/world_model/phase03_next_coverage_status.sh
)"
printf '%s\n' "${status_output}"

launch_allowed="$(
  printf '%s\n' "${status_output}" |
    awk -F= '$1 == "phase03_forward_backward_launch_allowed" { print $2 }' |
    tail -1
)"
launch_block_reasons="$(
  printf '%s\n' "${status_output}" |
    awk -F= '$1 == "phase03_forward_backward_launch_block_reasons" { print $2 }' |
    tail -1
)"

if [[ "${launch_allowed}" == "true" ]]; then
  echo "phase03_forward_backward_launch_required_ok=true"
  exit 0
fi

if [[ "${ALLOW_FAR_SCHEDULER_TEST_ONLY}" == "true" && "${launch_block_reasons}" == "scheduler_delay_exceeds_threshold" ]]; then
  echo "overriding_scheduler_delay_exceeds_threshold=true"
  echo "phase03_forward_backward_launch_required_ok=true"
  exit 0
fi

if [[ "${ALLOW_FAR_SCHEDULER_TEST_ONLY}" == "true" && "${launch_block_reasons}" == "scheduler_delay_exceeds_immediate_window,scheduler_delay_exceeds_threshold" ]]; then
  echo "scheduler_delay_override_requested=true"
  echo "scheduler_delay_override_refused_by_immediate_window=true"
  echo "phase03_forward_backward_launch_required_ok=false" >&2
  echo "phase03_forward_backward_launch_allowed=${launch_allowed:-missing}" >&2
  echo "phase03_forward_backward_launch_block_reasons=${launch_block_reasons:-missing}" >&2
  echo "reason=active_launcher_would_fail_salloc_immediate_window" >&2
  exit 44
fi

echo "phase03_forward_backward_launch_required_ok=false" >&2
echo "phase03_forward_backward_launch_allowed=${launch_allowed:-missing}" >&2
echo "phase03_forward_backward_launch_block_reasons=${launch_block_reasons:-missing}" >&2
echo "override_scheduler_delay_only_with_ALLOW_FAR_SCHEDULER_TEST_ONLY=true only when the active immediate-window gate is satisfied" >&2
exit 44
