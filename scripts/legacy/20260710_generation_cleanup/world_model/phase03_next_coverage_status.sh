#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"
INCLUDE_SCHEDULER_TEST="${INCLUDE_SCHEDULER_TEST:-false}"
PARTITION="${PARTITION:-gpu}"
JOB_NAME="${JOB_NAME:-p03_fb21}"
TIME_LIMIT="${TIME_LIMIT:-01:30:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
EXCLUDE_NODES="${EXCLUDE_NODES:-server02,server21,server27,server28,server30,server39,server53,server57}"
MANDATORY_EXCLUDE_NODES="${MANDATORY_EXCLUDE_NODES:-server02,server21,server27,server28,server30,server39,server53,server57}"
IMMEDIATE_SECONDS="${IMMEDIATE_SECONDS:-60}"
MAX_TEST_ONLY_DELAY_MINUTES="${MAX_TEST_ONLY_DELAY_MINUTES:-120}"
SCHEDULER_TEST_CACHE_SECONDS="${SCHEDULER_TEST_CACHE_SECONDS:-120}"
REQUIRE_IDLE_NODE_FOR_IMMEDIATE="${REQUIRE_IDLE_NODE_FOR_IMMEDIATE:-true}"
EXCLUDE_ARG_TEXT=""
if [[ -n "${EXCLUDE_NODES}" ]]; then
  EXCLUDE_ARG_TEXT="--exclude=${EXCLUDE_NODES}"
fi

is_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

if ! is_nonnegative_integer "${IMMEDIATE_SECONDS}"; then
  echo "phase03_next_coverage_status_ok=false"
  echo "reason=invalid_immediate_seconds"
  echo "immediate_seconds=${IMMEDIATE_SECONDS}"
  exit 2
fi

if ! is_nonnegative_integer "${MAX_TEST_ONLY_DELAY_MINUTES}"; then
  echo "phase03_next_coverage_status_ok=false"
  echo "reason=invalid_max_test_only_delay_minutes"
  echo "max_test_only_delay_minutes=${MAX_TEST_ONLY_DELAY_MINUTES}"
  exit 2
fi

if ! is_nonnegative_integer "${SCHEDULER_TEST_CACHE_SECONDS}"; then
  echo "phase03_next_coverage_status_ok=false"
  echo "reason=invalid_scheduler_test_cache_seconds"
  echo "scheduler_test_cache_seconds=${SCHEDULER_TEST_CACHE_SECONDS}"
  exit 2
fi

missing_mandatory_excludes=()
IFS=',' read -r -a mandatory_exclude_array <<<"${MANDATORY_EXCLUDE_NODES}"
for mandatory_node in "${mandatory_exclude_array[@]}"; do
  [[ -z "${mandatory_node}" ]] && continue
  if [[ ",${EXCLUDE_NODES}," != *",${mandatory_node},"* ]]; then
    missing_mandatory_excludes+=("${mandatory_node}")
  fi
done
if [[ "${#missing_mandatory_excludes[@]}" -gt 0 ]]; then
  mandatory_exclude_ok="false"
  missing_mandatory_exclude_nodes="$(IFS=,; echo "${missing_mandatory_excludes[*]}")"
else
  mandatory_exclude_ok="true"
  missing_mandatory_exclude_nodes="none"
fi

COMPLETION_OUT="$(mktemp)"
trap 'rm -f "${COMPLETION_OUT}"' EXIT

set +e
scripts/world_model/check_phase03_oracle_completion.sh >"${COMPLETION_OUT}"
completion_status="$?"
set -e
completion_check_ok="$(awk -F= '$1 == "phase03_oracle_completion_check_ok" { print $2 }' "${COMPLETION_OUT}" | tail -1)"
if [[ "${completion_check_ok}" == "true" && ( "${completion_status}" -eq 0 || "${completion_status}" -eq 3 ) ]]; then
  completion_gate_read_ok="true"
else
  completion_gate_read_ok="false"
fi

next_group="$(awk -F= '$1 == "next_required_coverage_group" { print $2 }' "${COMPLETION_OUT}" | tail -1)"
missing_items="$(awk -F= '$1 == "missing_coverage_items" { print $2 }' "${COMPLETION_OUT}" | tail -1)"
overall_complete="$(awk -F= '$1 == "phase03_oracle_overall_complete" { print $2 }' "${COMPLETION_OUT}" | tail -1)"
artifact_count=0
same_name_job_count=0
same_name_job_query_ok="false"
partition_idle_query_ok="unknown"
partition_idle_node_count="unknown"
partition_idle_immediate_ok="unknown"
scheduler_within_delay_threshold="unknown"
scheduler_within_immediate_window="unknown"
scheduler_launch_reason="scheduler_test_not_requested"

echo "phase03_next_coverage_status_ok=true"
echo "completion_gate_exit_status=${completion_status}"
echo "completion_gate_check_ok=${completion_check_ok:-unknown}"
echo "completion_gate_read_ok=${completion_gate_read_ok}"
echo "phase03_oracle_overall_complete=${overall_complete:-unknown}"
echo "missing_coverage_items=${missing_items:-unknown}"
echo "next_required_coverage_group=${next_group:-unknown}"
echo "mandatory_exclude_nodes=${MANDATORY_EXCLUDE_NODES}"
echo "mandatory_exclude_ok=${mandatory_exclude_ok}"
echo "missing_mandatory_exclude_nodes=${missing_mandatory_exclude_nodes}"

if [[ "${next_group}" == "forward_backward_target_motion" ]]; then
  echo "prepared_launcher=scripts/slurm/launch_phase03_forward_backward_probe_tmux.sh"
  echo "prepared_probe_script=scripts/slurm/phase03_forward_backward_probe.sh"
  echo "prepared_full_run_script=scripts/slurm/phase03_forward_backward_next.sh"
  echo "prepared_source_key=hole_late_reverse_seed1040038_idx0004"
  echo "prepared_run_group=h5_reverse"
  echo "prepared_run_name=try21"
  echo "prepared_probe_group=render_probe"
  echo "prepared_probe_name=fwdback21"
  echo "prepared_exclude_nodes=${EXCLUDE_NODES}"
fi

for path in \
  experiments/maniskill/runs/03_oracle/render_probe/fwdback21 \
  experiments/maniskill/runs/03_oracle/h5_reverse/try21 \
  logs/03_oracle/render_probe/fwdback21.log \
  logs/03_oracle/h5_reverse/try21.log
do
  if [[ -e "${path}" ]]; then
    echo "exists ${path}=true"
    artifact_count="$((artifact_count + 1))"
  else
    echo "exists ${path}=false"
  fi
done
echo "prepared_artifact_count=${artifact_count}"

if command -v squeue >/dev/null 2>&1; then
  echo "--- slurm_jobs_named_${JOB_NAME} ---"
  set +e
  same_name_jobs="$(
    squeue -h -u "${USER}" -n "${JOB_NAME}" -o "%.18i %.9P %.32j %.8T %.10M %.10l %.6D %R" 2>&1
  )"
  squeue_status="$?"
  set -e
  if [[ "${squeue_status}" -eq 0 ]]; then
    same_name_job_query_ok="true"
    if [[ -n "${same_name_jobs}" ]]; then
      printf '%s\n' "${same_name_jobs}"
      same_name_job_count="$(printf '%s\n' "${same_name_jobs}" | awk 'NF { count += 1 } END { print count + 0 }')"
    fi
  else
    same_name_job_query_ok="false"
    echo "same_name_slurm_job_query_error=${same_name_jobs}"
  fi
else
  echo "same_name_slurm_job_query_error=squeue_missing"
fi
echo "same_name_slurm_job_query_ok=${same_name_job_query_ok}"
echo "same_name_slurm_job_count=${same_name_job_count}"

if [[ "${REQUIRE_IDLE_NODE_FOR_IMMEDIATE}" == "true" && "${IMMEDIATE_SECONDS}" -gt 0 && "${next_group}" == "forward_backward_target_motion" ]]; then
  if command -v sinfo >/dev/null 2>&1; then
    set +e
    partition_idle_raw="$(sinfo -h -p "${PARTITION}" -o "%F" 2>&1)"
    sinfo_status="$?"
    set -e
    if [[ "${sinfo_status}" -eq 0 ]]; then
      partition_idle_query_ok="true"
      partition_idle_node_count="$(
        printf '%s\n' "${partition_idle_raw}" |
          awk -F/ 'NF >= 4 { idle += $2 } END { print idle + 0 }'
      )"
      if [[ "${partition_idle_node_count}" -gt 0 ]]; then
        partition_idle_immediate_ok="true"
      else
        partition_idle_immediate_ok="false"
      fi
    else
      partition_idle_query_ok="false"
      echo "partition_idle_query_error=${partition_idle_raw}"
    fi
  else
    partition_idle_query_ok="false"
    echo "partition_idle_query_error=sinfo_missing"
  fi
fi
echo "partition_idle_query_ok=${partition_idle_query_ok}"
echo "partition_idle_node_count=${partition_idle_node_count}"
echo "partition_idle_immediate_ok=${partition_idle_immediate_ok}"

pre_scheduler_block_reasons=()
if [[ "${completion_gate_read_ok}" != "true" ]]; then
  pre_scheduler_block_reasons+=("completion_gate_failed")
fi
if [[ "${next_group}" != "forward_backward_target_motion" ]]; then
  pre_scheduler_block_reasons+=("next_required_coverage_is_${next_group:-unknown}")
fi
if [[ "${artifact_count}" -ne 0 ]]; then
  pre_scheduler_block_reasons+=("prepared_artifacts_exist")
fi
if [[ "${same_name_job_count}" -ne 0 ]]; then
  pre_scheduler_block_reasons+=("same_name_slurm_job_exists")
fi
if [[ "${same_name_job_query_ok}" != "true" ]]; then
  pre_scheduler_block_reasons+=("same_name_slurm_job_query_failed")
fi
if [[ "${mandatory_exclude_ok}" != "true" ]]; then
  pre_scheduler_block_reasons+=("mandatory_exclude_nodes_missing")
fi
if [[ "${partition_idle_immediate_ok}" == "false" ]]; then
  pre_scheduler_block_reasons+=("partition_idle_nodes_zero")
fi
if [[ "${#pre_scheduler_block_reasons[@]}" -gt 0 ]]; then
  pre_scheduler_block_reasons_text="$(IFS=,; echo "${pre_scheduler_block_reasons[*]}")"
else
  pre_scheduler_block_reasons_text="none"
fi
echo "pre_scheduler_block_reasons=${pre_scheduler_block_reasons_text}"

if [[ "${INCLUDE_SCHEDULER_TEST}" == "true" && "${next_group}" == "forward_backward_target_motion" ]]; then
  if [[ "${pre_scheduler_block_reasons_text}" != "none" ]]; then
    echo "scheduler_test_skipped=true"
    echo "scheduler_test_skip_reason=${pre_scheduler_block_reasons_text}"
  else
    echo "scheduler_test_skipped=false"
    echo "--- scheduler_test_only_${JOB_NAME} ---"
    echo "partition=${PARTITION}"
    echo "job_name=${JOB_NAME}"
    echo "time_limit=${TIME_LIMIT}"
    echo "cpus_per_task=${CPUS_PER_TASK}"
    echo "memory=${MEMORY}"
    echo "immediate_seconds=${IMMEDIATE_SECONDS}"
    echo "scheduler_test_cache_seconds=${SCHEDULER_TEST_CACHE_SECONDS}"
    echo "exclude_nodes=${EXCLUDE_NODES}"
    cache_key="$(
      printf '%s\n' \
        "partition=${PARTITION}" \
        "job_name=${JOB_NAME}" \
        "time_limit=${TIME_LIMIT}" \
        "cpus_per_task=${CPUS_PER_TASK}" \
        "memory=${MEMORY}" \
        "exclude_nodes=${EXCLUDE_NODES}" |
        cksum |
        awk '{ print $1 }'
    )"
    cache_file="/tmp/phase03_scheduler_test_${USER}_${cache_key}.txt"
    test_only_output=""
    cache_hit="false"
    if (( SCHEDULER_TEST_CACHE_SECONDS > 0 )) && [[ -f "${cache_file}" ]]; then
      cache_mtime="$(stat -c %Y "${cache_file}" 2>/dev/null || true)"
      now_epoch="$(date +%s)"
      if [[ -n "${cache_mtime}" ]] && (( now_epoch - cache_mtime <= SCHEDULER_TEST_CACHE_SECONDS )); then
        test_only_output="$(cat "${cache_file}")"
        cache_hit="true"
      fi
    fi
    if [[ "${cache_hit}" != "true" ]]; then
      test_only_output="$(
        srun --test-only -p "${PARTITION}" -N1 -n1 --gres=gpu:1 \
          --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY}" \
          -t "${TIME_LIMIT}" --job-name="${JOB_NAME}" ${EXCLUDE_ARG_TEXT} \
          true 2>&1 || true
      )"
      if (( SCHEDULER_TEST_CACHE_SECONDS > 0 )); then
        printf '%s\n' "${test_only_output}" >"${cache_file}"
      fi
    fi
    echo "scheduler_test_cache_hit=${cache_hit}"
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
        if (( delay_seconds > IMMEDIATE_SECONDS )); then
          echo "scheduler_within_immediate_window=false"
          scheduler_within_immediate_window="false"
        else
          echo "scheduler_within_immediate_window=true"
          scheduler_within_immediate_window="true"
        fi
        if (( delay_seconds > max_delay_seconds )); then
          echo "scheduler_within_delay_threshold=false"
          scheduler_within_delay_threshold="false"
          scheduler_launch_reason="scheduler_delay_exceeds_threshold"
        else
          echo "scheduler_within_delay_threshold=true"
          scheduler_within_delay_threshold="true"
          scheduler_launch_reason="scheduler_delay_within_threshold"
        fi
      else
        echo "scheduler_delay_seconds=unknown"
        echo "scheduler_within_immediate_window=unknown"
        echo "scheduler_within_delay_threshold=unknown"
        scheduler_launch_reason="scheduler_delay_unknown"
      fi
    else
      echo "scheduler_delay_seconds=unknown"
      echo "scheduler_within_immediate_window=unknown"
      echo "scheduler_within_delay_threshold=unknown"
      scheduler_launch_reason="scheduler_estimate_missing"
    fi
  fi
fi

launch_allowed="true"
launch_block_reasons=()
if [[ "${#pre_scheduler_block_reasons[@]}" -gt 0 ]]; then
  launch_allowed="false"
  launch_block_reasons+=("${pre_scheduler_block_reasons[@]}")
fi
if [[ "${INCLUDE_SCHEDULER_TEST}" == "true" ]]; then
  if [[ "${pre_scheduler_block_reasons_text}" == "none" ]]; then
    if [[ "${scheduler_within_immediate_window}" != "true" ]]; then
      launch_allowed="false"
      launch_block_reasons+=("scheduler_delay_exceeds_immediate_window")
    fi
    if [[ "${scheduler_within_delay_threshold}" != "true" ]]; then
      launch_allowed="false"
      launch_block_reasons+=("${scheduler_launch_reason}")
    fi
  fi
else
  if [[ "${pre_scheduler_block_reasons_text}" == "none" ]]; then
    launch_allowed="unknown"
    launch_block_reasons+=("scheduler_test_not_requested")
  fi
fi

echo "phase03_forward_backward_launch_allowed=${launch_allowed}"
if [[ "${#launch_block_reasons[@]}" -gt 0 ]]; then
  joined_reasons="$(IFS=,; echo "${launch_block_reasons[*]}")"
else
  joined_reasons="none"
fi
echo "phase03_forward_backward_launch_block_reasons=${joined_reasons}"
