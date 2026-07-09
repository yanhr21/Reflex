#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
PROD_RUN_NAME="${PROD_RUN_NAME:-prod02}"
REVIEW_NAME="${REVIEW_NAME:-bcd_${PROD_RUN_NAME}_review_$(date +%Y%m%d).md}"
STATUS_ONLY=false

usage() {
  cat <<EOF
usage: build_dataset_bcd_prod02_review_index.sh [--status-only]

Checks the B/C/D expansion shards under <class>/${PROD_RUN_NAME}/<family>.
Default mode writes one combined human-review markdown index only after every
shard is complete and passes manifest/summary quality gates. --status-only
prints the same shard table and exits nonzero if anything is incomplete.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --status-only)
      STATUS_ONLY=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown_arg=$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

review_dir="${ROOT}/experiments/maniskill/runs/01_dataset/review"
review_file="${review_dir}/${REVIEW_NAME}"
tmp_table="$(mktemp)"
tmp_failures="$(mktemp)"
trap 'rm -f "${tmp_table}" "${tmp_failures}"' EXIT

families=(lr fb reverse stop sine cont)

expected_count() {
  local class="$1"
  local family="$2"
  case "${class}:${family}" in
    dynamic_rgb:lr|dynamic_rgb:fb) echo 170 ;;
    dynamic_rgb:reverse|dynamic_rgb:stop|dynamic_rgb:sine|dynamic_rgb:cont) echo 165 ;;
    frozen_dp_dynamic:lr|frozen_dp_dynamic:fb|future_teacher:lr|future_teacher:fb) echo 84 ;;
    frozen_dp_dynamic:reverse|frozen_dp_dynamic:stop|frozen_dp_dynamic:sine|frozen_dp_dynamic:cont) echo 83 ;;
    future_teacher:reverse|future_teacher:stop|future_teacher:sine|future_teacher:cont) echo 83 ;;
    *) echo "unknown_expected_count:${class}:${family}" >&2; exit 3 ;;
  esac
}

dataset_class_for() {
  case "$1" in
    dynamic_rgb) echo "B_dynamic_rgb_observation" ;;
    frozen_dp_dynamic) echo "C_frozen_dp_dynamic_failure" ;;
    future_teacher) echo "D_future_frame_cooperation_teacher" ;;
    *) echo "unknown_dataset_class:$1" >&2; exit 4 ;;
  esac
}

class_label_for() {
  case "$1" in
    dynamic_rgb) echo "B" ;;
    frozen_dp_dynamic) echo "C" ;;
    future_teacher) echo "D" ;;
    *) echo "?" ;;
  esac
}

teacher_expected_for() {
  case "$1" in
    future_teacher) echo "true" ;;
    dynamic_rgb|frozen_dp_dynamic) echo "false" ;;
    *) echo "false" ;;
  esac
}

record_failure() {
  printf '%s\n' "$*" >> "${tmp_failures}"
}

require_summary_pattern() {
  local summary="$1"
  local label="$2"
  local pattern="$3"
  if ! grep -qE "${pattern}" "${summary}"; then
    record_failure "${label}: missing pattern ${pattern} in ${summary}"
  fi
}

printf '| Class | Family | Expected videos | Videos | Review frames | Summary | Status |\n' > "${tmp_table}"
printf '| --- | --- | ---: | ---: | ---: | --- | --- |\n' >> "${tmp_table}"

total_expected=0
total_videos=0
total_review_frames=0
complete_shards=0

for class in dynamic_rgb frozen_dp_dynamic future_teacher; do
  dataset_class="$(dataset_class_for "${class}")"
  class_label="$(class_label_for "${class}")"
  teacher_expected="$(teacher_expected_for "${class}")"
  for family in "${families[@]}"; do
    expected="$(expected_count "${class}" "${family}")"
    dir="${ROOT}/experiments/maniskill/runs/01_dataset/${class}/${PROD_RUN_NAME}/${family}"
    summary="${dir}/summary.json"
    manifest="${dir}/manifest.txt"
    videos_dir="${dir}/videos"
    review_frames_dir="${dir}/review/frames"
    videos=0
    review_frames=0
    if [[ -d "${videos_dir}" ]]; then
      videos="$(find "${videos_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
    fi
    if [[ -d "${review_frames_dir}" ]]; then
      review_frames="$(find "${review_frames_dir}" -maxdepth 1 -type f | wc -l)"
    fi
    status="incomplete"

    total_expected=$((total_expected + expected))
    total_videos=$((total_videos + videos))
    total_review_frames=$((total_review_frames + review_frames))

    if [[ ! -f "${summary}" ]]; then
      record_failure "${class}/${family}: missing summary.json"
    elif [[ ! -f "${manifest}" ]]; then
      record_failure "${class}/${family}: missing manifest.txt"
    else
      status="summary_present"
      complete_shards=$((complete_shards + 1))
      require_summary_pattern "${summary}" "${class}/${family} dataset_class" "\"dataset_class\"[[:space:]]*:[[:space:]]*\"${dataset_class}\""
      require_summary_pattern "${summary}" "${class}/${family} status" '"status"[[:space:]]*:[[:space:]]*"production_complete"'
      require_summary_pattern "${summary}" "${class}/${family} dataset_smoke_only" '"dataset_smoke_only"[[:space:]]*:[[:space:]]*false'
      require_summary_pattern "${summary}" "${class}/${family} large_scale" '"large_scale_production_allowed"[[:space:]]*:[[:space:]]*true'
      require_summary_pattern "${summary}" "${class}/${family} method_evidence" '"method_evidence_allowed"[[:space:]]*:[[:space:]]*false'
      require_summary_pattern "${summary}" "${class}/${family} teacher_evidence" "\"teacher_evidence_allowed\"[[:space:]]*:[[:space:]]*${teacher_expected}"
      require_summary_pattern "${summary}" "${class}/${family} positive_policy" '"positive_policy_data_allowed"[[:space:]]*:[[:space:]]*false'
      require_summary_pattern "${summary}" "${class}/${family} state_intervention" '"state_intervention"[[:space:]]*:[[:space:]]*false'
      require_summary_pattern "${summary}" "${class}/${family} snap_or_teleport" '"snap_or_teleport"[[:space:]]*:[[:space:]]*false'
      require_summary_pattern "${summary}" "${class}/${family} target_assisted" '"target_assisted"[[:space:]]*:[[:space:]]*false'
      require_summary_pattern "${summary}" "${class}/${family} quality_gate" '"task_motion_quality_gate_passed"[[:space:]]*:[[:space:]]*true'
      require_summary_pattern "${summary}" "${class}/${family} trigger_mode" '"motion_trigger_mode"[[:space:]]*:[[:space:]]*"pre_insert_l2"'
      require_summary_pattern "${summary}" "${class}/${family} trigger_threshold" '"motion_trigger_threshold_m"[[:space:]]*:[[:space:]]*0\.2'
      require_summary_pattern "${summary}" "${class}/${family} min_trigger_to_insert" '"min_trigger_to_insert_steps"[[:space:]]*:[[:space:]]*8'
      require_summary_pattern "${summary}" "${class}/${family} preinsert_required" '"pre_insert_motion_required"[[:space:]]*:[[:space:]]*true'
      require_summary_pattern "${summary}" "${class}/${family} video_count" "\"video_count\"[[:space:]]*:[[:space:]]*${expected}"
    fi

    if [[ "${videos}" -ne "${expected}" ]]; then
      record_failure "${class}/${family}: video count ${videos} != expected ${expected}"
    fi
    if [[ "${review_frames}" -lt $((expected * 4)) ]]; then
      record_failure "${class}/${family}: review frame count ${review_frames} < expected minimum $((expected * 4))"
    fi

    printf '| %s | %s | %s | %s | %s | `%s` | %s |\n' \
      "${class_label}" "${family}" "${expected}" "${videos}" "${review_frames}" \
      "${summary#${ROOT}/}" "${status}" >> "${tmp_table}"
  done
done

failure_count="$(wc -l < "${tmp_failures}")"

echo "bcd_prod02_review_index_check=true"
echo "prod_run_name=${PROD_RUN_NAME}"
echo "complete_shards=${complete_shards}/18"
echo "total_expected_videos=${total_expected}"
echo "total_videos=${total_videos}"
echo "total_review_frames=${total_review_frames}"
echo "failure_count=${failure_count}"

if [[ "${failure_count}" -ne 0 ]]; then
  echo "bcd_prod02_review_index_built=false"
  echo "reason=quality_or_completion_check_failed"
  sed 's/^/failure=/' "${tmp_failures}"
  if [[ "${STATUS_ONLY}" == "true" ]]; then
    cat "${tmp_table}"
  fi
  exit 10
fi

if [[ "${STATUS_ONLY}" == "true" ]]; then
  echo "bcd_prod02_review_index_built=false"
  echo "reason=status_only"
  cat "${tmp_table}"
  exit 0
fi

mkdir -p "${review_dir}"
{
  echo "# B/C/D ${PROD_RUN_NAME} Expansion Review - $(date +%Y-%m-%d)"
  echo
  echo "This is the combined human-review index for the expanded B/C/D dynamic dataset."
  echo
  echo "Quality gates checked before this file was written:"
  echo
  echo "- All 18 shards under B/C/D ${PROD_RUN_NAME} are complete."
  echo "- Summary status is \`production_complete\` for every shard."
  echo "- \`dataset_smoke_only=false\`, \`large_scale_production_allowed=true\`, and \`method_evidence_allowed=false\`."
  echo "- \`state_intervention=false\`, \`snap_or_teleport=false\`, and \`target_assisted=false\`."
  echo "- Motion uses \`motion_trigger_mode=pre_insert_l2\`, threshold \`0.20\`, and \`min_trigger_to_insert_steps=8\`."
  echo "- C success/failure is only an outcome label; C is not rejected merely because frozen DP succeeds after a legal pre-insertion disturbance."
  echo
  echo "Totals:"
  echo
  echo "- Expected videos: ${total_expected}"
  echo "- Found videos: ${total_videos}"
  echo "- Review frames: ${total_review_frames}"
  echo
  cat "${tmp_table}"
  echo
  echo "Artifact roots:"
  echo
  for class in dynamic_rgb frozen_dp_dynamic future_teacher; do
    echo
    echo "## $(class_label_for "${class}") ${class}"
    for family in "${families[@]}"; do
      dir="${ROOT}/experiments/maniskill/runs/01_dataset/${class}/${PROD_RUN_NAME}/${family}"
      echo
      echo "- ${family}"
      echo "  - Summary: \`${dir#${ROOT}/}/summary.json\`"
      echo "  - Videos: \`${dir#${ROOT}/}/videos/\`"
      echo "  - Review frames: \`${dir#${ROOT}/}/review/frames/\`"
      echo "  - Manifest: \`${dir#${ROOT}/}/manifest.txt\`"
    done
  done
} > "${review_file}"

echo "review_file=${review_file}"
echo "bcd_prod02_review_index_built=true"
