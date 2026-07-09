#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
STAGE="${1:-${DATASET_STAGE:-b_dynamic_production}}"

case "${STAGE}" in
  b_dynamic_production)
    RUN_GROUP="${RUN_GROUP:-dynamic_rgb}"
    RUN_NAME="${RUN_NAME:-prod01}"
    DATASET_CLASS="B_dynamic_rgb_observation"
    TARGET_COUNT="${TARGET_COUNT:-1000}"
    COUNT_FIELD="episode_count"
    TRACE_REL="trace/demo_action_trace.json"
    EXPECT_TEACHER_EVIDENCE="false"
    ;;
  c_frozen_dp_production)
    RUN_GROUP="${RUN_GROUP:-frozen_dp_dynamic}"
    RUN_NAME="${RUN_NAME:-prod01}"
    DATASET_CLASS="C_frozen_dp_dynamic_failure"
    TARGET_COUNT="${TARGET_COUNT:-500}"
    COUNT_FIELD="rollout_count"
    TRACE_REL="trace/frozen_dp_trace.json"
    EXPECT_TEACHER_EVIDENCE="false"
    ;;
  d_future_teacher_production)
    RUN_GROUP="${RUN_GROUP:-future_teacher}"
    RUN_NAME="${RUN_NAME:-prod01}"
    DATASET_CLASS="D_future_frame_cooperation_teacher"
    TARGET_COUNT="${TARGET_COUNT:-500}"
    COUNT_FIELD="video_count"
    TRACE_REL="trace/demo_action_trace.json"
    EXPECT_TEACHER_EVIDENCE="true"
    ;;
  e_cosmos_predicted_production)
    RUN_GROUP="${RUN_GROUP:-cosmos_predicted}"
    RUN_NAME="${RUN_NAME:-prod01}"
    DATASET_CLASS="E_cosmos_predicted_cooperation"
    TARGET_COUNT="${TARGET_COUNT:-100}"
    COUNT_FIELD="rollout_count"
    TRACE_REL="trace/cosmos_predicted_trace.json"
    EXPECT_TEACHER_EVIDENCE="false"
    ;;
  *)
    echo "dataset_production_run_valid=false"
    echo "stage=${STAGE}"
    echo "reason=unknown_stage"
    exit 60
    ;;
esac

OUT_DIR="${OUT_DIR:-${ROOT}/experiments/maniskill/runs/01_dataset/${RUN_GROUP}/${RUN_NAME}}"
SUMMARY="${OUT_DIR}/summary.json"
MANIFEST="${OUT_DIR}/manifest.txt"
TRACE="${OUT_DIR}/${TRACE_REL}"
VIDEOS_DIR="${OUT_DIR}/videos"

echo "dataset_production_run_validation_ok=true"
echo "read_only=true"
echo "submits_slurm=false"
echo "stage=${STAGE}"
echo "dataset_class=${DATASET_CLASS}"
echo "run=${RUN_GROUP}/${RUN_NAME}"
echo "output_dir=${OUT_DIR}"
echo "target_count=${TARGET_COUNT}"
echo "count_field=${COUNT_FIELD}"
echo "summary=${SUMMARY}"
echo "manifest=${MANIFEST}"
echo "trace=${TRACE}"

if [[ ! -f "${SUMMARY}" && -d "${OUT_DIR}" ]]; then
  echo "shard_mode=true"
  mapfile -t shard_summaries < <(find "${OUT_DIR}" -mindepth 2 -maxdepth 2 -type f -name summary.json | sort)
  echo "shard_summary_count=${#shard_summaries[@]}"
  failures=0
  total_count=0
  total_video_count=0
  total_frame_count=0
  total_video_bytes=0
  total_video_files=0
  seen_constant_lr=false
  seen_constant_fb=false
  seen_reverse=false
  seen_move_stop=false
  seen_sine=false
  seen_continuous=false

  if [[ "${#shard_summaries[@]}" -eq 0 ]]; then
    echo "dataset_production_run_valid=false"
    echo "reason=no_single_summary_or_shard_summaries"
    exit 70
  fi

  json_number_from_summary() {
    local summary_path="$1"
    local field="$2"
    sed -nE "s/.*\"${field}\"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p" "${summary_path}" | head -n 1
  }

  for shard_summary in "${shard_summaries[@]}"; do
    shard_dir="$(dirname "${shard_summary}")"
    shard_name="$(basename "${shard_dir}")"
    shard_manifest="${shard_dir}/manifest.txt"
    shard_trace="${shard_dir}/${TRACE_REL}"
    shard_videos_dir="${shard_dir}/videos"
    echo "[shard_${shard_name}]"
    echo "  dir=${shard_dir}"
    echo "  summary=${shard_summary}"
    echo "  manifest=${shard_manifest}"
    echo "  trace=${shard_trace}"

    shard_failures=0
    for path_label in manifest trace; do
      path_var="shard_${path_label}"
      path="${!path_var}"
      if [[ -f "${path}" ]]; then
        echo "  ${path_label}_exists=true"
      else
        echo "  ${path_label}_exists=false"
        shard_failures=$((shard_failures + 1))
      fi
    done

    if [[ -d "${shard_videos_dir}" ]]; then
      shard_video_files=0
      shard_video_bytes=0
      shopt -s nullglob
      shard_video_paths=("${shard_videos_dir}"/*.mp4)
      shopt -u nullglob
      shard_video_files="${#shard_video_paths[@]}"
      for video_path in "${shard_video_paths[@]}"; do
        if [[ -f "${video_path}" ]]; then
          video_size="$(stat -c '%s' "${video_path}")"
          shard_video_bytes=$((shard_video_bytes + video_size))
        fi
      done
    else
      shard_video_files=0
      shard_video_bytes=0
      shard_failures=$((shard_failures + 1))
    fi
    echo "  video_files_count=${shard_video_files}"
    echo "  video_files_bytes=${shard_video_bytes}"

    for pattern in \
      "\"dataset_class\"[[:space:]]*:[[:space:]]*\"${DATASET_CLASS}\"" \
      '"status"[[:space:]]*:[[:space:]]*"production_complete"' \
      '"dataset_smoke_only"[[:space:]]*:[[:space:]]*false' \
      '"human_review_required"[[:space:]]*:[[:space:]]*false' \
      '"large_scale_production_allowed"[[:space:]]*:[[:space:]]*true' \
      '"method_evidence_allowed"[[:space:]]*:[[:space:]]*false' \
      "\"teacher_evidence_allowed\"[[:space:]]*:[[:space:]]*${EXPECT_TEACHER_EVIDENCE}" \
      '"positive_policy_data_allowed"[[:space:]]*:[[:space:]]*false' \
      '"state_intervention"[[:space:]]*:[[:space:]]*false' \
      '"snap_or_teleport"[[:space:]]*:[[:space:]]*false'; do
      if ! grep -qE "${pattern}" "${shard_summary}"; then
        shard_failures=$((shard_failures + 1))
      fi
    done

    if [[ -f "${shard_manifest}" ]]; then
      for expected in \
        "phase=01_dataset" \
        "dataset_class=${DATASET_CLASS}" \
        "method_evidence_allowed=false" \
        "teacher_evidence_allowed=${EXPECT_TEACHER_EVIDENCE}" \
        "forbidden_state_intervention_expected=false"; do
        if ! grep -qxF "${expected}" "${shard_manifest}"; then
          shard_failures=$((shard_failures + 1))
        fi
      done
    fi

    shard_count="$(json_number_from_summary "${shard_summary}" "${COUNT_FIELD}")"
    shard_video_count="$(json_number_from_summary "${shard_summary}" "video_count")"
    shard_frame_count="$(json_number_from_summary "${shard_summary}" "frame_count")"
    shard_summary_bytes="$(json_number_from_summary "${shard_summary}" "video_bytes")"
    shard_count="${shard_count:-0}"
    shard_video_count="${shard_video_count:-0}"
    shard_frame_count="${shard_frame_count:-0}"
    shard_summary_bytes="${shard_summary_bytes:-0}"
    echo "  summary_${COUNT_FIELD}=${shard_count}"
    echo "  summary_video_count=${shard_video_count}"
    echo "  summary_frame_count=${shard_frame_count}"
    echo "  summary_video_bytes=${shard_summary_bytes}"

    scenario="$(sed -nE 's/.*"scenario"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p' "${shard_summary}" | head -n 1)"
    echo "  scenario=${scenario}"
    case "${scenario}" in
      constant_lr) seen_constant_lr=true ;;
      constant_fb) seen_constant_fb=true ;;
      reverse) seen_reverse=true ;;
      move_stop) seen_move_stop=true ;;
      sine) seen_sine=true ;;
      continuous) seen_continuous=true ;;
    esac

    if [[ "${shard_video_files}" -lt "${shard_count}" || "${shard_video_count}" -lt "${shard_count}" || "${shard_frame_count}" -le 0 || "${shard_video_bytes}" -le 0 || "${shard_summary_bytes}" -le 0 ]]; then
      shard_failures=$((shard_failures + 1))
    fi

    echo "  shard_valid=$([[ "${shard_failures}" -eq 0 ]] && echo true || echo false)"
    echo "  shard_failure_count=${shard_failures}"
    failures=$((failures + shard_failures))
    total_count=$((total_count + shard_count))
    total_video_count=$((total_video_count + shard_video_count))
    total_frame_count=$((total_frame_count + shard_frame_count))
    total_video_bytes=$((total_video_bytes + shard_summary_bytes))
    total_video_files=$((total_video_files + shard_video_files))
  done

  echo "aggregate_${COUNT_FIELD}=${total_count}"
  echo "aggregate_video_count=${total_video_count}"
  echo "aggregate_video_files_count=${total_video_files}"
  echo "aggregate_frame_count=${total_frame_count}"
  echo "aggregate_video_bytes=${total_video_bytes}"
  echo "family_constant_lr=${seen_constant_lr}"
  echo "family_constant_fb=${seen_constant_fb}"
  echo "family_reverse=${seen_reverse}"
  echo "family_move_stop=${seen_move_stop}"
  echo "family_sine=${seen_sine}"
  echo "family_continuous=${seen_continuous}"

  if [[ "${total_count}" -ge "${TARGET_COUNT}" ]]; then
    echo "target_count_met=true"
  else
    echo "target_count_met=false"
    failures=$((failures + 1))
  fi

  if [[ "${total_video_count}" -ge "${TARGET_COUNT}" && "${total_video_files}" -ge "${TARGET_COUNT}" ]]; then
    echo "target_video_count_met=true"
  else
    echo "target_video_count_met=false"
    failures=$((failures + 1))
  fi

  if [[ "${STAGE}" != "e_cosmos_predicted_production" ]]; then
    for seen in "${seen_constant_lr}" "${seen_constant_fb}" "${seen_reverse}" "${seen_move_stop}" "${seen_sine}" "${seen_continuous}"; do
      if [[ "${seen}" != "true" ]]; then
        failures=$((failures + 1))
      fi
    done
  fi

  if [[ "${failures}" -ne 0 ]]; then
    echo "dataset_production_run_valid=false"
    echo "failure_count=${failures}"
    exit 70
  fi

  echo "dataset_production_run_valid=true"
  echo "failure_count=0"
  exit 0
fi

failures=0

require_file() {
  local label="$1"
  local path="$2"
  if [[ -f "${path}" ]]; then
    echo "${label}_exists=true"
  else
    echo "${label}_exists=false"
    failures=$((failures + 1))
  fi
}

require_summary_pattern() {
  local label="$1"
  local pattern="$2"
  if [[ -f "${SUMMARY}" ]] && grep -qE "${pattern}" "${SUMMARY}"; then
    echo "${label}=true"
  else
    echo "${label}=false"
    failures=$((failures + 1))
  fi
}

json_number_field() {
  local field="$1"
  if [[ ! -f "${SUMMARY}" ]]; then
    echo 0
    return
  fi
  sed -nE "s/.*\"${field}\"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p" "${SUMMARY}" | head -n 1
}

require_file summary "${SUMMARY}"
require_file manifest "${MANIFEST}"
require_file trace "${TRACE}"

if [[ -d "${VIDEOS_DIR}" ]]; then
  echo "videos_dir_exists=true"
  video_files_count="$(find "${VIDEOS_DIR}" -type f -name '*.mp4' | wc -l | tr -d ' ')"
  video_files_bytes="$(find "${VIDEOS_DIR}" -type f -name '*.mp4' -printf '%s\n' | awk '{s+=$1} END {print s+0}')"
else
  echo "videos_dir_exists=false"
  video_files_count=0
  video_files_bytes=0
  failures=$((failures + 1))
fi
echo "video_files_count=${video_files_count}"
echo "video_files_bytes=${video_files_bytes}"

require_summary_pattern required_dataset_class "\"dataset_class\"[[:space:]]*:[[:space:]]*\"${DATASET_CLASS}\""
require_summary_pattern required_status "\"status\"[[:space:]]*:[[:space:]]*\"production_complete\""
require_summary_pattern required_dataset_smoke_only_false "\"dataset_smoke_only\"[[:space:]]*:[[:space:]]*false"
require_summary_pattern required_human_review_required_false "\"human_review_required\"[[:space:]]*:[[:space:]]*false"
require_summary_pattern required_large_scale_production_allowed_true "\"large_scale_production_allowed\"[[:space:]]*:[[:space:]]*true"
require_summary_pattern required_method_evidence_allowed_false "\"method_evidence_allowed\"[[:space:]]*:[[:space:]]*false"
require_summary_pattern "required_teacher_evidence_allowed_${EXPECT_TEACHER_EVIDENCE}" "\"teacher_evidence_allowed\"[[:space:]]*:[[:space:]]*${EXPECT_TEACHER_EVIDENCE}"
require_summary_pattern required_positive_policy_data_allowed_false "\"positive_policy_data_allowed\"[[:space:]]*:[[:space:]]*false"
require_summary_pattern required_state_intervention_false "\"state_intervention\"[[:space:]]*:[[:space:]]*false"
require_summary_pattern required_snap_or_teleport_false "\"snap_or_teleport\"[[:space:]]*:[[:space:]]*false"

summary_count="$(json_number_field "${COUNT_FIELD}")"
summary_video_count="$(json_number_field "video_count")"
summary_frame_count="$(json_number_field "frame_count")"
summary_video_bytes="$(json_number_field "video_bytes")"
summary_count="${summary_count:-0}"
summary_video_count="${summary_video_count:-0}"
summary_frame_count="${summary_frame_count:-0}"
summary_video_bytes="${summary_video_bytes:-0}"
echo "summary_${COUNT_FIELD}=${summary_count}"
echo "summary_video_count=${summary_video_count}"
echo "summary_frame_count=${summary_frame_count}"
echo "summary_video_bytes=${summary_video_bytes}"

if [[ "${summary_count}" -ge "${TARGET_COUNT}" ]]; then
  echo "target_count_met=true"
else
  echo "target_count_met=false"
  failures=$((failures + 1))
fi

if [[ "${summary_video_count}" -ge "${TARGET_COUNT}" && "${video_files_count}" -ge "${TARGET_COUNT}" ]]; then
  echo "target_video_count_met=true"
else
  echo "target_video_count_met=false"
  failures=$((failures + 1))
fi

if [[ "${summary_frame_count}" -gt 0 && "${summary_video_bytes}" -gt 0 && "${video_files_bytes}" -gt 0 ]]; then
  echo "rgb_artifacts_nonempty=true"
else
  echo "rgb_artifacts_nonempty=false"
  failures=$((failures + 1))
fi

if [[ "${failures}" -ne 0 ]]; then
  echo "dataset_production_run_valid=false"
  echo "failure_count=${failures}"
  exit 70
fi

echo "dataset_production_run_valid=true"
echo "failure_count=0"
