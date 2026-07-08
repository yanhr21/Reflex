#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
STAGE_FILTER="${STAGE_FILTER:-all}"
FAMILY_FILTER="${FAMILY_FILTER:-all}"

case "${STAGE_FILTER}" in B|C|D|all) ;; *) echo "invalid_stage=${STAGE_FILTER}" >&2; exit 3 ;; esac
case "${FAMILY_FILTER}" in lr|fb|reverse|stop|sine|cont|all) ;; *) echo "invalid_family=${FAMILY_FILTER}" >&2; exit 4 ;; esac

echo "dataset_bcd_production_next_shard_ok=true"
echo "read_only=true"
echo "submits_slurm=false"
echo "stage_filter=${STAGE_FILTER}"
echo "family_filter=${FAMILY_FILTER}"

next_stage=""
next_family=""
next_reason=""
blocked_stage=""
blocked_family=""
blocked_reason=""
ready_count=0
missing_count=0
incomplete_count=0

json_number_field() {
  local summary="$1"
  local field="$2"
  sed -nE "s/.*\"${field}\"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p" "${summary}" | head -n 1
}

shard_status() {
  local stage="$1"
  local family="$2"
  local out_dir="$3"
  local expected_class="$4"
  local count_field="$5"
  local target_count="$6"
  local teacher_allowed="$7"
  local summary="${out_dir}/summary.json"
  local manifest="${out_dir}/manifest.txt"
  local videos_dir="${out_dir}/videos"

  if [[ ! -d "${out_dir}" ]]; then
    echo "missing"
    return
  fi
  if [[ ! -f "${summary}" || ! -f "${manifest}" || ! -d "${videos_dir}" ]]; then
    echo "incomplete"
    return
  fi
  for pattern in \
    "\"dataset_class\"[[:space:]]*:[[:space:]]*\"${expected_class}\"" \
    '"status"[[:space:]]*:[[:space:]]*"production_complete"' \
    '"dataset_smoke_only"[[:space:]]*:[[:space:]]*false' \
    '"human_review_required"[[:space:]]*:[[:space:]]*false' \
    '"large_scale_production_allowed"[[:space:]]*:[[:space:]]*true' \
    '"method_evidence_allowed"[[:space:]]*:[[:space:]]*false' \
    "\"teacher_evidence_allowed\"[[:space:]]*:[[:space:]]*${teacher_allowed}" \
    '"positive_policy_data_allowed"[[:space:]]*:[[:space:]]*false' \
    '"state_intervention"[[:space:]]*:[[:space:]]*false' \
    '"snap_or_teleport"[[:space:]]*:[[:space:]]*false'; do
    if ! grep -qE "${pattern}" "${summary}"; then
      echo "incomplete"
      return
    fi
  done
  count="$(json_number_field "${summary}" "${count_field}")"
  video_count="$(json_number_field "${summary}" "video_count")"
  count="${count:-0}"
  video_count="${video_count:-0}"
  file_count="$(find "${videos_dir}" -type f -name '*.mp4' | wc -l | tr -d ' ')"
  if [[ "${count}" -lt "${target_count}" || "${video_count}" -lt "${target_count}" || "${file_count}" -lt "${target_count}" ]]; then
    echo "incomplete"
    return
  fi
  echo "ready"
}

consider() {
  local stage="$1"
  local family="$2"
  local run_group="$3"
  local expected_class="$4"
  local count_field="$5"
  local target_count="$6"
  local teacher_allowed="$7"
  local out_dir="${ROOT}/experiments/maniskill/runs/01_dataset/${run_group}/prod01/${family}"

  if [[ "${STAGE_FILTER}" != "all" && "${STAGE_FILTER}" != "${stage}" ]]; then
    return
  fi
  if [[ "${FAMILY_FILTER}" != "all" && "${FAMILY_FILTER}" != "${family}" ]]; then
    return
  fi

  status="$(shard_status "${stage}" "${family}" "${out_dir}" "${expected_class}" "${count_field}" "${target_count}" "${teacher_allowed}")"
  echo "[${stage}_${family}]"
  echo "  output_dir=${out_dir}"
  echo "  target_count=${target_count}"
  echo "  status=${status}"

  case "${status}" in
    ready)
      ready_count=$((ready_count + 1))
      ;;
    missing)
      missing_count=$((missing_count + 1))
      if [[ -z "${next_stage}" ]]; then
        next_stage="${stage}"
        next_family="${family}"
        next_reason="missing"
      fi
      ;;
    incomplete)
      incomplete_count=$((incomplete_count + 1))
      if [[ -z "${blocked_stage}" ]]; then
        blocked_stage="${stage}"
        blocked_family="${family}"
        blocked_reason="incomplete_existing_output"
      fi
      ;;
  esac
}

consider B lr dynamic_rgb B_dynamic_rgb_observation episode_count 170 false
consider B fb dynamic_rgb B_dynamic_rgb_observation episode_count 170 false
consider B reverse dynamic_rgb B_dynamic_rgb_observation episode_count 165 false
consider B stop dynamic_rgb B_dynamic_rgb_observation episode_count 165 false
consider B sine dynamic_rgb B_dynamic_rgb_observation episode_count 165 false
consider B cont dynamic_rgb B_dynamic_rgb_observation episode_count 165 false
consider C lr frozen_dp_dynamic C_frozen_dp_dynamic_failure rollout_count 84 false
consider C fb frozen_dp_dynamic C_frozen_dp_dynamic_failure rollout_count 84 false
consider C reverse frozen_dp_dynamic C_frozen_dp_dynamic_failure rollout_count 83 false
consider C stop frozen_dp_dynamic C_frozen_dp_dynamic_failure rollout_count 83 false
consider C sine frozen_dp_dynamic C_frozen_dp_dynamic_failure rollout_count 83 false
consider C cont frozen_dp_dynamic C_frozen_dp_dynamic_failure rollout_count 83 false
consider D lr future_teacher D_future_frame_cooperation_teacher rollout_count 84 true
consider D fb future_teacher D_future_frame_cooperation_teacher rollout_count 84 true
consider D reverse future_teacher D_future_frame_cooperation_teacher rollout_count 83 true
consider D stop future_teacher D_future_frame_cooperation_teacher rollout_count 83 true
consider D sine future_teacher D_future_frame_cooperation_teacher rollout_count 83 true
consider D cont future_teacher D_future_frame_cooperation_teacher rollout_count 83 true

echo "[summary]"
echo "  ready_count=${ready_count}"
echo "  missing_count=${missing_count}"
echo "  incomplete_count=${incomplete_count}"
if [[ -n "${blocked_stage}" ]]; then
  echo "  next_shard_available=false"
  echo "  blocked_stage=${blocked_stage}"
  echo "  blocked_family=${blocked_family}"
  echo "  reason=${blocked_reason}"
  exit 70
fi
if [[ -n "${next_stage}" ]]; then
  echo "  next_shard_available=true"
  echo "  next_stage=${next_stage}"
  echo "  next_family=${next_family}"
  echo "  reason=${next_reason}"
  echo "  launch_command=scripts/slurm/launch_dataset_bcd_production_shards_tmux.sh --execute --stage ${next_stage} --family ${next_family} --max-launches 1"
  exit 0
fi
echo "  next_shard_available=false"
echo "  reason=all_matching_shards_ready"
