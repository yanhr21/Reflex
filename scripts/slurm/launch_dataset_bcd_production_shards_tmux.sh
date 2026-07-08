#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
EXECUTE=false
STAGE_FILTER="${STAGE_FILTER:-all}"
FAMILY_FILTER="${FAMILY_FILTER:-all}"
MAX_LAUNCHES="${MAX_LAUNCHES:-1}"

usage() {
  cat <<EOF
usage: launch_dataset_bcd_production_shards_tmux.sh [--execute] [--stage B|C|D|all] [--family lr|fb|reverse|stop|sine|cont|all] [--max-launches N]

Dry-runs the B/C/D production shard launch plan by default. Use --execute only
after B/C/D smoke videos are explicitly approved by the user and the class
approval files exist. Execution defaults to one shard so resource use stays
auditable; increase --max-launches only when intentionally queueing more.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --execute)
      EXECUTE=true
      shift
      ;;
    --stage)
      STAGE_FILTER="${2:-}"
      shift 2
      ;;
    --family)
      FAMILY_FILTER="${2:-}"
      shift 2
      ;;
    --max-launches)
      MAX_LAUNCHES="${2:-}"
      shift 2
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

case "${STAGE_FILTER}" in B|C|D|all) ;; *) echo "invalid_stage=${STAGE_FILTER}" >&2; exit 3 ;; esac
case "${FAMILY_FILTER}" in lr|fb|reverse|stop|sine|cont|all) ;; *) echo "invalid_family=${FAMILY_FILTER}" >&2; exit 4 ;; esac
if ! [[ "${MAX_LAUNCHES}" =~ ^[0-9]+$ ]] || [[ "${MAX_LAUNCHES}" -lt 1 ]]; then
  echo "invalid_max_launches=${MAX_LAUNCHES}" >&2
  exit 5
fi

echo "launch_bcd_production_shards=true"
echo "execute=${EXECUTE}"
echo "stage_filter=${STAGE_FILTER}"
echo "family_filter=${FAMILY_FILTER}"
echo "max_launches=${MAX_LAUNCHES}"
echo "requires_explicit_user_approval=true"

if [[ "${EXECUTE}" != "true" ]]; then
  "${ROOT}/scripts/world_model/dataset_bcd_production_shard_plan.sh"
  echo "dry_run_only=true"
  echo "no_slurm_submitted=true"
  exit 0
fi

for stage in b_dynamic_smoke c_frozen_dp_smoke d_future_teacher_smoke; do
  "${ROOT}/scripts/world_model/require_dataset_class_smoke_approved.sh" "${stage}"
done

launched=0
skipped=0
ready_skipped=0

json_number_field() {
  local summary="$1"
  local field="$2"
  sed -nE "s/.*\"${field}\"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p" "${summary}" | head -n 1
}

shard_status() {
  local out_dir="$1"
  local expected_class="$2"
  local count_field="$3"
  local target_count="$4"
  local teacher_allowed="$5"
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

launch_one() {
  local stage="$1"
  local family="$2"
  local launcher="$3"
  local session="$4"
  local run_group="$5"
  local run_name="$6"
  local scenario="$7"
  local count="$8"
  local dx="$9"
  local dy="${10}"
  local dz="${11}"
  local expected_class="${12}"
  local count_field="${13}"
  local teacher_allowed="${14}"

  if [[ "${STAGE_FILTER}" != "all" && "${STAGE_FILTER}" != "${stage}" ]]; then
    skipped=$((skipped + 1))
    return
  fi
  if [[ "${FAMILY_FILTER}" != "all" && "${FAMILY_FILTER}" != "${family}" ]]; then
    skipped=$((skipped + 1))
    return
  fi
  if [[ "${launched}" -ge "${MAX_LAUNCHES}" ]]; then
    skipped=$((skipped + 1))
    return
  fi

  out_dir="${ROOT}/experiments/maniskill/runs/01_dataset/${run_group}/${run_name}"
  status="$(shard_status "${out_dir}" "${expected_class}" "${count_field}" "${count}" "${teacher_allowed}")"
  if [[ "${status}" == "ready" ]]; then
    echo "[skip_ready_${stage}_${family}]"
    echo "output_dir=${out_dir}"
    ready_skipped=$((ready_skipped + 1))
    return
  fi
  if [[ "${status}" == "incomplete" ]]; then
    echo "refusing_incomplete_existing_shard=true" >&2
    echo "stage=${stage}" >&2
    echo "family=${family}" >&2
    echo "output_dir=${out_dir}" >&2
    echo "reason=archive_or_diagnose_incomplete_shard_before_relaunch" >&2
    exit 7
  fi

  echo "[launch_${stage}_${family}]"
  SESSION="${session}" RUN_GROUP="${run_group}" RUN_NAME="${run_name}" \
    SCENARIO="${scenario}" COUNT="${count}" \
    DELTA_X="${dx}" DELTA_Y="${dy}" DELTA_Z="${dz}" FPS=30 \
    "${ROOT}/${launcher}"
  launched=$((launched + 1))
}

launch_one B lr scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh dset_b_lr dynamic_rgb/prod01 lr constant_lr 170 0.0 0.08 0.0 B_dynamic_rgb_observation episode_count false
launch_one B fb scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh dset_b_fb dynamic_rgb/prod01 fb constant_fb 170 0.08 0.0 0.0 B_dynamic_rgb_observation episode_count false
launch_one B reverse scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh dset_b_reverse dynamic_rgb/prod01 reverse reverse 165 0.0 0.08 0.0 B_dynamic_rgb_observation episode_count false
launch_one B stop scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh dset_b_stop dynamic_rgb/prod01 stop move_stop 165 0.0 0.08 0.0 B_dynamic_rgb_observation episode_count false
launch_one B sine scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh dset_b_sine dynamic_rgb/prod01 sine sine 165 0.0 0.08 0.0 B_dynamic_rgb_observation episode_count false
launch_one B cont scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh dset_b_cont dynamic_rgb/prod01 cont continuous 165 0.0 0.10 0.0 B_dynamic_rgb_observation episode_count false

launch_one C lr scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh dset_c_lr frozen_dp_dynamic/prod01 lr constant_lr 84 0.0 0.08 0.0 C_frozen_dp_dynamic_failure rollout_count false
launch_one C fb scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh dset_c_fb frozen_dp_dynamic/prod01 fb constant_fb 84 0.08 0.0 0.0 C_frozen_dp_dynamic_failure rollout_count false
launch_one C reverse scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh dset_c_reverse frozen_dp_dynamic/prod01 reverse reverse 83 0.0 0.08 0.0 C_frozen_dp_dynamic_failure rollout_count false
launch_one C stop scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh dset_c_stop frozen_dp_dynamic/prod01 stop move_stop 83 0.0 0.08 0.0 C_frozen_dp_dynamic_failure rollout_count false
launch_one C sine scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh dset_c_sine frozen_dp_dynamic/prod01 sine sine 83 0.0 0.08 0.0 C_frozen_dp_dynamic_failure rollout_count false
launch_one C cont scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh dset_c_cont frozen_dp_dynamic/prod01 cont continuous 83 0.0 0.10 0.0 C_frozen_dp_dynamic_failure rollout_count false

launch_one D lr scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh dset_d_lr future_teacher/prod01 lr constant_lr 84 0.0 0.08 0.0 D_future_frame_cooperation_teacher rollout_count true
launch_one D fb scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh dset_d_fb future_teacher/prod01 fb constant_fb 84 0.08 0.0 0.0 D_future_frame_cooperation_teacher rollout_count true
launch_one D reverse scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh dset_d_reverse future_teacher/prod01 reverse reverse 83 0.0 0.08 0.0 D_future_frame_cooperation_teacher rollout_count true
launch_one D stop scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh dset_d_stop future_teacher/prod01 stop move_stop 83 0.0 0.08 0.0 D_future_frame_cooperation_teacher rollout_count true
launch_one D sine scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh dset_d_sine future_teacher/prod01 sine sine 83 0.0 0.08 0.0 D_future_frame_cooperation_teacher rollout_count true
launch_one D cont scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh dset_d_cont future_teacher/prod01 cont continuous 83 0.0 0.10 0.0 D_future_frame_cooperation_teacher rollout_count true

echo "launched_count=${launched}"
echo "skipped_count=${skipped}"
echo "ready_skipped_count=${ready_skipped}"
if [[ "${launched}" -eq 0 ]]; then
  echo "no_matching_shard_launched=true"
  exit 6
fi
