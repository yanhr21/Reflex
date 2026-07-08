#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
REVIEW_MD="${ROOT}/experiments/maniskill/runs/01_dataset/review/bcd_smoke_review_20260708.md"

echo "dataset_bcd_review_block_status_ok=true"
echo "read_only=true"
echo "submits_slurm=false"
echo "review_request=${REVIEW_MD}"
echo "review_request_exists=$([[ -f "${REVIEW_MD}" ]] && echo true || echo false)"
echo "approval_helper=${ROOT}/scripts/world_model/record_dataset_bcd_smoke_review_decision.sh"
echo "next_shard_helper=${ROOT}/scripts/world_model/dataset_bcd_production_next_shard.sh"
echo "production_launcher=${ROOT}/scripts/slurm/launch_dataset_bcd_production_shards_tmux.sh"

all_approved=true
any_rejected=false
missing_artifacts=0

report_one() {
  local label="$1"
  local run_group="$2"
  local run_name="$3"
  local out_dir="${ROOT}/experiments/maniskill/runs/01_dataset/${run_group}/${run_name}"
  local summary="${out_dir}/summary.json"
  local review_request="${out_dir}/review_request.md"
  local approval="${out_dir}/human_review_approved.txt"
  local rejection="${out_dir}/human_review_rejected.txt"
  local video_count=0
  local frame_count=0

  if [[ -d "${out_dir}" ]]; then
    video_count="$(find "${out_dir}" -type f -name '*.mp4' | wc -l | tr -d ' ')"
    frame_count="$(find "${out_dir}" -type f -path '*/review/frames/*.png' | wc -l | tr -d ' ')"
  fi

  echo "[${label}]"
  echo "  run=${run_group}/${run_name}"
  echo "  output_dir=${out_dir}"
  echo "  summary_exists=$([[ -f "${summary}" ]] && echo true || echo false)"
  echo "  review_request_exists=$([[ -f "${review_request}" ]] && echo true || echo false)"
  echo "  video_count=${video_count}"
  echo "  review_frame_count=${frame_count}"
  echo "  approval=${approval}"
  echo "  rejection=${rejection}"

  if [[ ! -f "${summary}" || ! -f "${review_request}" || "${video_count}" -eq 0 || "${frame_count}" -eq 0 ]]; then
    echo "  review_artifacts_ready=false"
    missing_artifacts=$((missing_artifacts + 1))
  else
    echo "  review_artifacts_ready=true"
  fi

  if [[ -f "${rejection}" ]]; then
    echo "  rejected=true"
    any_rejected=true
    all_approved=false
  else
    echo "  rejected=false"
  fi

  if [[ -f "${approval}" ]] && grep -qxF "approved=true" "${approval}"; then
    echo "  approved=true"
    echo "  blocked_on_human_review=false"
  else
    echo "  approved=false"
    echo "  blocked_on_human_review=true"
    all_approved=false
  fi
}

report_one B dynamic_rgb smoke01
report_one C frozen_dp_dynamic smoke01
report_one D future_teacher smoke01

echo "[summary]"
echo "  missing_artifact_classes=${missing_artifacts}"
echo "  any_rejected=${any_rejected}"
echo "  all_approved=${all_approved}"
if [[ "${missing_artifacts}" -ne 0 ]]; then
  echo "  goal_blocked=false"
  echo "  reason=review_artifacts_missing"
  exit 70
fi
if [[ "${any_rejected}" == "true" ]]; then
  echo "  goal_blocked=true"
  echo "  reason=human_review_rejected"
  exit 71
fi
if [[ "${all_approved}" != "true" ]]; then
  echo "  goal_blocked=true"
  echo "  reason=human_review_approval_missing"
  echo "  approve_all_after_user_review=scripts/world_model/record_dataset_bcd_smoke_review_decision.sh --decision approved --reviewer <name> --notes <text>"
  echo "  reject_all_after_user_review=scripts/world_model/record_dataset_bcd_smoke_review_decision.sh --decision rejected --reviewer <name> --notes <text>"
  exit 0
fi

echo "  goal_blocked=false"
echo "  reason=all_bcd_smokes_approved"
echo "  next_command=scripts/world_model/dataset_bcd_production_next_shard.sh"
