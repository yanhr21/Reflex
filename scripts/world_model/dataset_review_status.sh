#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
RUN_GROUP="${RUN_GROUP:-static_rgb}"
RUN_NAME="${RUN_NAME:-smoke05}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/maniskill/runs/01_dataset/${RUN_GROUP}/${RUN_NAME}}"
SUMMARY="${SUMMARY:-${OUT_DIR}/summary.json}"
REVIEW_MD="${REVIEW_MD:-${OUT_DIR}/review_request.md}"
APPROVAL="${APPROVAL:-${OUT_DIR}/human_review_approved.txt}"
REJECTION="${REJECTION:-${OUT_DIR}/human_review_rejected.txt}"

echo "dataset_review_status_ok=true"
echo "run=${RUN_GROUP}/${RUN_NAME}"
echo "output_dir=${OUT_DIR}"
echo "summary=${SUMMARY}"
echo "review_request=${REVIEW_MD}"
echo "approval=${APPROVAL}"
echo "rejection=${REJECTION}"

if [[ -f "${SUMMARY}" ]]; then
  echo "summary_exists=true"
else
  echo "summary_exists=false"
fi

if [[ -f "${REVIEW_MD}" ]]; then
  echo "review_request_exists=true"
else
  echo "review_request_exists=false"
fi

video_count=0
frame_count=0
if [[ -d "${OUT_DIR}" ]]; then
  video_count="$(find "${OUT_DIR}" -type f -name '*.mp4' | wc -l | tr -d ' ')"
  frame_count="$(find "${OUT_DIR}" -type f -path '*/review/frames/*.png' | wc -l | tr -d ' ')"
fi
echo "video_count=${video_count}"
echo "review_frame_count=${frame_count}"

if [[ -f "${APPROVAL}" ]]; then
  echo "approval_exists=true"
  if grep -q '^approved=true$' "${APPROVAL}"; then
    echo "approved=true"
    echo "goal_blocked_on_human_review=false"
  else
    echo "approved=false"
    echo "goal_blocked_on_human_review=true"
    echo "reason=approval_file_missing_approved_true"
  fi
else
  echo "approval_exists=false"
  echo "approved=false"
  echo "goal_blocked_on_human_review=true"
  if [[ -f "${REJECTION}" ]]; then
    echo "rejection_exists=true"
    echo "reason=human_review_rejected"
  else
    echo "rejection_exists=false"
    echo "reason=human_review_approval_missing"
  fi
fi

echo "next_if_approved=launch_dataset_static_rgb_full_tmux_after_gate_then_B_smoke_runner_when_implemented"
echo "must_not_auto_approve=true"
