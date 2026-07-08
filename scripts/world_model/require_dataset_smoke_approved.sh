#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
RUN_GROUP="${RUN_GROUP:-static_rgb}"
RUN_NAME="${RUN_NAME:-smoke05}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/maniskill/runs/01_dataset/${RUN_GROUP}/${RUN_NAME}}"
SUMMARY="${SUMMARY:-${OUT_DIR}/summary.json}"
APPROVAL="${APPROVAL:-${OUT_DIR}/human_review_approved.txt}"

if [[ ! -f "${SUMMARY}" ]]; then
  echo "dataset_smoke_approved=false"
  echo "reason=summary_missing"
  echo "summary=${SUMMARY}"
  exit 40
fi

if ! grep -q '"human_review_required": true' "${SUMMARY}"; then
  echo "dataset_smoke_approved=false"
  echo "reason=summary_missing_human_review_required_true"
  echo "summary=${SUMMARY}"
  exit 41
fi

if ! grep -q '"large_scale_production_allowed": false' "${SUMMARY}"; then
  echo "dataset_smoke_approved=false"
  echo "reason=summary_missing_large_scale_production_guard"
  echo "summary=${SUMMARY}"
  exit 42
fi

if [[ ! -f "${APPROVAL}" ]]; then
  echo "dataset_smoke_approved=false"
  echo "reason=human_review_approval_missing"
  echo "approval=${APPROVAL}"
  exit 43
fi

if ! grep -q '^approved=true$' "${APPROVAL}"; then
  echo "dataset_smoke_approved=false"
  echo "reason=human_review_approval_not_true"
  echo "approval=${APPROVAL}"
  exit 44
fi

echo "dataset_smoke_approved=true"
echo "summary=${SUMMARY}"
echo "approval=${APPROVAL}"
