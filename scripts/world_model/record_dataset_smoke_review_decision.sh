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

decision=""
reviewer="${REVIEWER:-}"
notes="${NOTES:-}"
dry_run=false

usage() {
  cat <<EOF
usage: record_dataset_smoke_review_decision.sh --decision approved|rejected --reviewer NAME --notes TEXT [--dry-run]

This records a human visual review decision for a dataset smoke run. It must
not be run by an agent to approve a review without explicit user instruction.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --decision)
      decision="${2:-}"
      shift 2
      ;;
    --reviewer)
      reviewer="${2:-}"
      shift 2
      ;;
    --notes)
      notes="${2:-}"
      shift 2
      ;;
    --dry-run)
      dry_run=true
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

if [[ "${decision}" != "approved" && "${decision}" != "rejected" ]]; then
  echo "review_decision_recorded=false"
  echo "reason=decision_required"
  usage >&2
  exit 20
fi

if [[ -z "${reviewer}" ]]; then
  echo "review_decision_recorded=false"
  echo "reason=reviewer_required"
  exit 21
fi

if [[ -z "${notes}" ]]; then
  echo "review_decision_recorded=false"
  echo "reason=notes_required"
  exit 22
fi

if [[ ! -f "${SUMMARY}" ]]; then
  echo "review_decision_recorded=false"
  echo "reason=summary_missing"
  echo "summary=${SUMMARY}"
  exit 23
fi

if [[ ! -f "${REVIEW_MD}" ]]; then
  echo "review_decision_recorded=false"
  echo "reason=review_request_missing"
  echo "review_request=${REVIEW_MD}"
  exit 24
fi

video_count="$(find "${OUT_DIR}" -type f -name '*.mp4' | wc -l | tr -d ' ')"
frame_count="$(find "${OUT_DIR}" -type f -path '*/review/frames/*.png' | wc -l | tr -d ' ')"
if [[ "${video_count}" -eq 0 || "${frame_count}" -eq 0 ]]; then
  echo "review_decision_recorded=false"
  echo "reason=review_artifacts_missing"
  echo "video_count=${video_count}"
  echo "review_frame_count=${frame_count}"
  exit 25
fi

target="${APPROVAL}"
approved_value=true
if [[ "${decision}" == "rejected" ]]; then
  target="${REJECTION}"
  approved_value=false
fi

echo "review_decision_recorded=$([[ "${dry_run}" == "true" ]] && echo dry_run || echo true)"
echo "decision=${decision}"
echo "target=${target}"
echo "run=${RUN_GROUP}/${RUN_NAME}"
echo "reviewer=${reviewer}"
echo "video_count=${video_count}"
echo "review_frame_count=${frame_count}"
echo "must_not_auto_approve=true"

if [[ "${dry_run}" == "true" ]]; then
  exit 0
fi

{
  echo "approved=${approved_value}"
  echo "decision=${decision}"
  echo "reviewer=${reviewer}"
  echo "timestamp=$(date -Is)"
  echo "run=${RUN_GROUP}/${RUN_NAME}"
  echo "summary=${SUMMARY}"
  echo "review_request=${REVIEW_MD}"
  echo "video_count=${video_count}"
  echo "review_frame_count=${frame_count}"
  echo "notes=${notes}"
} > "${target}"

if [[ "${decision}" == "approved" ]]; then
  rm -f "${REJECTION}"
else
  rm -f "${APPROVAL}"
fi
