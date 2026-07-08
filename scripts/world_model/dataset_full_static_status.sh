#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
RUN_GROUP="${RUN_GROUP:-static_rgb}"
RUN_NAME="${RUN_NAME:-full01}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/maniskill/runs/01_dataset/${RUN_GROUP}/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${ROOT}/logs/01_dataset/${RUN_GROUP}/${RUN_NAME}.log}"
SUMMARY="${OUT_DIR}/summary.json"
MANIFEST="${OUT_DIR}/manifest.txt"

echo "dataset_full_static_status_ok=true"
echo "run_group=${RUN_GROUP}"
echo "run_name=${RUN_NAME}"
echo "output_dir=${OUT_DIR}"
echo "log_file=${LOG_FILE}"
echo "shard_status_script=${ROOT}/scripts/world_model/dataset_static_full_shards_status.sh"

if [[ -f "${LOG_FILE}" ]]; then
  echo "log_exists=true"
  echo "log_size=$(wc -c < "${LOG_FILE}")"
else
  echo "log_exists=false"
  echo "log_size=0"
fi

if [[ -d "${OUT_DIR}" ]]; then
  echo "output_dir_exists=true"
else
  echo "output_dir_exists=false"
fi

if [[ -f "${MANIFEST}" ]]; then
  echo "manifest_exists=true"
else
  echo "manifest_exists=false"
fi

if [[ -f "${SUMMARY}" ]]; then
  echo "summary_exists=true"
else
  echo "summary_exists=false"
fi

video_count=0
video_bytes=0
frame_count=0
if [[ -d "${OUT_DIR}" ]]; then
  video_count="$(find "${OUT_DIR}" -type f -name '*.mp4' | wc -l | tr -d ' ')"
  video_bytes="$(find "${OUT_DIR}" -type f -name '*.mp4' -printf '%s\n' | awk '{s+=$1} END {print s+0}')"
  frame_count="$(find "${OUT_DIR}" -type f -path '*/review/frames/*.png' | wc -l | tr -d ' ')"
fi
echo "video_count=${video_count}"
echo "video_bytes=${video_bytes}"
echo "review_frame_count=${frame_count}"

if [[ -f "${SUMMARY}" ]] && grep -q '"dataset_smoke_only": false' "${SUMMARY}"; then
  echo "dataset_smoke_only=false"
else
  echo "dataset_smoke_only=unknown_or_true"
fi

if [[ -f "${SUMMARY}" ]] && grep -q '"human_review_required": false' "${SUMMARY}"; then
  echo "human_review_required=false"
else
  echo "human_review_required=unknown_or_true"
fi

if [[ -f "${SUMMARY}" ]] && grep -q '"large_scale_production_allowed": true' "${SUMMARY}"; then
  echo "large_scale_production_allowed=true"
else
  echo "large_scale_production_allowed=unknown_or_false"
fi

if [[ -f "${SUMMARY}" ]] && grep -qE '"status": "(render_complete|smoke_complete)"' "${SUMMARY}"; then
  echo "status=render_complete"
else
  echo "status=missing_or_incomplete"
fi

echo "[shards]"
shard_status="$(mktemp)"
if "${ROOT}/scripts/world_model/dataset_static_full_shards_status.sh" >"${shard_status}" 2>&1; then
  sed 's/^/  /' "${shard_status}"
else
  sed 's/^/  /' "${shard_status}"
fi
rm -f "${shard_status}"
