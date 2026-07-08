#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
RUN_GROUP="${RUN_GROUP:-static_rgb}"
RUN_NAME="${RUN_NAME:-smoke05}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/maniskill/runs/01_dataset/${RUN_GROUP}/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${ROOT}/logs/01_dataset/${RUN_GROUP}/${RUN_NAME}.log}"
SESSION="${SESSION:-dset_static_rgb_smoke01}"
JOB_NAME="${JOB_NAME:-dset_rgb_smoke}"

manifest="${OUT_DIR}/manifest.txt"
summary="${OUT_DIR}/summary.json"

echo "dataset_smoke_status_ok=true"
echo "run_group=${RUN_GROUP}"
echo "run_name=${RUN_NAME}"
echo "output_dir=${OUT_DIR}"
echo "log_file=${LOG_FILE}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux_session_alive=true"
else
  echo "tmux_session_alive=false"
fi

if command -v squeue >/dev/null 2>&1; then
  matching_jobs="$(squeue -u "${USER}" -h -o '%i %j %T %R' 2>/dev/null | awk -v name="${JOB_NAME}" '$2 == name {print}' || true)"
  if [[ -n "${matching_jobs}" ]]; then
    echo "slurm_job_present=true"
    echo "slurm_jobs<<EOF"
    echo "${matching_jobs}"
    echo "EOF"
  else
    echo "slurm_job_present=false"
  fi
else
  echo "slurm_job_present=unknown"
  echo "reason=squeue_missing"
fi

if [[ -f "${LOG_FILE}" ]]; then
  echo "log_exists=true"
  echo "log_size=$(wc -c < "${LOG_FILE}")"
  echo "allocation_failed_count=$(grep -c '^allocation_failed=' "${LOG_FILE}" || true)"
  if grep -q '^render_started=false' "${LOG_FILE}"; then
    echo "last_render_started=false"
  else
    echo "last_render_started=unknown_or_true"
  fi
else
  echo "log_exists=false"
  echo "log_size=0"
  echo "allocation_failed_count=0"
  echo "last_render_started=unknown"
fi

if [[ -d "${OUT_DIR}" ]]; then
  echo "output_dir_exists=true"
else
  echo "output_dir_exists=false"
fi

if [[ -f "${manifest}" ]]; then
  echo "manifest_exists=true"
else
  echo "manifest_exists=false"
fi

if [[ -f "${summary}" ]]; then
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

if [[ "${video_count}" -gt 0 && "${video_bytes}" -gt 0 ]]; then
  echo "smoke_render_artifacts_present=true"
else
  echo "smoke_render_artifacts_present=false"
fi

if [[ -f "${summary}" ]] && grep -q '"human_review_required": true' "${summary}"; then
  echo "human_review_required=true"
else
  echo "human_review_required=unknown_or_false"
fi

if [[ -f "${summary}" ]] && grep -q '"large_scale_production_allowed": false' "${summary}"; then
  echo "large_scale_production_allowed=false"
else
  echo "large_scale_production_allowed=unknown_or_true"
fi
