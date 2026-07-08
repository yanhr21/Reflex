#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
RUN_GROUP="${RUN_GROUP:-static_rgb}"
RUN_NAME="${RUN_NAME:-full01}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/maniskill/runs/01_dataset/${RUN_GROUP}/${RUN_NAME}}"
SUMMARY="${OUT_DIR}/summary.json"
MANIFEST="${OUT_DIR}/manifest.txt"
ALLOW_SHARDS="${ALLOW_STATIC_FULL_SHARDS:-true}"

echo "dataset_static_full_ready_check=true"
echo "run=${RUN_GROUP}/${RUN_NAME}"
echo "output_dir=${OUT_DIR}"
echo "summary=${SUMMARY}"
echo "manifest=${MANIFEST}"
echo "allow_shards=${ALLOW_SHARDS}"
echo "read_only=true"
echo "submits_slurm=false"

if [[ ! -f "${SUMMARY}" ]]; then
  if [[ "${ALLOW_SHARDS}" == "true" ]]; then
    shard_status="$(mktemp)"
    if "${ROOT}/scripts/world_model/dataset_static_full_shards_status.sh" >"${shard_status}" 2>&1; then
      sed 's/^/shards_/' "${shard_status}"
      rm -f "${shard_status}"
      echo "dataset_static_full_ready=true"
      echo "ready_source=shards"
      exit 0
    fi
    sed 's/^/shards_/' "${shard_status}"
    rm -f "${shard_status}"
  fi
  echo "dataset_static_full_ready=false"
  echo "reason=summary_missing"
  exit 70
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "dataset_static_full_ready=false"
  echo "reason=manifest_missing"
  exit 71
fi

manifest_status="$(mktemp)"
if ! "${ROOT}/scripts/world_model/validate_dataset_run_manifest.sh" "${OUT_DIR}" >"${manifest_status}" 2>&1; then
  echo "dataset_static_full_ready=false"
  echo "reason=manifest_validation_failed"
  sed 's/^/manifest_/' "${manifest_status}"
  rm -f "${manifest_status}"
  exit 72
fi
rm -f "${manifest_status}"

if ! grep -q '"dataset_smoke_only": false' "${SUMMARY}"; then
  echo "dataset_static_full_ready=false"
  echo "reason=summary_not_full_production"
  exit 73
fi

if ! grep -q '"human_review_required": false' "${SUMMARY}"; then
  echo "dataset_static_full_ready=false"
  echo "reason=summary_human_review_required_not_false"
  exit 74
fi

if ! grep -q '"large_scale_production_allowed": true' "${SUMMARY}"; then
  echo "dataset_static_full_ready=false"
  echo "reason=summary_large_scale_production_not_allowed"
  exit 75
fi

if ! grep -q '"video_fps"[[:space:]]*:[[:space:]]*30' "${SUMMARY}"; then
  echo "dataset_static_full_ready=false"
  echo "reason=summary_video_fps_not_30"
  exit 76
fi

if [[ -d "${OUT_DIR}/videos" ]]; then
  video_count="$({ find "${OUT_DIR}/videos" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null || true; } | wc -l | tr -d ' ')"
else
  video_count="$({ find "${OUT_DIR}" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null || true; } | wc -l | tr -d ' ')"
fi
if [[ -d "${OUT_DIR}/review/frames" ]]; then
  frame_count="$({ find "${OUT_DIR}/review/frames" -maxdepth 1 -type f -name '*.png' 2>/dev/null || true; } | wc -l | tr -d ' ')"
else
  frame_count=0
fi
echo "video_count=${video_count}"
echo "review_frame_count=${frame_count}"
if [[ "${video_count}" -eq 0 || "${frame_count}" -eq 0 ]]; then
  echo "dataset_static_full_ready=false"
  echo "reason=visual_artifacts_missing"
  exit 77
fi

echo "dataset_static_full_ready=true"
echo "ready_source=single_run"
