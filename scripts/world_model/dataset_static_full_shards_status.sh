#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
RUN_GROUP="${RUN_GROUP:-static_rgb}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/experiments/maniskill/runs/01_dataset/${RUN_GROUP}}"
SHARD_GLOB="${STATIC_FULL_SHARD_GLOB:-full_s*}"
TARGET_COUNT="${STATIC_FULL_TARGET_COUNT:-1000}"

echo "dataset_static_full_shards_status_ok=true"
echo "run_group=${RUN_GROUP}"
echo "run_root=${RUN_ROOT}"
echo "shard_glob=${SHARD_GLOB}"
echo "target_count=${TARGET_COUNT}"
echo "read_only=true"
echo "submits_slurm=false"

shopt -s nullglob
shards=("${RUN_ROOT}"/${SHARD_GLOB})
shopt -u nullglob

echo "shard_candidate_count=${#shards[@]}"

ready_shards=0
invalid_shards=0
total_count=0
total_video_count=0
total_frame_count=0
total_video_bytes=0

for shard_dir in "${shards[@]}"; do
  [[ -d "${shard_dir}" ]] || continue
  shard_name="$(basename "${shard_dir}")"
  summary="${shard_dir}/summary.json"
  manifest="${shard_dir}/manifest.txt"
  echo "[${shard_name}]"
  echo "  dir=${shard_dir}"
  if [[ ! -f "${summary}" ]]; then
    echo "  ready=false"
    echo "  reason=summary_missing"
    invalid_shards=$((invalid_shards + 1))
    continue
  fi
  if [[ ! -f "${manifest}" ]]; then
    echo "  ready=false"
    echo "  reason=manifest_missing"
    invalid_shards=$((invalid_shards + 1))
    continue
  fi
  validation_file="$(mktemp)"
  if ! "${ROOT}/scripts/world_model/validate_dataset_run_manifest.sh" "${shard_dir}" >"${validation_file}" 2>&1; then
    echo "  ready=false"
    echo "  reason=manifest_validation_failed"
    sed 's/^/  manifest_/' "${validation_file}"
    rm -f "${validation_file}"
    invalid_shards=$((invalid_shards + 1))
    continue
  fi
  rm -f "${validation_file}"

  shard_count="$(sed -n 's/.*"count"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "${summary}" | head -n 1)"
  shard_count="${shard_count:-0}"
  video_fps="$(sed -n 's/.*"video_fps"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "${summary}" | head -n 1)"
  video_fps="${video_fps:-unknown}"
  if [[ -d "${shard_dir}/videos" ]]; then
    video_dir="${shard_dir}/videos"
    video_count="$({ find "${video_dir}" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null || true; } | wc -l | tr -d ' ')"
    video_bytes="$({ find "${video_dir}" -maxdepth 1 -type f -name '*.mp4' -printf '%s\n' 2>/dev/null || true; } | awk '{s+=$1} END {print s+0}')"
  else
    video_count="$({ find "${shard_dir}" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null || true; } | wc -l | tr -d ' ')"
    video_bytes="$({ find "${shard_dir}" -maxdepth 1 -type f -name '*.mp4' -printf '%s\n' 2>/dev/null || true; } | awk '{s+=$1} END {print s+0}')"
  fi
  if [[ -d "${shard_dir}/review/frames" ]]; then
    frame_count="$({ find "${shard_dir}/review/frames" -maxdepth 1 -type f -name '*.png' 2>/dev/null || true; } | wc -l | tr -d ' ')"
  else
    frame_count=0
  fi
  echo "  count=${shard_count}"
  echo "  video_fps=${video_fps}"
  echo "  video_count=${video_count}"
  echo "  review_frame_count=${frame_count}"
  echo "  video_bytes=${video_bytes}"

  if ! grep -q '"dataset_smoke_only": false' "${summary}"; then
    echo "  ready=false"
    echo "  reason=summary_not_full_production"
    invalid_shards=$((invalid_shards + 1))
    continue
  fi
  if ! grep -q '"human_review_required": false' "${summary}"; then
    echo "  ready=false"
    echo "  reason=summary_human_review_required_not_false"
    invalid_shards=$((invalid_shards + 1))
    continue
  fi
  if ! grep -q '"large_scale_production_allowed": true' "${summary}"; then
    echo "  ready=false"
    echo "  reason=summary_large_scale_production_not_allowed"
    invalid_shards=$((invalid_shards + 1))
    continue
  fi
  if [[ "${video_count}" -eq 0 || "${frame_count}" -eq 0 ]]; then
    echo "  ready=false"
    echo "  reason=visual_artifacts_missing"
    invalid_shards=$((invalid_shards + 1))
    continue
  fi
  if [[ "${video_fps}" != "30" ]]; then
    echo "  ready=false"
    echo "  reason=video_fps_not_30"
    invalid_shards=$((invalid_shards + 1))
    continue
  fi

  echo "  ready=true"
  ready_shards=$((ready_shards + 1))
  total_count=$((total_count + shard_count))
  total_video_count=$((total_video_count + video_count))
  total_frame_count=$((total_frame_count + frame_count))
  total_video_bytes=$((total_video_bytes + video_bytes))
done

echo "ready_shard_count=${ready_shards}"
echo "invalid_shard_count=${invalid_shards}"
echo "total_count=${total_count}"
echo "total_video_count=${total_video_count}"
echo "total_review_frame_count=${total_frame_count}"
echo "total_video_bytes=${total_video_bytes}"

if [[ "${ready_shards}" -gt 0 && "${invalid_shards}" -eq 0 && "${total_count}" -ge "${TARGET_COUNT}" ]]; then
  echo "dataset_static_full_shards_ready=true"
  exit 0
fi

echo "dataset_static_full_shards_ready=false"
if [[ "${ready_shards}" -eq 0 ]]; then
  echo "reason=no_ready_shards"
elif [[ "${invalid_shards}" -ne 0 ]]; then
  echo "reason=invalid_shards_present"
else
  echo "reason=target_count_not_met"
fi
exit 80
