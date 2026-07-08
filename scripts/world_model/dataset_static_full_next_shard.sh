#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
RUN_GROUP="${RUN_GROUP:-static_rgb}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/experiments/maniskill/runs/01_dataset/${RUN_GROUP}}"
SHARD_SIZE="${STATIC_FULL_SHARD_SIZE:-50}"
TARGET_COUNT="${STATIC_FULL_TARGET_COUNT:-1000}"

echo "dataset_static_full_next_shard_ok=true"
echo "run_group=${RUN_GROUP}"
echo "run_root=${RUN_ROOT}"
echo "shard_size=${SHARD_SIZE}"
echo "target_count=${TARGET_COUNT}"
echo "read_only=true"
echo "submits_slurm=false"

completed_count=0
ready_shards=0
shopt -s nullglob
for summary in "${RUN_ROOT}"/full_s*/summary.json; do
  run_dir="$(dirname "${summary}")"
  status_file="$(mktemp)"
  if "${ROOT}/scripts/world_model/validate_dataset_run_manifest.sh" "${run_dir}" >"${status_file}" 2>&1 && \
    grep -q '"dataset_smoke_only": false' "${summary}" && \
    grep -q '"human_review_required": false' "${summary}" && \
    grep -q '"large_scale_production_allowed": true' "${summary}" && \
    grep -q '"video_fps"[[:space:]]*:[[:space:]]*30' "${summary}"; then
    count="$(sed -n 's/.*"count"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "${summary}" | head -n 1)"
    count="${count:-0}"
    completed_count=$((completed_count + count))
    ready_shards=$((ready_shards + 1))
  fi
  rm -f "${status_file}"
done
shopt -u nullglob

next_start="${completed_count}"
remaining=$((TARGET_COUNT - completed_count))
if [[ "${remaining}" -lt 0 ]]; then
  remaining=0
fi
next_count="${SHARD_SIZE}"
if [[ "${remaining}" -lt "${SHARD_SIZE}" ]]; then
  next_count="${remaining}"
fi

pair_index=$((next_start / 100))
within_pair=$(((next_start % 100) / SHARD_SIZE))
suffix="a"
if [[ "${within_pair}" -eq 1 ]]; then
  suffix="b"
fi
run_name="$(printf 'full_s%02d%s' "${pair_index}" "${suffix}")"

echo "ready_shard_count=${ready_shards}"
echo "completed_count=${completed_count}"
echo "remaining_count=${remaining}"
if [[ "${remaining}" -eq 0 ]]; then
  echo "next_shard_needed=false"
  echo "dataset_static_full_complete=true"
  exit 0
fi

echo "next_shard_needed=true"
echo "next_run_name=${run_name}"
echo "next_episode_start=${next_start}"
echo "next_count=${next_count}"
echo "dataset_static_full_complete=false"
