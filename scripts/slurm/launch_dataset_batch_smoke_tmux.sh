#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
DRY_RUN="${DRY_RUN:-false}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=true ;;
    *)
      echo "unknown_arg=${arg}" >&2
      echo "usage=$0 [--dry-run]" >&2
      exit 2
      ;;
  esac
done

echo "dataset_batch_smoke_launch=true"
echo "root=${ROOT}"
echo "policy=batch_smoke_then_combined_human_review"
echo "fps=${FPS:-30}"
echo "gpus_per_job=${GPUS:-1}"
echo "dry_run=${DRY_RUN}"
echo "readiness_requires_a_static_full=true"

stages=(
  b_dynamic_smoke
  c_frozen_dp_smoke
  d_future_teacher_smoke
  e_cosmos_predicted_smoke
)

launcher_for_stage() {
  case "$1" in
    b_dynamic_smoke) echo "${ROOT}/scripts/slurm/launch_dataset_dynamic_rgb_smoke_tmux.sh" ;;
    c_frozen_dp_smoke) echo "${ROOT}/scripts/slurm/launch_dataset_frozen_dp_dynamic_smoke_tmux.sh" ;;
    d_future_teacher_smoke) echo "${ROOT}/scripts/slurm/launch_dataset_future_frame_teacher_smoke_tmux.sh" ;;
    e_cosmos_predicted_smoke) echo "${ROOT}/scripts/slurm/launch_dataset_cosmos_predicted_coop_smoke_tmux.sh" ;;
    *) return 1 ;;
  esac
}

launched=0
skipped=0

for stage in "${stages[@]}"; do
  echo "[${stage}]"
  status_file="$(mktemp)"
  if "${ROOT}/scripts/world_model/require_dataset_stage_ready.sh" "${stage}" >"${status_file}" 2>&1; then
    echo "  ready=true"
  else
    echo "  ready=false"
    sed 's/^/  /' "${status_file}"
    rm -f "${status_file}"
    skipped=$((skipped + 1))
    continue
  fi
  rm -f "${status_file}"

  launcher="$(launcher_for_stage "${stage}")"
  echo "  launcher=${launcher}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "  launched=false"
    echo "  reason=dry_run"
    skipped=$((skipped + 1))
    continue
  fi
  if [[ ! -x "${launcher}" ]]; then
    echo "  launched=false"
    echo "  reason=launcher_missing"
    skipped=$((skipped + 1))
    continue
  fi

  if FPS="${FPS:-30}" GPUS="${GPUS:-1}" CPUS_PER_TASK="${CPUS_PER_TASK:-1}" \
    MEMORY="${MEMORY:-8G}" TIME_LIMIT="${TIME_LIMIT:-00:30:00}" \
    EXCLUDE_NODES="${EXCLUDE_NODES:-server36,server39,server43,server44,server46,server58,server63}" \
    "${launcher}"; then
    echo "  launched=true"
    launched=$((launched + 1))
  else
    echo "  launched=false"
    echo "  reason=launcher_failed"
    skipped=$((skipped + 1))
  fi
done

echo "batch_smoke_launched_count=${launched}"
echo "batch_smoke_skipped_count=${skipped}"
