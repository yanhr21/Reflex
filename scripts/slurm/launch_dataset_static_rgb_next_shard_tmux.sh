#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
JOB_PREFIX="${JOB_PREFIX:-dset_rgb}"
SESSION_PREFIX="${SESSION_PREFIX:-dset_static_rgb}"
VIDEO_FPS="${VIDEO_FPS:-30}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
MEMORY="${MEMORY:-8G}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
EXCLUDE_NODES="${EXCLUDE_NODES:-server10,server28,server30,server35,server36,server39,server43,server44,server46,server56,server57,server58,server59,server63}"
NODELIST="${NODELIST:-}"
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

next_file="$(mktemp)"
"${ROOT}/scripts/world_model/dataset_static_full_next_shard.sh" >"${next_file}"
cat "${next_file}"

if grep -qx 'next_shard_needed=false' "${next_file}"; then
  rm -f "${next_file}"
  echo "launch_needed=false"
  exit 0
fi

run_name="$(awk -F= '$1=="next_run_name"{print $2; exit}' "${next_file}")"
episode_start="$(awk -F= '$1=="next_episode_start"{print $2; exit}' "${next_file}")"
count="$(awk -F= '$1=="next_count"{print $2; exit}' "${next_file}")"
rm -f "${next_file}"

if [[ -z "${run_name}" || -z "${episode_start}" || -z "${count}" ]]; then
  echo "launch_needed=false"
  echo "reason=next_shard_parse_failed"
  exit 3
fi

session="${SESSION_PREFIX}_${run_name}"
job_name="${JOB_PREFIX}_${run_name#full_}"

echo "launch_needed=true"
echo "run_name=${run_name}"
echo "episode_start=${episode_start}"
echo "count=${count}"
echo "session=${session}"
echo "job_name=${job_name}"
echo "video_fps=${VIDEO_FPS}"
echo "exclude_nodes=${EXCLUDE_NODES}"
echo "dry_run=${DRY_RUN}"

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "launched=false"
  echo "reason=dry_run"
  exit 0
fi

SESSION="${session}" \
JOB_NAME="${job_name}" \
RUN_NAME="${run_name}" \
EPISODE_START="${episode_start}" \
COUNT="${count}" \
VIDEO_FPS="${VIDEO_FPS}" \
GPUS="${GPUS}" \
CPUS_PER_TASK="${CPUS_PER_TASK}" \
MEMORY="${MEMORY}" \
TIME_LIMIT="${TIME_LIMIT}" \
EXCLUDE_NODES="${EXCLUDE_NODES}" \
NODELIST="${NODELIST}" \
"${ROOT}/scripts/slurm/launch_dataset_static_rgb_full_tmux.sh"
