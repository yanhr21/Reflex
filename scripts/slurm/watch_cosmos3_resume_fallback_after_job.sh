#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
PRIMARY_JOB_ID="${PRIMARY_JOB_ID:?set PRIMARY_JOB_ID to the active training Slurm job id}"
FALLBACK_JOB_ID="${FALLBACK_JOB_ID:?set FALLBACK_JOB_ID to the held fallback allocation job id}"
SFT_ROOT="${SFT_ROOT:?set SFT_ROOT to the active SFT output root}"
CONDITION_ROOT="${CONDITION_ROOT:?set CONDITION_ROOT to the active condition root}"
JOB_NAME="${JOB_NAME:-vision_sft_droid_policy_full1000_rgb_300step_wam}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SFT_ROOT}/outputs/cosmos3/sft/${JOB_NAME}/checkpoints}"
TARGET_ITER="${TARGET_ITER:-1500}"
POLL_SECONDS="${POLL_SECONDS:-120}"
POST_STOP_STABLE_SECONDS="${POST_STOP_STABLE_SECONDS:-60}"
FALLBACK_NPROC="${FALLBACK_NPROC:-2}"
FALLBACK_CPUS="${FALLBACK_CPUS:-32}"
FALLBACK_MEM="${FALLBACK_MEM:-240G}"
MASTER_PORT="${MASTER_PORT:-50641}"

latest_checkpoint_name() {
  if [[ -s "${CHECKPOINT_ROOT}/latest_checkpoint.txt" ]]; then
    tr -d '[:space:]' < "${CHECKPOINT_ROOT}/latest_checkpoint.txt"
  fi
}

latest_checkpoint_iter() {
  local latest iter
  latest="$(latest_checkpoint_name || true)"
  if [[ "${latest}" =~ ^iter_0*([0-9]+)$ ]]; then
    iter="${BASH_REMATCH[1]}"
    printf '%d\n' "$((10#${iter}))"
  else
    printf '0\n'
  fi
}

main() {
  cd "${ROOT}"
  echo "timestamp=$(date -Is)"
  echo "fallback_boundary=no_concurrent_training_writer_start_only_after_primary_job_gone"
  echo "primary_job_id=${PRIMARY_JOB_ID}"
  echo "fallback_job_id=${FALLBACK_JOB_ID}"
  echo "sft_root=${SFT_ROOT}"
  echo "condition_root=${CONDITION_ROOT}"
  echo "checkpoint_root=${CHECKPOINT_ROOT}"
  echo "target_iter=${TARGET_ITER}"

  while squeue -j "${PRIMARY_JOB_ID}" -h | awk '{print $1}' | grep -qx "${PRIMARY_JOB_ID}"; do
    echo "primary_training_running=$(date -Is) latest_checkpoint=$(latest_checkpoint_name || true) latest_iter=$(latest_checkpoint_iter)"
    sleep "${POLL_SECONDS}"
  done

  echo "primary_training_gone=$(date -Is)"
  sleep "${POST_STOP_STABLE_SECONDS}"

  local latest_iter latest
  latest="$(latest_checkpoint_name || true)"
  latest_iter="$(latest_checkpoint_iter)"
  echo "post_stop_latest_checkpoint=${latest:-missing}"
  echo "post_stop_latest_iter=${latest_iter}"

  if (( latest_iter >= TARGET_ITER )); then
    echo "fallback_not_needed=target_iter_reached"
    exit 0
  fi

  echo "fallback_triggering_2gpu_resume_from=${latest:-missing}"
  OUTPUT_ROOT="${SFT_ROOT}" \
  CONDITION_ROOT="${CONDITION_ROOT}" \
  NPROC_PER_NODE="${FALLBACK_NPROC}" \
  DATA_PARALLEL_SHARD_DEGREE="${FALLBACK_NPROC}" \
  DATA_PARALLEL_REPLICATE_DEGREE=1 \
  CONTEXT_PARALLEL_SHARD_DEGREE=1 \
  MAX_ITER="${TARGET_ITER}" \
  SAVE_ITER=300 \
  VALIDATION_ITER=300 \
  MAX_VAL_ITER=40 \
  MASTER_PORT="${MASTER_PORT}" \
  srun --jobid="${FALLBACK_JOB_ID}" --overlap --ntasks=1 \
    --gres="gpu:${FALLBACK_NPROC}" \
    --cpus-per-task="${FALLBACK_CPUS}" \
    --mem="${FALLBACK_MEM}" \
    bash "${ROOT}/scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_fix1recipe_in_allocation.sh"
}

main "$@"
