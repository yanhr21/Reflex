#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
SFT_ROOT="${SFT_ROOT:?set SFT_ROOT to the active Cosmos3 SFT root}"
ALLOCATION_JOB_ID="${ALLOCATION_JOB_ID:?set ALLOCATION_JOB_ID to the held Slurm allocation}"
SOURCE_DATASET_ROOT="${SOURCE_DATASET_ROOT:?set SOURCE_DATASET_ROOT}"
CONDITION_ROOT="${CONDITION_ROOT:?set CONDITION_ROOT}"
EXPECTED_SOURCE_EPISODES="${EXPECTED_SOURCE_EPISODES:-733}"
RESUME_AFTER_ITER="${RESUME_AFTER_ITER:-1500}"
TARGET_MAX_ITER="${TARGET_MAX_ITER:-2100}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
STABLE_SECONDS="${STABLE_SECONDS:-90}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DATA_PARALLEL_SHARD_DEGREE="${DATA_PARALLEL_SHARD_DEGREE:-${NPROC_PER_NODE}}"
DATA_PARALLEL_REPLICATE_DEGREE="${DATA_PARALLEL_REPLICATE_DEGREE:-1}"
CONTEXT_PARALLEL_SHARD_DEGREE="${CONTEXT_PARALLEL_SHARD_DEGREE:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MASTER_PORT="${MASTER_PORT:-50631}"
SAVE_ITER="${SAVE_ITER:-300}"
VALIDATION_ITER="${VALIDATION_ITER:-300}"
MAX_VAL_ITER="${MAX_VAL_ITER:-40}"
SCHEDULER_CYCLE_LENGTH="${SCHEDULER_CYCLE_LENGTH:-${TARGET_MAX_ITER}}"

JOB_NAME="${JOB_NAME:-vision_sft_droid_policy_full1000_rgb_300step_wam}"
CKPT_DIR="${SFT_ROOT}/outputs/cosmos3/sft/${JOB_NAME}/checkpoints"
RESUME_ITER_NAME="$(printf 'iter_%09d' "${RESUME_AFTER_ITER}")"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
WATCH_LOG="${WATCH_LOG:-${SFT_ROOT}/sft_train_${NPROC_PER_NODE}gpu_auto_resume_after${RESUME_AFTER_ITER}_to${TARGET_MAX_ITER}_${STAMP}.watch.log}"

active_non_extern_steps() {
  squeue --steps -j "${ALLOCATION_JOB_ID}" -h -o '%i %j %M %N' 2>/dev/null \
    | awk '$2 != "extern" {print $1 ":" $2 ":" $3 ":" $4}' \
    | paste -sd, - || true
}

latest_checkpoint() {
  cat "${CKPT_DIR}/latest_checkpoint.txt" 2>/dev/null || true
}

{
  echo "timestamp=$(date -Is)"
  echo "purpose=auto_resume_after_checkpoint_without_concurrent_checkpoint_writer"
  echo "allocation_job_id=${ALLOCATION_JOB_ID}"
  echo "sft_root=${SFT_ROOT}"
  echo "resume_after_iter=${RESUME_AFTER_ITER}"
  echo "target_resume=${RESUME_ITER_NAME}"
  echo "target_max_iter=${TARGET_MAX_ITER}"
  echo "nproc_per_node=${NPROC_PER_NODE}"

  while true; do
    latest="$(latest_checkpoint)"
    has_target=no
    [[ -d "${CKPT_DIR}/${RESUME_ITER_NAME}" ]] && has_target=yes
    active_steps="$(active_non_extern_steps)"
    echo "poll $(date -Is) latest=${latest:-none} has_target=${has_target} active_steps=${active_steps:-none}"
    if [[ "${latest}" == "${RESUME_ITER_NAME}" && "${has_target}" == "yes" && -z "${active_steps}" ]]; then
      break
    fi
    sleep "${SLEEP_SECONDS}"
  done

  echo "stable_wait_seconds=${STABLE_SECONDS}"
  sleep "${STABLE_SECONDS}"
  latest_after_wait="$(latest_checkpoint)"
  if [[ "${latest_after_wait}" != "${RESUME_ITER_NAME}" ]]; then
    echo "refusing_resume_latest_changed=${latest_after_wait:-none}"
    exit 41
  fi
  active_after_wait="$(active_non_extern_steps)"
  if [[ -n "${active_after_wait}" ]]; then
    echo "refusing_resume_active_steps_after_wait=${active_after_wait}"
    exit 42
  fi

  cp -f "${SFT_ROOT}/sft_train.log" \
    "${SFT_ROOT}/sft_train_before_resume_after${RESUME_AFTER_ITER}_to${TARGET_MAX_ITER}_${STAMP}.log" 2>/dev/null || true
  cp -f "${SFT_ROOT}/sft_manifest.txt" \
    "${SFT_ROOT}/sft_manifest_before_resume_after${RESUME_AFTER_ITER}_to${TARGET_MAX_ITER}_${STAMP}.txt" 2>/dev/null || true

  echo "launch_resume $(date -Is)"
  cd "${ROOT}"
  set +e
  srun --jobid="${ALLOCATION_JOB_ID}" --nodes=1 --ntasks=1 \
    --gres="gpu:${NPROC_PER_NODE}" --cpus-per-task="${CPUS_PER_TASK}" \
    env SOURCE_DATASET_ROOT="${SOURCE_DATASET_ROOT}" \
      CONDITION_ROOT="${CONDITION_ROOT}" \
      OUTPUT_ROOT="${SFT_ROOT}" \
      EXPECTED_SOURCE_EPISODES="${EXPECTED_SOURCE_EPISODES}" \
      MAX_ITER="${TARGET_MAX_ITER}" SAVE_ITER="${SAVE_ITER}" \
      RUN_VALIDATION=true VALIDATION_ITER="${VALIDATION_ITER}" MAX_VAL_ITER="${MAX_VAL_ITER}" \
      MASTER_PORT="${MASTER_PORT}" \
      NPROC_PER_NODE="${NPROC_PER_NODE}" \
      DATA_PARALLEL_SHARD_DEGREE="${DATA_PARALLEL_SHARD_DEGREE}" \
      DATA_PARALLEL_REPLICATE_DEGREE="${DATA_PARALLEL_REPLICATE_DEGREE}" \
      CONTEXT_PARALLEL_SHARD_DEGREE="${CONTEXT_PARALLEL_SHARD_DEGREE}" \
      OPTIMIZER_LR=1.0e-4 SCHEDULER_CYCLE_LENGTH="${SCHEDULER_CYCLE_LENGTH}" \
      SCHEDULER_WARMUP_STEPS=10 SCHEDULER_F_MIN=0.5 \
      GRAD_CLIP_NORM=1.0 ACTION_LOSS_WEIGHT=2.0 \
      NORMALIZE_LOSS_BY_ACTIVE=true INDEPENDENT_ACTION_SCHEDULE=true SHIFT_ACTION=1 \
      ENFORCE_FIX1_ACTION_RECIPE=true \
      bash scripts/slurm/run_cosmos3_300f_full_episode_wam_in_allocation.sh

  code=$?
  set -e
  echo "resume_exit_code=${code} timestamp=$(date -Is)"
  exit "${code}"
} >> "${WATCH_LOG}" 2>&1
