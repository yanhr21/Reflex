#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this script only inside a compute-node srun step, for example: srun --jobid=$SLURM_JOB_ID --gres=gpu:1 --cpus-per-task=8 bash scripts/slurm/run_cosmos3_300f_task_state_readout_in_allocation.sh
EOF
  exit 30
fi

SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_20260609_204718}"
DATASET_MANIFEST="${DATASET_MANIFEST:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_dataset_full1000_maniskill_default_regen_20260606_0055/manifest.json}"
EVAL_ROOT="${EVAL_ROOT:-${SFT_ROOT}/eval_full_episode_wam_latest}"
READOUT_ROOT="${READOUT_ROOT:-${SFT_ROOT}/task_state_readout_reference_rgb_301f}"
READOUT_CHECKPOINT="${READOUT_CHECKPOINT:-${READOUT_ROOT}/best_model.pt}"
READOUT_EVAL_ROOT="${READOUT_EVAL_ROOT:-${EVAL_ROOT}/task_state_readout}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
TRAIN_READOUT="${TRAIN_READOUT:-true}"
RUN_EVAL_READOUT="${RUN_EVAL_READOUT:-true}"
TOTAL_VIDEO_FRAMES="${TOTAL_VIDEO_FRAMES:-301}"
FUTURE_START_FRAME="${FUTURE_START_FRAME:-29}"
IMAGE_SIZE="${IMAGE_SIZE:-160}"
READOUT_STEPS="${READOUT_STEPS:-2000}"
READOUT_BATCH_SIZE="${READOUT_BATCH_SIZE:-1}"
READOUT_MAX_EVAL_BATCHES="${READOUT_MAX_EVAL_BATCHES:-40}"

export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

write_manifest() {
  mkdir -p "${READOUT_ROOT}" "${READOUT_EVAL_ROOT}"
  {
    echo "timestamp=$(date -Is)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "sft_root=${SFT_ROOT}"
    echo "dataset_manifest=${DATASET_MANIFEST}"
    echo "eval_root=${EVAL_ROOT}"
    echo "readout_root=${READOUT_ROOT}"
    echo "readout_checkpoint=${READOUT_CHECKPOINT}"
    echo "readout_eval_root=${READOUT_EVAL_ROOT}"
    echo "total_video_frames=${TOTAL_VIDEO_FRAMES}"
    echo "hdf5_use_file_locking=${HDF5_USE_FILE_LOCKING}"
    echo "boundary=Reference-video readout and generated-video readout metrics are diagnostics on top of Cosmos3 RGB, not a replacement world model or controller evidence."
  } | tee "${READOUT_ROOT}/readout_run_manifest.txt"
}

main() {
  cd "${ROOT}"
  [[ -x "${PYTHON_BIN}" ]] || { echo "missing python: ${PYTHON_BIN}" >&2; exit 2; }
  [[ -s "${DATASET_MANIFEST}" ]] || { echo "missing dataset manifest: ${DATASET_MANIFEST}" >&2; exit 3; }
  write_manifest

  if [[ "${TRAIN_READOUT}" == "true" ]]; then
    "${PYTHON_BIN}" "${ROOT}/scripts/world_model/train_cosmos3_task_state_readout.py" \
      --dataset-manifest "${DATASET_MANIFEST}" \
      --output-dir "${READOUT_ROOT}" \
      --num-frames "${TOTAL_VIDEO_FRAMES}" \
      --future-start-frame "${FUTURE_START_FRAME}" \
      --image-size "${IMAGE_SIZE}" \
      --steps "${READOUT_STEPS}" \
      --batch-size "${READOUT_BATCH_SIZE}" \
      --max-eval-batches "${READOUT_MAX_EVAL_BATCHES}" \
      --require-exact-video-frames \
      --require-cuda \
      2>&1 | tee "${READOUT_ROOT}/train_readout.log"
  fi

  if [[ "${RUN_EVAL_READOUT}" == "true" ]]; then
    [[ -s "${READOUT_CHECKPOINT}" ]] || { echo "missing readout checkpoint: ${READOUT_CHECKPOINT}" >&2; exit 4; }
    [[ -s "${EVAL_ROOT}/eval_input_manifest.json" ]] || { echo "missing eval input manifest: ${EVAL_ROOT}/eval_input_manifest.json" >&2; exit 5; }
    "${PYTHON_BIN}" "${ROOT}/scripts/world_model/run_cosmos3_full_episode_readout_eval.py" \
      --eval-root "${EVAL_ROOT}" \
      --readout-checkpoint "${READOUT_CHECKPOINT}" \
      --output-root "${READOUT_EVAL_ROOT}" \
      --python-bin "${PYTHON_BIN}" \
      --num-frames "${TOTAL_VIDEO_FRAMES}" \
      --image-size "${IMAGE_SIZE}" \
      --strict \
      2>&1 | tee "${READOUT_EVAL_ROOT}/run_readout_eval.log"
  fi
}

main "$@"
