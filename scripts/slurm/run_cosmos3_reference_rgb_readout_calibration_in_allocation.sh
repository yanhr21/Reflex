#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this calibration only inside a compute-node srun step. It builds a reference-RGB eval root and runs task-state readout/profile there.
EOF
  exit 30
fi

SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_4gpu_warm_iter300_20260609_234018}"
SOURCE_EVAL_ROOT="${SOURCE_EVAL_ROOT:-${SFT_ROOT}/eval_full_episode_wam_iter_000001500}"
OUTPUT_EVAL_ROOT="${OUTPUT_EVAL_ROOT:-${SOURCE_EVAL_ROOT}/reference_rgb_readout_calibration}"
READOUT_ROOT="${READOUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_20260609_204718/task_state_readout_reference_rgb_301f}"
READOUT_CHECKPOINT="${READOUT_CHECKPOINT:-${READOUT_ROOT}/best_model.pt}"
READOUT_EVAL_ROOT="${READOUT_EVAL_ROOT:-${OUTPUT_EVAL_ROOT}/task_state_readout_best_current}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
TOTAL_VIDEO_FRAMES="${TOTAL_VIDEO_FRAMES:-301}"
IMAGE_SIZE="${IMAGE_SIZE:-160}"
THRESHOLDS_M="${THRESHOLDS_M:-0.002,0.005,0.01,0.02,0.05}"

export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

main() {
  cd "${ROOT}"
  [[ -x "${PYTHON_BIN}" ]] || { echo "missing python: ${PYTHON_BIN}" >&2; exit 2; }
  [[ -s "${SOURCE_EVAL_ROOT}/eval_input_manifest.json" ]] || {
    echo "missing source eval manifest: ${SOURCE_EVAL_ROOT}/eval_input_manifest.json" >&2
    exit 3
  }
  [[ -s "${READOUT_CHECKPOINT}" ]] || { echo "missing readout checkpoint: ${READOUT_CHECKPOINT}" >&2; exit 4; }

  mkdir -p "${OUTPUT_EVAL_ROOT}" "${READOUT_EVAL_ROOT}"
  {
    echo "timestamp=$(date -Is)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "step_id=${SLURM_STEP_ID}"
    echo "source_eval_root=${SOURCE_EVAL_ROOT}"
    echo "output_eval_root=${OUTPUT_EVAL_ROOT}"
    echo "readout_checkpoint=${READOUT_CHECKPOINT}"
    echo "readout_eval_root=${READOUT_EVAL_ROOT}"
    echo "boundary=Reference-RGB calibration only. GT RGB is used to test the readout/onset decoder, not to claim Cosmos3 world-model success."
  } | tee "${OUTPUT_EVAL_ROOT}/reference_rgb_readout_calibration_run_manifest.txt"

  "${PYTHON_BIN}" "${ROOT}/scripts/world_model/build_cosmos3_reference_rgb_eval_root.py" \
    --source-eval-root "${SOURCE_EVAL_ROOT}" \
    --output-root "${OUTPUT_EVAL_ROOT}" \
    --overwrite \
    2>&1 | tee "${OUTPUT_EVAL_ROOT}/build_reference_rgb_eval_root.log"

  "${PYTHON_BIN}" "${ROOT}/scripts/world_model/run_cosmos3_full_episode_readout_eval.py" \
    --eval-root "${OUTPUT_EVAL_ROOT}" \
    --readout-checkpoint "${READOUT_CHECKPOINT}" \
    --output-root "${READOUT_EVAL_ROOT}" \
    --python-bin "${PYTHON_BIN}" \
    --num-frames "${TOTAL_VIDEO_FRAMES}" \
    --image-size "${IMAGE_SIZE}" \
    --strict \
    2>&1 | tee "${READOUT_EVAL_ROOT}/run_readout_eval.log"

  "${PYTHON_BIN}" "${ROOT}/scripts/world_model/profile_cosmos3_readout_failure_modes.py" \
    --readout-root "${READOUT_EVAL_ROOT}" \
    --output-json "${READOUT_EVAL_ROOT}/readout_failure_profile.json" \
    --output-md "${READOUT_EVAL_ROOT}/readout_failure_profile.md" \
    --thresholds-m "${THRESHOLDS_M}" \
    2>&1 | tee "${READOUT_EVAL_ROOT}/run_readout_failure_profile.log"
}

main "$@"
