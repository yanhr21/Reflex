#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run target-motion calibration only inside a compute-node srun step.
EOF
  exit 30
fi

CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_full1000_rgb_300step_20260609_203902}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/target_motion_readout_calibration_20260610_0751}"
READOUT_ROOT="${READOUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_20260609_204718/task_state_readout_reference_rgb_301f}"
READOUT_CHECKPOINT="${READOUT_CHECKPOINT:-${READOUT_ROOT}/best_model.pt}"
GENERATED_READOUT_ROOT="${GENERATED_READOUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_4gpu_warm_iter300_20260609_234018/eval_full_episode_wam_iter_000001500/task_state_readout_best_current}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-120}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-88}"
TOTAL_VIDEO_FRAMES="${TOTAL_VIDEO_FRAMES:-301}"
IMAGE_SIZE="${IMAGE_SIZE:-160}"
HEAD_STEPS="${HEAD_STEPS:-1200}"

export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

run_readout() {
  local eval_root="$1"
  local output_root="$2"
  "${PYTHON_BIN}" "${ROOT}/scripts/world_model/run_cosmos3_full_episode_readout_eval.py" \
    --eval-root "${eval_root}" \
    --readout-checkpoint "${READOUT_CHECKPOINT}" \
    --output-root "${output_root}" \
    --python-bin "${PYTHON_BIN}" \
    --num-frames "${TOTAL_VIDEO_FRAMES}" \
    --image-size "${IMAGE_SIZE}" \
    --strict
}

main() {
  cd "${ROOT}"
  [[ -x "${PYTHON_BIN}" ]] || { echo "missing python: ${PYTHON_BIN}" >&2; exit 2; }
  [[ -s "${READOUT_CHECKPOINT}" ]] || { echo "missing readout checkpoint: ${READOUT_CHECKPOINT}" >&2; exit 3; }
  mkdir -p "${OUTPUT_ROOT}"
  {
    echo "timestamp=$(date -Is)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "step_id=${SLURM_STEP_ID}"
    echo "condition_root=${CONDITION_ROOT}"
    echo "output_root=${OUTPUT_ROOT}"
    echo "readout_checkpoint=${READOUT_CHECKPOINT}"
    echo "max_train_samples=${MAX_TRAIN_SAMPLES}"
    echo "max_val_samples=${MAX_VAL_SAMPLES}"
    echo "boundary=Target-motion head calibration over RGB-derived readout. Diagnostic only; not controller evidence."
  } | tee "${OUTPUT_ROOT}/run_manifest.txt"

  local train_eval_root="${OUTPUT_ROOT}/train_reference_rgb_eval"
  local val_eval_root="${OUTPUT_ROOT}/val_reference_rgb_eval"
  local train_readout_root="${train_eval_root}/task_state_readout_best_current"
  local val_readout_root="${val_eval_root}/task_state_readout_best_current"

  "${PYTHON_BIN}" "${ROOT}/scripts/world_model/build_cosmos3_reference_rgb_eval_root_from_jsonl.py" \
    --condition-root "${CONDITION_ROOT}" \
    --split train \
    --output-root "${train_eval_root}" \
    --max-samples "${MAX_TRAIN_SAMPLES}" \
    --overwrite | tee "${OUTPUT_ROOT}/build_train_reference_rgb_eval.log"

  "${PYTHON_BIN}" "${ROOT}/scripts/world_model/build_cosmos3_reference_rgb_eval_root_from_jsonl.py" \
    --condition-root "${CONDITION_ROOT}" \
    --split val \
    --output-root "${val_eval_root}" \
    --max-samples "${MAX_VAL_SAMPLES}" \
    --overwrite | tee "${OUTPUT_ROOT}/build_val_reference_rgb_eval.log"

  run_readout "${train_eval_root}" "${train_readout_root}" 2>&1 | tee "${OUTPUT_ROOT}/train_reference_readout.log"
  run_readout "${val_eval_root}" "${val_readout_root}" 2>&1 | tee "${OUTPUT_ROOT}/val_reference_readout.log"

  "${PYTHON_BIN}" "${ROOT}/scripts/world_model/train_target_motion_head_from_readout.py" \
    --train-readout-root "${train_readout_root}" \
    --eval-readout-root "${val_readout_root}" \
    --extra-eval-root "iter1500_generated_rgb=${GENERATED_READOUT_ROOT}" \
    --output-dir "${OUTPUT_ROOT}/target_motion_head" \
    --steps "${HEAD_STEPS}" \
    --cuda \
    2>&1 | tee "${OUTPUT_ROOT}/target_motion_head_train.log"
}

main "$@"
