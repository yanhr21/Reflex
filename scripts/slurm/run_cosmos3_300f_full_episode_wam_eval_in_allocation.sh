#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this script only inside a compute-node srun step, for example: srun --jobid=$SLURM_JOB_ID --gres=gpu:1 --cpus-per-task=16 bash scripts/slurm/run_cosmos3_300f_full_episode_wam_eval_in_allocation.sh
EOF
  exit 30
fi

SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_20260609_204718}"
CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_full1000_rgb_300step_20260609_203902}"
JOB_NAME="${JOB_NAME:-vision_sft_droid_policy_full1000_rgb_300step_wam}"
RUN_DIR="${RUN_DIR:-${SFT_ROOT}/outputs/cosmos3/sft/${JOB_NAME}}"
CONFIG_FILE="${CONFIG_FILE:-${RUN_DIR}/config.yaml}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${RUN_DIR}/checkpoints}"
LATEST_FILE="${LATEST_FILE:-${CHECKPOINT_ROOT}/latest_checkpoint.txt}"
COSMOS_VENV="${COSMOS_VENV:-${ROOT}/.venv_cosmos313}"
LOCAL_TOKENIZER_DIR="${LOCAL_TOKENIZER_DIR:-${ROOT}/checkpoints/cosmos3/Cosmos3-Nano-Policy-DROID}"
EVAL_ROOT="${EVAL_ROOT:-${SFT_ROOT}/eval_full_episode_wam_latest}"
EVAL_INPUT_JSONL="${EVAL_INPUT_JSONL:-${EVAL_ROOT}/inputs/val_full_episode_wam_policy_samples.jsonl}"
N_EVAL_SAMPLES="${N_EVAL_SAMPLES:-10}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-50217}"
TOTAL_VIDEO_FRAMES="${TOTAL_VIDEO_FRAMES:-301}"
TOTAL_ACTION_STEPS="${TOTAL_ACTION_STEPS:-300}"
ACTION_DIM="${ACTION_DIM:-32}"
RUN_INFERENCE="${RUN_INFERENCE:-true}"
RUN_INSPECTION="${RUN_INSPECTION:-true}"
INFERENCE_NUM_STEPS="${INFERENCE_NUM_STEPS:-30}"
LIBFFI_COMPAT_DIR="${LIBFFI_COMPAT_DIR:-${COSMOS_VENV}/lib_compat}"

cosmos_nvidia_lib_dirs() {
  local dirs=()
  local site d
  shopt -s nullglob
  for site in "${COSMOS_VENV}"/lib/python*/site-packages; do
    for d in "${site}"/nvidia/*/lib "${site}"/nvidia/*/lib64; do
      [[ -d "${d}" ]] && dirs+=("${d}")
    done
  done
  shopt -u nullglob
  local IFS=:
  printf '%s' "${dirs[*]}"
}

refresh_cosmos_ld_library_path() {
  local nvidia_libs
  nvidia_libs="$(cosmos_nvidia_lib_dirs)"
  if [[ -d "${LIBFFI_COMPAT_DIR}" && -n "${nvidia_libs}" ]]; then
    export LD_LIBRARY_PATH="${LIBFFI_COMPAT_DIR}:${nvidia_libs}:${LD_LIBRARY_PATH:-}"
  elif [[ -d "${LIBFFI_COMPAT_DIR}" ]]; then
    export LD_LIBRARY_PATH="${LIBFFI_COMPAT_DIR}:${LD_LIBRARY_PATH:-}"
  elif [[ -n "${nvidia_libs}" ]]; then
    export LD_LIBRARY_PATH="${nvidia_libs}:${LD_LIBRARY_PATH:-}"
  fi
}

repair_transformer_engine_cudart_path() {
  local site nvidia_dir
  shopt -s nullglob
  for site in "${COSMOS_VENV}"/lib/python*/site-packages; do
    nvidia_dir="${site}/nvidia"
    [[ -d "${nvidia_dir}/cuda_runtime" ]] || continue
    [[ -e "${nvidia_dir}/cudart" ]] || ln -s cuda_runtime "${nvidia_dir}/cudart"
    [[ -e "${nvidia_dir}/cuda_cudart" ]] || ln -s cuda_runtime "${nvidia_dir}/cuda_cudart"
  done
  shopt -u nullglob
}

resolve_checkpoint_path() {
  if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
    printf '%s\n' "${CHECKPOINT_PATH}"
    return
  fi
  [[ -s "${LATEST_FILE}" ]] || {
    echo "missing latest checkpoint file: ${LATEST_FILE}" >&2
    exit 31
  }
  local latest
  latest="$(tr -d '[:space:]' < "${LATEST_FILE}")"
  [[ -n "${latest}" ]] || {
    echo "empty latest checkpoint file: ${LATEST_FILE}" >&2
    exit 32
  }
  printf '%s\n' "${CHECKPOINT_ROOT}/${latest}"
}

write_manifest() {
  mkdir -p "${EVAL_ROOT}"
  {
    echo "timestamp=$(date -Is)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "sft_root=${SFT_ROOT}"
    echo "condition_root=${CONDITION_ROOT}"
    echo "run_dir=${RUN_DIR}"
    echo "config_file=${CONFIG_FILE}"
    echo "checkpoint_path=${CHECKPOINT_PATH_RESOLVED}"
    echo "eval_root=${EVAL_ROOT}"
    echo "eval_input_jsonl=${EVAL_INPUT_JSONL}"
    echo "total_video_frames=${TOTAL_VIDEO_FRAMES}"
    echo "total_action_steps=${TOTAL_ACTION_STEPS}"
    echo "action_dim=${ACTION_DIM}"
    echo "inference_num_steps=${INFERENCE_NUM_STEPS}"
    echo "evidence_boundary=Generated validation artifacts are eval evidence only after strict inspection and visual review; not controller evidence."
  } | tee "${EVAL_ROOT}/eval_manifest.txt"
}

main() {
  cd "${ROOT}"
  [[ -x "${COSMOS_VENV}/bin/python" ]] || { echo "missing Cosmos Python: ${COSMOS_VENV}/bin/python" >&2; exit 2; }
  [[ -s "${CONFIG_FILE}" ]] || { echo "missing config file: ${CONFIG_FILE}" >&2; exit 3; }
  [[ -d "${CONDITION_ROOT}" ]] || { echo "missing condition root: ${CONDITION_ROOT}" >&2; exit 4; }
  CHECKPOINT_PATH_RESOLVED="$(resolve_checkpoint_path)"
  export CHECKPOINT_PATH_RESOLVED
  [[ -d "${CHECKPOINT_PATH_RESOLVED}/model" || -s "${CHECKPOINT_PATH_RESOLVED}/model/.metadata" || -s "${CHECKPOINT_PATH_RESOLVED}/.metadata" ]] || {
    echo "checkpoint path does not look like a saved DCP checkpoint: ${CHECKPOINT_PATH_RESOLVED}" >&2
    exit 5
  }
  write_manifest
  repair_transformer_engine_cudart_path
  refresh_cosmos_ld_library_path

  "${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/build_cosmos3_full_episode_wam_eval_inputs.py" \
    --condition-root "${CONDITION_ROOT}" \
    --output-root "${EVAL_ROOT}" \
    --split val \
    --num-samples "${N_EVAL_SAMPLES}" \
    --expected-video-frames "${TOTAL_VIDEO_FRAMES}" \
    --expected-action-steps "${TOTAL_ACTION_STEPS}" \
    --expected-action-dim "${ACTION_DIM}" \
    2>&1 | tee "${EVAL_ROOT}/build_eval_inputs.log"

  if [[ "${RUN_INFERENCE}" == "true" ]]; then
    bash -lc "
      set -euo pipefail
      cd '${ROOT}/external/cosmos-framework'
      export PATH='${COSMOS_VENV}/bin':\"\${PATH}\"
      export PYTHONPATH='${ROOT}/external/cosmos-framework'
      export LD_LIBRARY_PATH='${LD_LIBRARY_PATH:-}'
      export COSMOS3_LOCAL_TOKENIZER_DIR='${LOCAL_TOKENIZER_DIR}'
      export COSMOS3_ACTION_PROMPT_STYLE='plain_sft'
      export AWS_EC2_METADATA_DISABLED=true
      unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
      torchrun --nproc_per_node='${NPROC_PER_NODE}' --master_port='${MASTER_PORT}' \
        -m cosmos_framework.scripts.inference \
        --parallelism-preset=throughput \
        -i '${EVAL_INPUT_JSONL}' \
        -o '${EVAL_ROOT}/inference' \
        --checkpoint-path '${CHECKPOINT_PATH_RESOLVED}' \
        --config-file '${CONFIG_FILE}' \
        --no-use-ema-weights \
        --no-guardrails \
        --seed=0 \
        --num-steps='${INFERENCE_NUM_STEPS}'
    " 2>&1 | tee "${EVAL_ROOT}/inference.log"
  fi

  if [[ "${RUN_INSPECTION}" == "true" ]]; then
    "${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/inspect_cosmos3_full_episode_wam_eval_artifacts.py" \
      --eval-root "${EVAL_ROOT}" \
      --expected-video-frames "${TOTAL_VIDEO_FRAMES}" \
      --expected-action-steps "${TOTAL_ACTION_STEPS}" \
      --expected-action-dim "${ACTION_DIM}" \
      --output-json "${EVAL_ROOT}/eval_artifact_inspection.json" \
      --output-md "${EVAL_ROOT}/eval_artifact_inspection.md" \
      --strict \
      2>&1 | tee "${EVAL_ROOT}/inspect_eval_artifacts.log"
  fi
}

main "$@"
