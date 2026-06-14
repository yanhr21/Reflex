#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run live-prefix Cosmos inference only inside a compute-node srun step.
EOF
  exit 30
fi

SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745}"
CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_rgb_300step_20260612_0245}"
JOB_NAME="${JOB_NAME:-vision_sft_droid_policy_full1000_rgb_300step_wam}"
RUN_DIR="${RUN_DIR:-${SFT_ROOT}/outputs/cosmos3/sft/${JOB_NAME}}"
CONFIG_FILE="${CONFIG_FILE:-${RUN_DIR}/config.yaml}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${RUN_DIR}/checkpoints}"
CHECKPOINT_ITER="${CHECKPOINT_ITER:-2100}"
printf -v CHECKPOINT_NAME 'iter_%09d' "${CHECKPOINT_ITER}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT_ROOT}/${CHECKPOINT_NAME}}"
COSMOS_VENV="${COSMOS_VENV:-${ROOT}/.venv_cosmos313}"
LOCAL_TOKENIZER_DIR="${LOCAL_TOKENIZER_DIR:-${ROOT}/checkpoints/cosmos3/Cosmos3-Nano-Policy-DROID}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SFT_ROOT}/live_prefix_receding_probe_${CHECKPOINT_NAME}_$(date +%Y%m%d_%H%M%S)}"
PREFIX_VIDEO="${PREFIX_VIDEO:?set PREFIX_VIDEO to the observed causal prefix video}"
PREFIX_FRAME_INDEX="${PREFIX_FRAME_INDEX:?set PREFIX_FRAME_INDEX to the current live frame index}"
SOURCE_H5="${SOURCE_H5:-}"
HISTORY_ACTION_PATH="${HISTORY_ACTION_PATH:-}"
SAMPLE_NAME="${SAMPLE_NAME:-live_prefix_receding_step}"
SCENARIO="${SCENARIO:-live_dynamic}"
PREFIX_ROLE="${PREFIX_ROLE:-live_receding}"
PROMPT="${PROMPT:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-50917}"
INFERENCE_NUM_STEPS="${INFERENCE_NUM_STEPS:-30}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-8}"
RUN_INFERENCE="${RUN_INFERENCE:-true}"
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

main() {
  cd "${ROOT}"
  [[ -x "${COSMOS_VENV}/bin/python" ]] || { echo "missing Cosmos Python: ${COSMOS_VENV}/bin/python" >&2; exit 2; }
  [[ -s "${CONFIG_FILE}" ]] || { echo "missing config file: ${CONFIG_FILE}" >&2; exit 3; }
  [[ -d "${CHECKPOINT_PATH}" ]] || { echo "missing checkpoint path: ${CHECKPOINT_PATH}" >&2; exit 4; }
  [[ -d "${CONDITION_ROOT}" ]] || { echo "missing condition root: ${CONDITION_ROOT}" >&2; exit 5; }
  [[ -s "${PREFIX_VIDEO}" ]] || { echo "missing prefix video: ${PREFIX_VIDEO}" >&2; exit 6; }

  mkdir -p "${OUTPUT_ROOT}"
  {
    echo "timestamp=$(date -Is)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "step_id=${SLURM_STEP_ID}"
    echo "sft_root=${SFT_ROOT}"
    echo "condition_root=${CONDITION_ROOT}"
    echo "checkpoint_path=${CHECKPOINT_PATH}"
    echo "config_file=${CONFIG_FILE}"
    echo "prefix_video=${PREFIX_VIDEO}"
    echo "prefix_frame_index=${PREFIX_FRAME_INDEX}"
    echo "source_h5=${SOURCE_H5}"
    echo "history_action_path=${HISTORY_ACTION_PATH}"
    echo "output_root=${OUTPUT_ROOT}"
    echo "run_inference=${RUN_INFERENCE}"
    echo "action_exec_horizon=${ACTION_EXEC_HORIZON}"
    echo "boundary=Single live-prefix Cosmos inference input/output for receding controller development; not method evidence by itself."
  } | tee "${OUTPUT_ROOT}/live_prefix_wrapper_manifest.txt"

  local builder_args=(
    --output-root "${OUTPUT_ROOT}"
    --prefix-video "${PREFIX_VIDEO}"
    --normalization-stats "${CONDITION_ROOT}/normalization_stats.json"
    --prefix-frame-index "${PREFIX_FRAME_INDEX}"
    --sample-name "${SAMPLE_NAME}"
    --scenario "${SCENARIO}"
    --prefix-role "${PREFIX_ROLE}"
    --condition-root "${CONDITION_ROOT}"
    --checkpoint-path "${CHECKPOINT_PATH}"
    --prompt "${PROMPT}"
  )
  if [[ -n "${HISTORY_ACTION_PATH}" ]]; then
    # Controller-facing live inference must condition on the real executed
    # history. Do not let a provenance SOURCE_H5 replace that history.
    :
  elif [[ -n "${SOURCE_H5}" ]]; then
    builder_args+=(--source-h5 "${SOURCE_H5}")
  fi
  if [[ -n "${HISTORY_ACTION_PATH}" ]]; then
    builder_args+=(--history-action-path "${HISTORY_ACTION_PATH}")
  fi

  "${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/build_cosmos3_live_prefix_wam_input.py" \
    "${builder_args[@]}" \
    2>&1 | tee "${OUTPUT_ROOT}/build_live_prefix_input.log"

  if [[ "${RUN_INFERENCE}" != "true" ]]; then
    echo "live_prefix_input_only=true" | tee -a "${OUTPUT_ROOT}/live_prefix_wrapper_manifest.txt"
    return 0
  fi

  repair_transformer_engine_cudart_path
  refresh_cosmos_ld_library_path
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
      -i '${OUTPUT_ROOT}/inputs/live_prefix_wam_policy_samples.jsonl' \
      -o '${OUTPUT_ROOT}/inference' \
      --checkpoint-path '${CHECKPOINT_PATH}' \
      --config-file '${CONFIG_FILE}' \
      --no-use-ema-weights \
      --no-guardrails \
      --seed=0 \
      --num-steps='${INFERENCE_NUM_STEPS}'
  " 2>&1 | tee "${OUTPUT_ROOT}/live_prefix_inference.log"

  "${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/extract_cosmos3_policy_action_chunk.py" \
    --sample-output-json "${OUTPUT_ROOT}/inference/${SAMPLE_NAME}/sample_outputs.json" \
    --normalization-stats "${CONDITION_ROOT}/normalization_stats.json" \
    --prefix-frame-index "${PREFIX_FRAME_INDEX}" \
    --action-exec-horizon "${ACTION_EXEC_HORIZON}" \
    --output-json "${OUTPUT_ROOT}/live_prefix_action_chunk.json" \
    2>&1 | tee "${OUTPUT_ROOT}/extract_live_prefix_action_chunk.log"
}

main "$@"
