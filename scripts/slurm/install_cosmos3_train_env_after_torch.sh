#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
COSMOS_VENV="${COSMOS_VENV:-${ROOT}/.venv_cosmos}"
COSMOS_PYTHON="${COSMOS_PYTHON:-${COSMOS_VENV}/bin/python}"
COSMOS_FRAMEWORK_DIR="${COSMOS_FRAMEWORK_DIR:-${ROOT}/external/cosmos-framework}"
DATASET_PATH="${DATASET_PATH:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/action_state_conditions_full1000_maniskill_default_regen_full301_joint_policy_clean_caption_20260608}"
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-${ROOT}/checkpoints/cosmos3/Cosmos3-Nano-Policy-DROID-DCP}"
WAN_VAE_PATH="${WAN_VAE_PATH:-${ROOT}/checkpoints/cosmos3/wan22_vae/Wan2.2_VAE.pth}"
DRYRUN_OUTPUT_ROOT="${DRYRUN_OUTPUT_ROOT:-/tmp/cosmos3_sft_env_dryrun_$(date +%Y%m%d_%H%M%S)}"
POLL_SECONDS="${POLL_SECONDS:-30}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/cosmos3}"
LIBFFI_COMPAT_DIR="${LIBFFI_COMPAT_DIR:-${COSMOS_VENV}/lib_compat}"
PRE_SYNC_TORCH_READY="${PRE_SYNC_TORCH_READY:-true}"
CREATE_VENV_IF_MISSING="${CREATE_VENV_IF_MISSING:-false}"
COSMOS_PYTHON_VERSION="${COSMOS_PYTHON_VERSION:-3.13}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/${USER}_uv_cache_cosmos3}"
UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
GIT_HTTP_VERSION="${GIT_HTTP_VERSION:-HTTP/1.1}"
GIT_HTTP_LOW_SPEED_LIMIT="${GIT_HTTP_LOW_SPEED_LIMIT:-0}"
GIT_HTTP_LOW_SPEED_TIME="${GIT_HTTP_LOW_SPEED_TIME:-999999}"
export UV_CACHE_DIR
export UV_HTTP_TIMEOUT
export GIT_HTTP_VERSION
export GIT_HTTP_LOW_SPEED_LIMIT
export GIT_HTTP_LOW_SPEED_TIME
mkdir -p "${UV_CACHE_DIR}"

if [[ "${CREATE_VENV_IF_MISSING}" == "true" && ! -x "${COSMOS_PYTHON}" ]]; then
  uv venv "${COSMOS_VENV}" --python "${COSMOS_PYTHON_VERSION}" --seed
fi
mkdir -p "${LIBFFI_COMPAT_DIR}"
if [[ ! -e "${LIBFFI_COMPAT_DIR}/libffi.so.6" && -e /lib/x86_64-linux-gnu/libffi.so.8 ]]; then
  ln -s /lib/x86_64-linux-gnu/libffi.so.8 "${LIBFFI_COMPAT_DIR}/libffi.so.6"
fi

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
  if [[ -n "${nvidia_libs}" ]]; then
    export LD_LIBRARY_PATH="${LIBFFI_COMPAT_DIR}:${nvidia_libs}:${LD_LIBRARY_PATH:-}"
  else
    export LD_LIBRARY_PATH="${LIBFFI_COMPAT_DIR}:${LD_LIBRARY_PATH:-}"
  fi
}

repair_transformer_engine_cudart_path() {
  local site nvidia_dir
  shopt -s nullglob
  for site in "${COSMOS_VENV}"/lib/python*/site-packages; do
    nvidia_dir="${site}/nvidia"
    [[ -d "${nvidia_dir}/cuda_runtime" ]] || continue
    if [[ ! -e "${nvidia_dir}/cudart" ]]; then
      ln -s cuda_runtime "${nvidia_dir}/cudart"
    fi
    if [[ ! -e "${nvidia_dir}/cuda_cudart" ]]; then
      ln -s cuda_runtime "${nvidia_dir}/cuda_cudart"
    fi
  done
  shopt -u nullglob
}

refresh_cosmos_ld_library_path

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

torch_ready() {
  [[ -x "${COSMOS_PYTHON}" ]] || return 1
  "${COSMOS_PYTHON}" - <<'PY' >/dev/null 2>&1
import torch
import torchvision
import torch.distributed.checkpoint.hf_storage  # noqa: F401
assert torch.__version__.startswith("2.10.0")
PY
}

env_import_ready() {
  PYTHONPATH="${COSMOS_FRAMEWORK_DIR}" "${COSMOS_PYTHON}" - <<'PY' >/dev/null 2>&1
import torch
import qwen_vl_utils  # noqa: F401
import torch.distributed.checkpoint.hf_storage  # noqa: F401
from cosmos_framework.scripts import convert_model_to_dcp, train  # noqa: F401
PY
}

run_dryrun() {
  PYTHONPATH="${COSMOS_FRAMEWORK_DIR}" \
  DATASET_PATH="${DATASET_PATH}" \
  BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH}" \
  WAN_VAE_PATH="${WAN_VAE_PATH}" \
  IMAGINAIRE_OUTPUT_ROOT="${DRYRUN_OUTPUT_ROOT}" \
  "${COSMOS_PYTHON}" -m cosmos_framework.scripts.train \
    --sft-toml="${COSMOS_FRAMEWORK_DIR}/examples/toml/sft_config/vision_sft_nano.toml" \
    --dryrun \
    -- \
    job.name=vision_sft_droid_policy_full1000_rgb_full301_env_dryrun \
    trainer.max_iter=500 \
    checkpoint.save_iter=100 \
    "model.config.resolution='256'" \
    model.config.ema.enabled=false \
    model.config.compile.enabled=false \
    model.config.parallelism.data_parallel_shard_degree=1 \
    model.config.parallelism.data_parallel_replicate_degree=1 \
    model.config.max_num_tokens_after_packing=16384 \
    dataloader_train.max_sequence_length=16384
}

main() {
  mkdir -p "${LOG_DIR}"
  {
    echo "timestamp=$(date -Is)"
    echo "cosmos_venv=${COSMOS_VENV}"
    echo "cosmos_framework_dir=${COSMOS_FRAMEWORK_DIR}"
    echo "dataset_path=${DATASET_PATH}"
    echo "dryrun_output_root=${DRYRUN_OUTPUT_ROOT}"
    echo "physical_reason=prepare_official_cosmos3_training_environment_for_clean_full301_rgb_droid_policy_sft"
    echo "boundary=environment_preflight_not_training_or_controller_evidence"
  } | tee "${LOG_DIR}/install_cosmos3_train_env_manifest.txt"

  if [[ "${PRE_SYNC_TORCH_READY}" == "true" ]]; then
    until torch_ready; do
      log "waiting_for_torch210_cu128 cosmos_python=${COSMOS_PYTHON}"
      sleep "${POLL_SECONDS}"
    done
  else
    log "skipping_pre_sync_torch_ready cosmos_python=${COSMOS_PYTHON}"
  fi

  log "torch_ready_installing_cosmos_train_dependencies"
  cd "${COSMOS_FRAMEWORK_DIR}"
  UV_PROJECT_ENVIRONMENT="${COSMOS_VENV}" uv sync --system-certs \
    --python "${COSMOS_PYTHON}" \
    --no-python-downloads \
    --frozen \
    --no-dev \
    --inexact \
    --extra train \
    --group cu128-train \
    --index-strategy unsafe-best-match
  uv pip install --system-certs -p "${COSMOS_PYTHON}" --no-deps -e .
  repair_transformer_engine_cudart_path
  refresh_cosmos_ld_library_path

  log "running_import_preflight"
  env_import_ready

  log "running_sft_config_dryrun"
  run_dryrun

  log "complete cosmos_train_env_ready=true"
}

main "$@"
