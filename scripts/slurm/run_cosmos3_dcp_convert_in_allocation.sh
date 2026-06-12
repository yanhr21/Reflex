#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
JOB_ID="${JOB_ID:?set JOB_ID to an existing Slurm allocation id}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/checkpoints/cosmos3/Cosmos3-Nano}"
DCP_OUTPUT_DIR="${DCP_OUTPUT_DIR:-${ROOT}/checkpoints/cosmos3/Cosmos3-Nano-DCP}"
COSMOS_PYTHON="${COSMOS_PYTHON:-${ROOT}/.venv_cosmos313/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LIBFFI_COMPAT_DIR="${LIBFFI_COMPAT_DIR:-${ROOT}/.venv_cosmos313/lib_compat}"
LOCAL_TOKENIZER_DIR="${LOCAL_TOKENIZER_DIR:-${CHECKPOINT_DIR}}"
WAN_VAE_PATH="${WAN_VAE_PATH:-${ROOT}/checkpoints/cosmos3/wan22_vae/Wan2.2_VAE.pth}"

cosmos_nvidia_lib_dirs() {
  local venv
  venv="$(dirname "$(dirname "${COSMOS_PYTHON}")")"
  local dirs=()
  local site d
  shopt -s nullglob
  for site in "${venv}"/lib/python*/site-packages; do
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
  local venv
  venv="$(dirname "$(dirname "${COSMOS_PYTHON}")")"
  local site nvidia_dir
  shopt -s nullglob
  for site in "${venv}"/lib/python*/site-packages; do
    nvidia_dir="${site}/nvidia"
    [[ -d "${nvidia_dir}/cuda_runtime" ]] || continue
    [[ -e "${nvidia_dir}/cudart" ]] || ln -s cuda_runtime "${nvidia_dir}/cudart"
    [[ -e "${nvidia_dir}/cuda_cudart" ]] || ln -s cuda_runtime "${nvidia_dir}/cuda_cudart"
  done
  shopt -u nullglob
}

repair_transformer_engine_cudart_path
refresh_cosmos_ld_library_path

required_files=(
  "checkpoint.json"
  "config.json"
  "model.safetensors.index.json"
  "model_index.json"
  "transformer/config.json"
  "transformer/diffusion_pytorch_model-00001-of-00007.safetensors"
  "transformer/diffusion_pytorch_model-00002-of-00007.safetensors"
  "transformer/diffusion_pytorch_model-00003-of-00007.safetensors"
  "transformer/diffusion_pytorch_model-00004-of-00007.safetensors"
  "transformer/diffusion_pytorch_model-00005-of-00007.safetensors"
  "transformer/diffusion_pytorch_model-00006-of-00007.safetensors"
  "transformer/diffusion_pytorch_model-00007-of-00007.safetensors"
  "vision_encoder/config.json"
  "vision_encoder/model.safetensors"
  "text_tokenizer/tokenizer_config.json"
  "text_tokenizer/tokenizer.json"
  "text_tokenizer/vocab.json"
  "text_tokenizer/merges.txt"
)

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

checkpoint_ready() {
  local rel
  for rel in "${required_files[@]}"; do
    [[ -s "${CHECKPOINT_DIR}/${rel}" ]] || return 1
  done
  return 0
}

cosmos_python_ready() {
  [[ -x "${COSMOS_PYTHON}" ]] || return 1
  PYTHONPATH="${ROOT}/external/cosmos-framework" timeout 180 "${COSMOS_PYTHON}" - "${ROOT}" "${LOCAL_TOKENIZER_DIR}" "${WAN_VAE_PATH}" <<'PY'
import os
import sys
import torch
import torch.distributed.checkpoint.hf_storage  # noqa: F401

root = sys.argv[1]
tokenizer_dir = sys.argv[2]
wan_vae_path = sys.argv[3]
required = ["tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"]
missing = [name for name in required if not os.path.isfile(os.path.join(tokenizer_dir, name))]
if missing:
    raise SystemExit(f"local tokenizer is incomplete: {tokenizer_dir}, missing={missing}")
if not os.path.isfile(wan_vae_path):
    raise SystemExit(f"missing Wan VAE: {wan_vae_path}")
if not os.path.isfile(os.path.join(root, "external/cosmos-framework/cosmos_framework/scripts/convert_model_to_dcp.py")):
    raise SystemExit("missing Cosmos3 DCP converter source")
PY
}

main() {
  cd "${ROOT}"
  mkdir -p "${DCP_OUTPUT_DIR}" "${ROOT}/logs/cosmos3"
  {
    echo "timestamp=$(date -Is)"
    echo "job_id=${JOB_ID}"
    echo "checkpoint_dir=${CHECKPOINT_DIR}"
    echo "dcp_output_dir=${DCP_OUTPUT_DIR}"
    echo "cosmos_python=${COSMOS_PYTHON}"
    echo "local_tokenizer_dir=${LOCAL_TOKENIZER_DIR}"
    echo "wan_vae_path=${WAN_VAE_PATH}"
    echo "disable_sound_for_vision_dcp=true"
    echo "physical_reason=convert_pretrained_cosmos3_diffusers_checkpoint_to_dcp_for_full1000_rgbd_sft_warm_start"
    echo "boundary=checkpoint_conversion_not_method_evidence"
  } | tee "${DCP_OUTPUT_DIR}/conversion_manifest.txt"

  until checkpoint_ready; do
    log "waiting_for_required_cosmos3_diffusers_files checkpoint_dir=${CHECKPOINT_DIR}"
    sleep "${POLL_SECONDS}"
  done

  until cosmos_python_ready; do
    log "waiting_for_cosmos_python_env cosmos_python=${COSMOS_PYTHON}"
    sleep "${POLL_SECONDS}"
  done

  if [[ -s "${DCP_OUTPUT_DIR}/model/.metadata" || -s "${DCP_OUTPUT_DIR}/checkpoint.json" ]]; then
    log "dcp_output_already_exists output_dir=${DCP_OUTPUT_DIR}"
    exit 0
  fi

  log "checkpoint_ready_running_dcp_conversion"
  COSMOS_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  srun --jobid="${JOB_ID}" --ntasks=1 --gres=gpu:1 \
    bash -lc "
      set -euo pipefail
      cd '${ROOT}'
      export LD_LIBRARY_PATH='${COSMOS_LD_LIBRARY_PATH}':\"\${LD_LIBRARY_PATH:-}\"
      export PYTHONPATH='${ROOT}/external/cosmos-framework'
      export COSMOS3_LOCAL_TOKENIZER_DIR='${LOCAL_TOKENIZER_DIR}'
      export WAN_VAE_PATH='${WAN_VAE_PATH}'
      export COSMOS3_DISABLE_SOUND_FOR_VISION_DCP=true
      export AWS_EC2_METADATA_DISABLED=true
      unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
      '${COSMOS_PYTHON}' -m cosmos_framework.scripts.convert_model_to_dcp \
        -o '${DCP_OUTPUT_DIR}' \
        --checkpoint-path '${CHECKPOINT_DIR}'
    " 2>&1 | tee "${DCP_OUTPUT_DIR}/convert_model_to_dcp.log"

  log "complete output_dir=${DCP_OUTPUT_DIR}"
}

main "$@"
