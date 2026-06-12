#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
SRC_DIR="${SRC_DIR:-/tmp/Cosmos3-Nano}"
DST_DIR="${DST_DIR:-${ROOT}/checkpoints/cosmos3/Cosmos3-Nano}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/cosmos3}"

required_files=(
  "checkpoint.json"
  "config.json"
  "generation_config.json"
  "model.safetensors.index.json"
  "model_index.json"
  "preprocessor_config.json"
  "video_preprocessor_config.json"
  "scheduler/scheduler_config.json"
  "text_tokenizer/added_tokens.json"
  "text_tokenizer/chat_template.jinja"
  "text_tokenizer/merges.txt"
  "text_tokenizer/special_tokens_map.json"
  "text_tokenizer/tokenizer.json"
  "text_tokenizer/tokenizer_config.json"
  "text_tokenizer/vocab.json"
  "transformer/config.json"
  "transformer/diffusion_pytorch_model.safetensors.index.json"
  "transformer/diffusion_pytorch_model-00001-of-00007.safetensors"
  "transformer/diffusion_pytorch_model-00002-of-00007.safetensors"
  "transformer/diffusion_pytorch_model-00003-of-00007.safetensors"
  "transformer/diffusion_pytorch_model-00004-of-00007.safetensors"
  "transformer/diffusion_pytorch_model-00005-of-00007.safetensors"
  "transformer/diffusion_pytorch_model-00006-of-00007.safetensors"
  "transformer/diffusion_pytorch_model-00007-of-00007.safetensors"
  "vae/config.json"
  "vae/diffusion_pytorch_model.safetensors"
  "vision_encoder/config.json"
  "vision_encoder/model.safetensors"
)

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

files_ready() {
  local base="$1"
  local rel
  for rel in "${required_files[@]}"; do
    [[ -s "${base}/${rel}" ]] || return 1
  done
  return 0
}

main() {
  mkdir -p "${DST_DIR}" "${LOG_DIR}"
  {
    echo "timestamp=$(date -Is)"
    echo "src_dir=${SRC_DIR}"
    echo "dst_dir=${DST_DIR}"
    echo "physical_reason=sync_completed_cosmos3_required_checkpoint_files_to_compute_visible_project_path"
    echo "boundary=file_staging_only_not_method_or_training_evidence"
  } | tee "${LOG_DIR}/sync_cosmos3_required_from_tmp_manifest.txt"

  until files_ready "${SRC_DIR}"; do
    log "waiting_for_tmp_cosmos3_required_files src_dir=${SRC_DIR}"
    sleep "${POLL_SECONDS}"
  done

  log "tmp_required_files_ready_syncing"
  rsync -a \
    --exclude='.cache' \
    --exclude='assets' \
    --exclude='images' \
    --exclude='sound_tokenizer' \
    "${SRC_DIR}/" "${DST_DIR}/"

  if ! files_ready "${DST_DIR}"; then
    log "sync_failed_missing_required_files dst_dir=${DST_DIR}"
    exit 2
  fi

  log "sync_complete dst_dir=${DST_DIR}"
}

main "$@"
