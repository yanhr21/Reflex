#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run inside a compute-node srun step from a tmux-held Slurm allocation.
EOF
  exit 30
fi

RUN_GROUP="${RUN_GROUP:-cosmos_sft_overfit}"
RUN_NAME="${RUN_NAME:-overfit01}"
RUN_DIR="${RUN_DIR:-${ROOT}/experiments/maniskill/runs/02_joint_training/${RUN_GROUP}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/02_joint_training/${RUN_GROUP}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"
COSMOS_PYTHON="${COSMOS_PYTHON:-${ROOT}/.venv_cosmos313/bin/python}"
COSMOS_TORCHRUN="${COSMOS_TORCHRUN:-${ROOT}/.venv_cosmos313/bin/torchrun}"
COSMOS_FRAMEWORK="${COSMOS_FRAMEWORK:-${ROOT}/external/cosmos-framework}"
COSMOS_FAST_IMPORT_PATCH_DIR="${COSMOS_FAST_IMPORT_PATCH_DIR:-${ROOT}/scripts/world_model/cosmos_fast_import_sitecustomize}"
CONDITION_RUN_DIR="${CONDITION_RUN_DIR:-${ROOT}/experiments/maniskill/runs/02_joint_training/joint_cosmos_condition/overfit05}"
CONDITION_ROOT="${CONDITION_ROOT:-${CONDITION_RUN_DIR}/condition_root}"
ACTIVE_COSMOS_ROOT="${ACTIVE_COSMOS_ROOT:-${ROOT}/experiments/maniskill/cosmos3_checkpoint/vision_sft_droid_policy_full1000_rgb_300step_wam}"
LATEST_CKPT="$(tr -d '[:space:]' < "${ACTIVE_COSMOS_ROOT}/checkpoints/latest_checkpoint.txt")"
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-${ACTIVE_COSMOS_ROOT}/checkpoints/${LATEST_CKPT}}"
WAN_VAE_PATH="${WAN_VAE_PATH:-${ROOT}/checkpoints/cosmos3/wan22_vae/Wan2.2_VAE.pth}"
COSMOS3_LOCAL_TOKENIZER_DIR="${COSMOS3_LOCAL_TOKENIZER_DIR:-${ROOT}/checkpoints/cosmos3/Cosmos3-Nano}"
TOML_FILE="${TOML_FILE:-${COSMOS_FRAMEWORK}/examples/toml/sft_config/vision_sft_nano.toml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-50741}"
MAX_ITER="${MAX_ITER:-5}"
SAVE_ITER="${SAVE_ITER:-5}"
VALIDATION_ITER="${VALIDATION_ITER:-5}"
MAX_VAL_ITER="${MAX_VAL_ITER:-2}"
RUN_VALIDATION="${RUN_VALIDATION:-true}"
RUN_VALIDATION_ON_START="${RUN_VALIDATION_ON_START:-false}"
COSMOS_JOB_NAME="${COSMOS_JOB_NAME:-joint_cosmos_sft_overfit01}"
COSMOS_OUTPUT_ROOT="${COSMOS_OUTPUT_ROOT:-${RUN_DIR}/cosmos_output}"
GRAD_ACCUM_ITER="${GRAD_ACCUM_ITER:-2}"
MODEL_COMPILE_ENABLED="${MODEL_COMPILE_ENABLED:-true}"
ENABLE_LORA="${ENABLE_LORA:-false}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen}"
USE_TORCHRUN="${USE_TORCHRUN:-true}"

LIBFFI_COMPAT_DIR="${LIBFFI_COMPAT_DIR:-${ROOT}/.venv_cosmos313/lib_compat}"
cosmos_nvidia_lib_dirs() {
  local venv
  local venv_bin
  venv_bin="$(dirname "${COSMOS_PYTHON}")"
  venv="$(dirname "${venv_bin}")"
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

case "${RUN_DIR}" in
  "${ROOT}/experiments/maniskill/runs/02_joint_training/"*) ;;
  *)
    echo "refusing_output_dir_outside_02_joint_training=true" >&2
    echo "run_dir=${RUN_DIR}" >&2
    exit 41
    ;;
esac
if [[ -e "${RUN_DIR}" ]] && [[ -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing_existing_nonempty_run_dir=true" >&2
  echo "run_dir=${RUN_DIR}" >&2
  exit 42
fi
if [[ ! -s "${CONDITION_RUN_DIR}/condition_preflight.json" ]]; then
  echo "refusing_missing_condition_preflight=true" >&2
  echo "condition_run_dir=${CONDITION_RUN_DIR}" >&2
  exit 43
fi
if ! grep -q '"strict_alignment_ok"[[:space:]]*:[[:space:]]*true' "${CONDITION_RUN_DIR}/condition_preflight.json"; then
  echo "refusing_condition_preflight_not_strict_ok=true" >&2
  echo "condition_preflight=${CONDITION_RUN_DIR}/condition_preflight.json" >&2
  exit 44
fi
if [[ ! -s "${CONDITION_ROOT}/train/video_dataset_file.jsonl" || ! -s "${CONDITION_ROOT}/val/video_dataset_file.jsonl" ]]; then
  echo "refusing_missing_cosmos_recipe_jsonl=true" >&2
  echo "condition_root=${CONDITION_ROOT}" >&2
  exit 45
fi
for required in "${COSMOS_PYTHON}" "${COSMOS_TORCHRUN}" "${TOML_FILE}" "${WAN_VAE_PATH}" "${BASE_CHECKPOINT_PATH}/model/.metadata"; do
  if [[ ! -e "${required}" ]]; then
    echo "refusing_missing_required_input=true" >&2
    echo "missing=${required}" >&2
    exit 46
  fi
done

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${COSMOS_OUTPUT_ROOT}"
cd "${ROOT}"

export ROOT
export DATASET_PATH="${CONDITION_ROOT}"
export BASE_CHECKPOINT_PATH
export WAN_VAE_PATH
export COSMOS3_LOCAL_TOKENIZER_DIR
export IMAGINAIRE_OUTPUT_ROOT="${COSMOS_OUTPUT_ROOT}"
export COSMOS_SKIP_PACKAGE_DISTRIBUTION_SCAN="${COSMOS_SKIP_PACKAGE_DISTRIBUTION_SCAN:-1}"
export PYTHONPATH="${COSMOS_FAST_IMPORT_PATCH_DIR}:${COSMOS_FRAMEWORK}:${PYTHONPATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export DISPLAY=
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export AWS_EC2_METADATA_DISABLED=true
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
refresh_cosmos_ld_library_path

{
  echo "timestamp=$(date -Is)"
  echo "phase=02_joint_training"
  echo "stage=cosmos_sft_overfit"
  echo "run_group=${RUN_GROUP}"
  echo "run_name=${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  echo "log_file=${LOG_FILE}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "slurm_step_id=${SLURM_STEP_ID}"
  echo "node=$(hostname)"
  echo "condition_run_dir=${CONDITION_RUN_DIR}"
  echo "condition_root=${CONDITION_ROOT}"
  echo "cosmos_framework=${COSMOS_FRAMEWORK}"
  echo "cosmos_fast_import_patch_dir=${COSMOS_FAST_IMPORT_PATCH_DIR}"
  echo "cosmos_skip_package_distribution_scan=${COSMOS_SKIP_PACKAGE_DISTRIBUTION_SCAN}"
  echo "cosmos_python=${COSMOS_PYTHON}"
  echo "cosmos_torchrun=${COSMOS_TORCHRUN}"
  echo "toml_file=${TOML_FILE}"
  echo "base_checkpoint_path=${BASE_CHECKPOINT_PATH}"
  echo "wan_vae_path=${WAN_VAE_PATH}"
  echo "cosmos3_local_tokenizer_dir=${COSMOS3_LOCAL_TOKENIZER_DIR}"
  echo "cosmos_output_root=${COSMOS_OUTPUT_ROOT}"
  echo "nproc_per_node=${NPROC_PER_NODE}"
  echo "max_iter=${MAX_ITER}"
  echo "save_iter=${SAVE_ITER}"
  echo "validation_iter=${VALIDATION_ITER}"
  echo "max_val_iter=${MAX_VAL_ITER}"
  echo "run_validation=${RUN_VALIDATION}"
  echo "run_validation_on_start=${RUN_VALIDATION_ON_START}"
  echo "grad_accum_iter=${GRAD_ACCUM_ITER}"
  echo "model_compile_enabled=${MODEL_COMPILE_ENABLED}"
  echo "enable_lora=${ENABLE_LORA}"
  echo "lora_rank=${LORA_RANK}"
  echo "lora_alpha=${LORA_ALPHA}"
  echo "lora_target_modules=${LORA_TARGET_MODULES}"
  echo "use_torchrun=${USE_TORCHRUN}"
  echo "method_evidence_allowed=false"
  echo "training_started=true"
  echo "training_scope=short_cosmos_sft_overfit_smoke"
  echo "uses_toy_model=false"
} | tee "${RUN_DIR}/manifest.txt" | tee -a "${LOG_FILE}"

nvidia-smi 2>&1 | tee -a "${LOG_FILE}" || true

echo "cosmos_train_start=$(date -Is)" | tee -a "${LOG_FILE}"
cosmos_overrides=(
  "job.name=${COSMOS_JOB_NAME}"
  "checkpoint.load_path=${BASE_CHECKPOINT_PATH}"
  "checkpoint.save_iter=${SAVE_ITER}"
  "trainer.max_iter=${MAX_ITER}"
  "trainer.logging_iter=1"
  "trainer.validation_iter=${VALIDATION_ITER}"
  "trainer.max_val_iter=${MAX_VAL_ITER}"
  "trainer.run_validation=${RUN_VALIDATION}"
  "trainer.run_validation_on_start=${RUN_VALIDATION_ON_START}"
  "trainer.grad_accum_iter=${GRAD_ACCUM_ITER}"
  "model.config.compile.enabled=${MODEL_COMPILE_ENABLED}"
  "dataloader_train.dataloader.num_workers=1"
  "dataloader_train.dataloader.prefetch_factor=1"
  "dataloader_val.dataloader.num_workers=1"
  "dataloader_val.dataloader.prefetch_factor=1"
  "model.config.tokenizer.vae_path=${WAN_VAE_PATH}"
)
if [[ "${ENABLE_LORA}" == "true" ]]; then
  cosmos_overrides+=(
    "model.config.lora_enabled=true"
    "model.config.lora_rank=${LORA_RANK}"
    "model.config.lora_alpha=${LORA_ALPHA}"
    "model.config.lora_target_modules='${LORA_TARGET_MODULES}'"
    "model.config.ema.enabled=false"
    "optimizer.keys_to_select=[lora_]"
    "checkpoint.keys_to_skip_loading=[net_ema.,lora_]"
  )
fi
if [[ "${USE_TORCHRUN}" == "true" ]]; then
  train_cmd=(
    "${COSMOS_TORCHRUN}"
    "--nproc_per_node=${NPROC_PER_NODE}"
    "--master_port=${MASTER_PORT}"
    -m cosmos_framework.scripts.train
    "--sft-toml=${TOML_FILE}"
    --
    "${cosmos_overrides[@]}"
  )
else
  export RANK="${RANK:-0}"
  export WORLD_SIZE="${WORLD_SIZE:-1}"
  export LOCAL_RANK="${LOCAL_RANK:-0}"
  export LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-1}"
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT}"
  train_cmd=(
    "${COSMOS_PYTHON}"
    -u
    -m cosmos_framework.scripts.train
    "--sft-toml=${TOML_FILE}"
    --
    "${cosmos_overrides[@]}"
  )
fi
{
  printf 'cosmos_train_invocation='
  printf '%q ' "${train_cmd[@]}"
  printf '\n'
} >> "${LOG_FILE}"
set +e
(
  cd "${COSMOS_FRAMEWORK}"
  "${train_cmd[@]}" >> "${LOG_FILE}" 2>&1 &
  train_pid="$!"
  echo "cosmos_train_pid=${train_pid}" >> "${LOG_FILE}"
  wait "${train_pid}"
)
train_status="$?"
set -e
echo "cosmos_train_exit_status=${train_status}" | tee -a "${LOG_FILE}"
if [[ "${train_status}" -ne 0 ]]; then
  echo "cosmos_sft_overfit_status=failed" | tee "${RUN_DIR}/classification.txt" | tee -a "${LOG_FILE}"
  exit "${train_status}"
fi

{
  echo "cosmos_sft_overfit_status=complete"
  echo "cosmos_output_root=${COSMOS_OUTPUT_ROOT}"
  echo "completed_at=$(date -Is)"
} | tee "${RUN_DIR}/classification.txt" | tee -a "${LOG_FILE}"
