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

RUN_GROUP="${RUN_GROUP:-cosmos_startup_diag}"
RUN_NAME="${RUN_NAME:-import_trace01}"
RUN_DIR="${RUN_DIR:-${ROOT}/experiments/maniskill/runs/02_joint_training/${RUN_GROUP}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/02_joint_training/${RUN_GROUP}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"
COSMOS_FRAMEWORK="${COSMOS_FRAMEWORK:-${ROOT}/external/cosmos-framework}"
COSMOS_FAST_IMPORT_PATCH_DIR="${COSMOS_FAST_IMPORT_PATCH_DIR:-${ROOT}/scripts/world_model/cosmos_fast_import_sitecustomize}"
COSMOS_PYTHON="${COSMOS_PYTHON:-${ROOT}/.venv_cosmos313/bin/python}"
CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/maniskill/runs/02_joint_training/joint_cosmos_condition/overfit09/condition_root}"
ACTIVE_COSMOS_ROOT="${ACTIVE_COSMOS_ROOT:-${ROOT}/experiments/maniskill/cosmos3_checkpoint/vision_sft_droid_policy_full1000_rgb_300step_wam}"
LATEST_CKPT="$(tr -d '[:space:]' < "${ACTIVE_COSMOS_ROOT}/checkpoints/latest_checkpoint.txt")"
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-${ACTIVE_COSMOS_ROOT}/checkpoints/${LATEST_CKPT}}"
WAN_VAE_PATH="${WAN_VAE_PATH:-${ROOT}/checkpoints/cosmos3/wan22_vae/Wan2.2_VAE.pth}"
COSMOS3_LOCAL_TOKENIZER_DIR="${COSMOS3_LOCAL_TOKENIZER_DIR:-${ROOT}/checkpoints/cosmos3/Cosmos3-Nano}"
TOML_FILE="${TOML_FILE:-${COSMOS_FRAMEWORK}/examples/toml/sft_config/vision_sft_nano.toml}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"

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

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
cd "${ROOT}"

export ROOT
export DATASET_PATH="${CONDITION_ROOT}"
export BASE_CHECKPOINT_PATH
export WAN_VAE_PATH
export COSMOS3_LOCAL_TOKENIZER_DIR
export IMAGINAIRE_OUTPUT_ROOT="${RUN_DIR}/cosmos_output"
export COSMOS_SKIP_PACKAGE_DISTRIBUTION_SCAN="${COSMOS_SKIP_PACKAGE_DISTRIBUTION_SCAN:-1}"
export PYTHONPATH="${COSMOS_FAST_IMPORT_PATCH_DIR}:${COSMOS_FRAMEWORK}:${PYTHONPATH:-}"
export AWS_EC2_METADATA_DISABLED=true
export PYTHONFAULTHANDLER=1
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy

{
  echo "timestamp=$(date -Is)"
  echo "phase=02_joint_training"
  echo "stage=cosmos_startup_import_diag"
  echo "run_group=${RUN_GROUP}"
  echo "run_name=${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  echo "log_file=${LOG_FILE}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "slurm_step_id=${SLURM_STEP_ID}"
  echo "node=$(hostname)"
  echo "condition_root=${CONDITION_ROOT}"
  echo "cosmos_framework=${COSMOS_FRAMEWORK}"
  echo "cosmos_fast_import_patch_dir=${COSMOS_FAST_IMPORT_PATCH_DIR}"
  echo "cosmos_skip_package_distribution_scan=${COSMOS_SKIP_PACKAGE_DISTRIBUTION_SCAN}"
  echo "cosmos_python=${COSMOS_PYTHON}"
  echo "toml_file=${TOML_FILE}"
  echo "base_checkpoint_path=${BASE_CHECKPOINT_PATH}"
  echo "wan_vae_path=${WAN_VAE_PATH}"
  echo "cosmos3_local_tokenizer_dir=${COSMOS3_LOCAL_TOKENIZER_DIR}"
  echo "timeout_seconds=${TIMEOUT_SECONDS}"
  echo "method_evidence_allowed=false"
  echo "training_started=false"
  echo "uses_toy_model=false"
} | tee "${RUN_DIR}/manifest.txt" | tee -a "${LOG_FILE}"

DIAG_SCRIPT="${RUN_DIR}/diag_load_config.py"
cat > "${DIAG_SCRIPT}" <<'PY'
import os

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

cfg = load_experiment_from_toml(
    "examples/toml/sft_config/vision_sft_nano.toml",
    extra_overrides=[
        "job.name=diag_import_trace",
        f"checkpoint.load_path={os.environ['BASE_CHECKPOINT_PATH']}",
        "trainer.max_iter=1",
        "model.config.compile.enabled=false",
        f"model.config.tokenizer.vae_path={os.environ['WAN_VAE_PATH']}",
    ],
)
print("CONFIG_LOADED", cfg.job.name, flush=True)
PY

echo "diag_start=$(date -Is)" | tee -a "${LOG_FILE}"
set +e
(
  cd "${COSMOS_FRAMEWORK}"
  timeout "${TIMEOUT_SECONDS}" \
    strace -f -tt -T -o "${RUN_DIR}/strace.txt" \
    "${COSMOS_PYTHON}" -X importtime -u "${DIAG_SCRIPT}" \
    > "${RUN_DIR}/importtime.stdout" \
    2> "${RUN_DIR}/importtime.stderr"
)
diag_status="$?"
set -e
echo "diag_exit_status=${diag_status}" | tee -a "${LOG_FILE}"
echo "diag_end=$(date -Is)" | tee -a "${LOG_FILE}"

{
  if [[ "${diag_status}" -eq 0 ]]; then
    echo "cosmos_startup_import_diag_status=complete"
  else
    echo "cosmos_startup_import_diag_status=failed_or_timeout"
  fi
  echo "diag_exit_status=${diag_status}"
  echo "diag_script=${DIAG_SCRIPT}"
  echo "strace=${RUN_DIR}/strace.txt"
  echo "importtime_stdout=${RUN_DIR}/importtime.stdout"
  echo "importtime_stderr=${RUN_DIR}/importtime.stderr"
  echo "completed_at=$(date -Is)"
} | tee "${RUN_DIR}/classification.txt" | tee -a "${LOG_FILE}"

exit "${diag_status}"
