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

RUN_GROUP="${RUN_GROUP:-cosmos_forward_eval}"
RUN_NAME="${RUN_NAME:-eval01}"
RUN_DIR="${RUN_DIR:-${ROOT}/experiments/maniskill/runs/02_joint_training/${RUN_GROUP}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/02_joint_training/${RUN_GROUP}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"
COSMOS_PYTHON="${COSMOS_PYTHON:-${ROOT}/.venv_cosmos313/bin/python}"
COSMOS_FRAMEWORK="${COSMOS_FRAMEWORK:-${ROOT}/external/cosmos-framework}"
COSMOS_FAST_IMPORT_PATCH_DIR="${COSMOS_FAST_IMPORT_PATCH_DIR:-${ROOT}/scripts/world_model/cosmos_fast_import_sitecustomize}"
CONDITION_RUN_DIR="${CONDITION_RUN_DIR:-${ROOT}/experiments/maniskill/runs/02_joint_training/joint_cosmos_condition/overfit09}"
CONDITION_ROOT="${CONDITION_ROOT:-${CONDITION_RUN_DIR}/condition_root}"
CONDITION_SPLIT="${CONDITION_SPLIT:-val}"
SAMPLE_INDEX="${SAMPLE_INDEX:-1}"
SAMPLE_UUID="${SAMPLE_UUID:-}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-${ROOT}/experiments/maniskill/runs/02_joint_training/cosmos_sft_overfit/overfit16}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${TRAIN_RUN_DIR}/cosmos_output/cosmos3/sft/joint_cosmos_sft_overfit16/checkpoints/iter_000000005}"
CONFIG_FILE="${CONFIG_FILE:-${TRAIN_RUN_DIR}/cosmos_output/cosmos3/sft/joint_cosmos_sft_overfit16/config.yaml}"
COSMOS3_LOCAL_TOKENIZER_DIR="${COSMOS3_LOCAL_TOKENIZER_DIR:-${ROOT}/checkpoints/cosmos3/Cosmos3-Nano}"
WAN_VAE_PATH="${WAN_VAE_PATH:-${ROOT}/checkpoints/cosmos3/wan22_vae/Wan2.2_VAE.pth}"
SEED="${SEED:-0}"
NUM_STEPS="${NUM_STEPS:-8}"
GUIDANCE="${GUIDANCE:-1.0}"
SHIFT="${SHIFT:-10.0}"
SIGMA_MAX="${SIGMA_MAX:-80.0}"
RESOLUTION="${RESOLUTION:-480}"
IMAGE_SIZE="${IMAGE_SIZE:-480}"
ACTION_DOMAIN="${ACTION_DOMAIN:-maniskill_peg_insertion}"
COSMOS3_ACTION_PROMPT_STYLE="${COSMOS3_ACTION_PROMPT_STYLE:-plain_sft}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"

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
CONDITION_JSONL="${CONDITION_ROOT}/${CONDITION_SPLIT}/video_dataset_file.jsonl"
for required in \
  "${COSMOS_PYTHON}" \
  "${CONDITION_JSONL}" \
  "${CHECKPOINT_PATH}/model/.metadata" \
  "${CONFIG_FILE}" \
  "${WAN_VAE_PATH}"; do
  if [[ ! -e "${required}" ]]; then
    echo "refusing_missing_required_input=true" >&2
    echo "missing=${required}" >&2
    exit 45
  fi
done

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
cd "${ROOT}"

export ROOT
export COSMOS3_LOCAL_TOKENIZER_DIR
export COSMOS_SKIP_PACKAGE_DISTRIBUTION_SCAN="${COSMOS_SKIP_PACKAGE_DISTRIBUTION_SCAN:-1}"
export COSMOS3_ACTION_PROMPT_STYLE
export PYTHONPATH="${COSMOS_FAST_IMPORT_PATCH_DIR}:${COSMOS_FRAMEWORK}:${PYTHONPATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export DISPLAY=
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export AWS_EC2_METADATA_DISABLED=true
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
refresh_cosmos_ld_library_path

SAMPLE_JSONL="${RUN_DIR}/inference_sample.jsonl"
ACTION_JSON="${RUN_DIR}/action_state_32.json"
SAMPLE_META_JSON="${RUN_DIR}/selected_condition_row.json"
INFERENCE_OUTPUT_DIR="${RUN_DIR}/inference_output"
METRICS_DIR="${RUN_DIR}/reconstruction_metrics"

{
  echo "timestamp=$(date -Is)"
  echo "phase=02_joint_training"
  echo "stage=cosmos_forward_dynamics_eval"
  echo "run_group=${RUN_GROUP}"
  echo "run_name=${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  echo "log_file=${LOG_FILE}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "slurm_step_id=${SLURM_STEP_ID}"
  echo "node=$(hostname)"
  echo "condition_run_dir=${CONDITION_RUN_DIR}"
  echo "condition_root=${CONDITION_ROOT}"
  echo "condition_split=${CONDITION_SPLIT}"
  echo "sample_index=${SAMPLE_INDEX}"
  echo "sample_uuid=${SAMPLE_UUID}"
  echo "train_run_dir=${TRAIN_RUN_DIR}"
  echo "checkpoint_path=${CHECKPOINT_PATH}"
  echo "config_file=${CONFIG_FILE}"
  echo "cosmos_framework=${COSMOS_FRAMEWORK}"
  echo "cosmos_fast_import_patch_dir=${COSMOS_FAST_IMPORT_PATCH_DIR}"
  echo "cosmos_python=${COSMOS_PYTHON}"
  echo "cosmos3_local_tokenizer_dir=${COSMOS3_LOCAL_TOKENIZER_DIR}"
  echo "wan_vae_path=${WAN_VAE_PATH}"
  echo "seed=${SEED}"
  echo "num_steps=${NUM_STEPS}"
  echo "guidance=${GUIDANCE}"
  echo "shift=${SHIFT}"
  echo "sigma_max=${SIGMA_MAX}"
  echo "resolution=${RESOLUTION}"
  echo "image_size=${IMAGE_SIZE}"
  echo "action_domain=${ACTION_DOMAIN}"
  echo "cosmos3_action_prompt_style=${COSMOS3_ACTION_PROMPT_STYLE}"
  echo "method_evidence_allowed=false"
  echo "eval_scope=diagnostic_forward_dynamics_generation"
  echo "uses_toy_model=false"
  echo "boundary=Generates a Cosmos future-state video from an overfit condition row. This is not closed-loop control or insertion success evidence."
} | tee "${RUN_DIR}/manifest.txt" | tee -a "${LOG_FILE}"

nvidia-smi 2>&1 | tee -a "${LOG_FILE}" || true

echo "prepare_inference_sample_start=$(date -Is)" | tee -a "${LOG_FILE}"
"${COSMOS_PYTHON}" - <<'PY' \
  "${CONDITION_JSONL}" \
  "${SAMPLE_INDEX}" \
  "${SAMPLE_UUID}" \
  "${SAMPLE_JSONL}" \
  "${ACTION_JSON}" \
  "${SAMPLE_META_JSON}" \
  "${ACTION_DOMAIN}" \
  "${IMAGE_SIZE}" \
  "${RESOLUTION}" \
  "${SEED}" \
  "${NUM_STEPS}" \
  "${GUIDANCE}" \
  "${SHIFT}" \
  "${SIGMA_MAX}"
import json
import sys
from pathlib import Path

import numpy as np

(
    condition_jsonl,
    sample_index,
    sample_uuid,
    sample_jsonl,
    action_json,
    sample_meta_json,
    action_domain,
    image_size,
    resolution,
    seed,
    num_steps,
    guidance,
    shift,
    sigma_max,
) = sys.argv[1:]

rows = [json.loads(line) for line in Path(condition_jsonl).read_text().splitlines() if line.strip()]
if sample_uuid:
    matches = [row for row in rows if row.get("source_uuid") == sample_uuid or row.get("uuid") == sample_uuid]
    if len(matches) != 1:
        raise SystemExit(f"expected_one_sample_uuid_match={sample_uuid} got={len(matches)}")
    row = matches[0]
else:
    idx = int(sample_index)
    if idx < 0 or idx >= len(rows):
        raise SystemExit(f"sample_index_out_of_range={idx} count={len(rows)}")
    row = rows[idx]

actions = np.load(row["action_path"], allow_pickle=False).astype(np.float32, copy=False)
if actions.ndim != 2:
    raise SystemExit(f"bad_action_rank={actions.ndim}")
if actions.shape[1] != int(row.get("raw_action_dim", 32)):
    raise SystemExit(f"bad_action_dim={actions.shape[1]} expected={row.get('raw_action_dim')}")
Path(action_json).write_text(json.dumps({"action": actions.tolist()}, sort_keys=True) + "\n")

caption = row.get("t2w_windows", [{}])[0].get("caption") or (
    "Observed PegInsertionSide history. Predict future RGB from the action/state channel."
)
sample_name = f"forward_{row['source_uuid']}"
sample = {
    "name": sample_name,
    "model_mode": "forward_dynamics",
    "prompt": caption,
    "vision_path": row["vision_path"],
    "action_path": str(Path(action_json).resolve()),
    "domain_name": action_domain,
    "view_point": "ego_view",
    "action_chunk_size": int(row["action_chunk_size"]),
    "condition_frame_indexes_vision": row.get("condition_frame_indexes_vision", []),
    "condition_frame_indexes_action": row.get("condition_frame_indexes_action", []),
    "fps": int(row.get("fps", 30)),
    "num_frames": int(row.get("num_video_frames", row["action_chunk_size"] + 1)),
    "image_size": int(image_size),
    "resolution": str(resolution),
    "aspect_ratio": "1,1",
    "video_save_quality": 10,
    "image_save_quality": 95,
    "num_steps": int(num_steps),
    "guidance": float(guidance),
    "shift": float(shift),
    "sigma_max": float(sigma_max),
    "normalize_cfg": False,
    "negative_prompt": "",
    "seed": int(seed),
    "extra": {
        "condition_source_uuid": row.get("source_uuid"),
        "condition_uuid": row.get("uuid"),
        "condition_policy": row.get("condition_policy"),
        "condition_prefix_frames": row.get("condition_prefix_frames"),
        "source_video_frames": row.get("source_video_frames"),
        "diagnostic_only": True,
        "method_evidence_allowed": False,
        "reference_window_contract": "official_action_inference_reads_first_action_chunk_plus_one_frames",
    },
}
Path(sample_jsonl).write_text(json.dumps(sample, sort_keys=True) + "\n")
Path(sample_meta_json).write_text(json.dumps({"condition_row": row, "inference_sample": sample}, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    "selected_source_uuid": row.get("source_uuid"),
    "selected_uuid": row.get("uuid"),
    "sample_name": sample_name,
    "action_shape": list(actions.shape),
    "vision_path": row.get("vision_path"),
    "condition_prefix_frames": row.get("condition_prefix_frames"),
}, sort_keys=True))
PY

SAMPLE_NAME="$("${COSMOS_PYTHON}" - <<'PY' "${SAMPLE_JSONL}"
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["name"])
PY
)"
PREDICTION_VIDEO="${INFERENCE_OUTPUT_DIR}/${SAMPLE_NAME}/vision.mp4"
REFERENCE_VIDEO="$("${COSMOS_PYTHON}" - <<'PY' "${SAMPLE_JSONL}"
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["vision_path"])
PY
)"
REFERENCE_FRAMES="$("${COSMOS_PYTHON}" - <<'PY' "${SAMPLE_JSONL}"
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["num_frames"])
PY
)"

echo "cosmos_forward_inference_start=$(date -Is)" | tee -a "${LOG_FILE}"
inference_cmd=(
  "${COSMOS_PYTHON}"
  -u
  -m cosmos_framework.scripts.inference
  --parallelism-preset=throughput
  --dp-shard-size=1
  --dp-replicate-size=1
  --cp-size=1
  --cfgp-size=1
  --no-use-torch-compile
  --no-use-cuda-graphs
  --no-guardrails
  -i "${SAMPLE_JSONL}"
  -o "${INFERENCE_OUTPUT_DIR}"
  --checkpoint-path "${CHECKPOINT_PATH}"
  --config-file "${CONFIG_FILE}"
  --seed "${SEED}"
)
if [[ -n "${MAX_MODEL_LEN}" ]]; then
  inference_cmd+=(--max-model-len "${MAX_MODEL_LEN}")
fi
if [[ -n "${MAX_NUM_SEQS}" ]]; then
  inference_cmd+=(--max-num-seqs "${MAX_NUM_SEQS}")
fi
{
  printf 'cosmos_forward_inference_invocation='
  printf '%q ' "${inference_cmd[@]}"
  printf '\n'
} >> "${LOG_FILE}"
set +e
(
  cd "${COSMOS_FRAMEWORK}"
  "${inference_cmd[@]}" >> "${LOG_FILE}" 2>&1
)
inference_status="$?"
set -e
echo "cosmos_forward_inference_exit_status=${inference_status}" | tee -a "${LOG_FILE}"
if [[ "${inference_status}" -ne 0 ]]; then
  {
    echo "cosmos_forward_eval_status=failed"
    echo "failure_stage=inference"
    echo "completed_at=$(date -Is)"
  } | tee "${RUN_DIR}/classification.txt" | tee -a "${LOG_FILE}"
  exit "${inference_status}"
fi
if [[ ! -s "${PREDICTION_VIDEO}" ]]; then
  {
    echo "cosmos_forward_eval_status=failed"
    echo "failure_stage=missing_prediction_video"
    echo "prediction_video=${PREDICTION_VIDEO}"
    echo "completed_at=$(date -Is)"
  } | tee "${RUN_DIR}/classification.txt" | tee -a "${LOG_FILE}"
  exit 61
fi

echo "reconstruction_eval_start=$(date -Is)" | tee -a "${LOG_FILE}"
set +e
"${COSMOS_PYTHON}" scripts/world_model/evaluate_cosmos3_rollout_reconstruction.py \
  --reference-video "${REFERENCE_VIDEO}" \
  --prediction-video "${PREDICTION_VIDEO}" \
  --output-dir "${METRICS_DIR}" \
  --reference-start 0 \
  --prediction-start 0 \
  --max-frames "${REFERENCE_FRAMES}" \
  --sheet-frames 8 \
  --allow-length-mismatch \
  --allow-truncation >> "${LOG_FILE}" 2>&1
metrics_status="$?"
set -e
echo "reconstruction_eval_exit_status=${metrics_status}" | tee -a "${LOG_FILE}"
if [[ "${metrics_status}" -ne 0 ]]; then
  {
    echo "cosmos_forward_eval_status=failed"
    echo "failure_stage=reconstruction_metrics"
    echo "prediction_video=${PREDICTION_VIDEO}"
    echo "reference_video=${REFERENCE_VIDEO}"
    echo "completed_at=$(date -Is)"
  } | tee "${RUN_DIR}/classification.txt" | tee -a "${LOG_FILE}"
  exit "${metrics_status}"
fi

{
  echo "cosmos_forward_eval_status=complete"
  echo "prediction_video=${PREDICTION_VIDEO}"
  echo "reference_video=${REFERENCE_VIDEO}"
  echo "metrics_json=${METRICS_DIR}/metrics.json"
  echo "comparison_sheet=${METRICS_DIR}/reconstruction_comparison_sheet.png"
  echo "sample_jsonl=${SAMPLE_JSONL}"
  echo "selected_condition_row=${SAMPLE_META_JSON}"
  echo "method_evidence_allowed=false"
  echo "closed_loop_evidence=false"
  echo "completed_at=$(date -Is)"
} | tee "${RUN_DIR}/classification.txt" | tee -a "${LOG_FILE}"
