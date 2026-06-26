#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this wrapper only inside a compute-node srun step. It waits for the current clean-dense generated eval, then trains/evaluates the task-state readout if needed.
EOF
  exit 30
fi

SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299}"
CHECKPOINT_ITER="${CHECKPOINT_ITER:-300}"
printf -v CHECKPOINT_NAME 'iter_%09d' "${CHECKPOINT_ITER}"
EVAL_ROOT="${EVAL_ROOT:-${SFT_ROOT}/eval_full_episode_wam_${CHECKPOINT_NAME}}"
DATASET_MANIFEST="${DATASET_MANIFEST:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_dataset_fix3_v7_user_override_733_rgb_512_20260612/manifest.json}"
READOUT_ROOT="${READOUT_ROOT:-${SFT_ROOT}/task_state_readout_fix3_v7_733_rgb_301f}"
READOUT_CHECKPOINT="${READOUT_CHECKPOINT:-${READOUT_ROOT}/best_model.pt}"
READOUT_EVAL_ROOT="${READOUT_EVAL_ROOT:-${EVAL_ROOT}/task_state_readout_fix3_v7_733}"
READOUT_STEPS="${READOUT_STEPS:-2000}"
READOUT_BATCH_SIZE="${READOUT_BATCH_SIZE:-1}"
READOUT_MAX_EVAL_BATCHES="${READOUT_MAX_EVAL_BATCHES:-40}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
WATCH_TIMEOUT_SECONDS="${WATCH_TIMEOUT_SECONDS:-14400}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"

cd "${ROOT}"
mkdir -p "${READOUT_ROOT}" "${READOUT_EVAL_ROOT}"
{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "step_id=${SLURM_STEP_ID}"
  echo "node_list=${SLURM_JOB_NODELIST:-}"
  echo "sft_root=${SFT_ROOT}"
  echo "checkpoint_iter=${CHECKPOINT_ITER}"
  echo "eval_root=${EVAL_ROOT}"
  echo "dataset_manifest=${DATASET_MANIFEST}"
  echo "readout_root=${READOUT_ROOT}"
  echo "readout_checkpoint=${READOUT_CHECKPOINT}"
  echo "readout_eval_root=${READOUT_EVAL_ROOT}"
  echo "readout_steps=${READOUT_STEPS}"
  echo "boundary=generated-RGB task-state readout diagnostic; not a replacement for closed-loop final-state evidence"
} | tee "${READOUT_ROOT}/watch_readout_manifest.txt"

start="$(date +%s)"
artifact_json="${EVAL_ROOT}/eval_artifact_inspection.json"
input_manifest="${EVAL_ROOT}/eval_input_manifest.json"
while [[ ! -s "${artifact_json}" || ! -s "${input_manifest}" ]]; do
  now="$(date +%s)"
  elapsed=$((now - start))
  if (( elapsed > WATCH_TIMEOUT_SECONDS )); then
    echo "readout_watch_timeout_seconds=${WATCH_TIMEOUT_SECONDS}" >&2
    exit 31
  fi
  echo "generated_eval_not_ready elapsed_seconds=${elapsed}"
  sleep "${CHECK_INTERVAL_SECONDS}"
done

"${PYTHON_BIN}" - "${artifact_json}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
ok = data.get("strict_eval_artifacts_ok")
if ok is not True:
    print(f"strict_eval_artifacts_ok={ok}", file=sys.stderr)
    raise SystemExit(40)
print("strict_eval_artifacts_ok=true")
PY

train_readout=true
if [[ -s "${READOUT_CHECKPOINT}" ]]; then
  train_readout=false
fi

SFT_ROOT="${SFT_ROOT}" \
DATASET_MANIFEST="${DATASET_MANIFEST}" \
EVAL_ROOT="${EVAL_ROOT}" \
READOUT_ROOT="${READOUT_ROOT}" \
READOUT_CHECKPOINT="${READOUT_CHECKPOINT}" \
READOUT_EVAL_ROOT="${READOUT_EVAL_ROOT}" \
TRAIN_READOUT="${train_readout}" \
RUN_EVAL_READOUT=true \
READOUT_STEPS="${READOUT_STEPS}" \
READOUT_BATCH_SIZE="${READOUT_BATCH_SIZE}" \
READOUT_MAX_EVAL_BATCHES="${READOUT_MAX_EVAL_BATCHES}" \
bash "${ROOT}/scripts/slurm/run_cosmos3_300f_task_state_readout_in_allocation.sh"

READOUT_EVAL_ROOT="${READOUT_EVAL_ROOT}" \
bash "${ROOT}/scripts/slurm/watch_cosmos3_readout_profile_in_allocation.sh"
