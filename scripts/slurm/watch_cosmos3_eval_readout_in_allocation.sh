#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this watcher only inside a compute-node srun step. It waits for a strict Cosmos3 eval root and then launches generated-RGB task-state readout from that same allocation.
EOF
  exit 30
fi

SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_20260609_204718}"
EVAL_ROOT="${EVAL_ROOT:-${SFT_ROOT}/eval_full_episode_wam_latest}"
READOUT_ROOT="${READOUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_20260609_204718/task_state_readout_reference_rgb_301f}"
READOUT_CHECKPOINT="${READOUT_CHECKPOINT:-${READOUT_ROOT}/best_model.pt}"
READOUT_EVAL_ROOT="${READOUT_EVAL_ROOT:-${EVAL_ROOT}/task_state_readout_best_current}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
TOTAL_VIDEO_FRAMES="${TOTAL_VIDEO_FRAMES:-301}"
IMAGE_SIZE="${IMAGE_SIZE:-160}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
WATCH_TIMEOUT_SECONDS="${WATCH_TIMEOUT_SECONDS:-10800}"

export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

main() {
  cd "${ROOT}"
  [[ -x "${PYTHON_BIN}" ]] || { echo "missing python: ${PYTHON_BIN}" >&2; exit 2; }
  [[ -s "${READOUT_CHECKPOINT}" ]] || { echo "missing readout checkpoint: ${READOUT_CHECKPOINT}" >&2; exit 3; }

  local artifact_json="${EVAL_ROOT}/eval_artifact_inspection.json"
  local input_manifest="${EVAL_ROOT}/eval_input_manifest.json"
  local start now elapsed
  start="$(date +%s)"
  echo "watch_eval_root=${EVAL_ROOT}"
  echo "watch_eval_artifact=${artifact_json}"
  echo "readout_checkpoint=${READOUT_CHECKPOINT}"
  echo "readout_eval_root=${READOUT_EVAL_ROOT}"
  echo "job_id=${SLURM_JOB_ID}"
  echo "step_id=${SLURM_STEP_ID}"

  while [[ ! -s "${artifact_json}" || ! -s "${input_manifest}" ]]; do
    now="$(date +%s)"
    elapsed=$((now - start))
    if (( elapsed > WATCH_TIMEOUT_SECONDS )); then
      echo "eval_readout_watch_timeout_seconds=${WATCH_TIMEOUT_SECONDS}" >&2
      exit 31
    fi
    echo "eval_not_ready elapsed_seconds=${elapsed}"
    sleep "${CHECK_INTERVAL_SECONDS}"
  done

  echo "eval_ready=${artifact_json}"
  "${PYTHON_BIN}" - "${artifact_json}" <<'PY'
import json
import sys

artifact_path = sys.argv[1]
with open(artifact_path, "r", encoding="utf-8") as f:
    data = json.load(f)
ok = data.get("strict_eval_artifacts_ok")
if ok is not True:
    print(f"strict_eval_artifacts_ok={ok}", file=sys.stderr)
    raise SystemExit(40)
print("strict_eval_artifacts_ok=true")
PY
  mkdir -p "${READOUT_EVAL_ROOT}"
  "${PYTHON_BIN}" "${ROOT}/scripts/world_model/run_cosmos3_full_episode_readout_eval.py" \
    --eval-root "${EVAL_ROOT}" \
    --readout-checkpoint "${READOUT_CHECKPOINT}" \
    --output-root "${READOUT_EVAL_ROOT}" \
    --python-bin "${PYTHON_BIN}" \
    --num-frames "${TOTAL_VIDEO_FRAMES}" \
    --image-size "${IMAGE_SIZE}" \
    --strict \
    2>&1 | tee "${READOUT_EVAL_ROOT}/run_readout_eval.log"
}

main "$@"
