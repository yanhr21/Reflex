#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this watcher only inside a compute-node srun step. It waits for a saved SFT checkpoint and then launches strict full-episode eval from that same allocation.
EOF
  exit 30
fi

SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_20260609_204718}"
JOB_NAME="${JOB_NAME:-vision_sft_droid_policy_full1000_rgb_300step_wam}"
RUN_DIR="${RUN_DIR:-${SFT_ROOT}/outputs/cosmos3/sft/${JOB_NAME}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${RUN_DIR}/checkpoints}"
CHECKPOINT_ITER="${CHECKPOINT_ITER:-600}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
WATCH_TIMEOUT_SECONDS="${WATCH_TIMEOUT_SECONDS:-7200}"
CHECKPOINT_STABLE_SECONDS="${CHECKPOINT_STABLE_SECONDS:-60}"
MASTER_PORT="${MASTER_PORT:-50219}"

printf -v CHECKPOINT_NAME 'iter_%09d' "${CHECKPOINT_ITER}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT_ROOT}/${CHECKPOINT_NAME}}"
EVAL_ROOT="${EVAL_ROOT:-${SFT_ROOT}/eval_full_episode_wam_${CHECKPOINT_NAME}}"

main() {
  cd "${ROOT}"
  local start now elapsed
  start="$(date +%s)"
  echo "watch_checkpoint=${CHECKPOINT_PATH}"
  echo "eval_root=${EVAL_ROOT}"
  echo "job_id=${SLURM_JOB_ID}"
  echo "step_id=${SLURM_STEP_ID}"
  while [[ ! -d "${CHECKPOINT_PATH}/model" || ! -s "${CHECKPOINT_PATH}/model/.metadata" ]]; do
    now="$(date +%s)"
    elapsed=$((now - start))
    if (( elapsed > WATCH_TIMEOUT_SECONDS )); then
      echo "checkpoint_watch_timeout_seconds=${WATCH_TIMEOUT_SECONDS}" >&2
      exit 31
    fi
    echo "checkpoint_not_ready elapsed_seconds=${elapsed}"
    sleep "${CHECK_INTERVAL_SECONDS}"
  done

  local stable_start previous_signature current_signature
  stable_start="$(date +%s)"
  previous_signature=""
  while true; do
    current_signature="$(find "${CHECKPOINT_PATH}" -type f -printf '%P %s\n' | sort)"
    now="$(date +%s)"
    elapsed=$((now - start))
    if (( elapsed > WATCH_TIMEOUT_SECONDS )); then
      echo "checkpoint_stability_timeout_seconds=${WATCH_TIMEOUT_SECONDS}" >&2
      exit 32
    fi
    if [[ -n "${current_signature}" && "${current_signature}" == "${previous_signature}" ]]; then
      if (( now - stable_start >= CHECKPOINT_STABLE_SECONDS )); then
        break
      fi
    else
      stable_start="${now}"
      previous_signature="${current_signature}"
    fi
    echo "checkpoint_waiting_for_stable_files elapsed_seconds=${elapsed} stable_seconds=$((now - stable_start))"
    sleep "${CHECK_INTERVAL_SECONDS}"
  done

  echo "checkpoint_ready=${CHECKPOINT_PATH}"
  CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
  EVAL_ROOT="${EVAL_ROOT}" \
  MASTER_PORT="${MASTER_PORT}" \
  bash "${ROOT}/scripts/slurm/run_cosmos3_300f_full_episode_wam_eval_in_allocation.sh"
}

main "$@"
