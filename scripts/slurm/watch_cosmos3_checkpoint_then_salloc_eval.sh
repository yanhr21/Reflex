#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
SFT_ROOT="${SFT_ROOT:?set SFT_ROOT}"
CONDITION_ROOT="${CONDITION_ROOT:?set CONDITION_ROOT}"
READOUT_CHECKPOINT="${READOUT_CHECKPOINT:?set READOUT_CHECKPOINT}"
CHECKPOINT_ITER="${CHECKPOINT_ITER:-1800}"
N_EVAL_SAMPLES="${N_EVAL_SAMPLES:-10}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-50851}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-60}"
STABLE_SECONDS="${STABLE_SECONDS:-90}"
VISUAL_REVIEW_NOTE="${VISUAL_REVIEW_NOTE:-pending_agent_visual_review}"
REQUIRE_LATEST_MATCH="${REQUIRE_LATEST_MATCH:-false}"

ALLOC_PARTITION="${ALLOC_PARTITION:-gpu}"
ALLOC_GRES="${ALLOC_GRES:-gpu:${NPROC_PER_NODE}}"
ALLOC_CPUS="${ALLOC_CPUS:-16}"
ALLOC_MEM="${ALLOC_MEM:-120G}"
ALLOC_TIME="${ALLOC_TIME:-1-00:00:00}"
ALLOC_EXCLUDE="${ALLOC_EXCLUDE:-server13}"
ALLOC_JOB_NAME="${ALLOC_JOB_NAME:-cosmos3_checkpoint_eval}"

JOB_NAME="${JOB_NAME:-vision_sft_droid_policy_full1000_rgb_300step_wam}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SFT_ROOT}/outputs/cosmos3/sft/${JOB_NAME}/checkpoints}"
printf -v CHECKPOINT_NAME 'iter_%09d' "${CHECKPOINT_ITER}"
EVAL_ROOT="${EVAL_ROOT:-${SFT_ROOT}/eval_full_episode_wam_iter_${CHECKPOINT_ITER_PADDED:-$(printf '%09d' "${CHECKPOINT_ITER}")}}"
READOUT_EVAL_ROOT="${READOUT_EVAL_ROOT:-${EVAL_ROOT}/task_state_readout_v7_733}"

latest_checkpoint() {
  tr -d '[:space:]' < "${CHECKPOINT_ROOT}/latest_checkpoint.txt" 2>/dev/null || true
}

echo "timestamp=$(date -Is)"
echo "purpose=wait_on_login_for_checkpoint_then_request_eval_allocation"
echo "sft_root=${SFT_ROOT}"
echo "checkpoint_root=${CHECKPOINT_ROOT}"
echo "checkpoint_name=${CHECKPOINT_NAME}"
echo "eval_root=${EVAL_ROOT}"
echo "boundary=login_node_only_polls_checkpoint_and_requests_salloc_no_gpu_or_cpu_heavy_work"
echo "require_latest_match=${REQUIRE_LATEST_MATCH}"

while true; do
  latest="$(latest_checkpoint)"
  has_dir=no
  has_model_metadata=no
  [[ -d "${CHECKPOINT_ROOT}/${CHECKPOINT_NAME}" ]] && has_dir=yes
  [[ -s "${CHECKPOINT_ROOT}/${CHECKPOINT_NAME}/model/.metadata" ]] && has_model_metadata=yes
  echo "poll $(date -Is) latest=${latest:-none} has_target_dir=${has_dir} has_model_metadata=${has_model_metadata}"
  if [[ "${has_dir}" == "yes" && "${has_model_metadata}" == "yes" ]]; then
    if [[ "${REQUIRE_LATEST_MATCH}" == "true" && "${latest}" != "${CHECKPOINT_NAME}" ]]; then
      sleep "${WAIT_INTERVAL_SECONDS}"
      continue
    fi
    break
  fi
  sleep "${WAIT_INTERVAL_SECONDS}"
done

echo "stable_wait_seconds=${STABLE_SECONDS}"
sleep "${STABLE_SECONDS}"
latest_after_wait="$(latest_checkpoint)"
if [[ "${REQUIRE_LATEST_MATCH}" == "true" && "${latest_after_wait}" != "${CHECKPOINT_NAME}" ]]; then
  echo "refusing_eval_latest_mismatch=${latest_after_wait:-none}"
  exit 41
fi
if [[ ! -d "${CHECKPOINT_ROOT}/${CHECKPOINT_NAME}" ]]; then
  echo "refusing_eval_target_dir_missing=${CHECKPOINT_ROOT}/${CHECKPOINT_NAME}"
  exit 42
fi
if [[ ! -s "${CHECKPOINT_ROOT}/${CHECKPOINT_NAME}/model/.metadata" ]]; then
  echo "refusing_eval_model_metadata_missing=${CHECKPOINT_ROOT}/${CHECKPOINT_NAME}/model/.metadata"
  exit 43
fi
if [[ "${latest_after_wait}" != "${CHECKPOINT_NAME}" ]]; then
  echo "latest_checkpoint_mismatch_allowed=${latest_after_wait:-none}"
fi

export ROOT SFT_ROOT CONDITION_ROOT READOUT_CHECKPOINT CHECKPOINT_ITER
export N_EVAL_SAMPLES NPROC_PER_NODE MASTER_PORT EVAL_ROOT READOUT_EVAL_ROOT
export CHECKPOINT_STABLE_SECONDS=1
export CHECK_INTERVAL_SECONDS=10
export WATCH_TIMEOUT_SECONDS=600
export VISUAL_REVIEW_NOTE
export ALLOC_CPUS

alloc_args=(-p "${ALLOC_PARTITION}" --nodes=1 --ntasks=1 --gres="${ALLOC_GRES}" \
  --cpus-per-task="${ALLOC_CPUS}" --mem="${ALLOC_MEM}" --time="${ALLOC_TIME}" \
  --job-name="${ALLOC_JOB_NAME}")
if [[ -n "${ALLOC_EXCLUDE}" ]]; then
  alloc_args+=(--exclude="${ALLOC_EXCLUDE}")
fi

echo "requesting_salloc $(date -Is) args=${alloc_args[*]}"
salloc "${alloc_args[@]}" bash -lc '
set -euo pipefail
cd "${ROOT}"
echo "eval_allocation_started job=${SLURM_JOB_ID:-unknown} host=$(hostname) timestamp=$(date -Is)"
srun --overlap --ntasks=1 --gres="gpu:${NPROC_PER_NODE}" --cpus-per-task="${ALLOC_CPUS:-16}" bash -lc '"'"'
set -euo pipefail
bash scripts/slurm/watch_cosmos3_300f_checkpoint_eval_in_allocation.sh
bash scripts/slurm/watch_cosmos3_eval_readout_in_allocation.sh
bash scripts/slurm/watch_cosmos3_readout_profile_in_allocation.sh
.venv/bin/python scripts/world_model/check_cosmos3_closed_loop_gate.py \
  --eval-root "$EVAL_ROOT" \
  --visual-review-status missing \
  --visual-review-note "$VISUAL_REVIEW_NOTE" \
  --output-json "$EVAL_ROOT/closed_loop_gate_pre_visual.json" \
  --allow-nonpassing-exit-zero
'"'"'
echo "eval_chain_finished timestamp=$(date -Is)"
exec bash
'
