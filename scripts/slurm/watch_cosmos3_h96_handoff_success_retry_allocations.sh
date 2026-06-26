#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

POLL_SECONDS="${POLL_SECONDS:-300}"
STATE_ROOT="${STATE_ROOT:-experiments/world_model_task_rebinding/cosmos3/h96_handoff_success_retry_autostart_20260621}"
mkdir -p "${STATE_ROOT}/logs" "${STATE_ROOT}/state"

echo "watch_start date=$(date --iso-8601=seconds) poll_seconds=${POLL_SECONDS} state_root=${STATE_ROOT}" | tee -a "${STATE_ROOT}/watch.log"

job_gpu_count() {
  local jobid="$1"
  local text
  text="$(scontrol show job "${jobid}" 2>/dev/null | tr '\n' ' ')"
  if [[ "${text}" =~ gres/gpu=([0-9]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  else
    printf '1\n'
  fi
}

start_training_once() {
  local jobid="$1"
  local jobname="$2"
  if [[ -s "${STATE_ROOT}/state/training_jobid.txt" ]]; then
    return 0
  fi
  local tag="hardphase_h96_handoff_success_retry_${jobid}_$(date +%Y%m%d_%H%M%S)"
  local scorer_root="experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_${tag}_formal3h"
  local session="cosmos3_h96_handoff_success_retry_train_${jobid}_20260621"
  local log="${STATE_ROOT}/logs/train_${jobid}.log"
  printf '%s\n' "${jobid}" > "${STATE_ROOT}/state/training_jobid.txt"
  printf '%s\n' "${scorer_root}" > "${STATE_ROOT}/state/training_scorer_root.txt"
  echo "launch_training date=$(date --iso-8601=seconds) jobid=${jobid} jobname=${jobname} session=${session} scorer_root=${scorer_root}" | tee -a "${STATE_ROOT}/watch.log"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && srun --jobid='${jobid}' --overlap --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=04:30:00 bash -lc 'export TAG=${tag}; export SCORER_ROOT=${scorer_root}; bash scripts/slurm/run_cosmos3_h96_handoff_success_scorer_retry_in_allocation.sh' 2>&1 | tee '${log}'"
}

start_keepalive_once() {
  local jobid="$1"
  local gpus="$2"
  local marker="${STATE_ROOT}/state/keepalive_${jobid}.txt"
  if [[ -s "${marker}" ]]; then
    return 0
  fi
  local scorer_root
  scorer_root="$(cat "${STATE_ROOT}/state/training_scorer_root.txt" 2>/dev/null || true)"
  if [[ -z "${scorer_root}" ]]; then
    return 0
  fi
  local session="cosmos3_h96_handoff_success_retry_keepalive_${jobid}_20260621"
  local log="${STATE_ROOT}/logs/keepalive_${jobid}.log"
  printf '%s\n' "${session}" > "${marker}"
  echo "launch_keepalive date=$(date --iso-8601=seconds) jobid=${jobid} gpus=${gpus} session=${session}" | tee -a "${STATE_ROOT}/watch.log"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && srun --jobid='${jobid}' --overlap --nodes=1 --ntasks=1 --gres=gpu:'${gpus}' --cpus-per-task=2 --mem=16G --time=1-00:00:00 bash -lc '.venv/bin/python scripts/slurm/gpu_keepalive_until_claims_done.py --claim-root ${STATE_ROOT}/state --min-done 999999 --stop-file ${scorer_root}/training_summary.json --matrix-size 8192 --inner-loops 8 --sleep-seconds 0.02' 2>&1 | tee '${log}'"
}

while true; do
  mapfile -t running < <(
    squeue -h -u yanhongru -t R -o '%i|%j' \
      | awk -F'|' '$2 ~ /^cosmos3_h96_formal_retry/ {print $0}'
  )
  if [[ "${#running[@]}" -gt 0 ]]; then
    for row in "${running[@]}"; do
      jobid="${row%%|*}"
      jobname="${row#*|}"
      start_training_once "${jobid}" "${jobname}"
    done
    for row in "${running[@]}"; do
      jobid="${row%%|*}"
      gpus="$(job_gpu_count "${jobid}")"
      start_keepalive_once "${jobid}" "${gpus}"
    done
  fi
  sleep "${POLL_SECONDS}"
done
