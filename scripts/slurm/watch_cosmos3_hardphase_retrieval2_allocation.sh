#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
JOB_ID="${JOB_ID:?JOB_ID is required}"
TARGET_SESSION="${TARGET_SESSION:?TARGET_SESSION is required}"
POLL_SECONDS="${POLL_SECONDS:-300}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/hardphase_retrieval2_watch_${STAMP}_job${JOB_ID}}"

mkdir -p "${LOG_ROOT}"
cat > "${LOG_ROOT}/watch_manifest.txt" <<EOF
schema=cosmos3_hardphase_retrieval2_allocation_watch_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
job_id=${JOB_ID}
target_session=${TARGET_SESSION}
poll_seconds=${POLL_SECONDS}
log_root=${LOG_ROOT}
boundary=Login-node queue watcher only. It launches compute-node srun shards inside a tmux-held allocation; it does not run replay, rendering, training, evaluation, or preflight on the login node.
EOF

while true; do
  state="$(squeue -h -j "${JOB_ID}" -o %T 2>/dev/null || true)"
  now="$(date --iso-8601=seconds)"
  echo "watch_time=${now} job=${JOB_ID} state=${state}" | tee -a "${LOG_ROOT}/watch.log"
  case "${state}" in
    RUNNING)
      cmd="cd ${ROOT}; mkdir -p ${LOG_ROOT}; ( pids=(); RUN_SCORER_AFTER_HEADROOM_GATE=false HARD_MAX_ROWS=64 HARD_SKIP_ROWS=0 STAMP=${STAMP}_shard0 srun --jobid=${JOB_ID} --exclusive --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=64G --chdir=${ROOT} bash scripts/slurm/run_cosmos3_hardphase_retrieval_smoke_in_allocation.sh > ${LOG_ROOT}/shard0.log 2>&1 & pids+=(\$!); RUN_SCORER_AFTER_HEADROOM_GATE=false HARD_MAX_ROWS=64 HARD_SKIP_ROWS=64 STAMP=${STAMP}_shard1 srun --jobid=${JOB_ID} --exclusive --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=64G --chdir=${ROOT} bash scripts/slurm/run_cosmos3_hardphase_retrieval_smoke_in_allocation.sh > ${LOG_ROOT}/shard1.log 2>&1 & pids+=(\$!); shard_wait_rc=0; for pid in \${pids[@]}; do wait \${pid} || shard_wait_rc=\$?; done; union_rc=0; STAMP=${STAMP}_union srun --jobid=${JOB_ID} --overlap --ntasks=1 --gpus=1 --cpus-per-task=4 --mem=32G --chdir=${ROOT} bash scripts/slurm/run_cosmos3_hardphase_retrieval_claim_union_in_allocation.sh > ${LOG_ROOT}/union.log 2>&1 || union_rc=\$?; echo retrieval2_parallel_done date=\$(date --iso-8601=seconds) shard_wait_rc=\${shard_wait_rc} union_rc=\${union_rc}; if [[ \${shard_wait_rc} -ne 0 ]]; then exit \${shard_wait_rc}; fi; exit \${union_rc} ) 2>&1 | tee ${LOG_ROOT}/parallel_driver.log"
      tmux send-keys -t "${TARGET_SESSION}" "${cmd}" C-m
      echo "launched=true date=$(date --iso-8601=seconds)" | tee -a "${LOG_ROOT}/watch.log"
      exit 0
      ;;
    CANCELLED|FAILED|TIMEOUT|COMPLETED)
      echo "terminal_state=${state}" | tee -a "${LOG_ROOT}/watch.log"
      exit 0
      ;;
    "")
      echo "job_missing=true" | tee -a "${LOG_ROOT}/watch.log"
      exit 0
      ;;
  esac
  sleep "${POLL_SECONDS}"
done
