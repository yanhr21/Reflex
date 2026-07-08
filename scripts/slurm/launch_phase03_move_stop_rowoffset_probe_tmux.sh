#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

SESSION="${SESSION:-p03_move17_offset}"
JOB_NAME="${JOB_NAME:-p03_mv17}"
PARTITION="${PARTITION:-gpu}"
IMMEDIATE_SECONDS="${IMMEDIATE_SECONDS:-60}"
TIME_LIMIT="${TIME_LIMIT:-01:30:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "refusing_existing_tmux_session=true session=${SESSION}" >&2
  exit 41
fi

if squeue -h -u "${USER}" -n "${JOB_NAME}" | grep -q .; then
  echo "refusing_existing_slurm_job=true job_name=${JOB_NAME}" >&2
  squeue -u "${USER}" -n "${JOB_NAME}" -o "%.18i %.9P %.20j %.8T %.10M %.9l %.6D %R" >&2
  exit 42
fi

tmux new-session -d -s "${SESSION}" "
cd '${ROOT}' &&
set -uo pipefail;
salloc --immediate=${IMMEDIATE_SECONDS} -p '${PARTITION}' -N1 -n1 --gres=gpu:1 --cpus-per-task='${CPUS_PER_TASK}' --mem='${MEMORY}' -t '${TIME_LIMIT}' --job-name='${JOB_NAME}' bash -lc '
  srun --ntasks=1 --cpus-per-task=${CPUS_PER_TASK} --gres=gpu:1 scripts/slurm/phase03_move_stop_rowoffset_probe.sh;
  echo phase03_move_stop_rowoffset_probe_done;
  sleep 20
';
status=\$?;
if [[ \${status} -ne 0 ]]; then
  echo phase03_move_stop_rowoffset_probe_launcher_exit=\${status};
fi;
for job_id in \$(squeue -h -u '${USER}' -n '${JOB_NAME}' -t PENDING -o '%i'); do
  echo canceling_pending_leftover_job=\${job_id};
  scancel \${job_id} || true;
done;
exit \${status}
"

echo "phase03_move_stop_rowoffset_probe_tmux_started=true"
echo "session=${SESSION}"
echo "job_name=${JOB_NAME}"
echo "immediate_seconds=${IMMEDIATE_SECONDS}"
