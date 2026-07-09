#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ "${SKIP_PHASE03_NEXT_COVERAGE_GUARD:-false}" == "true" ]]; then
  echo "refusing_skip_phase03_next_coverage_guard=true" >&2
  echo "reason=Phase03 forward/backward launchers must run readiness and launch gates." >&2
  exit 52
fi

SESSION="${SESSION:-p03_fwdback21}"
JOB_NAME="${JOB_NAME:-p03_fb21}"
PARTITION="${PARTITION:-gpu}"
IMMEDIATE_SECONDS="${IMMEDIATE_SECONDS:-60}"
TIME_LIMIT="${TIME_LIMIT:-01:30:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
MAX_TEST_ONLY_DELAY_MINUTES="${MAX_TEST_ONLY_DELAY_MINUTES:-120}"
ALLOW_FAR_SCHEDULER_TEST_ONLY="${ALLOW_FAR_SCHEDULER_TEST_ONLY:-false}"
EXCLUDE_NODES="${EXCLUDE_NODES:-server02,server21,server27,server28,server30,server39,server53,server57}"
MANDATORY_EXCLUDE_NODES="${MANDATORY_EXCLUDE_NODES:-server02,server21,server27,server28,server30,server39,server53,server57}"
EXCLUDE_ARG_TEXT=""
if [[ -n "${EXCLUDE_NODES}" ]]; then
  EXCLUDE_ARG_TEXT="--exclude=${EXCLUDE_NODES}"
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "refusing_existing_tmux_session=true session=${SESSION}" >&2
  exit 41
fi

if squeue -h -u "${USER}" -n "${JOB_NAME}" | grep -q .; then
  echo "refusing_existing_slurm_job=true job_name=${JOB_NAME}" >&2
  squeue -u "${USER}" -n "${JOB_NAME}" -o "%.18i %.9P %.20j %.8T %.10M %.9l %.6D %R" >&2
  exit 42
fi

PARTITION="${PARTITION}" \
  JOB_NAME="${JOB_NAME}" \
  TIME_LIMIT="${TIME_LIMIT}" \
  CPUS_PER_TASK="${CPUS_PER_TASK}" \
  MEMORY="${MEMORY}" \
  EXCLUDE_NODES="${EXCLUDE_NODES}" \
  MANDATORY_EXCLUDE_NODES="${MANDATORY_EXCLUDE_NODES}" \
  IMMEDIATE_SECONDS="${IMMEDIATE_SECONDS}" \
  SCHEDULER_TEST_CACHE_SECONDS=0 \
  MAX_TEST_ONLY_DELAY_MINUTES="${MAX_TEST_ONLY_DELAY_MINUTES}" \
  ALLOW_FAR_SCHEDULER_TEST_ONLY="${ALLOW_FAR_SCHEDULER_TEST_ONLY}" \
  scripts/world_model/require_phase03_forward_backward_launch_allowed.sh

tmux new-session -d -s "${SESSION}" "
cd '${ROOT}' &&
set -uo pipefail;
salloc --immediate=${IMMEDIATE_SECONDS} -p '${PARTITION}' -N1 -n1 --gres=gpu:1 --cpus-per-task='${CPUS_PER_TASK}' --mem='${MEMORY}' -t '${TIME_LIMIT}' --job-name='${JOB_NAME}' ${EXCLUDE_ARG_TEXT} bash -lc '
  srun --ntasks=1 --cpus-per-task=${CPUS_PER_TASK} --gres=gpu:1 scripts/slurm/phase03_forward_backward_probe.sh;
  echo phase03_forward_backward_probe_done;
  sleep 20
';
status=\$?;
if [[ \${status} -ne 0 ]]; then
  echo phase03_forward_backward_probe_launcher_exit=\${status};
fi;
for job_id in \$(squeue -h -u '${USER}' -n '${JOB_NAME}' -t PENDING -o '%i'); do
  echo canceling_pending_leftover_job=\${job_id};
  scancel \${job_id} || true;
done;
exit \${status}
"

echo "phase03_forward_backward_probe_tmux_started=true"
echo "session=${SESSION}"
echo "job_name=${JOB_NAME}"
echo "immediate_seconds=${IMMEDIATE_SECONDS}"
echo "exclude_nodes=${EXCLUDE_NODES}"
