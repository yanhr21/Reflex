#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

export RUN_GROUP="${RUN_GROUP:-render_probe}"
export RUN_NAME="${RUN_NAME:-fwdback21}"
export FULL_RUN_GROUP="${FULL_RUN_GROUP:-h5_reverse}"
export FULL_RUN_NAME="${FULL_RUN_NAME:-try21}"
export FULL_RUN_SCRIPT="${FULL_RUN_SCRIPT:-scripts/slurm/phase03_forward_backward_next.sh}"
export RENDER_SHADER_PACK="${RENDER_SHADER_PACK:-minimal}"
export RENDER_CANARY_API="${RENDER_CANARY_API:-gym}"

if [[ "${SKIP_PHASE03_NEXT_COVERAGE_GUARD:-false}" == "true" ]]; then
  echo "refusing_skip_phase03_next_coverage_guard=true" >&2
  echo "reason=Phase03 forward/backward probes must run readiness and coverage gates." >&2
  exit 52
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "refusing_login_node_execution=true" >&2
  echo "reason=Phase03 forward/backward probe, readiness, render, and rollout must run inside a tmux-held interactive Slurm allocation." >&2
  exit 53
fi

READINESS_PROBE_GROUP="${RUN_GROUP}"
READINESS_PROBE_NAME="${RUN_NAME}"
RUN_GROUP="${FULL_RUN_GROUP}" \
RUN_NAME="${FULL_RUN_NAME}" \
PROBE_GROUP="${READINESS_PROBE_GROUP}" \
PROBE_NAME="${READINESS_PROBE_NAME}" \
scripts/world_model/phase03_forward_backward_readiness.sh

if [[ "${ALLOW_EXISTING_PHASE03_RUN_DIR:-false}" != "true" ]]; then
  for path in \
    "${ROOT}/experiments/maniskill/runs/03_oracle/${RUN_GROUP}/${RUN_NAME}" \
    "${ROOT}/experiments/maniskill/runs/03_oracle/${FULL_RUN_GROUP}/${FULL_RUN_NAME}" \
    "${ROOT}/logs/03_oracle/${RUN_GROUP}/${RUN_NAME}.log" \
    "${ROOT}/logs/03_oracle/${FULL_RUN_GROUP}/${FULL_RUN_NAME}.log"
  do
    if [[ -e "${path}" ]]; then
      echo "refusing_existing_phase03_forward_backward_artifact=true path=${path}" >&2
      exit 50
    fi
  done
fi

bash scripts/slurm/phase03_render_gpu_probe.sh
