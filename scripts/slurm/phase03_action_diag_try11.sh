#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run inside a compute-node srun step from a tmux-held allocation.
EOF
  exit 30
fi

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

RUN_DIR="${RUN_DIR:-/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_fastshift/try06}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/maniskill/runs/03_oracle/action_diag/try11}"
LOG_FILE="${LOG_FILE:-${ROOT}/logs/03_oracle/action_diag/try11.log}"

mkdir -p "${OUT_DIR}" "$(dirname "${LOG_FILE}")"

{
  echo "diagnostic=phase03_oracle_action_interface_read_only"
  echo "source_run=${RUN_DIR}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "slurm_step_id=${SLURM_STEP_ID}"
  echo "node=$(hostname -s)"
  echo "method_evidence_allowed=false"
} | tee "${OUT_DIR}/manifest.txt" | tee "${LOG_FILE}"

"${ROOT}/.venv/bin/python" -u scripts/world_model/analyze_phase03_oracle_action_interface.py \
  --run-dir "${RUN_DIR}" \
  --output-json "${OUT_DIR}/action_interface_diagnostic.json" \
  2>&1 | tee -a "${LOG_FILE}"
