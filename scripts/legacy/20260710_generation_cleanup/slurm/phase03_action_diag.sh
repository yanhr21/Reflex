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

RUN_DIR="${RUN_DIR:?Set RUN_DIR to the archived or active Phase 03 run to diagnose.}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR to experiments/maniskill/runs/03_oracle/action_diag/tryNN.}"
LOG_FILE="${LOG_FILE:?Set LOG_FILE to logs/03_oracle/action_diag/tryNN.log.}"
OFFSET_WINDOW="${OFFSET_WINDOW:-16}"

mkdir -p "${OUT_DIR}" "$(dirname "${LOG_FILE}")"

{
  echo "diagnostic=phase03_oracle_action_interface_read_only"
  echo "source_run=${RUN_DIR}"
  echo "output_dir=${OUT_DIR}"
  echo "offset_window=${OFFSET_WINDOW}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "slurm_step_id=${SLURM_STEP_ID}"
  echo "node=$(hostname -s)"
  echo "method_evidence_allowed=false"
} | tee "${OUT_DIR}/manifest.txt" | tee "${LOG_FILE}"

"${ROOT}/.venv/bin/python" -u scripts/world_model/analyze_phase03_oracle_action_interface.py \
  --run-dir "${RUN_DIR}" \
  --output-json "${OUT_DIR}/action_interface_diagnostic.json" \
  --offset-window "${OFFSET_WINDOW}" \
  2>&1 | tee -a "${LOG_FILE}"

{
  echo "phase03_status=read_only_action_temporal_diagnostic_completed"
  echo "method_evidence_allowed=false"
  echo "physical_insertion_success=false"
  echo "oracle_success=false"
  echo "reason=Diagnostic-only comparison of existing action traces against source-H5 teacher actions; no rollout or teacher action execution."
} | tee "${OUT_DIR}/classification.txt" | tee -a "${LOG_FILE}"
