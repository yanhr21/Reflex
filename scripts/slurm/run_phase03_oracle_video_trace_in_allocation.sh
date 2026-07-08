#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_SUPERSEDED_PHASE03_BOUNDARY:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
refusing_superseded_phase03_boundary_video_wrapper=true
reason=Phase 03 active Oracle must use the full pipeline through Cosmos action control and a physical insertion finisher. Boundary-only video traces are archive context and do not satisfy the active objective.
override=Set ALLOW_SUPERSEDED_PHASE03_BOUNDARY=1 only for explicit archive/debug review, never for active evidence.
EOF
  exit 64
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "refusing_login_node_execution=true; run inside a tmux-held interactive Slurm allocation" >&2
  exit 2
fi

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

PHASE="03_oracle"
RUN_GROUP="${RUN_GROUP:-boundary_video}"
if [[ -z "${RUN_NAME:-}" ]]; then
  RUN_ROOT="${ROOT}/experiments/maniskill/runs/${PHASE}/${RUN_GROUP}"
  for try_idx in $(seq 1 99); do
    candidate="$(printf 'try%02d' "${try_idx}")"
    if [[ ! -e "${RUN_ROOT}/${candidate}" ]]; then
      RUN_NAME="${candidate}"
      break
    fi
  done
fi
RUN_NAME="${RUN_NAME:?no_available_short_run_name}"
RUN_ID="${RUN_ID:-${RUN_GROUP}/${RUN_NAME}}"
RUN_DIR="${RUN_DIR:-${ROOT}/experiments/maniskill/runs/${PHASE}/${RUN_GROUP}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/${PHASE}/${RUN_GROUP}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"

PROJECT_PYTHON="${PROJECT_PYTHON:-${ROOT}/.venv/bin/python}"
export PYTHONPATH="${ROOT}/deps/ManiSkill_clean/examples/baselines/diffusion_policy:${ROOT}/deps/ManiSkill_clean:${ROOT}:${PYTHONPATH:-}"
CKPT_PATH="${CKPT_PATH:-${ROOT}/experiments/maniskill/dp_checkpoint/run_90201/checkpoints/best_eval_success_at_end.pt}"
PHASE02_RUN="${PHASE02_RUN:-${ROOT}/experiments/maniskill/runs/02_cosmos_imagination/p02_cosmos_rgb6_config_20260703_012531_162203_server44}"
PHASE02_SAMPLE="${PHASE02_SAMPLE:-hole_late_fast_shift_seed10300001_idx5000}"
SEED="${SEED:-2}"
TARGET_MOTION_STEP="${TARGET_MOTION_STEP:-84}"
TARGET_MOTION_Y="${TARGET_MOTION_Y:-0.025}"
MAX_ORACLE_STEP_DISPLACEMENT="${MAX_ORACLE_STEP_DISPLACEMENT:-0.05}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "phase=${PHASE}"
  echo "run_id=${RUN_ID}"
  echo "run_group=${RUN_GROUP}"
  echo "run_name=${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  echo "log_file=${LOG_FILE}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "slurm_step_id=${SLURM_STEP_ID:-}"
  echo "node=$(hostname -s)"
  echo "project_python=${PROJECT_PYTHON}"
  echo "ckpt_path=${CKPT_PATH}"
  echo "phase02_run=${PHASE02_RUN}"
  echo "phase02_sample=${PHASE02_SAMPLE}"
  echo "seed=${SEED}"
  echo "target_motion_step=${TARGET_MOTION_STEP}"
  echo "target_motion_y=${TARGET_MOTION_Y}"
  echo "max_oracle_step_displacement=${MAX_ORACLE_STEP_DISPLACEMENT}"
  echo "method_evidence_allowed=false"
  echo "physical_insertion_success_claimed=false"
  echo "oracle_boundary=decision_diagnostic_only_no_peg_teleport"
  echo "oracle_peg_set_pose_allowed=false"
} | tee "${RUN_DIR}/manifest.txt" | tee "${LOG_FILE}"

nvidia-smi 2>&1 | tee -a "${LOG_FILE}" || true

"${PROJECT_PYTHON}" -u scripts/training/eval_dp_oracle_boundary_trace.py \
  --ckpt-path "${CKPT_PATH}" \
  --phase02-run "${PHASE02_RUN}" \
  --phase02-sample "${PHASE02_SAMPLE}" \
  --output-dir "${RUN_DIR}" \
  --seed "${SEED}" \
  --target-motion-step "${TARGET_MOTION_STEP}" \
  --target-motion-y "${TARGET_MOTION_Y}" \
  --max-oracle-step-displacement "${MAX_ORACLE_STEP_DISPLACEMENT}" \
  2>&1 | tee -a "${LOG_FILE}"

nvidia-smi 2>&1 | tee -a "${LOG_FILE}" || true
