#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_SUPERSEDED_PHASE03_BOUNDARY:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
refusing_superseded_phase03_boundary_wrapper=true
reason=Phase 03 active Oracle must use the full pipeline: DP static prefix, repeated RGB Cosmos prediction, Cosmos action control after target motion, then physical finisher. Boundary-only runs are archive context and do not satisfy the active objective.
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
RUN_GROUP="${RUN_GROUP:-boundary}"
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
CKPT_PATH="${CKPT_PATH:-${ROOT}/experiments/maniskill/dp_checkpoint/run_90201/checkpoints/best_eval_success_at_end.pt}"
PHASE02_RUN="${PHASE02_RUN:-${ROOT}/experiments/maniskill/runs/02_cosmos_imagination/p02_cosmos_rgb6_config_20260703_012531_162203_server44}"
PHASE02_SAMPLE="${PHASE02_SAMPLE:-hole_late_fast_shift_seed10300001_idx5000}"
SEED="${SEED:-2}"
TARGET_MOTION_STEP="${TARGET_MOTION_STEP:-84}"
TARGET_MOTION_Y="${TARGET_MOTION_Y:-0.025}"
MAX_PREFIX_STEPS="${MAX_PREFIX_STEPS:-140}"
ALLOW_NO_MOTION_TRIGGER="${ALLOW_NO_MOTION_TRIGGER:-0}"
RENDER_TIMEOUT_S="${RENDER_TIMEOUT_S:-20}"

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
  echo "max_prefix_steps=${MAX_PREFIX_STEPS}"
  echo "allow_no_motion_trigger=${ALLOW_NO_MOTION_TRIGGER}"
  echo "render_timeout_s=${RENDER_TIMEOUT_S}"
  echo "method_evidence_allowed=false"
  echo "physical_insertion_success_claimed=false"
  echo "oracle_boundary=upper_bound_diagnostic_only"
} | tee "${RUN_DIR}/manifest.txt" | tee "${LOG_FILE}"

nvidia-smi 2>&1 | tee -a "${LOG_FILE}" || true

allow_flag=()
if [[ "${ALLOW_NO_MOTION_TRIGGER}" == "1" ]]; then
  allow_flag=(--allow-no-motion-trigger)
fi

"${PROJECT_PYTHON}" -u scripts/world_model/phase03_oracle_boundary_probe.py \
  --ckpt-path "${CKPT_PATH}" \
  --phase02-run "${PHASE02_RUN}" \
  --phase02-sample "${PHASE02_SAMPLE}" \
  --output-dir "${RUN_DIR}" \
  --seed "${SEED}" \
  --target-motion-step "${TARGET_MOTION_STEP}" \
  --target-motion-y "${TARGET_MOTION_Y}" \
  --max-prefix-steps "${MAX_PREFIX_STEPS}" \
  --render-timeout-s "${RENDER_TIMEOUT_S}" \
  "${allow_flag[@]}" \
  2>&1 | tee -a "${LOG_FILE}"

nvidia-smi 2>&1 | tee -a "${LOG_FILE}" || true
