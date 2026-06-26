#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run clean contact-suffix LeRobot rebuild only inside a compute-node srun step.
example=srun --jobid=<held_job_id> --ntasks=1 --cpus-per-task=16 --mem=120G bash scripts/slurm/run_openpi_pi05_clean_contact_suffix_rebuild_in_allocation.sh
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LEROBOT_HOME_DIR="${LEROBOT_HOME_DIR:-${ROOT}/experiments/world_model_task_rebinding/openpi/lerobot_home}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/openpi/pi05_contact_suffix_clean_rebuild_${STAMP}_alloc${SLURM_JOB_ID}}"
QPOS8_REPO_ID="${QPOS8_REPO_ID:-yanhongru/maniskill_peg733_openpi_contact_suffix16_qpos8_clean_20260626}"
OBJECT17_REPO_ID="${OBJECT17_REPO_ID:-yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_clean_20260626}"
OPENPI_PYTHON="${OPENPI_PYTHON:-}"

mkdir -p "${OUTPUT_ROOT}" "${LEROBOT_HOME_DIR}"

cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=openpi_pi05_clean_contact_suffix_rebuild_driver_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
output_root=${OUTPUT_ROOT}
lerobot_home_dir=${LEROBOT_HOME_DIR}
qpos8_repo_id=${QPOS8_REPO_ID}
object17_repo_id=${OBJECT17_REPO_ID}
openpi_python=${OPENPI_PYTHON:-uv_locked_default}
resource_boundary=tmux-held interactive Slurm allocation; no login-node conversion/audit.
method_boundary=Clean OpenPI/pi0.5 LeRobot data repair only. No model training, replay, scorer, VAE, MLP, or custom policy.
EOF

run_convert() {
  local repo_id="$1"
  local state_mode="$2"
  local tag="$3"
  local out="${OUTPUT_ROOT}/${tag}_convert"
  REPO_ID="${repo_id}" \
  STATE_MODE="${state_mode}" \
  OVERWRITE=true \
  LEROBOT_HOME_DIR="${LEROBOT_HOME_DIR}" \
  OUTPUT_ROOT="${out}" \
  OPENPI_PYTHON="${OPENPI_PYTHON}" \
    bash "${ROOT}/scripts/slurm/run_openpi_pi05_contact_suffix_lerobot_convert_in_allocation.sh"
}

run_audit() {
  local repo_id="$1"
  local state_dim="$2"
  local tag="$3"
  local convert_out="${OUTPUT_ROOT}/${tag}_convert"
  local audit_out="${OUTPUT_ROOT}/${tag}_audit"
  REPO_ID="${repo_id}" \
  DATASET_ROOT="${LEROBOT_HOME_DIR}/${repo_id}" \
  CONVERSION_MANIFEST="${convert_out}/conversion_manifest.json" \
  OUTPUT_ROOT="${audit_out}" \
  OPENPI_PYTHON="${OPENPI_PYTHON}" \
  EXPECTED_STATE_DIM="${state_dim}" \
    bash "${ROOT}/scripts/slurm/run_openpi_pi05_contact_suffix_lerobot_audit_in_allocation.sh"
}

run_convert "${QPOS8_REPO_ID}" qpos8 qpos8
run_audit "${QPOS8_REPO_ID}" 8 qpos8

run_convert "${OBJECT17_REPO_ID}" object_state17 object17
run_audit "${OBJECT17_REPO_ID}" 17 object17

cat > "${OUTPUT_ROOT}/clean_rebuild_summary.json" <<EOF
{
  "schema": "openpi_pi05_clean_contact_suffix_rebuild_summary_v1",
  "slurm_job_id": "${SLURM_JOB_ID}",
  "slurm_step_id": "${SLURM_STEP_ID}",
  "host": "$(hostname)",
  "qpos8_repo_id": "${QPOS8_REPO_ID}",
  "qpos8_conversion_manifest": "${OUTPUT_ROOT}/qpos8_convert/conversion_manifest.json",
  "qpos8_audit_summary": "${OUTPUT_ROOT}/qpos8_audit/audit_summary.json",
  "object17_repo_id": "${OBJECT17_REPO_ID}",
  "object17_conversion_manifest": "${OUTPUT_ROOT}/object17_convert/conversion_manifest.json",
  "object17_audit_summary": "${OUTPUT_ROOT}/object17_audit/audit_summary.json",
  "passed": true
}
EOF
