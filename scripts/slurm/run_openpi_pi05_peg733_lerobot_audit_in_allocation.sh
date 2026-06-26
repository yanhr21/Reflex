#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run OpenPI 733 LeRobot audit only inside a compute-node srun step.
example=srun --jobid=<held_job_id> --ntasks=1 --cpus-per-task=8 --mem=40G bash scripts/slurm/run_openpi_pi05_peg733_lerobot_audit_in_allocation.sh
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/experiments/world_model_task_rebinding/openpi/lerobot_home/yanhongru/maniskill_peg733_openpi_libero}"
CONVERSION_MANIFEST="${CONVERSION_MANIFEST:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/openpi/pi05_peg733_lerobot_audit_${STAMP}_alloc${SLURM_JOB_ID}}"
OPENPI_PYTHON="${OPENPI_PYTHON:-python}"

mkdir -p "${OUTPUT_ROOT}"

cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=openpi_pi05_peg733_lerobot_audit_wrapper_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
dataset_root=${DATASET_ROOT}
conversion_manifest=${CONVERSION_MANIFEST}
output_root=${OUTPUT_ROOT}
openpi_python=${OPENPI_PYTHON}
resource_boundary=tmux-held interactive Slurm allocation; no login-node audit.
method_boundary=Dataset audit only. No model, VAE, MLP, diffusion, training, or weight mutation.
EOF

ARGS=(
  "${ROOT}/scripts/openpi/audit_maniskill_peg733_lerobot.py"
  --args.dataset-root "${DATASET_ROOT}"
  --args.output-json "${OUTPUT_ROOT}/audit_summary.json"
)

if [[ -n "${CONVERSION_MANIFEST}" ]]; then
  ARGS+=(--args.conversion-manifest "${CONVERSION_MANIFEST}")
fi

"${OPENPI_PYTHON}" "${ARGS[@]}" 2>&1 | tee "${OUTPUT_ROOT}/audit.log"
