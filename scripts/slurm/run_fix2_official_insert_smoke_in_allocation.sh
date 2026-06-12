#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run inside a compute-node srun step, for example:
  srun --jobid=$SLURM_JOB_ID --gres=gpu:1 --cpus-per-task=8 bash scripts/slurm/run_fix2_official_insert_smoke_in_allocation.sh
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
NUM_DEMOS="${NUM_DEMOS:-6}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-80}"
SEED="${SEED:-1002000}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/fix2_official_insert_repro_smoke${NUM_DEMOS}_${STAMP}}"
RENDER_ROOT="${RENDER_ROOT:-${OUTPUT_ROOT}/render_${WIDTH}_server${SLURM_JOB_NODELIST:-unknown}}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"

cd "${ROOT}"
mkdir -p "${OUTPUT_ROOT}" logs/slurm

export PYTHONPATH="${ROOT}/deps/ManiSkill_clean:${ROOT}:${PYTHONPATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export DISPLAY=
export HDF5_USE_FILE_LOCKING=FALSE

MANIFEST="${OUTPUT_ROOT}/allocation_run_manifest.txt"
{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "step_id=${SLURM_STEP_ID}"
  echo "node_list=${SLURM_JOB_NODELIST:-unknown}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "render_root=${RENDER_ROOT}"
  echo "num_demos=${NUM_DEMOS}"
  echo "max_attempts=${MAX_ATTEMPTS}"
echo "seed=${SEED}"
  echo "python=${PYTHON}"
  echo "boundary=fix2_official_static_insert_reproduction_no_target_motion_no_projection_not_fix3_sft"
} | tee "${MANIFEST}"

nvidia-smi | tee "${OUTPUT_ROOT}/nvidia_smi.txt"

"${PYTHON}" -m py_compile \
  scripts/world_model/generate_cosmos3_fix2_official_insert_smoke.py \
  scripts/world_model/generate_cosmos3_fix3_late_trigger_dynamic_experts.py \
  scripts/world_model/render_cosmos3_maniskill_sft_dataset.py

"${PYTHON}" -u scripts/world_model/generate_cosmos3_fix2_official_insert_smoke.py \
  --output-root "${OUTPUT_ROOT}" \
  --num-demos "${NUM_DEMOS}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --seed "${SEED}" \
  --overwrite 2>&1 | tee "${OUTPUT_ROOT}/generate.log"

"${PYTHON}" -u scripts/world_model/render_cosmos3_maniskill_sft_dataset.py \
  --paths-file "${OUTPUT_ROOT}/fix2_h5_paths.txt" \
  --output-root "${RENDER_ROOT}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --fps 30 \
  --frame-stride 1 \
  --max-frames 0 \
  --val-fraction 0.1 \
  --sheet-limit "${NUM_DEMOS}" \
  --sheet-frames 24 \
  --sheet-thumb-width 384 \
  --overwrite 2>&1 | tee "${OUTPUT_ROOT}/render.log"

for split in train val; do
  run_dir="${RENDER_ROOT}/${split}"
  if [[ -d "${run_dir}/videos" ]]; then
    "${PYTHON}" -u scripts/world_model/inspect_video_artifacts.py \
      --run-dir "${run_dir}" \
      --output-dir "${RENDER_ROOT}/${split}_dense_video_review" \
      --sample-count 30 \
      --max-frames-to-scan 400 \
      --thumb-width 320 \
      --require-video \
      --require-nonblank 2>&1 | tee "${OUTPUT_ROOT}/inspect_${split}.log"
  fi
done

echo "fix2_official_insert_smoke_complete=$(date -Is)" | tee -a "${MANIFEST}"
