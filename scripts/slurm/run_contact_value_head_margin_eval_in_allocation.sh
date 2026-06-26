#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this margin audit inside a compute-node srun step from a tmux-held allocation.
example=srun --jobid=<held_job_id> --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=80G bash scripts/slurm/run_contact_value_head_margin_eval_in_allocation.sh
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
TRAINING_ROOT="${TRAINING_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/contact_value_training_union_plus_panel0134_causal_suffix_20260623_204657_alloc146658}"
SCORER_ROOT="${SCORER_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/contact_value_head_union_plus_panel0134_causal_suffix_1gpu1h_20260623_204725_alloc146658}"
CHECKPOINT="${CHECKPOINT:-${SCORER_ROOT}/checkpoint_best_gate.pt}"
BASE_ROWS_JSONL="${BASE_ROWS_JSONL:-${TRAINING_ROOT}/live_snapshot_base_rows.jsonl}"
OUTCOME_JSONL="${OUTCOME_JSONL:-${TRAINING_ROOT}/candidate_outcome_labels.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/contact_value_head_margin_eval_union_plus_panel0134_causal_suffix_${STAMP}_alloc${SLURM_JOB_ID}}"
MARGINS="${MARGINS:-0,0.0025,0.005,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.3,0.5,1.0}"
SEED="${SEED:-20260623}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

mkdir -p "${OUTPUT_ROOT}"
cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=contact_value_head_margin_eval_wrapper_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
training_root=${TRAINING_ROOT}
scorer_root=${SCORER_ROOT}
checkpoint=${CHECKPOINT}
base_rows_jsonl=${BASE_ROWS_JSONL}
outcome_jsonl=${OUTCOME_JSONL}
output_root=${OUTPUT_ROOT}
margins=${MARGINS}
seed=${SEED}
resource_boundary=tmux-held interactive Slurm allocation; no sbatch; compute-node step only.
method_boundary=Offline margin audit over real replay labels. This is not live method evidence and cannot replace saved-snapshot selected replay or live video/final-state review.
EOF

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/eval_cosmos3_candidate_outcome_scorer_margins.py" \
  --contact-executor-jsonl "${BASE_ROWS_JSONL}" \
  --outcome-jsonl "${OUTCOME_JSONL}" \
  --checkpoint "${CHECKPOINT}" \
  --output-root "${OUTPUT_ROOT}" \
  --margins "${MARGINS}" \
  --val-fraction 0.2 \
  --seed "${SEED}" \
  --min-selected-handoff-success-improvement 0.02 \
  --min-selected-error-improvement 0.005 \
  --min-selected-progress-delta-improvement 0.0 \
  --min-non-dp-selected-fraction 0.1 \
  --require-cuda \
  2>&1 | tee "${OUTPUT_ROOT}/margin_eval.log"
