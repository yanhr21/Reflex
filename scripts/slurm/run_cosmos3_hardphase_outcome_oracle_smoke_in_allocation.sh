#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this hard-phase outcome-oracle smoke only inside a compute-node srun step from a tmux-held allocation.
example=srun --jobid=<held_job_id> --overlap --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=64G bash scripts/slurm/run_cosmos3_hardphase_outcome_oracle_smoke_in_allocation.sh
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
CONTACT_EXECUTOR_JSONL="${CONTACT_EXECUTOR_JSONL:-experiments/world_model_task_rebinding/cosmos3/contact_executor_dataset_rollout24_predpath_train512_20260617_server23_alloc135069/contact_executor_dataset_file.jsonl}"
OUTCOME_JSONL="${OUTCOME_JSONL:-experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_handoffgeom512_after_scorerfail_20260617_server61_alloc139069/candidate_outcome_labels.jsonl}"

OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/world_model_task_rebinding/cosmos3/outcome_oracle_candidate_executor_hardphase_balanced_smoke100_${STAMP}_alloc${SLURM_JOB_ID}}"
REPLAY_ROOT="${REPLAY_ROOT:-experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_balanced_smoke100_replay128_${STAMP}_alloc${SLURM_JOB_ID}}"
HEADROOM_ROOT="${HEADROOM_ROOT:-experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_hardphase_balanced_smoke100_replay128_${STAMP}_alloc${SLURM_JOB_ID}}"

HARD_PHASES="${HARD_PHASES:-far,lateral_align,preinsert_aligned}"
PHASE_SAMPLING="${PHASE_SAMPLING:-balanced}"
HARD_PHASE_ORACLE_MIN_IMPROVEMENT="${HARD_PHASE_ORACLE_MIN_IMPROVEMENT:-0.0}"
PREFER_SUCCESS_TARGETS="${PREFER_SUCCESS_TARGETS:-true}"
MAX_STEPS="${MAX_STEPS:-100}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-25}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
HIDDEN_DIM="${HIDDEN_DIM:-1024}"
NUM_LAYERS="${NUM_LAYERS:-4}"
CANDIDATE_SAMPLES="${CANDIDATE_SAMPLES:-16}"
CANDIDATE_TEMPS="${CANDIDATE_TEMPS:-0.5,1.0,1.5}"
CANDIDATE_SCALES="${CANDIDATE_SCALES:-0.2,0.5,1.0}"
MODEL_CANDIDATE_SAMPLES="${MODEL_CANDIDATE_SAMPLES:-16}"
MODEL_CANDIDATE_TEMPS="${MODEL_CANDIDATE_TEMPS:-0.5,1.0,1.5}"
MODEL_CANDIDATE_SCALES="${MODEL_CANDIDATE_SCALES:-0.2,0.5,1.0}"
LEGACY_CANDIDATE_SCALES="${LEGACY_CANDIDATE_SCALES:-0.05,0.1,0.2,0.5,1.0}"
REPLAY_MAX_SAMPLES="${REPLAY_MAX_SAMPLES:-128}"
EXEC_HORIZON="${EXEC_HORIZON:-24}"
VAL_FRACTION="${VAL_FRACTION:-0.25}"
SEED="${SEED:-20260618}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -s "${path}" ]]; then
    echo "missing_${label}=${path}" >&2
    exit 2
  fi
}

require_file "${CONTACT_EXECUTOR_JSONL}" "contact_executor_jsonl"
require_file "${OUTCOME_JSONL}" "outcome_jsonl"

mkdir -p "${OUTPUT_ROOT}" "${REPLAY_ROOT}" "${HEADROOM_ROOT}"
cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=cosmos3_hardphase_outcome_oracle_smoke_wrapper_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
contact_executor_jsonl=${CONTACT_EXECUTOR_JSONL}
outcome_jsonl=${OUTCOME_JSONL}
output_root=${OUTPUT_ROOT}
replay_root=${REPLAY_ROOT}
headroom_root=${HEADROOM_ROOT}
hard_phases=${HARD_PHASES}
phase_sampling=${PHASE_SAMPLING}
hard_phase_oracle_min_improvement=${HARD_PHASE_ORACLE_MIN_IMPROVEMENT}
prefer_success_targets=${PREFER_SUCCESS_TARGETS}
max_steps=${MAX_STEPS}
replay_max_samples=${REPLAY_MAX_SAMPLES}
exec_horizon=${EXEC_HORIZON}
resource_boundary=tmux-held allocation plus compute-node srun step only; no sbatch; no login-node project compute.
method_boundary=Short hard-phase candidate-generator smoke only. It is not formal training evidence and not live method evidence.
EOF

".venv/bin/python" -m py_compile \
  scripts/world_model/train_cosmos3_outcome_oracle_candidate_executor.py \
  scripts/world_model/export_cosmos3_candidate_outcome_labels.py \
  scripts/world_model/summarize_cosmos3_candidate_outcome_headroom.py

prefer_success_flag=()
if [[ "${PREFER_SUCCESS_TARGETS}" == "true" || "${PREFER_SUCCESS_TARGETS}" == "1" || "${PREFER_SUCCESS_TARGETS}" == "yes" ]]; then
  prefer_success_flag=(--prefer-success-targets)
else
  prefer_success_flag=(--no-prefer-success-targets)
fi

".venv/bin/python" scripts/world_model/train_cosmos3_outcome_oracle_candidate_executor.py \
  --contact-executor-jsonl "${CONTACT_EXECUTOR_JSONL}" \
  --outcome-jsonl "${OUTCOME_JSONL}" \
  --output-root "${OUTPUT_ROOT}" \
  --hard-phases "${HARD_PHASES}" \
  --phase-sampling "${PHASE_SAMPLING}" \
  --hard-phase-oracle-min-improvement "${HARD_PHASE_ORACLE_MIN_IMPROVEMENT}" \
  "${prefer_success_flag[@]}" \
  --max-steps "${MAX_STEPS}" \
  --eval-every-steps "${EVAL_EVERY_STEPS}" \
  --save-every-steps "${SAVE_EVERY_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --num-layers "${NUM_LAYERS}" \
  --candidate-samples "${CANDIDATE_SAMPLES}" \
  --candidate-temps "${CANDIDATE_TEMPS}" \
  --candidate-scales "${CANDIDATE_SCALES}" \
  --require-cuda \
  --seed "${SEED}"

require_file "${OUTPUT_ROOT}/checkpoint_best_offline.pt" "checkpoint_best_offline"

".venv/bin/python" scripts/world_model/export_cosmos3_candidate_outcome_labels.py \
  --contact-executor-jsonl "${CONTACT_EXECUTOR_JSONL}" \
  --output-root "${REPLAY_ROOT}" \
  --max-samples "${REPLAY_MAX_SAMPLES}" \
  --exec-horizon "${EXEC_HORIZON}" \
  --candidate-executor-checkpoint "${OUTPUT_ROOT}/checkpoint_best_offline.pt" \
  --model-candidate-samples "${MODEL_CANDIDATE_SAMPLES}" \
  --model-candidate-temps "${MODEL_CANDIDATE_TEMPS}" \
  --model-candidate-scales "${MODEL_CANDIDATE_SCALES}" \
  --candidate-scales "${LEGACY_CANDIDATE_SCALES}" \
  --include-legacy-teacher-scale-candidates

require_file "${REPLAY_ROOT}/candidate_outcome_labels.jsonl" "candidate_outcome_labels"

".venv/bin/python" scripts/world_model/summarize_cosmos3_candidate_outcome_headroom.py \
  --outcome-jsonl "${REPLAY_ROOT}/candidate_outcome_labels.jsonl" \
  --output-root "${HEADROOM_ROOT}" \
  --val-fraction "${VAL_FRACTION}" \
  --seed "${SEED}"

cat > "${HEADROOM_ROOT}/wrapper_done.txt" <<EOF
done=true
date=$(date --iso-8601=seconds)
output_root=${OUTPUT_ROOT}
replay_root=${REPLAY_ROOT}
headroom_root=${HEADROOM_ROOT}
checkpoint=${OUTPUT_ROOT}/checkpoint_best_offline.pt
boundary=Smoke completed inside compute-node allocation. Inspect summaries before deciding any next run.
EOF
