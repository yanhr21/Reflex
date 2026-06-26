#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run selected causal-suffix replay inside a compute-node srun step from a tmux-held allocation.
example=srun --jobid=<held_job_id> --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=80G bash scripts/slurm/run_selected_causal_suffix_value_replay_in_allocation.sh
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PANEL_ROOT="${PANEL_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_seed20260725_h8192_safegate1_source_suffix_offsets64_48_32_24_panel0134_20260623_alloc146658}"
CONVERSION_ROOT="${CONVERSION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_causal_suffix_diffusion_panel0134_exec8_dp96_20260623_204543_alloc146658}"
BASE_ROWS_JSONL="${BASE_ROWS_JSONL:-${CONVERSION_ROOT}/live_snapshot_base_rows.jsonl}"
OUTCOME_JSONL="${OUTCOME_JSONL:-${CONVERSION_ROOT}/candidate_outcome_labels.jsonl}"
SCORER_ROOT="${SCORER_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/contact_value_head_union_plus_panel0134_causal_suffix_1gpu1h_20260623_204725_alloc146658}"
CHECKPOINT="${CHECKPOINT:-${SCORER_ROOT}/checkpoint_best_gate.pt}"
CAUSAL_SUFFIX_DIFFUSION_CHECKPOINT="${CAUSAL_SUFFIX_DIFFUSION_CHECKPOINT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/causal_contact_action_suffix_diffusion_full733_1gpu1h_20260623_190108_alloc146658/checkpoint_best_eval.pt}"
SELECT_ROOT="${SELECT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/selected_causal_suffix_value_head_panel0134_margin${MARGIN:-0}_${STAMP}_alloc${SLURM_JOB_ID}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_selected_causal_suffix_value_head_panel0134_margin${MARGIN:-0}_exec8_dp96_${STAMP}_alloc${SLURM_JOB_ID}}"
MARGIN="${MARGIN:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-4}"
MAX_ITER_DIRS="${MAX_ITER_DIRS:-4}"
CAUSAL_SUFFIX_DIFFUSION_OFFSETS="${CAUSAL_SUFFIX_DIFFUSION_OFFSETS:-64,48,32,24,16,8}"
CAUSAL_SUFFIX_DIFFUSION_SAMPLES_PER_OFFSET="${CAUSAL_SUFFIX_DIFFUSION_SAMPLES_PER_OFFSET:-2}"
CAUSAL_SUFFIX_DIFFUSION_EXECUTE_STEPS="${CAUSAL_SUFFIX_DIFFUSION_EXECUTE_STEPS:-8}"
DP_ROLLOUT_CONTINUABILITY_HORIZON="${DP_ROLLOUT_CONTINUABILITY_HORIZON:-96}"
SEED="${SEED:-20260622}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

mkdir -p "${SELECT_ROOT}" "${OUTPUT_ROOT}"
cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=selected_causal_suffix_value_replay_wrapper_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
panel_root=${PANEL_ROOT}
conversion_root=${CONVERSION_ROOT}
base_rows_jsonl=${BASE_ROWS_JSONL}
outcome_jsonl=${OUTCOME_JSONL}
scorer_root=${SCORER_ROOT}
checkpoint=${CHECKPOINT}
causal_suffix_diffusion_checkpoint=${CAUSAL_SUFFIX_DIFFUSION_CHECKPOINT}
select_root=${SELECT_ROOT}
output_root=${OUTPUT_ROOT}
margin=${MARGIN}
max_samples=${MAX_SAMPLES}
max_iter_dirs=${MAX_ITER_DIRS}
causal_suffix_diffusion_offsets=${CAUSAL_SUFFIX_DIFFUSION_OFFSETS}
causal_suffix_diffusion_samples_per_offset=${CAUSAL_SUFFIX_DIFFUSION_SAMPLES_PER_OFFSET}
causal_suffix_diffusion_execute_steps=${CAUSAL_SUFFIX_DIFFUSION_EXECUTE_STEPS}
dp_rollout_continuability_horizon=${DP_ROLLOUT_CONTINUABILITY_HORIZON}
seed=${SEED}
resource_boundary=tmux-held interactive Slurm allocation; no sbatch; compute-node step only.
method_boundary=Selected saved-snapshot replay diagnostic only. It tests whether the value head can choose executable causal suffix candidates; it is not live controller evidence.
EOF

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/select_cosmos3_live_snapshot_candidates_with_value_head.py" \
  --contact-executor-jsonl "${BASE_ROWS_JSONL}" \
  --outcome-jsonl "${OUTCOME_JSONL}" \
  --checkpoint "${CHECKPOINT}" \
  --output-root "${SELECT_ROOT}" \
  --margin "${MARGIN}" \
  --allowed-candidate-families "dp_prior,causal_suffix_diffusion" \
  --require-cuda \
  2>&1 | tee "${SELECT_ROOT}/select.log"

SELECTED_FILTER="${SELECT_ROOT}/selected_candidate_filter.tsv"
if [[ ! -s "${SELECTED_FILTER}" ]]; then
  echo "selected_filter_missing=${SELECTED_FILTER}" >&2
  exit 31
fi

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/replay_cosmos3_live_action_bank_from_snapshots.py" \
  --panel-root "${PANEL_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --max-samples "${MAX_SAMPLES}" \
  --max-iter-dirs "${MAX_ITER_DIRS}" \
  --candidate-name-regex '^dp_prior$' \
  --max-candidates-per-iter 1 \
  --no-include-selected \
  --candidate-filter-tsv "${SELECTED_FILTER}" \
  --causal-suffix-diffusion-checkpoint "${CAUSAL_SUFFIX_DIFFUSION_CHECKPOINT}" \
  --causal-suffix-diffusion-offsets "${CAUSAL_SUFFIX_DIFFUSION_OFFSETS}" \
  --causal-suffix-diffusion-samples-per-offset "${CAUSAL_SUFFIX_DIFFUSION_SAMPLES_PER_OFFSET}" \
  --causal-suffix-diffusion-execute-steps "${CAUSAL_SUFFIX_DIFFUSION_EXECUTE_STEPS}" \
  --dp-rollout-continuability-horizon "${DP_ROLLOUT_CONTINUABILITY_HORIZON}" \
  --no-save-step-records \
  --seed "${SEED}" \
  2>&1 | tee "${OUTPUT_ROOT}/replay.log"
