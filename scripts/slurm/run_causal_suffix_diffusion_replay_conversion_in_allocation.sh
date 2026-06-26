#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this converter inside a compute-node srun step from a tmux-held allocation.
example=srun --jobid=<held_job_id> --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=80G bash scripts/slurm/run_causal_suffix_diffusion_replay_conversion_in_allocation.sh
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
REPLAY_ROOT="${REPLAY_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_causal_suffix_diffusion_panel0134_offsets64_48_32_24_16_8_s2_exec8_dp96_fix1_20260623_201146_alloc146658}"
LIVE_SNAPSHOT_JSONL="${LIVE_SNAPSHOT_JSONL:-${REPLAY_ROOT}/live_snapshot_action_bank_outcome_labels.jsonl}"
LIVE_SNAPSHOT_JSONL_EXTRA="${LIVE_SNAPSHOT_JSONL_EXTRA:-}"
CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_20260614_130050}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_causal_suffix_diffusion_panel0134_exec8_dp96_${STAMP}_alloc${SLURM_JOB_ID}}"
SNAPSHOT_NAMESPACE="${SNAPSHOT_NAMESPACE:-}"
SEED="${SEED:-20260622}"

mkdir -p "${OUTPUT_ROOT}"
cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=causal_suffix_diffusion_replay_conversion_wrapper_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
replay_root=${REPLAY_ROOT}
live_snapshot_jsonl=${LIVE_SNAPSHOT_JSONL}
live_snapshot_jsonl_extra=${LIVE_SNAPSHOT_JSONL_EXTRA}
condition_root=${CONDITION_ROOT}
output_root=${OUTPUT_ROOT}
snapshot_namespace=${SNAPSHOT_NAMESPACE}
seed=${SEED}
resource_boundary=tmux-held interactive Slurm allocation; no sbatch; compute-node step only.
method_boundary=Converts real saved-snapshot replay labels for generated causal-suffix actions into consequence/value training rows. This is not live controller evidence.
candidate_action_feature_mode=executed generated short chunk followed by DP prior fill, matching the candidate+DP96 label structure.
EOF

LIVE_ARGS=(--live-snapshot-jsonl "${LIVE_SNAPSHOT_JSONL}")
if [[ -n "${LIVE_SNAPSHOT_JSONL_EXTRA}" ]]; then
  IFS=':' read -r -a EXTRA_JSONL_ARRAY <<< "${LIVE_SNAPSHOT_JSONL_EXTRA}"
  for item in "${EXTRA_JSONL_ARRAY[@]}"; do
    if [[ -n "${item}" ]]; then
      LIVE_ARGS+=(--live-snapshot-jsonl "${item}")
    fi
  done
fi
NAMESPACE_ARGS=()
if [[ -n "${SNAPSHOT_NAMESPACE}" ]]; then
  NAMESPACE_ARGS=(--snapshot-namespace "${SNAPSHOT_NAMESPACE}")
fi

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/convert_cosmos3_live_snapshot_labels_for_outcome_scorer.py" \
  "${LIVE_ARGS[@]}" \
  --output-root "${OUTPUT_ROOT}" \
  --condition-root "${CONDITION_ROOT}" \
  "${NAMESPACE_ARGS[@]}" \
  --seed "${SEED}" \
  2>&1 | tee "${OUTPUT_ROOT}/conversion.log"
