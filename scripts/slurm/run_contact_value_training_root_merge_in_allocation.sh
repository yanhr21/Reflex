#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this merge inside a compute-node srun step from a tmux-held allocation.
example=srun --jobid=<held_job_id> --ntasks=1 --cpus-per-task=4 --mem=20G bash scripts/slurm/run_contact_value_training_root_merge_in_allocation.sh
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OLD_ROOT="${OLD_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_training_source_suffix_union_plus_panel0245_plus0204fixuuid_dp96_20260623_alloc146658}"
PANEL0134_ROOT="${PANEL0134_ROOT:?Set PANEL0134_ROOT to the merged DP-prior + causal-suffix conversion root.}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/contact_value_training_union_plus_panel0134_causal_suffix_${STAMP}_alloc${SLURM_JOB_ID}}"

mkdir -p "${OUTPUT_ROOT}"
cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=contact_value_training_root_merge_wrapper_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
old_root=${OLD_ROOT}
panel0134_root=${PANEL0134_ROOT}
output_root=${OUTPUT_ROOT}
resource_boundary=tmux-held interactive Slurm allocation; no sbatch; compute-node step only.
method_boundary=Dataset merge for consequence/value training over old live outcome union plus same-snapshot DP/causal generated labels. Not live method evidence.
EOF

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/merge_candidate_outcome_training_roots.py" \
  --base-jsonl "${OLD_ROOT}/live_snapshot_base_rows.jsonl" \
  --base-jsonl "${PANEL0134_ROOT}/live_snapshot_base_rows.jsonl" \
  --outcome-jsonl "${OLD_ROOT}/candidate_outcome_labels.jsonl" \
  --outcome-jsonl "${PANEL0134_ROOT}/candidate_outcome_labels.jsonl" \
  --output-root "${OUTPUT_ROOT}" \
  2>&1 | tee "${OUTPUT_ROOT}/merge.log"
