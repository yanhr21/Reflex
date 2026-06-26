#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this replay inside a compute-node srun step from a tmux-held allocation.
example=srun --jobid=<held_job_id> --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=80G bash scripts/slurm/run_panel0134_dp_prior_replay_in_allocation.sh
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PANEL_ROOT="${PANEL_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_seed20260725_h8192_safegate1_source_suffix_offsets64_48_32_24_panel0134_20260623_alloc146658}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_dp_prior_panel0134_exec8_dp96_${STAMP}_alloc${SLURM_JOB_ID}}"
MAX_SAMPLES="${MAX_SAMPLES:-4}"
MAX_ITER_DIRS="${MAX_ITER_DIRS:-4}"
DP_ROLLOUT_CONTINUABILITY_HORIZON="${DP_ROLLOUT_CONTINUABILITY_HORIZON:-96}"
SEED="${SEED:-20260622}"

mkdir -p "${OUTPUT_ROOT}"
cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=panel0134_dp_prior_replay_wrapper_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
panel_root=${PANEL_ROOT}
output_root=${OUTPUT_ROOT}
max_samples=${MAX_SAMPLES}
max_iter_dirs=${MAX_ITER_DIRS}
candidate_name_regex=^dp_prior$
dp_rollout_continuability_horizon=${DP_ROLLOUT_CONTINUABILITY_HORIZON}
seed=${SEED}
resource_boundary=tmux-held interactive Slurm allocation; no sbatch; compute-node step only.
method_boundary=Saved-snapshot DP-prior replay baseline for consequence/value training. This is not live method evidence.
EOF

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/replay_cosmos3_live_action_bank_from_snapshots.py" \
  --panel-root "${PANEL_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --max-samples "${MAX_SAMPLES}" \
  --max-iter-dirs "${MAX_ITER_DIRS}" \
  --candidate-name-regex '^dp_prior$' \
  --max-candidates-per-iter 1 \
  --no-include-selected \
  --no-save-step-records \
  --dp-rollout-continuability-horizon "${DP_ROLLOUT_CONTINUABILITY_HORIZON}" \
  --seed "${SEED}" \
  2>&1 | tee "${OUTPUT_ROOT}/replay.log"
