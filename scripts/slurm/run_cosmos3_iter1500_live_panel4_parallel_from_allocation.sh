#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
ALLOC_JOB_ID="${ALLOC_JOB_ID:?set ALLOC_JOB_ID to the held Slurm allocation id}"

SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299}"
CHECKPOINT_ITER="${CHECKPOINT_ITER:-1500}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${SFT_ROOT}/outputs/cosmos3/sft/vision_sft_droid_policy_full1000_rgb_300step_wam/checkpoints/iter_000001500}"
CONFIG_FILE="${CONFIG_FILE:-${SFT_ROOT}/outputs/cosmos3/sft/vision_sft_droid_policy_full1000_rgb_300step_wam/config.yaml}"
EVAL_ROOT="${EVAL_ROOT:-${SFT_ROOT}/eval_full_episode_wam_iter_000001500_formal_after_3h_abs4gpu_retry2}"
CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_20260614_130050}"
SOURCE_H5_ROOT="${SOURCE_H5_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/fix3_v7_dp_user_override_sft_source_20260612_733/canonical_h5}"
DP_MANIFEST="${DP_MANIFEST:-${ROOT}/experiments/dp_peg1000/run_90201/manifest.json}"
DP_CHECKPOINT="${DP_CHECKPOINT:-${ROOT}/experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_at_end.pt}"
CONTINUABILITY_STATS_JSON="${CONTINUABILITY_STATS_JSON:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/dp_static_continuability_stats_20260613/dp_static_continuability_stats.json}"

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-${SFT_ROOT}/live_receding_full300_panel_iter_000001500_clean_dense_parallel4_$(date +%Y%m%d_%H%M%S)}"
SAMPLE_INDICES_LIST="${SAMPLE_INDICES_LIST:-0 1 3 4}"
PORT_BASE="${PORT_BASE:-51040}"
CPUS_PER_SAMPLE="${CPUS_PER_SAMPLE:-8}"
MEM_PER_SAMPLE="${MEM_PER_SAMPLE:-50G}"

cd "${ROOT}"
mkdir -p "${BASE_OUTPUT_ROOT}"
{
  echo "timestamp=$(date -Is)"
  echo "alloc_job_id=${ALLOC_JOB_ID}"
  echo "sft_root=${SFT_ROOT}"
  echo "checkpoint_iter=${CHECKPOINT_ITER}"
  echo "checkpoint_path=${CHECKPOINT_PATH}"
  echo "config_file=${CONFIG_FILE}"
  echo "eval_root=${EVAL_ROOT}"
  echo "condition_root=${CONDITION_ROOT}"
  echo "source_h5_root=${SOURCE_H5_ROOT}"
  echo "sample_indices_list=${SAMPLE_INDICES_LIST}"
  echo "save_candidate_action_bank=${SAVE_CANDIDATE_ACTION_BANK:-false}"
  echo "cpus_per_sample=${CPUS_PER_SAMPLE}"
  echo "mem_per_sample=${MEM_PER_SAMPLE}"
  echo "boundary=Parallel launcher only. Heavy live rollouts run inside per-sample Slurm steps on the held compute allocation."
} | tee "${BASE_OUTPUT_ROOT}/parallel_launcher_manifest.txt"

pids=()
labels=()
offset=0
for sample_idx in ${SAMPLE_INDICES_LIST}; do
  port=$((PORT_BASE + offset))
  sample_output="${BASE_OUTPUT_ROOT}/sample_${sample_idx}_single_panel"
  sample_log="${BASE_OUTPUT_ROOT}/sample_${sample_idx}_launcher.log"
  mkdir -p "${sample_output}"
  echo "launch_sample=${sample_idx} master_port=${port} output_root=${sample_output}" | tee -a "${BASE_OUTPUT_ROOT}/parallel_launcher_manifest.txt"
  (
    NPROC_PER_NODE=1 \
    MASTER_PORT="${port}" \
    SFT_ROOT="${SFT_ROOT}" \
    CHECKPOINT_ITER="${CHECKPOINT_ITER}" \
    CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
    CONFIG_FILE="${CONFIG_FILE}" \
    EVAL_ROOT="${EVAL_ROOT}" \
    CONDITION_ROOT="${CONDITION_ROOT}" \
    SOURCE_H5_ROOT="${SOURCE_H5_ROOT}" \
    DP_MANIFEST="${DP_MANIFEST}" \
    DP_CHECKPOINT="${DP_CHECKPOINT}" \
    CONTINUABILITY_STATS_JSON="${CONTINUABILITY_STATS_JSON}" \
    OUTPUT_ROOT="${sample_output}" \
    SAMPLE_INDICES="${sample_idx}" \
    MAX_SAMPLES=1 \
    SAVE_LIVE_STATE_SNAPSHOTS=true \
    SAVE_CANDIDATE_ACTION_BANK="${SAVE_CANDIDATE_ACTION_BANK:-false}" \
    RUN_COSMOS_INFERENCE=true \
    srun --overlap --jobid="${ALLOC_JOB_ID}" --ntasks=1 --gres=gpu:1 \
      --cpus-per-task="${CPUS_PER_SAMPLE}" --mem="${MEM_PER_SAMPLE}" \
      bash scripts/slurm/run_cosmos3_live_receding_panel_in_allocation.sh
  ) >"${sample_log}" 2>&1 &
  pids+=("$!")
  labels+=("${sample_idx}:${sample_log}")
  offset=$((offset + 1))
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "sample_done ${labels[$i]}" | tee -a "${BASE_OUTPUT_ROOT}/parallel_launcher_manifest.txt"
  else
    rc=$?
    echo "sample_failed rc=${rc} ${labels[$i]}" | tee -a "${BASE_OUTPUT_ROOT}/parallel_launcher_manifest.txt"
    status=1
  fi
done

echo "parallel_launcher_complete=$(date -Is) status=${status}" | tee -a "${BASE_OUTPUT_ROOT}/parallel_launcher_manifest.txt"
exit "${status}"
