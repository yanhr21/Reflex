#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this wrapper only inside a compute-node srun step. It waits for strict current clean-dense generated eval and then launches live-receding closed-loop panel.
EOF
  exit 30
fi

if [[ "${SLURM_NTASKS:-1}" != "1" || "${SLURM_PROCID:-0}" != "0" ]]; then
  cat >&2 <<'EOF'
refusing_multi_task_execution=true
reason=Run this wrapper as exactly one Slurm task so live rollouts are serialized and auditable.
EOF
  exit 31
fi

SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299}"
CHECKPOINT_ITER="${CHECKPOINT_ITER:-300}"
printf -v CHECKPOINT_NAME 'iter_%09d' "${CHECKPOINT_ITER}"
JOB_NAME="${JOB_NAME:-vision_sft_droid_policy_full1000_rgb_300step_wam}"
RUN_DIR="${RUN_DIR:-${SFT_ROOT}/outputs/cosmos3/sft/${JOB_NAME}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${RUN_DIR}/checkpoints/${CHECKPOINT_NAME}}"
CONFIG_FILE="${CONFIG_FILE:-${RUN_DIR}/config.yaml}"
EVAL_ROOT="${EVAL_ROOT:-${SFT_ROOT}/eval_full_episode_wam_${CHECKPOINT_NAME}}"
CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_20260614_130050}"
SOURCE_H5_ROOT="${SOURCE_H5_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/fix3_v7_dp_user_override_sft_source_20260612_733/canonical_h5}"
DP_MANIFEST="${DP_MANIFEST:-${ROOT}/experiments/dp_peg1000/run_90201/manifest.json}"
DP_CHECKPOINT="${DP_CHECKPOINT:-${ROOT}/experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_at_end.pt}"
CONTINUABILITY_STATS_JSON="${CONTINUABILITY_STATS_JSON:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/dp_static_continuability_stats_20260613/dp_static_continuability_stats.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SFT_ROOT}/live_receding_full300_panel_${CHECKPOINT_NAME}_clean_dense_${STAMP}}"
SAMPLE_INDICES="${SAMPLE_INDICES:-0,1,3,4}"
MAX_SAMPLES="${MAX_SAMPLES:-4}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-8}"
DP_HANDOFF_HORIZON="${DP_HANDOFF_HORIZON:-32}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
WATCH_TIMEOUT_SECONDS="${WATCH_TIMEOUT_SECONDS:-14400}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"

cd "${ROOT}"
mkdir -p "${OUTPUT_ROOT}"
{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "step_id=${SLURM_STEP_ID}"
  echo "node_list=${SLURM_JOB_NODELIST:-}"
  echo "sft_root=${SFT_ROOT}"
  echo "checkpoint_iter=${CHECKPOINT_ITER}"
  echo "checkpoint_path=${CHECKPOINT_PATH}"
  echo "config_file=${CONFIG_FILE}"
  echo "eval_root=${EVAL_ROOT}"
  echo "condition_root=${CONDITION_ROOT}"
  echo "source_h5_root=${SOURCE_H5_ROOT}"
  echo "dp_manifest=${DP_MANIFEST}"
  echo "dp_checkpoint=${DP_CHECKPOINT}"
  echo "continuability_stats_json=${CONTINUABILITY_STATS_JSON}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "sample_indices=${SAMPLE_INDICES}"
  echo "max_samples=${MAX_SAMPLES}"
  echo "action_exec_horizon=${ACTION_EXEC_HORIZON}"
  echo "dp_handoff_horizon=${DP_HANDOFF_HORIZON}"
  echo "boundary=corrected full-300 live-receding closed-loop panel; real final-state/video evidence required before any method claim"
} | tee "${OUTPUT_ROOT}/watch_live_receding_manifest.txt"

artifact_json="${EVAL_ROOT}/eval_artifact_inspection.json"
start="$(date +%s)"
while [[ ! -d "${CHECKPOINT_PATH}/model" || ! -s "${artifact_json}" ]]; do
  now="$(date +%s)"
  elapsed=$((now - start))
  if (( elapsed > WATCH_TIMEOUT_SECONDS )); then
    echo "live_receding_watch_timeout_seconds=${WATCH_TIMEOUT_SECONDS}" >&2
    exit 32
  fi
  echo "live_receding_inputs_not_ready elapsed_seconds=${elapsed}"
  sleep "${CHECK_INTERVAL_SECONDS}"
done

"${PYTHON_BIN}" - "${artifact_json}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
ok = data.get("strict_eval_artifacts_ok")
if ok is not True:
    print(f"strict_eval_artifacts_ok={ok}", file=sys.stderr)
    raise SystemExit(40)
print("strict_eval_artifacts_ok=true")
PY

SFT_ROOT="${SFT_ROOT}" \
CHECKPOINT_ITER="${CHECKPOINT_ITER}" \
CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
CONFIG_FILE="${CONFIG_FILE}" \
EVAL_ROOT="${EVAL_ROOT}" \
SOURCE_H5_ROOT="${SOURCE_H5_ROOT}" \
CONDITION_ROOT="${CONDITION_ROOT}" \
DP_MANIFEST="${DP_MANIFEST}" \
DP_CHECKPOINT="${DP_CHECKPOINT}" \
CONTINUABILITY_STATS_JSON="${CONTINUABILITY_STATS_JSON}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
SAMPLE_INDICES="${SAMPLE_INDICES}" \
MAX_SAMPLES="${MAX_SAMPLES}" \
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON}" \
DP_HANDOFF_HORIZON="${DP_HANDOFF_HORIZON}" \
PREFIX_ROLE_MODE=auto \
PREFIX_START_MODE=target_motion_onset \
PRETRIGGER_CONTROL_MODE=frozen_dp_until_target_motion \
RUN_COSMOS_INFERENCE=true \
CAPTURE_LIVE_VIDEO=true \
SAVE_LIVE_STATE_SNAPSHOTS=true \
bash "${ROOT}/scripts/slurm/run_cosmos3_live_receding_panel_in_allocation.sh"
