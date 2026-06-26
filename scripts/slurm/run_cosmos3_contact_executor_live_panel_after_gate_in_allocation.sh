#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this gated contact-executor live panel only inside a compute-node Slurm step.
EOF
  exit 30
fi

if [[ "${SLURM_NTASKS:-1}" != "1" || "${SLURM_PROCID:-0}" != "0" ]]; then
  cat >&2 <<'EOF'
refusing_multi_task_execution=true
reason=Run this gated launcher as exactly one Slurm task so live rollouts stay serialized.
EOF
  exit 31
fi

FORMAL_ROOT="${FORMAL_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54_save20k}"
GATE_JSON="${GATE_JSON:-${FORMAL_ROOT}/formal_live_eval_gate.json}"
EXECUTOR_CHECKPOINT="${EXECUTOR_CHECKPOINT:-${FORMAL_ROOT}/checkpoint_final.pt}"

SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299}"
CHECKPOINT_ITER="${CHECKPOINT_ITER:-1500}"
EVAL_ROOT="${EVAL_ROOT:-${SFT_ROOT}/eval_full_episode_wam_iter_000001500_formal_after_3h_abs4gpu_retry2}"
CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_20260614_130050}"
SOURCE_H5_ROOT="${SOURCE_H5_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/fix3_v7_dp_user_override_sft_source_20260612_733/canonical_h5}"
DP_MANIFEST="${DP_MANIFEST:-${ROOT}/experiments/dp_peg1000/run_90201/manifest.json}"
DP_CHECKPOINT="${DP_CHECKPOINT:-${ROOT}/experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_at_end.pt}"
CONTINUABILITY_STATS_JSON="${CONTINUABILITY_STATS_JSON:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/dp_static_continuability_stats_20260613/dp_static_continuability_stats.json}"

SAMPLE_INDICES="${SAMPLE_INDICES:-0,1,3,4}"
MAX_SAMPLES="${MAX_SAMPLES:-4}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SAMPLE_TAG="${SAMPLE_INDICES//,/_}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SFT_ROOT}/live_receding_contact_executor_iter1500_panel${MAX_SAMPLES}_${STAMP}_samples${SAMPLE_TAG}_after_formal_gate}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -s "${path}" ]]; then
    echo "missing_${label}=${path}" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "missing_${label}=${path}" >&2
    exit 2
  fi
}

require_file "${GATE_JSON}" "formal_gate_json"
require_file "${FORMAL_ROOT}/training_summary.json" "training_summary"
require_file "${FORMAL_ROOT}/post_training_group_metrics.json" "post_training_group_metrics"
require_file "${EXECUTOR_CHECKPOINT}" "executor_checkpoint"
require_dir "${SFT_ROOT}" "sft_root"
require_dir "${EVAL_ROOT}" "eval_root"
require_dir "${CONDITION_ROOT}" "condition_root"
require_dir "${SOURCE_H5_ROOT}" "source_h5_root"
require_file "${DP_MANIFEST}" "dp_manifest"
require_file "${DP_CHECKPOINT}" "dp_checkpoint"
require_file "${CONTINUABILITY_STATS_JSON}" "continuability_stats_json"

"${ROOT}/.venv/bin/python" - "${GATE_JSON}" "${FORMAL_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

gate_path = Path(sys.argv[1]).resolve()
formal_root = Path(sys.argv[2]).resolve()
payload = json.loads(gate_path.read_text())
training_root = Path(payload.get("training_root", formal_root)).resolve()
allowed = payload.get("live_eval_allowed") is True

if training_root != formal_root:
    print("refusing_contact_executor_live_eval=true", file=sys.stderr)
    print(f"reason=gate_training_root_mismatch gate={training_root} expected={formal_root}", file=sys.stderr)
    sys.exit(44)

if not allowed:
    print("refusing_contact_executor_live_eval=true", file=sys.stderr)
    print("reason=formal_live_eval_gate_false", file=sys.stderr)
    print("failure_reasons=" + ",".join(map(str, payload.get("failure_reasons", []))), file=sys.stderr)
    missing = payload.get("missing_files", [])
    if missing:
        print("missing_files=" + ",".join(map(str, missing)), file=sys.stderr)
    sys.exit(44)

print("contact_executor_formal_gate_ok=true")
print(f"gate_json={gate_path}")
PY

cd "${ROOT}"
mkdir -p "${OUTPUT_ROOT}"
{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "step_id=${SLURM_STEP_ID}"
  echo "node_list=${SLURM_JOB_NODELIST:-}"
  echo "formal_root=${FORMAL_ROOT}"
  echo "gate_json=${GATE_JSON}"
  echo "executor_checkpoint=${EXECUTOR_CHECKPOINT}"
  echo "sft_root=${SFT_ROOT}"
  echo "checkpoint_iter=${CHECKPOINT_ITER}"
  echo "eval_root=${EVAL_ROOT}"
  echo "condition_root=${CONDITION_ROOT}"
  echo "source_h5_root=${SOURCE_H5_ROOT}"
  echo "sample_indices=${SAMPLE_INDICES}"
  echo "max_samples=${MAX_SAMPLES}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "controller_action_source=contact_executor"
  echo "executor_residual_scale=1.0"
  echo "boundary=Gated contact/progress executor live panel. This launcher refuses to run unless the formal 2GPU/3h final gate allows live eval. It binds the live run to clean-dense iter1500 Cosmos SFT and the current formal contact-executor final checkpoint."
} | tee "${OUTPUT_ROOT}/contact_executor_after_gate_launcher_manifest.txt"

SFT_ROOT="${SFT_ROOT}" \
CHECKPOINT_ITER="${CHECKPOINT_ITER}" \
EVAL_ROOT="${EVAL_ROOT}" \
CONDITION_ROOT="${CONDITION_ROOT}" \
SOURCE_H5_ROOT="${SOURCE_H5_ROOT}" \
DP_MANIFEST="${DP_MANIFEST}" \
DP_CHECKPOINT="${DP_CHECKPOINT}" \
CONTINUABILITY_STATS_JSON="${CONTINUABILITY_STATS_JSON}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
SAMPLE_INDICES="${SAMPLE_INDICES}" \
MAX_SAMPLES="${MAX_SAMPLES}" \
CONTROLLER_ACTION_SOURCE=contact_executor \
EXECUTOR_CHECKPOINT="${EXECUTOR_CHECKPOINT}" \
EXECUTOR_RESIDUAL_SCALE=1.0 \
RUN_COSMOS_INFERENCE=true \
SAVE_LIVE_STATE_SNAPSHOTS=true \
ALLOW_LIVE_RECEDING_DIAGNOSTIC=false \
bash "${ROOT}/scripts/slurm/run_cosmos3_live_receding_panel_in_allocation.sh"
