#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
if [[ "${SET_NVIDIA_VK_ICD:-false}" == "true" ]]; then
  export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
fi
if [[ -n "${DISPLAY:-}" ]]; then
  export DISPLAY
else
  unset DISPLAY
fi

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this gated candidate-executor live panel only inside a compute-node Slurm step.
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

FORMAL_ROOT="${FORMAL_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_phasecap_meanpen5_4096_finalgate}"
TRAINING_SUMMARY="${TRAINING_SUMMARY:-${FORMAL_ROOT}/training_summary.json}"
EXECUTOR_CHECKPOINT="${EXECUTOR_CHECKPOINT:-auto}"
CANDIDATE_OUTCOME_SCORER_CHECKPOINT="${CANDIDATE_OUTCOME_SCORER_CHECKPOINT:-}"
CANDIDATE_OUTCOME_SCORER_SUMMARY="${CANDIDATE_OUTCOME_SCORER_SUMMARY:-}"
CANDIDATE_OUTCOME_SCORER_DP_MARGIN="${CANDIDATE_OUTCOME_SCORER_DP_MARGIN:-0.0}"
CANDIDATE_OUTCOME_SCORER_MIN_PROGRESS_DELTA="${CANDIDATE_OUTCOME_SCORER_MIN_PROGRESS_DELTA:--1000000000.0}"
CANDIDATE_OUTCOME_SCORER_MIN_CONTINUABLE_PROB="${CANDIDATE_OUTCOME_SCORER_MIN_CONTINUABLE_PROB:-0.0}"
CANDIDATE_OUTCOME_SCORER_MIN_INSERTED_PROB="${CANDIDATE_OUTCOME_SCORER_MIN_INSERTED_PROB:-0.0}"
REQUIRE_DIFFUSION_GENERATOR="${REQUIRE_DIFFUSION_GENERATOR:-false}"

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${SFT_ROOT}/live_receding_candidate_executor_iter1500_panel${MAX_SAMPLES}_${STAMP}_samples${SAMPLE_TAG}_after_formal_gate}"

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

require_file "${TRAINING_SUMMARY}" "training_summary"
require_dir "${SFT_ROOT}" "sft_root"
require_dir "${EVAL_ROOT}" "eval_root"
require_dir "${CONDITION_ROOT}" "condition_root"
require_dir "${SOURCE_H5_ROOT}" "source_h5_root"
require_file "${DP_MANIFEST}" "dp_manifest"
require_file "${DP_CHECKPOINT}" "dp_checkpoint"
require_file "${CONTINUABILITY_STATS_JSON}" "continuability_stats_json"

if [[ -z "${EXECUTOR_CHECKPOINT}" || "${EXECUTOR_CHECKPOINT}" == "auto" ]]; then
  EXECUTOR_CHECKPOINT="$("${ROOT}/.venv/bin/python" - "${TRAINING_SUMMARY}" "${FORMAL_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1]).resolve()
formal_root = Path(sys.argv[2]).resolve()
payload = json.loads(summary_path.read_text())
candidate = payload.get("formal_live_eval_checkpoint")
if candidate:
    print(Path(str(candidate)).resolve())
else:
    print((formal_root / "checkpoint_final.pt").resolve())
PY
)"
fi
require_file "${EXECUTOR_CHECKPOINT}" "executor_checkpoint"

"${ROOT}/.venv/bin/python" - "${TRAINING_SUMMARY}" "${FORMAL_ROOT}" "${EXECUTOR_CHECKPOINT}" "${REQUIRE_DIFFUSION_GENERATOR}" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

summary_path = Path(sys.argv[1]).resolve()
formal_root = Path(sys.argv[2]).resolve()
checkpoint_path = Path(sys.argv[3]).resolve()
require_diffusion = sys.argv[4].lower() in {"1", "true", "yes"}
payload = json.loads(summary_path.read_text())
summary_root = Path(payload.get("output_root", formal_root)).resolve()
if summary_root != formal_root:
    print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
    print(f"reason=summary_output_root_mismatch summary={summary_root} expected={formal_root}", file=sys.stderr)
    sys.exit(44)

expected_checkpoint_raw = payload.get("formal_live_eval_checkpoint")
expected_checkpoint = Path(str(expected_checkpoint_raw)).resolve() if expected_checkpoint_raw else formal_root / "checkpoint_final.pt"
if checkpoint_path != expected_checkpoint:
    print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
    print(
        f"reason=executor_checkpoint_not_summary_live_eval_checkpoint checkpoint={checkpoint_path} "
        f"expected={expected_checkpoint}",
        file=sys.stderr,
    )
    sys.exit(44)

if payload.get("formal_training_floor_met") is not True:
    print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
    print("reason=formal_training_floor_not_met", file=sys.stderr)
    sys.exit(44)

if require_diffusion:
    if payload.get("generator_type") != "diffusion":
        print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
        print(f"reason=generator_type_not_diffusion generator_type={payload.get('generator_type')}", file=sys.stderr)
        sys.exit(44)
    try:
        candidate_samples = int(payload.get("candidate_samples"))
        rank_diffusion_count = int(payload.get("candidate_rank_diffusion_count"))
    except (TypeError, ValueError):
        print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
        print("reason=invalid_diffusion_candidate_metadata", file=sys.stderr)
        sys.exit(44)
    if candidate_samples <= 0 or rank_diffusion_count <= 0:
        print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
        print(
            f"reason=diffusion_candidate_metadata_disabled candidate_samples={candidate_samples} "
            f"candidate_rank_diffusion_count={rank_diffusion_count}",
            file=sys.stderr,
        )
        sys.exit(44)

def env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
        print(f"reason=invalid_gate_threshold {name}={raw}", file=sys.stderr)
        sys.exit(44)


thresholds = {
    "max_teacher_progress_mse": env_float("MAX_TEACHER_PROGRESS_MSE", 0.05),
    "max_teacher_value_mse": env_float("MAX_TEACHER_VALUE_MSE", 0.25),
    "min_teacher_inserted_acc": env_float("MIN_TEACHER_INSERTED_ACC", 0.75),
    "min_teacher_dp_continuable_acc": env_float("MIN_TEACHER_DP_CONTINUABLE_ACC", 0.75),
}

metrics_source = str(payload.get("formal_live_eval_metrics_source") or "final")
if metrics_source == "best_gate":
    metric_container = payload.get("best_gate_metrics") if isinstance(payload.get("best_gate_metrics"), dict) else {}
elif metrics_source == "final":
    metric_container = payload.get("final_metrics") if isinstance(payload.get("final_metrics"), dict) else {}
else:
    print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
    print(f"reason=unknown_formal_live_eval_metrics_source source={metrics_source}", file=sys.stderr)
    sys.exit(44)
eval_metrics = metric_container.get("eval") if isinstance(metric_container, dict) else None
if not isinstance(eval_metrics, dict):
    print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
    print(f"reason=missing_{metrics_source}_eval_metrics", file=sys.stderr)
    sys.exit(44)

def metric_float(name):
    value = eval_metrics.get(name)
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
        print(f"reason=invalid_{metrics_source}_metric {name}={value}", file=sys.stderr)
        sys.exit(44)
    if not math.isfinite(value_f):
        print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
        print(f"reason=nonfinite_{metrics_source}_metric {name}={value_f}", file=sys.stderr)
        sys.exit(44)
    return value_f


selected_mse_f = metric_float("selected_action_mse")
dp_prior_mse_f = metric_float("dp_prior_action_mse")
teacher_progress_mse_f = metric_float("teacher_progress_mse")
teacher_value_mse_f = metric_float("teacher_value_mse")
teacher_inserted_acc_f = metric_float("teacher_inserted_acc")
teacher_dp_continuable_acc_f = metric_float("teacher_dp_continuable_acc")

if teacher_progress_mse_f > thresholds["max_teacher_progress_mse"]:
    print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
    print("reason=teacher_progress_mse_above_gate", file=sys.stderr)
    print(f"eval_teacher_progress_mse={teacher_progress_mse_f}", file=sys.stderr)
    print(f"max_teacher_progress_mse={thresholds['max_teacher_progress_mse']}", file=sys.stderr)
    sys.exit(44)
if teacher_value_mse_f > thresholds["max_teacher_value_mse"]:
    print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
    print("reason=teacher_value_mse_above_gate", file=sys.stderr)
    print(f"eval_teacher_value_mse={teacher_value_mse_f}", file=sys.stderr)
    print(f"max_teacher_value_mse={thresholds['max_teacher_value_mse']}", file=sys.stderr)
    sys.exit(44)
if teacher_inserted_acc_f < thresholds["min_teacher_inserted_acc"]:
    print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
    print("reason=teacher_inserted_acc_below_gate", file=sys.stderr)
    print(f"eval_teacher_inserted_acc={teacher_inserted_acc_f}", file=sys.stderr)
    print(f"min_teacher_inserted_acc={thresholds['min_teacher_inserted_acc']}", file=sys.stderr)
    sys.exit(44)
if teacher_dp_continuable_acc_f < thresholds["min_teacher_dp_continuable_acc"]:
    print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
    print("reason=teacher_dp_continuable_acc_below_gate", file=sys.stderr)
    print(f"eval_teacher_dp_continuable_acc={teacher_dp_continuable_acc_f}", file=sys.stderr)
    print(f"min_teacher_dp_continuable_acc={thresholds['min_teacher_dp_continuable_acc']}", file=sys.stderr)
    sys.exit(44)
if selected_mse_f > dp_prior_mse_f:
    print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
    print(f"reason={metrics_source}_selected_action_mse_worse_than_dp_prior", file=sys.stderr)
    print(f"eval_selected_action_mse={selected_mse_f}", file=sys.stderr)
    print(f"eval_dp_prior_action_mse={dp_prior_mse_f}", file=sys.stderr)
    sys.exit(44)
source_counts = eval_metrics.get("candidate_source_counts")
source_counts = source_counts if isinstance(source_counts, dict) else {}
non_dp_selected = 0
for name, count in source_counts.items():
    if str(name) == "dp_prior":
        continue
    try:
        non_dp_selected += int(count)
    except (TypeError, ValueError):
        pass
if non_dp_selected <= 0:
    print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
    print(f"reason={metrics_source}_selected_candidate_collapsed_to_dp_prior", file=sys.stderr)
    print(f"eval_candidate_source_counts={source_counts}", file=sys.stderr)
    sys.exit(44)

if payload.get("ready_for_formal_live_eval") is not True:
    print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
    print("reason=ready_for_formal_live_eval_false_after_metric_gate", file=sys.stderr)
    print(f"formal_training_floor_met={payload.get('formal_training_floor_met')}", file=sys.stderr)
    print(f"ready_for_offline_gate={payload.get('ready_for_offline_gate')}", file=sys.stderr)
    sys.exit(44)

import torch

try:
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
except Exception as exc:
    print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
    print(f"reason=checkpoint_load_failed {exc!r}", file=sys.stderr)
    sys.exit(45)
checkpoint_args = checkpoint_payload.get("args") if isinstance(checkpoint_payload.get("args"), dict) else {}
if checkpoint_args.get("generator_type") != payload.get("generator_type"):
    print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
    print(
        f"reason=checkpoint_generator_type_mismatch checkpoint={checkpoint_args.get('generator_type')} "
        f"summary={payload.get('generator_type')}",
        file=sys.stderr,
    )
    sys.exit(45)
try:
    checkpoint_candidate_samples = int(checkpoint_args.get("candidate_samples"))
    checkpoint_rank_diffusion_count = int(checkpoint_args.get("candidate_rank_diffusion_count"))
except (TypeError, ValueError):
    print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
    print("reason=checkpoint_invalid_diffusion_candidate_metadata", file=sys.stderr)
    sys.exit(45)
if require_diffusion and (checkpoint_candidate_samples <= 0 or checkpoint_rank_diffusion_count <= 0):
    print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
    print(
        f"reason=checkpoint_diffusion_candidate_metadata_disabled "
        f"candidate_samples={checkpoint_candidate_samples} "
        f"candidate_rank_diffusion_count={checkpoint_rank_diffusion_count}",
        file=sys.stderr,
    )
    sys.exit(45)
try:
    feature_dim = int(checkpoint_payload.get("feature_dim"))
    target_dim = int(checkpoint_payload.get("target_dim"))
except (TypeError, ValueError):
    print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
    print("reason=checkpoint_missing_feature_or_target_dim", file=sys.stderr)
    sys.exit(45)
if target_dim % 7 != 0:
    print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
    print(f"reason=checkpoint_target_dim_not_robot_action_multiple target_dim={target_dim}", file=sys.stderr)
    sys.exit(45)
expected_feature_dim = 35 + (target_dim // 7) * 14 + target_dim + 15
if feature_dim != expected_feature_dim:
    print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
    print(
        f"reason=checkpoint_feature_contract_mismatch feature_dim={feature_dim} "
        f"expected={expected_feature_dim} target_dim={target_dim}",
        file=sys.stderr,
    )
    sys.exit(45)
for key, checkpoint_value in (("feature_dim", feature_dim), ("target_dim", target_dim), ("action_horizon", target_dim // 7)):
    if payload.get(key) is None:
        continue
    try:
        summary_value = int(payload.get(key))
    except (TypeError, ValueError):
        print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
        print(f"reason=invalid_summary_{key} value={payload.get(key)}", file=sys.stderr)
        sys.exit(45)
    if summary_value != int(checkpoint_value):
        print("refusing_candidate_executor_live_eval=true", file=sys.stderr)
        print(
            f"reason=summary_checkpoint_{key}_mismatch "
            f"summary={summary_value} checkpoint={int(checkpoint_value)}",
            file=sys.stderr,
        )
        sys.exit(45)

print("candidate_executor_formal_gate_ok=true")
print(f"training_summary={summary_path}")
print(f"checkpoint={checkpoint_path}")
print(f"formal_live_eval_metrics_source={metrics_source}")
print(f"checkpoint_feature_dim={feature_dim}")
print(f"checkpoint_target_dim={target_dim}")
print(f"checkpoint_action_horizon={target_dim // 7}")
print("candidate_executor_metric_gate_ok=true")
print("gate_thresholds=" + json.dumps(thresholds, sort_keys=True))
print(
    "gate_metrics="
    + json.dumps(
        {
            "selected_action_mse": selected_mse_f,
            "dp_prior_action_mse": dp_prior_mse_f,
            "teacher_progress_mse": teacher_progress_mse_f,
            "teacher_value_mse": teacher_value_mse_f,
            "teacher_inserted_acc": teacher_inserted_acc_f,
            "teacher_dp_continuable_acc": teacher_dp_continuable_acc_f,
            "selected_non_dp_candidate_count": non_dp_selected,
            "metrics_source": metrics_source,
        },
        sort_keys=True,
    )
)
PY

cd "${ROOT}"
mkdir -p "${OUTPUT_ROOT}"
{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "step_id=${SLURM_STEP_ID}"
  echo "node_list=${SLURM_JOB_NODELIST:-}"
  echo "formal_root=${FORMAL_ROOT}"
  echo "training_summary=${TRAINING_SUMMARY}"
  echo "executor_checkpoint=${EXECUTOR_CHECKPOINT}"
  echo "candidate_outcome_scorer_checkpoint=${CANDIDATE_OUTCOME_SCORER_CHECKPOINT:-none}"
  echo "candidate_outcome_scorer_summary=${CANDIDATE_OUTCOME_SCORER_SUMMARY:-none}"
  echo "candidate_outcome_scorer_dp_margin=${CANDIDATE_OUTCOME_SCORER_DP_MARGIN}"
  echo "candidate_outcome_scorer_min_progress_delta=${CANDIDATE_OUTCOME_SCORER_MIN_PROGRESS_DELTA}"
  echo "candidate_outcome_scorer_min_continuable_prob=${CANDIDATE_OUTCOME_SCORER_MIN_CONTINUABLE_PROB}"
  echo "candidate_outcome_scorer_min_inserted_prob=${CANDIDATE_OUTCOME_SCORER_MIN_INSERTED_PROB}"
  echo "require_diffusion_generator=${REQUIRE_DIFFUSION_GENERATOR}"
  echo "sft_root=${SFT_ROOT}"
  echo "checkpoint_iter=${CHECKPOINT_ITER}"
  echo "eval_root=${EVAL_ROOT}"
  echo "condition_root=${CONDITION_ROOT}"
  echo "source_h5_root=${SOURCE_H5_ROOT}"
  echo "sample_indices=${SAMPLE_INDICES}"
  echo "max_samples=${MAX_SAMPLES}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "controller_action_source=candidate_executor"
  echo "executor_residual_scale=1.0"
  echo "boundary=Gated candidate executor live panel. This launcher refuses to run unless the formal candidate-executor summary satisfies the active 1/2/4-GPU plus 3-hour floor, allows live eval, and the summary-selected live-eval checkpoint loads. It binds the live run to clean-dense iter1500 Cosmos SFT and executes only scorer-selected candidate chunks with real reobservation."
} | tee "${OUTPUT_ROOT}/candidate_executor_after_gate_launcher_manifest.txt"

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
CONTROLLER_ACTION_SOURCE=candidate_executor \
EXECUTOR_CHECKPOINT="${EXECUTOR_CHECKPOINT}" \
CANDIDATE_OUTCOME_SCORER_CHECKPOINT="${CANDIDATE_OUTCOME_SCORER_CHECKPOINT}" \
CANDIDATE_OUTCOME_SCORER_SUMMARY="${CANDIDATE_OUTCOME_SCORER_SUMMARY}" \
CANDIDATE_OUTCOME_SCORER_DP_MARGIN="${CANDIDATE_OUTCOME_SCORER_DP_MARGIN}" \
CANDIDATE_OUTCOME_SCORER_MIN_PROGRESS_DELTA="${CANDIDATE_OUTCOME_SCORER_MIN_PROGRESS_DELTA}" \
CANDIDATE_OUTCOME_SCORER_MIN_CONTINUABLE_PROB="${CANDIDATE_OUTCOME_SCORER_MIN_CONTINUABLE_PROB}" \
CANDIDATE_OUTCOME_SCORER_MIN_INSERTED_PROB="${CANDIDATE_OUTCOME_SCORER_MIN_INSERTED_PROB}" \
EXECUTOR_RESIDUAL_SCALE=1.0 \
RUN_COSMOS_INFERENCE=true \
SAVE_LIVE_STATE_SNAPSHOTS=true \
ALLOW_LIVE_RECEDING_DIAGNOSTIC=false \
bash "${ROOT}/scripts/slurm/run_cosmos3_live_receding_panel_in_allocation.sh"
