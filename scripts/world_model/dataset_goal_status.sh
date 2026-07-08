#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
RUN_GROUP="${RUN_GROUP:-static_rgb}"
RUN_NAME="${RUN_NAME:-smoke05}"
SMOKE_OUT_DIR="${SMOKE_OUT_DIR:-${ROOT}/experiments/maniskill/runs/01_dataset/${RUN_GROUP}/${RUN_NAME}}"

echo "dataset_goal_status_ok=true"
echo "timestamp=$(date -Is)"
echo "active_smoke_run=${RUN_GROUP}/${RUN_NAME}"
if [[ -f "${SMOKE_OUT_DIR}/summary.json" && ! -f "${SMOKE_OUT_DIR}/human_review_approved.txt" ]]; then
  echo "goal_blocked_on_human_review=true"
  echo "human_review_target=${SMOKE_OUT_DIR}/0.mp4"
else
  echo "goal_blocked_on_human_review=false"
fi

echo "[stage1_static_rgb_smoke]"
RUN_GROUP="${RUN_GROUP}" RUN_NAME="${RUN_NAME}" "${ROOT}/scripts/world_model/dataset_smoke_status.sh" | sed 's/^/  /'

echo "[stage1_static_rgb_manifest]"
if "${ROOT}/scripts/world_model/validate_dataset_run_manifest.sh" "${SMOKE_OUT_DIR}" >/tmp/reflex_manifest_status_$$.txt 2>&1; then
  sed 's/^/  /' /tmp/reflex_manifest_status_$$.txt
else
  sed 's/^/  /' /tmp/reflex_manifest_status_$$.txt
fi
rm -f /tmp/reflex_manifest_status_$$.txt

echo "[stage1_static_rgb_review]"
RUN_GROUP="${RUN_GROUP}" RUN_NAME="${RUN_NAME}" "${ROOT}/scripts/world_model/dataset_review_status.sh" | sed 's/^/  /'

echo "[class_review_gates]"
"${ROOT}/scripts/world_model/dataset_class_review_status.sh" | sed 's/^/  /'

echo "[stage1_static_rgb_production_gate]"
if RUN_GROUP="${RUN_GROUP}" RUN_NAME="${RUN_NAME}" "${ROOT}/scripts/world_model/require_dataset_smoke_approved.sh" >/tmp/reflex_dataset_gate_$$.txt 2>&1; then
  sed 's/^/  /' /tmp/reflex_dataset_gate_$$.txt
else
  sed 's/^/  /' /tmp/reflex_dataset_gate_$$.txt
fi
rm -f /tmp/reflex_dataset_gate_$$.txt

echo "[stage1_static_rgb_full]"
"${ROOT}/scripts/world_model/dataset_full_static_status.sh" | sed 's/^/  /'

echo "[next_stage_readiness]"
"${ROOT}/scripts/world_model/dataset_next_stage_status.sh" | sed 's/^/  /'

echo "[post_approval_plan]"
"${ROOT}/scripts/world_model/dataset_post_approval_plan.sh" | sed 's/^/  /'

echo "[render_risk_status]"
"${ROOT}/scripts/world_model/dataset_render_risk_status.sh" | sed 's/^/  /'

echo "[production_status]"
"${ROOT}/scripts/world_model/dataset_production_status.sh" | sed 's/^/  /'

echo "[source_recovery]"
"${ROOT}/scripts/world_model/dataset_source_recovery_status.sh" | sed 's/^/  /'

echo "[dynamic_adapter]"
if "${ROOT}/scripts/world_model/dataset_dynamic_adapter_status.sh" >/tmp/reflex_dynamic_adapter_$$.txt 2>&1; then
  sed 's/^/  /' /tmp/reflex_dynamic_adapter_$$.txt
else
  sed 's/^/  /' /tmp/reflex_dynamic_adapter_$$.txt
fi
rm -f /tmp/reflex_dynamic_adapter_$$.txt

echo "[active_data_registry]"
registry="${ROOT}/experiments/maniskill/data/active"
if [[ -d "${registry}" ]]; then
  echo "  active_registry_exists=true"
else
  echo "  active_registry_exists=false"
fi

for path in \
  "${registry}/a_static/official_state_pd_ee_delta_pose.h5" \
  "${registry}/a_static/official_state_pd_ee_delta_pose.json" \
  "${registry}/legacy_733/fix3_h5_paths_canonical.txt" \
  "${registry}/legacy_733/manifest.json" \
  "${registry}/b_dynamic_legacy_bootstrap/rgbd_root" \
  "${registry}/b_dynamic_legacy_bootstrap/inspection.json" \
  "${registry}/b_dynamic_legacy_bootstrap/inspection.md"; do
  label="$(basename "${path}")"
  if [[ -e "${path}" ]]; then
    echo "  ${label}_exists=true"
    echo "  ${label}_target=$(readlink -f "${path}")"
  else
    echo "  ${label}_exists=false"
  fi
done

bootstrap_root="${registry}/b_dynamic_legacy_bootstrap/rgbd_root"
echo "[b_dynamic_legacy_bootstrap]"
if "${ROOT}/scripts/world_model/dataset_bootstrap_status.sh" >/tmp/reflex_bootstrap_status_$$.txt 2>&1; then
  sed 's/^/  /' /tmp/reflex_bootstrap_status_$$.txt
else
  sed 's/^/  /' /tmp/reflex_bootstrap_status_$$.txt
fi
rm -f /tmp/reflex_bootstrap_status_$$.txt

echo "[training_inputs]"
if "${ROOT}/scripts/world_model/dataset_training_inputs_status.sh" >/tmp/reflex_training_inputs_$$.txt 2>&1; then
  sed 's/^/  /' /tmp/reflex_training_inputs_$$.txt
else
  sed 's/^/  /' /tmp/reflex_training_inputs_$$.txt
fi
rm -f /tmp/reflex_training_inputs_$$.txt

echo "[dataset_classes]"
echo "  A_static_expert_state_source=available"
if [[ -f "${SMOKE_OUT_DIR}/summary.json" ]]; then
  echo "  A_static_expert_rgb_smoke=present"
  echo "  A_static_expert_rgb_smoke_path=${SMOKE_OUT_DIR}/summary.json"
else
  echo "  A_static_expert_rgb_smoke=missing_or_pending"
fi
echo "  A_static_expert_full_rgb=blocked_until_smoke_approval"
if [[ -d "${bootstrap_root}" ]]; then
  echo "  B_dynamic_legacy_bootstrap=available_limited_use_not_new_production"
else
  echo "  B_dynamic_legacy_bootstrap=missing"
fi
echo "  B_dynamic_rgb_observation_new_production=blocked_until_stage1_rgb_smoke_review_and_runner"
echo "  C_frozen_dp_dynamic_failure=blocked_until_stage1_rgb_smoke_review"
echo "  D_future_frame_teacher=blocked_until_stage1_rgb_smoke_review_and_interface_check"
echo "  E_cosmos_predicted_cooperation=blocked_until_B_D_and_cosmos_readout_validation"

echo "[slurm]"
if "${ROOT}/scripts/world_model/dataset_slurm_status.sh" >/tmp/reflex_slurm_status_$$.txt 2>&1; then
  sed 's/^/  /' /tmp/reflex_slurm_status_$$.txt
else
  sed 's/^/  /' /tmp/reflex_slurm_status_$$.txt
fi
rm -f /tmp/reflex_slurm_status_$$.txt
