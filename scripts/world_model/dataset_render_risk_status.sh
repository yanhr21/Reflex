#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
SMOKE_RUN_GROUP="${SMOKE_RUN_GROUP:-static_rgb}"
SMOKE_RUN_NAME="${SMOKE_RUN_NAME:-smoke05}"
SMOKE_OUT_DIR="${SMOKE_OUT_DIR:-${ROOT}/experiments/maniskill/runs/01_dataset/${SMOKE_RUN_GROUP}/${SMOKE_RUN_NAME}}"
SMOKE_LOG="${SMOKE_LOG:-${ROOT}/logs/01_dataset/${SMOKE_RUN_GROUP}/${SMOKE_RUN_NAME}.log}"
FULL_LAUNCHER="${FULL_LAUNCHER:-${ROOT}/scripts/slurm/launch_dataset_static_rgb_full_tmux.sh}"
STAGE_SMOKE_LAUNCHER="${STAGE_SMOKE_LAUNCHER:-${ROOT}/scripts/slurm/launch_dataset_stage_smoke_tmux_common.sh}"
STAGE_PRODUCTION_LAUNCHER="${STAGE_PRODUCTION_LAUNCHER:-${ROOT}/scripts/slurm/launch_dataset_stage_production_tmux_common.sh}"
STATIC_SMOKE_LAUNCHER="${STATIC_SMOKE_LAUNCHER:-${ROOT}/scripts/slurm/launch_dataset_static_rgb_smoke_srun_tmux.sh}"
STATIC_SMOKE_RUNNER="${STATIC_SMOKE_RUNNER:-${ROOT}/scripts/slurm/run_dataset_static_rgb_smoke_in_allocation.sh}"
RUNBOOK="${RUNBOOK:-${ROOT}/docs/dataset_smoke_runbook.md}"
LEGACY_RENDER_DOC="${LEGACY_RENDER_DOC:-${ROOT}/docs/legacy/RENDERING_CLUSTER_VULKAN.md}"

echo "dataset_render_risk_status_ok=true"
echo "read_only=true"
echo "submits_slurm=false"
echo "smoke_run=${SMOKE_RUN_GROUP}/${SMOKE_RUN_NAME}"
echo "smoke_out_dir=${SMOKE_OUT_DIR}"
echo "smoke_log=${SMOKE_LOG}"
echo "runbook=${RUNBOOK}"
echo "legacy_render_doc=${LEGACY_RENDER_DOC}"

default_excludes() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    sed -nE 's/^EXCLUDE_NODES="\$\{EXCLUDE_NODES:-([^}]*)\}".*/\1/p' "${path}" | head -n 1
  fi
}

echo "[render_environment_contract]"
for path in "${STATIC_SMOKE_LAUNCHER}" "${STATIC_SMOKE_RUNNER}" "${STAGE_SMOKE_LAUNCHER}" "${STAGE_PRODUCTION_LAUNCHER}" "${FULL_LAUNCHER}"; do
  label="$(basename "${path}")"
  echo "  ${label}=${path}"
  echo "  ${label}_exists=$([[ -f "${path}" ]] && echo true || echo false)"
done
if grep -q 'VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json"' \
  "${STATIC_SMOKE_LAUNCHER}" "${STATIC_SMOKE_RUNNER}" "${STAGE_SMOKE_LAUNCHER}" "${STAGE_PRODUCTION_LAUNCHER}" "${FULL_LAUNCHER}" 2>/dev/null; then
  echo "  vk_icd_system_nvidia_configured=true"
else
  echo "  vk_icd_system_nvidia_configured=false"
fi
if grep -q 'HDF5_USE_FILE_LOCKING=FALSE' \
  "${STATIC_SMOKE_LAUNCHER}" "${STATIC_SMOKE_RUNNER}" "${STAGE_SMOKE_LAUNCHER}" "${STAGE_PRODUCTION_LAUNCHER}" "${FULL_LAUNCHER}" 2>/dev/null; then
  echo "  hdf5_file_locking_disabled=true"
else
  echo "  hdf5_file_locking_disabled=false"
fi
if grep -q 'RUN_RENDER_CANARY.*true' \
  "${STATIC_SMOKE_LAUNCHER}" "${STATIC_SMOKE_RUNNER}" "${STAGE_SMOKE_LAUNCHER}" "${STAGE_PRODUCTION_LAUNCHER}" "${FULL_LAUNCHER}" 2>/dev/null; then
  echo "  render_canary_default_true=true"
else
  echo "  render_canary_default_true=false"
fi
if grep -q 'RENDER_SHADER_PACK.*minimal' \
  "${STATIC_SMOKE_LAUNCHER}" "${STATIC_SMOKE_RUNNER}" "${STAGE_SMOKE_LAUNCHER}" "${STAGE_PRODUCTION_LAUNCHER}" "${FULL_LAUNCHER}" 2>/dev/null; then
  echo "  minimal_shader_default_present=true"
else
  echo "  minimal_shader_default_present=false"
fi

echo "[current_static_smoke_render_evidence]"
if [[ -f "${SMOKE_OUT_DIR}/manifest.txt" ]]; then
  echo "  manifest_exists=true"
  sed -nE 's/^(job_id|node_list|vk_icd_filenames|hdf5_use_file_locking|run_render_canary|render_shader_pack|render_canary_api|replay_shader|render_canary_exit_code|status)=/  manifest_\1=/p' \
    "${SMOKE_OUT_DIR}/manifest.txt"
else
  echo "  manifest_exists=false"
fi
echo "  summary_exists=$([[ -f "${SMOKE_OUT_DIR}/summary.json" ]] && echo true || echo false)"
if [[ -f "${SMOKE_OUT_DIR}/render_canary/frame.png" ]]; then
  echo "  render_canary_frame_exists=true"
  echo "  render_canary_frame_bytes=$(stat -c '%s' "${SMOKE_OUT_DIR}/render_canary/frame.png")"
else
  echo "  render_canary_frame_exists=false"
fi
static_video_count="$(find "${SMOKE_OUT_DIR}" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')"
static_review_frame_count="$(find "${SMOKE_OUT_DIR}/review/frames" -type f -name '*.png' 2>/dev/null | wc -l | tr -d ' ')"
echo "  video_count=${static_video_count}"
echo "  review_frame_count=${static_review_frame_count}"

echo "[current_bcd_smoke_render_evidence]"
for spec in \
  "B dynamic_rgb smoke01" \
  "C frozen_dp_dynamic smoke01" \
  "D future_teacher smoke01"; do
  set -- ${spec}
  label="$1"
  run_group="$2"
  run_name="$3"
  run_dir="${ROOT}/experiments/maniskill/runs/01_dataset/${run_group}/${run_name}"
  summary="${run_dir}/summary.json"
  manifest="${run_dir}/manifest.txt"
  echo "  [${label}]"
  echo "    run=${run_group}/${run_name}"
  echo "    manifest_exists=$([[ -f "${manifest}" ]] && echo true || echo false)"
  echo "    summary_exists=$([[ -f "${summary}" ]] && echo true || echo false)"
  if [[ -f "${manifest}" ]]; then
    sed -nE 's/^(job_id|node_list|fps|run_render_canary|render_shader_pack|render_canary_api)=/    manifest_\1=/p' "${manifest}"
  fi
  if [[ -f "${summary}" ]]; then
    sed -nE 's/.*"(frame_count|video_count)"[[:space:]]*:[[:space:]]*([^,]+).*/    summary_\1=\2/p' "${summary}"
    sed -nE 's/.*"(state_intervention|snap_or_teleport)"[[:space:]]*:[[:space:]]*([^,]+).*/    summary_\1=\2/p' "${summary}"
  fi
done

echo "[node_risk_observations]"
echo "  current_static_success_nodes=server02,server20,server27,server44,server60"
echo "  current_bcd_smoke_success_nodes=server10,server57"
echo "  current_device_lost_nodes=server34,server39,server43,server53,server58"
echo "  current_canary_timeout_nodes=server28,server36,server46,server53,server56,server60,server63"
echo "  current_scheduler_or_resource_risk_nodes=server57,server59"
echo "  mixed_evidence_nodes=server10,server44,server57"
echo "  evidence_scope=dataset_static_rgb_and_bcd_smoke_2026_07_06_to_2026_07_08"
echo "  policy=not_static_blacklist_use_job_local_canary_and_manifest_evidence"
echo "  stage_smoke_default_exclude_nodes=$(default_excludes "${STAGE_SMOKE_LAUNCHER}")"
echo "  stage_production_default_exclude_nodes=$(default_excludes "${STAGE_PRODUCTION_LAUNCHER}")"
echo "  static_full_default_exclude_nodes=$(default_excludes "${FULL_LAUNCHER}")"
echo "  smoke05_exclude_nodes=$(sed -nE 's/^exclude_nodes=(.*)$/\1/p' "${SMOKE_LOG}" 2>/dev/null | tail -n 1)"

echo "[known_incidents_from_runbook]"
if [[ -f "${RUNBOOK}" ]]; then
  for node in server28 server34 server36 server39 server43 server46 server53 server56 server57 server58 server59 server60 server63; do
    if grep -q "${node}" "${RUNBOOK}"; then
      echo "  ${node}_recorded=true"
    else
      echo "  ${node}_recorded=false"
    fi
  done
else
  echo "  runbook_exists=false"
fi

echo "[production_guidance]"
echo "  default_gpus=1"
echo "  use_tmux_held_srun=true"
echo "  run_render_canary_before_replay=true"
echo "  keep_vk_icd=/etc/vulkan/icd.d/nvidia_icd.json"
echo "  keep_hdf5_use_file_locking_false=true"
echo "  keep_replay_shader=minimal"
echo "  if_canary_fails=stop_before_collection_and_classify_node_or_render_failure"
echo "  if_scheduler_slow=reduce_cpu_memory_walltime_when_scientifically_acceptable"
echo "  if_known_risk_node_is_used=smoke_or_canary_only_record_node_evidence_before_scaling"
