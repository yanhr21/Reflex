#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this render smoke inside a compute-node srun step from a tmux-held allocation.
EOF
  exit 30
fi

RUN_GROUP="${RUN_GROUP:-static_rgb}"
RUN_NAME="${RUN_NAME:-smoke01}"
COUNT="${COUNT:-4}"
EPISODE_START="${EPISODE_START:-0}"
NUM_ENVS="${NUM_ENVS:-1}"
MAX_VIDEOS="${MAX_VIDEOS:-8}"
RUN_RENDER_CANARY="${RUN_RENDER_CANARY:-true}"
RENDER_CANARY_TIMEOUT="${RENDER_CANARY_TIMEOUT:-3m}"
RENDER_SHADER_PACK="${RENDER_SHADER_PACK:-minimal}"
RENDER_CANARY_API="${RENDER_CANARY_API:-gym}"
REPLAY_SHADER="${REPLAY_SHADER:-minimal}"
VIDEO_FPS="${VIDEO_FPS:-30}"
DATASET_SMOKE_ONLY="${DATASET_SMOKE_ONLY:-true}"
HUMAN_REVIEW_REQUIRED="${HUMAN_REVIEW_REQUIRED:-true}"
LARGE_SCALE_PRODUCTION_ALLOWED="${LARGE_SCALE_PRODUCTION_ALLOWED:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/experiments/maniskill/runs/01_dataset/${RUN_GROUP}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/01_dataset/${RUN_GROUP}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"

if [[ "${DATASET_SMOKE_ONLY}" == "true" ]]; then
  RUN_STATUS="${RUN_STATUS:-smoke_complete}"
  RUN_NOTES="${RUN_NOTES:-Stage 1 static expert RGB smoke for human review before production.}"
else
  RUN_STATUS="${RUN_STATUS:-render_complete}"
  RUN_NOTES="${RUN_NOTES:-Stage 1 static expert RGB full production after human smoke approval.}"
fi

if [[ "${RUN_GROUP}" =~ p03_|full_pipeline|[0-9]{8}|server[0-9]+|job[0-9]+ ]]; then
  echo "refusing_unreadable_run_group=true" >&2
  echo "run_group=${RUN_GROUP}" >&2
  exit 41
fi
if [[ "${RUN_NAME}" =~ p03_|full_pipeline|[0-9]{8}|server[0-9]+|job[0-9]+ ]]; then
  echo "refusing_unreadable_run_name=true" >&2
  echo "run_name=${RUN_NAME}" >&2
  exit 42
fi

MS_DIR="${ROOT}/deps/ManiSkill_clean"
VENV="${ROOT}/.venv"
SRC_DIR="${ROOT}/data/official_replay/PegInsertionSide-v1/motionplanning"
SRC_H5="${SRC_DIR}/trajectory.h5"
SRC_JSON="${SRC_DIR}/trajectory.json"
STATE_H5="${SRC_DIR}/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
STATE_JSON="${SRC_DIR}/trajectory.state.pd_ee_delta_pose.physx_cpu.json"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

cd "${ROOT}"

export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export DISPLAY="${DISPLAY:-}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export PYTHONPATH="${MS_DIR}/examples/baselines/diffusion_policy:${ROOT}/scripts/training:${PYTHONPATH:-}"

MANIFEST="${OUTPUT_DIR}/manifest.txt"
SUMMARY="${OUTPUT_DIR}/summary.json"
REVIEW_DIR="${OUTPUT_DIR}/review"
WORK_H5="${OUTPUT_DIR}/trajectory.h5"
WORK_JSON="${OUTPUT_DIR}/trajectory.json"

if [[ "${EPISODE_START}" == "0" && "${COUNT}" == "1000" ]]; then
  cp -f "${SRC_H5}" "${WORK_H5}"
  cp -f "${SRC_JSON}" "${WORK_JSON}"
else
  "${VENV}/bin/python" -u "${ROOT}/scripts/world_model/make_static_replay_shard.py" \
    --src-h5 "${SRC_H5}" \
    --src-json "${SRC_JSON}" \
    --out-h5 "${WORK_H5}" \
    --out-json "${WORK_JSON}" \
    --episode-start "${EPISODE_START}" \
    --episode-count "${COUNT}"
fi

{
  echo "timestamp=$(date -Is)"
  echo "phase=01_dataset"
  echo "dataset_class=A_static_expert"
  echo "run_group=${RUN_GROUP}"
  echo "run_name=${RUN_NAME}"
  echo "job_id=${SLURM_JOB_ID}"
  echo "step_id=${SLURM_STEP_ID}"
  echo "node_list=${SLURM_JOB_NODELIST:-unknown}"
  echo "log_file=${LOG_FILE}"
  echo "vk_icd_filenames=${VK_ICD_FILENAMES}"
  echo "hdf5_use_file_locking=${HDF5_USE_FILE_LOCKING}"
  echo "source_paths=${SRC_H5},${SRC_JSON},${STATE_H5},${STATE_JSON}"
  echo "src_h5=${SRC_H5}"
  echo "src_json=${SRC_JSON}"
  echo "state_h5=${STATE_H5}"
  echo "state_json=${STATE_JSON}"
  echo "work_h5=${WORK_H5}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "count=${COUNT}"
  echo "episode_start=${EPISODE_START}"
  echo "num_envs=${NUM_ENVS}"
  echo "run_render_canary=${RUN_RENDER_CANARY}"
  echo "render_canary_timeout=${RENDER_CANARY_TIMEOUT}"
  echo "render_shader_pack=${RENDER_SHADER_PACK}"
  echo "render_canary_api=${RENDER_CANARY_API}"
  echo "replay_shader=${REPLAY_SHADER}"
  echo "video_fps=${VIDEO_FPS}"
  echo "method_evidence_allowed=false"
  echo "teacher_evidence_allowed=false"
  echo "dataset_smoke_only=${DATASET_SMOKE_ONLY}"
  echo "human_review_required=${HUMAN_REVIEW_REQUIRED}"
  echo "large_scale_production_allowed=${LARGE_SCALE_PRODUCTION_ALLOWED}"
  echo "controller=official_motionplanning_replay_to_pd_ee_delta_pose_video"
  echo "action_contract=pd_ee_delta_pose"
  echo "allowed_losses=cosmos_static_future,phase_extraction_after_alignment"
  echo "disallowed_losses=dp_bc_from_rgb_render_only,positive_dynamic_policy_data,final_method_evidence"
  echo "rgb_required=true"
  echo "state_based_dp_reference=true"
  echo "forbidden_state_intervention_expected=false"
  echo "notes=${RUN_NOTES}"
  echo "maniskill_commit=$(git -C "${MS_DIR}" rev-parse HEAD)"
} | tee "${MANIFEST}"

nvidia-smi || true

if [[ "${RUN_RENDER_CANARY}" == "true" ]]; then
  {
    echo "dataset_smoke_status=render_canary_in_progress_no_replay"
    echo "render_canary_dir=${OUTPUT_DIR}/render_canary"
  } | tee -a "${MANIFEST}"
  set +e
  timeout -k 15s "${RENDER_CANARY_TIMEOUT}" "${VENV}/bin/python" -u \
    "${ROOT}/scripts/world_model/render_min_canary.py" \
      --output-dir "${OUTPUT_DIR}/render_canary" \
      --shader-pack "${RENDER_SHADER_PACK}" \
      --render-api "${RENDER_CANARY_API}"
  RENDER_CANARY_STATUS="${PIPESTATUS[0]}"
  set -e
  echo "render_canary_exit_code=${RENDER_CANARY_STATUS}" | tee -a "${MANIFEST}"
  if [[ "${RENDER_CANARY_STATUS}" -ne 0 ]]; then
    echo "dataset_smoke_status=blocked_render_canary_failed_no_replay" | tee -a "${MANIFEST}"
    exit "${RENDER_CANARY_STATUS}"
  fi
  echo "dataset_smoke_status=render_canary_passed_starting_replay" | tee -a "${MANIFEST}"
fi

cd "${MS_DIR}"
"${VENV}/bin/python" -m mani_skill.trajectory.replay_trajectory \
  --traj-path "${WORK_H5}" \
  --use-first-env-state \
  -c pd_ee_delta_pose \
  -o state \
  --shader "${REPLAY_SHADER}" \
  --video-fps "${VIDEO_FPS}" \
  --save-video \
  --allow-failure \
  --count "${COUNT}" \
  --num-envs "${NUM_ENVS}" \
  -b physx_cpu

cd "${ROOT}"
"${VENV}/bin/python" "${ROOT}/scripts/tools/extract_video_frames.py" \
  --video-dir "${OUTPUT_DIR}" \
  --output-dir "${REVIEW_DIR}/frames" \
  --max-videos "${MAX_VIDEOS}"

video_count="$(find "${OUTPUT_DIR}" -type f -name '*.mp4' | wc -l | tr -d ' ')"
frame_count="$(find "${REVIEW_DIR}/frames" -type f -name '*.png' 2>/dev/null | wc -l | tr -d ' ')"
video_bytes="$(find "${OUTPUT_DIR}" -type f -name '*.mp4' -printf '%s\n' | awk '{s+=$1} END {print s+0}')"

cat > "${SUMMARY}" <<EOF
{
  "phase": "01_dataset",
  "dataset_class": "A_static_expert",
  "run_group": "${RUN_GROUP}",
  "run_name": "${RUN_NAME}",
  "output_dir": "${OUTPUT_DIR}",
  "log_file": "${LOG_FILE}",
  "job_id": "${SLURM_JOB_ID}",
  "step_id": "${SLURM_STEP_ID}",
  "node_list": "${SLURM_JOB_NODELIST:-unknown}",
  "count": ${COUNT},
  "episode_start": ${EPISODE_START},
  "num_envs": ${NUM_ENVS},
  "run_render_canary": ${RUN_RENDER_CANARY},
  "render_canary_timeout": "${RENDER_CANARY_TIMEOUT}",
  "render_shader_pack": "${RENDER_SHADER_PACK}",
  "render_canary_api": "${RENDER_CANARY_API}",
  "replay_shader": "${REPLAY_SHADER}",
  "video_fps": ${VIDEO_FPS},
  "video_count": ${video_count},
  "frame_count": ${frame_count},
  "video_bytes": ${video_bytes},
  "human_review_required": ${HUMAN_REVIEW_REQUIRED},
  "large_scale_production_allowed": ${LARGE_SCALE_PRODUCTION_ALLOWED},
  "dataset_smoke_only": ${DATASET_SMOKE_ONLY},
  "method_evidence_allowed": false,
  "allowed_losses": ["cosmos_static_future", "phase_extraction_after_alignment"],
  "disallowed_losses": ["dp_bc_from_rgb_render_only", "positive_dynamic_policy_data", "final_method_evidence"],
  "disallowed_losses_until_alignment": ["dp_bc_from_rgb_render_only"],
  "status": "${RUN_STATUS}"
}
EOF

{
  echo "video_files:"
  find "${OUTPUT_DIR}" -type f -name '*.mp4' -printf '%p %s\n' | sort
  echo "frame_files:"
  find "${REVIEW_DIR}/frames" -type f -name '*.png' -printf '%p %s\n' | sort | head -n 80
  echo "summary=${SUMMARY}"
  echo "status=complete"
} | tee -a "${MANIFEST}"
