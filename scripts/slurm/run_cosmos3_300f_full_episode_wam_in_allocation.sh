#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this script only inside a compute-node srun step, for example from the tmux-held salloc shell: srun --jobid=$SLURM_JOB_ID --gres=gpu:1 --cpus-per-task=16 bash scripts/slurm/run_cosmos3_300f_full_episode_wam_in_allocation.sh
EOF
  exit 30
fi

JOB_ID="${SLURM_JOB_ID}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SOURCE_DATASET_ROOT="${SOURCE_DATASET_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_dataset_full1000_maniskill_default_regen_20260606_0055}"
CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_full1000_rgb_300step_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_${STAMP}}"
FIX3_USER_APPROVAL_FILE="${FIX3_USER_APPROVAL_FILE:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/FIX3_USER_APPROVED_FOR_SFT.txt}"
REQUIRE_FIX3_USER_APPROVAL="${REQUIRE_FIX3_USER_APPROVAL:-auto}"
ALLOW_OLD_FULL1000_INVALID_SOURCE_DIAGNOSTIC="${ALLOW_OLD_FULL1000_INVALID_SOURCE_DIAGNOSTIC:-false}"
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-${ROOT}/checkpoints/cosmos3/Cosmos3-Nano-Policy-DROID-DCP}"
WAN_VAE_PATH="${WAN_VAE_PATH:-${ROOT}/checkpoints/cosmos3/wan22_vae/Wan2.2_VAE.pth}"
COSMOS_VENV="${COSMOS_VENV:-${ROOT}/.venv_cosmos313}"
LOCAL_TOKENIZER_DIR="${LOCAL_TOKENIZER_DIR:-${ROOT}/checkpoints/cosmos3/Cosmos3-Nano-Policy-DROID}"
SFT_TOML="${SFT_TOML:-${ROOT}/external/cosmos-framework/examples/toml/sft_config/vision_sft_nano.toml}"
JOB_NAME="${JOB_NAME:-vision_sft_droid_policy_full1000_rgb_300step_wam}"
WAM_SFT_MODE="${WAM_SFT_MODE:-full_episode_joint_policy_history_action_300step}"
MAX_ITER="${MAX_ITER:-1500}"
SAVE_ITER="${SAVE_ITER:-300}"
RUN_VALIDATION="${RUN_VALIDATION:-true}"
VALIDATION_ITER="${VALIDATION_ITER:-300}"
MAX_VAL_ITER="${MAX_VAL_ITER:-40}"
MASTER_PORT="${MASTER_PORT:-50183}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-45056}"
OPTIMIZER_KEYS_TO_SELECT="${OPTIMIZER_KEYS_TO_SELECT:-moe_gen,time_embedder,vae2llm,llm2vae,action2llm,llm2action,action_modality_embed}"
OPTIMIZER_LR="${OPTIMIZER_LR:-1.0e-4}"
SCHEDULER_CYCLE_LENGTH="${SCHEDULER_CYCLE_LENGTH:-${MAX_ITER}}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-10}"
SCHEDULER_F_MIN="${SCHEDULER_F_MIN:-0.5}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-2.0}"
NORMALIZE_LOSS_BY_ACTIVE="${NORMALIZE_LOSS_BY_ACTIVE:-true}"
INDEPENDENT_ACTION_SCHEDULE="${INDEPENDENT_ACTION_SCHEDULE:-true}"
SHIFT_ACTION="${SHIFT_ACTION:-1}"
ENFORCE_FIX1_ACTION_RECIPE="${ENFORCE_FIX1_ACTION_RECIPE:-true}"
ALLOW_NON_FIX1_ACTION_RECIPE_DIAGNOSTIC="${ALLOW_NON_FIX1_ACTION_RECIPE_DIAGNOSTIC:-false}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DATA_PARALLEL_SHARD_DEGREE="${DATA_PARALLEL_SHARD_DEGREE:-1}"
DATA_PARALLEL_REPLICATE_DEGREE="${DATA_PARALLEL_REPLICATE_DEGREE:-1}"
CONTEXT_PARALLEL_SHARD_DEGREE="${CONTEXT_PARALLEL_SHARD_DEGREE:-1}"
EXPORT_CPUS="${EXPORT_CPUS:-8}"
LIBFFI_COMPAT_DIR="${LIBFFI_COMPAT_DIR:-${COSMOS_VENV}/lib_compat}"
TOTAL_VIDEO_FRAMES="${TOTAL_VIDEO_FRAMES:-301}"
TOTAL_ACTION_STEPS="${TOTAL_ACTION_STEPS:-300}"
EXPECTED_SOURCE_EPISODES="${EXPECTED_SOURCE_EPISODES:-1000}"
MAX_RECORDS="${MAX_RECORDS:-0}"
CONDITIONING_LATENT_FRAMES="${CONDITIONING_LATENT_FRAMES:-8}"
TEMPORAL_COMPRESSION_FACTOR="${TEMPORAL_COMPRESSION_FACTOR:-4}"
TEMPORAL_INTERVAL_MODE="${TEMPORAL_INTERVAL_MODE:-force_one}"
ACTION_CONDITIONED_SFT="${ACTION_CONDITIONED_SFT:-true}"
REQUIRE_STATE_TARGETS="${REQUIRE_STATE_TARGETS:-true}"
STRICT_FULL_PREFLIGHT="${STRICT_FULL_PREFLIGHT:-true}"
SIDECAR_TARGET_MODE="${SIDECAR_TARGET_MODE:-future_aligned_state}"
PREFIX_ROLE_SOURCE="${PREFIX_ROLE_SOURCE:-sampled}"
DENSE_RECEDING_PREFIX_STRIDE="${DENSE_RECEDING_PREFIX_STRIDE:-0}"
ROLE_WEIGHT_CONFIG="${ROLE_WEIGHT_CONFIG:-}"
LATE_REBIND_WEIGHT="${LATE_REBIND_WEIGHT:-1}"
LATE_REBIND_ROLES="${LATE_REBIND_ROLES:-target_motion_observed,target_post_motion,insert_resume}"
LATE_REBIND_MIN_ABS_X="${LATE_REBIND_MIN_ABS_X:-0.05}"
LATE_REBIND_MIN_ABS_Y="${LATE_REBIND_MIN_ABS_Y:-0.01}"
LATE_REBIND_MIN_ABS_Z="${LATE_REBIND_MIN_ABS_Z:-0.004}"
MIN_LATE_REBIND_CANDIDATES="${MIN_LATE_REBIND_CANDIDATES:-0}"
RUN_LIVE_QUERY_COVERAGE_AUDIT="${RUN_LIVE_QUERY_COVERAGE_AUDIT:-false}"
LIVE_QUERY_COVERAGE_SUMMARIES="${LIVE_QUERY_COVERAGE_SUMMARIES:-}"
LIVE_QUERY_COVERAGE_PREFIX_TOLERANCE="${LIVE_QUERY_COVERAGE_PREFIX_TOLERANCE:-16}"
LIVE_QUERY_COVERAGE_REL_L2_TOLERANCE="${LIVE_QUERY_COVERAGE_REL_L2_TOLERANCE:-0.05}"
LIVE_QUERY_COVERAGE_REL_YZ_TOLERANCE="${LIVE_QUERY_COVERAGE_REL_YZ_TOLERANCE:-0.03}"
LIVE_QUERY_COVERAGE_MAX_UNDERCOVERED_COUNT="${LIVE_QUERY_COVERAGE_MAX_UNDERCOVERED_COUNT:-0}"
LIVE_QUERY_COVERAGE_MAX_UNDERCOVERED_FRACTION="${LIVE_QUERY_COVERAGE_MAX_UNDERCOVERED_FRACTION:-0.0}"
CLEAN_DENSE_PREFLIGHT_DIAGNOSTIC_NOT_READY_REASON="${CLEAN_DENSE_PREFLIGHT_DIAGNOSTIC_NOT_READY_REASON:-}"
FORCE_EXPORT="${FORCE_EXPORT:-false}"
RUN_SFT="${RUN_SFT:-true}"

TRAIN_JSONL="${TRAIN_JSONL:-${CONDITION_ROOT}/train/video_action_dataset_file.jsonl}"
VAL_JSONL="${VAL_JSONL:-${CONDITION_ROOT}/val/video_action_dataset_file.jsonl}"

find_cosmos_ffmpeg_bin() {
  local bin
  shopt -s nullglob
  for bin in "${COSMOS_VENV}"/lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*; do
    if [[ -x "${bin}" ]]; then
      printf '%s\n' "${bin}"
      shopt -u nullglob
      return 0
    fi
  done
  shopt -u nullglob
  return 1
}

cosmos_nvidia_lib_dirs() {
  local dirs=()
  local site d
  shopt -s nullglob
  for site in "${COSMOS_VENV}"/lib/python*/site-packages; do
    for d in "${site}"/nvidia/*/lib "${site}"/nvidia/*/lib64; do
      [[ -d "${d}" ]] && dirs+=("${d}")
    done
  done
  shopt -u nullglob
  local IFS=:
  printf '%s' "${dirs[*]}"
}

refresh_cosmos_ld_library_path() {
  local nvidia_libs
  nvidia_libs="$(cosmos_nvidia_lib_dirs)"
  if [[ -d "${LIBFFI_COMPAT_DIR}" && -n "${nvidia_libs}" ]]; then
    export LD_LIBRARY_PATH="${LIBFFI_COMPAT_DIR}:${nvidia_libs}:${LD_LIBRARY_PATH:-}"
  elif [[ -d "${LIBFFI_COMPAT_DIR}" ]]; then
    export LD_LIBRARY_PATH="${LIBFFI_COMPAT_DIR}:${LD_LIBRARY_PATH:-}"
  elif [[ -n "${nvidia_libs}" ]]; then
    export LD_LIBRARY_PATH="${nvidia_libs}:${LD_LIBRARY_PATH:-}"
  fi
}

repair_transformer_engine_cudart_path() {
  local site nvidia_dir
  shopt -s nullglob
  for site in "${COSMOS_VENV}"/lib/python*/site-packages; do
    nvidia_dir="${site}/nvidia"
    [[ -d "${nvidia_dir}/cuda_runtime" ]] || continue
    [[ -e "${nvidia_dir}/cudart" ]] || ln -s cuda_runtime "${nvidia_dir}/cudart"
    [[ -e "${nvidia_dir}/cuda_cudart" ]] || ln -s cuda_runtime "${nvidia_dir}/cuda_cudart"
  done
  shopt -u nullglob
}

reject_invalid_root() {
  local path="$1"
  local label="$2"
  if [[ -e "${path}/INVALID_DO_NOT_USE_20260609.md" ]]; then
    echo "refusing_invalid_${label}=${path}" >&2
    echo "invalid_marker=${path}/INVALID_DO_NOT_USE_20260609.md" >&2
    exit 24
  fi
}

run_in_alloc() {
  "$@"
}

hydra_list_from_csv() {
  local csv="$1"
  local out="["
  local sep=""
  local item trimmed
  IFS=',' read -ra items <<< "${csv}"
  for item in "${items[@]}"; do
    trimmed="$(printf '%s' "${item}" | xargs)"
    [[ -z "${trimmed}" ]] && continue
    out+="${sep}'${trimmed}'"
    sep=","
  done
  out+="]"
  printf '%s\n' "${out}"
}

write_manifest() {
  mkdir -p "${OUTPUT_ROOT}" "${CONDITION_ROOT}"
  {
    echo "timestamp=$(date -Is)"
    echo "job_id=${JOB_ID}"
    echo "source_dataset_root=${SOURCE_DATASET_ROOT}"
    echo "condition_root=${CONDITION_ROOT}"
    echo "output_root=${OUTPUT_ROOT}"
    echo "fix3_user_approval_file=${FIX3_USER_APPROVAL_FILE}"
    echo "require_fix3_user_approval=${REQUIRE_FIX3_USER_APPROVAL}"
    echo "allow_old_full1000_invalid_source_diagnostic=${ALLOW_OLD_FULL1000_INVALID_SOURCE_DIAGNOSTIC}"
    echo "base_checkpoint_path=${BASE_CHECKPOINT_PATH}"
    echo "wan_vae_path=${WAN_VAE_PATH}"
    echo "cosmos_venv=${COSMOS_VENV}"
    echo "local_tokenizer_dir=${LOCAL_TOKENIZER_DIR}"
    echo "sft_toml=${SFT_TOML}"
    echo "job_name=${JOB_NAME}"
    echo "wam_sft_mode=${WAM_SFT_MODE}"
    echo "total_video_frames=${TOTAL_VIDEO_FRAMES}"
    echo "total_action_steps=${TOTAL_ACTION_STEPS}"
    echo "expected_source_episodes=${EXPECTED_SOURCE_EPISODES}"
    echo "max_records=${MAX_RECORDS}"
    echo "action_conditioned_sft=${ACTION_CONDITIONED_SFT}"
    echo "require_state_targets=${REQUIRE_STATE_TARGETS}"
    echo "strict_full_preflight=${STRICT_FULL_PREFLIGHT}"
    echo "condition_prefix_policy=multi_mode_full_episode_masks"
    echo "conditioned_vision=causal_prefix_mask_over_full_301_frames_mapped_to_latent_indexes"
    echo "temporal_compression_factor=${TEMPORAL_COMPRESSION_FACTOR}"
    echo "nproc_per_node=${NPROC_PER_NODE}"
    echo "data_parallel_shard_degree=${DATA_PARALLEL_SHARD_DEGREE}"
    echo "data_parallel_replicate_degree=${DATA_PARALLEL_REPLICATE_DEGREE}"
    echo "context_parallel_shard_degree=${CONTEXT_PARALLEL_SHARD_DEGREE}"
    echo "optimizer_keys_to_select=${OPTIMIZER_KEYS_TO_SELECT}"
    echo "optimizer_lr=${OPTIMIZER_LR}"
    echo "scheduler_cycle_length=${SCHEDULER_CYCLE_LENGTH}"
    echo "scheduler_warmup_steps=${SCHEDULER_WARMUP_STEPS}"
    echo "scheduler_f_min=${SCHEDULER_F_MIN}"
    echo "grad_clip_norm=${GRAD_CLIP_NORM}"
    echo "action_loss_weight=${ACTION_LOSS_WEIGHT}"
    echo "normalize_loss_by_active=${NORMALIZE_LOSS_BY_ACTIVE}"
    echo "independent_action_schedule=${INDEPENDENT_ACTION_SCHEDULE}"
    echo "shift_action=${SHIFT_ACTION}"
    echo "enforce_fix1_action_recipe=${ENFORCE_FIX1_ACTION_RECIPE}"
    echo "allow_non_fix1_action_recipe_diagnostic=${ALLOW_NON_FIX1_ACTION_RECIPE_DIAGNOSTIC}"
    echo "conditioned_actions=history_actions_before_prefix_only"
    echo "sidecar_target_mode=${SIDECAR_TARGET_MODE}"
    echo "prefix_role_source=${PREFIX_ROLE_SOURCE}"
    echo "dense_receding_prefix_stride=${DENSE_RECEDING_PREFIX_STRIDE}"
    echo "role_weight_config=${ROLE_WEIGHT_CONFIG}"
    echo "late_rebind_weight=${LATE_REBIND_WEIGHT}"
    echo "late_rebind_roles=${LATE_REBIND_ROLES}"
    echo "late_rebind_min_abs_x=${LATE_REBIND_MIN_ABS_X}"
    echo "late_rebind_min_abs_y=${LATE_REBIND_MIN_ABS_Y}"
    echo "late_rebind_min_abs_z=${LATE_REBIND_MIN_ABS_Z}"
    echo "min_late_rebind_candidates=${MIN_LATE_REBIND_CANDIDATES}"
    echo "run_live_query_coverage_audit=${RUN_LIVE_QUERY_COVERAGE_AUDIT}"
    echo "live_query_coverage_summaries=${LIVE_QUERY_COVERAGE_SUMMARIES}"
    echo "live_query_coverage_prefix_tolerance=${LIVE_QUERY_COVERAGE_PREFIX_TOLERANCE}"
    echo "live_query_coverage_rel_l2_tolerance=${LIVE_QUERY_COVERAGE_REL_L2_TOLERANCE}"
    echo "live_query_coverage_rel_yz_tolerance=${LIVE_QUERY_COVERAGE_REL_YZ_TOLERANCE}"
    echo "live_query_coverage_max_undercovered_count=${LIVE_QUERY_COVERAGE_MAX_UNDERCOVERED_COUNT}"
    echo "live_query_coverage_max_undercovered_fraction=${LIVE_QUERY_COVERAGE_MAX_UNDERCOVERED_FRACTION}"
    echo "clean_dense_preflight_diagnostic_not_ready_reason=${CLEAN_DENSE_PREFLIGHT_DIAGNOSTIC_NOT_READY_REASON}"
    echo "visual_input=RGB only; depth is not used"
    echo "data_contract=301 RGB/state frames and 300 action/state rows per row; no 128-frame sampling; no 129-frame sliced t2w window; no min(pred,ref) metrics"
    echo "physical_reason=learn when the target hole starts moving, predict its path/final task frame, and jointly predict executable robot/peg/contact continuation under full-episode supervision"
    echo "evidence_boundary=SFT loss alone is not method evidence; post-SFT generated validation videos/action/readout metrics are required before controller integration"
  } | tee "${OUTPUT_ROOT}/sft_manifest.txt"
}

preflight_static_paths() {
  reject_invalid_root "${SOURCE_DATASET_ROOT}" "source_dataset_root"
  reject_invalid_root "${CONDITION_ROOT}" "condition_root"
  reject_invalid_root "${OUTPUT_ROOT}" "output_root"
  [[ -d "${SOURCE_DATASET_ROOT}" ]] || { echo "missing source dataset root: ${SOURCE_DATASET_ROOT}" >&2; exit 2; }
  [[ -d "${BASE_CHECKPOINT_PATH}/model" ]] || { echo "missing DCP model directory: ${BASE_CHECKPOINT_PATH}/model" >&2; exit 3; }
  [[ -s "${WAN_VAE_PATH}" ]] || { echo "missing Wan VAE: ${WAN_VAE_PATH}" >&2; exit 4; }
  [[ -x "${COSMOS_VENV}/bin/python" ]] || { echo "missing Cosmos Python: ${COSMOS_VENV}/bin/python" >&2; exit 5; }
  [[ -s "${LOCAL_TOKENIZER_DIR}/tokenizer.json" ]] || { echo "missing local tokenizer: ${LOCAL_TOKENIZER_DIR}/tokenizer.json" >&2; exit 6; }
  [[ -s "${SFT_TOML}" ]] || { echo "missing SFT TOML: ${SFT_TOML}" >&2; exit 7; }
  if [[ "${SOURCE_DATASET_ROOT}" == *"sft_dataset_full1000_rgbd" ]]; then
    echo "refusing old dirty preview-linked Cosmos dataset: ${SOURCE_DATASET_ROOT}" >&2
    exit 21
  fi
  if [[ "${ALLOW_OLD_FULL1000_INVALID_SOURCE_DIAGNOSTIC}" != "true" && "${SOURCE_DATASET_ROOT}" == *"sft_dataset_full1000_maniskill_default_regen_20260606_0055"* ]]; then
    cat >&2 <<EOF
refusing_invalid_old_full1000_source=true
source_dataset_root=${SOURCE_DATASET_ROOT}
reason=The 2026-06-10 fix3 audit found this source has many failed final insertions and narrow target motion. It is historical/diagnostic only, not active SFT data.
override_for_non_method_diagnostic=ALLOW_OLD_FULL1000_INVALID_SOURCE_DIAGNOSTIC=true
EOF
    exit 26
  fi
}

validate_action_training_recipe() {
  if [[ "${ACTION_CONDITIONED_SFT}" != "true" ]]; then
    echo "fix1_action_recipe_check=skipped_non_action_conditioned" | tee -a "${OUTPUT_ROOT}/sft_manifest.txt"
    return
  fi
  if [[ "${ENFORCE_FIX1_ACTION_RECIPE}" != "true" || "${ALLOW_NON_FIX1_ACTION_RECIPE_DIAGNOSTIC}" == "true" ]]; then
    echo "fix1_action_recipe_check=diagnostic_override" | tee -a "${OUTPUT_ROOT}/sft_manifest.txt"
    return
  fi

  local bad=()
  [[ "${OPTIMIZER_KEYS_TO_SELECT}" == *"action2llm"* ]] || bad+=("missing_action2llm")
  [[ "${OPTIMIZER_KEYS_TO_SELECT}" == *"llm2action"* ]] || bad+=("missing_llm2action")
  [[ "${OPTIMIZER_KEYS_TO_SELECT}" == *"action_modality_embed"* ]] || bad+=("missing_action_modality_embed")
  case "${OPTIMIZER_LR}" in 1.0e-4|1e-4|0.0001) ;; *) bad+=("optimizer_lr=${OPTIMIZER_LR}") ;; esac
  case "${SCHEDULER_WARMUP_STEPS}" in 10) ;; *) bad+=("scheduler_warmup_steps=${SCHEDULER_WARMUP_STEPS}") ;; esac
  case "${SCHEDULER_F_MIN}" in 0.5|0.50|0.500) ;; *) bad+=("scheduler_f_min=${SCHEDULER_F_MIN}") ;; esac
  case "${GRAD_CLIP_NORM}" in 1|1.0|1.00) ;; *) bad+=("grad_clip_norm=${GRAD_CLIP_NORM}") ;; esac
  case "${ACTION_LOSS_WEIGHT}" in 2|2.0|2.00) ;; *) bad+=("action_loss_weight=${ACTION_LOSS_WEIGHT}") ;; esac
  [[ "${NORMALIZE_LOSS_BY_ACTIVE}" == "true" ]] || bad+=("normalize_loss_by_active=${NORMALIZE_LOSS_BY_ACTIVE}")
  [[ "${INDEPENDENT_ACTION_SCHEDULE}" == "true" ]] || bad+=("independent_action_schedule=${INDEPENDENT_ACTION_SCHEDULE}")
  case "${SHIFT_ACTION}" in 1|1.0) ;; *) bad+=("shift_action=${SHIFT_ACTION}") ;; esac

  if (( ${#bad[@]} )); then
    {
      echo "refusing_non_fix1_action_recipe=true"
      printf 'bad_fields=%s\n' "${bad[*]}"
      echo "reason=Full-episode Cosmos3 WAM SFT must default to the overfit-validated fix1 action recipe unless an explicit diagnostic override is set."
      echo "override_for_non_method_diagnostic=ALLOW_NON_FIX1_ACTION_RECIPE_DIAGNOSTIC=true"
    } >&2
    exit 65
  fi
  echo "fix1_action_recipe_check=passed" | tee -a "${OUTPUT_ROOT}/sft_manifest.txt"
}

fix3_approval_required() {
  case "${REQUIRE_FIX3_USER_APPROVAL}" in
    true)
      return 0
      ;;
    false)
      return 1
      ;;
    auto)
      if [[ "${SOURCE_DATASET_ROOT}" == *fix3* || "${CONDITION_ROOT}" == *fix3* || "${OUTPUT_ROOT}" == *fix3* ]]; then
        return 0
      fi
      return 1
      ;;
    *)
      echo "invalid REQUIRE_FIX3_USER_APPROVAL=${REQUIRE_FIX3_USER_APPROVAL}; expected true, false, or auto" >&2
      exit 35
      ;;
  esac
}

preflight_fix3_user_approval_for_sft() {
  if ! fix3_approval_required; then
    echo "fix3_user_approval_required=false" | tee -a "${OUTPUT_ROOT}/sft_manifest.txt"
    return
  fi
  if [[ ! -s "${FIX3_USER_APPROVAL_FILE}" ]]; then
    cat >&2 <<EOF
refusing_fix3_sft_before_user_approval=true
fix3_user_approval_file=${FIX3_USER_APPROVAL_FILE}
reason=After fix3 data construction, render exactly 10 videos per fix3 scenario and stop for user approval before SFT.
expected_file_content=approved_for_sft=true
EOF
    exit 36
  fi
  if ! grep -Eq '^(approved_for_sft|user_approved)=true([[:space:]]|$)' "${FIX3_USER_APPROVAL_FILE}"; then
    cat >&2 <<EOF
refusing_fix3_sft_before_user_approval=true
fix3_user_approval_file=${FIX3_USER_APPROVAL_FILE}
reason=Approval file exists but does not contain approved_for_sft=true or user_approved=true.
EOF
    exit 37
  fi
  echo "fix3_user_approval_required=true" | tee -a "${OUTPUT_ROOT}/sft_manifest.txt"
  echo "fix3_user_approval_file=${FIX3_USER_APPROVAL_FILE}" | tee -a "${OUTPUT_ROOT}/sft_manifest.txt"
}

export_conditions() {
  if [[ "${FORCE_EXPORT}" != "true" && -s "${CONDITION_ROOT}/manifest.json" && -s "${TRAIN_JSONL}" && -s "${VAL_JSONL}" ]]; then
    echo "condition_export_reuse_existing=${CONDITION_ROOT}" | tee -a "${OUTPUT_ROOT}/sft_manifest.txt"
    return
  fi
  rm -rf "${CONDITION_ROOT}"
  mkdir -p "${CONDITION_ROOT}"
  run_in_alloc bash -lc "
    set -euo pipefail
    cd '${ROOT}'
    export HDF5_USE_FILE_LOCKING=FALSE
    export PYTHONPATH='${ROOT}/scripts/world_model':\"\${PYTHONPATH:-}\"
    '${ROOT}/.venv/bin/python' '${ROOT}/scripts/world_model/export_cosmos3_maniskill_full_episode_wam_conditions.py' \
      --dataset-root '${SOURCE_DATASET_ROOT}' \
      --output-root '${CONDITION_ROOT}' \
      --total-video-frames '${TOTAL_VIDEO_FRAMES}' \
      --total-action-steps '${TOTAL_ACTION_STEPS}' \
      --require-video-frames '${TOTAL_VIDEO_FRAMES}' \
      --max-records '${MAX_RECORDS}' \
      --prefix-policy multi_mode \
      --action-condition-mode joint_policy_history_action \
      --sidecar-target-mode '${SIDECAR_TARGET_MODE}' \
      --prefix-role-source '${PREFIX_ROLE_SOURCE}' \
      --dense-receding-prefix-stride '${DENSE_RECEDING_PREFIX_STRIDE}' \
      --temporal-compression-factor '${TEMPORAL_COMPRESSION_FACTOR}' \
      --progress-every 10
  " 2>&1 | tee "${OUTPUT_ROOT}/condition_export.log"
}

apply_role_weights() {
  if [[ -z "${ROLE_WEIGHT_CONFIG}" && "${LATE_REBIND_WEIGHT}" == "1" ]]; then
    echo "role_weighting=disabled" | tee -a "${OUTPUT_ROOT}/sft_manifest.txt"
    return
  fi
  local weighted_train_jsonl="${CONDITION_ROOT}/train/video_action_dataset_file_role_weighted.jsonl"
  (
    set -euo pipefail
    cd "${ROOT}"
    "${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/build_cosmos3_role_weighted_sft_jsonl.py" \
      --input-jsonl "${CONDITION_ROOT}/train/video_action_dataset_file.jsonl" \
      --output-jsonl "${weighted_train_jsonl}" \
      --role-weights "${ROLE_WEIGHT_CONFIG}" \
      --late-rebind-weight "${LATE_REBIND_WEIGHT}" \
      --late-rebind-roles "${LATE_REBIND_ROLES}" \
      --late-rebind-min-abs-x "${LATE_REBIND_MIN_ABS_X}" \
      --late-rebind-min-abs-y "${LATE_REBIND_MIN_ABS_Y}" \
      --late-rebind-min-abs-z "${LATE_REBIND_MIN_ABS_Z}" \
      --manifest-out "${CONDITION_ROOT}/train/role_weighted_manifest.json"
  ) 2>&1 | tee "${OUTPUT_ROOT}/role_weighting.log"
  TRAIN_JSONL="${weighted_train_jsonl}"
  echo "role_weighting=enabled" | tee -a "${OUTPUT_ROOT}/sft_manifest.txt"
  echo "role_weighted_train_jsonl=${TRAIN_JSONL}" | tee -a "${OUTPUT_ROOT}/sft_manifest.txt"
}

audit_conditions() {
  run_in_alloc bash -lc "
    set -euo pipefail
    cd '${ROOT}'
    export HDF5_USE_FILE_LOCKING=FALSE
    '${ROOT}/.venv/bin/python' '${ROOT}/scripts/world_model/preflight_cosmos3_full_episode_wam_contract.py' \
      --condition-root '${CONDITION_ROOT}' \
      --expected-source-episodes '${EXPECTED_SOURCE_EPISODES}' \
      --expected-video-frames '${TOTAL_VIDEO_FRAMES}' \
      --expected-action-steps '${TOTAL_ACTION_STEPS}' \
      --expected-action-dim 32 \
      --temporal-compression-factor '${TEMPORAL_COMPRESSION_FACTOR}' \
      --output-json '${OUTPUT_ROOT}/full_episode_wam_preflight.json' \
      --output-md '${OUTPUT_ROOT}/full_episode_wam_preflight.md'
    '${ROOT}/.venv/bin/python' '${ROOT}/scripts/world_model/audit_cosmos3_sft_preflight.py' \
      --train-jsonl '${TRAIN_JSONL}' \
      --val-jsonl '${VAL_JSONL}' \
      --num-video-frames '${TOTAL_VIDEO_FRAMES}' \
      --action-conditioned-sft '${ACTION_CONDITIONED_SFT}' \
      --wam-sft-mode '${WAM_SFT_MODE}' \
      --require-state-targets '${REQUIRE_STATE_TARGETS}' \
      --strict-full-preflight '${STRICT_FULL_PREFLIGHT}'
    '${ROOT}/.venv/bin/python' '${ROOT}/scripts/world_model/audit_cosmos3_action_targets.py' \
      --condition-root '${CONDITION_ROOT}' \
      --output-dir '${OUTPUT_ROOT}/action_target_audit' \
      --expected-action-steps '${TOTAL_ACTION_STEPS}' \
      --expected-action-dim 32 \
      --strict
    '${ROOT}/.venv/bin/python' '${ROOT}/scripts/world_model/audit_cosmos3_receding_training_distribution.py' \
      --condition-root '${CONDITION_ROOT}' \
      --output-json '${OUTPUT_ROOT}/receding_training_distribution_audit.json' \
      --prefix-role-source '${PREFIX_ROLE_SOURCE}' \
      --require-no-condition-mask-errors \
      --min-late-rebind-candidates '${MIN_LATE_REBIND_CANDIDATES}'
    live_query_coverage_flag=()
    if [[ '${RUN_LIVE_QUERY_COVERAGE_AUDIT}' == 'true' ]]; then
      live_query_coverage_args=()
      IFS=',' read -r -a live_query_coverage_summaries <<< '${LIVE_QUERY_COVERAGE_SUMMARIES}'
      for live_summary in \"\${live_query_coverage_summaries[@]}\"; do
        [[ -n \"\${live_summary}\" ]] && live_query_coverage_args+=(--live-summary \"\${live_summary}\")
      done
      if [[ \"\${#live_query_coverage_args[@]}\" -eq 0 ]]; then
        echo 'missing_live_query_coverage_summaries=true' >&2
        exit 47
      fi
      '${ROOT}/.venv/bin/python' '${ROOT}/scripts/world_model/audit_cosmos3_live_query_training_coverage.py' \
        --condition-root '${CONDITION_ROOT}' \
        \"\${live_query_coverage_args[@]}\" \
        --prefix-tolerance '${LIVE_QUERY_COVERAGE_PREFIX_TOLERANCE}' \
        --rel-l2-tolerance '${LIVE_QUERY_COVERAGE_REL_L2_TOLERANCE}' \
        --rel-yz-tolerance '${LIVE_QUERY_COVERAGE_REL_YZ_TOLERANCE}' \
        --output-json '${OUTPUT_ROOT}/live_query_training_coverage_audit.json' \
        --output-md '${OUTPUT_ROOT}/live_query_training_coverage_audit.md'
      live_query_coverage_flag+=(--live-query-coverage-audit-json '${OUTPUT_ROOT}/live_query_training_coverage_audit.json')
      live_query_coverage_flag+=(--max-live-query-undercovered-count '${LIVE_QUERY_COVERAGE_MAX_UNDERCOVERED_COUNT}')
      live_query_coverage_flag+=(--max-live-query-undercovered-fraction '${LIVE_QUERY_COVERAGE_MAX_UNDERCOVERED_FRACTION}')
    fi
    if [[ '${PREFIX_ROLE_SOURCE}' == 'physical_mode' || '${DENSE_RECEDING_PREFIX_STRIDE}' != '0' ]]; then
      weighted_flag=()
      if [[ '${LATE_REBIND_WEIGHT}' != '1' || -n '${ROLE_WEIGHT_CONFIG}' ]]; then
        weighted_flag+=(--require-weighted-train-jsonl)
      fi
      diagnostic_not_ready_flag=()
      if [[ -n '${CLEAN_DENSE_PREFLIGHT_DIAGNOSTIC_NOT_READY_REASON}' ]]; then
        diagnostic_not_ready_flag+=(--diagnostic-not-ready-reason '${CLEAN_DENSE_PREFLIGHT_DIAGNOSTIC_NOT_READY_REASON}')
      fi
      '${ROOT}/.venv/bin/python' '${ROOT}/scripts/world_model/summarize_cosmos3_clean_dense_preflight.py' \
        --condition-root '${CONDITION_ROOT}' \
        --output-root '${OUTPUT_ROOT}' \
        --expected-source-episodes '${EXPECTED_SOURCE_EPISODES}' \
        --expected-video-frames '${TOTAL_VIDEO_FRAMES}' \
        --expected-action-steps '${TOTAL_ACTION_STEPS}' \
        --expected-prefix-role-source '${PREFIX_ROLE_SOURCE}' \
        --min-late-rebind-candidates '${MIN_LATE_REBIND_CANDIDATES}' \
        \"\${weighted_flag[@]}\" \
        \"\${live_query_coverage_flag[@]}\" \
        \"\${diagnostic_not_ready_flag[@]}\"
    fi
  " 2>&1 | tee "${OUTPUT_ROOT}/condition_audit.log"
}

run_sft() {
  if [[ "${RUN_SFT}" != "true" ]]; then
    echo "run_sft=false" | tee -a "${OUTPUT_ROOT}/sft_manifest.txt"
    return
  fi
  preflight_fix3_user_approval_for_sft

  FFMPEG_BIN="${FFMPEG_BIN:-$(find_cosmos_ffmpeg_bin || true)}"
  COSMOS_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  OPTIMIZER_KEYS_LIST="$(hydra_list_from_csv "${OPTIMIZER_KEYS_TO_SELECT}")"
  (
    set -euo pipefail
    cd "${ROOT}/external/cosmos-framework"
    export PATH="${COSMOS_VENV}/bin:${PATH}"
    export PYTHONPATH="${ROOT}/external/cosmos-framework"
    export LD_LIBRARY_PATH="${COSMOS_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
    export COSMOS3_LOCAL_TOKENIZER_DIR="${LOCAL_TOKENIZER_DIR}"
    export DATASET_PATH="${CONDITION_ROOT}"
    export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH}"
    export WAN_VAE_PATH="${WAN_VAE_PATH}"
    export IMAGINAIRE_OUTPUT_ROOT="${OUTPUT_ROOT}/outputs"
    export OUTPUT_ROOT="${OUTPUT_ROOT}/outputs"
    export FFMPEG_BIN="${FFMPEG_BIN}"
    export HDF5_USE_FILE_LOCKING=FALSE
    export AWS_EC2_METADATA_DISABLED=true
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
      -m cosmos_framework.scripts.train \
      --sft-toml="${SFT_TOML}" \
      -- \
      "job.name=${JOB_NAME}" \
      "trainer.max_iter=${MAX_ITER}" \
      "trainer.run_validation=${RUN_VALIDATION}" \
      trainer.run_validation_on_start=true \
      "trainer.validation_iter=${VALIDATION_ITER}" \
      "trainer.max_val_iter=${MAX_VAL_ITER}" \
      "checkpoint.save_iter=${SAVE_ITER}" \
      "optimizer.lr=${OPTIMIZER_LR}" \
      "optimizer.keys_to_select=${OPTIMIZER_KEYS_LIST}" \
      "scheduler.cycle_lengths=[${SCHEDULER_CYCLE_LENGTH}]" \
      "scheduler.f_min=[${SCHEDULER_F_MIN}]" \
      "scheduler.warm_up_steps=[${SCHEDULER_WARMUP_STEPS}]" \
      "trainer.callbacks.grad_clip.clip_norm=${GRAD_CLIP_NORM}" \
      checkpoint.load_from_object_store.enabled=false \
      checkpoint.save_to_object_store.enabled=false \
      checkpoint.enable_gcs_patch_in_boto3=false \
      "model.config.resolution='256'" \
      model.config.vlm_config.pretrained_weights.enabled=false \
      model.config.vlm_config.pretrained_weights.enable_gcs_patch_in_boto3=false \
      model.config.ema.enabled=false \
      model.config.compile.enabled=false \
      "model.config.parallelism.data_parallel_shard_degree=${DATA_PARALLEL_SHARD_DEGREE}" \
      "model.config.parallelism.data_parallel_replicate_degree=${DATA_PARALLEL_REPLICATE_DEGREE}" \
      "model.config.parallelism.context_parallel_shard_degree=${CONTEXT_PARALLEL_SHARD_DEGREE}" \
      "model.config.max_num_tokens_after_packing=${MAX_SEQUENCE_LENGTH}" \
      "model.config.rectified_flow_training_config.action_loss_weight=${ACTION_LOSS_WEIGHT}" \
      "model.config.rectified_flow_training_config.normalize_loss_by_active=${NORMALIZE_LOSS_BY_ACTIVE}" \
      "model.config.rectified_flow_training_config.independent_action_schedule=${INDEPENDENT_ACTION_SCHEDULE}" \
      "model.config.rectified_flow_training_config.shift_action=${SHIFT_ACTION}" \
      "dataloader_train.max_sequence_length=${MAX_SEQUENCE_LENGTH}" \
      "dataloader_train.dataloader.datasets.video.dataset.jsonl_paths=['${TRAIN_JSONL}']" \
      "dataloader_val.dataloader.datasets.video.dataset.jsonl_paths=['${VAL_JSONL}']" \
      "dataloader_train.dataloader.datasets.video.dataset.num_video_frames=${TOTAL_VIDEO_FRAMES}" \
      "dataloader_val.dataloader.datasets.video.dataset.num_video_frames=${TOTAL_VIDEO_FRAMES}" \
      "dataloader_train.dataloader.datasets.video.dataset.temporal_interval_mode=${TEMPORAL_INTERVAL_MODE}" \
      "dataloader_val.dataloader.datasets.video.dataset.temporal_interval_mode=${TEMPORAL_INTERVAL_MODE}" \
      "dataloader_train.dataloader.datasets.video.dataset.conditioning_config={${CONDITIONING_LATENT_FRAMES}:1.0}" \
      "dataloader_val.dataloader.datasets.video.dataset.conditioning_config={${CONDITIONING_LATENT_FRAMES}:1.0}" \
      dataloader_train.dataloader.datasets.video.dataset.max_action_dim=64 \
      dataloader_val.dataloader.datasets.video.dataset.max_action_dim=64
  ) 2>&1 | tee "${OUTPUT_ROOT}/sft_train.log"

  "${ROOT}/.venv/bin/python" - "${OUTPUT_ROOT}/sft_train.log" "${OUTPUT_ROOT}/val_loss_summary.json" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
text = log_path.read_text(errors="replace") if log_path.exists() else ""
rows = []
for match in re.finditer(r"Validation loss \(iteration (\d+)\):\s*([0-9.eE+-]+)", text):
    rows.append({"iteration": int(match.group(1)), "val_loss": float(match.group(2))})
summary = {
    "log_path": str(log_path),
    "num_validation_points": len(rows),
    "validation": rows,
    "latest_val_loss": rows[-1]["val_loss"] if rows else None,
    "evidence_boundary": "Validation loss is a training diagnostic only. Generated same-length validation videos and action/readout metrics are still required.",
}
if rows and not math.isfinite(rows[-1]["val_loss"]):
    raise SystemExit("latest val loss is not finite")
out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
if not rows:
    raise SystemExit("No validation loss lines found in SFT log.")
print(json.dumps(summary, sort_keys=True))
PY
}

main() {
  cd "${ROOT}"
  repair_transformer_engine_cudart_path
  refresh_cosmos_ld_library_path
  write_manifest
  preflight_static_paths
  validate_action_training_recipe
  trap 'code=$?; if [[ $code -ne 0 && ! -s "${OUTPUT_ROOT}/sft_completed" ]]; then { echo "timestamp=$(date -Is)"; echo "exit_code=${code}"; echo "reason=full_episode_wam_wrapper_exited_before_completion"; } > "${OUTPUT_ROOT}/sft_failed"; fi' EXIT
  rm -f "${OUTPUT_ROOT}/sft_failed"
  export_conditions
  apply_role_weights
  audit_conditions
  run_sft
  date -Is > "${OUTPUT_ROOT}/sft_completed"
}

main "$@"
