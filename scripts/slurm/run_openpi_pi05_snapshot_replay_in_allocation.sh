#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
OPENPI_ROOT="${OPENPI_ROOT:-/public/home/yanhongru/ICLR2027/openpi}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run OpenPI pi0.5 snapshot inference/replay only inside a compute-node srun step.
example=srun --jobid=<held_job_id> --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=120G bash scripts/slurm/run_openpi_pi05_snapshot_replay_in_allocation.sh
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
CONFIG_NAME="${CONFIG_NAME:-pi05_maniskill_peg733}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/experiments/world_model_task_rebinding/openpi/checkpoints_local_preserved/pi05_maniskill_peg733/pi05_peg733_1gpu1h_20260625_after_nfs_enolck_localckpt_1600_skipnorm_alloc150773/1599}"
PANEL_ROOT="${PANEL_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_seed20260725_h8192_safegate1_source_suffix_offsets64_48_32_24_panel0134_20260623_alloc146658}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/openpi/openpi_pi05_snapshot_replay_${STAMP}_alloc${SLURM_JOB_ID}}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
MAX_ITER_DIRS="${MAX_ITER_DIRS:-1}"
ITERATION_INDICES="${ITERATION_INDICES:-}"
EXECUTE_STEPS="${EXECUTE_STEPS:-8}"
PREPARE_STATE_MODE="${PREPARE_STATE_MODE:-qpos8}"
DP_ROLLOUT_CONTINUABILITY_HORIZON="${DP_ROLLOUT_CONTINUABILITY_HORIZON:-96}"
UV_SHARED_CACHE_DIR="${UV_SHARED_CACHE_DIR:-${ROOT}/experiments/world_model_task_rebinding/openpi/uv_cache}"
UV_RUN_PYTHON="${UV_RUN_PYTHON:-}"
UV_PYTHON_PLATFORM="${UV_PYTHON_PLATFORM:-}"

mkdir -p "${OUTPUT_ROOT}" "${UV_SHARED_CACHE_DIR}"

export OPENPI_ROOT
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${ROOT}/experiments/world_model_task_rebinding/openpi/lerobot_home}"
unset LEROBOT_HOME
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${ROOT}/experiments/world_model_task_rebinding/openpi/openpi_data_home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${UV_SHARED_CACHE_DIR}}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/openpi_uv_env_${USER}_${SLURM_JOB_ID}_${SLURM_STEP_ID}}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export GIT_HTTP_VERSION="${GIT_HTTP_VERSION:-HTTP/1.1}"
export GIT_TERMINAL_PROMPT=0

cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=openpi_pi05_snapshot_replay_wrapper_v2_split_env
date=$(date --iso-8601=seconds)
root=${ROOT}
openpi_root=${OPENPI_ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
config_name=${CONFIG_NAME}
checkpoint_dir=${CHECKPOINT_DIR}
panel_root=${PANEL_ROOT}
output_root=${OUTPUT_ROOT}
max_samples=${MAX_SAMPLES}
max_iter_dirs=${MAX_ITER_DIRS}
iteration_indices=${ITERATION_INDICES}
execute_steps=${EXECUTE_STEPS}
prepare_state_mode=${PREPARE_STATE_MODE}
dp_rollout_continuability_horizon=${DP_ROLLOUT_CONTINUABILITY_HORIZON}
project_python=${ROOT}/.venv/bin/python
uv_cache_dir=${UV_CACHE_DIR}
uv_project_environment=${UV_PROJECT_ENVIRONMENT}
uv_link_mode=${UV_LINK_MODE}
uv_python_platform=${UV_PYTHON_PLATFORM:-native}
uv_run_python=${UV_RUN_PYTHON:-uv_default}
resource_boundary=tmux-held interactive Slurm allocation; no login-node inference/replay/render.
method_boundary=Official OpenPI pi0.5 checkpoint inference plus existing ManiSkill saved-snapshot replay. No custom VAE/MLP/diffusion/intermediate model.
env_boundary=prepare/replay use project .venv with SAPIEN/ManiSkill; inference uses official OpenPI uv environment with JAX/Orbax.
EOF

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/openpi/prepare_openpi_pi05_snapshot_observations.py" \
  --panel-root "${PANEL_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --max-samples "${MAX_SAMPLES}" \
  --max-iter-dirs "${MAX_ITER_DIRS}" \
  --iteration-indices "${ITERATION_INDICES}" \
  --state-mode "${PREPARE_STATE_MODE}" \
  2>&1 | tee "${OUTPUT_ROOT}/prepare_observations.log"

cd "${OPENPI_ROOT}"
UV_RUN_ARGS=(uv run --frozen)
if [[ -n "${UV_RUN_PYTHON}" ]]; then
  UV_RUN_ARGS+=(--python "${UV_RUN_PYTHON}")
fi
if [[ -n "${UV_PYTHON_PLATFORM}" ]]; then
  UV_RUN_ARGS+=(--python-platform "${UV_PYTHON_PLATFORM}")
fi

"${UV_RUN_ARGS[@]}" "${ROOT}/scripts/openpi/infer_openpi_pi05_from_prepared_observations.py" \
  --prepared-manifest "${OUTPUT_ROOT}/prepared_observations_manifest.json" \
  --config-name "${CONFIG_NAME}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --execute-steps "${EXECUTE_STEPS}" \
  2>&1 | tee "${OUTPUT_ROOT}/openpi_pi05_inference.log"

cd "${ROOT}"
rm -f "${OUTPUT_ROOT}/replay_labels.list"
while IFS= read -r action_json; do
  [[ -n "${action_json}" ]] || continue
  base="$(basename "${action_json}" .action_chunk.json)"
  replay_dir="${OUTPUT_ROOT}/replay/${base}"
  mkdir -p "${replay_dir}"
  "${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/replay_policy_droid_action_chunk_from_snapshot.py" \
    --action-chunk-json "${action_json}" \
    --snapshot-state-h5 "$("${ROOT}/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["snapshot_state_h5"])' "${action_json}")" \
    --history-action-state-json "$("${ROOT}/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["history_action_state_json"])' "${action_json}")" \
    --source-h5 "$("${ROOT}/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_h5"])' "${action_json}")" \
    --output-root "${replay_dir}" \
    --execute-steps "${EXECUTE_STEPS}" \
    --dp-rollout-continuability-horizon "${DP_ROLLOUT_CONTINUABILITY_HORIZON}" \
    2>&1 | tee "${replay_dir}/replay.log"
  echo "${replay_dir}/policy_droid_snapshot_action_replay_label.json" >> "${OUTPUT_ROOT}/replay_labels.list"
done < <(find "${OUTPUT_ROOT}/action_chunks" -name '*.action_chunk.json' -type f | sort)

"${ROOT}/.venv/bin/python" - <<'PY' "${OUTPUT_ROOT}"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
labels = []
list_path = root / "replay_labels.list"
if list_path.exists():
    for line in list_path.read_text().splitlines():
        path = pathlib.Path(line)
        if path.exists():
            labels.append(json.loads(path.read_text()))
summary = {
    "schema": "openpi_pi05_split_env_snapshot_replay_summary_v1",
    "output_root": str(root),
    "label_count": len(labels),
    "after_success_count": sum(bool(x.get("after_success")) for x in labels),
    "after_inserted_live_pose_count": sum(bool(x.get("after_inserted_live_pose")) for x in labels),
    "after_contact_stable_proxy_count": sum(bool(x.get("after_contact_stable_proxy")) for x in labels),
    "after_grasped_count": sum(bool(x.get("after_grasped")) for x in labels),
    "dp96_success_count": sum(bool((x.get("dp_rollout_continuability") or {}).get("success")) for x in labels),
    "dp96_continuable_count": sum(bool((x.get("dp_rollout_continuability") or {}).get("continuable")) for x in labels),
    "boundary": "Saved-snapshot OpenPI pi0.5 replay summary; video/contact review is still required before success claims.",
}
(root / "openpi_pi05_snapshot_replay_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY
