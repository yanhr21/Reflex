#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}
JOB_ID=${JOB_ID:-${SLURM_JOB_ID:-}}
if [[ -z "${JOB_ID}" ]]; then
  echo "JOB_ID or SLURM_JOB_ID must be set for an existing held allocation" >&2
  exit 2
fi

srun --overlap --jobid="${JOB_ID}" --gres=gpu:4 --ntasks=1 --cpus-per-task=24 bash -lc "
set -euo pipefail
cd '${ROOT}'
export PYTHONPATH='${ROOT}/deps/ManiSkill_clean:${ROOT}':\${PYTHONPATH:-}
export HDF5_USE_FILE_LOCKING=FALSE
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export DISPLAY=

BASE_ROOT='${ROOT}/experiments/world_model_task_rebinding/cosmos3'
GEN='.venv/bin/python scripts/world_model/generate_cosmos3_fix3_original_protocol_large_motion_dp.py'

start_one() {
  local gpu_id=\"\$1\"
  local name=\"\$2\"
  local scenario_sequence=\"\$3\"
  local scenario_quotas=\"\$4\"
  local scenario_seed_bases=\"\$5\"
  local num_demos=\"\$6\"
  local accepted_offset=\"\$7\"
  local output_root=\"\${BASE_ROOT}/fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_\${name}\"
  local log_path=\"\${output_root}.generate.log\"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=\"\${gpu_id}\"
    echo timestamp=\$(date -Is)
    echo job_id=\${SLURM_JOB_ID:-unknown}
    echo node_list=\${SLURM_JOB_NODELIST:-unknown}
    echo cuda_visible_devices=\${CUDA_VISIBLE_DEVICES}
    echo output_root=\"\${output_root}\"
    echo physical_reason=v7_complete9_nonpeg_focus_pinned_generation
    \${GEN} \
      --output-root \"\${output_root}\" \
      --num-demos \"\${num_demos}\" \
      --max-attempts 60000 \
      --seed 18000000 \
      --val-fraction 0.1 \
      --scenario-seed-bases \"\${scenario_seed_bases}\" \
      --accepted-index-offset \"\${accepted_offset}\" \
      --non-priority-seed-offset 0 \
      --policy-rng-seed-base -1 \
      --save-reject-log-limit 3000 \
      --reject-log-every 100 \
      --scenario-sequence \"\${scenario_sequence}\" \
      --scenario-quotas \"\${scenario_quotas}\" \
      --no-use-priority-seeds
  ) 2>&1 | tee \"\${log_path}\" &
}

echo timestamp=\$(date -Is)
echo job_id=\${SLURM_JOB_ID:-unknown}
echo node_list=\${SLURM_JOB_NODELIST:-unknown}
echo outer_cuda_visible_devices=\${CUDA_VISIBLE_DEVICES:-unset}
echo bundle=fix3_v7_nonpeg_focus3_pinned
echo reason=pin scarce non-peg classes to otherwise unused GPUs inside held allocation

start_one 1 nonpeg_focus3_constant_aux90_seedbase18250000 hole_late_constant hole_late_constant=90 hole_late_constant=18250000 90 9000
start_one 2 nonpeg_focus3_sine_aux90_seedbase18251000 hole_late_sine hole_late_sine=90 hole_late_sine=18251000 90 9100
start_one 3 nonpeg_focus3_continuous_aux120_seedbase18241000 hole_late_continuous_insert hole_late_continuous_insert=120 hole_late_continuous_insert=18241000 120 9200

wait
echo focus3_pinned_finished=\$(date -Is)
"
