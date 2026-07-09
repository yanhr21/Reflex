#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
PROD_RUN_NAME="${PROD_RUN_NAME:-prod02}"

echo "dataset_bcd_expansion_shard_plan_ok=true"
echo "read_only=true"
echo "submits_slurm=false"
echo "root=${ROOT}"
echo "prod_run_name=${PROD_RUN_NAME}"
echo "production_root=experiments/maniskill/runs/01_dataset"
echo "log_root=logs/01_dataset"
echo "fps=30"
echo "gpus_per_shard=1"
echo "requires_bcd_human_review_approval=true"
echo "source_policy=repeat_approved_prod01_motion_families_with_new_output_root"

print_gate() {
  local stage="$1"
  local label="$2"
  echo "[gate_${label}]"
  if RUN_GROUP= RUN_NAME= "${ROOT}/scripts/world_model/require_dataset_class_smoke_approved.sh" "${stage}" >/tmp/bcd_expansion_gate_$$ 2>&1; then
    sed 's/^/  /' /tmp/bcd_expansion_gate_$$
  else
    sed 's/^/  /' /tmp/bcd_expansion_gate_$$
  fi
  rm -f /tmp/bcd_expansion_gate_$$
}

print_gate b_dynamic_smoke b
print_gate c_frozen_dp_smoke c
print_gate d_future_teacher_smoke d

echo "[motion_families]"
cat <<'EOF'
  family=lr scenario=constant_lr count_b=170 count_c=84 count_d=84 delta_x=0.0 delta_y=0.08 delta_z=0.0 max_step_delta_m=0.005
  family=fb scenario=constant_fb count_b=170 count_c=84 count_d=84 delta_x=0.08 delta_y=0.0 delta_z=0.0 max_step_delta_m=0.005
  family=reverse scenario=reverse count_b=165 count_c=83 count_d=83 delta_x=0.0 delta_y=0.08 delta_z=0.0 max_step_delta_m=0.005
  family=stop scenario=move_stop count_b=165 count_c=83 count_d=83 delta_x=0.0 delta_y=0.08 delta_z=0.0 max_step_delta_m=0.005
  family=sine scenario=sine count_b=165 count_c=83 count_d=83 delta_x=0.0 delta_y=0.08 delta_z=0.0 max_step_delta_m=0.005
  family=cont scenario=continuous count_b=165 count_c=83 count_d=83 delta_x=0.0 delta_y=0.10 delta_z=0.0 max_step_delta_m=0.005
EOF

echo "[targets]"
echo "  b_total=1000"
echo "  c_total=500"
echo "  d_total=500"
echo "  b_output_root=experiments/maniskill/runs/01_dataset/dynamic_rgb/${PROD_RUN_NAME}"
echo "  c_output_root=experiments/maniskill/runs/01_dataset/frozen_dp_dynamic/${PROD_RUN_NAME}"
echo "  d_output_root=experiments/maniskill/runs/01_dataset/future_teacher/${PROD_RUN_NAME}"

emit_command() {
  local stage="$1"
  local launcher="$2"
  local run_group="$3"
  local family="$4"
  local scenario="$5"
  local count="$6"
  local dx="$7"
  local dy="$8"
  local dz="$9"
  local session="${10}"
  printf '  %s_COUNT=%s ' "${stage}" "${count}"
  printf 'SESSION=%q RUN_GROUP=%q RUN_NAME=%q SCENARIO=%q COUNT=%q DELTA_X=%q DELTA_Y=%q DELTA_Z=%q MAX_STEP_DELTA_M=0.005 ' \
    "${session}" "${run_group}/${PROD_RUN_NAME}" "${family}" "${scenario}" "${count}" "${dx}" "${dy}" "${dz}"
  printf 'FPS=30 %q\n' "${ROOT}/${launcher}"
}

echo "[launch_commands_after_approval]"
emit_command B scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh dynamic_rgb lr constant_lr 170 0.0 0.08 0.0 dset_b_prod02_lr
emit_command B scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh dynamic_rgb fb constant_fb 170 0.08 0.0 0.0 dset_b_prod02_fb
emit_command B scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh dynamic_rgb reverse reverse 165 0.0 0.08 0.0 dset_b_prod02_reverse
emit_command B scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh dynamic_rgb stop move_stop 165 0.0 0.08 0.0 dset_b_prod02_stop
emit_command B scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh dynamic_rgb sine sine 165 0.0 0.08 0.0 dset_b_prod02_sine
emit_command B scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh dynamic_rgb cont continuous 165 0.0 0.10 0.0 dset_b_prod02_cont
emit_command C scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh frozen_dp_dynamic lr constant_lr 84 0.0 0.08 0.0 dset_c_prod02_lr
emit_command C scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh frozen_dp_dynamic fb constant_fb 84 0.08 0.0 0.0 dset_c_prod02_fb
emit_command C scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh frozen_dp_dynamic reverse reverse 83 0.0 0.08 0.0 dset_c_prod02_reverse
emit_command C scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh frozen_dp_dynamic stop move_stop 83 0.0 0.08 0.0 dset_c_prod02_stop
emit_command C scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh frozen_dp_dynamic sine sine 83 0.0 0.08 0.0 dset_c_prod02_sine
emit_command C scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh frozen_dp_dynamic cont continuous 83 0.0 0.10 0.0 dset_c_prod02_cont
emit_command D scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh future_teacher lr constant_lr 84 0.0 0.08 0.0 dset_d_prod02_lr
emit_command D scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh future_teacher fb constant_fb 84 0.08 0.0 0.0 dset_d_prod02_fb
emit_command D scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh future_teacher reverse reverse 83 0.0 0.08 0.0 dset_d_prod02_reverse
emit_command D scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh future_teacher stop move_stop 83 0.0 0.08 0.0 dset_d_prod02_stop
emit_command D scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh future_teacher sine sine 83 0.0 0.08 0.0 dset_d_prod02_sine
emit_command D scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh future_teacher cont continuous 83 0.0 0.10 0.0 dset_d_prod02_cont
