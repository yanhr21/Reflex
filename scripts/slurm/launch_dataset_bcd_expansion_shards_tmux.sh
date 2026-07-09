#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
EXECUTE=false
STAGE_FILTER="${STAGE_FILTER:-all}"
FAMILY_FILTER="${FAMILY_FILTER:-all}"
MAX_LAUNCHES="${MAX_LAUNCHES:-1}"
PROD_RUN_NAME="${PROD_RUN_NAME:-prod02}"
EXPANSION_EXCLUDE_NODES="${EXPANSION_EXCLUDE_NODES:-server02,server05,server07,server10,server18,server23,server27,server28,server30,server34,server35,server36,server39,server43,server44,server46,server51,server52,server53,server56,server57,server58,server59,server60,server63}"

usage() {
  cat <<EOF
usage: launch_dataset_bcd_expansion_shards_tmux.sh [--execute] [--stage B|C|D|all] [--family lr|fb|reverse|stop|sine|cont|all] [--max-launches N] [--prod-run-name NAME]

Dry-runs the B/C/D expansion shard launch plan by default. Execution launches
new shards under <class>/<prod-run-name>/<family> and refuses existing output
directories. Use --execute only after the approved B/C/D smoke gate is open.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --execute)
      EXECUTE=true
      shift
      ;;
    --stage)
      STAGE_FILTER="${2:-}"
      shift 2
      ;;
    --family)
      FAMILY_FILTER="${2:-}"
      shift 2
      ;;
    --max-launches)
      MAX_LAUNCHES="${2:-}"
      shift 2
      ;;
    --prod-run-name)
      PROD_RUN_NAME="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown_arg=$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${STAGE_FILTER}" in B|C|D|all) ;; *) echo "invalid_stage=${STAGE_FILTER}" >&2; exit 3 ;; esac
case "${FAMILY_FILTER}" in lr|fb|reverse|stop|sine|cont|all) ;; *) echo "invalid_family=${FAMILY_FILTER}" >&2; exit 4 ;; esac
if ! [[ "${MAX_LAUNCHES}" =~ ^[0-9]+$ ]] || [[ "${MAX_LAUNCHES}" -lt 1 ]]; then
  echo "invalid_max_launches=${MAX_LAUNCHES}" >&2
  exit 5
fi
if [[ "${PROD_RUN_NAME}" =~ /|\\.\\.|^$|[[:space:]] ]]; then
  echo "invalid_prod_run_name=${PROD_RUN_NAME}" >&2
  exit 6
fi

echo "launch_bcd_expansion_shards=true"
echo "execute=${EXECUTE}"
echo "stage_filter=${STAGE_FILTER}"
echo "family_filter=${FAMILY_FILTER}"
echo "max_launches=${MAX_LAUNCHES}"
echo "prod_run_name=${PROD_RUN_NAME}"
echo "expansion_exclude_nodes=${EXPANSION_EXCLUDE_NODES}"
echo "requires_explicit_user_approval=true"
echo "target_total_b=1000"
echo "target_total_c=500"
echo "target_total_d=500"

if [[ "${EXECUTE}" != "true" ]]; then
  PROD_RUN_NAME="${PROD_RUN_NAME}" "${ROOT}/scripts/world_model/dataset_bcd_expansion_shard_plan.sh"
  echo "dry_run_only=true"
  echo "no_slurm_submitted=true"
  exit 0
fi

for stage in b_dynamic_smoke c_frozen_dp_smoke d_future_teacher_smoke; do
  RUN_GROUP= RUN_NAME= "${ROOT}/scripts/world_model/require_dataset_class_smoke_approved.sh" "${stage}"
done

launched=0
skipped=0
existing_skipped=0

launch_one() {
  local stage="$1"
  local family="$2"
  local launcher="$3"
  local session="$4"
  local run_group="$5"
  local run_name="$6"
  local scenario="$7"
  local count="$8"
  local dx="$9"
  local dy="${10}"
  local dz="${11}"

  if [[ "${STAGE_FILTER}" != "all" && "${STAGE_FILTER}" != "${stage}" ]]; then
    skipped=$((skipped + 1))
    return
  fi
  if [[ "${FAMILY_FILTER}" != "all" && "${FAMILY_FILTER}" != "${family}" ]]; then
    skipped=$((skipped + 1))
    return
  fi
  if [[ "${launched}" -ge "${MAX_LAUNCHES}" ]]; then
    skipped=$((skipped + 1))
    return
  fi

  out_dir="${ROOT}/experiments/maniskill/runs/01_dataset/${run_group}/${run_name}"
  if [[ -e "${out_dir}" ]]; then
    echo "[skip_existing_${stage}_${family}]"
    echo "output_dir=${out_dir}"
    echo "reason=existing_output_dir_not_relaunched"
    existing_skipped=$((existing_skipped + 1))
    return
  fi

  echo "[launch_${stage}_${family}]"
  SESSION="${session}" RUN_GROUP="${run_group}" RUN_NAME="${run_name}" \
    SCENARIO="${scenario}" COUNT="${count}" \
    DELTA_X="${dx}" DELTA_Y="${dy}" DELTA_Z="${dz}" \
    MAX_STEP_DELTA_M=0.005 FPS=30 EXCLUDE_NODES="${EXPANSION_EXCLUDE_NODES}" \
    "${ROOT}/${launcher}"
  launched=$((launched + 1))
}

launch_one B lr scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh dset_b_${PROD_RUN_NAME}_lr dynamic_rgb/${PROD_RUN_NAME} lr constant_lr 170 0.0 0.08 0.0
launch_one B fb scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh dset_b_${PROD_RUN_NAME}_fb dynamic_rgb/${PROD_RUN_NAME} fb constant_fb 170 0.08 0.0 0.0
launch_one B reverse scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh dset_b_${PROD_RUN_NAME}_reverse dynamic_rgb/${PROD_RUN_NAME} reverse reverse 165 0.0 0.08 0.0
launch_one B stop scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh dset_b_${PROD_RUN_NAME}_stop dynamic_rgb/${PROD_RUN_NAME} stop move_stop 165 0.0 0.08 0.0
launch_one B sine scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh dset_b_${PROD_RUN_NAME}_sine dynamic_rgb/${PROD_RUN_NAME} sine sine 165 0.0 0.08 0.0
launch_one B cont scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh dset_b_${PROD_RUN_NAME}_cont dynamic_rgb/${PROD_RUN_NAME} cont continuous 165 0.0 0.10 0.0

launch_one C lr scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh dset_c_${PROD_RUN_NAME}_lr frozen_dp_dynamic/${PROD_RUN_NAME} lr constant_lr 84 0.0 0.08 0.0
launch_one C fb scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh dset_c_${PROD_RUN_NAME}_fb frozen_dp_dynamic/${PROD_RUN_NAME} fb constant_fb 84 0.08 0.0 0.0
launch_one C reverse scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh dset_c_${PROD_RUN_NAME}_reverse frozen_dp_dynamic/${PROD_RUN_NAME} reverse reverse 83 0.0 0.08 0.0
launch_one C stop scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh dset_c_${PROD_RUN_NAME}_stop frozen_dp_dynamic/${PROD_RUN_NAME} stop move_stop 83 0.0 0.08 0.0
launch_one C sine scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh dset_c_${PROD_RUN_NAME}_sine frozen_dp_dynamic/${PROD_RUN_NAME} sine sine 83 0.0 0.08 0.0
launch_one C cont scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh dset_c_${PROD_RUN_NAME}_cont frozen_dp_dynamic/${PROD_RUN_NAME} cont continuous 83 0.0 0.10 0.0

launch_one D lr scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh dset_d_${PROD_RUN_NAME}_lr future_teacher/${PROD_RUN_NAME} lr constant_lr 84 0.0 0.08 0.0
launch_one D fb scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh dset_d_${PROD_RUN_NAME}_fb future_teacher/${PROD_RUN_NAME} fb constant_fb 84 0.08 0.0 0.0
launch_one D reverse scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh dset_d_${PROD_RUN_NAME}_reverse future_teacher/${PROD_RUN_NAME} reverse reverse 83 0.0 0.08 0.0
launch_one D stop scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh dset_d_${PROD_RUN_NAME}_stop future_teacher/${PROD_RUN_NAME} stop move_stop 83 0.0 0.08 0.0
launch_one D sine scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh dset_d_${PROD_RUN_NAME}_sine future_teacher/${PROD_RUN_NAME} sine sine 83 0.0 0.08 0.0
launch_one D cont scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh dset_d_${PROD_RUN_NAME}_cont future_teacher/${PROD_RUN_NAME} cont continuous 83 0.0 0.10 0.0

echo "launched_count=${launched}"
echo "skipped_count=${skipped}"
echo "existing_skipped_count=${existing_skipped}"
if [[ "${launched}" -eq 0 ]]; then
  echo "no_matching_shard_launched=true"
  exit 7
fi
