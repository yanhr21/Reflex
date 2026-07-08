#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
export DATASET_STAGE="d_future_teacher_production"
exec "${ROOT}/scripts/slurm/launch_dataset_stage_production_tmux_common.sh"
