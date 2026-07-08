#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

export SOURCE_KEY="${SOURCE_KEY:-hole_late_move_stop_seed17280909_idx8226}"
export RUN_NAME="${RUN_NAME:-try17}"
export COSMOS_ACTION_ROW_OFFSET="${COSMOS_ACTION_ROW_OFFSET:-9}"
export COSMOS_ACTION_ROW_OFFSET_SOURCE="${COSMOS_ACTION_ROW_OFFSET_SOURCE:-action_diag_try13_best_teacher_temporal_offset_minus9}"
export COSMOS_ACTION_DIRECTION_GUARD="${COSMOS_ACTION_DIRECTION_GUARD:-none}"
export REQUIRE_TARGET_MOTION_COMPLETE_BEFORE_FINISHER="${REQUIRE_TARGET_MOTION_COMPLETE_BEFORE_FINISHER:-true}"
export FINISHER_CONTROLLER="${FINISHER_CONTROLLER:-diffusion_policy}"
export METHOD_EVIDENCE_ALLOWED="${METHOD_EVIDENCE_ALLOWED:-false}"

if [[ "${SOURCE_KEY}" != hole_late_move_stop_* ]]; then
  cat >&2 <<'EOF'
refusing_non_move_stop_row_offset=true
reason=This launcher is only for the documented Phase 03 move-stop row-offset diagnostic. Do not reuse a future-label offset on other case groups without a fresh action_diag result.
EOF
  exit 46
fi

if [[ "${COSMOS_ACTION_ROW_OFFSET}" == "0" ]]; then
  cat >&2 <<'EOF'
refusing_missing_row_offset=true
reason=This launcher is specifically for the Phase 03 move-stop action-row-offset diagnostic. Use COSMOS_ACTION_ROW_OFFSET=9 unless a newer documented diagnostic replaces it.
EOF
  exit 47
fi

bash scripts/slurm/phase03_h5_source.sh
