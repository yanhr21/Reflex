#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
ACTION="${1:-${ACTION:-}}"

if [[ -z "${ACTION}" ]]; then
  cat >&2 <<'EOF'
missing_action=true
usage=bash scripts/world_model/check_current_cosmos3_next_action_gate.sh <action>
actions=resume_current_condition_sft,launch_broad_panel_current_checkpoint,clean_dense_preflight_after_user_approval,clean_dense_overfit_sft_after_user_approval
EOF
  exit 64
fi

STATUS_JSON="${STATUS_JSON:-${ROOT}/docs/world_model_task_rebinding/2026-06-13_cosmos3_sft_closed_loop_status_auto.json}"
STATUS_MD="${STATUS_MD:-${ROOT}/docs/world_model_task_rebinding/2026-06-13_cosmos3_sft_closed_loop_status_auto.md}"
REQUIREMENT_AUDIT_JSON="${REQUIREMENT_AUDIT_JSON:-${ROOT}/docs/world_model_task_rebinding/2026-06-14_iter2700_closed_loop_requirement_audit.json}"
CLEAN_DENSE_PREFLIGHT_SUMMARY="${CLEAN_DENSE_PREFLIGHT_SUMMARY:-}"
SAFE_ACTION_NAME="$(printf '%s' "${ACTION}" | tr -c 'A-Za-z0-9_-' '_')"
OUTPUT_JSON="${OUTPUT_JSON:-${ROOT}/docs/world_model_task_rebinding/2026-06-13_cosmos3_next_action_gate_${SAFE_ACTION_NAME}.json}"
USER_APPROVED="${USER_APPROVED:-false}"
ALLOW_NONPASSING_EXIT_ZERO="${ALLOW_NONPASSING_EXIT_ZERO:-false}"

cd "${ROOT}"

OUTPUT_JSON="${STATUS_JSON}" OUTPUT_MD="${STATUS_MD}" \
  bash "${ROOT}/scripts/world_model/monitor_current_cosmos3_sft_closed_loop_status.sh" >/dev/null

gate_args=(
  python3 "${ROOT}/scripts/world_model/check_cosmos3_next_action_gate.py"
  --status-json "${STATUS_JSON}"
  --action "${ACTION}"
  --output-json "${OUTPUT_JSON}"
)

if [[ -s "${REQUIREMENT_AUDIT_JSON}" ]]; then
  gate_args+=(--requirement-audit-json "${REQUIREMENT_AUDIT_JSON}")
fi
if [[ -n "${CLEAN_DENSE_PREFLIGHT_SUMMARY}" && -s "${CLEAN_DENSE_PREFLIGHT_SUMMARY}" ]]; then
  gate_args+=(--clean-dense-preflight-summary-json "${CLEAN_DENSE_PREFLIGHT_SUMMARY}")
fi
if [[ "${USER_APPROVED}" == "true" ]]; then
  gate_args+=(--user-approved)
fi
if [[ "${ALLOW_NONPASSING_EXIT_ZERO}" == "true" ]]; then
  gate_args+=(--allow-nonpassing-exit-zero)
fi

"${gate_args[@]}"
