#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
COMMIT="${COMMIT:-852976723d813352cabd5690f0acaab910f86c4e^}"

files=(
  scripts/world_model/generate_cosmos3_fix3_hard_dynamic_teacher.py
  scripts/world_model/generate_cosmos3_fix3_late_trigger_dynamic_experts.py
  scripts/world_model/generate_cosmos3_fix3_successful_dynamic_dataset.py
  scripts/world_model/render_cosmos3_maniskill_sft_dataset.py
)

cd "${ROOT}"

echo "dataset_source_recovery_status_ok=true"
echo "source_commit=${COMMIT}"
echo "active_route_ready=false"
echo "reference_only=true"

for file in "${files[@]}"; do
  echo "[${file}]"
  if git cat-file -e "${COMMIT}:${file}" 2>/dev/null; then
    echo "  recoverable_from_git=true"
    echo "  line_count=$(git show "${COMMIT}:${file}" | wc -l | tr -d ' ')"
    echo "  active_worktree_exists=$([[ -f "${file}" ]] && echo true || echo false)"
    if git grep -q 'world_model_task_rebinding' "${COMMIT}" -- "${file}"; then
      echo "  old_world_model_task_rebinding_path=true"
    else
      echo "  old_world_model_task_rebinding_path=false"
    fi
    if git grep -q -E 'set_pose|set_state|set_state_dict' "${COMMIT}" -- "${file}"; then
      echo "  state_edit_or_state_restore_reference=true"
    else
      echo "  state_edit_or_state_restore_reference=false"
    fi
  else
    echo "  recoverable_from_git=false"
  fi
done

echo "must_not_restore_as_active_runner_without_review=true"
echo "audit_doc=${ROOT}/docs/legacy_dynamic_source_recovery.md"
