#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/public/home/yanhongru/ICLR2027/archive/Reflex}"
cd "${ROOT}"

KEY="${SOURCE_KEY:-hole_late_reverse_seed1040038_idx0004}"
H5="${ROOT}/experiments/maniskill/data/fix3_733/canonical_h5/${KEY}.fix3/${KEY}.h5"

echo "phase03_forward_backward_candidate_status_ok=true"
echo "source_key=${KEY}"
echo "source_h5_path=${H5}"
echo "source_h5_exists=$([[ -f "${H5}" ]] && echo true || echo false)"
echo "run_group=h5_reverse"
echo "planned_run_name=try21"

echo "--- existing_artifacts_for_key ---"
rg -l "\"source_key\"[[:space:]]*:[[:space:]]*\"${KEY}\"" \
  experiments/maniskill/runs/03_oracle \
  "${ARCHIVE_ROOT}/experiments/maniskill/runs/03_oracle" \
  2>/dev/null | sort || true

echo "--- closest_prior_forward_backward_failures ---"
for f in \
  "${ARCHIVE_ROOT}"/experiments/maniskill/runs/03_oracle/h5_reverse/try*/summary.json \
  "${ARCHIVE_ROOT}"/experiments/maniskill/runs/03_oracle/h5_move_stop/try*/summary.json
do
  [[ -f "${f}" ]] || continue
  rel="${f#${ARCHIVE_ROOT}/experiments/maniskill/runs/03_oracle/}"
  jq -r --arg rel "${rel}" '
    select((.dynamic_controller.future_label_teacher_dynamic_diagnostic // false) != true)
    | select((.finisher.future_label_teacher_suffix_diagnostic // false) != true)
    | select((.finisher_controller // "") != "source_h5_teacher_suffix")
    | select((.classification // "") | startswith("diagnostic_") | not)
    | [
        $rel,
        (.source_key // ""),
        (.classification // ""),
        (.simulator_success_metric // false),
        (.cosmos_dynamic_action_count // 0),
        (.finisher_controller // ""),
        (.final_eval.peg_head_l2 // .final_peg_head_l2 // 999),
        (.best_finisher_eval.peg_head_l2 // .best_peg_head_l2 // 999)
      ] | @tsv
  ' "${f}"
done | sort -t "$(printf '\t')" -k8,8n -k7,7n | sed -n '1,12p'
