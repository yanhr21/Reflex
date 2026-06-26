#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this selector-blocker audit inside a compute-node srun step from a tmux-held allocation.
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTCOME_JSONL="${OUTCOME_JSONL:-}"
if [[ -z "${OUTCOME_JSONL}" ]]; then
  mapfile -t OUTCOME_FILES < <(
    while IFS= read -r done_file; do
      out_root="$(awk -F= '$1=="output_root" {print $2}' "${done_file}" | tail -n 1)"
      if [[ -n "${out_root}" && -s "${out_root}/candidate_outcome_labels.jsonl" ]]; then
        printf '%s\n' "${out_root}/candidate_outcome_labels.jsonl"
      fi
    done < <(
      find experiments/world_model_task_rebinding/cosmos3/hardphase_h96_retrieval_shard_claims_20260621_1242_h96shard64_alloc145276 \
        -path '*/done.txt' -type f | sort
    )
  )
else
  IFS=':' read -r -a OUTCOME_FILES <<< "${OUTCOME_JSONL}"
fi

if [[ "${#OUTCOME_FILES[@]}" -lt 1 ]]; then
  echo "no_outcome_jsonl_found=true" >&2
  exit 31
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/world_model_task_rebinding/cosmos3/h96_selector_blocker_audit_${STAMP}_alloc${SLURM_JOB_ID}}"
mkdir -p "${OUTPUT_ROOT}"

OUTCOME_ARGS=()
for item in "${OUTCOME_FILES[@]}"; do
  OUTCOME_ARGS+=(--outcome-jsonl "${item}")
done

SCORER_ARGS=()
for item in \
  experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_handoff_success_retry_gatefix_145813_20260621_2018_formal3h/training_summary.json \
  experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_handoff_success_retry_rank02_gatefix_145814_20260621_2032_formal3h/training_summary.json
do
  if [[ -s "${item}" ]]; then
    SCORER_ARGS+=(--scorer-summary "${item}")
  fi
done
for item in \
  experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_handoff_success_retry_gatefix_145813_20260621_2018_formal3h/training_history.json \
  experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_handoff_success_retry_rank02_gatefix_145814_20260621_2032_formal3h/training_history.json
do
  if [[ -s "${item}" ]]; then
    SCORER_ARGS+=(--scorer-history "${item}")
  fi
done

".venv/bin/python" scripts/world_model/audit_cosmos3_h96_selector_blockers.py \
  "${OUTCOME_ARGS[@]}" \
  "${SCORER_ARGS[@]}" \
  --output-root "${OUTPUT_ROOT}" \
  --val-fraction "${VAL_FRACTION:-0.25}" \
  --seed "${SEED:-20260618}"

cat > "${OUTPUT_ROOT}/audit_done.txt" <<EOF
done=true
date=$(date --iso-8601=seconds)
output_root=${OUTPUT_ROOT}
boundary=Offline selector-blocker audit only. Not live method evidence.
EOF
