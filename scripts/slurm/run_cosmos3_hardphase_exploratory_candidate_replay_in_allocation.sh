#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run hard-phase exploratory candidate replay only inside a compute-node srun step from a tmux-held allocation.
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
CONTACT_EXECUTOR_JSONL="${CONTACT_EXECUTOR_JSONL:-experiments/world_model_task_rebinding/cosmos3/contact_executor_dataset_rollout24_predpath_train512_20260617_server23_alloc135069/contact_executor_dataset_file.jsonl}"
CHECKPOINT="${CHECKPOINT:-experiments/world_model_task_rebinding/cosmos3/outcome_oracle_candidate_executor_hardphase_balanced_smoke100_20260618_alloc139764/checkpoint_best_offline.pt}"
HARD_PHASES="${HARD_PHASES:-far,lateral_align,preinsert_aligned}"
HARD_MAX_ROWS="${HARD_MAX_ROWS:-64}"
HARD_SKIP_ROWS="${HARD_SKIP_ROWS:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_explore64_samples48_${STAMP}_alloc${SLURM_JOB_ID}}"
HEADROOM_ROOT="${HEADROOM_ROOT:-experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_hardphase_explore64_samples48_${STAMP}_alloc${SLURM_JOB_ID}}"
FILTER_ROOT="${FILTER_ROOT:-experiments/world_model_task_rebinding/cosmos3/contact_executor_dataset_hardphase_explore64_${STAMP}_alloc${SLURM_JOB_ID}}"

EXEC_HORIZON="${EXEC_HORIZON:-24}"
DP_ROLLOUT_CONTINUABILITY_HORIZON="${DP_ROLLOUT_CONTINUABILITY_HORIZON:-8}"
DP_ROLLOUT_CONTINUABILITY_MIN_STABLE_STEPS="${DP_ROLLOUT_CONTINUABILITY_MIN_STABLE_STEPS:-4}"
CONTACT_STABLE_MIN_REL_X="${CONTACT_STABLE_MIN_REL_X:--0.06}"
CONTACT_STABLE_MAX_REL_X="${CONTACT_STABLE_MAX_REL_X:-0.03}"
CONTACT_STABLE_MAX_ABS_Y="${CONTACT_STABLE_MAX_ABS_Y:-0.018}"
CONTACT_STABLE_MAX_ABS_Z="${CONTACT_STABLE_MAX_ABS_Z:-0.012}"
MODEL_CANDIDATE_SAMPLES="${MODEL_CANDIDATE_SAMPLES:-48}"
MODEL_CANDIDATE_TEMPS="${MODEL_CANDIDATE_TEMPS:-0.25,0.5,0.75,1.0,1.5,2.0}"
MODEL_CANDIDATE_SCALES="${MODEL_CANDIDATE_SCALES:-0.2,0.5,1.0,1.5}"
LEGACY_CANDIDATE_SCALES="${LEGACY_CANDIDATE_SCALES:-0.05,0.1,0.2,0.5,1.0}"
CANDIDATE_SHORT_PREFIX_STEPS="${CANDIDATE_SHORT_PREFIX_STEPS:-}"
INCLUDE_LEGACY_TEACHER_SCALE_CANDIDATES="${INCLUDE_LEGACY_TEACHER_SCALE_CANDIDATES:-true}"
INCLUDE_RETRIEVAL_RESIDUAL_CANDIDATES="${INCLUDE_RETRIEVAL_RESIDUAL_CANDIDATES:-false}"
RETRIEVAL_SOURCE_JSONL="${RETRIEVAL_SOURCE_JSONL:-${CONTACT_EXECUTOR_JSONL}}"
RETRIEVAL_K="${RETRIEVAL_K:-4}"
RETRIEVAL_POSITIVE_FIELDS="${RETRIEVAL_POSITIVE_FIELDS:-future_inserted_within_chunk,future_dp_continuable_within_chunk}"
RETRIEVAL_RESIDUAL_SCALES="${RETRIEVAL_RESIDUAL_SCALES:-1.0}"
VAL_FRACTION="${VAL_FRACTION:-0.25}"
SEED="${SEED:-20260618}"
SHARD_CLAIM_ROOT="${SHARD_CLAIM_ROOT:-}"
ALLOW_DUPLICATE_SHARD="${ALLOW_DUPLICATE_SHARD:-false}"
ALLOW_RERUN_FAILED_SHARD="${ALLOW_RERUN_FAILED_SHARD:-true}"
ALLOW_RERUN_ORPHANED_SHARD="${ALLOW_RERUN_ORPHANED_SHARD:-true}"

sanitize_claim_key() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '_'
}

claim_hash() {
  if command -v sha1sum >/dev/null 2>&1; then
    printf '%s' "$1" | sha1sum | awk '{print $1}'
  else
    printf '%s' "$1" | cksum | awk '{print $1 "_" $2}'
  fi
}

claim_value() {
  local path="$1"
  local key="$2"
  awk -F= -v target="${key}" '$1 == target {print substr($0, length($1) + 2); exit}' "${path}" 2>/dev/null || true
}

claim_job_is_active() {
  local owner_job="$1"
  if [[ -z "${owner_job}" ]]; then
    return 1
  fi
  local state
  state="$(squeue -h -j "${owner_job}" -o %T 2>/dev/null | head -n 1 || true)"
  case "${state}" in
    PENDING|CONFIGURING|RUNNING|COMPLETING|SUSPENDED)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -s "${path}" ]]; then
    echo "missing_${label}=${path}" >&2
    exit 2
  fi
}

require_file "${CONTACT_EXECUTOR_JSONL}" "contact_executor_jsonl"
require_file "${CHECKPOINT}" "checkpoint"

mkdir -p "${FILTER_ROOT}" "${OUTPUT_ROOT}" "${HEADROOM_ROOT}"
FILTERED_JSONL="${FILTER_ROOT}/contact_executor_dataset_file.jsonl"

CLAIM_DIR=""
if [[ -n "${SHARD_CLAIM_ROOT}" ]]; then
  CLAIM_KEY_RAW="contact=${CONTACT_EXECUTOR_JSONL}|checkpoint=${CHECKPOINT}|hard_phases=${HARD_PHASES}|hardrows=${HARD_MAX_ROWS}|skip=${HARD_SKIP_ROWS}|horizon=${EXEC_HORIZON}|dp_rollout_horizon=${DP_ROLLOUT_CONTINUABILITY_HORIZON}|dp_rollout_min_stable=${DP_ROLLOUT_CONTINUABILITY_MIN_STABLE_STEPS}|contact_stable=${CONTACT_STABLE_MIN_REL_X},${CONTACT_STABLE_MAX_REL_X},${CONTACT_STABLE_MAX_ABS_Y},${CONTACT_STABLE_MAX_ABS_Z}|legacy_scales=${LEGACY_CANDIDATE_SCALES}|short_prefix_steps=${CANDIDATE_SHORT_PREFIX_STEPS}|model_samples=${MODEL_CANDIDATE_SAMPLES}|model_temps=${MODEL_CANDIDATE_TEMPS}|model_scales=${MODEL_CANDIDATE_SCALES}|retrieval=${INCLUDE_RETRIEVAL_RESIDUAL_CANDIDATES}|retrieval_source=${RETRIEVAL_SOURCE_JSONL}|retrieval_k=${RETRIEVAL_K}|retrieval_fields=${RETRIEVAL_POSITIVE_FIELDS}|retrieval_scales=${RETRIEVAL_RESIDUAL_SCALES}"
  CLAIM_KEY_PREFIX="hardrows${HARD_MAX_ROWS}_skip${HARD_SKIP_ROWS}_retrieval${INCLUDE_RETRIEVAL_RESIDUAL_CANDIDATES}_k${RETRIEVAL_K}"
  CLAIM_KEY_HASH="$(claim_hash "${CLAIM_KEY_RAW}")"
  CLAIM_DIR="${SHARD_CLAIM_ROOT}/$(sanitize_claim_key "${CLAIM_KEY_PREFIX}")_${CLAIM_KEY_HASH}"
  mkdir -p "${SHARD_CLAIM_ROOT}"
  if [[ -d "${CLAIM_DIR}" && -s "${CLAIM_DIR}/failed.txt" && ! -s "${CLAIM_DIR}/done.txt" && "${ALLOW_RERUN_FAILED_SHARD}" == "true" ]]; then
    FAILED_ARCHIVE="${CLAIM_DIR}.failed_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}_step${SLURM_STEP_ID}_pid$$"
    mv "${CLAIM_DIR}" "${FAILED_ARCHIVE}"
  fi
  if [[ -d "${CLAIM_DIR}" && ! -s "${CLAIM_DIR}/done.txt" && ! -s "${CLAIM_DIR}/failed.txt" && -s "${CLAIM_DIR}/claim.txt" && "${ALLOW_RERUN_ORPHANED_SHARD}" == "true" ]]; then
    CLAIM_OWNER_JOB="$(claim_value "${CLAIM_DIR}/claim.txt" "slurm_job_id")"
    if ! claim_job_is_active "${CLAIM_OWNER_JOB}"; then
      ORPHAN_ARCHIVE="${CLAIM_DIR}.orphaned_$(date +%Y%m%d_%H%M%S)_oldjob${CLAIM_OWNER_JOB:-unknown}_newjob${SLURM_JOB_ID}_step${SLURM_STEP_ID}_pid$$"
      mv "${CLAIM_DIR}" "${ORPHAN_ARCHIVE}"
    fi
  fi
  if mkdir "${CLAIM_DIR}" 2>/dev/null; then
    cat > "${CLAIM_DIR}/claim.txt" <<EOF
schema=cosmos3_hardphase_candidate_replay_shard_claim_v1
date=$(date --iso-8601=seconds)
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
claim_key_hash=${CLAIM_KEY_HASH}
claim_key_raw=${CLAIM_KEY_RAW}
contact_executor_jsonl=${CONTACT_EXECUTOR_JSONL}
checkpoint=${CHECKPOINT}
hard_phases=${HARD_PHASES}
hard_max_rows=${HARD_MAX_ROWS}
hard_skip_rows=${HARD_SKIP_ROWS}
exec_horizon=${EXEC_HORIZON}
dp_rollout_continuability_horizon=${DP_ROLLOUT_CONTINUABILITY_HORIZON}
dp_rollout_continuability_min_stable_steps=${DP_ROLLOUT_CONTINUABILITY_MIN_STABLE_STEPS}
contact_stable_min_rel_x=${CONTACT_STABLE_MIN_REL_X}
contact_stable_max_rel_x=${CONTACT_STABLE_MAX_REL_X}
contact_stable_max_abs_y=${CONTACT_STABLE_MAX_ABS_Y}
contact_stable_max_abs_z=${CONTACT_STABLE_MAX_ABS_Z}
include_retrieval_residual_candidates=${INCLUDE_RETRIEVAL_RESIDUAL_CANDIDATES}
retrieval_source_jsonl=${RETRIEVAL_SOURCE_JSONL}
retrieval_k=${RETRIEVAL_K}
retrieval_positive_fields=${RETRIEVAL_POSITIVE_FIELDS}
retrieval_residual_scales=${RETRIEVAL_RESIDUAL_SCALES}
model_candidate_samples=${MODEL_CANDIDATE_SAMPLES}
dp_rollout_continuability_horizon=${DP_ROLLOUT_CONTINUABILITY_HORIZON}
model_candidate_temps=${MODEL_CANDIDATE_TEMPS}
model_candidate_scales=${MODEL_CANDIDATE_SCALES}
legacy_candidate_scales=${LEGACY_CANDIDATE_SCALES}
candidate_short_prefix_steps=${CANDIDATE_SHORT_PREFIX_STEPS}
output_root=${OUTPUT_ROOT}
headroom_root=${HEADROOM_ROOT}
allow_rerun_failed_shard=${ALLOW_RERUN_FAILED_SHARD}
allow_rerun_orphaned_shard=${ALLOW_RERUN_ORPHANED_SHARD}
EOF
    finish_claim() {
      local rc=$?
      if [[ -n "${CLAIM_DIR}" && -d "${CLAIM_DIR}" ]]; then
        if [[ "${rc}" -eq 0 ]]; then
          cat > "${CLAIM_DIR}/done.txt" <<EOF
done=true
date=$(date --iso-8601=seconds)
output_root=${OUTPUT_ROOT}
headroom_root=${HEADROOM_ROOT}
EOF
        else
          cat > "${CLAIM_DIR}/failed.txt" <<EOF
done=false
date=$(date --iso-8601=seconds)
rc=${rc}
output_root=${OUTPUT_ROOT}
headroom_root=${HEADROOM_ROOT}
EOF
        fi
      fi
    }
    trap finish_claim EXIT
  elif [[ "${ALLOW_DUPLICATE_SHARD}" != "true" ]]; then
    cat > "${HEADROOM_ROOT}/shard_claim_skipped.txt" <<EOF
skipped=true
reason=shard_already_claimed
date=$(date --iso-8601=seconds)
claim_dir=${CLAIM_DIR}
hard_max_rows=${HARD_MAX_ROWS}
hard_skip_rows=${HARD_SKIP_ROWS}
claim_key_hash=${CLAIM_KEY_HASH}
include_retrieval_residual_candidates=${INCLUDE_RETRIEVAL_RESIDUAL_CANDIDATES}
retrieval_k=${RETRIEVAL_K}
retrieval_residual_scales=${RETRIEVAL_RESIDUAL_SCALES}
model_candidate_samples=${MODEL_CANDIDATE_SAMPLES}
allow_rerun_failed_shard=${ALLOW_RERUN_FAILED_SHARD}
allow_rerun_orphaned_shard=${ALLOW_RERUN_ORPHANED_SHARD}
EOF
    exit 0
  fi
fi

".venv/bin/python" - "${CONTACT_EXECUTOR_JSONL}" "${FILTERED_JSONL}" "${HARD_PHASES}" "${HARD_MAX_ROWS}" "${HARD_SKIP_ROWS}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
hard_phases = {item.strip() for item in sys.argv[3].split(",") if item.strip()}
limit = int(sys.argv[4])
skip = int(sys.argv[5])
rows = []
matched = 0
for line in src.read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if str(row.get("current_phase") or "") not in hard_phases:
        continue
    matched += 1
    if matched <= skip:
        continue
    rows.append(row)
    if limit > 0 and len(rows) >= limit:
        break
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w") as f:
    for row in rows:
        f.write(json.dumps(row, sort_keys=True) + "\n")
summary = {
    "schema": "cosmos3_hardphase_contact_executor_filter_v1",
    "source_jsonl": str(src),
    "output_jsonl": str(dst),
    "hard_phases": sorted(hard_phases),
    "limit": limit,
    "skip": skip,
    "num_rows": len(rows),
    "phase_counts": dict(sorted(Counter(str(row.get("current_phase") or "unknown") for row in rows).items())),
}
(dst.parent / "filter_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
if not rows:
    raise SystemExit("no hard-phase rows selected")
PY

cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=cosmos3_hardphase_exploratory_candidate_replay_wrapper_v1
date=$(date --iso-8601=seconds)
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
contact_executor_jsonl=${CONTACT_EXECUTOR_JSONL}
filtered_jsonl=${FILTERED_JSONL}
checkpoint=${CHECKPOINT}
output_root=${OUTPUT_ROOT}
headroom_root=${HEADROOM_ROOT}
hard_phases=${HARD_PHASES}
hard_max_rows=${HARD_MAX_ROWS}
hard_skip_rows=${HARD_SKIP_ROWS}
exec_horizon=${EXEC_HORIZON}
dp_rollout_continuability_horizon=${DP_ROLLOUT_CONTINUABILITY_HORIZON}
dp_rollout_continuability_min_stable_steps=${DP_ROLLOUT_CONTINUABILITY_MIN_STABLE_STEPS}
contact_stable_min_rel_x=${CONTACT_STABLE_MIN_REL_X}
contact_stable_max_rel_x=${CONTACT_STABLE_MAX_REL_X}
contact_stable_max_abs_y=${CONTACT_STABLE_MAX_ABS_Y}
contact_stable_max_abs_z=${CONTACT_STABLE_MAX_ABS_Z}
model_candidate_samples=${MODEL_CANDIDATE_SAMPLES}
model_candidate_temps=${MODEL_CANDIDATE_TEMPS}
model_candidate_scales=${MODEL_CANDIDATE_SCALES}
include_retrieval_residual_candidates=${INCLUDE_RETRIEVAL_RESIDUAL_CANDIDATES}
include_legacy_teacher_scale_candidates=${INCLUDE_LEGACY_TEACHER_SCALE_CANDIDATES}
candidate_short_prefix_steps=${CANDIDATE_SHORT_PREFIX_STEPS}
retrieval_source_jsonl=${RETRIEVAL_SOURCE_JSONL}
retrieval_k=${RETRIEVAL_K}
retrieval_positive_fields=${RETRIEVAL_POSITIVE_FIELDS}
retrieval_residual_scales=${RETRIEVAL_RESIDUAL_SCALES}
shard_claim_root=${SHARD_CLAIM_ROOT}
shard_claim_dir=${CLAIM_DIR}
allow_rerun_failed_shard=${ALLOW_RERUN_FAILED_SHARD}
allow_rerun_orphaned_shard=${ALLOW_RERUN_ORPHANED_SHARD}
boundary=Exploratory hard-phase candidate distribution replay. This tests whether broader stochastic action chunks contain successful hard-state candidates; it is not live method evidence.
EOF

RETRIEVAL_ARGS=()
if [[ "${INCLUDE_RETRIEVAL_RESIDUAL_CANDIDATES}" == "true" ]]; then
  RETRIEVAL_ARGS=(
    --include-retrieval-residual-candidates
    --retrieval-source-jsonl "${RETRIEVAL_SOURCE_JSONL}"
    --retrieval-k "${RETRIEVAL_K}"
    --retrieval-positive-fields "${RETRIEVAL_POSITIVE_FIELDS}"
    --retrieval-residual-scales "${RETRIEVAL_RESIDUAL_SCALES}"
  )
fi

LEGACY_ARGS=(--include-legacy-teacher-scale-candidates)
case "${INCLUDE_LEGACY_TEACHER_SCALE_CANDIDATES}" in
  0|false|FALSE|no|NO|n|N) LEGACY_ARGS=(--no-include-legacy-teacher-scale-candidates) ;;
esac

EXPORT_CMD=(
  ".venv/bin/python" scripts/world_model/export_cosmos3_candidate_outcome_labels.py
  --contact-executor-jsonl "${FILTERED_JSONL}"
  --output-root "${OUTPUT_ROOT}"
  --max-samples "${HARD_MAX_ROWS}"
  --exec-horizon "${EXEC_HORIZON}"
  --dp-rollout-continuability-horizon "${DP_ROLLOUT_CONTINUABILITY_HORIZON}"
  --dp-rollout-continuability-min-stable-steps "${DP_ROLLOUT_CONTINUABILITY_MIN_STABLE_STEPS}"
  --contact-stable-min-rel-x "${CONTACT_STABLE_MIN_REL_X}"
  --contact-stable-max-rel-x "${CONTACT_STABLE_MAX_REL_X}"
  --contact-stable-max-abs-y "${CONTACT_STABLE_MAX_ABS_Y}"
  --contact-stable-max-abs-z "${CONTACT_STABLE_MAX_ABS_Z}"
  --candidate-executor-checkpoint "${CHECKPOINT}"
  --model-candidate-samples "${MODEL_CANDIDATE_SAMPLES}"
  --model-candidate-temps "${MODEL_CANDIDATE_TEMPS}"
  --model-candidate-scales "${MODEL_CANDIDATE_SCALES}"
  --candidate-scales "${LEGACY_CANDIDATE_SCALES}"
  --candidate-short-prefix-steps "${CANDIDATE_SHORT_PREFIX_STEPS}"
)
EXPORT_CMD+=("${LEGACY_ARGS[@]}")
EXPORT_CMD+=("${RETRIEVAL_ARGS[@]}")
"${EXPORT_CMD[@]}"

".venv/bin/python" scripts/world_model/summarize_cosmos3_candidate_outcome_headroom.py \
  --outcome-jsonl "${OUTPUT_ROOT}/candidate_outcome_labels.jsonl" \
  --output-root "${HEADROOM_ROOT}" \
  --val-fraction "${VAL_FRACTION}" \
  --seed "${SEED}"

cat > "${HEADROOM_ROOT}/wrapper_done.txt" <<EOF
done=true
date=$(date --iso-8601=seconds)
output_root=${OUTPUT_ROOT}
headroom_root=${HEADROOM_ROOT}
filtered_jsonl=${FILTERED_JSONL}
boundary=Exploratory replay complete. Use only to decide the next candidate-distribution repair.
EOF
