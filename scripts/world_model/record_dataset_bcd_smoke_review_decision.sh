#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

decision=""
reviewer="${REVIEWER:-}"
notes="${NOTES:-}"
dry_run=false

usage() {
  cat <<EOF
usage: record_dataset_bcd_smoke_review_decision.sh --decision approved|rejected --reviewer NAME --notes TEXT [--dry-run]

Records the same human visual-review decision for B/C/D dataset smoke runs.
Agents must not run this with --decision approved unless the user explicitly
approves the B/C/D smoke visual quality in the current conversation.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --decision)
      decision="${2:-}"
      shift 2
      ;;
    --reviewer)
      reviewer="${2:-}"
      shift 2
      ;;
    --notes)
      notes="${2:-}"
      shift 2
      ;;
    --dry-run)
      dry_run=true
      shift
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

if [[ "${decision}" != "approved" && "${decision}" != "rejected" ]]; then
  echo "bcd_review_decision_recorded=false"
  echo "reason=decision_required"
  usage >&2
  exit 20
fi

if [[ -z "${reviewer}" ]]; then
  echo "bcd_review_decision_recorded=false"
  echo "reason=reviewer_required"
  exit 21
fi

if [[ -z "${notes}" ]]; then
  echo "bcd_review_decision_recorded=false"
  echo "reason=notes_required"
  exit 22
fi

echo "bcd_review_decision_recorded=$([[ "${dry_run}" == "true" ]] && echo dry_run || echo true)"
echo "decision=${decision}"
echo "reviewer=${reviewer}"
echo "must_not_auto_approve=true"

run_one() {
  local run_group="$1"
  local run_name="$2"
  local label="$3"
  local args=(
    --decision "${decision}"
    --reviewer "${reviewer}"
    --notes "${notes}"
  )
  if [[ "${dry_run}" == "true" ]]; then
    args+=(--dry-run)
  fi

  echo "[${label}]"
  RUN_GROUP="${run_group}" RUN_NAME="${run_name}" \
    "${ROOT}/scripts/world_model/record_dataset_smoke_review_decision.sh" "${args[@]}"
}

run_one "dynamic_rgb" "smoke01" "b_dynamic_smoke"
run_one "frozen_dp_dynamic" "smoke01" "c_frozen_dp_smoke"
run_one "future_teacher" "smoke01" "d_future_teacher_smoke"
