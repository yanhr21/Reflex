#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
EXECUTE=false

usage() {
  cat <<EOF
usage: launch_dataset_bcd_next_production_shard_tmux.sh [--execute]

Dry-runs the next missing B/C/D production shard by default. Use --execute only
after the B/C/D smoke review is explicitly approved and approval files exist.
The script submits at most one shard.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --execute)
      EXECUTE=true
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

echo "launch_bcd_next_production_shard=true"
echo "execute=${EXECUTE}"
echo "max_launches=1"
echo "requires_bcd_human_review_approval=true"

review_status="$("${ROOT}/scripts/world_model/dataset_bcd_review_block_status.sh")"
review_reason="$(printf '%s\n' "${review_status}" | sed -nE 's/[[:space:]]*reason=([^[:space:]]+).*/\1/p' | tail -n 1)"
all_approved="$(printf '%s\n' "${review_status}" | sed -nE 's/[[:space:]]*all_approved=([^[:space:]]+).*/\1/p' | tail -n 1)"
all_approved="${all_approved:-false}"

if [[ "${all_approved}" != "true" ]]; then
  echo "ready_to_launch=false"
  echo "reason=${review_reason:-human_review_approval_missing}"
  echo "review_status_command=scripts/world_model/dataset_bcd_review_block_status.sh"
  echo "approval_command=scripts/world_model/record_dataset_bcd_smoke_review_decision.sh --decision approved --reviewer <name> --notes <text>"
  if [[ "${EXECUTE}" == "true" ]]; then
    exit 8
  fi
  exit 0
fi

set +e
next_status="$("${ROOT}/scripts/world_model/dataset_bcd_production_next_shard.sh" 2>&1)"
next_rc=$?
set -e
printf '%s\n' "${next_status}"
if [[ "${next_rc}" -ne 0 ]]; then
  echo "ready_to_launch=false"
  echo "reason=next_shard_status_failed"
  exit "${next_rc}"
fi

next_available="$(printf '%s\n' "${next_status}" | sed -nE 's/[[:space:]]*next_shard_available=([^[:space:]]+).*/\1/p' | tail -n 1)"
next_stage="$(printf '%s\n' "${next_status}" | sed -nE 's/[[:space:]]*next_stage=([^[:space:]]+).*/\1/p' | tail -n 1)"
next_family="$(printf '%s\n' "${next_status}" | sed -nE 's/[[:space:]]*next_family=([^[:space:]]+).*/\1/p' | tail -n 1)"

if [[ "${next_available}" != "true" ]]; then
  echo "ready_to_launch=false"
  echo "reason=all_matching_shards_ready"
  exit 0
fi

echo "ready_to_launch=true"
echo "next_stage=${next_stage}"
echo "next_family=${next_family}"
echo "launch_command=scripts/slurm/launch_dataset_bcd_production_shards_tmux.sh --execute --stage ${next_stage} --family ${next_family} --max-launches 1"

if [[ "${EXECUTE}" != "true" ]]; then
  echo "dry_run_only=true"
  echo "no_slurm_submitted=true"
  exit 0
fi

"${ROOT}/scripts/slurm/launch_dataset_bcd_production_shards_tmux.sh" \
  --execute \
  --stage "${next_stage}" \
  --family "${next_family}" \
  --max-launches 1
