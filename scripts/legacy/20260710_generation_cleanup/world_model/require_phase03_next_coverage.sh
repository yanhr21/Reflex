#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

EXPECTED_NEXT="${1:?Usage: require_phase03_next_coverage.sh <expected_next_required_coverage_group>}"

out="$(mktemp)"
trap 'rm -f "${out}"' EXIT

set +e
scripts/world_model/check_phase03_oracle_completion.sh >"${out}"
status="$?"
set -e

overall_complete="$(awk -F= '$1 == "phase03_oracle_overall_complete" { print $2 }' "${out}" | tail -1)"
next_group="$(awk -F= '$1 == "next_required_coverage_group" { print $2 }' "${out}" | tail -1)"
missing_items="$(awk -F= '$1 == "missing_coverage_items" { print $2 }' "${out}" | tail -1)"

echo "completion_gate_exit_status=${status}"
echo "phase03_oracle_overall_complete=${overall_complete:-unknown}"
echo "missing_coverage_items=${missing_items:-unknown}"
echo "next_required_coverage_group=${next_group:-unknown}"
echo "expected_next_required_coverage_group=${EXPECTED_NEXT}"

if [[ "${overall_complete:-unknown}" == "true" ]]; then
  echo "phase03_next_coverage_guard_ok=false"
  echo "reason=phase03_oracle_already_complete"
  exit 3
fi

if [[ "${next_group:-}" != "${EXPECTED_NEXT}" ]]; then
  echo "phase03_next_coverage_guard_ok=false"
  echo "reason=unexpected_next_required_coverage_group"
  exit 4
fi

echo "phase03_next_coverage_guard_ok=true"
