#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
WRAPPER="${ROOT}/scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_preflight_in_allocation.sh"

tmp_dir="$(mktemp -d /tmp/cosmos3_clean_dense_preflight_guard_selftest_XXXXXX)"
trap 'rm -rf "${tmp_dir}"' EXIT

run_expect_failure() {
  local expected_code="$1"
  local expected_text="$2"
  local name="$3"
  shift 3
  local log="${tmp_dir}/${name}.log"
  set +e
  "$@" >"${log}" 2>&1
  local code="$?"
  set -e
  if [[ "${code}" != "${expected_code}" ]]; then
    echo "expected ${name} exit ${expected_code}, got ${code}" >&2
    sed -n '1,120p' "${log}" >&2
    exit 1
  fi
  if ! grep -q "${expected_text}" "${log}"; then
    echo "expected ${name} log to contain ${expected_text}" >&2
    sed -n '1,120p' "${log}" >&2
    exit 1
  fi
}

run_expect_failure \
  46 \
  "refusing_clean_dense_preflight_without_live_query_coverage=true" \
  "coverage_disabled" \
  env \
    ALLOW_CLEAN_DENSE_PREFLIGHT=true \
    RUN_LIVE_QUERY_COVERAGE_AUDIT=false \
    OUTPUT_ROOT="${tmp_dir}/disabled_out" \
    CONDITION_ROOT="${tmp_dir}/disabled_condition" \
    bash "${WRAPPER}"

run_expect_failure \
  47 \
  "missing_live_query_coverage_summary=" \
  "coverage_missing_summary" \
  env \
    ALLOW_CLEAN_DENSE_PREFLIGHT=true \
    RUN_LIVE_QUERY_COVERAGE_AUDIT=true \
    LIVE_QUERY_COVERAGE_SUMMARIES="${tmp_dir}/does_not_exist.json" \
    OUTPUT_ROOT="${tmp_dir}/missing_out" \
    CONDITION_ROOT="${tmp_dir}/missing_condition" \
    bash "${WRAPPER}"

dry_log="${tmp_dir}/dry_run.log"
DRY_RUN_CONFIG_ONLY=true RUN_LIVE_QUERY_COVERAGE_AUDIT=false bash "${WRAPPER}" >"${dry_log}" 2>&1
grep -q "dry_run_config_only=true" "${dry_log}"
grep -q "run_live_query_coverage_audit=false" "${dry_log}"

echo "cosmos3_clean_dense_preflight_guard_selftest=passed"
