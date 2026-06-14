#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
FULL_WRAPPER="${ROOT}/scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_fix1recipe_in_allocation.sh"
OVERFIT_WRAPPER="${ROOT}/scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_overfit2_fix1recipe_in_allocation.sh"

tmp_dir="$(mktemp -d /tmp/cosmos3_legacy_sft_wrapper_guard_selftest_XXXXXX)"
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
  66 \
  "refusing_legacy_v7_733_sampled_role_sft=true" \
  "full_legacy_refusal" \
  bash "${FULL_WRAPPER}"

run_expect_failure \
  67 \
  "refusing_legacy_v7_733_overfit2_sft=true" \
  "overfit_legacy_refusal" \
  bash "${OVERFIT_WRAPPER}"

DRY_RUN_CONFIG_ONLY=true bash "${FULL_WRAPPER}" >"${tmp_dir}/full_dry.log" 2>&1
grep -q "dry_run_config_only=true" "${tmp_dir}/full_dry.log"
grep -q "would_refuse_legacy_training=true" "${tmp_dir}/full_dry.log"

DRY_RUN_CONFIG_ONLY=true bash "${OVERFIT_WRAPPER}" >"${tmp_dir}/overfit_dry.log" 2>&1
grep -q "dry_run_config_only=true" "${tmp_dir}/overfit_dry.log"
grep -q "would_refuse_legacy_training=true" "${tmp_dir}/overfit_dry.log"

echo "cosmos3_legacy_sft_wrapper_guard_selftest=passed"
