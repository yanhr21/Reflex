#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cosmos3_wrapper_guard_selftest_XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT

run_expect_refusal() {
  local name="$1"
  local expected_code="$2"
  local expected_text="$3"
  local output_root="$4"
  shift 4

  local stdout="${TMP_DIR}/${name}.stdout"
  local stderr="${TMP_DIR}/${name}.stderr"
  set +e
  (
    cd "${ROOT}"
    "$@"
  ) >"${stdout}" 2>"${stderr}"
  local code=$?
  set -e

  if [[ "${code}" != "${expected_code}" ]]; then
    echo "${name}: expected exit ${expected_code}, got ${code}" >&2
    echo "--- stdout ---" >&2
    cat "${stdout}" >&2
    echo "--- stderr ---" >&2
    cat "${stderr}" >&2
    exit 1
  fi
  if ! grep -q "${expected_text}" "${stderr}"; then
    echo "${name}: missing refusal marker ${expected_text}" >&2
    echo "--- stderr ---" >&2
    cat "${stderr}" >&2
    exit 1
  fi
  if [[ -e "${output_root}" ]]; then
    echo "${name}: output root was created before refusal: ${output_root}" >&2
    exit 1
  fi
}

run_expect_refusal \
  "live_panel_manifest_prefix" \
  "42" \
  "refusing_non_method_live_receding_panel=true" \
  "${TMP_DIR}/live_panel_manifest_prefix_out" \
  env \
    SLURM_JOB_ID=1 \
    SLURM_STEP_ID=1 \
    OUTPUT_ROOT="${TMP_DIR}/live_panel_manifest_prefix_out" \
    PREFIX_START_MODE=manifest \
    bash scripts/slurm/run_cosmos3_live_receding_panel_in_allocation.sh

run_expect_refusal \
  "live_loop_explicit_role" \
  "42" \
  "refusing_non_method_live_receding_loop=true" \
  "${TMP_DIR}/live_loop_explicit_role_out" \
  env \
    SLURM_JOB_ID=1 \
    SLURM_STEP_ID=1 \
    SOURCE_H5="${TMP_DIR}/missing.h5" \
    OUTPUT_ROOT="${TMP_DIR}/live_loop_explicit_role_out" \
    PREFIX_ROLE=target_motion_observed \
    bash scripts/slurm/run_cosmos3_live_receding_loop_in_allocation.sh

run_expect_refusal \
  "old_panel_default_refusal" \
  "43" \
  "refusing_old_oneshot_closed_loop_panel=true" \
  "${TMP_DIR}/old_panel_default_refusal_out" \
  env \
    SLURM_JOB_ID=1 \
    SLURM_STEP_ID=1 \
    OUTPUT_ROOT="${TMP_DIR}/old_panel_default_refusal_out" \
    VISUAL_REVIEW_STATUS=pass \
    bash scripts/slurm/run_cosmos3_closed_loop_panel_in_allocation.sh

run_expect_refusal \
  "old_loop_default_refusal" \
  "43" \
  "refusing_old_oneshot_closed_loop=true" \
  "${TMP_DIR}/old_loop_default_refusal_out" \
  env \
    SLURM_JOB_ID=1 \
    SLURM_STEP_ID=1 \
    OUTPUT_ROOT="${TMP_DIR}/old_loop_default_refusal_out" \
    bash scripts/slurm/run_cosmos3_receding_closed_loop_in_allocation.sh

echo "cosmos3_closed_loop_wrapper_guard_selftest=passed"
