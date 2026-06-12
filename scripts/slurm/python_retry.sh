#!/usr/bin/env bash

run_python_with_retry() {
  local max_attempts="${PYTHON_RETRY_MAX:-5}"
  local delay_seconds="${PYTHON_RETRY_DELAY_SECONDS:-5}"
  local attempt rc
  for attempt in $(seq 1 "${max_attempts}"); do
    python "$@" && return 0
    rc=$?
    if [[ "${rc}" == "64" || "${rc}" == "65" || "${rc}" == "66" ]]; then
      return "${rc}"
    fi
    echo "python command failed with rc=${rc}; retry ${attempt}/${max_attempts}" >&2
    sleep "${delay_seconds}"
  done
  return "${rc}"
}

run_python_stdin_with_retry() {
  local tmp
  tmp="$(mktemp)"
  cat > "${tmp}"
  local max_attempts="${PYTHON_RETRY_MAX:-5}"
  local delay_seconds="${PYTHON_RETRY_DELAY_SECONDS:-5}"
  local attempt rc
  for attempt in $(seq 1 "${max_attempts}"); do
    python "${tmp}" "$@" && {
      rm -f "${tmp}"
      return 0
    }
    rc=$?
    if [[ "${rc}" == "64" || "${rc}" == "65" || "${rc}" == "66" ]]; then
      rm -f "${tmp}"
      return "${rc}"
    fi
    echo "python stdin command failed with rc=${rc}; retry ${attempt}/${max_attempts}" >&2
    sleep "${delay_seconds}"
  done
  rm -f "${tmp}"
  return "${rc}"
}
