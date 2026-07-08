#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

echo "dataset_slurm_status_ok=true"
echo "read_only=true"
echo "submits_slurm=false"
echo "root=${ROOT}"

if ! command -v squeue >/dev/null 2>&1; then
  echo "squeue_available=false"
  echo "reflex_job_count=0"
  echo "non_reflex_job_count=0"
  exit 0
fi
echo "squeue_available=true"

job_lines="$(squeue -u "${USER}" -h -o '%i|%j|%T|%M|%l|%D|%R|%b|%C' 2>/dev/null || true)"
if [[ -z "${job_lines}" ]]; then
  echo "total_user_job_count=0"
  echo "reflex_job_count=0"
  echo "non_reflex_job_count=0"
  exit 0
fi

total_count=0
reflex_count=0
non_reflex_count=0
unknown_count=0

while IFS='|' read -r job_id job_name job_state elapsed time_limit node_count reason tres_per_node cpus; do
  [[ -z "${job_id}" ]] && continue
  total_count=$((total_count + 1))
  job_info="$(scontrol show job "${job_id}" 2>/dev/null || true)"
  work_dir="$(printf '%s\n' "${job_info}" | sed -nE 's/.*WorkDir=([^[:space:]]+).*/\1/p' | tail -n 1)"
  node_list="$(printf '%s\n' "${job_info}" | sed -nE 's/(^|[[:space:]])NodeList=([^[:space:]]*).*/\2/p' | tail -n 1 | xargs)"
  req_node_list="$(printf '%s\n' "${job_info}" | sed -nE 's/.*ReqNodeList=([^[:space:]]+).*/\1/p' | tail -n 1)"
  exc_node_list="$(printf '%s\n' "${job_info}" | sed -nE 's/.*ExcNodeList=([^[:space:]]+).*/\1/p' | tail -n 1)"
  sched_node_list="$(printf '%s\n' "${job_info}" | sed -nE 's/.*SchedNodeList=([^[:space:]]+).*/\1/p' | tail -n 1)"
  alloc_node="$(printf '%s\n' "${job_info}" | sed -nE 's/.*AllocNode:Sid=([^[:space:]]+).*/\1/p' | tail -n 1)"

  if [[ -n "${work_dir}" && "${work_dir}" == "${ROOT}"* ]]; then
    class="reflex"
    reflex_count=$((reflex_count + 1))
  elif [[ -n "${work_dir}" ]]; then
    class="non_reflex"
    non_reflex_count=$((non_reflex_count + 1))
  else
    class="unknown_workdir"
    unknown_count=$((unknown_count + 1))
  fi

  echo "[job_${job_id}]"
  echo "  class=${class}"
  echo "  name=${job_name}"
  echo "  state=${job_state}"
  echo "  elapsed=${elapsed}"
  echo "  time_limit=${time_limit}"
  echo "  node_count=${node_count}"
  echo "  node_list=${node_list:-${reason}}"
  echo "  req_node_list=${req_node_list:-none}"
  echo "  exc_node_list=${exc_node_list:-none}"
  echo "  sched_node_list=${sched_node_list:-none}"
  echo "  reason=${reason}"
  echo "  tres_per_node=${tres_per_node}"
  echo "  cpus=${cpus}"
  echo "  alloc_node=${alloc_node:-unknown}"
  echo "  work_dir=${work_dir:-unknown}"
done <<< "${job_lines}"

echo "total_user_job_count=${total_count}"
echo "reflex_job_count=${reflex_count}"
echo "non_reflex_job_count=${non_reflex_count}"
echo "unknown_workdir_job_count=${unknown_count}"
