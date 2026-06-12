#!/usr/bin/env bash
#SBATCH --job-name=reflex_replay_peg
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --export=ALL,VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json,DISPLAY=
#SBATCH --output=/public/home/yanhongru/ICLR2027/Reflex/logs/slurm/%x-%j.out
#SBATCH --error=/public/home/yanhongru/ICLR2027/Reflex/logs/slurm/%x-%j.err

set -euo pipefail

ROOT=/public/home/yanhongru/ICLR2027/Reflex
MS_DIR="${ROOT}/deps/ManiSkill_clean"
VENV="${ROOT}/.venv"
SRC_DIR=/public/home/yanhongru/.maniskill/demos/PegInsertionSide-v1/motionplanning
OUT_DIR="${ROOT}/data/official_replay/PegInsertionSide-v1/motionplanning"
NUM_ENVS="${NUM_ENVS:-16}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-1000}"

cd "${ROOT}"
source "${VENV}/bin/activate"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export DISPLAY="${DISPLAY:-}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
mkdir -p "${OUT_DIR}" logs/replay

SRC_H5="${SRC_DIR}/trajectory.h5"
SRC_JSON="${SRC_DIR}/trajectory.json"
WORK_H5="${OUT_DIR}/trajectory.h5"
WORK_JSON="${OUT_DIR}/trajectory.json"
MANIFEST="${OUT_DIR}/replay-${SLURM_JOB_ID:-manual}.manifest"
OUT_PREFIX="${OUT_DIR}/trajectory.state.pd_ee_delta_pose.physx_cpu"

cp -f "${SRC_H5}" "${WORK_H5}"
cp -f "${SRC_JSON}" "${WORK_JSON}"
rm -f "${OUT_PREFIX}.h5" "${OUT_PREFIX}.json" "${OUT_PREFIX}".*.h5 "${OUT_PREFIX}".*.json

{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID:-none}"
  echo "node_list=${SLURM_JOB_NODELIST:-none}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "vk_icd_filenames=${VK_ICD_FILENAMES:-unset}"
  echo "hdf5_use_file_locking=${HDF5_USE_FILE_LOCKING:-unset}"
  echo "root=${ROOT}"
  echo "maniskill_dir=${MS_DIR}"
  echo "maniskill_commit=$(git -C "${MS_DIR}" rev-parse HEAD)"
  echo "src_h5=${SRC_H5}"
  echo "src_json=${SRC_JSON}"
  echo "work_h5=${WORK_H5}"
  echo "num_envs=${NUM_ENVS}"
  echo "expected_episodes=${EXPECTED_EPISODES}"
  echo "allow_failure=true"
  echo "source_sizes:"
  ls -l "${SRC_H5}" "${SRC_JSON}"
  echo "source_sha256:"
  sha256sum "${SRC_H5}" "${SRC_JSON}"
} | tee "${MANIFEST}"

cd "${MS_DIR}"
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path "${WORK_H5}" \
  --use-first-env-state \
  -c pd_ee_delta_pose \
  -o state \
  --save-traj \
  --allow-failure \
  --num-envs "${NUM_ENVS}" \
  -b physx_cpu

OUT_H5="${OUT_DIR}/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
OUT_JSON="${OUT_DIR}/trajectory.state.pd_ee_delta_pose.physx_cpu.json"

python - <<PY
import json
from pathlib import Path
out_json = Path("${OUT_JSON}")
out_h5 = Path("${OUT_H5}")
assert out_h5.exists(), f"missing {out_h5}"
assert out_json.exists(), f"missing {out_json}"
with out_json.open("r") as f:
    data = json.load(f)
episodes = data.get("episodes", [])
print("output_h5", out_h5, out_h5.stat().st_size)
print("output_json", out_json, out_json.stat().st_size)
print("episodes", len(episodes))
assert len(episodes) == int("${EXPECTED_EPISODES}"), len(episodes)
PY

{
  echo "output_files:"
  ls -l "${OUT_H5}" "${OUT_JSON}"
  echo "output_sha256:"
  sha256sum "${OUT_H5}" "${OUT_JSON}"
  echo "status=complete"
} | tee -a "${MANIFEST}"

echo "Replay complete: ${OUT_H5}"
