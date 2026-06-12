#!/usr/bin/env bash
#SBATCH --job-name=reflex_check_gpu_env
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --export=ALL,VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json,DISPLAY=
#SBATCH --output=/public/home/yanhongru/ICLR2027/Reflex/logs/slurm/%x-%j.out
#SBATCH --error=/public/home/yanhongru/ICLR2027/Reflex/logs/slurm/%x-%j.err

set -euo pipefail

ROOT=/public/home/yanhongru/ICLR2027/Reflex
MS_DIR="${ROOT}/deps/ManiSkill_clean"
VENV="${ROOT}/.venv"

cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${MS_DIR}/examples/baselines/diffusion_policy:${PYTHONPATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export DISPLAY="${DISPLAY:-}"

echo "timestamp=$(date -Is)"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "node_list=${SLURM_JOB_NODELIST:-none}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
git -C "${MS_DIR}" rev-parse HEAD

nvidia-smi || true

python - <<'PY'
import torch
import mani_skill
import diffusion_policy
import diffusers

print("torch", torch.__version__)
print("torch_cuda_available", torch.cuda.is_available())
print("torch_cuda_version", torch.version.cuda)
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_name", torch.cuda.get_device_name(0))
print("mani_skill", getattr(mani_skill, "__version__", "version_unknown"))
print("diffusion_policy", getattr(diffusion_policy, "__version__", "version_unknown"))
print("diffusers", diffusers.__version__)
PY

echo "GPU environment check complete"
