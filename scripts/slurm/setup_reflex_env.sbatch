#!/usr/bin/env bash
#SBATCH --job-name=reflex_setup_env
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/yanhongru/ICLR2027/Reflex/logs/slurm/%x-%j.out
#SBATCH --error=/public/home/yanhongru/ICLR2027/Reflex/logs/slurm/%x-%j.err

set -euo pipefail

ROOT=/public/home/yanhongru/ICLR2027/Reflex
MS_DIR="${ROOT}/deps/ManiSkill_clean"
VENV="${ROOT}/.venv"
UV_BIN="${UV_BIN:-/public/home/yanhongru/.local/bin/uv}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-/usr/bin/python3.10}"

cd "${ROOT}"
mkdir -p logs/slurm logs/env

if [[ "${NO_PROXY_RETRY:-0}" == "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
fi

export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"

{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID:-none}"
  echo "node_list=${SLURM_JOB_NODELIST:-none}"
  echo "root=${ROOT}"
  echo "maniskill_dir=${MS_DIR}"
  echo "maniskill_commit=$(git -C "${MS_DIR}" rev-parse HEAD)"
  echo "uv_bin=${UV_BIN}"
  echo "python_bootstrap=${PYTHON_BOOTSTRAP}"
} | tee "logs/env/setup-${SLURM_JOB_ID:-manual}.manifest"

if [[ "${CLEAR_VENV:-0}" == "1" || ! -x "${VENV}/bin/python" ]]; then
  if [[ "${CLEAR_VENV:-0}" == "1" ]]; then
    "${UV_BIN}" venv --clear --python "${PYTHON_BOOTSTRAP}" "${VENV}"
  else
    "${UV_BIN}" venv --python "${PYTHON_BOOTSTRAP}" "${VENV}"
  fi
else
  echo "Reusing existing virtual environment: ${VENV}"
fi
source "${VENV}/bin/activate"

uv pip install --upgrade pip setuptools wheel
uv pip install torch torchvision --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

python -m pip install --progress-bar on -e "${MS_DIR}"
python -m pip install --progress-bar on -e "${MS_DIR}/examples/baselines/diffusion_policy"

python - <<'PY'
import importlib
mods = ["torch", "mani_skill", "diffusion_policy", "diffusers", "sapien"]
for name in mods:
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "version_unknown"))
PY

echo "Environment setup complete: ${VENV}"
