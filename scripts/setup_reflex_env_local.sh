#!/usr/bin/env bash
# Local venv setup for Reflex.
# Run on login node only for dependency installation. Do not run replay/train/eval here.

set -e
set -o pipefail

ROOT=/public/home/yanhongru/ICLR2027/Reflex
MS_DIR="${ROOT}/deps/ManiSkill_clean"
VENV="${ROOT}/.venv"
UV_BIN="${UV_BIN:-/public/home/yanhongru/.local/bin/uv}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-/usr/bin/python3.10}"

cd "${ROOT}"

export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "=== Reflex local venv setup ==="
echo "root=${ROOT}"
echo "maniskill_dir=${MS_DIR}"
echo "maniskill_commit=$(git -C "${MS_DIR}" rev-parse HEAD)"
echo "pip_index=${PIP_INDEX_URL}"
echo "date=$(date -Is)"

if [[ ! -x "${VENV}/bin/python" ]]; then
  "${UV_BIN}" venv --python "${PYTHON_BOOTSTRAP}" "${VENV}"
fi

source "${VENV}/bin/activate"
python -V
python -m pip --version

echo ""
echo "=== Stage A: base packaging ==="
python -m pip install --upgrade pip setuptools wheel

echo ""
echo "=== Stage B: torch 2.5.1+cu121 ==="
python -m pip install \
  "numpy==1.26.4" \
  "torch==2.5.1" \
  "torchvision==0.20.1" \
  "torchaudio==2.5.1" \
  --index-url https://download.pytorch.org/whl/cu121 \
  --trusted-host download.pytorch.org

echo ""
echo "=== Stage C: ManiSkill clean clone ==="
python -m pip install -e "${MS_DIR}"

echo ""
echo "=== Stage D: ManiSkill Diffusion Policy baseline package ==="
python -m pip install -e "${MS_DIR}/examples/baselines/diffusion_policy"

echo ""
echo "=== Stage E: final numpy repin ==="
python -m pip install "numpy==1.26.4"
python -m pip install "opencv-python==4.11.0.86"

echo ""
echo "=== Stage F: import verification only ==="
export PYTHONPATH="${MS_DIR}/examples/baselines/diffusion_policy:${PYTHONPATH:-}"
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
import numpy
print("numpy:", numpy.__version__)
assert numpy.__version__.startswith("1.26."), numpy.__version__
import gymnasium
print("gymnasium:", gymnasium.__version__)
import mani_skill
print("mani_skill:", getattr(mani_skill, "__version__", "version_unknown"))
import sapien
print("sapien:", getattr(sapien, "__version__", "version_unknown"))
import cv2
print("opencv:", cv2.__version__)
import diffusers
print("diffusers:", diffusers.__version__)
import diffusion_policy
print("diffusion_policy: OK")
PY

echo "=== DONE $(date -Is) ==="
