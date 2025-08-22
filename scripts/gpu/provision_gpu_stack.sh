#!/usr/bin/env bash
# Idempotent-ish setup for Ubuntu 24.04 (Noble)
# DRY-RUN by default: set EXEC=1 to actually run commands.

set -euo pipefail

# -------- configurable knobs ----------
EXEC="${EXEC:-0}"                # 0 = print, 1 = execute
CUDA_MINOR="${CUDA_MINOR:-12-4}" # toolkit minor (12-4, 12-5, …)
USE_RECOMMENDED_DRIVER="${USE_RECOMMENDED_DRIVER:-1}"
DRIVER_PACKAGE="${DRIVER_PACKAGE:-}"  # e.g. nvidia-driver-555 (optional explicit)
VENV="${VENV:-$PWD/.venv}"      # path to your project venv
PIN_NUMPY_RANGE="${PIN_NUMPY_RANGE:->=1.26,<3}"
PIN_ORT_GPU="${PIN_ORT_GPU:-onnxruntime-gpu>=1.22}"
PIN_INSIGHT="${PIN_INSIGHT:-insightface>=0.7}"
PIN_OCV="${PIN_OCV:-opencv-python-headless>=4.10}"
# --------------------------------------

_run() { echo "+ $*"; [ "$EXEC" = "1" ] && eval "$*"; }

echo "=== STEP 0: base tools ==="
_run "sudo apt-get update"
_run "sudo apt-get install -y software-properties-common curl wget gnupg ca-certificates lsb-release build-essential"

echo "=== STEP 1: NVIDIA driver ==="
if [ -n "$DRIVER_PACKAGE" ]; then
  _run "sudo apt-get install -y ${DRIVER_PACKAGE}"
elif [ "$USE_RECOMMENDED_DRIVER" = "1" ]; then
  _run "sudo ubuntu-drivers autoinstall"
else
  echo "Skipping driver install (set DRIVER_PACKAGE or USE_RECOMMENDED_DRIVER=1)"
fi

echo "=== STEP 2: CUDA ${CUDA_MINOR} repo & toolkit ==="
# NVIDIA CUDA repo keyring
_run "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb"
_run "sudo dpkg -i /tmp/cuda-keyring.deb"
_run "sudo apt-get update"
_run "sudo apt-get install -y cuda-toolkit-${CUDA_MINOR}"

# Add environment to /etc/profile.d
CUDA_HOME="/usr/local/cuda-${CUDA_MINOR/./-}"
if [ ! -d "$CUDA_HOME" ]; then CUDA_HOME="/usr/local/cuda-${CUDA_MINOR}"; fi
_run "echo 'export PATH=${CUDA_HOME}/bin:\$PATH' | sudo tee /etc/profile.d/cuda_path.sh > /dev/null"
_run "echo 'export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:\$LD_LIBRARY_PATH' | sudo tee -a /etc/profile.d/cuda_path.sh > /dev/null"

echo "=== STEP 3: cuDNN for CUDA 12 ==="
# cuDNN 9 for CUDA 12
_run "sudo apt-get install -y cudnn9-cuda-12"

echo "=== STEP 4: NVENC/NVDEC headers + ffmpeg ==="
# nv-codec-headers (libffmpeg-nvenc-dev) + ffmpeg from Ubuntu 24.04
_run "sudo apt-get install -y libffmpeg-nvenc-dev ffmpeg"

echo "=== STEP 5: GCC sanity (optional, for building things against CUDA) ==="
# CUDA 12.4 pairs well with GCC<=13.x; ensure gcc-13 present and selectable
_run "sudo apt-get install -y gcc-13 g++-13"
# You can pin via update-alternatives if needed:
# _run "sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130 --slave /usr/bin/g++ g++ /usr/bin/g++-13"

echo "=== STEP 6: Python GPU wheel stack in venv ==="
if [ ! -d "$VENV" ]; then
  _run "python3 -m venv \"$VENV\""
fi
# shellcheck disable=SC1090
_run "source \"$VENV/bin/activate\" && pip install -U pip wheel setuptools"
# Clean conflicts: remove CPU ORT & GUI OpenCV
_run "source \"$VENV/bin/activate\" && pip uninstall -y onnxruntime || true"
_run "source \"$VENV/bin/activate\" && pip uninstall -y opencv-python || true"
# Install GPU-first stack (pins can be edited at top)
_run "source \"$VENV/bin/activate\" && pip install '${PIN_ORT_GPU}' '${PIN_INSIGHT}' '${PIN_OCV}' 'numpy${PIN_NUMPY_RANGE}'"
# Basic smoke checks
_run "source \"$VENV/bin/activate\" && python - <<'PY'\nimport onnxruntime as ort, cv2, numpy\nprint('ORT providers:', ort.get_available_providers())\nprint('cv2', cv2.__version__)\nprint('numpy', numpy.__version__)\nPY"

echo "=== STEP 7: Smoke tests ==="
_run "nvidia-smi || true"
_run "bash -lc 'source /etc/profile.d/cuda_path.sh && nvcc --version || true'"
_run "ffmpeg -hide_banner -encoders | grep -E \"nvenc|nvdec\" || true"
echo "Done. Reboot may be required after driver install."
