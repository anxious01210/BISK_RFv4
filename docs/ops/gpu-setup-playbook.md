# GPU Setup Playbook (Ubuntu 24.04) — BISK_RFv4

This playbook gives you **two scripts** and the exact steps to set up a **GPU-first** environment for both your **dev laptop (GTX 1060)** and your **production server (RTX 20/40/50 series)**. It standardizes on **CUDA 12.x + cuDNN 9**, **ONNX Runtime GPU**, **InsightFace**, and **FFmpeg with NVENC**.

- **Script A — `snapshot_env.sh`**: Captures an exact manifest of your current GPU/toolchain/python stack on a working box.
- **Script B — `provision_gpu_stack.sh`**: Reproduces that stack on a fresh Ubuntu 24.04 machine (driver → CUDA 12.x → cuDNN → NVENC → Python GPU wheels).

> ✅ You can keep these files in your repo at `docs/ops/` for documentation and `scripts/gpu/` for actual execution.  
> ✅ Both scripts are **idempotent-ish**; the provisioner runs in **dry-run** mode by default—enable with `EXEC=1` to actually apply.

---

## Quick Map

- [Prerequisites](#prerequisites)
- [Where to put the scripts](#where-to-put-the-scripts)
- [Script A: snapshot\_env.sh](#script-a-snapshot_envsh)
  - [What it does](#what-it-does)
  - [Create the script](#create-the-script)
  - [Run it](#run-it)
- [Script B: provision\_gpu\_stack.sh](#script-b-provision_gpu_stacksh)
  - [What it does](#what-it-does-1)
  - [Create the script](#create-the-script-1)
  - [Dry-run vs real run](#dry-run-vs-real-run)
  - [Verification commands](#verification-commands)
- [Recommended versions & notes](#recommended-versions--notes)
- [Troubleshooting](#troubleshooting)
- [Appendix: One-liners](#appendix-one-liners)

---

## Prerequisites

- Ubuntu **24.04** on both dev and production boxes.
- Ability to `sudo`.
- Internet access to NVIDIA repositories.
- A Python **virtual environment** for BISK\_RFv4 (the provision script can create one if missing).

---

## Where to put the scripts

**Repository layout (suggested):**
```
BISK_RFv4/
├─ docs/
│  └─ ops/
│     └─ gpu-setup-playbook.md   <-- this file
└─ scripts/
   └─ gpu/
      ├─ snapshot_env.sh
      └─ provision_gpu_stack.sh
```

> You’ll **run** the scripts from `scripts/gpu/`. Keep this `.md` in `docs/ops/` for reference.

---

## Script A: `snapshot_env.sh`

### What it does
- Records:
  - **GPU/driver** (`nvidia-smi`)
  - **CUDA/NVCC** (paths and versions)
  - **cuDNN** / CUDA packages installed via `dpkg`
  - **FFmpeg** NVENC/NVDEC encoder availability
  - **Python wheels** in your active venv (ORT, InsightFace, OpenCV, NumPy, Torch if present)
  - ORT available providers (`CUDAExecutionProvider`, etc.)
- Writes to a manifest file: `gpu_env_manifest_<HOSTNAME>_<DATE>.txt`

### Create the script
Create `scripts/gpu/snapshot_env.sh` with the content below:

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-gpu_env_manifest_$(hostname)_$(date +%Y%m%d).txt}"

{
  echo "===== GPU/Driver ====="
  command -v nvidia-smi >/dev/null && nvidia-smi || echo "nvidia-smi not found"
  echo
  echo "===== CUDA/NVCC ====="
  command -v nvcc >/dev/null && nvcc --version || echo "nvcc not found"
  echo "CUDA_PATH=${CUDA_PATH:-}"
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
  echo
  echo "===== GCC ====="
  gcc --version | head -n1 || true
  echo
  echo "===== cuDNN (dpkg) ====="
  dpkg -l | awk '/cudnn9|libcudnn/ {print $1,$2,$3}' || true
  echo
  echo "===== CUDA packages (dpkg) ====="
  dpkg -l | awk '/cuda-toolkit-12|cuda-12/ {print $1,$2,$3}' || true
  echo
  echo "===== FFmpeg/NVENC ====="
  command -v ffmpeg >/dev/null && ffmpeg -hide_banner -loglevel error -encoders | grep -E "nvenc|nvdec" || echo "ffmpeg not found"
  command -v ffmpeg >/dev/null && ffmpeg -hide_banner -hwaccels || true
  dpkg -l | awk '/ffmpeg|libffmpeg-nvenc-dev|nv-codec-headers/ {print $1,$2,$3}' || true
  echo
  echo "===== Python (venv) ====="
  python -c "import sys; print('python', sys.version.replace('\n',' '))"
  python - <<'PY'
import importlib, json
pkgs = ["onnxruntime","onnxruntime-gpu","insightface","numpy","opencv-python","opencv-python-headless","torch"]
out = {}
for p in pkgs:
    modname = p.replace("-","_")
    try:
        m = importlib.import_module(modname)
        out[p] = getattr(m, "__version__", "unknown")
    except Exception:
        out[p] = None
print(json.dumps(out, indent=2))
PY
  python - <<'PY'
try:
    import onnxruntime as ort
    print("ORT providers:", ort.get_available_providers())
except Exception as e:
    print("ORT import failed:", e)
PY
} | tee "${OUT}"

echo "Wrote ${OUT}"
```

Then make it executable:
```bash
chmod +x scripts/gpu/snapshot_env.sh
```

### Run it
Activate your venv and run:
```bash
source .venv/bin/activate
cd scripts/gpu
./snapshot_env.sh
# -> gpu_env_manifest_<HOST>_<DATE>.txt
```

Commit the manifest to your repo (or keep it somewhere safe) so prod can mirror the versions.

---

## Script B: `provision_gpu_stack.sh`

### What it does
- Installs the **NVIDIA driver** (recommended or explicit package)
- Installs **CUDA 12.x toolkit** and sets PATH/LD\_LIBRARY\_PATH
- Installs **cuDNN 9 for CUDA 12**
- Installs **NVENC/NVDEC headers** and **FFmpeg**
- Ensures **GCC 13** is available (CUDA 12.x friendly)
- Creates/updates your **Python venv** with GPU wheels:
  - `onnxruntime-gpu` (≥ 1.22)
  - `insightface` (≥ 0.7)
  - `opencv-python-headless` (≥ 4.10)
  - `numpy` pinned to a compatible range
- Runs **smoke checks** (nvidia-smi, nvcc, ffmpeg nvenc encoders, ORT providers)

### Create the script
Create `scripts/gpu/provision_gpu_stack.sh` with the content below:

```bash
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
# cuDNN 9 for CUDA 12 (package name may vary slightly by mirror)
_run "sudo apt-get install -y cudnn9-cuda-12 || sudo apt-get install -y libcudnn9-cuda-12 || true"

echo "=== STEP 4: NVENC/NVDEC headers + ffmpeg ==="
# nv-codec-headers (libffmpeg-nvenc-dev) + ffmpeg from Ubuntu 24.04
_run "sudo apt-get install -y libffmpeg-nvenc-dev ffmpeg"

echo "=== STEP 5: GCC sanity (optional, for building things against CUDA) ==="
# CUDA 12.4 pairs well with GCC<=13.x; ensure gcc-13 present and selectable
_run "sudo apt-get install -y gcc-13 g++-13"
# Optionally pin via update-alternatives if needed:
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
```

Make it executable:
```bash
chmod +x scripts/gpu/provision_gpu_stack.sh
```

### Dry-run vs real run

- **Dry-run (default):** prints all actions without changing the system.
  ```bash
  cd scripts/gpu
  ./provision_gpu_stack.sh
  ```

- **Real run (apply changes):**
  ```bash
  EXEC=1 ./provision_gpu_stack.sh
  ```

- **Pin explicit versions (optional):**
  ```bash
  EXEC=1 DRIVER_PACKAGE=nvidia-driver-555 CUDA_MINOR=12-4 ./provision_gpu_stack.sh
  ```

- **Custom venv path (optional):**
  ```bash
  EXEC=1 VENV=/opt/bisk/.venv ./provision_gpu_stack.sh
  ```

> 🔁 **Reboot** after the driver install step completes: `sudo reboot`

### Verification commands

After the real run (and reboot if driver was installed), verify:

```bash
# Driver & CUDA
nvidia-smi
source /etc/profile.d/cuda_path.sh
nvcc --version

# cuDNN & CUDA packages
dpkg -l | grep -E 'cudnn9|cuda-toolkit-12'

# FFmpeg with NVENC
ffmpeg -hide_banner -hwaccels
ffmpeg -hide_banner -encoders | grep nvenc

# Python GPU wheels
source /path/to/.venv/bin/activate
python - <<'PY'
import onnxruntime as ort
print('Providers:', ort.get_available_providers())
from insightface.app import FaceAnalysis
app = FaceAnalysis(providers=['CUDAExecutionProvider','CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640,640))
print('InsightFace OK ->', app.models['recognition'].session.get_providers())
PY
```

You should see `CUDAExecutionProvider` in both ORT and InsightFace outputs.

---

## Recommended versions & notes

- **CUDA 12.x** keeps your **GTX 1060 (Pascal)** working and supports **RTX 20/40/50** series.
- **cuDNN 9** pairs with CUDA 12 for ONNX Runtime GPU.
- **GCC 13** is Ubuntu 24.04 default—compatible with CUDA 12 toolchain.
- Prefer **`opencv-python-headless`** on servers (no GUI deps).

If you *must* use **onnxruntime-gpu** with **NumPy 2.x**, pick an ORT-GPU build that supports it (≥ 1.22). If you are stuck on an older ORT-GPU, pin NumPy `< 2.0`.

---

## Troubleshooting

- **`CUDAExecutionProvider` missing**
  - Ensure driver installed (`nvidia-smi` works).
  - Ensure CUDA 12.x & cuDNN packages are present.
  - Ensure your venv uses `onnxruntime-gpu` (not CPU) and no conflicts.
- **FFmpeg lacks NVENC encoders**
  - Install `libffmpeg-nvenc-dev`.
  - If Ubuntu’s FFmpeg lacks NVENC, compile FFmpeg with `--enable-nvenc` (optional advanced step).
- **Pascal (GTX 1060) quirks**
  - Stay on CUDA **12.x**; newer major toolkits may drop Pascal toolchains.

---

## Appendix: One-liners

- Show ORT providers:
  ```bash
  python - <<'PY'
import onnxruntime as ort; print(ort.get_available_providers())
PY
  ```

- Show InsightFace providers:
  ```bash
  python - <<'PY'
from insightface.app import FaceAnalysis
app = FaceAnalysis(providers=['CUDAExecutionProvider','CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640,640))
print(app.models['recognition'].session.get_providers())
PY
  ```

- Verify FFmpeg NVENC:
  ```bash
  ffmpeg -hide_banner -encoders | grep nvenc
  ```

- Snapshot env:
  ```bash
  cd scripts/gpu && ./snapshot_env.sh
  ```

- Provision (real run, explicit driver/toolkit):
  ```bash
  cd scripts/gpu
  EXEC=1 DRIVER_PACKAGE=nvidia-driver-555 CUDA_MINOR=12-4 ./provision_gpu_stack.sh
  ```

---

**That’s it.** Copy the two scripts into `scripts/gpu/`, make them executable, and keep this guide at `docs/ops/gpu-setup-playbook.md`. Use **Script A** on your *working* laptop to capture a manifest. Use **Script B** on a *fresh* server to reproduce the environment (then verify).
