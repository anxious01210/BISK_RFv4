#!/usr/bin/env bash
  # chmod +x scripts/gpu/snapshot_env.sh
  #./scripts/gpu/snapshot_env.sh
  ## => gpu_env_manifest_<host>_<date>.txt
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
import pkgutil, importlib, json
pkgs = ["onnxruntime", "onnxruntime_gpu", "insightface", "numpy", "opencv-python", "opencv-python-headless", "torch"]
out = {}
for p in pkgs:
    try:
        m = importlib.import_module(p.replace('-', '_'))
        v = getattr(m, '__version__', 'unknown')
        out[p] = v
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
