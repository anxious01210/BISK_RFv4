#!/usr/bin/env bash
set -euo pipefail
echo "===== GPU ====="
nvidia-smi -L || true
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv || true
echo
echo "===== Python / ORT / InsightFace ====="
python -V
python - <<'PY'
import onnxruntime as ort
import insightface
print("ORT providers:", ort.get_available_providers())
print("insightface version:", insightface.__version__)
PY
echo
echo "===== FFmpeg ====="
which ffmpeg && ffmpeg -hide_banner -hwaccels
echo
echo "===== RTSP Probe Cam-01 ====="
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,avg_frame_rate \
  -of default=noprint_wrappers=1:nokey=0 -rtsp_transport tcp \
  "rtsp://admin:B!sk2025@192.168.137.95:554/Streaming/Channels/101/" || echo "Cam-01 probe failed"
echo
echo "===== RTSP Probe Cam-02 ====="
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,avg_frame_rate \
  -of default=noprint_wrappers=1:nokey=0 -rtsp_transport tcp \
  "rtsp://admin:B!sk2025@192.168.137.96:554/Streaming/Channels/101/" || echo "Cam-02 probe failed"
echo
echo "===== Snapshots dir ====="
ls -l --full-time media/snapshots || true
echo
echo "===== Heartbeat echo ====="
curl -sS -X POST 'http://127.0.0.1:8000/api/runner/heartbeat/?echo=1' \
  -H 'Content-Type: application/json' -H 'X-BISK-KEY: dev-key-change-me' \
  -d '{"camera_id":1,"profile_id":1,"pid":'"$$"',"camera_fps":2,"snapshot_every":30}' || true
echo
echo "Done."
