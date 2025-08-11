#!/usr/bin/env python3
# Make them executable (optional):
# chmod +x extras/recognize_ffmpeg.py extras/recognize_opencv.py
import argparse, os, signal, sys, time, json, urllib.request


def post(url, payload, timeout=2):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=timeout).read()


def main():
    p = argparse.ArgumentParser(description="OpenCV runner stub (heartbeat only).")
    p.add_argument("--camera", required=True, help="Camera ID (DB)")
    p.add_argument("--profile", required=True, help="StreamProfile ID (DB)")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--det_set", default="auto")
    p.add_argument("--hb", required=True, help="Heartbeat URL")
    p.add_argument("--rtsp", default="", help="RTSP URL (optional)")
    args, _ = p.parse_known_args()

    stop = False

    def handler(signum, frame):
        nonlocal stop;
        stop = True

    signal.signal(signal.SIGTERM, handler)

    detected = matched = 0
    last_beat = 0.0

    while not stop:
        now = time.monotonic()
        if now - last_beat >= 5.0:
            payload = {
                "camera_id": int(args.camera),
                "profile_id": int(args.profile),
                "fps": float(args.fps),
                "detected": detected,
                "matched": matched,
                "latency_ms": 0.0,
                "last_error": "",
            }
            try:
                post(args.hb, payload, timeout=2)
            except Exception:
                pass
            last_beat = now
        time.sleep(0.25)

    sys.exit(0)


if __name__ == "__main__":
    main()
