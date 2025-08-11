#!/usr/bin/env python3
# Make them executable (optional):
# chmod +x extras/recognize_ffmpeg.py extras/recognize_opencv.py
# Not needed for our enforcer, because we invoke them like:
# python /abs/path/extras/recognize_ffmpeg.py …
# You only need chmod +x if you plan to run them directly as ./recognize_ffmpeg.py using the shebang.
import argparse, os, signal, sys, time, json, urllib.request


def post(url, payload, key, timeout=2):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json", "X-BISK-KEY": key})
    urllib.request.urlopen(req, timeout=timeout).read()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--det_set", default="auto")
    p.add_argument("--hb", required=True)
    p.add_argument("--hb_key", required=True)
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
                post(args.hb, payload, args.hb_key, timeout=2)
            except Exception:
                pass
            last_beat = now
        time.sleep(0.25)

    sys.exit(0)


if __name__ == "__main__":
    main()
