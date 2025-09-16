#!/usr/bin/env python3
"""
mjpeg_uplink_runner.py
Pull frames via FFmpeg and POST JPEGs to the Django uplink endpoint:
  /attendance/stream/uplink/<session>/?key=...

Usage examples:

# RTSP (Hikvision etc.)
python extras/mjpeg_uplink_runner.py \
  --source rtsp \
  --rtsp "rtsp://admin:PASS@192.168.1.10:554/Streaming/Channels/101/" \
  --session cap_demo \
  --key dev-stream-key-change-me \
  --server http://127.0.0.1:8000 \
  --fps 6 --size 1280x720 --quality 3

# Webcam (Linux V4L2)
python extras/mjpeg_uplink_runner.py \
  --source webcam \
  --device /dev/video0 \
  --session cap_demo \
  --key dev-stream-key-change-me \
  --server http://127.0.0.1:8000 \
  --fps 6 --size 1280x720 --quality 3
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from typing import Tuple

import requests

JPEG_SOI = b"\xff\xd8"  # start of image
JPEG_EOI = b"\xff\xd9"  # end of image


def parse_size(arg: str) -> Tuple[int, int]:
    if "x" not in arg:
        raise argparse.ArgumentTypeError("size must look like 1280x720")
    w, h = arg.lower().split("x", 1)
    return int(w), int(h)


def build_ffmpeg_cmd(args) -> list[str]:
    w, h = args.size
    vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease"

    common = [
        args.ffmpeg, "-hide_banner", "-loglevel", "error",
        "-an",  # no audio
        "-vf", vf,
        "-r", str(args.fps),  # output frame rate (controls POST rate)
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-q:v", str(args.quality),  # 2..31 (2=best, 31=worst)
        "pipe:1",
    ]

    if args.source == "rtsp":
        # low latency, TCP RTSP
        return [
            args.ffmpeg, "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer", "-flags", "low_delay",
            "-use_wallclock_as_timestamps", "1",
            "-i", args.rtsp,
        ] + common[2:]  # reuse from "-an" onward
    elif args.source == "webcam":
        # Linux V4L2 webcam
        return [
            args.ffmpeg, "-hide_banner", "-loglevel", "error",
            "-f", "v4l2",
            "-framerate", str(max(1, min(args.fps, 30))),
            "-video_size", f"{w}x{h}",
            "-i", args.device,
        ] + common[2:]
    else:
        raise ValueError("unknown source")


def post_frame(session: requests.Session, url: str, frame: bytes, timeout: float = 3.0) -> None:
    session.post(url, data=frame, headers={"Content-Type": "image/jpeg"}, timeout=timeout)


def run(args) -> int:
    uplink_url = f"{args.server.rstrip('/')}/attendance/stream/uplink/{args.session}/"
    if args.key:
        connector = "&" if "?" in uplink_url else "?"
        uplink_url = f"{uplink_url}{connector}key={args.key}"

    cmd = build_ffmpeg_cmd(args)
    print("FFmpeg:", " ".join(shlex.quote(c) for c in cmd), file=sys.stderr)
    print("Uplink:", uplink_url, file=sys.stderr)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
    )

    s = requests.Session()
    buf = bytearray()
    last_post = time.time()
    frames = 0

    try:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(8192)
            if not chunk:
                # FFmpeg ended (error or EOF)
                break
            buf.extend(chunk)

            # Trim leading noise until we find a SOI
            soi_idx = buf.find(JPEG_SOI)
            if soi_idx > 0:
                del buf[:soi_idx]

            # Search for end of image
            eoi_idx = buf.find(JPEG_EOI)
            if eoi_idx != -1 and eoi_idx + 2 <= len(buf):
                frame = bytes(buf[: eoi_idx + 2])
                del buf[: eoi_idx + 2]

                try:
                    post_frame(s, uplink_url, frame, timeout=3.0)
                    frames += 1
                    # Optional: crude throttle safeguard
                    if args.max_fps and args.max_fps > 0:
                        now = time.time()
                        min_interval = 1.0 / float(args.max_fps)
                        sleep = min_interval - (now - last_post)
                        if sleep > 0:
                            time.sleep(sleep)
                        last_post = time.time()
                except requests.RequestException as e:
                    print(f"[WARN] POST failed: {e}", file=sys.stderr)
                    # Keep reading and try next frame
                    time.sleep(0.1)
    finally:
        # drain & log ffmpeg stderr
        try:
            if proc.stderr:
                err = proc.stderr.read().decode(errors="ignore").strip()
                if err:
                    print("FFmpeg stderr:\n" + err, file=sys.stderr)
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass

    print(f"Posted frames: {frames}", file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Post MJPEG frames to BISK uplink endpoint")
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"), help="ffmpeg binary path")
    ap.add_argument("--server", default="http://127.0.0.1:8000", help="Django server base URL")
    ap.add_argument("--session", required=True, help="session id (use the same in viewer)")
    ap.add_argument("--key", default=os.environ.get("STREAM_UPLINK_KEY"),
                    help="uplink key (matches settings.STREAM_UPLINK_KEY)")

    ap.add_argument("--source", choices=["rtsp", "webcam"], required=True)
    ap.add_argument("--rtsp", help="RTSP URL (when --source=rtsp)")
    ap.add_argument("--device", default="/dev/video0", help="Webcam device (when --source=webcam)")

    ap.add_argument("--fps", type=int, default=6, help="target output FPS")
    ap.add_argument("--max-fps", type=int, default=0, help="optional soft cap on post rate")
    ap.add_argument("--size", type=parse_size, default=(1280, 720), help="WxH, e.g. 1280x720")
    ap.add_argument("--quality", type=int, default=3, help="JPEG quality 2..31 (2=best)")

    args = ap.parse_args()

    if args.source == "rtsp" and not args.rtsp:
        ap.error("--rtsp URL is required when --source=rtsp")

    sys.exit(run(args))


if __name__ == "__main__":
    main()
