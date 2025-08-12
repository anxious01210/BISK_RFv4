#!/usr/bin/env python3
"""
FFmpeg runner (Type 1) — lightweight heartbeat + periodic snapshot.

- FPS is probed via ffprobe (fast & robust).
- Snapshot uses ffmpeg with -update 1 so we always overwrite <camera_id>.jpg.
- Heartbeats are sent at a fixed cadence (default 10s) using a monotonic clock.
- Snapshot is taken every N heartbeats (default every 3).

Invoke: python /abs/path/extras/recognize_ffmpeg.py --camera 1 --profile 1 ...
"""

import argparse
import json
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
import re


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------
def post(url: str, payload: dict, key: str, timeout: float = 3.0) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-BISK-KEY": key,
        },
    )
    urllib.request.urlopen(req, timeout=timeout).read()


# ---------------------------------------------------------------------------
# FPS via ffprobe (reliable across ffmpeg builds)
# ---------------------------------------------------------------------------
def probe_fps(ffprobe: str, rtsp: str) -> float | None:
    """
    Returns a float fps or None.
    Tries r_frame_rate then avg_frame_rate. Accepts forms like '4/1', '30000/1001', or '25'.
    """
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,avg_frame_rate",
        "-of",
        "default=nw=1:nk=1",
        rtsp,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=8).decode().strip()
        # ffprobe often prints two lines (r_frame_rate then avg_frame_rate). Try each.
        for line in (ln.strip() for ln in out.splitlines() if ln.strip()):
            if "/" in line:
                num, den = line.split("/", 1)
                try:
                    num_i = int(num.strip())
                    den_i = int(den.strip() or "1")
                    if den_i:
                        return num_i / den_i
                except Exception:
                    pass
            else:
                try:
                    return float(line)
                except Exception:
                    pass
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Snapshot via ffmpeg (overwrite single file)
# ---------------------------------------------------------------------------
def save_snapshot(ffmpeg: str, rtsp: str, out_path: Path) -> tuple[bool, str | None]:
    """
    Returns (ok, error_message). Tries TCP then UDP.
    """

    def run(transport: str):
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            transport,
            "-i",
            rtsp,
            "-frames:v",
            "1",
            "-q:v",
            "3",
            "-update",
            "1",  # write single image file (no sequence)
            "-y",
            str(out_path),
        ]
        return subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=12,
        )

    try:
        r = run("tcp")
        if r.returncode != 0:
            r2 = run("udp")
            if r2.returncode != 0:
                msg = r2.stderr.decode(errors="ignore")[-200:] or "snapshot failed"
                return False, msg
        return True, None
    except subprocess.TimeoutExpired:
        return False, "snapshot timeout"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="FFmpeg runner (heartbeat + snapshot)")
    p.add_argument("--camera", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--fps", type=float, default=1.0)  # not used here; future detector setting
    p.add_argument("--det_set", default="auto")  # not used here; future detector setting

    p.add_argument("--hb", required=True)  # heartbeat URL
    p.add_argument("--hb_key", required=True)  # shared key header
    p.add_argument("--rtsp", required=True)  # RTSP URL
    p.add_argument("--ffmpeg", default="/usr/bin/ffmpeg")
    p.add_argument("--ffprobe", default="/usr/bin/ffprobe")
    p.add_argument("--snapshots", required=True, help="Directory to write <camera_id>.jpg")

    # timing knobs
    p.add_argument("--hb_interval", type=float, default=10.0,
                   help="Seconds between heartbeats (default 10)")
    p.add_argument("--snapshot_every", type=int, default=3,
                   help="Take a snapshot every N heartbeats (default 3)")

    args, _ = p.parse_known_args()

    cam_id = int(args.camera)
    prof_id = int(args.profile)
    snap_dir = Path(args.snapshots)
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_path = snap_dir / f"{cam_id}.jpg"

    # graceful stop
    running = True

    def _stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    interval = max(1.0, float(args.hb_interval))
    every_snap = max(1, int(args.snapshot_every))

    tick = 0
    next_tick = time.monotonic()

    while running:
        tick += 1
        last_error = ""

        # Lightweight: probe fps quickly
        fps_val = probe_fps(args.ffprobe, args.rtsp) or 0.0

        # Heavier: snapshot every N ticks
        if tick % every_snap == 0:
            ok, err = save_snapshot(args.ffmpeg, args.rtsp, snap_path)
            if not ok and err:
                last_error = err

        # Heartbeat payload
        payload = {
            "camera_id": cam_id,
            "profile_id": prof_id,
            "fps": float(fps_val),
            "detected": 0,
            "matched": 0,
            "latency_ms": 0.0,
            "last_error": (last_error or "")[:200],
        }

        try:
            post(args.hb, payload, args.hb_key, timeout=3.0)
        except Exception:
            # swallow network hiccups; next tick will retry
            pass

        # Fixed schedule (no drift)
        next_tick += interval
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            # overran the interval; re-anchor
            next_tick = time.monotonic()

    # clean exit
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# #!/usr/bin/env python3
# # Make them executable (optional):
# # chmod +x extras/recognize_ffmpeg.py extras/recognize_opencv.py
# # Not needed for our enforcer, because we invoke them like:
# # python /abs/path/extras/recognize_ffmpeg.py …
# # You only need chmod +x if you plan to run them directly as ./recognize_ffmpeg.py using the shebang.
# import argparse, os, signal, sys, time, json, urllib.request, subprocess, shlex, re
# from pathlib import Path
#
#
# def post(url, payload, key, timeout=3):
#     data = json.dumps(payload).encode()
#     req = urllib.request.Request(url, data=data,
#                                  headers={"Content-Type": "application/json", "X-BISK-KEY": key})
#     urllib.request.urlopen(req, timeout=timeout).read()
#
#
# # --- helper: probe fps with ffprobe ---
# def probe_fps(ffprobe, rtsp):
#     cmd = [
#         ffprobe, "-v", "error",
#         "-select_streams", "v:0",
#         "-show_entries", "stream=r_frame_rate",
#         "-of", "default=nw=1:nk=1",
#         rtsp
#     ]
#     try:
#         out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=8).decode().strip()
#         # out like "4/1" or "25/1" or "30000/1001"
#         m = re.match(r"(\d+)\s*/\s*(\d+)", out)
#         if m:
#             num, den = int(m.group(1)), int(m.group(2))
#             if den != 0:
#                 return num / den
#         # sometimes ffprobe prints a plain number
#         try:
#             return float(out)
#         except:
#             return None
#     except Exception:
#         return None
#
#
# # --- replace your measure_fps with this version ---
# def measure_fps(ffmpeg, ffprobe, rtsp, seconds=5):
#     def run(transport):
#         cmd = [
#             ffmpeg, "-hide_banner", "-loglevel", "info", "-stats",
#             "-rtsp_transport", transport,
#             "-i", rtsp, "-an",
#             "-t", str(seconds),
#             "-f", "null", "-"  # some builds lack null; we'll fall back if this fails
#         ]
#         frames, fps, lines = 0, 0.0, []
#         try:
#             proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1)
#             for line in proc.stderr:
#                 line = line.strip()
#                 if line:
#                     lines.append(line);
#                     lines = lines[-8:]
#                 if "frame=" in line:
#                     parts = line.replace("=", " = ").split()
#                     for i, tok in enumerate(parts):
#                         if tok == "frame" and i + 2 < len(parts) and parts[i + 1] == "=":
#                             try:
#                                 frames = int(parts[i + 2]);
#                             except:
#                                 pass
#                         if tok == "fps" and i + 2 < len(parts) and parts[i + 1] == "=":
#                             try:
#                                 fps = float(parts[i + 2]);
#                             except:
#                                 pass
#             rc = proc.wait(timeout=seconds + 5)
#             return rc, (fps if fps > 0 else (frames / seconds if frames > 0 else 0.0)), frames, lines[-4:]
#         except Exception as e:
#             return 8, 0.0, 0, [str(e)]
#
#     rc, fps, frames, tail = run("tcp")
#     if frames == 0:
#         rc2, fps2, frames2, tail2 = run("udp")
#         if frames2 > 0:
#             return fps2, frames2, None
#         # both decode paths failed → try ffprobe
#         pfps = probe_fps(ffprobe, rtsp)
#         if pfps:
#             return pfps, 0, None  # we accept probed rate; not a decode error
#         return 0.0, 0, f"ffmpeg rc={rc2}; {' | '.join(tail2)}"
#     return fps, frames, None
#
#
# # --- keep your snapshot writer but ensure -update 1 is present ---
# def save_snapshot(ffmpeg, rtsp, out_path):
#     def run(transport):
#         cmd = [
#             ffmpeg, "-hide_banner", "-loglevel", "error",
#             "-rtsp_transport", transport,
#             "-i", rtsp,
#             "-frames:v", "1", "-q:v", "3",
#             "-update", "1",  # <-- important
#             "-y", str(out_path)
#         ]
#         return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=12)
#
#     try:
#         r = run("tcp")
#         if r.returncode != 0:
#             r2 = run("udp")
#             if r2.returncode != 0:
#                 return False, (r2.stderr.decode(errors="ignore")[-200:] or "snapshot failed")
#         return True, None
#     except subprocess.TimeoutExpired:
#         return False, "snapshot timeout"
#     except Exception as e:
#         return False, str(e)
#
#
# def main():
#     p = argparse.ArgumentParser(description="FFmpeg runner (decode-only heartbeat + snapshot)")
#     p.add_argument("--camera", required=True)
#     p.add_argument("--profile", required=True)
#     p.add_argument("--fps", type=float, default=1.0)
#     p.add_argument("--det_set", default="auto")
#     p.add_argument("--hb", required=True)
#     p.add_argument("--hb_key", required=True)
#     p.add_argument("--rtsp", required=True)
#     p.add_argument("--ffmpeg", default="/usr/bin/ffmpeg")
#     p.add_argument("--ffprobe", default="/usr/bin/ffprobe")
#     p.add_argument("--snapshots", required=True, help="Directory for snapshots")
#     p.add_argument("--hb_interval", type=float, default=10.0, help="Seconds between heartbeats (default 10)")
#     p.add_argument("--snapshot_every", type=int, default=3, help="Take a snapshot every N heartbeats (default 3)")
#     args, _ = p.parse_known_args()
#
#     running = True
#
#     def _stop(*_):
#         nonlocal running
#         running = False
#
#     signal.signal(signal.SIGTERM, _stop)
#     signal.signal(signal.SIGINT, _stop)
#
#     stop = False
#
#     def handler(sig, frame):
#         nonlocal stop;
#         stop = True
#
#     signal.signal(signal.SIGTERM, handler)
#
#     cam_id = int(args.camera)
#     prof_id = int(args.profile)
#     snap_dir = Path(args.snapshots)
#     snap_dir.mkdir(parents=True, exist_ok=True)
#     snap_path = snap_dir / f"{cam_id}.jpg"
#
#     beat_i = 0
#     last_error = ""
#
#     while not stop:
#         # measure actual fps over ~5s
#         fps_measured, frames, err = measure_fps(args.ffmpeg, args.ffprobe, args.rtsp, seconds=5)
#         if err:
#             last_error = err
#         else:
#             last_error = ""
#
#         # every ~15s, try snapshot
#         if beat_i % 3 == 0:
#             ok, serr = save_snapshot(args.ffmpeg, args.rtsp, snap_path)
#             if serr and not last_error:
#                 last_error = serr
#
#         # send heartbeat
#         payload = {
#             "camera_id": cam_id,
#             "profile_id": prof_id,
#             "fps": float(fps_measured),
#             "detected": 0,
#             "matched": 0,
#             "latency_ms": 0.0,
#             "last_error": last_error[:200] if last_error else "",
#         }
#         try:
#             post(args.hb, payload, args.hb_key, timeout=3)
#         except Exception:
#             # swallow; next loop will try again
#             pass
#
#         beat_i += 1
#         # small idle between cycles
#         for _ in range(10):
#             if stop: break
#             time.sleep(0.5)
#
#     sys.exit(0)
#
#
# if __name__ == "__main__":
#     main()
