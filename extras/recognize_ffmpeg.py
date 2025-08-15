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
import os


def ffmpeg_cmd(ffmpeg_bin, rtsp_url, out_jpg, args):
    """
    Build a single-process FFmpeg command that overwrites one JPG
    every N seconds using -update 1. Uses transport + hwaccel knobs.
    """
    cmd = [
        ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "warning",
    ]

    # RTSP transport (auto|tcp|udp)
    if getattr(args, "rtsp_transport", None) and args.rtsp_transport != "auto":
        cmd += ["-rtsp_transport", args.rtsp_transport]

    # hardware decode (nvdec => -hwaccel cuda). No index switch required for decode.
    if getattr(args, "hwaccel", None) == "nvdec":
        cmd += ["-hwaccel", "cuda"]

    # input
    cmd += ["-i", rtsp_url]

    # output: overwrite same file every snapshot_every seconds
    snap_every = max(1, int(getattr(args, "snapshot_every", 3)))
    cmd += [
        "-vf", f"fps=1/{snap_every}",
        "-q:v", "2",
        "-f", "image2",
        "-update", "1",
        "-y", out_jpg,
    ]
    return cmd


def spawn_ffmpeg(args, rtsp_url, out_jpg):
    """
    Start FFmpeg in the SAME process group as the runner (important!).
    Do NOT use preexec_fn=os.setsid here, so killpg from the scheduler
    will terminate both runner and ffmpeg together.
    """
    cmd = ffmpeg_cmd(args.ffmpeg, rtsp_url, out_jpg, args)
    return subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,   # <-- no pipes; we are not reading logs
        stderr=subprocess.DEVNULL,
        close_fds=True,
        # NOTE: no preexec_fn=os.setsid here on purpose
    )


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
    p.add_argument("--hb_interval", type=int, default=10, help="Seconds between heartbeats (default 10)")
    p.add_argument("--snapshot_every", type=int, default=3, help="Take a snapshot every N heartbeats (default 3)")
    p.add_argument("--rtsp_transport", choices=["auto", "tcp", "udp"], default="auto")
    p.add_argument("--hwaccel", choices=["none", "nvdec"], default="none")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--gpu_index", type=int, default=0)

    args, _ = p.parse_known_args()

    cam_id = int(args.camera)
    prof_id = int(args.profile)
    snap_dir = Path(args.snapshots)
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_path = snap_dir / f"{cam_id}.jpg"

    # Start a single ffmpeg process that overwrites <camera_id>.jpg
    ff_proc = spawn_ffmpeg(args, args.rtsp, str(snap_path))

    # respawn state
    backoff = 1
    last_error_msg = ""

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
        # Reap / respawn FFmpeg if it died
        rc = ff_proc.poll()
        if rc is not None:
            try:
                ff_proc.wait(timeout=0)  # reap zombie if any
            except Exception:
                pass
            last_error_msg = f"ffmpeg exited rc={rc}"
            time.sleep(backoff)  # simple backoff
            backoff = min(10, backoff * 2)  # cap at 10s
            ff_proc = spawn_ffmpeg(args, args.rtsp, str(snap_path))
        else:
            backoff = 1

        tick += 1

        # Lightweight: probe fps quickly
        fps_val = probe_fps(args.ffprobe, args.rtsp) or 0.0

        # Heartbeat payload
        payload = {
            "camera_id": cam_id,
            "profile_id": prof_id,
            "fps": float(fps_val),
            "detected": 0,
            "matched": 0,
            "latency_ms": 0.0,
            "pid": os.getpid(),
            "last_error": (last_error_msg or "")[:200],
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

    try:
        if ff_proc and ff_proc.poll() is None:
            ff_proc.terminate()
            try:
                ff_proc.wait(timeout=3)
            except Exception:
                try:
                    ff_proc.kill()
                except Exception:
                    pass
    except Exception:
        pass

    # clean exit
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
