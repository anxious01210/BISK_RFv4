#!/usr/bin/env python3
"""
FFmpeg runner (Type 1) — lightweight heartbeat + periodic snapshot.

- FPS is probed via ffprobe (fast & robust).
- Snapshot uses ffmpeg with -update 1 so we always overwrite <camera_id>.jpg.
- Heartbeats are sent at a fixed cadence (default 10s) using a monotonic clock.
- Snapshot is taken every N heartbeats (default every 3).

Invoke: python /abs/path/extras/recognize_ffmpeg.py --camera 1 --profile 1 ...
"""

import argparse, json, signal, subprocess, time, urllib.request, threading, os, re
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv4.settings")
import django

django.setup()
# --- Resource resolver imports (after django.setup()) ---
from apps.scheduler.resources import resolve_effective
from apps.cameras.models import Camera


# --- RESOURCES ENFORCEMENT (paste near the top of main, after Django setup) ---


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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,  # <-- read stderr to map errors
        close_fds=True,
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
    Return a float fps (prefer avg_frame_rate over r_frame_rate) or None.
    We ask ffprobe to print keys so we can reliably pick avg_frame_rate.
    """
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate",
        "-of", "default=nw=1",   # keep keys; don't use nk=1
        rtsp,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=8).decode().strip()
        got = {}
        for ln in (ln.strip() for ln in out.splitlines() if ln.strip()):
            if "=" not in ln:
                continue
            k, v = ln.split("=", 1)
            v = v.strip()
            if v.lower() == "n/a":
                continue
            # parse like "8/1", "30000/1001", or "25"
            try:
                if "/" in v:
                    num, den = v.split("/", 1)
                    num_i = float(num.strip())
                    den_i = float(den.strip() or "1")
                    if den_i != 0:
                        got[k] = num_i / den_i
                else:
                    got[k] = float(v)
            except Exception:
                pass
        # Prefer avg_frame_rate; fall back to r_frame_rate
        if "avg_frame_rate" in got and got["avg_frame_rate"] > 0:
            return got["avg_frame_rate"]
        if "r_frame_rate" in got and got["r_frame_rate"] > 0:
            return got["r_frame_rate"]
        return None
    except Exception:
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
    p.add_argument("--hb_interval", type=int, default=None, help="Seconds between heartbeats (default 10)")
    p.add_argument("--snapshot_every", type=int, default=None, help="Take a snapshot every N heartbeats (default 3)")
    p.add_argument("--rtsp_transport", choices=["auto", "tcp", "udp"], default="auto")
    p.add_argument("--hwaccel", choices=["none", "nvdec"], default="none")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--gpu_index", type=int, default=0)

    args, _ = p.parse_known_args()

    #

    # Resolve effective resources by camera ID
    eff = resolve_effective(camera_id=args.camera)

    # If you need the Camera instance later, you can fetch it here:
    try:
        cam = Camera.objects.get(pk=args.camera)
    except Camera.DoesNotExist:
        cam = None

    # GPU visibility must be set before importing GPU frameworks
    if eff.gpu_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = eff.gpu_visible_devices

    # CPU niceness / affinity (best-effort)
    try:
        import psutil
        p = psutil.Process(os.getpid())
        if eff.cpu_nice is not None:
            p.nice(eff.cpu_nice)
        if eff.cpu_affinity:
            p.cpu_affinity(eff.cpu_affinity)
    except Exception:
        # non-fatal: permission or platform might block it
        pass

    # Decide the frame cap we’ll honor in the loop
    # Prefer per-camera override → global default → CLI --fps → fallback 6
    TARGET_FPS = (
            (eff.max_fps if eff.max_fps is not None else None)
            or (args.fps if hasattr(args, "fps") and args.fps else None)
            or 6
    )

    # Optional: cap detector input size if your pipeline supports it
    DET_SET_MAX = eff.det_set_max  # e.g., "1600" or None

    # Optional: GPU soft util target (you can use it to tune sleep dynamically)
    GPU_UTIL_TARGET = eff.gpu_target_util_percent or None

    # Optional: CPU quota target (soft) for extra sleeps under high load
    CPU_QUOTA = eff.cpu_quota_percent or 100
    # --- END RESOURCES ENFORCEMENT ---

    # Safe to import GPU frameworks now
    try:
        import torch
        if eff.gpu_memory_fraction:
            # Maps to device 0 within CUDA_VISIBLE_DEVICES space
            torch.cuda.set_per_process_memory_fraction(eff.gpu_memory_fraction, device=0)
    except Exception:
        pass

    # If you use ONNXRuntime-GPU or InsightFace, import them here too
    # import onnxruntime as ort
    # from insightface.app import FaceAnalysis

    # 2) after args parsed:
    hb_interval = int(os.getenv("BISK_HB_INTERVAL", args.hb_interval or 10))

    # Derive snapshot_every from TARGET_FPS if available.
    # TARGET_FPS was resolved from resource settings earlier.
    # formula: snapshots per HB tick = hb_interval / fps  => every N seconds we write one frame
    if TARGET_FPS and TARGET_FPS > 0:
        target_snap_every = max(1, int(round(hb_interval / float(TARGET_FPS))))
    else:
        target_snap_every = 3  # sane fallback

    # Precedence: Camera/Global resource cap (TARGET_FPS) takes priority over CLI, but you can flip this.
    snapshot_every = int(os.getenv(
        "BISK_SNAPSHOT_EVERY",
        target_snap_every if args.snapshot_every is None else args.snapshot_every
    ))

    cam_id = int(args.camera)
    prof_id = int(args.profile)
    snap_dir = Path(args.snapshots)
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_path = snap_dir / f"{cam_id}.jpg"

    # IMPORTANT: pass the resolved snapshot_every to ffmpeg_cmd via args
    setattr(args, "snapshot_every", snapshot_every)

    # Start a single ffmpeg process that overwrites .jpg
    ff_proc = spawn_ffmpeg(args, args.rtsp, str(snap_path))

    # respawn state
    backoff = 1
    last_error_msg = ""
    last_error_seen_at = 0.0  # monotonic timestamp of most recent mapped error

    def _map_err(line: str) -> str | None:
        L = line.lower()
        if "401" in L and ("unauthorized" in L or "authorization" in L): return "401 unauthorized"
        if "describe" in L and "404" in L: return "rtsp describe 404"
        if "connection refused" in L: return "connection refused"
        if "timed out" in L or "timeout" in L: return "connection timeout"
        if "unrecognized option" in L or "option not found" in L: return "ffmpeg option not found"
        if "unsupported transport" in L or "461" in L: return "transport mismatch"
        if "invalid data found" in L: return "invalid input data"
        if "server returned 5" in L: return "server 5xx error"
        return None

    def _stderr_reader():
        nonlocal last_error_msg, last_error_seen_at
        if not ff_proc.stderr:
            return
        for raw in iter(ff_proc.stderr.readline, b""):
            try:
                line = raw.decode("utf-8", "ignore").strip()
            except Exception:
                continue
            mapped = _map_err(line)
            if mapped:
                last_error_msg = mapped
                last_error_seen_at = time.monotonic()

    t = threading.Thread(target=_stderr_reader, daemon=True)
    t.start()

    # graceful stop
    running = True

    def _stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    interval = max(1.0, float(hb_interval))
    every_snap = max(1, int(snapshot_every))

    tick = 0
    next_tick = time.monotonic()
    # throttle ffprobe to every N ticks
    probe_every = 3
    last_probed_fps = 0.0

    while running:
        # Reap / respawn FFmpeg if it died
        rc = ff_proc.poll()
        if rc is not None:
            try:
                ff_proc.wait(timeout=0)  # reap zombie if any
            except Exception:
                pass
            last_error_msg = last_error_msg or f"ffmpeg exited rc={rc}"
            time.sleep(backoff)  # simple backoff
            backoff = min(10, backoff * 2)  # cap at 10s
            ff_proc = spawn_ffmpeg(args, args.rtsp, str(snap_path))
            # restart reader
            t = threading.Thread(target=_stderr_reader, daemon=True)
            t.start()
        else:
            backoff = 1

        tick += 1

        # Lightweight: probe fps (throttled)
        if tick % probe_every == 0:
            last_probed_fps = probe_fps(args.ffprobe, args.rtsp) or last_probed_fps
        fps_val = last_probed_fps

        # Clear stale error if stream seems healthy for a while
        if last_error_msg and (time.monotonic() - last_error_seen_at) > max(2 * interval, 60):
            last_error_msg = ""

        # Compute camera vs processed rates
        camera_fps_val = float(fps_val or 0.0)
        target_fps_val = float(TARGET_FPS or 0.0)

        if target_fps_val > 0.0 and camera_fps_val > 0.0:
            processed_fps_val = min(camera_fps_val, target_fps_val)
        elif target_fps_val > 0.0:
            processed_fps_val = target_fps_val
        else:
            processed_fps_val = camera_fps_val

        # Heartbeat payload
        payload = {
            "camera_id": cam_id,
            "profile_id": prof_id,

            # legacy 'fps' stays = camera fps for back-compat
            "fps": camera_fps_val,

            # new fields used by the UI:
            "camera_fps": camera_fps_val,
            "processed_fps": processed_fps_val,

            "detected": 0,
            "matched": 0,
            "latency_ms": 0.0,
            "pid": os.getpid(),
            "last_error": (last_error_msg or "")[:200],
            "target_fps": target_fps_val,
            "snapshot_every": int(snapshot_every),
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
# mirror test Sun Aug 17 08:26:22 PM +03 2025
