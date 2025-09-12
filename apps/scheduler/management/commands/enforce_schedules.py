# apps/scheduler/management/commands/enforce_schedules.py
import os, sys, signal, subprocess
from django.core.management.base import BaseCommand
from apps.scheduler.services import enforce_schedules


def _in_window(now_t, s, e):
    # same-day or overnight
    return (s <= e and s <= now_t < e) or (s > e and (now_t >= s or now_t < e))


def _start(camera, profile):
    py = sys.executable
    # script = "extras/recognize_ffmpeg.py" if profile.script_type == 1 else "extras/recognize_opencv.py"
    # args = [py, script, "--camera", str(camera.id), "--fps", str(profile.fps), "--det_set", str(profile.detection_set)]
    # for k, v in (profile.extra_args or {}).items(): args += [f"--{k}", str(v)]
    # Map script_type → runner
    if profile.script_type == 1:
        script = "extras/recognize_runner_all_ffmpeg.py"  # the new, high-quality pipeline
    else:
        script = "extras/recognize_opencv.py"  # optional legacy path

    # Resolve required args for the all-FFmpeg runner
    # Assumptions:
    # - camera.url holds the RTSP URL
    # - profile.extra_args may contain hb, hb_key, snapshots, rtsp_transport, hwaccel, device, threshold, etc.
    rtsp_url = getattr(camera, "url", None) or (profile.extra_args or {}).get("rtsp")
    hb = (profile.extra_args or {}).get("hb") or "http://127.0.0.1:8000/api/runner/heartbeat/"
    hb_key = (profile.extra_args or {}).get("hb_key") or "dev-key-change-me"
    snapshots = (profile.extra_args or {}).get("snapshots") or "media/snapshots"
    rtsp_tx = (profile.extra_args or {}).get("rtsp_transport") or "auto"
    hwaccel = (profile.extra_args or {}).get("hwaccel") or "none"  # "nvdec" if you want GPU decode
    device = (profile.extra_args or {}).get("device") or "auto"
    gpu_index = (profile.extra_args or {}).get("gpu_index") or 0
    threshold = (profile.extra_args or {}).get("threshold")  # None → pull from RecognitionSettings
    fps_target = str(profile.fps)

    args = [
        py, script,
        "--camera", str(camera.id),
        "--profile", str(profile.id),
        "--rtsp", str(rtsp_url),
        "--hb", str(hb),
        "--hb_key", str(hb_key),
        "--snapshots", str(snapshots),
        "--rtsp_transport", str(rtsp_tx),
        "--hwaccel", str(hwaccel),
        "--device", str(device),
        "--gpu_index", str(gpu_index),
        "--fps", fps_target,
    ]
    if threshold is not None:
        args += ["--threshold", str(threshold)]

    # still allow ad-hoc overrides (e.g., ffmpeg path, snapshot_every, hb_interval…)
    for k, v in (profile.extra_args or {}).items():
        if k in {"rtsp", "hb", "hb_key", "snapshots", "rtsp_transport", "hwaccel", "device", "gpu_index", "threshold"}:
            continue
        args += [f"--{k}", str(v)]
    proc = subprocess.Popen(args, preexec_fn=os.setsid, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            close_fds=True)
    return proc.pid


def _stop(pid):
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


class Command(BaseCommand):
    help = "Align running processes with DB schedules (safe every minute)."

    def handle(self, *args, **opts):
        result = enforce_schedules()
        self.stdout.write(self.style.SUCCESS(
            f"Enforced. Started: {len(result.started)}, Stopped: {len(result.stopped)}, "
            f"Desired={result.desired_count}, Running(before)={result.running_count}"
        ))
