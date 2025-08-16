# apps/scheduler/management/commands/enforce_schedules.py
import os, sys, signal, subprocess
from django.core.management.base import BaseCommand
from apps.scheduler.services import enforce_schedules


def _in_window(now_t, s, e):
    # same-day or overnight
    return (s <= e and s <= now_t < e) or (s > e and (now_t >= s or now_t < e))


def _start(camera, profile):
    py = sys.executable
    script = "extras/recognize_ffmpeg.py" if profile.script_type == 1 else "extras/recognize_opencv.py"
    args = [py, script, "--camera", str(camera.id), "--fps", str(profile.fps), "--det_set", str(profile.detection_set)]
    for k, v in (profile.extra_args or {}).items(): args += [f"--{k}", str(v)]
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

