# apps/scheduler/management/commands/enforce_schedules.py
import os, sys, signal, subprocess
from django.core.management.base import BaseCommand
from django.utils import timezone
from apps.scheduler.models import SchedulePolicy, RunningProcess
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

# class Command(BaseCommand):
#     help = "Align running processes with DB schedules (safe every minute)."
#
#     def handle(self, *args, **opts):
#         now_local = timezone.localtime()  # aware, in Asia/Baghdad
#         desired = set()
#
#         policies = (SchedulePolicy.objects.filter(is_enabled=True)
#                     .prefetch_related("cameras", "windows", "exceptions", "windows__profile"))
#
#         # Compute desired (respect OFF exceptions; ON/Window can be added later)
#         for p in policies:
#             exc = {x.date: x for x in p.exceptions.all()}.get(now_local.date())
#             if exc and exc.mode == "off":
#                 continue
#             for w in p.windows.filter(day_of_week=now_local.weekday()):
#                 if _in_window(now_local.timetz().replace(tzinfo=None), w.start_time, w.end_time):
#                     for cam in p.cameras.all():
#                         desired.add((cam.id, w.profile_id))
#
#         running = {(r.camera_id, r.profile_id): r for r in RunningProcess.objects.select_related("camera", "profile")}
#
#         # start missing
#         for cam_id, prof_id in desired - set(running.keys()):
#             pol = policies.filter(cameras__id=cam_id, windows__profile_id=prof_id).first()
#             camera = pol.cameras.get(id=cam_id)
#             profile = next(ww.profile for ww in pol.windows.all() if ww.profile_id == prof_id)
#             pid = _start(camera, profile)
#             RunningProcess.objects.create(camera=camera, profile=profile, pid=pid)
#
#         # stop extras
#         for pair, r in running.items():
#             if pair not in desired:
#                 _stop(r.pid)
#                 r.status = "stopping";
#                 r.save(update_fields=["status"])
