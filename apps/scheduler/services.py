# apps/scheduler/services.py
import os, sys, signal, subprocess
from dataclasses import dataclass
from typing import Iterable, Optional
from django.utils import timezone
from .models import SchedulePolicy, RunningProcess
from pathlib import Path
from django.conf import settings
import psutil

hb_url = getattr(settings, "RUNNER_HEARTBEAT_URL", "http://127.0.0.1:8000/api/runner/heartbeat/")
hb_key = getattr(settings, "RUNNER_HEARTBEAT_KEY", "dev-key-change-me")


def _pid_alive(pid: int) -> bool:
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _in_window(now_t, s, e):
    # same-day or overnight
    # return (s <= e and s <= now_t < e) or (s > e and (now_t >= s or now_t < e))
    # Full-day window
    if s == e:
        return True
    # Same-day
    if s < e:
        return s <= now_t < e
    # Overnight (spans midnight)
    return (now_t >= s) or (now_t < e)


def _start(camera, profile):
    py = sys.executable
    script_name = "recognize_ffmpeg.py" if profile.script_type == 1 else "recognize_opencv.py"
    script = str(Path(settings.BASE_DIR) / "extras" / script_name)

    hb_url = getattr(settings, "RUNNER_HEARTBEAT_URL", "http://127.0.0.1:8000/api/runner/heartbeat/")
    hb_key = getattr(settings, "RUNNER_HEARTBEAT_KEY", "dev-key-change-me")

    args = [
        py, script,
        "--camera", str(camera.id),
        "--profile", str(profile.id),
        "--fps", str(profile.fps),
        "--det_set", str(profile.detection_set),
        "--hb", hb_url,
        "--hb_key", hb_key,
        "--rtsp", camera.rtsp_url,
        "--ffmpeg", settings.FFMPEG_PATH,
        "--ffprobe", settings.FFPROBE_PATH,
        "--snapshots", str(settings.SNAPSHOT_DIR),
        # NEW: drive runner timing from settings
        "--hb_interval", str(getattr(settings, "HEARTBEAT_INTERVAL_SEC", 10)),
        "--snapshot_every", str(getattr(settings, "HEARTBEAT_SNAPSHOT_EVERY", 3)),
    ]
    for k, v in (profile.extra_args or {}).items():
        args += [f"--{k}", str(v)]

    proc = subprocess.Popen(
        args,
        preexec_fn=os.setsid, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True
    )
    return proc.pid


def _stop(pid: int):
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


@dataclass
class EnforceResult:
    started: list
    stopped: list
    desired_count: int
    running_count: int


def enforce_schedules(policies: Optional[Iterable[SchedulePolicy]] = None) -> EnforceResult:
    """
    Idempotent enforcer. If `policies` passed, only those policies are considered;
    otherwise all enabled policies.
    """
    now_local = timezone.localtime()
    desired = set()

    qs = (policies if policies is not None
          else SchedulePolicy.objects.filter(is_enabled=True).prefetch_related(
        "cameras", "windows", "exceptions", "windows__profile"
    ))

    # Build desired state
    for p in qs:
        exc = {x.date: x for x in p.exceptions.all()}.get(now_local.date())
        if exc and exc.mode == "off":
            continue
        for w in p.windows.filter(day_of_week=now_local.weekday()):
            if _in_window(now_local.timetz().replace(tzinfo=None), w.start_time, w.end_time):
                for cam in p.cameras.all():
                    desired.add((cam.id, w.profile_id))

    # running = {(r.camera_id, r.profile_id): r for r in RunningProcess.objects.select_related("camera", "profile")}
    running = {}
    for r in RunningProcess.objects.select_related("camera", "profile"):
        if _pid_alive(r.pid):
            running[(r.camera_id, r.profile_id)] = r
        else:
            if r.status != "dead":
                r.status = "dead"
                r.save(update_fields=["status"])

    # --- ADD THIS BLOCK: treat 'stale' (no heartbeat) as not-running ---
    def _stale_secs(r):
        if not r.last_heartbeat:
            return 10 ** 9
        return (timezone.now() - r.last_heartbeat).total_seconds()

    for pair, r in list(running.items()):
        if _stale_secs(r) > 60:  # tweak threshold as you like
            try:
                _stop(r.pid)  # SIGTERM process group
            except Exception:
                pass
            r.status = "stopping"
            r.save(update_fields=["status"])
            running.pop(pair, None)  # remove so it will be restarted if desired

    started, stopped = [], []

    # Start missing
    for cam_id, prof_id in desired - set(running.keys()):
        pol = (qs if policies is not None else SchedulePolicy.objects).filter(
            cameras__id=cam_id, windows__profile_id=prof_id
        ).first()
        camera = pol.cameras.get(id=cam_id)
        profile = next(ww.profile for ww in pol.windows.all() if ww.profile_id == prof_id)
        pid = _start(camera, profile)

        # create one row for this spawn; avoid duplicates if the same pid was recorded
        row, created = RunningProcess.objects.get_or_create(
            camera=camera, profile=profile, pid=pid,
            defaults={"status": "running"}  # optional; status will be updated by heartbeat
        )
        # ensure only the newest row for this (camera, profile) remains active
        RunningProcess.objects.filter(camera=camera, profile=profile) \
            .exclude(id=row.id).update(status="dead")

        # if you keep a 'started' list for reporting, record it
        # started.append((camera.id, profile.id))

        row = RunningProcess.objects.create(camera=camera, profile=profile, pid=pid)
        # mark any older rows for same pair as dead so they don’t get “revived” by heartbeats
        RunningProcess.objects.filter(camera=camera, profile=profile).exclude(id=row.id).update(status="dead")
        started.append((camera.name, profile.name, pid))

    # Stop extras
    for pair, r in running.items():
        if pair not in desired:
            _stop(r.pid)
            r.status = "stopping"
            r.save(update_fields=["status"])
            stopped.append((r.camera.name, r.profile.name, r.pid))

    return EnforceResult(started=started, stopped=stopped, desired_count=len(desired), running_count=len(running))
