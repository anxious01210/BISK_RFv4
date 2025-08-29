# apps/scheduler/services/enforcer.py
import os, sys, signal, subprocess, time, shlex, psutil, re
from dataclasses import dataclass
from typing import Iterable, Optional
from django.utils import timezone
from apps.scheduler.models import SchedulePolicy, RunningProcess, StreamProfile
from pathlib import Path
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
from django.db import transaction, IntegrityError
from django.core.cache import cache
from apps.cameras.models import Camera
from django.db.models import Max, Q
from datetime import timedelta
from django.conf import settings


# How long a row must be Offline before we consider pruning (minutes).
# Prefer minutes-based; fall back to hours for BC; default 6h.
PRUNE_MINUTES = int(getattr(settings, "RUNPROC_PRUNE_OFFLINE_MINUTES",
                            getattr(settings, "RUNPROC_PRUNE_OFFLINE_HOURS", 6) * 60))

# What "Offline" means in seconds (keep consistent with your admin thresholds)
OFFLINE_SECS = int(getattr(settings, "HEARTBEAT_OFFLINE_SEC", 120))

# hb_url = getattr(settings, "RUNNER_HEARTBEAT_URL", "http://127.0.0.1:8000/api/runner/heartbeat/")
# hb_key = getattr(settings, "RUNNER_HEARTBEAT_KEY", "dev-key-change-me")

# heartbeat freshness (keep same numbers you show in admin)
STALE_SECS = getattr(settings, "HEARTBEAT_STALE_SEC", 45)

# --- defaults (place near other module constants) ---
DEFAULTS = {
    "rtsp_transport": getattr(settings, "DEFAULT_RTSP_TRANSPORT", "auto"),
    "hwaccel": getattr(settings, "DEFAULT_HWACCEL", "none"),
    "device": getattr(settings, "DEFAULT_DEVICE", "cpu"),
    "gpu_index": getattr(settings, "DEFAULT_GPU_INDEX", 0),
    "hb_interval": getattr(settings, "DEFAULT_HB_INTERVAL", 10),  # seconds
    "snapshot_every": getattr(settings, "DEFAULT_SNAPSHOT_EVERY", 3),  # heartbeats
    "nice": getattr(settings, "DEFAULT_RUNNER_NICE", None),
    "cpu_affinity": getattr(settings, "DEFAULT_RUNNER_CPU_AFFINITY", ""),  # e.g. "0,1"
    "fps": getattr(settings, "DEFAULT_FPS", 6),
    "detection_set": getattr(settings, "DEFAULT_DET_SET", "auto"),
}


def _val(v):
    return None if v in (None, "", []) else v


def resolve_knob(camera, profile, field):
    """Profile-first unless camera.prefer_camera_over_profile=True."""
    prof = _val(getattr(profile, field, None)) if profile else None
    cam = _val(getattr(camera, field, None))
    default = DEFAULTS.get(field)
    camera_first = bool(getattr(camera, "prefer_camera_over_profile", False))
    if camera_first:
        return cam if cam is not None else (prof if prof is not None else default)
    return prof if prof is not None else (cam if cam is not None else default)


def _policy_is_off_now(policy, now_local):
    """
    Returns True if any 'off' exception covers now.
    Supports either date-only or datetime-range exceptions.
    Adjust the related name 'exceptions' if yours differs.
    """
    try:
        ex_qs = policy.exceptions.filter(mode="off")
    except Exception:
        # if related name is different, try the default
        ex_qs = getattr(policy, "scheduleexception_set", None)
        if ex_qs is None:
            return False
        ex_qs = ex_qs.filter(mode="off")

    # date-only exceptions (today off)
    try:
        if ex_qs.filter(date=now_local.date()).exists():
            return True
    except Exception:
        pass

    # datetime-range exceptions
    try:
        if ex_qs.filter(start_at__lte=now_local, end_at__gte=now_local).exists():
            return True
    except Exception:
        pass

    return False


def _reap_if_child(pid: int) -> None:
    """
    Reap 'pid' if we're its parent, to clear <defunct> zombies.
    Safe to call even if we're not the parent.
    """
    try:
        # POSIX: will only succeed if we're the parent
        import os
        os.waitpid(pid, os.WNOHANG)
    except Exception:
        # Fallback / no-op if not our child
        pass


def _pid_alive(pid: int) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        p = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False

    # Must be running and not a zombie
    try:
        st = p.status()
        if st == psutil.STATUS_ZOMBIE:
            return False
    except psutil.Error:
        # If status can't be read but process object exists, keep checking
        pass

    if not p.is_running():
        return False

    # Extra safety: only consider it "our" runner if cmdline includes recognize_ffmpeg.py
    # Below is the fix of double-recognize .py Processes.
    try:
        cmd = " ".join(p.cmdline())
        if "recognize_runner_all_ffmpeg.py" not in cmd and "recognize_runner_ffmpeg.py" not in cmd:
            return False
    except psutil.Error:
        # If we can't read cmdline, err on the side of "not alive"
        return False

    return True


# TODO Test
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
    """
    Start a runner for (camera, profile).
    Returns (pid, cmdline_str).
    """

    py = sys.executable
    runner = Path(settings.BASE_DIR) / "extras" / "recognize_ffmpeg.py"

    hb_interval = int(resolve_knob(camera, profile, "hb_interval") or 10)
    snapshot_every = int(resolve_knob(camera, profile, "snapshot_every") or 3)

    rtsp_transport = resolve_knob(camera, profile, "rtsp_transport") or "auto"  # auto/tcp/udp
    hwaccel = resolve_knob(camera, profile, "hwaccel") or "none"  # none/nvdec
    device = resolve_knob(camera, profile, "device") or "cpu"  # cpu/cuda
    gpu_index = int(resolve_knob(camera, profile, "gpu_index") or 0)

    affinity_csv = resolve_knob(camera, profile, "cpu_affinity") or ""
    cpu_affinity_s = affinity_csv.strip()
    nice_raw = resolve_knob(camera, profile, "nice")
    nice_value = int(nice_raw) if (nice_raw is not None and str(nice_raw) != "") else 0

    # Still profile-driven (not affected by the toggle)
    fps = int(getattr(profile, "fps", 6) or 6)
    # Prefer detection_set; fall back to det_set for backward-compat
    det_set = str(getattr(profile, "detection_set", getattr(profile, "det_set", "auto")) or "auto")

    # Select which Python runner script to launch
    impl = getattr(settings, "RUNNER_IMPL", "ffmpeg_all")
    runner = {
        "ffmpeg_all": getattr(settings, "RUNNER_SCRIPT_ALL", BASE_DIR / "extras" / "recognize_runner_all_ffmpeg.py"),
        "ffmpeg_one": getattr(settings, "RUNNER_SCRIPT_ONE", BASE_DIR / "extras" / "recognize_runner_ffmpeg.py"),
    }.get(impl, getattr(settings, "RUNNER_SCRIPT_ALL", BASE_DIR / "extras" / "recognize_runner_all_ffmpeg.py"))

    # build the runner command (your runner builds/executes ffmpeg internally)
    rtsp_cli = []
    if rtsp_transport and rtsp_transport != "auto":
        rtsp_cli = ["--rtsp_transport", rtsp_transport]
    cmd = [
        str(py), str(runner),
        "--camera", str(camera.id),
        "--profile", str(profile.id),
        "--fps", str(fps),
        "--det_set", det_set,
        "--hb", settings.RUNNER_HEARTBEAT_URL,
        "--hb_key", settings.RUNNER_HEARTBEAT_KEY,
        "--rtsp", camera.rtsp_url,  # keep your current field
        "--ffmpeg", settings.FFMPEG_PATH,
        "--ffprobe", settings.FFPROBE_PATH,
        "--snapshots", str(settings.SNAPSHOT_DIR),
        "--hb_interval", str(hb_interval),
        "--snapshot_every", str(snapshot_every),
        "--hwaccel", hwaccel,
        "--device", device,
        "--gpu_index", str(gpu_index),
    ] + rtsp_cli

    # Build env for the runner
    env = os.environ.copy()
    env["BISK_HB_INTERVAL"] = str(hb_interval)  # already computed above
    env["BISK_SNAPSHOT_EVERY"] = str(snapshot_every)  # already computed above

    # spawn in its own session so killpg() works
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
        close_fds=True,
        env=env,
    )

    # set affinity / nice for the runner (ffmpeg inherits)
    try:
        proc = psutil.Process(p.pid)
        if cpu_affinity_s:
            cores = [int(x) for x in cpu_affinity_s.split(",") if x.strip().isdigit()]
            if cores:
                proc.cpu_affinity(cores)
        if nice_value:
            proc.nice(nice_value)
    except Exception:
        pass

    # Helpers to mask any credentials in RTSP URLs before persisting
    def _mask_url(s: str) -> str:
        # rtsp://user:pass@host -> rtsp://user:***@host
        return re.sub(r'(rtsp://[^:@\s]+:)[^@/\s]+(@)', r'\1***\2', s)

    def _mask_cmdline(argv):
        return " ".join(shlex.quote(_mask_url(x)) for x in argv)

    # return PID and the masked command we ran (for audit/UI)
    return p.pid, _mask_cmdline(cmd)



def _stop(pid: int, deadline: float = 3.0) -> bool:
    """
    Stop a runner: SIGTERM the process group, wait up to `deadline`,
    then SIGKILL if still alive. Reap if child. Return True if gone.
    """
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return True

    # soft stop
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        _reap_if_child(pid)
        return True

    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline:
        if not _pid_alive(pid):
            _reap_if_child(pid)
            return True
        time.sleep(0.1)

    # hard stop
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        _reap_if_child(pid)
        return True

    time.sleep(0.1)
    gone = not _pid_alive(pid)
    _reap_if_child(pid)
    return gone


def _kill_stray_runners_for_camera(cam_id: int) -> int:
    """
    Kill any recognize_ffmpeg.py processes for this camera that
    are NOT represented by a RunningProcess row (strays).
    Returns number of PIDs signalled.
    """
    killed = 0
    try:
        for p in psutil.process_iter(["pid", "cmdline"]):
            info = p.info
            cmd = info.get("cmdline") or []
            if not cmd:
                continue
            # Look for our runner script
            joined = " ".join(cmd)
            if ("recognize_ffmpeg.py" not in joined) and ("recognize_runner_all_ffmpeg.py" not in joined):
                continue

            # Match --camera <id>
            try:
                idx = cmd.index("--camera")
                cam_val = cmd[idx + 1]
            except Exception:
                continue
            if str(cam_val) != str(cam_id):
                continue

            # If this PID has no DB row, it's a stray → kill its group
            if not RunningProcess.objects.filter(pid=info["pid"]).exists():
                try:
                    pgid = os.getpgid(info["pid"])
                    os.killpg(pgid, signal.SIGTERM)
                    time.sleep(0.3)
                    os.killpg(pgid, signal.SIGKILL)
                    _reap_if_child(info["pid"])
                    killed += 1
                except Exception:
                    pass
    except Exception:
        pass
    return killed


def _prune_long_dead(now):
    """
    Delete old RunningProcess rows that:
      - are older than PRUNE_MINUTES,
      - have last_heartbeat older than OFFLINE_SECS,
      - are NOT the newest row for their (camera, profile),
      - and do NOT have a live OS process.
    Returns number of rows deleted.
    """
    off_cut = now - timedelta(seconds=OFFLINE_SECS)
    prune_cut = now - timedelta(minutes=PRUNE_MINUTES)

    # Keep the single newest row per (camera, profile)
    latest_ids = list(
        RunningProcess.objects
        .values("camera_id", "profile_id")
        .annotate(latest_id=Max("id"))
        .values_list("latest_id", flat=True)
    )

    # Candidates to prune
    qs = (RunningProcess.objects
          .filter(started_at__lt=prune_cut, last_heartbeat__lt=off_cut)
          .exclude(id__in=latest_ids)
          .only("id", "pid"))

    to_delete = []
    for rp in qs:
        if not _pid_alive(rp.pid):
            to_delete.append(rp.id)

    if to_delete:
        RunningProcess.objects.filter(id__in=to_delete).delete()
    return len(to_delete)


@dataclass
class EnforceResult:
    started: list
    stopped: list
    desired_count: int
    running_count: int
    pruned_count: int = 0


def enforce_schedules(policies: Optional[Iterable[SchedulePolicy]] = None) -> EnforceResult:
    """
    Idempotent enforcer. If `policies` passed, only those policies are considered;
    otherwise all enabled policies.
    """
    now = timezone.now()
    now_local = timezone.localtime(now)
    lock_key = "bisk:enforce_schedules:lock"
    token = f"{os.getpid()}-{time.time()}"

    # acquire (succeeds only if key doesn't exist)
    if not cache.add(lock_key, token, timeout=30):
        return EnforceResult(started=[], stopped=[], desired_count=0, running_count=0)
    try:
        # now_local = timezone.localtime()
        desired = set()

        qs = (policies if policies is not None
              else SchedulePolicy.objects.filter(is_enabled=True).prefetch_related(
            "cameras", "windows", "exceptions", "windows__profile"
        ))

        # Build desired state
        for p in qs:
            # Respect exceptions (date-only or datetime range)
            if _policy_is_off_now(p, now_local):
                # Proactively stop any runners under this policy
                cam_ids = list(p.cameras.values_list("id", flat=True))
                for r in RunningProcess.objects.filter(camera_id__in=cam_ids):
                    was_stopped = _stop(r.pid)
                    r.status = "dead" if was_stopped else "stopping"
                    r.save(update_fields=["status"])
                continue  # skip adding any desired for this policy

            for w in p.windows.filter(day_of_week=now_local.weekday()):
                if _in_window(now_local.timetz().replace(tzinfo=None), w.start_time, w.end_time):
                    for cam in p.cameras.all():
                        pu = cam.pause_until
                        if pu and now < pu:
                            continue  # still paused
                        if pu and now >= pu:
                            cam.pause_until = None
                            cam.save(update_fields=["pause_until"])
                        desired.add((cam.id, w.profile_id))

        # Keep only the newest row per (camera, profile); mark others dead and try to stop them.
        latest_ids = (RunningProcess.objects
                      .values("camera_id", "profile_id")
                      .annotate(latest_id=Max("id"))
                      .values_list("latest_id", flat=True))

        for r in RunningProcess.objects.exclude(id__in=latest_ids):
            try:
                if _pid_alive(r.pid):
                    _stop(r.pid)
            except Exception:
                pass
            if r.status != "dead":
                r.status = "dead"
                r.save(update_fields=["status"])

        # Stop anything whose camera is paused; keep only truly alive, non-paused
        alive_by_pair = {}  # (cam_id, prof_id) -> [RunningProcess, ...]
        for r in RunningProcess.objects.select_related("camera", "profile"):
            pu = r.camera.pause_until
            if pu is not None and now < pu:
                # camera paused → tear it down now
                stopped = _stop(r.pid)  # your SIGTERM→SIGKILL helper
                r.status = "dead" if stopped else "stopping"
                r.save(update_fields=["status"])
                continue

            if _pid_alive(r.pid):
                alive_by_pair.setdefault((r.camera_id, r.profile_id), []).append(r)
            else:
                if r.status != "dead":
                    r.status = "dead"
                    r.save(update_fields=["status"])

        # Kill strays for currently paused cameras (no DB row, but still running)
        paused_cam_ids = set(Camera.objects.filter(pause_until__gt=now).values_list("id", flat=True))
        for cid in paused_cam_ids:
            _kill_stray_runners_for_camera(cid)

        running_count = sum(len(v) for v in alive_by_pair.values())
        started_list, stopped_list = [], []

        # Stop extras: any alive pair that is NOT desired must be terminated
        for pair, rows in list(alive_by_pair.items()):
            if pair not in desired:
                for r in rows:
                    if _stop(r.pid):
                        r.status = "dead"
                    else:
                        r.status = "stopping"
                    r.save(update_fields=["status"])
                    stopped_list.append((r.camera.name, r.profile.name, r.pid))
                alive_by_pair.pop(pair, None)  # ← remove after stopping

        # Auto-prune long-dead rows (minute-based threshold)
        pruned = _prune_long_dead(now)

        # Start missing: desired pairs with no alive PID
        for cam_id, prof_id in desired - set(alive_by_pair.keys()):
            camera = Camera.objects.filter(id=cam_id).first()
            profile = StreamProfile.objects.filter(id=prof_id).first()
            if not camera or not profile:
                continue

            # defensive cleanup: kill any lingering rows for this pair
            for old in RunningProcess.objects.filter(camera_id=cam_id, profile_id=prof_id):
                try:
                    if _pid_alive(old.pid):
                        _stop(old.pid)
                except Exception:
                    pass
                if old.status != "dead":
                    old.status = "dead"
                    old.save(update_fields=["status"])

            pid, cmdline = _start(camera, profile)

            affinity_csv = resolve_knob(camera, profile, "cpu_affinity") or ""
            cpu_affinity_s = affinity_csv.strip()

            nice_raw = resolve_knob(camera, profile, "nice")
            nice_value = int(nice_raw) if (nice_raw is not None and str(nice_raw) != "") else None


            # Build a snapshot of the knobs we effectively used (camera-first/profile-first aware)
            env_snapshot = {
                "hb_interval": str(int(resolve_knob(camera, profile, "hb_interval") or 10)),
                "snapshot_every": str(int(resolve_knob(camera, profile, "snapshot_every") or 3)),
                "rtsp_transport": (resolve_knob(camera, profile, "rtsp_transport") or "auto"),
                "hwaccel": (resolve_knob(camera, profile, "hwaccel") or "none"),
                "device": (resolve_knob(camera, profile, "device") or "cpu"),
                "gpu_index": int(resolve_knob(camera, profile, "gpu_index") or 0),
            }
            with transaction.atomic():
                try:
                    row, created = RunningProcess.objects.get_or_create(
                        camera=camera,
                        profile=profile,
                        pid=pid,
                        defaults={
                            "status": "running",
                            "effective_args": cmdline,  # masked
                            "effective_env": env_snapshot,
                            "nice": (nice_value if (nice_raw is not None and str(nice_raw) != "") else None),
                            "cpu_affinity": cpu_affinity_s,
                        },
                    )
                    updates = []

                    if not created and row.effective_args != cmdline:
                        row.effective_args = cmdline
                        updates.append("effective_args")

                    if not created and row.effective_env != env_snapshot:
                        row.effective_env = env_snapshot
                        updates.append("effective_env")

                    # persist knobs we applied OS-side
                    expected_nice = nice_value if (nice_raw is not None and str(nice_raw) != "") else None
                    if not created and row.nice != expected_nice:
                        row.nice = expected_nice
                        updates.append("nice")

                    if not created and row.cpu_affinity != cpu_affinity_s:
                        row.cpu_affinity = cpu_affinity_s
                        updates.append("cpu_affinity")

                    if updates:
                        row.save(update_fields=updates)

                except IntegrityError:
                    row = RunningProcess.objects.get(camera=camera, profile=profile, pid=pid)

                # ensure only this row is active for the pair
                RunningProcess.objects.filter(camera=camera, profile=profile) \
                    .exclude(id=row.id).update(status="dead")

            started_list.append((camera.name, profile.name, pid))

        cut = now - timedelta(minutes=10)
        # RunningProcess.objects.filter(status="dead", last_heartbeat__lt=cut).delete()
        RunningProcess.objects.filter(status="dead").filter(
            Q(last_heartbeat__lt=cut) | Q(last_heartbeat__isnull=True)).delete()

        return EnforceResult(started=started_list, stopped=stopped_list, desired_count=len(desired),
                             running_count=running_count,
                             pruned_count=pruned, )


    finally:
        # release only if we're still the owner
        if cache.get(lock_key) == token:
            cache.delete(lock_key)
