# apps/scheduler/services/enforcer.py
import os, sys, signal, subprocess, time, shlex, psutil, re
from dataclasses import dataclass
from typing import Iterable, Optional
from django.utils import timezone

from apps.scheduler.models import (
    SchedulePolicy,
    RunningProcess,
    StreamProfile,
    GlobalResourceSettings,
    CameraResourceOverride,
)
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


# --- Policy/Resource normalization (Phase 2) ---
def _as_int_or_none(x):
    try:
        if x in (None, "", "auto"): return None
        return int(x)
    except Exception:
        return None


def _choose_policy(camera, profile):
    """
    Requested policy values. Camera overrides win only if prefer flag is ON and value is non-null.
    Returns dict with requested 'det_set', 'target_fps', 'hb_interval', 'snapshot_every',
    'rtsp_transport', plus optional items if present on your models (min_score, model_tag).
    """
    prefer = bool(getattr(camera, "prefer_camera_over_profile", False))
    # Profile (baseline)
    det_req_p = getattr(profile, "det_set_req", getattr(profile, "detection_set", "auto"))
    fps_req_p = getattr(profile, "target_fps_req", getattr(profile, "fps", None))
    hb_p = getattr(profile, "hb_interval", None)
    snap_p = getattr(profile, "snapshot_every", None)
    tr_p = getattr(profile, "rtsp_transport", "") or ""
    min_p = getattr(profile, "min_score", None)
    model_p = getattr(profile, "model_tag", None)

    # Camera overrides (nullable)
    det_req_c = getattr(camera, "det_set_req", None)
    fps_req_c = getattr(camera, "target_fps_req", None)
    hb_c = getattr(camera, "hb_interval", None)
    snap_c = getattr(camera, "snapshot_every", None)
    tr_c = getattr(camera, "rtsp_transport", "") or ""
    min_c = getattr(camera, "min_score", None)
    model_c = getattr(camera, "model_tag", None)

    def pick(cam_val, prof_val):
        return cam_val if (prefer and cam_val not in (None, "")) else prof_val

    return {
        "det_req": str(pick(det_req_c, det_req_p) or "auto"),
        "fps_req": _as_int_or_none(pick(fps_req_c, fps_req_p)),
        "hb_interval": _as_int_or_none(pick(hb_c, hb_p)),
        "snapshot_every": _as_int_or_none(pick(snap_c, snap_p)),
        "rtsp_transport": (pick(tr_c, tr_p) or "auto"),
        "min_score": pick(min_c, min_p),
        "model_tag": pick(model_c, model_p),
    }


def _choose_resources(camera):
    """
    Effective placement defaults and caps:
      start with Global (if active), then override with CRO (if active & non-null).
    Returns dict with device, hwaccel, gpu_index, cpu_affinity, cpu_nice,
    gpu_memory_fraction, gpu_target_util_percent, max_fps_cap, det_set_max.
    """
    # Global defaults
    grs = GlobalResourceSettings.get_solo() if hasattr(GlobalResourceSettings, "get_solo") else GlobalResourceSettings.objects.order_by("pk").first()
    res = {
        "device": None, "hwaccel": None, "gpu_index": None,
        "cpu_affinity": None, "cpu_nice": None,
        "gpu_memory_fraction": None, "gpu_target_util_percent": None,
        "cpu_quota_percent": None,  # <--- ADD THIS
        "max_fps_cap": None, "det_set_max": None,
        # NEW knobs (nullable = inherit)
        "model_tag": None,
        "pipe_mjpeg_q": None,
        "crop_format": None,
        "crop_jpeg_quality": None,
        "min_face_px": None,  # per-camera
        "min_face_px_default": None,  # global default
        "quality_version": None,
        "save_debug_unmatched": None,
        "use_global": False, "use_cro": False,
    }
    if grs and getattr(grs, "is_active", True):
        res["use_global"] = True
        for k in ("device","hwaccel","gpu_index","cpu_affinity","cpu_nice",
                  "gpu_memory_fraction","gpu_target_util_percent","cpu_quota_percent",
                  # NEW (global defaults)
                  "model_tag","pipe_mjpeg_q","crop_format","crop_jpeg_quality",
                  "quality_version"):
            v = getattr(grs, k, None)
            if v not in (None, "", []): res[k] = v
        if getattr(grs, "max_fps_default", None) not in (None, ""):
            res["max_fps_cap"] = getattr(grs, "max_fps_default")
        if getattr(grs, "det_set_max", None) not in (None, ""):
            res["det_set_max"] = str(getattr(grs, "det_set_max"))
        if getattr(grs, "min_face_px_default", None) not in (None, ""):
            res["min_face_px_default"] = int(getattr(grs, "min_face_px_default"))

    # Camera override
    cro = CameraResourceOverride.objects.filter(camera=camera).first()
    if cro and getattr(cro, "is_active", False):
        res["use_cro"] = True
        for k in ("device","hwaccel","gpu_index","cpu_affinity","cpu_nice",
                  "gpu_memory_fraction","gpu_target_util_percent","cpu_quota_percent",
                  # NEW (per-camera overrides)
                  "model_tag","pipe_mjpeg_q","crop_format","crop_jpeg_quality",
                  "min_face_px","quality_version","save_debug_unmatched","max_fps","det_set_max"):
            v = getattr(cro, k, None)
            if v not in (None, "", []):
                if k == "max_fps": res["max_fps_cap"] = v
                elif k == "det_set_max": res["det_set_max"] = str(v)
                else: res[k] = v

    return res


def _clamp_and_finalize(policy_req: dict, res: dict):
    """
    Clamp det_set and fps to caps; return (final_det_set:str, final_fps:int|None).
    det_set policy is a string like 'auto','640','800','1024','1600','2048'.
    """
    # det_set
    det_req = policy_req.get("det_req") or "auto"
    det_max = res.get("det_set_max")

    def det_to_int(s):
        v = _as_int_or_none(s)
        return v if v is not None else 10 ** 9  # 'auto' treated as huge so min(auto,cap)=cap

    if det_max not in (None, "", "auto"):
        det_final = str(min(det_to_int(det_req), det_to_int(det_max)))
    else:
        det_final = str(det_req)

    # fps
    fps_req = policy_req.get("fps_req")
    fps_cap = res.get("max_fps_cap")
    if fps_req is None and fps_cap is None:
        fps_final = None
    elif fps_req is None:
        fps_final = int(fps_cap)
    elif fps_cap is None:
        fps_final = int(fps_req)
    else:
        fps_final = int(min(int(fps_req), int(fps_cap)))

    return det_final, fps_final


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
    py = sys.executable

    # 1) Normalize policy/resources
    policy_req = _choose_policy(camera, profile)
    res_eff = _choose_resources(camera)
    det_set, fps = _clamp_and_finalize(policy_req, res_eff)

    # 2) Fill remaining policy knobs
    hb_interval = policy_req.get("hb_interval") or getattr(settings, "HEARTBEAT_INTERVAL_DEFAULT", 10)
    snapshot_every = policy_req.get("snapshot_every") or getattr(settings, "SNAPSHOT_EVERY_DEFAULT", 30)
    rtsp_transport = policy_req.get("rtsp_transport") or "auto"
    min_score = policy_req.get("min_score", None)
    model_tag = policy_req.get("model_tag", None)

    # 3) Runner script
    impl = getattr(settings, "RUNNER_IMPL", "ffmpeg_all")
    runner = {
        "ffmpeg_all": getattr(settings, "RUNNER_SCRIPT_ALL", BASE_DIR / "extras" / "recognize_runner_all_ffmpeg.py"),
        "ffmpeg_one": getattr(settings, "RUNNER_SCRIPT_ONE", BASE_DIR / "extras" / "recognize_runner_ffmpeg.py"),
    }.get(impl, getattr(settings, "RUNNER_SCRIPT_ALL", BASE_DIR / "extras" / "recognize_runner_all_ffmpeg.py"))

    # 4) Build command (NO legacy placement here)
    rtsp_cli = ["--rtsp_transport", rtsp_transport] if (rtsp_transport and rtsp_transport != "auto") else []
    cmd = [
              str(py), str(runner),
              "--camera", str(camera.id),
              "--profile", str(profile.id),
              "--fps", str(fps if fps is not None else 0),
              "--det_set", det_set,
              "--hb", settings.RUNNER_HEARTBEAT_URL,
              "--hb_key", settings.RUNNER_HEARTBEAT_KEY,
              "--rtsp", camera.rtsp_url,
              "--ffmpeg", settings.FFMPEG_PATH,
              "--ffprobe", settings.FFPROBE_PATH,
              "--snapshots", str(settings.SNAPSHOT_DIR),
              "--hb_interval", str(hb_interval),
              "--snapshot_every", str(snapshot_every),
          ] + rtsp_cli

    if min_score is not None:
        cmd += ["--min_score", str(min_score)]
    if model_tag:

        cmd += ["--model", str(model_tag)]

    # 5) Placement ONLY from normalized resources (res_eff)
    if res_eff.get("device"):
        cmd += ["--device", str(res_eff["device"])]
    if res_eff.get("gpu_index") not in (None, "", "auto"):
        cmd += ["--gpu_index", str(res_eff["gpu_index"])]
    if res_eff.get("hwaccel"):
        cmd += ["--hwaccel", str(res_eff["hwaccel"])]

    # 6) OS-level knobs come from resources too
    cpu_affinity_s = (res_eff.get("cpu_affinity") or "").strip()
    nice_value = int(res_eff.get("cpu_nice")) if res_eff.get("cpu_nice") not in (None, "") else 0

    # 7) run
    env = os.environ.copy()
    env["BISK_HB_INTERVAL"] = str(hb_interval)
    env["BISK_SNAPSHOT_EVERY"] = str(snapshot_every)

    # --- pass resource targets to the runner ---
    if res_eff.get("gpu_target_util_percent") not in (None, ""):
        env["BISK_GPU_TARGET_UTIL"] = str(int(res_eff["gpu_target_util_percent"]))
    # Optional smoothing window; leave unset if you don't have a model field
    if res_eff.get("gpu_util_window_ms") not in (None, ""):
        env["BISK_GPU_UTIL_WINDOW_MS"] = str(int(res_eff["gpu_util_window_ms"]))
    if res_eff.get("cpu_quota_percent") not in (None, ""):
        env["BISK_CPU_QUOTA_PERCENT"] = str(int(res_eff["cpu_quota_percent"]))
    if res_eff.get("gpu_memory_fraction") not in (None, ""):
        env["BISK_GPU_MEMORY_FRAC"] = str(float(res_eff["gpu_memory_fraction"]))
    # let runners know the placement index too (handy for pacer)
    if res_eff.get("gpu_index") not in (None, ""):
        env["BISK_PLACE_GPU_INDEX"] = str(res_eff["gpu_index"])

    # When building the child process environment (you already export BISK_CAP_* etc.)
    env["BISK_PIPE_MJPEG_Q"] = str(res_eff.get("pipe_mjpeg_q") or "")
    env["BISK_MIN_FACE_PX"] = str(res_eff.get("min_face_px") or res_eff.get("min_face_px_default") or "")
    env["BISK_CROP_FMT"] = str(res_eff.get("crop_format") or "")
    env["BISK_CROP_JPEG_Q"] = str(res_eff.get("crop_jpeg_quality") or "")
    env["BISK_QUALITY_VERSION"] = str(res_eff.get("quality_version") or "")
    env["BISK_SAVE_DEBUG_UNMATCHED"] = (
        "1" if res_eff.get("save_debug_unmatched") is True
        else ("0" if res_eff.get("save_debug_unmatched") is False else "")
    )

    p = subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid, close_fds=True, env=env
    )

    # apply affinity/nice at OS level
    try:
        proc = psutil.Process(p.pid)
        if cpu_affinity_s:
            cores = [int(x) for x in cpu_affinity_s.split(",") if x.strip().isdigit()]
            if cores: proc.cpu_affinity(cores)
        if nice_value: proc.nice(nice_value)
    except Exception:
        pass

    # Build the **env_snapshot** here (it needs res_eff)
    env_snapshot = {
        "hb_interval": str(hb_interval),
        "snapshot_every": str(snapshot_every),
        "rtsp_transport": rtsp_transport or "auto",
        "BISK_PLACE_DEVICE": str(res_eff.get("device") or ""),
        "BISK_PLACE_GPU_INDEX": str(res_eff.get("gpu_index") or ""),
        "BISK_PLACE_HWACCEL": str(res_eff.get("hwaccel") or ""),
        "BISK_CAP_MAX_FPS": str(res_eff.get("max_fps_cap") or ""),
        "BISK_CAP_DET_SET_MAX": str(res_eff.get("det_set_max") or ""),
        "_det_set": str(det_set),
        "_fps": str(fps if fps is not None else 0),
    }
    if res_eff.get("cpu_nice") not in (None, ""):
        env_snapshot["BISK_CPU_NICE"] = str(res_eff["cpu_nice"])
    if res_eff.get("cpu_affinity") not in (None, ""):
        env_snapshot["BISK_CPU_AFFINITY"] = str(res_eff["cpu_affinity"])

    if res_eff.get("gpu_target_util_percent") not in (None, ""):
        env_snapshot["BISK_GPU_TARGET_UTIL"] = str(res_eff["gpu_target_util_percent"])
    if res_eff.get("cpu_quota_percent") not in (None, ""):
        env_snapshot["BISK_CPU_QUOTA_PERCENT"] = str(res_eff["cpu_quota_percent"])
    if res_eff.get("gpu_memory_fraction") not in (None, ""):
        env_snapshot["BISK_GPU_MEMORY_FRAC"] = str(res_eff["gpu_memory_fraction"])
    if res_eff.get("gpu_index") not in (None, ""):
        env_snapshot["BISK_PLACE_GPU_INDEX"] = str(res_eff["gpu_index"])

    env_snapshot.update({
        "BISK_PIPE_MJPEG_Q": str(res_eff.get("pipe_mjpeg_q") or ""),
        "BISK_MIN_FACE_PX": str(res_eff.get("min_face_px") or res_eff.get("min_face_px_default") or ""),
        "BISK_CROP_FMT": str(res_eff.get("crop_format") or ""),
        "BISK_CROP_JPEG_Q": str(res_eff.get("crop_jpeg_quality") or ""),
        "BISK_QUALITY_VERSION": str(res_eff.get("quality_version") or ""),
        "BISK_SAVE_DEBUG_UNMATCHED": (
            "1" if res_eff.get("save_debug_unmatched") is True
            else ("0" if res_eff.get("save_debug_unmatched") is False else "")
        )
    })

    # mask and return
    def _mask_url(s: str) -> str:
        return re.sub(r'(rtsp://[^:@\s]+:)[^@/\s]+(@)', r'\1***\2', s)

    def _mask_cmdline(argv):
        return " ".join(shlex.quote(_mask_url(x)) for x in argv)

    return p.pid, _mask_cmdline(cmd), env_snapshot, (nice_value if nice_value else None), cpu_affinity_s


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
            if ("recognize_ffmpeg.py" not in joined
                    and "recognize_runner_all_ffmpeg.py" not in joined
                    and "recognize_runner_ffmpeg.py" not in joined):
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
                        if hasattr(cam, "is_active") and cam.is_active is False:
                            continue
                        if hasattr(cam, "scan") and cam.scan is False:
                            continue
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

        # --- NEW: spec-drift detection (restart runners whose config no longer matches) ---
        for pair, rows in list(alive_by_pair.items()):
            cam_id, prof_id = pair
            try:
                camera = Camera.objects.get(id=cam_id)
                profile = StreamProfile.objects.get(id=prof_id)
            except Camera.DoesNotExist:
                continue

            # Recompute the intended, normalized spec
            pol = _choose_policy(camera, profile)
            res = _choose_resources(camera)
            det_final, fps_final = _clamp_and_finalize(pol, res)

            # Build the minimal “desired env” view that we compare against what we saved at start
            desired_env = {
                "hb_interval": str(pol.get("hb_interval") or getattr(settings, "HEARTBEAT_INTERVAL_DEFAULT", 10)),
                "snapshot_every": str(pol.get("snapshot_every") or getattr(settings, "SNAPSHOT_EVERY_DEFAULT", 30)),
                "rtsp_transport": (pol.get("rtsp_transport") or "auto"),
                "BISK_PLACE_DEVICE": str(res.get("device") or ""),
                "BISK_PLACE_GPU_INDEX": str(res.get("gpu_index") or ""),
                "BISK_PLACE_HWACCEL": str(res.get("hwaccel") or ""),
                "BISK_CAP_MAX_FPS": str(res.get("max_fps_cap") or ""),
                "BISK_CAP_DET_SET_MAX": str(res.get("det_set_max") or ""),
                # Not strictly required, but cheap to include:
                "_det_set": str(det_final),
                "_fps": str(fps_final if fps_final is not None else 0),
            }
            # Optional CPU knobs if present
            if res.get("cpu_nice") not in (None, ""):
                desired_env["BISK_CPU_NICE"] = str(res["cpu_nice"])
            if res.get("cpu_affinity") not in (None, ""):
                desired_env["BISK_CPU_AFFINITY"] = str(res["cpu_affinity"])

            # Compare against each alive row (if any differ, stop them now)
            for r in rows:
                current_env = dict(r.effective_env or {})
                # embed det/fps from effective_args if you didn’t store them in env:
                # (Cheap parse heuristic; safe to skip if you already store them in env)
                # -- we rely mainly on the keys written by _start(), which you already do.
                needs_restart = False
                for k, v in desired_env.items():
                    if str(current_env.get(k, "")) != str(v):
                        needs_restart = True
                        break
                if needs_restart:
                    if _stop(r.pid):
                        r.status = "dead"
                    else:
                        r.status = "stopping"
                    r.save(update_fields=["status"])
                    # Remove this pair from 'alive' so the “start missing” pass will relaunch it
                    alive_by_pair.pop(pair, None)
                    break  # if there were multiple rows, one stop is enough
        # --- end spec-drift detection ---

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

            pid, cmdline, env_snapshot, nice_value, cpu_affinity_s = _start(camera, profile)

            with transaction.atomic():
                row, created = RunningProcess.objects.get_or_create(
                    camera=camera, profile=profile, pid=pid,
                    defaults={
                        "status": "running",
                        "effective_args": cmdline,
                        "effective_env": env_snapshot,
                        "nice": nice_value,
                        "cpu_affinity": cpu_affinity_s,
                    },
                )
                updates = []
                if not created and row.effective_args != cmdline:
                    row.effective_args = cmdline;
                    updates.append("effective_args")
                if not created and row.effective_env != env_snapshot:
                    row.effective_env = env_snapshot;
                    updates.append("effective_env")
                if not created and row.nice != nice_value:
                    row.nice = nice_value;
                    updates.append("nice")
                if not created and row.cpu_affinity != cpu_affinity_s:
                    row.cpu_affinity = cpu_affinity_s;
                    updates.append("cpu_affinity")
                if updates: row.save(update_fields=updates)

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
