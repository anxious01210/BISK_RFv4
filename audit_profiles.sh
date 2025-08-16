#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f manage.py ]]; then
  echo "Run this from your Django project root (where manage.py lives)." >&2
  exit 1
fi

python manage.py shell <<'PY'
from django.conf import settings
from django.utils import timezone

# Import models (adjust paths if your app labels differ)
from apps.cameras.models import Camera
try:
    from apps.scheduler.models import StreamProfile, RunningProcess
except Exception as e:
    StreamProfile = None
    RunningProcess = None

# Minimal defaults (used only when both camera/profile omit a field)
DEFAULTS = {
    "rtsp_transport": "auto",
    "hwaccel":        "none",
    "device":         "cpu",
    "gpu_index":      0,
    "hb_interval":    10,
    "snapshot_every": 3,
    "nice":           0,
    "cpu_affinity":   "",
}

FIELDS = [
    "rtsp_transport",
    "hwaccel",
    "device",
    "gpu_index",
    "hb_interval",
    "snapshot_every",
    "nice",
    "cpu_affinity",
]

def gval(obj, field):
    """Return value or None if 'unset' ('' or None). Numeric 0 is considered a value."""
    if obj is None:
        return None
    if not hasattr(obj, field):
        return None
    v = getattr(obj, field)
    if isinstance(v, str):
        v = v.strip()
        return v if v != "" else None
    return v  # allow 0 / numbers / etc.

def resolve(camera, profile, field, camera_first: bool):
    cv = gval(camera, field)
    pv = gval(profile, field)
    if camera_first:
        return cv if cv is not None else (pv if pv is not None else DEFAULTS.get(field))
    else:
        return pv if pv is not None else (cv if cv is not None else DEFAULTS.get(field))

def as_csv(v):
    if v is None: return "-"
    if isinstance(v, (list, tuple)): return ",".join(str(x) for x in v)
    return str(v)

# Choose the single profile (your case: 1 profile). If multiple, prefer active ones.
profile = None
if StreamProfile:
    qs = StreamProfile.objects.all()
    if hasattr(StreamProfile, "is_active"):
        qs = qs.filter(is_active=True) or StreamProfile.objects.all()
    if qs.count() == 1:
        profile = qs.first()
    else:
        # If multiple, we won't guess—just print them all below.
        pass

def print_profile_summary():
    print("=== StreamProfiles ===")
    if not StreamProfile:
        print("  (StreamProfile model not found)")
        return
    qs = StreamProfile.objects.all().order_by("id")
    if not qs.exists():
        print("  (no profiles)")
        return
    for sp in qs:
        print(f"  - id={sp.id} name={getattr(sp,'name','(no-name)')} is_active={getattr(sp,'is_active',None)}")
        for f in FIELDS + ["fps", "detection_set", "det_set"]:
            if hasattr(sp, f):
                print(f"      {f}: {getattr(sp,f)}")

def print_effective_args_for_cam(cam):
    if not RunningProcess:
        print("  (RunningProcess model not found)")
        return
    rp = RunningProcess.objects.filter(camera=cam).order_by("-id").first()
    if not rp:
        print("  (no RunningProcess rows for this camera)")
        return
    print(f"  latest RP id={rp.id} pid={rp.pid}")
    if hasattr(rp, "effective_args"):
        print("  effective_args:")
        print(f"    {rp.effective_args}")
    else:
        print("  (effective_args not present on RunningProcess)")

    # Try to show child ffmpeg cmdline if psutil is available
    try:
        import psutil
        p = psutil.Process(rp.pid)
        kids = p.children(recursive=True)
        for c in kids:
            try:
                cl = " ".join(c.cmdline())
            except Exception:
                cl = ""
            if "ffmpeg" in cl:
                print(f"  ffmpeg child pid={c.pid}")
                print(f"    {cl}")
    except Exception as e:
        print(f"  (psutil not available or cannot inspect children: {e})")

print_profile_summary()
print()
print("=== Cameras (with resolution) ===")
cams = Camera.objects.all().order_by("id")
if not cams.exists():
    print("  (no cameras)")
else:
    for cam in cams:
        print(f"\n-- Camera id={cam.id} name={cam.name!r} prefer_camera_over_profile={getattr(cam,'prefer_camera_over_profile',False)}")
        print(f"   rtsp_url: {cam.rtsp_url}")
        # Camera values
        print("   Camera values:")
        for f in FIELDS:
            print(f"     {f}: {getattr(cam,f,None)}")
        # Profile values (using the single selected profile or None)
        if profile:
            print(f"   Using profile id={profile.id} name={getattr(profile,'name','(no-name)')}")
        else:
            print("   (No single active profile selected; showing 'first' per-field value if exists.)")
        print("   Profile values:")
        if StreamProfile and profile:
            for f in FIELDS + ["fps","detection_set","det_set"]:
                if hasattr(profile, f):
                    print(f"     {f}: {getattr(profile,f)}")
        elif StreamProfile:
            # fall back: show values of first profile (if any), just for visibility
            sp = StreamProfile.objects.order_by("id").first()
            if sp:
                print(f"     (multiple/none active) first profile id={sp.id} name={getattr(sp,'name','(no-name)')}")
                for f in FIELDS + ["fps","detection_set","det_set"]:
                    if hasattr(sp, f):
                        print(f"       {f}: {getattr(sp,f)}")

        # Resolved values according to the camera toggle
        camera_first = bool(getattr(cam, "prefer_camera_over_profile", False))
        prof = profile if profile else (StreamProfile.objects.order_by("id").first() if StreamProfile else None)
        print("   Resolved (effective) values:")
        for f in FIELDS:
            eff = resolve(cam, prof, f, camera_first)
            print(f"     {f}: {eff}")
        # FPS / detection set are profile-driven (print what runner would see)
        fps_val = None
        det_val = None
        if prof is not None:
            if hasattr(prof, "fps"): fps_val = getattr(prof,"fps")
            if hasattr(prof, "detection_set"): det_val = getattr(prof,"detection_set")
            elif hasattr(prof, "det_set"): det_val = getattr(prof,"det_set")
        print(f"     fps: {fps_val if fps_val is not None else '(default)'}")
        print(f"     detection_set: {det_val if det_val is not None else '(default)'}")

        # Show latest RunningProcess command
        print_effective_args_for_cam(cam)
PY
