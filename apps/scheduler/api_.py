# apps/scheduler/api.py
import json
from django.http import JsonResponse, HttpResponseForbidden, HttpResponseNotAllowed
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.conf import settings
from django.core.cache import cache
from apps.scheduler.models import RunningProcess, RunnerHeartbeat

SCRUB_KEYS = {"hb_key", "key", "rtsp", "password", "Authorization", "auth"}


def _scrub_payload(d: dict) -> dict:
    if not isinstance(d, dict):
        return {}
    out = {}
    for k, v in d.items():
        out[k] = "***redacted***" if k in SCRUB_KEYS else v
    return out


def _auth_ok(request) -> bool:
    want = getattr(settings, "RUNNER_HEARTBEAT_KEY", "")
    if not want:
        return True
    if request.headers.get("X-BISK-KEY") == want:
        return True
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        body = {}
    return (request.GET.get("key") == want
            or body.get("hb_key") == want
            or body.get("key") == want)


def _to_int(v):
    try:
        return int(v) if v is not None else None
    except Exception:
        return None


def _to_float(v):
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


@csrf_exempt
def heartbeat(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    if not _auth_ok(request):
        return HttpResponseForbidden("bad key")

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    # Back-compat: older runners send "fps" instead of "camera_fps"
    camera_fps = payload.get("camera_fps")
    if camera_fps is None:
        camera_fps = payload.get("fps")
    processed_fps = payload.get("processed_fps")

    cam_id = _to_int(payload.get("camera_id") or payload.get("camera"))
    prof_id = _to_int(payload.get("profile_id") or payload.get("profile"))
    pid = _to_int(payload.get("pid"))

    detected = _to_int(payload.get("detected")) or 0
    matched = _to_int(payload.get("matched")) or 0
    latency_ms = _to_float(payload.get("latency_ms")) or 0.0
    last_err = (payload.get("last_error") or "").strip()[:200]
    target_fps = _to_float(payload.get("target_fps"))
    snapshot_every = _to_int(payload.get("snapshot_every"))

    now = timezone.now()

    # ---- Resolve the RP row (PID > (camera,profile) > camera-only)
    rp = None
    if pid:
        rp = RunningProcess.objects.filter(pid=pid).order_by("-id").first()
    if not rp and cam_id and prof_id:
        rp = (RunningProcess.objects
              .filter(camera_id=cam_id, profile_id=prof_id)
              .order_by("-id").first())
    if not rp and cam_id:
        rp = (RunningProcess.objects
              .filter(camera_id=cam_id)
              .order_by("-id").first())

    if not rp:
        # Nothing to update; OK 200 so runner won’t spin on errors
        return JsonResponse({"ok": True, "resolved": None})

    # Back-fill any missing identifiers from the resolved RP
    cam_id = cam_id or rp.camera_id
    prof_id = prof_id or rp.profile_id
    pid = pid or rp.pid

    # ---- ALWAYS refresh RP timestamps & quick stats (no rate limit)
    update_fields = []
    if hasattr(rp, "last_heartbeat"):
        rp.last_heartbeat = now;
        update_fields.append("last_heartbeat")
    if hasattr(rp, "last_heartbeat_at"):
        rp.last_heartbeat_at = now;
        update_fields.append("last_heartbeat_at")

    # Optional: carry a few live metrics on the RP row for the table
    if camera_fps is not None and hasattr(rp, "camera_fps"):
        rp.camera_fps = _to_float(camera_fps);
        update_fields.append("camera_fps")
    if processed_fps is not None and hasattr(rp, "processed_fps"):
        rp.processed_fps = _to_float(processed_fps);
        update_fields.append("processed_fps")
    if target_fps is not None and hasattr(rp, "target_fps"):
        rp.target_fps = target_fps;
        update_fields.append("target_fps")
    if snapshot_every is not None and hasattr(rp, "snapshot_every"):
        rp.snapshot_every = snapshot_every;
        update_fields.append("snapshot_every")
    if update_fields:
        rp.save(update_fields=update_fields)

    # ---- Rate-limited RunnerHeartbeat row (history/troubleshooting)
    log_every = int(getattr(settings, "HB_LOG_EVERY_SEC", 10))
    key = f"hb:log:{cam_id}:{prof_id}"
    if cache.add(key, "1", timeout=log_every):
        RunnerHeartbeat.objects.create(
            camera_id=cam_id, profile_id=prof_id, ts=now,
            fps=_to_float(camera_fps) or 0.0,
            target_fps=target_fps,
            snapshot_every=snapshot_every or 0,
            detected=detected, matched=matched,
            latency_ms=latency_ms, last_error=last_err or "",
        )

    # Debug/echo support that you already added
    if request.GET.get("echo"):
        return JsonResponse({
            "ok": True,
            "variant": "hb-tolerant-v2",
            "resolved": {"camera_id": cam_id, "profile_id": prof_id, "pid": pid, "rp_id": rp.id},
            "payload": payload,
        })

    return JsonResponse({
        "ok": True,
        "variant": "hb-tolerant-v2",
        "resolved": {"camera_id": cam_id, "profile_id": prof_id, "pid": pid, "rp_id": rp.id},
    })
