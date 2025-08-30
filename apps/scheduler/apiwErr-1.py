# apps/scheduler/api.py
import json
from django.http import JsonResponse, HttpResponseForbidden, HttpResponseNotAllowed
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.conf import settings
from django.core.cache import cache

from apps.scheduler.models import RunningProcess, RunnerHeartbeat

# #  Helper_01.1 ( Later delete 11:23 and the other part is before the return at the end of this file)
# SCRUB_KEYS = {"hb_key", "key", "rtsp", "password", "Authorization", "auth"}
#
# def _scrub_payload(d: dict) -> dict:
#     if not isinstance(d, dict):
#         return {}
#     out = {}
#     for k, v in d.items():
#         if k in SCRUB_KEYS:
#             out[k] = "***redacted***"
#         else:
#             out[k] = v
#     return out


def _auth_ok(request) -> bool:
    """
    If RUNNER_HEARTBEAT_KEY is set, accept it via:
      - header: X-BISK-KEY
      - query:  ?key=
      - json:   {"hb_key": "..."} or {"key": "..."}
    """
    want = getattr(settings, "RUNNER_HEARTBEAT_KEY", "")
    if not want:
        return True

    if request.headers.get("X-BISK-KEY") == want:
        return True

    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        body = {}

    return (
            request.GET.get("key") == want
            or body.get("hb_key") == want
            or body.get("key") == want
    )


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
    """
    Tolerant runner heartbeat:
    - Resolves RunningProcess by (camera_id, profile_id) or pid, then falls back to latest for camera.
    - Back-fills any missing camera_id/profile_id/pid from the resolved RP.
    - Updates last_heartbeat (+ last_heartbeat_at if present).
    - Logs a RunnerHeartbeat row on a rate-limited cadence (HB_LOG_EVERY_SEC).
    """
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    if not _auth_ok(request):
        return HttpResponseForbidden("bad key")

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    # Back-compat aliases
    camera_fps = payload.get("camera_fps")
    if camera_fps is None:
        camera_fps = payload.get("fps")  # older runners

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

    # ---- Resolve the RunningProcess row as robustly as possible
    rp = None
    if cam_id and prof_id:
        rp = (
            RunningProcess.objects.filter(camera_id=cam_id, profile_id=prof_id)
            .order_by("-id")
            .first()
        )
    if not rp and pid:
        rp = RunningProcess.objects.filter(pid=pid).order_by("-id").first()
    if not rp and cam_id:
        # Fall back when profile_id is missing: newest RP for this camera
        rp = (
            RunningProcess.objects.filter(camera_id=cam_id)
            .order_by("-id")
            .first()
        )

    # Back-fill any missing identifiers so we always touch/log the correct RP

    if rp:
        cam_id = cam_id or rp.camera_id
        prof_id = prof_id or rp.profile_id
        pid = pid or rp.pid

        # NEW: always touch the newest row for this (camera, profile)
        latest = (RunningProcess.objects
                  .filter(camera_id=cam_id, profile_id=prof_id)
                  .order_by('-id').first())
        if latest and latest.id != rp.id:
            rp = latest

        # NEW: keep the PID in sync on that newest row
        if pid and getattr(rp, 'pid', None) != pid:
            rp.pid = pid

        update_fields = []
        if hasattr(rp, "last_error"):
            rp.last_error = last_err
            update_fields.append("last_error")
        if hasattr(rp, "last_heartbeat_at"):
            rp.last_heartbeat_at = now
            update_fields.append("last_heartbeat_at")
        if hasattr(rp, "last_heartbeat"):
            rp.last_heartbeat = now
            update_fields.append("last_heartbeat")

        # If we changed pid above, persist it too
        if 'pid' in rp.__dict__:  # model has pid field
            update_fields.append('pid')

        rp.save(update_fields=list(dict.fromkeys(update_fields)) or None)

    # ---- Persist a heartbeat row (rate-limited per (camera, profile))
    if cam_id and prof_id:
        period = int(getattr(settings, "HB_LOG_EVERY_SEC", 60))  # 0 disables inserts
        if period > 0:
            def _insert(last_error_text=None):
                RunnerHeartbeat.objects.create(
                    camera_id=cam_id,
                    profile_id=prof_id,
                    pid=pid,
                    # store camera fps in legacy column 'fps'
                    fps=_to_float(camera_fps) or 0.0,
                    processed_fps=_to_float(processed_fps),
                    target_fps=_to_float(target_fps),
                    snapshot_every=_to_int(snapshot_every),
                    detected=int(detected or 0),
                    matched=int(matched or 0),
                    latency_ms=float(latency_ms or 0.0),
                    last_error=last_error_text,
                )

            if last_err:
                # Errors are always logged immediately
                _insert(last_err)
            else:
                exists = RunnerHeartbeat.objects.filter(
                    camera_id=cam_id, profile_id=prof_id
                ).only("id").exists()
                allowed = not exists
                if not allowed:
                    gate = f"bisk:hb:rl:{cam_id}:{prof_id}"
                    try:
                        # cache.add returns True once per period
                        allowed = cache.add(gate, "1", timeout=period)
                    except Exception:
                        allowed = True  # don't block inserts if cache is down
                if allowed:
                    _insert(None)
                    if not exists:
                        try:
                            cache.set(f"bisk:hb:rl:{cam_id}:{prof_id}", "1", timeout=period)
                        except Exception:
                            pass

    # # --- Helper_01.1 185:212 DEBUG: always write a scrubbed copy of the last payload to /tmp
    # try:
    #     with open("/tmp/last_hb.json", "w") as fh:
    #         import json as _json
    #         _json.dump(_scrub_payload(payload), fh, ensure_ascii=False, indent=2)
    # except Exception:
    #     pass
    #
    # # Optional echo in the HTTP response when ?echo=1 is present
    # extra = {}
    # if request.GET.get("echo") == "1":
    #     extra["payload"] = _scrub_payload(payload)
    #
    # # --- DEBUG: write the last payload to /tmp/last_hb.json
    # try:
    #     with open("/tmp/last_hb.json", "w") as fh:
    #         import json as _json
    #         fh.write(_json.dumps(payload, indent=2, ensure_ascii=False))
    # except Exception as e:
    #     # don’t crash if disk permission issue
    #     pass
    #
    # return JsonResponse({
    #     "ok": True,
    #     "variant": "hb-tolerant-v2",
    #     "resolved": {"camera_id": cam_id, "profile_id": prof_id, "pid": pid},
    #     **extra,
    # })

    return JsonResponse({"ok": True})
