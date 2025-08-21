# apps/scheduler/api.py
import json
from django.http import JsonResponse, HttpResponseForbidden, HttpResponseNotAllowed
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.conf import settings
from apps.cameras.models import Camera
from apps.scheduler.models import RunningProcess
# Import RunnerHeartbeat directly so we fail loudly if there is a real problem.
from apps.scheduler.models import RunnerHeartbeat
from django.core.cache import cache


def _auth_ok(request) -> bool:
    """If RUNNER_HEARTBEAT_KEY is set, require X-BISK-KEY header to match."""
    key_required = getattr(settings, "RUNNER_HEARTBEAT_KEY", "")
    if not key_required:
        return True
    return request.headers.get("X-BISK-KEY") == key_required


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

    # Back-compat:
    camera_fps = payload.get("camera_fps")
    # if camera_fps is None:
    #     camera_fps = payload.get("fps")  # older runners send 'fps'

    processed_fps = payload.get("processed_fps")  # may be None
    cam_id = payload.get("camera_id")
    prof_id = payload.get("profile_id")
    pid = payload.get("pid")
    detected = payload.get("detected") or 0
    matched = payload.get("matched") or 0
    latency = payload.get("latency_ms") or 0
    last_err = (payload.get("last_error") or "").strip()[:200]
    # NEW: resource-derived telemetry
    target_fps = payload.get("target_fps")
    snapshot_every = payload.get("snapshot_every")

    now = timezone.now()

    # ---- Update latest RunningProcess for this camera/profile (or by pid as fallback)
    rp = None
    if cam_id and prof_id:
        rp = (RunningProcess.objects
              .filter(camera_id=cam_id, profile_id=prof_id)
              .order_by("-id")
              .first())
    if not rp and pid:
        rp = RunningProcess.objects.filter(pid=pid).order_by("-id").first()

    if rp:
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
        if update_fields:
            rp.save(update_fields=update_fields)
        else:
            rp.save()

    # ---- Persist a heartbeat row, rate-limited per (camera,profile), but never block on cache errors
    if cam_id and prof_id:
        period = int(getattr(settings, "HB_LOG_EVERY_SEC", 60))  # 0 disables inserts entirely
        if period > 0:
            if last_err:
                # Errors are always logged immediately
                RunnerHeartbeat.objects.create(
                    camera_id=cam_id, profile_id=prof_id, pid=pid,
                    # store camera fps in legacy column 'fps'
                    fps=float(camera_fps or 0.0),
                    processed_fps=float(processed_fps) if processed_fps is not None else None,
                    target_fps=float(target_fps) if target_fps is not None else None,
                    snapshot_every=int(snapshot_every) if snapshot_every is not None else None,
                    detected=int(detected or 0),
                    matched=int(matched or 0),
                    latency_ms=float(latency or 0.0),
                    last_error=last_err if last_err else None,
                )

            else:
                # First-ever row for (cam, prof)?
                exists = RunnerHeartbeat.objects.filter(
                    camera_id=cam_id, profile_id=prof_id
                ).only("id").exists()
                allowed = not exists
                if not allowed:
                    gate = f"bisk:hb:rl:{cam_id}:{prof_id}"
                    try:
                        # cache.add returns True only once per period
                        allowed = cache.add(gate, "1", timeout=period)
                    except Exception:
                        # If cache is down, don't block inserts
                        allowed = True
                if allowed:
                    RunnerHeartbeat.objects.create(
                        camera_id=cam_id, profile_id=prof_id, pid=pid,
                        fps=float(camera_fps or 0.0),
                        processed_fps=float(processed_fps) if processed_fps is not None else None,
                        target_fps=float(target_fps) if target_fps is not None else None,
                        snapshot_every=int(snapshot_every) if snapshot_every is not None else None,
                        detected=int(detected or 0),
                        matched=int(matched or 0),
                        latency_ms=float(latency or 0.0),
                        last_error=None,
                    )

                    # Best-effort: set the gate; ignore cache errors
                    if not exists:
                        try:
                            cache.set(f"bisk:hb:rl:{cam_id}:{prof_id}", "1", timeout=period)
                        except Exception:
                            pass

    return JsonResponse({"ok": True})
