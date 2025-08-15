from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponseForbidden
from django.conf import settings
from django.utils import timezone
from .models import RunnerHeartbeat, RunningProcess
import json


@csrf_exempt
def heartbeat(request):
    # Shared-secret
    if request.headers.get("X-BISK-KEY") != getattr(settings, "RUNNER_HEARTBEAT_KEY", ""):
        return HttpResponseForbidden("invalid key")

    if request.method != "POST":
        return HttpResponseBadRequest("POST only")

    # Parse JSON (Django view, not DRF → use request.body)
    try:
        data = json.loads(request.body or b"{}")
        camera_id = int(data["camera_id"])
        profile_id = int(data["profile_id"])
        pid = int(data.get("pid") or 0)
        fps = float(data.get("fps", 0))
        detected = int(data.get("detected", 0))
        matched = int(data.get("matched", 0))
        latency_ms = float(data.get("latency_ms", 0))
        last_error = (data.get("last_error", "") or "")[:200]
    except Exception:
        return HttpResponseBadRequest("invalid payload")

    now = timezone.now()

    # Upsert "latest heartbeat" row for this (camera, profile)
    RunnerHeartbeat.objects.update_or_create(
        camera_id=camera_id,
        profile_id=profile_id,
        defaults=dict(
            ts=now,
            fps=fps,
            detected=detected,
            matched=matched,
            latency_ms=latency_ms,
            last_error=last_error,
        ),
    )

    # Touch ONLY the matching RunningProcess (camera, profile, pid)
    qs = RunningProcess.objects.filter(camera_id=camera_id, profile_id=profile_id)
    if pid > 0:
        qs = qs.filter(pid=pid)  # <- critical: update only this PID

    # If no row with this PID exists, do nothing (don’t refresh older rows!)
    if qs.exists():
        qs.update(last_heartbeat=now, status="running")

    return JsonResponse({"ok": True, "ts": now.isoformat(timespec="seconds")})
