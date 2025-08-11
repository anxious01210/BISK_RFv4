from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponseForbidden
from django.utils import timezone
from django.conf import settings
from .models import RunnerHeartbeat, RunningProcess
import json


@csrf_exempt
def heartbeat(request):
    # Simple shared-secret check
    if request.headers.get("X-BISK-KEY") != getattr(settings, "RUNNER_HEARTBEAT_KEY", ""):
        return HttpResponseForbidden("invalid key")

    if request.method != "POST":
        return HttpResponseBadRequest("POST only")
    try:
        data = json.loads(request.body or "{}")
        camera_id = int(data["camera_id"])
        profile_id = int(data["profile_id"])
        fps = float(data.get("fps", 0))
        detected = int(data.get("detected", 0))
        matched = int(data.get("matched", 0))
        latency_ms = float(data.get("latency_ms", 0))
        last_error = (data.get("last_error", "") or "")[:200]
    except Exception:
        return HttpResponseBadRequest("invalid payload")

    # upsert heartbeat
    RunnerHeartbeat.objects.update_or_create(
        camera_id=camera_id, profile_id=profile_id,
        defaults=dict(fps=fps, detected=detected, matched=matched, latency_ms=latency_ms, last_error=last_error)
    )
    # touch the corresponding RunningProcess (if any)
    RunningProcess.objects.filter(camera_id=camera_id, profile_id=profile_id).update(
        last_heartbeat=timezone.now(), status="running"
    )
    return JsonResponse({"ok": True, "ts": timezone.now().isoformat(timespec="seconds")})
