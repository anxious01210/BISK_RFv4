import os, platform, shutil, subprocess, psutil
from dataclasses import dataclass
from typing import Optional, List
from django.http import JsonResponse
from django.conf import settings
# from django.contrib.admin.views.decorators import staff_member_required
from apps.scheduler.models import RunningProcess, RunnerHeartbeat
from django.contrib import admin
from django.contrib.admin.views.decorators import staff_member_required
from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_POST
from django.contrib import messages
from django.shortcuts import redirect
from django.core.cache import cache, caches
from .services.enforcer import enforce_schedules
from django.db.models import OuterRef, Subquery, Value
from django.utils import timezone



@staff_member_required
@require_POST
@csrf_protect
def enforce_now(request):
    next_url = request.POST.get("next") or request.META.get("HTTP_REFERER") or "/admin/system/"
    if not cache.add("bisk:enforce:now", "1", timeout=10):
        messages.warning(request, "Please wait ~10s between manual enforces.")
        return redirect(next_url)
    res = enforce_schedules()
    messages.success(
        request,
        f"Enforced ✓ — started {len(res.started)}, stopped {len(res.stopped)}, "
        f"desired {res.desired_count}, running {res.running_count}, pruned {res.pruned_count}."
    )
    return redirect(next_url)
# def enforce_now(request):
#     if not cache.add("bisk:enforce:now", "1", timeout=10):
#         messages.warning(request, "Please wait ~10s between manual enforces.")
#         return redirect("/admin/system/")
#     res = enforce_schedules()
#     messages.success(
#         request,
#         f"Enforced ✓ — started {len(res.started)}, stopped {len(res.stopped)}, "
#         f"desired {res.desired_count}, running {res.running_count}, pruned {res.pruned_count}."
#     )
#     return redirect("/admin/system/")

@dataclass
class GpuInfo:
    name: str
    util: Optional[int]
    mem_used: Optional[int]
    mem_total: Optional[int]


def _gpu_query() -> List[GpuInfo]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=2)
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                name, util, mu, mt = parts

                def _int(x):
                    try:
                        return int(x)
                    except:
                        return None

                gpus.append(GpuInfo(name=name, util=_int(util), mem_used=_int(mu), mem_total=_int(mt)))
        return gpus
    except Exception:
        return []


def _runner_counts():
    th = getattr(settings, "HEARTBEAT_THRESHOLDS", {"online": 15, "stale": 45, "offline": 120})
    now = timezone.now()
    online = stale = offline = 0

    # We attempt to read a flexible heartbeat field; fallback to updated timestamp.
    hb_fields = ("last_heartbeat", "last_heartbeat_at", "hb_at", "updated_at")

    for rp in RunningProcess.objects.all():
        hb = None
        for f in hb_fields:
            hb = getattr(rp, f, None)
            if hb:
                break
        if not hb:
            offline += 1
            continue
        age = (now - hb).total_seconds()
        if age <= th["online"]:
            online += 1
        elif age <= th["stale"]:
            stale += 1
        else:
            offline += 1
    return {"online": online, "stale": stale, "offline": offline}


def _collect_system_info():
    """Single source of truth for system stats used by all views."""
    cache = caches["default"]
    # CPU & load
    cpu_percent = psutil.cpu_percent(interval=0.0)
    try:
        load1, load5, load15 = os.getloadavg()
    except (AttributeError, OSError):
        load1 = load5 = load15 = None
    # RAM & disk
    vm = psutil.virtual_memory()
    du = shutil.disk_usage("/")
    # GPUs & runners
    gpus = _gpu_query()
    runner_counts = _runner_counts()

    # Use the same thresholds everywhere (dashboard + admin)
    hb_thresholds = getattr(settings, "HEARTBEAT_THRESHOLDS", {"online": 15, "stale": 45, "offline": 120}, )


    # ---- Runner Summary rows (latest heartbeat per camera/profile) ----
    procs = (RunningProcess.objects
             .select_related('camera', 'profile')
             .order_by('id'))

    # Subquery: latest heartbeat for this (camera_id, profile_id)
    _latest_hb = (RunnerHeartbeat.objects
                  .filter(camera_id=OuterRef('camera_id'), profile_id=OuterRef('profile_id'))
                  .order_by('-id'))

    # NOTE: RunnerHeartbeat has 'ts' (not 'created_at')
    procs = procs.annotate(
        hb_target_fps=Subquery(_latest_hb.values('target_fps')[:1]),
        hb_snapshot_every=Subquery(_latest_hb.values('snapshot_every')[:1]),
        hb_fps=Subquery(_latest_hb.values('fps')[:1]),     # runner-reported processed/observed fps (if sent)
        hb_processed_fps=Subquery(_latest_hb.values('processed_fps')[:1]),
        hb_ts=Subquery(_latest_hb.values('ts')[:1]),
    )

    runner_rows = []
    now_ts = timezone.now()
    for p in procs:
        # Build a readable label
        if getattr(p, 'camera', None):
            label = f'#{p.id} · {p.camera.name}'
        elif getattr(p, 'profile', None):
            label = f'#{p.id} · Profile {p.profile_id}'
        else:
            label = f'#{p.id}'

        hb_ts = getattr(p, 'hb_ts', None)
        hb_age_s = None
        if hb_ts:
            try:
                hb_age_s = max(0, int((now_ts - hb_ts).total_seconds()))
            except Exception:
                hb_age_s = None

        runner_rows.append({
            "id": p.id,
            "label": label,
            "pid": getattr(p, "pid", None),
            "status": getattr(p, "status", None),
            "target_fps": getattr(p, "hb_target_fps", None),
            "camera_fps": getattr(p, "hb_fps", None),
            "processed_fps": getattr(p, "hb_processed_fps", None),
            "snapshot_every": getattr(p, "hb_snapshot_every", None),
            "hb_ts": hb_ts,
            "hb_age_s": hb_age_s,
        })

    # paused cameras
    try:
        from django.utils.timezone import now
        paused = RunningProcess.objects.filter(
            camera__pause_until__isnull=False, camera__pause_until__gt=now()
        ).count()
    except Exception:
        paused = "n/a"
    ctx = {
        "host": platform.node(),
        "host_name": platform.node(),
        "cpu_percent": cpu_percent,
        "load": (load1, load5, load15),
        "ram": {"used": vm.used, "total": vm.total, "percent": vm.percent},
        "disk": {"used": du.used, "total": du.total, "percent": round(du.used / du.total * 100, 1)},
        "gpus": gpus,
        "runners": runner_counts,
        "runner_rows": runner_rows,
        "paused": paused,
        "enforcer": {
            "last_run": cache.get("enforcer:last_run"),
            "last_ok": cache.get("enforcer:last_ok"),
            "last_error": cache.get("enforcer:last_error"),
            "pid": cache.get("enforcer:running_pid"),
            "interval": getattr(settings, "ENFORCER_INTERVAL_SECONDS", 15),
        },
        "now": timezone.localtime(),
        "hb_thresholds": hb_thresholds,

    }
    return ctx


@staff_member_required
def system_dash(request):
    return render(request, "scheduler/system.html", _collect_system_info())


@staff_member_required
def system_json(request):
    ctx = _collect_system_info()
    data = {
        "cpu_percent": ctx["cpu_percent"],
        "load": {"1m": ctx["load"][0], "5m": ctx["load"][1], "15m": ctx["load"][2]},
        "memory": ctx["ram"],
        "disk_root": ctx["disk"],
        "gpus": [{"name": g.name, "util": g.util, "mem_used": g.mem_used, "mem_total": g.mem_total} for g in
                 ctx["gpus"]],
        "runners": ctx["runners"],
        "paused": ctx["paused"],
        "enforcer": ctx["enforcer"],
        "host": ctx["host"],
        "now": ctx["now"],
    }
    return JsonResponse(data)


@staff_member_required
def admin_system(request):
    """Admin-wrapped version of the system dashboard."""
    ctx = _collect_system_info()
    # enrich with admin context so breadcrumbs/top bar work
    ctx.update(admin.site.each_context(request))
    ctx["title"] = "System status"
    return render(request, "admin/system_dash.html", ctx)


# --- HTMX partial: system panel only ---
from django.views.decorators.http import require_GET

@require_GET
def system_panel_partial(request):
    ctx = _collect_system_info()      # you already have this
    # IMPORTANT: render the *panel* template only (not the whole page)
    return render(request, "scheduler/_system_panel.html", ctx)
