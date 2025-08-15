import os, platform, shutil, subprocess, psutil
from dataclasses import dataclass
from typing import Optional, List
from django.http import JsonResponse
from django.conf import settings
# from django.contrib.admin.views.decorators import staff_member_required
# from django.core.cache import caches
# from django.shortcuts import render
from django.utils import timezone
from apps.scheduler.models import RunningProcess  # adjust import if your model path differs
from django.contrib import admin
from django.contrib.admin.views.decorators import staff_member_required
from django.core.cache import caches
from django.shortcuts import render
from apps.scheduler.models import RunningProcess


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
        "cpu_percent": cpu_percent,
        "load": (load1, load5, load15),
        "ram": {"used": vm.used, "total": vm.total, "percent": vm.percent},
        "disk": {"used": du.used, "total": du.total, "percent": round(du.used / du.total * 100, 1)},
        "gpus": gpus,
        "runners": runner_counts,
        "paused": paused,
        "enforcer": {
            "last_run": cache.get("enforcer:last_run"),
            "last_ok": cache.get("enforcer:last_ok"),
            "last_error": cache.get("enforcer:last_error"),
            "pid": cache.get("enforcer:running_pid"),
            "interval": getattr(settings, "ENFORCER_INTERVAL_SECONDS", 15),
        },
        "now": timezone.localtime(),
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
