# bisk/views.py
from django.contrib.admin.views.decorators import staff_member_required
from django.http import JsonResponse
import psutil, shutil, os
from django.shortcuts import render
from django.utils import timezone
from apps.scheduler.models import RunnerHeartbeat
from apps.cameras.models import Camera
from datetime import datetime


def portal_home(request):
    u = request.user

    can_admin = (
            u.is_authenticated and (
            u.is_superuser
            or u.groups.filter(name="supervisor").exists()
            or u.is_staff
    )
    )
    can_lunch = u.is_authenticated and (
            u.groups.filter(name="lunch_supervisor").exists()
            or can_admin
    )
    can_system = u.is_authenticated and can_admin  # adjust if you want broader access

    return render(request, "core/home.html", {
        "year": datetime.now().year,
        "can_admin": can_admin,
        "can_lunch": can_lunch,
        "can_system": can_system,
    })


@staff_member_required
def dashboard(request):
    return render(request, "dashboard.html")


def cameras_dashboard(request):
    now = timezone.now()
    # keep only the latest heartbeat per (camera, profile)
    latest = {}
    qs = RunnerHeartbeat.objects.select_related("camera", "profile").order_by("camera__name", "-ts")
    for hb in qs:
        key = (hb.camera_id, hb.profile_id)
        if key not in latest:
            latest[key] = hb

    grouped = {}
    for hb in latest.values():
        age = (now - hb.ts).total_seconds()
        status = "online" if age <= 15 else ("stale" if age <= 45 else "offline")
        grouped.setdefault(hb.camera, []).append({
            "hb": hb,
            "age": int(age),
            "status": status,
        })

    return render(request, "dashboard_cameras.html", {"grouped": grouped, "now": now})


def system_stats(request):
    # CPU & load
    cpu_percent = psutil.cpu_percent(interval=None)
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1 = load5 = load15 = None

    # Memory
    vm = psutil.virtual_memory()
    mem = {"total": vm.total, "used": vm.used, "percent": vm.percent}

    # Disk (root)
    du = shutil.disk_usage("/")
    disk = {"total": du.total, "used": du.used, "percent": round(du.used / du.total * 100, 2)}

    # GPU (optional; safe if no NVIDIA/driver)
    gpus = []
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h).decode("utf-8", errors="ignore")
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(h)
            gpus.append({
                "index": i,
                "name": name,
                "mem_total": meminfo.total,
                "mem_used": meminfo.used,
                "mem_percent": round(meminfo.used / meminfo.total * 100, 2) if meminfo.total else 0.0,
            })
        pynvml.nvmlShutdown()
    except Exception:
        gpus = []

    return JsonResponse({
        "cpu_percent": cpu_percent,
        "load": {"1m": load1, "5m": load5, "15m": load15},
        "memory": mem,
        "disk_root": disk,
        "gpus": gpus,
    })
