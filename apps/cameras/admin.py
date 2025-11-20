# apps/cameras/admin.py
from django.contrib import admin, messages
from django.utils.safestring import mark_safe
from django.utils.html import format_html
from django.conf import settings
from django.utils import timezone
from datetime import timedelta, datetime, time
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from apps.scheduler.services import enforce_schedules
from .models import Camera
from .utils import snapshot_url_for
from ..scheduler.models import RunningProcess


def _ffprobe_test(url: str, timeout_sec: int = 8):
    """Run ffprobe on the given RTSP URL and return (ok, message)."""
    exe = settings.FFPROBE_PATH
    cmd = [
        exe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate",
        "-of", "default=nw=1",
        "-rtsp_transport", "tcp",
        url,
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        if p.returncode == 0 and p.stdout.strip():
            return True, p.stdout.strip().replace("\n", " | ")
        return False, (p.stderr or "Unknown error / no video stream").strip()
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout_sec}s"
    except Exception as e:
        return False, str(e)


def _apply_pause(request, queryset, until, label: str):
    """Helper to set pause_until, enforce schedules, and show messages."""
    updated = queryset.update(pause_until=until)
    try:
        enforce_schedules()
    except Exception as e:
        messages.error(request, f"{label} {updated} camera(s), but enforce failed: {e}")
        return
    if until:
        until_local = timezone.localtime(until)
        messages.success(
            request,
            f"{label} {updated} camera(s) until {until_local:%Y-%m-%d %H:%M:%S %Z} and enforced schedules."
        )
    else:
        messages.success(request, f"{label} {updated} camera(s) and enforced schedules.")


@admin.action(description="Pause 30 minutes")
def pause_30_min(modeladmin, request, queryset):
    until = timezone.now() + timedelta(minutes=30)
    _apply_pause(request, queryset, until, "Paused")


@admin.action(description="Pause 2 hours")
def pause_2_hours(modeladmin, request, queryset):
    until = timezone.now() + timedelta(hours=2)
    _apply_pause(request, queryset, until, "Paused")


@admin.action(description="Pause until tomorrow 08:00 (Asia/Baghdad)")
def pause_until_tomorrow_08(modeladmin, request, queryset):
    tz = ZoneInfo("Asia/Baghdad")
    now_baghdad = timezone.now().astimezone(tz)
    target_baghdad = datetime.combine(
        now_baghdad.date() + timedelta(days=1),
        time(8, 0),
        tzinfo=tz,
    )
    # Convert back to project TZ
    target_store = target_baghdad.astimezone(timezone.get_current_timezone())
    _apply_pause(request, queryset, target_store, "Paused")


@admin.action(description="Unpause (clear pause_until)")
def unpause_cameras(modeladmin, request, queryset):
    _apply_pause(request, queryset, None, "Unpaused")


@admin.action(description="Restart with latest config")
def restart_with_latest_config(modeladmin, request, queryset):
    # queryset should be Camera (or RunningProcess) depending on where you register it
    from apps.scheduler.services.enforcer import _stop
    for cam in queryset:
        for rp in RunningProcess.objects.filter(camera=cam):
            _stop(rp.pid)
            rp.status = "dead"
            rp.save(update_fields=["status"])
    # NEW: kick the enforcer now so it starts with the fresh spec immediately
    enforce_schedules()


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    fieldsets = (
        (None, {
            "fields": ("name", "rtsp_url", "location", "scan_station", "script_type_default", "is_active",
                       "pause_until"),
        }),
        ("Dashboard tags", {
            "fields": ("usage_tags",),
        }),
        ("Attendance schedule", {
            "fields": ("period_templates",),
            "description": (
                "Optional: if you select one or more periods here, this camera's "
                "stream will only auto-run while an active PeriodOccurrence exists "
                "for those periods. Leave empty to keep the old behaviour "
                "(policy-based only)."
            ),
        }),
        ("Streaming / Policy", {
            "fields": (
                "rtsp_transport",
                "hb_interval",
                "snapshot_every",
                "prefer_camera_over_profile",
            ),
            "description": "Baseline streaming flags that are still policy-level."
        }),
        ("Policy overrides (camera-first when enabled)", {
            "fields": ("target_fps_req", "det_set_req"),
            "classes": ("collapse",),
            "description": "Nullable. Used only when 'prefer_camera_over_profile' is enabled."
        }),

    )

    list_display = ("name", "location", "is_active", "script_type_default", "prefer_camera_over_profile",
                    "target_fps_req", "det_set_req", "pause_until", "snapshot_thumb",)
    list_editable = ("is_active", "pause_until", "prefer_camera_over_profile",)
    search_fields = ("name", "location")
    list_filter = ("script_type_default", "scan_station", "is_active", "usage_tags", "period_templates",)
    filter_horizontal = (
        "usage_tags",
        "period_templates",
    )  # NEW: nice dual list UI for tags

    actions = [
        pause_30_min,
        pause_2_hours,
        pause_until_tomorrow_08,
        unpause_cameras,
        "action_test_rtsp",
        restart_with_latest_config,
    ]

    @admin.action(description="Test RTSP (ffprobe, 8s)")
    def action_test_rtsp(self, request, queryset):
        reports = []
        for cam in queryset:
            ok, msg = _ffprobe_test(cam.rtsp_url)
            icon = "✅" if ok else "❌"
            reports.append(f"{icon} <b>{cam.name}</b>: {msg}")
        messages.info(request, mark_safe("<br>".join(reports) or "No cameras selected."))

    def snapshot_thumb(self, obj):
        """Show the latest snapshot thumbnail for this camera."""
        url = snapshot_url_for(obj.id)
        if not url:
            return "—"
        return format_html(
            '<a href="{0}" target="_blank" title="Open full snapshot">'
            '<img src="{0}" alt="snapshot" '
            'style="height:90px;border-radius:6px;box-shadow:0 0 2px rgba(0,0,0,.25);" />'
            '</a>',
            url
        )

    snapshot_thumb.short_description = "Snapshot"
