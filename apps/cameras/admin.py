# apps/cameras/admin.py
from django.contrib import admin, messages
from django.utils.safestring import mark_safe
from django.utils.html import format_html
from django.conf import settings
from django.utils import timezone
from datetime import timedelta
from pathlib import Path
import subprocess
from apps.scheduler.services import enforce_schedules
from .models import Camera


def _ffprobe_test(url: str, timeout_sec: int = 8):
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


@admin.action(description="Pause 30 minutes")
def pause_30_min(modeladmin, request, queryset):
    until = timezone.now() + timedelta(minutes=30)
    updated = queryset.update(pause_until=until)
    # Optional: call your scheduler.enforce_schedules() here if you want instant stop
    try:
        enforce_schedules()
    except Exception as e:
        messages.error(request, f"Paused {updated} camera(s), but enforce failed: {e}")
        return

    # show the timestamp in your local timezone (UTC+3)
    until_local = timezone.localtime(until)
    messages.success(
        request,
        f"Paused {updated} camera(s) until {until_local:%Y-%m-%d %H:%M:%S %Z} and enforced schedules."
    )


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = (
        "name", "location", "script_type_default", "scan_station", "is_active",
        "rtsp_transport", "hwaccel", "device", "gpu_index",
        "cpu_affinity", "nice", "hb_interval", "snapshot_every",
        "pause_until", "snapshot_thumb",
    )
    list_editable = (
        "rtsp_transport", "hwaccel", "device", "gpu_index",
        "cpu_affinity", "nice", "hb_interval", "snapshot_every",
    )
    search_fields = ("name", "location")
    list_filter = ("script_type_default", "scan_station", "is_active")
    actions = [pause_30_min, "action_test_rtsp", ]

    @admin.action(description="Test RTSP (ffprobe, 8s)")
    def action_test_rtsp(self, request, queryset):
        reports = []
        for cam in queryset:
            ok, msg = _ffprobe_test(cam.rtsp_url)
            icon = "✅" if ok else "❌"
            reports.append(f"{icon} <b>{cam.name}</b>: {msg}")
        messages.info(request, mark_safe("<br>".join(reports) or "No cameras selected."))

    def snapshot_thumb(self, obj):
        snap_path = Path(settings.SNAPSHOT_DIR) / f"{obj.id}.jpg"
        if not snap_path.exists():
            return "—"
        try:
            mtime = int(snap_path.stat().st_mtime)
        except Exception:
            mtime = 0
        url = f"{settings.MEDIA_URL}snapshots/{obj.id}.jpg?v={mtime}"
        return format_html(
            '<a href="{}" target="_blank" title="Open full snapshot">'
            '<img src="{}" alt="snapshot" '
            'style="height:90px;border-radius:6px;box-shadow:0 0 2px rgba(0,0,0,.25);" />'
            '</a>',
            url, url
        )

    snapshot_thumb.short_description = "Snapshot"
