# apps/cameras/admin.py
from django.contrib import admin, messages
from django.utils.safestring import mark_safe
from django.conf import settings
import subprocess, shutil

from .models import Camera


def _ffprobe_test(url: str, timeout_sec: int = 8):
    exe = settings.FFPROBE_PATH  # Linux fixed path
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


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ("name", "location", "script_type_default", "scan_station", "is_active")
    search_fields = ("name", "location")
    list_filter = ("script_type_default", "scan_station", "is_active")
    actions = ["action_test_rtsp"]

    @admin.action(description="Test RTSP (ffprobe, 6s)")
    def action_test_rtsp(self, request, queryset):
        reports = []
        for cam in queryset:
            ok, msg = _ffprobe_test(cam.rtsp_url)
            icon = "✅" if ok else "❌"
            reports.append(f"{icon} <b>{cam.name}</b>: {msg}")
        messages.info(request, mark_safe("<br>".join(reports) or "No cameras selected."))
