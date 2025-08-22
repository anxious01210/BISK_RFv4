# apps/attendance/apps.py
from django.apps import AppConfig

class AttendanceConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = 'apps.attendance'
    verbose_name = "Attendance"

    def ready(self):
        # Avoid touching filesystem when Django builds migrations or checks
        import sys
        argv = " ".join(sys.argv).lower()
        if any(k in argv for k in ("makemigrations", "migrate", "collectstatic", "test")):
            return
        try:
            from .utils.media_paths import ensure_media_tree
            ensure_media_tree()
        except Exception:
            # Never crash the app if media path is missing at boot — just defer to manual command
            pass
