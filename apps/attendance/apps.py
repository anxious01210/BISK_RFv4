from django.apps import AppConfig
from django.db.models.signals import post_migrate


class AttendanceConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.attendance"
    verbose_name = "Attendance"

    def ready(self):
        import sys
        argv = " ".join(sys.argv).lower()
        if any(k in argv for k in ("makemigrations", "migrate", "collectstatic", "test")):
            return

        from .signals import ensure_media_tree_signal
        post_migrate.connect(ensure_media_tree_signal, sender=self)

        from . import signals  # noqa: F401