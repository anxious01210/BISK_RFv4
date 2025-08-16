# apps/scheduler/apps.py
import os, sys
from django.apps import AppConfig

EXCLUDE_CMDS = {"migrate", "makemigrations", "collectstatic", "shell", "dbshell", "check", "test"}


class SchedulerConfig(AppConfig):
    name = "apps.scheduler"
    verbose_name = "Scheduler"

    def ready(self):
        # Don't start during one-off mgmt commands
        cmd = sys.argv[1] if len(sys.argv) > 1 else ""
        if cmd in EXCLUDE_CMDS:
            return
        # Dev server: only start in the reloader child (avoid double-start)
        if cmd == "runserver" and os.environ.get("RUN_MAIN") != "true":
            return

        # Try to start — real single-instance guard lives inside start_background_scheduler()
        try:
            from .services.scheduler import start_background_scheduler
            start_background_scheduler()
        except Exception:
            # Optional: log, but never crash app startup
            pass
