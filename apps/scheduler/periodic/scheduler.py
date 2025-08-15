# apps/scheduler/services/scheduler.py
import logging, os
from datetime import datetime, timezone as py_tz

from apscheduler.schedulers.background import BackgroundScheduler
from django.conf import settings
from django.core.cache import caches
from django.utils import timezone

log = logging.getLogger(__name__)

def _import_enforce():
    """
    Lazy-import the enforcer to avoid import cycles. Adjust dotted paths here if needed.
    """
    try:
        # Preferred: apps.scheduler.services.enforcer:enforce_schedules
        from apps.scheduler.services.enforcer import enforce_schedules
        return enforce_schedules
    except Exception:
        # Fallback: apps.scheduler.services import enforce_schedules
        from apps.scheduler.services import enforce_schedules  # type: ignore
        return enforce_schedules

def _tick():
    cache = caches["default"]
    now = timezone.now()
    cache.set("enforcer:last_run", now, timeout=None)
    cache.set("enforcer:running_pid", str(os.getpid()), timeout=None)

    enforce_schedules = _import_enforce()
    try:
        enforce_schedules()
        cache.set("enforcer:last_ok", now, timeout=None)
        cache.delete("enforcer:last_error")
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        log.exception("enforce_schedules() failed: %s", msg)
        cache.set("enforcer:last_error", msg[:500], timeout=None)

def start_background_scheduler():
    """
    Create + start a BackgroundScheduler configured for our 15s cadence.
    """
    scheduler = BackgroundScheduler(
        timezone=settings.TIME_ZONE,
        job_defaults={
            "coalesce": True,        # catch-up collapses to one run
            "max_instances": 1,      # don't overlap this job
            "misfire_grace_time": 10
        },
    )
    interval = int(getattr(settings, "ENFORCER_INTERVAL_SECONDS", 15))
    scheduler.add_job(_tick, "interval", seconds=interval, next_run_time=datetime.now(py_tz.utc))
    scheduler.start()
    return scheduler
