# apps/scheduler/services/scheduler.py
import logging, os, time
from apscheduler.schedulers.background import BackgroundScheduler
from django.conf import settings
from django.core.cache import caches
from django.utils import timezone
from .lock import acquire_enforcer_lock, release_enforcer_lock, lock_holder_pid

log = logging.getLogger(__name__)


def _import_enforce():
    """
    Import enforce_schedules() without creating circular imports.
    Adjust the import path only if your project stores it elsewhere.
    """
    # Preferred: the function exposed by your mgmt command module
    # from apps.scheduler.management.commands.enforce_schedules import enforce_schedules
    # return enforce_schedules
    from apps.scheduler.services.enforcer import enforce_schedules
    return enforce_schedules


def _tick():
    """
    One APScheduler tick:
      - set cache heartbeat/metadata keys
      - call enforce_schedules()
      - record last_ok/last_error
    Keys are read by /admin/system/ (“Enforcer” box).
    """
    cache = caches["default"]
    now = timezone.now()

    # heartbeat: who’s running and when
    cache.set("enforcer:last_run", now, timeout=None)
    cache.set("enforcer:running_pid", str(os.getpid()), timeout=None)

    enforce = _import_enforce()
    try:
        enforce()
        cache.set("enforcer:last_ok", now, timeout=None)
        cache.set("enforcer:last_error", None, timeout=None)
    except Exception as e:
        # keep going; record the error so the admin widget shows it
        cache.set("enforcer:last_error", str(e), timeout=None)
        log.exception("enforce_schedules() failed")


def start_background_scheduler():
    """
    Boot the background APScheduler and keep the process alive.
    Single-instance enforced by a POSIX file lock.
    """
    if not acquire_enforcer_lock():
        # Another enforcer is already running; no-op.
        try:
            holder = lock_holder_pid()
            log.info("Enforcer not started; lock held by pid=%s", holder)
        except Exception:
            pass
        return False

    interval = int(getattr(settings, "ENFORCER_INTERVAL_SECONDS", 15))
    tz = str(timezone.get_current_timezone())

    sched = BackgroundScheduler(timezone=tz)
    # Coalesce missed ticks into one run; avoid overlapping ticks.
    sched.add_job(
        _tick,
        trigger="interval",
        seconds=interval,
        id="bisk_enforcer",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=interval,
        replace_existing=True,
    )

    sched.start()
    log.info("BISK enforcer APScheduler started: interval=%ss tz=%s pid=%s",
             interval, tz, os.getpid())

    # Block so systemd (or the mgmt command) keeps the process alive.
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sched.shutdown(wait=False)
        except Exception:
            pass
        release_enforcer_lock()
        log.info("BISK enforcer APScheduler stopped")
    return True