# apps/scheduler/services/__init__.py
from .lock import acquire_enforcer_lock, release_enforcer_lock, lock_holder_pid
"""
Package facade for scheduler services.

Keeps compatibility with older imports like:
    from apps.scheduler.services import _stop, _pid_alive, enforce_schedules
and exposes the APScheduler bootstrap:
    from apps.scheduler.services import start_background_scheduler
"""

def _stop(pid: int, deadline: float = 3.0):
    from .enforcer import _stop as impl
    return impl(pid, deadline)

def _pid_alive(pid: int) -> bool:
    from .enforcer import _pid_alive as impl
    return impl(pid)

def enforce_schedules(*args, **kwargs):
    from .enforcer import enforce_schedules as impl
    return impl(*args, **kwargs)

def start_background_scheduler(*args, **kwargs):
    from .scheduler import start_background_scheduler as impl
    return impl(*args, **kwargs)
