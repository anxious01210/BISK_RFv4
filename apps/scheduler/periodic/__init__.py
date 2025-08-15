"""
Compat shim so older code that calls
  apps.scheduler.periodic._pid_alive(...)
keeps working. Real helpers live in apps.scheduler.services.
"""
try:
    # Prefer the canonical implementation from services.py
    from apps.scheduler.services import _pid_alive  # type: ignore
except Exception:
    # Fallback (very small local implementation)
    import psutil


    def _pid_alive(pid) -> bool:  # type: ignore
        try:
            p = psutil.Process(int(pid))
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except Exception:
            return False

__all__ = ["_pid_alive"]
