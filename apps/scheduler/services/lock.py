# apps/scheduler/services/lock.py
import os, fcntl, atexit

# Default path; allow env override without touching Django settings
DEFAULT_LOCK = "/tmp/bisk_enforcer.lock"
LOCK_PATH = os.environ.get("ENFORCER_LOCK_FILE", DEFAULT_LOCK)

# Try to read from Django settings only if configured
try:
    from django.conf import settings  # noqa: F401

    LOCK_PATH = getattr(settings, "ENFORCER_LOCK_FILE", LOCK_PATH)
except Exception:
    pass

_lock_fh = None  # <— IMPORTANT: define the module global


def acquire_enforcer_lock() -> bool:
    global _lock_fh
    if _lock_fh:
        return True
    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
    _lock_fh = open(LOCK_PATH, "w")
    try:
        fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        _lock_fh.close(); _lock_fh = None
        return False
    _lock_fh.truncate(0)
    _lock_fh.write(str(os.getpid()))
    _lock_fh.flush()
    os.fsync(_lock_fh.fileno())
    atexit.register(release_enforcer_lock)
    return True


def release_enforcer_lock() -> None:
    """Release and remove the lock (best-effort)."""
    global _lock_fh
    if not _lock_fh:
        return
    try:
        fcntl.flock(_lock_fh, fcntl.LOCK_UN)
    finally:
        try:
            _lock_fh.close()
        finally:
            _lock_fh = None
            try:
                os.unlink(LOCK_PATH)
            except FileNotFoundError:
                pass


def lock_holder_pid() -> int | None:
    """Read the PID from the lock file (diagnostics)."""
    try:
        with open(LOCK_PATH, "r") as f:
            txt = (f.read() or "").strip()
            return int(txt) if txt.isdigit() else None
    except Exception:
        return None
