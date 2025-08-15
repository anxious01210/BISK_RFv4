# apps/scheduler/management/commands/run_enforcer.py
import fcntl
import os
import signal
import sys
import time

from django.conf import settings
from django.core.management.base import BaseCommand

from apps.scheduler.periodic.scheduler import start_background_scheduler

class Command(BaseCommand):
    help = "Run the BISK periodic enforcer (APScheduler) with a single-instance file lock."

    def add_arguments(self, parser):
        parser.add_argument("--lock-file", default=None, help="Override lock file path")

    def handle(self, *args, **opts):
        lock_path = opts["lock_file"] or getattr(settings, "ENFORCER_LOCK_FILE", "/tmp/bisk_enforcer.lock")
        # Ensure the lock directory exists
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)

        with open(lock_path, "w+") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self.stderr.write(self.style.WARNING(f"Another enforcer holds the lock: {lock_path}"))
                sys.exit(0)

            lock_file.write(str(os.getpid()))
            lock_file.flush()

            scheduler = start_background_scheduler()
            self.stdout.write(self.style.SUCCESS("Enforcer started. Press Ctrl+C to stop."))

            # Graceful shutdown
            def _shutdown(signum, frame):
                try:
                    scheduler.shutdown(wait=False)
                finally:
                    try:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)  # release
                    except Exception:
                        pass
                    sys.exit(0)

            signal.signal(signal.SIGINT, _shutdown)
            signal.signal(signal.SIGTERM, _shutdown)

            # Sleep-loop to keep process alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                _shutdown(None, None)
