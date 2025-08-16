from django.core.management.base import BaseCommand
from apps.scheduler.services.scheduler import start_background_scheduler
from apps.scheduler.services.lock import LOCK_PATH, lock_holder_pid


class Command(BaseCommand):
    help = "Start the background enforcer (idempotent)."

    def handle(self, *args, **opts):
        started = start_background_scheduler()
        if not started:
            holder = lock_holder_pid()
            self.stderr.write(
                f"Another enforcer holds the lock: {LOCK_PATH}"
                + (f" (pid={holder})" if holder else "")
            )
            raise SystemExit(1)  # exit non-zero so '&& echo started' will NOT run
        # If started=True, start_background_scheduler() blocks until stopped.
