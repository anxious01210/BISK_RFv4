# apps/scheduler/management/commands/start_enforce.py
from django.core.management.base import BaseCommand
from apps.scheduler.services import enforce_schedules  # ← public import

class Command(BaseCommand):
    help = "Run enforce_schedules() once and exit."

    def handle(self, *args, **kwargs):
        res = enforce_schedules()  # returns your result object
        self.stdout.write(self.style.SUCCESS(
            f"enforce_schedules() completed: started={len(res.started)} "
            f"stopped={len(res.stopped)} desired={res.desired_count} running(before)={res.running_count}"
        ))
