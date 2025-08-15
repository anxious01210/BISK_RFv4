# apps/scheduler/management/commands/start_enforce.py
from django.core.management.base import BaseCommand
from apps.scheduler.periodic.scheduler import _import_enforce

class Command(BaseCommand):
    help = "Run enforce_schedules() once and exit."

    def handle(self, *args, **kwargs):
        enforce = _import_enforce()
        enforce()
        self.stdout.write(self.style.SUCCESS("enforce_schedules() completed"))
