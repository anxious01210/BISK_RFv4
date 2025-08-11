from django.core.management.base import BaseCommand
from django.utils import timezone
from apps.scheduler.models import RunnerHeartbeat
from datetime import timedelta


class Command(BaseCommand):
    help = "Delete runner heartbeats older than --days (default 7)."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=7)

    def handle(self, *args, **opts):
        cutoff = timezone.now() - timedelta(days=opts["days"])
        qs = RunnerHeartbeat.objects.filter(ts__lt=cutoff)
        n = qs.count()
        qs.delete()
        self.stdout.write(self.style.SUCCESS(f"Deleted {n} heartbeats older than {opts['days']} days."))
