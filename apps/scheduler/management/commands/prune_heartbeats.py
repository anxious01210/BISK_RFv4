from django.core.management.base import BaseCommand
from django.utils import timezone
from apps.scheduler.models import RunnerHeartbeat
from datetime import timedelta


class Command(BaseCommand):
    help = "Delete runner heartbeats older than the given age (days/hours/minutes)."

    def add_arguments(self, parser):

        # Backwards compatible long flags  short double-dash aliases: --d, --h, --m
        parser.add_argument("--days", "--d", type=int, default=7, help="Age in days (default: 7)")
        parser.add_argument("--hours", "--hrs", "--h", dest="hours", type=int, default=0,
                            help="Additional age in hours (default: 0)")
        parser.add_argument("--minutes", "--min", "--m", dest="minutes", type=int, default=0,
                            help="Additional age in minutes (default: 0)")

    def handle(self, *args, **opts):

        days = max(0, int(opts.get("days", 0)))
        hours = max(0, int(opts.get("hours", 0)))
        minutes = max(0, int(opts.get("minutes", 0)))

        cutoff = timezone.now() - timedelta(days=days, hours=hours, minutes=minutes)
        qs = RunnerHeartbeat.objects.filter(ts__lt=cutoff)

        # .delete() returns (num_deleted, details)
        deleted, _ = qs.delete()

        # Human-readable age string
        parts = []
        if days: parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours: parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes or not parts:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        age_str = ", ".join(parts)

        self.stdout.write(self.style.SUCCESS(f"Deleted {deleted} heartbeat rows older than {age_str}."))
