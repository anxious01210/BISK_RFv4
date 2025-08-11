from django.core.management.base import BaseCommand
from apps.attendance.services import roll_periods


class Command(BaseCommand):
    help = "Generate PeriodOccurrence rows."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=7)

    def handle(self, *args, **opts):
        created = roll_periods(days=opts["days"])
        self.stdout.write(self.style.SUCCESS(f"Created {created} period occurrences"))

# # apps/attendance/management/commands/roll_periods.py
# from django.core.management.base import BaseCommand
# from django.utils import timezone
# from datetime import timedelta, datetime
# from apps.attendance.models import PeriodTemplate, PeriodOccurrence
#
#
# class Command(BaseCommand):
#     help = "Generate PeriodOccurrence rows for the next N days."
#
#     def add_arguments(self, parser):
#         parser.add_argument("--days", type=int, default=7)
#
#     def handle(self, *args, **opts):
#         tz = timezone.get_current_timezone()
#         today = timezone.localdate()
#         for i in range(opts["days"]):
#             d = today + timedelta(days=i)
#             dow = d.weekday()
#             for t in PeriodTemplate.objects.filter(is_enabled=True):
#                 if not t.is_active_on(dow): continue
#                 sdt = timezone.make_aware(datetime.combine(d, t.start_time), tz) - timedelta(
#                     minutes=t.early_grace_minutes)
#                 edt = timezone.make_aware(datetime.combine(d, t.end_time), tz) + timedelta(minutes=t.late_grace_minutes)
#                 PeriodOccurrence.objects.get_or_create(template=t, date=d, defaults={"start_dt": sdt, "end_dt": edt})
