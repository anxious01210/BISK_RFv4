from __future__ import annotations
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from django.core.management import BaseCommand
from django.utils.timezone import make_aware

from apps.attendance.models import PeriodTemplate, PeriodOccurrence

TZ = ZoneInfo("Asia/Baghdad")


def _template_active_on(template: PeriodTemplate, dow: int) -> bool:
    """Prefer template.is_active_on(dow) if it exists; otherwise check weekdays_mask"""
    if hasattr(template, "is_active_on") and callable(template.is_active_on):
        return template.is_active_on(dow)
    # Fallback bitmask: Monday=0 … Sunday=6
    return bool(template.weekdays_mask & (1 << dow))


class Command(BaseCommand):
    help = "Upsert PeriodOccurrence for the next N days from enabled PeriodTemplates."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=14)

    def handle(self, *args, **opts):
        days = opts["days"]
        today = date.today()
        created = updated = 0

        for t in PeriodTemplate.objects.filter(is_enabled=True).order_by("order", "id"):
            for offset in range(days):
                d = today + timedelta(days=offset)
                dow = d.weekday()  # 0=Mon
                if not _template_active_on(t, dow):
                    continue

                start_naive = datetime.combine(
                    d, t.start_time
                ) - timedelta(minutes=t.early_grace_minutes or 0)

                end_naive = datetime.combine(
                    d, t.end_time
                ) + timedelta(minutes=t.late_grace_minutes or 0)

                start_dt = make_aware(start_naive, TZ)
                end_dt = make_aware(end_naive, TZ)

                obj, was_created = PeriodOccurrence.objects.update_or_create(
                    template=t, date=d,
                    defaults={
                        "start_dt": start_dt,
                        "end_dt": end_dt,
                        "is_school_day": True,
                    },
                )
                if was_created:
                    created += 1
                else:
                    updated += 1

        self.stdout.write(self.style.SUCCESS(f"Done. created={created}, updated={updated}"))
