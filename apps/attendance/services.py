# apps/attendance/services.py
from datetime import timedelta, datetime
from typing import Optional
from django.utils import timezone
from .models import PeriodTemplate, PeriodOccurrence


def roll_periods(days: int = 7, start_date: Optional[datetime.date] = None) -> int:
    """
    Generate PeriodOccurrence rows for the next `days` days, starting from `start_date` (or today).
    Returns number of occurrences created.
    """
    tz = timezone.get_current_timezone()
    created = 0
    today = start_date or timezone.localdate()
    for i in range(days):
        d = today + timedelta(days=i)
        dow = d.weekday()
        for t in PeriodTemplate.objects.filter(is_enabled=True):
            if not t.is_active_on(dow):
                continue
            sdt = timezone.make_aware(datetime.combine(d, t.start_time), tz) - timedelta(minutes=t.early_grace_minutes)
            edt = timezone.make_aware(datetime.combine(d, t.end_time), tz) + timedelta(minutes=t.late_grace_minutes)
            _, was_created = PeriodOccurrence.objects.get_or_create(
                template=t, date=d, defaults={"start_dt": sdt, "end_dt": edt}
            )
            created += 1 if was_created else 0
    return created
