from __future__ import annotations
# apps/attendance/services.py

from datetime import timedelta, datetime
from typing import Optional, Tuple
from django.db import transaction
from django.utils.timezone import now
from django.utils import timezone

from .models import (
    Student,
    PeriodOccurrence,
    AttendanceEvent,
    AttendanceRecord,
    RecognitionSettings,
    PeriodTemplate,
)


def _resolve_open_occurrence(ts):
    # Finds the occurrence whose window contains ts
    return (
        PeriodOccurrence.objects
        .filter(is_school_day=True, start_dt__lte=ts, end_dt__gte=ts)
        .order_by("start_dt")
        .first()
    )


@transaction.atomic
def record_recognition(
        *,
        student: Student,
        score: float,
        camera: str,
        ts=None,
        crop_path: Optional[str] = None
) -> Tuple[AttendanceEvent, Optional[AttendanceRecord]]:
    """
    Append an AttendanceEvent and, if above threshold and inside a period window,
    create/upgrade the AttendanceRecord for (student, occurrence).

    Returns: (event, record_or_none)
    """
    ts = ts or now()
    cfg = RecognitionSettings.get_solo()

    min_score = cfg.min_score
    improve_delta = cfg.min_improve_delta or 0.0
    window = timedelta(seconds=cfg.re_register_window_sec or 0)

    occ = _resolve_open_occurrence(ts)

    # Always write the event (period may be None if no open occurrence)
    ev = AttendanceEvent.objects.create(
        student=student,
        period=occ,
        camera=camera,
        ts=ts,
        score=score,
        crop_path=crop_path,
    )

    # If below threshold or no matching period window, stop after event.
    if score < min_score or not occ:
        return ev, None

    # Enforce close-out: do not upgrade past end_dt + window (but creation inside window is OK).
    allow_upgrades_until = occ.end_dt + window

    # Enforce optional "max per day" cap BEFORE creating a new Record.
    # NOTE: If the record for THIS occurrence already exists, we will update it (cap is about new periods in the same day).
    cap = getattr(cfg, "max_periods_per_day", None)
    if cap:
        present_today = (
            AttendanceRecord.objects
            .filter(student=student, period__date=occ.date, status="present")
            .exclude(period=occ)  # exclude this occurrence (we're about to create/update it anyway)
            .count()
        )
        if present_today >= cap:
            # Mark overflow for this occurrence (record exists or not)
            rec, _ = AttendanceRecord.objects.get_or_create(
                student=student, period=occ,
                defaults={"status": "overflow"}
            )
            # If a 'present' record already exists, we do not downgrade it here.
            if rec.status != "present":
                rec.status = "overflow"
                rec.save(update_fields=["status"])
            return ev, rec

    # Create or update the single record for (student, occurrence)
    rec, created = AttendanceRecord.objects.get_or_create(
        student=student, period=occ,
        defaults=dict(
            first_seen=ts,
            last_seen=ts,
            best_seen=ts,
            best_score=score,
            best_camera=camera,
            best_crop=crop_path,
            sightings=1,
            status="present",
        )
    )

    if created:
        return ev, rec

    # Update existing record
    # Always bump last_seen and sightings
    rec.last_seen = ts
    rec.sightings = (rec.sightings or 0) + 1

    # Only upgrade best_* inside window and if score is materially better
    if ts <= allow_upgrades_until and (score >= (rec.best_score or 0) + improve_delta):
        rec.best_score = score
        rec.best_seen = ts
        rec.best_camera = camera
        if crop_path:
            # Respect delete_old_cropped if you later implement async deletion
            rec.best_crop = crop_path

    rec.save(update_fields=["last_seen", "sightings", "best_score", "best_seen", "best_camera", "best_crop"])
    return ev, rec


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
