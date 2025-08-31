from __future__ import annotations
# apps/attendance/services.py

from django.conf import settings
import os
from datetime import timedelta, datetime
from django.db.models import F
from django.db import transaction
from typing import Optional, Tuple
from django.utils import timezone

from .models import (
    Student,
    PeriodOccurrence,
    AttendanceEvent,
    AttendanceRecord,
    RecognitionSettings,
    PeriodTemplate,
)


# --- Private: resolve today's occurrence that contains ts (inclusive) ---
def _resolve_occurrence(ts):
    ts_local = timezone.localtime(ts if ts else timezone.now())
    from apps.attendance.models import PeriodOccurrence
    return (
        PeriodOccurrence.objects
        .filter(
            is_school_day=True,
            start_dt__lte=ts_local,
            end_dt__gte=ts_local,
        )
        .select_related("template")
        .first()
    )


# --- Private: single source of truth for Events + Records upsert ---
@transaction.atomic
def _write_from_match(
        *,
        student,
        camera,  # Camera instance (FK), not a str
        score: float,
        ts=None,
        crop_path: Optional[str] = None,
        use_policies: bool = True,
):
    """
    Creates AttendanceEvent (always), attaches PeriodOccurrence if found,
    and upserts AttendanceRecord(student, period) with first/last/best fields.
    Returns: (event, record_or_None, meta_flags_dict)
    """
    from apps.attendance.models import AttendanceEvent, AttendanceRecord, RecognitionSettings

    ts_local = timezone.localtime(ts if ts else timezone.now())
    rel_crop = _to_media_rel(crop_path)

    # Policies (min_score, re_register_window_sec, min_improve_delta, max_periods_per_day, ...)
    rs = RecognitionSettings.get_solo() if use_policies else None
    if rs and rs.min_score and score is not None and score < float(rs.min_score):
        # Still persist the event for audit, but it won't update the record.
        ev = AttendanceEvent.objects.create(
            student=student, camera=camera, ts=ts_local, score=score, crop_path=rel_crop
        )
        return ev, None, {"accepted": False, "reason": "below_min_score"}

    occ = _resolve_occurrence(ts_local)

    # Always log the event (even if no period window)
    ev = AttendanceEvent.objects.create(
        student=student,
        period=occ,
        camera=camera,
        ts=ts_local,
        score=score,
        crop_path=rel_crop,
    )

    if not occ:
        return ev, None, {"accepted": False, "reason": "no_period"}

    # Optional policy: max periods per day (skip if exceeded)
    if rs and getattr(rs, "max_periods_per_day", None):
        # Count distinct records for this student on this date
        day_records = AttendanceRecord.objects.filter(
            student=student, period__date=occ.date
        ).values_list("period_id", flat=True).distinct().count()
        if day_records >= int(rs.max_periods_per_day):
            return ev, None, {"accepted": False, "reason": "max_periods_reached"}

    # Upsert the per-period record
    rec, created = AttendanceRecord.objects.get_or_create(
        student=student,
        period=occ,
        defaults=dict(
            first_seen=ts_local,
            last_seen=ts_local,
            best_seen=ts_local,
            best_score=score or 0.0,
            best_camera=camera,
            best_crop=rel_crop or "",
            sightings=1,
            status="present",
        ),
    )
    if created:
        return ev, rec, {"accepted": True, "created": True}

    # De-dup / re-register window + improvement delta
    if rs and getattr(rs, "re_register_window_sec", None):
        delta = (ts_local - rec.last_seen).total_seconds()
        if delta < float(rs.re_register_window_sec):
            # only update if score improves by min_improve_delta (if set)
            if getattr(rs, "min_improve_delta", None):
                need = float(rs.min_improve_delta)
                if (score or 0.0) < (rec.best_score or 0.0) + need:
                    # Just update last_seen & sightings; keep best as-is
                    rec.last_seen = max(rec.last_seen, ts_local)
                    rec.sightings = (rec.sightings or 0) + 1
                    rec.save(update_fields=["last_seen", "sightings"])
                    return ev, rec, {"accepted": True, "updated": True, "improved": False}

    # Default: update last_seen, and improve best if score is higher
    improved = False
    if score is not None and (rec.best_score is None or score > rec.best_score):
        rec.best_score = score
        rec.best_seen = ts_local
        rec.best_camera = camera
        if rel_crop:
            rec.best_crop = rel_crop
        improved = True

    rec.last_seen = max(rec.last_seen, ts_local)
    rec.sightings = (rec.sightings or 0) + 1
    rec.status = rec.status or "present"
    rec.save()
    return ev, rec, {"accepted": True, "updated": True, "improved": improved}


def _resolve_open_occurrence(ts):
    # Finds the occurrence whose window contains ts
    return (
        PeriodOccurrence.objects
        .filter(is_school_day=True, start_dt__lte=ts, end_dt__gte=ts)
        .order_by("start_dt")
        .first()
    )

def _to_media_rel(path: str | None) -> str | None:
    if not path:
        return None
    import os
    p = os.path.normpath(str(path))
    mr = os.path.normpath(str(settings.MEDIA_ROOT or ""))

    try:
        if os.path.isabs(p) and mr:
            # Is p inside MEDIA_ROOT?
            if os.path.commonpath([os.path.realpath(p), os.path.realpath(mr)]) == os.path.realpath(mr):
                rel = os.path.relpath(p, mr)
                return rel.replace(os.sep, "/")
    except Exception:
        pass

    # Fallback: if the string contains '/media/', strip up to it
    marker = "/media/"
    i = p.find(marker)
    if i != -1:
        return p[i + len(marker):].lstrip("/")

    return p.replace(os.sep, "/")


@transaction.atomic
def record_recognition(
        *, student: "Student", score: float, camera, ts=None, crop_path: Optional[str] = None
):
    """
    Public entrypoint used by recognizers when they already have a Student instance.
    Normalizes timestamp to localtime, expects `camera` to be a Camera model.
    Returns (AttendanceEvent, AttendanceRecord|None).
    """
    ev, rec, _meta = _write_from_match(
        student=student,
        camera=camera,
        score=score,
        ts=ts,
        crop_path=crop_path,
        use_policies=True,
    )
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


def _get_settings() -> RecognitionSettings:
    # If no row, fall back to defaults of the model
    return RecognitionSettings.objects.first() or RecognitionSettings()


def _find_occurrence(ts) -> Optional[PeriodOccurrence]:
    return (PeriodOccurrence.objects
            .filter(start_dt__lte=ts, end_dt__gte=ts, is_school_day=True)
            .order_by("start_dt")
            .first())


@transaction.atomic
def ingest_match(
        *, h_code: str, score: float, camera, ts=None, crop_path: Optional[str] = None
) -> dict:
    """
    Public entrypoint used by pipelines that receive H-code and need us to resolve the Student.
    Returns a dictionary of flags for quick feedback.
    """
    from apps.attendance.models import Student
    try:
        student = Student.objects.get(h_code=h_code)
    except Student.DoesNotExist:
        return {"ok": False, "error": "student_not_found", "h_code": h_code}

    ev, rec, meta = _write_from_match(
        student=student,
        camera=camera,
        score=score,
        ts=ts,
        crop_path=crop_path,
        use_policies=True,
    )
    return {"ok": True, "event_id": ev.id, "record_id": getattr(rec, "id", None), **meta}
