#!/usr/bin/env bash
# Seed lunch AttendanceRecord data for testing (RESPECTS unique (student, period))
# Usage: bash scripts/seed_lunch_dummy.sh [path_to_manage_py] [camera_name]
# Example: bash scripts/seed_lunch_dummy.sh ./manage.py Dept_01
set -euo pipefail

MANAGE_PY="${1:-./manage.py}"
CAMERA_NAME="${2:-C01}"
export SEED_CAMERA_NAME="$CAMERA_NAME"

python "$MANAGE_PY" shell <<'PY'
from datetime import datetime, timedelta, time
import os, random, sys
from django.utils import timezone
from django.db import transaction
from django.apps import apps as djapps

# --- get models by app label (independent of module path) ---
def M(app_label, model_name):
    try:
        return djapps.get_model(app_label, model_name)
    except LookupError:
        return None

AttendanceRecord = M('attendance', 'AttendanceRecord')
Student          = M('attendance', 'Student')
PeriodTemplate   = M('attendance', 'PeriodTemplate')
PeriodOccurrence = M('attendance', 'PeriodOccurrence')
Camera           = M('attendance', 'Camera')

if AttendanceRecord is None or Student is None:
    print("[ERR] Could not resolve models in app label 'attendance'. Check INSTALLED_APPS / AppConfig.label.", file=sys.stderr)
    raise SystemExit(1)

HCODES = [
"H210237","H250036","H230083","H230020","H210008","H210007",
"H250013","H180126","H190018","H180125","H250029","H220284",
"H210262","H210261","H210260","H250047","H250046","H210014"
]
PERIOD_NAMES = ["lunch-pri","lunch-sec"]  # your view filters by period__template__name
# views.py shows period filter by template name and ordering/paging on best_seen/newest-first. :contentReference[oaicite:2]{index=2}

tz    = timezone.get_current_timezone()
now   = timezone.localtime()
today = now.date()

aware = lambda dt: timezone.make_aware(dt, tz)

def at_local(h, m=0, s=0): return aware(datetime(today.year, today.month, today.day, h, m, s))

# Realistic lunch window (HH:MM:SS renders nicely in the template). :contentReference[oaicite:3]{index=3}
WINDOW_START = at_local(13, 0, 0)
WINDOW_END   = at_local(13, 59, 0)
clamp = lambda dt: max(WINDOW_START, min(WINDOW_END, dt))

def get_or_create_period_for_today(name: str):
    """
    Ensure a PeriodTemplate named `name` and today's PeriodOccurrence spanning the day.
    Your rows view filters by template name (period__template__name). :contentReference[oaicite:4]{index=4}
    """
    if PeriodTemplate and PeriodOccurrence:
        tpl, _ = PeriodTemplate.objects.get_or_create(
            name=name,
            defaults=dict(order=1, start_time=time(0,0), end_time=time(23,59), weekdays_mask=127, is_enabled=True)
        )
        occ, _ = PeriodOccurrence.objects.get_or_create(
            template=tpl, date=today,
            defaults=dict(
                start_dt=aware(datetime(today.year, today.month, today.day, 0, 0, 0)),
                end_dt=aware(datetime(today.year, today.month, today.day, 23, 59, 0)),
                is_enabled=True
            )
        )
        return occ
    return None  # If your schema differs, the record inserts still happen (but the default period filter may hide them)

cam = Camera.objects.filter(name=os.environ.get("SEED_CAMERA_NAME","C01")).first() if Camera else None

upserted = 0
with transaction.atomic():
    for i, h in enumerate(HCODES):
        st, _ = Student.objects.get_or_create(
            h_code=h,
            defaults=dict(first_name=f"Student{h[-3:]}", last_name="Dummy", grade=9, has_lunch=True)
        )
        for p_name in PERIOD_NAMES:
            per = get_or_create_period_for_today(p_name)

            # --- single row per (student, period) → respect unique constraint ---
            # Generate realistic times & values
            base = i % 40
            dt_best = clamp(WINDOW_START + timedelta(minutes=base, seconds=random.randint(0, 40)))
            dt_first = dt_best - timedelta(seconds=random.randint(5, 25))
            dt_last  = dt_best + timedelta(seconds=random.randint(5, 25))
            dt_pass  = dt_best + timedelta(seconds=random.randint(10, 50))
            best_score = round(random.uniform(0.43, 0.95), 2)
            pass_count = random.choice([1,1,1,2])

            defaults = dict(
                first_seen=dt_first,
                best_seen=dt_best,
                last_seen=dt_last,
                last_pass_at=dt_pass,
                best_score=best_score,
                pass_count=pass_count,
                confirmed=False,
            )
            if cam: defaults["best_camera"] = cam

            if per is not None:
                rec, created = AttendanceRecord.objects.update_or_create(
                    student=st, period=per, defaults=defaults
                )
            else:
                # If period occurrences aren't used in your schema, try without period to avoid hiding rows.
                rec, created = AttendanceRecord.objects.update_or_create(
                    student=st, defaults=defaults
                )
                if cam and getattr(rec, "best_camera_id", None) != getattr(cam, "id", None):
                    rec.best_camera = cam
                    rec.save(update_fields=["best_camera"])

            upserted += 1

print(f"[OK] upserted {upserted} AttendanceRecord row(s) (1 per student×period).")
PY


# chmod +x scripts/seed_lunch_dummy.sh
# bash scripts/seed_lunch_dummy.sh ./manage.py Dept_01