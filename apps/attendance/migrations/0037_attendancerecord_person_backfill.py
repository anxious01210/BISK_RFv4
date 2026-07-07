# apps/attendance/migrations/0037_attendancerecord_person_backfill.py
"""Backfill AttendanceRecord.person_id from student via StudentProfile.legacy_student.

M3 of the attendance identity adoption (Phase 1.5 of the Person architecture).

For each AttendanceRecord row whose ``person_id`` is NULL and whose ``student_id``
resolves to a migrated identity Person (via ``StudentProfile.legacy_student``),
set ``person_id`` to that Person.

Properties:
  * Idempotent: ``filter(person__isnull=True)`` skips already-backfilled rows.
  * Batched: ``iterator(chunk_size=500)`` keeps memory constant; one
    ``update(pk__in=...)`` per person_id group per batch — short transactions,
    no long table lock.
  * Rows whose Student is not yet migrated to identity remain ``person=None``
    (non-fatal; legacy ``student`` path keeps working).
  * Reverse is a no-op — the column drop happens in a future release, not here.

The actual logic lives in ``_0037_helper`` so that tests can import it
(this module's name starts with a digit and is not importable normally).
"""
from django.db import migrations

from . _0037_helper import forwards, backwards  # noqa: F401


class Migration(migrations.Migration):

    dependencies = [
        ("attendance", "0036_attendancerecord_person"),
    ]

    operations = [
        migrations.RunPython(forwards, backwards),
    ]
