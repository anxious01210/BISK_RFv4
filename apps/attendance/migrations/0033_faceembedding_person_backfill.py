# apps/attendance/migrations/0033_faceembedding_person_backfill.py
"""Backfill FaceEmbedding.person_id from student via StudentProfile.legacy_student.

M1 of the attendance identity adoption (Phase 1.5 of the Person architecture).

For each FaceEmbedding row whose ``person_id`` is NULL and whose ``student_id``
resolves to a migrated identity Person (via ``StudentProfile.legacy_student``),
set ``person_id`` to that Person.

Properties:
  * Idempotent: ``filter(person__isnull=True)`` skips already-backfilled rows.
  * Batched: ``iterator(chunk_size=500)`` keeps memory constant.
  * One ``save(update_fields=["person_id"])`` per row — short transaction, no
    long table lock.
  * Rows whose Student is not yet migrated to identity remain ``person=None``
    (non-fatal; legacy ``student`` path keeps working).
  * Reverse is a no-op — the column drop happens in a future release, not here.

The actual logic lives in ``_0033_helper`` so that tests can import it
(this module's name starts with a digit and is not importable normally).
"""
from django.db import migrations

from . _0033_helper import forwards, backwards  # noqa: F401


class Migration(migrations.Migration):

    dependencies = [
        ("attendance", "0032_faceembedding_person"),
    ]

    operations = [
        migrations.RunPython(forwards, backwards),
    ]
