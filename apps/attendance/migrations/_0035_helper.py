# apps/attendance/migrations/_0035_helper.py
"""Helper functions for the 0035 AttendanceEvent.person_id backfill migration.

Lives outside the ``0035_attendanceevent_person_backfill.py`` module (whose
name starts with a digit and is therefore not importable via normal Python
imports) so that tests can import ``forwards`` / ``backwards`` directly.

Django's migration loader ignores modules starting with an underscore, so
this file is not treated as a migration.

Strategy:
  * Build a ``legacy_student_id -> person_id`` mapping once (small, in-memory).
  * Iterate ``AttendanceEvent`` rows needing backfill
    (``person IS NULL AND student IS NOT NULL``) in batches of 500 using
    ``iterator(chunk_size=500)`` so memory stays constant.
  * For each batch, group rows by ``person_id`` and issue one
    ``update(pk__in=ev_ids, person_id=person_id)`` per group. This avoids
    per-row ``save()`` and minimizes write traffic on the (potentially
    large) append-only events table.
  * Rows whose Student is not yet migrated to identity are skipped —
    ``person`` stays NULL (non-fatal; events are append-only audit).

Properties:
  * Idempotent: ``filter(person__isnull=True)`` skips already-backfilled rows.
  * Reversible: ``backwards`` is a no-op (the column drop happens in a future
    release, not here).
  * No long table lock: short transactions per batch via ``update()``.
"""


BATCH_SIZE = 500


def forwards(apps, schema_editor):
    AttendanceEvent = apps.get_model("attendance", "AttendanceEvent")
    StudentProfile = apps.get_model("identity", "StudentProfile")

    # Build legacy_student_id -> person_id mapping (loaded once, small).
    mapping = dict(
        StudentProfile.objects
        .exclude(legacy_student__isnull=True)
        .exclude(person__isnull=True)
        .values_list("legacy_student_id", "person_id")
    )
    if not mapping:
        # No students have been migrated to identity yet; nothing to backfill.
        return

    qs = (
        AttendanceEvent.objects
        .filter(person__isnull=True)
        .exclude(student__isnull=True)
        .only("id", "student_id")
    )

    # Group rows by person_id within a batch window, then flush with one
    # update() per group. This keeps memory constant (chunk_size=500) while
    # minimizing the number of UPDATE statements.
    pending_by_person = {}  # person_id -> [event_id, ...]

    for row in qs.iterator(chunk_size=BATCH_SIZE):
        person_id = mapping.get(row.student_id)
        if person_id is None:
            # Student not yet migrated to identity; leave person NULL.
            continue
        pending_by_person.setdefault(person_id, []).append(row.pk)

        # Flush when we have collected at least BATCH_SIZE pending ids total.
        total = sum(len(v) for v in pending_by_person.values())
        if total >= BATCH_SIZE:
            _flush(AttendanceEvent, pending_by_person)
            pending_by_person.clear()

    # Final flush for any remaining rows.
    if pending_by_person:
        _flush(AttendanceEvent, pending_by_person)


def _flush(model, pending_by_person):
    """Issue one ``update(pk__in=ev_ids, person_id=...)`` per person_id group."""
    for person_id, ev_ids in pending_by_person.items():
        model.objects.filter(pk__in=ev_ids).update(person_id=person_id)


def backwards(apps, schema_editor):
    # Noop reverse — the person_id column is dropped by the reverse of
    # migration 0034 (RemoveField). We deliberately do NOT null out
    # person_id here so that a rollback-then-re-apply of 0035 is a no-op
    # (idempotent), and a rollback of 0034 cleanly drops the column.
    pass
