# apps/attendance/migrations/_0033_helper.py
"""Helper functions for the 0033 FaceEmbedding.person_id backfill migration.

Lives outside the ``0033_faceembedding_person_backfill.py`` module (whose name
starts with a digit and is therefore not importable via normal Python imports)
so that tests can import ``forwards`` / ``backwards`` directly.

Django's migration loader ignores modules starting with an underscore, so this
file is not treated as a migration.
"""


def forwards(apps, schema_editor):
    FaceEmbedding = apps.get_model("attendance", "FaceEmbedding")
    StudentProfile = apps.get_model("identity", "StudentProfile")

    # Build legacy_student_id -> person_id mapping (loaded once, small).
    mapping = dict(
        StudentProfile.objects.exclude(legacy_student__isnull=True)
        .values_list("legacy_student_id", "person_id")
    )
    if not mapping:
        # No students have been migrated to identity yet; nothing to backfill.
        return

    qs = (
        FaceEmbedding.objects
        .filter(person__isnull=True)
        .exclude(student__isnull=True)
        .only("id", "student_id")
    )

    for row in qs.iterator(chunk_size=500):
        person_id = mapping.get(row.student_id)
        if person_id is None:
            # Student not yet migrated to identity; leave person NULL.
            continue
        row.person_id = person_id
        row.save(update_fields=["person_id"])


def backwards(apps, schema_editor):
    # Noop reverse — the person_id column is dropped by the reverse of
    # migration 0032 (RemoveField). We deliberately do NOT null out
    # person_id here so that a rollback-then-re-apply of 0033 is a no-op
    # (idempotent), and a rollback of 0032 cleanly drops the column.
    pass
