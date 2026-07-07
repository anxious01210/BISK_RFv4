"""Attendance identity reconciliation — read-only verification.

This module verifies that the ``person`` FK on the three attendance-domain
models (``FaceEmbedding``, ``AttendanceEvent``, ``AttendanceRecord``) is
correctly populated after the M1/M2/M3 dual-FK migrations. It is the
**post-migration verification** counterpart to the backfill migrations
(0033, 0035, 0037) and the dual-write code in ``_write_from_match`` /
``EnrollView``.

**This module is read-only.** No functions write to the database.
They inspect the attendance tables and the identity tables and produce
:class:`AttendanceIdentityReconciliationReport` objects listing any
discrepancies.

Reconciliation checks per model (FaceEmbedding, AttendanceEvent,
AttendanceRecord):

* **missing_person_id:** row has ``person_id IS NULL`` but ``student_id``
  points to a migrated student (StudentProfile.legacy_student exists with
  a non-null person). This is a backfill gap — the backfill migration
  skipped this row because the student was not yet migrated at backfill
  time.
* **mismatched_person_id:** row has ``person_id`` set but it doesn't match
  the Person resolved from ``student → StudentProfile.legacy_student →
  Person``. Indicates data corruption or a student re-linked to a
  different Person after the row was written.
* **orphan_person_no_student:** row has ``person_id`` set but
  ``student_id`` is NULL. Should be impossible (student FK is
  non-nullable), but defensive.
* **dangling_person_id:** row has ``person_id`` pointing to a Person that
  doesn't exist. Should be impossible (FK constraint), but defensive.

Cross-cutting checks:

* **duplicate_person_periods:** two or more ``AttendanceRecord`` rows with
  the same ``(person_id, period_id)`` where ``person_id IS NOT NULL``.
  Violates the 0038 constraint — indicates pre-migration drift or
  constraint creation failure.
* **duplicate_active_embeddings:** two or more active ``FaceEmbedding``
  rows (``is_active=True``) for the same ``person_id``. Violates the
  ``uniq_active_embedding_per_person`` constraint.

Usage::

    from apps.attendance.identity_reconciliation import (
        reconcile_all_attendance_identity,
    )

    report = reconcile_all_attendance_identity()
    print(f"Total issues: {report.total_issues}")
    if report.is_clean:
        print("S9 gate passes — safe to switch upsert key.")
    else:
        print("S9 gate fails — fix issues before switching upsert key.")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from django.apps import apps as django_apps
from django.db.models import Count


# ---------------------------------------------------------------------------
# Dataclasses (immutable — no Django models, no migrations)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttendanceIdentityIssue:
    """A single issue found during reconciliation of one row."""

    issue_type: str
    # "missing_person_id" | "mismatched_person_id"
    # | "orphan_person_no_student" | "dangling_person_id"
    model_name: str
    # "FaceEmbedding" | "AttendanceEvent" | "AttendanceRecord"
    row_id: int
    student_id: Optional[int]
    person_id: Optional[int]
    expected_person_id: Optional[int]
    detail: str


@dataclass(frozen=True)
class ModelReconciliationReport:
    """Per-model reconciliation report."""

    model_name: str
    total_rows: int
    rows_with_person_id: int
    rows_without_person_id: int
    issues: list[AttendanceIdentityIssue] = field(default_factory=list)

    @property
    def issues_found(self) -> int:
        return len(self.issues)

    @property
    def issue_types(self) -> dict[str, int]:
        """Count of issues by type."""
        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.issue_type] = counts.get(issue.issue_type, 0) + 1
        return counts


@dataclass(frozen=True)
class AttendanceIdentityReconciliationReport:
    """Aggregate report across all three attendance models."""

    face_embeddings: ModelReconciliationReport
    attendance_events: ModelReconciliationReport
    attendance_records: ModelReconciliationReport
    duplicate_person_periods: list[tuple[int, int, int]] = field(default_factory=list)
    # (person_id, period_id, count) for AttendanceRecord duplicates
    duplicate_active_embeddings: list[tuple[int, int]] = field(default_factory=list)
    # (person_id, count) for FaceEmbedding active duplicates

    @property
    def total_issues(self) -> int:
        return (
            self.face_embeddings.issues_found
            + self.attendance_events.issues_found
            + self.attendance_records.issues_found
            + len(self.duplicate_person_periods)
            + len(self.duplicate_active_embeddings)
        )

    @property
    def is_clean(self) -> bool:
        """True if zero issues found — the S9 gate condition."""
        return self.total_issues == 0


# ---------------------------------------------------------------------------
# Shared helper: build student_id → person_id mapping
# ---------------------------------------------------------------------------


def _build_student_to_person_mapping() -> dict[int, int]:
    """Build ``{legacy_student_id: person_id}`` for all migrated students.

    Read-only. Small (number of migrated students, typically < 10K).
    Reused across all three model checks to avoid repeated queries.
    """
    StudentProfile = django_apps.get_model("identity", "StudentProfile")
    return dict(
        StudentProfile.objects
        .exclude(legacy_student__isnull=True)
        .exclude(person__isnull=True)
        .values_list("legacy_student_id", "person_id")
    )


# ---------------------------------------------------------------------------
# Per-model reconciliation
# ---------------------------------------------------------------------------


def _reconcile_model(model_name: str, app_label: str, model_cls_name: str,
                     mapping: dict[int, int]) -> ModelReconciliationReport:
    """Generic per-model reconciliation logic.

    Works for any attendance model with ``student`` and ``person`` FKs.
    Read-only — never writes.
    """
    Model = django_apps.get_model(app_label, model_cls_name)
    Person = django_apps.get_model("identity", "Person")

    total = Model.objects.count()
    with_person = Model.objects.exclude(person__isnull=True).count()
    without_person = total - with_person

    issues: list[AttendanceIdentityIssue] = []

    # C1: missing_person_id — person_id IS NULL but student is migrated.
    if mapping:
        missing_qs = (
            Model.objects
            .filter(person__isnull=True)
            .exclude(student__isnull=True)
            .filter(student_id__in=mapping.keys())
            .values_list("id", "student_id")
        )
        for row_id, student_id in missing_qs.iterator(chunk_size=500):
            issues.append(AttendanceIdentityIssue(
                issue_type="missing_person_id",
                model_name=model_name,
                row_id=row_id,
                student_id=student_id,
                person_id=None,
                expected_person_id=mapping[student_id],
                detail=(
                    f"person_id is NULL but student_id={student_id} is "
                    f"migrated (expected person_id={mapping[student_id]})."
                ),
            ))

    # C2: mismatched_person_id — person_id set but doesn't match StudentProfile.
    if mapping:
        mismatched_qs = (
            Model.objects
            .exclude(person__isnull=True)
            .exclude(student__isnull=True)
            .filter(student_id__in=mapping.keys())
            .values_list("id", "student_id", "person_id")
        )
        for row_id, student_id, person_id in mismatched_qs.iterator(chunk_size=500):
            expected = mapping[student_id]
            if person_id != expected:
                issues.append(AttendanceIdentityIssue(
                    issue_type="mismatched_person_id",
                    model_name=model_name,
                    row_id=row_id,
                    student_id=student_id,
                    person_id=person_id,
                    expected_person_id=expected,
                    detail=(
                        f"person_id={person_id} but StudentProfile says "
                        f"person_id={expected} for student_id={student_id}."
                    ),
                ))

    # C3: orphan_person_no_student — person_id set but student_id is NULL.
    # Should be impossible (student FK is non-nullable), but defensive.
    orphan_qs = (
        Model.objects
        .exclude(person__isnull=True)
        .filter(student__isnull=True)
        .values_list("id", "person_id")
    )
    for row_id, person_id in orphan_qs.iterator(chunk_size=500):
        issues.append(AttendanceIdentityIssue(
            issue_type="orphan_person_no_student",
            model_name=model_name,
            row_id=row_id,
            student_id=None,
            person_id=person_id,
            expected_person_id=None,
            detail=f"person_id={person_id} set but student_id is NULL.",
        ))

    # C4: dangling_person_id — person_id points to non-existent Person.
    # Should be impossible (FK constraint), but defensive.
    valid_person_ids = set(
        Person.objects.values_list("pk", flat=True)
    )
    if valid_person_ids:
        dangling_qs = (
            Model.objects
            .exclude(person__isnull=True)
            .exclude(person_id__in=valid_person_ids)
            .values_list("id", "student_id", "person_id")
        )
        for row_id, student_id, person_id in dangling_qs.iterator(chunk_size=500):
            issues.append(AttendanceIdentityIssue(
                issue_type="dangling_person_id",
                model_name=model_name,
                row_id=row_id,
                student_id=student_id,
                person_id=person_id,
                expected_person_id=None,
                detail=f"person_id={person_id} does not exist in identity.Person.",
            ))

    return ModelReconciliationReport(
        model_name=model_name,
        total_rows=total,
        rows_with_person_id=with_person,
        rows_without_person_id=without_person,
        issues=issues,
    )


def reconcile_face_embeddings() -> ModelReconciliationReport:
    """Reconcile ``FaceEmbedding`` rows for missing/mismatched ``person_id``.

    **Read-only** — never writes.
    """
    mapping = _build_student_to_person_mapping()
    return _reconcile_model("FaceEmbedding", "attendance", "FaceEmbedding", mapping)


def reconcile_attendance_events() -> ModelReconciliationReport:
    """Reconcile ``AttendanceEvent`` rows for missing/mismatched ``person_id``.

    **Read-only** — never writes.
    """
    mapping = _build_student_to_person_mapping()
    return _reconcile_model("AttendanceEvent", "attendance", "AttendanceEvent", mapping)


def reconcile_attendance_records() -> ModelReconciliationReport:
    """Reconcile ``AttendanceRecord`` rows for missing/mismatched ``person_id``.

    **Read-only** — never writes.
    """
    mapping = _build_student_to_person_mapping()
    return _reconcile_model("AttendanceRecord", "attendance", "AttendanceRecord", mapping)


# ---------------------------------------------------------------------------
# Cross-cutting: duplicate detection
# ---------------------------------------------------------------------------


def _detect_duplicate_person_periods() -> list[tuple[int, int, int]]:
    """Detect duplicate ``(person_id, period_id)`` on ``AttendanceRecord``.

    Read-only. Returns ``[(person_id, period_id, count), ...]``.
    """
    AttendanceRecord = django_apps.get_model("attendance", "AttendanceRecord")
    qs = (
        AttendanceRecord.objects
        .exclude(person__isnull=True)
        .values("person_id", "period_id")
        .annotate(c=Count("id"))
        .filter(c__gt=1)
        .order_by("-c")
    )
    return [(r["person_id"], r["period_id"], r["c"]) for r in qs]


def _detect_duplicate_active_embeddings() -> list[tuple[int, int]]:
    """Detect duplicate active ``FaceEmbedding`` per ``person_id``.

    Read-only. Returns ``[(person_id, count), ...]``.
    """
    FaceEmbedding = django_apps.get_model("attendance", "FaceEmbedding")
    qs = (
        FaceEmbedding.objects
        .filter(is_active=True)
        .exclude(person__isnull=True)
        .values("person_id")
        .annotate(c=Count("id"))
        .filter(c__gt=1)
        .order_by("-c")
    )
    return [(r["person_id"], r["c"]) for r in qs]


# ---------------------------------------------------------------------------
# Full reconciliation
# ---------------------------------------------------------------------------


def reconcile_all_attendance_identity() -> AttendanceIdentityReconciliationReport:
    """Reconcile all three attendance models.

    **Read-only** — never writes.

    Returns an :class:`AttendanceIdentityReconciliationReport` with per-model
    reports and cross-cutting duplicate detection.
    """
    return AttendanceIdentityReconciliationReport(
        face_embeddings=reconcile_face_embeddings(),
        attendance_events=reconcile_attendance_events(),
        attendance_records=reconcile_attendance_records(),
        duplicate_person_periods=_detect_duplicate_person_periods(),
        duplicate_active_embeddings=_detect_duplicate_active_embeddings(),
    )
