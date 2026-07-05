"""Identity migration reconciliation — read-only verification.

This module verifies that legacy ``attendance.Student`` rows are
correctly represented in the new identity domain
(``Person`` + ``StudentProfile`` + ``PersonRole``). It is the
**post-migration verification** counterpart to the execution engine
(:mod:`apps.identity.migration_execution`).

**This module is read-only.** No functions write to the database.
They inspect both legacy and identity tables and produce
:class:`MigrationReconciliationReport` objects listing any
discrepancies.

Reconciliation checks per legacy Student:

* **Unmigrated:** no ``StudentProfile.legacy_student`` link exists.
* **Missing Person:** ``StudentProfile`` exists but its ``person`` is
  missing or None.
* **Missing PersonRole:** ``Person`` + ``StudentProfile`` exist but
  no ``PersonRole`` with ``role_type__code="student"``.
* **Code mismatch (Person):** ``Student.h_code != Person.code``.
* **Code mismatch (StudentProfile):** ``Student.h_code !=
  StudentProfile.code``.
* **Name mismatch:** ``Student.first_name / last_name`` differ from
  ``Person.first_name / last_name`` (informational, not blocking).

Cross-cutting checks:

* **Duplicate Person codes:** reuses
  :func:`~apps.identity.migration_planning.detect_duplicate_person_codes`.
* **Duplicate StudentProfile codes:** reuses
  :func:`~apps.identity.migration_planning.detect_duplicate_student_profile_codes`.
* **Orphan StudentProfile:** ``StudentProfile`` rows with
  ``legacy_student=None`` (not linked to any legacy Student).
* **Orphan Person:** ``Person`` rows with no ``StudentProfile`` (not
  necessarily an error — staff persons have no student profile).

Usage::

    from apps.identity.migration_reconciliation import reconcile_all

    report = reconcile_all()
    print(f"Total: {report.total_students}")
    print(f"Migrated: {report.migrated}")
    print(f"Unmigrated: {report.unmigrated}")
    print(f"Issues: {report.issues_found}")
    for row in report.rows:
        if row.issues:
            print(f"  {row.h_code}: {row.issues}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from django.apps import apps as django_apps

from .migration_planning import (
    detect_duplicate_person_codes,
    detect_duplicate_student_profile_codes,
    is_migrated,
)


# ---------------------------------------------------------------------------
# Dataclasses (immutable — no Django models, no migrations)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StudentReconciliationIssue:
    """A single issue found during reconciliation of one student."""

    issue_type: str  # e.g. "unmigrated", "missing_person", "code_mismatch"
    detail: str


@dataclass(frozen=True)
class StudentReconciliationRow:
    """Reconciliation result for one legacy ``Student``."""

    legacy_student_id: int
    h_code: str
    is_migrated: bool
    person_id: Optional[int]
    person_code: Optional[str]
    student_profile_id: Optional[int]
    student_profile_code: Optional[str]
    has_student_role: bool
    issues: list[StudentReconciliationIssue] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0


@dataclass(frozen=True)
class MigrationReconciliationReport:
    """Aggregate reconciliation report for all legacy ``Student`` rows
    plus cross-cutting checks."""

    total_students: int = 0
    migrated: int = 0
    unmigrated: int = 0
    issues_found: int = 0
    rows: list[StudentReconciliationRow] = field(default_factory=list)

    # Cross-cutting
    duplicate_person_codes: list[str] = field(default_factory=list)
    duplicate_student_profile_codes: list[str] = field(default_factory=list)
    orphan_student_profiles: list[int] = field(default_factory=list)
    orphan_persons: list[int] = field(default_factory=list)

    @property
    def rows_with_issues(self) -> list[StudentReconciliationRow]:
        return [r for r in self.rows if r.has_issues]


# ---------------------------------------------------------------------------
# Single-student reconciliation
# ---------------------------------------------------------------------------


def reconcile_student(student) -> StudentReconciliationRow:
    """Reconcile one legacy ``Student`` against the identity domain.

    **Read-only** — never writes.

    Returns a :class:`StudentReconciliationRow` listing any issues
    found (missing Person, missing StudentProfile, missing PersonRole,
    code mismatches, name mismatches).
    """
    StudentProfile = django_apps.get_model("identity", "StudentProfile")
    PersonRole = django_apps.get_model("identity", "PersonRole")

    h_code = (student.h_code or "").strip()
    issues: list[StudentReconciliationIssue] = []

    migrated = is_migrated(student)

    person_id: Optional[int] = None
    person_code: Optional[str] = None
    profile_id: Optional[int] = None
    profile_code: Optional[str] = None
    has_role = False

    if not migrated:
        issues.append(StudentReconciliationIssue(
            issue_type="unmigrated",
            detail="No StudentProfile.legacy_student link exists.",
        ))
    else:
        # StudentProfile exists — check Person + PersonRole + codes.
        try:
            profile = StudentProfile.objects.select_related("person").get(
                legacy_student=student
            )
        except StudentProfile.DoesNotExist:
            # is_migrated said True but the row is gone — race or
            # data inconsistency.
            issues.append(StudentReconciliationIssue(
                issue_type="missing_student_profile",
                detail="is_migrated() returned True but StudentProfile not found.",
            ))
            return StudentReconciliationRow(
                legacy_student_id=student.pk,
                h_code=h_code,
                is_migrated=True,
                person_id=None,
                person_code=None,
                student_profile_id=None,
                student_profile_code=None,
                has_student_role=False,
                issues=issues,
            )

        profile_id = profile.pk
        profile_code = profile.code
        person = profile.person
        if person is not None:
            person_id = person.pk
            person_code = person.code

        # Check Person exists.
        if person is None:
            issues.append(StudentReconciliationIssue(
                issue_type="missing_person",
                detail="StudentProfile exists but person is None.",
            ))
        else:
            # Check code mismatch: Student.h_code vs Person.code.
            if person.code != h_code:
                issues.append(StudentReconciliationIssue(
                    issue_type="code_mismatch_person",
                    detail=f"Student.h_code='{h_code}' but Person.code='{person.code}'.",
                ))

            # Check name mismatch (informational).
            if (student.first_name or "").strip() != (person.first_name or "").strip():
                issues.append(StudentReconciliationIssue(
                    issue_type="name_mismatch_first",
                    detail=f"Student.first_name='{student.first_name}' but Person.first_name='{person.first_name}'.",
                ))
            if (student.last_name or "").strip() != (person.last_name or "").strip():
                issues.append(StudentReconciliationIssue(
                    issue_type="name_mismatch_last",
                    detail=f"Student.last_name='{student.last_name}' but Person.last_name='{person.last_name}'.",
                ))

            # Check PersonRole (student role).
            has_role = PersonRole.objects.filter(
                person=person, role_type__code="student"
            ).exists()
            if not has_role:
                issues.append(StudentReconciliationIssue(
                    issue_type="missing_student_role",
                    detail="Person has no PersonRole with role_type='student'.",
                ))

        # Check code mismatch: Student.h_code vs StudentProfile.code.
        if profile.code != h_code:
            issues.append(StudentReconciliationIssue(
                issue_type="code_mismatch_profile",
                detail=f"Student.h_code='{h_code}' but StudentProfile.code='{profile.code}'.",
            ))

    return StudentReconciliationRow(
        legacy_student_id=student.pk,
        h_code=h_code,
        is_migrated=migrated,
        person_id=person_id,
        person_code=person_code,
        student_profile_id=profile_id,
        student_profile_code=profile_code,
        has_student_role=has_role,
        issues=issues,
    )


# ---------------------------------------------------------------------------
# Full reconciliation
# ---------------------------------------------------------------------------


def reconcile_all() -> MigrationReconciliationReport:
    """Reconcile all legacy ``Student`` rows against the identity
    domain.

    **Read-only** — never writes.

    Iterates every ``attendance.Student``, calls
    :func:`reconcile_student` for each, and runs cross-cutting checks
    (duplicate detection, orphan profiles/persons).

    Returns a :class:`MigrationReconciliationReport`.
    """
    Student = django_apps.get_model("attendance", "Student")
    StudentProfile = django_apps.get_model("identity", "StudentProfile")
    Person = django_apps.get_model("identity", "Person")

    rows: list[StudentReconciliationRow] = []
    total = 0
    migrated_count = 0
    unmigrated_count = 0
    issues_count = 0

    for student in Student.objects.all().order_by("h_code"):
        total += 1
        row = reconcile_student(student)
        rows.append(row)

        if row.is_migrated:
            migrated_count += 1
        else:
            unmigrated_count += 1

        if row.has_issues:
            issues_count += 1

    # Cross-cutting: duplicate detection (reuse planning helpers).
    dup_persons = detect_duplicate_person_codes()
    dup_profiles = detect_duplicate_student_profile_codes()

    # Cross-cutting: orphan StudentProfile (legacy_student=None).
    orphan_profile_ids = list(
        StudentProfile.objects.filter(legacy_student__isnull=True)
        .values_list("pk", flat=True)
    )

    # Cross-cutting: orphan Person (no StudentProfile).
    # A Person without a StudentProfile is not necessarily an error
    # (staff persons, manually created persons). We report them for
    # informational purposes.
    person_ids_with_profile = set(
        StudentProfile.objects.values_list("person_id", flat=True)
    )
    orphan_person_ids = list(
        Person.objects.exclude(pk__in=person_ids_with_profile)
        .values_list("pk", flat=True)
    )

    return MigrationReconciliationReport(
        total_students=total,
        migrated=migrated_count,
        unmigrated=unmigrated_count,
        issues_found=issues_count,
        rows=rows,
        duplicate_person_codes=dup_persons,
        duplicate_student_profile_codes=dup_profiles,
        orphan_student_profiles=orphan_profile_ids,
        orphan_persons=orphan_person_ids,
    )
