"""Identity migration planning infrastructure.

This module provides **dry-run-only** planning, validation, and
reporting helpers for the future ``attendance.Student`` →
``identity.Person`` + ``StudentProfile`` + ``PersonRole`` data
migration (Phase 1.5 of the identity architecture,
``docs/architecture/person_identity_architecture.md`` §21).

**This is infrastructure, not the migration itself.** No functions in
this module write to the database. They inspect legacy
``attendance.Student`` rows and produce :class:`StudentMigrationPlan`
objects describing what *would* be created. The actual migration
(execution) will be a separately-approved task that calls
``apps.identity.services.create_person`` /
``create_student_profile`` / ``assign_role`` inside a data migration.

Design goals:

* **Dry-run capable** — every function is read-only.
* **Idempotent** — planning the same Student twice produces the same
  plan; planning a Student that is already migrated reports it as
  skipped.
* **Validation** — detects missing required fields (``h_code``,
  ``first_name``, ``last_name``) and duplicate codes before the
  migration runs.
* **No legacy mutation** — never writes to ``attendance.Student``,
  ``identity.Person``, ``StudentProfile``, or ``PersonRole``.
* **No Django models** — :class:`MigrationReport` and
  :class:`StudentMigrationPlan` are plain Python dataclasses; no
  migration is needed for this module.

Usage::

    from apps.identity.migration_planning import plan_all_student_migrations

    report = plan_all_student_migrations()
    print(f"Total: {report.total}")
    print(f"Already migrated: {report.already_migrated}")
    print(f"Ready: {report.ready}")
    print(f"Blocked: {report.blocked}")
    for plan in report.blocked_plans:
        print(f"  BLOCKED {plan.h_code}: {plan.issues}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from django.apps import apps as django_apps


# ---------------------------------------------------------------------------
# Dataclasses (no Django models — no migrations needed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StudentMigrationPlan:
    """The dry-run plan for migrating a single legacy ``Student``.

    Describes what *would* be created if the migration were executed.
    No fields are written to the database by this object or by the
    functions that produce it.
    """

    legacy_student_id: int
    h_code: str
    person_code: str
    first_name: str
    middle_name: str
    last_name: str
    gender: Optional[str]
    grade: str
    has_meal: bool
    has_bus: bool
    is_active: bool
    student_profile_code: str
    role_type_code: str  # "student"
    already_migrated: bool
    issues: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        """True if the plan has no blocking issues and is not already
        migrated."""
        return not self.issues and not self.already_migrated


@dataclass(frozen=True)
class MigrationReport:
    """Aggregate dry-run report for all legacy ``Student`` rows."""

    total: int = 0
    already_migrated: int = 0
    ready: int = 0
    blocked: int = 0
    plans: list[StudentMigrationPlan] = field(default_factory=list)

    @property
    def blocked_plans(self) -> list[StudentMigrationPlan]:
        return [p for p in self.plans if p.issues]

    @property
    def ready_plans(self) -> list[StudentMigrationPlan]:
        return [p for p in self.plans if p.is_ready]

    @property
    def skipped_plans(self) -> list[StudentMigrationPlan]:
        return [p for p in self.plans if p.already_migrated]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_student_model():
    """Return the legacy ``attendance.Student`` model class via the
    Django app registry (avoids importing ``apps.attendance.models``
    at module load time)."""
    return django_apps.get_model("attendance", "Student")


def _get_student_profile_model():
    """Return the ``identity.StudentProfile`` model class."""
    return django_apps.get_model("identity", "StudentProfile")


def _get_person_model():
    return django_apps.get_model("identity", "Person")


def _normalize_gender(gender: Optional[str]) -> Optional[str]:
    """Normalize legacy gender values to Person.Gender choices."""
    if not gender:
        return None
    g = gender.strip().upper()
    if g in ("MALE", "M"):
        return "MALE"
    if g in ("FEMALE", "F"):
        return "FEMALE"
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_migration_data(student) -> list[str]:
    """Return a list of issue strings for a legacy ``Student``.

    Checks for missing required fields that would block the migration:

    * ``h_code`` must be non-empty (becomes ``Person.code``).
    * ``first_name`` should be non-empty (becomes ``Person.first_name``).
    * ``last_name`` should be non-empty (becomes ``Person.last_name``).

    Returns an empty list if all required fields are present. This is
    a **read-only** inspection — it never writes.
    """
    issues: list[str] = []

    if not getattr(student, "h_code", None) or not student.h_code.strip():
        issues.append("Missing h_code (required for Person.code).")
    if not getattr(student, "first_name", None) or not student.first_name.strip():
        issues.append("Missing first_name.")
    if not getattr(student, "last_name", None) or not student.last_name.strip():
        issues.append("Missing last_name.")

    return issues


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def detect_duplicate_person_codes() -> list[str]:
    """Return a list of ``h_code`` values that would produce duplicate
    ``Person.code`` values.

    A duplicate arises when two legacy ``Student`` rows share the same
    ``h_code`` (should not happen because ``h_code`` is unique, but
    this is a defensive check), or when a ``Person`` with the same
    code already exists but is not linked to the corresponding
    ``StudentProfile``.

    This is a **read-only** query.
    """
    Student = _get_student_model()
    Person = _get_person_model()
    StudentProfile = _get_student_profile_model()

    duplicates: list[str] = []

    for student in Student.objects.all():
        code = (student.h_code or "").strip()
        if not code:
            continue

        # Check if a Person with this code exists but is NOT linked
        # to a StudentProfile that references this Student.
        existing_person = Person.objects.filter(code=code).first()
        if existing_person is not None:
            # Is there a StudentProfile for this Person?
            profile = StudentProfile.objects.filter(
                person=existing_person
            ).first()
            if profile is None or profile.legacy_student_id != student.pk:
                duplicates.append(code)

    return duplicates


def detect_duplicate_student_profile_codes() -> list[str]:
    """Return a list of student profile codes that already exist in
    the ``StudentProfile`` table but would conflict with a planned
    migration.

    The migration plan uses ``h_code`` as the ``StudentProfile.code``
    (since ``h_code`` is the school-assigned student identifier). If a
    ``StudentProfile`` with that code already exists but is not linked
    to the corresponding legacy ``Student``, the migration would
    produce a unique-constraint violation.

    This is a **read-only** query.
    """
    Student = _get_student_model()
    StudentProfile = _get_student_profile_model()

    duplicates: list[str] = []

    for student in Student.objects.all():
        code = (student.h_code or "").strip()
        if not code:
            continue

        existing = StudentProfile.objects.filter(code=code).first()
        if existing is not None and existing.legacy_student_id != student.pk:
            duplicates.append(code)

    return duplicates


# ---------------------------------------------------------------------------
# Migration status
# ---------------------------------------------------------------------------


def is_migrated(student) -> bool:
    """Return ``True`` if a legacy ``Student`` already has a linked
    ``StudentProfile`` (i.e. the migration has already been executed
    for this student).

    This is a **read-only** query.
    """
    StudentProfile = _get_student_profile_model()
    return StudentProfile.objects.filter(legacy_student=student).exists()


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def plan_student_migration(student) -> StudentMigrationPlan:
    """Produce a dry-run migration plan for a single legacy
    ``Student``.

    Inspects the student's fields and returns a
    :class:`StudentMigrationPlan` describing what *would* be created:

    * One ``Person`` (code = h_code, names, gender).
    * One ``StudentProfile`` (code = h_code, grade, has_meal, has_bus).
    * One ``PersonRole`` (role_type = "student").

    **No database writes occur.** This is purely a read-only
    inspection. If the student is already migrated
    (``StudentProfile.legacy_student`` exists), the plan reports
    ``already_migrated=True`` and ``is_ready=False``.
    """
    issues = validate_migration_data(student)
    migrated = is_migrated(student)

    h_code = (student.h_code or "").strip()

    return StudentMigrationPlan(
        legacy_student_id=student.pk,
        h_code=h_code,
        person_code=h_code,
        first_name=(student.first_name or "").strip(),
        middle_name=(student.middle_name or "").strip(),
        last_name=(student.last_name or "").strip(),
        gender=_normalize_gender(student.gender),
        grade=(student.grade or "").strip(),
        has_meal=getattr(student, "has_meal", False),
        has_bus=getattr(student, "has_bus", False),
        is_active=getattr(student, "is_active", True),
        student_profile_code=h_code,
        role_type_code="student",
        already_migrated=migrated,
        issues=issues,
    )


def plan_all_student_migrations() -> MigrationReport:
    """Produce a dry-run migration report for **all** legacy
    ``Student`` rows.

    Iterates over every ``attendance.Student`` row, calls
    :func:`plan_student_migration` for each, and aggregates the
    results into a :class:`MigrationReport`.

    **No database writes occur.** The report is a read-only summary.

    The report is **idempotent**: calling this function multiple times
    produces the same result (assuming no writes occurred between
    calls).
    """
    Student = _get_student_model()

    plans: list[StudentMigrationPlan] = []
    total = 0
    already_migrated = 0
    ready = 0
    blocked = 0

    for student in Student.objects.all().order_by("h_code"):
        total += 1
        plan = plan_student_migration(student)
        plans.append(plan)

        if plan.already_migrated:
            already_migrated += 1
        elif plan.issues:
            blocked += 1
        else:
            ready += 1

    return MigrationReport(
        total=total,
        already_migrated=already_migrated,
        ready=ready,
        blocked=blocked,
        plans=plans,
    )
