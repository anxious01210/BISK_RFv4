"""Identity migration execution — single-student and bulk migration.

This module implements the **write** path for migrating legacy
``attendance.Student`` rows into the new identity domain
(``Person`` + ``StudentProfile`` + ``PersonRole``). It is the
execution counterpart to the read-only
:mod:`apps.identity.migration_planning` module.

Two layers:

* :func:`migrate_student` — migrates ONE student inside one
  ``@transaction.atomic`` block. Idempotent, retryable, reuses the
  existing ``get_or_create_*`` service helpers.
* :func:`migrate_all_students` — the **orchestration layer**.
  Iterates all legacy Students, calls ``migrate_student`` per
  student (one transaction per student), catches per-student
  failures, and returns a :class:`BulkMigrationResult` summary.

Design goals:

* **One transaction per student.** ``migrate_student`` is wrapped in
  ``@transaction.atomic``. The bulk migrator does NOT wrap the whole
  batch in one transaction — partial progress is preserved.
* **Idempotent.** Both layers are safe to re-run. ``migrate_student``
  skips already-migrated students; ``migrate_all_students`` produces
  the same result on repeated calls.
* **Service reuse.** ``migrate_student`` calls
  :func:`~apps.identity.services.get_or_create_person_by_code`,
  :func:`~apps.identity.services.get_or_create_student_profile`, and
  :func:`~apps.identity.services.get_or_create_role`. The bulk
  migrator calls ``migrate_student`` — no execution logic is
  duplicated.
* **RoleType prerequisite.** Requires ``RoleType(code="student")`` to
  exist (seeded by the ``0002_seed_roletypes`` data migration).
* **No legacy mutation.** Never writes to ``attendance.Student``.
* **Planning reuse.** Uses :func:`~apps.identity.migration_planning.validate_migration_data`,
  :func:`~apps.identity.migration_planning.is_migrated`,
  :func:`~apps.identity.migration_planning.detect_duplicate_person_codes`,
  and :func:`~apps.identity.migration_planning.detect_duplicate_student_profile_codes`.

Usage::

    from apps.identity.migration_execution import migrate_all_students

    result = migrate_all_students()
    print(f"Total: {result.total_students}")
    print(f"Migrated: {result.migrated}")
    print(f"Already migrated: {result.already_migrated}")
    print(f"Failed: {result.failed}")
    for err in result.errors:
        print(f"  FAILED {err.h_code}: {err.error}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from django.apps import apps as django_apps
from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone

from .migration_planning import (
    _normalize_gender,
    detect_duplicate_person_codes,
    detect_duplicate_student_profile_codes,
    is_migrated,
    validate_migration_data,
)
from .models import Person, PersonRole, RoleType, StudentProfile
from .services import (
    get_or_create_person_by_code,
    get_or_create_role,
    get_or_create_student_profile,
)


# ---------------------------------------------------------------------------
# Result dataclass (immutable — no Django model, no migration)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MigrationExecutionResult:
    """Immutable result of migrating one legacy ``Student``.

    Returned by :func:`migrate_student`. Carries the created/existing
    ``Person``, ``StudentProfile``, ``PersonRole`` and flags indicating
    which rows were newly created vs. already present.
    """

    person: Person
    student_profile: StudentProfile
    person_role: PersonRole
    created_person: bool
    created_profile: bool
    created_role: bool
    already_migrated: bool


# ---------------------------------------------------------------------------
# Single-student migration
# ---------------------------------------------------------------------------


@transaction.atomic
def migrate_student(student) -> MigrationExecutionResult:
    """Migrate one legacy ``attendance.Student`` into the identity
    domain.

    Creates (or reuses, if already present from a partial run):

    * One :class:`~apps.identity.models.Person` (``code = student.h_code``).
    * One :class:`~apps.identity.models.StudentProfile`
      (``code = student.h_code``, ``legacy_student = student``).
    * One :class:`~apps.identity.models.PersonRole`
      (``role_type = RoleType(code="student")``).

    All three operations run inside **one** ``@transaction.atomic``
    block. If any fails, all three roll back — the student remains
    unmigrated and can be retried.

    **Idempotent:** if the student is already migrated
    (``StudentProfile.legacy_student`` link exists), the function
    returns the existing rows with ``already_migrated=True`` and all
    ``created_*`` flags ``False``.

    **RoleType prerequisite:** ``RoleType(code="student")`` must exist
    (seeded by ``0002_seed_roletypes``). Raises ``ValidationError`` if
    missing.

    **Validation:** calls
    :func:`~apps.identity.migration_planning.validate_migration_data`
    to check required fields (``h_code``, ``first_name``,
    ``last_name``). Raises ``ValidationError`` if any are missing.

    **No legacy mutation:** never writes to ``attendance.Student``.

    :param student: A legacy ``attendance.Student`` instance.
    :returns: :class:`MigrationExecutionResult` with the
        Person / StudentProfile / PersonRole and creation flags.
    :raises ValidationError: if required fields are missing or
        RoleType is not seeded.
    """

    # ---------------------------------------------------------------
    # 1. Validate required fields (reuse planning helper).
    # ---------------------------------------------------------------
    issues = validate_migration_data(student)
    if issues:
        raise ValidationError(issues)

    # ---------------------------------------------------------------
    # 2. Check if already migrated (reuse planning helper).
    # ---------------------------------------------------------------
    if is_migrated(student):
        # The student already has a StudentProfile link. Return the
        # existing rows without creating duplicates.
        return _build_already_migrated_result(student)

    # ---------------------------------------------------------------
    # 3. Obtain the "student" RoleType (must be pre-seeded).
    # ---------------------------------------------------------------
    try:
        role_type = RoleType.objects.get(code="student")
    except RoleType.DoesNotExist:
        raise ValidationError(
            "RoleType with code='student' not found. "
            "Run the 0002_seed_roletypes migration first."
        )

    # ---------------------------------------------------------------
    # 4. Create or reuse Person.
    # ---------------------------------------------------------------
    h_code = (student.h_code or "").strip()
    gender = _normalize_gender(student.gender)

    person, created_person = get_or_create_person_by_code(
        code=h_code,
        first_name=(student.first_name or "").strip(),
        last_name=(student.last_name or "").strip(),
        middle_name=(student.middle_name or "").strip(),
        gender=gender or "",
        is_active=getattr(student, "is_active", True),
    )

    # ---------------------------------------------------------------
    # 5. Create or reuse StudentProfile (with legacy_student backlink).
    # ---------------------------------------------------------------
    profile, created_profile = get_or_create_student_profile(
        person=person,
        code=h_code,
        grade=(student.grade or "").strip(),
        has_meal=getattr(student, "has_meal", False),
        has_bus=getattr(student, "has_bus", False),
        legacy_student=student,
    )

    # ---------------------------------------------------------------
    # 6. Create or reuse PersonRole.
    #    For inactive students, set end_date so the role-active-window
    #    validator (validate_role_active_window) does not reject the
    #    creation. Active students keep end_date=None.
    # ---------------------------------------------------------------
    student_is_active = getattr(student, "is_active", True)
    role_end_date = None if student_is_active else timezone.localdate()
    role, created_role = get_or_create_role(
        person=person,
        role_type=role_type,
        is_active=student_is_active,
        end_date=role_end_date,
    )

    # ---------------------------------------------------------------
    # 7. Return the immutable result.
    # ---------------------------------------------------------------
    return MigrationExecutionResult(
        person=person,
        student_profile=profile,
        person_role=role,
        created_person=created_person,
        created_profile=created_profile,
        created_role=created_role,
        already_migrated=False,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_already_migrated_result(student) -> MigrationExecutionResult:
    """Build a result for a student that is already migrated.

    Looks up the existing ``StudentProfile`` via the
    ``legacy_student`` backlink, then the ``Person`` and
    ``PersonRole`` from the profile.
    """
    from django.apps import apps as django_apps

    StudentProfileModel = django_apps.get_model("identity", "StudentProfile")
    PersonRoleModel = django_apps.get_model("identity", "PersonRole")

    profile = StudentProfileModel.objects.select_related("person").get(
        legacy_student=student
    )
    person = profile.person

    # The PersonRole may or may not exist (a partial migration could
    # have created Person + StudentProfile but not PersonRole).
    role = PersonRoleModel.objects.filter(
        person=person, role_type__code="student"
    ).first()

    if role is None:
        # Edge case: Person + StudentProfile exist but no PersonRole.
        # This is a partial migration that was interrupted. We should
        # NOT silently create the role here (that would be a write
        # outside the normal flow). Instead, return the result with
        # person_role=None and let the caller decide. However, the
        # dataclass requires a PersonRole. We raise to signal the
        # partial state — the caller should call migrate_student
        # again, which will NOT skip (is_migrated is True, but the
        # role is missing). Actually, is_migrated checks for the
        # StudentProfile link, which exists. So migrate_student would
        # skip. We need to handle this: create the missing role.
        #
        # The cleanest approach: if the role is missing, create it
        # here (inside the outer @transaction.atomic). This completes
        # the partial migration.
        try:
            role_type = RoleType.objects.get(code="student")
        except RoleType.DoesNotExist:
            raise ValidationError(
                "RoleType with code='student' not found. "
                "Run the 0002_seed_roletypes migration first."
            )
        student_is_active = getattr(student, "is_active", True)
        role_end_date = None if student_is_active else timezone.localdate()
        role, created_role = get_or_create_role(
            person=person,
            role_type=role_type,
            is_active=student_is_active,
            end_date=role_end_date,
        )
    else:
        created_role = False

    return MigrationExecutionResult(
        person=person,
        student_profile=profile,
        person_role=role,
        created_person=False,
        created_profile=False,
        created_role=created_role,
        already_migrated=True,
    )


# ===========================================================================
# Bulk migration — orchestration layer
# ===========================================================================


@dataclass(frozen=True)
class BulkMigrationError:
    """Error record for a single student that failed during bulk
    migration."""

    student_id: int
    h_code: str
    error: str


@dataclass(frozen=True)
class BulkMigrationResult:
    """Immutable summary of a bulk migration run.

    Returned by :func:`migrate_all_students`. Carries per-student
    results (successful + already-migrated) and per-student errors
    (failures that did not abort the batch).
    """

    total_students: int
    migrated: int
    already_migrated: int
    blocked: int
    failed: int
    results: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def migrate_all_students(
    *,
    limit: Optional[int] = None,
    student_queryset=None,
) -> BulkMigrationResult:
    """Migrate all legacy ``attendance.Student`` rows into the identity
    domain.

    **Orchestration responsibilities** (this function does NOT
    duplicate per-student logic — it calls :func:`migrate_student`):

    1. **Pre-flight duplicate detection.** Calls
       :func:`~apps.identity.migration_planning.detect_duplicate_person_codes`
       and
       :func:`~apps.identity.migration_planning.detect_duplicate_student_profile_codes`.
       If any duplicates are found, raises ``ValidationError`` listing
       the conflicting codes. **No students are migrated** if
       duplicates exist.

    2. **Pre-flight RoleType check.** Resolves
       ``RoleType(code="student")`` once. Raises ``ValidationError``
       if missing. (``migrate_student`` also resolves it per-student,
       but this early check fails fast before any writes.)

    3. **Iterate students.** Builds the queryset (or uses the
       supplied ``student_queryset``), applies ``limit`` if given,
       and calls ``migrate_student(student)`` for each.

    4. **One transaction per student.** ``migrate_student`` is
       ``@transaction.atomic`` — each student is its own transaction.
       The bulk migrator does **NOT** wrap the whole batch in one
       transaction. If student #50 fails, students #1–49 are
       committed and students #51+ are still attempted.

    5. **Catch per-student failures.** If ``migrate_student`` raises,
       the error is recorded in :attr:`BulkMigrationResult.errors`
       and the batch continues. The student remains unmigrated.

    6. **Return summary.** :class:`BulkMigrationResult` with counts
       (``migrated`` / ``already_migrated`` / ``failed``) and the
       per-student results/errors.

    **Retry behavior:** the function is idempotent. Calling it again
    after a partial failure will:

    * Skip already-migrated students (``already_migrated`` count
      increases).
    * Retry previously-failed students (if the underlying issue is
      fixed, they succeed; otherwise they fail again).
    * Not create duplicates (``get_or_create_*`` helpers).

    :param limit: Optional maximum number of students to process.
        ``None`` means no limit (process all).
    :param student_queryset: Optional pre-filtered queryset of
        ``attendance.Student`` rows. If ``None``, all students are
        used (``Student.objects.all().order_by("h_code")``).
    :returns: :class:`BulkMigrationResult`.
    :raises ValidationError: if duplicate Person/StudentProfile codes
        are detected or RoleType is missing.
    """

    # ---------------------------------------------------------------
    # 1. Pre-flight: duplicate detection.
    # ---------------------------------------------------------------
    dup_persons = detect_duplicate_person_codes()
    dup_profiles = detect_duplicate_student_profile_codes()

    if dup_persons or dup_profiles:
        parts = []
        if dup_persons:
            parts.append(
                f"Duplicate Person codes: {', '.join(dup_persons)}"
            )
        if dup_profiles:
            parts.append(
                f"Duplicate StudentProfile codes: {', '.join(dup_profiles)}"
            )
        raise ValidationError(
            "Bulk migration aborted — duplicate codes detected. "
            "Resolve conflicts before migrating. " + " | ".join(parts)
        )

    # ---------------------------------------------------------------
    # 2. Pre-flight: RoleType check (resolve once, fail fast).
    # ---------------------------------------------------------------
    try:
        RoleType.objects.get(code="student")
    except RoleType.DoesNotExist:
        raise ValidationError(
            "RoleType with code='student' not found. "
            "Run the 0002_seed_roletypes migration first."
        )

    # ---------------------------------------------------------------
    # 3. Build the queryset.
    # ---------------------------------------------------------------
    Student = django_apps.get_model("attendance", "Student")

    if student_queryset is not None:
        qs = student_queryset
    else:
        qs = Student.objects.all().order_by("h_code")

    if limit is not None:
        qs = qs[:limit]

    # ---------------------------------------------------------------
    # 4. Iterate — one transaction per student (migrate_student
    #    has its own @transaction.atomic).
    # ---------------------------------------------------------------
    results: list[MigrationExecutionResult] = []
    errors: list[BulkMigrationError] = []
    migrated = 0
    already_migrated = 0
    failed = 0
    total = 0

    for student in qs:
        total += 1
        try:
            result = migrate_student(student)
            results.append(result)
            if result.already_migrated:
                already_migrated += 1
            else:
                migrated += 1
        except Exception as exc:
            # Record the failure and continue. The student's
            # transaction has already rolled back (migrate_student's
            # @transaction.atomic). The batch continues.
            failed += 1
            h_code = (getattr(student, "h_code", None) or "").strip()
            errors.append(
                BulkMigrationError(
                    student_id=student.pk,
                    h_code=h_code,
                    error=str(exc),
                )
            )

    # ---------------------------------------------------------------
    # 5. Return the immutable summary.
    # ---------------------------------------------------------------
    return BulkMigrationResult(
        total_students=total,
        migrated=migrated,
        already_migrated=already_migrated,
        blocked=0,  # blocked students are counted as failed (they raised)
        failed=failed,
        results=results,
        errors=errors,
    )
