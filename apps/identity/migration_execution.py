"""Identity migration execution — single-student migration.

This module implements the **write** path for migrating one legacy
``attendance.Student`` into the new identity domain
(``Person`` + ``StudentProfile`` + ``PersonRole``). It is the
execution counterpart to the read-only
:mod:`apps.identity.migration_planning` module.

**This module migrates ONE student at a time.** Bulk migration
(iterating all students, one transaction per student) and the
``execute_identity_migration`` management command are **not**
implemented here — they are separate future milestones.

Design goals:

* **One transaction per student.** ``migrate_student`` is wrapped in
  ``@transaction.atomic``. If any of the three creates (Person,
  StudentProfile, PersonRole) fails, all three roll back.
* **Idempotent.** If the student is already migrated
  (``StudentProfile.legacy_student`` exists), the function returns
  the existing Person / StudentProfile / PersonRole without creating
  duplicates. If a previous run partially completed (e.g. Person was
  created but StudentProfile was not), the function resumes safely
  via the ``get_or_create_*`` helpers.
* **Service reuse.** Calls
  :func:`~apps.identity.services.get_or_create_person_by_code`,
  :func:`~apps.identity.services.get_or_create_student_profile`, and
  :func:`~apps.identity.services.get_or_create_role`. No creation
  logic is duplicated.
* **RoleType prerequisite.** Requires ``RoleType(code="student")`` to
  exist (seeded by the ``0002_seed_roletypes`` data migration). Does
  NOT create RoleType rows — that's the seed migration's job. Raises
  ``ValidationError`` if the RoleType is missing.
* **No legacy mutation.** Never writes to ``attendance.Student``.
* **Planning reuse.** Uses :func:`~apps.identity.migration_planning.validate_migration_data`
  and :func:`~apps.identity.migration_planning.is_migrated` from the
  planning module.

Usage::

    from apps.identity.migration_execution import migrate_student

    result = migrate_student(legacy_student)
    if result.already_migrated:
        print(f"{legacy_student.h_code} already migrated")
    else:
        print(f"Created: person={result.created_person}, "
              f"profile={result.created_profile}, role={result.created_role}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from django.core.exceptions import ValidationError
from django.db import transaction

from .migration_planning import (
    _normalize_gender,
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
    # ---------------------------------------------------------------
    role, created_role = get_or_create_role(
        person=person,
        role_type=role_type,
        is_active=getattr(student, "is_active", True),
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
        role, created_role = get_or_create_role(
            person=person,
            role_type=role_type,
            is_active=getattr(student, "is_active", True),
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
