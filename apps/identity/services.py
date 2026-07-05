from __future__ import annotations

from typing import Optional

from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone

from .models import Person, PersonRole, RoleType, StaffProfile, StudentProfile
from .validators import (
    validate_email_unique_within_persons,
    validate_person_code,
    validate_role_active_window,
    validate_role_unique_for_person,
    validate_staff_code,
    validate_student_code,
)


def _normalize_name(value: str) -> str:
    return (value or "").strip()


@transaction.atomic
def create_person(
    *,
    code: str,
    first_name: str,
    last_name: str,
    middle_name: str = "",
    gender: str = "",
    date_of_birth=None,
    email: str = "",
    phone: str = "",
    address: str = "",
    is_active: bool = True,
    user=None,
) -> Person:
    code = (code or "").strip()
    first_name = _normalize_name(first_name)
    last_name = _normalize_name(last_name)
    email = (email or "").strip()

    validate_person_code(code)
    validate_email_unique_within_persons(email)

    return Person.objects.create(
        code=code,
        first_name=first_name,
        middle_name=_normalize_name(middle_name),
        last_name=last_name,
        gender=gender or None,
        date_of_birth=date_of_birth,
        email=email,
        phone=(phone or "").strip(),
        address=address or "",
        is_active=is_active,
        user=user,
    )


@transaction.atomic
def update_person(
    person: Person,
    *,
    first_name: Optional[str] = None,
    middle_name: Optional[str] = None,
    last_name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    address: Optional[str] = None,
    gender: Optional[str] = None,
    is_active: Optional[bool] = None,
    user=None,
) -> Person:
    if first_name is not None:
        person.first_name = _normalize_name(first_name)
    if middle_name is not None:
        person.middle_name = _normalize_name(middle_name)
    if last_name is not None:
        person.last_name = _normalize_name(last_name)
    if email is not None:
        email = email.strip()
        validate_email_unique_within_persons(email, instance=person)
        person.email = email
    if phone is not None:
        person.phone = phone.strip()
    if address is not None:
        person.address = address
    if gender is not None:
        person.gender = gender or None
    if is_active is not None:
        person.is_active = is_active
    if user is not None:
        person.user = user

    person.save()
    return person


def get_or_create_person_by_code(
    *, code: str, first_name: str, last_name: str, **kwargs
) -> tuple[Person, bool]:
    existing = Person.objects.filter(code=code).first()
    if existing is not None:
        return existing, False
    return create_person(
        code=code, first_name=first_name, last_name=last_name, **kwargs
    ), True


@transaction.atomic
def create_student_profile(
    *,
    person: Person,
    code: str,
    grade: str = "",
    homeroom: str = "",
    has_meal: bool = False,
    has_bus: bool = False,
    legacy_student=None,
) -> StudentProfile:
    code = (code or "").strip()
    validate_student_code(code)
    return StudentProfile.objects.create(
        person=person,
        code=code,
        grade=grade or "",
        homeroom=homeroom or "",
        has_meal=has_meal,
        has_bus=has_bus,
        legacy_student=legacy_student,
    )


@transaction.atomic
def create_staff_profile(
    *,
    person: Person,
    code: str,
    job_title: str = "",
    hire_date=None,
    is_teacher: bool = False,
) -> StaffProfile:
    code = (code or "").strip()
    validate_staff_code(code)
    return StaffProfile.objects.create(
        person=person,
        code=code,
        job_title=job_title or "",
        hire_date=hire_date,
        is_teacher=is_teacher,
    )


@transaction.atomic
def assign_role(
    *,
    person: Person,
    role_type: RoleType,
    is_active: bool = True,
    start_date=None,
    end_date=None,
    notes: str = "",
    assigned_by=None,
) -> PersonRole:
    validate_role_active_window(
        start_date=start_date, end_date=end_date, is_active=is_active
    )
    validate_role_unique_for_person(
        person_id=person.pk, role_type_id=role_type.pk
    )
    return PersonRole.objects.create(
        person=person,
        role_type=role_type,
        is_active=is_active,
        start_date=start_date,
        end_date=end_date,
        notes=notes or "",
        assigned_by=assigned_by,
    )


@transaction.atomic
def deactivate_role(role: PersonRole, *, end_date=None) -> PersonRole:
    role.is_active = False
    role.end_date = end_date or timezone.localdate()
    role.save(update_fields=["is_active", "end_date"])
    return role


@transaction.atomic
def deactivate_person(person: Person) -> Person:
    person.is_active = False
    person.save(update_fields=["is_active"])
    person.roles.filter(is_active=True).update(
        is_active=False, end_date=timezone.localdate()
    )
    return person


# ===========================================================================
# Get-or-create helpers (idempotent; safe for retry after partial migration)
# ===========================================================================


@transaction.atomic
def get_or_create_student_profile(
    *,
    person: Person,
    code: str,
    grade: str = "",
    homeroom: str = "",
    has_meal: bool = False,
    has_bus: bool = False,
    legacy_student=None,
) -> tuple[StudentProfile, bool]:
    """Return ``(StudentProfile, created)`` for the given ``code``.

    Idempotent: if a ``StudentProfile`` with this ``code`` already
    exists, returns it. If it does not exist, creates it.

    **Unsafe conflict detection:** if a profile with this ``code``
    exists but belongs to a **different** ``person``, this is an
    unsafe conflict (two different persons claiming the same student
    code). A ``ValidationError`` is raised — the conflict is not
    silently hidden.

    If ``legacy_student`` is supplied and the existing profile has a
    **different** ``legacy_student``, that is also an unsafe conflict
    and raises ``ValidationError``.
    """
    code = (code or "").strip()
    existing = StudentProfile.objects.filter(code=code).first()

    if existing is not None:
        # Unsafe conflict: same code, different person.
        if existing.person_id != person.pk:
            raise ValidationError(
                f"StudentProfile with code {code!r} exists but belongs "
                f"to a different person (pk={existing.person_id}, "
                f"expected pk={person.pk})."
            )
        # Unsafe conflict: same code, same person, but different
        # legacy_student link.
        if (
            legacy_student is not None
            and existing.legacy_student_id is not None
            and existing.legacy_student_id != legacy_student.pk
        ):
            raise ValidationError(
                f"StudentProfile with code {code!r} exists but is linked "
                f"to a different legacy student "
                f"(pk={existing.legacy_student_id}, "
                f"expected pk={legacy_student.pk})."
            )
        return existing, False

    # Create new profile. Reuse the existing create service so all
    # validators run.
    profile = create_student_profile(
        person=person,
        code=code,
        grade=grade,
        homeroom=homeroom,
        has_meal=has_meal,
        has_bus=has_bus,
        legacy_student=legacy_student,
    )
    return profile, True


@transaction.atomic
def get_or_create_role(
    *,
    person: Person,
    role_type: RoleType,
    is_active: bool = True,
    start_date=None,
    end_date=None,
    notes: str = "",
    assigned_by=None,
) -> tuple[PersonRole, bool]:
    """Return ``(PersonRole, created)`` for the given
    ``(person, role_type)`` pair.

    Idempotent: if a ``PersonRole`` for this ``(person, role_type)``
    already exists, returns it. If it does not exist, creates it via
    :func:`assign_role` (which runs all validators).

    Does **not** silently hide conflicts — if a ``PersonRole`` exists
    but with different ``is_active`` / ``start_date`` / ``end_date``
    values, the existing row is returned unchanged (the caller can
    inspect it). This is by design: the migration execution should
    not silently overwrite existing role assignments.
    """
    existing = PersonRole.objects.filter(
        person=person, role_type=role_type
    ).first()

    if existing is not None:
        return existing, False

    role = assign_role(
        person=person,
        role_type=role_type,
        is_active=is_active,
        start_date=start_date,
        end_date=end_date,
        notes=notes,
        assigned_by=assigned_by,
    )
    return role, True
