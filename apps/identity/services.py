from __future__ import annotations

from typing import Optional

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
