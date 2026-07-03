from __future__ import annotations

from typing import Optional

from django.db.models import QuerySet, Q

from .models import Person, PersonRole, RoleType, StaffProfile, StudentProfile


def get_person_by_code(code: str) -> Optional[Person]:
    return (
        Person.objects.select_related("user", "student_profile", "staff_profile")
        .filter(code=code)
        .first()
    )


def get_person_by_display_code(display_code: str) -> Optional[Person]:
    student = StudentProfile.objects.filter(code=display_code).first()
    if student is not None:
        return student.person
    staff = StaffProfile.objects.filter(code=display_code).first()
    if staff is not None:
        return staff.person
    return get_person_by_code(display_code)


def list_people(
    *,
    is_active: Optional[bool] = None,
    search: str = "",
) -> QuerySet[Person]:
    qs = Person.objects.select_related("user", "student_profile", "staff_profile")
    if is_active is not None:
        qs = qs.filter(is_active=is_active)
    search = (search or "").strip()
    if search:
        qs = qs.filter(
            Q(code__icontains=search)
            | Q(first_name__icontains=search)
            | Q(middle_name__icontains=search)
            | Q(last_name__icontains=search)
            | Q(email__icontains=search)
            | Q(phone__icontains=search)
        )
    return qs


def list_student_profiles(*, search: str = "") -> QuerySet[StudentProfile]:
    qs = StudentProfile.objects.select_related("person", "person__user")
    search = (search or "").strip()
    if search:
        qs = qs.filter(
            Q(code__icontains=search)
            | Q(person__code__icontains=search)
            | Q(person__first_name__icontains=search)
            | Q(person__last_name__icontains=search)
            | Q(grade__icontains=search)
            | Q(homeroom__icontains=search)
        )
    return qs


def list_staff_profiles(*, search: str = "") -> QuerySet[StaffProfile]:
    qs = StaffProfile.objects.select_related("person", "person__user")
    search = (search or "").strip()
    if search:
        qs = qs.filter(
            Q(code__icontains=search)
            | Q(person__code__icontains=search)
            | Q(person__first_name__icontains=search)
            | Q(person__last_name__icontains=search)
            | Q(job_title__icontains=search)
        )
    return qs


def list_person_roles(
    *, person_id=None, is_active: Optional[bool] = None
) -> QuerySet[PersonRole]:
    qs = PersonRole.objects.select_related("person", "role_type", "assigned_by")
    if person_id is not None:
        qs = qs.filter(person_id=person_id)
    if is_active is not None:
        qs = qs.filter(is_active=is_active)
    return qs


def active_roles_for_person(person: Person) -> QuerySet[PersonRole]:
    return list_person_roles(person_id=person.pk, is_active=True)


def list_role_types(*, is_active: Optional[bool] = None) -> QuerySet[RoleType]:
    qs = RoleType.objects.all()
    if is_active is not None:
        qs = qs.filter(is_active=is_active)
    return qs
