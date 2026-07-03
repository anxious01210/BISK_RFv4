from __future__ import annotations

from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

from .models import Person, PersonRole, StaffProfile, StudentProfile


def validate_person_code(code: str) -> None:
    code = (code or "").strip()
    if not code:
        raise ValidationError(_("Person code is required."))
    if Person.objects.filter(code=code).exists():
        raise ValidationError(
            _("A person with code %(code)s already exists.") % {"code": code}
        )


def validate_student_code(code: str) -> None:
    code = (code or "").strip()
    if not code:
        raise ValidationError(_("Student code is required."))
    if StudentProfile.objects.filter(code=code).exists():
        raise ValidationError(
            _("A student profile with code %(code)s already exists.")
            % {"code": code}
        )


def validate_staff_code(code: str) -> None:
    code = (code or "").strip()
    if not code:
        raise ValidationError(_("Staff code is required."))
    if StaffProfile.objects.filter(code=code).exists():
        raise ValidationError(
            _("A staff profile with code %(code)s already exists.") % {"code": code}
        )


def validate_role_active_window(
    *,
    start_date=None,
    end_date=None,
    is_active: bool = True,
) -> None:
    if start_date and end_date and end_date < start_date:
        raise ValidationError(_("End date cannot be before start date."))
    if not is_active and not end_date:
        raise ValidationError(
            _("An inactive role assignment should have an end date.")
        )


def validate_role_unique_for_person(
    *,
    person_id,
    role_type_id,
    instance: PersonRole | None = None,
) -> None:
    qs = PersonRole.objects.filter(person_id=person_id, role_type_id=role_type_id)
    if instance is not None and instance.pk:
        qs = qs.exclude(pk=instance.pk)
    if qs.exists():
        raise ValidationError(
            _("This person already has this role assigned.")
        )


def validate_email_unique_within_persons(
    email: str, *, instance: Person | None = None
) -> None:
    email = (email or "").strip()
    if not email:
        return
    qs = Person.objects.filter(email__iexact=email)
    if instance is not None and instance.pk:
        qs = qs.exclude(pk=instance.pk)
    if qs.exists():
        raise ValidationError(
            _("Another person already uses email %(email)s.") % {"email": email}
        )
