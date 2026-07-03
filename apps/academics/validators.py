from __future__ import annotations

from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

from .models import AcademicYear, StudentEnrollment, StudentEnrollmentSectionPlacement


def validate_enrollment_unique_per_year(
    *, student_id, academic_year_id, instance: StudentEnrollment | None = None
) -> None:
    qs = StudentEnrollment.objects.filter(
        student_id=student_id, academic_year_id=academic_year_id
    )
    if instance is not None and instance.pk:
        qs = qs.exclude(pk=instance.pk)
    if qs.exists():
        raise ValidationError(
            _("This student already has an enrollment for this academic year.")
        )


def validate_section_consistency(
    *,
    academic_year_id,
    grade_id,
    section,
    instance: StudentEnrollment | None = None,
) -> None:
    """A section must belong to the same academic year and grade as the enrollment."""
    if section is None:
        return
    if section.academic_year_id != academic_year_id:
        raise ValidationError(
            {
                "section": _(
                    "Section's academic year must match the enrollment's "
                    "academic year."
                )
            }
        )
    if section.grade_id != grade_id:
        raise ValidationError(
            {
                "section": _(
                    "Section's grade must match the enrollment's grade."
                )
            }
        )


def validate_section_capacity(
    *, section, instance: StudentEnrollment | None = None
) -> None:
    """Capacity counts only *current* section placements.

    A withdrawn/graduated/repeated enrollment keeps its historical
    `StudentEnrollment.section` FK for reference, but its current placement
    is closed. Such enrollments must not consume capacity. We therefore count
    `StudentEnrollmentSectionPlacement` rows with `is_current=True`, not
    `StudentEnrollment.section` rows.
    """
    if section is None or not section.capacity:
        return
    qs = StudentEnrollmentSectionPlacement.objects.filter(
        section=section, is_current=True
    )
    if instance is not None and instance.pk:
        qs = qs.exclude(enrollment_id=instance.pk)
    if qs.count() >= section.capacity:
        raise ValidationError(
            {
                "section": _(
                    "Section %(section)s is at capacity (%(cap)d)."
                    % {"section": section, "cap": section.capacity}
                )
            }
        )


def validate_withdrawal_date(
    *, enrollment_date, withdrawal_date
) -> None:
    if withdrawal_date and enrollment_date and withdrawal_date < enrollment_date:
        raise ValidationError(
            {"withdrawal_date": _("Withdrawal date cannot be before enrollment date.")}
        )


def validate_active_academic_year_uniqueness(
    *, instance: AcademicYear | None = None
) -> None:
    qs = AcademicYear.objects.filter(is_active=True)
    if instance is not None and instance.pk:
        qs = qs.exclude(pk=instance.pk)
    if qs.exists():
        raise ValidationError(
            {"is_active": _("Another academic year is already active.")}
        )
