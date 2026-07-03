from __future__ import annotations

from typing import Optional

from django.db import transaction
from django.utils import timezone

from .models import (
    AcademicYear,
    Grade,
    SchoolLevel,
    Section,
    StudentEnrollment,
    StudentEnrollmentSectionPlacement,
)
from .validators import (
    validate_active_academic_year_uniqueness,
    validate_enrollment_unique_per_year,
    validate_section_capacity,
    validate_section_consistency,
    validate_withdrawal_date,
)


@transaction.atomic
def create_academic_year(
    *,
    name: str,
    code: str,
    start_date,
    end_date,
    is_active: bool = False,
) -> AcademicYear:
    name = (name or "").strip()
    code = (code or "").strip()
    if is_active:
        validate_active_academic_year_uniqueness()
    return AcademicYear.objects.create(
        name=name,
        code=code,
        start_date=start_date,
        end_date=end_date,
        is_active=is_active,
    )


@transaction.atomic
def set_active_academic_year(academic_year: AcademicYear) -> AcademicYear:
    AcademicYear.objects.filter(is_active=True).exclude(pk=academic_year.pk).update(
        is_active=False
    )
    academic_year.is_active = True
    academic_year.save(update_fields=["is_active", "updated_at"])
    return academic_year


@transaction.atomic
def create_school_level(*, name: str, code: str, order: int = 0) -> SchoolLevel:
    return SchoolLevel.objects.create(
        name=(name or "").strip(), code=(code or "").strip(), order=order
    )


@transaction.atomic
def create_grade(
    *,
    name: str,
    code: str,
    level: Optional[SchoolLevel] = None,
    order: int = 0,
    is_active: bool = True,
) -> Grade:
    return Grade.objects.create(
        name=(name or "").strip(),
        code=(code or "").strip(),
        level=level,
        order=order,
        is_active=is_active,
    )


@transaction.atomic
def create_section(
    *,
    academic_year: AcademicYear,
    grade: Grade,
    code: str,
    name: str = "",
    homeroom_adviser=None,
    capacity=None,
    is_active: bool = True,
) -> Section:
    code = (code or "").strip()
    name = (name or "").strip() or code
    return Section.objects.create(
        academic_year=academic_year,
        grade=grade,
        code=code,
        name=name,
        homeroom_adviser=homeroom_adviser,
        capacity=capacity,
        is_active=is_active,
    )


def _close_current_placement(*, enrollment, end_date) -> None:
    """Close the active section placement for an enrollment, preserving history."""
    if end_date is None:
        end_date = timezone.localdate()
    StudentEnrollmentSectionPlacement.objects.filter(
        enrollment=enrollment, is_current=True
    ).update(is_current=False, end_date=end_date)


@transaction.atomic
def enroll_student(
    *,
    student,
    academic_year: AcademicYear,
    grade: Grade,
    section: Optional[Section] = None,
    enrollment_date=None,
    status: str = StudentEnrollment.Status.ACTIVE,
    prior_enrollment: Optional[StudentEnrollment] = None,
    notes: str = "",
) -> StudentEnrollment:
    validate_enrollment_unique_per_year(
        student_id=student.pk, academic_year_id=academic_year.pk
    )
    validate_section_consistency(
        academic_year_id=academic_year.pk,
        grade_id=grade.pk,
        section=section,
    )
    if section:
        validate_section_capacity(section=section)
    if enrollment_date is None:
        enrollment_date = timezone.localdate()
    enrollment = StudentEnrollment.objects.create(
        student=student,
        academic_year=academic_year,
        grade=grade,
        section=section,
        enrollment_date=enrollment_date,
        status=status,
        prior_enrollment=prior_enrollment,
        notes=notes or "",
    )
    if section is not None:
        StudentEnrollmentSectionPlacement.objects.create(
            enrollment=enrollment,
            section=section,
            start_date=enrollment_date,
            is_current=True,
        )
    return enrollment


@transaction.atomic
def activate_future_enrollments(*, academic_year: AcademicYear) -> int:
    updated = StudentEnrollment.objects.filter(
        academic_year=academic_year,
        status=StudentEnrollment.Status.FUTURE,
    ).update(status=StudentEnrollment.Status.ACTIVE)
    return updated


@transaction.atomic
def transfer_section(
    *,
    enrollment: StudentEnrollment,
    new_section: Section,
    changed_by=None,
    transfer_date=None,
) -> StudentEnrollment:
    validate_section_consistency(
        academic_year_id=enrollment.academic_year_id,
        grade_id=enrollment.grade_id,
        section=new_section,
        instance=enrollment,
    )
    validate_section_capacity(section=new_section, instance=enrollment)
    if transfer_date is None:
        transfer_date = timezone.localdate()

    previous = (
        StudentEnrollmentSectionPlacement.objects.filter(
            enrollment=enrollment, is_current=True
        )
        .order_by("-start_date")
        .first()
    )
    if previous is not None:
        previous.is_current = False
        previous.end_date = transfer_date
        previous.save(update_fields=["is_current", "end_date", "updated_at"])

    StudentEnrollmentSectionPlacement.objects.create(
        enrollment=enrollment,
        section=new_section,
        start_date=transfer_date,
        is_current=True,
    )

    enrollment.section = new_section
    enrollment.save(update_fields=["section", "updated_at"])
    return enrollment


@transaction.atomic
def withdraw(
    *,
    enrollment: StudentEnrollment,
    withdrawal_date=None,
    changed_by=None,
) -> StudentEnrollment:
    if withdrawal_date is None:
        withdrawal_date = timezone.localdate()
    validate_withdrawal_date(
        enrollment_date=enrollment.enrollment_date, withdrawal_date=withdrawal_date
    )
    _close_current_placement(enrollment=enrollment, end_date=withdrawal_date)
    enrollment.status = StudentEnrollment.Status.WITHDRAWN
    enrollment.withdrawal_date = withdrawal_date
    enrollment.save(update_fields=["status", "withdrawal_date", "updated_at"])
    return enrollment


@transaction.atomic
def graduate(
    *,
    enrollment: StudentEnrollment,
    graduation_date=None,
    changed_by=None,
) -> StudentEnrollment:
    if graduation_date is None:
        graduation_date = timezone.localdate()
    _close_current_placement(enrollment=enrollment, end_date=graduation_date)
    enrollment.status = StudentEnrollment.Status.GRADUATED
    enrollment.save(update_fields=["status", "updated_at"])
    return enrollment


@transaction.atomic
def repeat_year(
    *,
    enrollment: StudentEnrollment,
    next_academic_year: AcademicYear,
    changed_by=None,
    repeat_date=None,
) -> StudentEnrollment:
    if repeat_date is None:
        repeat_date = timezone.localdate()
    _close_current_placement(enrollment=enrollment, end_date=repeat_date)
    enrollment.status = StudentEnrollment.Status.REPEATED
    enrollment.save(update_fields=["status", "updated_at"])
    new_enrollment = enroll_student(
        student=enrollment.student,
        academic_year=next_academic_year,
        grade=enrollment.grade,
        section=None,
        enrollment_date=repeat_date,
        status=StudentEnrollment.Status.ACTIVE,
        prior_enrollment=enrollment,
    )
    return new_enrollment


@transaction.atomic
def archive_year(*, academic_year: AcademicYear) -> int:
    updated = StudentEnrollment.objects.filter(
        academic_year=academic_year,
        status__in=[
            StudentEnrollment.Status.WITHDRAWN,
            StudentEnrollment.Status.GRADUATED,
            StudentEnrollment.Status.REPEATED,
            StudentEnrollment.Status.TRANSFERRED,
        ],
    ).update(status=StudentEnrollment.Status.ARCHIVED)
    return updated
