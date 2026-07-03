from __future__ import annotations

from typing import Optional

from django.db.models import QuerySet, Q

from .models import (
    AcademicYear,
    Grade,
    SchoolLevel,
    Section,
    StudentEnrollment,
    StudentEnrollmentSectionPlacement,
)


def get_active_academic_year() -> Optional[AcademicYear]:
    return AcademicYear.objects.filter(is_active=True).first()


def get_academic_year_by_code(code: str) -> Optional[AcademicYear]:
    return AcademicYear.objects.filter(code=code).first()


def list_academic_years(*, is_active: Optional[bool] = None) -> QuerySet[AcademicYear]:
    qs = AcademicYear.objects.all()
    if is_active is not None:
        qs = qs.filter(is_active=is_active)
    return qs


def list_school_levels(*, is_active: Optional[bool] = None) -> QuerySet[SchoolLevel]:
    qs = SchoolLevel.objects.all()
    if is_active is not None:
        qs = qs.filter(is_active=is_active)
    return qs


def list_grades(
    *, level=None, is_active: Optional[bool] = None, search: str = ""
) -> QuerySet[Grade]:
    qs = Grade.objects.select_related("level")
    if level is not None:
        qs = qs.filter(level=level)
    if is_active is not None:
        qs = qs.filter(is_active=is_active)
    search = (search or "").strip()
    if search:
        qs = qs.filter(Q(name__icontains=search) | Q(code__icontains=search))
    return qs


def get_grade_by_code(code: str) -> Optional[Grade]:
    return Grade.objects.filter(code=code).first()


def list_sections(
    *,
    academic_year=None,
    grade=None,
    is_active: Optional[bool] = None,
    search: str = "",
) -> QuerySet[Section]:
    qs = Section.objects.select_related(
        "academic_year", "grade", "grade__level", "homeroom_adviser", "homeroom_adviser__person"
    )
    if academic_year is not None:
        qs = qs.filter(academic_year=academic_year)
    if grade is not None:
        qs = qs.filter(grade=grade)
    if is_active is not None:
        qs = qs.filter(is_active=is_active)
    search = (search or "").strip()
    if search:
        qs = qs.filter(
            Q(name__icontains=search)
            | Q(code__icontains=search)
            | Q(grade__name__icontains=search)
            | Q(grade__code__icontains=search)
        )
    return qs


def get_section(*, academic_year, grade, code: str) -> Optional[Section]:
    return (
        Section.objects.select_related("academic_year", "grade", "homeroom_adviser")
        .filter(academic_year=academic_year, grade=grade, code=code)
        .first()
    )


def active_enrollment_for(*, student, academic_year) -> Optional[StudentEnrollment]:
    return (
        StudentEnrollment.objects.select_related(
            "student", "student__person", "academic_year", "grade", "section"
        )
        .filter(student=student, academic_year=academic_year, status=StudentEnrollment.Status.ACTIVE)
        .first()
    )


def enrollments_for_year(*, academic_year, status: Optional[str] = None) -> QuerySet[StudentEnrollment]:
    qs = StudentEnrollment.objects.select_related(
        "student", "student__person", "academic_year", "grade", "section"
    ).filter(academic_year=academic_year)
    if status is not None:
        qs = qs.filter(status=status)
    return qs


def enrollments_for_student(*, student) -> QuerySet[StudentEnrollment]:
    return (
        StudentEnrollment.objects.select_related(
            "academic_year", "grade", "section", "section__homeroom_adviser"
        )
        .filter(student=student)
        .order_by("-academic_year__start_date")
    )


def students_in_section(*, section, status: Optional[str] = None) -> QuerySet[StudentEnrollment]:
    """Historical view: all enrollments whose `section` FK references this section.

    Includes enrollments that still carry the section FK after their current
    placement was closed (withdrawn/graduated/repeated). Use
    `current_enrollments_in_section()` for the operational roster.
    """
    qs = StudentEnrollment.objects.select_related(
        "student", "student__person", "grade", "section"
    ).filter(section=section)
    if status is not None:
        qs = qs.filter(status=status)
    return qs


def current_enrollments_in_section(*, section) -> QuerySet[StudentEnrollment]:
    """Operational roster: enrollments with a *current* placement in this section.

    A withdrawn/graduated/repeated enrollment keeps its historical
    `StudentEnrollment.section` FK but has its current placement closed, so it
    is excluded here. Only enrollments whose `StudentEnrollmentSectionPlacement`
    for this section is `is_current=True` are returned.
    """
    current_enrollment_ids = StudentEnrollmentSectionPlacement.objects.filter(
        section=section, is_current=True
    ).values_list("enrollment_id", flat=True)
    return (
        StudentEnrollment.objects.select_related(
            "student", "student__person", "grade", "section", "academic_year"
        )
        .filter(pk__in=list(current_enrollment_ids))
        .order_by("student__person__last_name", "student__person__first_name")
    )


def current_placement_for_enrollment(*, enrollment) -> Optional[StudentEnrollmentSectionPlacement]:
    return (
        StudentEnrollmentSectionPlacement.objects.select_related("section")
        .filter(enrollment=enrollment, is_current=True)
        .first()
    )


def placement_history_for_enrollment(*, enrollment) -> QuerySet[StudentEnrollmentSectionPlacement]:
    """All placements (current and closed) for an enrollment, most recent first."""
    return (
        StudentEnrollmentSectionPlacement.objects.select_related(
            "enrollment", "section", "section__academic_year", "section__grade"
        )
        .filter(enrollment=enrollment)
        .order_by("-start_date")
    )
