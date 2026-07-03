from datetime import date
from django.core.exceptions import ValidationError
from django.test import TestCase

from apps.identity.models import Person, StaffProfile, StudentProfile
from apps.academics.models import (
    AcademicYear,
    Grade,
    SchoolLevel,
    Section,
    StudentEnrollment,
    StudentEnrollmentSectionPlacement,
)
from apps.academics.selectors import (
    active_enrollment_for,
    current_enrollments_in_section,
    current_placement_for_enrollment,
    enrollments_for_student,
    enrollments_for_year,
    get_active_academic_year,
    get_grade_by_code,
    placement_history_for_enrollment,
    students_in_section,
)
from apps.academics.services import (
    archive_year,
    create_academic_year,
    create_grade,
    create_school_level,
    create_section,
    enroll_student,
    graduate,
    repeat_year,
    set_active_academic_year,
    transfer_section,
    withdraw,
)
from apps.academics.validators import (
    validate_active_academic_year_uniqueness,
    validate_enrollment_unique_per_year,
    validate_section_capacity,
    validate_section_consistency,
)


class AcademicsBaseData(TestCase):
    def setUp(self):
        self.year = create_academic_year(
            name="2026-2027", code="2026-27",
            start_date=date(2026, 9, 1), end_date=date(2027, 6, 30),
            is_active=True,
        )
        self.next_year = create_academic_year(
            name="2027-2028", code="2027-28",
            start_date=date(2027, 9, 1), end_date=date(2028, 6, 30),
        )
        self.level = create_school_level(name="Primary", code="PRI", order=1)
        self.grade = create_grade(name="Grade 1", code="G1", level=self.level, order=1)
        self.section_a = create_section(
            academic_year=self.year, grade=self.grade, code="A", name="Section A", capacity=2
        )

        self.person_student = Person.objects.create(
            code="P-S1", first_name="Stu", last_name="Dent"
        )
        self.student = StudentProfile.objects.create(person=self.person_student, code="S-1001")
        self.person_staff = Person.objects.create(
            code="P-T1", first_name="Tea", last_name="Cher"
        )
        self.adviser = StaffProfile.objects.create(person=self.person_staff, code="T-2001")


class AcademicYearTests(AcademicsBaseData):
    def test_active_year_returned(self):
        self.assertEqual(get_active_academic_year(), self.year)

    def test_only_one_active_year(self):
        set_active_academic_year(self.next_year)
        self.year.refresh_from_db()
        self.assertFalse(self.year.is_active)
        self.next_year.refresh_from_db()
        self.assertTrue(self.next_year.is_active)

    def test_create_active_rejects_second_active(self):
        with self.assertRaises(ValidationError):
            create_academic_year(
                name="2028-2029", code="2028-29",
                start_date=date(2028, 9, 1), end_date=date(2029, 6, 30),
                is_active=True,
            )

    def test_end_before_start_invalid(self):
        from django.db.utils import IntegrityError
        with self.assertRaises(IntegrityError):
            AcademicYear.objects.create(
                name="Bad", code="BAD",
                start_date=date(2027, 6, 30), end_date=date(2027, 6, 1),
            )


class GradeSectionTests(AcademicsBaseData):
    def test_grade_lookup_by_code(self):
        self.assertEqual(get_grade_by_code("G1"), self.grade)

    def test_section_unique_per_year_grade_code(self):
        with self.assertRaises(Exception):
            create_section(
                academic_year=self.year, grade=self.grade, code="A", name="Dup"
            )

    def test_section_year_scoped_identity(self):
        section_next = create_section(
            academic_year=self.next_year, grade=self.grade, code="A", name="Section A"
        )
        self.assertNotEqual(section_next.pk, self.section_a.pk)
        self.assertEqual(section_next.academic_year, self.next_year)

    def test_section_str_includes_year(self):
        self.assertIn("2026-27", str(self.section_a))
        self.assertIn("G1", str(self.section_a))


class EnrollmentServicesTests(AcademicsBaseData):
    def test_enroll_student(self):
        enrollment = enroll_student(
            student=self.student, academic_year=self.year,
            grade=self.grade, section=self.section_a,
            enrollment_date=date(2026, 9, 5),
        )
        self.assertEqual(enrollment.status, StudentEnrollment.Status.ACTIVE)
        self.assertEqual(active_enrollment_for(student=self.student, academic_year=self.year), enrollment)

    def test_duplicate_enrollment_per_year_rejected(self):
        enroll_student(
            student=self.student, academic_year=self.year, grade=self.grade,
            enrollment_date=date(2026, 9, 5),
        )
        with self.assertRaises(ValidationError):
            enroll_student(
                student=self.student, academic_year=self.year, grade=self.grade,
                enrollment_date=date(2026, 9, 6),
            )

    def test_section_year_mismatch_rejected(self):
        section_next = create_section(
            academic_year=self.next_year, grade=self.grade, code="A"
        )
        with self.assertRaises(ValidationError):
            enroll_student(
                student=self.student, academic_year=self.year, grade=self.grade,
                section=section_next, enrollment_date=date(2026, 9, 5),
            )

    def test_capacity_enforced(self):
        s2 = StudentProfile.objects.create(
            person=Person.objects.create(code="P-S2", first_name="Two", last_name="Student"),
            code="S-1002",
        )
        s3 = StudentProfile.objects.create(
            person=Person.objects.create(code="P-S3", first_name="Three", last_name="Student"),
            code="S-1003",
        )
        enroll_student(student=self.student, academic_year=self.year, grade=self.grade,
                       section=self.section_a, enrollment_date=date(2026, 9, 5))
        enroll_student(student=s2, academic_year=self.year, grade=self.grade,
                       section=self.section_a, enrollment_date=date(2026, 9, 5))
        with self.assertRaises(ValidationError):
            enroll_student(student=s3, academic_year=self.year, grade=self.grade,
                           section=self.section_a, enrollment_date=date(2026, 9, 5))

    def test_transfer_section(self):
        enrollment = enroll_student(
            student=self.student, academic_year=self.year, grade=self.grade,
            section=self.section_a, enrollment_date=date(2026, 9, 5),
        )
        section_b = create_section(
            academic_year=self.year, grade=self.grade, code="B", name="Section B"
        )
        transfer_section(enrollment=enrollment, new_section=section_b)
        enrollment.refresh_from_db()
        self.assertEqual(enrollment.section, section_b)
        self.assertEqual(enrollment.status, StudentEnrollment.Status.ACTIVE)

    def test_withdraw_then_history_preserved(self):
        enrollment = enroll_student(
            student=self.student, academic_year=self.year, grade=self.grade,
            section=self.section_a, enrollment_date=date(2026, 9, 5),
        )
        withdraw(enrollment=enrollment, withdrawal_date=date(2026, 11, 1))
        enrollment.refresh_from_db()
        self.assertEqual(enrollment.status, StudentEnrollment.Status.WITHDRAWN)
        self.assertIsNotNone(enrollment.withdrawal_date)
        self.assertEqual(enrollments_for_student(student=self.student).count(), 1)

    def test_withdrawal_before_enrollment_rejected(self):
        enrollment = enroll_student(
            student=self.student, academic_year=self.year, grade=self.grade,
            enrollment_date=date(2026, 9, 5),
        )
        with self.assertRaises(ValidationError):
            withdraw(enrollment=enrollment, withdrawal_date=date(2026, 8, 1))

    def test_graduate(self):
        enrollment = enroll_student(
            student=self.student, academic_year=self.year, grade=self.grade,
            enrollment_date=date(2026, 9, 5),
        )
        graduate(enrollment=enrollment)
        enrollment.refresh_from_db()
        self.assertEqual(enrollment.status, StudentEnrollment.Status.GRADUATED)

    def test_repeat_year_preserves_history(self):
        enrollment = enroll_student(
            student=self.student, academic_year=self.year, grade=self.grade,
            section=self.section_a, enrollment_date=date(2026, 9, 5),
        )
        new_enrollment = repeat_year(
            enrollment=enrollment, next_academic_year=self.next_year
        )
        enrollment.refresh_from_db()
        self.assertEqual(enrollment.status, StudentEnrollment.Status.REPEATED)
        self.assertEqual(new_enrollment.status, StudentEnrollment.Status.ACTIVE)
        self.assertEqual(new_enrollment.prior_enrollment, enrollment)
        self.assertEqual(enrollment.grade, new_enrollment.grade)
        self.assertEqual(enrollments_for_student(student=self.student).count(), 2)

    def test_archive_year(self):
        enrollment = enroll_student(
            student=self.student, academic_year=self.year, grade=self.grade,
            enrollment_date=date(2026, 9, 5),
        )
        graduate(enrollment=enrollment)
        count = archive_year(academic_year=self.year)
        self.assertEqual(count, 1)
        enrollment.refresh_from_db()
        self.assertEqual(enrollment.status, StudentEnrollment.Status.ARCHIVED)


class SelectorTests(AcademicsBaseData):
    def setUp(self):
        super().setUp()
        self.enrollment = enroll_student(
            student=self.student, academic_year=self.year, grade=self.grade,
            section=self.section_a, enrollment_date=date(2026, 9, 5),
        )

    def test_enrollments_for_year(self):
        self.assertIn(self.enrollment, enrollments_for_year(academic_year=self.year))

    def test_students_in_section(self):
        self.assertIn(self.enrollment, students_in_section(section=self.section_a))


class ValidatorTests(AcademicsBaseData):
    def test_validate_active_uniqueness(self):
        with self.assertRaises(ValidationError):
            validate_active_academic_year_uniqueness()

    def test_validate_section_year_consistency(self):
        section_next = create_section(
            academic_year=self.next_year, grade=self.grade, code="A"
        )
        with self.assertRaises(ValidationError):
            validate_section_consistency(
                academic_year_id=self.year.pk,
                grade_id=self.grade.pk,
                section=section_next,
            )

    def test_validate_section_grade_consistency(self):
        other_grade = create_grade(name="Grade 2", code="G2", order=2)
        section_g2 = create_section(
            academic_year=self.year, grade=other_grade, code="A"
        )
        with self.assertRaises(ValidationError):
            validate_section_consistency(
                academic_year_id=self.year.pk,
                grade_id=self.grade.pk,
                section=section_g2,
            )

    def test_validate_capacity(self):
        s2 = StudentProfile.objects.create(
            person=Person.objects.create(code="P-S2", first_name="Two", last_name="Student"),
            code="S-1002",
        )
        enroll_student(student=self.student, academic_year=self.year, grade=self.grade,
                       section=self.section_a, enrollment_date=date(2026, 9, 5))
        enroll_student(student=s2, academic_year=self.year, grade=self.grade,
                       section=self.section_a, enrollment_date=date(2026, 9, 5))
        with self.assertRaises(ValidationError):
            validate_section_capacity(section=self.section_a)


class SectionPlacementTests(AcademicsBaseData):
    def setUp(self):
        super().setUp()
        self.section_b = create_section(
            academic_year=self.year, grade=self.grade, code="B", name="Section B"
        )
        self.enrollment = enroll_student(
            student=self.student,
            academic_year=self.year,
            grade=self.grade,
            section=self.section_a,
            enrollment_date=date(2026, 9, 5),
        )

    def test_initial_placement_created(self):
        placements = placement_history_for_enrollment(enrollment=self.enrollment)
        self.assertEqual(placements.count(), 1)
        placement = placements.first()
        self.assertEqual(placement.section, self.section_a)
        self.assertTrue(placement.is_current)
        self.assertIsNone(placement.end_date)
        self.assertEqual(placement.start_date, date(2026, 9, 5))

    def test_enrollment_without_section_creates_no_placement(self):
        enrollment = enroll_student(
            student=StudentProfile.objects.create(
                person=Person.objects.create(code="P-S9", first_name="Nine", last_name="Student"),
                code="S-1009",
            ),
            academic_year=self.year,
            grade=self.grade,
            enrollment_date=date(2026, 9, 5),
        )
        self.assertEqual(placement_history_for_enrollment(enrollment=enrollment).count(), 0)

    def test_transfer_preserves_previous_placement(self):
        transfer_section(
            enrollment=self.enrollment,
            new_section=self.section_b,
            transfer_date=date(2026, 10, 1),
        )
        history = placement_history_for_enrollment(enrollment=self.enrollment)
        self.assertEqual(history.count(), 2)

        old = history.exclude(is_current=True).first()
        self.assertEqual(old.section, self.section_a)
        self.assertFalse(old.is_current)
        self.assertEqual(old.end_date, date(2026, 10, 1))

        current = current_placement_for_enrollment(enrollment=self.enrollment)
        self.assertEqual(current.section, self.section_b)
        self.assertTrue(current.is_current)
        self.assertIsNone(current.end_date)

    def test_only_one_current_placement_after_transfer(self):
        transfer_section(
            enrollment=self.enrollment,
            new_section=self.section_b,
            transfer_date=date(2026, 10, 1),
        )
        current_qs = StudentEnrollmentSectionPlacement.objects.filter(
            enrollment=self.enrollment, is_current=True
        )
        self.assertEqual(current_qs.count(), 1)
        self.assertEqual(current_qs.first().section, self.section_b)

    def test_current_enrollment_section_points_to_latest(self):
        transfer_section(
            enrollment=self.enrollment,
            new_section=self.section_b,
            transfer_date=date(2026, 10, 1),
        )
        self.enrollment.refresh_from_db()
        self.assertEqual(self.enrollment.section, self.section_b)

    def test_db_blocks_two_current_placements(self):
        from django.db.utils import IntegrityError
        with self.assertRaises(IntegrityError):
            StudentEnrollmentSectionPlacement.objects.create(
                enrollment=self.enrollment,
                section=self.section_b,
                start_date=date(2026, 11, 1),
                is_current=True,
            )

    def test_current_placement_with_end_date_blocked(self):
        from django.db.utils import IntegrityError
        with self.assertRaises(IntegrityError):
            StudentEnrollmentSectionPlacement.objects.create(
                enrollment=self.enrollment,
                section=self.section_b,
                start_date=date(2026, 10, 1),
                end_date=date(2026, 12, 1),
                is_current=True,
            )

    def test_transfer_section_year_mismatch_rejected(self):
        section_next = create_section(
            academic_year=self.next_year, grade=self.grade, code="A"
        )
        with self.assertRaises(ValidationError):
            transfer_section(enrollment=self.enrollment, new_section=section_next)


class EnrollmentModelCleanTests(AcademicsBaseData):
    def test_enrollment_clean_rejects_section_year_mismatch(self):
        section_next = create_section(
            academic_year=self.next_year, grade=self.grade, code="A"
        )
        enrollment = StudentEnrollment(
            student=self.student,
            academic_year=self.year,
            grade=self.grade,
            section=section_next,
            enrollment_date=date(2026, 9, 5),
        )
        with self.assertRaises(ValidationError):
            enrollment.clean()

    def test_enrollment_clean_rejects_section_grade_mismatch(self):
        other_grade = create_grade(name="Grade 2", code="G2", order=2)
        section_g2 = create_section(
            academic_year=self.year, grade=other_grade, code="A"
        )
        enrollment = StudentEnrollment(
            student=self.student,
            academic_year=self.year,
            grade=self.grade,
            section=section_g2,
            enrollment_date=date(2026, 9, 5),
        )
        with self.assertRaises(ValidationError):
            enrollment.clean()

    def test_enrollment_clean_rejects_withdrawal_before_enrollment(self):
        enrollment = StudentEnrollment(
            student=self.student,
            academic_year=self.year,
            grade=self.grade,
            enrollment_date=date(2026, 9, 5),
            withdrawal_date=date(2026, 8, 1),
        )
        with self.assertRaises(ValidationError):
            enrollment.clean()


class AcademicYearActiveConstraintTests(AcademicsBaseData):
    def test_db_blocks_second_active_year(self):
        from django.db.utils import IntegrityError
        with self.assertRaises(IntegrityError):
            AcademicYear.objects.create(
                name="2028-2029", code="2028-29",
                start_date=date(2028, 9, 1), end_date=date(2029, 6, 30),
                is_active=True,
            )

    def test_inactive_years_unconstrained(self):
        AcademicYear.objects.create(
            name="2028-2029", code="2028-29",
            start_date=date(2028, 9, 1), end_date=date(2029, 6, 30),
            is_active=False,
        )
        AcademicYear.objects.create(
            name="2029-2030", code="2029-30",
            start_date=date(2029, 9, 1), end_date=date(2030, 6, 30),
            is_active=False,
        )
        self.assertTrue(AcademicYear.objects.filter(code="2028-29", is_active=False).exists())
        self.assertTrue(AcademicYear.objects.filter(code="2029-30", is_active=False).exists())


class PlacementClosingTests(AcademicsBaseData):
    """When an enrollment ends for the year, the current placement must be closed."""

    def setUp(self):
        super().setUp()
        self.enrollment = enroll_student(
            student=self.student,
            academic_year=self.year,
            grade=self.grade,
            section=self.section_a,
            enrollment_date=date(2026, 9, 5),
        )

    def _current_placement(self):
        return current_placement_for_enrollment(enrollment=self.enrollment)

    def test_withdraw_closes_current_placement(self):
        withdraw(enrollment=self.enrollment, withdrawal_date=date(2026, 11, 1))
        placement = self._current_placement()
        self.assertIsNone(placement)
        closed = StudentEnrollmentSectionPlacement.objects.get(
            enrollment=self.enrollment, section=self.section_a
        )
        self.assertFalse(closed.is_current)
        self.assertEqual(closed.end_date, date(2026, 11, 1))

    def test_graduate_closes_current_placement(self):
        graduate(enrollment=self.enrollment, graduation_date=date(2027, 6, 15))
        placement = self._current_placement()
        self.assertIsNone(placement)
        closed = StudentEnrollmentSectionPlacement.objects.get(
            enrollment=self.enrollment, section=self.section_a
        )
        self.assertFalse(closed.is_current)
        self.assertEqual(closed.end_date, date(2027, 6, 15))

    def test_repeat_year_closes_current_placement(self):
        new_enrollment = repeat_year(
            enrollment=self.enrollment,
            next_academic_year=self.next_year,
            repeat_date=date(2027, 6, 15),
        )
        self.enrollment.refresh_from_db()
        self.assertEqual(self.enrollment.status, StudentEnrollment.Status.REPEATED)
        self.assertIsNone(self._current_placement())
        closed = StudentEnrollmentSectionPlacement.objects.get(
            enrollment=self.enrollment, section=self.section_a
        )
        self.assertFalse(closed.is_current)
        self.assertEqual(closed.end_date, date(2027, 6, 15))

    def test_repeat_year_new_enrollment_has_no_current_placement(self):
        new_enrollment = repeat_year(
            enrollment=self.enrollment,
            next_academic_year=self.next_year,
            repeat_date=date(2027, 6, 15),
        )
        self.assertIsNone(current_placement_for_enrollment(enrollment=new_enrollment))

    def test_withdraw_preserves_placement_history(self):
        section_b = create_section(
            academic_year=self.year, grade=self.grade, code="B", name="Section B"
        )
        transfer_section(
            enrollment=self.enrollment,
            new_section=section_b,
            transfer_date=date(2026, 10, 1),
        )
        withdraw(enrollment=self.enrollment, withdrawal_date=date(2026, 12, 1))
        self.assertIsNone(self._current_placement())
        history = placement_history_for_enrollment(enrollment=self.enrollment)
        self.assertEqual(history.count(), 2)
        for placement in history:
            self.assertFalse(placement.is_current)
            self.assertIsNotNone(placement.end_date)


class CapacityUsesCurrentPlacementsTests(AcademicsBaseData):
    """Capacity must count only *current* placements, not stale section FKs."""

    def setUp(self):
        super().setUp()
        self.s2 = StudentProfile.objects.create(
            person=Person.objects.create(code="P-S2", first_name="Two", last_name="Student"),
            code="S-1002",
        )
        self.s3 = StudentProfile.objects.create(
            person=Person.objects.create(code="P-S3", first_name="Three", last_name="Student"),
            code="S-1003",
        )

    def test_withdrawn_does_not_consume_capacity(self):
        enrollment = enroll_student(
            student=self.student, academic_year=self.year, grade=self.grade,
            section=self.section_a, enrollment_date=date(2026, 9, 5),
        )
        withdraw(enrollment=enrollment, withdrawal_date=date(2026, 10, 1))
        self.section_a.refresh_from_db()
        self.assertEqual(enrollment.section, self.section_a)
        self.assertIsNone(current_placement_for_enrollment(enrollment=enrollment))
        enroll_student(
            student=self.s2, academic_year=self.year, grade=self.grade,
            section=self.section_a, enrollment_date=date(2026, 10, 5),
        )
        enroll_student(
            student=self.s3, academic_year=self.year, grade=self.grade,
            section=self.section_a, enrollment_date=date(2026, 10, 6),
        )

    def test_graduated_does_not_consume_capacity(self):
        enrollment = enroll_student(
            student=self.student, academic_year=self.year, grade=self.grade,
            section=self.section_a, enrollment_date=date(2026, 9, 5),
        )
        graduate(enrollment=enrollment, graduation_date=date(2026, 10, 1))
        enroll_student(
            student=self.s2, academic_year=self.year, grade=self.grade,
            section=self.section_a, enrollment_date=date(2026, 10, 5),
        )
        enroll_student(
            student=self.s3, academic_year=self.year, grade=self.grade,
            section=self.section_a, enrollment_date=date(2026, 10, 6),
        )

    def test_repeated_does_not_consume_capacity(self):
        enrollment = enroll_student(
            student=self.student, academic_year=self.year, grade=self.grade,
            section=self.section_a, enrollment_date=date(2026, 9, 5),
        )
        repeat_year(
            enrollment=enrollment, next_academic_year=self.next_year,
            repeat_date=date(2026, 10, 1),
        )
        enrollment.refresh_from_db()
        self.assertEqual(enrollment.section, self.section_a)
        self.assertIsNone(current_placement_for_enrollment(enrollment=enrollment))
        enroll_student(
            student=self.s2, academic_year=self.year, grade=self.grade,
            section=self.section_a, enrollment_date=date(2026, 10, 5),
        )
        enroll_student(
            student=self.s3, academic_year=self.year, grade=self.grade,
            section=self.section_a, enrollment_date=date(2026, 10, 6),
        )

    def test_capacity_counts_only_current_placements(self):
        enroll_student(
            student=self.student, academic_year=self.year, grade=self.grade,
            section=self.section_a, enrollment_date=date(2026, 9, 5),
        )
        enroll_student(
            student=self.s2, academic_year=self.year, grade=self.grade,
            section=self.section_a, enrollment_date=date(2026, 9, 6),
        )
        with self.assertRaises(ValidationError):
            validate_section_capacity(section=self.section_a)

    def test_capacity_exclude_instance_allows_own_reenroll(self):
        enrollment = enroll_student(
            student=self.student, academic_year=self.year, grade=self.grade,
            section=self.section_a, enrollment_date=date(2026, 9, 5),
        )
        enroll_student(
            student=self.s2, academic_year=self.year, grade=self.grade,
            section=self.section_a, enrollment_date=date(2026, 9, 6),
        )
        validate_section_capacity(section=self.section_a, instance=enrollment)


class CurrentEnrollmentsInSectionTests(AcademicsBaseData):
    """The operational roster must exclude withdrawn/graduated/repeated enrollments."""

    def setUp(self):
        super().setUp()
        self.enrollment = enroll_student(
            student=self.student, academic_year=self.year, grade=self.grade,
            section=self.section_a, enrollment_date=date(2026, 9, 5),
        )
        self.s2 = StudentProfile.objects.create(
            person=Person.objects.create(code="P-S2", first_name="Two", last_name="Student"),
            code="S-1002",
        )
        self.enrollment2 = enroll_student(
            student=self.s2, academic_year=self.year, grade=self.grade,
            section=self.section_a, enrollment_date=date(2026, 9, 6),
        )

    def test_current_roster_includes_active(self):
        roster = list(current_enrollments_in_section(section=self.section_a))
        self.assertIn(self.enrollment, roster)
        self.assertIn(self.enrollment2, roster)
        self.assertEqual(len(roster), 2)

    def test_current_roster_excludes_withdrawn(self):
        withdraw(enrollment=self.enrollment, withdrawal_date=date(2026, 10, 1))
        roster = list(current_enrollments_in_section(section=self.section_a))
        self.assertNotIn(self.enrollment, roster)
        self.assertIn(self.enrollment2, roster)
        self.assertEqual(len(roster), 1)

    def test_current_roster_excludes_graduated(self):
        graduate(enrollment=self.enrollment, graduation_date=date(2026, 10, 1))
        roster = list(current_enrollments_in_section(section=self.section_a))
        self.assertNotIn(self.enrollment, roster)
        self.assertEqual(len(roster), 1)

    def test_historical_selector_still_includes_withdrawn(self):
        withdraw(enrollment=self.enrollment, withdrawal_date=date(2026, 10, 1))
        historical = list(students_in_section(section=self.section_a))
        self.assertIn(self.enrollment, historical)


class PlacementModelCleanTests(AcademicsBaseData):
    """StudentEnrollmentSectionPlacement.clean() consistency checks."""

    def setUp(self):
        super().setUp()
        self.enrollment = enroll_student(
            student=self.student, academic_year=self.year, grade=self.grade,
            section=self.section_a, enrollment_date=date(2026, 9, 5),
        )

    def _make_placement(self, **overrides):
        defaults = dict(
            enrollment=self.enrollment,
            section=self.section_a,
            start_date=date(2026, 9, 5),
            is_current=False,
        )
        defaults.update(overrides)
        return StudentEnrollmentSectionPlacement(**defaults)

    def test_placement_clean_rejects_section_year_mismatch(self):
        section_next = create_section(
            academic_year=self.next_year, grade=self.grade, code="A"
        )
        placement = self._make_placement(section=section_next)
        with self.assertRaises(ValidationError):
            placement.clean()

    def test_placement_clean_rejects_section_grade_mismatch(self):
        other_grade = create_grade(name="Grade 2", code="G2", order=2)
        section_g2 = create_section(
            academic_year=self.year, grade=other_grade, code="A"
        )
        placement = self._make_placement(section=section_g2)
        with self.assertRaises(ValidationError):
            placement.clean()

    def test_placement_clean_rejects_end_before_start(self):
        placement = self._make_placement(
            start_date=date(2026, 10, 1), end_date=date(2026, 9, 1)
        )
        with self.assertRaises(ValidationError):
            placement.clean()

    def test_placement_clean_rejects_start_before_enrollment_date(self):
        placement = self._make_placement(start_date=date(2026, 8, 1))
        with self.assertRaises(ValidationError):
            placement.clean()

    def test_placement_clean_accepts_valid_closed_placement(self):
        placement = self._make_placement(
            start_date=date(2026, 9, 5), end_date=date(2026, 10, 1), is_current=False
        )
        placement.clean()
