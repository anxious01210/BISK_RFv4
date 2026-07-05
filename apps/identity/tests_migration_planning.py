"""Tests for the identity migration planning infrastructure.

Covers:
- dry-run behavior (plan produces no writes)
- no legacy mutation (Student table unchanged)
- duplicate detection (Person.code / StudentProfile.code conflicts)
- missing required data validation
- idempotent planning behavior (same result on repeated calls)
- already-migrated detection (is_migrated, skip in plan)
"""

from django.test import TestCase

from apps.attendance.models import Student as LegacyStudent
from apps.identity.models import Person, StudentProfile

from .migration_planning import (
    MigrationReport,
    StudentMigrationPlan,
    detect_duplicate_person_codes,
    detect_duplicate_student_profile_codes,
    is_migrated,
    plan_all_student_migrations,
    plan_student_migration,
    validate_migration_data,
)


# ---------------------------------------------------------------------------
# Test base data
# ---------------------------------------------------------------------------

class MigrationPlanningBaseData(TestCase):
    def setUp(self):
        self.student1 = LegacyStudent.objects.create(
            h_code="H-MIG01",
            first_name="Alice",
            middle_name="",
            last_name="Wonderland",
            gender="MALE",
            grade="G5",
            has_meal=True,
            has_bus=False,
            is_active=True,
        )
        self.student2 = LegacyStudent.objects.create(
            h_code="H-MIG02",
            first_name="Bob",
            middle_name="",
            last_name="Builder",
            gender="FEMALE",
            grade="G3",
            has_meal=False,
            has_bus=True,
            is_active=True,
        )
        # A student with missing data.
        self.student_incomplete = LegacyStudent.objects.create(
            h_code="H-MIG03",
            first_name="",
            middle_name="",
            last_name="",
            gender=None,
            grade="",
            has_meal=False,
            has_bus=False,
            is_active=True,
        )
        # A student with no h_code (edge case).
        # h_code is unique and required at the DB level, but we test
        # the validator defensively.
        self.student_no_hcode = LegacyStudent.objects.create(
            h_code="H-MIG04",
            first_name="Has",
            middle_name="",
            last_name="Code",
        )


# ---------------------------------------------------------------------------
# validate_migration_data
# ---------------------------------------------------------------------------

class ValidateMigrationDataTests(MigrationPlanningBaseData):
    def test_valid_student_no_issues(self):
        issues = validate_migration_data(self.student1)
        self.assertEqual(issues, [])

    def test_missing_first_name(self):
        self.student1.first_name = ""
        issues = validate_migration_data(self.student1)
        self.assertIn("Missing first_name.", issues)

    def test_missing_last_name(self):
        self.student1.last_name = ""
        issues = validate_migration_data(self.student1)
        self.assertIn("Missing last_name.", issues)

    def test_missing_first_and_last(self):
        issues = validate_migration_data(self.student_incomplete)
        self.assertIn("Missing first_name.", issues)
        self.assertIn("Missing last_name.", issues)

    def test_empty_h_code(self):
        self.student1.h_code = ""
        issues = validate_migration_data(self.student1)
        self.assertIn("Missing h_code (required for Person.code).", issues)


# ---------------------------------------------------------------------------
# is_migrated
# ---------------------------------------------------------------------------

class IsMigratedTests(MigrationPlanningBaseData):
    def test_not_migrated(self):
        self.assertFalse(is_migrated(self.student1))

    def test_migrated(self):
        # Create a Person + StudentProfile linked to student1.
        person = Person.objects.create(
            code="H-MIG01", first_name="Alice", last_name="Wonderland"
        )
        StudentProfile.objects.create(
            person=person, code="H-MIG01", legacy_student=self.student1
        )
        self.assertTrue(is_migrated(self.student1))


# ---------------------------------------------------------------------------
# plan_student_migration (dry-run)
# ---------------------------------------------------------------------------

class PlanStudentMigrationTests(MigrationPlanningBaseData):
    def test_plan_happy_path(self):
        plan = plan_student_migration(self.student1)
        self.assertIsInstance(plan, StudentMigrationPlan)
        self.assertEqual(plan.h_code, "H-MIG01")
        self.assertEqual(plan.person_code, "H-MIG01")
        self.assertEqual(plan.first_name, "Alice")
        self.assertEqual(plan.last_name, "Wonderland")
        self.assertEqual(plan.gender, "MALE")
        self.assertEqual(plan.grade, "G5")
        self.assertTrue(plan.has_meal)
        self.assertFalse(plan.has_bus)
        self.assertEqual(plan.student_profile_code, "H-MIG01")
        self.assertEqual(plan.role_type_code, "student")
        self.assertFalse(plan.already_migrated)
        self.assertEqual(plan.issues, [])
        self.assertTrue(plan.is_ready)

    def test_plan_already_migrated(self):
        person = Person.objects.create(
            code="H-MIG01", first_name="Alice", last_name="Wonderland"
        )
        StudentProfile.objects.create(
            person=person, code="H-MIG01", legacy_student=self.student1
        )
        plan = plan_student_migration(self.student1)
        self.assertTrue(plan.already_migrated)
        self.assertFalse(plan.is_ready)

    def test_plan_with_issues(self):
        plan = plan_student_migration(self.student_incomplete)
        self.assertFalse(plan.is_ready)
        self.assertIn("Missing first_name.", plan.issues)
        self.assertIn("Missing last_name.", plan.issues)

    def test_plan_gender_normalization(self):
        plan = plan_student_migration(self.student1)
        self.assertEqual(plan.gender, "MALE")
        plan2 = plan_student_migration(self.student2)
        self.assertEqual(plan2.gender, "FEMALE")

    def test_plan_gender_none(self):
        plan = plan_student_migration(self.student_incomplete)
        self.assertIsNone(plan.gender)

    def test_plan_no_writes(self):
        """Planning must NOT create any Person / StudentProfile rows."""
        before_persons = Person.objects.count()
        before_profiles = StudentProfile.objects.count()
        plan_student_migration(self.student1)
        self.assertEqual(Person.objects.count(), before_persons)
        self.assertEqual(StudentProfile.objects.count(), before_profiles)


# ---------------------------------------------------------------------------
# plan_all_student_migrations (dry-run aggregate)
# ---------------------------------------------------------------------------

class PlanAllStudentMigrationsTests(MigrationPlanningBaseData):
    def test_report_counts(self):
        report = plan_all_student_migrations()
        self.assertIsInstance(report, MigrationReport)
        self.assertEqual(report.total, 4)  # 4 students in setUp
        self.assertEqual(report.already_migrated, 0)
        self.assertEqual(report.ready, 3)  # student1, student2, student_no_hcode
        self.assertEqual(report.blocked, 1)  # student_incomplete

    def test_report_properties(self):
        report = plan_all_student_migrations()
        self.assertEqual(len(report.ready_plans), 3)
        self.assertEqual(len(report.blocked_plans), 1)
        self.assertEqual(len(report.skipped_plans), 0)

    def test_report_no_writes(self):
        before_persons = Person.objects.count()
        before_profiles = StudentProfile.objects.count()
        plan_all_student_migrations()
        self.assertEqual(Person.objects.count(), before_persons)
        self.assertEqual(StudentProfile.objects.count(), before_profiles)

    def test_report_includes_all_students(self):
        report = plan_all_student_migrations()
        h_codes = {p.h_code for p in report.plans}
        self.assertIn("H-MIG01", h_codes)
        self.assertIn("H-MIG02", h_codes)
        self.assertIn("H-MIG03", h_codes)
        self.assertIn("H-MIG04", h_codes)

    def test_report_with_partial_migration(self):
        # Migrate student1.
        person = Person.objects.create(
            code="H-MIG01", first_name="Alice", last_name="Wonderland"
        )
        StudentProfile.objects.create(
            person=person, code="H-MIG01", legacy_student=self.student1
        )
        report = plan_all_student_migrations()
        self.assertEqual(report.total, 4)
        self.assertEqual(report.already_migrated, 1)
        self.assertEqual(report.ready, 2)
        self.assertEqual(report.blocked, 1)
        self.assertEqual(len(report.skipped_plans), 1)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class IdempotencyTests(MigrationPlanningBaseData):
    def test_planning_twice_same_result(self):
        report1 = plan_all_student_migrations()
        report2 = plan_all_student_migrations()
        self.assertEqual(report1.total, report2.total)
        self.assertEqual(report1.ready, report2.ready)
        self.assertEqual(report1.blocked, report2.blocked)
        self.assertEqual(report1.already_migrated, report2.already_migrated)

    def test_plan_single_student_twice_same_result(self):
        plan1 = plan_student_migration(self.student1)
        plan2 = plan_student_migration(self.student1)
        self.assertEqual(plan1, plan2)

    def test_planning_does_not_mark_as_migrated(self):
        """Planning a student must NOT create the StudentProfile link."""
        plan_student_migration(self.student1)
        self.assertFalse(is_migrated(self.student1))
        # Plan again — should still be "not migrated".
        plan = plan_student_migration(self.student1)
        self.assertFalse(plan.already_migrated)


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

class DuplicateDetectionTests(MigrationPlanningBaseData):
    def test_no_duplicates_when_identity_empty(self):
        self.assertEqual(detect_duplicate_person_codes(), [])
        self.assertEqual(detect_duplicate_student_profile_codes(), [])

    def test_person_code_conflict(self):
        # Create a Person with the same code as student1 but NOT linked.
        Person.objects.create(
            code="H-MIG01", first_name="Existing", last_name="Person"
        )
        dupes = detect_duplicate_person_codes()
        self.assertIn("H-MIG01", dupes)

    def test_student_profile_code_conflict(self):
        # Create a StudentProfile with the same code as student1 but
        # linked to a DIFFERENT legacy student.
        person = Person.objects.create(
            code="P-OTHER", first_name="Other", last_name="Person"
        )
        StudentProfile.objects.create(
            person=person, code="H-MIG01", legacy_student=self.student2
        )
        dupes = detect_duplicate_student_profile_codes()
        self.assertIn("H-MIG01", dupes)

    def test_no_conflict_when_properly_linked(self):
        # Person + StudentProfile properly linked to student1 — no conflict.
        person = Person.objects.create(
            code="H-MIG01", first_name="Alice", last_name="Wonderland"
        )
        StudentProfile.objects.create(
            person=person, code="H-MIG01", legacy_student=self.student1
        )
        self.assertNotIn("H-MIG01", detect_duplicate_person_codes())
        self.assertNotIn("H-MIG01", detect_duplicate_student_profile_codes())


# ---------------------------------------------------------------------------
# No legacy mutation
# ---------------------------------------------------------------------------

class NoLegacyMutationTests(MigrationPlanningBaseData):
    def test_student_table_unch_after_planning(self):
        before_count = LegacyStudent.objects.count()
        before_h_code = self.student1.h_code
        before_first = self.student1.first_name

        plan_all_student_migrations()

        self.student1.refresh_from_db()
        self.assertEqual(LegacyStudent.objects.count(), before_count)
        self.assertEqual(self.student1.h_code, before_h_code)
        self.assertEqual(self.student1.first_name, before_first)

    def test_student_not_deleted(self):
        plan_all_student_migrations()
        self.assertTrue(LegacyStudent.objects.filter(pk=self.student1.pk).exists())
        self.assertTrue(LegacyStudent.objects.filter(pk=self.student_incomplete.pk).exists())
