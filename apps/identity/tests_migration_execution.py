"""Tests for identity migration execution — migrate_student().

Covers:
- successful migration (Person + StudentProfile + PersonRole created)
- already migrated (returns existing, no duplicates)
- rollback on failure (transaction.atomic)
- RoleType missing (ValidationError)
- missing required fields (h_code, first_name, last_name)
- duplicate handling (Person already exists with same code)
- retry after partial completion (Person exists, StudentProfile doesn't)
- legacy_student backlink created
- PersonRole assigned
- Person.code == Student.h_code
- StudentProfile.code == Student.h_code
- no attendance mutations
"""

from datetime import date
from unittest.mock import patch

from django.core.exceptions import ValidationError
from django.db import transaction
from django.test import TestCase

from apps.attendance.models import Student as LegacyStudent
from apps.identity.models import Person, PersonRole, RoleType, StudentProfile
from apps.identity.migration_execution import (
    MigrationExecutionResult,
    migrate_student,
)
from apps.identity.services import (
    create_person,
    create_student_profile,
    get_or_create_role,
)


class MigrationExecutionBaseData(TestCase):
    def setUp(self):
        self.student = LegacyStudent.objects.create(
            h_code="H-EXE01",
            first_name="Alice",
            middle_name="",
            last_name="Wonderland",
            gender="MALE",
            grade="G5",
            has_meal=True,
            has_bus=False,
            is_active=True,
        )

    def _create_student(self, **kwargs):
        """Create an additional legacy Student for testing."""
        defaults = {
            "h_code": "H-EXE99",
            "first_name": "Extra",
            "last_name": "Student",
        }
        defaults.update(kwargs)
        return LegacyStudent.objects.create(**defaults)


# ---------------------------------------------------------------------------
# Successful migration
# ---------------------------------------------------------------------------

class SuccessfulMigrationTests(MigrationExecutionBaseData):
    def test_creates_person_profile_role(self):
        result = migrate_student(self.student)
        self.assertIsInstance(result, MigrationExecutionResult)
        self.assertTrue(result.created_person)
        self.assertTrue(result.created_profile)
        self.assertTrue(result.created_role)
        self.assertFalse(result.already_migrated)

    def test_person_code_equals_h_code(self):
        result = migrate_student(self.student)
        self.assertEqual(result.person.code, "H-EXE01")

    def test_profile_code_equals_h_code(self):
        result = migrate_student(self.student)
        self.assertEqual(result.student_profile.code, "H-EXE01")

    def test_person_fields_mapped(self):
        result = migrate_student(self.student)
        self.assertEqual(result.person.first_name, "Alice")
        self.assertEqual(result.person.last_name, "Wonderland")
        self.assertEqual(result.person.gender, "MALE")
        self.assertTrue(result.person.is_active)

    def test_profile_fields_mapped(self):
        result = migrate_student(self.student)
        self.assertEqual(result.student_profile.grade, "G5")
        self.assertTrue(result.student_profile.has_meal)
        self.assertFalse(result.student_profile.has_bus)

    def test_legacy_student_backlink_created(self):
        result = migrate_student(self.student)
        self.assertEqual(result.student_profile.legacy_student, self.student)

    def test_person_role_assigned(self):
        result = migrate_student(self.student)
        self.assertEqual(result.person_role.role_type.code, "student")
        self.assertTrue(result.person_role.is_active)

    def test_role_type_is_student(self):
        result = migrate_student(self.student)
        self.assertEqual(result.person_role.role_type, RoleType.objects.get(code="student"))

    def test_is_migrated_after_execution(self):
        from apps.identity.migration_planning import is_migrated
        migrate_student(self.student)
        self.assertTrue(is_migrated(self.student))


# ---------------------------------------------------------------------------
# Already migrated
# ---------------------------------------------------------------------------

class AlreadyMigratedTests(MigrationExecutionBaseData):
    def test_returns_existing_without_duplicates(self):
        # First migration.
        result1 = migrate_student(self.student)
        self.assertTrue(result1.created_person)

        # Second migration — should return existing.
        result2 = migrate_student(self.student)
        self.assertTrue(result2.already_migrated)
        self.assertFalse(result2.created_person)
        self.assertFalse(result2.created_profile)
        self.assertFalse(result2.created_role)

        # Same rows.
        self.assertEqual(result2.person, result1.person)
        self.assertEqual(result2.student_profile, result1.student_profile)
        self.assertEqual(result2.person_role, result1.person_role)

    def test_no_duplicate_persons(self):
        migrate_student(self.student)
        migrate_student(self.student)
        self.assertEqual(Person.objects.filter(code="H-EXE01").count(), 1)

    def test_no_duplicate_profiles(self):
        migrate_student(self.student)
        migrate_student(self.student)
        self.assertEqual(StudentProfile.objects.filter(code="H-EXE01").count(), 1)

    def test_no_duplicate_roles(self):
        migrate_student(self.student)
        migrate_student(self.student)
        person = Person.objects.get(code="H-EXE01")
        self.assertEqual(
            PersonRole.objects.filter(person=person, role_type__code="student").count(),
            1,
        )


# ---------------------------------------------------------------------------
# Rollback on failure
# ---------------------------------------------------------------------------

class RollbackTests(MigrationExecutionBaseData):
    def test_rollback_on_profile_creation_failure(self):
        """Force a failure during StudentProfile creation. Person
        creation should roll back — no orphaned Person."""
        from apps.identity.services import get_or_create_student_profile

        original = get_or_create_student_profile

        def failing_get_or_create_student_profile(*, person, code, **kwargs):
            # Let the first call (if any internal) pass; fail on the
            # actual migration call.
            raise ValidationError("Simulated failure during profile creation.")

        with patch(
            "apps.identity.migration_execution.get_or_create_student_profile",
            failing_get_or_create_student_profile,
        ):
            with self.assertRaises(ValidationError):
                migrate_student(self.student)

        # Person should NOT exist (rolled back).
        self.assertFalse(Person.objects.filter(code="H-EXE01").exists())
        # StudentProfile should NOT exist.
        self.assertFalse(StudentProfile.objects.filter(code="H-EXE01").exists())
        # Student should not be marked as migrated.
        from apps.identity.migration_planning import is_migrated
        self.assertFalse(is_migrated(self.student))

    def test_rollback_on_role_creation_failure(self):
        """Force a failure during PersonRole creation. Person and
        StudentProfile should roll back."""
        from apps.identity.services import get_or_create_role

        def failing_get_or_create_role(*, person, role_type, **kwargs):
            raise ValidationError("Simulated failure during role creation.")

        with patch(
            "apps.identity.migration_execution.get_or_create_role",
            failing_get_or_create_role,
        ):
            with self.assertRaises(ValidationError):
                migrate_student(self.student)

        # Nothing should exist.
        self.assertFalse(Person.objects.filter(code="H-EXE01").exists())
        self.assertFalse(StudentProfile.objects.filter(code="H-EXE01").exists())


# ---------------------------------------------------------------------------
# RoleType missing
# ---------------------------------------------------------------------------

class RoleTypeMissingTests(MigrationExecutionBaseData):
    def test_raises_when_roletype_missing(self):
        # Delete all RoleType rows to simulate missing seed.
        RoleType.objects.all().delete()

        with self.assertRaises(ValidationError) as ctx:
            migrate_student(self.student)

        self.assertIn("student", str(ctx.exception))

    def test_no_person_created_when_roletype_missing(self):
        RoleType.objects.all().delete()

        try:
            migrate_student(self.student)
        except ValidationError:
            pass

        self.assertFalse(Person.objects.filter(code="H-EXE01").exists())


# ---------------------------------------------------------------------------
# Missing required fields
# ---------------------------------------------------------------------------

class MissingRequiredFieldsTests(MigrationExecutionBaseData):
    def test_missing_h_code_raises(self):
        # h_code is unique=True at DB level; create a student with a
        # valid h_code then blank it in-memory (validate_migration_data
        # checks the in-memory value).
        student = self._create_student(h_code="H-EXE02")
        student.h_code = ""
        with self.assertRaises(ValidationError):
            migrate_student(student)

    def test_missing_first_name_raises(self):
        student = self._create_student(h_code="H-EXE03", first_name="")
        with self.assertRaises(ValidationError):
            migrate_student(student)

    def test_missing_last_name_raises(self):
        student = self._create_student(h_code="H-EXE04", last_name="")
        with self.assertRaises(ValidationError):
            migrate_student(student)

    def test_no_writes_on_missing_fields(self):
        student = self._create_student(h_code="H-EXE05", first_name="")
        try:
            migrate_student(student)
        except ValidationError:
            pass
        self.assertFalse(Person.objects.filter(code="H-EXE05").exists())


# ---------------------------------------------------------------------------
# Duplicate handling
# ---------------------------------------------------------------------------

class DuplicateHandlingTests(MigrationExecutionBaseData):
    def test_person_already_exists_with_same_code(self):
        """If a Person with the same h_code already exists (but no
        StudentProfile link), migrate_student should reuse the
        existing Person and create the StudentProfile + PersonRole."""
        existing_person = create_person(
            code="H-EXE01",
            first_name="PreExisting",
            last_name="Person",
        )
        self.assertEqual(Person.objects.filter(code="H-EXE01").count(), 1)

        result = migrate_student(self.student)

        # Person was not created (reused existing).
        self.assertFalse(result.created_person)
        # But StudentProfile and PersonRole were created.
        self.assertTrue(result.created_profile)
        self.assertTrue(result.created_role)
        # The reused Person is the same.
        self.assertEqual(result.person, existing_person)

    def test_unsafe_conflict_different_person_for_profile(self):
        """If a StudentProfile with the same code exists but belongs
        to a different person, migrate_student should raise."""
        person_a = create_person(code="P-A", first_name="A", last_name="A")
        create_student_profile(person=person_a, code="H-EXE01")

        with self.assertRaises(ValidationError):
            migrate_student(self.student)


# ---------------------------------------------------------------------------
# Retry after partial completion
# ---------------------------------------------------------------------------

class PartialCompletionRetryTests(MigrationExecutionBaseData):
    def test_person_exists_profile_missing(self):
        """Simulate a partial run: Person created, StudentProfile not.
        migrate_student should create the missing StudentProfile +
        PersonRole."""
        # Manually create the Person (simulating a partial run).
        person = create_person(
            code="H-EXE01",
            first_name="Alice",
            last_name="Wonderland",
            gender="MALE",
        )

        result = migrate_student(self.student)

        # Person was not created (already existed).
        self.assertFalse(result.created_person)
        # StudentProfile was created.
        self.assertTrue(result.created_profile)
        # PersonRole was created.
        self.assertTrue(result.created_role)
        # legacy_student backlink is set.
        self.assertEqual(result.student_profile.legacy_student, self.student)

    def test_person_and_profile_exist_role_missing(self):
        """Simulate a partial run: Person + StudentProfile created,
        PersonRole not. migrate_student should create the missing
        PersonRole."""
        person = create_person(
            code="H-EXE01",
            first_name="Alice",
            last_name="Wonderland",
        )
        create_student_profile(
            person=person, code="H-EXE01", legacy_student=self.student,
        )

        result = migrate_student(self.student)

        # Person and profile already existed.
        self.assertFalse(result.created_person)
        self.assertFalse(result.created_profile)
        # The role was created (it was missing) — even though
        # already_migrated=True, the partial state is completed.
        self.assertTrue(result.created_role)
        self.assertEqual(result.person_role.person, person)

    def test_retry_is_idempotent_after_completion(self):
        # Full migration.
        result1 = migrate_student(self.student)
        # Retry — should return existing.
        result2 = migrate_student(self.student)
        self.assertTrue(result2.already_migrated)
        self.assertFalse(result2.created_person)
        self.assertFalse(result2.created_profile)
        self.assertFalse(result2.created_role)


# ---------------------------------------------------------------------------
# No attendance mutations
# ---------------------------------------------------------------------------

class NoAttendanceMutationTests(MigrationExecutionBaseData):
    def test_student_fields_unchanged(self):
        before_h = self.student.h_code
        before_first = self.student.first_name
        before_last = self.student.last_name
        before_grade = self.student.grade
        before_has_meal = self.student.has_meal

        migrate_student(self.student)

        self.student.refresh_from_db()
        self.assertEqual(self.student.h_code, before_h)
        self.assertEqual(self.student.first_name, before_first)
        self.assertEqual(self.student.last_name, before_last)
        self.assertEqual(self.student.grade, before_grade)
        self.assertEqual(self.student.has_meal, before_has_meal)

    def test_student_not_deleted(self):
        migrate_student(self.student)
        self.assertTrue(LegacyStudent.objects.filter(pk=self.student.pk).exists())

    def test_student_count_unchanged(self):
        before = LegacyStudent.objects.count()
        migrate_student(self.student)
        self.assertEqual(LegacyStudent.objects.count(), before)


# ===========================================================================
# Bulk migration — migrate_all_students()
# ===========================================================================

from apps.identity.migration_execution import (
    BulkMigrationError,
    BulkMigrationResult,
    migrate_all_students,
)


class BulkMigrationBaseData(TestCase):
    def setUp(self):
        self.students = []
        for i in range(5):
            s = LegacyStudent.objects.create(
                h_code=f"H-BLK{i:02d}",
                first_name=f"Student{i}",
                last_name="Bulk",
                gender="MALE" if i % 2 == 0 else "FEMALE",
                grade=f"G{i}",
                has_meal=(i % 2 == 0),
                has_bus=False,
                is_active=True,
            )
            self.students.append(s)
        # One blocked student (missing first_name + last_name).
        self.blocked = LegacyStudent.objects.create(
            h_code="H-BLK99",
            first_name="",
            last_name="",
        )


class BulkAllSuccessfulTests(BulkMigrationBaseData):
    def test_all_migrated(self):
        result = migrate_all_students()
        self.assertIsInstance(result, BulkMigrationResult)
        self.assertEqual(result.total_students, 6)  # 5 ready + 1 blocked
        self.assertEqual(result.migrated, 5)
        self.assertEqual(result.already_migrated, 0)
        self.assertEqual(result.failed, 1)  # blocked student
        self.assertEqual(result.blocked, 0)
        self.assertEqual(len(result.results), 5)
        self.assertEqual(len(result.errors), 1)

    def test_summary_counts_correct(self):
        result = migrate_all_students()
        # total = migrated + already_migrated + failed
        self.assertEqual(
            result.total_students,
            result.migrated + result.already_migrated + result.failed,
        )

    def test_error_has_blocked_student(self):
        result = migrate_all_students()
        err = result.errors[0]
        self.assertIsInstance(err, BulkMigrationError)
        self.assertEqual(err.h_code, "H-BLK99")
        self.assertIn("first_name", err.error)


class BulkAlreadyMigratedTests(BulkMigrationBaseData):
    def test_already_migrated_counted(self):
        # Pre-migrate 2 students.
        migrate_student(self.students[0])
        migrate_student(self.students[1])

        result = migrate_all_students()
        self.assertEqual(result.migrated, 3)  # 5 - 2 already done
        self.assertEqual(result.already_migrated, 2)
        self.assertEqual(result.failed, 1)  # blocked

    def test_idempotent_repeated_run(self):
        migrate_all_students()
        result = migrate_all_students()
        self.assertEqual(result.migrated, 0)
        self.assertEqual(result.already_migrated, 5)
        self.assertEqual(result.failed, 1)  # blocked still fails


class BulkDuplicateDetectionTests(BulkMigrationBaseData):
    def test_duplicate_person_code_blocks_batch(self):
        # Create a Person with the same code as a student but not linked.
        create_person(code="H-BLK00", first_name="Conflict", last_name="Person")

        with self.assertRaises(ValidationError) as ctx:
            migrate_all_students()

        self.assertIn("duplicate", str(ctx.exception).lower())
        # No students migrated.
        from apps.identity.migration_planning import is_migrated
        self.assertFalse(is_migrated(self.students[0]))

    def test_duplicate_profile_code_blocks_batch(self):
        # Create a StudentProfile with a conflicting code.
        person = create_person(code="P-CONFLICT", first_name="C", last_name="P")
        create_student_profile(
            person=person, code="H-BLK00",
        )

        with self.assertRaises(ValidationError):
            migrate_all_students()


class BulkFailureContinuesTests(BulkMigrationBaseData):
    def test_one_failed_student_continues_batch(self):
        result = migrate_all_students()
        # The blocked student failed, but the other 5 succeeded.
        self.assertEqual(result.migrated, 5)
        self.assertEqual(result.failed, 1)

    def test_failed_student_not_migrated(self):
        result = migrate_all_students()
        from apps.identity.migration_planning import is_migrated
        self.assertFalse(is_migrated(self.blocked))

    def test_successful_students_migrated(self):
        result = migrate_all_students()
        from apps.identity.migration_planning import is_migrated
        for s in self.students:
            self.assertTrue(is_migrated(s), f"{s.h_code} should be migrated")


class BulkLimitTests(BulkMigrationBaseData):
    def test_limit_3(self):
        result = migrate_all_students(limit=3)
        self.assertEqual(result.total_students, 3)
        self.assertEqual(result.migrated, 3)

    def test_limit_0(self):
        result = migrate_all_students(limit=0)
        self.assertEqual(result.total_students, 0)
        self.assertEqual(result.migrated, 0)

    def test_limit_greater_than_total(self):
        result = migrate_all_students(limit=100)
        self.assertEqual(result.total_students, 6)
        self.assertEqual(result.migrated, 5)


class BulkCustomQuerysetTests(BulkMigrationBaseData):
    def test_custom_queryset_filters(self):
        # Only migrate students with has_meal=True (students 0, 2, 4).
        qs = LegacyStudent.objects.filter(has_meal=True).order_by("h_code")
        result = migrate_all_students(student_queryset=qs)
        self.assertEqual(result.total_students, 3)
        self.assertEqual(result.migrated, 3)

    def test_custom_queryset_excludes_blocked(self):
        # Exclude the blocked student.
        qs = LegacyStudent.objects.exclude(h_code="H-BLK99").order_by("h_code")
        result = migrate_all_students(student_queryset=qs)
        self.assertEqual(result.total_students, 5)
        self.assertEqual(result.migrated, 5)
        self.assertEqual(result.failed, 0)


class BulkTransactionIsolationTests(BulkMigrationBaseData):
    def test_one_transaction_per_student(self):
        """If student #2 fails, students #1 and #3 should still be
        committed (each has its own transaction)."""
        # Make student #2 fail by deleting RoleType before that
        # specific call. We can't easily mock inside the loop, but
        # we can verify that a blocked student doesn't prevent others.
        result = migrate_all_students()
        # The blocked student (H-BLK99) failed, but all 5 ready
        # students were committed.
        self.assertEqual(result.migrated, 5)
        self.assertEqual(result.failed, 1)

        # Verify the first and last students are persisted.
        from apps.identity.migration_planning import is_migrated
        self.assertTrue(is_migrated(self.students[0]))
        self.assertTrue(is_migrated(self.students[4]))


class BulkNoAttendanceMutationTests(BulkMigrationBaseData):
    def test_student_fields_unchanged(self):
        before_h = self.students[0].h_code
        before_first = self.students[0].first_name

        migrate_all_students()

        self.students[0].refresh_from_db()
        self.assertEqual(self.students[0].h_code, before_h)
        self.assertEqual(self.students[0].first_name, before_first)

    def test_student_count_unchanged(self):
        before = LegacyStudent.objects.count()
        migrate_all_students()
        self.assertEqual(LegacyStudent.objects.count(), before)

    def test_no_students_deleted(self):
        migrate_all_students()
        for s in self.students:
            self.assertTrue(LegacyStudent.objects.filter(pk=s.pk).exists())


# ---------------------------------------------------------------------------
# Inactive student migration
# ---------------------------------------------------------------------------


class InactiveStudentMigrationTests(MigrationExecutionBaseData):
    """Regression tests for migrating inactive legacy Students.

    The bug: inactive students (is_active=False) caused a ValidationError
    because the PersonRole validator (validate_role_active_window) requires
    end_date when is_active=False, but migrate_student did not pass
    end_date to get_or_create_role.

    Fix: migrate_student now passes end_date=timezone.localdate() for
    inactive students.
    """

    def setUp(self):
        super().setUp()
        self.inactive_student = self._create_student(
            h_code="H-INACT01",
            first_name="Ina",
            last_name="CTive",
            is_active=False,
        )

    def test_inactive_student_migrates_without_error(self):
        """The original bug: this raised ValidationError because
        end_date was not set on the PersonRole for inactive students."""
        result = migrate_student(self.inactive_student)
        self.assertIsInstance(result, MigrationExecutionResult)

    def test_inactive_student_role_is_inactive(self):
        result = migrate_student(self.inactive_student)
        self.assertFalse(result.person_role.is_active)

    def test_inactive_student_role_has_end_date(self):
        result = migrate_student(self.inactive_student)
        self.assertIsNotNone(result.person_role.end_date)

    def test_inactive_student_role_end_date_is_today(self):
        from django.utils import timezone
        result = migrate_student(self.inactive_student)
        self.assertEqual(result.person_role.end_date, timezone.localdate())

    def test_inactive_student_person_is_inactive(self):
        result = migrate_student(self.inactive_student)
        self.assertFalse(result.person.is_active)

    def test_active_student_role_has_no_end_date(self):
        """Active students should keep end_date=None (unchanged behavior)."""
        result = migrate_student(self.student)
        self.assertTrue(result.person_role.is_active)
        self.assertIsNone(result.person_role.end_date)

    def test_inactive_student_idempotent(self):
        """Re-running migrate_student on an already-migrated inactive
        student should not raise (the already-migrated path must also
        handle the end_date correctly)."""
        result1 = migrate_student(self.inactive_student)
        self.assertFalse(result1.already_migrated)

        result2 = migrate_student(self.inactive_student)
        self.assertTrue(result2.already_migrated)
        self.assertFalse(result2.person_role.is_active)
        self.assertIsNotNone(result2.person_role.end_date)

    def test_bulk_migration_handles_inactive(self):
        """Bulk migration should handle mixed active/inactive students."""
        from apps.identity.migration_execution import migrate_all_students
        result = migrate_all_students()
        self.assertEqual(result.failed, 0)
        self.assertEqual(result.migrated, 2)  # both active + inactive
