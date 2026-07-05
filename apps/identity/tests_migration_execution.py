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
