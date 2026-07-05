"""Tests for identity migration write-path preparation.

Covers:
- RoleType seed (all 14 system roles present after migration)
- create_student_profile with legacy_student parameter
- create_student_profile existing behavior (without legacy_student)
- get_or_create_student_profile creates when missing
- get_or_create_student_profile returns existing when present
- get_or_create_student_profile detects unsafe conflicts (different person)
- get_or_create_student_profile detects unsafe conflicts (different legacy_student)
- get_or_create_role creates when missing
- get_or_create_role returns existing when present
- get_or_create_role does not create duplicates
- no legacy Student mutation
"""

from django.core.exceptions import ValidationError
from django.test import TestCase

from apps.attendance.models import Student as LegacyStudent
from apps.identity.models import Person, PersonRole, RoleType, StudentProfile
from apps.identity.services import (
    assign_role,
    create_person,
    create_student_profile,
    get_or_create_role,
    get_or_create_student_profile,
)


# ---------------------------------------------------------------------------
# RoleType seed
# ---------------------------------------------------------------------------

class RoleTypeSeedTests(TestCase):
    """Verify the 0002_seed_roletypes migration ran and seeded all
    system roles. These tests run after the test database is created
    (which applies all migrations including the seed)."""

    def test_all_system_roles_seeded(self):
        seeded_codes = set(
            RoleType.objects.filter(is_system=True).values_list("code", flat=True)
        )
        expected_codes = {choice[0] for choice in RoleType.RoleChoices.choices}
        self.assertEqual(seeded_codes, expected_codes)

    def test_role_count(self):
        self.assertEqual(RoleType.objects.filter(is_system=True).count(), 14)

    def test_student_role_exists(self):
        rt = RoleType.objects.get(code="student")
        self.assertTrue(rt.is_system)
        self.assertTrue(rt.is_active)
        self.assertEqual(rt.name, "Student")

    def test_seed_is_idempotent(self):
        """Re-running the seed logic should not create duplicates."""
        # Replicate the seed logic (update_or_create for each role).
        system_roles = [
            ("student", "Student"), ("staff", "Staff"),
            ("teacher", "Teacher"), ("parent", "Parent"),
            ("guardian", "Guardian"), ("guest", "Guest"),
            ("vendor", "Vendor"), ("administrator", "Administrator"),
            ("finance", "Finance"), ("hr", "HR"),
            ("principal", "Principal"), ("vice_principal", "Vice Principal"),
            ("librarian", "Librarian"), ("nurse", "Nurse"),
        ]
        before = RoleType.objects.count()
        for code, name in system_roles:
            RoleType.objects.update_or_create(
                code=code,
                defaults={"name": name, "is_active": True, "is_system": True},
            )
        after = RoleType.objects.count()
        self.assertEqual(before, after)

    def test_no_legacy_student_mutation(self):
        """RoleType seed does not touch legacy Student table."""
        # This is implicitly verified — the seed migration only writes
        # to identity_roletype. We verify the Student table is empty
        # (no test data created by the seed).
        self.assertEqual(LegacyStudent.objects.count(), 0)


# ---------------------------------------------------------------------------
# create_student_profile with legacy_student
# ---------------------------------------------------------------------------

class CreateStudentProfileLegacyTests(TestCase):
    def setUp(self):
        self.person = Person.objects.create(
            code="P-LG1", first_name="Legacy", last_name="Test"
        )
        self.legacy_student = LegacyStudent.objects.create(
            h_code="H-LG01", first_name="Legacy", last_name="Student"
        )

    def test_create_with_legacy_student(self):
        profile = create_student_profile(
            person=self.person,
            code="S-LG01",
            legacy_student=self.legacy_student,
        )
        self.assertEqual(profile.legacy_student, self.legacy_student)

    def test_create_without_legacy_student_defaults_none(self):
        profile = create_student_profile(
            person=self.person,
            code="S-LG02",
        )
        self.assertIsNone(profile.legacy_student)

    def test_existing_callers_not_affected(self):
        """Callers that don't pass legacy_student should work as before."""
        profile = create_student_profile(
            person=self.person,
            code="S-LG03",
            grade="G5",
            has_meal=True,
        )
        self.assertEqual(profile.code, "S-LG03")
        self.assertEqual(profile.grade, "G5")
        self.assertTrue(profile.has_meal)
        self.assertIsNone(profile.legacy_student)


# ---------------------------------------------------------------------------
# get_or_create_student_profile
# ---------------------------------------------------------------------------

class GetOrCreateStudentProfileTests(TestCase):
    def setUp(self):
        self.person = Person.objects.create(
            code="P-GOC1", first_name="Get", last_name="Create"
        )
        self.person2 = Person.objects.create(
            code="P-GOC2", first_name="Other", last_name="Person"
        )
        self.legacy_student = LegacyStudent.objects.create(
            h_code="H-GOC01", first_name="Legacy", last_name="Student"
        )
        self.legacy_student2 = LegacyStudent.objects.create(
            h_code="H-GOC02", first_name="Legacy2", last_name="Student2"
        )

    def test_creates_when_missing(self):
        profile, created = get_or_create_student_profile(
            person=self.person,
            code="S-GOC01",
            legacy_student=self.legacy_student,
        )
        self.assertTrue(created)
        self.assertEqual(profile.code, "S-GOC01")
        self.assertEqual(profile.legacy_student, self.legacy_student)

    def test_returns_existing_when_present(self):
        # Create first.
        get_or_create_student_profile(
            person=self.person,
            code="S-GOC02",
            legacy_student=self.legacy_student,
        )
        # Second call should return existing.
        profile, created = get_or_create_student_profile(
            person=self.person,
            code="S-GOC02",
            legacy_student=self.legacy_student,
        )
        self.assertFalse(created)
        self.assertEqual(profile.code, "S-GOC02")

    def test_returns_existing_without_legacy_student_arg(self):
        """If legacy_student is not passed but the profile exists,
        return it (don't raise)."""
        create_student_profile(
            person=self.person, code="S-GOC03",
            legacy_student=self.legacy_student,
        )
        profile, created = get_or_create_student_profile(
            person=self.person, code="S-GOC03",
        )
        self.assertFalse(created)
        self.assertEqual(profile.legacy_student, self.legacy_student)

    def test_detects_unsafe_conflict_different_person(self):
        """Profile with same code but different person → ValidationError."""
        create_student_profile(
            person=self.person, code="S-GOC04",
        )
        with self.assertRaises(ValidationError):
            get_or_create_student_profile(
                person=self.person2, code="S-GOC04",
            )

    def test_detects_unsafe_conflict_different_legacy_student(self):
        """Profile with same code, same person, but different
        legacy_student → ValidationError."""
        create_student_profile(
            person=self.person, code="S-GOC05",
            legacy_student=self.legacy_student,
        )
        with self.assertRaises(ValidationError):
            get_or_create_student_profile(
                person=self.person, code="S-GOC05",
                legacy_student=self.legacy_student2,
            )

    def test_no_legacy_student_mutation(self):
        """get_or_create_student_profile does not modify legacy Student."""
        before_h = self.legacy_student.h_code
        before_first = self.legacy_student.first_name
        get_or_create_student_profile(
            person=self.person, code="S-GOC06",
            legacy_student=self.legacy_student,
        )
        self.legacy_student.refresh_from_db()
        self.assertEqual(self.legacy_student.h_code, before_h)
        self.assertEqual(self.legacy_student.first_name, before_first)


# ---------------------------------------------------------------------------
# get_or_create_role
# ---------------------------------------------------------------------------

class GetOrCreateRoleTests(TestCase):
    def setUp(self):
        self.person = Person.objects.create(
            code="P-ROL1", first_name="Role", last_name="Test"
        )
        self.role_type = RoleType.objects.get(code="student")

    def test_creates_when_missing(self):
        role, created = get_or_create_role(
            person=self.person, role_type=self.role_type,
        )
        self.assertTrue(created)
        self.assertEqual(role.person, self.person)
        self.assertEqual(role.role_type, self.role_type)
        self.assertTrue(role.is_active)

    def test_returns_existing_when_present(self):
        get_or_create_role(
            person=self.person, role_type=self.role_type,
        )
        role, created = get_or_create_role(
            person=self.person, role_type=self.role_type,
        )
        self.assertFalse(created)
        self.assertEqual(role.person, self.person)

    def test_does_not_create_duplicates(self):
        get_or_create_role(
            person=self.person, role_type=self.role_type,
        )
        get_or_create_role(
            person=self.person, role_type=self.role_type,
        )
        self.assertEqual(
            PersonRole.objects.filter(
                person=self.person, role_type=self.role_type
            ).count(),
            1,
        )

    def test_with_optional_params(self):
        from datetime import date
        role, created = get_or_create_role(
            person=self.person,
            role_type=self.role_type,
            start_date=date(2026, 1, 1),
            notes="migration",
        )
        self.assertTrue(created)
        self.assertEqual(role.notes, "migration")
