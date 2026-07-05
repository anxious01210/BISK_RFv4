"""Tests for identity migration reconciliation.

Covers:
- fully migrated student has no issues
- unmigrated student is reported
- missing student role is reported
- code mismatch (Person) is reported
- code mismatch (StudentProfile) is reported
- name mismatch is reported (informational)
- duplicate/conflict detection is included
- orphan StudentProfile detection
- orphan Person detection
- no database writes
- summary counts are correct
"""

from django.test import TestCase

from apps.attendance.models import Student as LegacyStudent
from apps.identity.models import Person, PersonRole, RoleType, StudentProfile
from apps.identity.migration_reconciliation import (
    MigrationReconciliationReport,
    StudentReconciliationIssue,
    StudentReconciliationRow,
    reconcile_all,
    reconcile_student,
)
from apps.identity.services import (
    create_person,
    create_student_profile,
    get_or_create_role,
)


class ReconciliationBaseData(TestCase):
    def setUp(self):
        self.student_ok = LegacyStudent.objects.create(
            h_code="H-REC01",
            first_name="Alice",
            last_name="Wonderland",
            gender="MALE",
            grade="G5",
            has_meal=True,
        )
        self.student_unmigrated = LegacyStudent.objects.create(
            h_code="H-REC02",
            first_name="Bob",
            last_name="Builder",
        )

    def _migrate_student(self, student):
        """Fully migrate a student (Person + StudentProfile + PersonRole)."""
        person = create_person(
            code=student.h_code,
            first_name=student.first_name,
            last_name=student.last_name,
            gender=student.gender or "",
        )
        profile = create_student_profile(
            person=person,
            code=student.h_code,
            grade=student.grade or "",
            has_meal=student.has_meal,
            has_bus=student.has_bus,
            legacy_student=student,
        )
        role_type = RoleType.objects.get(code="student")
        get_or_create_role(person=person, role_type=role_type)
        return person, profile


# ---------------------------------------------------------------------------
# reconcile_student — fully migrated (no issues)
# ---------------------------------------------------------------------------

class FullyMigratedTests(ReconciliationBaseData):
    def test_no_issues(self):
        self._migrate_student(self.student_ok)
        row = reconcile_student(self.student_ok)
        self.assertIsInstance(row, StudentReconciliationRow)
        self.assertTrue(row.is_migrated)
        self.assertEqual(row.issues, [])
        self.assertFalse(row.has_issues)

    def test_person_code_matches(self):
        self._migrate_student(self.student_ok)
        row = reconcile_student(self.student_ok)
        self.assertEqual(row.person_code, "H-REC01")

    def test_profile_code_matches(self):
        self._migrate_student(self.student_ok)
        row = reconcile_student(self.student_ok)
        self.assertEqual(row.student_profile_code, "H-REC01")

    def test_has_student_role(self):
        self._migrate_student(self.student_ok)
        row = reconcile_student(self.student_ok)
        self.assertTrue(row.has_student_role)


# ---------------------------------------------------------------------------
# reconcile_student — unmigrated
# ---------------------------------------------------------------------------

class UnmigratedTests(ReconciliationBaseData):
    def test_unmigrated_reported(self):
        row = reconcile_student(self.student_unmigrated)
        self.assertFalse(row.is_migrated)
        self.assertTrue(row.has_issues)
        issue_types = [i.issue_type for i in row.issues]
        self.assertIn("unmigrated", issue_types)

    def test_unmigrated_detail(self):
        row = reconcile_student(self.student_unmigrated)
        detail = row.issues[0].detail
        self.assertIn("legacy_student", detail.lower())


# ---------------------------------------------------------------------------
# reconcile_student — missing student role
# ---------------------------------------------------------------------------

class MissingStudentRoleTests(ReconciliationBaseData):
    def test_missing_role_reported(self):
        # Create Person + StudentProfile but no PersonRole.
        person = create_person(
            code="H-REC01", first_name="Alice", last_name="Wonderland",
        )
        create_student_profile(
            person=person, code="H-REC01", legacy_student=self.student_ok,
        )
        row = reconcile_student(self.student_ok)
        self.assertTrue(row.is_migrated)
        issue_types = [i.issue_type for i in row.issues]
        self.assertIn("missing_student_role", issue_types)


# ---------------------------------------------------------------------------
# reconcile_student — code mismatch
# ---------------------------------------------------------------------------

class CodeMismatchTests(ReconciliationBaseData):
    def test_person_code_mismatch(self):
        person = create_person(
            code="WRONG-CODE", first_name="Alice", last_name="Wonderland",
        )
        create_student_profile(
            person=person, code="H-REC01", legacy_student=self.student_ok,
        )
        role_type = RoleType.objects.get(code="student")
        get_or_create_role(person=person, role_type=role_type)

        row = reconcile_student(self.student_ok)
        issue_types = [i.issue_type for i in row.issues]
        self.assertIn("code_mismatch_person", issue_types)

    def test_profile_code_mismatch(self):
        person = create_person(
            code="H-REC01", first_name="Alice", last_name="Wonderland",
        )
        # Profile with wrong code.
        create_student_profile(
            person=person, code="WRONG-PROFILE", legacy_student=self.student_ok,
        )
        role_type = RoleType.objects.get(code="student")
        get_or_create_role(person=person, role_type=role_type)

        row = reconcile_student(self.student_ok)
        issue_types = [i.issue_type for i in row.issues]
        self.assertIn("code_mismatch_profile", issue_types)


# ---------------------------------------------------------------------------
# reconcile_student — name mismatch (informational)
# ---------------------------------------------------------------------------

class NameMismatchTests(ReconciliationBaseData):
    def test_first_name_mismatch_reported(self):
        person = create_person(
            code="H-REC01", first_name="Different", last_name="Wonderland",
        )
        create_student_profile(
            person=person, code="H-REC01", legacy_student=self.student_ok,
        )
        role_type = RoleType.objects.get(code="student")
        get_or_create_role(person=person, role_type=role_type)

        row = reconcile_student(self.student_ok)
        issue_types = [i.issue_type for i in row.issues]
        self.assertIn("name_mismatch_first", issue_types)

    def test_last_name_mismatch_reported(self):
        person = create_person(
            code="H-REC01", first_name="Alice", last_name="Different",
        )
        create_student_profile(
            person=person, code="H-REC01", legacy_student=self.student_ok,
        )
        role_type = RoleType.objects.get(code="student")
        get_or_create_role(person=person, role_type=role_type)

        row = reconcile_student(self.student_ok)
        issue_types = [i.issue_type for i in row.issues]
        self.assertIn("name_mismatch_last", issue_types)


# ---------------------------------------------------------------------------
# reconcile_all — summary counts + cross-cutting
# ---------------------------------------------------------------------------

class ReconcileAllTests(ReconciliationBaseData):
    def test_summary_counts(self):
        self._migrate_student(self.student_ok)
        report = reconcile_all()
        self.assertIsInstance(report, MigrationReconciliationReport)
        self.assertEqual(report.total_students, 2)
        self.assertEqual(report.migrated, 1)
        self.assertEqual(report.unmigrated, 1)
        # student_ok has no issues; student_unmigrated has 1 issue.
        self.assertEqual(report.issues_found, 1)

    def test_rows_with_issues(self):
        self._migrate_student(self.student_ok)
        report = reconcile_all()
        issue_rows = report.rows_with_issues
        self.assertEqual(len(issue_rows), 1)
        self.assertEqual(issue_rows[0].h_code, "H-REC02")

    def test_all_rows_present(self):
        report = reconcile_all()
        h_codes = {r.h_code for r in report.rows}
        self.assertIn("H-REC01", h_codes)
        self.assertIn("H-REC02", h_codes)


# ---------------------------------------------------------------------------
# Duplicate detection (reused from planning)
# ---------------------------------------------------------------------------

class DuplicateDetectionTests(ReconciliationBaseData):
    def test_duplicate_person_codes_reported(self):
        # Create a Person with same code as student_ok but not linked.
        create_person(code="H-REC01", first_name="Conflict", last_name="Person")
        report = reconcile_all()
        self.assertIn("H-REC01", report.duplicate_person_codes)

    def test_duplicate_profile_codes_reported(self):
        person = create_person(code="P-X", first_name="X", last_name="Y")
        create_student_profile(person=person, code="H-REC01")
        report = reconcile_all()
        self.assertIn("H-REC01", report.duplicate_student_profile_codes)

    def test_no_duplicates_when_clean(self):
        self._migrate_student(self.student_ok)
        report = reconcile_all()
        self.assertEqual(report.duplicate_person_codes, [])
        self.assertEqual(report.duplicate_student_profile_codes, [])


# ---------------------------------------------------------------------------
# Orphan detection
# ---------------------------------------------------------------------------

class OrphanDetectionTests(ReconciliationBaseData):
    def test_orphan_student_profile_detected(self):
        # Create a StudentProfile with legacy_student=None.
        person = create_person(code="P-ORPH", first_name="Orphan", last_name="Profile")
        create_student_profile(person=person, code="S-ORPH")
        report = reconcile_all()
        self.assertTrue(len(report.orphan_student_profiles) > 0)

    def test_orphan_person_detected(self):
        # Create a Person with no StudentProfile.
        create_person(code="P-ORPH2", first_name="Orphan", last_name="Person")
        report = reconcile_all()
        self.assertTrue(len(report.orphan_persons) > 0)

    def test_no_orphans_when_clean(self):
        self._migrate_student(self.student_ok)
        report = reconcile_all()
        self.assertEqual(report.orphan_student_profiles, [])
        # student_unmigrated has no Person, so no orphan either.
        # But if there are no Person rows without StudentProfile, it's clean.
        # (Person created in _migrate_student has a StudentProfile.)
        self.assertEqual(report.orphan_persons, [])


# ---------------------------------------------------------------------------
# No database writes
# ---------------------------------------------------------------------------

class NoDatabaseWritesTests(ReconciliationBaseData):
    def test_no_persons_created(self):
        before = Person.objects.count()
        reconcile_all()
        self.assertEqual(Person.objects.count(), before)

    def test_no_profiles_created(self):
        before = StudentProfile.objects.count()
        reconcile_all()
        self.assertEqual(StudentProfile.objects.count(), before)

    def test_no_student_mutations(self):
        before_h = self.student_ok.h_code
        before_first = self.student_ok.first_name
        reconcile_all()
        self.student_ok.refresh_from_db()
        self.assertEqual(self.student_ok.h_code, before_h)
        self.assertEqual(self.student_ok.first_name, before_first)

    def test_no_students_deleted(self):
        before = LegacyStudent.objects.count()
        reconcile_all()
        self.assertEqual(LegacyStudent.objects.count(), before)

    def test_reconcile_student_no_writes(self):
        before_persons = Person.objects.count()
        before_profiles = StudentProfile.objects.count()
        reconcile_student(self.student_ok)
        self.assertEqual(Person.objects.count(), before_persons)
        self.assertEqual(StudentProfile.objects.count(), before_profiles)
