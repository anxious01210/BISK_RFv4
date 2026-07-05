"""Tests for the reconcile_identity_migration management command.

Covers:
- default output (human-readable)
- JSON output
- --only-issues
- exit 0 when clean
- exit 1 when issues exist
- no database writes
- duplicate/orphan information included
"""

import io
import json

from django.core.management import call_command
from django.test import TestCase

from apps.attendance.models import Student as LegacyStudent
from apps.identity.models import Person, StudentProfile
from apps.identity.services import (
    create_person,
    create_student_profile,
    get_or_create_role,
)
from apps.identity.models import RoleType


class ReconcileCommandBaseData(TestCase):
    def setUp(self):
        self.student_ok = LegacyStudent.objects.create(
            h_code="H-RCMD01",
            first_name="Alice",
            last_name="Wonderland",
            gender="MALE",
            grade="G5",
            has_meal=True,
        )
        self.student_unmigrated = LegacyStudent.objects.create(
            h_code="H-RCMD02",
            first_name="Bob",
            last_name="Builder",
        )

    def _migrate_student(self, student):
        person = create_person(
            code=student.h_code,
            first_name=student.first_name,
            last_name=student.last_name,
            gender=student.gender or "",
        )
        create_student_profile(
            person=person,
            code=student.h_code,
            grade=student.grade or "",
            has_meal=student.has_meal,
            legacy_student=student,
        )
        role_type = RoleType.objects.get(code="student")
        get_or_create_role(person=person, role_type=role_type)

    def _call(self, *args, **kwargs) -> str:
        out = io.StringIO()
        try:
            call_command(
                "reconcile_identity_migration", *args,
                stdout=out, stderr=out, **kwargs,
            )
        except SystemExit:
            pass
        return out.getvalue()


# ---------------------------------------------------------------------------
# Default output
# ---------------------------------------------------------------------------

class DefaultOutputTests(ReconcileCommandBaseData):
    def test_shows_summary(self):
        out = self._call()
        self.assertIn("Total students:", out)
        self.assertIn("Migrated:", out)
        self.assertIn("Unmigrated:", out)
        self.assertIn("Students with issues:", out)

    def test_shows_unmigrated_student(self):
        out = self._call()
        self.assertIn("H-RCMD02", out)
        self.assertIn("unmigrated", out)

    def test_shows_duplicate_info(self):
        out = self._call()
        self.assertIn("Duplicate person codes:", out)
        self.assertIn("Duplicate profile codes:", out)

    def test_shows_orphan_info(self):
        out = self._call()
        self.assertIn("Orphan student profiles:", out)
        self.assertIn("Orphan persons:", out)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

class JsonOutputTests(ReconcileCommandBaseData):
    def test_json_valid(self):
        out = self._call(json=True)
        data = json.loads(out)
        self.assertIn("total_students", data)
        self.assertIn("migrated", data)
        self.assertIn("unmigrated", data)
        self.assertIn("issues_found", data)
        self.assertIn("rows", data)

    def test_json_counts(self):
        out = self._call(json=True)
        data = json.loads(out)
        self.assertEqual(data["total_students"], 2)
        self.assertEqual(data["migrated"], 0)
        self.assertEqual(data["unmigrated"], 2)
        self.assertEqual(data["issues_found"], 2)

    def test_json_rows_contain_issues(self):
        out = self._call(json=True)
        data = json.loads(out)
        h_codes = {r["h_code"] for r in data["rows"]}
        self.assertIn("H-RCMD01", h_codes)
        self.assertIn("H-RCMD02", h_codes)

    def test_json_includes_cross_cutting(self):
        out = self._call(json=True)
        data = json.loads(out)
        self.assertIn("duplicate_person_codes", data)
        self.assertIn("duplicate_student_profile_codes", data)
        self.assertIn("orphan_student_profiles", data)
        self.assertIn("orphan_persons", data)


# ---------------------------------------------------------------------------
# --only-issues
# ---------------------------------------------------------------------------

class OnlyIssuesTests(ReconcileCommandBaseData):
    def test_only_issues_shows_issues(self):
        self._migrate_student(self.student_ok)
        out = self._call(only_issues=True)
        # student_ok has no issues; student_unmigrated has issues.
        self.assertIn("H-RCMD02", out)
        self.assertIn("unmigrated", out)

    def test_only_issues_hides_clean(self):
        self._migrate_student(self.student_ok)
        out = self._call(only_issues=True)
        # student_ok should not appear (it's clean).
        lines = [l for l in out.splitlines() if "H-RCMD01" in l]
        self.assertEqual(lines, [])

    def test_only_issues_json(self):
        self._migrate_student(self.student_ok)
        out = self._call(only_issues=True, json=True)
        data = json.loads(out)
        h_codes = {r["h_code"] for r in data["rows"]}
        self.assertIn("H-RCMD02", h_codes)
        self.assertNotIn("H-RCMD01", h_codes)


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

class ExitCodeTests(ReconcileCommandBaseData):
    def test_exit_1_when_issues_exist(self):
        out = io.StringIO()
        with self.assertRaises(SystemExit) as ctx:
            call_command(
                "reconcile_identity_migration",
                stdout=out, stderr=out,
            )
        self.assertEqual(ctx.exception.code, 1)

    def test_exit_0_when_clean(self):
        self._migrate_student(self.student_ok)
        self._migrate_student(self.student_unmigrated)
        out = io.StringIO()
        call_command(
            "reconcile_identity_migration",
            stdout=out, stderr=out,
        )
        # No SystemExit raised → exit 0.

    def test_exit_0_only_issues_when_clean(self):
        self._migrate_student(self.student_ok)
        self._migrate_student(self.student_unmigrated)
        out = io.StringIO()
        call_command(
            "reconcile_identity_migration",
            only_issues=True,
            stdout=out, stderr=out,
        )


# ---------------------------------------------------------------------------
# No database writes
# ---------------------------------------------------------------------------

class NoDatabaseWritesTests(ReconcileCommandBaseData):
    def test_no_persons_created(self):
        before = Person.objects.count()
        self._call()
        self.assertEqual(Person.objects.count(), before)

    def test_no_profiles_created(self):
        before = StudentProfile.objects.count()
        self._call()
        self.assertEqual(StudentProfile.objects.count(), before)

    def test_no_student_mutations(self):
        before_h = self.student_ok.h_code
        before_first = self.student_ok.first_name
        self._call()
        self.student_ok.refresh_from_db()
        self.assertEqual(self.student_ok.h_code, before_h)
        self.assertEqual(self.student_ok.first_name, before_first)

    def test_no_students_deleted(self):
        before = LegacyStudent.objects.count()
        self._call()
        self.assertEqual(LegacyStudent.objects.count(), before)
