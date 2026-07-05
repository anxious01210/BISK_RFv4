"""Tests for the plan_identity_migration management command.

Covers:
- default text output (summary + blocked + ready + already migrated)
- JSON output
- --only-blocked filtering
- --student H_CODE single-student lookup (text + JSON)
- unknown student handling (exit code 2)
- non-zero exit when blocked records exist
- zero exit when no blocked records exist
- no database writes
"""

import io
import json
from dataclasses import asdict

from django.core.management import call_command
from django.test import TestCase

from apps.attendance.models import Student as LegacyStudent
from apps.identity.models import Person, StudentProfile


class CommandBaseData(TestCase):
    """Create legacy Student rows for the command to plan against."""

    def setUp(self):
        self.student_ok = LegacyStudent.objects.create(
            h_code="H-CMD01",
            first_name="Alice",
            last_name="Wonderland",
            gender="MALE",
            grade="G5",
            has_meal=True,
        )
        self.student_ok2 = LegacyStudent.objects.create(
            h_code="H-CMD02",
            first_name="Bob",
            last_name="Builder",
            gender="FEMALE",
            grade="G3",
        )
        self.student_blocked = LegacyStudent.objects.create(
            h_code="H-CMD03",
            first_name="",  # missing first_name
            last_name="",   # missing last_name
        )

    def _call(self, *args, **kwargs) -> str:
        """Call the command via ``call_command`` and return stdout text.

        If the command raises ``SystemExit`` (non-zero exit code for
        blocked records), it is silently caught — use
        ``_call_direct`` for exit-code assertions.
        """
        out = io.StringIO()
        try:
            call_command(
                "plan_identity_migration", *args, stdout=out, **kwargs
            )
        except SystemExit:
            pass  # Expected for non-zero exit codes.
        return out.getvalue()

    def _call_direct(self, *args, **kwargs) -> tuple[str, int]:
        """Call the Command class directly to capture the exit code.

        ``handle()`` calls ``sys.exit(code)`` for non-zero exit codes
        (following Django's ``makemigrations --check`` pattern). This
        helper catches ``SystemExit`` and extracts ``.code``. If no
        ``SystemExit`` is raised, the exit code is 0.

        Returns ``(stdout_text, exit_code)``.
        """
        from apps.identity.management.commands.plan_identity_migration import Command

        out = io.StringIO()
        cmd = Command(stdout=out)
        try:
            cmd.handle(
                json=kwargs.get("json", False),
                only_blocked=kwargs.get("only_blocked", False),
                student=kwargs.get("student", None),
            )
        except SystemExit as exc:
            return out.getvalue(), exc.code
        return out.getvalue(), 0


# ---------------------------------------------------------------------------
# Default text output
# ---------------------------------------------------------------------------

class DefaultTextOutputTests(CommandBaseData):
    def test_shows_summary_counts(self):
        out = self._call()
        self.assertIn("Total students:", out)
        self.assertIn("Ready to migrate:", out)
        self.assertIn("Already migrated:", out)
        self.assertIn("Blocked:", out)

    def test_total_is_3(self):
        out = self._call()
        self.assertIn("3", out)

    def test_blocked_section_lists_issues(self):
        out = self._call()
        self.assertIn("H-CMD03", out)
        self.assertIn("Missing first_name", out)
        self.assertIn("Missing last_name", out)

    def test_ready_students_not_in_blocked_section(self):
        out = self._call()
        # H-CMD01 and H-CMD02 should not appear in the blocked section.
        # They are ready, so they won't be listed under "Blocked records".
        # The summary line "Ready to migrate: 2" should appear.
        self.assertIn("2", out)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

class JsonOutputTests(CommandBaseData):
    def test_json_is_valid(self):
        out = self._call(json=True)
        data = json.loads(out)
        self.assertIn("total", data)
        self.assertIn("ready", data)
        self.assertIn("blocked", data)
        self.assertIn("already_migrated", data)
        self.assertIn("plans", data)

    def test_json_counts(self):
        out = self._call(json=True)
        data = json.loads(out)
        self.assertEqual(data["total"], 3)
        self.assertEqual(data["ready"], 2)
        self.assertEqual(data["blocked"], 1)
        self.assertEqual(data["already_migrated"], 0)

    def test_json_plans_contain_h_codes(self):
        out = self._call(json=True)
        data = json.loads(out)
        h_codes = {p["h_code"] for p in data["plans"]}
        self.assertIn("H-CMD01", h_codes)
        self.assertIn("H-CMD02", h_codes)
        self.assertIn("H-CMD03", h_codes)

    def test_json_blocked_plan_has_issues(self):
        out = self._call(json=True)
        data = json.loads(out)
        blocked = [p for p in data["plans"] if p["issues"]]
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0]["h_code"], "H-CMD03")
        self.assertTrue(len(blocked[0]["issues"]) > 0)


# ---------------------------------------------------------------------------
# --only-blocked
# ---------------------------------------------------------------------------

class OnlyBlockedTests(CommandBaseData):
    def test_only_blocked_shows_blocked(self):
        out = self._call(only_blocked=True)
        self.assertIn("H-CMD03", out)
        self.assertIn("Missing first_name", out)

    def test_only_blocked_hides_ready(self):
        out = self._call(only_blocked=True)
        # H-CMD01 is ready; it should not appear in the blocked-only output.
        # The summary "Blocked records (1 of 3)" should appear.
        self.assertIn("1", out)
        # H-CMD01 should not be listed as blocked.
        lines = [l for l in out.splitlines() if "H-CMD01" in l]
        self.assertEqual(lines, [])

    def test_only_blocked_json(self):
        out = self._call(only_blocked=True, json=True)
        data = json.loads(out)
        self.assertEqual(data["blocked"], 1)
        self.assertEqual(len(data["blocked_plans"]), 1)
        self.assertEqual(data["blocked_plans"][0]["h_code"], "H-CMD03")


# ---------------------------------------------------------------------------
# --student H_CODE
# ---------------------------------------------------------------------------

class SingleStudentTests(CommandBaseData):
    def test_single_student_text(self):
        out = self._call(student="H-CMD01")
        self.assertIn("H-CMD01", out)
        self.assertIn("Alice", out)
        self.assertIn("Wonderland", out)
        self.assertIn("Ready to migrate", out)

    def test_single_student_json(self):
        out = self._call(student="H-CMD01", json=True)
        data = json.loads(out)
        self.assertEqual(data["h_code"], "H-CMD01")
        self.assertEqual(data["first_name"], "Alice")
        self.assertEqual(data["last_name"], "Wonderland")
        self.assertFalse(data["already_migrated"])
        self.assertEqual(data["issues"], [])

    def test_single_student_blocked(self):
        out = self._call(student="H-CMD03")
        self.assertIn("H-CMD03", out)
        self.assertIn("Missing first_name", out)
        self.assertIn("Missing last_name", out)

    def test_single_student_already_migrated(self):
        # Create a Person + StudentProfile linked to student_ok.
        person = Person.objects.create(
            code="H-CMD01", first_name="Alice", last_name="Wonderland"
        )
        StudentProfile.objects.create(
            person=person, code="H-CMD01", legacy_student=self.student_ok
        )
        out = self._call(student="H-CMD01")
        self.assertIn("Already migrated", out)


# ---------------------------------------------------------------------------
# Unknown student handling
# ---------------------------------------------------------------------------

class UnknownStudentTests(CommandBaseData):
    def test_unknown_student_text(self):
        out, ret = self._call_direct(student="H-UNKNOWN")
        self.assertIn("not found", out)
        self.assertEqual(ret, 2)

    def test_unknown_student_json(self):
        out, ret = self._call_direct(student="H-UNKNOWN", json=True)
        data = json.loads(out)
        self.assertEqual(data["error"], "student_not_found")
        self.assertEqual(data["h_code"], "H-UNKNOWN")
        self.assertEqual(ret, 2)


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

class ExitCodeTests(CommandBaseData):
    def test_nonzero_when_blocked_exist(self):
        _, ret = self._call_direct()
        self.assertEqual(ret, 1)

    def test_zero_when_no_blocked(self):
        # Fix the blocked student.
        self.student_blocked.first_name = "Fixed"
        self.student_blocked.last_name = "Name"
        self.student_blocked.save()
        _, ret = self._call_direct()
        self.assertEqual(ret, 0)

    def test_zero_when_all_migrated(self):
        # Migrate all 3 students.
        for s in [self.student_ok, self.student_ok2, self.student_blocked]:
            person = Person.objects.create(
                code=s.h_code, first_name=s.first_name or "X",
                last_name=s.last_name or "Y",
            )
            StudentProfile.objects.create(
                person=person, code=s.h_code, legacy_student=s
            )
        _, ret = self._call_direct()
        self.assertEqual(ret, 0)

    def test_only_blocked_nonzero_when_blocked_exist(self):
        _, ret = self._call_direct(only_blocked=True)
        self.assertEqual(ret, 1)

    def test_only_blocked_zero_when_no_blocked(self):
        self.student_blocked.first_name = "Fixed"
        self.student_blocked.last_name = "Name"
        self.student_blocked.save()
        _, ret = self._call_direct(only_blocked=True)
        self.assertEqual(ret, 0)

    def test_single_student_nonzero_when_blocked(self):
        _, ret = self._call_direct(student="H-CMD03")
        self.assertEqual(ret, 1)

    def test_single_student_zero_when_ready(self):
        _, ret = self._call_direct(student="H-CMD01")
        self.assertEqual(ret, 0)


# ---------------------------------------------------------------------------
# No database writes
# ---------------------------------------------------------------------------

class NoDatabaseWritesTests(CommandBaseData):
    def test_no_persons_created(self):
        before = Person.objects.count()
        self._call()
        self.assertEqual(Person.objects.count(), before)

    def test_no_student_profiles_created(self):
        before = StudentProfile.objects.count()
        self._call()
        self.assertEqual(StudentProfile.objects.count(), before)

    def test_no_legacy_students_modified(self):
        before_h = self.student_ok.h_code
        before_first = self.student_ok.first_name
        self._call()
        self.student_ok.refresh_from_db()
        self.assertEqual(self.student_ok.h_code, before_h)
        self.assertEqual(self.student_ok.first_name, before_first)

    def test_no_legacy_students_deleted(self):
        before = LegacyStudent.objects.count()
        self._call()
        self.assertEqual(LegacyStudent.objects.count(), before)

    def test_json_no_writes(self):
        before_persons = Person.objects.count()
        before_profiles = StudentProfile.objects.count()
        self._call(json=True)
        self.assertEqual(Person.objects.count(), before_persons)
        self.assertEqual(StudentProfile.objects.count(), before_profiles)
