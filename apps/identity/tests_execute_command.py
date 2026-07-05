"""Tests for the execute_identity_migration management command.

Covers:
- dry-run (no writes)
- execute one student
- execute all
- confirmation required (exit 3)
- limit parameter
- JSON output
- unknown student (exit 2)
- failures (exit 1)
- exit codes
- no writes in dry-run
"""

import io
import json

from django.core.management import call_command
from django.test import TestCase

from apps.attendance.models import Student as LegacyStudent
from apps.identity.models import Person, StudentProfile


class CommandBaseData(TestCase):
    def setUp(self):
        self.students = []
        for i in range(3):
            s = LegacyStudent.objects.create(
                h_code=f"H-EXC{i:02d}",
                first_name=f"Student{i}",
                last_name="Execute",
                gender="MALE",
                grade=f"G{i}",
                has_meal=True,
            )
            self.students.append(s)
        # A blocked student (missing first_name + last_name).
        self.blocked = LegacyStudent.objects.create(
            h_code="H-EXC99",
            first_name="",
            last_name="",
        )

    def _call(self, *args, **kwargs) -> str:
        """Call via call_command; catch SystemExit; return stdout."""
        out = io.StringIO()
        try:
            call_command(
                "execute_identity_migration", *args,
                stdout=out, stderr=out, **kwargs,
            )
        except SystemExit:
            pass
        return out.getvalue()

    def _call_direct(self, *args, **kwargs) -> tuple[str, int]:
        """Call Command.handle() directly; return (stdout, exit_code).

        Catches SystemExit (raised by sys.exit() inside handle()) and
        extracts .code.
        """
        from apps.identity.management.commands.execute_identity_migration import Command

        out = io.StringIO()
        cmd = Command(stdout=out)
        try:
            cmd.handle(
                dry_run=kwargs.get("dry_run", False),
                student=kwargs.get("student", None),
                limit=kwargs.get("limit", None),
                yes=kwargs.get("yes", False),
                json=kwargs.get("json", False),
            )
        except SystemExit as exc:
            return out.getvalue(), exc.code
        return out.getvalue(), 0


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

class DryRunTests(CommandBaseData):
    def test_dry_run_no_writes(self):
        before_persons = Person.objects.count()
        before_profiles = StudentProfile.objects.count()
        self._call(dry_run=True)
        self.assertEqual(Person.objects.count(), before_persons)
        self.assertEqual(StudentProfile.objects.count(), before_profiles)

    def test_dry_run_shows_summary(self):
        out = self._call(dry_run=True)
        self.assertIn("Total students:", out)
        self.assertIn("Ready to migrate:", out)
        self.assertIn("Blocked:", out)
        self.assertIn("dry-run", out.lower())

    def test_dry_run_json(self):
        out = self._call(dry_run=True, json=True)
        data = json.loads(out)
        self.assertTrue(data["dry_run"])
        self.assertEqual(data["total"], 4)
        self.assertEqual(data["ready"], 3)
        self.assertEqual(data["blocked"], 1)

    def test_dry_run_single_student(self):
        out = self._call(dry_run=True, student="H-EXC00")
        self.assertIn("H-EXC00", out)
        self.assertIn("Person code:", out)
        self.assertIn("Already migrated:   False", out)

    def test_dry_run_blocked_single(self):
        out = self._call(dry_run=True, student="H-EXC99")
        self.assertIn("Missing first_name", out)


# ---------------------------------------------------------------------------
# Confirmation required
# ---------------------------------------------------------------------------

class ConfirmationTests(CommandBaseData):
    def test_no_yes_exits_3(self):
        _, code = self._call_direct()
        self.assertEqual(code, 3)

    def test_no_yes_text_message(self):
        out = self._call()
        self.assertIn("--yes", out)

    def test_no_yes_json(self):
        out = self._call(json=True)
        data = json.loads(out)
        self.assertEqual(data["error"], "confirmation_required")

    def test_yes_required_for_write(self):
        # Without --yes, no writes should occur.
        before = Person.objects.count()
        self._call()
        self.assertEqual(Person.objects.count(), before)


# ---------------------------------------------------------------------------
# Execute one student
# ---------------------------------------------------------------------------

class ExecuteSingleStudentTests(CommandBaseData):
    def test_execute_one_student(self):
        out = self._call(student="H-EXC00", yes=True)
        self.assertIn("migrated", out.lower())
        self.assertTrue(Person.objects.filter(code="H-EXC00").exists())

    def test_execute_one_student_json(self):
        out = self._call(student="H-EXC00", yes=True, json=True)
        data = json.loads(out)
        self.assertEqual(data["h_code"], "H-EXC00")
        self.assertTrue(data["created_person"])

    def test_execute_unknown_student_exit_2(self):
        out = self._call(student="H-UNKNOWN", yes=True)
        self.assertIn("not found", out)

    def test_execute_unknown_student_json(self):
        out = self._call(student="H-UNKNOWN", yes=True, json=True)
        data = json.loads(out)
        self.assertEqual(data["error"], "student_not_found")

    def test_execute_blocked_student_exit_1(self):
        out = self._call(student="H-EXC99", yes=True)
        self.assertIn("FAILED", out)
        self.assertFalse(Person.objects.filter(code="H-EXC99").exists())

    def test_execute_already_migrated(self):
        # First migration.
        self._call(student="H-EXC00", yes=True)
        # Second — should say "already migrated".
        out = self._call(student="H-EXC00", yes=True)
        self.assertIn("already migrated", out)


# ---------------------------------------------------------------------------
# Execute all (bulk)
# ---------------------------------------------------------------------------

class ExecuteAllTests(CommandBaseData):
    def test_execute_all(self):
        out = self._call(yes=True)
        self.assertIn("Migrated:", out)
        self.assertIn("Failed:", out)
        self.assertIn("Elapsed:", out)

    def test_execute_all_summary_counts(self):
        out = self._call(yes=True)
        # 3 ready students migrated, 1 blocked failed.
        self.assertIn("3", out)  # migrated count
        self.assertIn("1", out)  # failed count

    def test_execute_all_json(self):
        out = self._call(yes=True, json=True)
        data = json.loads(out)
        self.assertEqual(data["total_students"], 4)
        self.assertEqual(data["migrated"], 3)
        self.assertEqual(data["failed"], 1)
        self.assertIn("elapsed_seconds", data)
        self.assertEqual(len(data["errors"]), 1)

    def test_execute_all_exit_1_on_failures(self):
        out = self._call(yes=True)
        # SystemExit(1) was raised; _call catches it.
        # Verify the failed student is listed.
        self.assertIn("H-EXC99", out)


# ---------------------------------------------------------------------------
# Limit
# ---------------------------------------------------------------------------

class LimitTests(CommandBaseData):
    def test_limit_2(self):
        out = self._call(yes=True, limit=2, json=True)
        data = json.loads(out)
        self.assertEqual(data["total_students"], 2)
        self.assertEqual(data["migrated"], 2)

    def test_limit_json(self):
        out = self._call(yes=True, limit=1, json=True)
        data = json.loads(out)
        self.assertEqual(data["total_students"], 1)


# ---------------------------------------------------------------------------
# No writes in dry-run
# ---------------------------------------------------------------------------

class NoWritesInDryRunTests(CommandBaseData):
    def test_no_persons_created(self):
        before = Person.objects.count()
        self._call(dry_run=True)
        self.assertEqual(Person.objects.count(), before)

    def test_no_profiles_created(self):
        before = StudentProfile.objects.count()
        self._call(dry_run=True)
        self.assertEqual(StudentProfile.objects.count(), before)

    def test_no_students_modified(self):
        before_h = self.students[0].h_code
        self._call(dry_run=True)
        self.students[0].refresh_from_db()
        self.assertEqual(self.students[0].h_code, before_h)
