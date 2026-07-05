"""Tests for the scheduler system dashboard.

Phase 5A — Scheduler adopts Identity.

These tests verify that ``system_panel_partial`` reports the active
student count from the new Identity model (``StudentProfile`` linked to
an active ``Person``) rather than from the legacy ``attendance.Student``
table. Dashboard behavior is otherwise preserved.
"""

from django.test import TestCase

from apps.attendance.models import Student as LegacyStudent
from apps.identity.models import Person, StudentProfile


class SystemPanelTotalStudentsTests(TestCase):
    """Verify ``total_students`` in the dashboard context reflects the
    active student count under the Identity model."""

    url = "/dash/system/panel/"

    def _make_person(self, code: str, *, is_active: bool = True) -> Person:
        return Person.objects.create(
            code=code,
            first_name=code,
            last_name="Test",
            is_active=is_active,
        )

    def _make_student_profile(
        self, person: Person, code: str
    ) -> StudentProfile:
        return StudentProfile.objects.create(person=person, code=code)

    def test_total_students_is_zero_when_no_identity_data(self):
        """Empty Identity tables → dashboard shows 0 students.

        Critically, the dashboard must NOT fall back to the legacy
        ``attendance.Student`` table — even when that table has rows —
        because Phase 5A removes that production dependency.
        """
        # A legacy student exists but must be ignored by the dashboard.
        LegacyStudent.objects.create(
            h_code="H-LEG01", first_name="Legacy", last_name="Student"
        )

        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["total_students"], 0)

    def test_total_students_counts_active_student_profiles(self):
        """Each active Person with a StudentProfile contributes one to
        the displayed count."""
        p1 = self._make_person("P-ACT1")
        p2 = self._make_person("P-ACT2")
        self._make_student_profile(p1, "S-ACT1")
        self._make_student_profile(p2, "S-ACT2")

        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["total_students"], 2)

    def test_inactive_person_excluded_from_total_students(self):
        """An inactive Person's StudentProfile must not be counted."""
        active = self._make_person("P-ACT3", is_active=True)
        inactive = self._make_person("P-INA3", is_active=False)
        self._make_student_profile(active, "S-ACT3")
        self._make_student_profile(inactive, "S-INA3")

        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["total_students"], 1)

    def test_total_students_ignores_legacy_student_table(self):
        """The dashboard count is derived solely from Identity data;
        legacy ``attendance.Student`` rows do not influence it."""
        # Three legacy rows — must be ignored.
        for i in range(3):
            LegacyStudent.objects.create(
                h_code=f"H-LEG{i:02d}",
                first_name="Legacy",
                last_name=f"Student{i}",
            )

        # A single active Identity student.
        person = self._make_person("P-IDX")
        self._make_student_profile(person, "S-IDX")

        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["total_students"], 1)
