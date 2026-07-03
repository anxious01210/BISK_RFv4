from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.test import TestCase

from .models import Person, PersonRole, RoleType, StaffProfile, StudentProfile
from .selectors import (
    active_roles_for_person,
    get_person_by_code,
    get_person_by_display_code,
    list_people,
    list_staff_profiles,
    list_student_profiles,
)
from .services import (
    assign_role,
    create_person,
    create_staff_profile,
    create_student_profile,
    deactivate_person,
    deactivate_role,
    get_or_create_person_by_code,
    update_person,
)
from .validators import (
    validate_person_code,
    validate_role_active_window,
    validate_student_code,
)

User = get_user_model()


class PersonServicesTests(TestCase):
    def test_create_person_normalizes_fields(self):
        person = create_person(
            code="P-001",
            first_name="  Alice  ",
            middle_name="",
            last_name="Wonderland",
            email="alice@example.com",
        )
        self.assertEqual(person.first_name, "Alice")
        self.assertEqual(person.code, "P-001")
        self.assertTrue(person.is_active)

    def test_create_person_rejects_duplicate_code(self):
        create_person(code="P-002", first_name="Bob", last_name="Builder")
        with self.assertRaises(ValidationError):
            create_person(code="P-002", first_name="Dup", last_name="Licate")

    def test_update_person_validates_unique_email(self):
        create_person(
            code="P-003", first_name="A", last_name="B", email="a@x.com"
        )
        target = create_person(
            code="P-004", first_name="C", last_name="D", email="c@x.com"
        )
        with self.assertRaises(ValidationError):
            update_person(target, email="a@x.com")

    def test_get_or_create_person_by_code(self):
        person, created = get_or_create_person_by_code(
            code="P-010", first_name="New", last_name="Guy"
        )
        self.assertTrue(created)
        again, created2 = get_or_create_person_by_code(
            code="P-010", first_name="New", last_name="Guy"
        )
        self.assertFalse(created2)
        self.assertEqual(person.pk, again.pk)


class ProfileServicesTests(TestCase):
    def setUp(self):
        self.person = create_person(
            code="P-100", first_name="Stu", last_name="Dent"
        )

    def test_create_student_profile(self):
        profile = create_student_profile(
            person=self.person, code="S-100", grade="G1", homeroom="H1"
        )
        self.assertEqual(profile.code, "S-100")
        self.assertEqual(self.person.student_profile, profile)

    def test_duplicate_student_code_rejected(self):
        create_student_profile(person=self.person, code="S-200")
        other = create_person(code="P-101", first_name="Other", last_name="One")
        with self.assertRaises(ValidationError):
            create_student_profile(person=other, code="S-200")

    def test_create_staff_profile(self):
        profile = create_staff_profile(
            person=self.person, code="T-100", job_title="Teacher", is_teacher=True
        )
        self.assertEqual(profile.code, "T-100")
        self.assertTrue(profile.is_teacher)


class RoleServicesTests(TestCase):
    def setUp(self):
        self.person = create_person(
            code="P-200", first_name="Role", last_name="Player"
        )
        self.role_type = RoleType.objects.create(
            code="student", name="Student"
        )

    def test_assign_role(self):
        role = assign_role(person=self.person, role_type=self.role_type)
        self.assertTrue(role.is_active)
        self.assertEqual(active_roles_for_person(self.person).count(), 1)

    def test_assign_duplicate_role_rejected(self):
        assign_role(person=self.person, role_type=self.role_type)
        with self.assertRaises(ValidationError):
            assign_role(person=self.person, role_type=self.role_type)

    def test_deactivate_role_sets_end_date(self):
        role = assign_role(person=self.person, role_type=self.role_type)
        deactivate_role(role)
        role.refresh_from_db()
        self.assertFalse(role.is_active)
        self.assertIsNotNone(role.end_date)

    def test_invalid_role_window_rejected(self):
        from datetime import date

        with self.assertRaises(ValidationError):
            assign_role(
                person=self.person,
                role_type=self.role_type,
                start_date=date(2024, 1, 10),
                end_date=date(2024, 1, 1),
            )

    def test_deactivate_person_cascades_to_roles(self):
        assign_role(person=self.person, role_type=self.role_type)
        deactivate_person(self.person)
        self.person.refresh_from_db()
        self.assertFalse(self.person.is_active)
        self.assertEqual(active_roles_for_person(self.person).count(), 0)


class SelectorsTests(TestCase):
    def setUp(self):
        self.person = create_person(
            code="P-300", first_name="Sel", last_name="Ector", email="sel@x.com"
        )
        create_student_profile(person=self.person, code="S-300", grade="G2")

    def test_get_person_by_code(self):
        self.assertEqual(get_person_by_code("P-300"), self.person)

    def test_get_person_by_display_code_resolves_student(self):
        self.assertEqual(get_person_by_display_code("S-300"), self.person)

    def test_get_person_by_display_code_falls_back_to_person_code(self):
        self.assertEqual(get_person_by_display_code("P-300"), self.person)

    def test_list_people_search(self):
        qs = list_people(search="Sel")
        self.assertIn(self.person, qs)

    def test_list_student_profiles_search(self):
        qs = list_student_profiles(search="S-300")
        self.assertEqual(qs.count(), 1)


class ValidatorsTests(TestCase):
    def test_validate_person_code_blank(self):
        with self.assertRaises(ValidationError):
            validate_person_code("")

    def test_validate_student_code_existing(self):
        person = create_person(code="P-400", first_name="V", last_name="X")
        create_student_profile(person=person, code="S-400")
        with self.assertRaises(ValidationError):
            validate_student_code("S-400")

    def test_validate_role_active_window_inactive_requires_end_date(self):
        with self.assertRaises(ValidationError):
            validate_role_active_window(is_active=False)
