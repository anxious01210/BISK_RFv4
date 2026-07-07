# apps/attendance/tests.py
"""M1 tests — FaceEmbedding person FK (attendance identity adoption).

Covers:
  * The new nullable ``person`` FK on ``FaceEmbedding``.
  * The ``uniq_active_embedding_per_person`` partial unique constraint.
  * Dual-write behavior in ``EnrollView`` (and resolver helper).
  * The 0033 backfill migration (idempotent, noop reverse, skips unmigrated).
  * Read-path regression (``GalleryView`` response shape unchanged).

Tests follow the ``TestCase`` pattern established by ``apps/identity/tests_*``,
using ``LegacyStudent`` + ``Person`` + ``StudentProfile`` fixtures.
"""
import base64

from django.test import TestCase, override_settings
from django.db import IntegrityError

from apps.attendance.models import FaceEmbedding, Student
from apps.identity.models import Person, RoleType, StudentProfile


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _vec_bytes(dim: int = 512) -> bytes:
    """Return deterministic non-zero bytes of length dim*4 for a FaceEmbedding
    vector field (avoids needing numpy in tests)."""
    return bytes((i % 256) for i in range(dim * 4))


class FaceEmbeddingPersonBase(TestCase):
    """Shared fixtures for FaceEmbedding person-FK tests."""

    @classmethod
    def setUpTestData(cls):
        # RoleType "student" is seeded by an identity data migration.
        cls.role_student, _ = RoleType.objects.get_or_create(
            code="student", defaults={"name": "Student", "is_system": True}
        )
        # Legacy attendance.Student (migrated to identity)
        cls.student = Student.objects.create(
            h_code="H-TEST01",
            first_name="Test",
            last_name="Student",
            is_active=True,
        )
        cls.person = Person.objects.create(
            code="H-TEST01",
            first_name="Test",
            last_name="Student",
        )
        cls.profile = StudentProfile.objects.create(
            person=cls.person,
            code="H-TEST01",
            legacy_student=cls.student,
        )

    def _make_embedding(self, *, student=None, person=None, is_active=True, dim=512):
        return FaceEmbedding.objects.create(
            student=student if student is not None else self.student,
            person=person,
            dim=dim,
            vector=_vec_bytes(dim),
            is_active=is_active,
        )


# ---------------------------------------------------------------------------
# 1. Schema correctness
# ---------------------------------------------------------------------------


class FaceEmbeddingPersonFKTests(FaceEmbeddingPersonBase):
    """Verify the person FK field and the unchanged student FK."""

    def test_person_field_exists_and_nullable(self):
        field = FaceEmbedding._meta.get_field("person")
        self.assertTrue(field.null)
        self.assertTrue(field.blank)
        self.assertEqual(field.remote_field.on_delete.__name__, "CASCADE")
        self.assertEqual(field.remote_field.related_name, "embeddings")
        self.assertEqual(field.remote_field.model.__name__, "Person")

    def test_person_field_related_name(self):
        emb = self._make_embedding(person=self.person)
        self.assertIn(emb, self.person.embeddings.all())

    def test_existing_embedding_defaults_person_null(self):
        emb = self._make_embedding(person=None)
        self.assertIsNone(emb.person_id)

    def test_create_embedding_with_person(self):
        emb = self._make_embedding(person=self.person)
        self.assertEqual(emb.person_id, self.person.pk)
        self.assertEqual(emb.student_id, self.student.pk)

    def test_student_fk_unchanged(self):
        field = FaceEmbedding._meta.get_field("student")
        self.assertEqual(field.remote_field.on_delete.__name__, "CASCADE")
        self.assertEqual(field.remote_field.related_name, "embeddings")

    def test_existing_constraint_unchanged(self):
        names = {c.name for c in FaceEmbedding._meta.constraints}
        self.assertIn("uniq_active_embedding_per_student", names)


# ---------------------------------------------------------------------------
# 2. uniq_active_embedding_per_person constraint
# ---------------------------------------------------------------------------


class FaceEmbeddingConstraintTests(FaceEmbeddingPersonBase):
    """Verify the new uniq_active_embedding_per_person partial constraint."""

    def test_constraint_exists(self):
        names = {c.name for c in FaceEmbedding._meta.constraints}
        self.assertIn("uniq_active_embedding_per_person", names)

    def test_two_active_same_person_raises(self):
        # A second active embedding for a different student but the SAME
        # person (the dual-FK edge case) must violate the constraint.
        other_student = Student.objects.create(
            h_code="H-TEST02", first_name="Other", last_name="Student"
        )
        self._make_embedding(student=self.student, person=self.person, is_active=True)
        with self.assertRaises(IntegrityError):
            self._make_embedding(
                student=other_student, person=self.person, is_active=True
            )

    def test_two_active_same_student_raises(self):
        # Legacy constraint still enforced.
        self._make_embedding(person=self.person, is_active=True)
        with self.assertRaises(IntegrityError):
            self._make_embedding(person=self.person, is_active=True)

    def test_inactive_same_person_allowed(self):
        self._make_embedding(person=self.person, is_active=True)
        # A second, INACTIVE embedding for the same person is allowed.
        second = self._make_embedding(person=self.person, is_active=False)
        self.assertFalse(second.is_active)

    def test_null_person_active_allowed(self):
        # Multiple active embeddings with person=None are allowed (NULL in
        # unique index is not counted).
        s2 = Student.objects.create(h_code="H-NULL2", first_name="N", last_name="N")
        self._make_embedding(person=None, is_active=True)
        emb2 = self._make_embedding(student=s2, person=None, is_active=True)
        self.assertIsNone(emb2.person_id)

    def test_different_persons_active_allowed(self):
        s2 = Student.objects.create(
            h_code="H-TEST02", first_name="Other", last_name="Student"
        )
        p2 = Person.objects.create(code="H-TEST02", first_name="Other", last_name="S")
        StudentProfile.objects.create(person=p2, code="H-TEST02", legacy_student=s2)
        self._make_embedding(person=self.person, is_active=True)
        emb2 = self._make_embedding(student=s2, person=p2, is_active=True)
        self.assertEqual(emb2.person_id, p2.pk)


# ---------------------------------------------------------------------------
# 3. _resolve_person_from_student helper
# ---------------------------------------------------------------------------


class ResolvePersonFromStudentTests(FaceEmbeddingPersonBase):
    """Verify the resolver helper used by the dual-write path."""

    def test_resolves_when_migrated(self):
        from apps.attendance.services import _resolve_person_from_student
        p = _resolve_person_from_student(self.student)
        self.assertEqual(p, self.person)

    def test_returns_none_when_not_migrated(self):
        from apps.attendance.services import _resolve_person_from_student
        unmigrated = Student.objects.create(
            h_code="H-NOMIG", first_name="No", last_name="Mig"
        )
        self.assertIsNone(_resolve_person_from_student(unmigrated))

    def test_returns_none_for_none_input(self):
        from apps.attendance.services import _resolve_person_from_student
        self.assertIsNone(_resolve_person_from_student(None))


# ---------------------------------------------------------------------------
# 4. EnrollView dual-write
# ---------------------------------------------------------------------------


class EnrollViewDualWriteTests(FaceEmbeddingPersonBase):
    """Verify the EnrollView POST dual-writes person when the student is
    migrated."""

    # Disable the X-BISK-KEY gate for these tests (no key configured in
    # test settings -> the view treats the header as optional).
    @override_settings(RUNNER_HEARTBEAT_KEY="")
    def _post(self, h_code, dim=512):
        url = "/api/attendance/enroll/"
        return self.client.post(
            url,
            data={
                "h_code": h_code,
                "dim": dim,
                "vec": base64.b64encode(_vec_bytes(dim)).decode("ascii"),
            },
            content_type="application/json",
        )

    @classmethod
    def setUpTestData(cls):
        super().setUpTestData()
        # An unmigrated student for the negative case.
        cls.unmigrated = Student.objects.create(
            h_code="H-NOMIG", first_name="No", last_name="Mig", is_active=True
        )

    def test_enroll_view_sets_person_when_migrated(self):
        resp = self._post("H-TEST01")
        self.assertEqual(resp.status_code, 201, resp.content)
        emb = FaceEmbedding.objects.get(student=self.student)
        self.assertEqual(emb.person_id, self.person.pk)
        self.assertEqual(emb.student_id, self.student.pk)

    def test_enroll_view_person_null_when_not_migrated(self):
        resp = self._post("H-NOMIG")
        self.assertEqual(resp.status_code, 201, resp.content)
        emb = FaceEmbedding.objects.get(student=self.unmigrated)
        self.assertIsNone(emb.person_id)
        self.assertEqual(emb.student_id, self.unmigrated.pk)

    def test_enroll_view_unknown_student_404(self):
        resp = self._post("H-DOESNOTEXIST")
        self.assertEqual(resp.status_code, 404)

    def test_enroll_view_response_shape_unchanged(self):
        resp = self._post("H-TEST01")
        self.assertIn("ok", resp.json())
        self.assertIn("id", resp.json())


# ---------------------------------------------------------------------------
# 5. Backfill migration 0033
# ---------------------------------------------------------------------------


class BackfillMigrationTests(FaceEmbeddingPersonBase):
    """Verify the 0033 backfill RunPython directly (idempotent, noop reverse,
    skips unmigrated students)."""

    def _run_backfill(self):
        from apps.attendance.migrations._0033_helper import forwards, backwards
        from django.apps import apps
        forwards(apps, None)

    def test_backfill_sets_person_id(self):
        # An embedding whose student is migrated but person_id is NULL.
        emb = self._make_embedding(person=None)
        self.assertIsNone(emb.person_id)
        self._run_backfill()
        emb.refresh_from_db()
        self.assertEqual(emb.person_id, self.person.pk)

    def test_backfill_skips_already_backfilled(self):
        emb = self._make_embedding(person=self.person)
        original_person_id = emb.person_id
        self._run_backfill()
        emb.refresh_from_db()
        self.assertEqual(emb.person_id, original_person_id)

    def test_backfill_skips_unmigrated_students(self):
        unmigrated = Student.objects.create(
            h_code="H-NOMIG", first_name="No", last_name="Mig", is_active=True
        )
        emb = self._make_embedding(student=unmigrated, person=None)
        self._run_backfill()
        emb.refresh_from_db()
        self.assertIsNone(emb.person_id)

    def test_backfill_skips_null_student(self):
        # Edge case: embedding with student=None cannot exist because student
        # FK is non-null, but the backfill code defends against it via
        # .exclude(student__isnull=True). Verify no crash by calling directly
        # with an empty queryset path.
        self._run_backfill()  # should not raise even with no rows to backfill

    def test_backfill_reverse_is_noop(self):
        from apps.attendance.migrations._0033_helper import backwards
        from django.apps import apps
        emb = self._make_embedding(person=self.person)
        backwards(apps, None)  # must not raise or change data
        emb.refresh_from_db()
        self.assertEqual(emb.person_id, self.person.pk)


# ---------------------------------------------------------------------------
# 6. Read-path regression (GalleryView)
# ---------------------------------------------------------------------------


class GalleryViewRegressionTests(FaceEmbeddingPersonBase):
    """Confirm the additive person FK does not break the GalleryView response."""

    @override_settings(RUNNER_HEARTBEAT_KEY="")
    def test_gallery_view_returns_h_code(self):
        self._make_embedding(person=self.person, dim=512, is_active=True)
        url = "/api/attendance/gallery/?dim=512&active=1"
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200, resp.content)
        body = resp.json()
        self.assertEqual(body["dim"], 512)
        self.assertEqual(body["count"], 1)
        item = body["embeddings"][0]
        self.assertEqual(item["h_code"], "H-TEST01")
        # person_id must NOT leak into the response (backward compat).
        self.assertNotIn("person_id", item)
        self.assertNotIn("person", item)

    @override_settings(RUNNER_HEARTBEAT_KEY="")
    def test_gallery_view_filter_by_dim(self):
        self._make_embedding(person=self.person, dim=512, is_active=True)
        # Second embedding with a different dim must be inactive to avoid
        # violating uniq_active_embedding_per_student / per_person.
        self._make_embedding(person=self.person, dim=128, is_active=False)
        url = "/api/attendance/gallery/?dim=512"
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["count"], 1)
