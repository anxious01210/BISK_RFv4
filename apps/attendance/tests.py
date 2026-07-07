# apps/attendance/tests.py
"""Attendance identity adoption tests.

M1 — FaceEmbedding person FK:

  * The new nullable ``person`` FK on ``FaceEmbedding``.
  * The ``uniq_active_embedding_per_person`` partial unique constraint.
  * Dual-write behavior in ``EnrollView`` (and resolver helper).
  * The 0033 backfill migration (idempotent, noop reverse, skips unmigrated).
  * Read-path regression (``GalleryView`` response shape unchanged).

M2 — AttendanceEvent person FK:

  * The new nullable ``person`` FK on ``AttendanceEvent``.
  * No new unique constraint on ``AttendanceEvent`` (append-only audit log).
  * Resolver helper reuse (read-only).
  * Dual-write in ``_write_from_match`` across all four
    ``AttendanceEvent.objects.create`` branches (below_min_score, no_period,
    winners loop, max_periods_reached fallback), via the public entrypoints
    ``ingest_match`` and ``record_recognition``.
  * The 0035 backfill migration (idempotent, batched, noop reverse,
    skips unmigrated students).
  * Read-path regression: ``latest_event_qs`` subqueries still key on
    ``student_id``; admin list_display still references ``student__*`` only;
    camera health query unaffected.
  * Meals bridge fallback regression: the bridge resolves Person via the
    ``attendance_event.student → StudentProfile.legacy_student → Person``
    backlink when ``attendance_event.person`` is NULL or set (M2 keeps the
    bridge on the fallback chain; switching it to prefer the event column is
    design step S11).
  * ``IngestView`` wire contract regression (no ``person_id`` in response).

M3 — AttendanceRecord person FK:

  * The new nullable ``person`` FK on ``AttendanceRecord``.
  * Coexisting ``(person, period)`` unique_together + index alongside the
    legacy ``(student, period)`` constraints.
  * Dual-write via ``get_or_create`` defaults (NOT upsert key switch — the key
    remains ``(student, period)`` until design step S9).
  * Defensive ``person_id`` set on found records (edge case: student migrated
    after backfill).
  * ``save(update_fields=...)`` in the re-register window path includes
    ``person_id`` when defensively set.
  * The 0037 backfill migration (idempotent, batched, noop reverse).
  * Upsert invariant regression (no duplicate records created).
  * Read-path regression: admin/serializer unchanged.
  * Meals bridge fallback regression.

Tests follow the ``TestCase`` pattern established by ``apps/identity/tests_*``,
using ``LegacyStudent`` + ``Person`` + ``StudentProfile`` fixtures.
"""
import base64
from datetime import timedelta

from django.test import TestCase, override_settings
from django.db import IntegrityError
from django.utils import timezone

from apps.attendance.models import (
    AttendanceEvent,
    AttendanceRecord,
    FaceEmbedding,
    PeriodOccurrence,
    PeriodTemplate,
    RecognitionSettings,
    Student,
)
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


# ===========================================================================
# M2 — AttendanceEvent person FK
# ===========================================================================


# ---------------------------------------------------------------------------
# M2 Shared fixtures
# ---------------------------------------------------------------------------


class AttendanceEventPersonBase(TestCase):
    """Shared fixtures for AttendanceEvent person-FK tests (M2)."""

    @classmethod
    def setUpTestData(cls):
        # RoleType "student" is seeded by an identity data migration.
        cls.role_student, _ = RoleType.objects.get_or_create(
            code="student", defaults={"name": "Student", "is_system": True}
        )
        # Migrated legacy Student
        cls.student = Student.objects.create(
            h_code="H-EVT01", first_name="Eve", last_name="Ent",
            is_active=True,
        )
        cls.person = Person.objects.create(
            code="H-EVT01", first_name="Eve", last_name="Ent",
        )
        cls.profile = StudentProfile.objects.create(
            person=cls.person, code="H-EVT01", legacy_student=cls.student,
        )
        # Unmigrated legacy Student (no StudentProfile link)
        cls.unmigrated = Student.objects.create(
            h_code="H-EVT02", first_name="Un", last_name="Mig",
            is_active=True,
        )
        # Minimal period template + occurrence that is "always open" so tests
        # can deterministically trigger the winners-loop branch of
        # ``_write_from_match``.
        cls.template = PeriodTemplate.objects.create(
            name="Test Block", order=1,
            start_time=timezone.datetime.min.time(),
            end_time=timezone.datetime.max.time(),
            weekdays_mask=127,  # every day
            is_enabled=True,
        )
        cls.now = timezone.localtime()
        cls.occ = PeriodOccurrence.objects.create(
            template=cls.template, date=cls.now.date(),
            start_dt=cls.now - timedelta(hours=1),
            end_dt=cls.now + timedelta(hours=1),
            is_school_day=True,
        )

    def _make_event(self, *, student=None, person=None, period=None,
                    camera=None, ts=None, score=0.9, crop_path=""):
        """Create an AttendanceEvent directly (bypasses the service layer,
        mirroring the ``apps.meals.tests_integrations_attendance`` helper)."""
        return AttendanceEvent.objects.create(
            student=student if student is not None else self.student,
            person=person,
            period=period,
            camera=camera,
            ts=ts or self.now,
            score=score,
            crop_path=crop_path,
        )

    def _reset_settings(self, **overrides):
        """Reset the RecognitionSettings singleton to deterministic defaults
        for tests that mutate it. Must be called from ``setUp`` (not
        ``setUpTestData``) because the singleton is shared across tests."""
        rs, _ = RecognitionSettings.objects.get_or_create(pk=1)
        rs.min_score = 0.75
        rs.re_register_window_sec = 10
        rs.min_improve_delta = 0.01
        rs.max_periods_per_day = None
        rs.save()
        return rs


# ---------------------------------------------------------------------------
# M2 Section 7 — Schema correctness
# ---------------------------------------------------------------------------


class AttendanceEventPersonFKTests(AttendanceEventPersonBase):
    """Verify the person FK field and the unchanged student FK."""

    def test_person_field_exists_and_nullable(self):
        field = AttendanceEvent._meta.get_field("person")
        self.assertTrue(field.null)
        self.assertTrue(field.blank)
        self.assertEqual(field.remote_field.on_delete.__name__, "CASCADE")
        self.assertEqual(field.remote_field.related_name, "attendance_events")
        self.assertEqual(field.remote_field.model.__name__, "Person")

    def test_person_field_db_index(self):
        field = AttendanceEvent._meta.get_field("person")
        self.assertTrue(field.db_index)

    def test_person_field_related_name(self):
        ev = self._make_event(person=self.person)
        self.assertIn(ev, self.person.attendance_events.all())

    def test_existing_event_defaults_person_null(self):
        ev = self._make_event(person=None)
        self.assertIsNone(ev.person_id)

    def test_create_event_with_person(self):
        ev = self._make_event(person=self.person)
        self.assertEqual(ev.person_id, self.person.pk)
        self.assertEqual(ev.student_id, self.student.pk)

    def test_student_fk_unchanged(self):
        field = AttendanceEvent._meta.get_field("student")
        self.assertEqual(field.remote_field.on_delete.__name__, "CASCADE")
        # Legacy student FK has no related_name override (Django default).
        self.assertFalse(field.remote_field.related_name)

    def test_no_new_constraint_on_event(self):
        # M2 adds NO constraint (events are append-only audit log; no upsert
        # invariant, no is_active flag). Mirror M1's uniq_active_embedding
        # _per_person is NOT done here.
        names = {c.name for c in AttendanceEvent._meta.constraints}
        self.assertEqual(names, set())

    def test_no_unique_together_on_event(self):
        # AttendanceEvent has no unique_together (neither on student nor person).
        self.assertEqual(tuple(AttendanceEvent._meta.unique_together), ())


# ---------------------------------------------------------------------------
# M2 Section 8 — Resolver reuse
# ---------------------------------------------------------------------------


class ResolvePersonFromStudentReuseTests(AttendanceEventPersonBase):
    """Verify the M1 resolver helper still works for event-side students.
    Guards against accidental refactors of the helper breaking M2."""

    def test_resolver_returns_person_for_migrated_student(self):
        from apps.attendance.services import _resolve_person_from_student
        p = _resolve_person_from_student(self.student)
        self.assertEqual(p, self.person)

    def test_resolver_returns_none_for_unmigrated_student(self):
        from apps.attendance.services import _resolve_person_from_student
        self.assertIsNone(_resolve_person_from_student(self.unmigrated))

    def test_resolver_returns_none_for_none_input(self):
        from apps.attendance.services import _resolve_person_from_student
        self.assertIsNone(_resolve_person_from_student(None))


# ---------------------------------------------------------------------------
# M2 Section 9 — _write_from_match dual-write
# ---------------------------------------------------------------------------


@override_settings(ATTENDANCE_DEDUP_SECONDS=0, RUNNER_HEARTBEAT_KEY="")
class WriteFromMatchDualWriteTests(AttendanceEventPersonBase):
    """Verify the dual-write path in ``_write_from_match`` across all four
    ``AttendanceEvent.objects.create`` branches. Uses the public entrypoints
    ``ingest_match`` (resolves Student by h_code) and ``record_recognition``
    (accepts a Student instance) to exercise the full path."""

    def setUp(self):
        super().setUp()
        # Reset the singleton to deterministic defaults for every test.
        self._reset_settings()

    # -- winners loop (line 170) via ingest_match ----------------------------

    def test_ingest_match_dual_writes_person_when_migrated(self):
        from apps.attendance.services import ingest_match
        res = ingest_match(
            h_code="H-EVT01", score=0.95, camera=None, ts=self.now, crop_path="captures/test.jpg",
        )
        self.assertTrue(res.get("ok"), res)
        ev = AttendanceEvent.objects.get(student=self.student)
        self.assertEqual(ev.student_id, self.student.pk)
        self.assertEqual(ev.person_id, self.person.pk)

    def test_ingest_match_person_null_when_unmigrated(self):
        from apps.attendance.services import ingest_match
        res = ingest_match(
            h_code="H-EVT02", score=0.95, camera=None, ts=self.now, crop_path="captures/test.jpg",
        )
        self.assertTrue(res.get("ok"), res)
        ev = AttendanceEvent.objects.get(student=self.unmigrated)
        self.assertEqual(ev.student_id, self.unmigrated.pk)
        self.assertIsNone(ev.person_id)

    def test_ingest_match_unknown_h_code_returns_error(self):
        from apps.attendance.services import ingest_match
        res = ingest_match(
            h_code="H-NOPE", score=0.95, camera=None, ts=self.now, crop_path="captures/test.jpg",
        )
        self.assertFalse(res.get("ok"))
        self.assertEqual(res.get("error"), "student_not_found")
        self.assertFalse(AttendanceEvent.objects.filter(student__h_code="H-NOPE").exists())

    # -- below_min_score (line 131) ------------------------------------------

    def test_below_min_score_branch_dual_writes_person(self):
        from apps.attendance.services import ingest_match
        rs = RecognitionSettings.get_solo()
        rs.min_score = 0.99  # score 0.50 will be below threshold
        rs.save()
        res = ingest_match(
            h_code="H-EVT01", score=0.50, camera=None, ts=self.now, crop_path="captures/test.jpg",
        )
        self.assertFalse(res.get("accepted"))
        self.assertEqual(res.get("reason"), "below_min_score")
        ev = AttendanceEvent.objects.get(student=self.student)
        # Audit event still dual-writes person (the M2 invariant).
        self.assertEqual(ev.person_id, self.person.pk)
        self.assertIsNone(ev.period_id)  # no record created

    # -- no_period (line 142) ------------------------------------------------

    def test_no_period_branch_dual_writes_person(self):
        from apps.attendance.services import ingest_match
        # Delete the only occurrence so no period window is open at self.now.
        PeriodOccurrence.objects.all().delete()
        res = ingest_match(
            h_code="H-EVT01", score=0.95, camera=None, ts=self.now, crop_path="captures/test.jpg",
        )
        self.assertFalse(res.get("accepted"))
        self.assertEqual(res.get("reason"), "no_period")
        ev = AttendanceEvent.objects.get(student=self.student)
        self.assertEqual(ev.person_id, self.person.pk)
        self.assertIsNone(ev.period_id)

    # -- max_periods_reached fallback (line 250) -----------------------------

    def test_max_periods_reached_branch_dual_writes_person(self):
        from apps.attendance.services import ingest_match
        rs = RecognitionSettings.get_solo()
        rs.max_periods_per_day = 1  # cap to one distinct period/day
        rs.save()

        # First call: creates a record on self.occ.
        res1 = ingest_match(
            h_code="H-EVT01", score=0.95, camera=None, ts=self.now, crop_path="captures/test.jpg",
        )
        self.assertTrue(res1.get("accepted"))

        # Create a second distinct occurrence for the same day. The cap
        # forces _write_from_match into the max_periods_reached fallback.
        # PeriodOccurrence has unique_together(template, date) so we need a
        # second template.
        template2 = PeriodTemplate.objects.create(
            name="Test Block 2", order=2,
            start_time=timezone.datetime.min.time(),
            end_time=timezone.datetime.max.time(),
            weekdays_mask=127,
            is_enabled=True,
        )
        occ2 = PeriodOccurrence.objects.create(
            template=template2, date=self.now.date(),
            start_dt=self.now + timedelta(hours=2),
            end_dt=self.now + timedelta(hours=3),
            is_school_day=True,
        )
        res2 = ingest_match(
            h_code="H-EVT01", score=0.90, camera=None, ts=occ2.start_dt + timedelta(minutes=5),
            crop_path="captures/test.jpg",
        )
        self.assertFalse(res2.get("accepted"))
        self.assertEqual(res2.get("reason"), "max_periods_reached")

        # The fallback event must also have person set.
        fallback_events = AttendanceEvent.objects.filter(
            student=self.student, period__isnull=True
        )
        self.assertGreaterEqual(fallback_events.count(), 1)
        self.assertEqual(fallback_events.first().person_id, self.person.pk)

    # -- record_recognition path (Student instance, not h_code) --------------

    def test_record_recognition_dual_writes_person(self):
        from apps.attendance.services import record_recognition
        ev, rec = record_recognition(
            student=self.student, score=0.95, camera=None, ts=self.now, crop_path="captures/test.jpg",
        )
        self.assertIsNotNone(ev)
        self.assertEqual(ev.student_id, self.student.pk)
        self.assertEqual(ev.person_id, self.person.pk)

    def test_record_recognition_person_null_unmigrated(self):
        from apps.attendance.services import record_recognition
        ev, rec = record_recognition(
            student=self.unmigrated, score=0.95, camera=None, ts=self.now, crop_path="captures/test.jpg",
        )
        self.assertIsNotNone(ev)
        self.assertEqual(ev.student_id, self.unmigrated.pk)
        self.assertIsNone(ev.person_id)

    # -- multi-period tie: each event must have person set ------------------

    @override_settings(MULTI_PERIOD_ON_TIES=True)
    def test_multi_period_tie_each_event_has_person(self):
        from apps.attendance.services import ingest_match
        # Create a second winning occurrence for the same date and same
        # lowest order. MULTI_PERIOD_ON_TIES=True writes an event per winner.
        template2 = PeriodTemplate.objects.create(
            name="Test Block 2", order=1,  # same lowest order
            start_time=timezone.datetime.min.time(),
            end_time=timezone.datetime.max.time(),
            weekdays_mask=127,
            is_enabled=True,
        )
        occ2 = PeriodOccurrence.objects.create(
            template=template2, date=self.now.date(),
            start_dt=self.now - timedelta(hours=1),
            end_dt=self.now + timedelta(hours=1),
            is_school_day=True,
        )
        res = ingest_match(
            h_code="H-EVT01", score=0.95, camera=None, ts=self.now, crop_path="captures/test.jpg",
        )
        self.assertTrue(res.get("accepted"))
        self.assertEqual(res.get("winners"), 2)
        events = list(AttendanceEvent.objects.filter(student=self.student))
        # Should have written one event per winner (2 events here).
        self.assertGreaterEqual(len(events), 2)
        for ev in events:
            self.assertEqual(ev.person_id, self.person.pk)


# ---------------------------------------------------------------------------
# M2 Section 10 — Backfill migration 0035
# ---------------------------------------------------------------------------


class EventBackfillMigrationTests(AttendanceEventPersonBase):
    """Verify the 0035 backfill RunPython directly (idempotent, noop reverse,
    skips unmigrated students, batches correctly)."""

    def _run_backfill(self):
        from apps.attendance.migrations._0035_helper import forwards, backwards
        from django.apps import apps
        forwards(apps, None)

    def _run_backfill_backwards(self):
        from apps.attendance.migrations._0035_helper import backwards
        from django.apps import apps
        backwards(apps, None)

    def test_backfill_sets_person_id(self):
        ev = self._make_event(person=None)
        self.assertIsNone(ev.person_id)
        self._run_backfill()
        ev.refresh_from_db()
        self.assertEqual(ev.person_id, self.person.pk)

    def test_backfill_skips_already_backfilled(self):
        ev = self._make_event(person=self.person)
        original_person_id = ev.person_id
        self._run_backfill()
        ev.refresh_from_db()
        self.assertEqual(ev.person_id, original_person_id)

    def test_backfill_skips_unmigrated_students(self):
        ev = self._make_event(student=self.unmigrated, person=None)
        self._run_backfill()
        ev.refresh_from_db()
        self.assertIsNone(ev.person_id)

    def test_backfill_skips_null_student(self):
        # Edge case: events with student=None cannot exist (student FK is
        # non-null), but the backfill defends via .exclude(student__isnull=True).
        # Verify no crash when there are no rows needing backfill.
        self._run_backfill()  # should not raise

    def test_backfill_reverse_is_noop(self):
        ev = self._make_event(person=self.person)
        self._run_backfill_backwards()  # must not raise or change data
        ev.refresh_from_db()
        self.assertEqual(ev.person_id, self.person.pk)

    def test_backfill_idempotent(self):
        ev = self._make_event(person=None)
        self._run_backfill()
        ev.refresh_from_db()
        first = ev.person_id
        # Second run must not change anything (filter person__isnull=True).
        self._run_backfill()
        ev.refresh_from_db()
        self.assertEqual(ev.person_id, first)

    def test_backfill_batches_large_event_set(self):
        # Create 12 events with mixed migrated/unmigrated students to
        # exercise the batch flush logic (BATCH_SIZE=500 internally, so
        # this fits in one batch — but the test guards the grouping path).
        for _ in range(7):
            self._make_event(student=self.student, person=None)
        for _ in range(5):
            self._make_event(student=self.unmigrated, person=None)
        before = AttendanceEvent.objects.filter(student=self.student).count()
        self.assertEqual(before, 7)
        self._run_backfill()
        migrated_count = (
            AttendanceEvent.objects.filter(student=self.student)
            .exclude(person__isnull=True).count()
        )
        self.assertEqual(migrated_count, 7)
        unmigrated_count = (
            AttendanceEvent.objects.filter(student=self.unmigrated, person__isnull=True).count()
        )
        self.assertEqual(unmigrated_count, 5)


# ---------------------------------------------------------------------------
# M2 Section 11 — Read-path regression
# ---------------------------------------------------------------------------


class ReadPathRegressionTests(AttendanceEventPersonBase):
    """Verify legacy read paths still work after M2 (student FK intact)."""

    def test_latest_event_subquery_still_filters_on_student_id(self):
        # Mirror the subquery used in apps.attendance.views.meal_period_cards
        # and meal_stream_rows:
        #   AttendanceEvent.objects
        #     .filter(student_id=OuterRef("student_id"), period_id=OuterRef("period_id"))
        #     .order_by("-ts", "-id")
        from django.db.models import OuterRef, Subquery
        ev1 = self._make_event(student=self.student, person=self.person,
                                period=self.occ, ts=self.now - timedelta(minutes=10),
                                score=0.80)
        ev2 = self._make_event(student=self.student, person=self.person,
                                period=self.occ, ts=self.now, score=0.95)
        # The subquery returns the latest event's score for this (student, period).
        latest_score_sq = (
            AttendanceEvent.objects
            .filter(student_id=OuterRef("student_id"), period_id=OuterRef("period_id"))
            .order_by("-ts", "-id")
            .values("score")[:1]
        )
        from apps.attendance.models import AttendanceRecord
        # Create a record so we have something to annotate onto.
        rec = AttendanceRecord.objects.create(
            student=self.student, period=self.occ,
            first_seen=self.now, last_seen=self.now, best_seen=self.now,
            best_score=0.95, status="present",
        )
        recs = list(
            AttendanceRecord.objects
            .annotate(latest_event_score=Subquery(latest_score_sq))
            .values("pk", "latest_event_score")
        )
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["latest_event_score"], 0.95)

    def test_admin_event_list_unchanged(self):
        # M2 must NOT add person to AttendanceEventAdmin.list_display /
        # search_fields / list_filter. Only student__* should appear.
        from apps.attendance.admin import AttendanceEventAdmin
        self.assertNotIn("person", AttendanceEventAdmin.list_display)
        self.assertFalse(
            any("person" in str(f) for f in AttendanceEventAdmin.search_fields)
        )
        self.assertFalse(
            any(str(f).startswith("person") for f in AttendanceEventAdmin.list_filter)
        )

    def test_camera_health_query_still_works(self):
        # meal_camera_health filters AttendanceEvent by camera only. M2 does
        # not touch that path. Verify a plain camera-filtered query works.
        ev = self._make_event(student=self.student, person=self.person,
                               period=None, ts=self.now, camera=None)
        # No camera set in fixtures; just confirm the query does not raise
        # and that .filter(camera=cam).order_by("-ts","-id").first() works
        # for the None camera case.
        latest = (
            AttendanceEvent.objects.filter(camera=None)
            .order_by("-ts", "-id").first()
        )
        self.assertEqual(latest, ev)

    def test_no_unique_constraint_on_event_person(self):
        # Anti-test: M2 must NOT add a uniq constraint on AttendanceEvent.person
        # (unlike M1's uniq_active_embedding_per_person).
        names = {c.name for c in AttendanceEvent._meta.constraints}
        self.assertEqual(names, set())

    def test_no_person_filter_in_admin_event_list(self):
        # Anti-test: M2 must NOT add a person filter to the admin.
        from apps.attendance.admin import AttendanceEventAdmin
        for f in AttendanceEventAdmin.list_filter:
            self.assertFalse(str(f).startswith("person"), f"unexpected person filter: {f}")


# ---------------------------------------------------------------------------
# M2 Section 12 — Meals bridge fallback regression
# ---------------------------------------------------------------------------


class BridgeFallbackRegressionTests(AttendanceEventPersonBase):
    """Verify the meals bridge still resolves Person via the
    ``attendance_event.student → StudentProfile.legacy_student → Person``
    backlink when ``attendance_event.person`` is NULL or set.

    M2 does NOT modify the bridge — these tests guard against accidental
    premature S11 switching in this milestone."""

    def test_bridge_resolves_person_when_event_person_null(self):
        # Bypass the service layer (mirroring _create_attendance_event in
        # apps.meals.tests_integrations_attendance) so person stays NULL.
        ev = self._make_event(student=self.student, person=None)
        self.assertIsNone(ev.person_id)
        from apps.meals.integrations.attendance import resolve_person_from_student
        resolved = resolve_person_from_student(ev.student)
        self.assertEqual(resolved, self.person)

    def test_bridge_resolves_person_when_event_person_set(self):
        # Even when attendance_event.person is set, the bridge reads
        # attendance_event.student and re-resolves via the backlink (M2 keeps
        # the bridge on the fallback chain).
        ev = self._make_event(student=self.student, person=self.person)
        self.assertEqual(ev.person_id, self.person.pk)
        from apps.meals.integrations.attendance import resolve_person_from_student
        resolved = resolve_person_from_student(ev.student)
        self.assertEqual(resolved, self.person)


# ---------------------------------------------------------------------------
# M2 Section 13 — IngestView wire contract regression
# ---------------------------------------------------------------------------


@override_settings(RUNNER_HEARTBEAT_KEY="")
class IngestViewWireContractTests(AttendanceEventPersonBase):
    """Verify the IngestView POST response shape is unchanged by M2 — no
    ``person_id`` / ``person`` leaks into the response."""

    def _post(self, h_code, score=0.95):
        import json
        from django.utils.http import urlencode
        # IngestView reads request.data (DRF JSON parser). Use JSON body.
        return self.client.post(
            "/api/attendance/ingest/",
            data=json.dumps({
                "h_code": h_code, "score": score, "ts": self.now.isoformat(),
                "camera_id": None, "crop_path": "captures/test.jpg",
            }),
            content_type="application/json",
        )

    def setUp(self):
        super().setUp()
        self._reset_settings()

    def test_ingest_view_response_shape_unchanged(self):
        resp = self._post("H-EVT01")
        self.assertEqual(resp.status_code, 200, resp.content)
        body = resp.json()
        # The response dict contains the legacy keys (ok, event_id, ...).
        self.assertTrue(body.get("ok"))
        self.assertIn("event_id", body)
        # M2 must NOT leak person_id / person into the response.
        self.assertNotIn("person_id", body)
        self.assertNotIn("person", body)

    def test_ingest_view_unknown_student_response_unchanged(self):
        resp = self._post("H-NOPE")
        self.assertEqual(resp.status_code, 200, resp.content)
        body = resp.json()
        self.assertFalse(body.get("ok"))
        self.assertEqual(body.get("error"), "student_not_found")
        self.assertNotIn("person_id", body)
        self.assertNotIn("person", body)


# ===========================================================================
# M3 — AttendanceRecord person FK
# ===========================================================================


# ---------------------------------------------------------------------------
# M3 Shared fixtures
# ---------------------------------------------------------------------------


class AttendanceRecordPersonBase(TestCase):
    """Shared fixtures for AttendanceRecord person-FK tests (M3)."""

    @classmethod
    def setUpTestData(cls):
        cls.role_student, _ = RoleType.objects.get_or_create(
            code="student", defaults={"name": "Student", "is_system": True}
        )
        # Migrated legacy Student
        cls.student = Student.objects.create(
            h_code="H-REC01", first_name="Rec", last_name="Ent",
            is_active=True,
        )
        cls.person = Person.objects.create(
            code="H-REC01", first_name="Rec", last_name="Ent",
        )
        cls.profile = StudentProfile.objects.create(
            person=cls.person, code="H-REC01", legacy_student=cls.student,
        )
        # Unmigrated legacy Student (no StudentProfile link)
        cls.unmigrated = Student.objects.create(
            h_code="H-REC02", first_name="Un", last_name="Mig",
            is_active=True,
        )
        # Minimal period template + occurrence that is "always open" so tests
        # can deterministically trigger the winners-loop branch.
        cls.template = PeriodTemplate.objects.create(
            name="Test Block Rec", order=1,
            start_time=timezone.datetime.min.time(),
            end_time=timezone.datetime.max.time(),
            weekdays_mask=127, is_enabled=True,
        )
        cls.now = timezone.localtime()
        cls.occ = PeriodOccurrence.objects.create(
            template=cls.template, date=cls.now.date(),
            start_dt=cls.now - timedelta(hours=1),
            end_dt=cls.now + timedelta(hours=1),
            is_school_day=True,
        )

    def _make_record(self, *, student=None, person=None, period=None,
                     score=0.9, ts=None):
        """Create an AttendanceRecord directly (bypasses the service layer)."""
        ts = ts or self.now
        return AttendanceRecord.objects.create(
            student=student if student is not None else self.student,
            person=person,
            period=period if period is not None else self.occ,
            first_seen=ts,
            last_seen=ts,
            best_seen=ts,
            best_score=score,
            best_crop="captures/test.jpg",
            sightings=1,
            status="present",
            pass_count=1,
            last_pass_at=ts,
        )

    def _reset_settings(self):
        """Reset the RecognitionSettings singleton to deterministic defaults."""
        rs, _ = RecognitionSettings.objects.get_or_create(pk=1)
        rs.min_score = 0.75
        rs.re_register_window_sec = 10
        rs.min_improve_delta = 0.01
        rs.max_periods_per_day = None
        rs.save()
        return rs


# ---------------------------------------------------------------------------
# M3 Section 14 — Schema correctness
# ---------------------------------------------------------------------------


class AttendanceRecordPersonFKTests(AttendanceRecordPersonBase):
    """Verify the person FK field, the coexisting unique_together, and indexes."""

    def test_person_field_exists_and_nullable(self):
        field = AttendanceRecord._meta.get_field("person")
        self.assertTrue(field.null)
        self.assertTrue(field.blank)
        self.assertEqual(field.remote_field.on_delete.__name__, "CASCADE")
        self.assertEqual(field.remote_field.related_name, "attendance_records")
        self.assertEqual(field.remote_field.model.__name__, "Person")

    def test_person_field_db_index(self):
        field = AttendanceRecord._meta.get_field("person")
        self.assertTrue(field.db_index)

    def test_person_field_related_name(self):
        rec = self._make_record(person=self.person)
        self.assertIn(rec, self.person.attendance_records.all())

    def test_existing_record_defaults_person_null(self):
        rec = self._make_record(person=None)
        self.assertIsNone(rec.person_id)

    def test_create_record_with_person(self):
        rec = self._make_record(person=self.person)
        self.assertEqual(rec.person_id, self.person.pk)
        self.assertEqual(rec.student_id, self.student.pk)

    def test_student_fk_unchanged(self):
        field = AttendanceRecord._meta.get_field("student")
        self.assertEqual(field.remote_field.on_delete.__name__, "CASCADE")
        self.assertFalse(field.remote_field.related_name)

    def test_student_period_unique_together_unchanged(self):
        uts = set(AttendanceRecord._meta.unique_together)
        self.assertIn(("student", "period"), uts)

    def test_person_period_unique_together_added(self):
        uts = set(AttendanceRecord._meta.unique_together)
        self.assertIn(("person", "period"), uts)

    def test_student_period_index_unchanged(self):
        idx_fields = {tuple(i.fields) for i in AttendanceRecord._meta.indexes}
        self.assertIn(("student", "period"), idx_fields)

    def test_person_period_index_added(self):
        idx_fields = {tuple(i.fields) for i in AttendanceRecord._meta.indexes}
        self.assertIn(("person", "period"), idx_fields)


# ---------------------------------------------------------------------------
# M3 Section 15 — Constraint behavior
# ---------------------------------------------------------------------------


class AttendanceRecordConstraintTests(AttendanceRecordPersonBase):
    """Verify the (person, period) unique constraint coexists with (student, period)."""

    def test_two_records_same_person_same_period_raises(self):
        self._make_record(person=self.person)
        with self.assertRaises(IntegrityError):
            self._make_record(person=self.person)

    def test_two_records_same_student_same_period_raises(self):
        self._make_record(person=None)
        with self.assertRaises(IntegrityError):
            self._make_record(person=None)

    def test_null_person_same_period_allowed(self):
        # PostgreSQL: multiple NULLs in a UNIQUE constraint are allowed.
        # Two unmigrated students with records for the same period should be fine.
        s2 = Student.objects.create(h_code="H-NULL2", first_name="N", last_name="N")
        rec1 = self._make_record(student=self.unmigrated, person=None)
        rec2 = self._make_record(student=s2, person=None)
        self.assertIsNone(rec1.person_id)
        self.assertIsNone(rec2.person_id)

    def test_different_persons_same_period_allowed(self):
        s2 = Student.objects.create(h_code="H-REC03", first_name="Other", last_name="P")
        p2 = Person.objects.create(code="H-REC03", first_name="Other", last_name="P")
        StudentProfile.objects.create(person=p2, code="H-REC03", legacy_student=s2)
        rec1 = self._make_record(student=self.student, person=self.person)
        rec2 = self._make_record(student=s2, person=p2)
        self.assertEqual(rec1.person_id, self.person.pk)
        self.assertEqual(rec2.person_id, p2.pk)

    def test_same_person_different_periods_allowed(self):
        occ2 = PeriodOccurrence.objects.create(
            template=self.template, date=self.now.date() + timedelta(days=1),
            start_dt=self.now + timedelta(days=1, hours=-1),
            end_dt=self.now + timedelta(days=1, hours=1),
            is_school_day=True,
        )
        rec1 = self._make_record(person=self.person, period=self.occ)
        rec2 = self._make_record(person=self.person, period=occ2)
        self.assertEqual(rec1.person_id, self.person.pk)
        self.assertEqual(rec2.person_id, self.person.pk)


# ---------------------------------------------------------------------------
# M3 Section 16 — Backfill migration 0037
# ---------------------------------------------------------------------------


class RecordBackfillMigrationTests(AttendanceRecordPersonBase):
    """Verify the 0037 backfill RunPython directly (idempotent, noop reverse,
    skips unmigrated students, batches correctly)."""

    def _run_backfill(self):
        from apps.attendance.migrations._0037_helper import forwards, backwards
        from django.apps import apps
        forwards(apps, None)

    def _run_backfill_backwards(self):
        from apps.attendance.migrations._0037_helper import backwards
        from django.apps import apps
        backwards(apps, None)

    def test_backfill_sets_person_id(self):
        rec = self._make_record(person=None)
        self.assertIsNone(rec.person_id)
        self._run_backfill()
        rec.refresh_from_db()
        self.assertEqual(rec.person_id, self.person.pk)

    def test_backfill_skips_already_backfilled(self):
        rec = self._make_record(person=self.person)
        original_person_id = rec.person_id
        self._run_backfill()
        rec.refresh_from_db()
        self.assertEqual(rec.person_id, original_person_id)

    def test_backfill_skips_unmigrated_students(self):
        rec = self._make_record(student=self.unmigrated, person=None)
        self._run_backfill()
        rec.refresh_from_db()
        self.assertIsNone(rec.person_id)

    def test_backfill_skips_null_student(self):
        # Defensive: calling forwards with no rows to backfill does not raise.
        self._run_backfill()

    def test_backfill_reverse_is_noop(self):
        rec = self._make_record(person=self.person)
        self._run_backfill_backwards()
        rec.refresh_from_db()
        self.assertEqual(rec.person_id, self.person.pk)

    def test_backfill_idempotent(self):
        rec = self._make_record(person=None)
        self._run_backfill()
        rec.refresh_from_db()
        first = rec.person_id
        self._run_backfill()
        rec.refresh_from_db()
        self.assertEqual(rec.person_id, first)

    def test_backfill_batches_large_record_set(self):
        # Create records with mixed migrated/unmigrated students to exercise
        # the batch flush logic. Each record needs a distinct period to avoid
        # violating the (student, period) unique constraint.
        periods = []
        for i in range(1, 13):  # start from day 1 to avoid colliding with self.occ
            p = PeriodOccurrence.objects.create(
                template=self.template, date=self.now.date() + timedelta(days=i),
                start_dt=self.now + timedelta(days=i, hours=-1),
                end_dt=self.now + timedelta(days=i, hours=1),
                is_school_day=True,
            )
            periods.append(p)
        for i in range(7):
            self._make_record(student=self.student, person=None, period=periods[i])
        for i in range(5):
            self._make_record(student=self.unmigrated, person=None, period=periods[7 + i])
        self._run_backfill()
        migrated_count = (
            AttendanceRecord.objects.filter(student=self.student)
            .exclude(person__isnull=True).count()
        )
        self.assertEqual(migrated_count, 7)
        unmigrated_count = (
            AttendanceRecord.objects.filter(student=self.unmigrated, person__isnull=True).count()
        )
        self.assertEqual(unmigrated_count, 5)


# ---------------------------------------------------------------------------
# M3 Section 17 — Dual-write via get_or_create defaults
# ---------------------------------------------------------------------------


@override_settings(ATTENDANCE_DEDUP_SECONDS=0, RUNNER_HEARTBEAT_KEY="")
class RecordDualWriteDefaultsTests(AttendanceRecordPersonBase):
    """Verify new records get person via defaults and existing records get
    person_id defensively set when found by (student, period)."""

    def setUp(self):
        super().setUp()
        self._reset_settings()

    def test_new_record_gets_person_via_defaults(self):
        from apps.attendance.services import ingest_match
        res = ingest_match(
            h_code="H-REC01", score=0.95, camera=None, ts=self.now,
            crop_path="captures/test.jpg",
        )
        self.assertTrue(res.get("ok"), res)
        rec = AttendanceRecord.objects.get(student=self.student)
        self.assertEqual(rec.student_id, self.student.pk)
        self.assertEqual(rec.person_id, self.person.pk)

    def test_new_record_unmigrated_person_null(self):
        from apps.attendance.services import ingest_match
        res = ingest_match(
            h_code="H-REC02", score=0.95, camera=None, ts=self.now,
            crop_path="captures/test.jpg",
        )
        self.assertTrue(res.get("ok"), res)
        rec = AttendanceRecord.objects.get(student=self.unmigrated)
        self.assertEqual(rec.student_id, self.unmigrated.pk)
        self.assertIsNone(rec.person_id)

    def test_existing_record_found_by_student_period(self):
        from apps.attendance.services import ingest_match
        # Create a record directly with person=None (bypass service layer).
        self._make_record(person=None)
        self.assertEqual(AttendanceRecord.objects.filter(student=self.student).count(), 1)
        # Now call ingest_match — should find the existing record, NOT create a duplicate.
        res = ingest_match(
            h_code="H-REC01", score=0.96, camera=None, ts=self.now,
            crop_path="captures/test.jpg",
        )
        self.assertTrue(res.get("ok"), res)
        self.assertEqual(AttendanceRecord.objects.filter(student=self.student).count(), 1)

    def test_existing_record_defensive_person_set(self):
        from apps.attendance.services import ingest_match
        # Create a record directly with person=None.
        rec = self._make_record(person=None)
        self.assertIsNone(rec.person_id)
        # Call ingest_match — should find the record and defensively set person_id.
        ingest_match(
            h_code="H-REC01", score=0.96, camera=None, ts=self.now,
            crop_path="captures/test.jpg",
        )
        rec.refresh_from_db()
        self.assertEqual(rec.person_id, self.person.pk)

    def test_existing_record_person_not_overwritten(self):
        from apps.attendance.services import ingest_match
        # Create a record with person already set.
        rec = self._make_record(person=self.person)
        original_person_id = rec.person_id
        # Call ingest_match with a higher score — should update the record,
        # not overwrite person_id.
        ingest_match(
            h_code="H-REC01", score=0.99, camera=None, ts=self.now,
            crop_path="captures/test.jpg",
        )
        rec.refresh_from_db()
        self.assertEqual(rec.person_id, original_person_id)

    def test_existing_record_unmigrated_stays_null(self):
        from apps.attendance.services import ingest_match
        rec = self._make_record(student=self.unmigrated, person=None)
        self.assertIsNone(rec.person_id)
        ingest_match(
            h_code="H-REC02", score=0.95, camera=None, ts=self.now,
            crop_path="captures/test.jpg",
        )
        rec.refresh_from_db()
        self.assertIsNone(rec.person_id)

    def test_no_duplicate_records_created(self):
        from apps.attendance.services import ingest_match
        for i in range(5):
            ingest_match(
                h_code="H-REC01", score=0.90 + i * 0.01, camera=None,
                ts=self.now + timedelta(seconds=i),
                crop_path="captures/test.jpg",
            )
        self.assertEqual(AttendanceRecord.objects.filter(student=self.student).count(), 1)


# ---------------------------------------------------------------------------
# M3 Section 18 — Re-register window path
# ---------------------------------------------------------------------------


@override_settings(ATTENDANCE_DEDUP_SECONDS=0, RUNNER_HEARTBEAT_KEY="")
class RecordReregisterWindowTests(AttendanceRecordPersonBase):
    """Verify the defensive person_id set works in the re-register window
    short-circuit path (the save(update_fields=...) path)."""

    def setUp(self):
        super().setUp()
        rs = self._reset_settings()
        # Large re-register window so the short-circuit path is always taken.
        rs.re_register_window_sec = 3600
        # Small delta so the short-circuit is taken when score does NOT improve.
        rs.min_improve_delta = 0.01
        rs.save()

    def test_reregister_window_sets_person_id_in_update_fields(self):
        from apps.attendance.services import ingest_match
        # Create a record directly with person=None and a high best_score.
        rec = self._make_record(person=None, score=0.95)
        self.assertIsNone(rec.person_id)
        # Send a second event with a LOWER score that does NOT improve by
        # min_improve_delta → the re-register short-circuit path is taken.
        ingest_match(
            h_code="H-REC01", score=0.95, camera=None, ts=self.now + timedelta(seconds=5),
            crop_path="captures/test.jpg",
        )
        rec.refresh_from_db()
        # person_id must have been persisted (not just set in memory).
        self.assertEqual(rec.person_id, self.person.pk)

    def test_reregister_window_does_not_overwrite_existing_person(self):
        from apps.attendance.services import ingest_match
        rec = self._make_record(person=self.person, score=0.95)
        original_person_id = rec.person_id
        ingest_match(
            h_code="H-REC01", score=0.95, camera=None, ts=self.now + timedelta(seconds=5),
            crop_path="captures/test.jpg",
        )
        rec.refresh_from_db()
        self.assertEqual(rec.person_id, original_person_id)


# ---------------------------------------------------------------------------
# M3 Section 19 — Upsert invariant regression
# ---------------------------------------------------------------------------


@override_settings(ATTENDANCE_DEDUP_SECONDS=0, RUNNER_HEARTBEAT_KEY="")
class UpsertInvariantRegressionTests(AttendanceRecordPersonBase):
    """Verify the upsert invariant is preserved under various scenarios."""

    def setUp(self):
        super().setUp()
        self._reset_settings()

    def test_multiple_events_same_student_period_one_record(self):
        from apps.attendance.services import ingest_match
        for i in range(5):
            ingest_match(
                h_code="H-REC01", score=0.90 + i * 0.01, camera=None,
                ts=self.now + timedelta(seconds=i),
                crop_path="captures/test.jpg",
            )
        self.assertEqual(AttendanceRecord.objects.filter(student=self.student).count(), 1)
        rec = AttendanceRecord.objects.get(student=self.student)
        self.assertGreaterEqual(rec.sightings, 5)
        self.assertAlmostEqual(rec.best_score, 0.94)

    def test_multiple_events_migrated_then_unmigrated(self):
        from apps.attendance.services import ingest_match
        ingest_match(
            h_code="H-REC01", score=0.95, camera=None, ts=self.now,
            crop_path="captures/test.jpg",
        )
        ingest_match(
            h_code="H-REC02", score=0.95, camera=None, ts=self.now,
            crop_path="captures/test.jpg",
        )
        self.assertEqual(AttendanceRecord.objects.count(), 2)
        rec_migrated = AttendanceRecord.objects.get(student=self.student)
        rec_unmigrated = AttendanceRecord.objects.get(student=self.unmigrated)
        self.assertEqual(rec_migrated.person_id, self.person.pk)
        self.assertIsNone(rec_unmigrated.person_id)

    @override_settings(MULTI_PERIOD_ON_TIES=True)
    def test_multi_period_tie_one_record_per_period(self):
        from apps.attendance.services import ingest_match
        template2 = PeriodTemplate.objects.create(
            name="Test Block Rec 2", order=1,  # same lowest order
            start_time=timezone.datetime.min.time(),
            end_time=timezone.datetime.max.time(),
            weekdays_mask=127, is_enabled=True,
        )
        occ2 = PeriodOccurrence.objects.create(
            template=template2, date=self.now.date(),
            start_dt=self.now - timedelta(hours=1),
            end_dt=self.now + timedelta(hours=1),
            is_school_day=True,
        )
        res = ingest_match(
            h_code="H-REC01", score=0.95, camera=None, ts=self.now,
            crop_path="captures/test.jpg",
        )
        self.assertTrue(res.get("ok"))
        self.assertEqual(res.get("winners"), 2)
        # One record per period.
        self.assertEqual(AttendanceRecord.objects.filter(student=self.student).count(), 2)
        for rec in AttendanceRecord.objects.filter(student=self.student):
            self.assertEqual(rec.person_id, self.person.pk)


# ---------------------------------------------------------------------------
# M3 Section 20 — Read-path regression
# ---------------------------------------------------------------------------


class RecordReadPathRegressionTests(AttendanceRecordPersonBase):
    """Verify legacy read paths still work after M3 (student FK intact)."""

    def test_admin_record_list_unchanged(self):
        from apps.attendance.admin import AttendanceRecordAdmin
        self.assertNotIn("person", AttendanceRecordAdmin.list_display)
        self.assertFalse(
            any("person" in str(f) for f in AttendanceRecordAdmin.search_fields)
        )
        self.assertFalse(
            any(str(f).startswith("person") for f in AttendanceRecordAdmin.list_filter)
        )

    def test_no_person_filter_in_admin_record_list(self):
        from apps.attendance.admin import AttendanceRecordAdmin
        for f in AttendanceRecordAdmin.list_filter:
            self.assertFalse(str(f).startswith("person"), f"unexpected person filter: {f}")

    def test_serializer_sources_student(self):
        from apps.attendance.serializers import AttendanceRecordSerializer
        # The serializer must NOT have person_code / person_full_name fields.
        field_names = set(AttendanceRecordSerializer._declared_fields.keys())
        self.assertNotIn("person_code", field_names)
        self.assertNotIn("person_full_name", field_names)
        self.assertNotIn("person_id", field_names)
        self.assertNotIn("person", field_names)


# ---------------------------------------------------------------------------
# M3 Section 21 — Bridge fallback regression
# ---------------------------------------------------------------------------


class RecordBridgeFallbackRegressionTests(AttendanceRecordPersonBase):
    """Verify the meals bridge still resolves Person via the backlink when
    attendance_record.person is NULL or set."""

    def test_bridge_resolves_person_when_record_person_null(self):
        rec = self._make_record(person=None)
        self.assertIsNone(rec.person_id)
        from apps.meals.integrations.attendance import resolve_person_from_student
        resolved = resolve_person_from_student(rec.student)
        self.assertEqual(resolved, self.person)

    def test_bridge_resolves_person_when_record_person_set(self):
        rec = self._make_record(person=self.person)
        self.assertEqual(rec.person_id, self.person.pk)
        from apps.meals.integrations.attendance import resolve_person_from_student
        resolved = resolve_person_from_student(rec.student)
        self.assertEqual(resolved, self.person)
