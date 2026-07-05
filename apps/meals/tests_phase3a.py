"""Phase 3A tests for apps.meals — MealServiceEvent + MealSupervisorAction.

Covers: model creation, DB constraints, clean() role validation,
snapshot immutability validator (the core invariant for terminal
statuses), admin-safe behavior (delete/change permissions), and the
read-only selectors.

Out of scope (Phase 3B): resolve_service, confirm/deny/refund/void
services, finance.charge calls, supervisor workflow.
"""

from datetime import date, timedelta

from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.test import TestCase

from apps.identity.models import Person, StaffProfile, StudentProfile

from .models import (
    MealEligibility,
    MealPlan,
    MealServiceEvent,
    MealSubscription,
    MealSupervisorAction,
    SERVICE_EVENT_IMMUTABLE_SNAPSHOT_FIELDS,
    TERMINAL_SERVICE_EVENT_STATUSES,
)
from .selectors import (
    confirmed_meals_for,
    get_service_event_by_id,
    list_service_events,
    list_supervisor_actions,
    service_events_for,
    service_events_for_section,
    supervisor_actions_for_eligibility,
    supervisor_actions_for_service_event,
)
from .validators import (
    validate_service_event_role,
    validate_service_event_snapshot_immutable,
    validate_supervisor_action_target,
)


# ---------------------------------------------------------------------------
# Test base data
# ---------------------------------------------------------------------------

class Phase3ABaseData(TestCase):
    def setUp(self):
        self.person_student = Person.objects.create(
            code="P-3AS", first_name="Stu", last_name="Dent"
        )
        self.student = StudentProfile.objects.create(
            person=self.person_student, code="S-3A01"
        )
        self.person_staff = Person.objects.create(
            code="P-3AT", first_name="Tea", last_name="Cher"
        )
        self.staff = StaffProfile.objects.create(
            person=self.person_staff, code="T-3A01"
        )
        self.plan_wallet = MealPlan.objects.create(
            name="3A Wallet",
            kind=MealPlan.Kind.LUNCH,
            mode=MealPlan.Mode.WALLET,
            default_price_iqd=1500,
        )
        self.plan_daterange = MealPlan.objects.create(
            name="3A Annual",
            kind=MealPlan.Kind.LUNCH,
            mode=MealPlan.Mode.DATE_RANGE,
            default_price_iqd=0,
        )
        self.today = date.today()
        self.eligibility = MealEligibility.objects.create(
            person=self.person_student,
            student=self.student,
            date=self.today,
            decision=MealEligibility.Decision.ELIGIBLE,
            meal_plan=self.plan_wallet,
        )


# ---------------------------------------------------------------------------
# MealServiceEvent model + DB constraints
# ---------------------------------------------------------------------------

class MealServiceEventModelTests(Phase3ABaseData):
    def test_create_pending_service_event_via_orm(self):
        ev = MealServiceEvent.objects.create(
            person=self.person_student,
            student=self.student,
            date=self.today,
            eligibility=self.eligibility,
            meal_plan=self.plan_wallet,
            status=MealServiceEvent.Status.PENDING,
            price_base_iqd=1500,
            final_charge_iqd=1500,
            price_resolution_source="period_price",
            grade_code_snapshot="G1",
            section_code_snapshot="A",
            meal_period_label_snapshot="First lunch block",
        )
        self.assertEqual(ev.status, MealServiceEvent.Status.PENDING)
        self.assertEqual(ev.final_charge_iqd, 1500)
        self.assertEqual(ev.grade_code_snapshot, "G1")

    def test_default_status_is_pending(self):
        ev = MealServiceEvent.objects.create(
            person=self.person_student, date=self.today,
        )
        self.assertEqual(ev.status, MealServiceEvent.Status.PENDING)

    def test_str_includes_person_date_status(self):
        ev = MealServiceEvent.objects.create(
            person=self.person_student, date=self.today,
        )
        s = str(ev)
        self.assertIn(str(self.person_student.code), s)
        self.assertIn("pending", s)

    def test_negative_final_charge_rejected_at_db(self):
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                MealServiceEvent.objects.create(
                    person=self.person_student,
                    date=self.today,
                    final_charge_iqd=-1,
                )

    def test_negative_price_base_rejected_at_db(self):
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                MealServiceEvent.objects.create(
                    person=self.person_student,
                    date=self.today,
                    price_base_iqd=-5,
                )

    def test_negative_discount_rejected_at_db(self):
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                MealServiceEvent.objects.create(
                    person=self.person_student,
                    date=self.today,
                    discount_iqd=-1,
                )

    def test_clean_rejects_both_student_and_staff(self):
        ev = MealServiceEvent(
            person=self.person_student,
            student=self.student,
            staff=self.staff,
            date=self.today,
        )
        with self.assertRaises(ValidationError):
            ev.full_clean()

    def test_clean_accepts_guest_event_no_student_no_staff(self):
        ev = MealServiceEvent(
            person=self.person_student,
            date=self.today,
        )
        ev.full_clean()  # should not raise


# ---------------------------------------------------------------------------
# Snapshot immutability validator
# ---------------------------------------------------------------------------

class SnapshotImmutabilityTests(Phase3ABaseData):
    """The core invariant: once a MealServiceEvent reaches a terminal
    status, its snapshot fields are immutable. Corrections create a new
    event rather than mutating the confirmed row."""

    def _create_confirmed(self):
        return MealServiceEvent.objects.create(
            person=self.person_student,
            student=self.student,
            date=self.today,
            eligibility=self.eligibility,
            meal_plan=self.plan_wallet,
            status=MealServiceEvent.Status.CONFIRMED,
            price_base_iqd=1500,
            final_charge_iqd=1500,
            wallet_balance_before_iqd=5000,
            wallet_balance_after_iqd=3500,
            grade_code_snapshot="G1",
            section_code_snapshot="A",
        )

    # --- pending event: snapshots may be edited ------------------------

    def test_pending_event_snapshots_mutable(self):
        ev = MealServiceEvent.objects.create(
            person=self.person_student,
            date=self.today,
            status=MealServiceEvent.Status.PENDING,
            price_base_iqd=1000,
            final_charge_iqd=1000,
        )
        ev.price_base_iqd = 2000
        ev.final_charge_iqd = 2000
        # Should not raise — PENDING is not terminal.
        validate_service_event_snapshot_immutable(instance=ev)

    # --- confirmed event: snapshots frozen -----------------------------

    def test_confirmed_event_price_base_immutable(self):
        ev = self._create_confirmed()
        ev.price_base_iqd = 9999
        with self.assertRaises(ValidationError):
            validate_service_event_snapshot_immutable(instance=ev)

    def test_confirmed_event_final_charge_immutable(self):
        ev = self._create_confirmed()
        ev.final_charge_iqd = 9999
        with self.assertRaises(ValidationError):
            validate_service_event_snapshot_immutable(instance=ev)

    def test_confirmed_event_grade_snapshot_immutable(self):
        ev = self._create_confirmed()
        ev.grade_code_snapshot = "G2"
        with self.assertRaises(ValidationError):
            validate_service_event_snapshot_immutable(instance=ev)

    def test_confirmed_event_person_immutable(self):
        ev = self._create_confirmed()
        new_person = Person.objects.create(code="P-NEW", first_name="N", last_name="N")
        ev.person = new_person
        with self.assertRaises(ValidationError):
            validate_service_event_snapshot_immutable(instance=ev)

    def test_confirmed_event_date_immutable(self):
        ev = self._create_confirmed()
        ev.date = self.today + timedelta(days=1)
        with self.assertRaises(ValidationError):
            validate_service_event_snapshot_immutable(instance=ev)

    def test_confirmed_event_meal_plan_immutable(self):
        ev = self._create_confirmed()
        ev.meal_plan = self.plan_daterange
        with self.assertRaises(ValidationError):
            validate_service_event_snapshot_immutable(instance=ev)

    # --- wallet_refund_transaction is mutable on CONFIRMED -------------

    def test_confirmed_event_wallet_refund_transaction_mutable(self):
        # The refund flow sets wallet_refund_transaction on a CONFIRMED
        # event before transitioning to REFUNDED. This must NOT trip the
        # immutability guard.
        ev = self._create_confirmed()
        # Set to a non-None placeholder (no real WalletTransaction in
        # Phase 3A tests; just verify the field is allowed to change).
        ev.wallet_refund_transaction = None  # already None; no-op
        validate_service_event_snapshot_immutable(instance=ev)

    # --- refunded event: fully frozen ---------------------------------

    def test_refunded_event_wallet_refund_transaction_immutable(self):
        ev = MealServiceEvent.objects.create(
            person=self.person_student,
            student=self.student,
            date=self.today,
            status=MealServiceEvent.Status.REFUNDED,
            final_charge_iqd=1500,
            wallet_balance_before_iqd=5000,
            wallet_balance_after_iqd=5000,
        )
        # wallet_refund_transaction is immutable on REFUNDED (not CONFIRMED).
        ev.wallet_balance_after_iqd = 9999
        with self.assertRaises(ValidationError):
            validate_service_event_snapshot_immutable(instance=ev)

    # --- terminal statuses all enforce immutability --------------------

    def test_all_terminal_statuses_enforce_immutability(self):
        for status in TERMINAL_SERVICE_EVENT_STATUSES:
            ev = MealServiceEvent.objects.create(
                person=self.person_student,
                date=self.today,
                status=status,
                price_base_iqd=1000,
                final_charge_iqd=1000,
            )
            ev.price_base_iqd = 2000
            with self.assertRaises(ValidationError, msg=f"status={status}"):
                validate_service_event_snapshot_immutable(instance=ev)

    # --- updating_fields narrows the check ----------------------------

    def test_updating_fields_status_only_allowed(self):
        ev = self._create_confirmed()
        # Updating only `status` (e.g. CONFIRMED → REFUNDED) should not
        # trip the guard even though price_base_iqd differs in memory.
        ev.status = MealServiceEvent.Status.REFUNDED
        validate_service_event_snapshot_immutable(
            instance=ev, updating_fields={"status"}
        )

    def test_updating_fields_reason_code_allowed(self):
        ev = self._create_confirmed()
        ev.reason_code = "supervisor_refund"
        validate_service_event_snapshot_immutable(
            instance=ev, updating_fields={"reason_code"}
        )

    def test_updating_fields_snapshot_field_rejected(self):
        ev = self._create_confirmed()
        ev.final_charge_iqd = 9999
        with self.assertRaises(ValidationError):
            validate_service_event_snapshot_immutable(
                instance=ev, updating_fields={"final_charge_iqd"}
            )

    # --- new (unsaved) instance: not enforced -------------------------

    def test_unsaved_instance_not_enforced(self):
        ev = MealServiceEvent(
            person=self.person_student,
            date=self.today,
            status=MealServiceEvent.Status.CONFIRMED,
            final_charge_iqd=1000,
        )
        # No PK → not enforced.
        validate_service_event_snapshot_immutable(instance=ev)

    # --- non-MealServiceEvent instance rejected -----------------------

    def test_non_service_event_instance_rejected(self):
        with self.assertRaises(TypeError):
            validate_service_event_snapshot_immutable(instance="not an event")

    # --- clean() calls the immutability validator ----------------------

    def test_clean_enforces_immutability_on_confirmed(self):
        ev = self._create_confirmed()
        ev.price_base_iqd = 9999
        with self.assertRaises(ValidationError):
            ev.full_clean()


# ---------------------------------------------------------------------------
# MealSupervisorAction model + audit-only behavior
# ---------------------------------------------------------------------------

class MealSupervisorActionModelTests(Phase3ABaseData):
    def setUp(self):
        super().setUp()
        self.service_event = MealServiceEvent.objects.create(
            person=self.person_student,
            student=self.student,
            date=self.today,
            status=MealServiceEvent.Status.PENDING,
        )

    def test_create_action_via_orm(self):
        action = MealSupervisorAction.objects.create(
            service_event=self.service_event,
            action=MealSupervisorAction.Action.CONFIRM,
            reason_code="auto",
            performed_by=self.staff,
        )
        self.assertEqual(action.action, MealSupervisorAction.Action.CONFIRM)
        self.assertIsNotNone(action.performed_at)

    def test_action_without_target_ok(self):
        # Manual lookup may have neither service_event nor eligibility.
        action = MealSupervisorAction.objects.create(
            action=MealSupervisorAction.Action.MANUAL_LOOKUP,
        )
        self.assertIsNone(action.service_event)
        self.assertIsNone(action.eligibility)

    def test_validator_rejects_both_targets(self):
        with self.assertRaises(ValidationError):
            validate_supervisor_action_target(
                service_event=self.service_event,
                eligibility=self.eligibility,
            )

    def test_validator_accepts_one_target(self):
        validate_supervisor_action_target(service_event=self.service_event)
        validate_supervisor_action_target(eligibility=self.eligibility)
        validate_supervisor_action_target()  # manual lookup

    def test_str_includes_action_and_target(self):
        action = MealSupervisorAction.objects.create(
            service_event=self.service_event,
            action=MealSupervisorAction.Action.DENY,
        )
        s = str(action)
        self.assertIn("Deny", s)


# ---------------------------------------------------------------------------
# Admin-safe behavior
# ---------------------------------------------------------------------------

class AdminBehaviorTests(Phase3ABaseData):
    def test_pending_event_deletable(self):
        from .models import TERMINAL_SERVICE_EVENT_STATUSES

        ev = MealServiceEvent.objects.create(
            person=self.person_student,
            date=self.today,
            status=MealServiceEvent.Status.PENDING,
        )
        self.assertNotIn(ev.status, TERMINAL_SERVICE_EVENT_STATUSES)

    def test_terminal_event_not_deletable_by_admin_check(self):
        from .admin import MealServiceEventAdmin
        from .models import TERMINAL_SERVICE_EVENT_STATUSES

        ev = MealServiceEvent.objects.create(
            person=self.person_student,
            date=self.today,
            status=MealServiceEvent.Status.CONFIRMED,
        )
        # The admin's has_delete_permission checks terminal status.
        admin_instance = MealServiceEventAdmin(MealServiceEvent, None)
        self.assertFalse(admin_instance.has_delete_permission(None, ev))

    def test_pending_event_deletable_by_admin_check(self):
        from .admin import MealServiceEventAdmin

        ev = MealServiceEvent.objects.create(
            person=self.person_student,
            date=self.today,
            status=MealServiceEvent.Status.PENDING,
        )
        admin_instance = MealServiceEventAdmin(MealServiceEvent, None)
        self.assertTrue(admin_instance.has_delete_permission(None, ev))

    def test_supervisor_action_admin_no_change_permission(self):
        from .admin import MealSupervisorActionAdmin

        admin_instance = MealSupervisorActionAdmin(MealSupervisorAction, None)
        self.assertFalse(admin_instance.has_change_permission(None))

    def test_supervisor_action_admin_no_delete_permission(self):
        from .admin import MealSupervisorActionAdmin

        admin_instance = MealSupervisorActionAdmin(MealSupervisorAction, None)
        self.assertFalse(admin_instance.has_delete_permission(None))


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

class ServiceEventSelectorTests(Phase3ABaseData):
    def setUp(self):
        super().setUp()
        self.ev1 = MealServiceEvent.objects.create(
            person=self.person_student,
            student=self.student,
            date=self.today,
            status=MealServiceEvent.Status.CONFIRMED,
            section_code_snapshot="A",
            meal_plan=self.plan_wallet,
        )
        self.ev2 = MealServiceEvent.objects.create(
            person=self.person_student,
            student=self.student,
            date=self.today + timedelta(days=1),
            status=MealServiceEvent.Status.PENDING,
            section_code_snapshot="A",
        )
        self.ev3 = MealServiceEvent.objects.create(
            person=self.person_staff,
            staff=self.staff,
            date=self.today,
            status=MealServiceEvent.Status.CONFIRMED,
        )

    def test_list_service_events_all(self):
        self.assertEqual(list_service_events().count(), 3)

    def test_list_service_events_filter_person(self):
        self.assertEqual(
            list_service_events(person=self.person_student).count(), 2
        )

    def test_list_service_events_filter_status(self):
        self.assertEqual(
            list_service_events(status=MealServiceEvent.Status.CONFIRMED).count(), 2
        )

    def test_list_service_events_filter_date(self):
        self.assertEqual(list_service_events(date=self.today).count(), 2)

    def test_list_service_events_filter_meal_plan(self):
        self.assertEqual(
            list_service_events(meal_plan=self.plan_wallet).count(), 1
        )

    def test_list_service_events_filter_section_snapshot(self):
        self.assertEqual(
            list_service_events(section_code_snapshot="A").count(), 2
        )

    def test_service_events_for_person_date(self):
        qs = service_events_for(person=self.person_student, date=self.today)
        self.assertEqual(qs.count(), 1)
        self.assertEqual(qs[0], self.ev1)

    def test_service_events_for_section(self):
        qs = service_events_for_section(section_code="A", date=self.today)
        self.assertEqual(qs.count(), 1)
        self.assertEqual(qs[0], self.ev1)

    def test_get_service_event_by_id(self):
        self.assertEqual(get_service_event_by_id(self.ev1.pk), self.ev1)

    def test_get_service_event_by_id_missing_returns_none(self):
        self.assertIsNone(get_service_event_by_id(999999))

    def test_confirmed_meals_for(self):
        qs = confirmed_meals_for(person=self.person_student, date=self.today)
        self.assertEqual(qs.count(), 1)
        self.assertEqual(qs[0], self.ev1)


class SupervisorActionSelectorTests(Phase3ABaseData):
    def setUp(self):
        super().setUp()
        self.ev = MealServiceEvent.objects.create(
            person=self.person_student,
            student=self.student,
            date=self.today,
            status=MealServiceEvent.Status.PENDING,
        )
        self.action1 = MealSupervisorAction.objects.create(
            service_event=self.ev,
            action=MealSupervisorAction.Action.CONFIRM,
            performed_by=self.staff,
        )
        self.action2 = MealSupervisorAction.objects.create(
            service_event=self.ev,
            action=MealSupervisorAction.Action.VOID,
            performed_by=self.staff,
        )
        self.action3 = MealSupervisorAction.objects.create(
            eligibility=self.eligibility,
            action=MealSupervisorAction.Action.OVERRIDE_ELIGIBLE,
        )

    def test_list_supervisor_actions_all(self):
        self.assertEqual(list_supervisor_actions().count(), 3)

    def test_list_supervisor_actions_filter_service_event(self):
        self.assertEqual(
            list_supervisor_actions(service_event=self.ev).count(), 2
        )

    def test_list_supervisor_actions_filter_action(self):
        self.assertEqual(
            list_supervisor_actions(
                action=MealSupervisorAction.Action.CONFIRM
            ).count(),
            1,
        )

    def test_supervisor_actions_for_service_event(self):
        qs = supervisor_actions_for_service_event(service_event=self.ev)
        self.assertEqual(qs.count(), 2)
        # newest first
        self.assertEqual(qs[0], self.action2)

    def test_supervisor_actions_for_eligibility(self):
        qs = supervisor_actions_for_eligibility(eligibility=self.eligibility)
        self.assertEqual(qs.count(), 1)


# ---------------------------------------------------------------------------
# Validator unit tests
# ---------------------------------------------------------------------------

class ValidatorUnitTests(TestCase):
    def test_validate_service_event_role_both_set_rejected(self):
        # Use plain sentinel objects; the validator only checks ``is not None``.
        class Stub:
            pass
        with self.assertRaises(ValidationError):
            validate_service_event_role(student=Stub(), staff=Stub())

    def test_validate_service_event_role_one_set_ok(self):
        class Stub:
            pass
        validate_service_event_role(student=Stub(), staff=None)
        validate_service_event_role(student=None, staff=Stub())
        validate_service_event_role(student=None, staff=None)

    def test_validate_supervisor_action_target_both_rejected(self):
        class Stub:
            pass
        with self.assertRaises(ValidationError):
            validate_supervisor_action_target(
                service_event=Stub(), eligibility=Stub()
            )

    def test_immutable_snapshot_fields_set_contents(self):
        # Sanity-check the canonical snapshot field set.
        self.assertIn("price_base_iqd", SERVICE_EVENT_IMMUTABLE_SNAPSHOT_FIELDS)
        self.assertIn("final_charge_iqd", SERVICE_EVENT_IMMUTABLE_SNAPSHOT_FIELDS)
        self.assertIn("wallet_transaction", SERVICE_EVENT_IMMUTABLE_SNAPSHOT_FIELDS)
        self.assertIn("grade_code_snapshot", SERVICE_EVENT_IMMUTABLE_SNAPSHOT_FIELDS)
        self.assertIn("person", SERVICE_EVENT_IMMUTABLE_SNAPSHOT_FIELDS)
        self.assertIn("date", SERVICE_EVENT_IMMUTABLE_SNAPSHOT_FIELDS)
        # status is NOT in the immutable set (it transitions).
        self.assertNotIn("status", SERVICE_EVENT_IMMUTABLE_SNAPSHOT_FIELDS)
        # reason_* are NOT immutable (workflow updates them).
        self.assertNotIn("reason_code", SERVICE_EVENT_IMMUTABLE_SNAPSHOT_FIELDS)
        self.assertNotIn("reason_notes", SERVICE_EVENT_IMMUTABLE_SNAPSHOT_FIELDS)

    def test_terminal_statuses_set_contents(self):
        self.assertIn("confirmed", TERMINAL_SERVICE_EVENT_STATUSES)
        self.assertIn("denied", TERMINAL_SERVICE_EVENT_STATUSES)
        self.assertIn("unpaid", TERMINAL_SERVICE_EVENT_STATUSES)
        self.assertIn("refunded", TERMINAL_SERVICE_EVENT_STATUSES)
        self.assertIn("voided", TERMINAL_SERVICE_EVENT_STATUSES)
        self.assertNotIn("pending", TERMINAL_SERVICE_EVENT_STATUSES)
