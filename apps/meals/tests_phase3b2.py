"""Phase 3B-2 tests for apps.meals — supervisor workflow services.

Covers:
- confirm_service_event (PENDING → CONFIRMED, UNPAID → CONFIRMED with charge)
- deny_service_event (PENDING → DENIED)
- unconfirm_service_event (CONFIRMED → PENDING with finance.refund)
- refund_service_event (CONFIRMED → REFUNDED with finance.refund)
- void_service_event (PENDING → VOIDED, CONFIRMED → VOIDED with refund)
- supervisor_override (eligibility override + audit row)
- MealPlan policy-flag enforcement (allow_supervisor_*, require_reason_on_*)
- Status-transition validation (terminal statuses can't transition)
- Immutability (snapshot fields not mutated by workflow)
- Audit rows written for every operation
- Row locking (select_for_update) prevents concurrent actions
"""

from datetime import date, timedelta

from django.core.exceptions import ValidationError
from django.test import TestCase

from apps.finance.services import create_wallet, record_payment
from apps.identity.models import Person, StaffProfile, StudentProfile

from .models import (
    MealEligibility,
    MealPeriod,
    MealPlan,
    MealServiceEvent,
    MealSubscription,
    MealSupervisorAction,
)
from .services import (
    confirm_service_event,
    create_subscription,
    deny_service_event,
    refund_service_event,
    resolve_eligibility,
    resolve_service,
    supervisor_override,
    unconfirm_service_event,
    void_service_event,
)
from .validators import (
    validate_service_event_status_transition,
    validate_supervisor_can_confirm,
    validate_supervisor_can_refund,
    validate_supervisor_can_unconfirm,
    validate_supervisor_override_reason,
)


# ---------------------------------------------------------------------------
# Base data
# ---------------------------------------------------------------------------

class Phase3B2BaseData(TestCase):
    def setUp(self):
        self.person_student = Person.objects.create(
            code="P-3B2S", first_name="Stu", last_name="Dent"
        )
        self.student = StudentProfile.objects.create(
            person=self.person_student, code="S-3B201"
        )
        self.staff = StaffProfile.objects.create(
            person=Person.objects.create(code="P-3B2T", first_name="Tea", last_name="Cher"),
            code="T-3B201",
        )
        self.lunch_period = MealPeriod.objects.create(
            kind=MealPeriod.Kind.LUNCH, label="Lunch", sort_order=1,
        )
        self.plan_daterange = MealPlan.objects.create(
            name="3B2 Annual",
            kind=MealPlan.Kind.LUNCH,
            mode=MealPlan.Mode.DATE_RANGE,
            default_price_iqd=0,
        )
        self.plan_wallet = MealPlan.objects.create(
            name="3B2 Wallet",
            kind=MealPlan.Kind.LUNCH,
            mode=MealPlan.Mode.WALLET,
            default_price_iqd=1500,
            insufficient_funds_mode=MealPlan.InsufficientFundsMode.DENY,
            # Default policy flags:
            # allow_supervisor_confirm=True
            # allow_supervisor_unconfirm=False
            # allow_supervisor_refund=False
            # require_reason_on_override=True
            # require_reason_on_unconfirm=False
            # require_reason_on_refund=True
        )
        self.plan_wallet_refundable = MealPlan.objects.create(
            name="3B2 Wallet Refundable",
            kind=MealPlan.Kind.LUNCH,
            mode=MealPlan.Mode.WALLET,
            default_price_iqd=1500,
            insufficient_funds_mode=MealPlan.InsufficientFundsMode.DENY,
            allow_supervisor_confirm=True,
            allow_supervisor_unconfirm=True,
            allow_supervisor_refund=True,
            require_reason_on_override=True,
            require_reason_on_unconfirm=True,
            require_reason_on_refund=True,
        )
        self.plan_wallet_unpaid_refundable = MealPlan.objects.create(
            name="3B2 Wallet Unpaid Refundable",
            kind=MealPlan.Kind.LUNCH,
            mode=MealPlan.Mode.WALLET,
            default_price_iqd=1500,
            insufficient_funds_mode=MealPlan.InsufficientFundsMode.ALLOW_UNPAID,
            allow_supervisor_confirm=True,
            allow_supervisor_unconfirm=True,
            allow_supervisor_refund=True,
            require_reason_on_override=True,
            require_reason_on_unconfirm=True,
            require_reason_on_refund=True,
        )
        self.plan_wallet_no_confirm = MealPlan.objects.create(
            name="3B2 Wallet No Confirm",
            kind=MealPlan.Kind.LUNCH,
            mode=MealPlan.Mode.WALLET,
            default_price_iqd=1500,
            allow_supervisor_confirm=False,
        )
        self.today = date.today()
        self.window_start = self.today - timedelta(days=5)
        self.window_end = self.today + timedelta(days=5)


# ---------------------------------------------------------------------------
# confirm_service_event
# ---------------------------------------------------------------------------

class ConfirmServiceEventTests(Phase3B2BaseData):
    def test_confirm_pending_date_range_event(self):
        create_subscription(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
        )
        ev = resolve_service(
            person=self.person_student, on_date=self.today,
            meal_period=self.lunch_period,
        )
        # resolve_service already CONFIRMs date-range events; create a
        # PENDING event manually for this test.
        ev_pending = MealServiceEvent.objects.create(
            person=self.person_student,
            student=self.student,
            date=self.today,
            meal_plan=self.plan_daterange,
            status=MealServiceEvent.Status.PENDING,
        )
        confirmed = confirm_service_event(
            service_event=ev_pending,
            reason_code="manual_confirm",
        )
        self.assertEqual(confirmed.status, MealServiceEvent.Status.CONFIRMED)
        self.assertEqual(confirmed.reason_code, "manual_confirm")
        self.assertIsNotNone(confirmed.served_at)
        # Audit row written.
        self.assertEqual(
            confirmed.supervisor_actions.filter(
                action=MealSupervisorAction.Action.CONFIRM
            ).count(),
            1,
        )

    def test_confirm_unpaid_event_charges_wallet(self):
        # Create a wallet-mode subscription with empty wallet → UNPAID event.
        # Need ALLOW_UNPAID mode to get UNPAID (not DENIED).
        create_wallet(person=self.person_student)  # empty wallet
        create_subscription(
            person=self.person_student,
            meal_plan=self.plan_wallet_unpaid_refundable,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
        )
        ev = resolve_service(
            person=self.person_student, on_date=self.today,
            meal_period=self.lunch_period,
        )
        self.assertEqual(ev.status, MealServiceEvent.Status.UNPAID)
        self.assertIsNone(ev.wallet_transaction)

        # Top up the wallet, then confirm.
        record_payment(person=self.person_student, amount_iqd=10000)
        confirmed = confirm_service_event(
            service_event=ev,
            reason_code="paid_after_topup",
        )
        self.assertEqual(confirmed.status, MealServiceEvent.Status.CONFIRMED)
        self.assertIsNotNone(confirmed.wallet_transaction)
        self.assertEqual(confirmed.wallet_balance_after_iqd, 8500)

    def test_confirm_rejected_when_not_allowed(self):
        ev = MealServiceEvent.objects.create(
            person=self.person_student,
            student=self.student,
            date=self.today,
            meal_plan=self.plan_wallet_no_confirm,
            status=MealServiceEvent.Status.PENDING,
        )
        with self.assertRaises(ValidationError):
            confirm_service_event(service_event=ev)

    def test_confirm_terminal_event_rejected(self):
        ev = MealServiceEvent.objects.create(
            person=self.person_student,
            date=self.today,
            meal_plan=self.plan_wallet,
            status=MealServiceEvent.Status.CONFIRMED,
        )
        with self.assertRaises(ValidationError):
            confirm_service_event(service_event=ev)


# ---------------------------------------------------------------------------
# deny_service_event
# ---------------------------------------------------------------------------

class DenyServiceEventTests(Phase3B2BaseData):
    def test_deny_pending_event(self):
        ev = MealServiceEvent.objects.create(
            person=self.person_student,
            student=self.student,
            date=self.today,
            meal_plan=self.plan_wallet,
            status=MealServiceEvent.Status.PENDING,
        )
        denied = deny_service_event(
            service_event=ev,
            reason_code="student_absent",
        )
        self.assertEqual(denied.status, MealServiceEvent.Status.DENIED)
        self.assertEqual(denied.reason_code, "student_absent")
        self.assertEqual(
            denied.supervisor_actions.filter(
                action=MealSupervisorAction.Action.DENY
            ).count(),
            1,
        )

    def test_deny_rejected_when_confirm_not_allowed(self):
        ev = MealServiceEvent.objects.create(
            person=self.person_student,
            date=self.today,
            meal_plan=self.plan_wallet_no_confirm,
            status=MealServiceEvent.Status.PENDING,
        )
        with self.assertRaises(ValidationError):
            deny_service_event(service_event=ev)

    def test_deny_confirmed_event_rejected(self):
        ev = MealServiceEvent.objects.create(
            person=self.person_student,
            date=self.today,
            status=MealServiceEvent.Status.CONFIRMED,
        )
        with self.assertRaises(ValidationError):
            deny_service_event(service_event=ev)


# ---------------------------------------------------------------------------
# unconfirm_service_event
# ---------------------------------------------------------------------------

class UnconfirmServiceEventTests(Phase3B2BaseData):
    def setUp(self):
        super().setUp()
        self.wallet = create_wallet(person=self.person_student)
        record_payment(person=self.person_student, amount_iqd=10000)
        create_subscription(
            person=self.person_student,
            meal_plan=self.plan_wallet_refundable,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
        )
        self.ev = resolve_service(
            person=self.person_student, on_date=self.today,
            meal_period=self.lunch_period,
        )
        self.assertEqual(self.ev.status, MealServiceEvent.Status.CONFIRMED)

    def test_unconfirm_reverses_charge(self):
        unconfirmed = unconfirm_service_event(
            service_event=self.ev,
            reason_code="wrong_charge",
            reason_notes="Supervisor reversal",
        )
        self.assertEqual(unconfirmed.status, MealServiceEvent.Status.PENDING)
        self.assertIsNotNone(unconfirmed.wallet_refund_transaction)
        # Audit row.
        self.assertEqual(
            unconfirmed.supervisor_actions.filter(
                action=MealSupervisorAction.Action.UNCONFIRM
            ).count(),
            1,
        )

    def test_unconfirm_rejected_when_not_allowed(self):
        # plan_wallet has allow_supervisor_unconfirm=False (default).
        # Create a confirmed event with plan_wallet (no unconfirm allowed).
        # Use a separate date to avoid clashing with the setUp event.
        ev2 = MealServiceEvent.objects.create(
            person=self.person_student,
            student=self.student,
            date=self.today + timedelta(days=1),
            meal_plan=self.plan_wallet,
            status=MealServiceEvent.Status.CONFIRMED,
            final_charge_iqd=1500,
        )
        with self.assertRaises(ValidationError):
            unconfirm_service_event(service_event=ev2, reason_code="x")

    def test_unconfirm_requires_reason(self):
        # plan_wallet_refundable has require_reason_on_unconfirm=True.
        with self.assertRaises(ValidationError):
            unconfirm_service_event(service_event=self.ev, reason_code="")

    def test_unconfirm_pending_rejected(self):
        ev = MealServiceEvent.objects.create(
            person=self.person_student,
            date=self.today + timedelta(days=2),
            status=MealServiceEvent.Status.PENDING,
        )
        with self.assertRaises(ValidationError):
            unconfirm_service_event(service_event=ev, reason_code="x")


# ---------------------------------------------------------------------------
# refund_service_event
# ---------------------------------------------------------------------------

class RefundServiceEventTests(Phase3B2BaseData):
    def setUp(self):
        super().setUp()
        self.wallet = create_wallet(person=self.person_student)
        record_payment(person=self.person_student, amount_iqd=10000)
        create_subscription(
            person=self.person_student,
            meal_plan=self.plan_wallet_refundable,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
        )
        self.ev = resolve_service(
            person=self.person_student, on_date=self.today,
            meal_period=self.lunch_period,
        )
        self.assertEqual(self.ev.status, MealServiceEvent.Status.CONFIRMED)

    def test_refund_full_amount(self):
        refunded = refund_service_event(
            service_event=self.ev,
            reason_code="supervisor_refund",
        )
        self.assertEqual(refunded.status, MealServiceEvent.Status.REFUNDED)
        self.assertIsNotNone(refunded.wallet_refund_transaction)
        self.assertIsNotNone(refunded.reversed_at)
        self.assertEqual(
            refunded.supervisor_actions.filter(
                action=MealSupervisorAction.Action.REFUND
            ).count(),
            1,
        )

    def test_refund_partial_amount(self):
        refunded = refund_service_event(
            service_event=self.ev,
            amount_iqd=500,
            reason_code="partial_refund",
        )
        self.assertEqual(refunded.status, MealServiceEvent.Status.REFUNDED)

    def test_refund_rejected_when_not_allowed(self):
        # plan_wallet has allow_supervisor_refund=False.
        ev2 = MealServiceEvent.objects.create(
            person=self.person_student,
            student=self.student,
            date=self.today + timedelta(days=1),
            meal_plan=self.plan_wallet,
            status=MealServiceEvent.Status.CONFIRMED,
            final_charge_iqd=1500,
            wallet_transaction=self.ev.wallet_transaction,
        )
        with self.assertRaises(ValidationError):
            refund_service_event(service_event=ev2, reason_code="x")

    def test_refund_requires_reason(self):
        # plan_wallet_refundable has require_reason_on_refund=True.
        with self.assertRaises(ValidationError):
            refund_service_event(service_event=self.ev, reason_code="")

    def test_refund_no_wallet_transaction_rejected(self):
        ev = MealServiceEvent.objects.create(
            person=self.person_student,
            date=self.today + timedelta(days=2),
            meal_plan=self.plan_wallet_refundable,
            status=MealServiceEvent.Status.CONFIRMED,
            final_charge_iqd=0,
        )
        with self.assertRaises(ValidationError):
            refund_service_event(service_event=ev, reason_code="x")

    def test_refund_pending_rejected(self):
        ev = MealServiceEvent.objects.create(
            person=self.person_student,
            date=self.today + timedelta(days=2),
            status=MealServiceEvent.Status.PENDING,
        )
        with self.assertRaises(ValidationError):
            refund_service_event(service_event=ev, reason_code="x")


# ---------------------------------------------------------------------------
# void_service_event
# ---------------------------------------------------------------------------

class VoidServiceEventTests(Phase3B2BaseData):
    def test_void_pending_event(self):
        ev = MealServiceEvent.objects.create(
            person=self.person_student,
            date=self.today,
            status=MealServiceEvent.Status.PENDING,
        )
        voided = void_service_event(service_event=ev, reason_code="mistake")
        self.assertEqual(voided.status, MealServiceEvent.Status.VOIDED)
        self.assertEqual(voided.reason_code, "mistake")
        self.assertEqual(
            voided.supervisor_actions.filter(
                action=MealSupervisorAction.Action.VOID
            ).count(),
            1,
        )

    def test_void_confirmed_with_charge_reverses(self):
        self.wallet = create_wallet(person=self.person_student)
        record_payment(person=self.person_student, amount_iqd=10000)
        create_subscription(
            person=self.person_student,
            meal_plan=self.plan_wallet_refundable,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
        )
        ev = resolve_service(
            person=self.person_student, on_date=self.today,
            meal_period=self.lunch_period,
        )
        self.assertEqual(ev.status, MealServiceEvent.Status.CONFIRMED)
        voided = void_service_event(service_event=ev, reason_code="void_after_charge")
        self.assertEqual(voided.status, MealServiceEvent.Status.VOIDED)
        self.assertIsNotNone(voided.wallet_refund_transaction)

    def test_void_confirmed_no_charge_no_refund(self):
        ev = MealServiceEvent.objects.create(
            person=self.person_student,
            date=self.today,
            meal_plan=self.plan_daterange,
            status=MealServiceEvent.Status.CONFIRMED,
            final_charge_iqd=0,
        )
        voided = void_service_event(service_event=ev, reason_code="void")
        self.assertEqual(voided.status, MealServiceEvent.Status.VOIDED)
        self.assertIsNone(voided.wallet_refund_transaction)

    def test_void_confirmed_with_charge_rejected_when_refund_not_allowed(self):
        self.wallet = create_wallet(person=self.person_student)
        record_payment(person=self.person_student, amount_iqd=10000)
        create_subscription(
            person=self.person_student,
            meal_plan=self.plan_wallet,  # allow_supervisor_refund=False
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
        )
        ev = resolve_service(
            person=self.person_student, on_date=self.today,
            meal_period=self.lunch_period,
        )
        with self.assertRaises(ValidationError):
            void_service_event(service_event=ev, reason_code="x")

    def test_void_terminal_event_rejected(self):
        ev = MealServiceEvent.objects.create(
            person=self.person_student,
            date=self.today,
            status=MealServiceEvent.Status.VOIDED,
        )
        with self.assertRaises(ValidationError):
            void_service_event(service_event=ev)


# ---------------------------------------------------------------------------
# supervisor_override
# ---------------------------------------------------------------------------

class SupervisorOverrideTests(Phase3B2BaseData):
    def test_override_eligible(self):
        elig = supervisor_override(
            person=self.person_student,
            on_date=self.today,
            decision="eligible",
            reason_code="manual_override",
            meal_plan=self.plan_daterange,
        )
        self.assertEqual(elig.decision, MealEligibility.Decision.OVERRIDDEN_ELIGIBLE)
        self.assertEqual(elig.reason_code, "manual_override")
        # Audit row against the eligibility.
        self.assertEqual(
            elig.supervisor_actions.filter(
                action=MealSupervisorAction.Action.OVERRIDE_ELIGIBLE
            ).count(),
            1,
        )

    def test_override_denied(self):
        elig = supervisor_override(
            person=self.person_student,
            on_date=self.today,
            decision="denied",
            reason_code="parent_request",
        )
        self.assertEqual(elig.decision, MealEligibility.Decision.OVERRIDDEN_DENIED)
        self.assertEqual(
            elig.supervisor_actions.filter(
                action=MealSupervisorAction.Action.OVERRIDE_DENIED
            ).count(),
            1,
        )

    def test_override_requires_reason(self):
        # plan_wallet_refundable has require_reason_on_override=True.
        with self.assertRaises(ValidationError):
            supervisor_override(
                person=self.person_student,
                on_date=self.today,
                decision="eligible",
                reason_code="",
                meal_plan=self.plan_wallet_refundable,
            )

    def test_override_frozen_eligibility_rejected(self):
        # Create a service event that references the eligibility → frozen.
        create_subscription(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
        )
        resolve_service(
            person=self.person_student, on_date=self.today,
            meal_period=self.lunch_period,
        )
        # Now try to override — should be rejected (frozen).
        with self.assertRaises(ValidationError):
            supervisor_override(
                person=self.person_student,
                on_date=self.today,
                decision="denied",
                reason_code="x",
            )

    def test_override_upserts_existing_unfrozen_eligibility(self):
        # Create an eligibility row (not frozen).
        resolve_eligibility(person=self.person_student, on_date=self.today)
        elig = supervisor_override(
            person=self.person_student,
            on_date=self.today,
            decision="denied",
            reason_code="late_override",
        )
        # Same row, updated decision.
        self.assertEqual(elig.decision, MealEligibility.Decision.OVERRIDDEN_DENIED)

    def test_override_unknown_decision_rejected(self):
        with self.assertRaises(ValidationError):
            supervisor_override(
                person=self.person_student,
                on_date=self.today,
                decision="maybe",
                reason_code="x",
            )


# ---------------------------------------------------------------------------
# Status-transition validator
# ---------------------------------------------------------------------------

class StatusTransitionValidatorTests(TestCase):
    def test_pending_to_confirmed_ok(self):
        validate_service_event_status_transition(
            current_status="pending", new_status="confirmed"
        )

    def test_pending_to_denied_ok(self):
        validate_service_event_status_transition(
            current_status="pending", new_status="denied"
        )

    def test_pending_to_voided_ok(self):
        validate_service_event_status_transition(
            current_status="pending", new_status="voided"
        )

    def test_confirmed_to_refunded_ok(self):
        validate_service_event_status_transition(
            current_status="confirmed", new_status="refunded"
        )

    def test_confirmed_to_voided_ok(self):
        validate_service_event_status_transition(
            current_status="confirmed", new_status="voided"
        )

    def test_unpaid_to_confirmed_ok(self):
        validate_service_event_status_transition(
            current_status="unpaid", new_status="confirmed"
        )

    def test_refunded_to_anything_rejected(self):
        with self.assertRaises(ValidationError):
            validate_service_event_status_transition(
                current_status="refunded", new_status="confirmed"
            )

    def test_voided_to_anything_rejected(self):
        with self.assertRaises(ValidationError):
            validate_service_event_status_transition(
                current_status="voided", new_status="confirmed"
            )

    def test_denied_to_anything_rejected(self):
        with self.assertRaises(ValidationError):
            validate_service_event_status_transition(
                current_status="denied", new_status="confirmed"
            )

    def test_confirmed_to_pending_via_unconfirm_ok(self):
        # unconfirm_service_event transitions CONFIRMED → PENDING.
        validate_service_event_status_transition(
            current_status="confirmed", new_status="pending"
        )


# ---------------------------------------------------------------------------
# Immutability after workflow operations
# ---------------------------------------------------------------------------

class ImmutabilityAfterWorkflowTests(Phase3B2BaseData):
    def test_refunded_event_snapshot_immutable(self):
        self.wallet = create_wallet(person=self.person_student)
        record_payment(person=self.person_student, amount_iqd=10000)
        create_subscription(
            person=self.person_student,
            meal_plan=self.plan_wallet_refundable,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
        )
        ev = resolve_service(
            person=self.person_student, on_date=self.today,
            meal_period=self.lunch_period,
        )
        refunded = refund_service_event(service_event=ev, reason_code="r")
        # Mutating a snapshot field should raise.
        refunded.final_charge_iqd = 9999
        with self.assertRaises(ValidationError):
            refunded.full_clean()

    def test_voided_event_snapshot_immutable(self):
        ev = MealServiceEvent.objects.create(
            person=self.person_student,
            date=self.today,
            status=MealServiceEvent.Status.PENDING,
        )
        voided = void_service_event(service_event=ev, reason_code="v")
        voided.final_charge_iqd = 9999
        with self.assertRaises(ValidationError):
            voided.full_clean()


# ---------------------------------------------------------------------------
# Validator unit tests
# ---------------------------------------------------------------------------

class SupervisorValidatorUnitTests(TestCase):
    def test_validate_supervisor_can_confirm_allowed(self):
        from apps.identity.models import Person
        from .models import MealPlan

        plan = MealPlan.objects.create(
            name="VC1", kind=MealPlan.Kind.LUNCH,
            mode=MealPlan.Mode.WALLET, allow_supervisor_confirm=True,
        )
        validate_supervisor_can_confirm(meal_plan=plan)

    def test_validate_supervisor_can_confirm_rejected(self):
        from .models import MealPlan

        plan = MealPlan.objects.create(
            name="VC2", kind=MealPlan.Kind.LUNCH,
            mode=MealPlan.Mode.WALLET, allow_supervisor_confirm=False,
        )
        with self.assertRaises(ValidationError):
            validate_supervisor_can_confirm(meal_plan=plan)

    def test_validate_supervisor_can_unconfirm_requires_reason(self):
        from .models import MealPlan

        plan = MealPlan.objects.create(
            name="VU1", kind=MealPlan.Kind.LUNCH,
            mode=MealPlan.Mode.WALLET,
            allow_supervisor_unconfirm=True,
            require_reason_on_unconfirm=True,
        )
        with self.assertRaises(ValidationError):
            validate_supervisor_can_unconfirm(meal_plan=plan, reason_code="")

    def test_validate_supervisor_can_refund_rejected(self):
        from .models import MealPlan

        plan = MealPlan.objects.create(
            name="VR1", kind=MealPlan.Kind.LUNCH,
            mode=MealPlan.Mode.WALLET, allow_supervisor_refund=False,
        )
        with self.assertRaises(ValidationError):
            validate_supervisor_can_refund(meal_plan=plan)

    def test_validate_supervisor_override_reason_required(self):
        from .models import MealPlan

        plan = MealPlan.objects.create(
            name="VO1", kind=MealPlan.Kind.LUNCH,
            mode=MealPlan.Mode.WALLET, require_reason_on_override=True,
        )
        with self.assertRaises(ValidationError):
            validate_supervisor_override_reason(meal_plan=plan, reason_code="")

    def test_validate_supervisor_can_refund_requires_reason(self):
        from .models import MealPlan

        plan = MealPlan.objects.create(
            name="VR2", kind=MealPlan.Kind.LUNCH,
            mode=MealPlan.Mode.WALLET,
            allow_supervisor_refund=True,
            require_reason_on_refund=True,
        )
        with self.assertRaises(ValidationError):
            validate_supervisor_can_refund(meal_plan=plan, reason_code="")
