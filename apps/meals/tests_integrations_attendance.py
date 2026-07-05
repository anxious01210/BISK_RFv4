"""Tests for the Attendance-to-Meals integration bridge.

Covers:
- resolve_person_from_student (happy path, no profile, None input)
- resolve_person_from_h_code (happy path, unknown h_code, empty)
- bridge_recognition_to_service_event (DATE_RANGE → CONFIRMED,
  WALLET → CONFIRMED with finance transaction, no Person → None)
- MealServiceEvent.recognition_event FK link
- No legacy mutation (MealRecord / Wallet / WalletTransaction)
- Idempotency / duplicate behavior (eligibility freeze)
"""

from datetime import date, timedelta

from django.core.exceptions import ValidationError
from django.test import TestCase
from django.utils import timezone

from apps.attendance.models import (
    AttendanceEvent,
    AttendanceRecord,
    PeriodOccurrence,
    Student as LegacyStudent,
)
from apps.finance.services import create_wallet, record_payment
from apps.identity.models import Person, StaffProfile, StudentProfile

from .integrations.attendance import (
    bridge_recognition_to_service_event,
    resolve_person_from_h_code,
    resolve_person_from_student,
)
from .models import (
    MealPeriod,
    MealPlan,
    MealServiceEvent,
    MealSubscription,
)
from .services import create_subscription


# ---------------------------------------------------------------------------
# Base data
# ---------------------------------------------------------------------------

class BridgeBaseData(TestCase):
    def setUp(self):
        # Create identity Person + StudentProfile (new domain).
        self.person = Person.objects.create(
            code="P-BR1", first_name="Bridge", last_name="Tester"
        )
        self.student_profile = StudentProfile.objects.create(
            person=self.person, code="S-BR01"
        )

        # Create legacy Student and link via StudentProfile.legacy_student.
        self.legacy_student = LegacyStudent.objects.create(
            h_code="H-BR01",
            first_name="Bridge",
            last_name="Tester",
        )
        self.student_profile.legacy_student = self.legacy_student
        self.student_profile.save()

        # Create a MealPeriod and MealPlans.
        self.lunch_period = MealPeriod.objects.create(
            kind=MealPeriod.Kind.LUNCH, label="Lunch", sort_order=1,
        )
        self.plan_daterange = MealPlan.objects.create(
            name="Bridge Annual",
            kind=MealPlan.Kind.LUNCH,
            mode=MealPlan.Mode.DATE_RANGE,
            default_price_iqd=0,
        )
        self.plan_wallet = MealPlan.objects.create(
            name="Bridge Wallet",
            kind=MealPlan.Kind.LUNCH,
            mode=MealPlan.Mode.WALLET,
            default_price_iqd=1500,
            insufficient_funds_mode=MealPlan.InsufficientFundsMode.DENY,
        )

        self.today = date.today()
        self.window_start = self.today - timedelta(days=5)
        self.window_end = self.today + timedelta(days=5)
        self.now = timezone.now()

        # Create a PeriodOccurrence for the AttendanceRecord.
        from apps.attendance.models import PeriodTemplate
        self.period_template = PeriodTemplate.objects.create(
            name="Lunch Block", order=1,
            start_time="12:00", end_time="13:00",
        )
        self.period_occurrence = PeriodOccurrence.objects.create(
            template=self.period_template,
            date=self.today,
            start_dt=timezone.make_aware(timezone.datetime.combine(self.today, timezone.datetime.min.time())),
            end_dt=timezone.make_aware(timezone.datetime.combine(self.today, timezone.datetime.max.time())),
        )

    def _create_attendance_event(self, student=None, ts=None):
        """Create a legacy AttendanceEvent for the given (or default)
        legacy Student."""
        if student is None:
            student = self.legacy_student
        if ts is None:
            ts = self.now
        return AttendanceEvent.objects.create(
            student=student,
            period=self.period_occurrence,
            camera=None,
            ts=ts,
            score=0.92,
            crop_path="",
        )


# ---------------------------------------------------------------------------
# resolve_person_from_student
# ---------------------------------------------------------------------------

class ResolvePersonFromStudentTests(BridgeBaseData):
    def test_happy_path(self):
        person = resolve_person_from_student(self.legacy_student)
        self.assertIsNotNone(person)
        self.assertEqual(person, self.person)

    def test_returns_none_if_no_student_profile(self):
        # Create a legacy Student without a StudentProfile link.
        unlinked = LegacyStudent.objects.create(
            h_code="H-UNLINKED", first_name="Un", last_name="Linked"
        )
        person = resolve_person_from_student(unlinked)
        self.assertIsNone(person)

    def test_returns_none_for_none_input(self):
        self.assertIsNone(resolve_person_from_student(None))


# ---------------------------------------------------------------------------
# resolve_person_from_h_code
# ---------------------------------------------------------------------------

class ResolvePersonFromHCodeTests(BridgeBaseData):
    def test_happy_path(self):
        person = resolve_person_from_h_code("H-BR01")
        self.assertIsNotNone(person)
        self.assertEqual(person, self.person)

    def test_unknown_h_code_returns_none(self):
        self.assertIsNone(resolve_person_from_h_code("H-UNKNOWN"))

    def test_empty_h_code_returns_none(self):
        self.assertIsNone(resolve_person_from_h_code(""))
        self.assertIsNone(resolve_person_from_h_code(None))

    def test_unlinked_student_returns_none(self):
        LegacyStudent.objects.create(
            h_code="H-UNLINKED2", first_name="Un2", last_name="Linked2"
        )
        self.assertIsNone(resolve_person_from_h_code("H-UNLINKED2"))


# ---------------------------------------------------------------------------
# bridge_recognition_to_service_event
# ---------------------------------------------------------------------------

class BridgeDateRangeTests(BridgeBaseData):
    def setUp(self):
        super().setUp()
        create_subscription(
            person=self.person,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student_profile,
            priority=1,
        )

    def test_date_range_creates_confirmed_event(self):
        event = self._create_attendance_event()
        service_event = bridge_recognition_to_service_event(
            attendance_event=event,
            meal_period=self.lunch_period,
        )
        self.assertIsNotNone(service_event)
        self.assertEqual(service_event.status, MealServiceEvent.Status.CONFIRMED)
        self.assertEqual(service_event.final_charge_iqd, 0)

    def test_recognition_event_fk_linked(self):
        event = self._create_attendance_event()
        service_event = bridge_recognition_to_service_event(
            attendance_event=event,
            meal_period=self.lunch_period,
        )
        self.assertEqual(service_event.recognition_event, event)

    def test_on_date_derived_from_event_ts(self):
        # Event on a different date.
        other_day = self.today - timedelta(days=2)
        ts = timezone.make_aware(
            timezone.datetime.combine(other_day, timezone.datetime.min.time())
        )
        event = self._create_attendance_event(ts=ts)
        # Extend subscription window to cover other_day.
        create_subscription(
            person=self.person,
            meal_plan=self.plan_daterange,
            start_date=self.window_start - timedelta(days=10),
            end_date=self.window_end,
            student=self.student_profile,
            priority=2,
        )
        service_event = bridge_recognition_to_service_event(
            attendance_event=event,
            meal_period=self.lunch_period,
        )
        self.assertIsNotNone(service_event)
        self.assertEqual(service_event.date, other_day)

    def test_explicit_on_date_overrides_ts(self):
        event = self._create_attendance_event()
        service_event = bridge_recognition_to_service_event(
            attendance_event=event,
            meal_period=self.lunch_period,
            on_date=self.today,
        )
        self.assertEqual(service_event.date, self.today)


class BridgeWalletTests(BridgeBaseData):
    def setUp(self):
        super().setUp()
        self.wallet = create_wallet(person=self.person)
        record_payment(person=self.person, amount_iqd=10000)
        create_subscription(
            person=self.person,
            meal_plan=self.plan_wallet,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student_profile,
            priority=1,
        )

    def test_wallet_creates_confirmed_with_transaction(self):
        event = self._create_attendance_event()
        service_event = bridge_recognition_to_service_event(
            attendance_event=event,
            meal_period=self.lunch_period,
        )
        self.assertIsNotNone(service_event)
        self.assertEqual(service_event.status, MealServiceEvent.Status.CONFIRMED)
        self.assertEqual(service_event.final_charge_iqd, 1500)
        self.assertIsNotNone(service_event.wallet_transaction)
        self.assertEqual(service_event.wallet_balance_before_iqd, 10000)
        self.assertEqual(service_event.wallet_balance_after_iqd, 8500)

    def test_recognition_event_fk_linked(self):
        event = self._create_attendance_event()
        service_event = bridge_recognition_to_service_event(
            attendance_event=event,
            meal_period=self.lunch_period,
        )
        self.assertEqual(service_event.recognition_event, event)


class BridgeNoPersonTests(BridgeBaseData):
    def test_returns_none_if_no_person(self):
        # Create a legacy Student without a StudentProfile link.
        unlinked = LegacyStudent.objects.create(
            h_code="H-NOBRIDGE", first_name="No", last_name="Bridge"
        )
        event = self._create_attendance_event(student=unlinked)
        result = bridge_recognition_to_service_event(
            attendance_event=event,
            meal_period=self.lunch_period,
        )
        self.assertIsNone(result)

    def test_returns_none_for_none_event(self):
        self.assertIsNone(
            bridge_recognition_to_service_event(attendance_event=None)
        )


# ---------------------------------------------------------------------------
# No legacy mutation
# ---------------------------------------------------------------------------

class NoLegacyMutationTests(BridgeBaseData):
    def setUp(self):
        super().setUp()
        create_subscription(
            person=self.person,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student_profile,
            priority=1,
        )

    def test_no_legacy_meal_record_created(self):
        from apps.attendance.models import MealRecord
        event = self._create_attendance_event()
        before = MealRecord.objects.count()
        bridge_recognition_to_service_event(
            attendance_event=event,
            meal_period=self.lunch_period,
        )
        self.assertEqual(MealRecord.objects.count(), before)

    def test_no_legacy_wallet_transaction_created(self):
        from apps.attendance.models import WalletTransaction as LegacyWT
        event = self._create_attendance_event()
        before = LegacyWT.objects.count()
        bridge_recognition_to_service_event(
            attendance_event=event,
            meal_period=self.lunch_period,
        )
        self.assertEqual(LegacyWT.objects.count(), before)

    def test_no_legacy_wallet_mutated(self):
        from apps.attendance.models import Wallet as LegacyWallet
        # Create a legacy wallet for the student.
        LegacyWallet.objects.create(student=self.legacy_student, balance_iqd=5000)
        event = self._create_attendance_event()
        bridge_recognition_to_service_event(
            attendance_event=event,
            meal_period=self.lunch_period,
        )
        # Legacy wallet balance unchanged.
        lw = LegacyWallet.objects.get(student=self.legacy_student)
        self.assertEqual(lw.balance_iqd, 5000)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class IdempotencyTests(BridgeBaseData):
    def setUp(self):
        super().setUp()
        create_subscription(
            person=self.person,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student_profile,
            priority=1,
        )

    def test_second_call_raises_validation_error(self):
        event = self._create_attendance_event()
        # First call creates a CONFIRMED event + freezes eligibility.
        ev1 = bridge_recognition_to_service_event(
            attendance_event=event,
            meal_period=self.lunch_period,
        )
        self.assertEqual(ev1.status, MealServiceEvent.Status.CONFIRMED)

        # Second call for the same (person, date) raises because the
        # eligibility is now frozen (referenced by ev1).
        with self.assertRaises(ValidationError):
            bridge_recognition_to_service_event(
                attendance_event=event,
                meal_period=self.lunch_period,
            )

        # The original event is untouched.
        ev1.refresh_from_db()
        self.assertEqual(ev1.status, MealServiceEvent.Status.CONFIRMED)

    def test_only_one_service_event_created(self):
        event = self._create_attendance_event()
        bridge_recognition_to_service_event(
            attendance_event=event,
            meal_period=self.lunch_period,
        )
        # A second call that raises still should not create a duplicate.
        try:
            bridge_recognition_to_service_event(
                attendance_event=event,
                meal_period=self.lunch_period,
            )
        except ValidationError:
            pass
        self.assertEqual(
            MealServiceEvent.objects.filter(person=self.person, date=self.today).count(),
            1,
        )
