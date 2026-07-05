"""Phase 2 tests for apps.meals — subscriptions, exceptions, eligibility.

These tests are split into a separate module (``tests_phase2.py``) to
keep the Phase 1 ``tests.py`` file reviewable. The default Django test
runner discovers ``test*.py`` so both modules run under
``python manage.py test apps.meals``.
"""

from datetime import date, timedelta

from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.test import TestCase

from apps.identity.models import Person, StaffProfile, StudentProfile

from .models import (
    MealEligibility,
    MealException,
    MealPlan,
    MealSubscription,
)
from .selectors import (
    active_subscriptions_for,
    eligibility_for,
    exceptions_for,
    list_eligibilities,
    list_exceptions,
    list_subscriptions,
)
from .services import (
    REASON_DATE_RANGE_NO_CHARGE,
    REASON_GUEST_ELIGIBLE,
    REASON_NO_SUBSCRIPTION,
    REASON_ONE_TIME_ELIGIBLE,
    REASON_TEMPORARY_DENY,
    cancel_subscription,
    create_subscription,
    grant_one_time_permission,
    pause_subscription,
    resolve_eligibility,
    resume_subscription,
)
from .validators import (
    validate_exception_window,
    validate_subscription_dates,
    validate_subscription_overlap,
    validate_subscription_role,
    validate_subscription_status_transition,
)


# ---------------------------------------------------------------------------
# Test base data
# ---------------------------------------------------------------------------

class Phase2BaseData(TestCase):
    def setUp(self):
        self.person_student = Person.objects.create(
            code="P-S2", first_name="Stu", last_name="Dent"
        )
        self.student = StudentProfile.objects.create(
            person=self.person_student, code="S-2001"
        )
        self.person_staff = Person.objects.create(
            code="P-T2", first_name="Tea", last_name="Cher"
        )
        self.staff = StaffProfile.objects.create(
            person=self.person_staff, code="T-2001"
        )
        self.plan_daterange = MealPlan.objects.create(
            name="Phase2 Lunch Annual",
            kind=MealPlan.Kind.LUNCH,
            mode=MealPlan.Mode.DATE_RANGE,
            default_price_iqd=0,
        )
        self.plan_wallet = MealPlan.objects.create(
            name="Phase2 Lunch Wallet",
            kind=MealPlan.Kind.LUNCH,
            mode=MealPlan.Mode.WALLET,
            default_price_iqd=1000,
        )
        self.today = date.today()
        self.window_start = self.today - timedelta(days=10)
        self.window_end = self.today + timedelta(days=10)


# ---------------------------------------------------------------------------
# MealSubscription model + constraints
# ---------------------------------------------------------------------------

class MealSubscriptionModelTests(Phase2BaseData):
    def test_create_subscription_via_orm(self):
        sub = MealSubscription.objects.create(
            person=self.person_student,
            student=self.student,
            meal_plan=self.plan_daterange,
            status=MealSubscription.Status.ACTIVE,
            start_date=self.window_start,
            end_date=self.window_end,
            priority=1,
        )
        self.assertEqual(sub.status, MealSubscription.Status.ACTIVE)
        self.assertEqual(sub.priority, 1)
        self.assertEqual(sub.student, self.student)

    def test_str_includes_priority_and_window(self):
        sub = MealSubscription.objects.create(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            priority=1,
        )
        s = str(sub)
        self.assertIn("Phase2 Lunch Annual", s)
        self.assertIn("p1", s)

    def test_end_before_start_rejected_at_db(self):
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                MealSubscription.objects.create(
                    person=self.person_student,
                    start_date=self.window_end,
                    end_date=self.window_start,
                )

    def test_priority_zero_rejected_at_db(self):
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                MealSubscription.objects.create(
                    person=self.person_student,
                    start_date=self.window_start,
                    end_date=self.window_end,
                    priority=0,
                )

    def test_clean_rejects_both_student_and_staff(self):
        sub = MealSubscription(
            person=self.person_student,
            student=self.student,
            staff=self.staff,
            start_date=self.window_start,
            end_date=self.window_end,
        )
        with self.assertRaises(ValidationError):
            sub.full_clean()

    def test_clean_rejects_invalid_dates(self):
        sub = MealSubscription(
            person=self.person_student,
            start_date=self.window_end,
            end_date=self.window_start,
        )
        with self.assertRaises(ValidationError):
            sub.full_clean()

    def test_clean_accepts_guest_subscription_no_student_no_staff(self):
        sub = MealSubscription(
            person=self.person_student,
            start_date=self.window_start,
            end_date=self.window_end,
        )
        sub.full_clean()  # should not raise


# ---------------------------------------------------------------------------
# Subscription validators
# ---------------------------------------------------------------------------

class SubscriptionValidatorTests(Phase2BaseData):
    def test_validate_subscription_dates_invalid(self):
        with self.assertRaises(ValidationError):
            validate_subscription_dates(
                start_date=self.today, end_date=self.today - timedelta(days=1)
            )

    def test_validate_subscription_dates_missing(self):
        with self.assertRaises(ValidationError):
            validate_subscription_dates(start_date=None, end_date=self.today)

    def test_validate_subscription_role_both_set_rejected(self):
        with self.assertRaises(ValidationError):
            validate_subscription_role(student=self.student, staff=self.staff)

    def test_validate_subscription_role_one_set_ok(self):
        validate_subscription_role(student=self.student, staff=None)
        validate_subscription_role(student=None, staff=self.staff)
        validate_subscription_role(student=None, staff=None)

    def test_validate_subscription_overlap_blocks_same_priority_active(self):
        MealSubscription.objects.create(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            status=MealSubscription.Status.ACTIVE,
            start_date=self.window_start,
            end_date=self.window_end,
            priority=1,
        )
        with self.assertRaises(ValidationError):
            validate_subscription_overlap(
                person=self.person_student,
                priority=1,
                start_date=self.window_start,
                end_date=self.window_end,
                status=MealSubscription.Status.ACTIVE,
            )

    def test_validate_subscription_overlap_allows_different_priority(self):
        MealSubscription.objects.create(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            status=MealSubscription.Status.ACTIVE,
            start_date=self.window_start,
            end_date=self.window_end,
            priority=1,
        )
        validate_subscription_overlap(
            person=self.person_student,
            priority=2,
            start_date=self.window_start,
            end_date=self.window_end,
            status=MealSubscription.Status.ACTIVE,
        )

    def test_validate_subscription_overlap_ignores_cancelled(self):
        MealSubscription.objects.create(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            status=MealSubscription.Status.CANCELLED,
            start_date=self.window_start,
            end_date=self.window_end,
            priority=1,
        )
        validate_subscription_overlap(
            person=self.person_student,
            priority=1,
            start_date=self.window_start,
            end_date=self.window_end,
            status=MealSubscription.Status.ACTIVE,
        )

    def test_validate_subscription_overlap_excludes_self(self):
        sub = MealSubscription.objects.create(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            status=MealSubscription.Status.ACTIVE,
            start_date=self.window_start,
            end_date=self.window_end,
            priority=1,
        )
        validate_subscription_overlap(
            person=self.person_student,
            priority=1,
            start_date=self.window_start,
            end_date=self.window_end,
            status=MealSubscription.Status.ACTIVE,
            exclude_pk=sub.pk,
        )

    def test_status_transition_active_to_paused_ok(self):
        validate_subscription_status_transition(
            current_status=MealSubscription.Status.ACTIVE,
            new_status=MealSubscription.Status.PAUSED,
        )

    def test_status_transition_paused_to_active_ok(self):
        validate_subscription_status_transition(
            current_status=MealSubscription.Status.PAUSED,
            new_status=MealSubscription.Status.ACTIVE,
        )

    def test_status_transition_active_to_cancelled_ok(self):
        validate_subscription_status_transition(
            current_status=MealSubscription.Status.ACTIVE,
            new_status=MealSubscription.Status.CANCELLED,
        )

    def test_status_transition_cancelled_to_active_invalid(self):
        with self.assertRaises(ValidationError):
            validate_subscription_status_transition(
                current_status=MealSubscription.Status.CANCELLED,
                new_status=MealSubscription.Status.ACTIVE,
            )

    def test_status_transition_expired_to_anything_invalid(self):
        with self.assertRaises(ValidationError):
            validate_subscription_status_transition(
                current_status=MealSubscription.Status.EXPIRED,
                new_status=MealSubscription.Status.ACTIVE,
            )

    def test_status_transition_active_to_active_invalid(self):
        with self.assertRaises(ValidationError):
            validate_subscription_status_transition(
                current_status=MealSubscription.Status.ACTIVE,
                new_status=MealSubscription.Status.ACTIVE,
            )


# ---------------------------------------------------------------------------
# Subscription services
# ---------------------------------------------------------------------------

class SubscriptionServiceTests(Phase2BaseData):
    def test_create_subscription_defaults_active_when_in_window(self):
        sub = create_subscription(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
        )
        self.assertEqual(sub.status, MealSubscription.Status.ACTIVE)

    def test_create_subscription_defaults_future_when_before_start(self):
        future_start = self.today + timedelta(days=5)
        future_end = self.today + timedelta(days=20)
        sub = create_subscription(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=future_start,
            end_date=future_end,
            student=self.student,
            priority=1,
        )
        self.assertEqual(sub.status, MealSubscription.Status.FUTURE)

    def test_create_subscription_explicit_status_overrides_default(self):
        sub = create_subscription(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
            status=MealSubscription.Status.PAUSED,
        )
        self.assertEqual(sub.status, MealSubscription.Status.PAUSED)

    def test_create_subscription_rejects_overlap_same_priority(self):
        create_subscription(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
        )
        with self.assertRaises(ValidationError):
            create_subscription(
                person=self.person_student,
                meal_plan=self.plan_daterange,
                start_date=self.window_start,
                end_date=self.window_end,
                student=self.student,
                priority=1,
            )

    def test_create_subscription_allows_primary_plus_fallback(self):
        primary = create_subscription(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
        )
        fallback = create_subscription(
            person=self.person_student,
            meal_plan=self.plan_wallet,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=2,
        )
        self.assertEqual(primary.priority, 1)
        self.assertEqual(fallback.priority, 2)

    def test_create_subscription_rejects_both_student_and_staff(self):
        with self.assertRaises(ValidationError):
            create_subscription(
                person=self.person_student,
                start_date=self.window_start,
                end_date=self.window_end,
                student=self.student,
                staff=self.staff,
            )

    def test_pause_subscription(self):
        sub = create_subscription(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
        )
        paused = pause_subscription(subscription=sub)
        self.assertEqual(paused.status, MealSubscription.Status.PAUSED)

    def test_resume_subscription(self):
        sub = create_subscription(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
        )
        pause_subscription(subscription=sub)
        resumed = resume_subscription(subscription=sub)
        self.assertEqual(resumed.status, MealSubscription.Status.ACTIVE)

    def test_pause_rejects_already_paused(self):
        sub = create_subscription(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
            status=MealSubscription.Status.PAUSED,
        )
        with self.assertRaises(ValidationError):
            pause_subscription(subscription=sub)

    def test_cancel_subscription_marks_cancelled_and_keeps_row(self):
        sub = create_subscription(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
        )
        cancelled = cancel_subscription(
            subscription=sub, reason_code="parent_request"
        )
        self.assertEqual(cancelled.status, MealSubscription.Status.CANCELLED)
        self.assertIn("parent_request", cancelled.notes)
        # Row is NOT deleted.
        self.assertTrue(MealSubscription.objects.filter(pk=sub.pk).exists())

    def test_cancel_rejects_already_cancelled(self):
        sub = create_subscription(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
            status=MealSubscription.Status.CANCELLED,
        )
        with self.assertRaises(ValidationError):
            cancel_subscription(subscription=sub)


# ---------------------------------------------------------------------------
# Subscription selectors
# ---------------------------------------------------------------------------

class SubscriptionSelectorTests(Phase2BaseData):
    def setUp(self):
        super().setUp()
        self.active_sub = create_subscription(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
        )
        self.fallback_sub = create_subscription(
            person=self.person_student,
            meal_plan=self.plan_wallet,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=2,
        )
        self.expired_sub = create_subscription(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=self.today - timedelta(days=30),
            end_date=self.today - timedelta(days=20),
            student=self.student,
            priority=1,
            status=MealSubscription.Status.EXPIRED,
        )

    def test_active_subscriptions_for_filters_by_date(self):
        qs = active_subscriptions_for(
            person=self.person_student, on_date=self.today
        )
        # active_sub (p1) + fallback_sub (p2), but expired_sub is outside window.
        self.assertEqual(qs.count(), 2)
        # Ordered by priority.
        self.assertEqual(qs[0], self.active_sub)
        self.assertEqual(qs[1], self.fallback_sub)

    def test_active_subscriptions_for_excludes_outside_window(self):
        yesterday = self.today - timedelta(days=25)
        qs = active_subscriptions_for(
            person=self.person_student, on_date=yesterday
        )
        self.assertEqual(qs.count(), 0)

    def test_list_subscriptions_filter_by_status(self):
        qs = list_subscriptions(status=MealSubscription.Status.EXPIRED)
        self.assertEqual(qs.count(), 1)
        self.assertEqual(qs[0], self.expired_sub)

    def test_list_subscriptions_filter_by_person(self):
        qs = list_subscriptions(person=self.person_student)
        self.assertEqual(qs.count(), 3)

    def test_list_subscriptions_filter_by_meal_plan(self):
        qs = list_subscriptions(meal_plan=self.plan_wallet)
        self.assertEqual(qs.count(), 1)

    def test_list_subscriptions_is_active_period_filter(self):
        active_qs = list_subscriptions(is_active_period=True)
        self.assertEqual(active_qs.count(), 2)
        inactive_qs = list_subscriptions(is_active_period=False)
        self.assertEqual(inactive_qs.count(), 1)


# ---------------------------------------------------------------------------
# MealException model + services
# ---------------------------------------------------------------------------

class MealExceptionTests(Phase2BaseData):
    def test_create_exception_via_orm(self):
        ex = MealException.objects.create(
            person=self.person_student,
            kind=MealException.Kind.ONE_TIME_ELIGIBLE,
            effective_date=self.today,
        )
        self.assertEqual(ex.kind, MealException.Kind.ONE_TIME_ELIGIBLE)
        self.assertIsNone(ex.end_date)

    def test_exception_end_before_effective_rejected_at_db(self):
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                MealException.objects.create(
                    person=self.person_student,
                    kind=MealException.Kind.ONE_TIME_ELIGIBLE,
                    effective_date=self.today,
                    end_date=self.today - timedelta(days=1),
                )

    def test_clean_rejects_invalid_exception_window(self):
        ex = MealException(
            person=self.person_student,
            kind=MealException.Kind.ONE_TIME_ELIGIBLE,
            effective_date=self.today,
            end_date=self.today - timedelta(days=1),
        )
        with self.assertRaises(ValidationError):
            ex.full_clean()

    def test_grant_one_time_permission_service(self):
        ex = grant_one_time_permission(
            person=self.person_student,
            effective_date=self.today,
            reason_code="forgot_card",
            reason_notes="Parent called to confirm",
        )
        self.assertEqual(ex.kind, MealException.Kind.ONE_TIME_ELIGIBLE)
        self.assertEqual(ex.reason_code, "forgot_card")
        self.assertIsNone(ex.end_date)
        self.assertTrue(ex.is_active)

    def test_grant_one_time_permission_with_kind_override(self):
        ex = grant_one_time_permission(
            person=self.person_student,
            effective_date=self.today,
            kind=MealException.Kind.TEMPORARY_DENY,
        )
        self.assertEqual(ex.kind, MealException.Kind.TEMPORARY_DENY)

    def test_exception_window_validator(self):
        validate_exception_window(
            effective_date=self.today, end_date=None
        )
        validate_exception_window(
            effective_date=self.today, end_date=self.today
        )
        with self.assertRaises(ValidationError):
            validate_exception_window(
                effective_date=self.today,
                end_date=self.today - timedelta(days=1),
            )

    def test_selector_exceptions_for_single_date(self):
        grant_one_time_permission(
            person=self.person_student, effective_date=self.today
        )
        qs = exceptions_for(person=self.person_student, on_date=self.today)
        self.assertEqual(qs.count(), 1)

    def test_selector_exceptions_for_window(self):
        MealException.objects.create(
            person=self.person_student,
            kind=MealException.Kind.ONE_TIME_ELIGIBLE,
            effective_date=self.today - timedelta(days=2),
            end_date=self.today + timedelta(days=2),
        )
        qs = exceptions_for(person=self.person_student, on_date=self.today)
        self.assertEqual(qs.count(), 1)

    def test_selector_exceptions_for_excludes_outside_window(self):
        MealException.objects.create(
            person=self.person_student,
            kind=MealException.Kind.ONE_TIME_ELIGIBLE,
            effective_date=self.today + timedelta(days=5),
        )
        qs = exceptions_for(person=self.person_student, on_date=self.today)
        self.assertEqual(qs.count(), 0)

    def test_selector_exceptions_for_excludes_inactive(self):
        MealException.objects.create(
            person=self.person_student,
            kind=MealException.Kind.ONE_TIME_ELIGIBLE,
            effective_date=self.today,
            is_active=False,
        )
        qs = exceptions_for(person=self.person_student, on_date=self.today)
        self.assertEqual(qs.count(), 0)

    def test_selector_list_exceptions_filter_by_kind(self):
        MealException.objects.create(
            person=self.person_student,
            kind=MealException.Kind.ONE_TIME_ELIGIBLE,
            effective_date=self.today,
        )
        MealException.objects.create(
            person=self.person_student,
            kind=MealException.Kind.TEMPORARY_DENY,
            effective_date=self.today,
        )
        self.assertEqual(
            list_exceptions(
                person=self.person_student,
                kind=MealException.Kind.ONE_TIME_ELIGIBLE,
            ).count(),
            1,
        )


# ---------------------------------------------------------------------------
# resolve_eligibility
# ---------------------------------------------------------------------------

class ResolveEligibilityTests(Phase2BaseData):
    # --- DATE_RANGE primary ---------------------------------------------

    def test_date_range_subscription_grants_eligible_no_charge(self):
        create_subscription(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
        )
        elig = resolve_eligibility(
            person=self.person_student, on_date=self.today
        )
        self.assertEqual(elig.decision, MealEligibility.Decision.ELIGIBLE)
        self.assertEqual(elig.reason_code, REASON_DATE_RANGE_NO_CHARGE)
        self.assertEqual(elig.subscription.meal_plan, self.plan_daterange)
        self.assertEqual(elig.meal_plan, self.plan_daterange)

    def test_date_range_subscription_outside_window_not_eligible(self):
        create_subscription(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=self.today + timedelta(days=5),
            end_date=self.today + timedelta(days=20),
            student=self.student,
            priority=1,
        )
        elig = resolve_eligibility(
            person=self.person_student, on_date=self.today
        )
        self.assertEqual(elig.decision, MealEligibility.Decision.NOT_ELIGIBLE)
        self.assertEqual(elig.reason_code, REASON_NO_SUBSCRIPTION)

    def test_paused_date_range_subscription_not_eligible(self):
        sub = create_subscription(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
        )
        pause_subscription(subscription=sub)
        elig = resolve_eligibility(
            person=self.person_student, on_date=self.today
        )
        self.assertEqual(elig.decision, MealEligibility.Decision.NOT_ELIGIBLE)

    # --- Priority ordering ----------------------------------------------

    def test_priority_lower_number_wins(self):
        primary = create_subscription(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
        )
        create_subscription(
            person=self.person_student,
            meal_plan=self.plan_wallet,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=2,
        )
        elig = resolve_eligibility(
            person=self.person_student, on_date=self.today
        )
        self.assertEqual(elig.decision, MealEligibility.Decision.ELIGIBLE)
        self.assertEqual(elig.subscription, primary)
        self.assertEqual(elig.reason_code, REASON_DATE_RANGE_NO_CHARGE)

    def test_fallback_evaluated_when_primary_does_not_cover_date(self):
        # Primary covers a future window; fallback covers today.
        create_subscription(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=self.today + timedelta(days=5),
            end_date=self.today + timedelta(days=20),
            student=self.student,
            priority=1,
        )
        # Fallback is a wallet plan covering today.
        create_subscription(
            person=self.person_student,
            meal_plan=self.plan_wallet,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=2,
        )
        elig = resolve_eligibility(
            person=self.person_student, on_date=self.today
        )
        # Phase 3B-1: wallet-mode subscriptions now grant ELIGIBLE
        # (the charging decision — CONFIRMED vs UNPAID vs DENIED — is
        # made by resolve_service based on insufficient_funds_mode).
        self.assertEqual(elig.decision, MealEligibility.Decision.ELIGIBLE)
        self.assertEqual(elig.reason_code, "wallet_subscription")

    # --- Wallet mode (Phase 3B-1 grants ELIGIBLE; charging in resolve_service) ---

    def test_wallet_only_subscription_not_eligible_in_phase2(self):
        # Phase 3B-1 update: wallet-mode subscriptions now grant
        # ELIGIBLE at the eligibility layer. The charging/deny decision
        # moved to resolve_service (Phase 3B-1).
        create_subscription(
            person=self.person_student,
            meal_plan=self.plan_wallet,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
        )
        elig = resolve_eligibility(
            person=self.person_student, on_date=self.today
        )
        # Phase 3B-1: wallet-mode subscriptions now grant ELIGIBLE.
        self.assertEqual(elig.decision, MealEligibility.Decision.ELIGIBLE)
        self.assertEqual(elig.reason_code, "wallet_subscription")

    # --- Exceptions -----------------------------------------------------

    def test_one_time_eligible_exception_grants_eligible(self):
        grant_one_time_permission(
            person=self.person_student, effective_date=self.today
        )
        elig = resolve_eligibility(
            person=self.person_student, on_date=self.today
        )
        self.assertEqual(elig.decision, MealEligibility.Decision.ELIGIBLE)
        self.assertEqual(elig.reason_code, REASON_ONE_TIME_ELIGIBLE)

    def test_guest_eligible_exception_grants_eligible(self):
        MealException.objects.create(
            person=self.person_student,
            kind=MealException.Kind.GUEST_ELIGIBLE,
            effective_date=self.today,
        )
        elig = resolve_eligibility(
            person=self.person_student, on_date=self.today
        )
        self.assertEqual(elig.decision, MealEligibility.Decision.ELIGIBLE)
        self.assertEqual(elig.reason_code, REASON_GUEST_ELIGIBLE)

    def test_temporary_deny_exception_marks_overridden_denied(self):
        MealException.objects.create(
            person=self.person_student,
            kind=MealException.Kind.TEMPORARY_DENY,
            effective_date=self.today,
        )
        elig = resolve_eligibility(
            person=self.person_student, on_date=self.today
        )
        self.assertEqual(
            elig.decision, MealEligibility.Decision.OVERRIDDEN_DENIED
        )
        self.assertEqual(elig.reason_code, REASON_TEMPORARY_DENY)

    def test_exception_outside_window_ignored(self):
        MealException.objects.create(
            person=self.person_student,
            kind=MealException.Kind.ONE_TIME_ELIGIBLE,
            effective_date=self.today + timedelta(days=5),
        )
        elig = resolve_eligibility(
            person=self.person_student, on_date=self.today
        )
        self.assertEqual(elig.decision, MealEligibility.Decision.NOT_ELIGIBLE)
        self.assertEqual(elig.reason_code, REASON_NO_SUBSCRIPTION)

    def test_temporary_deny_wins_over_one_time_eligible_same_date(self):
        MealException.objects.create(
            person=self.person_student,
            kind=MealException.Kind.ONE_TIME_ELIGIBLE,
            effective_date=self.today,
        )
        MealException.objects.create(
            person=self.person_student,
            kind=MealException.Kind.TEMPORARY_DENY,
            effective_date=self.today,
        )
        elig = resolve_eligibility(
            person=self.person_student, on_date=self.today
        )
        self.assertEqual(
            elig.decision, MealEligibility.Decision.OVERRIDDEN_DENIED
        )

    # --- Default --------------------------------------------------------

    def test_no_subscription_no_exception_not_eligible(self):
        elig = resolve_eligibility(
            person=self.person_student, on_date=self.today
        )
        self.assertEqual(elig.decision, MealEligibility.Decision.NOT_ELIGIBLE)
        self.assertEqual(elig.reason_code, REASON_NO_SUBSCRIPTION)
        self.assertIsNone(elig.subscription)
        self.assertIsNone(elig.meal_plan)

    # --- Idempotency / upsert -------------------------------------------

    def test_resolve_eligibility_upserts_existing_row(self):
        create_subscription(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
        )
        elig1 = resolve_eligibility(
            person=self.person_student, on_date=self.today
        )
        self.assertEqual(elig1.decision, MealEligibility.Decision.ELIGIBLE)
        # Re-resolve after cancelling the subscription.
        cancel_subscription(subscription=elig1.subscription)
        elig2 = resolve_eligibility(
            person=self.person_student, on_date=self.today
        )
        self.assertEqual(elig2.pk, elig1.pk)  # upsert, not new row
        self.assertEqual(elig2.decision, MealEligibility.Decision.NOT_ELIGIBLE)

    def test_eligibility_unique_per_person_date(self):
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                MealEligibility.objects.create(
                    person=self.person_student,
                    date=self.today,
                    decision=MealEligibility.Decision.ELIGIBLE,
                )
                MealEligibility.objects.create(
                    person=self.person_student,
                    date=self.today,
                    decision=MealEligibility.Decision.NOT_ELIGIBLE,
                )

    # --- Selectors ------------------------------------------------------

    def test_selector_eligibility_for_returns_row(self):
        create_subscription(
            person=self.person_student,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            student=self.student,
            priority=1,
        )
        resolve_eligibility(
            person=self.person_student, on_date=self.today
        )
        elig = eligibility_for(person=self.person_student, on_date=self.today)
        self.assertIsNotNone(elig)
        self.assertEqual(elig.decision, MealEligibility.Decision.ELIGIBLE)

    def test_selector_eligibility_for_returns_none_when_absent(self):
        self.assertIsNone(
            eligibility_for(
                person=self.person_student, on_date=self.today
            )
        )

    def test_selector_list_eligibilities_filter_by_decision(self):
        # Person with no subscription → NOT_ELIGIBLE.
        resolve_eligibility(
            person=self.person_student, on_date=self.today
        )
        # Person with date-range subscription → ELIGIBLE.
        create_subscription(
            person=self.person_staff,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            staff=self.staff,
            priority=1,
        )
        resolve_eligibility(person=self.person_staff, on_date=self.today)
        eligible_qs = list_eligibilities(
            decision=MealEligibility.Decision.ELIGIBLE
        )
        self.assertEqual(eligible_qs.count(), 1)
        not_eligible_qs = list_eligibilities(
            decision=MealEligibility.Decision.NOT_ELIGIBLE
        )
        self.assertEqual(not_eligible_qs.count(), 1)

    # --- Staff support --------------------------------------------------

    def test_staff_subscription_resolves_eligible(self):
        create_subscription(
            person=self.person_staff,
            meal_plan=self.plan_daterange,
            start_date=self.window_start,
            end_date=self.window_end,
            staff=self.staff,
            priority=1,
        )
        elig = resolve_eligibility(
            person=self.person_staff, on_date=self.today
        )
        self.assertEqual(elig.decision, MealEligibility.Decision.ELIGIBLE)
        self.assertEqual(elig.reason_code, REASON_DATE_RANGE_NO_CHARGE)
