from datetime import date, timedelta

from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.test import TestCase

from apps.identity.models import Person

from .models import (
    MealPeriod,
    MealPeriodPrice,
    MealPersonPriceOverride,
    MealPlan,
)
from .selectors import (
    active_override_for_date,
    get_meal_plan_by_name,
    get_person_override,
    get_period_price,
    list_meal_periods,
    list_meal_plans,
    list_period_prices,
    list_person_overrides,
)
from .services import (
    SOURCE_DEFAULT,
    SOURCE_DEFAULT_ZERO,
    SOURCE_PERIOD_PRICE,
    SOURCE_PERSON_OVERRIDE,
    SOURCE_PERSON_OVERRIDE_PERIOD,
    create_meal_period,
    create_meal_plan,
    create_period_price,
    create_person_override,
    resolve_price,
)
from .validators import (
    validate_effective_window,
    validate_non_negative_price,
    validate_period_price_unique,
    validate_person_override_unique,
)


# ---------------------------------------------------------------------------
# Test base data
# ---------------------------------------------------------------------------

class MealsBaseData(TestCase):
    def setUp(self):
        self.person = Person.objects.create(
            code="P-M1", first_name="Meal", last_name="Tester"
        )
        self.lunch_period_1 = MealPeriod.objects.create(
            kind=MealPeriod.Kind.LUNCH,
            label="First lunch block",
            sort_order=1,
        )
        self.lunch_period_2 = MealPeriod.objects.create(
            kind=MealPeriod.Kind.LUNCH,
            label="Second lunch block",
            sort_order=2,
        )
        self.breakfast_period = MealPeriod.objects.create(
            kind=MealPeriod.Kind.BREAKFAST,
            label="Breakfast",
            sort_order=1,
        )
        self.plan_wallet = MealPlan.objects.create(
            name="Lunch Wallet",
            kind=MealPlan.Kind.LUNCH,
            mode=MealPlan.Mode.WALLET,
            default_price_iqd=1000,
        )
        self.plan_daterange = MealPlan.objects.create(
            name="Lunch Annual",
            kind=MealPlan.Kind.LUNCH,
            mode=MealPlan.Mode.DATE_RANGE,
            default_price_iqd=0,
        )
        self.today = date.today()


# ---------------------------------------------------------------------------
# MealPeriod
# ---------------------------------------------------------------------------

class MealPeriodTests(MealsBaseData):
    def test_default_kind_is_lunch(self):
        self.assertEqual(self.lunch_period_1.kind, MealPeriod.Kind.LUNCH)

    def test_str_includes_label_when_present(self):
        self.assertIn("First lunch block", str(self.lunch_period_1))

    def test_str_falls_back_to_pk_when_no_label(self):
        period = MealPeriod.objects.create(kind=MealPeriod.Kind.SNACK)
        self.assertIn("#", str(period))

    def test_template_ref_consistency_both_blank_ok(self):
        period = MealPeriod.objects.create(
            kind=MealPeriod.Kind.LUNCH,
            period_template_source="",
            period_template_ref_id=None,
        )
        self.assertIsNone(period.period_template_ref_id)

    def test_template_ref_consistency_source_only_rejected_at_clean(self):
        period = MealPeriod(
            kind=MealPeriod.Kind.LUNCH,
            period_template_source="attendance",
            period_template_ref_id=None,
        )
        with self.assertRaises(ValidationError):
            period.full_clean()

    def test_template_ref_consistency_ref_only_rejected_at_clean(self):
        period = MealPeriod(
            kind=MealPeriod.Kind.LUNCH,
            period_template_source="",
            period_template_ref_id=42,
        )
        with self.assertRaises(ValidationError):
            period.full_clean()

    def test_unique_kind_and_template_ref_enforced_at_db(self):
        MealPeriod.objects.create(
            kind=MealPeriod.Kind.LUNCH,
            period_template_source="attendance",
            period_template_ref_id=7,
        )
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                MealPeriod.objects.create(
                    kind=MealPeriod.Kind.LUNCH,
                    period_template_source="attendance",
                    period_template_ref_id=7,
                )

    def test_duplicate_kind_template_ref_across_kinds_allowed(self):
        # Same ref under a different kind is fine.
        MealPeriod.objects.create(
            kind=MealPeriod.Kind.LUNCH,
            period_template_source="attendance",
            period_template_ref_id=7,
        )
        MealPeriod.objects.create(
            kind=MealPeriod.Kind.BREAKFAST,
            period_template_source="attendance",
            period_template_ref_id=7,
        )

    def test_selectors_list_meal_periods(self):
        qs = list_meal_periods(kind=MealPeriod.Kind.LUNCH)
        self.assertEqual(qs.count(), 2)
        qs = list_meal_periods(is_active=True)
        self.assertEqual(qs.count(), 3)

    def test_selector_get_meal_period_by_template_ref(self):
        MealPeriod.objects.create(
            kind=MealPeriod.Kind.LUNCH,
            period_template_source="attendance",
            period_template_ref_id=99,
        )
        again = list_meal_periods().filter(
            period_template_ref_id=99
        ).first()
        self.assertIsNotNone(again)


# ---------------------------------------------------------------------------
# MealPlan
# ---------------------------------------------------------------------------

class MealPlanTests(MealsBaseData):
    def test_str_returns_name(self):
        self.assertEqual(str(self.plan_wallet), "Lunch Wallet")

    def test_default_mode_is_date_range(self):
        plan = MealPlan.objects.create(name="X")
        self.assertEqual(plan.mode, MealPlan.Mode.DATE_RANGE)

    def test_default_price_non_negative_constraint(self):
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                MealPlan.objects.create(
                    name="Bad", default_price_iqd=-1
                )

    def test_unique_name(self):
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                MealPlan.objects.create(name="Lunch Wallet")

    def test_selectors_list_meal_plans_by_kind(self):
        self.assertEqual(
            list_meal_plans(kind=MealPlan.Kind.LUNCH).count(), 2
        )
        self.assertEqual(
            list_meal_plans(mode=MealPlan.Mode.WALLET).count(), 1
        )

    def test_selector_get_meal_plan_by_name(self):
        self.assertEqual(get_meal_plan_by_name("Lunch Wallet"), self.plan_wallet)


# ---------------------------------------------------------------------------
# MealPeriodPrice
# ---------------------------------------------------------------------------

class MealPeriodPriceTests(MealsBaseData):
    def test_create_period_price(self):
        price = MealPeriodPrice.objects.create(
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=1500,
        )
        self.assertEqual(price.price_iqd, 1500)
        self.assertTrue(price.is_enabled)

    def test_unique_plan_period(self):
        MealPeriodPrice.objects.create(
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=1500,
        )
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                MealPeriodPrice.objects.create(
                    meal_plan=self.plan_wallet,
                    meal_period=self.lunch_period_1,
                    price_iqd=2000,
                )

    def test_negative_price_rejected_at_db(self):
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                MealPeriodPrice.objects.create(
                    meal_plan=self.plan_wallet,
                    meal_period=self.lunch_period_1,
                    price_iqd=-5,
                )

    def test_selector_get_period_price(self):
        MealPeriodPrice.objects.create(
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=1500,
        )
        price = get_period_price(
            meal_plan=self.plan_wallet, meal_period=self.lunch_period_1
        )
        self.assertIsNotNone(price)
        self.assertEqual(price.price_iqd, 1500)

    def test_selector_get_period_price_skips_disabled(self):
        MealPeriodPrice.objects.create(
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=1500,
            is_enabled=False,
        )
        self.assertIsNone(
            get_period_price(
                meal_plan=self.plan_wallet, meal_period=self.lunch_period_1
            )
        )

    def test_selector_list_period_prices(self):
        MealPeriodPrice.objects.create(
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=1500,
        )
        MealPeriodPrice.objects.create(
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_2,
            price_iqd=1200,
        )
        self.assertEqual(
            list_period_prices(meal_plan=self.plan_wallet).count(), 2
        )
        self.assertEqual(
            list_period_prices(
                meal_plan=self.plan_wallet,
                meal_period=self.lunch_period_1,
            ).count(),
            1,
        )

    def test_validator_period_price_unique(self):
        MealPeriodPrice.objects.create(
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=1500,
        )
        with self.assertRaises(ValidationError):
            validate_period_price_unique(
                meal_plan=self.plan_wallet,
                meal_period=self.lunch_period_1,
            )


# ---------------------------------------------------------------------------
# MealPersonPriceOverride
# ---------------------------------------------------------------------------

class MealPersonPriceOverrideTests(MealsBaseData):
    def test_create_any_period_override(self):
        override = MealPersonPriceOverride.objects.create(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=None,
            price_iqd=700,
        )
        self.assertIsNone(override.meal_period)
        self.assertEqual(override.price_iqd, 700)

    def test_create_per_period_override(self):
        override = MealPersonPriceOverride.objects.create(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=700,
        )
        self.assertEqual(override.meal_period, self.lunch_period_1)

    def test_unique_per_plan_period(self):
        MealPersonPriceOverride.objects.create(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=700,
        )
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                MealPersonPriceOverride.objects.create(
                    person=self.person,
                    meal_plan=self.plan_wallet,
                    meal_period=self.lunch_period_1,
                    price_iqd=800,
                )

    def test_any_period_and_per_period_can_coexist(self):
        # NULL is treated as distinct by PostgreSQL.
        MealPersonPriceOverride.objects.create(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=None,
            price_iqd=700,
        )
        MealPersonPriceOverride.objects.create(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=750,
        )
        self.assertEqual(
            list_person_overrides(person=self.person).count(), 2
        )

    def test_negative_override_price_rejected_at_db(self):
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                MealPersonPriceOverride.objects.create(
                    person=self.person,
                    meal_plan=self.plan_wallet,
                    meal_period=self.lunch_period_1,
                    price_iqd=-1,
                )

    def test_invalid_effective_window_rejected_at_db(self):
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                MealPersonPriceOverride.objects.create(
                    person=self.person,
                    meal_plan=self.plan_wallet,
                    meal_period=self.lunch_period_1,
                    price_iqd=700,
                    effective_from=self.today + timedelta(days=1),
                    effective_until=self.today,
                )

    def test_selector_get_person_override_any_period(self):
        MealPersonPriceOverride.objects.create(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=None,
            price_iqd=700,
        )
        override = get_person_override(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=None,
        )
        self.assertIsNotNone(override)
        self.assertEqual(override.price_iqd, 700)

    def test_selector_get_person_override_per_period(self):
        MealPersonPriceOverride.objects.create(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=750,
        )
        override = get_person_override(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
        )
        self.assertIsNotNone(override)
        self.assertEqual(override.price_iqd, 750)

    def test_selector_active_override_for_date_respects_effective_window(self):
        # Past override (until yesterday) — should not match today.
        MealPersonPriceOverride.objects.create(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=700,
            effective_from=self.today - timedelta(days=2),
            effective_until=self.today - timedelta(days=1),
        )
        self.assertIsNone(
            active_override_for_date(
                person=self.person,
                meal_plan=self.plan_wallet,
                meal_period=self.lunch_period_1,
                on_date=self.today,
            )
        )

    def test_selector_active_override_for_date_matches_current_window(self):
        MealPersonPriceOverride.objects.create(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=700,
            effective_from=self.today - timedelta(days=1),
            effective_until=self.today + timedelta(days=1),
        )
        override = active_override_for_date(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            on_date=self.today,
        )
        self.assertIsNotNone(override)
        self.assertEqual(override.price_iqd, 700)

    def test_selector_active_override_skips_disabled(self):
        MealPersonPriceOverride.objects.create(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=700,
            is_enabled=False,
        )
        self.assertIsNone(
            active_override_for_date(
                person=self.person,
                meal_plan=self.plan_wallet,
                meal_period=self.lunch_period_1,
                on_date=self.today,
            )
        )

    def test_validator_person_override_unique(self):
        MealPersonPriceOverride.objects.create(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=700,
        )
        with self.assertRaises(ValidationError):
            validate_person_override_unique(
                person=self.person,
                meal_plan=self.plan_wallet,
                meal_period=self.lunch_period_1,
            )


# ---------------------------------------------------------------------------
# Services — create_*
# ---------------------------------------------------------------------------

class ServiceCreateTests(MealsBaseData):
    def test_create_meal_period_service(self):
        period = create_meal_period(
            kind=MealPeriod.Kind.SNACK,
            label="Snack",
            sort_order=5,
        )
        self.assertEqual(period.kind, MealPeriod.Kind.SNACK)
        self.assertEqual(period.label, "Snack")

    def test_create_meal_period_service_rejects_inconsistent_ref(self):
        from django.core.exceptions import ValidationError as VE

        with self.assertRaises(VE):
            create_meal_period(
                kind=MealPeriod.Kind.SNACK,
                period_template_source="attendance",
                period_template_ref_id=None,
            )

    def test_create_meal_plan_service(self):
        plan = create_meal_plan(
            name="Service Plan",
            kind=MealPlan.Kind.BREAKFAST,
            mode=MealPlan.Mode.WALLET,
            default_price_iqd=500,
        )
        self.assertEqual(plan.name, "Service Plan")
        self.assertEqual(plan.kind, MealPlan.Kind.BREAKFAST)

    def test_create_meal_plan_service_rejects_negative_default(self):
        with self.assertRaises(ValidationError):
            create_meal_plan(
                name="Bad Plan",
                default_price_iqd=-1,
            )

    def test_create_period_price_service(self):
        price = create_period_price(
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=1200,
        )
        self.assertEqual(price.price_iqd, 1200)

    def test_create_period_price_service_rejects_duplicate(self):
        create_period_price(
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=1200,
        )
        with self.assertRaises(ValidationError):
            create_period_price(
                meal_plan=self.plan_wallet,
                meal_period=self.lunch_period_1,
                price_iqd=999,
            )

    def test_create_person_override_service(self):
        override = create_person_override(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=700,
            reason_code="staff_discount",
        )
        self.assertEqual(override.price_iqd, 700)
        self.assertEqual(override.reason_code, "staff_discount")

    def test_create_person_override_service_rejects_invalid_window(self):
        with self.assertRaises(ValidationError):
            create_person_override(
                person=self.person,
                meal_plan=self.plan_wallet,
                meal_period=self.lunch_period_1,
                price_iqd=700,
                effective_from=self.today + timedelta(days=1),
                effective_until=self.today,
            )


# ---------------------------------------------------------------------------
# Services — resolve_price (pure function)
# ---------------------------------------------------------------------------

class ResolvePriceTests(MealsBaseData):
    # --- DATE_RANGE plans short-circuit to 0 ---------------------------

    def test_date_range_plan_returns_zero(self):
        base, delta, source = resolve_price(
            person=self.person,
            meal_plan=self.plan_daterange,
            meal_period=self.lunch_period_1,
            on_date=self.today,
        )
        self.assertEqual(base, 0)
        self.assertEqual(delta, 0)
        self.assertEqual(source, "date_range_no_charge")

    # --- Default price fallback ---------------------------------------

    def test_wallet_plan_no_period_price_no_override_falls_back_to_default(self):
        base, delta, source = resolve_price(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            on_date=self.today,
        )
        self.assertEqual(base, 1000)
        self.assertEqual(delta, 0)
        self.assertEqual(source, SOURCE_DEFAULT)

    def test_wallet_plan_zero_default_returns_default_zero(self):
        plan = MealPlan.objects.create(
            name="Free Wallet",
            kind=MealPlan.Kind.LUNCH,
            mode=MealPlan.Mode.WALLET,
            default_price_iqd=0,
        )
        base, delta, source = resolve_price(
            person=self.person,
            meal_plan=plan,
            meal_period=self.lunch_period_1,
            on_date=self.today,
        )
        self.assertEqual(base, 0)
        self.assertEqual(delta, 0)
        self.assertEqual(source, SOURCE_DEFAULT_ZERO)

    # --- Per-MealPeriod list price ------------------------------------

    def test_wallet_plan_period_price_wins(self):
        MealPeriodPrice.objects.create(
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=1500,
        )
        base, delta, source = resolve_price(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            on_date=self.today,
        )
        self.assertEqual(base, 1500)
        self.assertEqual(delta, 0)
        self.assertEqual(source, SOURCE_PERIOD_PRICE)

    def test_wallet_plan_disabled_period_price_skipped(self):
        MealPeriodPrice.objects.create(
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=1500,
            is_enabled=False,
        )
        base, delta, source = resolve_price(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            on_date=self.today,
        )
        self.assertEqual(base, 1000)  # default
        self.assertEqual(source, SOURCE_DEFAULT)

    # --- Per-Person × Per-MealPeriod override -------------------------

    def test_per_period_override_wins_over_period_price(self):
        MealPeriodPrice.objects.create(
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=1500,
        )
        MealPersonPriceOverride.objects.create(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=1000,  # cheaper than the 1500 list price
        )
        base, delta, source = resolve_price(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            on_date=self.today,
        )
        self.assertEqual(base, 1000)
        self.assertEqual(delta, 500)  # list - override = 1500 - 1000
        self.assertEqual(source, SOURCE_PERSON_OVERRIDE_PERIOD)

    def test_per_period_override_with_no_list_price_delta_zero(self):
        MealPersonPriceOverride.objects.create(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=800,
        )
        base, delta, source = resolve_price(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            on_date=self.today,
        )
        self.assertEqual(base, 800)
        self.assertEqual(delta, 0)  # no period price → list = 0
        self.assertEqual(source, SOURCE_PERSON_OVERRIDE_PERIOD)

    def test_per_period_override_outside_effective_window_skipped(self):
        MealPersonPriceOverride.objects.create(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=800,
            effective_from=self.today + timedelta(days=1),
            effective_until=self.today + timedelta(days=2),
        )
        base, delta, source = resolve_price(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            on_date=self.today,
        )
        self.assertEqual(source, SOURCE_DEFAULT)
        self.assertEqual(base, 1000)

    def test_per_period_override_disabled_skipped(self):
        MealPersonPriceOverride.objects.create(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=800,
            is_enabled=False,
        )
        base, delta, source = resolve_price(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            on_date=self.today,
        )
        self.assertEqual(source, SOURCE_DEFAULT)

    # --- Per-Person × any-period override -----------------------------

    def test_any_period_override_wins_over_default(self):
        MealPersonPriceOverride.objects.create(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=None,  # any-period override
            price_iqd=600,
        )
        base, delta, source = resolve_price(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            on_date=self.today,
        )
        self.assertEqual(base, 600)
        # No period price exists, so delta=0 per §14.1.
        self.assertEqual(delta, 0)
        self.assertEqual(source, SOURCE_PERSON_OVERRIDE)

    def test_any_period_override_does_not_win_when_per_period_exists(self):
        # Per-period override takes precedence over any-period override.
        MealPersonPriceOverride.objects.create(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=None,
            price_iqd=600,
        )
        MealPersonPriceOverride.objects.create(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=700,
        )
        base, delta, source = resolve_price(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            on_date=self.today,
        )
        self.assertEqual(base, 700)
        self.assertEqual(source, SOURCE_PERSON_OVERRIDE_PERIOD)

    def test_any_period_override_wins_over_period_price_when_no_per_period(self):
        MealPeriodPrice.objects.create(
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=1500,
        )
        MealPersonPriceOverride.objects.create(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=None,
            price_iqd=600,
        )
        base, delta, source = resolve_price(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            on_date=self.today,
        )
        # Per-period override? none → fall to any-period override.
        self.assertEqual(base, 600)
        self.assertEqual(delta, 900)  # list 1500 - 600
        self.assertEqual(source, SOURCE_PERSON_OVERRIDE)

    # --- Surcharged override (override > list price) ------------------

    def test_surcharged_override_produces_negative_delta(self):
        MealPeriodPrice.objects.create(
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=1000,
        )
        MealPersonPriceOverride.objects.create(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=1200,  # more expensive than the list price
        )
        base, delta, source = resolve_price(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            on_date=self.today,
        )
        self.assertEqual(base, 1200)
        self.assertEqual(delta, -200)  # list 1000 - 1200
        self.assertEqual(source, SOURCE_PERSON_OVERRIDE_PERIOD)

    # --- Purity: resolve_price does not write -------------------------

    def test_resolve_price_does_not_create_any_rows(self):
        MealPeriodPrice.objects.create(
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            price_iqd=1500,
        )
        before_periods = MealPeriod.objects.count()
        before_plans = MealPlan.objects.count()
        before_prices = MealPeriodPrice.objects.count()
        before_overrides = MealPersonPriceOverride.objects.count()
        resolve_price(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
            on_date=self.today,
        )
        self.assertEqual(MealPeriod.objects.count(), before_periods)
        self.assertEqual(MealPlan.objects.count(), before_plans)
        self.assertEqual(MealPeriodPrice.objects.count(), before_prices)
        self.assertEqual(
            MealPersonPriceOverride.objects.count(), before_overrides
        )

    def test_resolve_price_defaults_on_date_to_today(self):
        # Calling without on_date must not blow up; it uses date.today().
        base, _, source = resolve_price(
            person=self.person,
            meal_plan=self.plan_wallet,
            meal_period=self.lunch_period_1,
        )
        self.assertEqual(source, SOURCE_DEFAULT)
        self.assertEqual(base, 1000)


# ---------------------------------------------------------------------------
# Validators (unit)
# ---------------------------------------------------------------------------

class ValidatorUnitTests(TestCase):
    def test_validate_non_negative_price_rejects_negative(self):
        with self.assertRaises(ValidationError):
            validate_non_negative_price(-1)

    def test_validate_non_negative_price_accepts_zero(self):
        validate_non_negative_price(0)

    def test_validate_effective_window_invalid(self):
        with self.assertRaises(ValidationError):
            validate_effective_window(
                effective_from=date(2026, 7, 5),
                effective_until=date(2026, 7, 4),
            )

    def test_validate_effective_window_one_sided_ok(self):
        validate_effective_window(effective_from=date(2026, 7, 5))
        validate_effective_window(effective_until=date(2026, 7, 5))
        validate_effective_window()
