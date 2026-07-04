# Finance Domain Architecture — BISK_RFv4

Date: 2026-07-03 (revised 2026-07-04)
Branch: feature/person-architecture
Version: 1.1 Draft — pending review
Status: Architecture design only. No code, models, or migrations are produced by this document.

> **Reconciliation note (2026-07-04):** This document has been revised
> to reconcile it with the finalized Meals domain architecture
> (`docs/architecture/meals_domain_architecture.md` v1.1). Under the
> decided architecture, **Meals owns meal price resolution** and
> **Finance records pre-resolved meal charges only**. Finance does
> **not** resolve meal prices and does **not** call the Discounts
> service for meal pricing. The meal product list price
> (`MealPeriodPrice`), per-Person overrides
> (`MealPersonPriceOverride`), and the discount-composition call now
> all live in `apps.meals`. The `PriceList` / `PricingRule` placeholder
> in §4.7 is retained for **non-meal** ERP billing (tuition, transport,
> etc.) but is no longer the owner of meal prices. The future meal app
> is `apps.meals` (plural) and the meal product model is `MealPlan`
> (legacy `attendance.MealProfile` maps to it). See
> `docs/architecture/meals_domain_architecture.md` for the
> authoritative Meals design.

---

## 1. Purpose

This document designs the **Finance domain** for BISK_RFv4 — wallets,
charges, payments, refunds, adjustments, and the foundation for future
ERP billing modules (tuition, transport, library fines, school shop,
event fees, uniforms).

> Domain boundary (decided): **Finance owns wallets, ledger, payments,
> refunds, charges, and adjustments only.** Finance records money. It
> does **not** own meal product pricing, meal periods, meal entitlement,
> meal subscriptions, or discount rules. For meal billing, Finance
> receives a **pre-resolved** charge amount from Meals and records the
> ledger row. See `meals_domain_architecture.md` §5 and §20.

The current implementation lives scattered inside `apps.attendance`
(`Wallet`, `WalletTransaction`, `DiscountProfile`, `DiscountRule`),
all currently FK-keyed to the legacy `attendance.Student`. This document
designs a clean `apps.finance` domain that:

- Anchors to `Person` (and where role-specific, `StudentProfile`), never
  to legacy `Student`.
- Enforces an **immutable ledger**: every balance change is a permanent
  `WalletTransaction`; the running balance is a derived/cached value that
  must always be reconstructable from the ledger.
- Exposes a **service boundary** so the meals domain (and future ERP
  modules) can request charges, refunds, and balance checks without
  embedding wallet math.
- **Records pre-resolved charges only.** For meal billing, Meals resolves
  the price (and, in the future, calls `apps.discounts`) and passes a
  final `amount_iqd` to `finance.charge`. Finance does **not** resolve
  meal prices and does **not** call the Discounts service for meal
  pricing. (Discounts remain a separate future domain, `apps.discounts`,
  that Meals will call directly when implemented.)
- Does **not** own discount rules, profiles, or assignments. Discount
  implementation is a separate future domain (`apps.discounts`); Finance
  only snapshots a `discount_iqd` value on `Charge` when the caller
  (Meals) supplies one as part of the pre-resolved amount's audit
  context.

This is a **design document**, not an implementation order. Each entity
below is a blueprint for a future, separately-approved implementation step.

### Scope

| In scope | Out of scope |
|---|---|
| Wallet, WalletTransaction, Charge, Payment, Refund, Adjustment | AI recognition engine (attendance) |
| Wallet lifecycle and immutable ledger | Meal subscriptions / eligibility / service events (meals domain) |
| Transaction model: credit/debit, source module, reversal | Meal product pricing, meal periods, per-Person meal pricing (meals domain) |
| Meal billing integration boundary (Finance records pre-resolved charges) | Grade/Section/Enrollment models (academics) |
| Non-meal ERP billing list-price placeholder (`PriceList` / `PricingRule` for tuition/transport/etc.) | Menu/kitchen inventory |
| Future ERP billing module support | Bank/payment-gateway processing |
| Licensing & audit support | Full double-entry accounting engine |
|  | Discount rules/profiles/assignments (future `apps.discounts`; Meals calls it, not Finance) |
|  | Payroll / staff salary computation |

### Constraints respected

- Do **not** recreate the old `Student` model. Finance references
  `Person` and (where role-specific) `StudentProfile` via `apps.identity`.
- Finance **must preserve historical ledger records**: transactions are
  append-only and immutable; corrections create reversal rows, never
  mutate prior rows.
- **Avoid destructive balance updates**: the `Wallet.balance` field is a
  cached projection of the ledger, updated only inside the same
  transaction that appends the ledger row.
- **Avoid storing only a mutable balance without ledger history**: a
  wallet with a balance column but no transaction rows is forbidden; the
  ledger is the source of truth.
- **Lunch/meals must not contain direct wallet math**: meals calls finance
  services; it never reads/writes `Wallet.balance` or creates
  `WalletTransaction` rows directly. (Wallet mutations happen only
  through Finance services.)
- **Finance records pre-resolved meal charges only.** For meal billing,
  Meals resolves the price and passes a final `amount_iqd` to
  `finance.charge`. Finance does **not** resolve meal prices and does
  **not** call `apps.discounts` for meal pricing. Meals will call
  `apps.discounts` directly when discounts are implemented.
- Uses the `services.py` / `selectors.py` / `validators.py` pattern
  established by `apps.identity` and `apps.academics`, and follows
  `DOMAIN_INTEGRATION_GUIDE.md`.
- Follows `education_domain_architecture.md`,
  `academics_domain_architecture.md`, `meals_domain_architecture.md`
  (authoritative for the meals domain),
  `person_identity_architecture.md`, `PROJECT_ARCHITECTURE.md`, and
  `AI_DEVELOPMENT_GUIDE.md`.

---

## 2. Design Principles

| # | Principle | Application to Finance |
|---|---|---|
| 1 | **Identity before role** | A wallet belongs to a `Person`. Whether that person is a student is answered by `PersonRole`, not by the wallet. Student-specific billing context may reference `StudentProfile`, but the ledger anchor is `Person`. |
| 2 | **Ledger is the source of truth** | The `WalletTransaction` rows are the authoritative record. `Wallet.balance` is a cached projection, always reconstructable as `sum(transactions.amount)`. A balance update without a ledger row is a bug. |
| 3 | **Append-only ledger** | Transactions are never edited or deleted. Corrections create a new reversal transaction referencing the original via `reverses`. Historical rows remain valid forever. |
| 4 | **Money is auditable** | Every charge, top-up, refund, and adjustment creates a permanent transaction with `balance_before`/`balance_after` snapshots, a source module, a reference, an actor, and a reason. No silent balance changes. |
| 5 | **Service-layer boundary** | Balance checks, charges, and refunds are performed by `apps.finance.services` (e.g. `charge()`, `refund()`, `check_balance()`). Other domains call these services; they never touch `Wallet.balance` or create `WalletTransaction` rows directly. |
| 6 | **Workflow as state machine** | Wallet lifecycle (`active`/`suspended`/`closed`) and payment status use explicit `status` enums with documented transitions, not scattered booleans. |
| 7 | **Domain ownership first** | `apps.finance` owns Wallet/WalletTransaction/Charge/Payment/Refund/Adjustment and (for non-meal ERP billing) `PriceList`/`PricingRule`. It does **not** own meal product pricing, meal periods, meal entitlement, or meal subscriptions — those belong to `apps.meals`. It does **not** own discount rules — those belong to future `apps.discounts`. Meals, attendance, and future ERP modules depend on finance via a service boundary; finance does not depend on them. |
| 8 | **Snapshot only when justified** | Financial snapshots (price, discount, charge, balance before/after) are stored on immutable historical records in the *calling* domain (e.g. `MealServiceEvent`) because they cannot be reconstructed later; finance itself stores the authoritative ledger. |
| 9 | **display_code in templates** | Wallet/billing reports render `person.display_code`, never legacy `h_code`. |
| 10 | **No duplicated identity** | No `h_code`, no `first_name`/`last_name`/`grade` on finance entities. Names are read via `wallet.person.full_name`. |
| 11 | **One logical feature per migration** | Each wallet/transaction schema change is its own migration; no unrelated changes ride along (per `DEVELOPMENT_STANDARDS.md`). |
| 12 | **Backward compatibility always** | Legacy `Wallet`/`WalletTransaction` keep working during migration. New models are additive; legacy FKs are removed only after verification. |
| 13 | **Integer money, explicit currency** | Amounts are integer IQD (matching legacy). Currency is explicit on the wallet/transaction. Decimal is a future option (open question Q11), but mixing int and decimal is forbidden. |

---

## 3. Proposed App / Module Name

```
apps/finance/
├── __init__.py
├── apps.py
├── models.py        # Wallet, WalletTransaction, Charge, Payment,
│                    # Refund, Adjustment, PriceList, PricingRule (non-meal ERP only)
├── admin.py
├── services.py      # create_wallet, charge, refund, top_up, adjust,
│                    # check_balance  (charge records pre-resolved amounts;
│                    # does NOT resolve meal prices or call discounts)
├── selectors.py     # balance_for, transaction_history, ledger_for_person, ...
├── validators.py    # sufficient_funds, wallet_active, reversal rules, ...
└── migrations/
```

**App label:** `finance` (Django app). Python module path: `apps.finance`.

### Dependency direction

```
apps.finance ──depends on──► apps.identity   (Person, StudentProfile, StaffProfile)
apps.finance ──depended on by──► apps.meals          (meal billing; Meals calls finance)
apps.finance ──depended on by──► future apps.tuition, apps.transport, apps.shop, ...
apps.finance ──may consume──► apps.academics  (AcademicYear for year-scoped billing, read-only)
```

> Note: `apps.discounts` does not exist yet. Under the decided
> architecture, **Finance does not call `apps.discounts`** for meal
> pricing. Meals calls `apps.discounts` directly during its own
> `resolve_price` and passes the pre-resolved `final_charge_iqd` to
> `finance.charge`. Finance only snapshots a `discount_iqd` value on
> `Charge` when the caller (Meals) supplies one as part of the
> pre-resolved amount's audit context. For future **non-meal** ERP
> billing modules that do not perform their own discount resolution,
> Finance may still expose a `resolve_pricing` helper, but it must not
> call `apps.discounts` on the meal code path.

`apps.finance` must **not** import from `apps.meals`, `apps.attendance`,
or any future billing module. Those modules call finance; the reverse is
forbidden to avoid circular ownership. A meal charge is requested by the
meals domain via `finance.charge(...)` with a **pre-resolved**
`amount_iqd`; finance records the ledger and returns the transaction;
meals stores the snapshot.

A dependency `meals → finance` is allowed (meals calls finance). A
dependency `finance → meals` is **not** allowed.

---

## 4. Core Entities

All designs below are **conceptual blueprints**. No migration is produced by
this document. Field types are indicative; exact choices are made at
implementation time.

### 4.1 Wallet

- **Owner:** `apps.finance`
- **Purpose:** A prepaid balance and the anchor for a Person's ledger. The
  migration target for legacy `Wallet` (currently FK to `attendance.Student`).
- **Conceptual fields:**

```python
class Wallet(models.Model):
    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        SUSPENDED = "suspended", "Suspended"
        CLOSED = "closed", "Closed"

    person = models.OneToOneField(
        "identity.Person", on_delete=models.PROTECT,
        related_name="wallet",
    )
    currency = models.CharField(max_length=8, default="IQD")
    balance_iqd = models.IntegerField(
        default=0,
        help_text="Cached projection of the ledger. Always = sum(transactions.amount).",
    )
    status = models.CharField(max_length=20, choices=Status.choices,
                              default=Status.ACTIVE, db_index=True)
    credit_limit_iqd = models.IntegerField(
        default=0,
        help_text="Negative balance allowed down to -credit_limit (0 = no negative).",
    )
    notes = models.CharField(max_length=200, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["person__code"]
        indexes = [models.Index(fields=["status"])]
```

- **Rules:**
  - `person` is a `OneToOneField` to `Person` (PROTECT — never delete a
    person who has a wallet without explicit handling). This replaces the
    legacy `student` FK.
  - `balance_iqd` is a **cached projection**, not the source of truth. It
    is updated only inside `@transaction.atomic` service methods that also
    append a `WalletTransaction`. Direct writes to `balance_iqd` outside
    the ledger service are forbidden (enforced by convention + audit; a DB
    trigger or signal-based guard is an open question Q9).
  - `credit_limit_iqd` allows controlled negative balances (e.g. -1000 IQD)
    for trust-based billing; `0` means no negative allowed.
  - `status` is the wallet lifecycle state (Section 5).

### 4.2 WalletTransaction

- **Owner:** `apps.finance`
- **Purpose:** The **immutable ledger row**. Every balance change is one
  row. The migration target for legacy `WalletTransaction` (currently FK
  to `attendance.Student` and `attendance.AttendanceRecord`).
- **Conceptual fields:**

```python
class WalletTransaction(models.Model):
    class TxType(models.TextChoices):
        TOPUP = "topup", "Top-up"            # credit (payment in)
        DEBIT = "debit", "Debit"             # charge (money out)
        REFUND = "refund", "Refund"          # credit (reversal of a debit)
        ADJUSTMENT = "adjustment", "Adjustment"  # correction (credit or debit)
        UNPAID = "unpaid", "Unpaid"          # recorded charge, no wallet impact

    wallet = models.ForeignKey(
        Wallet, on_delete=models.PROTECT,
        related_name="transactions",
    )
    person = models.ForeignKey(
        "identity.Person", on_delete=models.PROTECT,
        related_name="wallet_transactions",
    )
    tx_type = models.CharField(max_length=20, choices=TxType.choices, db_index=True)
    amount_iqd = models.IntegerField(
        help_text="Signed. +topup, -debit, +refund, +/-adjustment, 0 unpaid.",
    )
    balance_before_iqd = models.IntegerField(default=0)
    balance_after_iqd = models.IntegerField(default=0)

    # provenance: who/what created this row
    source_module = models.CharField(max_length=40, db_index=True,
        help_text="e.g. 'meal', 'tuition', 'manual_topup'.")
    reference_type = models.CharField(max_length=40, blank=True, default="")
    reference_id = models.PositiveBigIntegerField(null=True, blank=True)

    reason_code = models.CharField(max_length=32, blank=True, default="")
    notes = models.CharField(max_length=200, blank=True, default="")

    # reversal chain (append-only: a reversal is itself a transaction)
    is_reversal = models.BooleanField(default=False)
    reverses = models.ForeignKey(
        "self", on_delete=models.PROTECT, related_name="reversals",
        null=True, blank=True,
    )

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name="created_wallet_transactions",
    )
    created_by_staff = models.ForeignKey(
        "identity.StaffProfile", on_delete=models.SET_NULL,
        null=True, blank=True, related_name="created_wallet_transactions",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-id"]
        indexes = [
            models.Index(fields=["wallet", "created_at"]),
            models.Index(fields=["person", "created_at"]),
            models.Index(fields=["source_module", "created_at"]),
            models.Index(fields=["tx_type", "created_at"]),
        ]
```

- **Rules:**
  - **Immutable after creation.** No field on a `WalletTransaction` is
    ever updated. Corrections create a new row (`is_reversal=True`,
    `reverses=<original>`). The original row is never mutated or deleted.
  - `amount_iqd` is signed: top-up/refund are positive, debit is negative,
    adjustment may be either, unpaid is zero (records an intent without
    balance impact).
  - `balance_before_iqd` / `balance_after_iqd` snapshot the wallet balance
    around this transaction for audit; they must equal the wallet's
    balance before/after the row is applied.
  - `source_module` + `reference_type`/`reference_id` form a **generic
    reference** to the originating object (e.g.
    `source_module="meals", reference_type="MealServiceEvent",
    reference_id=42`). A typed FK to `MealServiceEvent` is **not** used
    because finance must not import meals models (circular ownership). The
    meals domain stores the reverse typed FK
    (`MealServiceEvent.wallet_transaction`) for its own lookups.
  - `person` is denormalized from `wallet.person` for fast person-scoped
    queries without joining through wallet; it must equal
    `wallet.person_id` (enforced in `clean()`).

### 4.3 Charge

- **Owner:** `apps.finance`
- **Purpose:** A **charge request** — an intent to debit a wallet for a
  product/service. A charge produces a `WalletTransaction` (DEBIT) if the
  wallet has funds (or credit), or an UNPAID row if not. Charges decouple
  "I want to bill this person" from "the ledger entry".
- **Conceptual fields:**

```python
class Charge(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        SETTLED = "settled", "Settled"          # debit transaction created
        UNPAID = "unpaid", "Unpaid"             # recorded, no balance impact
        REVERSED = "reversed", "Reversed"       # fully refunded
        VOIDED = "voided", "Voided"             # cancelled before settlement

    wallet = models.ForeignKey(Wallet, on_delete=models.PROTECT, related_name="charges")
    person = models.ForeignKey("identity.Person", on_delete=models.PROTECT,
                               related_name="finance_charges")
    academic_year = models.ForeignKey(
        "academics.AcademicYear", on_delete=models.PROTECT,
        related_name="charges", null=True, blank=True,
    )
    product_code = models.CharField(max_length=64, db_index=True,
        help_text="e.g. 'meal_lunch', 'tuition_2026_term1'.")
    description = models.CharField(max_length=200, blank=True, default="")
    price_base_iqd = models.IntegerField(default=0)
    # discount snapshots (resolved by apps.discounts; finance stores the result only)
    discount_iqd = models.IntegerField(default=0)
    discount_reason_code = models.CharField(max_length=32, blank=True, default="")
    discount_notes = models.CharField(max_length=200, blank=True, default="")
    final_charge_iqd = models.IntegerField(default=0)
    status = models.CharField(max_length=20, choices=Status.choices,
                              default=Status.PENDING, db_index=True)
    settled_transaction = models.ForeignKey(
        WalletTransaction, on_delete=models.PROTECT,
        related_name="settled_charges", null=True, blank=True,
    )
    source_module = models.CharField(max_length=40, db_index=True)
    reference_type = models.CharField(max_length=40, blank=True, default="")
    reference_id = models.PositiveBigIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["person", "status"]),
            models.Index(fields=["academic_year", "product_code"]),
            models.Index(fields=["source_module", "status"]),
        ]
```

- **Rules:**
  - A `Charge` is the canonical billing record per product instance. It
    snapshots `price_base_iqd` / `discount_iqd` / `final_charge_iqd`
    because pricing/discount rules change over time and the charged amount
    is historical truth.
  - `settled_transaction` points to the DEBIT `WalletTransaction` once
    settled, or null if UNPAID.
  - `product_code` + `source_module` identify what the charge is for
    (meal, tuition, transport) without a typed FK to those domains
    (avoiding circular imports).

### 4.4 Payment

- **Owner:** `apps.finance`
- **Purpose:** Money **received** into the wallet (a top-up source). A
  payment produces a TOPUP `WalletTransaction`. Distinct from a charge so
  that payment methods, receipts, and reconciliation are auditable.
- **Conceptual fields:**

```python
class Payment(models.Model):
    class Method(models.TextChoices):
        CASH = "cash", "Cash"
        BANK_TRANSFER = "bank_transfer", "Bank transfer"
        CARD = "card", "Card"
        WALLET_TRANSFER = "wallet_transfer", "Wallet transfer"
        OTHER = "other", "Other"

    wallet = models.ForeignKey(Wallet, on_delete=models.PROTECT, related_name="payments")
    person = models.ForeignKey("identity.Person", on_delete=models.PROTECT,
                               related_name="finance_payments")
    amount_iqd = models.IntegerField()
    method = models.CharField(max_length=20, choices=Method.choices, default=Method.CASH)
    reference = models.CharField(max_length=128, blank=True, default="",
        help_text="External receipt/transfer reference.")
    settled_transaction = models.ForeignKey(
        WalletTransaction, on_delete=models.PROTECT,
        related_name="settled_payments", null=True, blank=True,
    )
    received_at = models.DateTimeField()
    received_by = models.ForeignKey(
        "identity.StaffProfile", on_delete=models.SET_NULL,
        null=True, blank=True, related_name="received_payments",
    )
    notes = models.CharField(max_length=200, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-received_at"]
        indexes = [
            models.Index(fields=["wallet", "received_at"]),
            models.Index(fields=["method", "received_at"]),
        ]
```

- **Rules:**
  - A `Payment` is the **audit record of money in**. It references the
    TOPUP transaction it produced.
  - Finance records payments; it does **not** process card/bank
    transactions (that is a future payment-gateway boundary, Q8).

### 4.5 Refund

- **Owner:** `apps.finance`
- **Purpose:** Money returned to the wallet, reversing a prior charge. A
  refund produces a REFUND `WalletTransaction` that `reverses` the
  original DEBIT.
- **Conceptual fields:**

```python
class Refund(models.Model):
    wallet = models.ForeignKey(Wallet, on_delete=models.PROTECT, related_name="refunds")
    person = models.ForeignKey("identity.Person", on_delete=models.PROTECT,
                               related_name="finance_refunds")
    original_charge = models.ForeignKey(
        Charge, on_delete=models.PROTECT, related_name="refunds",
    )
    amount_iqd = models.IntegerField()
    reason_code = models.CharField(max_length=32, blank=True, default="")
    reason_notes = models.CharField(max_length=200, blank=True, default="")
    settled_transaction = models.ForeignKey(
        WalletTransaction, on_delete=models.PROTECT,
        related_name="settled_refunds", null=True, blank=True,
    )
    approved_by = models.ForeignKey(
        "identity.StaffProfile", on_delete=models.SET_NULL,
        null=True, blank=True, related_name="approved_refunds",
    )
    refunded_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-refunded_at"]
```

- **Rules:**
  - A refund references the original `Charge`, not just the transaction,
    so partial refunds and refund history per charge are queryable.
  - The refund transaction's `reverses` FK points to the original DEBIT
    transaction for the ledger reversal chain.

### 4.6 Adjustment

- **Owner:** `apps.finance`
- **Purpose:** A manual correction to the wallet balance that is not a
  charge, payment, or refund — e.g. fixing a data-entry error, a goodwill
  credit, a written-off debit. Produces an ADJUSTMENT `WalletTransaction`.
- **Conceptual fields:**

```python
class Adjustment(models.Model):
    wallet = models.ForeignKey(Wallet, on_delete=models.PROTECT, related_name="adjustments")
    person = models.ForeignKey("identity.Person", on_delete=models.PROTECT,
                               related_name="finance_adjustments")
    amount_iqd = models.IntegerField(
        help_text="Signed: positive = credit, negative = debit.",
    )
    reason_code = models.CharField(max_length=32, db_index=True)
    reason_notes = models.CharField(max_length=200, blank=True, default="")
    settled_transaction = models.ForeignKey(
        WalletTransaction, on_delete=models.PROTECT,
        related_name="settled_adjustments", null=True, blank=True,
    )
    approved_by = models.ForeignKey(
        "identity.StaffProfile", on_delete=models.SET_NULL,
        null=True, blank=True, related_name="approved_adjustments",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
```

- **Rules:**
  - Adjustments are the **escape hatch** for balance correction, but they
    are themselves audited ledger rows — they never silently mutate the
    balance.
  - Require `reason_code` and (by policy) staff approval.

### 4.7 PricingRule / PriceList (non-meal ERP placeholder)

- **Owner:** `apps.finance`
- **Purpose:** A **placeholder** for **non-meal** list-price definitions
  per product, period, and academic year (tuition, transport, library
  fines, school shop, event fees, uniforms). This generalizes the
  legacy `MealProfilePeriodPrice` *concept* to non-meal ERP billing.
- **Meals is not a consumer of `PriceList` / `PricingRule`.** Under the
  decided architecture, **meal product pricing lives in `apps.meals`**
  (`MealPeriodPrice`, `MealPersonPriceOverride`, `MealPlan.default_price_iqd`),
  not in Finance. The meals domain resolves meal prices itself and passes
  a pre-resolved `amount_iqd` to `finance.charge`. See
  `meals_domain_architecture.md` §14–§18. `PriceList` / `PricingRule`
  must not be used as the source of meal prices.
- **Conceptual fields:**

```python
class PriceList(models.Model):
    name = models.CharField(max_length=100, unique=True)
    currency = models.CharField(max_length=8, default="IQD")
    is_active = models.BooleanField(default=True)
    academic_year = models.ForeignKey(
        "academics.AcademicYear", on_delete=models.PROTECT,
        related_name="price_lists", null=True, blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class PricingRule(models.Model):
    price_list = models.ForeignKey(PriceList, on_delete=models.CASCADE, related_name="rules")
    product_code = models.CharField(max_length=64, db_index=True)
    price_iqd = models.IntegerField(default=0)
    is_enabled = models.BooleanField(default=True)
    # optional scope keys
    grade = models.ForeignKey("academics.Grade", on_delete=models.PROTECT,
                              related_name="pricing_rules", null=True, blank=True)
    period_template = models.ForeignKey(
        "scheduler.PeriodTemplate", on_delete=models.PROTECT,
        related_name="pricing_rules", null=True, blank=True,
    )
    notes = models.CharField(max_length=200, blank=True, default="")

    class Meta:
        unique_together = [["price_list", "product_code"]]
        ordering = ["price_list__name", "product_code"]
```

- **Rules:**
  - `PricingRule` returns the **list price** for a **non-meal** product in
    a context. It must not be used for meal products (`product_code`
    starting with `meal_*`); meal prices live in `apps.meals`.
  - Discounts for non-meal products may be applied on top by a
    `resolve_pricing` helper; for **meal** products, Meals resolves
    prices and discounts itself (see `meals_domain_architecture.md` §14
    and §19).
  - This is a placeholder for non-meal ERP billing; the meals domain's
    `MealPeriodPrice` / `MealPersonPriceOverride` / `MealPlan.default_price_iqd`
    are the source for meal prices, owned by `apps.meals`.

### 4.8 Relationship to StudentProfile

- `Wallet.person` → `Person` (the ledger anchor). A wallet belongs to a
  Person, not a StudentProfile, because staff/parents may eventually have
  wallets too.
- Where student-specific billing context is needed (e.g. grade-scoped
  tuition), `Charge.academic_year` + the meal/academics domain's
  enrollment provide the grade/section. Finance does **not** store
  `student` FKs on ledger rows; it stores `person`.
- Legacy `WalletTransaction.student` (FK to `attendance.Student`) migrates
  to `WalletTransaction.person` (FK to `identity.Person`), denormalized
  from `wallet.person`.

### 4.9 Relationship to Person

- Every wallet, transaction, charge, payment, refund, and adjustment
  carries a `person` FK to `identity.Person`.
- Person identity (name, code, photo) is read via `person.full_name` /
  `person.display_code`. **No** name/code fields are duplicated on finance
  entities.
- "Does this person have a wallet?" → `Wallet.objects.filter(person=p)`.
  "Is this person a student being billed?" combines `PersonRole(person,
  student)` with a `Charge` for a student-scoped `product_code`.

### 4.10 Future relationship to Enrollment

- `Charge.academic_year` scopes a charge to a year for per-year billing
  and reporting.
- Grade-scoped pricing (e.g. "Grade 10 tuition") uses
  `PricingRule.grade` + the student's active `StudentEnrollment.grade`
  (read from academics).
- Finance does **not** store `enrollment` FKs on ledger rows. Enrollment
  context is resolved at charge time from academics and snapshotted onto
  the immutable `Charge` (price/discount/grade) if needed.

---

## 5. Wallet Lifecycle

Modeled as a state machine. Each transition is a service method.

### 5.1 Status values

| Status | Meaning |
|---|---|
| `active` | Wallet is operational; charges/payments allowed. |
| `suspended` | Temporarily frozen (e.g. disputed balance, graduated-but-unsettled). No charges allowed; balance queries allowed. |
| `closed` | Permanently closed (e.g. student left the school). No charges/payments; ledger remains queryable forever. |

### 5.2 Valid transitions

```
active ──(suspend)──► suspended
suspended ──(reactivate)──► active
active/suspended ──(close)──► closed   (terminal)
```

### 5.3 Creation

- A wallet is created by `create_wallet(person, currency="IQD")` in
  `apps.finance.services`. It starts with `balance_iqd=0`,
  `status=active`, and an optional initial-top-up `Payment` that produces
  the first ledger row.
- A wallet is **never** created without at least the possibility of ledger
  rows; the ledger is the source of truth from creation.

### 5.4 Balance calculation

- `Wallet.balance_iqd` is a **cached projection**, updated atomically with
  each new `WalletTransaction`:
  `balance_after = balance_before + amount`.
- The authoritative balance is `sum(tx.amount_iqd for tx in
  wallet.transactions.all())`. A reconciliation service
  (`recompute_balance`) verifies the cache matches the ledger and is run
  periodically (open question Q9 on automated reconciliation).
- Direct writes to `balance_iqd` outside `services.charge/refund/top_up/
  adjust` are forbidden.

### 5.5 Ledger / history preservation

- `WalletTransaction` rows are append-only and immutable.
- A closed wallet's transactions remain forever queryable for audit and
  reporting.
- Balance is never "reset"; closing a wallet zeroes it via a final
  ADJUSTMENT if needed, with a reason code, leaving the ledger intact.

---

## 6. Transaction Model

### 6.1 Immutable ledger concept

- Every balance change is one `WalletTransaction` row.
- Rows are **never** updated or deleted (no `UPDATE`/`DELETE` on the
  ledger table from any code path). The model enforces this by convention
  and audit; a DB-level guard is open question Q9.
- `balance_before`/`balance_after` are snapshotted at creation and never
  changed, even by reversals (a reversal is a *new* row with its own
  before/after).

### 6.2 Credit / Debit

| `tx_type` | Sign of `amount_iqd` | Effect on balance |
|---|---|---|
| `topup` (Payment in) | positive | increases |
| `debit` (Charge) | negative | decreases |
| `refund` (reverses a debit) | positive | increases |
| `adjustment` | signed | increases or decreases |
| `unpaid` | zero | no effect (records intent) |

### 6.3 Source module

- `source_module` identifies the originating domain
  (`"meal"`, `"tuition"`, `"transport"`, `"manual_topup"`, ...).
- This decouples finance from those domains: finance does not import their
  models; it records a string tag and a generic reference.

### 6.4 Reference object

- `reference_type` + `reference_id` form a generic FK to the originating
  object (e.g. `("MealServiceEvent", 42)`). Finance does not resolve
  these to typed objects; the owning domain stores the reverse typed FK
  (`MealServiceEvent.wallet_transaction`) for its own joins.
- A typed FK from `WalletTransaction` to `MealServiceEvent` is
  **forbidden** (would create `finance → meals` circular dependency).

### 6.5 Audit notes

- Every transaction carries `reason_code`, `notes`, `created_by` (auth.User),
  and `created_by_staff` (StaffProfile). Meals-supervisor and finance-staff
  actions are auditable to a person.

### 6.6 Reversal strategy

- A reversal is a **new transaction** with `is_reversal=True` and
  `reverses=<original_id>`.
- The reversal's `amount_iqd` is the negation of the original's effect
  (e.g. a -5000 debit is reversed by a +5000 refund).
- The original transaction is **never** mutated. Its `balance_before`/
  `balance_after` remain the historical truth of that moment.
- Partial reversals are allowed (a refund of 2000 against a 5000 debit),
  with `reverses` pointing to the original; multiple reversals may
  reference one original.
- A reversal cannot itself be reversed directly; instead a new correction
  adjustment is created (avoids deep chains; open question Q7).

---

## 7. Meals Integration

### 7.1 Meals passes a pre-resolved charge to Finance

> Decided boundary: **Meals owns meal price resolution; Finance records
> pre-resolved charges only.** Finance does **not** resolve meal prices
> and does **not** call `apps.discounts` on the meal code path. Meals
> calls `apps.discounts` itself during `resolve_price`
> (`meals_domain_architecture.md` §14, §19) and passes the resulting
> `final_charge_iqd` to `finance.charge`.

- When a `MealServiceEvent` is confirmed and `MealPlan.mode == WALLET`,
  the meals domain resolves the price itself (per-Person / per-period /
  default), calls `apps.discounts` if available, and then calls Finance
  with a **pre-resolved** amount:
  ```python
  finance.charge(
      person=student.person,
      amount_iqd=final_charge_iqd,    # pre-resolved by Meals
      source_module="meals",
      reference_type="MealServiceEvent",
      reference_id=service_event.pk,
      academic_year=<current>,
      description=f"meal_lunch base={base} discount={discount_iqd}",
  ) -> WalletTransaction
  ```
- Finance validates the wallet is active and funds/credit suffice (or
  returns an UNPAID transaction per finance policy), appends the DEBIT
  row, updates `Wallet.balance_iqd` atomically inside the same
  transaction, and returns the `WalletTransaction` (with
  `balance_before`/`after` snapshots) to Meals.
- Finance does **not** look up `MealPeriodPrice` or
  `MealPersonPriceOverride`, does **not** consult `MealPlan`, and does
  **not** call `apps.discounts`. It records the pre-resolved amount.
- Meals stores the returned `WalletTransaction` FK and the
  price/discount/charge/balance snapshots on the immutable
  `MealServiceEvent` (`meals_domain_architecture.md` §12, §20).
- For `MealPlan.mode == DATE_RANGE` subscriptions, Meals makes **no**
  `finance.charge` call — the service event records `final_charge_iqd = 0`
  and `price_resolution_source = "date_range_no_charge"`
  (`meals_domain_architecture.md` §13 step 4a).

### 7.2 Meals must not own wallet calculations

- Meals never reads `Wallet.balance_iqd`, never creates
  `WalletTransaction` rows, never computes `balance_before`/`after`.
- Meals calls `finance.check_balance(person, amount)` to decide whether to
  proceed, and `finance.charge(...)` / `finance.refund(...)` to mutate.
- Wallet mutations happen only through Finance services.
- The "insufficient funds" policy (`MealPlan.insufficient_funds_mode`)
  is **meals-domain policy** (how the meal product reacts to a balance
  state returned by finance); it is not finance logic
  (`meals_domain_architecture.md` §13 step 4b, §17).

### 7.3 Finance service/selector boundary for balance and charge checks

```python
# apps/finance/services.py
def check_balance(*, person, amount_iqd: int = 0) -> tuple[int, bool]:
    """Return (current_balance, sufficient_for_amount)."""

def charge(*, person, amount_iqd: int, source_module: str,
           reference_type: str = "", reference_id: int | None = None,
           academic_year=None, description: str = "") -> WalletTransaction:
    """Record a pre-resolved charge. `amount_iqd` is the final amount to
    debit, resolved by the caller (e.g. Meals). Finance does NOT resolve
    pricing and does NOT call discounts for meal charges."""

def refund(*, person, original_transaction, amount_iqd=None, reason_code="",
           approved_by=None) -> WalletTransaction: ...

def top_up(*, person, amount_iqd, method="cash", reference="",
           received_by=None) -> tuple[Payment, WalletTransaction]: ...

def adjust(*, person, amount_iqd, reason_code, reason_notes="",
           approved_by=None) -> tuple[Adjustment, WalletTransaction]: ...

# apps/finance/selectors.py
def balance_for(*, person) -> int: ...
def transaction_history(*, person, since=None) -> QuerySet[WalletTransaction]: ...
def ledger_for_person(*, person) -> QuerySet[WalletTransaction]: ...
def charges_for_person(*, person, product_code=None) -> QuerySet[Charge]: ...
```

- These are the **only** entry points other domains use. They encapsulate
  the ledger math, the atomic balance update, and the audit snapshots.
- `charge()` takes `amount_iqd` (the pre-resolved final amount), **not**
  `price_base_iqd` — Finance does not resolve prices. (Earlier drafts of
  this document had `charge(..., price_base_iqd=...)` resolving pricing
  internally; that design is superseded by the Meals v1.1 architecture.)

---

## 8. Discount Integration

> Decided boundary (revised): **Finance does not call `apps.discounts`
> for meal pricing.** Meals calls `apps.discounts` directly during its
> own `resolve_price` and passes the pre-resolved `final_charge_iqd` to
> `finance.charge`. Finance may still snapshot a `discount_iqd` value on
> `Charge` when the caller supplies one (for audit self-containment),
> but it does not *compute* it. This section is retained to document
> the boundary and the snapshot semantics; the discount *resolution*
> ownership has moved to Meals (for meal products) and remains open for
> future non-meal ERP billing modules.

### 8.1 Discounts are a separate domain, clearly bounded

- Discount **profiles, rules, and assignments** do **not** live in
  `apps.finance`. They belong to a future separate discount domain/app,
  likely `apps.discounts`, which owns `DiscountProfile`, `DiscountRule`,
  and a future `DiscountAssignment` to `Person`/`Family`/`Grade` scoped by
  `AcademicYear` (migrating the legacy `DiscountProfile`/`DiscountRule`
  from `apps.attendance` when that domain is built).
- Finance **does not own** `DiscountProfile`, `DiscountRule`, or
  `DiscountAssignment`. Finance owns the `Charge` (which may store discount
  *snapshot* fields — see 8.3) and the ledger, not discount rule logic.
- **Meals calls `apps.discounts` for meal pricing**, not Finance. Finance
  records the pre-resolved amount. For future **non-meal** ERP billing
  modules that do not perform their own discount resolution, the question
  of who calls `apps.discounts` is deferred (open question Q14).
- Discount implementation is **deferred** until after the Finance + Meals
  foundation is in place, unless a concrete requirement forces it
  earlier. Until then, the Meals `resolve_price` discount call is a no-op
  stub returning `discount_iqd=0` (`meals_domain_architecture.md` §19.6).

### 8.2 Meals consumes discount results; Finance records the pre-resolved amount

- When the discount domain exists, **Meals** calls it during
  `resolve_price` (`meals_domain_architecture.md` §14.2, §19):
  ```python
  discounts.resolve(person, price_base, product_code,
                     academic_year, context) -> (discount_iqd,
                     final_charge_iqd, applied_profiles)
  ```
  This function lives in `apps.discounts.services`, **not** in
  `apps.finance` and **not** in `apps.meals` (Meals imports the discount
  service interface, not the discount models).
- Meals passes the resulting `final_charge_iqd` to `finance.charge(...)`.
  Finance records the pre-resolved amount and may snapshot the
  `discount_iqd` on `Charge` if Meals supplies it (for audit
  self-containment). Finance does **not** call `apps.discounts`.
- The result is also snapshotted on Meals' immutable `MealServiceEvent`
  (`meals_domain_architecture.md` §12).
- Until `apps.discounts` exists, Meals' `resolve_price` skips the
  discount call and charges the full `price_base_iqd`; Finance sees
  `amount_iqd = price_base_iqd`.
- **Finance does not call `apps.discounts` on the meal code path.** An
  earlier draft of this section described `finance.charge` calling the
  discount service internally; that design is superseded by the Meals
  v1.1 architecture.

### 8.3 Charge may store discount snapshots, not discount rule logic

- `Charge` may store **snapshot** fields: `discount_iqd`,
  `discount_reason_code`, and `discount_notes` (the resolved amount and a
  human-readable reason at charge time). These are immutable once the
  charge is settled.
- `Charge` does **not** store discount-rule IDs, discount-profile FKs, or
  discount-rule logic. Hardcoding discount rules on the charge/transaction
  would couple finance to discount-rule versions and break immutability (a
  rule rename would corrupt historical rows).
- `WalletTransaction` stores `amount_iqd` (the signed final effect) and
  the `reference` to the `Charge` that broke down price/discount/final. It
  stores no discount fields at all.

### 8.4 Meals calls Discounts directly; Finance does not

- The meals domain calls `apps.discounts` directly during `resolve_price`
  and receives the resolved `discount_iqd`; it never asks Finance to
  compute discounts. Finance records the pre-resolved `amount_iqd` only.
- Meals stores the discount *snapshot* on its immutable
  `MealServiceEvent` for audit; it does not own the rules that produced
  it (`meals_domain_architecture.md` §19.4).

### 8.5 Discount types (resolved by the discount domain, not finance)

The discount domain (`apps.discounts`) is responsible for resolving these
types; finance only receives the final `discount_iqd`:

| Type | Resolved by |
|---|---|
| Fixed amount | `apps.discounts` (`discount_iqd = min(value, price_base)`) |
| Percentage | `apps.discounts` (`discount_iqd = price_base * value_percent / 100`) |
| Subsidy / scholarship | `apps.discounts` via `DiscountAssignment` to Person/Family/Grade |
| Stacking / exclusive | `apps.discounts` (`DiscountProfile.priority` + stacking policy; open question Q6) |

---

## 9. Future ERP Support

The finance domain is designed so future billing modules plug in via the
**same service boundary** meals uses, without finance needing to know
they exist.

| Future module | Product code example | Integration |
|---|---|---|
| Meals (current) | `meal_lunch`, `meal_breakfast` | Meals resolves price + discounts itself, passes `amount_iqd` (pre-resolved) to `finance.charge`. See `meals_domain_architecture.md`. |
| Tuition | `tuition_2026_term1` | Calls `finance.charge(person, amount_iqd, source_module="tuition", ...)`; the module resolves its own price (or uses `PricingRule`) and passes a pre-resolved amount. Finance settles from wallet or records UNPAID for invoiced billing. |
| Transportation | `transport_term1` | Same pattern; `PricingRule.grade` may scope bus-fee-by-grade for non-meal products. |
| Library fines | `library_fine_overdue` | Same pattern; `source_module="library"`. |
| School shop | `shop_item_42` | Same pattern; `source_module="shop"`. |
| Event fees | `event_field_trip_7` | Same pattern; `source_module="events"`. |
| Uniforms | `uniform_size_M` | Same pattern; `source_module="uniforms"`. |

### Why this works

- Each module calls `finance.charge(...)` with its own `source_module` and
  a **pre-resolved** `amount_iqd`. Finance records the ledger and returns
  the transaction. (For meal products, Meals resolves the price itself;
  for non-meal products, the calling module may use `PricingRule` or its
  own price source.)
- No new finance model is needed per module; `Charge.product_code` +
  `source_module` distinguish them.
- Reports group by `source_module`/`product_code` without finance importing
  any module's models.
- A future `Invoice`/`Receipt` module (open question Q5) can aggregate
  multiple `Charge` rows per billing period.

---

## 10. Recommended Model Boundaries

### Belongs in `apps.finance`

| Belongs in finance | Why |
|---|---|
| `Wallet`, `WalletTransaction` | Ledger and balance anchor. |
| `Charge`, `Payment`, `Refund`, `Adjustment` | Billing/money-in/money-out/correction records. |
| `PriceList`, `PricingRule` | **Non-meal** ERP list-price definitions (tuition, transport, etc.). Meal prices live in `apps.meals`. |
| `charge()`, `refund()`, `top_up()`, `adjust()`, `check_balance()` services | Ledger math lives here. `charge()` records **pre-resolved** amounts; it does not resolve meal prices or call discounts for meals. |

### Does NOT belong in `apps.finance`

| Does NOT belong in finance | Where it belongs |
|---|---|
| Meal subscriptions / eligibility / service events | `apps.meals` (`meals_domain_architecture.md`) |
| **Meal product pricing** (`MealPeriodPrice`, `MealPersonPriceOverride`, `MealPlan.default_price_iqd`) | `apps.meals` — Meals owns meal price resolution |
| Meal periods / meal entitlement / date-range vs wallet mode | `apps.meals` |
| Recognition events/cameras/embeddings | `apps.attendance` |
| Grade/Section/Enrollment/Placement | `apps.academics` |
| Person/StudentProfile/StaffProfile identity | `apps.identity` |
| Period templates/occurrences | `apps.scheduler` |
| **Discount profiles/rules/assignments** | future `apps.discounts` (Meals calls it for meal pricing; finance does not call it on the meal code path) |
| Invoice/Receipt aggregation (future) | future `apps.finance.Invoice` or a separate billing app (Q5) |
| Bank/payment-gateway processing | future payment-gateway connector (Q8) |
| Payroll / staff salary | future `apps.hr` / payroll |

### Boundary justification

- Placing `Wallet.balance` writes in the meals domain would couple meals
  to the ledger and bypass audit. Hence balance mutation is finance-only
  (wallet mutations happen only through Finance services).
- Placing discount rules on `WalletTransaction` would corrupt historical
  rows when rules change. Hence the breakdown lives on `Charge`.
- Placing meal product pricing (`MealPeriodPrice`,
  `MealPersonPriceOverride`) in Finance would couple Finance to
  meal-product concepts and break the "Finance records money only"
  invariant. Hence meal price resolution lives in `apps.meals`.
- A typed FK from `WalletTransaction` to `MealServiceEvent` would create
  `finance → meals` circular dependency. Hence generic `reference_type`/
  `reference_id` + the reverse typed FK on the meals domain.

---

## 11. What Should NOT Be Implemented Yet

| Item | Why deferred |
|---|---|
| The `apps.finance` app itself | Design-only. Implementation needs a separate approved task following Section 14. |
| `Invoice` / `Receipt` aggregation | Defer until multi-charge billing periods are needed (Q5). |
| `DiscountAssignment` to Person/Family/Grade | Defer until family/guardian domain exists; start with `DiscountProfile` + `DiscountRule` migrated from legacy. |
| Payment-gateway integration (card/bank) | Defer until online payments are required (Q8). |
| Decimal money fields | Stay with integer IQD (legacy); Decimal is a future migration (Q11). |
| DB-level immutability triggers on ledger | Start with service-layer + audit; add triggers only if needed (Q9). |
| Automated reconciliation cron | Implement `recompute_balance` service; auto-run is deferred (Q9). |
| Multi-currency support | Single currency (IQD) for now; multi-currency is a future expansion. |
| Payroll / staff salary | Out of scope; future HR/payroll module. |
| The `apps.discounts` domain (DiscountProfile/DiscountRule/DiscountAssignment) | Deferred until after Finance + Meals foundation unless a requirement forces it earlier. **Meals** calls it for meal pricing; Finance does not call it on the meal code path. Meals resolves with `discount_iqd=0` until then. |
| Removing legacy `Wallet`/`WalletTransaction` | Keep through dual-FK migration; remove only after verification. |
| Removing legacy `DiscountProfile`/`DiscountRule` | Keep until `apps.discounts` is built and migration is verified. |
| Typed FKs from `WalletTransaction` to calling-domain models | Forbidden by design (circular ownership); always use generic reference. |
| Meal product pricing in `PriceList`/`PricingRule` | Forbidden by the decided architecture. Meal prices live in `apps.meals` (`MealPeriodPrice`, `MealPersonPriceOverride`, `MealPlan.default_price_iqd`). |

---

## 12. Migration Safety Notes

When the finance app is eventually implemented, these rules apply. They
are **not** actions for this document.

### General rules

- **Additive first.** The finance app creates new tables only. It does not
  alter `Person` or `StudentProfile`. Legacy `Wallet`/`WalletTransaction`
  remain.
- **Dual-FK pattern** for migrating off legacy `Student`:
  1. Add `person` FK (nullable) to the new finance models.
  2. Data migration: populate `person_id` from
     `legacy_wallet.student.migrated_to.person` (via StudentProfile).
  3. Enforce NOT NULL.
  4. Flip read paths.
  5. Later release: drop legacy `Wallet`/`WalletTransaction`.
- **PROTECT** on wallet FKs so historical transactions survive wallet
  closure/person deactivation.
- **One logical feature per migration.** Do not combine the finance app's
  `0001_initial` with unrelated schema changes.
- **Run `python manage.py makemigrations --check --dry-run` and
  `python manage.py check`** after each step.
- **Tests run under the CREATEDB-enabled role** (`TEST_DB_USER` /
  `TEST_DB_PASSWORD`), already configured in `bisk/settings.py`. New
  migrations must pass `python manage.py test apps.finance` (and
  `apps.identity`) before merge.
- **Backup the production database** before backfilling wallets/
  transactions from legacy tables.

### Ledger immutability

- Once a `WalletTransaction` is created, no code path may `UPDATE` or
  `DELETE` it. Enforce in service methods (only `create` is allowed) and
  via admin `readonly_fields`.
- Reversals create new rows; they never mutate the original.
- A reconciliation service verifies `Wallet.balance == sum(transactions)`;
  any drift is a bug to fix, not a balance to silently overwrite.

### Rollback

- Because the finance app is purely additive (new tables only), rollback
  means un-applying its migrations and removing the app from
  `INSTALLED_APPS`. No existing table is altered, so rollback is safe
  before the legacy removal step.

---

## 13. Open Questions / Decisions Needed

1. **Wallet owner: Person vs StudentProfile.** This document recommends
   `Wallet.person` (Person) so staff/parents can have wallets too.
   Confirm vs. legacy `Wallet.student`. Recommendation: `person`.

2. **Balance cache vs. computed.** Keep `Wallet.balance_iqd` as a cached
   projection (fast reads) or compute from the ledger on every query?
   Recommendation: cache, with periodic reconciliation.

3. **Credit limit per-wallet vs. per-product.** Should negative balances be
   allowed per wallet (`credit_limit_iqd`) or per product/plan (meal plan
   allows negative, tuition does not)? Affects `charge()` validation.

4. **Charge settlement timing.** Should `charge()` always settle
   immediately (DEBIT or UNPAID), or support a `PENDING` charge settled
   later (e.g. end-of-day batch)? Affects the `Charge` state machine.

5. **Invoice/Receipt model.** Should `Invoice` aggregate multiple charges
   per billing period, or is `Charge` sufficient until a real invoicing
   requirement exists? Defer.

6. **Discount stacking.** When multiple discounts apply, do they stack
   (additive) or is only the highest-priority applied (exclusive)?
   Affects `resolve_discount`.

7. **Reversal-of-reversal.** Should reversing a reversal be allowed, or
   must corrections use a fresh `Adjustment`? Recommendation: fresh
   adjustment to avoid deep chains.

8. **Payment-gateway boundary.** Where does card/bank processing live? A
   future `apps.payments` connector that calls `finance.top_up` on
   confirmation? Defer until online payments are needed.

9. **Ledger immutability enforcement.** Service-layer convention only, or
   DB-level triggers/row-level security, or a `post_save` guard signal?
   Recommendation: service-layer + audit first; add DB guards only if
   violations appear.

10. **`created_by` (auth.User) vs `created_by_staff` (StaffProfile).**
    Which is required on a transaction? Recommendation: `created_by`
    (auth.User) required for system actions; `created_by_staff` optional
    until staff-assignment modeling exists.

11. **Integer IQD vs Decimal money.** Legacy uses `IntegerField`. Should
    finance migrate to `DecimalField(max_digits=12, decimal_places=3)`?
    Affects all money fields. Recommendation: stay integer until a
    fractional-currency requirement appears; do not mix.

12. **Generic reference vs. typed FK with `null=True`.** The generic
    `reference_type`/`reference_id` avoids circular imports but loses
    referential integrity. Is a set of nullable typed FKs
    (`meal_service_event`, `tuition_invoice`, ...) preferable? Trade-off:
    integrity vs. coupling. Recommendation: generic reference; the calling
    domain holds the reverse typed FK.

13. **AcademicYear on Charge.** Should `Charge.academic_year` be required
    or nullable? Recommendation: nullable until academics is rolled out,
    then recommended for year-scoped reporting.

14. **Discount caller for non-meal ERP billing (revised).** Under the
    decided architecture, Meals calls `apps.discounts` for meal pricing
    and Finance records the pre-resolved amount. For future **non-meal**
    ERP billing modules (tuition, transport, etc.) that do not perform
    their own discount resolution, who calls `apps.discounts`? Options:
    (a) each non-meal module calls `apps.discounts` itself (consistent
    with Meals); (b) Finance exposes a `resolve_pricing` helper for
    non-meal modules that calls `apps.discounts`. Recommendation: (a)
    for consistency — each billing module resolves its own discounts and
    passes a pre-resolved amount to `finance.charge`. Defer the final
    call until a non-meal billing module is implemented.

---

## 14. Recommended Implementation Order

Each step is a **separate, approved task**. This document implements
nothing.

| Step | Deliverable | Depends on | Notes |
|---|---|---|---|
| 0 | Identity + Academics foundations | — | **DONE (identity)** / **DESIGNED (academics)**. Finance needs `Person` and `AcademicYear`. |
| 1 | `apps.finance` app skeleton (`apps.py`, empty `models.py`, `INSTALLED_APPS`) | identity | Additive; no models yet. |
| 2 | `Wallet` model (FK to `Person`) + admin + `create_wallet` service | step 1 | Ledger anchor. |
| 3 | `WalletTransaction` model (immutable ledger, generic reference, reversal chain) + admin | step 2 | The core ledger row. |
| 4 | `services.charge` / `refund` / `top_up` / `adjust` / `check_balance` + `selectors.balance_for` / `transaction_history` + validators | steps 2–3 | The service boundary other domains call. |
| 5 | `Charge`, `Payment`, `Refund`, `Adjustment` models + admin | step 4 | Billing/money records referencing transactions. |
| 6 | `PriceList` / `PricingRule` (placeholder) + `resolve_pricing` | step 5 | Generalized list-price source. |
| 7 | Meals integration: meals calls `finance.charge` (with pre-resolved `amount_iqd`) / `check_balance` / `refund` (replace dry-run); `discount_iqd=0` until `apps.discounts` exists | step 4, `apps.meals` | Live meal billing without discounts. Finance records pre-resolved charges; Meals owns price resolution. |
| 8 | Wallet lifecycle services (`suspend`, `reactivate`, `close`) + `recompute_balance` reconciliation | step 4 | Wallet state machine + audit. |
| 9 | (Later phase) `apps.discounts` domain: `DiscountProfile` / `DiscountRule` / `DiscountAssignment` migrated from `apps.attendance` + `discounts.resolve(...)` service boundary | Finance + Meals foundations verified | Discount resolution. **Meals** calls it for meal pricing; Finance does not call it on the meal code path. |
| 10 | Meals ↔ discounts integration: Meals' `resolve_price` calls `discounts.resolve(...)`, snapshots `discount_iqd` on `MealServiceEvent`, passes pre-resolved `final_charge_iqd` to `finance.charge` | step 9 | Live discounts in meal billing. (Finance's `Charge` snapshots `discount_iqd` only when Meals supplies it.) |
| 11 | (Later phase) `Invoice`/`Receipt` aggregation | step 5 | Multi-charge billing periods (Q5). |
| 12 | (Later phase) Payment-gateway connector | step 5 | Online card/bank payments (Q8). |
| 13 | (Later phase) Legacy `Wallet`/`WalletTransaction` removal | steps 2–8 verified | Drop legacy wallet tables. |
| 14 | (Later phase) Legacy `DiscountProfile`/`DiscountRule` removal | step 9 verified | Drop legacy discount tables. |

### Ordering rationale

- `Wallet` (step 2) and `WalletTransaction` (step 3) precede the service
  boundary (step 4) because the services operate on them.
- The service boundary (step 4) is the unblocker for meals integration
  (step 7); it can happen before `Charge`/`Payment`/`Refund`/`Adjustment`
  (step 5) by operating on raw transactions, but step 5 is recommended
  first for clean audit records.
- Meal billing (step 7) ships **without** discounts (`discount_iqd=0`);
  Meals resolves the price itself and passes a pre-resolved `amount_iqd`
  to `finance.charge`. The `apps.discounts` domain (step 9) is built only
  after the Finance + Meals foundation is verified, per the deferral rule
  in Section 8.1.
- Meals↔discounts integration (step 10) is the point at which Meals'
  `resolve_price` begins calling `discounts.resolve(...)` and passing
  the pre-resolved `final_charge_iqd` to `finance.charge`; until then,
  the discount call is a no-op in Meals. Finance never calls
  `discounts.resolve(...)` on the meal code path.
- Legacy removal (steps 13–14) is last and conditional on full
  verification; wallet legacy and discount legacy are removed
  independently.

---

## 15. How This Supports Future Licensing and Audit Requirements

### 15.1 Licensing

- A future `apps.licensing` module can charge license fees via the same
  `finance.charge(person, product_code="license_...", source_module=
  "licensing", ...)` boundary — no finance changes needed.
- Per-license billing is just another `source_module`/`product_code` in
  the ledger, queryable and auditable like any other charge.

### 15.2 Audit requirements

- **Immutable ledger:** every balance change is a permanent row with
  before/after balances, source, reference, actor, reason, and timestamp.
  This satisfies internal audit and external financial-review
  requirements.
- **Reversal chain:** corrections are explicit reversal rows, never
  silent edits, so the full history of any balance is reconstructable.
- **Reconciliation:** `recompute_balance` verifies the cached balance
  matches the ledger, detecting drift or tampering.
- **Source-module tagging:** every transaction names the system that
  created it (`meal`, `tuition`, `manual_topup`), enabling per-module
  audit reports.
- **Person-staffed actions:** `created_by`/`created_by_staff`/
  `approved_by` identify the human behind each balance change, supporting
  segregation-of-duties audits.
- **Snapshot on caller:** the calling domain (e.g. meals) stores an
  immutable financial snapshot (price/discount/charge/balance) on its own
  historical record, so meal-service audit and wallet-ledger audit are
  independently verifiable and cross-referenceable via
  `WalletTransaction.reference_id`.

### 15.3 External reporting

- The ledger's `source_module`/`product_code`/`academic_year` indexes
  support period reports, per-student statements, per-module revenue
  breakdowns, and year-over-year comparisons without finance importing any
  other domain's models.
- A future reporting module reads finance selectors (`ledger_for_person`,
  `charges_for_person`) and academics context to produce bursar/audit
  reports.

---

## Appendix A — Finance Domain Relationship Diagram (text)

```
                         auth.User (login identity)
                              │ optional O2O
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│                            Person                                 │
│  code  = global ERP identifier                                   │
│  display_code → StudentProfile.code / StaffProfile.code / code    │
└───────────────┬───────────────────────────────────────────────────┘
                │ O2O (wallet owner)
                ▼
           ┌─────────┐
           │ Wallet  │  (person, balance cache, status, credit_limit)
           └────┬────┘
                │ 1:N (immutable ledger)
                ▼
           ┌──────────────────────┐
           │ WalletTransaction    │  (tx_type, signed amount,
           │  amount_iqd (signed) │   before/after, source_module,
           │  balance_before/after│   reference_type/id, reverses,
           │  source_module       │   created_by/staff, reason)
           │  reference_type/id   │
           │  reverses (self-FK)  │
           └──────────┬───────────┘
                      │ settled_transaction (reverse: from Charge/Payment/Refund/Adjustment)
        ┌─────────────┼─────────────┬─────────────┐
        ▼             ▼             ▼             ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐
   │ Charge  │  │ Payment │  │ Refund  │  │ Adjustment  │
   │ product │  │ method  │  │ reverses│  │ signed      │
   │ price/  │  │ receipt │  │ charge  │  │ reason      │
   │ discount│  │         │  │         │  │ approved_by │
   └─────────┘  └─────────┘  └─────────┘  └─────────────┘

PriceList ──► PricingRule ──► (grade, period_template, product_code, price_iqd)

(future) apps.discounts:
  DiscountProfile ──► DiscountRule ──► (type, value, priority)
  DiscountAssignment ──► Person/Family/Grade + AcademicYear
  finance.charge() calls discounts.resolve(...) via service boundary

AcademicYear ──► Charge.academic_year (year-scoped billing)
```

All arrows point from the depending module to the owning module. Finance
owns the Wallet/WalletTransaction/Charge/Payment/Refund/Adjustment/
PricingRule entities; a future `apps.discounts` owns Discount* entities;
identity owns Person/StudentProfile/
StaffProfile; academics owns AcademicYear/Grade/Section; meal/attendance/
future modules call finance services and store reverse typed FKs to
WalletTransaction. Finance consumes discount results via a service
boundary; it does not own discount models.

---

## Appendix B — Ledger Immutability & Reversal Summary

| Operation | Ledger effect | Reversal |
|---|---|---|
| `top_up` (Payment) | +amount TOPUP row | Not typically reversed; if needed, an ADJUSTMENT debit. |
| `charge` (Charge settled) | -amount DEBIT row | A REFUND row (+amount, `reverses=debit`). |
| `charge` (Charge unpaid) | 0 UNPAID row | Void the Charge (status=VOIDED); no ledger reversal needed (amount was 0). |
| `refund` (Refund) | +amount REFUND row (`reverses=original debit`) | A new ADJUSTMENT debit (not a reversal of the refund). |
| `adjust` (Adjustment) | signed ADJUSTMENT row | A new ADJUSTMENT of opposite sign. |

**Rule:** no row is ever `UPDATE`d or `DELETE`d. Every correction is a new
row. The original row's `balance_before`/`balance_after` remain the
historical truth of that moment.

---

## Appendix C — Snapshot vs. Ledger Ownership

| Data | Owner | Why |
|---|---|---|
| Wallet balance (cached) | `apps.finance` (Wallet) | Fast reads; reconciled against ledger. |
| Transaction rows (authoritative) | `apps.finance` (WalletTransaction) | Immutable ledger; source of truth. |
| Price/discount/charge breakdown | `apps.finance` (Charge) | Immutable billing record per product instance. For meal products, Meals computes the breakdown and passes a pre-resolved `amount_iqd`; Finance snapshots it on `Charge` only when the caller supplies it. |
| Meal service financial snapshot | `apps.meals` (`MealServiceEvent`) | Immutable historical truth of the meal service; cross-references `WalletTransaction` via reverse typed FK. Owned by Meals, not Finance. |
| Meal product pricing (`MealPeriodPrice`, `MealPersonPriceOverride`, `MealPlan.default_price_iqd`) | `apps.meals` | Meals owns meal price resolution. Finance does not own meal prices. |
| Grade/section at charge time | `apps.academics` (read) / snapshot on `Charge` if needed | Finance does not own academic data. |
| Person identity (name/code) | `apps.identity` (Person) | Finance reads via `wallet.person.full_name`. |

No live (non-snapshot) identity, academic, meal, or discount-rule fields
exist on finance ledger entities other than the FKs needed to reference
the owning rows.

---

## Appendix D — Document Hierarchy

```
docs/development/PROJECT_ARCHITECTURE.md            (standards)
docs/development/AI_DEVELOPMENT_GUIDE.md            (AI workflow)
docs/development/DOMAIN_INTEGRATION_GUIDE.md       (cross-app integration rules)
docs/architecture/person_identity_architecture.md   (identity foundation — implemented)
docs/architecture/erp_foundation_architecture.md    (top-level ERP blueprint)
docs/architecture/education_domain_architecture.md  (education domain — broad)
docs/architecture/academics_domain_architecture.md  (academics domain — enrollment/section)
docs/architecture/meals_domain_architecture.md      (meals domain — AUTHORITATIVE for meals: app `apps.meals`, `MealPlan`, price resolution, resolver flow)
docs/architecture/lunch_domain_architecture.md      (legacy/transitional lunch design — superseded by meals_domain_architecture.md for the future meals app)
docs/architecture/finance_domain_architecture.md    (THIS document — finance/wallet/ledger)
```

This document is consistent with all of the above. Where they describe
implemented models, this document references them; where they describe
future models, this document narrows the finance subset and adds ledger,
wallet-lifecycle, meals/discount integration, ERP-billing, audit, and
open-question detail specific to the finance domain. The meals domain
pricing placement is governed by `meals_domain_architecture.md`, which
is authoritative where this document and it might otherwise disagree.

---

End of document.
