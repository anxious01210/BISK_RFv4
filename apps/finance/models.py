from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models


class Wallet(models.Model):
    """A prepaid financial account owned by one Person.

    ``balance_iqd`` is a **cached projection** of the immutable
    :class:`WalletTransaction` ledger. The ledger is the source of truth;
    this field is updated only inside the same atomic service call that
    appends a ledger row. Direct writes outside ``apps.finance.services``
    are forbidden.
    """

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        SUSPENDED = "suspended", "Suspended"
        CLOSED = "closed", "Closed"

    person = models.OneToOneField(
        "identity.Person",
        on_delete=models.PROTECT,
        related_name="wallet",
    )
    currency = models.CharField(max_length=8, default="IQD")
    balance_iqd = models.IntegerField(
        default=0,
        help_text="Cached projection of the ledger. Always equals sum(transactions.amount_iqd).",
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.ACTIVE,
        db_index=True,
    )
    credit_limit_iqd = models.IntegerField(
        default=0,
        help_text="Negative balance allowed down to -credit_limit (0 = no negative).",
    )
    notes = models.CharField(max_length=200, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["person__code"]
        indexes = [
            models.Index(fields=["status"]),
        ]
        constraints = [
            models.CheckConstraint(
                check=models.Q(credit_limit_iqd__gte=0),
                name="wallet_credit_limit_non_negative",
            ),
            models.CheckConstraint(
                check=models.Q(balance_iqd__gte=models.F("credit_limit_iqd") * -1),
                name="wallet_balance_within_credit_limit",
            ),
        ]

    def __str__(self) -> str:
        return f"Wallet {self.person} - {self.balance_iqd} {self.currency}"

    @property
    def available_balance_iqd(self) -> int:
        """Balance plus the authorized credit limit (may be negative)."""
        return self.balance_iqd + self.credit_limit_iqd

    def clean(self):
        super().clean()
        if self.currency and self.currency != self.currency.upper():
            raise ValidationError({"currency": "Currency code must be uppercase."})


class WalletTransaction(models.Model):
    """An immutable ledger row.

    Every balance change on a wallet is one row here. Rows are never
    edited or deleted; corrections create a new row with
    ``is_reversal=True`` pointing back via ``reverses``.
    """

    class TxType(models.TextChoices):
        TOPUP = "topup", "Top-up"
        DEBIT = "debit", "Debit"
        REFUND = "refund", "Refund"
        ADJUSTMENT = "adjustment", "Adjustment"
        UNPAID = "unpaid", "Unpaid"

    wallet = models.ForeignKey(
        Wallet,
        on_delete=models.PROTECT,
        related_name="transactions",
    )
    person = models.ForeignKey(
        "identity.Person",
        on_delete=models.PROTECT,
        related_name="wallet_transactions",
    )
    tx_type = models.CharField(
        max_length=20, choices=TxType.choices, db_index=True
    )
    amount_iqd = models.IntegerField(
        help_text="Signed. +topup, -debit, +refund, +/-adjustment, 0 unpaid.",
    )
    balance_before_iqd = models.IntegerField(default=0)
    balance_after_iqd = models.IntegerField(default=0)

    source_module = models.CharField(
        max_length=40, db_index=True,
        help_text="Originating domain, e.g. 'meal', 'tuition', 'manual_topup'.",
    )
    reference_type = models.CharField(max_length=40, blank=True, default="")
    reference_id = models.PositiveBigIntegerField(null=True, blank=True)

    reason_code = models.CharField(max_length=32, blank=True, default="")
    notes = models.CharField(max_length=200, blank=True, default="")

    is_reversal = models.BooleanField(default=False)
    reverses = models.ForeignKey(
        "self",
        on_delete=models.PROTECT,
        related_name="reversals",
        null=True,
        blank=True,
    )

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="finance_created_wallet_transactions",
    )
    created_by_staff = models.ForeignKey(
        "identity.StaffProfile",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="finance_created_wallet_transactions",
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

    def __str__(self) -> str:
        return f"{self.tx_type} {self.amount_iqd} for {self.person}"

    def clean(self):
        super().clean()
        if self.wallet_id is not None and self.person_id is not None:
            if self.wallet.person_id != self.person_id:
                raise ValidationError(
                    {"person": "Transaction person must match wallet person."}
                )


class Charge(models.Model):
    """A billing record — an amount owed for a product/service.

    Generic across modules (meal, tuition, transport, library, shop,
    events, uniforms). Identified by ``product_code`` + ``source_module``
    rather than a typed FK to the calling domain, so finance never imports
    those models.
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        SETTLED = "settled", "Settled"
        UNPAID = "unpaid", "Unpaid"
        REVERSED = "reversed", "Reversed"
        VOIDED = "voided", "Voided"

    wallet = models.ForeignKey(
        Wallet, on_delete=models.PROTECT, related_name="charges"
    )
    person = models.ForeignKey(
        "identity.Person",
        on_delete=models.PROTECT,
        related_name="finance_charges",
    )
    academic_year = models.ForeignKey(
        "academics.AcademicYear",
        on_delete=models.PROTECT,
        related_name="charges",
        null=True,
        blank=True,
    )
    product_code = models.CharField(max_length=64, db_index=True)
    description = models.CharField(max_length=200, blank=True, default="")
    price_base_iqd = models.IntegerField(default=0)
    discount_iqd = models.IntegerField(default=0)
    discount_reason_code = models.CharField(max_length=32, blank=True, default="")
    discount_notes = models.CharField(max_length=200, blank=True, default="")
    final_charge_iqd = models.IntegerField(default=0)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
        db_index=True,
    )
    settled_transaction = models.ForeignKey(
        WalletTransaction,
        on_delete=models.PROTECT,
        related_name="settled_charges",
        null=True,
        blank=True,
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
        constraints = [
            models.CheckConstraint(
                check=models.Q(price_base_iqd__gte=0),
                name="charge_price_base_non_negative",
            ),
            models.CheckConstraint(
                check=models.Q(discount_iqd__gte=0),
                name="charge_discount_non_negative",
            ),
            models.CheckConstraint(
                check=models.Q(final_charge_iqd__gte=0),
                name="charge_final_non_negative",
            ),
            models.CheckConstraint(
                check=models.Q(discount_iqd__lte=models.F("price_base_iqd")),
                name="charge_discount_within_price",
            ),
            models.CheckConstraint(
                check=models.Q(
                    final_charge_iqd__exact=models.F("price_base_iqd")
                    - models.F("discount_iqd")
                ),
                name="charge_final_equals_price_minus_discount",
            ),
        ]

    def __str__(self) -> str:
        return f"Charge {self.product_code} {self.final_charge_iqd} for {self.person}"

    def clean(self):
        super().clean()
        if self.wallet_id is not None and self.person_id is not None:
            if self.wallet.person_id != self.person_id:
                raise ValidationError(
                    {"person": "Charge person must match wallet person."}
                )


class Payment(models.Model):
    """Money received into a wallet. Produces a TOPUP ledger row."""

    class Method(models.TextChoices):
        CASH = "cash", "Cash"
        BANK_TRANSFER = "bank_transfer", "Bank transfer"
        CARD = "card", "Card"
        WALLET_TRANSFER = "wallet_transfer", "Wallet transfer"
        OTHER = "other", "Other"

    wallet = models.ForeignKey(
        Wallet, on_delete=models.PROTECT, related_name="payments"
    )
    person = models.ForeignKey(
        "identity.Person",
        on_delete=models.PROTECT,
        related_name="finance_payments",
    )
    amount_iqd = models.IntegerField()
    method = models.CharField(
        max_length=20, choices=Method.choices, default=Method.CASH
    )
    reference = models.CharField(
        max_length=128, blank=True, default="",
        help_text="External receipt/transfer reference.",
    )
    settled_transaction = models.ForeignKey(
        WalletTransaction,
        on_delete=models.PROTECT,
        related_name="settled_payments",
        null=True,
        blank=True,
    )
    received_at = models.DateTimeField()
    received_by = models.ForeignKey(
        "identity.StaffProfile",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="received_payments",
    )
    notes = models.CharField(max_length=200, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-received_at"]
        indexes = [
            models.Index(fields=["wallet", "received_at"]),
            models.Index(fields=["method", "received_at"]),
        ]
        constraints = [
            models.CheckConstraint(
                check=models.Q(amount_iqd__gt=0),
                name="payment_amount_positive",
            ),
        ]

    def __str__(self) -> str:
        return f"Payment {self.amount_iqd} via {self.method} for {self.person}"

    def clean(self):
        super().clean()
        if self.wallet_id is not None and self.person_id is not None:
            if self.wallet.person_id != self.person_id:
                raise ValidationError(
                    {"person": "Payment person must match wallet person."}
                )


class Refund(models.Model):
    """Money returned to a wallet, reversing a prior Charge's DEBIT row."""

    wallet = models.ForeignKey(
        Wallet, on_delete=models.PROTECT, related_name="refunds"
    )
    person = models.ForeignKey(
        "identity.Person",
        on_delete=models.PROTECT,
        related_name="finance_refunds",
    )
    original_charge = models.ForeignKey(
        Charge, on_delete=models.PROTECT, related_name="refunds"
    )
    amount_iqd = models.IntegerField()
    reason_code = models.CharField(max_length=32, blank=True, default="")
    reason_notes = models.CharField(max_length=200, blank=True, default="")
    settled_transaction = models.ForeignKey(
        WalletTransaction,
        on_delete=models.PROTECT,
        related_name="settled_refunds",
        null=True,
        blank=True,
    )
    approved_by = models.ForeignKey(
        "identity.StaffProfile",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="approved_refunds",
    )
    refunded_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-refunded_at"]
        indexes = [
            models.Index(fields=["wallet", "refunded_at"]),
            models.Index(fields=["original_charge", "refunded_at"]),
        ]
        constraints = [
            models.CheckConstraint(
                check=models.Q(amount_iqd__gt=0),
                name="refund_amount_positive",
            ),
        ]

    def __str__(self) -> str:
        return f"Refund {self.amount_iqd} for {self.person}"

    def clean(self):
        super().clean()
        if self.wallet_id is not None and self.person_id is not None:
            if self.wallet.person_id != self.person_id:
                raise ValidationError(
                    {"person": "Refund person must match wallet person."}
                )
        if self.original_charge_id is not None and self.wallet_id is not None:
            if self.original_charge.wallet_id != self.wallet_id:
                raise ValidationError(
                    {"original_charge": "Refunded charge must belong to the same wallet."}
                )


class Adjustment(models.Model):
    """A manual accounting correction. Produces an ADJUSTMENT ledger row."""

    wallet = models.ForeignKey(
        Wallet, on_delete=models.PROTECT, related_name="adjustments"
    )
    person = models.ForeignKey(
        "identity.Person",
        on_delete=models.PROTECT,
        related_name="finance_adjustments",
    )
    amount_iqd = models.IntegerField(
        help_text="Signed: positive = credit, negative = debit.",
    )
    reason_code = models.CharField(max_length=32, db_index=True)
    reason_notes = models.CharField(max_length=200, blank=True, default="")
    settled_transaction = models.ForeignKey(
        WalletTransaction,
        on_delete=models.PROTECT,
        related_name="settled_adjustments",
        null=True,
        blank=True,
    )
    approved_by = models.ForeignKey(
        "identity.StaffProfile",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="approved_adjustments",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["wallet", "created_at"]),
            models.Index(fields=["reason_code", "created_at"]),
        ]
        constraints = [
            models.CheckConstraint(
                check=~models.Q(amount_iqd__exact=0),
                name="adjustment_amount_nonzero",
            ),
        ]

    def __str__(self) -> str:
        return f"Adjustment {self.amount_iqd} ({self.reason_code}) for {self.person}"

    def clean(self):
        super().clean()
        if self.wallet_id is not None and self.person_id is not None:
            if self.wallet.person_id != self.person_id:
                raise ValidationError(
                    {"person": "Adjustment person must match wallet person."}
                )


class PriceList(models.Model):
    """A generic, admin-configurable collection of list prices.

    Future ERP modules (tuition, transport, library, shop, events,
    uniforms) read prices from :class:`PricingRule` rows attached to a
    ``PriceList`` rather than carrying their own price tables.
    """

    name = models.CharField(max_length=100, unique=True)
    currency = models.CharField(max_length=8, default="IQD")
    is_active = models.BooleanField(default=True, db_index=True)
    academic_year = models.ForeignKey(
        "academics.AcademicYear",
        on_delete=models.PROTECT,
        related_name="price_lists",
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        indexes = [
            models.Index(fields=["is_active", "academic_year"]),
        ]

    def __str__(self) -> str:
        return self.name

    def clean(self):
        super().clean()
        if self.currency and self.currency != self.currency.upper():
            raise ValidationError({"currency": "Currency code must be uppercase."})


class PricingRule(models.Model):
    """A list-price definition for one ``product_code`` within a
    :class:`PriceList`.

    Optional scope keys (``grade``, ``period_key``) narrow a rule to a
    specific academic context. Discounts are NOT owned by finance and are
    not represented here; they are resolved by the future ``apps.discounts``
    domain at charge time.
    """

    price_list = models.ForeignKey(
        PriceList, on_delete=models.CASCADE, related_name="rules"
    )
    product_code = models.CharField(max_length=64, db_index=True)
    price_iqd = models.IntegerField(default=0)
    is_enabled = models.BooleanField(default=True)
    grade = models.ForeignKey(
        "academics.Grade",
        on_delete=models.PROTECT,
        related_name="pricing_rules",
        null=True,
        blank=True,
    )
    period_key = models.CharField(
        max_length=64, blank=True, default="",
        help_text="Opaque scope key for the period context (e.g. 'lunch', 'term1').",
    )
    notes = models.CharField(max_length=200, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [["price_list", "product_code"]]
        ordering = ["price_list__name", "product_code"]
        indexes = [
            models.Index(fields=["product_code", "is_enabled"]),
            models.Index(fields=["grade", "is_enabled"]),
        ]
        constraints = [
            models.CheckConstraint(
                check=models.Q(price_iqd__gte=0),
                name="pricing_rule_price_non_negative",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.price_list} / {self.product_code} = {self.price_iqd}"
