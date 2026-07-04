from django.contrib import admin

from .models import (
    Adjustment,
    Charge,
    Payment,
    PriceList,
    PricingRule,
    Refund,
    Wallet,
    WalletTransaction,
)


@admin.register(Wallet)
class WalletAdmin(admin.ModelAdmin):
    list_display = (
        "person",
        "currency",
        "balance_iqd",
        "credit_limit_iqd",
        "available_balance_iqd",
        "status",
        "updated_at",
    )
    list_display_links = ("person",)
    list_filter = ("status", "currency", "created_at", "updated_at")
    search_fields = (
        "person__code",
        "person__first_name",
        "person__last_name",
        "notes",
    )
    ordering = ("person__code",)
    date_hierarchy = "created_at"
    readonly_fields = ("created_at", "updated_at", "balance_iqd")
    autocomplete_fields = ("person",)
    fieldsets = (
        ("Owner", {"fields": ("person",)}),
        ("Money", {"fields": ("currency", "balance_iqd", "credit_limit_iqd")}),
        ("State", {"fields": ("status", "notes")}),
        ("Timestamps", {"fields": ("created_at", "updated_at"), "classes": ("collapse",)}),
    )

    @admin.display(description="Available")
    def available_balance_iqd(self, obj):
        return obj.available_balance_iqd

    def has_delete_permission(self, request, obj=None):
        # Wallets should not be deleted while they may carry ledger history.
        return False


@admin.register(WalletTransaction)
class WalletTransactionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "wallet",
        "person",
        "tx_type",
        "amount_iqd",
        "balance_before_iqd",
        "balance_after_iqd",
        "source_module",
        "is_reversal",
        "created_at",
    )
    list_display_links = ("id", "wallet")
    list_filter = (
        "tx_type",
        "source_module",
        "is_reversal",
        "created_at",
    )
    search_fields = (
        "wallet__person__code",
        "wallet__person__first_name",
        "wallet__person__last_name",
        "source_module",
        "reference_type",
        "reason_code",
        "notes",
    )
    ordering = ("-created_at", "-id")
    date_hierarchy = "created_at"
    # The ledger is immutable after creation.
    readonly_fields = (
        "wallet",
        "person",
        "tx_type",
        "amount_iqd",
        "balance_before_iqd",
        "balance_after_iqd",
        "source_module",
        "reference_type",
        "reference_id",
        "reason_code",
        "notes",
        "is_reversal",
        "reverses",
        "created_by",
        "created_by_staff",
        "created_at",
    )
    autocomplete_fields = ("wallet", "person")
    fieldsets = (
        ("Ledger", {"fields": ("wallet", "person", "tx_type", "amount_iqd")}),
        ("Balance snapshot", {"fields": ("balance_before_iqd", "balance_after_iqd")}),
        ("Provenance", {"fields": ("source_module", "reference_type", "reference_id")}),
        ("Reason", {"fields": ("reason_code", "notes")}),
        ("Reversal", {"fields": ("is_reversal", "reverses"), "classes": ("collapse",)}),
        ("Audit", {"fields": ("created_by", "created_by_staff", "created_at"), "classes": ("collapse",)}),
    )

    def has_add_permission(self, request):
        # Ledger rows are created only via the finance service layer.
        return False

    def has_change_permission(self, request, obj=None):
        # Immutable after creation.
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(Charge)
class ChargeAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "person",
        "product_code",
        "source_module",
        "price_base_iqd",
        "discount_iqd",
        "final_charge_iqd",
        "status",
        "academic_year",
        "created_at",
    )
    list_display_links = ("id", "person")
    list_filter = ("status", "source_module", "product_code", "academic_year", "created_at")
    search_fields = (
        "person__code",
        "person__first_name",
        "person__last_name",
        "product_code",
        "source_module",
        "reference_type",
        "description",
    )
    ordering = ("-created_at",)
    date_hierarchy = "created_at"
    readonly_fields = ("created_at", "updated_at", "settled_transaction")
    autocomplete_fields = ("wallet", "person", "academic_year")
    fieldsets = (
        ("Owner", {"fields": ("wallet", "person", "academic_year")}),
        ("Product", {"fields": ("product_code", "description", "source_module")}),
        ("Pricing", {"fields": ("price_base_iqd", "discount_iqd", "discount_reason_code", "discount_notes", "final_charge_iqd")}),
        ("Reference", {"fields": ("reference_type", "reference_id")}),
        ("State", {"fields": ("status", "settled_transaction")}),
        ("Timestamps", {"fields": ("created_at", "updated_at"), "classes": ("collapse",)}),
    )


@admin.register(Payment)
class PaymentAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "person",
        "amount_iqd",
        "method",
        "reference",
        "received_by",
        "received_at",
    )
    list_display_links = ("id", "person")
    list_filter = ("method", "received_at", "created_at")
    search_fields = (
        "person__code",
        "person__first_name",
        "person__last_name",
        "reference",
        "notes",
    )
    ordering = ("-received_at",)
    date_hierarchy = "received_at"
    readonly_fields = ("created_at", "settled_transaction")
    autocomplete_fields = ("wallet", "person", "received_by")
    fieldsets = (
        ("Owner", {"fields": ("wallet", "person")}),
        ("Money", {"fields": ("amount_iqd", "method", "reference")}),
        ("Receipt", {"fields": ("received_at", "received_by", "settled_transaction")}),
        ("Notes", {"fields": ("notes",), "classes": ("collapse",)}),
        ("Timestamps", {"fields": ("created_at",), "classes": ("collapse",)}),
    )


@admin.register(Refund)
class RefundAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "person",
        "original_charge",
        "amount_iqd",
        "reason_code",
        "approved_by",
        "refunded_at",
    )
    list_display_links = ("id", "person")
    list_filter = ("reason_code", "refunded_at", "created_at")
    search_fields = (
        "person__code",
        "person__first_name",
        "person__last_name",
        "reason_code",
        "reason_notes",
    )
    ordering = ("-refunded_at",)
    date_hierarchy = "refunded_at"
    readonly_fields = ("created_at", "settled_transaction")
    autocomplete_fields = ("wallet", "person", "original_charge", "approved_by")
    fieldsets = (
        ("Owner", {"fields": ("wallet", "person", "original_charge")}),
        ("Money", {"fields": ("amount_iqd", "settled_transaction")}),
        ("Reason", {"fields": ("reason_code", "reason_notes")}),
        ("Approval", {"fields": ("approved_by", "refunded_at")}),
        ("Timestamps", {"fields": ("created_at",), "classes": ("collapse",)}),
    )


@admin.register(Adjustment)
class AdjustmentAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "person",
        "amount_iqd",
        "reason_code",
        "approved_by",
        "created_at",
    )
    list_display_links = ("id", "person")
    list_filter = ("reason_code", "created_at")
    search_fields = (
        "person__code",
        "person__first_name",
        "person__last_name",
        "reason_code",
        "reason_notes",
    )
    ordering = ("-created_at",)
    date_hierarchy = "created_at"
    readonly_fields = ("created_at", "settled_transaction")
    autocomplete_fields = ("wallet", "person", "approved_by")
    fieldsets = (
        ("Owner", {"fields": ("wallet", "person")}),
        ("Money", {"fields": ("amount_iqd", "settled_transaction")}),
        ("Reason", {"fields": ("reason_code", "reason_notes")}),
        ("Approval", {"fields": ("approved_by",)}),
        ("Timestamps", {"fields": ("created_at",), "classes": ("collapse",)}),
    )


@admin.register(PriceList)
class PriceListAdmin(admin.ModelAdmin):
    list_display = ("name", "currency", "is_active", "academic_year", "updated_at")
    list_display_links = ("name",)
    list_filter = ("is_active", "currency", "academic_year", "created_at", "updated_at")
    search_fields = ("name",)
    ordering = ("name",)
    date_hierarchy = "created_at"
    readonly_fields = ("created_at", "updated_at")
    autocomplete_fields = ("academic_year",)
    fieldsets = (
        ("List", {"fields": ("name", "currency", "is_active", "academic_year")}),
        ("Timestamps", {"fields": ("created_at", "updated_at"), "classes": ("collapse",)}),
    )


@admin.register(PricingRule)
class PricingRuleAdmin(admin.ModelAdmin):
    list_display = (
        "price_list",
        "product_code",
        "price_iqd",
        "is_enabled",
        "grade",
        "period_key",
    )
    list_display_links = ("price_list", "product_code")
    list_filter = ("is_enabled", "price_list", "grade", "product_code")
    search_fields = ("product_code", "period_key", "notes")
    ordering = ("price_list__name", "product_code")
    readonly_fields = ("created_at", "updated_at")
    autocomplete_fields = ("price_list", "grade")
    fieldsets = (
        ("Rule", {"fields": ("price_list", "product_code", "price_iqd", "is_enabled")}),
        ("Scope", {"fields": ("grade", "period_key"), "classes": ("collapse",)}),
        ("Notes", {"fields": ("notes",), "classes": ("collapse",)}),
        ("Timestamps", {"fields": ("created_at", "updated_at"), "classes": ("collapse",)}),
    )
