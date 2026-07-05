from django.contrib import admin

from .models import (
    MealEligibility,
    MealException,
    MealPeriod,
    MealPeriodPrice,
    MealPersonPriceOverride,
    MealPlan,
    MealServiceEvent,
    MealSubscription,
    MealSupervisorAction,
)


@admin.register(MealPeriod)
class MealPeriodAdmin(admin.ModelAdmin):
    list_display = (
        "kind",
        "label",
        "period_template_source",
        "period_template_ref_id",
        "sort_order",
        "is_active",
    )
    list_display_links = ("kind", "label")
    list_filter = ("kind", "is_active", "created_at", "updated_at")
    search_fields = ("label", "period_template_source")
    ordering = ("kind", "sort_order", "id")
    date_hierarchy = "created_at"
    readonly_fields = ("created_at", "updated_at")
    fieldsets = (
        (
            "Period",
            {
                "fields": (
                    "kind",
                    "label",
                    "sort_order",
                    "is_active",
                )
            },
        ),
        (
            "Generic template reference",
            {
                "fields": (
                    "period_template_source",
                    "period_template_ref_id",
                ),
                "classes": ("collapse",),
                "description": (
                    "Used until apps.scheduler owns PeriodTemplate. Both "
                    "fields must be set together or both blank."
                ),
            },
        ),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )


@admin.register(MealPlan)
class MealPlanAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "kind",
        "mode",
        "default_price_iqd",
        "insufficient_funds_mode",
        "credit_limit_iqd",
        "is_active",
    )
    list_display_links = ("name",)
    list_filter = ("kind", "mode", "insufficient_funds_mode", "is_active")
    search_fields = ("name", "notes")
    ordering = ("name",)
    date_hierarchy = "created_at"
    readonly_fields = ("created_at", "updated_at")
    fieldsets = (
        ("Plan", {"fields": ("name", "kind", "mode", "is_active")}),
        (
            "Pricing",
            {"fields": ("default_price_iqd",)},
        ),
        (
            "Supervisor policy",
            {
                "fields": (
                    "allow_supervisor_confirm",
                    "allow_supervisor_unconfirm",
                    "allow_supervisor_refund",
                    "require_reason_on_override",
                    "require_reason_on_unconfirm",
                    "require_reason_on_refund",
                ),
                "classes": ("collapse",),
            },
        ),
        (
            "Insufficient funds (wallet mode)",
            {
                "fields": ("insufficient_funds_mode", "credit_limit_iqd"),
                "classes": ("collapse",),
            },
        ),
        ("Notes", {"fields": ("notes",), "classes": ("collapse",)}),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )


@admin.register(MealPeriodPrice)
class MealPeriodPriceAdmin(admin.ModelAdmin):
    list_display = ("meal_plan", "meal_period", "price_iqd", "is_enabled")
    list_display_links = ("meal_plan", "meal_period")
    list_filter = ("is_enabled", "meal_plan", "meal_period")
    search_fields = ("notes",)
    ordering = ("meal_plan__name", "meal_period__sort_order")
    date_hierarchy = "created_at"
    readonly_fields = ("created_at", "updated_at")
    autocomplete_fields = ("meal_plan", "meal_period")
    fieldsets = (
        ("Pair", {"fields": ("meal_plan", "meal_period")}),
        ("Price", {"fields": ("price_iqd", "is_enabled")}),
        ("Notes", {"fields": ("notes",), "classes": ("collapse",)}),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )


@admin.register(MealPersonPriceOverride)
class MealPersonPriceOverrideAdmin(admin.ModelAdmin):
    list_display = (
        "person",
        "meal_plan",
        "meal_period",
        "price_iqd",
        "is_enabled",
        "effective_from",
        "effective_until",
    )
    list_display_links = ("person", "meal_plan")
    list_filter = (
        "is_enabled",
        "meal_plan",
        "meal_period",
        "created_at",
        "updated_at",
    )
    search_fields = (
        "person__code",
        "person__first_name",
        "person__last_name",
        "meal_plan__name",
        "reason_code",
        "notes",
    )
    ordering = ("person", "meal_plan", "meal_period__sort_order")
    date_hierarchy = "created_at"
    readonly_fields = ("created_at", "updated_at")
    autocomplete_fields = ("person", "meal_plan", "meal_period")
    fieldsets = (
        ("Owner", {"fields": ("person",)}),
        ("Scope", {"fields": ("meal_plan", "meal_period")}),
        ("Price", {"fields": ("price_iqd", "is_enabled")}),
        (
            "Effective window",
            {"fields": ("effective_from", "effective_until"), "classes": ("collapse",)},
        ),
        (
            "Reason",
            {"fields": ("reason_code", "notes"), "classes": ("collapse",)},
        ),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )


# ===========================================================================
# Phase 2 — Subscription / Exception / Eligibility admin
# ===========================================================================


@admin.register(MealSubscription)
class MealSubscriptionAdmin(admin.ModelAdmin):
    list_display = (
        "person",
        "meal_plan",
        "status",
        "priority",
        "start_date",
        "end_date",
        "plan_type",
        "source",
    )
    list_display_links = ("person", "meal_plan")
    list_filter = (
        "status",
        "priority",
        "plan_type",
        "source",
        "meal_plan",
        "academic_year",
    )
    search_fields = (
        "person__code",
        "person__first_name",
        "person__last_name",
        "student__code",
        "staff__code",
        "meal_plan__name",
        "notes",
    )
    ordering = ("person", "priority", "start_date", "id")
    date_hierarchy = "created_at"
    readonly_fields = ("created_at", "updated_at")
    autocomplete_fields = (
        "person", "student", "staff", "meal_plan", "academic_year",
    )
    fieldsets = (
        ("Owner", {"fields": ("person", "student", "staff")}),
        ("Plan", {"fields": ("meal_plan", "academic_year")}),
        (
            "Window",
            {"fields": ("start_date", "end_date", "priority", "plan_type")},
        ),
        ("State", {"fields": ("status", "source")}),
        ("Notes", {"fields": ("notes",), "classes": ("collapse",)}),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )


@admin.register(MealException)
class MealExceptionAdmin(admin.ModelAdmin):
    list_display = (
        "person",
        "kind",
        "effective_date",
        "end_date",
        "meal_plan",
        "is_active",
    )
    list_display_links = ("person", "kind")
    list_filter = ("kind", "is_active", "meal_plan", "effective_date")
    search_fields = (
        "person__code",
        "person__first_name",
        "person__last_name",
        "reason_code",
        "reason_notes",
    )
    ordering = ("-effective_date", "-id")
    date_hierarchy = "effective_date"
    readonly_fields = ("created_at", "updated_at")
    autocomplete_fields = ("person", "meal_plan", "approved_by")
    fieldsets = (
        ("Owner", {"fields": ("person",)}),
        ("Exception", {"fields": ("kind", "meal_plan")}),
        (
            "Effective window",
            {"fields": ("effective_date", "end_date")},
        ),
        ("Reason", {"fields": ("reason_code", "reason_notes")}),
        ("Approval", {"fields": ("approved_by", "is_active")}),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )


@admin.register(MealEligibility)
class MealEligibilityAdmin(admin.ModelAdmin):
    list_display = (
        "person",
        "date",
        "decision",
        "subscription",
        "meal_plan",
        "resolved_at",
    )
    list_display_links = ("person", "date")
    list_filter = ("decision", "date", "meal_plan")
    search_fields = (
        "person__code",
        "person__first_name",
        "person__last_name",
        "student__code",
        "reason_code",
        "reason_notes",
    )
    ordering = ("-date", "-resolved_at", "-id")
    date_hierarchy = "date"
    readonly_fields = ("resolved_at",)
    autocomplete_fields = (
        "person", "student", "subscription", "meal_plan", "resolved_by",
    )
    fieldsets = (
        ("Owner", {"fields": ("person", "student", "date")}),
        ("Decision", {"fields": ("decision", "subscription", "meal_plan")}),
        ("Academic snapshot", {
            "fields": ("grade_code_snapshot", "section_code_snapshot"),
            "classes": ("collapse",),
        }),
        ("Reason", {"fields": ("reason_code", "reason_notes")}),
        ("Audit", {"fields": ("resolved_by", "resolved_at")}),
    )


# ===========================================================================
# Phase 3A — Service event / supervisor action admin
# ===========================================================================


@admin.register(MealServiceEvent)
class MealServiceEventAdmin(admin.ModelAdmin):
    list_display = (
        "person",
        "date",
        "status",
        "meal_plan",
        "meal_period",
        "final_charge_iqd",
        "served_at",
    )
    list_display_links = ("person", "date")
    list_filter = (
        "status",
        "date",
        "meal_plan",
        "meal_period",
    )
    search_fields = (
        "person__code",
        "person__first_name",
        "person__last_name",
        "student__code",
        "staff__code",
        "section_code_snapshot",
        "grade_code_snapshot",
        "reason_code",
        "reason_notes",
        "price_resolution_source",
    )
    ordering = ("-date", "-id")
    date_hierarchy = "date"
    readonly_fields = (
        "created_at",
        "updated_at",
        "performed_actions",
    )
    autocomplete_fields = (
        "person",
        "student",
        "staff",
        "eligibility",
        "subscription",
        "meal_plan",
        "meal_period",
        "served_by",
        "reversed_by",
    )
    fieldsets = (
        ("Owner", {"fields": ("person", "student", "staff", "date")}),
        (
            "Links",
            {
                "fields": (
                    "eligibility",
                    "subscription",
                    "meal_plan",
                    "meal_period",
                )
            },
        ),
        ("State", {"fields": ("status",)}),
        (
            "Price snapshot",
            {
                "fields": (
                    "price_base_iqd",
                    "price_override_iqd",
                    "discount_iqd",
                    "final_charge_iqd",
                    "price_resolution_source",
                ),
            },
        ),
        (
            "Financial snapshot",
            {
                "fields": (
                    "wallet_balance_before_iqd",
                    "wallet_balance_after_iqd",
                    "wallet_transaction",
                    "wallet_refund_transaction",
                ),
            },
        ),
        (
            "Academic snapshot",
            {
                "fields": (
                    "grade_code_snapshot",
                    "section_code_snapshot",
                    "meal_period_label_snapshot",
                ),
                "classes": ("collapse",),
            },
        ),
        ("Reason", {"fields": ("reason_code", "reason_notes")}),
        (
            "Service / reversal",
            {
                "fields": ("served_at", "served_by", "reversed_at", "reversed_by"),
                "classes": ("collapse",),
            },
        ),
        (
            "Audit",
            {
                "fields": ("performed_actions", "created_at", "updated_at"),
                "classes": ("collapse",),
            },
        ),
    )

    @admin.display(description="Supervisor actions")
    def performed_actions(self, obj):
        if obj is None or not obj.pk:
            return "—"
        actions = obj.supervisor_actions.all()
        if not actions:
            return "—"
        return ", ".join(
            f"{a.get_action_display()} @ {a.performed_at:%Y-%m-%d %H:%M}"
            for a in actions
        )

    def has_delete_permission(self, request, obj=None):
        # Service events in terminal statuses should not be deleted
        # (they are immutable historical records). PENDING events may
        # be deleted by an admin if needed.
        if obj is None:
            return True
        from .models import TERMINAL_SERVICE_EVENT_STATUSES
        return obj.status not in TERMINAL_SERVICE_EVENT_STATUSES


@admin.register(MealSupervisorAction)
class MealSupervisorActionAdmin(admin.ModelAdmin):
    list_display = (
        "action",
        "service_event",
        "eligibility",
        "performed_by",
        "performed_by_user",
        "performed_at",
    )
    list_display_links = ("action",)
    list_filter = ("action", "performed_at")
    search_fields = (
        "reason_code",
        "reason_notes",
        "performed_by__code",
        "performed_by__person__first_name",
        "performed_by__person__last_name",
    )
    ordering = ("-performed_at", "-id")
    date_hierarchy = "performed_at"
    readonly_fields = ("performed_at",)
    autocomplete_fields = (
        "service_event",
        "eligibility",
        "performed_by",
        "performed_by_user",
    )
    fieldsets = (
        ("Target", {"fields": ("service_event", "eligibility")}),
        ("Action", {"fields": ("action", "reason_code", "reason_notes")}),
        (
            "Performer",
            {
                "fields": ("performed_by", "performed_by_user"),
            },
        ),
        ("Timestamp", {"fields": ("performed_at",), "classes": ("collapse",)}),
    )

    def has_change_permission(self, request, obj=None):
        # Audit rows are append-only — never editable.
        return False

    def has_delete_permission(self, request, obj=None):
        # Audit rows are append-only — never deletable.
        return False
