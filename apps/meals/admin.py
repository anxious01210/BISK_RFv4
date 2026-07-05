from django.contrib import admin

from .models import (
    MealPeriod,
    MealPeriodPrice,
    MealPersonPriceOverride,
    MealPlan,
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
