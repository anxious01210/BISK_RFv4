from django.contrib import admin

from .models import Person, PersonRole, RoleType, StaffProfile, StudentProfile


@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = (
        "code",
        "display_code",
        "full_name",
        "gender",
        "is_active",
        "user",
    )
    list_display_links = ("code", "display_code")
    list_filter = ("gender", "is_active", "created_at", "updated_at")
    search_fields = (
        "code",
        "first_name",
        "middle_name",
        "last_name",
        "email",
        "phone",
    )
    ordering = ("code",)
    date_hierarchy = "created_at"
    readonly_fields = ("created_at", "updated_at")
    autocomplete_fields = ("user",)
    fieldsets = (
        (
            "Identity",
            {
                "fields": (
                    "code",
                    "first_name",
                    "middle_name",
                    "last_name",
                    "gender",
                    "date_of_birth",
                )
            },
        ),
        (
            "Contact",
            {"fields": ("email", "phone", "address")},
        ),
        (
            "Account",
            {"fields": ("user", "is_active")},
        ),
        (
            "Media",
            {"fields": ("photo",), "classes": ("collapse",)},
        ),
        (
            "Timestamps",
            {
                "fields": ("created_at", "updated_at"),
                "classes": ("collapse",),
            },
        ),
    )

    @admin.display(description="Display code", ordering="code")
    def display_code(self, obj):
        return obj.display_code


@admin.register(RoleType)
class RoleTypeAdmin(admin.ModelAdmin):
    list_display = ("code", "name", "is_active", "is_system")
    list_filter = ("is_active", "is_system")
    search_fields = ("code", "name")
    ordering = ("code",)
    fieldsets = (
        ("Role", {"fields": ("code", "name")}),
        ("State", {"fields": ("is_active", "is_system")}),
    )


@admin.register(PersonRole)
class PersonRoleAdmin(admin.ModelAdmin):
    list_display = ("person", "role_type", "is_active", "start_date", "end_date", "assigned_at")
    list_filter = ("role_type", "is_active", "assigned_at")
    search_fields = (
        "person__code",
        "person__first_name",
        "person__last_name",
        "role_type__code",
        "role_type__name",
    )
    ordering = ("person__code", "role_type__code")
    date_hierarchy = "assigned_at"
    readonly_fields = ("assigned_at",)
    autocomplete_fields = ("person", "role_type", "assigned_by")
    fieldsets = (
        (
            "Assignment",
            {"fields": ("person", "role_type", "assigned_by")},
        ),
        (
            "Validity",
            {"fields": ("is_active", "start_date", "end_date")},
        ),
        (
            "Details",
            {"fields": ("notes",), "classes": ("collapse",)},
        ),
        (
            "Timestamps",
            {"fields": ("assigned_at",), "classes": ("collapse",)},
        ),
    )


@admin.register(StudentProfile)
class StudentProfileAdmin(admin.ModelAdmin):
    list_display = (
        "person",
        "code",
        "grade",
        "homeroom",
        "has_meal",
        "has_bus",
        "is_active",
        "legacy_student",
    )
    list_display_links = ("person", "code")
    list_filter = ("grade", "homeroom", "has_meal", "has_bus", "created_at", "updated_at")
    search_fields = (
        "person__code",
        "person__first_name",
        "person__last_name",
        "code",
        "grade",
        "homeroom",
    )
    ordering = ("person__code",)
    date_hierarchy = "created_at"
    readonly_fields = ("created_at", "updated_at")
    autocomplete_fields = ("person",)
    fieldsets = (
        (
            "Profile",
            {"fields": ("person", "code")},
        ),
        (
            "Schooling",
            {"fields": ("grade", "homeroom")},
        ),
        (
            "Services",
            {"fields": ("has_meal", "has_bus")},
        ),
        (
            "Legacy",
            {"fields": ("legacy_student",), "classes": ("collapse",)},
        ),
        (
            "Timestamps",
            {
                "fields": ("created_at", "updated_at"),
                "classes": ("collapse",),
            },
        ),
    )

    @admin.display(boolean=True, description="Active")
    def is_active(self, obj):
        return obj.person.is_active


@admin.register(StaffProfile)
class StaffProfileAdmin(admin.ModelAdmin):
    list_display = (
        "person",
        "code",
        "job_title",
        "is_teacher",
        "is_active",
        "hire_date",
    )
    list_display_links = ("person", "code")
    list_filter = ("is_teacher", "hire_date", "created_at", "updated_at")
    search_fields = (
        "person__code",
        "person__first_name",
        "person__last_name",
        "code",
        "job_title",
    )
    ordering = ("person__code",)
    date_hierarchy = "created_at"
    readonly_fields = ("created_at", "updated_at")
    autocomplete_fields = ("person",)
    fieldsets = (
        (
            "Profile",
            {"fields": ("person", "code")},
        ),
        (
            "Employment",
            {"fields": ("job_title", "hire_date", "is_teacher")},
        ),
        (
            "Timestamps",
            {
                "fields": ("created_at", "updated_at"),
                "classes": ("collapse",),
            },
        ),
    )

    @admin.display(boolean=True, description="Active")
    def is_active(self, obj):
        return obj.person.is_active
