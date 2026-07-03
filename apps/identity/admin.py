from django.contrib import admin

from .models import Person, PersonRole, RoleType, StaffProfile, StudentProfile


@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = ("code", "full_name", "gender", "is_active", "user")
    list_filter = ("gender", "is_active")
    search_fields = ("code", "first_name", "middle_name", "last_name", "email", "phone")


@admin.register(RoleType)
class RoleTypeAdmin(admin.ModelAdmin):
    list_display = ("code", "name", "is_active", "is_system")
    list_filter = ("is_active", "is_system")
    search_fields = ("code", "name")


@admin.register(PersonRole)
class PersonRoleAdmin(admin.ModelAdmin):
    list_display = ("person", "role_type", "is_active", "start_date", "end_date")
    list_filter = ("role_type", "is_active")
    search_fields = (
        "person__code",
        "person__first_name",
        "person__last_name",
        "role_type__code",
    )


@admin.register(StudentProfile)
class StudentProfileAdmin(admin.ModelAdmin):
    list_display = (
        "person",
        "grade",
        "homeroom",
        "has_meal",
        "has_bus",
        "legacy_student",
    )
    list_filter = ("grade", "has_meal", "has_bus")
    search_fields = (
        "person__code",
        "person__first_name",
        "person__last_name",
        "grade",
        "homeroom",
    )


@admin.register(StaffProfile)
class StaffProfileAdmin(admin.ModelAdmin):
    list_display = ("person", "code", "job_title", "is_teacher")
    list_filter = ("is_teacher",)
    search_fields = (
        "person__code",
        "person__first_name",
        "person__last_name",
        "code",
        "job_title",
    )
