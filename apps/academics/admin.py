from django.contrib import admin

from .models import (
    AcademicYear,
    Grade,
    SchoolLevel,
    Section,
    StudentEnrollment,
    StudentEnrollmentSectionPlacement,
)


@admin.register(AcademicYear)
class AcademicYearAdmin(admin.ModelAdmin):
    list_display = ("name", "code", "start_date", "end_date", "is_active", "created_at", "updated_at")
    list_display_links = ("name", "code")
    list_filter = ("is_active", "created_at", "updated_at")
    search_fields = ("name", "code")
    ordering = ("-start_date",)
    date_hierarchy = "start_date"
    readonly_fields = ("created_at", "updated_at")
    fieldsets = (
        ("Year", {"fields": ("name", "code")}),
        ("Dates", {"fields": ("start_date", "end_date")}),
        ("State", {"fields": ("is_active",)}),
        ("Timestamps", {"fields": ("created_at", "updated_at"), "classes": ("collapse",)}),
    )


@admin.register(SchoolLevel)
class SchoolLevelAdmin(admin.ModelAdmin):
    list_display = ("name", "code", "order", "is_active")
    list_filter = ("is_active",)
    search_fields = ("name", "code")
    ordering = ("order",)
    readonly_fields = ("created_at", "updated_at")
    fieldsets = (
        ("Level", {"fields": ("name", "code", "order")}),
        ("State", {"fields": ("is_active",)}),
        ("Timestamps", {"fields": ("created_at", "updated_at"), "classes": ("collapse",)}),
    )


@admin.register(Grade)
class GradeAdmin(admin.ModelAdmin):
    list_display = ("name", "code", "level", "order", "is_active")
    list_display_links = ("name", "code")
    list_filter = ("level", "is_active")
    search_fields = ("name", "code")
    ordering = ("order",)
    readonly_fields = ("created_at", "updated_at")
    autocomplete_fields = ("level",)
    fieldsets = (
        ("Grade", {"fields": ("name", "code", "level", "order")}),
        ("State", {"fields": ("is_active",)}),
        ("Timestamps", {"fields": ("created_at", "updated_at"), "classes": ("collapse",)}),
    )


@admin.register(Section)
class SectionAdmin(admin.ModelAdmin):
    list_display = ("academic_year", "grade", "code", "name", "homeroom_adviser", "capacity", "is_active")
    list_display_links = ("code", "name")
    list_filter = ("academic_year", "grade", "is_active", "created_at", "updated_at")
    search_fields = (
        "name",
        "code",
        "grade__name",
        "grade__code",
        "academic_year__name",
        "academic_year__code",
    )
    ordering = ("-academic_year__start_date", "grade__order", "code")
    date_hierarchy = "created_at"
    readonly_fields = ("created_at", "updated_at")
    autocomplete_fields = ("academic_year", "grade", "homeroom_adviser")
    fieldsets = (
        ("Section", {"fields": ("academic_year", "grade", "code", "name")}),
        ("Adviser", {"fields": ("homeroom_adviser",)}),
        ("Roster", {"fields": ("capacity", "is_active")}),
        ("Timestamps", {"fields": ("created_at", "updated_at"), "classes": ("collapse",)}),
    )


@admin.register(StudentEnrollment)
class StudentEnrollmentAdmin(admin.ModelAdmin):
    list_display = (
        "student",
        "academic_year",
        "grade",
        "section",
        "status",
        "enrollment_date",
        "withdrawal_date",
    )
    list_display_links = ("student",)
    list_filter = ("academic_year", "grade", "status", "enrollment_date", "updated_at")
    search_fields = (
        "student__code",
        "student__person__first_name",
        "student__person__last_name",
        "academic_year__name",
        "academic_year__code",
        "grade__name",
        "grade__code",
        "section__name",
        "section__code",
        "notes",
    )
    ordering = ("-academic_year__start_date", "student")
    date_hierarchy = "enrollment_date"
    readonly_fields = ("created_at", "updated_at")
    autocomplete_fields = ("student", "academic_year", "grade", "section", "prior_enrollment")
    fieldsets = (
        ("Enrollment", {"fields": ("student", "academic_year", "grade", "section")}),
        ("Lifecycle", {"fields": ("status", "enrollment_date", "withdrawal_date", "prior_enrollment")}),
        ("Notes", {"fields": ("notes",), "classes": ("collapse",)}),
        ("Timestamps", {"fields": ("created_at", "updated_at"), "classes": ("collapse",)}),
    )


@admin.register(StudentEnrollmentSectionPlacement)
class StudentEnrollmentSectionPlacementAdmin(admin.ModelAdmin):
    list_display = ("enrollment", "section", "start_date", "end_date", "is_current", "created_at", "updated_at")
    list_display_links = ("enrollment",)
    list_filter = ("is_current", "start_date", "end_date", "created_at", "updated_at")
    search_fields = (
        "enrollment__student__code",
        "enrollment__student__person__first_name",
        "enrollment__student__person__last_name",
        "enrollment__academic_year__code",
        "section__name",
        "section__code",
        "notes",
    )
    ordering = ("-start_date",)
    date_hierarchy = "start_date"
    readonly_fields = ("created_at", "updated_at")
    autocomplete_fields = ("enrollment", "section")
    fieldsets = (
        ("Placement", {"fields": ("enrollment", "section")}),
        ("Validity", {"fields": ("start_date", "end_date", "is_current")}),
        ("Notes", {"fields": ("notes",), "classes": ("collapse",)}),
        ("Timestamps", {"fields": ("created_at", "updated_at"), "classes": ("collapse",)}),
    )
