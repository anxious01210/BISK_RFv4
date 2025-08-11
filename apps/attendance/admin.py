# apps/attendance/admin.py
from django.urls import path, reverse
from django.shortcuts import redirect
from django.contrib import admin, messages
from django.utils.html import format_html
from django.conf import settings
from .models import Student, PeriodTemplate, PeriodOccurrence, AttendanceRecord
from .services import roll_periods


@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ("h_code", "full_name", "is_active")
    search_fields = ("h_code", "full_name")
    list_filter = ("is_active",)


# @admin.register(PeriodTemplate)
# class PeriodTemplateAdmin(admin.ModelAdmin):
#     list_display = ("name", "order", "start_time", "end_time", "weekdays_mask", "is_enabled")

@admin.register(PeriodTemplate)
class PeriodTemplateAdmin(admin.ModelAdmin):
    list_display = ("name", "order", "start_time", "end_time", "weekdays_mask", "is_enabled")
    actions = ["action_generate_next_7_days"]

    @admin.action(description="Generate Period Occurrences for next 7 days")
    def action_generate_next_7_days(self, request, queryset):
        # You can ignore queryset and generate for all enabled templates (simplest),
        # or filter by selected only. Let's respect selection:
        # created_total = 0 # It was enabled.
        # Temporarily disable is_active_on filter by selection; roll_periods() already checks templates.
        created_total = roll_periods(days=7)
        messages.success(request, f"Generated {created_total} occurrences (next 7 days).")

    def get_urls(self):
        urls = super().get_urls()
        extra = [
            path("roll-7d/", self.admin_site.admin_view(self.roll_7d_view), name="attendance_roll_7d")
        ]
        return extra + urls

    def roll_7d_view(self, request):
        created = roll_periods(days=7)
        messages.success(request, f"Generated {created} occurrences (next 7 days).")
        return redirect("admin:attendance_periodtemplate_changelist")


@admin.register(PeriodOccurrence)
class PeriodOccurrenceAdmin(admin.ModelAdmin):
    list_display = ("template", "date", "start_dt", "end_dt", "is_school_day")
    list_filter = ("template", "is_school_day")
    date_hierarchy = "date"


@admin.register(AttendanceRecord)
class AttendanceRecordAdmin(admin.ModelAdmin):
    date_hierarchy = "best_seen"
    list_display = ("student_col", "face_preview", "score_col", "period_col", "best_camera", "best_seen")
    list_filter = ("period__template", "best_camera")
    search_fields = ("student__h_code", "student__full_name")
    ordering = ("-best_seen",)
    list_select_related = ("student", "period__template", "best_camera")

    def student_col(self, obj):
        return format_html("<b>{} – {}</b>", obj.student.h_code, obj.student.full_name)

    def face_preview(self, obj):
        if not obj.best_crop: return "—"
        return format_html('<img src="{}{}" style="height:72px;border-radius:6px;">', settings.MEDIA_URL, obj.best_crop)

    def score_col(self, obj):
        g = getattr(settings, "ATTENDANCE_SCORE_GREEN", 0.80)
        o = getattr(settings, "ATTENDANCE_SCORE_ORANGE", 0.60)
        s = obj.best_score or 0.0
        color = "lime" if s >= g else ("orange" if s >= o else "red")
        return format_html('<span style="color:{};font-weight:600;">{:.2f}</span>', color, s)

    def period_col(self, obj):
        t = obj.period.template
        s = obj.period.start_dt.astimezone().time().strftime("%H:%M:%S")
        e = obj.period.end_dt.astimezone().time().strftime("%H:%M:%S")
        return f"{t.name} ({s} - {e})"
