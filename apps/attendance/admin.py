# apps/attendance/admin.py
from django.urls import path, reverse
from django.shortcuts import redirect
from django.contrib import admin, messages
from django.utils.html import format_html
from django.conf import settings
from django.utils import timezone

from .models import (
    Student,
    PeriodTemplate,
    PeriodOccurrence,
    AttendanceRecord,
    AttendanceEvent,
    RecognitionSettings,
    FaceEmbedding,
)
from .services import roll_periods

import csv, os
from django.http import HttpResponse
from apps.attendance.models import Student
from apps.attendance.utils.media_paths import DIRS, student_gallery_dir
from apps.attendance.utils.embeddings import enroll_student_from_folder
from django.core.management import call_command


def _first_image_rel(h_code: str) -> str | None:
    folder = student_gallery_dir(h_code)
    if not os.path.isdir(folder):
        return None
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            # Return MEDIA-relative path
            return os.path.relpath(os.path.join(folder, name), settings.MEDIA_ROOT)
    return None


@admin.action(description="Sort gallery intake (inbox → per-student)")
def sort_gallery_intake_action(modeladmin, request, queryset):
    # Works regardless of selection; leverages the command (rename policy is safe default)
    try:
        call_command("sort_gallery_intake", policy="rename", clean_empty=True)
        messages.success(request,
                         "Sorted inbox into per-student gallery folders (policy=rename). See _unsorted/ for leftovers.")
    except Exception as e:
        messages.error(request, f"Sort failed: {e}")


@admin.action(description="Enroll from folder (selected or all)")
def enroll_from_folder_action(modeladmin, request, queryset):
    try:
        if queryset.exists():
            ok = fail = 0
            for st in queryset:
                res = enroll_student_from_folder(st.h_code, k=3, force=True)
                if res.get("ok"):
                    ok += 1
                    messages.success(request,
                                     f"[{st.h_code}] OK -> {res.get('embedding_path')} (created={res.get('created')})")
                else:
                    fail += 1
                    messages.warning(request, f"[{st.h_code}] FAIL: {res.get('reason')}")
            messages.info(request, f"Enroll complete. ok={ok} fail={fail}")
        else:
            # No selection → all actives
            call_command("enroll_from_folder", k=3, force=True)
            messages.success(request, "Enrollment run started for all active students (k=3, force).")
    except Exception as e:
        messages.error(request, f"Enroll failed: {e}")


@admin.action(description="Build packed embeddings (PKL)")
def build_embeddings_pkl_action(modeladmin, request, queryset):
    try:
        call_command("build_embeddings_pkl", dim=512, force=True)
        messages.success(request, "Built embeddings_dim512.pkl for all active students.")
    except Exception as e:
        messages.error(request, f"PKL build failed: {e}")


@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ("h_code", "full_name", "gallery_count", "gallery_thumb", "is_active")
    search_fields = ("h_code", "first_name", "middle_name", "last_name")
    list_filter = ("is_active",)
    actions = [sort_gallery_intake_action, enroll_from_folder_action, build_embeddings_pkl_action]

    def gallery_count(self, obj):
        folder = student_gallery_dir(obj.h_code)
        if not os.path.isdir(folder):
            return 0
        total = 0
        for name in os.listdir(folder):
            if os.path.splitext(name)[1].lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                total += 1
        return total

    gallery_count.short_description = "Images"

    def gallery_thumb(self, obj):
        rel = _first_image_rel(obj.h_code)
        if not rel:
            return "-"
        # Use MEDIA_URL
        url = f"{settings.MEDIA_URL}{rel}"
        # Tiny thumb
        return format_html('<img src="{}" style="height:36px;width:auto;border-radius:4px;" />', url)

    gallery_thumb.short_description = "Thumb"


@admin.register(PeriodTemplate)
class PeriodTemplateAdmin(admin.ModelAdmin):
    list_display = ("name", "order", "start_time", "end_time", "weekdays_mask", "is_enabled")
    actions = ["action_generate_next_7_days"]

    @admin.action(description="Generate Period Occurrences for next 7 days")
    def action_generate_next_7_days(self, request, queryset):
        """
        Uses your existing roll_periods(days=7). It already respects template flags,
        weekday masks, and grace. We call it once (global) for simplicity.
        """
        created_total = roll_periods(days=7)
        messages.success(request, f"Generated {created_total} occurrences (next 7 days).")

    # Keep your custom URL/view so you can click an admin link to roll 7 days.
    def get_urls(self):
        urls = super().get_urls()
        extra = [path("roll-7d/", self.admin_site.admin_view(self.roll_7d_view), name="attendance_roll_7d")]
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
        return format_html("<b>{} – {}</b>", obj.student.h_code, obj.student.full_name())

    def face_preview(self, obj):
        if not obj.best_crop:
            return "—"
        return format_html('<img src="{}{}" style="height:72px;border-radius:6px;">',
                           settings.MEDIA_URL, obj.best_crop)

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


@admin.register(AttendanceEvent)
class AttendanceEventAdmin(admin.ModelAdmin):
    date_hierarchy = "ts"
    list_display = ("student", "period", "camera", "score", "ts")
    list_filter = ("camera", "period__template")
    search_fields = ("student__h_code", "student__full_name")
    ordering = ("-ts",)
    list_select_related = ("student", "period__template", "camera")


@admin.register(RecognitionSettings)
class RecognitionSettingsAdmin(admin.ModelAdmin):
    list_display = ("min_score", "re_register_window_sec", "min_improve_delta",
                    "delete_old_cropped", "save_all_crops", "use_cosine_similarity")


@admin.action(description="Activate selected")
def activate_embeddings(modeladmin, request, queryset):
    updated = queryset.update(is_active=True)
    messages.success(request, f"Activated {updated} embeddings.")

@admin.action(description="Deactivate selected")
def deactivate_embeddings(modeladmin, request, queryset):
    updated = queryset.update(is_active=False)
    messages.success(request, f"Deactivated {updated} embeddings.")

@admin.action(description="Export CSV (metadata)")
def export_embeddings_csv(modeladmin, request, queryset):
    fields = [
        "id", "student__h_code", "is_active", "dim", "created_at",
        "last_enrolled_at", "last_used_k", "last_used_det_size",
        "images_considered", "images_used", "avg_sharpness", "avg_brightness",
        "embedding_norm", "embedding_sha256", "arcface_model", "provider",
        "enroll_runtime_ms",
    ]
    # build response
    resp = HttpResponse(content_type="text/csv")
    resp["Content-Disposition"] = 'attachment; filename="face_embeddings.csv"'
    w = csv.writer(resp)
    w.writerow(fields)
    for fe in queryset.select_related("student").iterator():
        row = [
            fe.id,
            getattr(fe.student, "h_code", ""),
            fe.is_active, fe.dim, fe.created_at,
            fe.last_enrolled_at, fe.last_used_k, fe.last_used_det_size,
            fe.images_considered, fe.images_used, fe.avg_sharpness, fe.avg_brightness,
            fe.embedding_norm, fe.embedding_sha256, fe.arcface_model, fe.provider,
            fe.enroll_runtime_ms,
        ]
        w.writerow(row)
    return resp

@admin.register(FaceEmbedding)
class FaceEmbeddingAdmin(admin.ModelAdmin):
    list_display = (
        "id", "student_link", "is_active", "dim",
        "last_enrolled_at", "last_used_k", "last_used_det_size",
        "images_used", "provider", "arcface_model",
        "enroll_runtime_ms", "created_at",
    )
    list_display_links = ("student_link",)
    ordering = ("-last_enrolled_at", "-created_at")
    list_filter = ("is_active", "dim", "provider", "arcface_model", "last_used_det_size")
    search_fields = ("student__h_code", "embedding_sha256")
    actions = [activate_embeddings, deactivate_embeddings, export_embeddings_csv]

    readonly_fields = (
        "student", "dim", "created_at",
        "last_enrolled_at", "last_used_k", "last_used_det_size",
        "images_considered", "images_used",
        "avg_sharpness", "avg_brightness",
        "used_images",
        "embedding_norm", "embedding_sha256",
        "arcface_model", "provider",
        "enroll_runtime_ms", "enroll_notes",
        "source_path",
    )

    fieldsets = (
        (None, {
            "fields": ("student", "is_active", "dim", "created_at", "source_path")
        }),
        ("Enrollment metadata", {
            "fields": (
                "last_enrolled_at", "last_used_k", "last_used_det_size",
                "images_considered", "images_used",
                "avg_sharpness", "avg_brightness",
                "used_images",
                "embedding_norm", "embedding_sha256",
                "arcface_model", "provider",
                "enroll_runtime_ms", "enroll_notes",
            )
        }),
    )

    def student_link(self, obj):
        return format_html("<b>{}</b>", getattr(obj.student, "h_code", "-"))
    student_link.short_description = "Student"