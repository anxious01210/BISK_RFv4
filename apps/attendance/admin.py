# apps/attendance/admin.py
import hashlib
import subprocess
import sys
from pathlib import Path
import numpy as np
from django.views.decorators.http import require_POST
from django.utils.decorators import method_decorator

from apps.cameras.models import Camera
from django.urls import path, reverse
from django.shortcuts import redirect, render, get_object_or_404
from django.contrib import admin, messages
from django.utils.html import format_html, escape, conditional_escape
from django.conf import settings
from django.utils import timezone
from django import forms
from django.db import transaction

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

import csv, os, re
from django.http import HttpResponse
from apps.attendance.models import Student
from apps.attendance.utils.media_paths import DIRS, student_gallery_dir, ensure_media_tree
from apps.attendance.utils.embeddings import enroll_student_from_folder, set_det_size
from apps.attendance.utils.facescore import score_images  # ROI + cascade scorer

from django.core.management import call_command
from import_export import resources, fields
from import_export.widgets import BooleanWidget
from import_export.admin import ImportExportMixin
from import_export.formats import base_formats

# Build a static label like "4 Used images" (falls back to "K Used images")
_THUMBS_N = getattr(settings, "EMBEDDING_LIST_MAX_THUMBS", None)
_THUMBS_TITLE = (f"{int(_THUMBS_N)} Used images" if isinstance(_THUMBS_N, int) and _THUMBS_N > 0 else "K Used images")


def _coerce_raw01(value):
    """
    Accept both canonical raw01 in [0..1] and legacy percent/other.
    If value>1, treat as percent and divide by 100.
    Returns float in [0..1] or None.
    """
    try:
        x = float(value)
    except Exception:
        return None
    if x > 1.0:
        x = x / 100.0  # legacy percent
    if x < 0.0:
        x = 0.0
    if x > 1.0:
        x = 1.0
    return x


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
        call_command(
            "sort_gallery_intake",
            policy="rename",
            clean_empty=True,
            crop_faces=False,  # move only
            det_size=getattr(settings, "FACE_DET_SIZE_DEFAULT", 640),
        )
        messages.success(request,
                         "Sorted inbox into per-student gallery folders (policy=rename). See _unsorted/ for leftovers.")
    except Exception as e:
        messages.error(request, f"Sort failed: {e}")


def sort_gallery_intake_crop_keep_action(modeladmin, request, queryset):
    try:
        call_command(
            "sort_gallery_intake",
            policy="rename",
            clean_empty=True,
            crop_faces=True,
            keep_raw=True,
            enhance=True,  # improve crops while keeping originals
            det_size=getattr(settings, "FACE_DET_SIZE_DEFAULT", 640),
        )
        messages.success(request, "Sorted inbox with CROPPING (kept originals under raw/).")
    except Exception as e:
        messages.error(request, f"Sort failed: {e}")


def sort_gallery_intake_crop_discard_action(modeladmin, request, queryset):
    try:
        call_command(
            "sort_gallery_intake",
            policy="rename",
            clean_empty=True,
            crop_faces=True,
            keep_raw=False,  # discard originals after crop
            enhance=True,  # improve crops
            det_size=getattr(settings, "FACE_DET_SIZE_DEFAULT", 640),
        )
        messages.success(request, "Sorted inbox with CROPPING (originals discarded).")
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


# class StudentResource(resources.ModelResource):
#     class Meta:
#         model = Student
#         exclude = ("id",)  # include everything EXCEPT id
#         import_id_fields = ("h_code",)  # upsert identity
#         # fields = ("h_code","first_name","middle_name","last_name","is_active")
#         # export_order = ("h_code","first_name","middle_name","last_name","is_active")
#         skip_unchanged = True
#         use_bulk = True
#
#     def before_import_row(self, row, **kwargs):
#         if not (row.get("first_name") or row.get("middle_name") or row.get("last_name")):
#             full = (row.get("full_name") or "").strip()
#             if full:
#                 parts = [p for p in full.split() if p]
#                 if len(parts) == 1:
#                     row["first_name"], row["middle_name"], row["last_name"] = parts[0], "", ""
#                 elif len(parts) == 2:
#                     row["first_name"], row["middle_name"], row["last_name"] = parts[0], "", parts[1]
#                 else:
#                     row["first_name"], row["middle_name"], row["last_name"] = parts[0], " ".join(parts[1:-1]), parts[-1]
#
#     # Optional: make 'h_code' the first column in exports (everything else follows automatically)
#     def get_export_fields(self):
#         fields = super().get_export_fields()
#         fields.sort(key=lambda f: (f.attribute != "h_code",))
#         return fields


@admin.register(Student)
class StudentAdmin(ImportExportMixin, admin.ModelAdmin):
    list_display = ("h_code", "is_active", "full_name", "grade", "gallery_count", "gallery_thumb", "has_lunch", "has_bus")
    search_fields = ("h_code", "first_name", "middle_name", "last_name")
    list_filter = ("is_active",)
    actions = [
        sort_gallery_intake_action,  # move only
        sort_gallery_intake_crop_keep_action,  # crop + keep raw
        sort_gallery_intake_crop_discard_action,  # crop + discard raw (enhanced)
        enroll_from_folder_action, build_embeddings_pkl_action
    ]

    change_list_template = "admin/attendance/student/change_list.html"

    # resource_class = StudentResource

    # URL: /admin/attendance/student/upload-inbox/
    def get_urls(self):
        urls = super().get_urls()
        extra = [
            path(
                "upload-inbox/",
                self.admin_site.admin_view(self.upload_inbox_view),
                name="attendance_student_upload_inbox",
            ),
        ]
        return extra + urls

    def upload_inbox_view(self, request):
        """
        Bulk upload images directly into MEDIA/face_gallery_inbox/.
        Safe filename sanitation + no overwrite (adds _1, _2, ...).
        """
        inbox_dir = DIRS["FACE_GALLERY_INBOX"]
        ensure_media_tree()
        if request.method == "POST":
            files = request.FILES.getlist("files") or []
            if not files:
                messages.warning(request, "No files selected.")
                return redirect("admin:attendance_student_upload_inbox")

            allowed = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            saved = 0
            for f in files:
                name = os.path.basename(f.name)
                base, ext = os.path.splitext(name)
                if ext.lower() not in allowed:
                    # skip silently but inform once at end
                    continue
                # sanitize filename
                safe_base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_") or "img"
                candidate = f"{safe_base}{ext.lower()}"
                path = os.path.join(inbox_dir, candidate)
                i = 1
                while os.path.exists(path):
                    candidate = f"{safe_base}_{i}{ext.lower()}"
                    path = os.path.join(inbox_dir, candidate)
                    i += 1
                with open(path, "wb+") as out:
                    for chunk in f.chunks():
                        out.write(chunk)
                saved += 1

            if saved:
                messages.success(request, f"Uploaded {saved} image(s) to inbox: {inbox_dir}")
            else:
                messages.warning(request, "No images were uploaded (unsupported types or empty selection).")
            # back to Student changelist so you can run the sorting action next
            return redirect("admin:attendance_student_changelist")

        ctx = {
            "opts": Student._meta,
            "inbox_dir": inbox_dir,
            "upload_url": reverse("admin:attendance_student_upload_inbox"),
        }
        return render(request, "admin/attendance/student/upload_inbox.html", ctx)

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
    list_display = ("student_col", "score_col", "period_col", "best_camera", "best_seen", "face_preview",)
    list_filter = ("period__template", "best_camera")
    search_fields = ("student__h_code", "student__full_name")
    ordering = ("-best_seen",)
    list_select_related = ("student", "period__template", "best_camera")

    readonly_fields = ("student", "period", "first_seen", "last_seen", "best_seen", "best_camera",
                       "best_crop_preview", "best_crop_url", "best_score", "sightings", "status")
    exclude = ("best_crop",)

    def _image_url(self, path: str | None) -> str | None:
        """
        Accepts:
          - MEDIA-relative path (preferred)
          - absolute path under MEDIA_ROOT
          - full http(s) URL
        Returns a safe web URL.
        """
        import os
        if not path:
            return None
        p = str(path)

        # If it's already a full URL, just return it
        if p.startswith("http://") or p.startswith("https://"):
            return p

        # If it's an absolute filesystem path under MEDIA_ROOT → make it MEDIA-relative
        try:
            mr = os.path.normpath(str(settings.MEDIA_ROOT or ""))
            if os.path.isabs(p) and mr:
                if os.path.commonpath([os.path.realpath(p), os.path.realpath(mr)]) == os.path.realpath(mr):
                    p = os.path.relpath(p, mr)
                    p = p.replace(os.sep, "/")
        except Exception:
            pass

        # Build URL: MEDIA_URL + relative path (and trim stray slashes)
        base = settings.MEDIA_URL or "/media/"
        return f"{base.rstrip('/')}/{p.lstrip('/')}"

    # new_tab-based face_preview
    # def face_preview(self, obj):
    #     url = self._image_url(getattr(obj, "best_crop", None))
    #     if not url:
    #         return "—"
    #     return format_html(
    #         '<a href="{}" target="_blank" rel="noopener">'
    #         '<img src="{}" style="height:72px;border-radius:6px;" /></a>',
    #         url, url
    #     )

    # modal-based face_preview
    def face_preview(self, obj):
        url = self._image_url(getattr(obj, "best_crop", None))
        if not url:
            return "—"
        # Modal hook: no target, href="#", and data-url for JS
        # return format_html(
        #     '<a href="#" class="bisk-img-modal" data-url="{}" title="Open preview">'
        #     '  <img src="{}" style="height:72px;border-radius:6px;box-shadow:0 0 0 1px rgba(0,0,0,.08);" />'
        #     '</a>',
        #     url, url
        # )
        return format_html(
            '<span style="display:inline-flex;gap:6px;align-items:center">'
            '  <a href="#" class="bisk-img-modal" data-url="{}" title="Open preview">'
            '    <img src="{}" style="height:72px;border-radius:6px;box-shadow:0 0 0 1px rgba(0,0,0,.08);" />'
            '  </a>'
            '  <a href="{}" target="_blank" rel="noopener" title="Open original" '
            '     style="text-decoration:none;font-size:18px;line-height:1;">↗︎</a>'
            '</span>',
            url, url, url
        )

    def best_crop_url(self, obj):
        url = self._image_url(getattr(obj, "best_crop", None))
        if not url:
            return "—"
        return format_html('<a href="{0}" target="_blank" rel="noopener">{0}</a>', url)

    best_crop_url.short_description = "Best crop (URL)"

    # best_crop_preview.short_description = "Best crop"

    def best_crop_preview(self, obj):
        # reuse the same normalization logic as above (you may refactor into a helper)
        return self.face_preview(obj)

    best_crop_preview.short_description = "Best crop"

    def score_col(self, obj):
        # 1) Coerce to float and pre-format to text to avoid SafeString :f errors
        s = float(obj.best_score or 0.0)
        s_txt = f"{s:.2f}"

        # 2) Read bands & colors from settings (with safe defaults)
        bands = getattr(settings, "ATTENDANCE_SCORE_BANDS", {})
        g = float(bands.get("green_min", 0.80))
        y = float(bands.get("yellow_min", 0.65))
        o = float(bands.get("orange_min", 0.50))

        c_green = getattr(settings, "ATTENDANCE_COLOR_GREEN", "lime")
        c_yellow = getattr(settings, "ATTENDANCE_COLOR_YELLOW", "#facc15")
        c_orange = getattr(settings, "ATTENDANCE_COLOR_ORANGE", "orange")
        c_red = getattr(settings, "ATTENDANCE_COLOR_RED", "red")

        # 3) Decide color by band
        if s >= g:
            color = c_green
        elif s >= y:
            color = c_yellow
        elif s >= o:
            color = c_orange
        else:
            color = c_red

        return format_html('<span style="color:{};font-weight:600;">{}</span>', color, s_txt)

    def student_col(self, obj):
        return format_html("<b>{} – {}</b>", obj.student.h_code, obj.student.full_name())

    def period_col(self, obj):
        t = obj.period.template
        s = obj.period.start_dt.astimezone().time().strftime("%H:%M:%S")
        e = obj.period.end_dt.astimezone().time().strftime("%H:%M:%S")
        return f"{t.name} ({s} - {e})"

    # ModelAdmin Media.js = ("attendance/crop_modal.js",) asks Django to serve that path via the staticfiles pipeline. That succeeds only if a finder can locate a file with that relative path in:
    # any app’s static/ directory (e.g. apps/attendance/static/attendance/crop_modal.js), or
    # any directory listed in STATICFILES_DIRS (e.g. <BASE_DIR>/static/attendance/crop_modal.js).
    class Media:
        js = ("attendance/crop_modal.js",)


# @admin.register(AttendanceEvent)
# class AttendanceEventAdmin(admin.ModelAdmin):
#     date_hierarchy = "ts"
#     list_display = ("student", "period", "camera", "score", "ts")
#     list_filter = ("camera", "period__template")
#     search_fields = ("student__h_code", "student__full_name")
#     ordering = ("-ts",)
#     list_select_related = ("student", "period__template", "camera")

@admin.register(AttendanceEvent)
class AttendanceEventAdmin(admin.ModelAdmin):
    date_hierarchy = "ts"
    list_display = ("student_col", "score_col", "period_col", "camera", "ts", "face_preview",)
    list_filter = ("camera", "period__template")
    search_fields = ("student__h_code", "student__full_name")
    ordering = ("-ts",)
    list_select_related = ("student", "period__template", "camera")
    readonly_fields = ("student_col", "score_col", "period_col", "camera", "ts", "face_preview", "crop_path")
    exclude = ["period", "score", "student"]

    # Reuse the same normalization approach as AttendanceRecord preview
    def _image_url(self, path: str | None) -> str | None:
        if not path:
            return None
        p = str(path)
        if p.startswith("http://") or p.startswith("https://"):
            return p
        try:
            import os
            mr = str(settings.MEDIA_ROOT or "")
            if os.path.isabs(p) and mr and os.path.normpath(p).startswith(os.path.normpath(mr)):
                p = os.path.relpath(p, mr).replace(os.sep, "/")
        except Exception:
            pass
        base = settings.MEDIA_URL or "/media/"
        return f"{base.rstrip('/')}/{p.lstrip('/')}"

    def face_preview(self, obj):
        url = self._image_url(getattr(obj, "crop_path", None))
        if not url:
            return "—"
        # modal on click + a small ↗︎ to open original
        return format_html(
            '<span style="display:inline-flex;gap:6px;align-items:center">'
            '  <a href="#" class="bisk-img-modal" data-url="{0}" title="Open preview">'
            '    <img src="{0}" style="height:60px;border-radius:6px;box-shadow:0 0 0 1px rgba(0,0,0,.08);" />'
            '  </a>'
            '  <a href="{0}" target="_blank" rel="noopener" title="Open original" '
            '     style="text-decoration:none;font-size:16px;line-height:1;">↗︎</a>'
            '</span>',
            url
        )

    def student_col(self, obj):
        return format_html("<b>{} — {}</b>", obj.student.h_code, obj.student.full_name())

    def score_col(self, obj):
        s = float(obj.score or 0.0);
        s_txt = f"{s:.2f}"
        bands = getattr(settings, "ATTENDANCE_SCORE_BANDS", {})
        g = float(bands.get("green_min", 0.80));
        y = float(bands.get("yellow_min", 0.65));
        o = float(bands.get("orange_min", 0.50))
        c_green = getattr(settings, "ATTENDANCE_COLOR_GREEN", "lime")
        c_yellow = getattr(settings, "ATTENDANCE_COLOR_YELLOW", "#facc15")
        c_orange = getattr(settings, "ATTENDANCE_COLOR_ORANGE", "orange")
        c_red = getattr(settings, "ATTENDANCE_COLOR_RED", "red")
        color = c_green if s >= g else c_yellow if s >= y else c_orange if s >= o else c_red
        return format_html('<span style="color:{};font-weight:600;">{}</span>', color, s_txt)

    def period_col(self, obj):
        if not obj.period:
            return "—"
        t = obj.period.template
        s = obj.period.start_dt.astimezone().time().strftime("%H:%M:%S")
        e = obj.period.end_dt.astimezone().time().strftime("%H:%M:%S")
        return f"{t.name} ({s} - {e})"

    class Media:
        js = ("attendance/crop_modal.js",)


@admin.register(RecognitionSettings)
class RecognitionSettingsAdmin(admin.ModelAdmin):
    list_display = ("id", "min_score", "re_register_window_sec", "min_improve_delta", "min_face_px", "changed_at",
                    "delete_old_cropped", "save_all_crops", "use_cosine_similarity",)
    readonly_fields = ("changed_at",)
    list_editable = ("min_score", "re_register_window_sec", "min_improve_delta", "min_face_px",
                     "delete_old_cropped", "save_all_crops", "use_cosine_similarity",)
    list_display_links = ('id',)


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


@admin.action(description="Re-enroll (refresh bytes/metadata)")
def reenroll_embeddings_action(modeladmin, request, queryset):
    from apps.attendance.utils.embeddings import set_det_size, enroll_student_from_folder
    ok = fail = 0
    # prefetch student to avoid N+1
    for fe in queryset.select_related("student").all():
        if not fe.student or not fe.student.is_active:
            fail += 1
            continue
        k = fe.last_used_k or 3
        det = fe.last_used_det_size or 640
        try:
            set_det_size(det)
            res = enroll_student_from_folder(fe.student.h_code, k=k, force=True)
            if res.get("ok"):
                ok += 1
            else:
                fail += 1
        except Exception:
            fail += 1
    if ok:
        messages.success(request, f"Re-enrolled {ok} embedding(s).")
    if fail:
        messages.warning(request, f"{fail} embedding(s) failed to re-enroll; check logs.")


@admin.register(FaceEmbedding)
class FaceEmbeddingAdmin(admin.ModelAdmin):
    list_display = (
        "id", "student_link", "is_active", "dim",
        "last_enrolled_at", "last_used_k", "last_used_det_size",
        "images_used", "avg_used_score", "thumbs_with_chips",
        "provider_short", "arcface_model",
        "enroll_runtime_ms", "created_at",
    )
    list_display_links = ("student_link",)
    ordering = ("-last_enrolled_at", "-created_at",)
    list_filter = ("is_active", "dim", "provider", "arcface_model", "last_used_det_size",)
    search_fields = ("student__h_code", "embedding_sha256",)
    actions = [reenroll_embeddings_action, activate_embeddings, deactivate_embeddings, export_embeddings_csv]

    readonly_fields = (
        "student", "dim", "created_at",
        "last_enrolled_at", "last_used_k", "last_used_det_size",
        "images_considered", "images_used",
        "avg_sharpness", "avg_brightness",
        "used_images",
        "embedding_norm", "embedding_sha256",
        "arcface_model", "provider",
        "enroll_runtime_ms", "enroll_notes_full",
        # "enroll_notes",
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
                "enroll_runtime_ms", "enroll_notes_full",
                # "enroll_notes",
            )
        }),
    )

    def enroll_notes_full(self, obj):
        txt = obj.enroll_notes or ""
        # preserve newlines, don’t truncate, keep it readable
        html = f"<div style='white-space:pre-wrap;max-width:80ch'>{escape(txt)}</div>"
        return format_html(html)

    enroll_notes_full.short_description = "Enroll notes"

    # def enroll_notes_full(self, obj):
    #     txt = obj.enroll_notes or ""
    #     # textarea cannot be truncated by container CSS; preserves newlines and allows scrolling
    #     return format_html(
    #         "<textarea readonly rows='8' style='width:100%;white-space:pre-wrap;"
    #         "font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace;'>"
    #         "{}</textarea>",
    #         conditional_escape(txt),
    #     )
    #
    # enroll_notes_full.short_description = "Enroll notes"

    # def enroll_notes_full(self, obj):
    #     txt = obj.enroll_notes or ""
    #     safe = conditional_escape(txt)
    #
    #     # Heuristic: fit initial height to content lines, within sensible bounds
    #     # (so it opens 'just right' most of the time).
    #     lines = txt.count("\n") + 1
    #     rows = max(4, min(lines, 24))  # min 4 rows, max 24 rows
    #
    #     return format_html(
    #         "<textarea readonly rows='{rows}' wrap='soft' "
    #         "style='display:block;width:200%;max-width:100%;box-sizing:border-box;"
    #         "white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;"
    #         "overflow-x:hidden;resize:vertical;"
    #         "font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "
    #         "Liberation Mono, monospace;'>"
    #         "{content}</textarea>",
    #         rows=rows,
    #         content=safe,
    #     )
    #
    # enroll_notes_full.short_description = "Enroll notes"

    def student_link(self, obj):
        return format_html("<b>{}</b>", getattr(obj.student, "h_code", "-"))

    student_link.short_description = "Student"

    # def top3_preview(self, obj):
    #     """
    #     Render up to 3 small thumbnails with their scores from used_images_detail.
    #     Falls back to filenames if the images are missing.
    #     """
    #     det = getattr(obj, "used_images_detail", None) or []
    #     if not det:
    #         return "-"
    #     # Build media-relative URLs
    #     h = getattr(obj.student, "h_code", "")
    #     folder = student_gallery_dir(h)
    #     items = []
    #     for rec in det[:6]:
    #         name = rec.get("name", "")
    #         score = rec.get("score", 0.0)
    #         rel = os.path.relpath(os.path.join(folder, name), settings.MEDIA_ROOT)
    #         url = f"{settings.MEDIA_URL}{rel}"
    #         items.append(
    #             f'<div style="display:inline-block;margin-right:6px;text-align:center;">'
    #             f'  <img src="{url}" onerror="this.style.display=\'none\'" '
    #             f'       style="height:36px;width:auto;border-radius:4px;display:block;margin:auto;" />'
    #             f'  <div style="font-size:11px; color:blue;">{score:.2f}</div>'
    #             f'</div>'
    #         )
    #     return format_html("".join(items))
    #
    # top3_preview.short_description = "Top‑K (score)"
    #
    def avg_used_score(self, obj):
        det = getattr(obj, "used_images_detail", None) or []
        if isinstance(det, str):
            try:
                import json
                det = json.loads(det)
            except Exception:
                det = []
        vals = []
        for r in det:
            if not isinstance(r, dict):
                continue
            v = r.get("raw01", r.get("score", None))
            v = _coerce_raw01(v)
            if v is not None:
                vals.append(v)
        if not vals:
            return "-"
        return f"{(sum(vals) / len(vals)):.3f}"

    avg_used_score.short_description = "avg used score (raw01)"

    def provider_short(self, obj):
        p = (obj.provider or "").upper()
        if "CUDA" in p: return "CUDA"
        if "CPU" in p:  return "CPU"
        return (obj.provider or "").replace("ExecutionProvider", "").strip() or obj.provider

    provider_short.short_description = "Provider"

    def thumbs_preview(self, obj):
        """
        Render up to K (last_used_k) clickable thumbs with scores; cap via settings.EMBEDDING_LIST_MAX_THUMBS if set.
        """
        det = getattr(obj, "used_images_detail", None) or getattr(obj, "used_images", None) or []
        if not det:
            return "-"

        # Normalize records to dicts with at least 'name'
        if isinstance(det, list) and det and isinstance(det[0], str):
            det = [{"name": name, "score": None} for name in det]

        k = int(getattr(obj, "last_used_k", 0) or len(det))
        cap = getattr(settings, "EMBEDDING_LIST_MAX_THUMBS", None)
        n = min(k, len(det), cap if isinstance(cap, int) and cap > 0 else k)

        h = getattr(obj.student, "h_code", "")
        folder = student_gallery_dir(h)

        items = []
        for rec in det[:n]:
            name = rec.get("name") or rec.get("path") or ""
            # prefer normalized raw01; fall back to legacy score
            raw01 = _coerce_raw01(rec.get("raw01", rec.get("score", None)))
            rank = rec.get("rank", None)
            sharp = rec.get("sharp", None)
            bright = rec.get("bright", None)
            det_conf = rec.get("det_conf", None)
            det_size = rec.get("det_size", None)

            # build MEDIA url safely
            try:
                rel = os.path.relpath(os.path.join(folder, name), settings.MEDIA_ROOT)
                url = f"{settings.MEDIA_URL}{rel}"
            except Exception:
                url = "#"

            score_txt = f"{raw01:.2f}" if isinstance(raw01, float) else ""
            title = f"{name}"
            if raw01 is not None:
                title += f" | raw01={raw01:.3f}"
            if isinstance(rank, int):
                title += f" | rank={rank}"
            if det_size:
                title += f" | det={det_size}"
            if det_conf is not None:
                title += f" | conf={float(det_conf):.2f}"
            if sharp is not None:
                title += f" | sharp={float(sharp):.2f}"
            if bright is not None:
                title += f" | bright={float(bright):.2f}"

            items.append(
                f'<a href="{url}" target="_blank" rel="noopener" '
                f'style="display:inline-block;margin-right:6px;text-align:center;text-decoration:none;" '
                f'title="{title}">'
                f'  <img src="{url}" '
                f'       style="height:36px;width:auto;border-radius:4px;display:block;margin:auto;'
                f'              box-shadow:0 0 0 1px rgba(0,0,0,.08);" />'
                f'  <div style="font-size:11px;color:#555;">{score_txt}</div>'
                f'</a>'
            )

        more = len(det) - n
        if more > 0:
            items.append(
                f'<span style="display:inline-block;vertical-align:top;'
                f'background:#eee;border-radius:6px;padding:2px 6px;font-size:12px;color:#444;">+{more}</span>'
            )

        return format_html("".join(items))

    # after the method
    cap = getattr(settings, "EMBEDDING_LIST_MAX_THUMBS", None)
    thumbs_preview.short_description = f"{cap} Used images" if isinstance(cap, int) and cap > 0 else "Used images"

    # ---- New: thumbs + chips + +more ----
    def thumbs_with_chips(self, obj):
        # Build the thumbs HTML by reusing the existing method
        thumbs = self.thumbs_preview(obj)
        # Chips values
        k = obj.last_used_k or 0
        det = obj.last_used_det_size or 0
        ms = getattr(obj, "last_used_min_score", None)
        ms_txt = f"{ms:.2f}" if isinstance(ms, (float, int)) else "—"
        chips = format_html(
            '<span class="bisk-chips">'
            '<span class="bisk-chip bisk-chip--k">K: {}</span>'
            '<span class="bisk-chip bisk-chip--min">min: {}</span>'
            '<span class="bisk-chip bisk-chip--det">det: {}</span>'
            '</span>',
            k, ms_txt, det,
        )

        more = format_html(
            '<a class="button bisk-more js-reenroll-pop" data-id="{}" target="_blank" rel="noopener noreferrer" href="{}" '
            'style="">+more</a>',
            obj.id, self._re_enroll_modal_url(obj.id)
        )
        return format_html('<div class="bisk-cell">{more}{thumbs}</div><div style="font-size: 20px;">{chips}</div>',
                           thumbs=format_html("{}", thumbs), chips=chips, more=more)

    thumbs_with_chips.short_description = "Used images • chips"

    # ---- Admin modal URLs ----
    def get_urls(self):
        urls = super().get_urls()
        extra = [
            path("<int:pk>/re-enroll/", self.admin_site.admin_view(self.re_enroll_modal),
                 name="attendance_faceembedding_re_enroll"),
            path("<int:pk>/re-enroll/run/", self.admin_site.admin_view(self.re_enroll_run),
                 name="attendance_faceembedding_re_enroll_run"),
            # NEW:
            path("<int:pk>/capture/", self.admin_site.admin_view(self.re_enroll_capture_run),
                 name="attendance_faceembedding_capture_run"),
        ]
        return extra + urls

    def _re_enroll_modal_url(self, pk: int) -> str:
        return f"./{pk}/re-enroll/"

    # ---- Modal form ----
    class _ReEnrollForm(forms.Form):
        k = forms.IntegerField(min_value=1, required=False, help_text="Top-K images to average")
        det_size = forms.IntegerField(required=False, help_text="Detector input size (e.g., 640 / 1024)")
        min_score = forms.FloatField(required=False, help_text="Min quality score 0..1")
        strict_top = forms.BooleanField(required=False, initial=False, help_text="Filter by min-score before K")

    class _CaptureForm(forms.Form):
        camera = forms.ModelChoiceField(
            queryset=Camera.objects.all().order_by("name"),
            required=True, label="Camera"
        )
        k = forms.IntegerField(min_value=1, initial=3, required=True, label="Top-K images")
        det_size = forms.ChoiceField(
            choices=[(640, "640"), (800, "800"), (1024, "1024"), (1280, "1280"), (1600, "1600"), (2048, "2048")],
            initial=1024, required=True, label="det_set"
        )

    # ---- GET: render gallery + overrides ----
    def re_enroll_modal(self, request, pk: int):
        fe = get_object_or_404(FaceEmbedding, pk=pk)
        student = fe.student
        h = getattr(student, "h_code", "")
        folder = student_gallery_dir(h)
        rows = []
        if os.path.isdir(folder):
            # Score ALL images (ROI + cascade). Uses settings defaults.
            paths = [os.path.join(folder, n) for n in sorted(os.listdir(folder))]
            rows = [r for r in score_images(paths) if r]

        # Convert filesystem paths to MEDIA urls for the template
        def _fs_to_media_url(fs_path: str) -> str:
            if not fs_path:
                return ""
            try:
                rel = os.path.relpath(fs_path, settings.MEDIA_ROOT)
            except Exception:
                return ""  # outside MEDIA_ROOT; skip
            return settings.MEDIA_URL + rel.replace(os.sep, "/")

        for r in rows:
            r["url"] = _fs_to_media_url(r.get("path"))

        init = {
            "k": fe.last_used_k or None,
            "det_size": fe.last_used_det_size or None,
            "min_score": fe.last_used_min_score if fe.last_used_min_score is not None else None,
            "strict_top": False,
        }
        form = self._ReEnrollForm(initial=init)
        # ctx = {"opts": self.model._meta, "fe": fe, "student": student, "rows": rows, "form": form}
        # return render(request, "admin/attendance/faceembedding/re_enroll_modal.html", ctx)
        capture_form = self._CaptureForm(initial={"k": fe.last_used_k or 3, "det_size": fe.last_used_det_size or 1024})
        capture_url = reverse("admin:attendance_faceembedding_capture_run", args=[fe.pk])

        ctx = {
            "opts": self.model._meta,
            "fe": fe,
            "student": student,
            "rows": rows,
            "form": form,
            # NEW:
            "capture_form": capture_form,
            "capture_url": capture_url,
        }
        return render(request, "admin/attendance/faceembedding/re_enroll_modal.html", ctx)

    @transaction.atomic
    @method_decorator(require_POST)
    def re_enroll_capture_run(self, request, pk: int):
        fe = get_object_or_404(FaceEmbedding, pk=pk)
        student = fe.student
        hcode = getattr(student, "h_code", None)
        if not hcode:
            messages.error(request, "Student H-code missing.")
            return redirect(request.META.get("HTTP_REFERER") or "../../")

        form = self._CaptureForm(request.POST)
        if not form.is_valid():
            messages.error(request, "Invalid capture inputs.")
            return redirect(request.META.get("HTTP_REFERER") or "../../")

        cam = form.cleaned_data["camera"]
        k = int(form.cleaned_data["k"])
        det_size = int(form.cleaned_data["det_size"])

        # Resolve RTSP/URL field (be tolerant to different field names)
        rtsp = getattr(cam, "url", None) or getattr(cam, "rtsp", None) or getattr(cam, "rtsp_url", None)
        if not rtsp:
            messages.error(request, "Selected camera has no RTSP/URL configured.")
            return redirect(request.META.get("HTTP_REFERER") or "../../")

        # Build script path
        proj_root = Path(settings.BASE_DIR)  # your Django base
        script = proj_root / "extras" / "capture_embeddings_ffmpeg.py"
        if not script.exists():
            messages.error(request, f"Capture script not found: {script}")
            return redirect(request.META.get("HTTP_REFERER") or "../../")

        # Run capture (synchronously; keep it simple for now)
        cmd = [
            sys.executable, str(script),
            "--rtsp", str(rtsp),
            "--hcode", hcode,
            "--k", str(k),
            "--det_size", str(det_size),
            "--device", "auto",
            "--fps", "4",
            "--duration", "30",
            "--rtsp_transport", "tcp",
        ]
        try:
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        except subprocess.TimeoutExpired:
            messages.error(request, "Capture timed out.")
            return redirect(request.META.get("HTTP_REFERER") or "../../")

        if run.returncode != 0:
            messages.error(request, f"Capture failed:\n{run.stderr or run.stdout}")
            return redirect(request.META.get("HTTP_REFERER") or "../../")

        # Expect the script to have written media/embeddings/<HCODE>.npy
        emb_dir = Path(getattr(settings, "MEDIA_ROOT", "media")) / "embeddings"
        npy_path = emb_dir / f"{hcode}.npy"
        if not npy_path.exists():
            messages.error(request, f"Capture finished but .npy not found: {npy_path}")
            return redirect(request.META.get("HTTP_REFERER") or "../../")

        # Ingest into FaceEmbedding (active singleton per student)
        try:
            vec = np.load(str(npy_path)).astype(np.float32).reshape(-1)
            # L2 norm should be ~1.0; store as raw bytes
            norm = float(np.linalg.norm(vec)) if vec.size else 0.0
            raw = vec.tobytes()
            sha = hashlib.sha256(raw).hexdigest()

            # Deactivate existing active rows for this student
            FaceEmbedding.objects.filter(student=student, is_active=True).update(is_active=False)

            fe2 = FaceEmbedding.objects.create(
                student=student,
                dim=int(vec.size),
                vector=raw,
                source_path=str(npy_path.relative_to(Path(settings.MEDIA_ROOT))).replace("\\", "/"),
                camera=cam,
                is_active=True,
                last_enrolled_at=timezone.now(),
                last_used_k=k,
                last_used_det_size=det_size,
                images_considered=0,
                images_used=k,
                embedding_norm=norm,
                embedding_sha256=sha,
                arcface_model="buffalo_l",
                provider="CUDAExecutionProvider" if "CUDA" in (run.stdout or "") else "CPUExecutionProvider",
                enroll_runtime_ms=0,
                enroll_notes=(
                    f"live-capture ffmpeg; k={k}, det={det_size}, camera={getattr(cam, 'name', cam.pk)}; "
                    f"vec_dim={vec.size}; norm={norm:.4f}"
                ),
            )
            messages.success(request, f"Live capture OK → new embedding for {hcode} (id={fe2.id}).")
        except Exception as e:
            messages.error(request, f"Failed to save embedding from .npy: {e}")
            return redirect(request.META.get("HTTP_REFERER") or "../../")

        # persist last_used_min_score=None, we didn’t threshold here
        return redirect("../../")

    # ---- POST: run enrollment with overrides (force) ----
    @transaction.atomic
    @method_decorator(require_POST)
    def re_enroll_run(self, request, pk: int):
        fe = get_object_or_404(FaceEmbedding, pk=pk)
        form = self._ReEnrollForm(request.POST)
        if not form.is_valid():
            messages.error(request, "Invalid inputs.")
            return redirect(request.META.get("HTTP_REFERER") or "../../")

        k = form.cleaned_data.get("k")
        det = form.cleaned_data.get("det_size")
        min_score = form.cleaned_data.get("min_score")
        strict_top = bool(form.cleaned_data.get("strict_top"))

        # Apply detector size override to your embedding singleton
        if det:
            try:
                set_det_size(int(det))
            except Exception:
                pass

        try:
            res = enroll_student_from_folder(fe.student.h_code,
                                             k=int(k) if k else (fe.last_used_k or 3),
                                             force=True,
                                             min_score=float(min_score) if min_score is not None else 0.0,
                                             strict_top=bool(strict_top))
        except Exception as e:
            messages.error(request, f"Enroll failed: {e}")
            return redirect(request.META.get("HTTP_REFERER") or "../../")

        if not res or not res.get("ok"):
            messages.warning(request, f"Enroll did not succeed: {res.get('reason') if res else 'Unknown error'}")
            return redirect("../../")

        # Persist min-score to the *active* FaceEmbedding row (id provided by your enroll function)
        new_id = res.get("face_embedding_id")
        meta = (res or {}).get("meta", {}) or {}
        used_min = meta.get("min_score", min_score)
        try:
            if new_id:
                fe2 = FaceEmbedding.objects.select_for_update().get(pk=new_id)
            else:
                fe2 = fe  # fallback to the same row
            if used_min is not None:
                fe2.last_used_min_score = float(used_min)
                fe2.save(update_fields=["last_used_min_score"])
        except Exception:
            pass

        messages.success(request, f"Re-enrolled {fe.student.h_code} successfully.")
        return redirect("../../")  # back to changelist (refreshes row)

    # class Media:
    #     css = {"all": ("attendance/admin_chips.css",)}

    class Media:
        # Only defines a single CSS variable (--bisk-chip-fg) that flips in dark mode.
        css = {"all": ("attendance/admin_chips.css",)}
        # JS to open “+more” in a popup window (new window)
        js = ("attendance/re_enroll_popup.js",)
