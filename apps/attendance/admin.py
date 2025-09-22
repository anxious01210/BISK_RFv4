# apps/attendance/admin.py
import hashlib, subprocess, sys, csv, os, re, json, shutil, tempfile
from pathlib import Path
import numpy as np
from django.views.decorators.http import require_POST
from django.utils.decorators import method_decorator
from urllib.parse import urlparse, parse_qsl
from apps.cameras.models import Camera
from django.urls import path, reverse
from django.shortcuts import redirect, render, get_object_or_404
from django.contrib import admin, messages
from django.utils.html import format_html, escape, conditional_escape
from django.conf import settings
from django.utils import timezone
from django import forms
from django.db import transaction
from uuid import uuid4

from .models import (
    Student,
    PeriodTemplate,
    PeriodOccurrence,
    AttendanceRecord,
    AttendanceEvent,
    RecognitionSettings,
    FaceEmbedding,
    DashboardTag,
)
from .services import roll_periods

import urllib.request
import requests

from django.http import HttpResponse, HttpResponseBadRequest, QueryDict
from apps.attendance.models import Student
from apps.attendance.utils.media_paths import DIRS, student_gallery_dir, ensure_media_tree
from apps.attendance.utils.embeddings import enroll_student_from_folder, set_det_size
from apps.attendance.utils.facescore import score_images  # ROI + cascade scorer

from django.core.management import call_command
from import_export import resources, fields
from import_export.widgets import BooleanWidget
from import_export.admin import ImportExportMixin
from import_export.formats import base_formats

from django.template.loader import render_to_string
from django.utils.http import url_has_allowed_host_and_scheme  # if you need

# Build a static label like "4 Used images" (falls back to "K Used images")
_THUMBS_N = getattr(settings, "EMBEDDING_LIST_MAX_THUMBS", None)
_THUMBS_TITLE = (f"{int(_THUMBS_N)} Used images" if isinstance(_THUMBS_N, int) and _THUMBS_N > 0 else "K Used images")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}  # adjust if needed


def _score_to_color_and_text(raw_score: float):
    s = float(raw_score or 0.0)
    s_txt = f"{s:.2f}"

    bands = getattr(settings, "ATTENDANCE_SCORE_BANDS", {})
    b = float(bands.get("blue_min", 0.90))
    g = float(bands.get("green_min", 0.80))
    y = float(bands.get("yellow_min", 0.65))
    o = float(bands.get("orange_min", 0.50))

    c_blue = getattr(settings, "ATTENDANCE_COLOR_BLUE", "#2196F3")
    c_green = getattr(settings, "ATTENDANCE_COLOR_GREEN", "lime")
    c_yellow = getattr(settings, "ATTENDANCE_COLOR_YELLOW", "#facc15")
    c_orange = getattr(settings, "ATTENDANCE_COLOR_ORANGE", "orange")
    c_red = getattr(settings, "ATTENDANCE_COLOR_RED", "red")

    if s >= b:
        color = c_blue
    elif s >= g:
        color = c_green
    elif s >= y:
        color = c_yellow
    elif s >= o:
        color = c_orange
    else:
        color = c_red

    return color, s_txt


def _safe_media_path(rel: str) -> Path:
    base = Path(settings.MEDIA_ROOT).resolve()
    p = (base / rel).resolve()
    if base not in p.parents and p != base:
        raise ValueError("path outside MEDIA_ROOT")
    return p


def _expand_selected(rel_list):
    """Expand files & folders recursively; return list[Path] of files only."""
    files = []
    for rel in rel_list or []:
        p = _safe_media_path(rel)
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in ALLOWED_EXTS:
                    files.append(f)
        elif p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            files.append(p)
    return files


def _rel_media(p: Path) -> str:
    return os.path.relpath(str(p), settings.MEDIA_ROOT).replace(os.sep, "/")


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
    """
    Return MEDIA-relative path of the first image found directly under:
        MEDIA/face_gallery/<H_CODE>/
    Subfolders are ignored.
    """
    folder = student_gallery_dir(h_code)
    if not os.path.isdir(folder):
        return None

    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if not os.path.isfile(p):
            continue
        if os.path.splitext(name)[1].lower() in IMG_EXTS:
            rel = os.path.relpath(p, settings.MEDIA_ROOT)
            return rel.replace(os.sep, "/")
    return None

# def _first_image_rel(h_code: str) -> str | None:
#     folder = student_gallery_dir(h_code)
#     if not os.path.isdir(folder):
#         return None
#
#     # default subdir used by student_capture_dir(...)
#     subdir = "captures"
#     root = os.path.join(folder, subdir)
#     if not os.path.isdir(root):
#         return None
#
#     # Walk date / period / camera, return first existing image
#     for date in sorted(os.listdir(root)):
#         p_date = os.path.join(root, date)
#         if not os.path.isdir(p_date):
#             continue
#         for period in sorted(os.listdir(p_date)):
#             p_per = os.path.join(p_date, period)
#             if not os.path.isdir(p_per):
#                 continue
#             for cam in sorted(os.listdir(p_per)):
#                 p_cam = os.path.join(p_per, cam)
#                 if not os.path.isdir(p_cam):
#                     continue
#                 for name in sorted(os.listdir(p_cam)):
#                     if os.path.splitext(name)[1].lower() in IMG_EXTS:
#                         rel = os.path.relpath(os.path.join(p_cam, name), settings.MEDIA_ROOT)
#                         return rel.replace(os.sep, "/")
#     return None


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

# NEW: simple admin for controlled vocabulary
@admin.register(DashboardTag)
class DashboardTagAdmin(admin.ModelAdmin):
    list_display = ("name", "slug")
    list_editable = ("slug",)
    search_fields = ("name", "slug")

@admin.register(Student)
class StudentAdmin(ImportExportMixin, admin.ModelAdmin):
    list_display = ("h_code", "is_active", "full_name", "grade", "gallery_count", "gallery_thumb", "has_lunch",
                    "has_bus")
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
    list_display = ("name", "order", "start_time", "end_time", "weekdays_mask", "is_enabled", )
    list_filter = ("usage_tags",)
    search_fields = ("name",)
    filter_horizontal = ("usage_tags",)

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
    list_filter = ("period__template", "best_camera",)
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

    # modal-based face_preview
    def face_preview(self, obj):
        url = self._image_url(getattr(obj, "best_crop", None))
        if not url:
            return "—"

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
        color, s_txt = _score_to_color_and_text(getattr(obj, "best_score", None))  # for AttendanceRecordAdmin
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
        color, s_txt = _score_to_color_and_text(getattr(obj, "score", None))  # for AttendanceEventAdmin
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
    change_list_template = "admin/attendance/faceembedding/change_list.html"  # <-- add this
    list_display = (
        "student_link", "crops_opt_in", "is_active", "dim",
        "last_enrolled_at", "last_used_k", "last_used_det_size",
        "images_used", "avg_used_score", "thumbs_with_chips",
        "provider_short", "arcface_model",
        "enroll_runtime_ms", "created_at", "id",
    )
    list_display_links = ("student_link",)
    ordering = ("-last_enrolled_at", "-created_at",)
    list_filter = ("is_active", "dim", "provider", "arcface_model", "last_used_det_size",)
    search_fields = ("student__h_code", "embedding_sha256",)
    list_editable = ["crops_opt_in", ]
    actions = [reenroll_embeddings_action, activate_embeddings, deactivate_embeddings, export_embeddings_csv]

    readonly_fields = (
        "id", "student", "dim", "created_at",
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
            "fields": ("id", "student", "is_active", "dim", "created_at", "source_path")
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

    def _row_card(self, name: str, meta, *, rel: str | None = None,
                  thumb_url: str | None = None, is_folder: bool = False,
                  open_href: str | None = None) -> str:
        """
        meta can be either a dict with {score, sharp, bright} OR a string ("date"/"period"/"camera"/"image").
        Renders:
          - selection corner,
          - optional thumb (or folder icon),
          - single-line filename with ellipsis,
          - optional stats (3 lines),
          - tooltip with all info.
        """
        # unpack stats if provided
        score = sharp = bright = None
        if isinstance(meta, dict):
            score = meta.get("score")
            sharp = meta.get("sharp")
            bright = meta.get("bright")

        # tooltip text
        title_parts = [name]
        stat_parts = []
        if score is not None: stat_parts.append(f"score: {float(score):.2f}")
        if sharp is not None: stat_parts.append(f"sharp: {float(sharp):.2f}")
        if bright is not None: stat_parts.append(f"bright: {float(bright):.2f}")
        if stat_parts:
            title_parts.append(" | ".join(stat_parts))
        card_title = " — ".join(title_parts)

        # outer attrs (title for hover)
        sel_attrs = (f' class="card sel" data-rel="{rel}" title="{escape(card_title)}"'
                     if rel else f' class="card" title="{escape(card_title)}"')

        # tick box
        selbox = ''
        if rel:
            selbox = (
                '<div class="sel-box" title="Select">'
                '  <svg viewBox="0 0 24 24" class="tick"><path d="M20 6L9 17l-5-5"/></svg>'
                '</div>'
            )

        # thumb or folder icon
        if is_folder and not thumb_url:
            thumb = (
                '<div class="thumb" style="display:flex;align-items:center;justify-content:center;background:#111">'
                '  <svg width="48" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="opacity:.8">'
                '    <path d="M10 4l2 2h8v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h4z" stroke="currentColor" stroke-width="1.2"/>'
                '  </svg>'
                '</div>'
            )
        else:
            thumb = (f'<img class="thumb" src="{thumb_url}" onerror="this.style.display=\'none\'">'
                     if thumb_url else "")

        # filename + stats block
        if isinstance(meta, dict):
            stat_rows = []
            if score is not None: stat_rows.append(f'<div>score: {float(score):.2f}</div>')
            if sharp is not None: stat_rows.append(f'<div>sharp: {float(sharp):.2f}</div>')
            if bright is not None: stat_rows.append(f'<div>bright: {float(bright):.2f}</div>')
            stats_html = f'<div class="stats">{"".join(stat_rows)}</div>' if stat_rows else ''
            meta_html = (f'<div class="meta">'
                         f'  <div class="fname" title="{escape(card_title)}">{escape(name)}</div>'
                         f'  {stats_html}'
                         f'</div>')
        else:
            meta_html = (f'<div class="meta">'
                         f'  <div class="fname" title="{escape(card_title)}">{escape(name)}</div>'
                         f'  <div class="stats">{escape(str(meta))}</div>'
                         f'</div>')

        # optional "Open" for folders
        open_btn = ''
        if open_href:
            open_btn = (
                f'<div style="margin-top:4px">'
                f'  <a class="button button-small" '
                f'     hx-get="{open_href}" hx-target="#captures-browser" hx-swap="innerHTML">Open</a>'
                f'</div>'
            )

        return f'<div{sel_attrs}>{selbox}{thumb}{meta_html}{open_btn}</div>'

    def enroll_notes_full(self, obj):
        txt = obj.enroll_notes or ""
        # preserve newlines, don’t truncate, keep it readable
        html = f"<div style='white-space:pre-wrap;max-width:80ch'>{escape(txt)}</div>"
        return format_html(html)

    enroll_notes_full.short_description = "Enroll notes"

    def student_link(self, obj):
        return format_html("<b>{}</b>", getattr(obj.student, "h_code", "-"))

    student_link.short_description = "Student"

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

        # ... inside thumbs_preview() ...
        h = getattr(obj.student, "h_code", "")
        media_root = Path(settings.MEDIA_ROOT).resolve()
        gallery_root = media_root / "face_gallery" / h

        items = []
        for rec in det[:n]:
            raw_path = (rec.get("path") or "").strip()
            name = (rec.get("name") or "").strip()

            rel = None
            if raw_path:
                try:
                    p = Path(raw_path)
                    # normalize to an absolute filesystem path
                    p_abs = p if p.is_absolute() else (media_root / raw_path).resolve()
                    # try to compute MEDIA-relative path
                    rel_candidate = str(p_abs.relative_to(media_root)).replace("\\", "/")
                    # accept only if it's inside face_gallery/<HCODE>/
                    if rel_candidate.startswith(f"face_gallery/{h}/"):
                        rel = rel_candidate
                except Exception:
                    rel = None

            if not rel:
                # fall back to <MEDIA_ROOT>/face_gallery/<HCODE>/<name>
                rel = str((gallery_root / name).relative_to(media_root)).replace("\\", "/")

            url = f"{settings.MEDIA_URL}{rel}"

            raw01 = _coerce_raw01(rec.get("raw01", rec.get("score")))
            rank = rec.get("rank")
            sharp = rec.get("sharp");
            bright = rec.get("bright")
            det_conf = rec.get("det_conf");
            det_size = rec.get("det_size")

            title_name = Path(raw_path).name if raw_path else name
            title = f"{title_name}"
            if raw01 is not None: title += f" | raw01={float(raw01):.3f}"
            if isinstance(rank, int): title += f" | rank={rank}"
            if det_conf is not None: title += f" | conf={float(det_conf):.2f}"
            if det_size is not None: title += f" | det={det_size}"
            if sharp is not None: title += f" | sharp={float(sharp):.2f}"
            if bright is not None: title += f" | bright={float(bright):.2f}"

            items.append(
                f'<a href="{url}" target="_blank" rel="noopener" title="{title}" '
                f'style="display:inline-block;margin-right:6px;text-align:center;text-decoration:none;">'
                f'  <img src="{url}" style="height:36px;width:auto;border-radius:4px;display:block;margin:auto;'
                f'          box-shadow:0 0 0 1px rgba(0,0,0,.08);" />'
                f'  <div style="font-size:11px;color:#555;">{"" if raw01 is None else f"{raw01:.2f}"}</div>'
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
            '<a class="button bisk-more js-reenroll-pop" data-id="{}" href="{}">+more</a>',
            # '<a class="button bisk-more js-reenroll-pop" data-id="{}" '
            # 'target="_blank" rel="noopener noreferrer" href="{}">+more</a>',
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
            path("<int:pk>/captures/", self.admin_site.admin_view(self.re_enroll_captures),
                 name="attendance_faceembedding_captures"),
            path("<int:pk>/captures/delete/", self.admin_site.admin_view(self.re_enroll_captures_delete),
                 name="attendance_faceembedding_captures_delete"),

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
        # folder = student_gallery_dir(h)
        rows = []
        # active cameras for dropdown
        cameras = Camera.objects.filter(is_active=True).order_by("name")

        # all cameras (active & non-active)
        # cameras = Camera.objects.all().order_by("name")

        # if os.path.isdir(folder):
        #     # Score ALL images (ROI + cascade). Uses settings defaults.
        #     paths = [os.path.join(folder, n) for n in sorted(os.listdir(folder))]
        #     rows = [r for r in score_images(paths) if r]

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
            "cameras": cameras,
        }
        browser_url = reverse("admin:attendance_faceembedding_captures", args=[fe.pk])
        delete_url = reverse("admin:attendance_faceembedding_captures_delete", args=[fe.pk])
        ctx["browser_url"] = browser_url
        ctx["delete_url"] = delete_url
        # fe = get_object_or_404(FaceEmbedding, pk=pk)

        selected_top_k = str(request.GET.get("k") or (fe.last_used_k or 3))
        selected_det_set = str(request.GET.get("det_size") or "1024")

        ctx.update({
            "top_k_options": [1, 2, 3, 4, 5, 6, 8, 10],
            "det_set_options": [640, 800, 1024, 1600, 2048],
            "selected_top_k": selected_top_k,
            "selected_det_set": selected_det_set,
        })
        return render(request, "admin/attendance/faceembedding/re_enroll_modal.html", ctx)

    # ---- POST: live capture (FFmpeg) with extra tunables ----
    @method_decorator(require_POST)
    def re_enroll_capture_run(self, request, pk: int):
        """
        Kick off extras/capture_embeddings_ffmpeg.py and, optionally, save top crops
        under face_gallery/<HCODE>/live_YYYYmmdd_HHMMSS/.

        Always returns an HX-Trigger so the UI refreshes the captures grid
        without swapping it out (pair this with hx-swap="none" in the form).
        """

        def push_flag(cmd_list, key, value):
            # append "--key value" only if value is not None/blank and not already present
            if value is None:
                return
            sv = str(value).strip()
            if not sv:
                return
            if f"--{key}" in cmd_list:
                return
            cmd_list += [f"--{key}", sv]

        fe = get_object_or_404(FaceEmbedding, pk=pk)
        hcode = fe.student.h_code

        # keep current query in scope so the listing stays at the same level after refresh
        hx_url = request.headers.get("HX-Current-URL") or ""
        if hx_url and not request.GET:
            parsed = urlparse(hx_url)
            qd = QueryDict(mutable=True)
            qd.update(dict(parse_qsl(parsed.query or "")))
            request.GET = qd

        # posted knobs (all optional; keep your names exactly)
        k = int(request.POST.get("k") or 3)
        # det_set = int(request.POST.get("det_set") or 1024)
        det_raw = (request.POST.get("det_size") or request.POST.get("det_set") or "").strip()
        det_set = int(det_raw) if det_raw.isdigit() else 1024
        duration = int(request.POST.get("duration") or 30)
        # fps = int(float(request.POST.get("fps") or 4))  # script expects int
        fps_raw = (request.POST.get("fps") or "").strip()
        fps = int(float(fps_raw)) if fps_raw != "" else 0
        min_face = request.POST.get("min_face_px")
        min_face = int(min_face) if (min_face and str(min_face).isdigit()) else None
        model = (request.POST.get("model") or "").strip()  # blank = inherit
        pipe_q = (request.POST.get("pipe_mjpeg_q") or "").strip()  # blank = inherit
        crop_fmt = (request.POST.get("crop_fmt") or "").strip()  # blank = inherit
        crop_jq = (request.POST.get("crop_jpeg_quality") or "").strip()
        bbox_exp = (request.POST.get("bbox_expand") or "").strip()
        rtsp_tr = (request.POST.get("rtsp_transport") or "auto").strip()
        device_raw = (request.POST.get("device") or "auto").strip()
        device = device_raw
        # normalize 'cuda:0' / 'cuda 0' / 'CUDA:1' → 'cuda0'
        m = re.match(r"^\s*cuda[:\s]?(\d+)\s*$", device_raw, flags=re.I)
        if m:
            device = f"cuda{m.group(1)}"
        else:
            device = device_raw.lower()
        hwaccel = (request.POST.get("hwaccel") or "none").strip()
        save_crops = (request.POST.get("save_top_crops") == "1")
        # Parse the optional "sharp,bright,size" triple (e.g., "0,0,1920x1080" or "0,0,1080")
        preproc = (request.POST.get("preproc") or "").strip()
        # pipe_size = ""
        # if preproc:
        #     parts = [p.strip() for p in preproc.split(",")]
        #     if len(parts) >= 3 and parts[2]:
        #         # Accept "1920x1080" or just "1080" (height). We'll normalize in the script.
        #         pipe_size = parts[2]
        #         print(f"{pipe_size=}")

        # --- Pipe size: prefer explicit select, fallback to preproc[2] ---
        pipe_size = (request.POST.get("pipe_size") or "").strip()

        # Normalize common labels from the UI select
        norm = pipe_size.lower()
        if norm in ("", "inherit", "inherit (no scale)"):
            pipe_size = ""  # no scaling => don't pass the flag
        elif norm == "height 720 (keep aspect)":
            pipe_size = "720"
        elif norm == "height 1080 (keep aspect)":
            pipe_size = "1080"
        # "1280x720" / "1920x1080" pass through as-is

        # Fallback to the legacy preproc field (sharp,bright,size)
        if not pipe_size:
            preproc = (request.POST.get("preproc") or "").strip()
            if preproc:
                parts = [p.strip() for p in preproc.split(",")]
                if len(parts) >= 3:
                    s = parts[2]
                    if s and s.lower() not in ("inherit", "inherit (no scale)"):
                        pipe_size = s

        # preview-only: boxes/labels only (no crops, no .npy)
        preview_only = (request.POST.get("preview_only") == "1")

        if preview_only:
            k = 0
            save_crops = False

        weights = (request.POST.get("qual_weights") or "").strip()

        # Source (camera RTSP). Keep name you already use.
        cam_rtsp = (request.POST.get("camera_rtsp") or "").strip()

        # NEW: preview controls (all optional; safe to omit)
        preview = bool(request.POST.get("preview") or "")
        preview_fullscreen = bool(request.POST.get("preview_fullscreen") or "")
        preview_max_w = (request.POST.get("preview_max_w") or "").strip()
        preview_max_h = (request.POST.get("preview_max_h") or "").strip()
        preview_allow_upscale = bool(request.POST.get("preview_allow_upscale") or "")
        write_npy_now = bool(request.POST.get('write_npy_now') or request.POST.get('write_npy'))
        disable_uplink = (request.POST.get("disable_uplink") == "1")

        # Script path
        script = Path(settings.BASE_DIR) / "extras" / "capture_embeddings_ffmpeg.py"
        if not script.exists():
            # For HTMX: small body + HX-Trigger (no grid swap)
            if request.headers.get("HX-Request") == "true":
                resp = HttpResponse("<!-- script missing -->")
                resp["HX-Trigger"] = json.dumps({
                    "toast": {"text": f"Capture script not found: {script}"}
                })
                return resp
            messages.error(request, "Capture script not found.")
            return redirect(request.META.get("HTTP_REFERER") or "../../")

        # Build command (keep your flags 1:1 with the script)
        cmd = [
            sys.executable, str(script),
            "--hcode", hcode,
            "--k", str(k),
            "--duration", str(duration),
            "--fps", str(fps),
            "--det_size", str(det_set),
            "--device", device,
            "--rtsp_transport", rtsp_tr,
            "--hwaccel", hwaccel,
        ]
        if save_crops:
            cmd += ["--save_top_crops"]
        if pipe_size:
            cmd += ["--pipe-size", pipe_size]
            # optional debug:
            # print(f"[CAPTURE] pipe_size={pipe_size}")
        # optional overrides (inherit when blank)
        if cam_rtsp:
            push_flag(cmd, "rtsp", cam_rtsp)
        if model:
            push_flag(cmd, "model", model)
        if pipe_q.isdigit():
            push_flag(cmd, "pipe_mjpeg_q", pipe_q)
        if crop_fmt:
            push_flag(cmd, "crop_fmt", crop_fmt)
        if crop_jq.isdigit():
            push_flag(cmd, "crop_jpeg_quality", crop_jq)
        try:
            if bbox_exp and float(bbox_exp) == float(bbox_exp):
                push_flag(cmd, "bbox_expand", bbox_exp)
        except ValueError:
            pass
        if min_face is not None:
            push_flag(cmd, "min_face_px", min_face)
        if weights:
            push_flag(cmd, "weights", weights)

        # preview flags
        if preview:
            cmd += ["--preview"]
        if preview_fullscreen:
            cmd += ["--preview_fullscreen"]
        if preview_max_w.isdigit() and int(preview_max_w) > 0:
            push_flag(cmd, "preview_max_w", preview_max_w)
        if preview_max_h.isdigit() and int(preview_max_h) > 0:
            push_flag(cmd, "preview_max_h", preview_max_h)
        if preview_allow_upscale:
            cmd += ["--preview_allow_upscale"]
        if write_npy_now:
            cmd += ['--write-npy']

        # Reuse the same preview session if provided; otherwise mint a new one
        preview_session = (request.POST.get("preview_session") or request.GET.get("preview_session") or "").strip()
        cap_session = preview_session if preview_session else f"cap_{uuid4().hex[:6]}"

        # Build base URL from current request (e.g., http://127.0.0.1:8000)
        scheme = "https" if request.is_secure() else "http"
        server_base = f"{scheme}://{request.get_host()}"
        uplink_key = getattr(settings, "STREAM_UPLINK_KEY", "")

        # Hard-stop any lightweight preview runner on this session so frames don't interleave.
        if preview_session:
            stop_url = f"{server_base}/attendance/stream/run/stop/{preview_session}/"
            try:
                try:
                    requests.get(stop_url, timeout=1)
                except Exception:

                    urllib.request.urlopen(stop_url, timeout=1).read()
            except Exception:
                pass
            else:
                # brief breather so the process actually exits
                import time as _t;
                _t.sleep(0.25)
        # # Append preview/uplink flags (post annotated frames into the SAME session)
        # cmd += ["--preview-session", cap_session, "--server", server_base]
        # if uplink_key:
        #     cmd += ["--uplink-key", uplink_key]
        # Append preview/uplink flags only if not disabled
        if not disable_uplink:
            if cap_session:
                cmd += ["--preview-session", cap_session]
            if server_base:
                cmd += ["--server", server_base]
            if uplink_key:
                cmd += ["--uplink-key", uplink_key, "--uplink-maxfps", "6", "--uplink-quality", "85"]

        # cmd += ["--uplink-maxfps", "6", "--uplink-quality", "85"]
        # Step-B: make preview-only snappier (higher FPS, slightly lower JPEG quality)
        # uplink_maxfps = "12" if preview_only else "6"
        # uplink_quality = "80" if preview_only else "85"
        # cmd += ["--uplink-maxfps", uplink_maxfps, "--uplink-quality", uplink_quality]

        print("CAPTURE CMD:", " ".join(cmd))

        # Run (capture output for parsing; allow enough time for a real capture)
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 15)
        except subprocess.TimeoutExpired:
            if request.headers.get("HX-Request") == "true":
                resp = HttpResponse("<!-- timeout -->")
                resp["HX-Trigger"] = json.dumps({
                    "toast": {"text": "Live capture timed out."},
                    "bisk:refresh-captures": True
                })
                return resp
            messages.error(request, "Live capture timed out.")
            return redirect(request.META.get("HTTP_REFERER") or "../../")
        except Exception as e:
            if request.headers.get("HX-Request") == "true":
                resp = HttpResponse("<!-- error -->")
                resp["HX-Trigger"] = json.dumps({
                    "toast": {"text": f"Capture failed: {e}"},
                    "bisk:refresh-captures": True
                })
                return resp
            messages.error(request, f"Capture failed: {e}")
            return redirect(request.META.get("HTTP_REFERER") or "../../")

        # Parse results
        saved_dir = ""
        stdout = out.stdout or ""
        stderr = out.stderr or ""
        for line in stdout.splitlines():
            if line.startswith("[TOP_CROPS]"):
                # "[TOP_CROPS] saved=3 dir=face_gallery/H123456/live_YYYYmmdd_HHMMSS"
                try:
                    saved_dir = line.split("dir=", 1)[1].strip()
                except Exception:
                    pass

        ok = ("[OK] wrote embedding:" in stdout) or (out.returncode == 0)

        # parse optional "wrote embedding" path for nicer toast
        embed_path = ""
        for line in stdout.splitlines():
            if line.startswith("[OK] wrote embedding:"):
                try:
                    embed_path = line.split(":", 1)[1].strip()
                except Exception:
                    pass

        # # HTMX branch: tiny body + HX-Trigger (DON'T swap the grid)
        # if request.headers.get("HX-Request") == "true":
        #     msg = f"Live capture {'OK' if ok else 'error'} (k={k}, det={det_set})"
        #     if saved_dir:
        #         msg += f" · {saved_dir}"
        #     resp = HttpResponse("<!-- ok -->" if ok else "<!-- error -->")
        #     resp["HX-Trigger"] = json.dumps({
        #         "toast": {"text": msg if ok else f"Capture error: {(stderr or stdout).strip()[:300]}"},
        #         "bisk:refresh-captures": True
        #     })
        #     return resp
        # HTMX branch: tiny body + HX-Trigger (DON'T swap the grid)
        if request.headers.get("HX-Request") == "true":

            msg = f"Live capture {'OK' if ok else 'error'} (k={k}, det={det_set})"
            if saved_dir:
                msg += f" · {saved_dir}"
            if embed_path:
                # Show only the file name to keep toast short
                msg += f" · npy: {Path(embed_path).name}"

            resp = HttpResponse("<!-- ok -->" if ok else "<!-- error -->")
            resp["HX-Trigger"] = json.dumps({
                "toast": {
                    "text": msg if ok else f"Capture error: {(stderr or stdout).strip()[:300]}",
                    # keep toast visible longer (ms)
                    "duration": 12000,
                    # optional: if your global toast handler supports variants
                    "variant": "success" if ok else "error"
                },
                "bisk:refresh-captures": True
            })
            return resp

        # Non-HTMX (full page)
        if ok:
            messages.success(request, f"Live capture OK (k={k}, det={det_set}).")
            if saved_dir:
                messages.info(request, f"Saved crops: {saved_dir}")
        else:
            messages.error(request, f"Capture error: {stderr or stdout or 'unknown'}")

        return redirect(request.META.get("HTTP_REFERER") or "../../")

    # ---- POST: run enrollment with overrides (force) ----
    @transaction.atomic
    @method_decorator(require_POST)
    def re_enroll_run(self, request, pk: int):
        fe = get_object_or_404(FaceEmbedding, pk=pk)

        # Keep the same folder (querystring) when called via HTMX
        is_htmx = (request.headers.get("HX-Request") == "true")
        if is_htmx and not request.GET:
            hx_url = request.headers.get("HX-Current-URL") or ""
            if hx_url:
                parsed = urlparse(hx_url)
                qd = QueryDict(mutable=True)
                qd.update(dict(parse_qsl(parsed.query or "")))
                request.GET = qd

        form = self._ReEnrollForm(request.POST)
        if not form.is_valid():
            if is_htmx:
                resp = HttpResponse('<div class="error">Invalid inputs.</div>')
                resp["HX-Trigger"] = json.dumps({"toast": {"text": "Re-enroll: invalid inputs"}})
                return resp
            messages.error(request, "Invalid inputs.")
            return redirect(request.META.get("HTTP_REFERER") or "../../")

        k = form.cleaned_data.get("k")
        det = form.cleaned_data.get("det_size")
        min_score = form.cleaned_data.get("min_score")
        strict_top = bool(form.cleaned_data.get("strict_top"))

        # Optional selection from the client (MEDIA-relative list)
        try:
            sel = json.loads(request.POST.get("selected") or "[]")
            if not isinstance(sel, list):
                sel = []
        except Exception:
            sel = []

        # If your detector size lives globally, apply override up front
        if det:
            try:
                set_det_size(int(det))
            except Exception:
                pass

        # Stage files and build a robust mapping: basename(lowercased) -> MEDIA-relative path
        stage_dir: Path | None = None
        staged_count = 0
        name_to_rel: dict[str, str] = {}
        media_root = Path(settings.MEDIA_ROOT)

        def _is_img(p: Path) -> bool:
            return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        try:
            if sel:
                stage_dir = Path(tempfile.mkdtemp(prefix=f"tmp_enroll_{fe.student.h_code}_", dir=media_root))
                for rel in sel:
                    p = (media_root / rel).resolve()
                    if media_root not in p.parents and p != media_root:  # safety
                        continue
                    if p.is_file() and _is_img(p):
                        shutil.copy2(p, stage_dir / p.name)
                        staged_count += 1
                        name_to_rel[p.name.lower()] = str(p.relative_to(media_root)).replace(os.sep, "/")
                    elif p.is_dir():
                        for q in p.rglob("*"):
                            if q.is_file() and _is_img(q):
                                shutil.copy2(q, stage_dir / q.name)
                                staged_count += 1
                                name_to_rel[q.name.lower()] = str(q.relative_to(media_root)).replace(os.sep, "/")
        except Exception as e:
            if stage_dir and stage_dir.exists():
                shutil.rmtree(stage_dir, ignore_errors=True)
            if is_htmx:
                resp = HttpResponse(f'<div class="error">Preparing selection failed: {e}</div>')
                resp["HX-Trigger"] = json.dumps({"toast": {"text": f"Selection failed: {e}"}})
                return resp
            messages.error(request, f"Preparing selection failed: {e}")
            return redirect(request.META.get("HTTP_REFERER") or "../../")

        # Run your pipeline; support a few API variants
        try:
            source_hcode = fe.student.h_code
            common = dict(
                k=int(k) if k else (fe.last_used_k or 3),
                force=True,
                min_score=float(min_score) if min_score is not None else 0.0,
                strict_top=bool(strict_top),
            )
            if stage_dir and any(stage_dir.iterdir()):
                sd = str(stage_dir)
                try:
                    res = enroll_student_from_folder(source_hcode, folder_override=sd, **common)
                except TypeError:
                    try:
                        res = enroll_student_from_folder(source_hcode, path=sd, **common)
                    except TypeError:
                        res = enroll_student_from_folder(source_hcode, sd, **common)
            else:
                res = enroll_student_from_folder(source_hcode, **common)
        except Exception as e:
            if stage_dir and stage_dir.exists():
                shutil.rmtree(stage_dir, ignore_errors=True)
            if is_htmx:
                resp = HttpResponse(f'<div class="error">Enroll failed: {e}</div>')
                resp["HX-Trigger"] = json.dumps({"toast": {"text": f"Enroll failed: {e}"}})
                return resp
            messages.error(request, f"Enroll failed: {e}")
            return redirect(request.META.get("HTTP_REFERER") or "../../")
        finally:
            if stage_dir and stage_dir.exists():
                shutil.rmtree(stage_dir, ignore_errors=True)

        if not res or not res.get("ok"):
            msg = res.get("reason") if res else "Unknown error"
            if is_htmx:
                resp = HttpResponse(f'<div class="error">Enroll did not succeed: {msg}</div>')
                resp["HX-Trigger"] = json.dumps({"toast": {"text": f"Enroll failed: {msg}"}})
                return resp
            messages.warning(request, f"Enroll did not succeed: {msg}")
            return redirect("../../")

        # Persist min-score to the active row
        new_id = res.get("face_embedding_id")
        meta = (res or {}).get("meta", {}) or {}
        used_min = meta.get("min_score", min_score)
        try:
            fe_active = FaceEmbedding.objects.select_for_update().get(pk=new_id) if new_id else fe
            if used_min is not None:
                fe_active.last_used_min_score = float(used_min)
                fe_active.save(update_fields=["last_used_min_score"])
        except Exception:
            fe_active = fe  # safe fallback

        # >>> CRITICAL FIX: ensure every used_images_detail has a good MEDIA-relative 'path'
        try:
            if getattr(fe_active, "used_images_detail", None):
                detail = list(fe_active.used_images_detail or [])
                changed = False
                for rec in detail:
                    # normalize a key we can match against our staging map
                    nm_raw = rec.get("name") or rec.get("file") or rec.get("path") or ""
                    nm = os.path.basename(str(nm_raw)).lower()
                    rel = name_to_rel.get(nm)
                    if rel:
                        # Always overwrite; previous value may point to tmp_enroll or be missing
                        rec["path"] = rel
                        changed = True
                if changed:
                    fe_active.used_images_detail = detail
                    fe_active.save(update_fields=["used_images_detail"])
        except Exception:
            pass
        # <<< END FIX

        if is_htmx:
            used = meta.get("used") or meta.get("files_used") or meta.get("images_used") or staged_count
            accepted = meta.get("accepted") or meta.get("n_accepted")
            avg_score = meta.get("avg_score") or meta.get("mean_score")

            bits = [f"Re-enrolled with <b>{used}</b> image(s)"]
            if k:                     bits.append(f"k={k}")
            if det:                   bits.append(f"det_size={det}")
            if min_score is not None: bits.append(f"min_score={min_score}")
            if strict_top:            bits.append("strict-top")
            if accepted is not None:  bits.append(f"accepted={accepted}")
            if avg_score is not None: bits.append(f"avg={avg_score}")

            summary_html = '<div class="success" style="margin-top:8px">' + ", ".join(bits) + ".</div>"

            # Refresh the same folder view
            grid_resp = self.re_enroll_captures(request, pk)
            grid_html = grid_resp.content.decode("utf-8")
            oob = f'<div id="captures-browser" hx-swap-oob="innerHTML">{grid_html}</div>'

            resp = HttpResponse(summary_html + oob)
            resp["HX-Trigger"] = json.dumps({"toast": {"text": f"Re-enrolled {used} image(s)."}})
            return resp

        messages.success(request, f"Re-enrolled {fe.student.h_code} successfully.")
        return redirect("../../")

    def re_enroll_captures(self, request, pk: int):

        def _media_url(p: Path) -> str:
            base = (settings.MEDIA_URL or "/media/").rstrip("/") + "/"
            rel = os.path.relpath(str(p), settings.MEDIA_ROOT).replace(os.sep, "/")
            return base + rel

        def _rel_media(p: Path) -> str:
            return os.path.relpath(str(p), settings.MEDIA_ROOT).replace(os.sep, "/")

        def _is_img(p: Path) -> bool:
            return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        fe = get_object_or_404(FaceEmbedding, pk=pk)
        student = fe.student
        h = getattr(student, "h_code", "")
        if not h:
            return HttpResponse("<p>Missing student H-code.</p>")

        root = Path(student_gallery_dir(h))

        # navigation state
        q_date = ((request.GET.get("date") or request.POST.get("date") or "")).strip()
        q_period = ((request.GET.get("period") or request.POST.get("period") or "")).strip()
        q_cam = ((request.GET.get("camera") or request.POST.get("camera") or "")).strip()

        cur = root
        if q_date:
            cur = cur / q_date
        if q_period:
            cur = cur / q_period
        if q_cam:
            cur = cur / q_cam

        cards_html = []
        if not cur.exists():
            html = "<p>Nothing captured yet.</p>"
        else:
            # Folders first (date/period/camera levels)
            entries = sorted(cur.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))

            # 1) batch-score only the images in this folder once
            imgs = [p for p in entries if p.is_file() and _is_img(p)]
            stats_map = {}
            if imgs:
                try:
                    rows = score_images([str(p) for p in imgs]) or []
                    # index by absolute filesystem path (normalized)
                    import os as _os
                    stats_map = {_os.path.normpath(r.get("path", "")): r for r in rows if r}
                except Exception:
                    stats_map = {}

            # 2) render folders first, then images with stats
            for p in entries:
                if p.is_dir():
                    rel = _rel_media(p)
                    href = reverse("admin:attendance_faceembedding_captures", args=[fe.pk])
                    qs = []
                    if not q_date:
                        qs.append(f"date={p.name}")
                    elif not q_period:
                        qs.append(f"date={q_date}&period={p.name}")
                    elif not q_cam:
                        qs.append(f"date={q_date}&period={q_period}&camera={p.name}")
                    open_href = href + ("?" + "&".join(qs) if qs else "")
                    cards_html.append(
                        self._row_card(p.name,
                                       "date" if not q_date else ("period" if not q_period else "camera"),
                                       rel=rel, thumb_url=None, is_folder=True, open_href=open_href)
                    )
                elif _is_img(p):
                    rel = _rel_media(p)
                    url = _media_url(p)
                    key = __import__("os").path.normpath(str(p))
                    rec = stats_map.get(key) or {}

                    def _to_float(v):
                        try:
                            return float(v)
                        except Exception:
                            return None

                    meta = {
                        "score": _to_float(rec.get("score")),
                        "sharp": _to_float(rec.get("sharp")),
                        "bright": _to_float(rec.get("bright")),
                    }
                    cards_html.append(
                        self._row_card(p.name, meta, rel=rel, thumb_url=url, is_folder=False, open_href=None)
                    )

            # --- Breadcrumbs ---
            base_href = reverse("admin:attendance_faceembedding_captures", args=[fe.pk])

            def _hxlink(text: str, href: str) -> str:
                return (f'<a class="crumb" hx-get="{href}" '
                        f'hx-target="#captures-browser" hx-swap="innerHTML">{text}</a>')

            trail = [_hxlink(student.h_code, base_href)]
            if q_date:
                trail.append('<span class="sep">/</span>' + _hxlink(q_date, f'{base_href}?date={q_date}'))
            if q_period:
                trail.append(
                    '<span class="sep">/</span>' + _hxlink(q_period, f'{base_href}?date={q_date}&period={q_period}'))
            if q_cam:
                trail.append('<span class="sep">/</span>' + _hxlink(q_cam,
                                                                    f'{base_href}?date={q_date}&period={q_period}&camera={q_cam}'))

            parent_href = None
            if q_cam:
                parent_href = f'{base_href}?date={q_date}&period={q_period}'
            elif q_period:
                parent_href = f'{base_href}?date={q_date}'
            elif q_date:
                parent_href = base_href

            up_html = (f'<a class="up" hx-get="{parent_href}" '
                       f'hx-target="#captures-browser" hx-swap="innerHTML">← Up</a>') if parent_href else ''

            crumbs_html = f'<div class="crumbs">{up_html}<span class="trail">{" ".join(trail)}</span></div>'

            html = crumbs_html + '<div class="grid">' + "".join(cards_html) + "</div>"

        return HttpResponse(html)

        # return HttpResponse(html)

    @method_decorator(require_POST)
    def re_enroll_captures_delete(self, request, pk: int):
        """
        Delete selected files/folders under MEDIA_ROOT safely, then re-render current listing.
        Expects POST 'delete_selected' = JSON array of MEDIA-relative paths.
        """
        # preserve current query params when coming from HTMX
        hx_url = request.headers.get("HX-Current-URL") or ""
        if hx_url and not request.GET:
            from urllib.parse import urlparse, parse_qsl
            parsed = urlparse(hx_url)
            qd = QueryDict(mutable=True)
            qd.update(dict(parse_qsl(parsed.query or "")))
            request.GET = qd

        try:
            fe = get_object_or_404(FaceEmbedding, pk=pk)
            sel = json.loads(request.POST.get("delete_selected") or "[]")
            if not isinstance(sel, list):
                return HttpResponseBadRequest("bad payload")
        except Exception:
            return HttpResponseBadRequest("bad payload")

        base = Path(settings.MEDIA_ROOT).resolve()
        removed = 0
        deleted_names: list[str] = []  # NEW

        for rel in sel:
            try:
                p = (base / rel).resolve()
                # safety: keep within MEDIA_ROOT
                if base not in p.parents and p != base:
                    continue
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                    removed += 1
                    deleted_names.append(Path(rel).name)  # preview folder name
                elif p.exists():
                    p.unlink(missing_ok=True)
                    removed += 1
                    deleted_names.append(Path(rel).name)  # preview file name
            except Exception:
                # swallow; continue best-effort
                pass

        # optional Django message (won't show in the modal, toast will)
        messages.info(request, f"Removed {removed} item(s).")

        # Re-render the grid and attach a toast
        resp = self.re_enroll_captures(request, pk)  # returns updated grid HTML

        preview = ", ".join(deleted_names[:4]) + ("…" if len(deleted_names) > 4 else "")
        resp["HX-Trigger"] = json.dumps({
            "toast": {"text": f"Deleted {removed} item(s){(': ' + preview) if preview else ''}"}
        })
        return resp

    class Media:
        # Only defines a single CSS variable (--bisk-chip-fg) that flips in dark mode.
        css = {"all": ("attendance/admin_chips.css",)}
        js = (
            "attendance/re_enroll_popup.js",  # opens +more in a popup window
            "attendance/crop_modal.js",  # your image preview overlay (unchanged)
        )
