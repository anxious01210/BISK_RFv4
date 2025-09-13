# apps/attendance/utils/media_paths.py
from __future__ import annotations
import os
from typing import Dict
from django.conf import settings

# Canonical locations under MEDIA_ROOT
MEDIA = getattr(settings, "MEDIA_ROOT", ".")

DIRS: Dict[str, str] = {
    "FACE_GALLERY": os.path.join(MEDIA, "face_gallery"),
    "FACE_GALLERY_INBOX": os.path.join(MEDIA, "face_gallery_inbox"),
    "FACE_GALLERY_INBOX_UNSORTED": os.path.join(MEDIA, "face_gallery_inbox", "_unsorted"),
    "EMBEDDINGS": os.path.join(MEDIA, "embeddings"),
    "ATTN_CROPS": os.path.join(MEDIA, "attendance_crops"),
    "LOGS_DEBUG_FACES": os.path.join(MEDIA, "logs", "debug_faces"),
    "REPORTS_QUALITY": os.path.join(MEDIA, "reports", "gallery_quality"),
    "REPORTS_ENROLL": os.path.join(MEDIA, "reports", "enroll"),
    "UPLOAD_TMP": os.path.join(MEDIA, "upload_tmp"),
}


def ensure_media_tree() -> None:
    for path in DIRS.values():
        os.makedirs(path, exist_ok=True)


def student_gallery_dir(h_code: str) -> str:
    return os.path.join(DIRS["FACE_GALLERY"], h_code)


# Save crops under:
# media/face_gallery/<HCODE>/<SUBDIR>/<YYYY-MM-DD>/<PERIOD_OR__none_>/<CAMERA_NAME>/
def student_capture_dir(
        h_code: str,
        camera_name: str,
        period_id: str | int | None,
        when=None,
        *,
        subdir: str | None = None,
        include_period: bool = True,
) -> str:
    import datetime, os
    from django.conf import settings
    from .media_paths import DIRS, student_gallery_dir  # keep using canonical roots

    root = student_gallery_dir(h_code)
    when = when or datetime.datetime.now()
    date_str = when.strftime("%Y-%m-%d")

    # from DB or fallback
    subdir = (subdir or "captures").strip() or "captures"

    # sanitize cam name for filesystem
    safe_cam = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (camera_name or "camera"))

    parts = [root, subdir, date_str]
    if include_period:
        parts.append(str(period_id) if period_id not in (None, "",) else "_none_")
    parts.append(safe_cam)

    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path
