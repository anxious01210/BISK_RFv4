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
}


def ensure_media_tree() -> None:
    for path in DIRS.values():
        os.makedirs(path, exist_ok=True)


def student_gallery_dir(h_code: str) -> str:
    return os.path.join(DIRS["FACE_GALLERY"], h_code)
