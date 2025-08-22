# apps/attendance/management/commands/check_media_folders.py
# Manual: check/create folders => python3 manage.py check_media_folders
from __future__ import annotations
from django.core.management.base import BaseCommand
from apps.attendance.utils.media_paths import DIRS, ensure_media_tree

class Command(BaseCommand):
    help = "Create/verify required MEDIA subdirectories for gallery, embeddings, logs, and reports."

    def handle(self, *args, **opts):
        ensure_media_tree()
        for k, v in DIRS.items():
            self.stdout.write(f"{k}: {v}")
        self.stdout.write(self.style.SUCCESS("Media folder check complete."))
