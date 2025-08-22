# apps/attendance/management/commands/sort_gallery_intake.py
# python manage.py sort_gallery_intake --policy rename --clean-empty
# --policy overwrite → replace destination if exists
# --policy skip → leave existing files, don’t move duplicates
# --policy rename (default) → keeps both by adding __1, __2, …
from __future__ import annotations

import os, re, shutil, time
from typing import Tuple
from django.core.management.base import BaseCommand, CommandError
from apps.attendance.utils.media_paths import DIRS, student_gallery_dir

# Match an H-code anywhere in the filename, e.g. H123456, H123456-1, H123456_2
# We capture just the H123456 part; case-insensitive.
H_RE = re.compile(r"(?i)(H\d{6,})")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def collision_target(dst: str, policy: str) -> str | None:
    if not os.path.exists(dst):
        return dst
    if policy == "overwrite":
        return dst
    if policy == "skip":
        return None
    # rename
    base, ext = os.path.splitext(dst)
    i = 1
    while True:
        cand = f"{base}__{i}{ext}"
        if not os.path.exists(cand):
            return cand
        i += 1


class Command(BaseCommand):
    help = (
        "Sort MEDIA_ROOT/face_gallery_inbox/* into per-student folders under face_gallery/<H_CODE>/ "
        "by parsing H-codes from filenames. Unmatched go to _unsorted/."
    )

    def add_arguments(self, p):
        p.add_argument("--policy", choices=["overwrite", "skip", "rename"], default="rename",
                       help="Collision handling for destination files")
        p.add_argument("--dry-run", action="store_true", help="Do not move, only log actions")
        p.add_argument("--clean-empty", action="store_true", help="Remove empty subfolders in inbox after move")

    def handle(self, *args, **opts):
        inbox = DIRS["FACE_GALLERY_INBOX"]
        unsorted_dir = DIRS["FACE_GALLERY_INBOX_UNSORTED"]
        if not os.path.isdir(inbox):
            raise CommandError(f"Inbox not found: {inbox}")

        policy = opts["policy"]
        dry = bool(opts["dry_run"])
        clean = bool(opts["clean_empty"])

        moved = skipped = unmatched = 0
        t0 = time.time()

        for root, dirs, files in os.walk(inbox):
            # skip the _unsorted dir itself
            if os.path.abspath(root) == os.path.abspath(unsorted_dir):
                continue
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext not in IMG_EXTS:
                    continue
                src = os.path.join(root, name)
                m = H_RE.search(name)
                if not m:
                    # park to _unsorted
                    os.makedirs(unsorted_dir, exist_ok=True)
                    dst = os.path.join(unsorted_dir, name)
                    if not dry:
                        shutil.move(src, dst)
                    unmatched += 1
                    self.stdout.write(self.style.WARNING(f"UNMATCHED: {name} -> _unsorted/"))
                    continue

                h_code = m.group(1).upper()
                dst_dir = student_gallery_dir(h_code)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, name)

                tgt = collision_target(dst, policy=policy)
                if tgt is None:
                    skipped += 1
                    self.stdout.write(self.style.NOTICE(f"SKIP (exists): {dst}"))
                    continue

                if dry:
                    self.stdout.write(f"DRY MOVE {src} -> {tgt}")
                else:
                    shutil.move(src, tgt)
                moved += 1

                if (moved + skipped + unmatched) % 50 == 0:
                    self.stdout.write(f"... progress: moved={moved} skipped={skipped} unmatched={unmatched}")

        if clean and not dry:
            # remove empty folders inside inbox
            for root, dirs, files in os.walk(inbox, topdown=False):
                if os.path.abspath(root) == os.path.abspath(unsorted_dir):
                    continue
                if not dirs and not files:
                    try:
                        os.rmdir(root)
                    except OSError:
                        pass

        dt = time.time() - t0
        self.stdout.write(self.style.SUCCESS(
            f"Done in {dt:.1f}s. moved={moved} skipped={skipped} unmatched={unmatched}"
        ))
