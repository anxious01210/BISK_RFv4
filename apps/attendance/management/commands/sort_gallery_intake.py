# apps/attendance/management/commands/sort_gallery_intake.py
# python manage.py sort_gallery_intake --policy rename --clean-empty
# --policy overwrite → replace destination if exists
# --policy skip → leave existing files, don’t move duplicates
# --policy rename (default) → keeps both by adding __1, __2, …


# apps/attendance/management/commands/sort_gallery_intake.py
from __future__ import annotations
import os, re, shutil, time
from django.core.management.base import BaseCommand, CommandError
import argparse
from django.conf import settings
from apps.attendance.utils.media_paths import DIRS, student_gallery_dir
from apps.attendance.utils.facescore import crop_face_to_file, detect_with_cascade
import cv2
import numpy as np
from apps.attendance.models import Student

# EXACTLY H + 6 digits (7 chars total), and do not match when more digits follow
H_RE = re.compile(r'(?i)(H\d{6})(?!\d)')
ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


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
    help = "Sort MEDIA_ROOT/face_gallery_inbox into per-student folders under face_gallery/<Hnnnnnn>/ " \
           "Optionally crop faces; unmatched files go to _unsorted/."

    def add_arguments(self, p):
        p.add_argument("--policy", choices=("rename", "skip", "overwrite"), default="rename",
                       help="Collision handling at destination")
        p.add_argument("--dry-run", action="store_true",
                       help="Print what would happen; do not write/move/delete anything.")
        p.add_argument("--clean-empty", action="store_true", help="Remove empty dirs from inbox afterwards.")
        p.add_argument("--crop-faces", action=argparse.BooleanOptionalAction,
                       default=getattr(settings, "FACE_INTAKE_CROP_FACES", False),
                       help="Enable/disable face cropping regardless of settings.")
        p.add_argument("--keep-raw", action=argparse.BooleanOptionalAction,
                       default=getattr(settings, "FACE_INTAKE_SAVE_ORIGINAL", True),
                       help="When cropping, keep original in Hxxxxxx/raw/. Use --no-keep-raw to discard originals.")
        p.add_argument("--det-size", type=int,
                       default=getattr(settings, "FACE_DET_SIZE_DEFAULT", 800),
                       help="Starting det-size for detection cascade.")
        p.add_argument("--require-student", action=argparse.BooleanOptionalAction, default=True,
                       help="Only accept files whose H-code exists in Student table (default: on).")
        p.add_argument("--active-only", action=argparse.BooleanOptionalAction, default=True,
                       help="When --require-student, require student.is_active=True (default: on).")
        p.add_argument("--enhance", action=argparse.BooleanOptionalAction,
                       default=getattr(settings, "FACE_INTAKE_ENHANCE", False),
                       help="Apply mild enhancement to ROI crops (normalize brightness, CLAHE, unsharp).")

    def handle(self, *args, **opt):
        inbox = DIRS["FACE_GALLERY_INBOX"]
        unsorted_dir = DIRS.get("FACE_GALLERY_INBOX_UNSORTED") or os.path.join(inbox, "_unsorted")
        if not os.path.isdir(inbox):
            raise CommandError(f"Inbox not found: {inbox}")

        policy = opt["policy"]
        dry = bool(opt.get("dry_run"))
        clean = bool(opt.get("clean_empty"))
        crop_on = bool(opt.get("crop_faces"))
        keep_raw = bool(opt.get("keep_raw"))
        det_size = int(opt.get("det_size"))
        enhance = bool(opt.get("enhance"))

        moved = skipped = unmatched = 0
        unknown_student = 0
        fallback_original = 0
        t0 = time.time()

        for root, dirs, files in os.walk(inbox):
            # skip the _unsorted dir itself
            if os.path.abspath(root) == os.path.abspath(unsorted_dir):
                continue
            for name in sorted(files):
                ext = os.path.splitext(name)[1].lower()
                if ext not in ALLOWED:
                    continue
                src = os.path.join(root, name)
                m = H_RE.search(name)
                if not m:
                    # No valid Hnnnnnn → move to _unsorted
                    os.makedirs(unsorted_dir, exist_ok=True)
                    dst = os.path.join(unsorted_dir, name)
                    if dry:
                        self.stdout.write(f"UNMATCHED: {name} -> {dst}")
                    else:
                        shutil.move(src, dst)
                    unmatched += 1
                    continue

                h_code = m.group(1).upper()  # exactly H + 6 digits
                # Require a Student (optionally active)
                if opt.get("require_student"):
                    q = Student.objects.filter(h_code=h_code)
                    if opt.get("active_only"):
                        q = q.filter(is_active=True)
                    if not q.exists():
                        # Park to _unsorted (or just log on dry-run)
                        os.makedirs(unsorted_dir, exist_ok=True) if not dry else None
                        dst = os.path.join(unsorted_dir, name)
                        if dry:
                            self.stdout.write(f"UNKNOWN_STUDENT: {name} ({h_code}) -> {dst}")
                        else:
                            shutil.move(src, dst)
                        unknown_student += 1
                        continue

                dst_dir = student_gallery_dir(h_code)
                if not dry:
                    os.makedirs(dst_dir, exist_ok=True)

                if crop_on:
                    # Cropped file goes directly under Hxxxxxx/<name>
                    crop_dst = os.path.join(dst_dir, name)
                    if dry:
                        # Dry-run: detect only; DO NOT write/move/delete or create dirs.
                        try:
                            arr = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
                            bbox, conf, used, prov = detect_with_cascade(arr, det_size)
                            det_info = f"det={used}, conf={conf:.2f}" if bbox else "NO_FACE"
                        except Exception:
                            det_info = "detect=error"
                        raw_where = f"{h_code}/raw/{name}" if keep_raw else "(discarded)"
                        self.stdout.write(f"CROP: {src} -> {crop_dst}  [{det_info}]  raw->{raw_where}")
                    else:
                        # Write crop; if it fails, move original into main dest folder
                        meta = crop_face_to_file(src, crop_dst, start_size=det_size, enhance=enhance)
                        if meta.get("ok"):
                            if keep_raw:
                                raw_dir = os.path.join(dst_dir, "raw")
                                os.makedirs(raw_dir, exist_ok=True)
                                raw_dst = collision_target(os.path.join(raw_dir, name), policy)
                                if raw_dst:
                                    shutil.move(src, raw_dst)
                                else:
                                    skipped += 1
                            else:
                                try:
                                    os.remove(src)
                                except OSError:
                                    pass
                            moved += 1
                        else:
                            # crop failed → move original into main dest folder
                            main_dst = collision_target(os.path.join(dst_dir, name), policy)
                            if main_dst:
                                shutil.move(src, main_dst)
                                moved += 1
                                fallback_original += 1
                            else:
                                skipped += 1
                else:
                    # No crop: move original straight into gallery
                    main_dst = collision_target(os.path.join(dst_dir, name), policy)
                    if dry:
                        self.stdout.write(
                            f"MOVE: {src} -> {h_code}/{os.path.basename(main_dst) if main_dst else '(skipped)'}")
                    else:
                        if main_dst:
                            shutil.move(src, main_dst);
                            moved += 1
                        else:
                            skipped += 1

        if clean and not dry:
            # remove empty dirs inside inbox (but keep _unsorted if present)
            for r, dirs, files in os.walk(inbox, topdown=False):
                if os.path.abspath(r) == os.path.abspath(unsorted_dir):
                    continue
                if not dirs and not files:
                    try:
                        os.rmdir(r)
                    except OSError:
                        pass

        dt = time.time() - t0
        self.stdout.write(self.style.SUCCESS(
            f"Done in {dt:.1f}s. moved={moved} skipped={skipped} unmatched={unmatched}"
        ))

        dt = time.time() - t0
        summary = f"Done in {dt:.1f}s. moved={moved} skipped={skipped} unmatched={unmatched}"
        if opt.get('require_student'):
            summary += f" unknown_student={unknown_student}"
        if crop_on:
            summary += f" fallback_original={fallback_original}"
        self.stdout.write(self.style.SUCCESS(summary))
