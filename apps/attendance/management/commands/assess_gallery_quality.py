# Lightweight quality pass; writes CSV under reports/gallery_quality/
from __future__ import annotations
import csv, os, time
from typing import List, Tuple
import numpy as np
from django.core.management.base import BaseCommand
from apps.attendance.utils.media_paths import DIRS, student_gallery_dir
from apps.attendance.utils.embeddings import load_bgr, image_score
from apps.attendance.models import Student


class Command(BaseCommand):
    help = "Assess gallery quality (sharpness, brightness) and produce a CSV report."

    def add_arguments(self, p):
        p.add_argument("--student", type=str, help="Assess a single student by H_CODE")
        p.add_argument("--min-images", type=int, default=1, help="Warn when fewer than this number exist")

    def handle(self, *args, **opts):
        h = opts.get("student")
        min_images = int(opts["min_images"])

        students = (Student.objects.filter(h_code=h, is_active=True)
                    if h else Student.objects.filter(is_active=True))
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_csv = os.path.join(DIRS["REPORTS_QUALITY"], f"gallery_quality_{ts}.csv")
        os.makedirs(DIRS["REPORTS_QUALITY"], exist_ok=True)

        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "h_code", "image_path", "sharpness", "brightness", "flags"
            ])

            for st in students:
                folder = student_gallery_dir(st.h_code)
                imgs = []
                if os.path.isdir(folder):
                    for name in os.listdir(folder):
                        if os.path.splitext(name)[1].lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                            imgs.append(os.path.join(folder, name))
                if len(imgs) < min_images:
                    w.writerow([st.h_code, "", "", "", f"LOW_COUNT<{min_images}"])
                    continue

                # compute stats
                for p in imgs:
                    bgr = load_bgr(p)
                    if bgr is None:
                        w.writerow([st.h_code, p, "", "", "READ_FAIL"])
                        continue
                    _, sharp, bright = image_score(bgr)
                    flags = []
                    if sharp < 80:  # heuristic, tune later
                        flags.append("BLURRY")
                    if bright < 70:  # heuristic
                        flags.append("DARK")
                    if bright > 200:
                        flags.append("OVEREXPOSED")
                    w.writerow([st.h_code, os.path.relpath(p, DIRS["FACE_GALLERY"]), f"{sharp:.1f}", f"{bright:.1f}",
                                "|".join(flags)])

        self.stdout.write(self.style.SUCCESS(f"Wrote: {out_csv}"))
