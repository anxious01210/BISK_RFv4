# python manage.py enroll_from_folder --k 3 --force --det-size 1024
# What does --k 3 do exactly?
# We score all images in face_gallery/<H_CODE>/ by sharpness/brightness.
# We select the best K (max K = the number you pass).
# If there are 10 images and --k 3, it will pick the best 3 of the 10.
# apps/attendance/management/commands/enroll_from_folder.py
from __future__ import annotations
from django.core.management.base import BaseCommand
from django.conf import settings
from apps.attendance.utils.embeddings import enroll_student_from_folder, set_det_size
from apps.attendance.models import Student


class Command(BaseCommand):
    help = "Average top-K images per student into a 512D embedding, save .npy, and update FaceEmbedding bytes + metadata."

    def add_arguments(self, p):
        p.add_argument("--student", type=str, help="H_CODE of a single student")
        p.add_argument("--k", type=int, default=getattr(settings, "EMBEDDING_DEFAULT_K", 3),
                       help="Top-K images to average (default from settings)")
        p.add_argument("--force", action="store_true", help="Overwrite existing")
        p.add_argument("--det-size", type=int, default=getattr(settings, "EMBEDDING_DEFAULT_DET_SIZE", 640),
                       help="Detector input size (square), e.g. 640/1024 (default from settings)")
        p.add_argument("--min-score", type=float, default=getattr(settings, "EMBEDDING_MIN_SCORE", 0.0),
                       help="Normalized score floor 0..1; images below this are dropped if --strict-top")
        p.add_argument("--strict-top", action="store_true",
                       default=getattr(settings, "EMBEDDING_USE_STRICT_TOP", False),
                       help="Enable score floor filtering before taking top-K")

    def handle(self, *args, **opts):
        det = int(opts["det_size"])
        set_det_size(det)

        code = opts.get("student")
        k = int(opts.get("k") or 3)
        force = bool(opts.get("force"))
        min_score = float(opts.get("min_score"))
        strict = bool(opts.get("strict_top"))
        codes = [code] if code else list(Student.objects.filter(is_active=True).values_list("h_code", flat=True))

        ok = fail = 0
        for h in codes:
            res = enroll_student_from_folder(h, k=k, force=force, min_score=min_score, strict_top=strict)
            if res.get("ok"):
                ok += 1
                self.stdout.write(self.style.SUCCESS(
                    f"[{h}] OK -> {res.get('embedding_path')} (created={res.get('created')})"
                ))
            else:
                fail += 1
                self.stderr.write(self.style.WARNING(f"[{h}] FAIL: {res.get('reason')}"))
        self.stdout.write(self.style.NOTICE(f"Done. ok={ok} fail={fail}"))
