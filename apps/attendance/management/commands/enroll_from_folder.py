# python manage.py enroll_from_folder --k 3 --force --det-size 1024
# What does --k 3 do exactly?
# We score all images in face_gallery/<H_CODE>/ by sharpness/brightness.
# We select the best K (max K = the number you pass).
# If there are 10 images and --k 3, it will pick the best 3 of the 10.
# apps/attendance/management/commands/enroll_from_folder.py
from __future__ import annotations
from django.core.management.base import BaseCommand
from apps.attendance.utils.embeddings import enroll_student_from_folder, set_det_size
from apps.attendance.models import Student


class Command(BaseCommand):
    help = "Compute averaged 512D embeddings from face_gallery/<H_CODE>/, store npy + FaceEmbedding bytes/metadata."

    def add_arguments(self, p):
        p.add_argument("--student", type=str, help="H_CODE of a single student")
        p.add_argument("--k", type=int, default=3, help="Top-K images to average")
        p.add_argument("--force", action="store_true", help="Overwrite existing")
        p.add_argument("--det-size", type=int, default=640, help="Detector input size (square), e.g. 640/1024")

    def handle(self, *args, **opts):
        det = int(opts["det_size"])
        set_det_size(det)

        code = opts.get("student")
        k = int(opts.get("k") or 3)
        force = bool(opts.get("force"))
        codes = [code] if code else list(Student.objects.filter(is_active=True).values_list("h_code", flat=True))

        ok = fail = 0
        for h in codes:
            res = enroll_student_from_folder(h, k=k, force=force)
            if res.get("ok"):
                ok += 1
                self.stdout.write(self.style.SUCCESS(
                    f"[{h}] OK -> {res.get('embedding_path')} (created={res.get('created')})"
                ))
            else:
                fail += 1
                self.stderr.write(self.style.WARNING(f"[{h}] FAIL: {res.get('reason')}"))
        self.stdout.write(self.style.NOTICE(f"Done. ok={ok} fail={fail}"))
