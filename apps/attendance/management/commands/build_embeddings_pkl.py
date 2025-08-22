from __future__ import annotations
import os, pickle, numpy as np
from django.core.management.base import BaseCommand, CommandError
from apps.attendance.utils.media_paths import DIRS
from apps.attendance.utils.embeddings import l2_normalize
from apps.attendance.models import Student


class Command(BaseCommand):
    help = "Stack all embeddings/*.npy for ACTIVE students into embeddings_dim512.pkl with L2-normalized rows."

    def add_arguments(self, p):
        p.add_argument("--dim", type=int, default=512)
        p.add_argument("--force", action="store_true")

    def handle(self, *args, **opts):
        dim = int(opts["dim"])
        force = bool(opts["force"])
        emb_dir = DIRS["EMBEDDINGS"]
        out_pkl = os.path.join(emb_dir, f"embeddings_dim{dim}.pkl")
        if not os.path.isdir(emb_dir):
            raise CommandError(f"Embeddings dir not found: {emb_dir}")
        active = set(Student.objects.filter(is_active=True).values_list("h_code", flat=True))

        xs, codes, skipped = [], [], 0
        for name in os.listdir(emb_dir):
            if not name.lower().endswith(".npy"):
                continue
            h_code = os.path.splitext(name)[0]
            if h_code not in active:
                continue
            vec = np.load(os.path.join(emb_dir, name)).astype(np.float32)
            if vec.ndim != 1 or vec.shape[0] != dim:
                skipped += 1
                continue
            xs.append(l2_normalize(vec))
            codes.append(h_code)
        if not xs:
            raise CommandError("No embeddings found to pack.")

        X = np.stack(xs, axis=0).astype(np.float32)
        payload = {"dim": dim, "codes": codes, "embeddings": X}

        if os.path.exists(out_pkl) and not force:
            raise CommandError(f"Output exists: {out_pkl} (use --force)")

        with open(out_pkl, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.stdout.write(self.style.SUCCESS(f"Wrote: {out_pkl}  (N={len(codes)}, dim={dim}, skipped={skipped})"))
