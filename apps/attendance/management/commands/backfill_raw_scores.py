from __future__ import annotations
from django.core.management.base import BaseCommand
from django.db import transaction
from apps.attendance.models import FaceEmbedding


def _coerce_raw01(v):
    try:
        x = float(v)
    except Exception:
        return None
    if x > 1.0:  # legacy percent
        x = x / 100.0
    if x < 0.0:
        x = 0.0
    if x > 1.0:
        x = 1.0
    return x


class Command(BaseCommand):
    help = "Normalize used_images_detail to canonical raw01∈[0..1], add rank if missing, recompute avg_used_score."

    def add_arguments(self, p):
        p.add_argument("--dry-run", action="store_true", help="Print changes, do not write.")

    @transaction.atomic
    def handle(self, *args, **opts):
        dry = bool(opts["dry_run"])
        fixed_sets = 0
        total_rows = 0

        qs = FaceEmbedding.objects.all().only("id", "used_images_detail", "avg_used_score")
        for fe in qs.iterator():
            det = fe.used_images_detail or []
            if not isinstance(det, list) or not det:
                continue

            # Convert to canonical records and collect raw01
            new_det = []
            changes = False
            for rec in det:
                if not isinstance(rec, dict):
                    continue
                name = rec.get("name") or rec.get("path")
                raw01 = rec.get("raw01", rec.get("score", None))
                raw01 = _coerce_raw01(raw01)
                # carry across fields
                item = {
                    "name": name,
                    "raw01": raw01,
                    "det_size": rec.get("det_size"),
                    "det_conf": rec.get("det_conf"),
                    "sharp": rec.get("sharp"),
                    "bright": rec.get("bright"),
                    "selected": rec.get("selected", True),
                }
                # rank may be missing for old rows
                if "rank" in rec and isinstance(rec["rank"], int):
                    item["rank"] = rec["rank"]
                new_det.append(item)

            # Re-rank by raw01 desc for selected images
            selected = [r for r in new_det if r.get("selected")]
            selected_sorted = sorted(selected, key=lambda r: (r.get("raw01") or 0.0), reverse=True)
            for i, r in enumerate(selected_sorted, start=1):
                if r.get("rank") != i:
                    r["rank"] = i
                    changes = True

            # Compute avg over selected raw01 values
            vals = [float(r["raw01"]) for r in selected_sorted if r.get("raw01") is not None]
            new_avg = round(sum(vals) / len(vals), 6) if vals else 0.0
            if abs((fe.avg_used_score or 0.0) - new_avg) > 1e-6:
                changes = True

            if changes:
                fixed_sets += 1
                self.stdout.write(f"[{fe.id}] avg_used_score {fe.avg_used_score} -> {new_avg}  "
                                  f"(selected {len(selected_sorted)} imgs)")
                if not dry:
                    fe.used_images_detail = new_det
                    fe.avg_used_score = new_avg
                    fe.save(update_fields=["used_images_detail", "avg_used_score"])

            total_rows += 1

        self.stdout.write(self.style.SUCCESS(
            f"Checked {total_rows} FaceEmbedding rows. Fixed={fixed_sets}. Dry-run={dry}."
        ))
