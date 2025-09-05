# apps/scheduler/management/commands/bisk_sanity_policy_resources.py
from django.core.management.base import BaseCommand
import sys
from django.db.models import Exists, OuterRef

from apps.cameras.models import Camera
from apps.scheduler.models import (
    StreamProfile, CameraResourceOverride, GlobalResourceSettings
)

# Legacy resource fields that must be empty on Camera after Phase 4
LEGACY_CAMERA_FIELDS = ["device", "hwaccel", "gpu_index", "cpu_affinity", "nice"]

# Profile fields considered resource-ish in your schema; must be empty
# (rtsp_transport / fps / detection_set / hb_interval / snapshot_every are POLICY, keep them)
LEGACY_PROFILE_FIELDS = ["device", "hwaccel", "gpu_index", "cpu_affinity", "nice"]

# Values that mean "effectively empty" for each model
NEUTRAL_CAMERA = {
    "device": "cpu",       # Camera default
    "hwaccel": "none",     # Camera default
    "gpu_index": 0,        # Camera default
    "cpu_affinity": "",    # blank OK
    "nice": 0,             # Camera default
}
NEUTRAL_PROFILE = {
    "device": "",          # blank = inherit
    "hwaccel": "",         # blank = inherit
    "gpu_index": None,     # null OK
    "cpu_affinity": "",    # blank OK
    "nice": None,          # null OK
}



def _model_has_field(model, name: str) -> bool:
    return any(f.name == name for f in model._meta.get_fields())

def _is_effectively_set(obj, fields, neutral_map):
    """
    Return True if ANY field is set to a non-neutral, non-empty value.
    """
    for f in fields:
        if not _model_has_field(obj.__class__, f):
            continue
        v = getattr(obj, f, None)
        # Treat these as "empty"
        if v in (None, "", []):
            continue
        if f in neutral_map and v == neutral_map[f]:
            continue
        return True
    return False

def _model_has_field(model, name: str) -> bool:
    return any(f.name == name for f in model._meta.get_fields())


def _any_nonempty(obj, fields) -> bool:
    for f in fields:
        if not _model_has_field(obj.__class__, f):
            continue
        v = getattr(obj, f, None)
        if v not in (None, "", []):
            return True
    return False


class Command(BaseCommand):
    help = "Phase 4 posture check: policy vs resources. Exits non-zero on violations."

    def add_arguments(self, parser):
        parser.add_argument(
            "--strict-cro",
            action="store_true",
            help="Treat missing active CRO for an active camera as a violation instead of a warning.",
        )

    def handle(self, *args, **opts):
        strict_cro = opts.get("strict_cro", False)
        violations, warnings = [], []

        # 1) Exactly one active GRS
        has_is_active = _model_has_field(GlobalResourceSettings, "is_active")
        if has_is_active:
            n = GlobalResourceSettings.objects.filter(is_active=True).count()
            if n != 1:
                violations.append(f"[GRS] Expected exactly 1 active row, found {n}.")
        else:
            # Fallback if your GRS didn’t ship is_active (older migration)
            n = GlobalResourceSettings.objects.count()
            if n != 1:
                violations.append(f"[GRS] Expected exactly 1 row, found {n}.")

        # 2) Cameras must NOT carry legacy resource values anymore
        bad_cam_ids = [
            c.id for c in Camera.objects.all().only("id")
            if _is_effectively_set(c, LEGACY_CAMERA_FIELDS, NEUTRAL_CAMERA)
        ]
        if bad_cam_ids:
            violations.append(f"[Camera] Legacy resource fields still set: IDs {sorted(bad_cam_ids)}")

        # 3) StreamProfiles must NOT carry resource-ish values
        bad_sp_ids = [
            s.id for s in StreamProfile.objects.all().only("id")
            if _is_effectively_set(s, LEGACY_PROFILE_FIELDS, NEUTRAL_PROFILE)
        ]
        if bad_sp_ids:
            violations.append(f"[StreamProfile] Legacy resource-ish fields still set: IDs {sorted(bad_sp_ids)}")

        # 4) CRO presence for active cameras (warning by default; strict -> violation)
        cro_exists = CameraResourceOverride.objects.filter(
            camera_id=OuterRef("pk"),
            is_active=True,
        )
        cams = Camera.objects.filter(is_active=True).annotate(has_cro=Exists(cro_exists))
        cams_without_cro = list(cams.filter(has_cro=False).values_list("id", flat=True))

        if cams_without_cro:
            msg = f"[CRO] Active cameras without an active CRO: {sorted(cams_without_cro)}"
            (violations if strict_cro else warnings).append(
                msg + (" (strict)" if strict_cro else " (warn)")
            )

        # Emit results
        for w in warnings:
            self.stdout.write(self.style.WARNING(f"WARNING: {w}"))
        if violations:
            for v in violations:
                self.stdout.write(self.style.ERROR(f"VIOLATION: {v}"))
            sys.exit(1)

        self.stdout.write(self.style.SUCCESS("OK: policy/resources posture is clean."))
