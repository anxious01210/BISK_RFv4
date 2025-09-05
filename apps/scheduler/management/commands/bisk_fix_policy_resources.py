# apps/scheduler/management/commands/bisk_fix_policy_resources.py
from django.core.management.base import BaseCommand
from django.db import transaction

from apps.cameras.models import Camera
from apps.scheduler.models import (
    StreamProfile,
    CameraResourceOverride,
    GlobalResourceSettings,
)

LEGACY_CAMERA_FIELDS = ["device", "hwaccel", "gpu_index", "cpu_affinity", "nice"]
LEGACY_PROFILE_FIELDS = ["device", "hwaccel", "gpu_index", "cpu_affinity", "nice"]


def _has_field(model, name):  # tolerate staged schemas
    return any(f.name == name for f in model._meta.get_fields())


def _get(model, obj, name):
    return getattr(obj, name) if _has_field(model, name) else None


def _is_set(v):
    return v not in (None, "", [])


class Command(BaseCommand):
    help = "Backfill Phase-4 posture: move legacy resources → CRO/GRS, then clear on Camera/StreamProfile."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Show changes without writing.")
        parser.add_argument("--only", choices=["camera", "profile"], help="Limit to one side.")

    def handle(self, *args, **opts):
        dry = opts.get("dry_run", False)
        only = opts.get("only")

        wrote = False

        # ---- Cameras: legacy → CRO (per camera), then clear on Camera
        if only in (None, "camera"):
            for cam in Camera.objects.all():
                # collect legacy values present on Camera
                legacy = {}
                for f in LEGACY_CAMERA_FIELDS:
                    if not _has_field(Camera, f):
                        continue
                    v = getattr(cam, f, None)
                    if _is_set(v):
                        legacy[f] = v

                if not legacy:
                    continue

                # ensure CRO row
                cro, created = CameraResourceOverride.objects.get_or_create(camera=cam, defaults={"is_active": True})

                # write legacy values into CRO if not set there
                changed_cro = {}
                for k, v in legacy.items():
                    if _has_field(CameraResourceOverride, "cpu_nice") and k == "nice":
                        # field name differs between Camera (nice) and CRO (cpu_nice)
                        target = "cpu_nice"
                    else:
                        target = k
                    if not _has_field(CameraResourceOverride, target):
                        continue
                    existing = getattr(cro, target, None)
                    if not _is_set(existing):
                        setattr(cro, target, v)
                        changed_cro[target] = v

                # clear on Camera -> set to model defaults (NOT NULL)
                DEFAULTS_CAMERA = {
                    "device": "cpu",
                    "hwaccel": "none",
                    "gpu_index": 0,
                    "cpu_affinity": "",
                    "nice": 0,
                }

                cleared_cam = []
                for k in legacy.keys():
                    if not _has_field(Camera, k):
                        continue
                    setattr(cam, k, DEFAULTS_CAMERA.get(k, None))
                    cleared_cam.append(k)

                if dry:
                    self.stdout.write(f"[DRY] Camera #{cam.id} → CRO ({'new' if created else 'exists'}) "
                                      f"set {changed_cro or '{}'}; cleared Camera {cleared_cam}.")
                else:
                    with transaction.atomic():
                        if changed_cro:
                            cro.is_active = True
                            cro.save()
                        cam.save(update_fields=[f for f in cleared_cam if f != "nice"] + (["nice"] if "nice" in cleared_cam else []))
                    wrote = True
                    self.stdout.write(self.style.SUCCESS(
                        f"[OK] Camera #{cam.id}: moved {list(legacy.keys())} → CRO; cleared on Camera."
                    ))

        # ---- Profiles: legacy → GRS (singleton) if empty, else clear on Profile
        if only in (None, "profile"):
            # get/create singleton GRS; prefer is_active=True if the field exists
            if _has_field(GlobalResourceSettings, "is_active"):
                grs = GlobalResourceSettings.objects.filter(is_active=True).first()
                if not grs:
                    grs = GlobalResourceSettings.objects.create(is_active=True)
            else:
                grs = GlobalResourceSettings.objects.first() or GlobalResourceSettings.objects.create()

            for sp in StreamProfile.objects.all():
                legacy = {}
                for f in LEGACY_PROFILE_FIELDS:
                    if not _has_field(StreamProfile, f):
                        continue
                    v = getattr(sp, f, None)
                    if _is_set(v):
                        legacy[f] = v

                if not legacy:
                    continue

                promoted_to_grs = {}
                cleared_sp = []
                # promote each field to GRS only if GRS doesn't already specify it
                for k, v in legacy.items():
                    target = k
                    if k == "nice" and _has_field(GlobalResourceSettings, "cpu_nice"):
                        target = "cpu_nice"
                    if not _has_field(GlobalResourceSettings, target):
                        continue
                    if not _is_set(getattr(grs, target, None)):
                        setattr(grs, target, v)
                        promoted_to_grs[target] = v
                    # clear on profile -> set to model defaults / inheritance-safe values
                    DEFAULTS_SP = {
                        "device": "",  # CharField, blank=True, default=""
                        "hwaccel": "",  # CharField, blank=True, default=""
                        "gpu_index": None,  # null=True
                        "cpu_affinity": "",  # CharField, blank=True, default=""
                        "nice": None,  # null=True
                    }
                    setattr(sp, k, DEFAULTS_SP.get(k, None))

                if dry:
                    self.stdout.write(f"[DRY] StreamProfile #{sp.id}: promote→GRS {promoted_to_grs or '{}'}; "
                                      f"clear on profile {list(legacy.keys())}.")
                else:
                    with transaction.atomic():
                        if promoted_to_grs:
                            grs.save()
                        sp.save(update_fields=[f for f in legacy.keys() if hasattr(sp, f)])
                    wrote = True
                    self.stdout.write(self.style.SUCCESS(
                        f"[OK] StreamProfile #{sp.id}: promoted {list(promoted_to_grs.keys()) or '[]'} to GRS; "
                        f"cleared legacy on profile."
                    ))

        if not wrote and not dry:
            self.stdout.write("Nothing to change.")
