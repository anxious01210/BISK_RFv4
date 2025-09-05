from django.core.checks import register, Error, Warning, Tags
from django.db.models import Exists, OuterRef
from apps.cameras.models import Camera
from apps.scheduler.models import StreamProfile, CameraResourceOverride, GlobalResourceSettings

LEGACY_CAMERA_FIELDS = ["device", "hwaccel", "gpu_index", "cpu_affinity", "nice"]
LEGACY_PROFILE_FIELDS = ["device", "hwaccel", "gpu_index", "cpu_affinity", "nice"]

NEUTRAL_CAMERA = {"device": "cpu", "hwaccel": "none", "gpu_index": 0, "cpu_affinity": "", "nice": 0}
NEUTRAL_PROFILE = {"device": "", "hwaccel": "", "gpu_index": None, "cpu_affinity": "", "nice": None}


def _has(model, name): return any(f.name == name for f in model._meta.get_fields())


def _is_set(obj, fields, neutral):
    for f in fields:
        if not _has(obj.__class__, f): continue
        v = getattr(obj, f, None)
        if v in (None, "", []): continue
        if f in neutral and v == neutral[f]: continue
        return True
    return False


@register(Tags.models)
def policy_resources_posture_check(app_configs, **kwargs):
    errs, warns = [], []

    n_active = (GlobalResourceSettings.objects.filter(is_active=True).count()
                if _has(GlobalResourceSettings, "is_active") else
                GlobalResourceSettings.objects.count())
    if n_active != 1:
        errs.append(Error(f"Expected exactly 1 active GlobalResourceSettings, found {n_active}.", id="bisk.E001"))

    bad_cam = [c.id for c in Camera.objects.only("id") if _is_set(c, LEGACY_CAMERA_FIELDS, NEUTRAL_CAMERA)]
    if bad_cam:
        errs.append(Error(f"Legacy resource fields still set on Camera IDs {sorted(bad_cam)}.", id="bisk.E002"))

    bad_sp = [s.id for s in StreamProfile.objects.only("id") if _is_set(s, LEGACY_PROFILE_FIELDS, NEUTRAL_PROFILE)]
    if bad_sp:
        errs.append(
            Error(f"Legacy resource-ish fields still set on StreamProfile IDs {sorted(bad_sp)}.", id="bisk.E003"))

    cro_exists = CameraResourceOverride.objects.filter(camera_id=OuterRef("pk"), is_active=True)
    no_cro = list(Camera.objects.filter(is_active=True).annotate(has_cro=Exists(cro_exists))
                  .filter(has_cro=False).values_list("id", flat=True))
    if no_cro:
        warns.append(Warning(f"Active cameras without an active CRO: {sorted(no_cro)} (Phase 4 warn).", id="bisk.W001"))

    return errs + warns
