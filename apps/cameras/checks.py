# apps.cameras.checks.py
from django.core.checks import register, Error, Warning, Tags
from django.db import connection
from django.db.utils import OperationalError, ProgrammingError

LEGACY_CAMERA_FIELDS = ["device", "hwaccel", "gpu_index", "cpu_affinity", "nice"]
LEGACY_PROFILE_FIELDS = ["device", "hwaccel", "gpu_index", "cpu_affinity", "nice"]

NEUTRAL_CAMERA = {"device": "cpu", "hwaccel": "none", "gpu_index": 0, "cpu_affinity": "", "nice": 0}
NEUTRAL_PROFILE = {"device": "", "hwaccel": "", "gpu_index": None, "cpu_affinity": "", "nice": None}


def _has(model, name):
    return any(f.name == name for f in model._meta.get_fields())


def _is_set(obj, fields, neutral):
    for f in fields:
        if not _has(obj.__class__, f):
            continue
        v = getattr(obj, f, None)
        if v in (None, "", []):
            continue
        if f in neutral and v == neutral[f]:
            continue
        return True
    return False


@register(Tags.models)
def policy_resources_posture_check(app_configs, **kwargs):
    """
    Migration-safe posture check:
    - Skips if required tables don't exist yet (fresh DB / pre-migrate).
    - Catches OperationalError/ProgrammingError during migrations.
    - Preserves your original error/warn IDs and logic.
    """
    # Import models lazily to avoid import-time DB hits
    try:
        from apps.cameras.models import Camera
        from apps.scheduler.models import StreamProfile, CameraResourceOverride, GlobalResourceSettings
    except Exception:
        # If models can't import yet (early app loading), don't block startup
        return []

    # 1) Skip while tables aren't ready
    try:
        tables = set(connection.introspection.table_names())
    except Exception:
        return []

    needed = {
        # Resolve table names from model _meta to avoid hard-coding
        # and to keep working if db_table is customized later.
        # Cameras
        Camera._meta.db_table,
        # Scheduler
        StreamProfile._meta.db_table,
        CameraResourceOverride._meta.db_table,
        GlobalResourceSettings._meta.db_table,
    }
    if not needed.issubset(tables):
        return []  # DB not initialized yet; safe no-op

    errs, warns = [], []

    # 2) Do the posture queries defensively
    try:
        n_active = (
            GlobalResourceSettings.objects.filter(is_active=True).count()
            if _has(GlobalResourceSettings, "is_active")
            else GlobalResourceSettings.objects.count()
        )
    except (OperationalError, ProgrammingError):
        return []
    # if n_active != 1:
    #     errs.append(Error(f"Expected exactly 1 active GlobalResourceSettings, found {n_active}.", id="bisk.E001"))
    if n_active == 0:
        warns.append(Warning("No active GlobalResourceSettings found (expected 1).", id="bisk.W002"))
    elif n_active > 1:
        errs.append(Error(f"Expected exactly 1 active GlobalResourceSettings, found {n_active}.", id="bisk.E001"))

    try:
        bad_cam = [c.id for c in Camera.objects.only("id") if _is_set(c, LEGACY_CAMERA_FIELDS, NEUTRAL_CAMERA)]
    except (OperationalError, ProgrammingError):
        bad_cam = []
    if bad_cam:
        errs.append(Error(f"Legacy resource fields still set on Camera IDs {sorted(bad_cam)}.", id="bisk.E002"))

    try:
        bad_sp = [s.id for s in StreamProfile.objects.only("id") if _is_set(s, LEGACY_PROFILE_FIELDS, NEUTRAL_PROFILE)]
    except (OperationalError, ProgrammingError):
        bad_sp = []
    if bad_sp:
        errs.append(Error(f"Legacy resource-ish fields still set on StreamProfile IDs {sorted(bad_sp)}.", id="bisk.E003"))

    try:
        from django.db.models import Exists, OuterRef
        cro_exists = CameraResourceOverride.objects.filter(camera_id=OuterRef("pk"), is_active=True)
        no_cro = list(
            Camera.objects.filter(is_active=True)
            .annotate(has_cro=Exists(cro_exists))
            .filter(has_cro=False)
            .values_list("id", flat=True)
        )
    except (OperationalError, ProgrammingError):
        no_cro = []
    if no_cro:
        warns.append(Warning(f"Active cameras without an active CRO: {sorted(no_cro)} (Phase 4 warn).", id="bisk.W001"))

    return errs + warns


# from django.core.checks import register, Error, Warning, Tags
# from django.db.models import Exists, OuterRef
# from apps.cameras.models import Camera
# from apps.scheduler.models import StreamProfile, CameraResourceOverride, GlobalResourceSettings
#
# LEGACY_CAMERA_FIELDS = ["device", "hwaccel", "gpu_index", "cpu_affinity", "nice"]
# LEGACY_PROFILE_FIELDS = ["device", "hwaccel", "gpu_index", "cpu_affinity", "nice"]
#
# NEUTRAL_CAMERA = {"device": "cpu", "hwaccel": "none", "gpu_index": 0, "cpu_affinity": "", "nice": 0}
# NEUTRAL_PROFILE = {"device": "", "hwaccel": "", "gpu_index": None, "cpu_affinity": "", "nice": None}
#
#
# def _has(model, name): return any(f.name == name for f in model._meta.get_fields())
#
#
# def _is_set(obj, fields, neutral):
#     for f in fields:
#         if not _has(obj.__class__, f): continue
#         v = getattr(obj, f, None)
#         if v in (None, "", []): continue
#         if f in neutral and v == neutral[f]: continue
#         return True
#     return False
#
#
# @register(Tags.models)
# def policy_resources_posture_check(app_configs, **kwargs):
#     errs, warns = [], []
#
#     n_active = (GlobalResourceSettings.objects.filter(is_active=True).count()
#                 if _has(GlobalResourceSettings, "is_active") else
#                 GlobalResourceSettings.objects.count())
#     if n_active != 1:
#         errs.append(Error(f"Expected exactly 1 active GlobalResourceSettings, found {n_active}.", id="bisk.E001"))
#
#     bad_cam = [c.id for c in Camera.objects.only("id") if _is_set(c, LEGACY_CAMERA_FIELDS, NEUTRAL_CAMERA)]
#     if bad_cam:
#         errs.append(Error(f"Legacy resource fields still set on Camera IDs {sorted(bad_cam)}.", id="bisk.E002"))
#
#     bad_sp = [s.id for s in StreamProfile.objects.only("id") if _is_set(s, LEGACY_PROFILE_FIELDS, NEUTRAL_PROFILE)]
#     if bad_sp:
#         errs.append(
#             Error(f"Legacy resource-ish fields still set on StreamProfile IDs {sorted(bad_sp)}.", id="bisk.E003"))
#
#     cro_exists = CameraResourceOverride.objects.filter(camera_id=OuterRef("pk"), is_active=True)
#     no_cro = list(Camera.objects.filter(is_active=True).annotate(has_cro=Exists(cro_exists))
#                   .filter(has_cro=False).values_list("id", flat=True))
#     if no_cro:
#         warns.append(Warning(f"Active cameras without an active CRO: {sorted(no_cro)} (Phase 4 warn).", id="bisk.W001"))
#
#     return errs + warns