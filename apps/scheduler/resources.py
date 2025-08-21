from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Union

from .models import GlobalResourceSettings, CameraResourceOverride


@dataclass
class EffectiveResources:
    cpu_nice: Optional[int]
    cpu_affinity: Optional[Sequence[int]]
    cpu_quota_percent: Optional[int]
    gpu_visible_devices: Optional[str]  # e.g. "0" or "0,1"
    gpu_memory_fraction: Optional[float]
    gpu_target_util_percent: Optional[int]
    max_fps: Optional[int]
    det_set_max: Optional[str]


def _parse_affinity(s: Optional[str]):
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    vals = []
    for p in parts:
        if p:
            vals.append(int(p))
    return vals or None


def resolve_effective(*, camera=None, camera_id: Optional[int] = None) -> EffectiveResources:
    """
    Merge GlobalResourceSettings with optional per-camera overrides.
    Priority: Camera override (if exists) → Global defaults.
    """
    from apps.cameras.models import Camera as Cam  # lazy import to avoid cycles

    if camera is None and camera_id is not None:
        try:
            camera = Cam.objects.get(pk=camera_id)
        except Cam.DoesNotExist:
            camera = None

    gs = GlobalResourceSettings.get_solo()
    eff = EffectiveResources(
        cpu_nice=gs.cpu_nice,
        cpu_affinity=_parse_affinity(gs.cpu_affinity),
        cpu_quota_percent=gs.cpu_quota_percent,
        gpu_visible_devices=(gs.gpu_index or None),
        gpu_memory_fraction=gs.gpu_memory_fraction,
        gpu_target_util_percent=gs.gpu_target_util_percent,
        max_fps=gs.max_fps_default,
        det_set_max=gs.det_set_max,
    )

    if camera and hasattr(camera, "resource_override"):
        co: CameraResourceOverride = camera.resource_override
        if co.cpu_nice is not None: eff.cpu_nice = co.cpu_nice
        if co.cpu_affinity: eff.cpu_affinity = _parse_affinity(co.cpu_affinity)
        if co.cpu_quota_percent is not None: eff.cpu_quota_percent = co.cpu_quota_percent

        if co.gpu_index: eff.gpu_visible_devices = co.gpu_index
        if co.gpu_memory_fraction is not None: eff.gpu_memory_fraction = co.gpu_memory_fraction
        if co.gpu_target_util_percent is not None: eff.gpu_target_util_percent = co.gpu_target_util_percent

        if co.max_fps is not None: eff.max_fps = co.max_fps
        if co.det_set_max: eff.det_set_max = co.det_set_max

    return eff
