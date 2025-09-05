# apps/cameras/models.py
from django.db import models


class Camera(models.Model):
    FFMPEG, OPENCV = 1, 2
    SCRIPT_CHOICES = ((FFMPEG, "FFmpeg"), (OPENCV, "OpenCV"))

    name = models.CharField(max_length=80, unique=True)
    rtsp_url = models.CharField(max_length=300)
    location = models.CharField(max_length=120, blank=True, default="")
    scan_station = models.BooleanField(default=False)
    script_type_default = models.PositiveSmallIntegerField(choices=SCRIPT_CHOICES, default=FFMPEG)
    is_active = models.BooleanField(default=True)
    pause_until = models.DateTimeField(null=True, blank=True,
                                       help_text="If set, this camera will not launch until this time (Asia/Baghdad).")
    RTSP_TRANSPORT_CHOICES = [("auto", "auto"), ("tcp", "tcp"), ("udp", "udp")]
    rtsp_transport = models.CharField(max_length=8, choices=RTSP_TRANSPORT_CHOICES, default="auto",
                                      help_text="FFmpeg -rtsp_transport (tcp/udp/auto)")

    hb_interval = models.PositiveSmallIntegerField(default=10)
    snapshot_every = models.PositiveSmallIntegerField(default=3)
    prefer_camera_over_profile = models.BooleanField(
        default=False,
        help_text="If enabled, Camera settings override StreamProfile (camera-first). Otherwise profile-first."
    )
    # --- New: Camera-level POLICY overrides (nullable) ---
    # Treat these as requested values only if prefer_camera_over_profile=True.
    # Leave null to fall back to the StreamProfile value.
    target_fps_req = models.PositiveIntegerField(
        null=True, blank=True,
        help_text="Requested processed FPS for this camera (nullable = use profile)."
    )
    det_set_req = models.CharField(
        max_length=16, null=True, blank=True,
        help_text="Requested detector size, e.g. 'auto','640','800','1024','1600','2048' (nullable = use profile)."
    )
    # (Optional, add later if desired)
    # min_score = models.FloatField(null=True, blank=True)
    # model_tag = models.CharField(max_length=64, null=True, blank=True)

    @property
    def is_paused(self):
        from django.utils import timezone
        pu = self.pause_until
        return bool(pu and timezone.now() < pu)

    def __str__(self): return self.name
