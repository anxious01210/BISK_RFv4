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

    HWACCEL_CHOICES = [("none", "none"), ("nvdec", "nvdec")]
    hwaccel = models.CharField(max_length=8, choices=HWACCEL_CHOICES, default="none",
                               help_text="FFmpeg hardware decode (nvdec => -hwaccel cuda)")

    DEVICE_CHOICES = [("cpu", "CPU"), ("cuda", "CUDA")]
    device = models.CharField(max_length=8, choices=DEVICE_CHOICES, default="cpu",
                              help_text="Intended ML device (for future OpenCV/CUDA path)")
    gpu_index = models.PositiveSmallIntegerField(default=0)

    cpu_affinity = models.CharField(max_length=64, blank=True, help_text='Comma-separated cores, e.g. "0,1"')
    nice = models.SmallIntegerField(default=0, help_text="Process nice (-20..19)")

    hb_interval = models.PositiveSmallIntegerField(default=10)
    snapshot_every = models.PositiveSmallIntegerField(default=3)
    prefer_camera_over_profile = models.BooleanField(
        default=False,
        help_text="If enabled, Camera settings override StreamProfile (camera-first). Otherwise profile-first."
    )


    @property
    def is_paused(self):
        from django.utils import timezone
        pu = self.pause_until
        return bool(pu and timezone.now() < pu)

    def __str__(self): return self.name
