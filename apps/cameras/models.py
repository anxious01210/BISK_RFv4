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

    def __str__(self): return self.name
