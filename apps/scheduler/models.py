# apps/scheduler/models.py
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

HWACCEL_CHOICES = [("", "(inherit)"), ("none", "none"), ("nvdec", "nvdec")]
DEVICE_CHOICES = [("", "(inherit)"), ("cpu", "CPU"), ("cuda", "CUDA")]

class StreamProfile(models.Model):
    FFMPEG, OPENCV = 1, 2
    SCRIPT_CHOICES = ((FFMPEG, "FFmpeg"), (OPENCV, "OpenCV"))
    DET_CHOICES = [("auto", "auto"), ("640", "640"), ("800", "800"), ("1024", "1024"), ("1600", "1600"),
                   ("2048", "2048")]

    name = models.CharField(max_length=100, unique=True)
    script_type = models.PositiveSmallIntegerField(choices=SCRIPT_CHOICES, default=FFMPEG)
    fps = models.PositiveIntegerField(default=1)
    detection_set = models.CharField(max_length=16, choices=DET_CHOICES, default="auto")
    extra_args = models.JSONField(default=dict, blank=True)
    RTSP_TRANSPORT_CHOICES = [("", "(inherit)"), ("auto", "auto"), ("tcp", "tcp"), ("udp", "udp")]
    # HWACCEL_CHOICES = [("", "(inherit)"), ("none", "none"), ("nvdec", "nvdec")]
    # DEVICE_CHOICES = [("", "(inherit)"), ("cpu", "CPU"), ("cuda", "CUDA")]

    rtsp_transport = models.CharField(max_length=8, choices=RTSP_TRANSPORT_CHOICES, blank=True, default="")
    hb_interval = models.PositiveSmallIntegerField(null=True, blank=True, help_text="Seconds between heartbeats")
    snapshot_every = models.PositiveSmallIntegerField(null=True, blank=True, help_text="Snapshot every N heartbeats")
    is_active = models.BooleanField(default=True)

    def __str__(
            self): return f"{self.name} [{self.get_script_type_display()} @ {self.fps}fps, det={self.detection_set}]"


class SchedulePolicy(models.Model):
    name = models.CharField(max_length=120, unique=True)
    cameras = models.ManyToManyField("cameras.Camera", related_name="policies", blank=True)
    is_enabled = models.BooleanField(default=True)
    notes = models.CharField(max_length=200, blank=True, default="")

    def __str__(self): return self.name


class ScheduleWindow(models.Model):
    policy = models.ForeignKey(SchedulePolicy, on_delete=models.CASCADE, related_name="windows")
    day_of_week = models.PositiveSmallIntegerField()  # 0=Mon..6=Sun
    start_time = models.TimeField()
    end_time = models.TimeField()
    profile = models.ForeignKey(StreamProfile, on_delete=models.PROTECT)

    class Meta:
        indexes = [models.Index(fields=["policy", "day_of_week"])]


class ScheduleException(models.Model):
    policy = models.ForeignKey(SchedulePolicy, on_delete=models.CASCADE, related_name="exceptions")
    date = models.DateField()
    mode = models.CharField(max_length=10, choices=(("off", "off"), ("on", "on"), ("window", "window")))
    start_time = models.TimeField(null=True, blank=True)
    end_time = models.TimeField(null=True, blank=True)
    profile = models.ForeignKey(StreamProfile, null=True, blank=True, on_delete=models.PROTECT)

    class Meta:
        unique_together = [("policy", "date")]


class RunningProcess(models.Model):
    camera = models.ForeignKey("cameras.Camera", on_delete=models.CASCADE)
    profile = models.ForeignKey(StreamProfile, on_delete=models.PROTECT)
    pid = models.IntegerField()
    started_at = models.DateTimeField(auto_now_add=True)
    last_heartbeat = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=16, default="running")  # running|stopping|dead
    meta = models.JSONField(default=dict, blank=True)
    # Human-readable summary of what we actually ran (derived from StreamProfile).
    effective_opts = models.TextField(blank=True, default="")
    effective_args = models.TextField(blank=True, default="")  # full CLI used
    effective_env = models.JSONField(blank=True, default=dict)  # env snapshot we exported
    nice = models.IntegerField(null=True, blank=True)  # e.g. 10
    cpu_affinity = models.CharField(max_length=100, blank=True, default="")
    last_error = models.CharField(max_length=512, blank=True, default="")  # most recent friendly error from runner
    last_heartbeat_at = models.DateTimeField(null=True, blank=True)  # runner pings update this
    # Live telemetry mirrored from latest HB (optional, for admin tables)
    camera_fps = models.FloatField(null=True, blank=True)
    processed_fps = models.FloatField(null=True, blank=True)
    target_fps = models.FloatField(null=True, blank=True)
    snapshot_every = models.PositiveIntegerField(null=True, blank=True)

    class Meta:
        indexes = [models.Index(fields=["camera", "profile"])]
        constraints = [
            models.UniqueConstraint(fields=["camera", "profile", "pid"], name="uniq_cam_prof_pid")
        ]


class RunnerHeartbeat(models.Model):
    camera = models.ForeignKey("cameras.Camera", on_delete=models.CASCADE)
    profile = models.ForeignKey("scheduler.StreamProfile", on_delete=models.CASCADE)
    pid = models.IntegerField(null=True, blank=True)
    ts = models.DateTimeField(auto_now_add=True)
    fps = models.FloatField(default=0)
    detected = models.PositiveIntegerField(default=0)
    matched = models.PositiveIntegerField(default=0)
    latency_ms = models.FloatField(default=0)
    # Friendly error string extracted from runner/ffmpeg (optional).
    last_error = models.CharField(max_length=255, null=True, blank=True)
    target_fps = models.FloatField(null=True, blank=True)
    snapshot_every = models.PositiveIntegerField(null=True, blank=True)
    processed_fps = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ("-ts",)
        indexes = [
            models.Index(fields=["camera", "profile", "ts"], name="hb_cam_prof_ts"),
        ]

    def __str__(self):
        return f"HB cam={self.camera_id} prof={self.profile_id} @ {self.ts:%Y-%m-%d %H:%M:%S}"


class GlobalResourceSettings(models.Model):
    """
    Singleton defaults for runner processes.
    Leave values null to mean 'no default at this layer'.
    """
    # Placement defaults (nullable → no default at this layer)
    device = models.CharField(
        max_length=8, choices=DEVICE_CHOICES, null=True, blank=True,
        help_text="Default compute device (cuda/cpu). Null=not set here."
    )
    hwaccel = models.CharField(
        max_length=8, choices=HWACCEL_CHOICES, null=True, blank=True,
        help_text="Default decode accel (nvdec/none). Null=not set here."
    )

    # CPU
    cpu_nice = models.IntegerField(null=True, blank=True, help_text="e.g. 10 for background")
    cpu_affinity = models.CharField(
        max_length=64, null=True, blank=True,
        help_text="CSV of CPU core indexes, e.g. '0,1,2'. Empty=all cores."
    )
    cpu_quota_percent = models.PositiveSmallIntegerField(
        null=True, blank=True, validators=[MinValueValidator(1), MaxValueValidator(100)],
        help_text="Soft target used by runner throttling. 100 = no cap."
    )

    # GPU
    gpu_index = models.CharField(
        max_length=32, null=True, blank=True,
        help_text="Single or CSV for multi-GPU; e.g. '0' or '0,1'. Applied to CUDA_VISIBLE_DEVICES."
    )
    gpu_memory_fraction = models.FloatField(
        null=True, blank=True, validators=[MinValueValidator(0.05), MaxValueValidator(1.0)],
        help_text="Torch-only per-process memory fraction (0.05–1.0)."
    )
    gpu_target_util_percent = models.PositiveSmallIntegerField(
        null=True, blank=True, validators=[MinValueValidator(5), MaxValueValidator(100)],
        help_text="Runner attempts to keep GPU util near this via FPS throttling."
    )

    # Pipeline levers
    max_fps_default = models.PositiveSmallIntegerField(
        null=True, blank=True, help_text="Default FPS cap if camera/profile doesn't specify."
    )
    det_set_max = models.CharField(
        max_length=16, null=True, blank=True, help_text="Upper bound like '1600' to clamp detector size."
    )
    is_active = models.BooleanField(
        default=True,
        help_text="If disabled, global defaults/caps are ignored."
    )

    def __str__(self):
        return "Global Resource Settings"

    @classmethod
    def get_solo(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class CameraResourceOverride(models.Model):
    """
    Per-camera overrides. Non-null fields override global defaults.
    """
    camera = models.OneToOneField(
        # That string is the app label + model name. Your app label is cameras, not attendance.
        "cameras.Camera",  # string ref avoids circular import
        on_delete=models.CASCADE,
        related_name="resource_override"
    )


    # CPU
    cpu_nice = models.IntegerField(null=True, blank=True)
    cpu_affinity = models.CharField(max_length=64, null=True, blank=True)
    cpu_quota_percent = models.PositiveSmallIntegerField(null=True, blank=True)

    # GPU
    # Placement overrides (non-null wins)
    device = models.CharField(
        max_length=8, choices=DEVICE_CHOICES, null=True, blank=True,
        help_text="Override compute device for this camera only."
    )
    hwaccel = models.CharField(
        max_length=8, choices=HWACCEL_CHOICES, null=True, blank=True,
        help_text="Override decode accel for this camera only."
    )
    gpu_index = models.CharField(max_length=32, null=True, blank=True)
    gpu_memory_fraction = models.FloatField(null=True, blank=True)
    gpu_target_util_percent = models.PositiveSmallIntegerField(null=True, blank=True)

    # Pipeline levers
    max_fps = models.PositiveSmallIntegerField(null=True, blank=True)
    det_set_max = models.CharField(max_length=16, null=True, blank=True)
    is_active = models.BooleanField(
        default=False,
        help_text="Master toggle for considering any per-camera resource overrides."
    )

    def __str__(self):
        return f"Overrides for {self.camera.name if self.camera_id else 'unknown'}"
