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
    min_face_px = models.PositiveIntegerField(null=True, blank=True)

    class Meta:
        ordering = ("-ts",)
        indexes = [
            models.Index(fields=["camera", "profile", "ts"], name="hb_cam_prof_ts"),
        ]

    def __str__(self):
        return f"HB cam={self.camera_id} prof={self.profile_id} @ {self.ts:%Y-%m-%d %H:%M:%S}"


MODEL_CHOICES = [
    ("buffalo_l", "buffalo_l"),
    ("antelopev2", "antelopev2"),
]

CROP_FMT_CHOICES = [
    ("jpg", "JPEG"),
    ("png", "PNG"),
]


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
        null=True, blank=True,
        help_text="Default FPS cap if camera/profile doesn't specify."
    )
    det_set_max = models.CharField(
        max_length=16, null=True, blank=True,
        help_text="Upper bound like '1600' to clamp detector size."
    )
    # NEW: model/quality defaults (nullable = inherit/unspecified)
    model_tag = models.CharField(
        max_length=32, choices=MODEL_CHOICES, blank=True, default="",
        help_text="Default face model to use (e.g., 'buffalo_l' for speed or 'antelopev2' for stronger quality). "
                  "Blank = runner default."
    )
    pipe_mjpeg_q = models.PositiveSmallIntegerField(
        null=True, blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(31)],
        help_text="FFmpeg MJPEG quality for the live frame pipe (1=best, 31=worst). "
                  "Lower gives sharper frames but higher CPU/bitrate. Typical: 2."
    )
    crop_format = models.CharField(
        max_length=8, choices=CROP_FMT_CHOICES, blank=True, default="",
        help_text="Default image format for saved face crops. Leave blank to inherit runner default."
    )
    crop_jpeg_quality = models.PositiveSmallIntegerField(
        null=True, blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(100)],
        help_text="JPEG quality (1–100) for saved crops when format=JPEG. Higher = larger file. Ignored for PNG."
    )
    min_face_px_default = models.PositiveSmallIntegerField(
        null=True, blank=True,
        help_text="Global minimum face-box size in pixels required to process a detection. "
                  "Raise to reduce false matches from tiny/far faces; lower to catch smaller faces."
    )
    quality_version = models.PositiveSmallIntegerField(
        null=True, blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(9)],
        help_text="Optional tuning profile version for end-to-end heuristics (e.g., 1/2/3). "
                  "Runner may use this to switch thresholds/filters. Null = runner default."
    )
    save_debug_unmatched = models.BooleanField(
        null=True, blank=True,
        help_text=(
            "If True, save below-threshold (unmatched) face crops for diagnostics; "
            "if False, never save; if blank, cameras inherit or fallback to script default."
        ),
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
    cpu_nice = models.IntegerField(
        null=True, blank=True,
        help_text="Process niceness (lower = higher priority). Example: 10 for background."
    )
    cpu_affinity = models.CharField(
        max_length=64, null=True, blank=True,
        help_text="CSV of CPU core indexes (e.g., '0,1,2'). Empty = no pinning."
    )
    cpu_quota_percent = models.PositiveSmallIntegerField(
        null=True, blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(100)],
        help_text="Per-process CPU throttle target. 100 = no cap."
    )

    # GPU placement
    device = models.CharField(
        max_length=8, choices=DEVICE_CHOICES, null=True, blank=True,
        help_text="Override compute device for this camera (cpu/cuda)."
    )
    hwaccel = models.CharField(
        max_length=8, choices=HWACCEL_CHOICES, null=True, blank=True,
        help_text="Override video decode accel for this camera (nvdec/none)."
    )
    gpu_index = models.CharField(
        max_length=32, null=True, blank=True,
        help_text="Single or CSV for multi-GPU; e.g. '0' or '0,1'."
    )
    gpu_memory_fraction = models.FloatField(
        null=True, blank=True,
        validators=[MinValueValidator(0.05), MaxValueValidator(1.0)],
        help_text="Torch-only per-process memory fraction (0.05–1.0)."
    )
    gpu_target_util_percent = models.PositiveSmallIntegerField(
        null=True, blank=True,
        validators=[MinValueValidator(5), MaxValueValidator(100)],
        help_text="Runner attempts to keep GPU util near this via FPS throttling."
    )

    # Pipeline levers
    max_fps = models.PositiveSmallIntegerField(
        null=True, blank=True,
        help_text="Hard cap for this camera’s effective FPS (after downsampling). Blank = inherit."
    )
    det_set_max = models.CharField(
        max_length=16, null=True, blank=True,
        help_text="Upper bound to clamp detector size for this camera (e.g., '1024'). Blank = inherit."
    )
    min_face_px = models.PositiveSmallIntegerField(
        null=True, blank=True,
        help_text="Minimum face-box size (pixels) for this camera. Overrides global/default."
    )
    model_tag = models.CharField(
        max_length=32, choices=MODEL_CHOICES, blank=True, default="",
        help_text="Override ArcFace model just for this camera (e.g., 'antelopev2' for stronger recognition)."
    )
    pipe_mjpeg_q = models.PositiveSmallIntegerField(
        null=True, blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(31)],
        help_text="FFmpeg MJPEG quality for the frame pipe (1=best, 31=worst) for this camera only."
    )
    crop_format = models.CharField(
        max_length=8, choices=CROP_FMT_CHOICES, blank=True, default="",
        help_text="Image format for saved face crops from this camera."
    )
    crop_jpeg_quality = models.PositiveSmallIntegerField(
        null=True, blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(100)],
        help_text="JPEG quality (1–100) for this camera’s saved crops (ignored for PNG)."
    )
    quality_version = models.PositiveSmallIntegerField(
        null=True, blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(9)],
        help_text="Optional camera-specific tuning profile (1/2/3…). Blank = inherit."
    )
    save_debug_unmatched = models.BooleanField(
        null=True, blank=True,
        help_text="If True, save unmatched crops for tuning/debug for this camera only; "
                  "if False, never save; if blank, inherit."
    )

    is_active = models.BooleanField(
        default=False,
        help_text="Master toggle for considering any per-camera resource overrides."
    )

    def __str__(self):
        return f"Overrides for {self.camera.name if self.camera_id else 'unknown'}"

# model_tag lets you opt into a stronger ArcFace model (e.g., antelopev2) where you can afford it, while
# keeping buffalo_l elsewhere. We will plumb this so enforcer passes --model when set; it already supports a
# model_tag knob in policy logic, so we’ll extend the resource path as a fallback too
# pipe_mjpeg_q controls JPEG quality for FFmpeg’s image pipe to Python. Lower = sharper frames
# (better detection/landmarks) at the cost of CPU/bandwidth.
# min_face_px(_default) and per-camera min_face_px gate small faces that tend to be noisy; raising this improves
# precision; lowering helps recall at distance. Runner already emits min_face_px with heartbeats for observability
# crop_format / crop_jpeg_quality control saved-crop fidelity & size so you can tune storage vs inspection quality.
# quality_version gives you a single switch we can use in runner logic (and later in embedding/live scripts) to adjust
# multi-knob heuristics in concert (e.g., bbox expand, det NMS thresholds, top-K filtering) without changing many fields.
