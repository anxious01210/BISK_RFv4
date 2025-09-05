# apps/attendance/models.py
from django.db import models
from django.db.models import Q, UniqueConstraint
from django.utils import timezone


class Student(models.Model):
    # --- existing fields (keep yours, e.g. h_code, is_active, etc.) ---
    h_code = models.CharField(max_length=32, unique=True)
    is_active = models.BooleanField(default=True)

    # NEW: split name fields (use empty strings rather than NULLs)
    first_name = models.CharField(max_length=100, blank=True, default="")
    middle_name = models.CharField(max_length=100, blank=True, default="")
    last_name = models.CharField(max_length=100, blank=True, default="")

    # REMOVE the old full_name = models.CharField(...) field.
    # (Do NOT delete it here yet; we will remove it in the 2nd migration!)
    # For the code edit step: comment it out now so makemigrations detects removal.
    # full_name = models.CharField(max_length=255, blank=True, default="")  # <- to be removed

    # ... whatever else you already had ...

    def full_name(self) -> str:
        """
        Return a display-friendly full name assembled from parts.
        Django templates can call {{ student.full_name }} (method call allowed).
        """
        parts = [self.first_name.strip(), self.middle_name.strip(), self.last_name.strip()]
        return " ".join(p for p in parts if p).strip() or self.h_code

    full_name.short_description = "Full name"

    def __str__(self):
        # keep your existing logic if different
        return f"{self.h_code} — {self.full_name()}"


class PeriodTemplate(models.Model):
    name = models.CharField(
        max_length=64,
        help_text="Display name for this period (e.g., 'First Block')."
    )
    order = models.PositiveSmallIntegerField(
        help_text="Sort order within the day. Lower numbers appear first."
    )
    start_time = models.TimeField(
        help_text="Local start time of the period (HH:MM)."
    )
    end_time = models.TimeField(
        help_text="Local end time of the period (HH:MM)."
    )
    weekdays_mask = models.PositiveSmallIntegerField(
        default=31,  # Mon–Fri by default
        help_text=(
            "Bitmask of active weekdays. Add the values for the days you want:\n"
            "Mon=1, Tue=2, Wed=4, Thu=8, Fri=16, Sat=32, Sun=64. ALL=127.\n"
            "Examples:\n"
            "• Mon–Fri: 31  (1+2+4+8+16)\n"
            "• Sat+Sun (western weekend): 96  (32+64)\n"
            "• Fri+Sat (Gulf weekend): 48  (16+32)\n"
            "• Sun–Thu (Fri & Sat OFF): 79  (64+1+2+4+8)\n"
            "• Only Sun: 64  • Only Fri: 16\n"
            "At runtime we test: mask & (1 << date.weekday())."
        )
    )
    is_enabled = models.BooleanField(
        default=True,
        help_text="If off, this template is ignored when generating daily occurrences."
    )
    early_grace_minutes = models.PositiveSmallIntegerField(
        default=0,
        help_text="Extend the start earlier by this many minutes when generating the daily window."
    )
    late_grace_minutes = models.PositiveSmallIntegerField(
        default=0,
        help_text="Extend the end later by this many minutes when generating the daily window."
    )

    def __str__(self):
        return f"{self.name} ({self.start_time}-{self.end_time})"

    def is_active_on(self, dow: int) -> bool:
        """Return True if this template includes the weekday (Mon=0 … Sun=6)."""
        return bool(self.weekdays_mask & (1 << dow))


class PeriodOccurrence(models.Model):
    template = models.ForeignKey(PeriodTemplate, on_delete=models.PROTECT, related_name="occurrences")
    date = models.DateField()
    start_dt = models.DateTimeField()  # aware
    end_dt = models.DateTimeField()  # aware
    is_school_day = models.BooleanField(default=True)

    class Meta:
        unique_together = [("template", "date")]
        indexes = [models.Index(fields=["date", "start_dt", "end_dt"])]

    def __str__(self):
        try:
            s = self.start_dt.astimezone().strftime("%H:%M")
            e = self.end_dt.astimezone().strftime("%H:%M")
            return f"{self.template.name} ({s} - {e})"
        except Exception:
            return f"{self.template.name} on {self.date}"


class AttendanceRecord(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    period = models.ForeignKey(PeriodOccurrence, on_delete=models.CASCADE)
    first_seen = models.DateTimeField()
    last_seen = models.DateTimeField()
    best_seen = models.DateTimeField()
    best_score = models.FloatField()
    best_camera = models.ForeignKey("cameras.Camera", null=True, blank=True, on_delete=models.SET_NULL)
    best_crop = models.CharField(max_length=300, blank=True, default="")
    sightings = models.PositiveIntegerField(default=1)
    status = models.CharField(max_length=16, default="present")

    class Meta:
        unique_together = [("student", "period")]
        indexes = [models.Index(fields=["student", "period"])]


class AttendanceEvent(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    period = models.ForeignKey(PeriodOccurrence, on_delete=models.SET_NULL, null=True, blank=True)
    camera = models.ForeignKey("cameras.Camera", on_delete=models.SET_NULL, null=True, blank=True)
    ts = models.DateTimeField(help_text="The timestamp of the event where it was taken.")
    score = models.FloatField()
    crop_path = models.CharField(max_length=300, blank=True, default="")


class RecognitionSettings(models.Model):
    min_score = models.FloatField(default=0.75)
    re_register_window_sec = models.PositiveIntegerField(default=10)
    min_improve_delta = models.FloatField(default=0.01)
    delete_old_cropped = models.BooleanField(default=False)
    save_all_crops = models.BooleanField(default=False)
    use_cosine_similarity = models.BooleanField(default=True)
    max_periods_per_day = models.PositiveSmallIntegerField(
        null=True, blank=True,
        help_text="If set, cap how many periods can count as 'present' per student per day."
    )
    min_face_px = models.PositiveIntegerField(
        default=40,
        help_text="Drop faces smaller than this many pixels on the shortest bbox side. (ex. like 36px to have maximum up to 5m distance for recognitions.)"
    )
    changed_at = models.DateTimeField(auto_now=True, help_text="Bumps on every save to notify runners.")

    @classmethod
    def get_solo(cls):
        # Simple singleton: id=1
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class FaceEmbedding(models.Model):
    """
    One embedding (float32 vector) for a student.
    Stored as raw bytes to avoid forcing numpy on the server.
    """
    student = models.ForeignKey(
        "attendance.Student",
        on_delete=models.CASCADE,
        related_name="embeddings",
        help_text="The student this embedding belongs to."
    )
    dim = models.PositiveSmallIntegerField(
        default=512,
        help_text="Embedding dimensionality. We use 512 for ArcFace."
    )
    vector = models.BinaryField(
        help_text="L2‑normalized embedding stored as raw float32 bytes (length = dim*4)."
    )
    source_path = models.CharField(
        max_length=512, blank=True,
        help_text="MEDIA‑relative path for reference (e.g., saved .npy for this vector)."
    )
    camera = models.ForeignKey(
        "cameras.Camera", null=True, blank=True, on_delete=models.SET_NULL,
        help_text="Camera associated with this embedding, if known (optional)."
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Exactly one active embedding per student (enforced by DB constraint)."
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Row creation timestamp."
    )
    # --- Enrollment / quality metadata ---
    last_enrolled_at = models.DateTimeField(
        null=True, blank=True,
        help_text="When this embedding was (re)computed."
    )
    last_used_k = models.PositiveIntegerField(
        default=0,
        help_text="How many images were averaged to build this vector."
    )
    last_used_det_size = models.PositiveIntegerField(
        default=640,
        help_text="Face detector input size (square). Higher can help small faces but costs GPU."
    )
    # NEW: persist the exact score threshold used during the last (re)enrollment
    last_used_min_score = models.FloatField(
        null=True, blank=True,
        help_text="Minimum quality score threshold (0..1) used for the last enrollment."
    )
    images_considered = models.PositiveIntegerField(
        default=0,
        help_text="Total gallery images discovered for the student at enrollment time."
    )
    images_used = models.PositiveIntegerField(
        default=0,
        help_text="Images actually used after filtering/scoring."
    )

    avg_sharpness = models.FloatField(
        default=0.0,
        help_text="Mean Laplacian variance across used images (higher ≈ sharper)."
    )
    avg_brightness = models.FloatField(
        default=0.0,
        help_text="Mean grayscale intensity 0–255 across used images."
    )
    avg_used_score = models.FloatField(
        default=0.0,
        help_text="Mean of the per-image ROI quality scores (0..1) for the images used to build the .npy."
    )


    used_images = models.JSONField(
        default=list, blank=True,
        help_text="Filenames of images included in the K‑image average."
    )
    used_images_detail = models.JSONField(
        default=list, blank=True,
        help_text="List of objects for the K images actually used: "
                  "[{'name': 'H123456_1.jpg', 'score': 0.93, 'sharp': 123.4, 'bright': 108.7}, ...]. "
                  "Score is the combined ranking value used to pick top‑K."
    )

    embedding_norm = models.FloatField(
        default=0.0,
        help_text="L2 norm of the stored vector (should be ~1.0)."
    )
    embedding_sha256 = models.CharField(
        max_length=64, blank=True, default="",
        help_text="SHA‑256 hash of the raw float32 bytes (useful for dedup/versioning)."
    )

    arcface_model = models.CharField(
        max_length=64, blank=True, default="buffalo_l",
        help_text="InsightFace/ArcFace model name used to generate this embedding."
    )
    provider = models.CharField(
        max_length=64, blank=True, default="CUDAExecutionProvider",
        help_text="ONNX Runtime execution provider actually used (CUDA or CPU)."
    )

    # Optional extras (very useful in practice)
    enroll_runtime_ms = models.PositiveIntegerField(
        default=0,
        help_text="End‑to‑end enrollment runtime in milliseconds."
    )
    enroll_notes = models.TextField(
        blank=True, default="",
        help_text="Concise human‑readable one‑liner: k, det, counts, provider, norm, sample files."
    )

    class Meta:
        indexes = [
            models.Index(fields=["student"]),
            models.Index(fields=["is_active", "dim"]),
            models.Index(fields=["created_at"]),
            models.Index(fields=["embedding_sha256"]),
            models.Index(fields=["last_enrolled_at"]),
        ]
        constraints = [
            UniqueConstraint(
                fields=("student",),
                condition=Q(is_active=True),
                name="uniq_active_embedding_per_student",
            ),
        ]

    def __str__(self):
        return f"{self.student.h_code} • dim={self.dim} • {'on' if self.is_active else 'off'}"
