# apps/attendance/models.py
from django.db import models
from django.db.models import Q, UniqueConstraint
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator
from django.conf import settings
from pathlib import Path


class DashboardTag(models.Model):
    """
    Controlled vocabulary for dashboard scoping (e.g., 'lunch', 'bus', 'assembly').
    Use the 'slug' value in code/filters; 'name' is for display.
    """
    LUNCH = "lunch"
    BUS = "bus"
    ASSEMBLY = "assembly"
    SLUG_CHOICES = [
        (LUNCH, "Lunch"),
        (BUS, "Bus"),
        (ASSEMBLY, "Assembly"),
    ]

    name = models.CharField(max_length=64)
    slug = models.SlugField(max_length=32, choices=SLUG_CHOICES, unique=True, db_index=True)

    class Meta:
        verbose_name = "Dashboard tag"
        verbose_name_plural = "Dashboard tags"

    def __str__(self):
        return f"{self.name} ({self.slug})"


class Student(models.Model):
    # GRADE_CHOICES = [
    #     ("KG1", "Kindergarten 1"), ("KG2", "Kindergarten 2"),
    #     ("G1", "Grade 1"), ("G2", "Grade 2"), ("G3", "Grade 3"),
    #     ("G4", "Grade 4"), ("G5", "Grade 5"), ("G6", "Grade 6"),
    #     ("G7", "Grade 7"), ("G8", "Grade 8"), ("G9", "Grade 9"),
    #     ("G10", "Grade 10"), ("G11", "Grade 11"), ("G12", "Grade 12"),
    # ]
    # grade = models.CharField(max_length=32, choices=GRADE_CHOICES, blank=True, null=True, db_index=True)
    h_code = models.CharField(max_length=32, unique=True,
                              help_text="Human/School code (unique student identifier, e.g., H123456).")
    is_active = models.BooleanField(default=True)

    # NEW: split name fields (use empty strings rather than NULLs)
    first_name = models.CharField(max_length=100, blank=True, default="", help_text="Given name.")
    middle_name = models.CharField(max_length=100, blank=True, default="", help_text="Middle name(s), if any.")
    last_name = models.CharField(max_length=100, blank=True, default="", help_text="Family name / surname.")
    GENDER_CHOICES = [
        ("MALE", "male"),
        ("FEMALE", "female")
    ]
    gender = models.CharField(max_length=6, choices=GENDER_CHOICES, blank=True, null=True, db_index=True)

    grade = models.CharField(max_length=32, blank=True, null=True, db_index=True)
    has_lunch = models.BooleanField(default=True, db_index=True)
    has_bus = models.BooleanField(default=False, db_index=True)

    # --- helper: derive face_gallery first image (no ImageField yet) ---
    def gallery_photo_relurl(self) -> str | None:
        """
        Return "/media/face_gallery/<H_CODE>/first_image.jpg" or None if not found.
        """
        if not getattr(self, "h_code", None):
            return None
        base_fs = Path(settings.MEDIA_ROOT) / "face_gallery" / str(self.h_code)
        if not base_fs.exists():
            return None
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = sorted([p for p in base_fs.rglob("*") if p.is_file() and p.suffix.lower() in exts])
        if not files:
            return None
        rel = files[0].relative_to(settings.MEDIA_ROOT).as_posix()
        return f"{settings.MEDIA_URL}{rel}"

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
    # NEW: controlled tags
    usage_tags = models.ManyToManyField(
        "attendance.DashboardTag",
        blank=True,
        related_name="period_templates",
        help_text="Dashboards that should include this period"
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
        return f"{self.name} ({self.start_time}-{self.end_time})-id={self.id}"
        # return f"{self.name} ({self.start_time:%H:%M}-{self.end_time:%H:%M})"

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
    best_camera = models.ForeignKey("cameras.Camera", null=True, blank=True, on_delete=models.SET_NULL)
    best_score = models.FloatField(
        help_text="Highest similarity score observed during this record’s time window."
    )
    best_crop = models.CharField(
        max_length=300, blank=True, default="",
        help_text="MEDIA-relative path of the best face crop for this record."
    )
    sightings = models.PositiveIntegerField(
        default=1,
        help_text="How many frames contributed to this record (debounced within the window)."
    )
    status = models.CharField(
        max_length=16, default="present",
        help_text="Computed attendance status for the period (e.g., present/late/absent)."
    )
    pass_count = models.PositiveIntegerField(default=1)  # new
    last_pass_at = models.DateTimeField(blank=True, null=True, db_index=True)  # new
    confirmed = models.BooleanField(  # NEW
        default=False,
        db_index=True,
        help_text="Manually marked by a lunch supervisor as verified."  # NEW
    )
    lunch_eligible_at_time = models.BooleanField(
        null=True,
        blank=True,
        db_index=True,
        help_text=(
            "Snapshot: whether the student was eligible for lunch on this record's best_seen date. "
            "Saved at record creation time for audit/reporting."
        ),
    )

    # If eligible, store which subscription made them eligible.
    # PROTECT prevents deleting a subscription that was relied on by historical attendance.
    lunch_subscription = models.ForeignKey(
        "attendance.LunchSubscription",
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="attendance_records",
        help_text="Subscription that made this record eligible (snapshot).",
    )

    # Lightweight dropdown (reason code) + optional notes (mainly for supervisor overrides).
    REASON_NONE = ""
    REASON_PAID_NOT_ENTERED = "paid_not_entered"
    REASON_MANAGER_APPROVED = "manager_approved"
    REASON_GUEST = "guest"
    REASON_OTHER = "other"

    LUNCH_REASON_CHOICES = [
        (REASON_NONE, "—"),
        (REASON_PAID_NOT_ENTERED, "Paid but not entered yet"),
        (REASON_MANAGER_APPROVED, "Manager approved / exception"),
        (REASON_GUEST, "Guest / visitor"),
        (REASON_OTHER, "Other"),
    ]

    lunch_reason_code = models.CharField(
        max_length=32,
        blank=True,
        default=REASON_NONE,
        choices=LUNCH_REASON_CHOICES,
        db_index=True,
        help_text="Reason code when lunch confirmation needs explanation (esp. overrides).",
    )

    lunch_reason_notes = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Optional notes to clarify the reason.",
    )

    class Meta:
        unique_together = [("student", "period")]
        indexes = [models.Index(fields=["student", "period"])]

    def save(self, *args, **kwargs):
        """
        Fill lunch snapshot fields once (audit-safe).

        - Auto-compute ONLY when lunch_eligible_at_time is NULL.
        - If a supervisor/admin sets it explicitly, we keep that.
        """
        if self.lunch_eligible_at_time is None:
            try:
                day = timezone.localdate(self.best_seen) if self.best_seen else timezone.localdate()

                sub = LunchSubscription.objects.filter(
                    student_id=self.student_id,
                    status=LunchSubscription.STATUS_ACTIVE,
                    start_date__lte=day,
                    end_date__gte=day,
                ).order_by("start_date", "id").first()

                if sub:
                    self.lunch_eligible_at_time = True
                    self.lunch_subscription = sub
                else:
                    self.lunch_eligible_at_time = False
                    self.lunch_subscription = None

            except Exception:
                # Fail safe: never block attendance writes
                self.lunch_eligible_at_time = False

        super().save(*args, **kwargs)


class AttendanceEvent(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    period = models.ForeignKey(PeriodOccurrence, on_delete=models.SET_NULL, null=True, blank=True)
    camera = models.ForeignKey("cameras.Camera", on_delete=models.SET_NULL, null=True, blank=True)
    ts = models.DateTimeField(
        help_text="Timestamp (timezone-aware) when this event was generated."
    )
    score = models.FloatField(
        help_text="Similarity score for this event (0.00–1.00; higher means more similar to the enrolled embedding)."
    )
    crop_path = models.CharField(
        max_length=300, blank=True, default="",
        help_text="MEDIA-relative path of the face crop saved for this event (if any)."
    )


class RecognitionSettings(models.Model):
    min_score = models.FloatField(
        default=0.75,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=(
            "Minimum similarity score (0.00–1.00) required to accept a match. "
            "Raise this to reduce false accepts; lower it to recognize faces at longer distances "
            "or in poor lighting. Typical range: 0.50–0.85. Default 0.75."
        ),
    )
    re_register_window_sec = models.PositiveIntegerField(
        default=10,
        help_text=(
            "Debounce window in seconds per student. During this window, repeated detections of the "
            "same person on the same camera/period will not create new records; we only update "
            "last_seen/best_seen. Prevents duplicate ‘pops’ when a person stays in view."
        ),
    )
    pass_gap_window_sec = models.PositiveIntegerField(  # NEW
        default=120,
        help_text=(
            "Gap in seconds to count a *new pass* for pass_count. "
            "Used to increment AttendanceRecord.pass_count when the same student reappears "
            "after this gap within the same period. Typical 60–240."
            "We meant to use it for the LUNCH periods for the student waiting in queue, e.x. lunch-pri & lunch-sec"
        ),
    )
    min_improve_delta = models.FloatField(
        default=0.01,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=(
            "Within the debounce window, only update best_seen if the new score improves the previous "
            "best by at least this amount. Example: 0.02 avoids churn from tiny score fluctuations."
        ),
    )
    delete_old_cropped = models.BooleanField(
        default=False,
        help_text=(
            "If ON, old cropped face images will be deleted automatically to save disk space. "
            "Turn OFF if you need to keep historical crops for audit/debug."
        ),
    )
    save_all_crops = models.BooleanField(
        default=False,
        help_text=(
            "If ON, save every detected face crop (matched and unmatched). Useful for tuning but "
            "increases disk usage and I/O. If OFF, only matched/best crops are saved."
        ),
    )
    use_cosine_similarity = models.BooleanField(
        default=True,
        help_text=(
            "Similarity metric preference. Cosine similarity (ON) is the default and recommended. "
            "Note: current runtime uses cosine regardless; this flag is reserved for future switching."
        ),
    )
    max_periods_per_day = models.PositiveSmallIntegerField(
        null=True, blank=True,
        help_text=(
            "Optional cap on how many periods can count as ‘present’ per student per day. "
            "Leave blank for no limit. Example: 1 = only the first recognized period counts."
        ),
    )
    min_face_px = models.PositiveIntegerField(
        default=40,
        help_text=(
            "Reject faces smaller than this many pixels (shortest side of the face bounding box) "
            "before scoring. Use this to control the ‘working distance’: higher = requires closer/bigger "
            "faces (fewer false accepts), lower = allows farther/smaller faces (more recalls). "
            "Guideline for 1440p streams: 36–48 px ≈ ~4–6 m with good optics/bitrate."
        ),
    )
    changed_at = models.DateTimeField(
        auto_now=True,
        help_text=(
            "Auto-updated change marker. Runners poll this field and hot-reload settings within a few "
            "seconds after you save."
        ),
    )

    # --- Face-crop capture & curation policy ---
    crops_enabled = models.BooleanField(
        default=False,
        help_text="Master switch. If off, no per-student capture occurs."
    )
    crops_apply_all_students = models.BooleanField(
        default=False,
        help_text="If ON, capture applies to ALL students. If OFF, only FaceEmbeddings with per-student opt-in."
    )
    crops_save_threshold = models.FloatField(
        default=0.65,
        help_text="Archive when live match score ≥ this value (independent of recognition min-score)."
    )
    crops_keep_n = models.PositiveIntegerField(
        default=5,
        help_text="Keep only the top-N images per (date/period/camera) folder."
    )
    crops_padding = models.PositiveIntegerField(
        default=16,
        help_text="Extra pixels around face before saving (visual pad)."
    )
    crops_margin = models.PositiveIntegerField(
        default=8,
        help_text="Optional black margin/border in saved image."
    )
    crops_format = models.CharField(
        max_length=8, default="png",
        help_text="png or jpg"
    )
    crops_quality = models.PositiveIntegerField(
        default=95,
        help_text="If JPG, 1..100"
    )
    crops_subdir = models.CharField(
        max_length=32, default="captures",
        help_text="Subfolder under each student’s gallery (e.g., 'captures')."
    )
    crops_include_period = models.BooleanField(
        default=True,
        help_text="Insert /<period_id>/ between date and camera in the path."
    )

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
    # Per-student archive opt-in (used when RecognitionSettings.crops_apply_all_students=False)
    crops_opt_in = models.BooleanField(
        default=False,
        help_text="If true and global capture is enabled, archive top crops for this student."
    )
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


class LunchSubscription(models.Model):
    TYPE_ANNUAL = "annual"
    TYPE_MONTHLY = "monthly"
    TYPE_OTHER = "other"

    TYPE_CHOICES = [
        (TYPE_ANNUAL, "Annual"),
        (TYPE_MONTHLY, "Monthly"),
        (TYPE_OTHER, "Other"),
    ]

    STATUS_ACTIVE = "active"
    STATUS_CANCELLED = "cancelled"
    STATUS_EXPIRED = "expired"

    STATUS_CHOICES = [
        (STATUS_ACTIVE, "Active"),
        (STATUS_CANCELLED, "Cancelled"),
        (STATUS_EXPIRED, "Expired"),
    ]

    student = models.ForeignKey(
        "attendance.Student",
        on_delete=models.CASCADE,
        related_name="lunch_subscriptions",
    )
    plan_type = models.CharField(
        max_length=20,
        choices=TYPE_CHOICES,
        default=TYPE_MONTHLY,
        help_text="Annual / monthly / other – mainly for reporting.",
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_ACTIVE,
        db_index=True,
    )
    start_date = models.DateField(db_index=True)
    end_date = models.DateField(db_index=True)

    notes = models.CharField(max_length=200, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["student", "status", "start_date", "end_date"]),
        ]
        ordering = ["student", "start_date"]

    def __str__(self):
        return f"{self.student.h_code} {self.plan_type} [{self.start_date} → {self.end_date}]"

    def is_active_on(self, day):
        return (
                self.status == self.STATUS_ACTIVE
                and self.start_date <= day <= self.end_date
        )
