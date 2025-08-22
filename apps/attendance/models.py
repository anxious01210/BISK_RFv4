# apps/attendance/models.py
from django.db import models


class Student(models.Model):
    h_code = models.CharField(max_length=32, unique=True)
    full_name = models.CharField(max_length=120)
    is_active = models.BooleanField(default=True)

    def __str__(self): return f"{self.h_code} - {self.full_name}"


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
    ts = models.DateTimeField()
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

    @classmethod
    def get_solo(cls):
        # Simple singleton: id=1
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj
