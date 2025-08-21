# apps/attendance/models.py
from django.db import models


class Student(models.Model):
    h_code = models.CharField(max_length=32, unique=True)
    full_name = models.CharField(max_length=120)
    is_active = models.BooleanField(default=True)

    def __str__(self): return f"{self.h_code} - {self.full_name}"


class PeriodTemplate(models.Model):
    name = models.CharField(max_length=64)
    order = models.PositiveSmallIntegerField()
    start_time = models.TimeField()
    end_time = models.TimeField()
    weekdays_mask = models.PositiveSmallIntegerField(default=0)
    is_enabled = models.BooleanField(default=True)
    early_grace_minutes = models.PositiveSmallIntegerField(default=0)
    late_grace_minutes = models.PositiveSmallIntegerField(default=0)

    def __str__(self): return f"{self.name} ({self.start_time}-{self.end_time})"

    def is_active_on(self, dow: int) -> bool: return bool(self.weekdays_mask & (1 << dow))


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

