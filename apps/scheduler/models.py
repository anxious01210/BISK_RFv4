# apps/scheduler/models.py
from django.db import models


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

    class Meta:
        indexes = [models.Index(fields=["camera", "profile"])]
