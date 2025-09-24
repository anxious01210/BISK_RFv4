# apps/attendance/serializers.py
from rest_framework import serializers
from .models import AttendanceRecord


class AttendanceRecordSerializer(serializers.ModelSerializer):
    h_code = serializers.CharField(source="student.h_code")
    full_name = serializers.SerializerMethodField()
    period_name = serializers.CharField(source="period.template.name")
    period_date = serializers.DateField(source="period.date")
    period_start = serializers.DateTimeField(source="period.start_dt")
    period_end = serializers.DateTimeField(source="period.end_dt")
    # camera = serializers.CharField(source="best_camera.name")
    camera = serializers.SerializerMethodField()
    camera_id = serializers.SerializerMethodField()
    crop_url = serializers.SerializerMethodField()
    student_grade = serializers.SerializerMethodField()
    student_photo_url = serializers.SerializerMethodField()
    pass_count = serializers.IntegerField(read_only=True)
    confirmed = serializers.BooleanField(read_only=True)

    class Meta:
        model = AttendanceRecord
        fields = (
            "id",
            "h_code",
            "full_name",
            "period_name",
            "period_date",
            "period_start",
            "period_end",
            "first_seen",
            "last_seen",
            "best_seen",
            "best_score",
            # "camera",
            "camera",
            "camera_id",
            "crop_url",
            "pass_count",
            "confirmed",
            "student_grade",
            "student_photo_url",
        )

    def get_camera(self, obj):
        cam = getattr(obj, "best_camera", None)
        return getattr(cam, "name", None)

    def get_camera_id(self, obj):
        cam = getattr(obj, "best_camera", None)
        return getattr(cam, "id", None)

    def get_crop_url(self, obj):
        from django.conf import settings
        return f"{settings.MEDIA_URL}{obj.best_crop}" if obj.best_crop else None

    def get_full_name(self, obj):
        return obj.student.full_name()

    def get_student_grade(self, obj):
        return getattr(obj.student, "grade", None)

    def get_student_photo_url(self, obj):
        if hasattr(obj.student, "gallery_photo_relurl"):
            return obj.student.gallery_photo_relurl()
        return None
