# apps/attendance/serializers.py
from rest_framework import serializers
from .models import AttendanceRecord


class AttendanceRecordSerializer(serializers.ModelSerializer):
    h_code = serializers.CharField(source="student.h_code")
    full_name = serializers.CharField(source="student.full_name")
    period_name = serializers.CharField(source="period.template.name")
    period_date = serializers.DateField(source="period.date")
    period_start = serializers.DateTimeField(source="period.start_dt")
    period_end = serializers.DateTimeField(source="period.end_dt")
    camera = serializers.CharField(source="best_camera.name")
    crop_url = serializers.SerializerMethodField()

    class Meta:
        model = AttendanceRecord
        fields = ("id", "h_code", "full_name", "period_name", "period_date", "period_start", "period_end",
                  "first_seen", "last_seen", "best_seen", "best_score", "camera", "crop_url")

    def get_crop_url(self, obj):
        from django.conf import settings
        return f"{settings.MEDIA_URL}{obj.best_crop}" if obj.best_crop else None
