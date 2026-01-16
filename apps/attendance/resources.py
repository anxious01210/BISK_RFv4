# apps/attendance/resources.py
from import_export import resources, fields
from import_export.widgets import ForeignKeyWidget

from .models import AttendanceRecord, Student, PeriodOccurrence, LunchSubscription
from apps.cameras.models import Camera


class AttendanceRecordResource(resources.ModelResource):
    # Student info
    student_h_code = fields.Field(attribute="student__h_code", column_name="H-code")
    student_full_name = fields.Field(column_name="Full name")
    student_first_name = fields.Field(attribute="student__first_name", column_name="First name")
    student_middle_name = fields.Field(attribute="student__middle_name", column_name="Middle name")
    student_last_name = fields.Field(attribute="student__last_name", column_name="Last name")

    # Optional: if you have grade/section fields, add them here (example):
    # student_grade = fields.Field(attribute="student__grade", column_name="Grade")

    # Attendance info
    period_template = fields.Field(attribute="period__template__name", column_name="Period template")
    best_camera_name = fields.Field(attribute="best_camera__name", column_name="Best camera")
    best_seen = fields.Field(attribute="best_seen", column_name="Best seen")
    best_score = fields.Field(attribute="best_score", column_name="Best score")
    confirmed = fields.Field(attribute="confirmed", column_name="Confirmed")

    # Lunch snapshot fields (the important part)
    lunch_eligible_at_time = fields.Field(attribute="lunch_eligible_at_time", column_name="Lunch eligible at time")
    lunch_reason_code = fields.Field(attribute="lunch_reason_code", column_name="Lunch reason code")
    lunch_reason_notes = fields.Field(attribute="lunch_reason_notes", column_name="Lunch reason notes")
    lunch_subscription_id = fields.Field(attribute="lunch_subscription_id", column_name="Lunch subscription ID")

    def dehydrate_student_full_name(self, obj):
        # Student.full_name is a METHOD, so we call it
        return obj.student.full_name()

    class Meta:
        model = AttendanceRecord
        # Keep this list tight and human-friendly
        export_order = (
            "student_h_code",
            "student_full_name",
            "student_first_name",
            "student_middle_name",
            "student_last_name",
            "period_template",
            "best_camera_name",
            "best_seen",
            "best_score",
            "lunch_eligible_at_time",
            "confirmed",
            "lunch_reason_code",
            "lunch_reason_notes",
            "lunch_subscription_id",
        )
        fields = export_order
