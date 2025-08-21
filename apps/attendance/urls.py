# apps/attendance/urls.py
from django.urls import path
from rest_framework.routers import DefaultRouter
from .api import AttendanceRecordViewSet, IngestView

app_name = "attendance"  # <-- namespacing key

router = DefaultRouter()
# DRF route names under this namespace become:
#   attendance:attendance-records-list
#   attendance:attendance-records-detail
router.register(r"api/attendance/records", AttendanceRecordViewSet, basename="attendance-records")

urlpatterns = [
    # POST /api/attendance/ingest/
    # Reverse name (namespaced): attendance:attendance_ingest
    path("api/attendance/ingest/", IngestView.as_view(), name="attendance_ingest"),
]

urlpatterns += router.urls
