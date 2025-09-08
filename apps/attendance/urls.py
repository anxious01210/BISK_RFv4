# apps/attendance/urls.py
from django.urls import path
from rest_framework.routers import DefaultRouter
from .api import AttendanceRecordViewSet, IngestView, EnrollView, GalleryView
from . import views  # <-- add this

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
    path("api/attendance/enroll/", EnrollView.as_view(), name="attendance_enroll"),
    path("api/attendance/gallery/", GalleryView.as_view(), name="attendance_gallery"),
    # NEW — Lunch Supervisor panel
    path("dash/lunch/", views.lunch_page, name="lunch_page"),
    path("dash/lunch/stream/", views.lunch_stream_rows, name="lunch_stream_rows"),
]

urlpatterns += router.urls
