# apps/attendance/urls.py
from django.urls import path
from rest_framework.routers import DefaultRouter
from .api import AttendanceRecordViewSet, IngestView, EnrollView, GalleryView
from . import views  # <-- add this
# ✨ NEW: import the stream view from your chosen location
from .utils import views_stream as stream_views

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
    # MJPEG stream endpoint (staff-only via the view)
    path("stream/live/<str:session>.mjpg", stream_views.mjpeg_stream, name="att_mjpeg_stream"),
    # synthetic publisher controls
    path("stream/test/start/<str:session>/", stream_views.test_start, name="att_stream_test_start"),
    path("stream/test/stop/<str:session>/", stream_views.test_stop, name="att_stream_test_stop"),
    # HTTP uplink for posting frames
    path("stream/uplink/<str:session>/", stream_views.uplink_frame, name="att_stream_uplink"),
    # spawn/stop local runner
    path("stream/run/start/<str:session>/", stream_views.run_start, name="att_stream_run_start"),
    path("stream/run/stop/<str:session>/", stream_views.run_stop, name="att_stream_run_stop"),

]

urlpatterns += router.urls
