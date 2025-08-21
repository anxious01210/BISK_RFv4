# apps/attendance/api.py
from rest_framework import viewsets, permissions, filters
from django_filters.rest_framework import DjangoFilterBackend, FilterSet, DateFilter, NumberFilter, CharFilter
from .models import AttendanceRecord
from .serializers import AttendanceRecordSerializer

from rest_framework.views import APIView
from rest_framework.response import Response
from django.utils.dateparse import parse_datetime
from django.conf import settings
from apps.cameras.models import Camera  # adjust if your path differs
from .services import ingest_match


class AttendanceRecordFilter(FilterSet):
    date_from = DateFilter(field_name="period__date", lookup_expr="gte")
    date_to = DateFilter(field_name="period__date", lookup_expr="lte")
    min_score = NumberFilter(field_name="best_score", lookup_expr="gte")
    h_code = CharFilter(field_name="student__h_code", lookup_expr="iexact")
    camera = CharFilter(field_name="best_camera__name", lookup_expr="iexact")
    period = CharFilter(field_name="period__template__name", lookup_expr="iexact")

    class Meta:
        model = AttendanceRecord
        fields = []


class AttendanceRecordViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = AttendanceRecordSerializer
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_class = AttendanceRecordFilter
    search_fields = ["student__h_code", "student__full_name"]
    ordering_fields = ["best_seen", "best_score", "period__date"]
    ordering = ["-best_seen"]

    def get_queryset(self):
        return (AttendanceRecord.objects
                .select_related("student", "period__template", "best_camera")
                .all())


class IngestView(APIView):
    """
    POST /api/attendance/ingest/
    Headers:  X-BISK-KEY: <key>   (optional; enforced if settings.RUNNER_HEARTBEAT_KEY is set)
    Body (JSON): { "h_code": "H123456", "score": 0.82, "ts": "2025-08-21T16:40:00Z", "camera_id": 1, "crop_path": "attendance_crops/..." }
    """
    authentication_classes = []  # allow runners without session auth
    permission_classes = []  # we gate with the header key instead

    def post(self, request):
        key_required = getattr(settings, "RUNNER_HEARTBEAT_KEY", "")
        if key_required and request.headers.get("X-BISK-KEY") != key_required:
            return Response({"detail": "bad key"}, status=403)

        payload = request.data or {}
        h_code = (payload.get("h_code") or "").strip()
        score = float(payload.get("score") or 0.0)

        ts_raw = payload.get("ts")
        ts = parse_datetime(ts_raw) if ts_raw else None

        camera = None
        cam_id = payload.get("camera_id")
        if cam_id is not None:
            camera = Camera.objects.filter(id=cam_id).first()

        crop_path = payload.get("crop_path") or ""
        res = ingest_match(h_code=h_code, score=score, ts=ts, camera=camera, crop_path=crop_path)
        return Response(res)
