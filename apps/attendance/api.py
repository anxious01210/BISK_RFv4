# apps/attendance/api.py
from rest_framework import viewsets, permissions, filters
from django_filters.rest_framework import DjangoFilterBackend, FilterSet, DateFilter, NumberFilter, CharFilter
from .models import AttendanceRecord, Student, FaceEmbedding
from .serializers import AttendanceRecordSerializer

from rest_framework.views import APIView
from rest_framework.response import Response
from django.utils.dateparse import parse_datetime
from django.conf import settings
from apps.cameras.models import Camera  # adjust if your path differs
from .services import ingest_match

import base64
from rest_framework import status


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


def _auth_ok(request) -> bool:
    key_required = getattr(settings, "RUNNER_HEARTBEAT_KEY", "")
    return (not key_required) or (request.headers.get("X-BISK-KEY") == key_required)


class EnrollView(APIView):
    """
    POST /api/attendance/enroll/
    Body JSON:
      { "h_code": "H123456",
        "dim": 512,
        "vec": "<base64 float32 bytes>",  # len must be dim*4
        "camera_id": 1,                   # optional
        "source_path": "attendance_crops/..."  # optional
      }
    """
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        if not _auth_ok(request):
            return Response({"detail": "bad key"}, status=403)

        data = request.data or {}
        h_code = (data.get("h_code") or "").strip()
        dim = int(data.get("dim") or 512)
        vec_b64 = data.get("vec") or ""

        if not h_code or not vec_b64:
            return Response({"detail": "h_code and vec are required"}, status=400)

        try:
            raw = base64.b64decode(vec_b64)
        except Exception:
            return Response({"detail": "vec must be base64"}, status=400)

        if len(raw) != dim * 4:
            return Response({"detail": f"vector length mismatch for dim={dim}"}, status=400)

        student = Student.objects.filter(h_code=h_code, is_active=True).first()
        if not student:
            return Response({"detail": "unknown student"}, status=404)

        cam = None
        cam_id = data.get("camera_id")
        if cam_id is not None:
            cam = Camera.objects.filter(id=cam_id).first()

        emb = FaceEmbedding.objects.create(
            student=student,
            dim=dim,
            vector=raw,
            camera=cam,
            source_path=data.get("source_path") or "",
            is_active=True,
        )
        return Response({"ok": True, "id": emb.id}, status=201)


class GalleryView(APIView):
    """
    GET /api/attendance/gallery/?dim=512&active=1
    Returns:
      { "dim": 512, "count": N,
        "embeddings": [
          {"id": 1, "h_code": "H123456", "vec": "<base64 float32 bytes>"}, ...
        ]
      }
    """
    authentication_classes = []
    permission_classes = []

    def get(self, request):
        if not _auth_ok(request):
            return Response({"detail": "bad key"}, status=403)

        try:
            dim = int(request.GET.get("dim", 512))
        except ValueError:
            dim = 512
        active_only = request.GET.get("active", "1") not in ("0", "false", "False")

        qs = FaceEmbedding.objects.filter(dim=dim)
        if active_only:
            qs = qs.filter(is_active=True)

        qs = qs.select_related("student").order_by("student__h_code", "id")

        items = []
        for e in qs:
            items.append({
                "id": e.id,
                "h_code": e.student.h_code,
                "vec": base64.b64encode(e.vector).decode("ascii"),
            })

        return Response({"dim": dim, "count": len(items), "embeddings": items})
