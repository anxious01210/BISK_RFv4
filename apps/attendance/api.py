# apps/attendance/api.py
from .models import AttendanceRecord, Student, FaceEmbedding, PeriodTemplate
from rest_framework import viewsets, permissions
from rest_framework.filters import SearchFilter, OrderingFilter

from django_filters.rest_framework import DjangoFilterBackend, FilterSet
import django_filters as df  # <- alias

from django_filters.widgets import CSVWidget
from django.utils.html import escape, format_html

# from django_filters import rest_framework as filters
from .serializers import AttendanceRecordSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from django.utils.dateparse import parse_datetime
from django.conf import settings
from apps.cameras.models import Camera  # adjust if your path differs
from .services import ingest_match
from django.db.models import Q
import base64
import re
from rest_framework.permissions import BasePermission

# class IsApiUserOrStaff(BasePermission):
#     """
#     Allow only authenticated staff OR members of the 'api_user' group.
#     """
#     def has_permission(self, request, view):
#         u = request.user
#         return bool(u and u.is_authenticated and (u.is_staff or u.groups.filter(name="api_user").exists()))

class IsApiUserOrAdmin(BasePermission):
    """
    Allow only:
      - superusers, OR
      - members of 'supervisor' group (admin-level), OR
      - members of 'api_user' group
    """
    def has_permission(self, request, view):
        u = request.user
        if not (u and u.is_authenticated):
            return False
        if u.is_superuser or u.groups.filter(name="supervisor").exists():
            return True
        return u.groups.filter(name="api_user").exists()

class CharInFilter(df.BaseInFilter, df.CharFilter):
    """Accept comma-separated values (or repeated params) and apply SQL IN."""
    pass


# CSV / repeated param for integers
class NumberInFilter(df.BaseInFilter, df.NumberFilter):
    """Accept comma-separated integers (or repeat params) and apply SQL IN."""
    pass


def parse_iso(dt_str):
    if not dt_str:
        return None
    dt = parse_datetime(dt_str)
    return dt  # your project runs system time at UTC+3; keep consistent


class FilterHelpOnLabel(FilterSet):
    def get_form_class(self):
        Base = super().get_form_class()

        class FormWithInlineHelp(Base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.label_suffix = ""  # we add ":" ourselves
                for f in self.fields.values():
                    ht = getattr(f, "help_text", "")
                    if ht:
                        # label + colon on first line, help text on its own line and indented
                        f.label = format_html(
                            '{}:<span style="display:block; margin:2px 2px 0 8px; '
                            'color:#9aa0a6; font-weight:normal; white-space:normal;">{}</span>',
                            f.label,
                            escape(ht),
                        )
                        f.help_text = ""  # prevent duplicate help below the field

        return FormWithInlineHelp


class AttendanceRecordFilter(FilterHelpOnLabel):
    # class AttendanceRecordFilter(FilterSet):
    date_from = df.DateFilter(
        field_name="period__date",
        lookup_expr="gte",
        label="Date from (inclusive)",
        help_text=(
            "Lower bound on the **period date** (not a timestamp). "
            "Format: YYYY-MM-DD. Example: ?date_from=2025-09-20"
        ),
    )
    date_to = df.DateFilter(
        field_name="period__date",
        lookup_expr="lte",
        label="Date to (inclusive)",
        help_text=(
            "Upper bound on the **period date** (not a timestamp). "
            "Format: YYYY-MM-DD. Example: ?date_to=2025-09-21"
        ),
    )
    min_score = df.NumberFilter(
        field_name="best_score",
        lookup_expr="gte",
        label="Min score (≥)",
        help_text=(
            "Minimum **best_score** to include (0.00–1.00). "
            "Example: ?min_score=0.80"
        ),
    )
    max_score = df.NumberFilter(                                   # NEW
        field_name="best_score",
        lookup_expr="lte",
        label="Max score (≤)",
        help_text=(
            "Maximum **best_score** to include (0.00–1.00). "
            "Example: ?max_score=0.92"
        ),
    )
    score_between = df.CharFilter(                                  # NEW
        method="filter_score_between",
        label="Score between (inclusive)",
        help_text=(
            "Filter by score range **inclusive**. Accepts many formats:\n"
            "  ?score_between=0.70,0.85   (comma)\n"
            "  ?score_between=0.70..0.85 (two dots)\n"
            "  ?score_between=0.70-0.85  (dash)\n"
            "Values must be between 0.00 and 1.00."
        ),
    )

    # def filter_score_between(self, qs, name, value):                # NEW
    #     if not value:
    #         return qs
    #     # split on comma, two+ dots, or dash, and tolerate spaces
    #     parts = re.split(r"\s*(?:,|\.{2,}|-)\s*", str(value).strip())
    #     if len(parts) != 2:
    #         return qs
    #     try:
    #         lo, hi = float(parts[0]), float(parts[1])
    #     except ValueError:
    #         return qs
    #     if lo > hi:
    #         lo, hi = hi, lo
    #     lo = max(0.0, lo)
    #     hi = min(1.0, hi)
    #     return qs.filter(best_score__gte=lo, best_score__lte=hi)
    # --- NEW: pass_count filters ---
    pass_count = df.NumberFilter(
        field_name="pass_count", lookup_expr="exact",
        label="Pass count (=)",
        help_text="Exact value. Example: ?pass_count=2"
    )
    pass_count_min = df.NumberFilter(
        field_name="pass_count", lookup_expr="gte",
        label="Pass count (≥)",
        help_text="Minimum pass count. Example: ?pass_count_min=2"
    )
    pass_count_max = df.NumberFilter(
        field_name="pass_count", lookup_expr="lte",
        label="Pass count (≤)",
        help_text="Maximum pass count. Example: ?pass_count_max=3"
    )
    # --- NEW: confirmed flag ---
    confirmed = df.BooleanFilter(
        field_name="confirmed",
        label="Confirmed?",
        help_text="true/false (also 1/0, yes/no). Example: ?confirmed=true"
    )

    # 1) Single student by h_code (case-insensitive)
    h_code_exact = df.CharFilter(
        field_name="student__h_code",
        lookup_expr="iexact",
        label="Student h code (single)",
        help_text="Exact match (case-insensitive). Example: H123456. Leave blank for all.",
    )

    # 2) Many students by CSV (or repeated query params)
    h_code = CharInFilter(
        field_name="student__h_code",
        lookup_expr="in",
        label="Student h codes (CSV)",
        help_text="Comma-separated (or repeat ?h_code=). Examples: ?h_code=H1,H2 or ?h_code=H1&h_code=H2.",
    )

    # 3) Many students via multi-select (validated against DB)
    students = df.ModelMultipleChoiceFilter(
        field_name="student__h_code",
        to_field_name="h_code",
        queryset=Student.objects.order_by("h_code"),
        label="Students (multi-select)",
        help_text="Use Ctrl/⌘ to select multiple. Leave empty to include all students.",
    )
    camera = df.CharFilter(field_name="best_camera__name", lookup_expr="iexact")

    # --- Period filters (3 styles) ----------------------------------------------
    # 1) Single PeriodTemplate by ID (exact)
    period_template_id = df.NumberFilter(
        field_name="period__template_id",
        lookup_expr="exact",
        label="Period template ID (single)",
        help_text="Exact ID. Example: 3. Leave blank for all.",
    )

    # 2) Many PeriodTemplates by ID (CSV or repeated)  ← canonical param
    period_template_ids = NumberInFilter(
        field_name="period__template_id",
        lookup_expr="in",
        label="Period template IDs (CSV)",
        widget=CSVWidget,  # <-- forces a text box; commas allowed
        help_text=(
            "Comma-separated (or repeat ?period_template_ids=). Examples: "
            "?period_template_ids=3,5,9  or  "
            "?period_template_ids=3&period_template_ids=5."
        ),
    )

    # 3) Many PeriodTemplates via FK (multi-select of IDs)
    period_templates = df.ModelMultipleChoiceFilter(
        field_name="period__template",
        queryset=PeriodTemplate.objects.order_by("name"),
        label="Period templates (IDs)",
        help_text=(
            "Use Ctrl/⌘ to select multiple. Leave empty to include all. "
            "Values are PeriodTemplate IDs; the option text shows the name/time window."
        ),
    )

    # period = CharFilter(field_name="period__template__name", lookup_expr="iexact")

    # # Multi-select by template **IDs** (labels will still show names)
    # period_template = ModelMultipleChoiceFilter(
    #     field_name="period__template",
    #     queryset=PeriodTemplate.objects.order_by("name"),
    #     label="Period templates (IDs)",
    # )

    # # Multi-select by template **names** (values are the names)
    # period_template_name = ModelMultipleChoiceFilter(
    #     field_name="period__template__name",
    #     to_field_name="name",
    #     queryset=PeriodTemplate.objects.order_by("name"),
    #     label="Period templates (names)",
    # )
    def filter_score_between(self, qs, name, value):                # NEW
        if not value:
            return qs
        # split on comma, two+ dots, or dash, and tolerate spaces
        parts = re.split(r"\s*(?:,|\.{2,}|-)\s*", str(value).strip())
        if len(parts) != 2:
            return qs
        try:
            lo, hi = float(parts[0]), float(parts[1])
        except ValueError:
            return qs
        if lo > hi:
            lo, hi = hi, lo
        lo = max(0.0, lo)
        hi = min(1.0, hi)
        return qs.filter(best_score__gte=lo, best_score__lte=hi)

    class Meta:
        model = AttendanceRecord
        fields = []


class AttendanceRecordViewSet(viewsets.ReadOnlyModelViewSet):
    # permission_classes = [permissions.IsAuthenticated]
    # permission_classes = [IsApiUserOrStaff]
    permission_classes = [IsApiUserOrAdmin]
    serializer_class = AttendanceRecordSerializer
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_class = AttendanceRecordFilter
    search_fields = ["student__h_code", "student__full_name"]
    # ordering_fields = ["best_seen", "best_score", "period__date"]
    ordering_fields = ["best_seen", "best_score", "period__date", "pass_count", "confirmed"]
    ordering = ["-best_seen"]

    def get_queryset(self):
        qs = (AttendanceRecord.objects
              .select_related("student", "period__template", "best_camera")
              .all())

        # Multi-period: ?period=a,b,c (case-insensitive)
        period_name_in = self.request.GET.get("period")
        if period_name_in:
            parts = [p.strip() for p in period_name_in.split(",") if p.strip()]
            if parts:
                q = Q()
                for p in parts:
                    q |= Q(period__template__name__iexact=p)
                qs = qs.filter(q)

        # Incremental: ?after_ts=ISO8601  (on best_seen)
        after_ts = parse_iso(self.request.GET.get("after_ts"))
        if after_ts:
            qs = qs.filter(best_seen__gte=after_ts)
        qs = qs.distinct()
        return qs


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
