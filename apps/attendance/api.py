# apps/attendance/api.py
from rest_framework import viewsets, permissions, filters
from django_filters.rest_framework import DjangoFilterBackend, FilterSet, DateFilter, NumberFilter, CharFilter
from .models import AttendanceRecord
from .serializers import AttendanceRecordSerializer


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
