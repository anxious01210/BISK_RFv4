from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseForbidden
from django.shortcuts import render
from django.template.loader import render_to_string
from django.views.decorators.http import require_GET
from django.db.models import Q
from .models import AttendanceRecord
from django.utils import timezone
from django.utils.dateparse import parse_date, parse_datetime
from datetime import datetime, time


def _parse_date_or_dt(s):
    if not s:
        return None
    dt = parse_datetime(s)
    if dt:
        return timezone.make_aware(dt) if timezone.is_naive(dt) else dt
    d = parse_date(s)
    if d:
        # interpret as start/end of that local day
        tz = timezone.get_current_timezone()
        return tz.localize(datetime.combine(d, time.min))
    return None


# Group gate: lunch_supervisor only
def _is_lunch_supervisor(user) -> bool:
    return user.is_authenticated and user.groups.filter(name="lunch_supervisor").exists()


@login_required
def lunch_page(request):
    if not _is_lunch_supervisor(request.user):
        return HttpResponseForbidden("Requires lunch_supervisor")
    # initial load; rows will stream via HTMX
    return render(request, "attendance/dash/lunch.html")


@require_GET
@login_required
def lunch_stream_rows(request):
    if not _is_lunch_supervisor(request.user):
        return HttpResponseForbidden("Requires lunch_supervisor")

    qs = (AttendanceRecord.objects
          .select_related("student", "period__template", "best_camera")
          .all())

    # period can be comma-separated; default to both lunch windows
    period_param = request.GET.get("period", "lunch-pri,lunch-sec")
    parts = [p.strip() for p in period_param.split(",") if p.strip()]
    if parts:
        q = Q()
        for p in parts:
            q |= Q(period__template__name__iexact=p)
        qs = qs.filter(q)

    # optional date range (by best_seen timestamps)
    df_raw = request.GET.get("date_from")
    dt_raw = request.GET.get("date_to")
    start_dt = _parse_date_or_dt(df_raw)
    end_dt = _parse_date_or_dt(dt_raw)
    if start_dt:
        qs = qs.filter(best_seen__gte=start_dt)
    if end_dt:
        qs = qs.filter(best_seen__lte=end_dt)

    # incremental fetch (best_seen >= after_ts)
    after_ts = request.GET.get("after_ts")
    if after_ts:
        at = parse_datetime(after_ts)
        if at:
            qs = qs.filter(best_seen__gte=at)

    qs = qs.order_by("-best_seen")
    html = render_to_string("attendance/dash/_lunch_rows.html", {"records": qs[:50]})
    return HttpResponse(html)
