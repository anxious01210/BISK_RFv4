from django.db.models import Q
from django.db.models.functions import RowNumber
from django.db.models.expressions import Window
from django.utils import timezone
from django.utils.dateparse import parse_date, parse_datetime
from datetime import datetime, time
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseForbidden
from django.views.decorators.http import require_GET
from django.shortcuts import render
from django.utils.html import escape
from django.conf import settings

from .models import AttendanceRecord

def _parse_any(s):
    if not s:
        return None
    dt = parse_datetime(s)
    if dt:
        return timezone.make_aware(dt) if timezone.is_naive(dt) else dt
    d = parse_date(s)
    if d:
        tz = timezone.get_current_timezone()
        return tz.localize(datetime.combine(d, time.min))
    return None


def _is_lunch_supervisor(user) -> bool:
    return user.is_authenticated and user.groups.filter(name="lunch_supervisor").exists()


@login_required
def lunch_page(request):
    if not _is_lunch_supervisor(request.user):
        return HttpResponseForbidden("Requires lunch_supervisor")
    return render(request, "attendance/dash/lunch.html")


@login_required
@require_GET
def lunch_stream_rows(request):
    """
    Streams <tr> rows for the lunch table based on filters in #params.
    Returns HTML (tbody fragments). HTMX appends them with hx-swap="beforeend".
    """
    if not _is_lunch_supervisor(request.user):
        return HttpResponseForbidden("Requires lunch_supervisor")

    qs = (AttendanceRecord.objects
          .select_related("student", "period__template", "best_camera")
          .all())

    # Periods (comma list; case-insensitive on template name)
    period_param = request.GET.get("period", "lunch-pri,lunch-sec")
    parts = [p.strip() for p in period_param.split(",") if p.strip()]
    if parts:
        q = Q()
        for p in parts:
            q |= Q(period__template__name__iexact=p)
        qs = qs.filter(q)

    # Camera filter (comma list of camera names; case-insensitive per value)
    camera_param = request.GET.get("camera")
    if camera_param:
        cams = [c.strip() for c in camera_param.split(",") if c.strip()]
        if cams:
            cam_q = Q()
            for c in cams:
                cam_q |= Q(best_camera__name__iexact=c)
            qs = qs.filter(cam_q)

    # Eligible only
    if request.GET.get("eligible_only") in ("1", "true", "True", "yes"):
        qs = qs.filter(student__has_lunch=True)

    # Min score (optional)
    try:
        min_score = float(request.GET.get("min_score", "") or "nan")
        if min_score == min_score:  # not NaN
            qs = qs.filter(best_score__gte=min_score)
    except ValueError:
        pass

    # Search q (H-code or name)  <-- ensure '=' not '?'
    qtxt = request.GET.get("q")
    if qtxt:
        qs = qs.filter(
            Q(student__h_code__icontains=qtxt) |
            Q(student__first_name__icontains=qtxt) |
            Q(student__middle_name__icontains=qtxt) |
            Q(student__last_name__icontains=qtxt)
        )

    # date_from / date_to (accept date or datetime ISO)
    start_dt = _parse_any(request.GET.get("date_from"))
    end_dt = _parse_any(request.GET.get("date_to"))
    if start_dt:
        qs = qs.filter(best_seen__gte=start_dt)
    if end_dt:
        qs = qs.filter(best_seen__lte=end_dt)

    # Incremental: after_ts (best_seen >= after_ts)
    at = _parse_any(request.GET.get("after_ts"))
    if at:
        qs = qs.filter(best_seen__gte=at)

    # Mode: all vs latest-per-student-per-period
    mode = request.GET.get("mode", "latest")  # 'latest' or 'all'
    if mode == "latest":
        qs = qs.annotate(
            rn=Window(
                expression=RowNumber(),
                partition_by=["student_id", "period_id"],
                order_by=["-best_seen"],
            )
        ).filter(rn=1)

    # Order ascending by time so incremental appends make sense
    qs = qs.order_by("best_seen")[:200]

    # Build tbody rows
    rows = []
    media_url = settings.MEDIA_URL.rstrip("/") if getattr(settings, "MEDIA_URL", "") else ""
    for rec in qs:
        h = escape(getattr(rec.student, "h_code", ""))
        name = escape(rec.student.full_name() if hasattr(rec.student, "full_name") else "")
        per_name = escape(getattr(rec.period.template, "name", "")) if rec.period and rec.period.template else ""
        score = f"{float(rec.best_score or 0.0):.2f}"
        cam = escape(getattr(rec.best_camera, "name", "") or "-")
        ts = escape(rec.best_seen.astimezone().strftime("%H:%M:%S"))
        pc = int(getattr(rec, "pass_count", 1) or 1)
        badge_class = "badge r" if pc >= 2 else "badge g"
        crop = getattr(rec, "best_crop", "") or ""
        if crop.startswith("http://") or crop.startswith("https://"):
            img_url = crop
        else:
            img_url = f"{media_url}/{crop.lstrip('/')}" if (media_url and crop) else ""

        img_html = f'<img src="{img_url}" class="photo" loading="lazy"/>' if img_url else "—"

        rows.append(
            f"<tr>"
            f"<td>{img_html}</td>"
            f"<td><b>{h}</b><br/><span class='small'>{name}</span></td>"
            f"<td>{per_name}</td>"
            f"<td>{cam}</td>"
            f"<td><span class='{badge_class}'>{pc}</span></td>"
            f"<td>{score}</td>"
            f"<td class='small'>{ts}</td>"
            f"</tr>"
        )

    html = "\n".join(rows)
    if not rows and not at:
        html = "<tr><td colspan='7' class='small' style='opacity:.7;'>No matches for current filters.</td></tr>"

    headers = {}
    if rows:
        try:
            last_ts = qs[len(qs)-1].best_seen.astimezone().isoformat()
            headers["HX-Trigger"] = f'{{"lunch:last_ts": "{last_ts}"}}'
        except Exception:
            pass

    return HttpResponse(html, headers=headers)