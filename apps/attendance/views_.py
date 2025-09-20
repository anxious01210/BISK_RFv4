from datetime import datetime, time
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Q, OuterRef, Subquery
from django.http import HttpResponse, HttpResponseForbidden
from django.shortcuts import render
from django.utils import timezone
from django.utils.dateparse import parse_date, parse_datetime
from django.utils.html import escape
from django.views.decorators.http import require_GET, require_POST
from django.urls import reverse
from math import ceil
from django.db.models import F, Window
from django.db.models.functions import RowNumber
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
    if not _is_lunch_supervisor(request.user):
        return HttpResponseForbidden("Requires lunch_supervisor")

    is_supervisor = _is_lunch_supervisor(request.user)
    confirm_url = reverse("attendance:confirm_record")

    qs = (AttendanceRecord.objects
          .select_related("student", "period__template", "best_camera")
          .all())

    # Period filter
    period_param = request.GET.get("period", "lunch-pri,lunch-sec")
    parts = [p.strip() for p in (period_param or "").split(",") if p.strip()]
    if parts:
        q = Q()
        for p in parts:
            q |= Q(period__template__name__iexact=p)
        qs = qs.filter(q)

    # Camera filter
    camera_param = request.GET.get("camera")
    if camera_param:
        cams = [c.strip() for c in camera_param.split(",") if c.strip()]
        if cams:
            cam_q = Q()
            for c in cams:
                cam_q |= Q(best_camera__name__iexact=c)
            qs = qs.filter(cam_q)

    # Eligible
    if request.GET.get("eligible_only") in ("1", "true", "True", "yes"):
        qs = qs.filter(student__has_lunch=True)

    # Min score
    try:
        min_score = float(request.GET.get("min_score", "") or "nan")
        if min_score == min_score:
            qs = qs.filter(best_score__gte=min_score)
    except ValueError:
        pass

    # Search
    qtxt = request.GET.get("q")
    if qtxt:
        qs = qs.filter(
            Q(student__h_code__icontains=qtxt) |
            Q(student__first_name__icontains=qtxt) |
            Q(student__middle_name__icontains=qtxt) |
            Q(student__last_name__icontains=qtxt)
        )

    # Date range
    start_dt = _parse_any(request.GET.get("date_from"))
    end_dt = _parse_any(request.GET.get("date_to"))
    if start_dt:
        qs = qs.filter(best_seen__gte=start_dt)
    if end_dt:
        qs = qs.filter(best_seen__lte=end_dt)

    mode = request.GET.get("mode", "latest")

    # Cursor only for all mode
    at = _parse_any(request.GET.get("after_ts"))
    if at and mode != "latest":
        qs = qs.filter(best_seen__gte=at)

    if mode == "latest":
        latest_pk = (
            AttendanceRecord.objects
            .filter(student_id=OuterRef("student_id"),
                    period_id=OuterRef("period_id"))
            .order_by("-best_seen", "-id")
            .values("pk")[:1]
        )
        qs = qs.filter(pk=Subquery(latest_pk))

    qs = qs.order_by("best_seen")[:200]

    rows = []
    media_url = (settings.MEDIA_URL or "").rstrip("/")
    for rec in qs:
        st = rec.student
        h = escape(getattr(st, "h_code", ""))
        name = escape(st.full_name() if hasattr(st, "full_name") else "")
        per_name = escape(getattr(rec.period.template, "name", "")) if rec.period and rec.period.template else ""
        score = f"{float(rec.best_score or 0.0):.2f}"
        cam = escape(getattr(rec.best_camera, "name", "") or "-")
        eligible = "Yes" if getattr(st, "has_lunch", False) else "No"
        eligible_html = f"<span class='badge g'>LUNCH</span>" if st.has_lunch else f"<span class='badge r'>No Lunch</span>"
        ts = escape(rec.best_seen.astimezone().strftime("%H:%M:%S"))
        pc = int(getattr(rec, "pass_count", 1) or 1)
        badge_class = "badge r" if pc >= 2 else "badge g"

        # NEW: Confirm cell
        if getattr(rec, "confirmed", False):
            confirm_html = "<span class='badge g'>✔ Confirmed</span>"
        elif is_supervisor:
            # HTMX posts and asks server to return the updated <tr>, swapping in place
            confirm_html = (
                f"<button "
                f"  hx-post='{confirm_url}' "
                f"  hx-vals='{{\"id\": {rec.id}}}' "
                f"  hx-target='closest tr' "
                f"  hx-swap='outerHTML'"
                f">Confirm</button>"
            )
        else:
            confirm_html = "—"

        # crop = getattr(rec, "best_crop", "") or ""
        # crop_url = f"{media_url}/{crop.lstrip('/')}" if (media_url and crop) else ""
        # crop_html = f'<img src="{crop_url}" class="photo" loading="lazy"/>' if crop_url else "—"
        #
        # gallery_url = ""
        # try:
        #     if hasattr(st, "gallery_photo_relurl"):
        #         u = st.gallery_photo_relurl()
        #         gallery_url = u or ""
        # except Exception:
        #     gallery_url = ""
        # gallery_html = f'<img src="{gallery_url}" class="photo" loading="lazy"/>' if gallery_url else "—"
        # img_html = f'<div style="display:flex;gap:6px;align-items:center">{gallery_html}{crop_html}</div>'

        crop = getattr(rec, "best_crop", "") or ""
        crop_url = f"{media_url}/{crop.lstrip('/')}" if (media_url and crop) else ""

        gallery_url = ""
        try:
            if hasattr(st, "gallery_photo_relurl"):
                u = st.gallery_photo_relurl()
                gallery_url = u or ""
        except Exception:
            gallery_url = ""

        def _ph(url: str) -> str:
            if not url:
                return "<div class='photo ph-empty'></div>"
            return (
                "<div class='ph'>"
                f"  <img src='{url}' class='photo js-preview' data-full='{url}' loading='lazy'/>"
                f"  <a class='open' href='{url}' target='_blank' title='Open in new tab'>↗</a>"
                "</div>"
            )

        img_html = (
            "<div class='photos'>"
            f"{_ph(gallery_url)}{_ph(crop_url)}"
            "</div>"
        )

        rows.append(
            f"<tr>"
            f"<td>{img_html}</td>"
            # f"<td><b>{h}</b><br/><span class='small'>{name}</span></td><br/><span class='small'>{escape(getattr(st, 'get_grade_display', lambda: getattr(st, 'grade', '') or '—')())}</span>"
            f"<td>"
            # f"  <b>{h}</b>"
            f"  <span>{h}</span>"
            f"  <br/><span class='small'>{name}</span>"
            f"  <br/><span class='small'>{escape(getattr(st, 'get_grade_display', lambda: getattr(st, 'grade', '') or '—')())}</span>"
            f"</td>"
            f"<td>{per_name}</td>"
            f"<td>{cam}</td>"
            f"<td>{eligible_html}</td>"
            f"<td>{confirm_html}</td>"  # NEW column
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
            last_ts = qs[len(qs) - 1].best_seen.astimezone().isoformat()
            headers["HX-Trigger"] = f'{{"lunch:last_ts": "{last_ts}"}}'
        except Exception:
            pass

    return HttpResponse(html, headers=headers)


# put above lunch_stream_rows/confirm_record
def _photos_html(gallery_url: str, crop_url: str) -> str:
    def _ph(url: str) -> str:
        if not url:
            return "<div class='photo ph-empty'></div>"
        return (
            "<div class='ph'>"
            f"  <img src='{url}' class='photo js-preview' data-full='{url}' loading='lazy'/>"
            f"  <a class='open' href='{url}' target='_blank' title='Open in new tab'>↗</a>"
            "</div>"
        )

    return "<div class='photos'>" + _ph(gallery_url) + _ph(crop_url) + "</div>"


# NEW: Lunch supervisor confirms a record
@login_required
@require_POST
def confirm_record(request):
    if not _is_lunch_supervisor(request.user):
        return HttpResponseForbidden("Requires lunch_supervisor")

    rid = request.POST.get("id") or request.POST.get("record_id")
    if not rid:
        return HttpResponse("Missing id", status=400)

    try:
        rec = AttendanceRecord.objects.select_related(
            "student", "period__template", "best_camera"
        ).get(pk=int(rid))
    except AttendanceRecord.DoesNotExist:
        return HttpResponse("Not found", status=404)

    # Mark confirmed
    if not getattr(rec, "confirmed", False):
        rec.confirmed = True
        rec.save(update_fields=["confirmed"])

    # Rebuild a single <tr> (same shape as lunch_stream_rows)
    st = rec.student
    media_url = (settings.MEDIA_URL or "").rstrip("/")
    h = escape(getattr(st, "h_code", ""))
    name = escape(st.full_name() if hasattr(st, "full_name") else "")
    per_name = escape(getattr(rec.period.template, "name", "")) if rec.period and rec.period.template else ""
    score = f"{float(rec.best_score or 0.0):.2f}"
    cam = escape(getattr(rec.best_camera, "name", "") or "-")
    eligible_html = f"<span class='badge g'>LUNCH</span>" if getattr(st, 'has_lunch',
                                                                     False) else f"<span class='badge r'>No Lunch</span>"
    ts = escape(rec.best_seen.astimezone().strftime("%H:%M:%S"))
    pc = int(getattr(rec, "pass_count", 1) or 1)
    badge_class = "badge r" if pc >= 2 else "badge g"
    confirm_html = "<span class='badge g'>✔ Confirmed</span>"

    crop = getattr(rec, "best_crop", "") or ""
    crop_url = f"{media_url}/{crop.lstrip('/')}" if (media_url and crop) else ""
    crop_html = f'<img src="{crop_url}" class="photo" loading="lazy"/>' if crop_url else "—"

    gallery_url = ""
    try:
        if hasattr(st, "gallery_photo_relurl"):
            u = st.gallery_photo_relurl()
            gallery_url = u or ""
    except Exception:
        gallery_url = ""
    gallery_html = f'<img src="{gallery_url}" class="photo" loading="lazy"/>' if gallery_url else "—"
    # img_html = f'<div style="display:flex;gap:6px;align-items:center">{gallery_html}{crop_html}</div>'
    img_html = _photos_html(gallery_url, crop_url)

    html = (
        f"<tr>"
        f"<td>{img_html}</td>"
        # f"<td><b>{h}</b><br/><span class='small'>{name}</span></td>"
        f"<td>"
        # f"  <b>{h}</b>"
        f"  <span>{h}</span>"
        f"  <br/><span class='small'>{name}</span>"
        f"  <br/><span class='small'>{escape(getattr(st, 'get_grade_display', lambda: getattr(st, 'grade', '') or '—')())}</span>"
        f"</td>"
        f"<td>{per_name}</td>"
        f"<td>{cam}</td>"
        f"<td>{eligible_html}</td>"
        f"<td>{confirm_html}</td>"
        f"<td><span class='{badge_class}'>{pc}</span></td>"
        f"<td>{score}</td>"
        f"<td class='small'>{ts}</td>"
        f"</tr>"
    )
    return HttpResponse(html)
