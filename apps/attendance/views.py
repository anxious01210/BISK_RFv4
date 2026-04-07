from datetime import datetime, time
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Q, OuterRef, Subquery
from django.http import HttpResponse, HttpResponseForbidden
from django.shortcuts import render, get_object_or_404
from django.utils import timezone
from apps.cameras.models import Camera
from django.utils.dateparse import parse_date, parse_datetime
from django.utils.html import escape
from django.views.decorators.http import require_GET, require_POST
from django.urls import reverse
from math import ceil
from django.db.models import F, Window
from django.db.models.functions import RowNumber
from .models import (
    DashboardTag,
    PeriodTemplate,
    AttendanceRecord,
    MealSubscription,
    MealRecord,
    Wallet,
    WalletTransaction,
)

def get_meal_price(meal_profile, period_template):
    if not meal_profile or not period_template:
        return 0

    try:
        mpp = meal_profile.period_prices.filter(
            period_template=period_template,
            is_enabled=True
        ).first()
        return mpp.price_iqd if mpp else 0
    except Exception:
        return 0

def _parse_any(s):
    if not s:
        return None
    s = s.strip()
    # try datetime then date
    dt = parse_datetime(s)
    if dt:
        return dt
    d = parse_date(s)
    if d:
        return datetime.combine(d, time.min).replace(tzinfo=timezone.get_current_timezone())
    return None


# small helper so both the list view and confirm swap render the same photo+row
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


def _row_html(rec, media_url, is_supervisor, confirm_url):
    from django.utils import timezone

    def fmt_hms(dt):
        if not dt:
            return "—"
        try:
            return dt.astimezone(timezone.get_current_timezone()).strftime("%H:%M:%S")
        except Exception:
            try:
                return dt.strftime("%H:%M:%S")
            except Exception:
                return "—"

    st = rec.student
    h = escape(getattr(st, "h_code", ""))
    name = escape(st.full_name() if hasattr(st, "full_name") else "")
    grade = escape(getattr(st, "get_grade_display", lambda: getattr(st, "grade", "") or "—")())
    per_name = escape(getattr(rec.period.template, "name", "")) if rec.period and rec.period.template else ""
    score = f"{float(rec.best_score or 0.0):.2f}"
    cam = escape(getattr(rec.best_camera, "name", "") or "-")

    meal = getattr(rec, "meal_record", None)

    if meal and meal.eligible_at_time is True:
        eligible_html = "<span class='badge g'>MEAL</span>"
    elif meal and meal.eligible_at_time is False:
        eligible_html = "<span class='badge r'>No Meal</span>"
    else:
        eligible_html = "<span class='badge g'>MEAL</span>" if getattr(st, "has_meal", False) else "<span class='badge r'>No Meal</span>"
    pc = int(getattr(rec, "pass_count", 1) or 1)
    badge_class = "badge r" if pc >= 2 else "badge g"

    # Confirm cell
    if meal and meal.status == "confirmed":
        confirm_html = "<span class='badge g'>✔ Confirmed</span>"
    elif meal and meal.status == "denied":
        confirm_html = "<span class='badge r'>Denied</span>"
    elif meal and meal.status == "unpaid":
        confirm_html = "<span class='badge r'>Unpaid</span>"
    elif is_supervisor:
        confirm_html = (
            f"<button hx-post='{confirm_url}' "
            f"hx-vals='{{\"id\": {rec.id}}}' "
            f"hx-target='closest tr' hx-swap='outerHTML'>Confirm</button>"
        )
    else:
        confirm_html = "—"

    # Photos (gallery + crop)
    crop = getattr(rec, "best_crop", "") or ""
    crop_url = f"{media_url}/{crop.lstrip('/')}" if (media_url and crop) else ""
    gallery_url = ""
    try:
        if hasattr(st, "gallery_photo_relurl"):
            u = st.gallery_photo_relurl()
            gallery_url = u or ""
    except Exception:
        gallery_url = ""
    img_html = _photos_html(gallery_url, crop_url)

    # Times: first/best/last stacked, and last_pass_at separate
    t_first = fmt_hms(getattr(rec, "first_seen", None))
    t_best  = fmt_hms(getattr(rec, "best_seen",  None))
    t_last  = fmt_hms(getattr(rec, "last_seen",  None))
    t_pass  = fmt_hms(getattr(rec, "last_pass_at", None))

    times_html = (
        f"<span class='small'><span class='k'>first:</span> {t_first}</span><br/>"
        f"<span class='small'><span class='k'>best:</span> {t_best}</span><br/>"
        f"<span class='small'><span class='k'>last:</span> {t_last}</span>"
    )
    pass_html = f"<span class='small'>{t_pass}</span>"

    return (
        f"<tr>"
        f"<td>{img_html}</td>"
        f"<td><b>{h}</b><br/><span class='small'>{name}</span><br/><span class='small'>{grade}</span></td>"
        f"<td>{per_name}</td>"
        f"<td>{cam}</td>"
        f"<td>{eligible_html}</td>"
        f"<td>{confirm_html}</td>"
        f"<td><span class='{badge_class}'>{pc}</span></td>"
        f"<td>{score}</td>"
        f"<td class='small'>{times_html}</td>"   # ← first/best/last stacked
        f"<td class='small'>{pass_html}</td>"    # ← last_pass_at
        f"</tr>"
    )


def _is_meal_supervisor(user) -> bool:
    return user.is_authenticated and user.groups.filter(name="meal_supervisor").exists()


@login_required
def meal_page(request):
    if not _is_meal_supervisor(request.user):
        return HttpResponseForbidden("Requires meal_supervisor")

    # Precompute default lists for the toolbar (controlled vocabulary tag = 'meal')
    meal_tag = DashboardTag.objects.filter(slug=DashboardTag.MEAL).first()
    default_periods = []
    default_cameras = []
    if meal_tag:
        default_periods = list(
            PeriodTemplate.objects.filter(usage_tags=meal_tag).order_by("order").values_list("name", flat=True)
        )
        now = timezone.now()
        default_cameras = list(
            Camera.objects
            .filter(usage_tags=meal_tag, is_active=True)
            # .exclude(pause_until__gt=now)
            .order_by("name")
            .values_list("name", flat=True)
        )
    ctx = {
        "default_periods": ",".join(default_periods),
        "default_cameras": ",".join(default_cameras),
        "show_empty_notice": (not default_periods or not default_cameras),
    }
    print(f"{ctx}")
    # return render(request, "attendance/dash/meal.html")
    return render(request, "attendance/dash/meal.html", ctx)

@login_required
@require_GET
def meal_stream_rows(request):
    if not _is_meal_supervisor(request.user):
        return HttpResponseForbidden("Requires meal_supervisor")

    is_supervisor = _is_meal_supervisor(request.user)
    confirm_url = reverse("attendance:confirm_record")

    # ----- base queryset and filters (shared) -----
    qs = (AttendanceRecord.objects.select_related("student", "period__template", "best_camera", "meal_record"))

    # # Period filter
    # period_param = request.GET.get("period", "meal-pri,meal-sec")

    # Period filter — default to 'meal'-tagged PeriodTemplates if empty
    period_param = request.GET.get("period", "")
    parts = [p.strip() for p in (period_param or "").split(",") if p.strip()]
    if not parts:
        tag = DashboardTag.objects.filter(slug=DashboardTag.MEAL).first()
        if tag:
            parts = list(PeriodTemplate.objects.filter(usage_tags=tag).values_list("name", flat=True))

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
    else:
        tag = DashboardTag.objects.filter(slug=DashboardTag.MEAL).first()
        if tag:
            now = timezone.now()
            cams = list(
                Camera.objects.filter(usage_tags=tag, is_active=True)
                .exclude(pause_until__gt=now)
                .values_list("name", flat=True)
            )
            if cams:
                cam_q = Q()
                for c in cams:
                    cam_q |= Q(best_camera__name__iexact=c)
                qs = qs.filter(cam_q)

    # If neither a request filter nor tagged defaults exist → show nothing.
    if (not request.GET.get("period") and not parts) or (not request.GET.get("camera") and not camera_param and not locals().get("cams")):
        qs = qs.none()


    # Eligible
    if request.GET.get("eligible_only") in ("1", "true", "True", "yes"):
        qs = qs.filter(student__has_meal=True)

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

    mode = request.GET.get("mode", "latest").strip()  # 'latest' | 'all' | 'lastN'

    last_n = request.GET.get("last_n") or ""
    try:
        last_n = int(last_n) if last_n else None
    except Exception:
        last_n = None

    # paging params (used when mode is 'all' or 'lastN')
    try:
        page = max(1, int(request.GET.get("page", "1")))
    except Exception:
        page = 1
    try:
        page_size = max(1, int(request.GET.get("page_size", "20")))
    except Exception:
        page_size = 20

    at = _parse_any(request.GET.get("after_ts"))

    # ------------- branching by mode -------------
    totals = None  # (total, pages, page)
    if mode == "latest":
        # stream: allow cursor
        if at:
            qs = qs.filter(best_seen__gte=at)

        latest_pk = (
            AttendanceRecord.objects
            .filter(student_id=OuterRef("student_id"),
                    period_id=OuterRef("period_id"))
            .order_by("-best_seen", "-id")
            .values("pk")[:1]
        )
        qs = qs.filter(pk=Subquery(latest_pk)).order_by("best_seen")[:200]

    elif mode == "all":
        qs = qs.order_by("-best_seen", "-id")
        total = qs.count()
        pages = max(1, ceil(total / page_size))
        page = min(page, pages)
        start = (page - 1) * page_size
        qs = qs[start:start + page_size]
        totals = (total, pages, page)

    elif mode == "lastN" and (last_n or 0) > 0:
        # Prefer window function (Postgres, SQLite 3.25+) for accuracy and speed
        try:
            qs = (qs
                  .annotate(
                      rn=Window(
                          expression=RowNumber(),
                          partition_by=[F("student_id"), F("period_id")],
                          order_by=[F("best_seen").desc(), F("id").desc()],
                      )
                  )
                  .filter(rn__lte=last_n)
                  .order_by("-best_seen", "-id"))
            total = qs.count()
            pages = max(1, ceil(total / page_size))
            page = min(page, pages)
            start = (page - 1) * page_size
            qs = qs[start:start + page_size]
            totals = (total, pages, page)
        except Exception:
            # Fallback: coarse but portable
            qs = qs.order_by("-best_seen", "-id")[:5000]
            # group in Python
            from collections import defaultdict
            groups = defaultdict(list)
            for r in qs:
                groups[(r.student_id, r.period_id)].append(r)
            flat = []
            for key, rows in groups.items():
                flat.extend(rows[:last_n])
            # sort again newest first
            flat.sort(key=lambda r: (r.best_seen, r.id), reverse=True)
            total = len(flat)
            pages = max(1, ceil(total / page_size))
            page = min(page, pages)
            start = (page - 1) * page_size
            qs = flat[start:start + page_size]
            totals = (total, pages, page)

    else:
        # unknown -> default to latest stream semantics
        if at:
            qs = qs.filter(best_seen__gte=at)
        latest_pk = (
            AttendanceRecord.objects
            .filter(student_id=OuterRef("student_id"),
                    period_id=OuterRef("period_id"))
            .order_by("-best_seen", "-id")
            .values("pk")[:1]
        )
        qs = qs.filter(pk=Subquery(latest_pk)).order_by("best_seen")[:200]

    # ------------- render rows -------------
    rows = []
    media_url = (settings.MEDIA_URL or "").rstrip("/")
    is_supervisor = _is_meal_supervisor(request.user)
    confirm_url = reverse("attendance:confirm_record")


    for rec in qs:
        rows.append(_row_html(rec, media_url, is_supervisor, confirm_url))

    html = "\n".join(rows)
    if not rows and not at:
        # html = "<tr><td colspan='9' class='small' style='opacity:.7;'>No matches for current filters.</td></tr>"
        html = "<tr><td colspan='10' class='small' style='opacity:.7;'>No matches for current filters.</td></tr>"

    headers = {}
    try:
        if mode == "latest" and rows:
            last_ts = (qs[len(qs) - 1].best_seen if hasattr(qs, "__getitem__") else None)
            if last_ts:
                headers["HX-Trigger"] = f'{{"meal:last_ts": "{last_ts.astimezone().isoformat()}"}}'
        elif totals:
            total, pages, page = totals
            headers["HX-Trigger"] = f'{{"meal:total": {total}, "meal:pages": {pages}, "meal:page": {page}}}'
    except Exception:
        pass

    return HttpResponse(html, headers=headers)




# NEW: Meal supervisor confirms a record
@login_required
@require_POST
def confirm_record(request):
    if not _is_meal_supervisor(request.user):
        return HttpResponseForbidden("Requires meal_supervisor")

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
    # Find active subscription for the record date
    try:
        meal_day = timezone.localdate(rec.best_seen) if rec.best_seen else timezone.localdate()
    except Exception:
        meal_day = timezone.localdate()

    sub = (
        MealSubscription.objects
        .select_related("meal_profile")
        .filter(
            student_id=rec.student_id,
            status=MealSubscription.STATUS_ACTIVE,
            start_date__lte=meal_day,
            end_date__gte=meal_day,
        )
        .order_by("start_date", "id")
        .first()
    )

    meal, created = MealRecord.objects.get_or_create(
        attendance_record=rec,
        defaults={
            "status": MealRecord.STATUS_PENDING,
        }
    )

    # Snapshot subscription/profile/eligibility
    meal.meal_subscription = sub
    meal.meal_profile = sub.meal_profile if sub and sub.meal_profile_id else None
    meal.eligible_at_time = bool(sub)

    if meal.meal_profile:
        if meal.meal_profile.mode == MealRecord.MODE_WALLET:
            meal.mode_snapshot = MealRecord.MODE_WALLET
        else:
            meal.mode_snapshot = MealRecord.MODE_DATE_RANGE
    else:
        meal.mode_snapshot = MealRecord.MODE_DATE_RANGE if sub else MealRecord.MODE_NONE

    # For now, confirmation logic is still simple:
    # active subscription -> confirmed
    # no subscription -> denied
    from django.db import transaction

    if sub and meal.meal_profile:

        profile = meal.meal_profile

        # CASE 1: DATE RANGE MODE
        if profile.mode == MealRecord.MODE_DATE_RANGE:
            meal.status = MealRecord.STATUS_CONFIRMED
            meal.confirmed_at = timezone.now()
            meal.confirmed_by = request.user

        # CASE 2: WALLET MODE
        elif profile.mode == MealRecord.MODE_WALLET:

            wallet = Wallet.objects.filter(student_id=rec.student_id, is_active=True).first()

            period_template = rec.period.template if rec.period else None
            price = get_meal_price(profile, period_template)

            meal.price_base_iqd = price
            meal.discount_iqd = 0
            meal.final_charge_iqd = price
            meal.wallet_transaction = None
            meal.wallet_refund_transaction = None
            meal.wallet_balance_before_iqd = 0
            meal.wallet_balance_after_iqd = 0

            if wallet:
                meal.wallet_balance_before_iqd = wallet.balance_iqd
                meal.wallet_balance_after_iqd = wallet.balance_iqd

                # Enough balance
                if wallet.balance_iqd >= price:
                    wallet.balance_iqd -= price
                    wallet.save()

                    tx = WalletTransaction.objects.create(
                        wallet=wallet,
                        student=rec.student,
                        tx_type="debit",
                        amount_iqd=price,
                        balance_before_iqd=meal.wallet_balance_before_iqd,
                        balance_after_iqd=wallet.balance_iqd,
                        attendance_record=rec,
                        created_by=request.user,
                    )

                    meal.wallet_transaction = tx
                    meal.wallet_balance_after_iqd = wallet.balance_iqd
                    meal.status = MealRecord.STATUS_CONFIRMED
                    meal.confirmed_at = timezone.now()
                    meal.confirmed_by = request.user

                # Not enough balance
                else:
                    mode = profile.insufficient_funds_mode

                    if mode == "deny":
                        meal.status = MealRecord.STATUS_DENIED
                        meal.confirmed_at = None
                        meal.confirmed_by = None
                        meal.wallet_transaction = None

                    elif mode == "allow_unpaid":
                        meal.status = MealRecord.STATUS_UNPAID
                        meal.confirmed_at = timezone.now()
                        meal.confirmed_by = request.user
                        meal.wallet_balance_after_iqd = meal.wallet_balance_before_iqd
                        meal.wallet_transaction = None

                    elif mode == "allow_negative":
                        wallet.balance_iqd -= price
                        wallet.save()

                        tx = WalletTransaction.objects.create(
                            wallet=wallet,
                            student=rec.student,
                            tx_type="debit",
                            amount_iqd=price,
                            balance_before_iqd=meal.wallet_balance_before_iqd,
                            balance_after_iqd=wallet.balance_iqd,
                            attendance_record=rec,
                            created_by=request.user,
                        )

                        meal.wallet_transaction = tx
                        meal.wallet_balance_after_iqd = wallet.balance_iqd
                        meal.status = MealRecord.STATUS_CONFIRMED
                        meal.confirmed_at = timezone.now()
                        meal.confirmed_by = request.user

            else:
                meal.status = MealRecord.STATUS_DENIED
                meal.confirmed_at = None
                meal.confirmed_by = None
                meal.wallet_transaction = None
                meal.wallet_balance_before_iqd = 0
                meal.wallet_balance_after_iqd = 0

    else:
        meal.status = MealRecord.STATUS_DENIED
        meal.confirmed_at = None
        meal.confirmed_by = None
        meal.wallet_transaction = None
        meal.wallet_refund_transaction = None
        meal.wallet_balance_before_iqd = 0
        meal.wallet_balance_after_iqd = 0

    meal.save()


    # Rebuild a single <tr> (same shape as meal_stream_rows)
    st = rec.student
    media_url = (settings.MEDIA_URL or "").rstrip("/")
    h = escape(getattr(st, "h_code", ""))
    name = escape(st.full_name() if hasattr(st, "full_name") else "")
    per_name = escape(getattr(rec.period.template, "name", "")) if rec.period and rec.period.template else ""
    score = f"{float(rec.best_score or 0.0):.2f}"
    cam = escape(getattr(rec.best_camera, "name", "") or "-")
    if meal.eligible_at_time is True:
        eligible_html = "<span class='badge g'>MEAL</span>"
    elif meal.eligible_at_time is False:
        eligible_html = "<span class='badge r'>No Meal</span>"
    else:
        eligible_html = "<span class='badge g'>MEAL</span>" if getattr(st, "has_meal", False) else "<span class='badge r'>No Meal</span>"
    # ts = escape(rec.best_seen.astimezone().strftime("%H:%M:%S"))
    # ts = escape(rec.best_seen.astimezone().strftime("%Y-%m-%d %H:%M:%S"))
    pc = int(getattr(rec, "pass_count", 1) or 1)
    badge_class = "badge r" if pc >= 2 else "badge g"
    if meal.status == MealRecord.STATUS_CONFIRMED:
        confirm_html = "<span class='badge g'>✔ Confirmed</span>"
    elif meal.status == MealRecord.STATUS_DENIED:
        confirm_html = "<span class='badge r'>Denied</span>"
    elif meal.status == MealRecord.STATUS_UNPAID:
        confirm_html = "<span class='badge r'>Unpaid</span>"
    else:
        confirm_html = f"<span class='badge'>{escape(meal.status.title())}</span>"

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

    def fmt_hms(dt):
        if not dt:
            return "—"
        try:
            return dt.astimezone(timezone.get_current_timezone()).strftime("%H:%M:%S")
        except Exception:
            try:
                return dt.strftime("%H:%M:%S")
            except Exception:
                return "—"

    t_first = fmt_hms(getattr(rec, "first_seen", None))
    t_best = fmt_hms(getattr(rec, "best_seen", None))
    t_last = fmt_hms(getattr(rec, "last_seen", None))
    t_pass = fmt_hms(getattr(rec, "last_pass_at", None))

    times_html = (
        f"<span class='small'><span class='k'>first:</span> {t_first}</span><br/>"
        f"<span class='small'><span class='k'>best:</span> {t_best}</span><br/>"
        f"<span class='small'><span class='k'>last:</span> {t_last}</span>"
    )
    pass_html = f"<span class='small'>{t_pass}</span>"

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
        f"<td class='small'>{times_html}</td>"
        f"<td class='small'>{pass_html}</td>"
        f"</tr>"
    )
    return HttpResponse(html)
