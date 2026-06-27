from datetime import datetime, time, date, timedelta
from math import ceil

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Q, OuterRef, Subquery, F, Window, Count
from django.db.models.functions import RowNumber
from django.http import HttpResponse, HttpResponseForbidden
from django.shortcuts import render
from django.urls import reverse
from django.utils import timezone
from django.utils.dateparse import parse_date, parse_datetime
from django.utils.html import escape
from django.views.decorators.http import require_GET, require_POST

from apps.cameras.models import Camera
from .models import (
    DashboardTag,
    PeriodTemplate,
    PeriodOccurrence,
    RecognitionSettings,
    AttendanceEvent,
    AttendanceRecord,
    MealSubscription,
    MealRecord,
    Wallet,
    WalletTransaction,
    MealProfile,
)


def _academic_year_end(today):
    year = today.year
    if today.month >= 7:
        return date(year + 1, 6, 30)
    return date(year, 6, 30)


def _get_or_create_postpaid_profile():
    profile, _ = MealProfile.objects.get_or_create(
        name="Postpaid Wallet",
        defaults={
            "mode": MealRecord.MODE_WALLET,
            "insufficient_funds_mode": "allow_negative",
            "allow_supervisor_confirm": True,
            "allow_supervisor_unconfirm": True,
            "allow_supervisor_refund": True,
            "require_reason_on_override": False,
            "require_reason_on_unconfirm": False,
            "require_reason_on_refund": False,
            "is_active": True,
            "notes": "Reusable supervisor-created postpaid wallet profile",
        },
    )
    return profile


def get_meal_price(meal_profile, period_template):
    if not meal_profile or not period_template:
        return 0
    try:
        mpp = meal_profile.period_prices.filter(
            period_template=period_template,
            is_enabled=True,
        ).first()
        return mpp.price_iqd if mpp else 0
    except Exception:
        return 0


def _parse_any(s):
    if not s:
        return None
    s = s.strip()
    dt = parse_datetime(s)
    if dt:
        return dt
    d = parse_date(s)
    if d:
        return datetime.combine(d, time.min).replace(
            tzinfo=timezone.get_current_timezone()
        )
    return None


def _get_active_subscriptions_for_day(student_id, meal_day):
    return list(
        MealSubscription.objects
        .select_related("meal_profile")
        .filter(
            student_id=student_id,
            status=MealSubscription.STATUS_ACTIVE,
            start_date__lte=meal_day,
            end_date__gte=meal_day,
        )
        .order_by("priority", "start_date", "id")
    )


def _resolve_effective_meal_setup(student, meal_day, period_template=None):
    subs = _get_active_subscriptions_for_day(student.id, meal_day)
    wallet = Wallet.objects.filter(student=student, is_active=True).first()

    for sub in subs:
        profile = sub.meal_profile
        if not profile or not getattr(profile, "is_active", False):
            continue

        if profile.mode == MealRecord.MODE_DATE_RANGE:
            return {
                "sub": sub,
                "profile": profile,
                "wallet": None,
                "mode": MealRecord.MODE_DATE_RANGE,
                "priority": sub.priority,
                "is_fallback": sub.priority > 1,
                "blocked": False,
                "block_reason": "",
                "credit_limit_iqd": None,
                "projected_balance_iqd": None,
            }

        if profile.mode == MealRecord.MODE_WALLET:
            price = int(get_meal_price(profile, period_template) or 0)
            balance = int(getattr(wallet, "balance_iqd", 0) or 0) if wallet else 0
            projected = balance - price

            blocked = False
            block_reason = ""
            credit_limit_iqd = getattr(profile, "credit_limit_iqd", None)
            funds_mode = getattr(profile, "insufficient_funds_mode", "")

            if price <= 0:
                blocked = True
                block_reason = "No meal price is configured for this period."
            elif not wallet:
                blocked = True
                block_reason = "No active wallet exists."
            elif (
                    funds_mode == MealProfile.INSUFFICIENT_ALLOW_NEGATIVE
                    and credit_limit_iqd is not None
            ):
                if projected < -int(credit_limit_iqd):
                    blocked = True
                    block_reason = (
                        f"Credit limit reached. Balance {balance} IQD, charge {price} IQD, "
                        f"limit -{int(credit_limit_iqd)} IQD."
                    )

            return {
                "sub": sub,
                "profile": profile,
                "wallet": wallet,
                "mode": MealRecord.MODE_WALLET,
                "priority": sub.priority,
                "is_fallback": sub.priority > 1,
                "blocked": blocked,
                "block_reason": block_reason,
                "credit_limit_iqd": credit_limit_iqd,
                "projected_balance_iqd": projected,
            }

    return {
        "sub": None,
        "profile": None,
        "wallet": wallet,
        "mode": MealRecord.MODE_NONE,
        "priority": None,
        "is_fallback": False,
        "blocked": False,
        "block_reason": "",
        "credit_limit_iqd": None,
        "projected_balance_iqd": None,
    }


def _photos_html(gallery_url: str, best_crop_url: str, latest_crop_url: str) -> str:
    def _ph(url: str, label: str) -> str:
        if not url:
            return (
                "<div class='ph'>"
                f"  <div class='photo ph-empty'></div>"
                f"  <div class='small k' style='margin-top:4px;text-align:center'>{label}</div>"
                "</div>"
            )
        return (
            "<div class='ph'>"
            f"  <img src='{url}' class='photo js-preview' data-full='{url}' loading='lazy'/>"
            f"  <a class='open' href='{url}' target='_blank' title='Open in new tab'>↗</a>"
            f"  <div class='small k' style='margin-top:4px;text-align:center'>{label}</div>"
            "</div>"
        )

    return (
            "<div class='photos'>"
            + _ph(gallery_url, "ID")
            + _ph(best_crop_url, "Best")
            + _ph(latest_crop_url, "Latest")
            + "</div>"
    )


def _format_clock(value):
    if not value:
        return "—"
    try:
        return value.strftime("%H:%M")
    except Exception:
        return str(value)


def _build_period_cards(default_periods, default_cameras):
    tag = DashboardTag.objects.filter(slug=DashboardTag.MEAL).first()
    if not tag:
        return []

    periods = list(
        PeriodTemplate.objects
        .filter(usage_tags=tag)
        .order_by("order", "name")
    )

    now = timezone.localtime()
    today = now.date()
    occurrences = {
        occ.template_id: occ
        for occ in PeriodOccurrence.objects.filter(
            template__in=periods,
            date=today,
            is_school_day=True,
        ).select_related("template")
    }
    tomorrow = today + timedelta(days=1)

    tz = timezone.get_current_timezone()
    latest_event_qs = (
        AttendanceEvent.objects
        .filter(student_id=OuterRef("student_id"), period_id=OuterRef("period_id"))
        .order_by("-ts", "-id")
    )

    base_qs = (
        AttendanceRecord.objects
        .filter(
            period__template__in=periods,
            last_seen__gte=datetime.combine(today, time.min).replace(tzinfo=tz),
            last_seen__lt=datetime.combine(tomorrow, time.min).replace(tzinfo=tz),
        )
        .annotate(
            latest_event_camera_name=Subquery(latest_event_qs.values("camera__name")[:1]),
        )
    )

    if default_cameras:
        base_qs = base_qs.filter(latest_event_camera_name__in=default_cameras)

    rec_map = {
        row["period__template_id"]: row["c"]
        for row in base_qs.values("period__template_id").annotate(c=Count("id"))
    }
    dr_map = {
        row["period__template_id"]: row["c"]
        for row in base_qs.filter(
            meal_record__status=MealRecord.STATUS_CONFIRMED,
            meal_record__mode_snapshot=MealRecord.MODE_DATE_RANGE,
        ).values("period__template_id").annotate(c=Count("id"))
    }
    wal_map = {
        row["period__template_id"]: row["c"]
        for row in base_qs.filter(
            meal_record__status=MealRecord.STATUS_CONFIRMED,
            meal_record__mode_snapshot=MealRecord.MODE_WALLET,
        ).values("period__template_id").annotate(c=Count("id"))
    }
    blk_map = {
        row["period__template_id"]: row["c"]
        for row in base_qs.filter(
            meal_record__status=MealRecord.STATUS_DENIED,
        ).values("period__template_id").annotate(c=Count("id"))
    }

    cards = []
    now_t = now.time()

    for pt in periods:
        occ = occurrences.get(pt.id)

        if occ:
            st_dt = getattr(occ, "start_dt", None)
            et_dt = getattr(occ, "end_dt", None)

            st_local = timezone.localtime(st_dt) if st_dt else None
            et_local = timezone.localtime(et_dt) if et_dt else None

            st = st_local.time() if st_local else getattr(pt, "start_time", None)
            et = et_local.time() if et_local else getattr(pt, "end_time", None)

            is_active_now = bool(st_local and et_local and st_local <= now <= et_local)
        else:
            st = getattr(pt, "start_time", None)
            et = getattr(pt, "end_time", None)
            is_active_now = False

        cards.append({
            "name": pt.name,
            "label": pt.name,
            "start": _format_clock(st),
            "end": _format_clock(et),
            "is_active_now": is_active_now,
            "selected": pt.name in default_periods,
            "recognized_count": rec_map.get(pt.id, 0),
            "date_range_count": dr_map.get(pt.id, 0),
            "wallet_count": wal_map.get(pt.id, 0),
            "blocked_count": blk_map.get(pt.id, 0),
        })

    return cards


def _row_html(rec, media_url, is_supervisor, confirm_url, reverse_url):
    try:
        meal_day = timezone.localdate(rec.best_seen) if rec.best_seen else timezone.localdate()
    except Exception:
        meal_day = timezone.localdate()

    period_template = rec.period.template if rec.period else None
    resolved = _resolve_effective_meal_setup(rec.student, meal_day, period_template=period_template)
    active_sub = resolved["sub"]
    active_profile = resolved["profile"]
    resolved_wallet = resolved["wallet"]
    resolved_blocked = resolved["blocked"]
    resolved_is_fallback = resolved["is_fallback"]

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
    best_score_txt = f"{float(rec.best_score or 0.0):.2f}"
    latest_score_txt = f"{float(getattr(rec, 'latest_event_score', None) or rec.best_score or 0.0):.2f}"

    best_cam = escape(getattr(rec.best_camera, "name", "") or "-")
    latest_cam = escape(getattr(rec, "latest_event_camera_name", "") or best_cam or "-")

    meal = getattr(rec, "meal_record", None)
    plan_html = "<span class='small k'>—</span>"

    sub = active_sub
    profile = active_profile

    if sub and profile:
        profile_name = escape(getattr(profile, "name", "") or "Profile")
        mode = getattr(profile, "mode", "") or ""
        mode_label = "DATE-RANGE" if mode == MealRecord.MODE_DATE_RANGE else "WALLET"

        if mode == MealRecord.MODE_DATE_RANGE:
            end_date = getattr(sub, "end_date", None)
            end_txt = escape(end_date.isoformat()) if end_date else "—"
            priority_label = "Fallback" if resolved_is_fallback else "Primary"
            plan_html = (
                f"<span class='badge'>{mode_label}</span><br/>"
                f"<span class='small'>{profile_name}</span><br/>"
                f"<span class='small k'>{priority_label}</span><br/>"
                f"<span class='small k'>until:</span> <span class='small'>{end_txt}</span>"
            )
        else:
            wallet = resolved_wallet or getattr(rec.student, "wallet", None)
            bal = getattr(wallet, "balance_iqd", None)
            if bal is None and meal is not None:
                bal = getattr(meal, "wallet_balance_after_iqd", None)
            if bal is None and meal is not None:
                bal = getattr(meal, "wallet_balance_before_iqd", None)

            bal = int(bal or 0)
            bal_class = "g" if bal >= 0 else "r"

            policy = getattr(profile, "insufficient_funds_mode", "") or ""
            if policy == "allow_negative":
                policy_badge = "<span class='badge'>POSTPAID OK</span>"
            elif policy == "allow_unpaid":
                policy_badge = "<span class='badge'>UNPAID OK</span>"
            elif policy == "deny":
                policy_badge = "<span class='badge r'>DENY</span>"
            else:
                policy_badge = "<span class='badge'>—</span>"

            priority_label = "Fallback" if resolved_is_fallback else "Primary"
            extra_badge = ""
            if resolved_blocked:
                extra_badge = "<br/><span class='badge r'>LIMIT REACHED</span>"

            plan_html = (
                f"<span class='badge'>{mode_label}</span><br/>"
                f"<span class='small'>{profile_name}</span><br/>"
                f"<span class='small k'>{priority_label}</span><br/>"
                f"<span class='small {bal_class}'>Bal: {bal} IQD</span><br/>"
                f"{policy_badge}"
                f"{extra_badge}"
            )
    else:
        plan_html = "<span class='badge r'>No active plan</span>"

    eligible_html = "<span class='badge g'>MEAL</span>" if sub else "<span class='badge r'>No Meal</span>"
    pc = int(getattr(rec, "pass_count", 1) or 1)
    badge_class = "badge r" if pc >= 2 else "badge g"

    meal_profile = getattr(meal, "meal_profile", None) if meal else None
    allow_refund = bool(meal_profile and getattr(meal_profile, "allow_supervisor_refund", False))
    allow_unconfirm = bool(meal_profile and getattr(meal_profile, "allow_supervisor_unconfirm", False))

    if meal and meal.status == MealRecord.STATUS_CONFIRMED:
        status_html = "<span class='badge g'>✔ Confirmed</span>"
        if is_supervisor and meal.mode_snapshot == MealRecord.MODE_WALLET and allow_refund:
            status_html += (
                "<br/>"
                f"<button class='btn-mini' hx-post='{reverse_url}' "
                f"hx-vals='{{\"id\": {rec.id}}}' "
                f"hx-target='closest tr' hx-swap='outerHTML'>Refund</button>"
            )
        elif is_supervisor and meal.mode_snapshot == MealRecord.MODE_DATE_RANGE and allow_unconfirm:
            status_html += (
                "<br/>"
                f"<button class='btn-mini' hx-post='{reverse_url}' "
                f"hx-vals='{{\"id\": {rec.id}}}' "
                f"hx-target='closest tr' hx-swap='outerHTML'>Unconfirm</button>"
            )
    elif meal and meal.status == MealRecord.STATUS_UNPAID:
        status_html = "<span class='badge r'>Unpaid</span>"
        if is_supervisor and allow_unconfirm:
            status_html += (
                "<br/>"
                f"<button class='btn-mini' hx-post='{reverse_url}' "
                f"hx-vals='{{\"id\": {rec.id}}}' "
                f"hx-target='closest tr' hx-swap='outerHTML'>Void</button>"
            )
    elif meal and meal.status == MealRecord.STATUS_DENIED:
        if (
                is_supervisor
                and sub
                and profile
                and not resolved_blocked
                and getattr(profile, "allow_supervisor_confirm", False)
        ):
            status_html = (
                "<span class='badge r'>Denied</span><br/>"
                f"<button hx-post='{confirm_url}' "
                f"hx-vals='{{\"id\": {rec.id}}}' "
                f"hx-target='closest tr' hx-swap='outerHTML'>Reconfirm</button>"
            )
        else:
            status_html = "<span class='badge r'>Denied</span>"
    elif meal and meal.status == MealRecord.STATUS_REFUNDED:
        status_html = "<span class='badge'>Refunded</span>"
        if is_supervisor and meal_profile and getattr(meal_profile, "allow_supervisor_confirm", False):
            status_html += (
                "<br/>"
                f"<button class='btn-mini' hx-post='{confirm_url}' "
                f"hx-vals='{{\"id\": {rec.id}}}' "
                f"hx-target='closest tr' hx-swap='outerHTML'>Reconfirm</button>"
            )
    elif meal and meal.status == MealRecord.STATUS_VOIDED:
        status_html = "<span class='badge'>Voided</span>"
        if is_supervisor and meal_profile and getattr(meal_profile, "allow_supervisor_confirm", False):
            status_html += (
                "<br/>"
                f"<button class='btn-mini' hx-post='{confirm_url}' "
                f"hx-vals='{{\"id\": {rec.id}}}' "
                f"hx-target='closest tr' hx-swap='outerHTML'>Reconfirm</button>"
            )
    elif is_supervisor:
        if sub and profile and not resolved_blocked:
            status_html = (
                f"<button hx-post='{confirm_url}' "
                f"hx-vals='{{\"id\": {rec.id}}}' "
                f"hx-target='closest tr' hx-swap='outerHTML'>Confirm</button>"
            )
        elif sub and profile and resolved_blocked:
            credit_limit_iqd = getattr(profile, "credit_limit_iqd", None)
            bal_now = int(getattr(resolved_wallet, "balance_iqd", 0) or 0) if resolved_wallet else 0
            charge_iqd = int(get_meal_price(profile, rec.period.template if rec.period else None) or 0)
            status_html = (
                f"<span class='badge r'>Blocked</span><br/>"
                f"<span class='small status-reason'>"
                f"Credit limit reached.<br/>"
                # f"Balance {bal_now} • Charge {charge_iqd} • Limit -{int(credit_limit_iqd or 0)}"
                f"• Balance: {bal_now} IQD.<br/>"
                f"• Charge: {charge_iqd} IQD.<br/>"
                f"• Limit: -{int(credit_limit_iqd or 0)} IQD."
                f"</span>"
            )
        else:
            status_html = "<span class='small k'>No applicable plan</span>"
    else:
        status_html = "—"

    best_crop = getattr(rec, "best_crop", "") or ""
    best_crop_url = f"{media_url}/{best_crop.lstrip('/')}" if (media_url and best_crop) else ""

    latest_crop = getattr(rec, "latest_event_crop_path", "") or ""
    latest_crop_url = f"{media_url}/{latest_crop.lstrip('/')}" if (media_url and latest_crop) else ""

    gallery_url = ""
    try:
        if hasattr(st, "gallery_photo_relurl"):
            u = st.gallery_photo_relurl()
            gallery_url = u or ""
    except Exception:
        gallery_url = ""

    img_html = _photos_html(gallery_url, best_crop_url, latest_crop_url)

    t_first = fmt_hms(getattr(rec, "first_seen", None))
    t_best = fmt_hms(getattr(rec, "best_seen", None))
    t_latest = fmt_hms(getattr(rec, "last_seen", None))
    t_pass = fmt_hms(getattr(rec, "last_pass_at", None))

    times_html = (
        f"<span class='small'><span class='k'>First:</span> {t_first}</span><br/>"
        f"<span class='small'><span class='k'>Best:</span> {t_best}</span><br/>"
        f"<span class='small'><span class='k'>Latest:</span> {t_latest}</span>"
    )
    pass_html = f"<span class='small'>{t_pass}</span>"

    return (
        f"<tr>"
        f"<td>{img_html}</td>"
        f"<td><b>{h}</b><br/><span class='small'>{name}</span><br/><span class='small'>{grade}</span></td>"
        f"<td>{per_name}</td>"
        f"<td>"
        f"  <span class='small'><span class='k'>Best:</span> {best_cam}</span><br/>"
        f"  <span class='small'><span class='k'>Latest:</span> {latest_cam}</span>"
        f"</td>"
        f"<td>{eligible_html}</td>"
        f"<td>{plan_html}</td>"
        f"<td>{status_html}</td>"
        f"<td><span class='{badge_class}'>{pc}</span></td>"
        f"<td>"
        f"  <span class='small'><span class='k'>Best:</span> {best_score_txt}</span><br/>"
        f"  <span class='small'><span class='k'>Latest:</span> {latest_score_txt}</span>"
        f"</td>"
        f"<td class='small'>{times_html}</td>"
        f"<td class='small'>{pass_html}</td>"
        f"</tr>"
    )


def _is_meal_supervisor(user) -> bool:
    return user.is_authenticated and user.groups.filter(name="meal_supervisor").exists()


@login_required
def meal_page(request):
    if not _is_meal_supervisor(request.user):
        return HttpResponseForbidden("Requires meal_supervisor")

    meal_tag = DashboardTag.objects.filter(slug=DashboardTag.MEAL).first()
    default_periods = []
    default_cameras = []
    if meal_tag:
        default_periods = list(
            PeriodTemplate.objects.filter(usage_tags=meal_tag).order_by("order").values_list("name", flat=True)
        )
        default_cameras = list(
            Camera.objects
            .filter(usage_tags=meal_tag, is_active=True)
            .order_by("name")
            .values_list("name", flat=True)
        )

    period_cards = _build_period_cards(default_periods, default_cameras)

    rs = RecognitionSettings.get_solo()

    ctx = {
        "default_periods": ",".join(default_periods),
        "default_cameras": ",".join(default_cameras),
        "show_empty_notice": (not default_periods or not default_cameras),
        "period_cards": period_cards,
        "pass_gap_window_sec": int(getattr(rs, "pass_gap_window_sec", 120) or 120),
    }
    return render(request, "attendance/dash/meal.html", ctx)


@login_required
@require_GET
def meal_period_cards(request):
    if not _is_meal_supervisor(request.user):
        return HttpResponseForbidden("Requires meal_supervisor")

    period_param = request.GET.get("period", "")
    camera_param = request.GET.get("camera", "")

    selected_periods = [p.strip() for p in period_param.split(",") if p.strip()]
    selected_cameras = [c.strip() for c in camera_param.split(",") if c.strip()]

    # Fall back to meal-tag defaults if nothing passed
    if not selected_periods or not selected_cameras:
        meal_tag = DashboardTag.objects.filter(slug=DashboardTag.MEAL).first()
        if meal_tag:
            if not selected_periods:
                selected_periods = list(
                    PeriodTemplate.objects
                    .filter(usage_tags=meal_tag)
                    .order_by("order")
                    .values_list("name", flat=True)
                )
            if not selected_cameras:
                selected_cameras = list(
                    Camera.objects
                    .filter(usage_tags=meal_tag, is_active=True)
                    .order_by("name")
                    .values_list("name", flat=True)
                )

    period_cards = _build_period_cards(selected_periods, selected_cameras)

    html = []
    for card in period_cards:
        classes = ["period-card"]
        if card.get("selected"):
            classes.append("selected")
        if card.get("is_active_now"):
            classes.append("active-now")
        else:
            classes.append("dimmed")

        badge = "ACTIVE" if card.get("is_active_now") else "PERIOD"

        html.append(
            f"""
            <div
              class="{' '.join(classes)}"
              data-period-name="{escape(card['name'])}"
              data-selected="{'1' if card.get('selected') else '0'}"
              title="{escape(card['label'])} ({escape(card['start'])} - {escape(card['end'])})"
            >
              <div class="pc-top">
                <div class="pc-title">{escape(card['label'])}</div>
                <div class="pc-badge">{badge}</div>
              </div>
              <div class="pc-time">{escape(card['start'])} – {escape(card['end'])}</div>
              <div class="pc-counts">
                <span><em>Rec</em> <b>{card['recognized_count']}</b></span>
                <span><em>DR</em> <b>{card['date_range_count']}</b></span>
                <span><em>Wal</em> <b>{card['wallet_count']}</b></span>
                <span class="blk"><em>Blk</em> <b>{card['blocked_count']}</b></span>
              </div>
            </div>
            """
        )

    return HttpResponse("".join(html))


@login_required
@require_GET
def meal_camera_health(request):
    if not _is_meal_supervisor(request.user):
        return HttpResponseForbidden("Requires meal_supervisor")

    tag = DashboardTag.objects.filter(slug=DashboardTag.MEAL).first()
    if not tag:
        return HttpResponse("<div class='small k'>No meal dashboard tag found.</div>")

    now = timezone.now()

    cameras = list(
        Camera.objects
        .filter(usage_tags=tag, is_active=True)
        .order_by("name")
    )

    html = []
    html.append("<div class='camera-health-grid'>")

    for cam in cameras:
        latest_event = (
            AttendanceEvent.objects
            .filter(camera=cam)
            .order_by("-ts", "-id")
            .first()
        )

        if latest_event and latest_event.ts:
            age_sec = int((now - latest_event.ts).total_seconds())

            if age_sec <= 15:
                level = "ok"
                label = "Fresh"
            elif age_sec <= 60:
                level = "warn"
                label = "Stale"
            else:
                level = "bad"
                label = "No recent events"

            if age_sec < 60:
                age_txt = f"{age_sec}s ago"
            else:
                age_txt = f"{age_sec // 60}m {age_sec % 60}s ago"
        else:
            level = "bad"
            label = "No data"
            age_txt = "—"

        html.append(
            f"""
            <div class="camera-health-item {level}">
              <span class="dot"></span>
              <b>{escape(cam.name)}</b>
              <span>{label}</span>
              <span class="k">{age_txt}</span>
            </div>
            """
        )

    html.append("</div>")
    return HttpResponse("".join(html))


@login_required
@require_GET
def meal_stream_rows(request):
    if not _is_meal_supervisor(request.user):
        return HttpResponseForbidden("Requires meal_supervisor")

    is_supervisor = _is_meal_supervisor(request.user)
    confirm_url = reverse("attendance:confirm_record")
    reverse_url = reverse("attendance:reverse_record")

    latest_event_qs = (
        AttendanceEvent.objects
        .filter(student_id=OuterRef("student_id"), period_id=OuterRef("period_id"))
        .order_by("-ts", "-id")
    )

    qs = (
        AttendanceRecord.objects
        .select_related("student", "period__template", "best_camera", "meal_record")
        .annotate(
            latest_event_ts=Subquery(latest_event_qs.values("ts")[:1]),
            latest_event_camera_id=Subquery(latest_event_qs.values("camera_id")[:1]),
            latest_event_camera_name=Subquery(latest_event_qs.values("camera__name")[:1]),
            latest_event_score=Subquery(latest_event_qs.values("score")[:1]),
            latest_event_crop_path=Subquery(latest_event_qs.values("crop_path")[:1]),
        )
    )

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

    camera_param = request.GET.get("camera")
    if camera_param:
        cams = [c.strip() for c in camera_param.split(",") if c.strip()]
        if cams:
            cam_q = Q()
            for c in cams:
                cam_q |= Q(latest_event_camera_name__iexact=c)
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
                    cam_q |= Q(latest_event_camera_name__iexact=c)
                qs = qs.filter(cam_q)

    if (not request.GET.get("period") and not parts) or (
            not request.GET.get("camera") and not camera_param and not locals().get("cams")):
        qs = qs.none()

    if request.GET.get("eligible_only") in ("1", "true", "True", "yes"):
        qs = qs.filter(student__has_meal=True)

    try:
        min_score = float(request.GET.get("min_score", "") or "nan")
        if min_score == min_score:
            qs = qs.filter(best_score__gte=min_score)
    except ValueError:
        pass

    qtxt = request.GET.get("q")
    if qtxt:
        qs = qs.filter(
            Q(student__h_code__icontains=qtxt)
            | Q(student__first_name__icontains=qtxt)
            | Q(student__middle_name__icontains=qtxt)
            | Q(student__last_name__icontains=qtxt)
        )

    start_dt = _parse_any(request.GET.get("date_from"))
    end_dt = _parse_any(request.GET.get("date_to"))
    if start_dt:
        qs = qs.filter(last_seen__gte=start_dt)
    if end_dt:
        qs = qs.filter(last_seen__lte=end_dt)

    mode = request.GET.get("mode", "latest").strip()
    last_n = request.GET.get("last_n") or ""
    try:
        last_n = int(last_n) if last_n else None
    except Exception:
        last_n = None

    try:
        page = max(1, int(request.GET.get("page", "1")))
    except Exception:
        page = 1
    try:
        page_size = max(1, int(request.GET.get("page_size", "20")))
    except Exception:
        page_size = 20

    at = _parse_any(request.GET.get("after_ts"))
    totals = None

    if mode == "latest":
        if at:
            qs = qs.filter(last_seen__gte=at)

        latest_pk = (
            AttendanceRecord.objects
            .filter(student_id=OuterRef("student_id"), period_id=OuterRef("period_id"))
            .order_by("-last_seen", "-id")
            .values("pk")[:1]
        )
        qs = qs.filter(pk=Subquery(latest_pk)).order_by("last_seen")[:200]

    elif mode == "all":
        qs = qs.order_by("-last_seen", "-id")
        total = qs.count()
        pages = max(1, ceil(total / page_size))
        page = min(page, pages)
        start = (page - 1) * page_size
        qs = qs[start:start + page_size]
        totals = (total, pages, page)

    elif mode == "lastN" and (last_n or 0) > 0:
        try:
            qs = (
                qs.annotate(
                    rn=Window(
                        expression=RowNumber(),
                        partition_by=[F("student_id"), F("period_id")],
                        order_by=[F("last_seen").desc(), F("id").desc()],
                    )
                )
                .filter(rn__lte=last_n)
                .order_by("-last_seen", "-id")
            )
            total = qs.count()
            pages = max(1, ceil(total / page_size))
            page = min(page, pages)
            start = (page - 1) * page_size
            qs = qs[start:start + page_size]
            totals = (total, pages, page)
        except Exception:
            from collections import defaultdict
            qs = qs.order_by("-last_seen", "-id")[:5000]
            groups = defaultdict(list)
            for r in qs:
                groups[(r.student_id, r.period_id)].append(r)
            flat = []
            for rows in groups.values():
                flat.extend(rows[:last_n])
            flat.sort(key=lambda r: (r.last_seen, r.id), reverse=True)
            total = len(flat)
            pages = max(1, ceil(total / page_size))
            page = min(page, pages)
            start = (page - 1) * page_size
            qs = flat[start:start + page_size]
            totals = (total, pages, page)

    else:
        if at:
            qs = qs.filter(last_seen__gte=at)
        latest_pk = (
            AttendanceRecord.objects
            .filter(student_id=OuterRef("student_id"), period_id=OuterRef("period_id"))
            .order_by("-last_seen", "-id")
            .values("pk")[:1]
        )
        qs = qs.filter(pk=Subquery(latest_pk)).order_by("last_seen")[:200]

    rows = []
    media_url = (settings.MEDIA_URL or "").rstrip("/")
    for rec in qs:
        rows.append(_row_html(rec, media_url, is_supervisor, confirm_url, reverse_url))

    html = "\n".join(rows)
    if not rows and not at:
        html = "<tr><td colspan='11' class='small' style='opacity:.7;'>No matches for current filters.</td></tr>"

    headers = {}
    try:
        if mode == "latest" and rows:
            last_ts = (qs[len(qs) - 1].last_seen if hasattr(qs, "__getitem__") else None)
            if last_ts:
                headers["HX-Trigger"] = f'{{"meal:last_ts": "{last_ts.astimezone().isoformat()}"}}'
        elif totals:
            total, pages, page = totals
            headers["HX-Trigger"] = f'{{"meal:total": {total}, "meal:pages": {pages}, "meal:page": {page}}}'
    except Exception:
        pass

    return HttpResponse(html, headers=headers)


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

    try:
        meal_day = timezone.localdate(rec.best_seen) if rec.best_seen else timezone.localdate()
    except Exception:
        meal_day = timezone.localdate()

    period_template = rec.period.template if rec.period else None
    resolved = _resolve_effective_meal_setup(rec.student, meal_day, period_template=period_template)
    sub = resolved["sub"]
    profile = resolved["profile"]
    wallet = resolved["wallet"]
    resolved_blocked = resolved["blocked"]

    meal, _created = MealRecord.objects.get_or_create(
        attendance_record=rec,
        defaults={"status": MealRecord.STATUS_PENDING},
    )

    # Prevent duplicate charging if this row is already confirmed.
    # A stale browser tab or repeated POST should not debit the wallet again.
    if meal.status == MealRecord.STATUS_CONFIRMED:
        media_url = (settings.MEDIA_URL or "").rstrip("/")
        confirm_url = reverse("attendance:confirm_record")
        reverse_url = reverse("attendance:reverse_record")
        html = _row_html(rec, media_url, True, confirm_url, reverse_url)
        return HttpResponse(
            html,
            headers={"HX-Trigger": '{"meal:refresh_cards": true}'}
        )

    meal.reversed_at = None
    meal.reversed_by = None
    meal.wallet_refund_transaction = None

    # Clear old denial reason when attempting a fresh confirm/reconfirm
    meal.reason_code = ""
    meal.reason_notes = ""

    meal.meal_subscription = sub
    meal.meal_profile = profile
    meal.eligible_at_time = bool(sub)

    if profile:
        meal.mode_snapshot = (
            MealRecord.MODE_WALLET if profile.mode == MealRecord.MODE_WALLET else MealRecord.MODE_DATE_RANGE
        )
    else:
        meal.mode_snapshot = MealRecord.MODE_DATE_RANGE if sub else MealRecord.MODE_NONE

    if sub and profile and resolved_blocked:
        meal.status = MealRecord.STATUS_DENIED
        meal.confirmed_at = None
        meal.confirmed_by = None
        meal.wallet_transaction = None
        meal.wallet_refund_transaction = None
        meal.wallet_balance_before_iqd = getattr(wallet, "balance_iqd", 0) if wallet else 0
        meal.wallet_balance_after_iqd = meal.wallet_balance_before_iqd
        meal.reason_code = "credit_limit_reached"
        meal.reason_notes = resolved.get("block_reason", "")
        meal.save()

        media_url = (settings.MEDIA_URL or "").rstrip("/")
        confirm_url = reverse("attendance:confirm_record")
        reverse_url = reverse("attendance:reverse_record")
        html = _row_html(rec, media_url, True, confirm_url, reverse_url)
        return HttpResponse(
            html,
            headers={"HX-Trigger": '{"meal:refresh_cards": true}'}
        )

    if sub and profile:
        if profile.mode == MealRecord.MODE_DATE_RANGE:
            meal.wallet_transaction = None
            meal.wallet_refund_transaction = None
            meal.wallet_balance_before_iqd = 0
            meal.wallet_balance_after_iqd = 0
            meal.status = MealRecord.STATUS_CONFIRMED
            meal.confirmed_at = timezone.now()
            meal.confirmed_by = request.user

        elif profile.mode == MealRecord.MODE_WALLET:
            period_template = rec.period.template if rec.period else None
            price = int(get_meal_price(profile, period_template) or 0)

            if price <= 0:
                meal.price_base_iqd = 0
                meal.discount_iqd = 0
                meal.final_charge_iqd = 0
                meal.wallet_transaction = None
                meal.wallet_refund_transaction = None
                meal.wallet_balance_before_iqd = getattr(
                    Wallet.objects.filter(student_id=rec.student_id, is_active=True).first(),
                    "balance_iqd",
                    0,
                )
                meal.wallet_balance_after_iqd = meal.wallet_balance_before_iqd
                meal.status = MealRecord.STATUS_DENIED
                meal.confirmed_at = None
                meal.confirmed_by = None
                meal.save()

                media_url = (settings.MEDIA_URL or "").rstrip("/")
                confirm_url = reverse("attendance:confirm_record")
                reverse_url = reverse("attendance:reverse_record")
                html = _row_html(rec, media_url, True, confirm_url, reverse_url)
                return HttpResponse(
                    html,
                    headers={"HX-Trigger": '{"meal:refresh_cards": true}'}
                )

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
                        credit_limit_iqd = getattr(profile, "credit_limit_iqd", None)
                        projected_balance = wallet.balance_iqd - price

                        if credit_limit_iqd is not None and projected_balance < -int(credit_limit_iqd):
                            meal.status = MealRecord.STATUS_DENIED
                            meal.confirmed_at = None
                            meal.confirmed_by = None
                            meal.wallet_transaction = None
                            meal.wallet_balance_after_iqd = meal.wallet_balance_before_iqd
                            meal.reason_code = "credit_limit_reached"
                            meal.reason_notes = (
                                f"Credit limit reached. Balance {meal.wallet_balance_before_iqd} IQD, "
                                f"charge {price} IQD, limit -{int(credit_limit_iqd)} IQD."
                            )
                        else:
                            wallet.balance_iqd = projected_balance
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

    media_url = (settings.MEDIA_URL or "").rstrip("/")
    confirm_url = reverse("attendance:confirm_record")
    reverse_url = reverse("attendance:reverse_record")
    html = _row_html(rec, media_url, True, confirm_url, reverse_url)
    return HttpResponse(
        html,
        headers={"HX-Trigger": '{"meal:refresh_cards": true}'}
    )


@login_required
@require_POST
def reverse_record(request):
    if not _is_meal_supervisor(request.user):
        return HttpResponseForbidden("Requires meal_supervisor")

    rid = request.POST.get("id") or request.POST.get("record_id")
    if not rid:
        return HttpResponse("Missing id", status=400)

    try:
        rec = AttendanceRecord.objects.select_related(
            "student",
            "period__template",
            "best_camera",
            "meal_record__meal_profile",
            "meal_record__wallet_transaction__wallet",
        ).get(pk=int(rid))
    except AttendanceRecord.DoesNotExist:
        return HttpResponse("Not found", status=404)

    meal = getattr(rec, "meal_record", None)
    if not meal:
        return HttpResponse("No meal record", status=400)

    profile = meal.meal_profile

    if meal.status == MealRecord.STATUS_CONFIRMED and meal.mode_snapshot == MealRecord.MODE_WALLET:
        if not (profile and getattr(profile, "allow_supervisor_refund", False)):
            return HttpResponse("Refund not allowed", status=403)

        tx0 = meal.wallet_transaction
        if not tx0 or not tx0.wallet_id:
            return HttpResponse("Original wallet transaction missing", status=400)

        wallet = tx0.wallet
        refund_amount = meal.final_charge_iqd or meal.price_base_iqd or 0

        balance_before = wallet.balance_iqd
        wallet.balance_iqd += refund_amount
        wallet.save()

        refund_tx = WalletTransaction.objects.create(
            wallet=wallet,
            student=rec.student,
            tx_type=WalletTransaction.TYPE_REFUND,
            amount_iqd=refund_amount,
            balance_before_iqd=balance_before,
            balance_after_iqd=wallet.balance_iqd,
            attendance_record=rec,
            created_by=request.user,
            reason_code="refund",
            notes=f"Refund for meal record #{meal.id}",
        )

        meal.wallet_refund_transaction = refund_tx
        meal.status = MealRecord.STATUS_REFUNDED
        meal.reversed_at = timezone.now()
        meal.reversed_by = request.user
        meal.save()
    elif meal.status == MealRecord.STATUS_CONFIRMED and meal.mode_snapshot == MealRecord.MODE_DATE_RANGE:
        if not (profile and getattr(profile, "allow_supervisor_unconfirm", False)):
            return HttpResponse("Unconfirm not allowed", status=403)

        meal.status = MealRecord.STATUS_VOIDED
        meal.reversed_at = timezone.now()
        meal.reversed_by = request.user
        meal.save()
    elif meal.status == MealRecord.STATUS_UNPAID:
        if not (profile and getattr(profile, "allow_supervisor_unconfirm", False)):
            return HttpResponse("Void not allowed", status=403)

        meal.status = MealRecord.STATUS_VOIDED
        meal.reversed_at = timezone.now()
        meal.reversed_by = request.user
        meal.save()
    else:
        return HttpResponse("Nothing to reverse", status=400)

    media_url = (settings.MEDIA_URL or "").rstrip("/")
    confirm_url = reverse("attendance:confirm_record")
    reverse_url = reverse("attendance:reverse_record")
    html = _row_html(rec, media_url, True, confirm_url, reverse_url)
    return HttpResponse(
        html,
        headers={"HX-Trigger": '{"meal:refresh_cards": true}'}
    )


@login_required
@require_POST
def enable_postpaid(request):
    if not _is_meal_supervisor(request.user):
        return HttpResponseForbidden("Requires meal_supervisor")

    rid = request.POST.get("id") or request.POST.get("record_id")
    if not rid:
        return HttpResponse("Missing id", status=400)

    try:
        rec = AttendanceRecord.objects.select_related(
            "student",
            "period__template",
            "best_camera",
            "meal_record",
        ).get(pk=int(rid))
    except AttendanceRecord.DoesNotExist:
        return HttpResponse("Not found", status=404)

    student = rec.student
    today = timezone.localdate()
    end_date = _academic_year_end(today)

    profile = _get_or_create_postpaid_profile()

    wallet, _ = Wallet.objects.get_or_create(
        student=student,
        defaults={
            "balance_iqd": 0,
            "is_active": True,
            "notes": "Auto-created from dashboard postpaid enable",
        },
    )

    sub = (
        MealSubscription.objects
        .filter(
            student=student,
            meal_profile=profile,
            status=MealSubscription.STATUS_ACTIVE,
            end_date__gte=today,
        )
        .order_by("-end_date")
        .first()
    )

    if not sub:
        sub = MealSubscription.objects.create(
            student=student,
            plan_type="other",
            meal_profile=profile,
            status=MealSubscription.STATUS_ACTIVE,
            start_date=today,
            end_date=end_date,
            notes=f"Auto-created from dashboard by {request.user.username}",
            source=MealSubscription.SOURCE_DASHBOARD_POSTPAID,
        )

    media_url = (settings.MEDIA_URL or "").rstrip("/")
    confirm_url = reverse("attendance:confirm_record")
    reverse_url = reverse("attendance:reverse_record")
    html = _row_html(rec, media_url, True, confirm_url, reverse_url)
    return HttpResponse(
        html,
        headers={"HX-Trigger": '{"meal:refresh_cards": true}'}
    )
