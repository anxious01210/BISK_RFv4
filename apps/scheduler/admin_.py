# apps/scheduler/admin.py
import os, signal, psutil, json
from django.urls import path
from django.shortcuts import redirect
from django.contrib import admin, messages
from django.utils.safestring import mark_safe
from django import forms
from django.forms.models import BaseInlineFormSet
from django.core.exceptions import ValidationError
from django.db.models import F, Window
from django.db.models.functions import RowNumber

from .models import (
    StreamProfile,
    SchedulePolicy,
    ScheduleWindow,
    ScheduleException,
    RunningProcess,
    RunnerHeartbeat,
)
from django.utils import timezone
from django.db.models import Q
import datetime as _dt
from django.urls import reverse
from django.utils.html import format_html
from .services import _stop, enforce_schedules  # Linux-only SIGTERM to process group
from django.conf import settings
from datetime import timedelta
# from apps.scheduler import periodic  # for _pid_alive
from apps.scheduler.services import enforcer as _enf  # provides _pid_alive
from pathlib import Path
from django.utils.timesince import timesince

# Status thresholds (seconds)
ONLINE = getattr(settings, "HEARTBEAT_ONLINE_SEC", max(15, int(getattr(settings, "HEARTBEAT_INTERVAL_SEC", 10) * 1.5)))
STALE = getattr(settings, "HEARTBEAT_STALE_SEC", 45)
OFFLINE = getattr(settings, "HEARTBEAT_OFFLINE_SEC", 120)

admin.site.empty_value_display = "—"
from django.contrib import admin
from .models import GlobalResourceSettings, CameraResourceOverride


@admin.register(GlobalResourceSettings)
class GlobalResourceSettingsAdmin(admin.ModelAdmin):
    list_display = ("id", "cpu_nice", "cpu_affinity", "cpu_quota_percent",
                    "gpu_index", "gpu_memory_fraction", "gpu_target_util_percent",
                    "max_fps_default", "det_set_max")
    readonly_fields = ("id",)


@admin.register(CameraResourceOverride)
class CameraResourceOverrideAdmin(admin.ModelAdmin):
    list_display = ("camera", "cpu_nice", "cpu_affinity", "cpu_quota_percent",
                    "gpu_index", "gpu_memory_fraction", "gpu_target_util_percent",
                    "max_fps", "det_set_max")
    search_fields = ("camera__name",)


# ----------------------------
# Overlap detection (hard prevent)
# ----------------------------
def _to_ranges(start, end):
    """
    Return list of minute ranges for a window, handling overnight and 24h.
    - 24h: start == end -> [(0, 1440)]
    - Same-day: start < end -> [(s, e)]
    - Overnight: start > end -> [(s, 1440), (0, e)]
    """
    s = start.hour * 60 + start.minute
    e = end.hour * 60 + end.minute
    if s == e:
        return [(0, 1440)]
    if s < e:
        return [(s, e)]
    return [(s, 1440), (0, e)]


def _overlaps(a, b):
    """a,b are (lo,hi) in minutes; overlap if length > 0."""
    return not (a[1] <= b[0] or b[1] <= a[0])


class WindowInlineFormSet(BaseInlineFormSet):
    """
    Hard validation: prevent overlapping windows on the same weekday
    inside a single Policy.
    """

    def clean(self):
        super().clean()
        by_dow = {i: [] for i in range(7)}
        for form in self.forms:
            if not form.cleaned_data or form.cleaned_data.get("DELETE"):
                continue
            day = form.cleaned_data.get("day_of_week")
            start = form.cleaned_data.get("start_time")
            end = form.cleaned_data.get("end_time")
            if day is None or start is None or end is None:
                raise ValidationError("All schedule windows must have day, start, and end times.")
            for r in _to_ranges(start, end):
                by_dow[day].append(r)

        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for dow, ranges in by_dow.items():
            for i in range(len(ranges)):
                for j in range(i + 1, len(ranges)):
                    if _overlaps(ranges[i], ranges[j]):
                        def fmt(rr):
                            return f"{rr[0] // 60:02d}:{rr[0] % 60:02d}-{rr[1] // 60:02d}:{rr[1] % 60:02d}"

                        raise ValidationError(
                            f"Overlapping windows on {day_names[dow]}: "
                            f"{fmt(ranges[i])} conflicts with {fmt(ranges[j])}."
                        )


# ----------------------------
# Forms with user-facing help_text
# ----------------------------
class StreamProfileForm(forms.ModelForm):
    class Meta:
        model = StreamProfile
        fields = "__all__"
        help_texts = {
            "name": "Human-friendly label. Example: “F1 • FFmpeg @ 6 fps”.",
            "script_type": "Runner type: 1=FFmpeg, 2=OpenCV.",
            "fps": "Target frames per second the runner should aim for.",
            "detection_set": "Detection input size. Use ‘auto’ unless you know you need 640/800/1024/1600/2048.",
            "extra_args": "Advanced: JSON of extra flags for the runner. Example: {\"gpu\": 0, \"resize\": \"1280x720\"}.",
            "is_active": "Uncheck to hide this profile from use.",
        }


class SchedulePolicyForm(forms.ModelForm):
    class Meta:
        model = SchedulePolicy
        fields = "__all__"
        help_texts = {
            "name": "Name of this policy (e.g., “Main Gate – Weekdays”).",
            "cameras": "Cameras controlled by this policy. All windows below apply to ALL selected cameras.",
            "is_enabled": "If unchecked, this policy is ignored by the enforcer.",
            "notes": "Optional notes for administrators.",
        }


class ScheduleWindowForm(forms.ModelForm):
    class Meta:
        model = ScheduleWindow
        fields = "__all__"
        help_texts = {
            "day_of_week": "Day index (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun).",
            "start_time": "Local start time. If equal to End time, the window covers the FULL day.",
            "end_time": "Local end time. For overnight spans, set End earlier than Start (e.g., 20:00 → 03:00).",
            "profile": "StreamProfile to run during this window.",
        }


class ScheduleExceptionForm(forms.ModelForm):
    class Meta:
        model = ScheduleException
        fields = "__all__"
        help_texts = {
            "date": "Calendar date for this exception (local time).",
            "mode": "‘off’ disables all windows on this date. "
                    "‘on’ and ‘window’ are reserved for future use.",
            "start_time": "Only used for ‘window’ mode (ignored otherwise).",
            "end_time": "Only used for ‘window’ mode (ignored otherwise).",
            "profile": "Optional profile for ‘window’ mode (ignored otherwise).",
        }


# ----------------------------
# Admin registrations
# ----------------------------
@admin.register(StreamProfile)
class StreamProfileAdmin(admin.ModelAdmin):
    form = StreamProfileForm
    list_display = ("name", "script_type", "fps", "detection_set", "is_active")
    list_filter = ("script_type", "detection_set", "is_active")
    search_fields = ("name",)
    save_on_top = True


class WindowInline(admin.TabularInline):
    model = ScheduleWindow
    form = ScheduleWindowForm
    formset = WindowInlineFormSet  # <- overlap prevention applied here
    extra = 1
    verbose_name = "Schedule window"
    verbose_name_plural = "Schedule windows (weekly): 0=Mon..6=Sun; 00:00→00:00 = full day; overnight allowed."


class ExceptionInline(admin.TabularInline):
    model = ScheduleException
    form = ScheduleExceptionForm
    extra = 0
    verbose_name = "Schedule exception"
    verbose_name_plural = "Schedule exceptions (date-specific overrides; ‘off’ supported today)."


@admin.register(SchedulePolicy)
class SchedulePolicyAdmin(admin.ModelAdmin):
    form = SchedulePolicyForm
    list_display = ("name", "is_enabled")
    list_filter = ("is_enabled",)
    filter_horizontal = ("cameras",)
    save_on_top = True
    inlines = [WindowInline, ExceptionInline]
    actions = ["action_enforce_now"]

    # Brief rules summary at the top of the form
    fieldsets = (
        (None, {
            "fields": ("name", "is_enabled", "notes", "cameras"),
            "description": mark_safe(
                "<ul style='margin:8px 0 0 16px'>"
                "<li><b>Day of week</b>: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun.</li>"
                "<li><b>Full day</b>: set Start = End (00:00 → 00:00).</li>"
                "<li><b>Overnight</b>: Start &gt; End spans midnight (e.g., 20:00 → 03:00).</li>"
                "<li><b>Overlaps</b>: saving is blocked if two windows overlap on the same day.</li>"
                "<li><b>Exceptions</b>: ‘off’ disables all windows for that date. Other modes are reserved.</li>"
                "</ul>"
            )
        }),
    )

    # Action: enforce now for selected policies
    @admin.action(description="Enforce schedules now (selected policies)")
    def action_enforce_now(self, request, queryset):
        result = enforce_schedules(policies=list(queryset))
        msg = (
                      f"Enforced. Started: {len(result.started)}, Stopped: {len(result.stopped)}<br>"
                      + "<br>".join(
                  [f"➕ {c}/{p} (pid={pid})" for c, p, pid in result.started]
                  + [f"🛑 {c}/{p} (pid={pid})" for c, p, pid in result.stopped]
              )
              ) or "No changes."
        messages.info(request, mark_safe(msg))

    # Top-right button: enforce all
    def get_urls(self):
        urls = super().get_urls()
        extra = [path("enforce-now/", self.admin_site.admin_view(self.enforce_now_view), name="scheduler_enforce_now")]
        return extra + urls

    def enforce_now_view(self, request):
        result = enforce_schedules()
        messages.success(
            request,
            f"Enforced. Started {len(result.started)}, Stopped {len(result.stopped)}, "
            f"Pruned {getattr(result, 'pruned_count', 0)} dead row(s)."
        )
        return redirect("admin:scheduler_schedulepolicy_changelist")


# ---------- FORM (help texts) ----------
class RunningProcessForm(forms.ModelForm):
    class Meta:
        model = RunningProcess
        fields = "__all__"
        help_texts = {
            "camera": "Camera this OS process is serving. Created automatically by the scheduler enforcer.",
            "profile": "StreamProfile used to start this process.",
            "pid": "Operating System process ID (Linux). Spawned in its own session group so it can be SIGTERM’d.",
            "status": "Current state (running | stopping | dead). Managed by the enforcer and admin actions.",
            "started_at": "When the process was started (server local time).",
            "last_heartbeat": "Updated by incoming runner heartbeats.",
            "meta": "Optional JSON sent by the runner (e.g., build info).",
        }

    # render JSONField nicely if present
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "meta" in self.fields:
            self.fields["meta"].widget = forms.Textarea(
                attrs={"rows": 8, "class": "vLargeTextField monospace"}
            )


# =====================
# RunningProcess admin
# =====================
@admin.register(RunningProcess)
class RunningProcessAdmin(admin.ModelAdmin):
    form = RunningProcessForm
    change_list_template = "admin/scheduler/runningprocess/change_list.html"

    list_display = (
        "camera", "profile",
        "pid", "pgid",
        "status_badge",
        "uptime_short",
        "age_secs",
        "fps_latest",
        "mem_mb",
        "threads",
        "children",
        "last_heartbeat_at",
        "last_error_badge",
        "snapshot_thumb",
    )
    list_select_related = ("camera", "profile")
    list_filter = ("status", "camera", "profile")
    search_fields = ("camera__name", "profile__name", "pid")
    ordering = ("-started_at",)

    # Detail is read-only; act via actions
    readonly_fields = (
        "camera", "profile", "pid", "status",
        "started_at", "last_heartbeat",
        "pgid", "uptime_full",
        "age_secs", "fps_latest",
        "mem_mb", "threads", "children",
        "cpu_affinity_display",
        "meta_pretty",
        "snapshot_preview",
        "effective_args",
        "runner_flavor",
    )

    fieldsets = (
        (None, {
            "fields": ("camera", "profile", "pid", "status", "started_at", "last_heartbeat", "runner_flavor", "meta_pretty"),
            "description": format_html(
                "<div style='margin:6px 0 10px'>"
                "<b>What is this?</b> A live record of a runner process the scheduler started.<br>"
                "<b>How is it created?</b> Automatically by the enforcer when a schedule window is active.<br>"
                "<b>How to use?</b> Use actions to Stop / Kill / Restart. Do not add rows manually.<br>"
                "<b>Tip:</b> If status is <i>stale</i> or <i>offline</i>, check heartbeats and camera connectivity."
                "</div>"
            )
        }),
        ("Process", {
            "fields": ("pgid", "uptime_full", "cpu_affinity_display"),
            "description": format_html(
                "<div class='help'>"
                "<b>PGID</b>: POSIX process group ID (we kill the whole group). "
                "<b>Uptime</b>: time since <i>started_at</i>. "
                "<b>CPU affinity</b>: cores pinned for the runner (inherited by FFmpeg).</div>"
            )
        }),
        ("Metrics (live)", {
            "fields": ("age_secs", "fps_latest", "mem_mb", "threads", "children"),
            "description": format_html(
                "<div class='help'>"
                "<b>Age</b>: seconds since this row’s <i>last_heartbeat</i>. "
                "<b>FPS</b>: latest reported by the runner (0 means ‘No video’). "
                "<b>RSS</b>: resident memory of the runner process. "
                "<b>Threads</b>: thread count for the PID. "
                "<b>Children</b>: number of child processes (e.g., FFmpeg).</div>"
            )
        }),
        ("Snapshot", {
            "fields": ("snapshot_preview",),
            "description": format_html(
                "<div class='help'>Single JPG updated by FFmpeg every <i>snapshot_every</i> seconds.</div>"
            )
        }),
    )

    # Simple indicator of which Python script the Enforcer used
    def runner_flavor(self, obj):
        try:
            from django.conf import settings
            return getattr(settings, "RUNNER_IMPL", "ffmpeg_all")
        except Exception:
            return "unknown"
    runner_flavor.short_description = "Runner flavor"

    def last_error_badge(self, obj):
        if getattr(obj, "last_error", ""):
            text = (obj.last_error[:60] + "…") if len(obj.last_error) > 60 else obj.last_error
            return format_html(
                '<span style="background:#dc2626;color:#fff;padding:2px 6px;border-radius:12px;font-size:11px;" '
                'title="{}">{}</span>',
                obj.last_error, text
            )
        return "—"

    last_error_badge.short_description = "Last error"

    # ---------- helpers ----------
    def _proc(self, pid):
        try:
            return psutil.Process(pid)
        except Exception:
            return None

    # ---------- list/detail columns ----------
    def pgid(self, obj):
        try:
            return os.getpgid(obj.pid)
        except Exception:
            return "—"

    def uptime_short(self, obj):
        if not obj.started_at:
            return "—"
        return timesince(obj.started_at, timezone.now()).split(",")[0]

    uptime_short.short_description = "Uptime"

    def uptime_full(self, obj):
        if not obj.started_at:
            return "—"
        return timesince(obj.started_at, timezone.now())

    uptime_full.short_description = "Uptime"

    def age_secs(self, obj):
        ts = obj.last_heartbeat
        return int((timezone.now() - ts).total_seconds()) if ts else None

    age_secs.short_description = "Age (s)"

    def fps_latest(self, obj):
        hb = (RunnerHeartbeat.objects
              .filter(camera=obj.camera, profile=obj.profile)
              .only("fps", "ts")
              .order_by("-ts")
              .first())
        return getattr(hb, "fps", None)

    fps_latest.short_description = "FPS"

    def mem_mb(self, obj):
        p = self._proc(obj.pid)
        if not p:
            return None
        try:
            return round(p.memory_info().rss / (1024 * 1024), 1)
        except Exception:
            return None

    mem_mb.short_description = "RSS (MB)"

    def threads(self, obj):
        p = self._proc(obj.pid)
        if not p:
            return None
        try:
            return p.num_threads()
        except Exception:
            return None

    def children(self, obj):
        p = self._proc(obj.pid)
        if not p:
            return None
        try:
            return len(p.children(recursive=True))
        except Exception:
            return None

    def cpu_affinity_display(self, obj):
        p = self._proc(obj.pid)
        if not p:
            return "—"
        try:
            return ",".join(map(str, p.cpu_affinity()))
        except Exception:
            return "—"

    cpu_affinity_display.short_description = "CPU affinity"

    def meta_pretty(self, obj):
        data = getattr(obj, "meta", None)
        if not data:
            return "—"
        try:
            pretty = json.dumps(data, indent=2, ensure_ascii=False)
        except Exception:
            pretty = str(data)
        return format_html("<pre style='white-space:pre-wrap;margin:0'>{}</pre>", pretty)

    meta_pretty.short_description = "Meta"

    def snapshot_thumb(self, obj):
        snap = Path(settings.SNAPSHOT_DIR) / f"{obj.camera_id}.jpg"
        if not snap.exists():
            return "—"
        try:
            mtime = int(snap.stat().st_mtime)
        except Exception:
            mtime = 0
        url = f"{settings.MEDIA_URL}snapshots/{obj.camera_id}.jpg?v={mtime}"
        return format_html(
            '<a href="{}" target="_blank"><img src="{}" style="height:54px;border-radius:6px;box-shadow:0 0 2px rgba(0,0,0,.25);" /></a>',
            url, url
        )

    snapshot_thumb.short_description = "Snapshot"

    def snapshot_preview(self, obj):
        snap = Path(settings.SNAPSHOT_DIR) / f"{obj.camera_id}.jpg"
        if not snap.exists():
            return "—"
        try:
            mtime = int(snap.stat().st_mtime)
        except Exception:
            mtime = 0
        url = f"{settings.MEDIA_URL}snapshots/{obj.camera_id}.jpg?v={mtime}"
        return format_html(
            '<a href="{}" target="_blank"><img src="{}" style="max-width:100%;border-radius:8px;" /></a>', url, url)

    snapshot_preview.short_description = "Snapshot"

    # def status_badge(self, obj):
    #     ts = obj.last_heartbeat
    #     # live = periodic._pid_alive(obj.pid)
    #     live = _enf._pid_alive(obj.pid)
    #     if not ts:
    #         return format_html("<span style='color:#b91c1c;font-weight:600'>Offline</span>")
    #     age = (timezone.now() - ts).total_seconds()
    #     if live and age <= ONLINE:  # <-- uses your settings-backed thresholds
    #         return format_html("<span style='color:#16a34a;font-weight:600'>Online</span>")
    #     if age > OFFLINE:
    #         return format_html("<span style='color:#b91c1c;font-weight:600'>Offline</span>")
    #     return format_html("<span style='color:#d97706;font-weight:600'>Stale</span>")

    # apps/scheduler/admin.py

    def status_badge(self, obj):
        # choose the freshest RP timestamp available
        ts = getattr(obj, "last_heartbeat", None) or getattr(obj, "last_heartbeat_at", None)
        live = _enf._pid_alive(obj.pid)
        if not ts:
            return format_html("<span style='color:#b91c1c;font-weight:600'>Offline</span>")
        age = (timezone.now() - ts).total_seconds()
        if live and age <= ONLINE:
            return format_html("<span style='color:#16a34a;font-weight:600'>Online</span>")
        if age > OFFLINE:
            return format_html("<span style='color:#b91c1c;font-weight:600'>Offline</span>")
        return format_html("<span style='color:#d97706;font-weight:600'>Stale</span>")

    status_badge.short_description = "Status"

    # ---------- actions ----------
    actions = [
        "action_stop_selected", "action_kill_selected",
        "action_restart_selected", "action_mark_dead",
        "action_purge_dead", "action_enforce_now",
    ]

    def changelist_view(self, request, extra_context=None):
        extra_context = extra_context or {}
        # Only content; styling handled in the template (dark/light aware)
        extra_context["rp_help"] = format_html(
            """
            <div class="rp-title">What is this page?</div>
            <ul class="rp-bullets">
              <li><b>Live runner processes</b> that the scheduler started for each (camera, profile).</li>
              <li>Rows are created/updated automatically by the <i>enforcer</i>; don’t add rows manually.</li>
              <li>Use the actions to <b>Stop</b>, <b>Kill group</b>, <b>Restart</b>, <b>Mark dead</b>, or <b>Enforce now</b>.</li>
            </ul>
            <div class="rp-subtitle">Status timing</div>
            <div class="rp-text">‘Online’ ≤ <code>{online}s</code>, ‘Stale’ ≤ <code>{stale}s</code>, ‘Offline’ &gt; <code>{offline}s</code>. Online also requires the OS PID to be alive.</div>
            <div class="rp-tip">Tip: if a row is stale/offline, check heartbeats and camera connectivity.</div> <br>
            """,
            online=ONLINE, stale=STALE, offline=OFFLINE,
        )
        return super().changelist_view(request, extra_context=extra_context)

    @admin.action(description="Stop selected (SIGTERM group)")
    def action_stop_selected(self, request, queryset):
        n = 0
        for rp in queryset:
            try:
                _stop(rp.pid)
                rp.status = "stopping"
                rp.save(update_fields=["status"])
                n += 1
            except Exception as e:
                self.message_user(request, f"Stop failed PID {rp.pid}: {e}", level="warning")
        self.message_user(request, f"Sent SIGTERM to {n} process(es).")

    @admin.action(description="Kill selected (SIGKILL group)")
    def action_kill_selected(self, request, queryset):
        n = 0
        for rp in queryset:
            try:
                pg = os.getpgid(rp.pid)
                os.killpg(pg, signal.SIGKILL)
                n += 1
            except Exception as e:
                self.message_user(request, f"Kill failed PID {rp.pid}: {e}", level="warning")
        self.message_user(request, f"Sent SIGKILL to {n} process(es).")

    @admin.action(description="Restart selected (stop → enforce)")
    def action_restart_selected(self, request, queryset):
        for rp in queryset:
            try:
                _stop(rp.pid)
                rp.status = "stopping"
                rp.save(update_fields=["status"])
            except Exception as e:
                self.message_user(request, f"Stop failed PID {rp.pid}: {e}", level="warning")
        result = enforce_schedules()
        messages.success(
            request,
            f"Enforced. Started {len(result.started)}, Stopped {len(result.stopped)}, "
            f"Pruned {getattr(result, 'pruned_count', 0)} dead row(s)."
        )

    @admin.action(description="Mark selected as dead")
    def action_mark_dead(self, request, queryset):
        updated = queryset.update(status="dead")
        self.message_user(request, f"Marked {updated} row(s) as dead.")

    @admin.action(description="Purge dead (>10 min)")
    def action_purge_dead(self, request, queryset):
        cut = timezone.now() - timezone.timedelta(minutes=10)
        deleted, _ = RunningProcess.objects.filter(
            status="dead"
        ).filter(Q(last_heartbeat__lt=cut) | Q(last_heartbeat__isnull=True)).delete()
        self.message_user(request, f"Purged {deleted} dead row(s).")

    @admin.action(description="Enforce schedules now")
    def action_enforce_now(self, request, queryset):
        result = enforce_schedules()
        self.message_user(request, f"Enforced. Started {len(result.started)}, Stopped {len(result.stopped)}.")

    # No manual creation of rows
    def has_add_permission(self, request):
        return False


# ============================
# RunnerHeartbeat (admin)
# ============================
class RunnerHeartbeatForm(forms.ModelForm):
    class Meta:
        model = RunnerHeartbeat
        fields = "__all__"
        help_texts = {
            "camera": "Camera reporting this heartbeat.",
            "profile": "StreamProfile used by the running process.",
            "ts": "Timestamp of this heartbeat (auto-set on update).",
            "fps": "Frames per second reported by the runner (instant or smoothed).",
            "detected": "Cumulative detections since runner start (optional).",
            "matched": "Cumulative successful recognitions since runner start (optional).",
            "latency_ms": "End-to-end pipeline latency, in milliseconds (optional).",
            "last_error": "Most recent non-fatal error seen by the runner (trimmed to 200 chars).",
        }


@admin.register(RunnerHeartbeat)
class RunnerHeartbeatAdmin(admin.ModelAdmin):
    form = RunnerHeartbeatForm
    list_display = ("camera", "profile", "ts", "fps", "target_fps", "snapshot_every", "detected", "matched",
                    "latency_ms", "last_error", "age_col", "status_col")
    list_filter = ("profile", "camera")
    search_fields = ("camera__name", "profile__name", "last_error")
    ordering = ("-ts",)
    # Heartbeats are telemetry; make them read-only.
    readonly_fields = ("camera", "profile", "ts", "fps", "detected", "matched", "latency_ms", "last_error")
    actions = ["action_purge_selected", "action_purge_stale"]
    date_hierarchy = "ts"
    list_per_page = 25

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        # Add ?all=1 to the URL to see the full time series
        if request.GET.get("all") == "1":
            return qs.select_related("camera", "profile")
        return (
            qs.annotate(
                rn=Window(
                    expression=RowNumber(),
                    partition_by=[F("camera_id"), F("profile_id")],
                    order_by=F("ts").desc(),
                )
            )
            .filter(rn=1)
            .select_related("camera", "profile")
            .order_by("camera__name", "profile__name")
        )

    # No manual creation of heartbeats
    def has_add_permission(self, request):
        return False

    # Keep delete so “Purge selected” still works
    def has_delete_permission(self, request, obj=None):
        return True

    def changelist_view(self, request, extra_context=None):
        extra_context = extra_context or {}
        extra_context["title"] = "Runner heartbeats"
        toggle = (
            "<a href='?all=1'>Show all</a>" if request.GET.get("all") != "1" else "<a href='?'>Show latest only</a>")
        extra_context["subtitle"] = mark_safe(
            f"Latest telemetry per runner. ‘Stale’ &gt; {STALE}s; ‘Online’ ≤ {ONLINE}s. &nbsp; {toggle}"
        )
        return super().changelist_view(request, extra_context=extra_context)

    def age_col(self, obj):
        secs = (timezone.now() - obj.ts).total_seconds()
        return f"{int(secs)}s"

    age_col.short_description = "Age"

    # def status_col(self, obj):
    #     secs = (timezone.now() - obj.ts).total_seconds()
    #
    #     # require LIVE process for "Online"
    #     rp = (RunningProcess.objects
    #           .filter(camera=obj.camera, profile=obj.profile)
    #           .order_by("-id")
    #           .first())
    #     # live = bool(rp and periodic._pid_alive(rp.pid))
    #     live = bool(rp and _enf._pid_alive(rp.pid))
    #
    #     # Optional: treat repeated fps==0 as "No video"
    #     # (a runner can be alive but camera powered off; it will send fps=0)
    #     if secs <= STALE and obj.fps == 0 and live:
    #         return mark_safe("<span style='color:#d97706;font-weight:600;'>No video</span>")
    #
    #     if live and secs <= ONLINE:
    #         return mark_safe("<span style='color:#16a34a;font-weight:600;'>Online</span>")
    #     if secs > OFFLINE:
    #         return mark_safe("<span style='color:#b91c1c;font-weight:600;'>Offline</span>")
    #     return mark_safe("<span style='color:#d97706;font-weight:600;'>Stale</span>")

    # in RunnerHeartbeatAdmin.status_col (apps/scheduler/admin.py)
    def status_col(self, obj):
        rp = (RunningProcess.objects
              .filter(camera=obj.camera, profile=obj.profile)
              .order_by("-id").first())
        live = bool(rp and _enf._pid_alive(rp.pid))
        # Prefer RP freshness
        ts = getattr(rp, "last_heartbeat", None) or obj.ts
        secs = (timezone.now() - ts).total_seconds()

        if secs <= STALE and obj.fps == 0 and live:
            return mark_safe("<span style='color:#d97706;font-weight:600;'>No video</span>")
        if live and secs <= ONLINE:
            return mark_safe("<span style='color:#16a34a;font-weight:600;'>Online</span>")
        if secs > OFFLINE:
            return mark_safe("<span style='color:#b91c1c;font-weight:600;'>Offline</span>")
        return mark_safe("<span style='color:#d97706;font-weight:600;'>Stale</span>")

    status_col.short_description = "Status"

    @admin.action(description="Purge selected")
    def action_purge_selected(self, request, queryset):
        count = queryset.count()
        queryset.delete()
        messages.success(request, f"Deleted {count} heartbeat(s).")

    @admin.action(description="Purge stale (> 24h)")
    def action_purge_stale(self, request, queryset):
        cutoff = timezone.now() - _dt.timedelta(hours=24)
        qs = RunnerHeartbeat.objects.filter(ts__lt=cutoff)
        count = qs.count()
        qs.delete()
        messages.info(request, f"Deleted {count} stale heartbeat(s) older than 24h.")
