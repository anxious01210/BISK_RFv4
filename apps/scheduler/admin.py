# apps/scheduler/admin.py
from django.urls import path
from django.shortcuts import redirect
from django.contrib import admin, messages
from django.utils.safestring import mark_safe
from django import forms
from django.forms.models import BaseInlineFormSet
from django.core.exceptions import ValidationError

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

# Status thresholds (seconds)
ONLINE = getattr(settings, "HEARTBEAT_ONLINE_SEC", max(15, int(getattr(settings, "HEARTBEAT_INTERVAL_SEC", 10) * 1.5)))
STALE = getattr(settings, "HEARTBEAT_STALE_SEC", 45)
OFFLINE = getattr(settings, "HEARTBEAT_OFFLINE_SEC", 120)

admin.site.empty_value_display = "—"


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
        messages.success(request, f"Enforced. Started {len(result.started)}, Stopped {len(result.stopped)}.")
        return redirect("admin:scheduler_schedulepolicy_changelist")


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
            "last_heartbeat": "Updated by incoming runner heartbeats (if implemented).",
            "meta": "Optional JSON sent by the runner (e.g., build info).",
        }


@admin.register(RunningProcess)
class RunningProcessAdmin(admin.ModelAdmin):
    form = RunningProcessForm

    # Show useful, navigable columns
    list_display = (
        "camera_link", "profile_link", "pid",
        "status_col", "started_at", "last_heartbeat",
        "age_col",
    )
    list_filter = ("status", "profile__script_type", "profile")
    search_fields = ("camera__name", "profile__name", "pid")
    ordering = ("-started_at",)

    # System-managed: read-only in the form
    readonly_fields = ("camera", "profile", "pid", "status", "started_at", "last_heartbeat", "meta")

    # Don’t allow manual creation
    def has_add_permission(self, request):
        return False

    # Helpful description on the change form
    fieldsets = (
        (None, {
            "fields": ("camera", "profile", "pid", "status", "started_at", "last_heartbeat", "meta"),
            "description": format_html(
                "<div style='margin:4px 0 8px'>"
                "<b>What is this?</b> A live record of a runner process the scheduler started.<br>"
                "<b>How is it created?</b> Automatically by the enforcer when a schedule window is active.<br>"
                "<b>How to use?</b> Use actions to Stop/Restart. Do not add rows manually.<br>"
                "<b>Tip:</b> If status is <i>stale</i> or <i>offline</i>, check heartbeats and camera connectivity."
                "</div>"
            )
        }),
    )

    # Changelist subtitle
    def changelist_view(self, request, extra_context=None):
        extra_context = extra_context or {}
        extra_context["title"] = "Running processes"
        extra_context["subtitle"] = mark_safe(
            f"Live runner processes. Status derives from last heartbeat age: "
            f"Offline &gt; {OFFLINE}s, Stale &gt; {STALE}s, Online ≤ {ONLINE}s."
        )
        return super().changelist_view(request, extra_context=extra_context)

    # Computed columns
    def camera_link(self, obj):
        url = reverse("admin:cameras_camera_change", args=[obj.camera_id])
        return format_html('<a href="{}">{}</a>', url, obj.camera.name)

    camera_link.short_description = "Camera"

    def profile_link(self, obj):
        url = reverse("admin:scheduler_streamprofile_change", args=[obj.profile_id])
        return format_html('<a href="{}">{}</a>', url, obj.profile.name)

    profile_link.short_description = "Profile"

    def age_col(self, obj):
        if not obj.started_at:
            return "—"
        secs = int((timezone.now() - obj.started_at).total_seconds())
        return f"{secs}s"

    age_col.short_description = "Age"

    def status_col(self, obj):
        # Decorate status using heartbeats age if available
        label = obj.status or "—"
        if obj.last_heartbeat:
            secs = (timezone.now() - obj.last_heartbeat).total_seconds()
            if secs > OFFLINE:
                return format_html("<span style='color:#b33;font-weight:600;'>Offline</span>")
            if secs > STALE:
                return format_html("<span style='color:#d88;font-weight:600;'>Stale</span>")
            if secs <= ONLINE:
                return format_html("<span style='color:#2a7;font-weight:600;'>Online</span>")
        return label

    status_col.short_description = "Status"

    # Actions: Stop, Restart, Mark dead, Enforce now
    actions = ["action_stop_selected", "action_restart_selected", "action_mark_dead", "action_enforce_now"]

    @admin.action(description="Stop selected (SIGTERM group)")
    def action_stop_selected(self, request, queryset):
        stopped = 0
        for rp in queryset:
            try:
                _stop(rp.pid)  # best-effort SIGTERM to the process group (Linux)
                rp.status = "stopping"
                rp.save(update_fields=["status"])
                stopped += 1
            except Exception as e:
                messages.warning(request, f"Could not stop PID {rp.pid}: {e}")
        messages.success(request, f"Sent SIGTERM to {stopped} process(es).")

    @admin.action(description="Restart selected (stop, then enforce now)")
    def action_restart_selected(self, request, queryset):
        # Stop first
        for rp in queryset:
            try:
                _stop(rp.pid)
                rp.status = "stopping"
                rp.save(update_fields=["status"])
            except Exception as e:
                messages.warning(request, f"Could not stop PID {rp.pid}: {e}")
        # Reconcile immediately (will start again if still desired)
        result = enforce_schedules()
        messages.info(
            request,
            f"Restart triggered. Enforced: started {len(result.started)}, stopped {len(result.stopped)}."
        )

    @admin.action(description="Mark selected as dead")
    def action_mark_dead(self, request, queryset):
        updated = queryset.update(status="dead")
        messages.info(request, f"Marked {updated} process(es) as dead.")

    @admin.action(description="Enforce schedules now (all policies)")
    def action_enforce_now(self, request, queryset):
        result = enforce_schedules()
        messages.success(
            request,
            f"Enforced. Started {len(result.started)}, Stopped {len(result.stopped)}."
        )


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
    list_display = ("camera", "profile", "ts", "fps", "detected", "matched", "latency_ms", "last_error", "age_col",
                    "status_col")
    list_filter = ("profile", "camera")
    search_fields = ("camera__name", "profile__name", "last_error")
    ordering = ("-ts",)
    # Heartbeats are telemetry; make them read-only.
    readonly_fields = ("camera", "profile", "ts", "fps", "detected", "matched", "latency_ms", "last_error")
    actions = ["action_purge_selected", "action_purge_stale"]

    # No manual creation of heartbeats
    def has_add_permission(self, request):
        return False

    # Keep delete so “Purge selected” still works
    def has_delete_permission(self, request, obj=None):
        return True

    def changelist_view(self, request, extra_context=None):
        extra_context = extra_context or {}
        extra_context["title"] = "Runner heartbeats"
        extra_context["subtitle"] = mark_safe(
            f"Latest telemetry per runner. ‘Stale’ &gt; {STALE}s; ‘Online’ ≤ {ONLINE}s."
        )
        return super().changelist_view(request, extra_context=extra_context)

    def age_col(self, obj):
        secs = (timezone.now() - obj.ts).total_seconds()
        return f"{int(secs)}s"

    age_col.short_description = "Age"

    def status_col(self, obj):
        secs = (timezone.now() - obj.ts).total_seconds()
        if secs > OFFLINE:
            return mark_safe("<span style='color:#b33;font-weight:600;'>Offline</span>")
        if secs > STALE:
            return mark_safe("<span style='color:#d88;font-weight:600;'>Stale</span>")
        return mark_safe("<span style='color:#2a7;font-weight:600;'>Online</span>")

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
