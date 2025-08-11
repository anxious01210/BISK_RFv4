# apps/scheduler/admin.py
from django.urls import path, reverse
from django.shortcuts import redirect
from django.contrib import admin, messages
from django.utils.safestring import mark_safe
from .models import StreamProfile, SchedulePolicy, ScheduleWindow, ScheduleException, RunningProcess
from .services import enforce_schedules


@admin.register(StreamProfile)
class StreamProfileAdmin(admin.ModelAdmin):
    list_display = ("name", "script_type", "fps", "detection_set", "is_active")
    list_filter = ("script_type", "detection_set", "is_active")
    search_fields = ("name",)


class WindowInline(admin.TabularInline):
    model = ScheduleWindow
    extra = 1


class ExceptionInline(admin.TabularInline):
    model = ScheduleException
    extra = 0


# @admin.register(SchedulePolicy)
# class SchedulePolicyAdmin(admin.ModelAdmin):
#     list_display = ("name", "is_enabled")
#     list_filter = ("is_enabled",)
#     filter_horizontal = ("cameras",)
#     inlines = [WindowInline, ExceptionInline]

@admin.register(SchedulePolicy)
class SchedulePolicyAdmin(admin.ModelAdmin):
    list_display = ("name", "is_enabled")
    list_filter = ("is_enabled",)
    filter_horizontal = ("cameras",)
    inlines = [WindowInline, ExceptionInline]
    actions = ["action_enforce_now"]

    # 1) keep the row-based action
    @admin.action(description="Enforce schedules now (selected policies)")
    def action_enforce_now(self, request, queryset):
        result = enforce_schedules(policies=list(queryset))
        msg = (
                      f"Enforced. Started: {len(result.started)}, Stopped: {len(result.stopped)}<br>" +
                      "<br>".join(
                          [f"➕ {c}/{p} (pid={pid})" for c, p, pid in result.started] +
                          [f"🛑 {c}/{p} (pid={pid})" for c, p, pid in result.stopped]
                      )
              ) or "No changes."
        messages.info(request, mark_safe(msg))

    # 2) add a custom URL for the top-right button
    def get_urls(self):
        urls = super().get_urls()
        extra = [
            path("enforce-now/", self.admin_site.admin_view(self.enforce_now_view), name="scheduler_enforce_now")
        ]
        return extra + urls

    def enforce_now_view(self, request):
        result = enforce_schedules()
        messages.success(request, f"Enforced. Started {len(result.started)}, Stopped {len(result.stopped)}.")
        return redirect("admin:scheduler_schedulepolicy_changelist")


@admin.register(RunningProcess)
class RunningProcessAdmin(admin.ModelAdmin):
    list_display = ("camera", "profile", "pid", "status", "started_at", "last_heartbeat")
    list_filter = ("status", "profile__script_type")
