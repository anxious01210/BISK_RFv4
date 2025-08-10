# apps/scheduler/admin.py
from django.contrib import admin
from .models import StreamProfile, SchedulePolicy, ScheduleWindow, ScheduleException, RunningProcess


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


@admin.register(SchedulePolicy)
class SchedulePolicyAdmin(admin.ModelAdmin):
    list_display = ("name", "is_enabled")
    list_filter = ("is_enabled",)
    filter_horizontal = ("cameras",)
    inlines = [WindowInline, ExceptionInline]


@admin.register(RunningProcess)
class RunningProcessAdmin(admin.ModelAdmin):
    list_display = ("camera", "profile", "pid", "status", "started_at", "last_heartbeat")
    list_filter = ("status", "profile__script_type")
