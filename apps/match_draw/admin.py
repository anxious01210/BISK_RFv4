from django.contrib import admin
from .models import DrawEvent, DrawTeam, DrawPairRule, DrawMatch, DrawSpinLog


class DrawTeamInline(admin.TabularInline):
    model = DrawTeam
    extra = 0


class DrawPairRuleInline(admin.TabularInline):
    model = DrawPairRule
    extra = 0
    fk_name = "draw_event"


class DrawMatchInline(admin.TabularInline):
    model = DrawMatch
    extra = 0


@admin.register(DrawEvent)
class DrawEventAdmin(admin.ModelAdmin):
    list_display = ("title", "mode", "theme", "status", "teams_count", "created_by", "created_at")
    list_filter = ("mode", "theme", "status", "created_at")
    search_fields = ("title",)
    inlines = [DrawTeamInline, DrawPairRuleInline, DrawMatchInline]


@admin.register(DrawTeam)
class DrawTeamAdmin(admin.ModelAdmin):
    list_display = ("name", "draw_event", "display_order", "is_active")
    list_filter = ("draw_event", "is_active")
    search_fields = ("name",)


@admin.register(DrawPairRule)
class DrawPairRuleAdmin(admin.ModelAdmin):
    list_display = ("draw_event", "team_a", "team_b")
    list_filter = ("draw_event",)
    search_fields = ("team_a__name", "team_b__name", "draw_event__title")


@admin.register(DrawMatch)
class DrawMatchAdmin(admin.ModelAdmin):
    list_display = ("draw_event", "match_order", "team_1", "team_2", "revealed_at")
    list_filter = ("draw_event",)
    search_fields = ("team_1__name", "team_2__name", "draw_event__title")


@admin.register(DrawSpinLog)
class DrawSpinLogAdmin(admin.ModelAdmin):
    list_display = ("draw_event", "spin_number", "selected_team", "was_forced", "created_at")
    list_filter = ("draw_event", "was_forced")
    search_fields = ("selected_team__name", "draw_event__title")