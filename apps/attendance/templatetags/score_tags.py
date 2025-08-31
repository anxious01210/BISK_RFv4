from django import template
from django.conf import settings
from django.utils.safestring import mark_safe

register = template.Library()


def _bands_and_colors():
    bands = getattr(settings, "ATTENDANCE_SCORE_BANDS", {}) or {}
    g = float(bands.get("green_min", 0.80))
    y = float(bands.get("yellow_min", 0.65))
    o = float(bands.get("orange_min", 0.50))
    c_green = getattr(settings, "ATTENDANCE_COLOR_GREEN", "lime")
    c_yellow = getattr(settings, "ATTENDANCE_COLOR_YELLOW", "#facc15")
    c_orange = getattr(settings, "ATTENDANCE_COLOR_ORANGE", "orange")
    c_red = getattr(settings, "ATTENDANCE_COLOR_RED", "red")
    return (g, y, o, c_green, c_yellow, c_orange, c_red)


@register.filter
def colored_score(value):
    """Return <span> with admin-consistent color for a raw 0..1 score."""
    try:
        s = float(value)
    except Exception:
        return "-"
    g, y, o, c_green, c_yellow, c_orange, c_red = _bands_and_colors()
    color = c_green if s >= g else c_yellow if s >= y else c_orange if s >= o else c_red
    html = f'<span style="color:{color};font-weight:600;">{s:.2f}</span>'
    return mark_safe(html)
