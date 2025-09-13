from django import template
from django.conf import settings
from django.utils.safestring import mark_safe

register = template.Library()


def _bands_and_colors():
    bands = getattr(settings, "ATTENDANCE_SCORE_BANDS", {}) or {}
    b = float(bands.get("blue_min", 0.90))
    g = float(bands.get("green_min", 0.80))
    y = float(bands.get("yellow_min", 0.65))
    o = float(bands.get("orange_min", 0.50))
    c_blue = getattr(settings, "ATTENDANCE_COLOR_BLUE", "#2196F3")
    c_green = getattr(settings, "ATTENDANCE_COLOR_GREEN", "lime")
    c_yellow = getattr(settings, "ATTENDANCE_COLOR_YELLOW", "#facc15")
    c_orange = getattr(settings, "ATTENDANCE_COLOR_ORANGE", "orange")
    c_red = getattr(settings, "ATTENDANCE_COLOR_RED", "red")
    return (b, g, y, o, c_blue, c_green, c_yellow, c_orange, c_red)


@register.filter
def colored_score(value):
    """Return <span> with admin-consistent color for a raw 0..1 score."""
    try:
        s = float(value)
    except Exception:
        return "-"

    b, g, y, o, c_blue, c_green, c_yellow, c_orange, c_red = _bands_and_colors()

    if s >= b:
        color = c_blue
    elif s >= g:
        color = c_green
    elif s >= y:
        color = c_yellow
    elif s >= o:
        color = c_orange
    else:
        color = c_red

    html = f'<span style="color:{color};font-weight:600;">{s:.2f}</span>'
    return mark_safe(html)
