"""
URL configuration for bisk project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# bisk/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from apps.scheduler import views as sched_views

from apps.scheduler import api as sched_api
from bisk import views as core_views
from apps.scheduler.views import system_dash, system_json, admin_system, system_panel_partial
from django.views.generic import RedirectView
from django.shortcuts import redirect
from types import MethodType
from django.contrib.auth import logout as auth_logout


# --- new: portal logout that always clears session then returns to "/" ---
def portal_logout(request):
    if request.method == "POST":
        auth_logout(request)
    else:
        # if someone GETs it, still log them out for convenience
        auth_logout(request)
    return redirect("/")


# Only allow superusers OR members of "supervisor" to open /admin/
def _admin_has_permission(self, request):
    u = request.user
    return u.is_active and (u.is_staff or u.is_superuser or u.groups.filter(name="supervisor").exists())


admin.site.has_permission = MethodType(_admin_has_permission, admin.site)


def root_redirect(request):
    u = request.user
    # Anonymous → Admin login, but come back to "/" so we can route by role after auth
    if not u.is_authenticated:
        # return redirect(f"{settings.LOGIN_URL}?next=/")
        return redirect("/home/")
    # 1) staff/superuser/supervisor → admin
    if u.is_superuser or u.groups.filter(name="supervisor").exists():
        return redirect("/admin/")
    # 2) lunch supervisors (non-staff or staff) → lunch dashboard
    if u.groups.filter(name="lunch_supervisor").exists():
        return redirect("/dash/lunch/")
    # 3) api users → records endpoint
    if u.groups.filter(name="api_user").exists():
        return redirect("/api/attendance/records/")
    # 4) fallback
    # return redirect("/dash/lunch/")
    return redirect("/home/")


urlpatterns = [
    # Role-aware homepage first (must appear before the "" include below)
    path("", root_redirect),  # role-aware homepage
    # Admin-wrapped dashboard page
    path("admin/system/", admin.site.admin_view(admin_system), name="admin_system"),
    path("api-auth/", include("rest_framework.urls")),  # DRF session login/logout
    path("logout/", portal_logout, name="portal_logout"),
    path("home/", core_views.portal_home, name="portal_home"),
    path("", include(("apps.attendance.urls", "attendance"), namespace="attendance")),
    # path("api/attendance/records/", include(("apps.attendance.urls", "attendance"), namespace="attendance")),
    # path("admin/system/enforce-now/", sched_api.enforce_now, name="enforce_now") if hasattr(sched_api,
    #                                                                                         "enforce_now") else path(
    #     "admin/system/enforce-now/", admin_system, name="enforce_now"),  # keep your existing import if different
    path("admin/system/enforce-now/", sched_views.enforce_now, name="enforce_now"),

    path("admin/", admin.site.urls),

    # Runner heartbeat
    path("api/runner/heartbeat/", sched_api.heartbeat, name="runner-heartbeat"),

    # Attendance app (namespaced). This includes:
    #  - /api/attendance/ingest/   (attendance:attendance_ingest)
    #  - /api/attendance/records/  (DRF router in apps.attendance.urls)
    # path("", include(("apps.attendance.urls", "attendance"), namespace="attendance")),

    # Dashboard (public/dash)
    path("dash/system/panel/", system_panel_partial, name="system_panel_partial"),
    path("dash/system/", system_dash, name="system_dash"),
    path("dash/system.json", system_json, name="system_json"),

    # Legacy dashboard pages
    path("dashboard/", core_views.dashboard, name="dashboard"),
    path("dashboard/system_stats/", core_views.system_stats, name="system-stats"),
    path("dashboard/cameras/", core_views.cameras_dashboard, name="cameras-dashboard"),
    # path("attendance/", include(("apps.attendance.urls", "attendance"), namespace="attendance")),
    path("accounts/", include("django.contrib.auth.urls")),
]

# urlpatterns = [
#     path("api/", include(router.urls)),          # API lives here
#     path("", RedirectView.as_view(               # real homepage (pick what you want)
#         url="/admin/", permanent=False
#     )),
# ]

# Dev static/media
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

admin.site.site_title = "BISK Admin Portal"
admin.site.site_header = "Welcome to BISK Admin Area"
admin.site.index_title = "Dashboard Overview"
