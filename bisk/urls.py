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

urlpatterns = [
    # Admin-wrapped dashboard page
    path("admin/system/", admin.site.admin_view(admin_system), name="admin_system"),
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
    path("", include(("apps.attendance.urls", "attendance"), namespace="attendance")),
    # path("attendance/", include(("apps.attendance.urls", "attendance"), namespace="attendance")),

    # Dashboard (public/dash)
    path("dash/system/panel/", system_panel_partial, name="system_panel_partial"),
    path("dash/system/", system_dash, name="system_dash"),
    path("dash/system.json", system_json, name="system_json"),

    # Legacy dashboard pages
    path("dashboard/", core_views.dashboard, name="dashboard"),
    path("dashboard/system_stats/", core_views.system_stats, name="system-stats"),
    path("dashboard/cameras/", core_views.cameras_dashboard, name="cameras-dashboard"),

]

# Dev static/media
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

admin.site.site_title = "BISK Admin Portal"
admin.site.site_header = "Welcome to BISK Admin Area"
admin.site.index_title = "Dashboard Overview"
