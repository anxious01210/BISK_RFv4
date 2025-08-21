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
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from rest_framework.routers import DefaultRouter
from apps.attendance.api import AttendanceRecordViewSet  # will exist after step 5
from apps.scheduler import api as sched_api
from bisk import views as core_views
from django.views.generic import RedirectView
from django.urls import path, include
from django.contrib import admin
from apps.scheduler.views import system_dash, system_json, admin_system, system_panel_partial
from apps.scheduler import views as sched_views

router = DefaultRouter()
router.register(r"attendance/records", AttendanceRecordViewSet, basename="attendance-records")

urlpatterns = [
    path("admin/system/", admin.site.admin_view(admin_system), name="admin_system"),
    path("admin/system/enforce-now/", sched_views.enforce_now, name="enforce_now"),
    path('admin/', admin.site.urls),
    path("api/", include(router.urls)),
    path("api/runner/heartbeat/", sched_api.heartbeat, name="runner-heartbeat"),
    path("dash/system/panel/", system_panel_partial, name="system_panel_partial"),
    path("dash/system/", system_dash, name="system_dash"),
    path("dashboard/", core_views.dashboard, name="dashboard"),
    path("dashboard/system_stats/", core_views.system_stats, name="system-stats"),
    path("dashboard/cameras/", core_views.cameras_dashboard, name="cameras-dashboard"),
    path("dash/system.json", system_json, name="system_json"),
    # Admin-wrapped dashboard
    path("admin/system/", admin.site.admin_view(admin_system), name="admin_system"),
    # legacy dashboard routes → redirect to new system page/json
    # path("dashboard/", RedirectView.as_view(pattern_name="system_dash", permanent=False)),
    # path("dashboard/cameras/", RedirectView.as_view(pattern_name="system_dash", permanent=False)),
    # path("dashboard/system_stats/", RedirectView.as_view(pattern_name="system_json", permanent=False)),

]
# bisk/urls.py


# ✅ Serve media and static files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

admin.site.site_title = "BISK Admin Portal"
admin.site.site_header = "Welcome to BISK Admin Area"
admin.site.index_title = "Dashboard Overview"
