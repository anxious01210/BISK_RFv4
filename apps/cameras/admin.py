# apps/cameras/admin.py
from django.contrib import admin
from .models import Camera


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ("name", "location", "script_type_default", "scan_station", "is_active")
    search_fields = ("name", "location")
    list_filter = ("script_type_default", "scan_station", "is_active")
