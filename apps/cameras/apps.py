from django.apps import AppConfig


class CamerasConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.cameras'
    verbose_name = "Cameras"

    def ready(self):
        from . import checks  # registers the check