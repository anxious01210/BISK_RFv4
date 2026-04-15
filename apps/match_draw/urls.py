from django.urls import path
from . import views

app_name = "match_draw"

urlpatterns = [
    path("", views.draw_home, name="home"),
    path("<int:event_id>/", views.draw_detail, name="detail"),
    path("<int:event_id>/start/", views.start_draw, name="start"),
    path("<int:event_id>/live/", views.live_draw, name="live"),
    path("<int:event_id>/live/data/", views.live_draw_data, name="live_data"),
]