# apps/scheduler/signals.py
def start_scheduler_after_migrate(**kwargs):
    from .services.scheduler import start_background_scheduler
    start_background_scheduler()
