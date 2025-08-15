# python manage.py stop_runners            # stop everything
# python manage.py stop_runners --camera 2 # stop one camera

# 4) Deploy / reboot checklist
# python manage.py stop_runners
# pgrep -af "recognize_ffmpeg\.py|ffmpeg" → should be empty
# Stop Django (gunicorn/uvicorn/devserver)
# Pull/upgrade/migrate/etc.
# Start Django
# Run Enforce now (or your periodic enforcer) to bring RPs back exactly per schedule.

from django.core.management.base import BaseCommand
from apps.scheduler.models import RunningProcess
from apps.scheduler.services import _stop, _pid_alive


class Command(BaseCommand):
    help = "Stop all running recognize_ffmpeg runners (SIGTERM→SIGKILL) and mark rows dead."

    def add_arguments(self, parser):
        parser.add_argument("--camera", type=int, help="Only stop a specific camera_id")
        parser.add_argument("--profile", type=int, help="Only stop a specific profile_id")

    def handle(self, *args, **opts):
        qs = RunningProcess.objects.all()
        if opts.get("camera") is not None:
            qs = qs.filter(camera_id=opts["camera"])
        if opts.get("profile") is not None:
            qs = qs.filter(profile_id=opts["profile"])

        stopped = 0
        skipped = 0
        for rp in qs:
            if _pid_alive(rp.pid):
                if _stop(rp.pid):
                    rp.status = "dead"
                    rp.save(update_fields=["status"])
                    stopped += 1
            else:
                skipped += 1

        self.stdout.write(self.style.SUCCESS(
            f"Stopped {stopped} runner(s); skipped {skipped} already-dead."
        ))
