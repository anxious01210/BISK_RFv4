from django.core.management.base import BaseCommand
from django.contrib.auth.models import Group

TARGET_GROUPS = ("api_user", "lunch_supervisor")

class Command(BaseCommand):
    help = "Mark members of api_user & lunch_supervisor as staff so they can use the Admin login page."

    def handle(self, *args, **opts):
        changed = 0
        missing = []
        for name in TARGET_GROUPS:
            try:
                g = Group.objects.get(name=name)
            except Group.DoesNotExist:
                missing.append(name)
                continue
            for u in g.user_set.all():
                if not u.is_staff:
                    u.is_staff = True
                    u.save(update_fields=["is_staff"])
                    changed += 1
                    self.stdout.write(f"✔ set is_staff=True for user: {u.username}")
        if missing:
            self.stdout.write(self.style.WARNING(f"Groups not found: {', '.join(missing)}"))
        self.stdout.write(self.style.SUCCESS(f"Done. {changed} user(s) updated."))
        self.stdout.write("Note: Only superusers or members of 'supervisor' can open /admin/ UI.")
