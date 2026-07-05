"""Seed RoleType system roles.

Creates all 13 RoleType rows from ``RoleType.RoleChoices`` with
``is_system=True``. Idempotent: uses ``update_or_create`` so running
it multiple times (or on a DB that already has some roles) is safe —
existing rows are updated to ``is_system=True``; missing rows are
created.

No reverse migration: deleting system roles could break
``PersonRole`` FK references (PROTECT) and would lose the seeded
state. Rolling back this migration is a manual operation (delete the
``identity_roletype`` rows if no PersonRole references them).
"""

from django.db import migrations


def seed_roletypes(apps, schema_editor):
    RoleType = apps.get_model("identity", "RoleType")

    # Hardcoded from RoleType.RoleChoices (apps.get_model returns a
    # historical model that lacks inner classes). Keep in sync with
    # apps/identity/models.py RoleType.RoleChoices.
    system_roles = [
        ("student", "Student"),
        ("staff", "Staff"),
        ("teacher", "Teacher"),
        ("parent", "Parent"),
        ("guardian", "Guardian"),
        ("guest", "Guest"),
        ("vendor", "Vendor"),
        ("administrator", "Administrator"),
        ("finance", "Finance"),
        ("hr", "HR"),
        ("principal", "Principal"),
        ("vice_principal", "Vice Principal"),
        ("librarian", "Librarian"),
        ("nurse", "Nurse"),
    ]

    for code, name in system_roles:
        RoleType.objects.update_or_create(
            code=code,
            defaults={
                "name": name,
                "is_active": True,
                "is_system": True,
            },
        )


class Migration(migrations.Migration):
    dependencies = [
        ("identity", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(
            seed_roletypes,
            # No reverse: deleting system roles could break PersonRole
            # FK references. Use migrations.RunPython.noop if a reverse
            # is needed for testing; production rollback is manual.
            migrations.RunPython.noop,
        ),
    ]
