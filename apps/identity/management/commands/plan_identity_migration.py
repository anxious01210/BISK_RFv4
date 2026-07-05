"""Management command: plan_identity_migration.

Exposes the read-only identity migration planning infrastructure
(:mod:`apps.identity.migration_planning`) as a Django management
command.

**This command is read-only.** It does not create or modify any
database rows. It inspects legacy ``attendance.Student`` rows and
reports what *would* be migrated.

Usage::

    # Full report (human-readable)
    python manage.py plan_identity_migration

    # JSON output
    python manage.py plan_identity_migration --json

    # Only blocked records
    python manage.py plan_identity_migration --only-blocked

    # Single student
    python manage.py plan_identity_migration --student H12345

    # Single student, JSON
    python manage.py plan_identity_migration --student H12345 --json

Exit codes:

* ``0`` — all records are ready or already migrated (no blocked).
* ``1`` — one or more records are blocked (missing data or duplicates).
* ``2`` — ``--student`` supplied but the h_code was not found.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import Optional

from django.apps import apps as django_apps
from django.core.management.base import BaseCommand

from apps.identity.migration_planning import (
    MigrationReport,
    StudentMigrationPlan,
    plan_all_student_migrations,
    plan_student_migration,
)


class Command(BaseCommand):
    help = (
        "Dry-run planning for the Student → Person + StudentProfile + "
        "PersonRole identity migration. Read-only: does not create or "
        "modify any database rows."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--json",
            action="store_true",
            default=False,
            help="Output the report as JSON instead of human-readable text.",
        )
        parser.add_argument(
            "--only-blocked",
            action="store_true",
            default=False,
            help="Only show blocked records (students with missing data or duplicates).",
        )
        parser.add_argument(
            "--student",
            type=str,
            default=None,
            metavar="H_CODE",
            help="Only report the student with the given h_code.",
        )

    def handle(self, *args, **options):
        json_output: bool = options["json"]
        only_blocked: bool = options["only_blocked"]
        student_h_code: Optional[str] = options["student"]

        # --- Single-student mode (--student H_CODE) ----------------
        if student_h_code is not None:
            self._handle_single_student(
                h_code=student_h_code,
                json_output=json_output,
            )
            return

        # --- Full report mode (default) ---------------------------
        report = plan_all_student_migrations()

        if only_blocked:
            self._output_blocked_only(
                report=report, json_output=json_output
            )
        else:
            self._output_full_report(
                report=report, json_output=json_output
            )

        # Exit with code 1 if any blocked records exist.
        # This follows the same pattern as Django's own
        # ``makemigrations --check`` (sys.exit(1) when changes are
        # detected). The output has already been written to stdout;
        # sys.exit() just sets the process exit code.
        if report.blocked > 0:
            sys.exit(1)

    # -------------------------------------------------------------------
    # Output helpers
    # -------------------------------------------------------------------

    def _output_full_report(
        self, *, report: MigrationReport, json_output: bool
    ) -> None:
        if json_output:
            self.stdout.write(json.dumps(self._report_to_dict(report), indent=2))
        else:
            self.stdout.write(self.style.MIGRATE_HEADING(
                "Identity Migration Plan (dry-run)"
            ))
            self.stdout.write("")
            self.stdout.write(f"  Total students:     {report.total}")
            self.stdout.write(f"  Ready to migrate:   {report.ready}")
            self.stdout.write(f"  Already migrated:   {report.already_migrated}")
            self.stdout.write(f"  Blocked:            {report.blocked}")
            self.stdout.write("")

            if report.blocked_plans:
                self.stdout.write(self.style.WARNING("Blocked records:"))
                for plan in report.blocked_plans:
                    self.stdout.write(
                        f"  {plan.h_code}: {'; '.join(plan.issues)}"
                    )
                self.stdout.write("")

            if report.skipped_plans:
                self.stdout.write(self.style.SUCCESS("Already migrated:"))
                for plan in report.skipped_plans:
                    self.stdout.write(f"  {plan.h_code}")
                self.stdout.write("")

            if report.ready_plans and not report.blocked_plans:
                self.stdout.write(self.style.SUCCESS(
                    f"All {report.ready} ready student(s) can be migrated."
                ))
            elif report.ready_plans:
                self.stdout.write(
                    f"  ({report.ready} student(s) ready to migrate)"
                )

    def _output_blocked_only(
        self, *, report: MigrationReport, json_output: bool
    ) -> None:
        blocked = report.blocked_plans

        if json_output:
            data = {
                "total": report.total,
                "blocked": len(blocked),
                "blocked_plans": [asdict(p) for p in blocked],
            }
            self.stdout.write(json.dumps(data, indent=2))
        else:
            self.stdout.write(self.style.WARNING(
                f"Blocked records ({len(blocked)} of {report.total}):"
            ))
            self.stdout.write("")
            for plan in blocked:
                self.stdout.write(
                    f"  {plan.h_code}: {'; '.join(plan.issues)}"
                )

        if blocked:
            sys.exit(1)

    def _handle_single_student(
        self, *, h_code: str, json_output: bool
    ) -> None:
        Student = django_apps.get_model("attendance", "Student")
        try:
            student = Student.objects.get(h_code=h_code)
        except Student.DoesNotExist:
            if json_output:
                self.stdout.write(json.dumps({
                    "error": "student_not_found",
                    "h_code": h_code,
                }, indent=2))
            else:
                self.stdout.write(self.style.ERROR(
                    f"Student with h_code '{h_code}' not found."
                ))
            sys.exit(2)  # unknown student
            return  # unreachable; keeps linters happy

        plan = plan_student_migration(student)

        if json_output:
            self.stdout.write(json.dumps(asdict(plan), indent=2, default=str))
        else:
            self.stdout.write(self.style.MIGRATE_HEADING(
                f"Migration plan for {h_code}"
            ))
            self.stdout.write("")
            self.stdout.write(f"  Legacy student ID:  {plan.legacy_student_id}")
            self.stdout.write(f"  Person code:        {plan.person_code}")
            self.stdout.write(f"  First name:         {plan.first_name}")
            self.stdout.write(f"  Last name:          {plan.last_name}")
            self.stdout.write(f"  Gender:             {plan.gender or '(none)'}")
            self.stdout.write(f"  Grade:              {plan.grade or '(none)'}")
            self.stdout.write(f"  Has meal:           {plan.has_meal}")
            self.stdout.write(f"  Has bus:            {plan.has_bus}")
            self.stdout.write(f"  Is active:          {plan.is_active}")
            self.stdout.write(f"  Profile code:       {plan.student_profile_code}")
            self.stdout.write(f"  Role type:          {plan.role_type_code}")
            self.stdout.write(f"  Already migrated:   {plan.already_migrated}")
            self.stdout.write("")

            if plan.issues:
                self.stdout.write(self.style.WARNING("Issues:"))
                for issue in plan.issues:
                    self.stdout.write(f"  - {issue}")
            elif plan.already_migrated:
                self.stdout.write(self.style.SUCCESS(
                    "Already migrated — no action needed."
                ))
            else:
                self.stdout.write(self.style.SUCCESS(
                    "Ready to migrate."
                ))

        # Exit with code 1 if blocked (issues).
        if plan.issues:
            sys.exit(1)

    # -------------------------------------------------------------------
    # Serialization helper
    # -------------------------------------------------------------------

    def _report_to_dict(self, report: MigrationReport) -> dict:
        return {
            "total": report.total,
            "ready": report.ready,
            "already_migrated": report.already_migrated,
            "blocked": report.blocked,
            "plans": [asdict(p) for p in report.plans],
        }
