"""Management command: execute_identity_migration.

Thin CLI wrapper around the identity migration execution engine
(:mod:`apps.identity.migration_execution`). All business logic lives
in the service layer; this command only parses arguments, calls the
appropriate service function, formats the output, and sets the exit
code.

Usage::

    # Dry-run (no writes — uses planning infrastructure)
    python manage.py execute_identity_migration --dry-run

    # Execute all ready students (requires --yes)
    python manage.py execute_identity_migration --yes

    # Execute one student
    python manage.py execute_identity_migration --student H12345 --yes

    # Limit to N students
    python manage.py execute_identity_migration --limit 50 --yes

    # JSON output
    python manage.py execute_identity_migration --yes --json

Exit codes:

* ``0`` — success (all migrated or already migrated, no failures).
* ``1`` — one or more students failed.
* ``2`` — ``--student`` h_code not found.
* ``3`` — ``--yes`` not supplied for a write operation.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from typing import Optional

from django.apps import apps as django_apps
from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand

from apps.identity.migration_execution import (
    BulkMigrationResult,
    MigrationExecutionResult,
    migrate_all_students,
    migrate_student,
)
from apps.identity.migration_planning import (
    plan_all_student_migrations,
    plan_student_migration,
)


class Command(BaseCommand):
    help = (
        "Execute the Student → Person + StudentProfile + PersonRole "
        "identity migration. Use --dry-run for a safe preview. "
        "Use --yes to confirm write operations."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="Plan only — no writes. Uses the planning infrastructure.",
        )
        parser.add_argument(
            "--student",
            type=str,
            default=None,
            metavar="H_CODE",
            help="Execute (or plan) only the student with the given h_code.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            metavar="N",
            help="Limit bulk migration to N students (passed to migrate_all_students).",
        )
        parser.add_argument(
            "--yes",
            action="store_true",
            default=False,
            help="Confirm write operations. Required for non-dry-run execution.",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            default=False,
            help="Output the summary as JSON.",
        )

    def handle(self, *args, **options):
        dry_run: bool = options["dry_run"]
        student_h_code: Optional[str] = options["student"]
        limit: Optional[int] = options["limit"]
        confirmed: bool = options["yes"]
        json_output: bool = options["json"]

        start = time.monotonic()

        # --- Dry-run mode (no writes) -------------------------------
        if dry_run:
            return self._handle_dry_run(
                student_h_code=student_h_code,
                json_output=json_output,
                start=start,
            )

        # --- Write mode: confirmation check ------------------------
        if not confirmed:
            if json_output:
                self.stdout.write(json.dumps({
                    "error": "confirmation_required",
                    "message": "Use --yes to confirm write operations.",
                }, indent=2))
            else:
                self.stdout.write(self.style.ERROR(
                    "This command will modify the database. "
                    "Use --yes to confirm."
                ))
            sys.exit(3)

        # --- Write mode: single student ----------------------------
        if student_h_code is not None:
            return self._handle_single_student(
                h_code=student_h_code,
                json_output=json_output,
                start=start,
            )

        # --- Write mode: bulk migration ----------------------------
        return self._handle_bulk(
            limit=limit,
            json_output=json_output,
            start=start,
        )

    # -------------------------------------------------------------------
    # Dry-run handler (reuses planning — zero writes)
    # -------------------------------------------------------------------

    def _handle_dry_run(self, *, student_h_code, json_output, start):
        if student_h_code is not None:
            Student = django_apps.get_model("attendance", "Student")
            try:
                student = Student.objects.get(h_code=student_h_code)
            except Student.DoesNotExist:
                if json_output:
                    self.stdout.write(json.dumps({
                        "error": "student_not_found",
                        "h_code": student_h_code,
                    }, indent=2))
                else:
                    self.stdout.write(self.style.ERROR(
                        f"Student with h_code '{student_h_code}' not found."
                    ))
                sys.exit(2)

            plan = plan_student_migration(student)
            elapsed = time.monotonic() - start

            if json_output:
                self.stdout.write(json.dumps(
                    {**asdict(plan), "elapsed_seconds": round(elapsed, 3)},
                    indent=2, default=str,
                ))
            else:
                self._print_plan_text(plan, elapsed)
            return

        # Full dry-run report.
        report = plan_all_student_migrations()
        elapsed = time.monotonic() - start

        if json_output:
            self.stdout.write(json.dumps({
                "total": report.total,
                "ready": report.ready,
                "already_migrated": report.already_migrated,
                "blocked": report.blocked,
                "elapsed_seconds": round(elapsed, 3),
                "dry_run": True,
            }, indent=2))
        else:
            self.stdout.write(self.style.MIGRATE_HEADING(
                "Identity Migration Plan (dry-run — no writes)"
            ))
            self.stdout.write("")
            self.stdout.write(f"  Total students:     {report.total}")
            self.stdout.write(f"  Ready to migrate:   {report.ready}")
            self.stdout.write(f"  Already migrated:   {report.already_migrated}")
            self.stdout.write(f"  Blocked:            {report.blocked}")
            self.stdout.write(f"  Elapsed:            {elapsed:.3f}s")

            if report.blocked_plans:
                self.stdout.write("")
                self.stdout.write(self.style.WARNING("Blocked records:"))
                for p in report.blocked_plans:
                    self.stdout.write(f"  {p.h_code}: {'; '.join(p.issues)}")

        if report.blocked > 0:
            sys.exit(1)

    # -------------------------------------------------------------------
    # Single-student execution
    # -------------------------------------------------------------------

    def _handle_single_student(self, *, h_code, json_output, start):
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
            sys.exit(2)

        try:
            result = migrate_student(student)
        except ValidationError as exc:
            elapsed = time.monotonic() - start
            if json_output:
                self.stdout.write(json.dumps({
                    "h_code": h_code,
                    "failed": True,
                    "error": str(exc),
                    "elapsed_seconds": round(elapsed, 3),
                }, indent=2))
            else:
                self.stdout.write(self.style.ERROR(
                    f"FAILED {h_code}: {exc}"
                ))
            sys.exit(1)

        elapsed = time.monotonic() - start
        self._print_single_result(result, json_output, elapsed)

    # -------------------------------------------------------------------
    # Bulk execution
    # -------------------------------------------------------------------

    def _handle_bulk(self, *, limit, json_output, start):
        try:
            result = migrate_all_students(limit=limit)
        except ValidationError as exc:
            elapsed = time.monotonic() - start
            if json_output:
                self.stdout.write(json.dumps({
                    "error": "pre_flight_failure",
                    "message": str(exc),
                    "elapsed_seconds": round(elapsed, 3),
                }, indent=2))
            else:
                self.stdout.write(self.style.ERROR(
                    f"Pre-flight check failed: {exc}"
                ))
            sys.exit(1)

        elapsed = time.monotonic() - start
        self._print_bulk_result(result, json_output, elapsed)

        if result.failed > 0:
            sys.exit(1)

    # -------------------------------------------------------------------
    # Output helpers
    # -------------------------------------------------------------------

    def _print_plan_text(self, plan, elapsed):
        self.stdout.write(self.style.MIGRATE_HEADING(
            f"Dry-run plan for {plan.h_code}"
        ))
        self.stdout.write("")
        self.stdout.write(f"  Person code:        {plan.person_code}")
        self.stdout.write(f"  First name:         {plan.first_name}")
        self.stdout.write(f"  Last name:          {plan.last_name}")
        self.stdout.write(f"  Already migrated:   {plan.already_migrated}")
        self.stdout.write(f"  Elapsed:            {elapsed:.3f}s")
        if plan.issues:
            self.stdout.write(self.style.WARNING("Issues:"))
            for issue in plan.issues:
                self.stdout.write(f"  - {issue}")

    def _print_single_result(self, result: MigrationExecutionResult, json_output, elapsed):
        if json_output:
            self.stdout.write(json.dumps({
                "h_code": result.person.code,
                "already_migrated": result.already_migrated,
                "created_person": result.created_person,
                "created_profile": result.created_profile,
                "created_role": result.created_role,
                "elapsed_seconds": round(elapsed, 3),
            }, indent=2))
        else:
            if result.already_migrated:
                self.stdout.write(self.style.SUCCESS(
                    f"{result.person.code} — already migrated ({elapsed:.3f}s)"
                ))
            else:
                self.stdout.write(self.style.SUCCESS(
                    f"{result.person.code} — migrated "
                    f"(person={result.created_person}, "
                    f"profile={result.created_profile}, "
                    f"role={result.created_role}) "
                    f"({elapsed:.3f}s)"
                ))

    def _print_bulk_result(self, result: BulkMigrationResult, json_output, elapsed):
        if json_output:
            self.stdout.write(json.dumps({
                "total_students": result.total_students,
                "migrated": result.migrated,
                "already_migrated": result.already_migrated,
                "failed": result.failed,
                "elapsed_seconds": round(elapsed, 3),
                "errors": [
                    {"h_code": e.h_code, "error": e.error}
                    for e in result.errors
                ],
            }, indent=2))
        else:
            self.stdout.write(self.style.MIGRATE_HEADING(
                "Identity Migration Complete"
            ))
            self.stdout.write("")
            self.stdout.write(f"  Total students:     {result.total_students}")
            self.stdout.write(f"  Migrated:           {result.migrated}")
            self.stdout.write(f"  Already migrated:   {result.already_migrated}")
            self.stdout.write(f"  Failed:             {result.failed}")
            self.stdout.write(f"  Elapsed:            {elapsed:.3f}s")

            if result.errors:
                self.stdout.write("")
                self.stdout.write(self.style.WARNING("Failed students:"))
                for err in result.errors:
                    self.stdout.write(f"  {err.h_code}: {err.error}")
