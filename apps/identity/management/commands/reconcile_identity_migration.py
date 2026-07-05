"""Management command: reconcile_identity_migration.

Thin read-only CLI wrapper around
:func:`apps.identity.migration_reconciliation.reconcile_all`.

Verifies that legacy ``attendance.Student`` rows are correctly
represented in the identity domain (``Person`` + ``StudentProfile`` +
``PersonRole``). Reports any discrepancies (unmigrated students,
missing Person/PersonRole, code mismatches, duplicates, orphans).

**This command is read-only.** It does not create or modify any
database rows.

Usage::

    # Full reconciliation report (human-readable)
    python manage.py reconcile_identity_migration

    # JSON output
    python manage.py reconcile_identity_migration --json

    # Only show rows with issues
    python manage.py reconcile_identity_migration --only-issues

Exit codes:

* ``0`` — no issues found (all students migrated correctly).
* ``1`` — one or more issues found (unmigrated, missing data, code
  mismatches, duplicates, orphans).
"""

from __future__ import annotations

import json
import sys

from django.core.management.base import BaseCommand

from apps.identity.migration_reconciliation import (
    MigrationReconciliationReport,
    reconcile_all,
)


class Command(BaseCommand):
    help = (
        "Reconcile legacy Student rows against the identity domain. "
        "Read-only: reports discrepancies, does not modify data."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--json",
            action="store_true",
            default=False,
            help="Output the report as JSON instead of human-readable text.",
        )
        parser.add_argument(
            "--only-issues",
            action="store_true",
            default=False,
            help="Only show rows that have issues (skip clean rows).",
        )

    def handle(self, *args, **options):
        json_output: bool = options["json"]
        only_issues: bool = options["only_issues"]

        report = reconcile_all()

        if only_issues:
            self._output_issues_only(report, json_output)
        else:
            self._output_full(report, json_output)

        if report.issues_found or report.duplicate_person_codes or \
           report.duplicate_student_profile_codes or \
           report.orphan_student_profiles or report.orphan_persons:
            sys.exit(1)

    # -------------------------------------------------------------------
    # Output helpers
    # -------------------------------------------------------------------

    def _output_full(self, report: MigrationReconciliationReport, json_output: bool):
        if json_output:
            self._output_json(report, only_issues=False)
        else:
            self.stdout.write(self.style.MIGRATE_HEADING(
                "Identity Migration Reconciliation"
            ))
            self.stdout.write("")
            self.stdout.write(f"  Total students:              {report.total_students}")
            self.stdout.write(f"  Migrated:                    {report.migrated}")
            self.stdout.write(f"  Unmigrated:                  {report.unmigrated}")
            self.stdout.write(f"  Students with issues:        {report.issues_found}")
            self.stdout.write(f"  Duplicate person codes:      {len(report.duplicate_person_codes)}")
            self.stdout.write(f"  Duplicate profile codes:     {len(report.duplicate_student_profile_codes)}")
            self.stdout.write(f"  Orphan student profiles:     {len(report.orphan_student_profiles)}")
            self.stdout.write(f"  Orphan persons:              {len(report.orphan_persons)}")
            self.stdout.write("")

            if report.duplicate_person_codes:
                self.stdout.write(self.style.WARNING("Duplicate Person codes:"))
                for code in report.duplicate_person_codes:
                    self.stdout.write(f"  {code}")
                self.stdout.write("")

            if report.duplicate_student_profile_codes:
                self.stdout.write(self.style.WARNING("Duplicate StudentProfile codes:"))
                for code in report.duplicate_student_profile_codes:
                    self.stdout.write(f"  {code}")
                self.stdout.write("")

            if report.orphan_student_profiles:
                self.stdout.write(self.style.WARNING("Orphan StudentProfiles (legacy_student=None):"))
                for pk in report.orphan_student_profiles:
                    self.stdout.write(f"  StudentProfile pk={pk}")
                self.stdout.write("")

            if report.orphan_persons:
                self.stdout.write(self.style.WARNING("Orphan Persons (no StudentProfile):"))
                for pk in report.orphan_persons:
                    self.stdout.write(f"  Person pk={pk}")
                self.stdout.write("")

            if report.rows_with_issues:
                self.stdout.write(self.style.WARNING("Students with issues:"))
                for row in report.rows_with_issues:
                    self.stdout.write(f"  {row.h_code}:")
                    for issue in row.issues:
                        self.stdout.write(f"    [{issue.issue_type}] {issue.detail}")
                self.stdout.write("")

            if not report.rows_with_issues and \
               not report.duplicate_person_codes and \
               not report.duplicate_student_profile_codes and \
               not report.orphan_student_profiles and \
               not report.orphan_persons:
                self.stdout.write(self.style.SUCCESS(
                    "No issues found — all students reconciled correctly."
                ))

    def _output_issues_only(self, report: MigrationReconciliationReport, json_output: bool):
        if json_output:
            self._output_json(report, only_issues=True)
        else:
            issue_rows = report.rows_with_issues
            self.stdout.write(self.style.WARNING(
                f"Rows with issues ({len(issue_rows)} of {report.total_students}):"
            ))
            self.stdout.write("")
            for row in issue_rows:
                self.stdout.write(f"  {row.h_code}:")
                for issue in row.issues:
                    self.stdout.write(f"    [{issue.issue_type}] {issue.detail}")
            self.stdout.write("")

            if report.duplicate_person_codes:
                self.stdout.write(self.style.WARNING("Duplicate Person codes:"))
                for code in report.duplicate_person_codes:
                    self.stdout.write(f"  {code}")
                self.stdout.write("")

            if report.duplicate_student_profile_codes:
                self.stdout.write(self.style.WARNING("Duplicate StudentProfile codes:"))
                for code in report.duplicate_student_profile_codes:
                    self.stdout.write(f"  {code}")
                self.stdout.write("")

            if report.orphan_student_profiles:
                self.stdout.write(self.style.WARNING("Orphan StudentProfiles:"))
                for pk in report.orphan_student_profiles:
                    self.stdout.write(f"  pk={pk}")
                self.stdout.write("")

            if report.orphan_persons:
                self.stdout.write(self.style.WARNING("Orphan Persons:"))
                for pk in report.orphan_persons:
                    self.stdout.write(f"  pk={pk}")

    def _output_json(self, report: MigrationReconciliationReport, *, only_issues: bool):
        rows = report.rows_with_issues if only_issues else report.rows
        self.stdout.write(json.dumps({
            "total_students": report.total_students,
            "migrated": report.migrated,
            "unmigrated": report.unmigrated,
            "issues_found": report.issues_found,
            "duplicate_person_codes": report.duplicate_person_codes,
            "duplicate_student_profile_codes": report.duplicate_student_profile_codes,
            "orphan_student_profiles": report.orphan_student_profiles,
            "orphan_persons": report.orphan_persons,
            "rows": [
                {
                    "h_code": r.h_code,
                    "is_migrated": r.is_migrated,
                    "person_code": r.person_code,
                    "student_profile_code": r.student_profile_code,
                    "has_student_role": r.has_student_role,
                    "issues": [
                        {"issue_type": i.issue_type, "detail": i.detail}
                        for i in r.issues
                    ],
                }
                for r in rows
            ],
        }, indent=2))
