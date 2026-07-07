"""Management command: reconcile_attendance_identity.

Thin read-only CLI wrapper around
:func:`apps.attendance.identity_reconciliation.reconcile_all_attendance_identity`.

Verifies that the ``person`` FK on ``FaceEmbedding``, ``AttendanceEvent``,
and ``AttendanceRecord`` is correctly populated after the M1/M2/M3 dual-FK
migrations. Reports any discrepancies (missing person_id, mismatched
person_id, duplicates, orphans).

**This command is read-only.** It does not create or modify any
database rows. There is no ``--fix`` flag.

Usage::

    # Full reconciliation report (human-readable)
    python manage.py reconcile_attendance_identity

    # JSON output
    python manage.py reconcile_attendance_identity --json

    # Only show rows with issues
    python manage.py reconcile_attendance_identity --only-issues

    # Reconcile a single model
    python manage.py reconcile_attendance_identity --model attendance_record

Exit codes:

* ``0`` — no issues found (S9 gate passes).
* ``1`` — one or more issues found (S9 gate fails; fix before S9).
* ``2`` — unexpected error (exception during reconciliation).
"""

from __future__ import annotations

import json
import sys
import traceback

from django.core.management.base import BaseCommand

from apps.attendance.identity_reconciliation import (
    AttendanceIdentityReconciliationReport,
    ModelReconciliationReport,
    reconcile_all_attendance_identity,
    reconcile_attendance_events,
    reconcile_attendance_records,
    reconcile_face_embeddings,
)


class Command(BaseCommand):
    help = (
        "Reconcile attendance identity dual-FK data (FaceEmbedding, "
        "AttendanceEvent, AttendanceRecord). Read-only: reports "
        "discrepancies, does not modify data."
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
        parser.add_argument(
            "--model",
            choices=["face_embedding", "attendance_event", "attendance_record", "all"],
            default="all",
            help="Reconcile a single model (default: all).",
        )

    def handle(self, *args, **options):
        json_output: bool = options["json"]
        only_issues: bool = options["only_issues"]
        model_filter: str = options["model"]

        try:
            if model_filter == "all":
                report = reconcile_all_attendance_identity()
            elif model_filter == "face_embedding":
                fe = reconcile_face_embeddings()
                report = AttendanceIdentityReconciliationReport(
                    face_embeddings=fe,
                    attendance_events=ModelReconciliationReport(
                        model_name="AttendanceEvent", total_rows=0,
                        rows_with_person_id=0, rows_without_person_id=0,
                    ),
                    attendance_records=ModelReconciliationReport(
                        model_name="AttendanceRecord", total_rows=0,
                        rows_with_person_id=0, rows_without_person_id=0,
                    ),
                )
            elif model_filter == "attendance_event":
                ae = reconcile_attendance_events()
                report = AttendanceIdentityReconciliationReport(
                    face_embeddings=ModelReconciliationReport(
                        model_name="FaceEmbedding", total_rows=0,
                        rows_with_person_id=0, rows_without_person_id=0,
                    ),
                    attendance_events=ae,
                    attendance_records=ModelReconciliationReport(
                        model_name="AttendanceRecord", total_rows=0,
                        rows_with_person_id=0, rows_without_person_id=0,
                    ),
                )
            else:  # attendance_record
                ar = reconcile_attendance_records()
                report = AttendanceIdentityReconciliationReport(
                    face_embeddings=ModelReconciliationReport(
                        model_name="FaceEmbedding", total_rows=0,
                        rows_with_person_id=0, rows_without_person_id=0,
                    ),
                    attendance_events=ModelReconciliationReport(
                        model_name="AttendanceEvent", total_rows=0,
                        rows_with_person_id=0, rows_without_person_id=0,
                    ),
                    attendance_records=ar,
                )
        except Exception:
            traceback.print_exc()
            sys.exit(2)

        if only_issues:
            self._output_issues_only(report, json_output, model_filter)
        else:
            self._output_full(report, json_output, model_filter)

        if not report.is_clean:
            sys.exit(1)

    # -------------------------------------------------------------------
    # Output helpers
    # -------------------------------------------------------------------

    def _output_full(self, report: AttendanceIdentityReconciliationReport,
                     json_output: bool, model_filter: str):
        if json_output:
            self._output_json(report, only_issues=False, model_filter=model_filter)
        else:
            self.stdout.write(self.style.MIGRATE_HEADING(
                "Attendance Identity Reconciliation"
            ))
            self.stdout.write("")
            for mr in [report.face_embeddings, report.attendance_events,
                       report.attendance_records]:
                if model_filter != "all" and mr.total_rows == 0:
                    continue
                self._print_model_summary(mr)

            if report.duplicate_person_periods:
                self.stdout.write(self.style.WARNING(
                    f"Duplicate (person_id, period_id) on AttendanceRecord: "
                    f"{len(report.duplicate_person_periods)}"
                ))
                for person_id, period_id, count in report.duplicate_person_periods:
                    self.stdout.write(
                        f"  person_id={person_id}, period_id={period_id}, count={count}"
                    )
                self.stdout.write("")

            if report.duplicate_active_embeddings:
                self.stdout.write(self.style.WARNING(
                    f"Duplicate active FaceEmbedding per person_id: "
                    f"{len(report.duplicate_active_embeddings)}"
                ))
                for person_id, count in report.duplicate_active_embeddings:
                    self.stdout.write(
                        f"  person_id={person_id}, count={count}"
                    )
                self.stdout.write("")

            if report.is_clean:
                self.stdout.write(self.style.SUCCESS(
                    "No issues found — all attendance identity data reconciled "
                    "correctly. S9 gate passes."
                ))

    def _output_issues_only(self, report: AttendanceIdentityReconciliationReport,
                            json_output: bool, model_filter: str):
        if json_output:
            self._output_json(report, only_issues=True, model_filter=model_filter)
        else:
            any_issues = False
            for mr in [report.face_embeddings, report.attendance_events,
                       report.attendance_records]:
                if model_filter != "all" and mr.total_rows == 0:
                    continue
                if mr.issues:
                    any_issues = True
                    self.stdout.write(self.style.WARNING(
                        f"{mr.model_name} issues ({mr.issues_found}):"
                    ))
                    for issue in mr.issues:
                        self.stdout.write(
                            f"  [{issue.issue_type}] row_id={issue.row_id} "
                            f"student_id={issue.student_id} "
                            f"person_id={issue.person_id} "
                            f"expected={issue.expected_person_id}"
                        )
                    self.stdout.write("")

            if report.duplicate_person_periods:
                any_issues = True
                self.stdout.write(self.style.WARNING(
                    f"Duplicate (person_id, period_id): "
                    f"{len(report.duplicate_person_periods)}"
                ))
                for person_id, period_id, count in report.duplicate_person_periods:
                    self.stdout.write(
                        f"  person_id={person_id}, period_id={period_id}, "
                        f"count={count}"
                    )
                self.stdout.write("")

            if report.duplicate_active_embeddings:
                any_issues = True
                self.stdout.write(self.style.WARNING(
                    f"Duplicate active embeddings: "
                    f"{len(report.duplicate_active_embeddings)}"
                ))
                for person_id, count in report.duplicate_active_embeddings:
                    self.stdout.write(
                        f"  person_id={person_id}, count={count}"
                    )
                self.stdout.write("")

            if not any_issues:
                self.stdout.write(self.style.SUCCESS(
                    "No issues found. S9 gate passes."
                ))

    def _output_json(self, report: AttendanceIdentityReconciliationReport,
                     *, only_issues: bool, model_filter: str):
        models = []
        for mr in [report.face_embeddings, report.attendance_events,
                   report.attendance_records]:
            if model_filter != "all" and mr.total_rows == 0:
                continue
            if only_issues and not mr.issues:
                continue
            models.append({
                "model_name": mr.model_name,
                "total_rows": mr.total_rows,
                "rows_with_person_id": mr.rows_with_person_id,
                "rows_without_person_id": mr.rows_without_person_id,
                "issues_found": mr.issues_found,
                "issue_types": mr.issue_types,
                "issues": [
                    {
                        "issue_type": i.issue_type,
                        "row_id": i.row_id,
                        "student_id": i.student_id,
                        "person_id": i.person_id,
                        "expected_person_id": i.expected_person_id,
                        "detail": i.detail,
                    }
                    for i in mr.issues
                ] if not only_issues else [
                    {
                        "issue_type": i.issue_type,
                        "row_id": i.row_id,
                        "student_id": i.student_id,
                        "person_id": i.person_id,
                        "expected_person_id": i.expected_person_id,
                        "detail": i.detail,
                    }
                    for i in mr.issues
                ],
            })

        self.stdout.write(json.dumps({
            "total_issues": report.total_issues,
            "is_clean": report.is_clean,
            "duplicate_person_periods": [
                {"person_id": p, "period_id": d, "count": c}
                for p, d, c in report.duplicate_person_periods
            ],
            "duplicate_active_embeddings": [
                {"person_id": p, "count": c}
                for p, c in report.duplicate_active_embeddings
            ],
            "models": models,
        }, indent=2))

    def _print_model_summary(self, mr: ModelReconciliationReport):
        self.stdout.write(f"  {mr.model_name}:")
        self.stdout.write(f"    Total rows:                {mr.total_rows}")
        self.stdout.write(f"    Rows with person_id:       {mr.rows_with_person_id}")
        self.stdout.write(f"    Rows without person_id:    {mr.rows_without_person_id}")
        self.stdout.write(f"    Issues:                    {mr.issues_found}")
        if mr.issue_types:
            for itype, count in mr.issue_types.items():
                self.stdout.write(f"      {itype}: {count}")
        self.stdout.write("")
        if mr.issues:
            self.stdout.write(self.style.WARNING(
                f"  {mr.model_name} issues:"
            ))
            for issue in mr.issues:
                self.stdout.write(
                    f"    [{issue.issue_type}] row_id={issue.row_id} "
                    f"student_id={issue.student_id} "
                    f"person_id={issue.person_id} "
                    f"expected={issue.expected_person_id}"
                )
            self.stdout.write("")
