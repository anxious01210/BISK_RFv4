from django.core.exceptions import ValidationError
from django.db import models


class AcademicYear(models.Model):
    name = models.CharField(max_length=100, help_text="e.g., '2026-2027'.")
    code = models.CharField(
        max_length=32,
        unique=True,
        db_index=True,
        help_text="Natural key used by other models and imports, e.g., '2026-27'.",
    )
    start_date = models.DateField()
    end_date = models.DateField()
    is_active = models.BooleanField(
        default=False,
        db_index=True,
        help_text="At most one active academic year per school (enforced at DB level).",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-start_date"]
        constraints = [
            models.CheckConstraint(
                check=models.Q(end_date__gte=models.F("start_date")),
                name="academic_year_end_after_start",
            ),
            models.UniqueConstraint(
                fields=["is_active"],
                condition=models.Q(is_active=True),
                name="academic_year_single_active",
            ),
        ]

    def __str__(self) -> str:
        return self.name

    def clean(self):
        super().clean()
        if self.is_active:
            qs = AcademicYear.objects.filter(is_active=True)
            if self.pk:
                qs = qs.exclude(pk=self.pk)
            if qs.exists():
                from django.core.exceptions import ValidationError
                raise ValidationError(
                    {"is_active": "Another academic year is already active."}
                )


class SchoolLevel(models.Model):
    name = models.CharField(max_length=100, help_text="e.g., 'Primary', 'Secondary'.")
    code = models.CharField(max_length=32, unique=True, db_index=True)
    order = models.PositiveSmallIntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["order"]

    def __str__(self) -> str:
        return self.name


class Grade(models.Model):
    name = models.CharField(max_length=100, help_text="e.g., 'Grade 1'.")
    code = models.CharField(
        max_length=32,
        unique=True,
        db_index=True,
        help_text="Natural key for imports and FK references, e.g., 'G1'.",
    )
    level = models.ForeignKey(
        SchoolLevel,
        on_delete=models.PROTECT,
        related_name="grades",
        null=True,
        blank=True,
    )
    order = models.PositiveSmallIntegerField(default=0)
    is_active = models.BooleanField(default=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["order"]

    def __str__(self) -> str:
        return self.name


class Section(models.Model):
    name = models.CharField(max_length=100, help_text="e.g., 'Section A'.")
    code = models.CharField(max_length=32, help_text="e.g., 'A'.")
    grade = models.ForeignKey(
        Grade,
        on_delete=models.PROTECT,
        related_name="sections",
    )
    academic_year = models.ForeignKey(
        AcademicYear,
        on_delete=models.PROTECT,
        related_name="sections",
    )
    homeroom_adviser = models.ForeignKey(
        "identity.StaffProfile",
        on_delete=models.SET_NULL,
        related_name="advised_sections",
        null=True,
        blank=True,
    )
    capacity = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Year-specific. Enforced in StudentEnrollment.clean().",
    )
    is_active = models.BooleanField(default=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [("academic_year", "grade", "code")]
        ordering = ["academic_year__start_date", "grade__order", "code"]
        indexes = [
            models.Index(fields=["academic_year", "grade", "is_active"]),
        ]

    def __str__(self) -> str:
        return f"{self.academic_year.code} / {self.grade.code} / {self.code}"


class StudentEnrollment(models.Model):
    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        FUTURE = "future", "Future"
        TRANSFERRED = "transferred", "Transferred"
        WITHDRAWN = "withdrawn", "Withdrawn"
        GRADUATED = "graduated", "Graduated"
        REPEATED = "repeated", "Repeated"
        ARCHIVED = "archived", "Archived"

    student = models.ForeignKey(
        "identity.StudentProfile",
        on_delete=models.CASCADE,
        related_name="enrollments",
    )
    academic_year = models.ForeignKey(
        AcademicYear,
        on_delete=models.PROTECT,
        related_name="enrollments",
    )
    grade = models.ForeignKey(
        Grade,
        on_delete=models.PROTECT,
        related_name="enrollments",
    )
    section = models.ForeignKey(
        Section,
        on_delete=models.SET_NULL,
        related_name="enrollments",
        null=True,
        blank=True,
    )
    enrollment_date = models.DateField()
    withdrawal_date = models.DateField(null=True, blank=True)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.ACTIVE,
        db_index=True,
    )
    prior_enrollment = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        related_name="subsequent",
        null=True,
        blank=True,
        help_text="The enrollment this one continues from (transfer/repeat).",
    )
    notes = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-academic_year__start_date"]
        unique_together = [["student", "academic_year"]]
        indexes = [
            models.Index(fields=["academic_year", "status"]),
            models.Index(fields=["grade", "section", "status"]),
            models.Index(fields=["student", "status"]),
        ]

    def __str__(self) -> str:
        return f"{self.student} - {self.academic_year} - {self.grade}"

    def clean(self):
        super().clean()
        if self.section is not None:
            if self.section.academic_year_id != self.academic_year_id:
                raise ValidationError(
                    {"section": "Section's academic year must match the enrollment's academic year."}
                )
            if self.section.grade_id != self.grade_id:
                raise ValidationError(
                    {"section": "Section's grade must match the enrollment's grade."}
                )
        if (
            self.withdrawal_date
            and self.enrollment_date
            and self.withdrawal_date < self.enrollment_date
        ):
            raise ValidationError(
                {"withdrawal_date": "Withdrawal date cannot be before enrollment date."}
            )


class StudentEnrollmentSectionPlacement(models.Model):
    enrollment = models.ForeignKey(
        StudentEnrollment,
        on_delete=models.CASCADE,
        related_name="section_placements",
    )
    section = models.ForeignKey(
        Section,
        on_delete=models.PROTECT,
        related_name="placements",
    )
    start_date = models.DateField()
    end_date = models.DateField(null=True, blank=True)
    is_current = models.BooleanField(default=True, db_index=True)
    notes = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-start_date"]
        indexes = [
            models.Index(fields=["enrollment", "is_current"]),
            models.Index(fields=["section", "is_current"]),
        ]
        constraints = [
            models.CheckConstraint(
                check=models.Q(is_current=False) | models.Q(end_date__isnull=True),
                name="placement_current_has_no_end_date",
            ),
            models.UniqueConstraint(
                fields=["enrollment"],
                condition=models.Q(is_current=True),
                name="placement_single_current_per_enrollment",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.enrollment} - {self.section} - {'current' if self.is_current else 'closed'}"

    def clean(self):
        super().clean()
        if self.end_date and self.start_date and self.end_date < self.start_date:
            raise ValidationError(
                {"end_date": "Placement end date cannot be before start date."}
            )
        if self.section_id and self.enrollment_id:
            if self.section.academic_year_id != self.enrollment.academic_year_id:
                raise ValidationError(
                    {"section": "Placement section's academic year must match the enrollment's academic year."}
                )
            if self.section.grade_id != self.enrollment.grade_id:
                raise ValidationError(
                    {"section": "Placement section's grade must match the enrollment's grade."}
                )
        if (
            self.start_date
            and self.enrollment_id
            and self.enrollment.enrollment_date
            and self.start_date < self.enrollment.enrollment_date
        ):
            raise ValidationError(
                {"start_date": "Placement start date cannot be before the enrollment date."}
            )
