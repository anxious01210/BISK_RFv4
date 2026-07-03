from django.conf import settings
from django.db import models


class Person(models.Model):
    class Gender(models.TextChoices):
        MALE = "MALE", "Male"
        FEMALE = "FEMALE", "Female"

    code = models.CharField(max_length=32, unique=True, db_index=True)
    first_name = models.CharField(max_length=150)
    middle_name = models.CharField(max_length=150, blank=True)
    last_name = models.CharField(max_length=150)
    gender = models.CharField(
        max_length=10,
        choices=Gender.choices,
        null=True,
        blank=True,
    )
    date_of_birth = models.DateField(null=True, blank=True)
    email = models.EmailField(blank=True)
    phone = models.CharField(max_length=32, blank=True)
    address = models.TextField(blank=True)
    photo = models.ImageField(upload_to="person_photos/", blank=True)
    is_active = models.BooleanField(default=True, db_index=True)
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="person",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["code"]
        indexes = [
            models.Index(fields=["is_active"]),
            models.Index(fields=["last_name", "first_name"]),
        ]

    def __str__(self) -> str:
        return f"{self.code} - {self.full_name}"

    @property
    def full_name(self) -> str:
        parts = [self.first_name, self.middle_name, self.last_name]
        return " ".join(part for part in parts if part).strip()

    @property
    def display_code(self) -> str:
        """
        Return the preferred code for displaying this person.

        If a role-specific profile exists (Student, Staff, etc.),
        its code is used. Otherwise, the global Person.code is returned.
        """
        profile = getattr(self, "student_profile", None)
        if profile is not None:
            return profile.display_code

        profile = getattr(self, "staff_profile", None)
        if profile is not None:
            return profile.display_code

        return self.code

class RoleType(models.Model):
    class RoleChoices(models.TextChoices):
        STUDENT = "student", "Student"
        STAFF = "staff", "Staff"
        TEACHER = "teacher", "Teacher"
        PARENT = "parent", "Parent"
        GUARDIAN = "guardian", "Guardian"
        GUEST = "guest", "Guest"
        VENDOR = "vendor", "Vendor"
        ADMINISTRATOR = "administrator", "Administrator"
        FINANCE = "finance", "Finance"
        HR = "hr", "HR"
        PRINCIPAL = "principal", "Principal"
        VICE_PRINCIPAL = "vice_principal", "Vice Principal"
        LIBRARIAN = "librarian", "Librarian"
        NURSE = "nurse", "Nurse"

    code = models.CharField(max_length=32, unique=True, choices=RoleChoices.choices)
    name = models.CharField(max_length=100)
    is_active = models.BooleanField(default=True)
    is_system = models.BooleanField(default=False)

    class Meta:
        ordering = ["code"]

    def __str__(self) -> str:
        return self.name


class PersonRole(models.Model):
    person = models.ForeignKey(
        Person,
        on_delete=models.CASCADE,
        related_name="roles",
    )
    role_type = models.ForeignKey(
        RoleType,
        on_delete=models.PROTECT,
        related_name="person_roles",
    )
    is_active = models.BooleanField(default=True)
    start_date = models.DateField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)
    notes = models.TextField(blank=True)
    assigned_at = models.DateTimeField(auto_now_add=True)
    assigned_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )

    class Meta:
        ordering = ["person", "role_type"]
        unique_together = [["person", "role_type"]]
        indexes = [
            models.Index(fields=["person", "is_active"]),
            models.Index(fields=["role_type", "is_active"]),
        ]

    def __str__(self) -> str:
        return f"{self.person} - {self.role_type}"


class StudentProfile(models.Model):
    person = models.OneToOneField(
        Person,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="student_profile",
    )
    code = models.CharField(
        max_length=32,
        unique=True,
        db_index=True,
        help_text="Student code assigned by the school.",
    )
    grade = models.CharField(
        max_length=50,
        blank=True,
        db_index=True,
        help_text="Transitional compatibility field for legacy attendance.Student grade data.",
    )
    homeroom = models.CharField(max_length=50, blank=True)
    has_meal = models.BooleanField(
        default=False,
        db_index=True,
        help_text="Transitional compatibility field for legacy attendance.Student meal data.",
    )
    has_bus = models.BooleanField(
        default=False,
        db_index=True,
        help_text="Transitional compatibility field for legacy attendance.Student bus data.",
    )
    legacy_student = models.OneToOneField(
        "attendance.Student",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="migrated_to",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["person__code"]

    @property
    def display_code(self) -> str:
        return self.code

    def __str__(self) -> str:
        return f"Student profile - {self.person}"


class StaffProfile(models.Model):
    person = models.OneToOneField(
        Person,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="staff_profile",
    )
    code = models.CharField(
        max_length=32,
        unique=True,
        db_index=True,
        help_text="Staff code assigned by the school.",
    )
    job_title = models.CharField(max_length=100, blank=True)
    hire_date = models.DateField(null=True, blank=True)
    is_teacher = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["person__code"]

    @property
    def display_code(self) -> str:
        return self.code

    def __str__(self) -> str:
        return f"Staff profile - {self.person}"
