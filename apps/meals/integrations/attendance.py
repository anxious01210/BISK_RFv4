"""Attendance-to-Meals integration bridge.

This module is the **only** integration point between the legacy
``apps.attendance`` recognition pipeline and the new ``apps.meals``
domain. It resolves ``identity.Person`` from legacy
``attendance.Student`` via the ``StudentProfile.legacy_student``
backlink, then calls :func:`apps.meals.services.resolve_service` to
create/update a :class:`~apps.meals.models.MealServiceEvent` linked
to the :class:`~apps.attendance.models.AttendanceEvent` that triggered
it.

Design rules (per ``docs/architecture/meals_domain_architecture.md``
§6 and ``docs/development/DOMAIN_INTEGRATION_GUIDE.md`` §10):

* **Additive only.** This module does NOT modify any legacy code.
  The legacy ``confirm_record`` / ``reverse_record`` views continue
  to work unchanged.
* **No migration dependency.** This module queries
  ``attendance.AttendanceEvent`` at runtime via Django's app registry;
  it does not import attendance models into ``apps.meals.models``.
* **Dependency direction.** The bridge lives in ``apps.meals``, so
  ``apps.attendance`` does not import ``apps.meals``. The direction is
  ``bridge → meals.services → identity / finance``, with a read-only
  runtime query on ``attendance.AttendanceEvent``.
* **No legacy mutation.** The bridge does NOT touch legacy
  ``MealRecord``, ``Wallet``, or ``WalletTransaction`` tables. It
  creates only new-domain rows (``MealServiceEvent``,
  ``MealSupervisorAction``, and — via ``finance.charge`` — new-domain
  ``WalletTransaction``).
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from django.apps import apps as django_apps
from django.utils import timezone


def _get_attendance_student_model():
    """Return the legacy ``attendance.Student`` model class via the
    Django app registry (avoids importing ``apps.attendance.models``
    at module load time)."""
    return django_apps.get_model("attendance", "Student")


def _get_attendance_event_model():
    """Return the legacy ``attendance.AttendanceEvent`` model class."""
    return django_apps.get_model("attendance", "AttendanceEvent")


def resolve_person_from_student(student) -> Optional:
    """Resolve ``identity.Person`` from a legacy
    ``attendance.Student``.

    Resolution chain::

        Student.migrated_to  →  StudentProfile  →  StudentProfile.person  →  Person

    Returns ``None`` if the student has no ``StudentProfile`` (i.e. the
    identity data migration has not linked this student to a Person
    yet). This is expected in production until the Phase 1.5 data
    migration runs.
    """
    if student is None:
        return None

    # Use Django's app registry to avoid importing attendance models.
    StudentProfile = django_apps.get_model("identity", "StudentProfile")

    try:
        profile = StudentProfile.objects.get(legacy_student=student)
    except StudentProfile.DoesNotExist:
        return None

    return profile.person


def resolve_person_from_h_code(h_code: str) -> Optional:
    """Resolve ``identity.Person`` from a legacy student ``h_code``.

    Looks up ``attendance.Student`` by ``h_code``, then delegates to
    :func:`resolve_person_from_student`.

    Returns ``None`` if the ``h_code`` is not found or the student has
    no linked ``StudentProfile`` / ``Person``.
    """
    if not h_code:
        return None

    Student = _get_attendance_student_model()
    try:
        student = Student.objects.get(h_code=h_code)
    except Student.DoesNotExist:
        return None

    return resolve_person_from_student(student)


def bridge_recognition_to_service_event(
    *,
    attendance_event,
    meal_period=None,
    on_date: Optional[date] = None,
    created_by=None,
):
    """Bridge a legacy ``AttendanceEvent`` to a new-domain
    :class:`~apps.meals.models.MealServiceEvent`.

    1. Resolves ``Person`` from ``attendance_event.student`` via the
       ``StudentProfile.legacy_student`` backlink.
    2. If no ``Person`` is resolved, returns ``None`` — no event is
       created. This is expected for students that have not yet been
       migrated to the identity domain.
    3. Derives ``on_date`` from ``attendance_event.ts`` if not
       supplied.
    4. Calls :func:`apps.meals.services.resolve_service` with
       ``person``, ``on_date``, ``meal_period``, ``created_by``, and
       ``recognition_event=attendance_event``.
    5. Returns the resulting :class:`MealServiceEvent`.

    The ``MealServiceEvent.recognition_event`` FK (SET_NULL, string
    ref to ``"attendance.AttendanceEvent"``) links the service event
    to the recognition that triggered it (§6, §12).

    **Idempotency:** if ``resolve_service`` has already been called
    for the same ``(person, date, meal_period)`` and the resulting
    event is in a terminal status, the eligibility row is frozen.
    A second call will raise ``ValidationError`` from
    ``resolve_eligibility``. Callers should catch this if they need
    idempotent behaviour:

        .. code-block:: python

            from django.core.exceptions import ValidationError
            try:
                ev = bridge_recognition_to_service_event(...)
            except ValidationError:
                ev = None  # already resolved

    **No legacy mutation:** this function does NOT touch legacy
    ``MealRecord``, ``Wallet``, or ``WalletTransaction`` rows. It
    creates only new-domain rows.
    """
    from apps.meals.services import resolve_service

    if attendance_event is None:
        return None

    person = resolve_person_from_student(attendance_event.student)
    if person is None:
        return None

    # Derive on_date from the event timestamp if not supplied.
    if on_date is None:
        ts = getattr(attendance_event, "ts", None)
        if ts is not None:
            on_date = timezone.localdate(ts)
        else:
            on_date = date.today()

    ev = resolve_service(
        person=person,
        on_date=on_date,
        meal_period=meal_period,
        recognition_event=attendance_event,
        created_by=created_by,
    )
    return ev
