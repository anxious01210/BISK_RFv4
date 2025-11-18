# apps/attendance/utils/lunch.py
from django.utils import timezone
from apps.attendance.models import Student, LunchSubscription


def recalc_lunch_flags_for_students(qs=None, today=None, verbose=False):
    """
    Sync Student.has_lunch based on LunchSubscription for the given queryset.
    If qs is None, all students are processed.

    Logic:
      has_lunch = True  if there is at least one ACTIVE subscription
                         where start_date <= today <= end_date.
      has_lunch = False otherwise.
    """
    if today is None:
        today = timezone.localdate()

    if qs is None:
        qs = Student.objects.all()

    # Work with IDs only to avoid surprising queryset reuse.
    student_ids = list(qs.values_list("id", flat=True))
    if not student_ids:
        return 0, 0

    # Subscriptions that make a student eligible today.
    active_ids = set(
        LunchSubscription.objects.filter(
            status=LunchSubscription.STATUS_ACTIVE,
            start_date__lte=today,
            end_date__gte=today,
            student_id__in=student_ids,
        ).values_list("student_id", flat=True)
    )

    # First, mark all in this set as False…
    Student.objects.filter(id__in=student_ids).update(has_lunch=False)
    # …then mark only the eligible ones as True.
    if active_ids:
        Student.objects.filter(id__in=active_ids).update(has_lunch=True)

    if verbose:
        print(f"[lunch] Recalculated {len(student_ids)} students; {len(active_ids)} eligible on {today}.")

    return len(active_ids), len(student_ids)


def recalc_lunch_flags_all(today=None, verbose=False):
    """
    Convenience wrapper for 'all students'.
    """
    return recalc_lunch_flags_for_students(
        qs=Student.objects.all(),
        today=today,
        verbose=verbose,
    )
