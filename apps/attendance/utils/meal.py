# apps/attendance/utils/meal.py
from django.utils import timezone
from apps.attendance.models import Student, MealSubscription


def recalc_meal_flags_for_students(qs=None, today=None, verbose=False):
    """
    Sync Student.has_meal based on MealSubscription for the given queryset.
    If qs is None, all students are processed.

    Logic:
      has_meal = True  if there is at least one ACTIVE subscription
                         where start_date <= today <= end_date.
      has_meal = False otherwise.
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
        MealSubscription.objects.filter(
            status=MealSubscription.STATUS_ACTIVE,
            start_date__lte=today,
            end_date__gte=today,
            student_id__in=student_ids,
        ).values_list("student_id", flat=True)
    )

    # First, mark all in this set as False…
    Student.objects.filter(id__in=student_ids).update(has_meal=False)
    # …then mark only the eligible ones as True.
    if active_ids:
        Student.objects.filter(id__in=active_ids).update(has_meal=True)

    if verbose:
        print(f"[meal] Recalculated {len(student_ids)} students; {len(active_ids)} eligible on {today}.")

    return len(active_ids), len(student_ids)


def recalc_meal_flags_all(today=None, verbose=False):
    """
    Convenience wrapper for 'all students'.
    """
    return recalc_meal_flags_for_students(
        qs=Student.objects.all(),
        today=today,
        verbose=verbose,
    )
