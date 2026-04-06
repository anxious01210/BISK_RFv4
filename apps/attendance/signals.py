# apps/attendance/signals.py
def ensure_media_tree_signal(**kwargs):
    from .utils.media_paths import ensure_media_tree
    ensure_media_tree()


# apps/attendance/signals.py
from django.db import transaction
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.utils import timezone

from apps.attendance.models import MealSubscription, Student
from apps.attendance.utils.meal import recalc_meal_flags_for_students


def _recalc_student_meal_after_commit(student_id):
    """
    Recalculate has_meal for ONE student, safely after DB commit.
    """

    def _do():
        qs = Student.objects.filter(id=student_id)
        recalc_meal_flags_for_students(qs=qs, today=timezone.localdate(), verbose=False)

    transaction.on_commit(_do)


@receiver(post_save, sender=MealSubscription)
def meal_subscription_saved(sender, instance, **kwargs):
    """
    Triggered when accountant creates or edits a meal subscription.
    """
    if instance.student_id:
        _recalc_student_meal_after_commit(instance.student_id)


@receiver(post_delete, sender=MealSubscription)
def meal_subscription_deleted(sender, instance, **kwargs):
    """
    Triggered if a meal subscription is deleted
    (rare, but safe to handle).
    """
    if instance.student_id:
        _recalc_student_meal_after_commit(instance.student_id)
