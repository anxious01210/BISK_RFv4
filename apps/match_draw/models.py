from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models


class DrawEvent(models.Model):
    MODE_RANDOM = "random"
    MODE_RIGGED = "rigged"

    MODE_CHOICES = [
        (MODE_RANDOM, "Completely Random"),
        (MODE_RIGGED, "Rigged / Controlled"),
    ]

    STATUS_DRAFT = "draft"
    STATUS_READY = "ready"
    STATUS_LIVE = "live"
    STATUS_DONE = "done"

    STATUS_CHOICES = [
        (STATUS_DRAFT, "Draft"),
        (STATUS_READY, "Ready"),
        (STATUS_LIVE, "Live"),
        (STATUS_DONE, "Done"),
    ]

    THEME_DARK = "dark"
    THEME_LIGHT = "light"
    THEME_AUTO = "auto"

    THEME_CHOICES = [
        (THEME_DARK, "Dark"),
        (THEME_LIGHT, "Light"),
        (THEME_AUTO, "Auto"),
    ]

    title = models.CharField(max_length=200)
    mode = models.CharField(max_length=20, choices=MODE_CHOICES, default=MODE_RANDOM)
    theme = models.CharField(max_length=20, choices=THEME_CHOICES, default=THEME_DARK)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_DRAFT)

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="match_draw_events",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.title

    @property
    def teams_count(self):
        return self.teams.filter(is_active=True).count()


class DrawTeam(models.Model):
    draw_event = models.ForeignKey(
        DrawEvent,
        on_delete=models.CASCADE,
        related_name="teams",
    )
    name = models.CharField(max_length=120)
    display_order = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["display_order", "id"]
        unique_together = [("draw_event", "name")]

    def __str__(self):
        return f"{self.name} ({self.draw_event.title})"


class DrawPairRule(models.Model):
    draw_event = models.ForeignKey(
        DrawEvent,
        on_delete=models.CASCADE,
        related_name="pair_rules",
    )
    team_a = models.ForeignKey(
        DrawTeam,
        on_delete=models.CASCADE,
        related_name="pair_rule_as_a",
    )
    team_b = models.ForeignKey(
        DrawTeam,
        on_delete=models.CASCADE,
        related_name="pair_rule_as_b",
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["draw_event", "team_a"],
                name="unique_pair_rule_team_a_per_event",
            ),
            models.UniqueConstraint(
                fields=["draw_event", "team_b"],
                name="unique_pair_rule_team_b_per_event",
            ),
        ]

    def clean(self):
        if self.team_a_id == self.team_b_id:
            raise ValidationError("A team cannot be paired with itself.")

        if self.team_a.draw_event_id != self.draw_event_id or self.team_b.draw_event_id != self.draw_event_id:
            raise ValidationError("Both teams must belong to the same draw event.")

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.team_a.name} ↔ {self.team_b.name}"


class DrawMatch(models.Model):
    draw_event = models.ForeignKey(
        DrawEvent,
        on_delete=models.CASCADE,
        related_name="matches",
    )
    team_1 = models.ForeignKey(
        DrawTeam,
        on_delete=models.CASCADE,
        related_name="matches_as_team_1",
    )
    team_2 = models.ForeignKey(
        DrawTeam,
        on_delete=models.CASCADE,
        related_name="matches_as_team_2",
    )
    match_order = models.PositiveIntegerField(default=1)
    revealed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["match_order", "id"]

    def clean(self):
        if self.team_1_id == self.team_2_id:
            raise ValidationError("A match cannot contain the same team twice.")

        if self.team_1.draw_event_id != self.draw_event_id or self.team_2.draw_event_id != self.draw_event_id:
            raise ValidationError("Both teams must belong to the same draw event.")

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Match {self.match_order}: {self.team_1.name} vs {self.team_2.name}"


class DrawSpinLog(models.Model):
    draw_event = models.ForeignKey(
        DrawEvent,
        on_delete=models.CASCADE,
        related_name="spin_logs",
    )
    selected_team = models.ForeignKey(
        DrawTeam,
        on_delete=models.CASCADE,
        related_name="spin_logs",
    )
    spin_number = models.PositiveIntegerField()
    was_forced = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["spin_number", "id"]

    def __str__(self):
        return f"Spin {self.spin_number}: {self.selected_team.name}"