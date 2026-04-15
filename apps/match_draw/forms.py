from django import forms
from .models import DrawEvent


class DrawSetupForm(forms.Form):
    title = forms.CharField(
        max_length=200,
        label="Draw title",
        widget=forms.TextInput(attrs={
            "placeholder": "Example: Grade Football Draw 2026"
        })
    )

    mode = forms.ChoiceField(
        label="Draw mode",
        choices=[
            (DrawEvent.MODE_RANDOM, "Random Draw"),
            (DrawEvent.MODE_RIGGED, "Guided Draw"),
        ],
        initial=DrawEvent.MODE_RIGGED,
        widget=forms.RadioSelect
    )

    theme = forms.ChoiceField(
        label="Theme",
        choices=DrawEvent.THEME_CHOICES,
        initial=DrawEvent.THEME_DARK
    )

    teams_text = forms.CharField(
        label="Teams",
        required=True,
        widget=forms.Textarea(attrs={
            "rows": 10,
            "placeholder": "One team per line\n\nGrade 6A\nGrade 6B\nGrade 5A\nGrade 5B"
        }),
        help_text="Enter one team per line."
    )

    rigged_pairs_text = forms.CharField(
        label="Pair rules",
        required=False,
        widget=forms.Textarea(attrs={
            "rows": 8,
            "placeholder": "One pair per line using | between teams\n\nGrade 6A | Grade 6B\nGrade 5A | Grade 5B"
        }),
        help_text="Only used in guided draw mode. Format: Team A | Team B"
    )

    def clean_teams_text(self):
        raw = self.cleaned_data["teams_text"]
        teams = [line.strip() for line in raw.splitlines() if line.strip()]

        if len(teams) < 2:
            raise forms.ValidationError("Please enter at least 2 teams.")

        if len(teams) % 2 != 0:
            raise forms.ValidationError("The number of teams must be even.")

        if len(set(teams)) != len(teams):
            raise forms.ValidationError("Duplicate team names are not allowed.")

        return "\n".join(teams)

    def clean(self):
        cleaned = super().clean()
        mode = cleaned.get("mode")
        teams_text = cleaned.get("teams_text", "")
        rigged_pairs_text = cleaned.get("rigged_pairs_text", "")

        teams = [line.strip() for line in teams_text.splitlines() if line.strip()]
        team_set = set(teams)

        if mode == DrawEvent.MODE_RIGGED:
            pair_lines = [line.strip() for line in rigged_pairs_text.splitlines() if line.strip()]

            if not pair_lines:
                raise forms.ValidationError("Guided draw requires at least one pair rule.")

            seen = set()

            for line in pair_lines:
                if "|" not in line:
                    raise forms.ValidationError(
                        f"Invalid pair format: '{line}'. Use: Team A | Team B"
                    )

                left, right = [part.strip() for part in line.split("|", 1)]

                if not left or not right:
                    raise forms.ValidationError(
                        f"Invalid pair format: '{line}'. Both team names are required."
                    )

                if left == right:
                    raise forms.ValidationError(
                        f"Invalid pair: '{line}'. A team cannot be paired with itself."
                    )

                if left not in team_set or right not in team_set:
                    raise forms.ValidationError(
                        f"Invalid pair: '{line}'. Both teams must exist in the team list."
                    )

                if left in seen or right in seen:
                    raise forms.ValidationError(
                        f"Invalid pair: '{line}'. A team cannot appear in more than one pair rule."
                    )

                seen.add(left)
                seen.add(right)

            if seen != team_set:
                missing = sorted(team_set - seen)
                raise forms.ValidationError(
                    "In guided draw mode, every team must appear in exactly one pair rule. "
                    f"Missing teams: {', '.join(missing)}"
                )

        return cleaned