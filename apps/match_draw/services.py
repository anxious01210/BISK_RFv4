import random
from typing import List, Tuple

from django.db import transaction

from .models import DrawEvent, DrawMatch, DrawTeam, DrawPairRule


class DrawGenerationError(Exception):
    pass


def _active_teams(event: DrawEvent) -> List[DrawTeam]:
    teams = list(event.teams.filter(is_active=True).order_by("display_order", "id"))
    if len(teams) < 2:
        raise DrawGenerationError("At least 2 active teams are required.")
    if len(teams) % 2 != 0:
        raise DrawGenerationError("The number of active teams must be even.")
    return teams


def _generate_random_pairs(event: DrawEvent) -> List[Tuple[DrawTeam, DrawTeam]]:
    teams = _active_teams(event)
    random.shuffle(teams)

    pairs = []
    for i in range(0, len(teams), 2):
        pairs.append((teams[i], teams[i + 1]))
    return pairs


def _generate_guided_pairs(event: DrawEvent) -> List[Tuple[DrawTeam, DrawTeam]]:
    teams = _active_teams(event)
    team_ids = {t.id for t in teams}

    rules = list(
        event.pair_rules.select_related("team_a", "team_b")
        .all()
        .order_by("id")
    )

    if not rules:
        raise DrawGenerationError("Guided draw requires pair rules.")

    used = set()
    pairs = []

    for rule in rules:
        a = rule.team_a
        b = rule.team_b

        if a.id not in team_ids or b.id not in team_ids:
            raise DrawGenerationError("All pair rules must refer to active teams in this event.")

        if a.id in used or b.id in used:
            raise DrawGenerationError("A team appears more than once in pair rules.")

        used.add(a.id)
        used.add(b.id)
        pairs.append((a, b))

    if used != team_ids:
        raise DrawGenerationError("Every active team must appear in exactly one pair rule.")

    random.shuffle(pairs)
    return pairs


@transaction.atomic
def generate_matches_for_event(event: DrawEvent) -> List[DrawMatch]:
    event.matches.all().delete()
    event.spin_logs.all().delete()

    if event.mode == DrawEvent.MODE_RANDOM:
        pairs = _generate_random_pairs(event)
    elif event.mode == DrawEvent.MODE_RIGGED:
        pairs = _generate_guided_pairs(event)
    else:
        raise DrawGenerationError(f"Unsupported draw mode: {event.mode}")

    created_matches = []
    for idx, (team_1, team_2) in enumerate(pairs, start=1):
        match = DrawMatch.objects.create(
            draw_event=event,
            team_1=team_1,
            team_2=team_2,
            match_order=idx,
        )
        created_matches.append(match)

    event.status = DrawEvent.STATUS_LIVE
    event.save(update_fields=["status", "updated_at"])

    return created_matches