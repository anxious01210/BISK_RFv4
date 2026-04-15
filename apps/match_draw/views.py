from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import redirect, render, get_object_or_404

from .forms import DrawSetupForm
from .models import DrawEvent, DrawTeam, DrawPairRule
from .services import generate_matches_for_event, DrawGenerationError


@login_required
def draw_home(request):
    if request.method == "POST":
        form = DrawSetupForm(request.POST)
        if form.is_valid():
            event = DrawEvent.objects.create(
                title=form.cleaned_data["title"],
                mode=form.cleaned_data["mode"],
                theme=form.cleaned_data["theme"],
                status=DrawEvent.STATUS_READY,
                created_by=request.user,
            )

            team_names = [
                line.strip()
                for line in form.cleaned_data["teams_text"].splitlines()
                if line.strip()
            ]

            created_teams = {}
            for idx, team_name in enumerate(team_names, start=1):
                team = DrawTeam.objects.create(
                    draw_event=event,
                    name=team_name,
                    display_order=idx,
                    is_active=True,
                )
                created_teams[team_name] = team

            if event.mode == DrawEvent.MODE_RIGGED:
                pair_lines = [
                    line.strip()
                    for line in form.cleaned_data["rigged_pairs_text"].splitlines()
                    if line.strip()
                ]

                for line in pair_lines:
                    left, right = [part.strip() for part in line.split("|", 1)]
                    DrawPairRule.objects.create(
                        draw_event=event,
                        team_a=created_teams[left],
                        team_b=created_teams[right],
                    )

            messages.success(request, f"Draw event '{event.title}' created successfully.")
            return redirect("match_draw:detail", event_id=event.id)
    else:
        form = DrawSetupForm(initial={
            "mode": DrawEvent.MODE_RIGGED,
            "theme": DrawEvent.THEME_DARK,
            "teams_text": "Grade 6A\nGrade 6B\nGrade 5A\nGrade 5B\nGrade 4A\nGrade 4B",
            "rigged_pairs_text": "Grade 6A | Grade 6B\nGrade 5A | Grade 5B\nGrade 4A | Grade 4B",
        })

    recent_events = DrawEvent.objects.all()[:8]

    return render(request, "match_draw/home.html", {
        "page_title": "Match Draw Setup",
        "form": form,
        "recent_events": recent_events,
    })


@login_required
def draw_detail(request, event_id):
    event = get_object_or_404(
        DrawEvent.objects.prefetch_related(
            "teams",
            "pair_rules__team_a",
            "pair_rules__team_b",
            "matches__team_1",
            "matches__team_2",
        ),
        id=event_id,
    )

    return render(request, "match_draw/detail.html", {
        "page_title": event.title,
        "event": event,
        "teams": event.teams.all(),
        "pair_rules": event.pair_rules.all(),
        "matches": event.matches.all(),
    })


@login_required
def start_draw(request, event_id):
    if request.method != "POST":
        return redirect("match_draw:detail", event_id=event_id)

    event = get_object_or_404(DrawEvent, id=event_id)

    try:
        generate_matches_for_event(event)
        messages.success(request, f"Draw started for '{event.title}'. Matches generated successfully.")
    except DrawGenerationError as exc:
        messages.error(request, str(exc))

    return redirect("match_draw:detail", event_id=event.id)


@login_required
def live_draw(request, event_id):
    event = get_object_or_404(
        DrawEvent.objects.prefetch_related("matches__team_1", "matches__team_2"),
        id=event_id,
    )

    matches = list(event.matches.all().order_by("match_order", "id"))

    serialized_matches = [
        {
            "match_order": m.match_order,
            "team_1": m.team_1.name,
            "team_2": m.team_2.name,
        }
        for m in matches
    ]

    return render(request, "match_draw/live.html", {
        "page_title": f"{event.title} - Live Ceremony",
        "event": event,
        "matches_json": serialized_matches,
    })


@login_required
def live_draw_data(request, event_id):
    event = get_object_or_404(
        DrawEvent.objects.prefetch_related("matches__team_1", "matches__team_2"),
        id=event_id,
    )

    matches = list(event.matches.all().order_by("match_order", "id"))

    payload = {
        "event": {
            "id": event.id,
            "title": event.title,
            "theme": event.theme,
        },
        "matches": [
            {
                "match_order": m.match_order,
                "team_1": m.team_1.name,
                "team_2": m.team_2.name,
            }
            for m in matches
        ],
    }
    return JsonResponse(payload)