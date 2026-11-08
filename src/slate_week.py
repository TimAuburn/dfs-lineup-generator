"""Map an uploaded DraftKings slate to one NFL season and week.

The mapping is a lookup against scheduled games. It is not the Sleeper
current-week feed, and it is not ``(date - season start) // 7``. January
dates stay on the season the schedule names. A slate that covers two weeks,
or a date the schedule does not list, is reported and not given a guessed
week.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

# DraftKings lists kickoff dates in US Eastern time. ESPN instants are UTC.
EASTERN = ZoneInfo("America/New_York")

# Codes that show up on ESPN, Sleeper, or older DraftKings files.
_TEAM_ALIASES = {
    "WSH": "WAS",
    "JAC": "JAX",
    "LA": "LAR",
    "STL": "LAR",
    "SD": "LAC",
    "OAK": "LV",
}

# ESPN season.type on a scoreboard event.
ESPN_SEASON_TYPES = {1: "pre", 2: "regular", 3: "post"}

_GAME_RE = re.compile(
    r"(?P<away>[A-Za-z]{2,4})\s*@\s*(?P<home>[A-Za-z]{2,4})"
    r"(?:\s+(?P<date>\d{1,2}/\d{1,2}/\d{2,4}))?",
    re.IGNORECASE,
)


def dk_team(team: object) -> str:
    """DraftKings abbreviation for an ESPN, Sleeper, or DK team code."""
    abbrev = str(team or "").strip().upper()
    if abbrev in {"NAN", "NONE", "NAT"}:
        return ""
    return _TEAM_ALIASES.get(abbrev, abbrev)


@dataclass(frozen=True)
class SlateGame:
    """One matchup parsed from a DraftKings ``Game Info`` cell."""

    away: str
    home: str
    game_date: Optional[date]
    raw: str


@dataclass(frozen=True)
class ScheduledGame:
    """One real NFL game from a schedule payload or a supplied fixture."""

    game_date: date
    season: str
    week: int
    season_type: str
    away: str
    home: str


def parse_game_info(raw: object) -> Optional[SlateGame]:
    """Parse ``AWAY@HOME MM/DD/YYYY``. A missing date stays missing."""
    if raw is None:
        return None
    if isinstance(raw, float) and raw != raw:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "in progress", "final"}:
        return None
    match = _GAME_RE.search(text)
    if not match:
        return None
    away = dk_team(match.group("away"))
    home = dk_team(match.group("home"))
    if not away or not home:
        return None
    return SlateGame(
        away=away,
        home=home,
        game_date=_parse_mdy(match.group("date")),
        raw=text,
    )


def slate_games_from_pool(pool: object) -> list[SlateGame]:
    """Unique matchups on a parsed salary frame, in first-seen order."""
    columns = getattr(pool, "columns", None)
    if columns is None or "Game Info" not in columns or getattr(pool, "empty", True):
        return []
    seen: set[tuple] = set()
    games: list[SlateGame] = []
    for raw in pool["Game Info"].tolist():
        game = parse_game_info(raw)
        if game is None:
            continue
        key = (game.away, game.home, game.game_date)
        if key in seen:
            continue
        seen.add(key)
        games.append(game)
    return games


def parse_espn_scoreboard(payload: object) -> list[ScheduledGame]:
    """Games from an ESPN NFL scoreboard, dated in US Eastern time.

    The week and season are the values on each event. They are not derived
    from the calendar date.
    """
    if not isinstance(payload, dict):
        return []
    top_week = payload.get("week") if isinstance(payload.get("week"), dict) else {}
    games: list[ScheduledGame] = []
    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            continue
        season = event.get("season") if isinstance(event.get("season"), dict) else {}
        season_type = ESPN_SEASON_TYPES.get(season.get("type"))
        event_week = event.get("week") if isinstance(event.get("week"), dict) else {}
        week = event_week.get("number", top_week.get("number"))
        year = season.get("year")
        if season_type is None or week is None or year is None:
            continue
        try:
            week_num = int(week)
            season_year = str(int(year))
        except (TypeError, ValueError):
            continue
        if week_num < 1:
            continue
        competitions = event.get("competitions") or []
        if not competitions or not isinstance(competitions[0], dict):
            continue
        competition = competitions[0]
        kickoff = _eastern_date(competition.get("date") or event.get("date"))
        if kickoff is None:
            continue
        away = home = ""
        for competitor in competition.get("competitors") or []:
            if not isinstance(competitor, dict):
                continue
            team = competitor.get("team") if isinstance(competitor.get("team"), dict) else {}
            abbrev = dk_team(team.get("abbreviation"))
            side = str(competitor.get("homeAway") or "").lower()
            if side == "home":
                home = abbrev
            elif side == "away":
                away = abbrev
        if not away or not home:
            continue
        games.append(
            ScheduledGame(
                game_date=kickoff,
                season=season_year,
                week=week_num,
                season_type=season_type,
                away=away,
                home=home,
            )
        )
    return games


def resolve_slate_week(slate: list[SlateGame], scheduled: list[ScheduledGame]) -> dict:
    """Return one season and week, or an explicit reason not to pick one."""
    if not slate:
        return _failure(
            "missing_game_dates",
            "No game dates were found on the uploaded slate. The current NFL week was not used.",
        )
    dated = [game for game in slate if game.game_date is not None]
    undated = [game for game in slate if game.game_date is None]
    if not dated:
        return _failure(
            "missing_game_dates",
            "Slate games have teams but no dates. The current NFL week was not used.",
            undated_games=_labels(undated),
        )

    matched: list[tuple[str, str, int]] = []
    unmapped: list[str] = []
    team_mismatches: list[str] = []
    ambiguous_dates: list[str] = []
    for game in dated:
        on_date = [row for row in scheduled if row.game_date == game.game_date]
        hits = [row for row in on_date if {row.away, row.home} == {game.away, game.home}]
        if not on_date:
            unmapped.append(_label(game))
            continue
        if not hits:
            team_mismatches.append(_label(game))
            continue
        weeks = {(row.season, row.season_type, int(row.week)) for row in hits}
        if len(weeks) != 1:
            ambiguous_dates.append(_label(game))
            continue
        matched.append(weeks.pop())

    if unmapped:
        return _failure(
            "unmapped_game_dates",
            "These slate dates are not on the NFL schedule. The current NFL week was not used: "
            + ", ".join(unmapped),
            unmapped_games=unmapped,
        )
    if team_mismatches:
        return _failure(
            "slate_game_not_on_schedule",
            "These matchups are not on the NFL schedule for their listed date. "
            "The current NFL week was not used: " + ", ".join(team_mismatches),
            unmapped_games=team_mismatches,
        )
    if ambiguous_dates:
        return _failure(
            "ambiguous_game_date",
            "A slate date matched more than one NFL week: " + ", ".join(ambiguous_dates),
            ambiguous_games=ambiguous_dates,
        )
    distinct = set(matched)
    if len(distinct) != 1:
        detail = ", ".join(
            f"{season} {season_type} week {week}"
            for season, season_type, week in sorted(distinct, key=lambda item: (item[0], item[1], item[2]))
        )
        return _failure(
            "ambiguous_slate_week",
            "The slate covers more than one NFL week (" + detail + "). No week was chosen.",
            weeks=[list(item) for item in sorted(distinct)],
        )

    season, season_type, week = distinct.pop()
    if season_type not in {"pre", "regular", "post"}:
        return _failure(
            "unsupported_season_type",
            f"Schedule season type {season_type} is not a projection week.",
        )

    unconfirmed = _unconfirmed_undated(undated, scheduled, season, season_type, week)
    if unconfirmed:
        return _failure(
            "partial_game_dates",
            "Some slate games have no date and are not on the resolved week's schedule. "
            "No week was applied: " + ", ".join(unconfirmed),
            undated_games=_labels(undated),
            dated_games=_labels(dated),
        )

    message = (
        f"Slate week {season} {season_type} week {week} comes from the uploaded game dates."
    )
    if undated:
        message += " Undated matchups on this week's schedule: " + ", ".join(_labels(undated)) + "."
    return {
        "ok": True,
        "error": None,
        "message": message,
        "season": season,
        "week": int(week),
        "season_type": season_type,
        "undated_games": _labels(undated),
        "dated_games": _labels(dated),
    }


def validate_projection_records(
    records: object,
    slate: list[SlateGame],
    season: str,
    week: int,
    season_type: str,
) -> dict:
    """Check that a projection payload is the slate week, not another week.

    When the payload includes game dates and opponents, every dated slate
    game must appear. When it does not, the week label can still match and
    the missing dates are reported. Team confirmation is not invented.
    """
    expected = (str(season), str(season_type), int(week))
    if not isinstance(records, list) or not records:
        return {
            "ok": False,
            "error": "empty_projections",
            "message": (
                f"The projection source returned no players for {season} "
                f"{season_type} week {week}."
            ),
            "dates_confirmed": False,
        }

    declared: set[tuple] = set()
    dated_games: set[tuple] = set()
    for row in records:
        if not isinstance(row, dict):
            continue
        if row.get("season") is not None or row.get("week") is not None or row.get("season_type") is not None:
            try:
                declared.add(
                    (str(row.get("season")), str(row.get("season_type")), int(row.get("week") or 0))
                )
            except (TypeError, ValueError):
                declared.add(("unreadable", "unreadable", -1))
        team = dk_team(row.get("team"))
        opponent = dk_team(row.get("opponent"))
        raw_date = row.get("date")
        if team and opponent and raw_date:
            dated_games.add((frozenset({team, opponent}), str(raw_date)[:10]))

    if not declared or any(item != expected for item in declared):
        found = ", ".join(
            f"{item[0]} {item[1]} week {item[2]}" for item in sorted(declared, key=str)
        ) or "none"
        return {
            "ok": False,
            "error": "projection_week_mismatch",
            "message": (
                f"Projection rows are {found}, not {season} {season_type} week {week}. "
                "They were not applied."
            ),
            "dates_confirmed": False,
            "declared": [list(item) for item in sorted(declared, key=str)],
        }

    missing = []
    for game in slate:
        if game.game_date is None:
            continue
        key = (frozenset({game.away, game.home}), game.game_date.isoformat())
        if dated_games and key not in dated_games:
            missing.append(_label(game))
    if missing:
        return {
            "ok": False,
            "error": "projection_slate_mismatch",
            "message": (
                "Projection game dates do not include the uploaded slate: "
                + ", ".join(missing)
                + ". Those projections were not applied."
            ),
            "dates_confirmed": True,
            "missing_games": missing,
        }
    if dated_games:
        return {
            "ok": True,
            "error": None,
            "message": "Sleeper projection game dates match the uploaded slate.",
            "dates_confirmed": True,
        }
    return {
        "ok": True,
        "error": None,
        "message": (
            "The projection source did not include game dates or opponents. "
            f"Its week label matches {season} {season_type} week {week}. "
            "Team-by-team confirmation was unavailable."
        ),
        "dates_confirmed": False,
    }


def slate_fingerprint(slate: list[SlateGame], season: str, week: int, season_type: str) -> str:
    """Cache identity for one slate week. A different week does not match."""
    return f"v1|{season}|{season_type}|{int(week)}|{slate_key(slate)}"


def slate_key(slate: list[SlateGame]) -> str:
    """Matchup identity without a week, used only to recognize the same slate."""
    parts = sorted(
        f"{game.game_date.isoformat() if game.game_date else ''}:{game.away}@{game.home}"
        for game in slate
    )
    return "|".join(parts)


def coerce_schedule(schedule_games: object) -> list[ScheduledGame]:
    """Accept ``ScheduledGame`` values or plain dicts from a fixture."""
    if not schedule_games:
        return []
    games: list[ScheduledGame] = []
    for row in schedule_games:
        if isinstance(row, ScheduledGame):
            games.append(row)
            continue
        if not isinstance(row, dict):
            continue
        raw_date = row.get("game_date")
        if isinstance(raw_date, datetime):
            parsed = raw_date.date()
        elif isinstance(raw_date, date):
            parsed = raw_date
        else:
            parsed = date.fromisoformat(str(raw_date)[:10])
        games.append(
            ScheduledGame(
                game_date=parsed,
                season=str(row["season"]),
                week=int(row["week"]),
                season_type=str(row["season_type"]),
                away=dk_team(row["away"]),
                home=dk_team(row["home"]),
            )
        )
    return games


def _unconfirmed_undated(
    undated: list[SlateGame],
    scheduled: list[ScheduledGame],
    season: str,
    season_type: str,
    week: int,
) -> list[str]:
    if not undated:
        return []
    pairs = {
        frozenset({row.away, row.home})
        for row in scheduled
        if row.season == season and row.season_type == season_type and int(row.week) == int(week)
    }
    return [_label(game) for game in undated if frozenset({game.away, game.home}) not in pairs]


def _eastern_date(raw: object) -> Optional[date]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=EASTERN)
    return parsed.astimezone(EASTERN).date()


def _parse_mdy(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    parts = value.split("/")
    if len(parts) != 3:
        return None
    try:
        month, day, year = (int(part) for part in parts)
    except ValueError:
        return None
    if year < 100:
        year += 2000
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _label(game: SlateGame) -> str:
    when = game.game_date.isoformat() if game.game_date else "no date"
    return f"{game.away}@{game.home} {when}"


def _labels(games: list[SlateGame]) -> list[str]:
    return [_label(game) for game in games]


def _failure(code: str, message: str, **extra: object) -> dict:
    payload = {
        "ok": False,
        "error": code,
        "message": message,
        "season": None,
        "week": None,
        "season_type": None,
        "undated_games": [],
        "dated_games": [],
    }
    payload.update(extra)
    return payload
