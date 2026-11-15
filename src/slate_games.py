"""Slate games shown on the player-pool page, and the on/off generation filter.

The total and the spread come from the same Odds API lines the lineup rules
read. Each team line has ``total`` (the game over/under) and ``spread`` (that
team's own signed number, negative for the favorite). A missing number stays
missing. Nothing here is filled in with zero, and projections are not edited.
"""

from __future__ import annotations

from datetime import datetime
from typing import Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from .lineup_rules import ConstructionSettings, active_game_shortfall, normalize_spreads
from .optimizer import games_from_pool
from .player_pool import game_label, generation_pool, person_key
from .vegas_cache import CACHE_PATH as VEGAS_CACHE_PATH
from .vegas_cache import cache_status

ODDS_SOURCE = "The Odds API"
MISSING = "missing"
EASTERN = ZoneInfo("America/New_York")

_KICKOFF_ZONES = {
    "ET": "America/New_York",
    "EST": "America/New_York",
    "EDT": "America/New_York",
    "CT": "America/Chicago",
    "CST": "America/Chicago",
    "CDT": "America/Chicago",
    "MT": "America/Denver",
    "MST": "America/Denver",
    "MDT": "America/Denver",
    "PT": "America/Los_Angeles",
    "PST": "America/Los_Angeles",
    "PDT": "America/Los_Angeles",
}


def game_key(value: object) -> str:
    """``AWAY@HOME`` from a game label or a ``AWAY @ HOME`` matchup."""
    text = " ".join(str(value or "").strip().upper().split())
    if not text or text in {"NAN", "NONE", "NAT"}:
        return ""
    text = text.replace(" @ ", "@")
    if "@" not in text:
        return ""
    away, home = text.split("@", 1)
    away = away.strip()
    home = home.strip()
    if not away or not home or " " in away or " " in home:
        return ""
    return f"{away}@{home}"


def format_fetched(timestamp: Optional[float]) -> str:
    """Clock time for a cache stamp. A missing stamp stays 'not fetched'."""
    if timestamp is None:
        return "not fetched"
    try:
        moment = datetime.fromtimestamp(float(timestamp), tz=EASTERN)
    except (TypeError, ValueError, OSError, OverflowError):
        return "not fetched"
    return moment.strftime("%Y-%m-%d %I:%M %p ET")


def odds_fetched_at(cache_path=VEGAS_CACHE_PATH) -> Optional[float]:
    """When the Odds API cache was written, or None when it was not."""
    status = cache_status(cache_path)
    stamp = status.get("timestamp")
    if stamp is None:
        return None
    try:
        return float(stamp)
    except (TypeError, ValueError):
        return None


def odds_source_caption(fetched_at: Optional[float]) -> str:
    return (
        f"Odds source: {ODDS_SOURCE}. Fetched: {format_fetched(fetched_at)}. "
        "The total and the spread are the lines lineup rules use."
    )


def forecast_source_caption(fetched_at: Optional[float]) -> str:
    return (
        "Kickoff forecast: National Weather Service hourly (api.weather.gov). "
        f"Fetched: {format_fetched(fetched_at)}."
    )


class SlateMatchup:
    """One game parsed from the salary file, in first-seen order."""

    def __init__(
        self,
        key: str,
        away: str,
        home: str,
        kickoff: str,
        kickoff_at: Optional[datetime],
    ) -> None:
        self.key = key
        self.away = away
        self.home = home
        self.kickoff = kickoff
        self.kickoff_at = kickoff_at

    @property
    def matchup(self) -> str:
        return f"{self.away} @ {self.home}"


def kickoff_text(raw: object) -> str:
    """The date, clock, and zone DraftKings put after the matchup."""
    if raw is None:
        return MISSING
    try:
        if pd.isna(raw):
            return MISSING
    except (TypeError, ValueError):
        pass
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "in progress", "final"}:
        return MISSING
    if game_label(raw) == "":
        return MISSING
    parts = text.split(None, 1)
    if len(parts) < 2:
        return MISSING
    rest = parts[1].strip()
    return rest or MISSING


def kickoff_datetime(raw: object) -> Optional[datetime]:
    """Aware kickoff. DraftKings times are US Eastern unless a zone is named."""
    text = kickoff_text(raw)
    if text == MISSING:
        return None
    pieces = text.split()
    if len(pieces) < 2:
        return None
    date_text, clock_text = pieces[0], pieces[1].upper().replace(" ", "")
    zone_name = pieces[2].upper() if len(pieces) > 2 else "ET"
    zone_key = _KICKOFF_ZONES.get(zone_name)
    if zone_key is None:
        return None
    for date_fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            parsed = datetime.strptime(f"{date_text} {clock_text}", f"{date_fmt} %I:%M%p")
        except ValueError:
            continue
        return parsed.replace(tzinfo=ZoneInfo(zone_key))
    return None


def matchups_from_pool(pool: Optional[pd.DataFrame]) -> list[SlateMatchup]:
    """Unique games on a salary file, in first-seen order."""
    if pool is None or pool.empty or "Game Info" not in pool.columns:
        return []
    found: list[SlateMatchup] = []
    seen: set[str] = set()
    for raw in pool["Game Info"].tolist():
        key = game_label(raw)
        if not key or key in seen:
            continue
        seen.add(key)
        away, home = key.split("@", 1)
        found.append(
            SlateMatchup(
                key=key,
                away=away,
                home=home,
                kickoff=kickoff_text(raw),
                kickoff_at=kickoff_datetime(raw),
            )
        )
    return found


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _team_line(lines: object, team: str) -> Optional[dict]:
    if not isinstance(lines, Mapping):
        return None
    direct = lines.get(team)
    if isinstance(direct, Mapping):
        return dict(direct)
    for key, value in lines.items():
        if str(key).strip().upper() == team and isinstance(value, Mapping):
            return dict(value)
    return None


def display_total(lines: object, away: str, home: str) -> str:
    """Shared over/under, or ``missing`` when the source has no single total."""
    totals: list[float] = []
    for team in (away, home):
        info = _team_line(lines, team)
        if info is None or "total" not in info:
            continue
        value = _as_float(info.get("total"))
        if value is None:
            continue
        totals.append(value)
    if not totals:
        return MISSING
    first = round(totals[0], 2)
    if any(round(value, 2) != first for value in totals):
        return MISSING
    return f"{totals[0]:.1f}"


def display_spread(lines: object, away: str, home: str) -> str:
    """Favorite and signed line, or ``missing`` when either spread is absent."""
    away_info = _team_line(lines, away)
    home_info = _team_line(lines, home)
    if (
        away_info is None
        or home_info is None
        or "spread" not in away_info
        or "spread" not in home_info
    ):
        return MISSING
    away_spread = _as_float(away_info.get("spread"))
    home_spread = _as_float(home_info.get("spread"))
    if away_spread is None or home_spread is None:
        return MISSING
    if round(away_spread + home_spread, 2) != 0:
        return MISSING
    if away_spread < home_spread:
        favorite, line = away, away_spread
    elif home_spread < away_spread:
        favorite, line = home, home_spread
    else:
        return "Pick'em"
    return f"{favorite} {line:.1f}"


def _total_sort_number(lines: object, game: SlateMatchup) -> Optional[float]:
    """Displayed total as a number. ``missing`` stays missing."""
    shown = display_total(lines, game.away, game.home)
    if shown == MISSING:
        return None
    return _as_float(shown)


def _total_order_key(game: SlateMatchup, lines: object) -> tuple:
    """Highest total first. A missing total is last, then kickoff, then matchup."""
    number = _total_sort_number(lines, game)
    kickoff = game.kickoff_at if isinstance(game.kickoff_at, datetime) else None
    return (
        number is None,
        -(number if number is not None else 0.0),
        kickoff is None,
        kickoff.timestamp() if kickoff is not None else 0.0,
        game.kickoff,
        game.matchup,
    )


def order_matchups_by_total(
    matchups: Sequence[SlateMatchup],
    lines: object,
) -> list[SlateMatchup]:
    """Highest total first. Missing totals last. The total text is not filled in.

    Equal totals break by kickoff, earlier first. A missing kickoff sorts after
    a known kickoff at that total. The last key is the matchup text.
    """
    return sorted(matchups, key=lambda game: _total_order_key(game, lines))


def order_game_labels_by_total(
    labels: Sequence[str],
    matchups: Sequence[SlateMatchup],
    lines: object,
) -> list[str]:
    """Game-filter labels in the same order as the games table.

    A label that is not on the slate stays after the slate games, in its
    original place relative to the other leftovers.
    """
    rank = {
        game.key: index
        for index, game in enumerate(order_matchups_by_total(matchups, lines))
    }
    paired = list(enumerate(labels))
    paired.sort(key=lambda item: (rank.get(game_key(item[1]), len(rank)), item[0]))
    return [label for _, label in paired]


def games_table_frame(
    matchups: Sequence[SlateMatchup],
    lines: object,
    weather: Optional[Mapping[str, str]] = None,
    disabled: Optional[Sequence[str]] = None,
    *,
    showdown: bool = False,
) -> pd.DataFrame:
    """One row per game, highest total first.

    A missing total stays the text ``missing`` and sorts last. Equal totals
    break by kickoff (earlier first), then by matchup. Totals and spreads
    are text so a gap is not a number. ``On`` travels with the matchup.
    """
    off = {game_key(item) for item in (disabled or [])}
    labels = weather or {}
    rows = []
    for game in order_matchups_by_total(matchups, lines):
        row = {
            "Matchup": game.matchup,
            "Kickoff": game.kickoff,
            "Total": display_total(lines, game.away, game.home),
            "Spread": display_spread(lines, game.away, game.home),
            "Weather": labels.get(game.key) or "Forecast unavailable",
        }
        if not showdown:
            row = {"On": game.key not in off, **row}
        rows.append(row)
    columns = ["Matchup", "Kickoff", "Total", "Spread", "Weather"]
    if not showdown:
        columns = ["On", *columns]
    return pd.DataFrame(rows, columns=columns)


def disabled_for_slate(
    store: Optional[Mapping],
    slate_key: str,
    games: Sequence[str],
) -> list[str]:
    """Off games saved for this slate. Another slate's list is not reused."""
    if not isinstance(store, Mapping):
        return []
    saved = store.get(str(slate_key))
    if not isinstance(saved, (list, tuple)):
        return []
    known = {game_key(game) for game in games}
    chosen: list[str] = []
    for item in saved:
        key = game_key(item)
        if key in known and key not in chosen:
            chosen.append(key)
    return chosen


def with_disabled(
    store: Optional[Mapping],
    slate_key: str,
    disabled: Sequence[str],
) -> dict:
    """Copy of the selection store with this slate's off games replaced."""
    updated = dict(store) if isinstance(store, Mapping) else {}
    chosen: list[str] = []
    for item in disabled:
        key = game_key(item)
        if key and key not in chosen:
            chosen.append(key)
    updated[str(slate_key)] = chosen
    return updated


def pool_without_games(
    pool: pd.DataFrame,
    disabled: Sequence[str],
    *,
    showdown: bool = False,
) -> pd.DataFrame:
    """Drop every player, including DST, from games that are off.

    Showdown is one game and is returned unchanged. An empty off-list returns
    the same frame. Projections are not edited.
    """
    if showdown or pool is None or pool.empty:
        return pool
    off = {game_key(item) for item in disabled if game_key(item)}
    if not off or "Game Info" not in pool.columns:
        return pool
    labels = pool["Game Info"].map(game_label)
    return pool.loc[~labels.isin(off)].copy()


def generation_pool_for_games(
    pool: pd.DataFrame,
    disabled: Sequence[str],
    *,
    showdown: bool,
) -> pd.DataFrame:
    """Included players, then the Classic game filter."""
    eligible = generation_pool(pool, showdown=showdown)
    if showdown:
        return eligible
    return pool_without_games(eligible, disabled)


def _as_bool(value: object, *, default: bool) -> bool:
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return bool(value)


def _has_pin(row: pd.Series) -> bool:
    if "ProjectionOverride" not in row.index:
        return False
    value = pd.to_numeric(row.get("ProjectionOverride"), errors="coerce")
    return not pd.isna(value)


def _why(lock: bool, pin: bool, minimum: bool) -> str:
    if lock and pin and minimum:
        return "is locked, has a pinned projection, and is a min-exposure target"
    if lock and pin:
        return "is locked and has a pinned projection"
    if lock and minimum:
        return "is locked and a min-exposure target"
    if pin and minimum:
        return "has a pinned projection and is a min-exposure target"
    if lock:
        return "is locked"
    if pin:
        return "has a pinned projection"
    return "is a min-exposure target"


def off_game_blocks(
    pool: Optional[pd.DataFrame],
    disabled: Sequence[str],
    *,
    min_exposure_pct: Optional[float] = None,
) -> tuple[str, ...]:
    """Name locked, pinned, and min-exposure players who sit in a game that is off.

    Included players only. A player already removed with Include stays removed
    and is not named again. Min exposure uses the same percent the sidebar
    sends: a number above 0 marks every included player as a target.
    """
    off = {game_key(item) for item in disabled if game_key(item)}
    if not off or pool is None or pool.empty or "Game Info" not in pool.columns:
        return ()
    minimum = min_exposure_pct is not None and float(min_exposure_pct) > 0
    people: dict[tuple[str, str], dict] = {}
    for _, row in pool.iterrows():
        game = game_label(row.get("Game Info"))
        if game not in off:
            continue
        key = person_key(row.get("Name"), row.get("TeamAbbrev"), row.get("Position"))
        if not key[0]:
            continue
        slot = people.setdefault(
            key,
            {
                "name": str(row.get("Name") or "").strip(),
                "team": str(row.get("TeamAbbrev") or "").strip().upper(),
                "game": game,
                "include": False,
                "lock": False,
                "pin": False,
            },
        )
        if not _as_bool(row.get("Include"), default=True):
            continue
        slot["include"] = True
        if _as_bool(row.get("Lock"), default=False):
            slot["lock"] = True
        if _has_pin(row):
            slot["pin"] = True
    messages: list[str] = []
    for slot in people.values():
        if not slot["include"]:
            continue
        is_min = minimum
        if not slot["lock"] and not slot["pin"] and not is_min:
            continue
        label = slot["name"]
        if slot["team"]:
            label = f"{label} ({slot['team']})"
        reason = _why(slot["lock"], slot["pin"], is_min)
        messages.append(
            f"{label} {reason} in {slot['game']}, which is off. "
            "Turn that game on before generating."
        )
    if len(messages) <= 8:
        return tuple(messages)
    hidden = len(messages) - 8
    return tuple(
        [
            *messages[:8],
            f"And {hidden} more players are in a game that is off.",
        ]
    )


def spreads_for_pool(lines: object, pool: Optional[pd.DataFrame]) -> dict[str, float]:
    """Signed spreads for teams still in the pool. Other games are left out."""
    spreads = normalize_spreads(lines if isinstance(lines, Mapping) else None)
    if pool is None or pool.empty or "TeamAbbrev" not in pool.columns:
        return {}
    teams = {
        str(team).strip().upper()
        for team in pool["TeamAbbrev"]
        if str(team).strip() and str(team).strip().lower() != "nan"
    }
    return {team: value for team, value in spreads.items() if team in teams}


def generation_blockers(
    pool: Optional[pd.DataFrame],
    disabled: Sequence[str],
    *,
    showdown: bool,
    min_exposure_pct: Optional[float] = None,
    settings: Optional[ConstructionSettings] = None,
) -> tuple[str, ...]:
    """Pre-solve messages for Classic games that are off. Showdown has none."""
    if showdown:
        return ()
    off = {game_key(item) for item in disabled if game_key(item)}
    slate = {game.key for game in matchups_from_pool(pool)}
    removed = off & slate
    if not removed:
        return ()
    messages = list(
        off_game_blocks(pool, removed, min_exposure_pct=min_exposure_pct)
    )
    filtered = generation_pool_for_games(pool, removed, showdown=False)
    if settings is not None:
        messages.extend(active_game_shortfall(len(games_from_pool(filtered)), settings))
    if not messages and (filtered is None or filtered.empty):
        messages.append(
            "Every game is off, so there are no players left to generate. "
            "Turn at least one game on."
        )
    return tuple(messages)
