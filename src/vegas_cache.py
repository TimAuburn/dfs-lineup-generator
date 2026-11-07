"""Cached NFL Vegas lines from The Odds API.

The cache lives at ``data/vegas_cache.json`` so Streamlit reruns can reuse a
fresh file instead of spending an API request. Sunday morning uses a short
TTL; every other window keeps lines for six hours.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, time as clock_time, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "data" / "vegas_cache.json"
# Bump when the meaning of cached ``itt`` changes. Schema 1 (and files with
# no schema) stored the reversed favorite/underdog split and must not be read.
CACHE_SCHEMA_VERSION = 2
OUTDATED_CACHE_STATUS = "Vegas odds cache is out of date and was not used."
NO_USABLE_ODDS_WARNING = (
    "The odds feed returned no usable games, so no implied team totals were applied."
)

ODDS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
SLEEPER_STATE_URL = "https://api.sleeper.app/v1/state/nfl"
SUNDAY_TTL_SECONDS = 30 * 60
DEFAULT_TTL_SECONDS = 6 * 60 * 60
# Sleeper's season_start_date is the Wednesday that opens week 1.
NFL_WEEK_DAYS = 7

# Odds API full names -> DraftKings abbreviations.
TEAM_ABBREVIATIONS = {
    "arizona cardinals": "ARI",
    "atlanta falcons": "ATL",
    "baltimore ravens": "BAL",
    "buffalo bills": "BUF",
    "carolina panthers": "CAR",
    "chicago bears": "CHI",
    "cincinnati bengals": "CIN",
    "cleveland browns": "CLE",
    "dallas cowboys": "DAL",
    "denver broncos": "DEN",
    "detroit lions": "DET",
    "green bay packers": "GB",
    "houston texans": "HOU",
    "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC",
    "las vegas raiders": "LV",
    "los angeles chargers": "LAC",
    "los angeles rams": "LAR",
    "miami dolphins": "MIA",
    "minnesota vikings": "MIN",
    "new england patriots": "NE",
    "new orleans saints": "NO",
    "new york giants": "NYG",
    "new york jets": "NYJ",
    "philadelphia eagles": "PHI",
    "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF",
    "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN",
    "washington commanders": "WAS",
    "washington football team": "WAS",
    "oakland raiders": "LV",
    "san diego chargers": "LAC",
    "st. louis rams": "LAR",
}


def team_abbreviation(name: str) -> Optional[str]:
    """Map an Odds API team name to a DraftKings abbreviation."""
    key = " ".join(str(name).strip().lower().split())
    if not key:
        return None
    if key in TEAM_ABBREVIATIONS:
        return TEAM_ABBREVIATIONS[key]
    upper = key.upper()
    if upper in set(TEAM_ABBREVIATIONS.values()):
        return upper
    return None


def cache_ttl_seconds(now: Optional[datetime] = None) -> int:
    """30 minutes on Sunday 8:00 AM–1:00 PM local time, otherwise 6 hours."""
    current = now or datetime.now()
    start = clock_time(8, 0)
    end = clock_time(13, 0)
    if current.weekday() == 6 and start <= current.time() < end:
        return SUNDAY_TTL_SECONDS
    return DEFAULT_TTL_SECONDS


def implied_team_total(total: float, spread: float) -> float:
    """Implied points for one team.

    ``spread`` is that team's American line, the Odds API ``point``: negative
    for the favorite and positive for the underdog. The share is
    ``(game_total - signed_spread) / 2``, so the favorite is higher and the
    two teams add back to the game total. Subtracting the absolute spread
    for a negative line did the reverse (total 50 and -10 became 20 vs 30).
    """
    return (float(total) - float(spread)) / 2.0


def missing_odds_warning(game: str) -> str:
    """User-facing note that one game had no usable totals/spreads market."""
    return (
        f"No odds market for {game}. "
        "Implied team totals were not applied to this game."
    )


class ParsedOdds(dict):
    """Team lines from one payload, with coverage warnings attached."""

    def __init__(self, lines: dict, warnings: list[str]):
        super().__init__(lines)
        self.warnings = list(warnings)


def _load_json(path: Path) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _cache_payload_is_readable(payload: dict) -> bool:
    if "timestamp" not in payload or not isinstance(payload.get("data"), dict):
        return False
    try:
        float(payload["timestamp"])
    except (TypeError, ValueError):
        return False
    return True


def _read_cache(path: Path) -> Optional[dict]:
    """Return a current-schema cache, or None when the file cannot be used.

    Files written before ``CACHE_SCHEMA_VERSION`` stored reversed implied
    totals. They are rejected even as an offline fallback.
    """
    payload = _load_json(path)
    if payload is None or not _cache_payload_is_readable(payload):
        return None
    if payload.get("schema") != CACHE_SCHEMA_VERSION:
        return None
    return payload


def cache_is_outdated(path: Path) -> bool:
    """True when a readable cache file uses an older implied-total schema."""
    payload = _load_json(path)
    if payload is None or not _cache_payload_is_readable(payload):
        return False
    return payload.get("schema") != CACHE_SCHEMA_VERSION


def _write_cache(
    path: Path,
    data: dict,
    *,
    timestamp: Optional[float] = None,
    warnings: Optional[list] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "timestamp": float(time.time() if timestamp is None else timestamp),
        "data": data,
        "warnings": [str(item) for item in (warnings or [])],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def normalize_vegas_memory(state: Optional[dict] = None) -> dict:
    """Session fields to keep after an implied-total schema check.

    Streamlit holds the odds editor in memory across reruns. A frame from
    before ``CACHE_SCHEMA_VERSION`` is dropped, and ``vegas_editor_version``
    increments so the widget key changes and the old cells are not restored.
    A brand-new session is stamped with the current schema and left at its
    editor version.
    """
    current = dict(state or {})
    schema = current.get("vegas_schema")
    frame = current.get("vegas_frame")
    source = current.get("vegas_source_json")
    try:
        version = int(current.get("vegas_editor_version", 0) or 0)
    except (TypeError, ValueError):
        version = 0
    if schema == CACHE_SCHEMA_VERSION:
        return {
            "vegas_schema": CACHE_SCHEMA_VERSION,
            "vegas_frame": frame,
            "vegas_source_json": source,
            "vegas_editor_version": version,
        }
    has_prior_lines = frame is not None or schema is not None or source not in (None, "")
    return {
        "vegas_schema": CACHE_SCHEMA_VERSION,
        "vegas_frame": None,
        "vegas_source_json": None,
        "vegas_editor_version": version + 1 if has_prior_lines else version,
    }


def cache_status(
    cache_path: Path = CACHE_PATH,
    *,
    now: Optional[float] = None,
) -> dict:
    """Describe the on-disk cache without calling the network."""
    payload = _read_cache(cache_path)
    if payload is None:
        return {"cached": False, "age_minutes": None, "timestamp": None, "fresh": False}
    stamp = float(payload["timestamp"])
    current = time.time() if now is None else float(now)
    age_seconds = current - stamp
    age_minutes = max(0.0, age_seconds / 60.0)
    ttl = cache_ttl_seconds(datetime.fromtimestamp(current))
    fresh = 0 <= age_seconds <= ttl
    return {
        "cached": True,
        "age_minutes": age_minutes,
        "timestamp": stamp,
        "fresh": fresh,
    }


def format_cache_status(
    cache_path: Path = CACHE_PATH,
    *,
    now: Optional[float] = None,
) -> str:
    if cache_is_outdated(cache_path):
        return OUTDATED_CACHE_STATUS
    status = cache_status(cache_path, now=now)
    if not status["cached"]:
        return "No data cached"
    minutes = int(status["age_minutes"])
    payload = _read_cache(cache_path)
    data = payload.get("data") if isinstance(payload, dict) else None
    if not data:
        return f"No Vegas lines cached (Last updated: {minutes} minutes ago)."
    return f"Vegas Data: Cached (Last updated: {minutes} minutes ago)"


def _pick_bookmaker(bookmakers: list) -> Optional[dict]:
    preferred = ("draftkings", "fanduel", "betmgm", "williamhill_us")
    by_key = {str(book.get("key", "")): book for book in bookmakers if isinstance(book, dict)}
    for key in preferred:
        if key in by_key:
            return by_key[key]
    return bookmakers[0] if bookmakers else None


def _market_outcomes(book: dict, market_key: str) -> list:
    for market in book.get("markets") or []:
        if market.get("key") == market_key:
            outcomes = market.get("outcomes") or []
            return outcomes if isinstance(outcomes, list) else []
    return []


def nfl_week_window(season_start: str, week: int) -> tuple[datetime, datetime]:
    """UTC window for one regular-season week, Wednesday through the next Wednesday."""
    start_day = date.fromisoformat(str(season_start)[:10])
    begin_day = start_day + timedelta(days=(int(week) - 1) * NFL_WEEK_DAYS)
    begin = datetime.combine(begin_day, clock_time.min, tzinfo=timezone.utc)
    return begin, begin + timedelta(days=NFL_WEEK_DAYS)


def _commence_time(event: dict) -> Optional[datetime]:
    raw = event.get("commence_time")
    if not raw:
        return None
    text = str(raw).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _week_window_from_state(state: object) -> Optional[tuple[datetime, datetime]]:
    """Current regular-season week from the Sleeper NFL state payload."""
    if not isinstance(state, dict):
        return None
    if str(state.get("season_type") or "regular") != "regular":
        return None
    start = state.get("season_start_date")
    week = state.get("week")
    if not start or not week:
        return None
    try:
        return nfl_week_window(str(start), int(week))
    except (TypeError, ValueError):
        return None


def current_nfl_week_window(_now: Optional[datetime] = None) -> Optional[tuple[datetime, datetime]]:
    """Week window for ``now``. A Sleeper miss leaves the caller unfiltered."""
    try:
        state = _http_get(SLEEPER_STATE_URL)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return None
    return _week_window_from_state(state)


def _in_window(
    commence: Optional[datetime],
    commence_from: Optional[datetime],
    commence_to: Optional[datetime],
) -> bool:
    if commence_from is None and commence_to is None:
        return True
    if commence is None:
        return False
    if commence_from is not None and commence < commence_from:
        return False
    if commence_to is not None and commence >= commence_to:
        return False
    return True


def parse_odds_payload(
    payload: object,
    *,
    commence_from: Optional[datetime] = None,
    commence_to: Optional[datetime] = None,
) -> ParsedOdds:
    """Turn an Odds API response into ``{ABBR: {total, itt, spread, game}}``.

    Games outside ``[commence_from, commence_to)`` are dropped. The feed lists
    the next week as soon as those lines exist, and a later kickoff used to
    replace the current week's opponent for the same team.

    An in-window game with no totals or spreads market is omitted and named
    on ``warnings``. No total, spread, or implied total is filled in for it.
    ``spread`` is the team's own Odds API point (negative for the favorite).
    """
    if not isinstance(payload, list):
        raise ValueError("Odds API payload must be a list of games")
    lines: dict = {}
    warnings: list[str] = []
    events = [event for event in payload if isinstance(event, dict)]
    events.sort(key=lambda event: _commence_time(event) or datetime.max.replace(tzinfo=timezone.utc))
    for event in events:
        if not _in_window(_commence_time(event), commence_from, commence_to):
            continue
        home_name = str(event.get("home_team") or "")
        away_name = str(event.get("away_team") or "")
        home = team_abbreviation(home_name)
        away = team_abbreviation(away_name)
        if not home or not away:
            continue
        if home in lines or away in lines:
            continue
        game = f"{away}@{home}"
        bookmakers = event.get("bookmakers") or []
        book = _pick_bookmaker(bookmakers if isinstance(bookmakers, list) else [])
        total_outcomes = _market_outcomes(book, "totals") if book else []
        spread_outcomes = _market_outcomes(book, "spreads") if book else []
        if book is None or not total_outcomes or not spread_outcomes:
            warnings.append(missing_odds_warning(game))
            continue
        try:
            total = float(total_outcomes[0]["point"])
        except (KeyError, TypeError, ValueError):
            warnings.append(missing_odds_warning(game))
            continue
        spreads: dict = {}
        spread_broken = False
        for outcome in spread_outcomes:
            abbr = team_abbreviation(str(outcome.get("name") or ""))
            if abbr is None:
                continue
            try:
                spreads[abbr] = float(outcome["point"])
            except (KeyError, TypeError, ValueError):
                spread_broken = True
                break
        if spread_broken or home not in spreads or away not in spreads:
            warnings.append(missing_odds_warning(game))
            continue
        for abbr in (home, away):
            spread = spreads[abbr]
            lines[abbr] = {
                "total": round(total, 2),
                "spread": round(spread, 2),
                "itt": round(implied_team_total(total, spread), 2),
                "game": game,
            }
    return ParsedOdds(lines, warnings)


def _http_get(url: str, timeout: float = 20) -> object:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _cached_data(path: Path) -> dict:
    payload = _read_cache(path)
    if payload is None:
        return {}
    data = payload.get("data")
    return data if isinstance(data, dict) else {}


def coverage_warnings(cache_path: Path = CACHE_PATH) -> list[str]:
    """Missing-market warnings stored with the current odds cache."""
    payload = _read_cache(cache_path)
    if payload is None:
        return []
    raw = payload.get("warnings")
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item).strip()]


def fetch_vegas_lines(
    api_key: Optional[str] = None,
    force_refresh: bool = False,
    *,
    cache_path: Path = CACHE_PATH,
    now: Optional[datetime] = None,
) -> dict:
    """
    Return implied team totals keyed by DraftKings abbreviation.

    A fresh cache file is returned without a network call. A missing key or a
    failed request falls back to the file on disk, or ``{}`` when there is none.
    """
    current = now or datetime.now()
    payload = _read_cache(cache_path)
    if payload is not None and not force_refresh:
        age = current.timestamp() - float(payload["timestamp"])
        if 0 <= age <= cache_ttl_seconds(current):
            data = payload.get("data")
            return data if isinstance(data, dict) else {}

    key = (api_key or "").strip()
    if not key:
        return _cached_data(cache_path)

    url = (
        f"{ODDS_URL}?apiKey={quote(key)}&regions=us&markets=totals,spreads"
    )
    window = current_nfl_week_window(current)
    try:
        body = _http_get(url)
        parsed = parse_odds_payload(
            body,
            commence_from=None if window is None else window[0],
            commence_to=None if window is None else window[1],
        )
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return _cached_data(cache_path)

    lines = dict(parsed)
    warnings = list(parsed.warnings)
    # A successful empty body used to save a blank file and look like a fresh
    # cache. Name that outcome so the caption is not the only message.
    if not lines and not warnings:
        warnings.append(NO_USABLE_ODDS_WARNING)
    _write_cache(
        cache_path,
        lines,
        timestamp=current.timestamp(),
        warnings=warnings,
    )
    return lines
