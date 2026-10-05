"""Cached NFL Vegas lines from The Odds API.

The cache lives at ``data/vegas_cache.json`` so Streamlit reruns can reuse a
fresh file instead of spending an API request. Sunday morning uses a short
TTL; every other window keeps lines for six hours.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, time as clock_time
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "data" / "vegas_cache.json"

ODDS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
SUNDAY_TTL_SECONDS = 30 * 60
DEFAULT_TTL_SECONDS = 6 * 60 * 60

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
    """
    Implied team total from the game total and the team's signed spread.

    A negative spread is the favorite: ``(total / 2) - (|spread| / 2)``.
    A positive spread is the underdog: ``(total / 2) + (spread / 2)``.
    """
    total_f = float(total)
    spread_f = float(spread)
    if spread_f <= 0:
        return (total_f / 2.0) - (abs(spread_f) / 2.0)
    return (total_f / 2.0) + (spread_f / 2.0)


def _read_cache(path: Path) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if "timestamp" not in payload or not isinstance(payload.get("data"), dict):
        return None
    try:
        float(payload["timestamp"])
    except (TypeError, ValueError):
        return None
    return payload


def _write_cache(path: Path, data: dict, *, timestamp: Optional[float] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": float(time.time() if timestamp is None else timestamp), "data": data}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    status = cache_status(cache_path, now=now)
    if not status["cached"]:
        return "No data cached"
    minutes = int(status["age_minutes"])
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


def parse_odds_payload(payload: object) -> dict:
    """Turn an Odds API response into ``{ABBR: {total, itt, spread, game}}``."""
    if not isinstance(payload, list):
        raise ValueError("Odds API payload must be a list of games")
    lines: dict = {}
    for event in payload:
        if not isinstance(event, dict):
            continue
        home_name = str(event.get("home_team") or "")
        away_name = str(event.get("away_team") or "")
        home = team_abbreviation(home_name)
        away = team_abbreviation(away_name)
        if not home or not away:
            continue
        book = _pick_bookmaker(event.get("bookmakers") or [])
        if book is None:
            continue
        total_outcomes = _market_outcomes(book, "totals")
        spread_outcomes = _market_outcomes(book, "spreads")
        if not total_outcomes or not spread_outcomes:
            continue
        try:
            total = float(total_outcomes[0]["point"])
        except (KeyError, TypeError, ValueError):
            continue
        spreads: dict = {}
        for outcome in spread_outcomes:
            abbr = team_abbreviation(str(outcome.get("name") or ""))
            if abbr is None:
                continue
            try:
                spreads[abbr] = float(outcome["point"])
            except (KeyError, TypeError, ValueError):
                continue
        if home not in spreads or away not in spreads:
            continue
        game = f"{away}@{home}"
        for abbr in (home, away):
            spread = spreads[abbr]
            lines[abbr] = {
                "total": round(total, 2),
                "spread": round(spread, 2),
                "itt": round(implied_team_total(total, spread), 2),
                "game": game,
            }
    return lines


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
    try:
        body = _http_get(url)
        lines = parse_odds_payload(body)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return _cached_data(cache_path)

    _write_cache(cache_path, lines, timestamp=current.timestamp())
    return lines
