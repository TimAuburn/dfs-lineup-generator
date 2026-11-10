"""Kickoff-hour forecast from the National Weather Service hourly grid.

This is display only. It does not read the latest-observation endpoint and it
does not change a projection or the wind adjustment. Stadium coordinates and
roof type come from the local table in ``src/stadiums.py``, keyed by the
home team's DraftKings abbreviation. api.weather.gov requires a User-Agent
and does not use an API key.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .stadiums import get_stadium
from .weather import NWS_API_ROOT, NWS_USER_AGENT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "data" / "kickoff_forecast_cache.json"
SUCCESS_TTL_SECONDS = 60 * 60
ERROR_TTL_SECONDS = 15 * 60
# A sunny forecast with wind at least this strong uses the wind icon.
WIND_ICON_MPH = 20.0
UNAVAILABLE = "Forecast unavailable"

HttpGet = Callable[[str], object]


def nws_request(url: str) -> Request:
    """GET request with the User-Agent National Weather Service requires."""
    return Request(
        url,
        headers={
            "User-Agent": NWS_USER_AGENT,
            "Accept": "application/geo+json",
        },
    )


def _default_http_get(url: str, timeout: float = 20) -> object:
    request = nws_request(url)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class KickoffForecast:
    """One game's display forecast. ``summary`` is the cell text without an emoji."""

    def __init__(
        self,
        game: str,
        status: str,
        icon: str,
        summary: str,
        *,
        retractable: bool = False,
        fetched_at: Optional[float] = None,
        reason: str = "",
    ) -> None:
        self.game = game
        self.status = status
        self.icon = icon
        self.summary = summary
        self.retractable = retractable
        self.fetched_at = fetched_at
        self.reason = reason


def hourly_url_from_points(points: object) -> str:
    """Hourly grid URL from a points payload. Empty when the payload has neither."""
    if not isinstance(points, dict):
        return ""
    props = points.get("properties")
    if not isinstance(props, dict):
        return ""
    direct = str(props.get("forecastHourly") or "").strip()
    if direct:
        return direct
    grid = str(props.get("gridId") or "").strip()
    x = props.get("gridX")
    y = props.get("gridY")
    if not grid or x is None or y is None:
        return ""
    return f"{NWS_API_ROOT}/gridpoints/{grid}/{x},{y}/forecast/hourly"


def _parse_time(value: object) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def period_for_kickoff(periods: object, kickoff: Optional[datetime]) -> Optional[dict]:
    """The hourly period that contains kickoff, or None when it is outside."""
    if kickoff is None or not isinstance(periods, list):
        return None
    moment = kickoff if kickoff.tzinfo is not None else kickoff.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    for period in periods:
        if not isinstance(period, dict):
            continue
        start = _parse_time(period.get("startTime"))
        end = _parse_time(period.get("endTime"))
        if start is None or end is None:
            continue
        if start <= moment < end:
            return period
    return None


def _pop(period: dict) -> Optional[int]:
    node = period.get("probabilityOfPrecipitation")
    if not isinstance(node, dict) or node.get("value") is None:
        return None
    try:
        return int(round(float(node["value"])))
    except (TypeError, ValueError):
        return None


def _temperature(period: dict) -> Optional[float]:
    raw = period.get("temperature")
    if raw is None:
        return None
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None
    unit = str(period.get("temperatureUnit") or "F").strip().upper()
    if unit.startswith("C"):
        return number * 9.0 / 5.0 + 32.0
    return number


def _wind_numbers(text: str) -> list[float]:
    return [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]


def wind_phrase(raw: object) -> str:
    text = str(raw or "").strip()
    if not text:
        return "wind missing"
    numbers = _wind_numbers(text)
    if not numbers:
        return f"wind {text}"
    if len(numbers) == 1:
        shown = int(numbers[0]) if numbers[0].is_integer() else numbers[0]
        return f"wind {shown} mph"
    start = int(numbers[0]) if numbers[0].is_integer() else numbers[0]
    end = int(numbers[-1]) if numbers[-1].is_integer() else numbers[-1]
    return f"wind {start} to {end} mph"


def _conditions_kind(text: str, wind_mph: Optional[float]) -> str:
    folded = text.casefold()
    if any(word in folded for word in ("snow", "flurr", "blizzard", "sleet", "ice")):
        return "snow"
    if any(word in folded for word in ("rain", "shower", "drizzle", "thunder", "storm")):
        return "rain"
    windy = any(word in folded for word in ("wind", "breez", "bluster", "gust"))
    if windy or (wind_mph is not None and wind_mph >= WIND_ICON_MPH):
        return "wind"
    if any(word in folded for word in ("cloud", "overcast", "fog", "haze", "mist")):
        return "cloud"
    return "sun"


def describe_hourly_period(period: object) -> tuple[str, str]:
    """Icon name and one-line summary from one hourly forecast period.

    The icon is sun, cloud, rain, snow, or wind. A missing temperature, rain
    chance, or wind is named as missing and is not replaced with zero.
    """
    if not isinstance(period, dict):
        return "cloud", UNAVAILABLE
    conditions = str(period.get("shortForecast") or "").strip() or "Conditions missing"
    numbers = _wind_numbers(str(period.get("windSpeed") or ""))
    wind_mph = max(numbers) if numbers else None
    kind = _conditions_kind(conditions, wind_mph)
    temperature = _temperature(period)
    temp_text = "temperature missing" if temperature is None else f"{temperature:.0f}°F"
    chance = _pop(period)
    chance_name = "snow" if kind == "snow" else "rain"
    if chance is None:
        chance_text = f"{chance_name} chance missing"
    else:
        chance_text = f"{chance}% {chance_name}"
    summary = f"{conditions}, {temp_text}, {chance_text}, {wind_phrase(period.get('windSpeed'))}"
    return kind, summary


def _roof(home: str) -> str:
    stadium = get_stadium(home)
    if stadium is None:
        return "outdoor"
    return str(stadium.enclosure or "outdoor")


def _unavailable(
    game: str,
    reason: str,
    *,
    retractable: bool = False,
    fetched_at: Optional[float] = None,
) -> KickoffForecast:
    summary = UNAVAILABLE
    if retractable:
        summary = f"{UNAVAILABLE} · retractable roof"
    return KickoffForecast(
        game,
        "unavailable",
        "",
        summary,
        retractable=retractable,
        fetched_at=fetched_at,
        reason=reason,
    )


def _dome(game: str) -> KickoffForecast:
    return KickoffForecast(game, "dome", "dome", "Dome", reason="roof")


def weather_label(forecast: Optional[KickoffForecast]) -> str:
    """Cell text: an icon plus the summary, or the unavailable sentence."""
    if forecast is None:
        return UNAVAILABLE
    if forecast.status == "dome":
        return "🏟️ Dome"
    if forecast.status != "forecast":
        return forecast.summary or UNAVAILABLE
    icons = {
        "sun": "☀️",
        "cloud": "☁️",
        "rain": "🌧️",
        "snow": "❄️",
        "wind": "💨",
    }
    prefix = icons.get(forecast.icon, "")
    return f"{prefix} {forecast.summary}".strip()


def _read_cache(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    entries = payload.get("entries")
    return entries if isinstance(entries, dict) else {}


def _write_cache(path: Path, entries: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"entries": entries}, indent=2),
        encoding="utf-8",
    )


def _entry_fresh(entry: object, now: float, *, force: bool) -> bool:
    if force or not isinstance(entry, dict):
        return False
    try:
        fetched = float(entry.get("fetched_at"))
    except (TypeError, ValueError):
        return False
    age = now - fetched
    if age < 0:
        return False
    ttl = SUCCESS_TTL_SECONDS if entry.get("status") == "forecast" else ERROR_TTL_SECONDS
    return age <= ttl


def _forecast_from_entry(game: str, entry: dict) -> KickoffForecast:
    return KickoffForecast(
        game,
        str(entry.get("status") or "unavailable"),
        str(entry.get("icon") or ""),
        str(entry.get("summary") or UNAVAILABLE),
        retractable=bool(entry.get("retractable")),
        fetched_at=_as_stamp(entry.get("fetched_at")),
        reason=str(entry.get("reason") or ""),
    )


def _as_stamp(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _store_entry(forecast: KickoffForecast) -> dict:
    return {
        "status": forecast.status,
        "icon": forecast.icon,
        "summary": forecast.summary,
        "retractable": forecast.retractable,
        "fetched_at": forecast.fetched_at,
        "reason": forecast.reason,
    }


def _cache_key(slate_key: str, game: str, kickoff: Optional[datetime]) -> str:
    stamp = kickoff.isoformat() if kickoff is not None else ""
    return f"{slate_key}|{game}|{stamp}"


def _load_hourly(lat: float, lon: float, http_get: HttpGet) -> dict:
    points = http_get(f"{NWS_API_ROOT}/points/{lat:.4f},{lon:.4f}")
    hourly_url = hourly_url_from_points(points)
    if not hourly_url or "/forecast/hourly" not in hourly_url:
        raise ValueError("points payload has no hourly forecast url")
    payload = http_get(hourly_url)
    if not isinstance(payload, dict):
        raise ValueError("hourly forecast payload is not an object")
    return payload


def _from_payload(
    game: str,
    payload: dict,
    kickoff: Optional[datetime],
    *,
    retractable: bool,
    fetched_at: float,
) -> KickoffForecast:
    props = payload.get("properties") if isinstance(payload.get("properties"), dict) else payload
    periods = props.get("periods") if isinstance(props, dict) else None
    period = period_for_kickoff(periods, kickoff)
    if period is None:
        return _unavailable(
            game,
            "outside_window",
            retractable=retractable,
            fetched_at=fetched_at,
        )
    icon, summary = describe_hourly_period(period)
    if icon not in {"sun", "cloud", "rain", "snow", "wind"}:
        return _unavailable(game, "unreadable", retractable=retractable, fetched_at=fetched_at)
    if retractable:
        summary = f"{summary} · retractable roof"
    return KickoffForecast(
        game,
        "forecast",
        icon,
        summary,
        retractable=retractable,
        fetched_at=fetched_at,
        reason="",
    )


def forecasts_for_matchups(
    matchups: Sequence[object],
    *,
    slate_key: str,
    cache_path: Path = CACHE_PATH,
    now: Optional[datetime] = None,
    http_get: Optional[HttpGet] = None,
    force_refresh: bool = False,
) -> list[KickoffForecast]:
    """Forecast for each game. A failed fetch is 'Forecast unavailable' and does not raise.

    Results are cached per slate and game. Indoor and fixed-roof stadiums are
    Dome and do not call the network. Retractable roofs are forecast and marked.
    """
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    stamp = current.timestamp()
    getter = http_get or _default_http_get
    entries = _read_cache(cache_path)
    changed = False
    results: list[KickoffForecast] = []
    for matchup in matchups:
        game = str(getattr(matchup, "key", "") or "")
        home = str(getattr(matchup, "home", "") or "")
        kickoff = getattr(matchup, "kickoff_at", None)
        roof = _roof(home)
        if roof in {"dome", "fixed_roof"}:
            results.append(_dome(game))
            continue
        retractable = roof == "retractable"
        if kickoff is None:
            results.append(
                _unavailable(game, "missing_kickoff", retractable=retractable, fetched_at=None)
            )
            continue
        key = _cache_key(slate_key, game, kickoff)
        cached = entries.get(key)
        if _entry_fresh(cached, stamp, force=force_refresh):
            results.append(_forecast_from_entry(game, cached))
            continue
        try:
            stadium = get_stadium(home)
            if stadium is None:
                raise ValueError("unknown stadium")
            payload = _load_hourly(stadium.lat, stadium.lon, getter)
            forecast = _from_payload(
                game,
                payload,
                kickoff,
                retractable=retractable,
                fetched_at=stamp,
            )
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError, TypeError):
            forecast = _unavailable(
                game,
                "fetch_failed",
                retractable=retractable,
                fetched_at=stamp,
            )
        entries[key] = _store_entry(forecast)
        changed = True
        results.append(forecast)
    if changed:
        try:
            _write_cache(cache_path, entries)
        except OSError:
            pass
    return results
