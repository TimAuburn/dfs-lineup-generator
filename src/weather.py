"""National Weather Service fetch + wind-based projection downgrades.

Tunable wind constants live at module top. Pipeline order: Vegas ITT
adjustments first, then weather (call ``apply_weather_projections`` before
optimize). Indoor / enclosed home stadiums skip wind downgrades entirely.
api.weather.gov does not use an API key.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from .stadiums import STADIUMS, get_stadium, is_outdoor_stadium

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "data" / "weather_cache.json"

# National Weather Service. A User-Agent is required; an API key is not.
NWS_API_ROOT = "https://api.weather.gov"
NWS_USER_AGENT = "dfs-lineup-generator (https://github.com/TimAuburn/dfs-lineup-generator)"
DEFAULT_TTL_SECONDS = 60 * 60  # 1 hour
_KM_H_TO_MPH = 0.621371
_M_S_TO_MPH = 2.236936

# ---------------------------------------------------------------------------
# Wind downgrade knobs (easy to find / tune)
# ---------------------------------------------------------------------------
WIND_THRESHOLD_MPH = 18.0
# Sustained wind above the threshold multiplies projections by these factors.
WIND_PASS_FACTOR = 0.90  # QB / WR / TE: -10% (pass game)
WIND_RB_FACTOR = 1.00  # RB: unchanged (relative run-game tilt; no boost/cut)
WIND_K_FACTOR = 0.85  # K: -15% (accuracy / distance hurt by wind)
WIND_DST_FACTOR = 1.00  # DST: unchanged (mixed effects; incomplete passes help)

PASS_HEAVY_POSITIONS = frozenset({"QB", "WR", "TE"})
_GAME_MATCHUP_RE = re.compile(
    r"(?P<away>[A-Za-z]{2,3})\s*@\s*(?P<home>[A-Za-z]{2,3})"
)


def parse_home_team(game_info: object) -> Optional[str]:
    """Extract the home TeamAbbrev from a DK ``Game Info`` / Vegas ``game`` string."""
    if game_info is None:
        return None
    try:
        if pd.isna(game_info):
            return None
    except (TypeError, ValueError):
        pass
    text = str(game_info).strip()
    if not text or text in {"In Progress", "Final"}:
        return None
    match = _GAME_MATCHUP_RE.search(text)
    if not match:
        return None
    return match.group("home").upper()


def wind_factor_for_position(position: object) -> float:
    """Projection multiplier for one primary position when wind is high."""
    primary = str(position or "").split("/")[0].strip().upper()
    if primary in PASS_HEAVY_POSITIONS:
        return WIND_PASS_FACTOR
    if primary == "RB":
        return WIND_RB_FACTOR
    if primary == "K":
        return WIND_K_FACTOR
    if primary in {"DST", "DEF"}:
        return WIND_DST_FACTOR
    return 1.0


def should_apply_wind(home_team: Optional[str], wind_mph: Optional[float]) -> bool:
    """True when the home stadium is outdoor and sustained wind exceeds the threshold."""
    if home_team is None or wind_mph is None:
        return False
    if not is_outdoor_stadium(home_team):
        return False
    try:
        return float(wind_mph) > WIND_THRESHOLD_MPH
    except (TypeError, ValueError):
        return False


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
    payload = {
        "timestamp": float(time.time() if timestamp is None else timestamp),
        "data": data,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _cached_data(path: Path) -> dict:
    payload = _read_cache(path)
    if payload is None:
        return {}
    data = payload.get("data")
    return data if isinstance(data, dict) else {}


def format_weather_cache_status(
    cache_path: Path = CACHE_PATH,
    *,
    now: Optional[float] = None,
) -> str:
    payload = _read_cache(cache_path)
    if payload is None:
        return "No weather cached"
    stamp = float(payload["timestamp"])
    current = time.time() if now is None else float(now)
    minutes = max(0, int((current - stamp) / 60.0))
    return f"Weather: Cached (Last updated: {minutes} minutes ago)"


def _nws_quantity(node: object, *, to: str) -> Optional[float]:
    """Convert an NWS quantitative value to mph or degrees Fahrenheit."""
    if not isinstance(node, dict):
        return None
    raw = node.get("value")
    if raw is None:
        return None
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None
    unit = str(node.get("unitCode") or "").lower().replace(" ", "")
    if to == "mph":
        if "m_s" in unit:
            return number * _M_S_TO_MPH
        if "km_h" in unit or "km/h" in unit:
            return number * _KM_H_TO_MPH
        if "mph" in unit or "mi_h" in unit:
            return number
        # Observations report wind in kilometers per hour when the unit is omitted.
        return number * _KM_H_TO_MPH
    if "degf" in unit or unit.endswith(":f"):
        return number
    return number * 9.0 / 5.0 + 32.0


def parse_weather_payload(payload: object, *, home_team: str) -> dict:
    """Normalize one NWS station observation for a home stadium."""
    if not isinstance(payload, dict):
        raise ValueError("Weather payload must be an object")
    props = payload.get("properties")
    if not isinstance(props, dict):
        props = payload
    wind_mph = _nws_quantity(props.get("windSpeed"), to="mph")
    wind_gust = _nws_quantity(props.get("windGust"), to="mph")
    temperature = _nws_quantity(props.get("temperature"), to="f")
    phrase = props.get("textDescription") or ""
    stadium = get_stadium(home_team)
    return {
        "home_team": str(home_team).upper(),
        "stadium": stadium.name if stadium else "",
        "is_outdoor": bool(stadium.is_outdoor) if stadium else True,
        "wind_mph": wind_mph,
        "wind_gust_mph": wind_gust,
        "temperature_f": temperature,
        "phrase": str(phrase),
        "applies_wind": should_apply_wind(str(home_team).upper(), wind_mph),
    }


def _http_get(url: str, timeout: float = 20) -> object:
    request = Request(
        url,
        headers={
            "User-Agent": NWS_USER_AGENT,
            "Accept": "application/geo+json",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _first_station_id(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    features = payload.get("features") or []
    if not features or not isinstance(features[0], dict):
        return ""
    props = features[0].get("properties") or {}
    ident = props.get("stationIdentifier")
    if ident:
        return str(ident).strip()
    station_url = str(props.get("@id") or features[0].get("id") or "")
    return station_url.rstrip("/").rsplit("/", 1)[-1].strip()


def _fetch_one(home_team: str) -> dict:
    stadium = get_stadium(home_team)
    if stadium is None:
        return {
            "home_team": home_team,
            "stadium": "",
            "is_outdoor": True,
            "wind_mph": None,
            "wind_gust_mph": None,
            "temperature_f": None,
            "phrase": "",
            "applies_wind": False,
            "error": "unknown_stadium",
        }
    if not stadium.is_outdoor:
        return {
            "home_team": home_team,
            "stadium": stadium.name,
            "is_outdoor": False,
            "wind_mph": None,
            "wind_gust_mph": None,
            "temperature_f": None,
            "phrase": "Indoor / enclosed — weather skipped",
            "applies_wind": False,
        }
    points = _http_get(f"{NWS_API_ROOT}/points/{stadium.lat:.4f},{stadium.lon:.4f}")
    stations_url = ""
    if isinstance(points, dict):
        stations_url = str((points.get("properties") or {}).get("observationStations") or "")
    if not stations_url:
        raise ValueError(f"No NWS observation stations for {home_team}")
    station_id = _first_station_id(_http_get(stations_url))
    if not station_id:
        raise ValueError(f"No NWS station id for {home_team}")
    observation = _http_get(f"{NWS_API_ROOT}/stations/{station_id}/observations/latest")
    return parse_weather_payload(observation, home_team=home_team)


def home_teams_from_pool(pool: Optional[pd.DataFrame]) -> list[str]:
    """Unique home TeamAbbrevs from a DraftKings player pool."""
    if pool is None or pool.empty or "Game Info" not in pool.columns:
        return []
    homes: set[str] = set()
    for raw in pool["Game Info"].tolist():
        home = parse_home_team(raw)
        if home:
            homes.add(home)
    return sorted(homes)


def fetch_weather_for_homes(
    home_teams: Sequence[str],
    force_refresh: bool = False,
    *,
    cache_path: Path = CACHE_PATH,
    now: Optional[datetime] = None,
) -> dict:
    """
    Return ``{HOME_ABBR: weather_dict}`` for the given home stadiums.

    Indoor homes are recorded without an HTTP call. Outdoor homes use the
    National Weather Service latest observation. Fresh cache entries are reused.
    """
    current = now or datetime.now()
    wanted = [str(t).strip().upper() for t in home_teams if str(t).strip()]
    wanted = list(dict.fromkeys(wanted))
    if not wanted:
        return {}

    cached = _cached_data(cache_path)
    payload = _read_cache(cache_path)
    cache_fresh = False
    if payload is not None and not force_refresh:
        age = current.timestamp() - float(payload["timestamp"])
        cache_fresh = 0 <= age <= DEFAULT_TTL_SECONDS

    result: dict = {}
    missing: list[str] = []
    for team in wanted:
        stadium = get_stadium(team)
        if stadium is not None and not stadium.is_outdoor:
            result[team] = {
                "home_team": team,
                "stadium": stadium.name,
                "is_outdoor": False,
                "wind_mph": None,
                "wind_gust_mph": None,
                "temperature_f": None,
                "phrase": "Indoor / enclosed — weather skipped",
                "applies_wind": False,
            }
            continue
        if cache_fresh and team in cached and isinstance(cached[team], dict):
            result[team] = cached[team]
        else:
            missing.append(team)

    if not missing:
        return result

    updated = dict(cached)
    for team in missing:
        try:
            info = _fetch_one(team)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            if team in cached and isinstance(cached[team], dict):
                info = cached[team]
            else:
                info = {
                    "home_team": team,
                    "stadium": (get_stadium(team).name if get_stadium(team) else ""),
                    "is_outdoor": is_outdoor_stadium(team),
                    "wind_mph": None,
                    "wind_gust_mph": None,
                    "temperature_f": None,
                    "phrase": "",
                    "applies_wind": False,
                    "error": "fetch_failed",
                }
        result[team] = info
        updated[team] = info

    _write_cache(cache_path, updated, timestamp=current.timestamp())
    return result


def fetch_slate_weather(
    pool: Optional[pd.DataFrame],
    force_refresh: bool = False,
    *,
    cache_path: Path = CACHE_PATH,
    now: Optional[datetime] = None,
) -> dict:
    """Fetch weather for every home stadium on the loaded slate."""
    return fetch_weather_for_homes(
        home_teams_from_pool(pool),
        force_refresh=force_refresh,
        cache_path=cache_path,
        now=now,
    )


def apply_weather_projections(pool: pd.DataFrame, weather_by_home: dict) -> pd.DataFrame:
    """
    Downgrade projections for outdoor games with wind above ``WIND_THRESHOLD_MPH``.

    Factors (when wind applies):
    - QB / WR / TE → ``WIND_PASS_FACTOR`` (default 0.90)
    - RB → ``WIND_RB_FACTOR`` (default 1.00, unchanged)
    - K → ``WIND_K_FACTOR`` (default 0.85)
    - DST → ``WIND_DST_FACTOR`` (default 1.00, unchanged)

    Indoor / enclosed home stadiums never apply a wind factor.
    """
    adjusted = pool.copy()
    if adjusted.empty or not weather_by_home or "Projection" not in adjusted.columns:
        return adjusted
    if "Game Info" not in adjusted.columns or "Position" not in adjusted.columns:
        return adjusted

    projections = pd.to_numeric(adjusted["Projection"], errors="coerce")
    factors = pd.Series(1.0, index=adjusted.index, dtype="float64")
    for idx, raw_game in adjusted["Game Info"].items():
        home = parse_home_team(raw_game)
        if home is None:
            continue
        info = weather_by_home.get(home)
        if not isinstance(info, dict):
            continue
        wind = info.get("wind_mph")
        if not should_apply_wind(home, wind if wind is not None else None):
            continue
        factors.loc[idx] = wind_factor_for_position(adjusted.at[idx, "Position"])
    adjusted["Projection"] = (projections * factors).round(4)
    return adjusted


def weather_adjustment_labels(
    pool: Optional[pd.DataFrame],
    weather_by_home: dict,
) -> list[str]:
    """Short labels for outdoor homes where wind downgrades will apply."""
    if pool is None or pool.empty or not weather_by_home:
        return []
    homes = set(home_teams_from_pool(pool))
    labels = []
    for home in sorted(homes):
        info = weather_by_home.get(home)
        if not isinstance(info, dict):
            continue
        wind = info.get("wind_mph")
        if not should_apply_wind(home, wind if wind is not None else None):
            continue
        wind_txt = f"{float(wind):.0f}" if wind is not None else "?"
        labels.append(
            f"{home} wind {wind_txt} mph "
            f"(QB/WR/TE ×{WIND_PASS_FACTOR:.2f}, K ×{WIND_K_FACTOR:.2f})"
        )
    return labels


def outdoor_teams() -> list[str]:
    return sorted(t for t, s in STADIUMS.items() if s.is_outdoor)


def indoor_teams() -> list[str]:
    return sorted(t for t, s in STADIUMS.items() if not s.is_outdoor)
