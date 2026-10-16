"""Sleeper weekly projections and injury status for a DraftKings pool.

DraftKings salaries and player IDs stay on the CSV. The optimizer's
``Projection`` column is replaced with Sleeper ``pts_ppr`` (DraftKings is
PPR). Injury status comes from the same feed. OUT, IR, Doubtful, PUP,
suspended, and NA stay ineligible through the existing injury filter.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .parser import canon_injury_status, is_inactive_injury

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "data" / "sleeper_cache.json"
STATE_URL = "https://api.sleeper.app/v1/state/nfl"
PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
PROJECTIONS_URL = "https://api.sleeper.app/projections/nfl/{season}/{week}"
USER_AGENT = "dfs-lineup-generator (https://github.com/TimAuburn/dfs-lineup-generator)"
DEFAULT_TTL_SECONDS = 60 * 60

_NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})
_TEAM_ALIASES = {"WSH": "WAS", "JAC": "JAX", "LA": "LAR"}
_DST_POSITIONS = frozenset({"DST", "DEF"})
_SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE", "K", "FB"})


def normalize_player_name(name: object) -> str:
    """Lowercase name with punctuation and generational suffixes removed."""
    text = str(name or "").lower().replace(".", "").replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z\s-]", " ", text)
    parts = [part for part in text.split() if part not in _NAME_SUFFIXES]
    return " ".join(parts)


def dk_team(team: object) -> str:
    """DraftKings team abbreviation for a Sleeper team code."""
    abbrev = str(team or "").strip().upper()
    return _TEAM_ALIASES.get(abbrev, abbrev)


def _primary_position(position: object) -> str:
    token = str(position or "").split("/")[0].strip().upper()
    if token == "DEF":
        return "DST"
    return token


def _projection_points(stats: object) -> Optional[float]:
    if not isinstance(stats, dict):
        return None
    raw = stats.get("pts_ppr")
    if raw is None:
        raw = stats.get("pts_std")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def index_projections(records: object) -> dict:
    """Map ``(name, team)`` and ``dst:{team}`` to projection and injury."""
    if not isinstance(records, list):
        return {}
    index: dict = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        player = row.get("player") if isinstance(row.get("player"), dict) else {}
        team = dk_team(row.get("team") or player.get("team"))
        if not team:
            continue
        points = _projection_points(row.get("stats"))
        injury = canon_injury_status(player.get("injury_status"))
        position = _primary_position(player.get("position"))
        entry = {
            "projection": points,
            "injury": injury,
            "position": position,
            "depth_order": None,
            "depth_position": "",
        }
        if position == "DST":
            _keep_better(index, f"dst:{team}", entry)
            continue
        first = str(player.get("first_name") or "").strip()
        last = str(player.get("last_name") or "").strip()
        name = normalize_player_name(f"{first} {last}".strip())
        if not name:
            continue
        _keep_better(index, f"{name}|{team}", entry)
    return index


def index_depth_chart(players: object) -> dict:
    """Map ``name|team`` to Sleeper depth chart order and injury status.

    ``depth_chart_order`` 1 is the top player at that spot (QB, RB, LWR, RWR,
    SWR, TE, K). The weekly projections feed does not include this field.
    """
    if not isinstance(players, dict):
        return {}
    index: dict = {}
    for player in players.values():
        if not isinstance(player, dict):
            continue
        position = _primary_position(player.get("position"))
        if position not in _SKILL_POSITIONS:
            continue
        team = dk_team(player.get("team"))
        if not team:
            continue
        first = str(player.get("first_name") or "").strip()
        last = str(player.get("last_name") or "").strip()
        name = normalize_player_name(f"{first} {last}".strip())
        if not name:
            continue
        order = _depth_order(player.get("depth_chart_order"))
        slot = str(player.get("depth_chart_position") or "").strip().upper()
        entry = {
            "projection": None,
            "injury": canon_injury_status(player.get("injury_status")),
            "position": "RB" if position == "FB" else position,
            "depth_order": order,
            "depth_position": slot,
        }
        _keep_shallower(index, f"{name}|{team}", entry)
    return index


def _depth_order(value: object) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _keep_shallower(index: dict, key: str, entry: dict) -> None:
    """Prefer the listing that is higher on the depth chart."""
    current = index.get(key)
    if current is None:
        index[key] = entry
        return
    new_order = entry.get("depth_order")
    old_order = current.get("depth_order")
    if new_order is None:
        return
    if old_order is None or new_order < old_order:
        index[key] = entry


def merge_sleeper_indexes(projections: dict, depth: dict) -> dict:
    """Combine weekly points with the roster depth chart.

    A player present on the depth chart keeps that injury tag, including a
    blank tag when Sleeper lists them as healthy. Points still come from the
    weekly projection.
    """
    merged: dict = {}
    for key in set(projections) | set(depth):
        projected = projections.get(key) or {}
        charted = depth.get(key) or {}
        if key in depth:
            injury = charted.get("injury") or ""
        else:
            injury = projected.get("injury") or ""
        merged[key] = {
            "projection": projected.get("projection"),
            "injury": injury,
            "position": charted.get("position") or projected.get("position") or "",
            "depth_order": charted.get("depth_order"),
            "depth_position": charted.get("depth_position") or "",
        }
    return merged


def _index_has_depth(index: object) -> bool:
    if not isinstance(index, dict):
        return False
    return any(isinstance(entry, dict) and "depth_order" in entry for entry in index.values())


def _keep_better(index: dict, key: str, entry: dict) -> None:
    current = index.get(key)
    if current is None:
        index[key] = entry
        return
    new_points = entry["projection"]
    old_points = current["projection"]
    if new_points is None:
        return
    if old_points is None or new_points > old_points:
        index[key] = entry


def apply_sleeper_index(pool: pd.DataFrame, index: dict) -> tuple[pd.DataFrame, int]:
    """Replace projections and injury status for rows Sleeper recognizes.

    ``AvgPointsPerGame`` stays the DraftKings average. ``Sleeper`` stores the
    weekly number, blank when Sleeper has no match. ``Projection`` is what the
    optimizer uses: the Sleeper number when present, otherwise the DraftKings
    average. A matched quarterback with no weekly points is 0, not that
    average. Matched rows take Sleeper's status even when it is blank, so a
    healthy player is not left OUT from a stale DK tag.
    """
    updated = pool.copy()
    if updated.empty or not index:
        return updated, 0
    if "Projection" not in updated.columns:
        updated["Projection"] = pd.to_numeric(
            updated.get("AvgPointsPerGame", 0.0), errors="coerce"
        )
    updated["Projection"] = pd.to_numeric(updated["Projection"], errors="coerce").astype(float)
    updated["Sleeper"] = pd.Series(pd.NA, index=updated.index, dtype="Float64")
    if "Injury" not in updated.columns:
        updated["Injury"] = ""
    updated["DepthOrder"] = pd.Series(pd.NA, index=updated.index, dtype="Int64")
    updated["DepthPosition"] = ""
    matched = 0
    for idx, row in updated.iterrows():
        team = dk_team(row.get("TeamAbbrev"))
        position = _primary_position(row.get("Position"))
        if position in _DST_POSITIONS:
            hit = index.get(f"dst:{team}")
        else:
            hit = index.get(f"{normalize_player_name(row.get('Name'))}|{team}")
        if hit is None:
            continue
        matched += 1
        if hit.get("projection") is not None:
            points = float(hit["projection"])
            updated.at[idx, "Projection"] = points
            updated.at[idx, "Sleeper"] = points
        elif position == "QB" or str(hit.get("position") or "").upper() == "QB":
            # A depth-chart quarterback with no weekly Sleeper points is 0.
            # Leaving the DraftKings season average in place makes a backup
            # look like a starter.
            updated.at[idx, "Projection"] = 0.0
            updated.at[idx, "Sleeper"] = 0.0
        updated.at[idx, "Injury"] = hit.get("injury") or ""
        order = _depth_order(hit.get("depth_order"))
        if order is not None:
            updated.at[idx, "DepthOrder"] = order
        updated.at[idx, "DepthPosition"] = str(hit.get("depth_position") or "")
    return updated, matched


def _http_get(url: str, timeout: float = 30) -> object:
    request = Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _read_cache(path: Path) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("index"), dict):
        return None
    try:
        float(payload["timestamp"])
    except (TypeError, ValueError, KeyError):
        return None
    return payload


def _write_cache(path: Path, index: dict, meta: dict, *, timestamp: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": float(timestamp), "index": index, **meta}
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_sleeper_index(
    *,
    force_refresh: bool = False,
    cache_path: Path = CACHE_PATH,
    now: Optional[datetime] = None,
) -> tuple[dict, dict]:
    """Return the cached or freshly downloaded Sleeper index and a status dict."""
    current = now or datetime.now()
    cached = _read_cache(cache_path)
    fresh = False
    if cached is not None and not force_refresh:
        age = current.timestamp() - float(cached["timestamp"])
        fresh = 0 <= age <= DEFAULT_TTL_SECONDS
    if fresh and cached is not None and cached.get("has_depth") and _index_has_depth(cached["index"]):
        return cached["index"], _meta_from_cache(cached, from_cache=True)

    try:
        state = _http_get(STATE_URL)
        if not isinstance(state, dict):
            raise ValueError("Sleeper NFL state was not an object")
        season = str(state.get("season") or "").strip()
        week = int(state.get("week") or 0)
        season_type = str(state.get("season_type") or "regular").strip() or "regular"
        if not season or week < 1:
            raise ValueError("Sleeper NFL state is missing a season or week")
        query = urlencode({"season_type": season_type})
        records = _http_get(f"{PROJECTIONS_URL.format(season=season, week=week)}?{query}")
        projections = index_projections(records)
        depth: dict = {}
        try:
            depth = index_depth_chart(_http_get(PLAYERS_URL))
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            depth = {}
        index = merge_sleeper_indexes(projections, depth)
        meta = {
            "season": season,
            "week": week,
            "season_type": season_type,
            "players": len(index),
            "has_depth": bool(depth),
            "error": None,
        }
        _write_cache(cache_path, index, meta, timestamp=current.timestamp())
        return index, {**meta, "from_cache": False}
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        if cached is not None:
            meta = _meta_from_cache(cached, from_cache=True)
            meta["error"] = f"fetch_failed:{type(exc).__name__}"
            return cached["index"], meta
        return {}, {
            "season": None,
            "week": None,
            "season_type": None,
            "players": 0,
            "has_depth": False,
            "from_cache": False,
            "error": f"fetch_failed:{type(exc).__name__}",
        }


def _meta_from_cache(payload: dict, *, from_cache: bool) -> dict:
    return {
        "season": payload.get("season"),
        "week": payload.get("week"),
        "season_type": payload.get("season_type"),
        "players": payload.get("players", len(payload.get("index") or {})),
        "has_depth": bool(payload.get("has_depth")),
        "from_cache": from_cache,
        "error": None,
    }


def overlay_sleeper_projections(
    pool: pd.DataFrame,
    *,
    force_refresh: bool = False,
    cache_path: Path = CACHE_PATH,
    now: Optional[datetime] = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply the current Sleeper week onto a parsed DraftKings pool."""
    index, meta = load_sleeper_index(
        force_refresh=force_refresh,
        cache_path=cache_path,
        now=now,
    )
    updated, matched = apply_sleeper_index(pool, index)
    inactive = 0
    if "Injury" in updated.columns:
        inactive = int(updated["Injury"].map(is_inactive_injury).sum())
    summary = {
        **meta,
        "matched": matched,
        "rows": int(len(updated)),
        "inactive_rows": inactive,
        "ok": bool(index),
    }
    return updated, summary


def sleeper_status_caption(summary: Optional[dict]) -> str:
    """One line for the player-pool caption."""
    if not summary or not summary.get("ok"):
        detail = (summary or {}).get("error") or "unavailable"
        return f"Sleeper projections unavailable ({detail}). Using DraftKings averages."
    week = summary.get("week")
    season = summary.get("season")
    matched = int(summary.get("matched") or 0)
    rows = int(summary.get("rows") or 0)
    inactive = int(summary.get("inactive_rows") or 0)
    cached = " Cached." if summary.get("from_cache") else ""
    if summary.get("error"):
        cached = " Saved copy used after a refresh failed."
    return (
        f"Projections: Sleeper {season} week {week}, {matched} of {rows} rows. "
        f"DraftKings is the salary-file average. "
        f"{inactive} row(s) marked OUT, IR, or Doubtful.{cached}"
    )
