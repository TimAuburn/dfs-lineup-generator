"""Sleeper weekly projections and injury status for a DraftKings pool.

DraftKings salaries and player IDs stay on the CSV. When the weekly feed has
the component stats the DraftKings table needs, ``Projection`` is the
DraftKings total from ``src.dk_scoring``. Milestone bonuses are expected
values. A captain row is not multiplied here.

If those components are missing, the row falls back to Sleeper ``pts_ppr``
and is flagged. ``pts_std`` is never used as that fallback. A partial stat
line is not added up. Injury status still comes from this feed. OUT, IR,
Doubtful, PUP, suspended, and NA stay ineligible through the existing filter.
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

from .dk_scoring import SCORING_VERSION, canon_scoring_position, score_projection
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


def _ppr_points(stats: object) -> Optional[float]:
    """Sleeper PPR total. Standard scoring is not a stand-in for PPR."""
    if not isinstance(stats, dict) or "pts_ppr" not in stats:
        return None
    try:
        return float(stats["pts_ppr"])
    except (TypeError, ValueError):
        return None


def _adapted_projection(stats: object, position: str) -> dict:
    """DraftKings points from components, or a flagged PPR fallback.

    Kickers use the Showdown table, which is the only DraftKings NFL table
    that scores them. Classic lineups have no kicker slot. Offense and
    defense use the shared table. The captain multiplier is not applied.
    """
    pos = canon_scoring_position(position)
    contest = "showdown" if pos == "K" else "classic"
    ppr = _ppr_points(stats)
    if not isinstance(stats, dict):
        return _blank_score("Sleeper stats were not an object.")
    result = score_projection(stats, position=pos or position, contest=contest)
    if result.ok and result.points is not None:
        return {
            "projection": float(result.points),
            "sleeper_ppr": ppr,
            "score_basis": "draftkings",
            "score_warning": result.warning,
            "scoring": "DraftKings",
        }
    if ppr is not None:
        detail = result.warning or "Component stats were incomplete."
        return {
            "projection": ppr,
            "sleeper_ppr": ppr,
            "score_basis": "ppr_fallback",
            "score_warning": f"PPR fallback. {detail}",
            "scoring": "PPR fallback",
        }
    detail = result.warning or "Component stats were incomplete."
    if "pts_std" in stats and "pts_ppr" not in stats:
        detail = f"{detail} pts_std is not used as PPR."
    return {
        "projection": None,
        "sleeper_ppr": None,
        "score_basis": "unavailable",
        "score_warning": detail,
        "scoring": "Unavailable",
    }


def _blank_score(warning: str) -> dict:
    return {
        "projection": None,
        "sleeper_ppr": None,
        "score_basis": "unavailable",
        "score_warning": warning,
        "scoring": "Unavailable",
    }


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
        position = _primary_position(player.get("position"))
        adapted = _adapted_projection(row.get("stats"), position)
        injury = canon_injury_status(player.get("injury_status"))
        entry = {
            "projection": adapted["projection"],
            "sleeper_ppr": adapted["sleeper_ppr"],
            "score_basis": adapted["score_basis"],
            "score_warning": adapted["score_warning"],
            "scoring": adapted["scoring"],
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
            "sleeper_ppr": projected.get("sleeper_ppr"),
            "score_basis": projected.get("score_basis") or "",
            "score_warning": projected.get("score_warning") or "",
            "scoring": projected.get("scoring") or "",
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

    ``AvgPointsPerGame`` stays the salary-file average. ``Sleeper`` stores the
    weekly DraftKings total, or the flagged PPR fallback, and is blank when
    that number is missing. ``Projection`` is what the optimizer uses: that
    weekly number when present, otherwise the salary-file average. ``Scoring``
    and ``ScoringNote`` say which path was used. Matched rows take Sleeper's
    status even when it is blank, so a healthy player is not left OUT from a
    stale DK tag. Captain rows are stored at 1x.
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
    updated["PPR"] = pd.Series(pd.NA, index=updated.index, dtype="Float64")
    if "Injury" not in updated.columns:
        updated["Injury"] = ""
    updated["DepthOrder"] = pd.Series(pd.NA, index=updated.index, dtype="Int64")
    updated["DepthPosition"] = ""
    updated["Scoring"] = ""
    updated["ScoringNote"] = ""
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
        if hit.get("sleeper_ppr") is not None:
            updated.at[idx, "PPR"] = float(hit["sleeper_ppr"])
        updated.at[idx, "Scoring"] = str(hit.get("scoring") or "")
        updated.at[idx, "ScoringNote"] = str(hit.get("score_warning") or "")
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
    if payload.get("scoring") != SCORING_VERSION:
        # Older files stored raw PPR in ``projection``. Do not reuse them.
        return None
    try:
        float(payload["timestamp"])
    except (TypeError, ValueError, KeyError):
        return None
    return payload


def _write_cache(path: Path, index: dict, meta: dict, *, timestamp: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": float(timestamp),
        "scoring": SCORING_VERSION,
        "index": index,
        **meta,
    }
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
    fallback = 0
    if "Scoring" in updated.columns:
        fallback = int(updated["Scoring"].eq("PPR fallback").sum())
    summary = {
        **meta,
        "matched": matched,
        "rows": int(len(updated)),
        "inactive_rows": inactive,
        "ppr_fallback_rows": fallback,
        "scoring": SCORING_VERSION,
        "ok": bool(index),
    }
    return updated, summary


def scoring_fallback_lines(pool: pd.DataFrame) -> str:
    """One warning line naming players whose components were not scored."""
    if pool is None or pool.empty or "Scoring" not in pool.columns:
        return ""
    flagged = pool.loc[pool["Scoring"].astype(str).eq("PPR fallback")]
    if flagged.empty:
        return ""
    parts: list[str] = []
    seen: set[str] = set()
    for _, row in flagged.iterrows():
        label = f"{row.get('Name')} ({row.get('TeamAbbrev')})"
        if label in seen:
            continue
        seen.add(label)
        note = str(row.get("ScoringNote") or "").strip()
        parts.append(f"{label}: {note}" if note else label)
    preview = "; ".join(parts[:6])
    extra = f" +{len(parts) - 6} more" if len(parts) > 6 else ""
    return (
        "PPR fallback. Component stats were incomplete, so these projections "
        f"are Sleeper PPR and were not filled in from a partial line: {preview}{extra}."
    )


def scoring_approximation_lines(pool: pd.DataFrame) -> str:
    """Note defenses whose points-allowed tier is an expected value."""
    if pool is None or pool.empty or "ScoringNote" not in pool.columns:
        return ""
    notes = pool["ScoringNote"].astype(str)
    flagged = pool.loc[notes.str.contains("not a distribution", case=False, na=False)]
    if flagged.empty or "TeamAbbrev" not in flagged.columns:
        return ""
    teams = sorted({str(team).strip().upper() for team in flagged["TeamAbbrev"] if str(team).strip()})
    team_list = ", ".join(teams)
    return (
        "Defense points-allowed tiers are expected values from the average "
        f"allowed, not a realized tier ({team_list}). "
        "Yards allowed are not a DraftKings category."
    )


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
    fallback = int(summary.get("ppr_fallback_rows") or 0)
    return (
        f"Projections: DraftKings scoring from Sleeper {season} week {week} "
        f"component stats, {matched} of {rows} rows. "
        f"Milestone bonuses are expected values, not a flat +3. "
        f"{fallback} row(s) flagged as PPR fallback. "
        f"The salary-file average stays in the DraftKings column. "
        f"{inactive} row(s) marked OUT, IR, or Doubtful.{cached}"
    )
