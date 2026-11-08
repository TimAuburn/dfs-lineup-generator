"""Sleeper weekly projections and injury status for one uploaded slate.

DraftKings salaries, roster slots, and player IDs stay on the CSV. Candidate
matching chooses the Sleeper identity. The DraftKings component scorer then
turns that player's stat line into the weekly number. ``pts_std`` is never
that number. A captain row is stored at 1x.

The week comes from the slate's game dates and a real NFL schedule, then it
is checked against the projection payload. The Sleeper current-week endpoint
is not used. A saved copy is reused only for the same slate, season, and
week, and only inside the maximum age below.

Cache age
---------
``DEFAULT_TTL_SECONDS`` (1 hour) is how long a saved copy is reused without
contacting Sleeper again. ``MAX_CACHE_AGE_SECONDS`` (6 hours) is the oldest
copy that may be used when a refresh fails. Six hours covers a morning
session that loses the network and rejects yesterday's file, including a
file from a different slate. Both limits are arguments on the loader
(``ttl_seconds`` and ``max_cache_age_seconds``). They are not bankroll or
contest settings.

Retrieval times are when this process received each feed. A projection row's
``updated_at`` is the feed's own stamp and is stored separately. Sleeper does
not provide a per-player injury update time, so that value stays unavailable
instead of being copied from the retrieval time.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .dk_scoring import SCORING_VERSION, canon_scoring_position, score_projection
from .parser import canon_injury_status, injury_status_for_player, is_inactive_injury
from .slate_week import (
    ScheduledGame,
    coerce_schedule,
    dk_team,
    parse_espn_scoreboard,
    parse_game_info,
    resolve_slate_week,
    slate_fingerprint,
    slate_games_from_pool,
    slate_key,
    validate_projection_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "data" / "sleeper_cache.json"
PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
PROJECTIONS_URL = "https://api.sleeper.app/projections/nfl/{season}/{week}"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
USER_AGENT = "dfs-lineup-generator (https://github.com/TimAuburn/dfs-lineup-generator)"

# Prefer a new download after one hour. Never apply a copy older than six.
DEFAULT_TTL_SECONDS = 60 * 60
MAX_CACHE_AGE_SECONDS = 6 * 60 * 60
MATCH_CACHE_VERSION = 1

_NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})
_STATUS_TAG_RE = re.compile(
    r"\((?:q|o|d|ir|ir-r|irr|pup|sus|na|nwt|p|out|doubtful|questionable|probable|suspended)\)",
    re.IGNORECASE,
)
_DST_POSITIONS = frozenset({"DST", "DEF"})
_SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE", "K", "FB"})
_SLEEPER_ID_COLUMNS = ("SleeperID", "Sleeper Id", "sleeper_id")
_INJURY_RANK = {
    "OUT": 80,
    "IR": 70,
    "PUP": 60,
    "SUS": 50,
    "NA": 40,
    "DOUBTFUL": 30,
    "QUESTIONABLE": 20,
    "PROBABLE": 10,
}
_FETCH_ERRORS = (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError)


def normalize_player_name(name: object) -> str:
    """Lowercase name without punctuation, generational suffixes, or ``(Q)`` tags."""
    text = _STATUS_TAG_RE.sub(" ", str(name or ""))
    text = text.lower().replace(".", "").replace("'", "").replace("’", "").replace(",", " ")
    text = re.sub(r"[^a-z\s-]", " ", text)
    parts = [part for part in text.split() if part and part not in _NAME_SUFFIXES]
    return " ".join(parts)


def _primary_position(position: object) -> str:
    token = str(position or "").split("/")[0].strip().upper()
    if token in _DST_POSITIONS:
        return "DST"
    if token == "FB":
        return "RB"
    return token


def _positions_compatible(left: object, right: object) -> bool:
    return _primary_position(left) == _primary_position(right) and bool(_primary_position(left))


def _ppr_points(stats: object) -> Optional[float]:
    """Sleeper PPR total. Standard scoring is not a stand-in for PPR."""
    if not isinstance(stats, dict) or "pts_ppr" not in stats:
        return None
    try:
        return float(stats["pts_ppr"])
    except (TypeError, ValueError):
        return None


def _adapted_projection(stats: object, position: str) -> dict:
    """DraftKings points for one already chosen stat line.

    Kickers use the Showdown table. The captain multiplier is not applied.
    Missing components fall back to ``pts_ppr`` and are flagged. ``pts_std``
    is never that fallback.
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
            "score_warning": result.warning or "",
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
    if isinstance(stats, dict) and "pts_std" in stats and "pts_ppr" not in stats:
        detail = f"{detail} pts_std is not used as PPR."
    return _blank_score(detail)


def _blank_score(warning: str) -> dict:
    return {
        "projection": None,
        "sleeper_ppr": None,
        "score_basis": "unavailable",
        "score_warning": warning,
        "scoring": "Unavailable",
    }


def _score_candidate(row: dict) -> dict:
    """Score one source row after identity selection. Does not choose among rows."""
    adapted = _adapted_projection(row.get("stats") or {}, str(row.get("position") or ""))
    scored = dict(row)
    scored.update(adapted)
    return scored


def _feed_updated_at(row: dict) -> Optional[float]:
    for key in ("updated_at", "last_modified"):
        raw = row.get(key)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0:
            return float(raw)
    return None


def _candidate_from_projection(row: dict) -> Optional[dict]:
    player = row.get("player") if isinstance(row.get("player"), dict) else {}
    team = dk_team(row.get("team") or player.get("team"))
    if not team:
        return None
    position = _primary_position(player.get("position") or row.get("position"))
    if not position:
        return None
    first = str(player.get("first_name") or "").strip()
    last = str(player.get("last_name") or "").strip()
    player_id = str(row.get("player_id") or player.get("player_id") or "").strip()
    stats = row.get("stats") if isinstance(row.get("stats"), dict) else {}
    return {
        "player_id": player_id,
        "name": normalize_player_name(f"{first} {last}".strip()),
        "team": team,
        "position": position,
        "stats": stats,
        "projection": None,
        "injury": canon_injury_status(player.get("injury_status")),
        "depth_order": None,
        "depth_position": "",
        "game_date": str(row.get("date") or "")[:10] or None,
        "opponent": dk_team(row.get("opponent")) or None,
        "feed_updated_at": _feed_updated_at(row),
    }


def index_projections(records: object) -> dict:
    """Projection-feed candidates. Duplicates stay in the list."""
    players: list[dict] = []
    if isinstance(records, list):
        for row in records:
            if not isinstance(row, dict):
                continue
            candidate = _candidate_from_projection(row)
            if candidate is not None:
                players.append(candidate)
    return {"projections": players, "depth": []}


def index_depth_chart(players: object) -> dict:
    """Depth-chart candidates. The weekly feed does not include depth order."""
    depth: list[dict] = []
    if isinstance(players, dict):
        for key, player in players.items():
            if not isinstance(player, dict):
                continue
            position = _primary_position(player.get("position"))
            if position not in _SKILL_POSITIONS and position not in {"QB", "RB", "WR", "TE", "K"}:
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
            depth.append(
                {
                    "player_id": str(player.get("player_id") or key or "").strip(),
                    "name": name,
                    "team": team,
                    "position": "RB" if position == "FB" else position,
                    "projection": None,
                    "injury": canon_injury_status(player.get("injury_status")),
                    "depth_order": order,
                    "depth_position": str(player.get("depth_chart_position") or "").strip().upper(),
                    "game_date": None,
                    "opponent": None,
                    "feed_updated_at": None,
                }
            )
    return {"projections": [], "depth": depth}


def merge_sleeper_indexes(projections: object, depth: object) -> dict:
    """Keep both feeds. A blank depth injury does not erase a projection injury."""
    return {
        "projections": _candidate_rows(projections, "projections"),
        "depth": _candidate_rows(depth, "depth"),
    }


def _candidate_rows(payload: object, key: str) -> list[dict]:
    if isinstance(payload, dict):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _depth_order(value: object) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_injury(dk_status: object, source_statuses: list) -> tuple[str, str]:
    """Combine DraftKings and Sleeper injury tags.

    A blank value never clears a status from the other source, including OUT.
    When both sides name a status and they differ, the more severe one is
    kept: OUT, then IR, PUP, suspended, NA, Doubtful, Questionable, Probable.
    Questionable is not an inactive status.
    """
    dk = canon_injury_status(dk_status)
    present = []
    for raw in source_statuses:
        status = canon_injury_status(raw)
        if status:
            present.append(status)
    sleeper = ""
    note = ""
    if present:
        sleeper = max(present, key=lambda status: _INJURY_RANK.get(status, 0))
        if len(set(present)) > 1:
            note = "Sleeper statuses differ (" + ", ".join(sorted(set(present))) + f"). Using {sleeper}."
    if dk and not sleeper:
        return dk, f"Kept {dk} from DraftKings; Sleeper injury status was blank."
    if sleeper and not dk:
        if is_inactive_injury(sleeper):
            return sleeper, f"Kept {sleeper} from Sleeper; DraftKings injury status was blank."
        return sleeper, note
    if dk and sleeper and dk != sleeper:
        winner = dk if _INJURY_RANK.get(dk, 0) >= _INJURY_RANK.get(sleeper, 0) else sleeper
        conflict = f"Conflict: DraftKings {dk}, Sleeper {sleeper}. Using {winner}."
        return winner, f"{note} {conflict}".strip()
    if dk and sleeper:
        return dk, note
    return "", ""


def _blank(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "<na>"}


def _row_sleeper_id(row: pd.Series) -> str:
    for column in _SLEEPER_ID_COLUMNS:
        if column in row.index and not _blank(row.get(column)):
            return str(row.get(column)).strip()
    return ""


def _identity(name: object, team: object, position: object) -> tuple[str, str, str]:
    pos = _primary_position(position)
    if pos == "DST":
        return ("dst", dk_team(team), "DST")
    return (normalize_player_name(name), dk_team(team), pos)


def _same_player(left: dict, right: dict) -> bool:
    left_id = str(left.get("player_id") or "").strip()
    right_id = str(right.get("player_id") or "").strip()
    if left_id and right_id:
        return left_id == right_id
    if left_id or right_id:
        return False
    return (
        left.get("name") == right.get("name")
        and left.get("team") == right.get("team")
        and _positions_compatible(left.get("position"), right.get("position"))
    )


def _unique_players(rows: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for row in rows:
        placed = False
        for group in groups:
            if _same_player(group[0], row):
                group.append(row)
                placed = True
                break
        if not placed:
            groups.append([row])
    return groups


def _projection_values(rows: list[dict]) -> list[float]:
    values = []
    for row in rows:
        points = row.get("projection")
        if points is None:
            continue
        number = round(float(points), 4)
        if number not in values:
            values.append(number)
    return values


def _candidate_brief(row: dict) -> str:
    points = row.get("projection")
    points_text = "no projection" if points is None else f"projection {float(points):.2f}"
    player_id = str(row.get("player_id") or "").strip() or "no id"
    return f"id {player_id} {points_text}"


def _select_candidates(row: pd.Series, projections: list[dict], depth: list[dict]) -> dict:
    team = dk_team(row.get("TeamAbbrev"))
    position = _primary_position(row.get("Position"))
    name = normalize_player_name(row.get("Name"))
    sleeper_id = _row_sleeper_id(row)
    label = f"{row.get('Name')} ({team} {position})".strip()

    def team_pos(candidates: list[dict]) -> list[dict]:
        return [
            candidate
            for candidate in candidates
            if candidate.get("team") == team and _positions_compatible(position, candidate.get("position"))
        ]

    if position == "DST":
        proj_hits = team_pos(projections)
        depth_hits = team_pos(depth)
        if sleeper_id:
            by_id = [candidate for candidate in proj_hits if str(candidate.get("player_id") or "") == sleeper_id]
            other_ids = [candidate for candidate in projections if str(candidate.get("player_id") or "") == sleeper_id]
            if other_ids and not by_id:
                return {"status": "ambiguous", "reason": f"{label} Sleeper id {sleeper_id} failed the team or position check."}
            if sleeper_id and not other_ids and not by_id:
                return {"status": "unmatched", "reason": f"{label} Sleeper id {sleeper_id} was not on the week feed."}
            if by_id:
                proj_hits = by_id
                depth_hits = [candidate for candidate in depth_hits if str(candidate.get("player_id") or "") == sleeper_id]
    elif sleeper_id:
        by_id = [candidate for candidate in projections if str(candidate.get("player_id") or "") == sleeper_id]
        depth_by_id = [candidate for candidate in depth if str(candidate.get("player_id") or "") == sleeper_id]
        if not by_id and not depth_by_id:
            return {"status": "unmatched", "reason": f"{label} Sleeper id {sleeper_id} was not on the week feed."}
        proj_hits = team_pos(by_id)
        depth_hits = team_pos(depth_by_id)
        if (by_id or depth_by_id) and not proj_hits and not depth_hits:
            return {
                "status": "ambiguous",
                "reason": f"{label} Sleeper id {sleeper_id} failed the team or position check.",
                "candidates": by_id or depth_by_id,
            }
    else:
        proj_hits = [
            candidate
            for candidate in team_pos(projections)
            if candidate.get("name") == name
        ]
        depth_hits = [
            candidate
            for candidate in team_pos(depth)
            if candidate.get("name") == name
        ]
        if not proj_hits and not depth_hits:
            wrong_position = [
                candidate
                for candidate in projections
                if candidate.get("name") == name
                and candidate.get("team") == team
                and not _positions_compatible(position, candidate.get("position"))
            ]
            if wrong_position:
                return {"status": "unmatched", "reason": f"{label} position does not match the Sleeper player."}
            return {"status": "unmatched", "reason": f"{label} was not on the Sleeper week feed."}

    proj_groups = _unique_players(proj_hits)
    depth_groups = _unique_players(depth_hits)
    if len(proj_groups) > 1 or (not proj_groups and len(depth_groups) > 1):
        shown = [_score_candidate(candidate) for candidate in (proj_hits or depth_hits)]
        detail = "; ".join(_candidate_brief(candidate) for candidate in shown[:6])
        return {
            "status": "ambiguous",
            "reason": f"{label} matches more than one Sleeper player ({detail}). Neither projection was used.",
            "candidates": shown,
        }
    if proj_groups:
        scored = [_score_candidate(candidate) for candidate in proj_groups[0]]
        values = _projection_values(scored)
        if len(values) > 1:
            detail = "; ".join(_candidate_brief(candidate) for candidate in scored[:6])
            return {
                "status": "ambiguous",
                "reason": (
                    f"{label} has conflicting Sleeper projections ({detail}). "
                    "The higher number was not chosen."
                ),
                "candidates": scored,
            }
        points = values[0] if values else None
        chosen = scored
        chosen_depth = depth_hits
        if chosen[0].get("player_id"):
            chosen_depth = [
                candidate
                for candidate in depth_hits
                if str(candidate.get("player_id") or "") == str(chosen[0].get("player_id"))
            ] or depth_hits
        status = "projection" if points is not None else "no_projection"
        status = _qb_without_weekly_status(status, position, chosen_depth)
        return {
            "status": status,
            "projection": points,
            "score": chosen[0],
            "proj_rows": chosen,
            "depth_rows": chosen_depth,
        }
    status = _qb_without_weekly_status("depth_only", position, depth_groups[0])
    return {
        "status": status,
        "projection": None,
        "score": {},
        "proj_rows": [],
        "depth_rows": depth_groups[0],
    }


def _qb_without_weekly_status(status: str, position: object, depth_rows: list[dict]) -> str:
    """Confirmed starters keep the salary-file average. Other depth-chart QBs do not.

    A weekly number, including a PPR fallback, is left unchanged. A projection
    feed that cannot be scored stays ``no_projection`` so the average is kept
    and the under-12 cutoff still applies. Only a depth-chart quarterback with
    no weekly row is rewritten here.
    """
    if status == "projection" or _primary_position(position) != "QB":
        return status
    if status not in {"no_projection", "depth_only"}:
        return status
    orders = [_depth_order(item.get("depth_order")) for item in depth_rows]
    orders = [order for order in orders if order is not None]
    slots = [str(item.get("depth_position") or "").strip().upper() for item in depth_rows]
    slots = [slot for slot in slots if slot]
    order = min(orders) if orders else None
    confirmed = order == 1 and (not slots or "QB" in slots)
    if confirmed:
        return "starter_average"
    if status == "depth_only" and order is not None:
        return "backup_zero"
    return status


def match_sleeper_pool(pool: pd.DataFrame, index: object) -> tuple[pd.DataFrame, dict]:
    """Apply one matched identity to each row.

    Captain and flex rows share a name, team, and position, so they receive
    the same weekly points and injury tag. Salary and roster position are
    left as uploaded. Points are not added together and the captain row is
    not multiplied. A depth-chart-only player keeps the DraftKings average.
    """
    updated = pool.copy()
    report = _empty_report(len(updated))
    if updated.empty:
        return updated, report
    projections = _candidate_rows(index, "projections")
    depth = _candidate_rows(index, "depth")
    if "Projection" not in updated.columns:
        updated["Projection"] = pd.to_numeric(updated.get("AvgPointsPerGame", 0.0), errors="coerce")
    updated["Projection"] = pd.to_numeric(updated["Projection"], errors="coerce").astype(float)
    if "AvgPointsPerGame" in updated.columns:
        updated["AvgPointsPerGame"] = pd.to_numeric(updated["AvgPointsPerGame"], errors="coerce")
    updated["Sleeper"] = pd.Series(pd.NA, index=updated.index, dtype="Float64")
    if "Injury" not in updated.columns:
        updated["Injury"] = ""
    updated["DepthOrder"] = pd.Series(pd.NA, index=updated.index, dtype="Int64")
    updated["DepthPosition"] = ""
    updated["PreLock"] = ""
    updated["InjuryNote"] = ""
    updated["MatchStatus"] = ""
    updated["PPR"] = pd.Series(pd.NA, index=updated.index, dtype="Float64")
    updated["Scoring"] = ""
    updated["ScoringNote"] = ""
    updated["StarterFallback"] = False

    identities: dict[tuple, dict] = {}
    for idx, row in updated.iterrows():
        identity = _identity(row.get("Name"), row.get("TeamAbbrev"), row.get("Position"))
        choice = _select_candidates(row, projections, depth)
        status = choice["status"]
        dk_injury = injury_status_for_player(row.get("Name"), row.get("Injury"))
        note = ""
        injury = dk_injury
        points = None
        if status in {"projection", "no_projection", "depth_only"}:
            sources = [item.get("injury") for item in choice.get("proj_rows") or []]
            sources.extend(item.get("injury") for item in choice.get("depth_rows") or [])
            injury, note = resolve_injury(dk_injury, sources)
            points = choice.get("projection")
        elif status == "ambiguous":
            note = choice.get("reason") or ""
            injury = dk_injury
        else:
            note = ""
            injury = dk_injury

        updated.at[idx, "Injury"] = injury
        updated.at[idx, "InjuryNote"] = note
        updated.at[idx, "PreLock"] = "Q" if injury == "QUESTIONABLE" else ""
        updated.at[idx, "MatchStatus"] = status
        score = choice.get("score") if isinstance(choice.get("score"), dict) else {}
        if status == "projection" and points is not None:
            updated.at[idx, "Projection"] = float(points)
            updated.at[idx, "Sleeper"] = float(points)
            if score.get("sleeper_ppr") is not None:
                updated.at[idx, "PPR"] = float(score["sleeper_ppr"])
            updated.at[idx, "Scoring"] = str(score.get("scoring") or "")
            updated.at[idx, "ScoringNote"] = str(score.get("score_warning") or "")
        elif status == "no_projection":
            updated.at[idx, "Scoring"] = str(score.get("scoring") or "")
            updated.at[idx, "ScoringNote"] = str(score.get("score_warning") or "")
        elif status == "starter_average":
            updated.at[idx, "StarterFallback"] = True
            updated.at[idx, "Scoring"] = "Salary-file average"
            updated.at[idx, "ScoringNote"] = (
                "Confirmed starter has no weekly Sleeper projection. "
                "DraftKings salary-file average kept. This is not a projection match."
            )
        elif status == "backup_zero":
            updated.at[idx, "Projection"] = 0.0
            updated.at[idx, "Sleeper"] = 0.0
            updated.at[idx, "Scoring"] = "No weekly score"
            updated.at[idx, "ScoringNote"] = (
                "Unconfirmed quarterback has no weekly projection, "
                "so the salary-file average was not used."
            )
        depth_rows = choice.get("depth_rows") or []
        orders = [_depth_order(item.get("depth_order")) for item in depth_rows]
        orders = [order for order in orders if order is not None]
        if status in {"projection", "no_projection", "depth_only", "starter_average", "backup_zero"} and orders:
            updated.at[idx, "DepthOrder"] = min(orders)
            slots = [str(item.get("depth_position") or "") for item in depth_rows if item.get("depth_position")]
            if slots:
                updated.at[idx, "DepthPosition"] = slots[0]

        bucket = identities.setdefault(
            identity,
            {
                "name": str(row.get("Name") or ""),
                "team": identity[1],
                "position": identity[2],
                "rows": 0,
                "status": status,
                "reason": choice.get("reason") or note,
                "candidates": choice.get("candidates") or [],
                "prelock": injury == "QUESTIONABLE",
            },
        )
        bucket["rows"] += 1
        if status == "ambiguous":
            bucket["status"] = "ambiguous"
            bucket["reason"] = choice.get("reason") or bucket["reason"]
            bucket["candidates"] = choice.get("candidates") or bucket["candidates"]
        elif bucket["status"] != "ambiguous" and status == "projection":
            bucket["status"] = "projection"
        elif bucket["status"] not in {"ambiguous", "projection"} and status in {
            "starter_average",
            "backup_zero",
        }:
            bucket["status"] = status

    for identity, bucket in identities.items():
        report["player_count"] += 1
        if bucket["status"] == "projection":
            report["projection_players"] += 1
        elif bucket["status"] == "depth_only":
            report["depth_only_players"] += 1
        elif bucket["status"] == "starter_average":
            report["starter_fallbacks"].append(
                {"name": bucket["name"], "team": bucket["team"], "position": bucket["position"]}
            )
        elif bucket["status"] == "unmatched":
            report["unmatched"].append(
                {
                    "name": bucket["name"],
                    "team": bucket["team"],
                    "position": bucket["position"],
                    "rows": bucket["rows"],
                    "reason": bucket["reason"],
                }
            )
        elif bucket["status"] == "ambiguous":
            report["ambiguous"].append(
                {
                    "name": bucket["name"],
                    "team": bucket["team"],
                    "position": bucket["position"],
                    "rows": bucket["rows"],
                    "reason": bucket["reason"],
                    "candidates": [
                        {
                            "player_id": str(candidate.get("player_id") or ""),
                            "projection": candidate.get("projection"),
                        }
                        for candidate in bucket["candidates"]
                    ],
                }
            )
        if bucket["status"] in {
            "projection",
            "no_projection",
            "depth_only",
            "starter_average",
            "backup_zero",
        }:
            report["injury_players"] += 1
        if bucket["prelock"]:
            report["questionable"].append(
                {"name": bucket["name"], "team": bucket["team"], "position": bucket["position"]}
            )

    report["projection_rows"] = int(updated["MatchStatus"].eq("projection").sum()) if len(updated) else 0
    report["injury_rows"] = int(
        updated["MatchStatus"].isin(
            ["projection", "no_projection", "depth_only", "starter_average", "backup_zero"]
        ).sum()
    )
    report["depth_only_rows"] = int(updated["MatchStatus"].eq("depth_only").sum())
    report["ppr_fallback_rows"] = int(updated["Scoring"].eq("PPR fallback").sum()) if len(updated) else 0
    report["rows"] = int(len(updated))
    report["inactive_rows"] = int(updated["Injury"].map(is_inactive_injury).sum())
    return updated, report


def apply_sleeper_index(pool: pd.DataFrame, index: dict) -> tuple[pd.DataFrame, int]:
    """Apply Sleeper to a pool. The count is projection rows, not depth-only rows."""
    updated, report = match_sleeper_pool(pool, index)
    return updated, int(report["projection_rows"])


def _empty_report(rows: int) -> dict:
    return {
        "projection_rows": 0,
        "projection_players": 0,
        "injury_rows": 0,
        "injury_players": 0,
        "depth_only_rows": 0,
        "depth_only_players": 0,
        "player_count": 0,
        "rows": int(rows),
        "inactive_rows": 0,
        "unmatched": [],
        "ambiguous": [],
        "questionable": [],
        "starter_fallbacks": [],
        "ppr_fallback_rows": 0,
        "matching_ran": False,
    }


def _http_get(url: str, timeout: float = 30) -> object:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _clock(now: Optional[datetime]):
    def stamp() -> float:
        if now is None:
            return time.time()
        return float(now.timestamp())

    return stamp


def _fmt_ts(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number > 10**11:
        number = number / 1000.0
    return datetime.fromtimestamp(number, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _age_phrase(seconds: object) -> str:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "unknown"
    minutes = int(value // 60)
    if minutes < 90:
        return f"{minutes} minutes"
    return f"{value / 3600:.1f} hours"


def _limit_phrase(seconds: int) -> str:
    if seconds % 3600 == 0 and seconds >= 3600:
        hours = seconds // 3600
        unit = "hour" if hours == 1 else "hours"
        return f"{hours} {unit}"
    return f"{int(seconds)} seconds"


def _read_cache(path: Path) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("match_schema") != MATCH_CACHE_VERSION:
        return None
    if payload.get("scoring") != SCORING_VERSION:
        return None
    if not isinstance(payload.get("index"), dict):
        return None
    if not payload.get("fingerprint") or not payload.get("slate_key"):
        return None
    return payload


def _write_cache(path: Path, index: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **meta,
        "match_schema": MATCH_CACHE_VERSION,
        "scoring": SCORING_VERSION,
        "index": index,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _cache_age_seconds(payload: dict, now_ts: float) -> Optional[float]:
    retrieved = payload.get("retrieved_at")
    if not isinstance(retrieved, dict):
        return None
    stamps = []
    projections = retrieved.get("projections")
    if projections is None:
        return None
    try:
        stamps.append(float(projections))
    except (TypeError, ValueError):
        return None
    if payload.get("schedule_supplied"):
        pass
    else:
        schedule = retrieved.get("schedule")
        if schedule is None:
            return None
        try:
            stamps.append(float(schedule))
        except (TypeError, ValueError):
            return None
    players = retrieved.get("players")
    if players is not None:
        try:
            stamps.append(float(players))
        except (TypeError, ValueError):
            return None
    ages = [now_ts - stamp for stamp in stamps]
    if any(age < 0 for age in ages):
        return None
    return max(ages)


def _cache_matches(payload: dict, fingerprint: str, season: str, week: int, season_type: str) -> bool:
    try:
        same_week = (
            str(payload.get("season")) == str(season)
            and int(payload.get("week")) == int(week)
            and str(payload.get("season_type")) == str(season_type)
        )
    except (TypeError, ValueError):
        return False
    return bool(payload.get("validated")) and payload.get("fingerprint") == fingerprint and same_week


def _base_meta(
    *,
    max_cache_age_seconds: int,
    ttl_seconds: int,
) -> dict:
    return {
        "ok": False,
        "applied": False,
        "error": None,
        "message": "",
        "season": None,
        "week": None,
        "season_type": None,
        "from_cache": False,
        "has_depth": False,
        "players": 0,
        "dates_confirmed": None,
        "validation_message": "",
        "retrieved_at": {"schedule": None, "projections": None, "players": None},
        "injury_updated_at": None,
        "projections_feed_updated_at": None,
        "cache_age_seconds": None,
        "max_cache_age_seconds": int(max_cache_age_seconds),
        "ttl_seconds": int(ttl_seconds),
        "rejected_cache": None,
        "schedule_supplied": False,
        "matching_ran": False,
        "scoring": SCORING_VERSION,
        "match_schema": MATCH_CACHE_VERSION,
        "fingerprint": "",
        "slate_key": "",
    }


def _empty_index() -> dict:
    return {"projections": [], "depth": []}


def _fetch_schedule(dates: list, http_get, stamp) -> tuple[list[ScheduledGame], Optional[float]]:
    games: list[ScheduledGame] = []
    retrieved = None
    for day in sorted(set(dates)):
        url = f"{ESPN_SCOREBOARD_URL}?{urlencode({'dates': day.strftime('%Y%m%d')})}"
        payload = http_get(url)
        retrieved = stamp()
        games.extend(parse_espn_scoreboard(payload))
    return games, retrieved


def _projection_url(season: str, week: int, season_type: str) -> str:
    query = urlencode({"season_type": season_type})
    return f"{PROJECTIONS_URL.format(season=season, week=int(week))}?{query}"


def _max_feed_stamp(index: dict) -> Optional[float]:
    stamps = [
        row.get("feed_updated_at")
        for row in _candidate_rows(index, "projections")
        if isinstance(row.get("feed_updated_at"), (int, float))
    ]
    if not stamps:
        return None
    return float(max(stamps))


def load_sleeper_index(
    *,
    pool: Optional[pd.DataFrame] = None,
    force_refresh: bool = False,
    cache_path: Path = CACHE_PATH,
    now: Optional[datetime] = None,
    max_cache_age_seconds: int = MAX_CACHE_AGE_SECONDS,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    schedule_games: object = None,
    http_get=None,
) -> tuple[dict, dict]:
    """Return a slate-week index and a status dict. The index may be empty."""
    getter = http_get or _http_get
    stamp = _clock(now)
    now_ts = stamp()
    meta = _base_meta(max_cache_age_seconds=max_cache_age_seconds, ttl_seconds=ttl_seconds)
    slate = slate_games_from_pool(pool) if pool is not None else []
    key = slate_key(slate)
    meta["slate_key"] = key
    meta["fingerprint"] = f"unresolved|{key}"
    cached = _read_cache(cache_path)
    supplied = schedule_games is not None
    meta["schedule_supplied"] = supplied

    schedule: list[ScheduledGame] = []
    schedule_error = None
    if supplied:
        schedule = coerce_schedule(schedule_games)
    else:
        dates = [game.game_date for game in slate if game.game_date is not None]
        if dates:
            try:
                schedule, retrieved = _fetch_schedule(dates, getter, stamp)
                meta["retrieved_at"]["schedule"] = retrieved
            except _FETCH_ERRORS as exc:
                schedule_error = f"schedule_fetch_failed:{type(exc).__name__}"

    if schedule_error:
        fallback = _fallback_cache(
            cached,
            slate_key_value=key,
            now_ts=now_ts,
            max_cache_age_seconds=max_cache_age_seconds,
            fingerprint=None,
        )
        if fallback is not None:
            return fallback["index"], _meta_from_cache(
                fallback,
                meta,
                from_cache=True,
                error=schedule_error,
                now_ts=now_ts,
            )
        meta["error"] = schedule_error
        meta["message"] = (
            "The NFL schedule could not be fetched, so the slate week was not chosen. "
            "The current NFL week was not used."
        )
        meta["rejected_cache"] = _rejected_cache(cached, now_ts, max_cache_age_seconds, key)
        return _empty_index(), meta

    resolution = resolve_slate_week(slate, schedule)
    if not resolution["ok"]:
        meta["error"] = resolution["error"]
        meta["message"] = resolution["message"]
        meta["rejected_cache"] = _rejected_cache(cached, now_ts, max_cache_age_seconds, key)
        return _empty_index(), meta

    season = str(resolution["season"])
    week = int(resolution["week"])
    season_type = str(resolution["season_type"])
    fingerprint = slate_fingerprint(slate, season, week, season_type)
    meta["season"] = season
    meta["week"] = week
    meta["season_type"] = season_type
    meta["message"] = resolution["message"]
    meta["fingerprint"] = fingerprint
    meta["slate_key"] = key

    fresh_limit = min(int(ttl_seconds), int(max_cache_age_seconds))
    fresh = _matching_cache(cached, fingerprint, season, week, season_type, now_ts, fresh_limit)
    if fresh is not None and not force_refresh:
        return fresh["index"], _meta_from_cache(fresh, meta, from_cache=True, error=None, now_ts=now_ts)

    try:
        records = getter(_projection_url(season, week, season_type))
        projections_retrieved = stamp()
        validation = validate_projection_records(records, slate, season, week, season_type)
        if not validation["ok"]:
            meta["error"] = validation["error"]
            meta["message"] = validation["message"]
            meta["validation_message"] = validation["message"]
            meta["dates_confirmed"] = validation.get("dates_confirmed")
            meta["retrieved_at"]["projections"] = projections_retrieved
            meta["rejected_cache"] = _rejected_cache(cached, now_ts, max_cache_age_seconds, key)
            return _empty_index(), meta
        projections = index_projections(records)
        depth: dict = {"projections": [], "depth": []}
        players_retrieved = None
        players_error = None
        try:
            depth = index_depth_chart(getter(PLAYERS_URL))
            players_retrieved = stamp()
        except _FETCH_ERRORS as exc:
            players_error = f"players_fetch_failed:{type(exc).__name__}"
        index = merge_sleeper_indexes(projections, depth)
        feed_stamp = _max_feed_stamp(index)
        cache_meta = {
            "fingerprint": fingerprint,
            "slate_key": key,
            "season": season,
            "week": week,
            "season_type": season_type,
            "validated": True,
            "dates_confirmed": bool(validation.get("dates_confirmed")),
            "validation_message": validation.get("message") or "",
            "schedule_supplied": supplied,
            "retrieved_at": {
                "schedule": meta["retrieved_at"]["schedule"],
                "projections": projections_retrieved,
                "players": players_retrieved,
            },
            "injury_updated_at": None,
            "projections_feed_updated_at": feed_stamp,
            "players": len(index["projections"]),
            "has_depth": bool(index["depth"]),
            "players_error": players_error,
        }
        _write_cache(cache_path, index, cache_meta)
        loaded = _meta_from_cache(
            {**cache_meta, "index": index},
            meta,
            from_cache=False,
            error=players_error,
            now_ts=now_ts,
        )
        loaded["ok"] = True
        loaded["applied"] = True
        return index, loaded
    except _FETCH_ERRORS as exc:
        usable = _matching_cache(
            cached,
            fingerprint,
            season,
            week,
            season_type,
            now_ts,
            max_cache_age_seconds,
        )
        if usable is not None:
            return usable["index"], _meta_from_cache(
                usable,
                meta,
                from_cache=True,
                error=f"fetch_failed:{type(exc).__name__}",
                now_ts=now_ts,
            )
        if cached is not None and cached.get("slate_key") == key:
            age = _cache_age_seconds(cached, now_ts)
            if age is not None and age > max_cache_age_seconds:
                meta["error"] = "stale_cache"
                meta["message"] = (
                    f"The saved Sleeper copy is {_age_phrase(age)} old, older than the "
                    f"{_limit_phrase(max_cache_age_seconds)} maximum, and was not used. "
                    "No fresh retrieval time was assigned."
                )
                meta["rejected_cache"] = _rejected_cache(cached, now_ts, max_cache_age_seconds, key)
                return _empty_index(), meta
        if cached is not None and cached.get("slate_key") != key:
            meta["error"] = "wrong_week_cache"
            meta["message"] = (
                "A saved Sleeper copy is for a different slate, season, or week and was not used. "
                f"The refresh failed ({type(exc).__name__}). The current NFL week was not used."
            )
            meta["rejected_cache"] = _rejected_cache(cached, now_ts, max_cache_age_seconds, key)
            return _empty_index(), meta
        meta["error"] = f"fetch_failed:{type(exc).__name__}"
        meta["message"] = (
            f"Sleeper projections could not be fetched ({type(exc).__name__}). "
            "DraftKings averages were kept. No retrieval time was invented."
        )
        return _empty_index(), meta


def _matching_cache(payload, fingerprint, season, week, season_type, now_ts, limit) -> Optional[dict]:
    if payload is None or not _cache_matches(payload, fingerprint, season, week, season_type):
        return None
    age = _cache_age_seconds(payload, now_ts)
    if age is None or age > limit:
        return None
    return payload


def _fallback_cache(payload, *, slate_key_value: str, now_ts: float, max_cache_age_seconds: int, fingerprint) -> Optional[dict]:
    if payload is None or not payload.get("validated"):
        return None
    if payload.get("slate_key") != slate_key_value or not slate_key_value:
        return None
    if fingerprint and payload.get("fingerprint") != fingerprint:
        return None
    age = _cache_age_seconds(payload, now_ts)
    if age is None or age > max_cache_age_seconds:
        return None
    return payload


def _rejected_cache(payload, now_ts: float, max_cache_age_seconds: int, slate_key_value: str) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    age = _cache_age_seconds(payload, now_ts)
    reason = "different_slate"
    if payload.get("slate_key") == slate_key_value and age is not None and age > max_cache_age_seconds:
        reason = "stale_cache"
    elif payload.get("match_schema") != MATCH_CACHE_VERSION:
        reason = "obsolete_cache"
    return {
        "reason": reason,
        "age_seconds": age,
        "retrieved_at": payload.get("retrieved_at"),
        "season": payload.get("season"),
        "week": payload.get("week"),
        "season_type": payload.get("season_type"),
    }


def _meta_from_cache(payload: dict, meta: dict, *, from_cache: bool, error: Optional[str], now_ts: float) -> dict:
    age = _cache_age_seconds(payload, now_ts)
    retrieved = payload.get("retrieved_at") if isinstance(payload.get("retrieved_at"), dict) else {}
    message = meta.get("message") or ""
    if from_cache and error:
        message = (
            f"{message} Refresh failed ({error}). "
            f"The saved copy retrieved at {_fmt_ts((retrieved or {}).get('projections'))} was used. "
            f"Its age is {_age_phrase(age)}, under the {_limit_phrase(meta['max_cache_age_seconds'])} maximum."
        ).strip()
    elif from_cache:
        message = (
            f"{message} Saved copy used. Age {_age_phrase(age)}. "
            f"A new download is preferred after {_limit_phrase(meta['ttl_seconds'])}."
        ).strip()
    loaded = {
        **meta,
        "ok": True,
        "applied": True,
        "error": error,
        "message": message,
        "season": payload.get("season"),
        "week": payload.get("week"),
        "season_type": payload.get("season_type"),
        "from_cache": from_cache,
        "has_depth": bool(payload.get("has_depth")),
        "players": payload.get("players") or len(_candidate_rows(payload.get("index"), "projections")),
        "dates_confirmed": payload.get("dates_confirmed"),
        "validation_message": payload.get("validation_message") or "",
        "retrieved_at": {
            "schedule": retrieved.get("schedule"),
            "projections": retrieved.get("projections"),
            "players": retrieved.get("players"),
        },
        "injury_updated_at": None,
        "projections_feed_updated_at": payload.get("projections_feed_updated_at"),
        "cache_age_seconds": age,
        "schedule_supplied": bool(payload.get("schedule_supplied")),
        "players_error": payload.get("players_error"),
    }
    return loaded


def overlay_sleeper_projections(
    pool: pd.DataFrame,
    *,
    force_refresh: bool = False,
    cache_path: Path = CACHE_PATH,
    now: Optional[datetime] = None,
    max_cache_age_seconds: int = MAX_CACHE_AGE_SECONDS,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    schedule_games: object = None,
    http_get=None,
) -> tuple[pd.DataFrame, dict]:
    """Apply the slate's Sleeper week, or leave DraftKings values in place."""
    index, meta = load_sleeper_index(
        pool=pool,
        force_refresh=force_refresh,
        cache_path=cache_path,
        now=now,
        max_cache_age_seconds=max_cache_age_seconds,
        ttl_seconds=ttl_seconds,
        schedule_games=schedule_games,
        http_get=http_get,
    )
    if meta.get("applied"):
        updated, report = match_sleeper_pool(pool, index)
        report["matching_ran"] = True
    else:
        updated = pool.copy()
        report = _empty_report(len(updated))
        if "Injury" in updated.columns:
            report["inactive_rows"] = int(updated["Injury"].map(is_inactive_injury).sum())
    summary = {
        **meta,
        **report,
        "matched": int(report["projection_rows"]),
        "rows": int(len(updated)),
        "inactive_rows": int(report["inactive_rows"]),
        "ppr_fallback_rows": int(report.get("ppr_fallback_rows") or 0),
        "scoring": SCORING_VERSION,
        "match_schema": MATCH_CACHE_VERSION,
    }
    summary["message"] = _with_pool_gaps(pool, summary.get("message") or "")
    updated = _stamp_contract(updated, summary)
    return updated, summary


def _stamp_contract(pool: pd.DataFrame, summary: dict) -> pd.DataFrame:
    """Mark this frame as produced by the current scorer and matcher."""
    updated = pool.copy()
    if "PPR" not in updated.columns:
        updated["PPR"] = pd.Series(pd.NA, index=updated.index, dtype="Float64")
    if "Scoring" not in updated.columns:
        updated["Scoring"] = ""
    if "ScoringNote" not in updated.columns:
        updated["ScoringNote"] = ""
    if "StarterFallback" not in updated.columns:
        updated["StarterFallback"] = False
    updated.attrs["scoring_version"] = SCORING_VERSION
    updated.attrs["match_schema"] = MATCH_CACHE_VERSION
    updated.attrs["fingerprint"] = summary.get("fingerprint") or ""
    updated.attrs["slate_key"] = summary.get("slate_key") or ""
    return updated


def _with_pool_gaps(pool: pd.DataFrame, message: str) -> str:
    if pool is None or pool.empty or "Game Info" not in pool.columns:
        return message
    blank = 0
    for raw in pool["Game Info"].tolist():
        if parse_game_info(raw) is None:
            blank += 1
    if not blank:
        return message
    note = f"{blank} row(s) had no usable game date and were not used to choose the week."
    return f"{message} {note}".strip()


def _names(entries: list[dict], limit: int = 8) -> str:
    if not entries:
        return "none"
    labels = []
    for entry in entries:
        labels.append(f"{entry.get('name')} ({entry.get('team')} {entry.get('position')})")
    preview = ", ".join(labels[:limit])
    extra = len(labels) - limit
    if extra > 0:
        preview += f" +{extra} more"
    return preview


def sleeper_status_lines(summary: Optional[dict]) -> list[str]:
    """User-visible week, coverage, freshness, and match results."""
    if not summary:
        return [
            "Sleeper projections were not applied (unavailable). DraftKings averages were kept. "
            "The current NFL week was not used. No retrieval time was assigned."
        ]
    lines = [summary.get("message") or "Sleeper projections were not applied."]
    players = int(summary.get("player_count") or 0)
    rows = int(summary.get("rows") or 0)
    if summary.get("matching_ran"):
        lines.append(
            "Projection coverage: "
            f"{int(summary.get('projection_players') or 0)} of {players} players "
            f"({int(summary.get('projection_rows') or 0)} of {rows} rows). "
            "Injury coverage: "
            f"{int(summary.get('injury_players') or 0)} of {players} players "
            f"({int(summary.get('injury_rows') or 0)} of {rows} rows). "
            "Depth-only, not projection matches: "
            f"{int(summary.get('depth_only_players') or 0)} players "
            f"({int(summary.get('depth_only_rows') or 0)} rows)."
        )
        lines.append(
            "Unmatched players: "
            f"{len(summary.get('unmatched') or [])}. {_names(summary.get('unmatched') or [])}."
        )
        lines.append(
            "Ambiguous players: "
            f"{len(summary.get('ambiguous') or [])}. {_names(summary.get('ambiguous') or [])}."
        )
        questionable = summary.get("questionable") or []
        lines.append(
            "Questionable players stay eligible and are flagged before lock: "
            f"{len(questionable)}. {_names(questionable)}."
        )
        starters = summary.get("starter_fallbacks") or []
        lines.append(
            "Confirmed starters with no weekly projection keep the DraftKings average "
            f"and are not projection coverage: {len(starters)}. {_names(starters)}."
        )
        lines.append(
            "DraftKings scoring from Sleeper component stats. "
            "Milestone bonuses are expected values, not a flat +3. "
            f"{int(summary.get('ppr_fallback_rows') or 0)} row(s) flagged as PPR fallback."
        )
    else:
        lines.append(
            "Projection coverage: 0. Injury coverage: 0. "
            "Matching did not run, so unmatched and ambiguous players were not listed from a guessed week."
        )
        lines.append("Questionable pre-lock flags were not evaluated because the slate week was not applied.")
    lines.append(_freshness_line(summary))
    if summary.get("validation_message") and summary.get("validation_message") not in lines[0]:
        lines.append(str(summary["validation_message"]))
    return lines


def _freshness_line(summary: dict) -> str:
    retrieved = summary.get("retrieved_at") if isinstance(summary.get("retrieved_at"), dict) else {}
    max_age = _limit_phrase(int(summary.get("max_cache_age_seconds") or MAX_CACHE_AGE_SECONDS))
    ttl = _limit_phrase(int(summary.get("ttl_seconds") or DEFAULT_TTL_SECONDS))
    parts = [f"Maximum saved-copy age is {max_age}. A new download is preferred after {ttl}."]
    if summary.get("schedule_supplied"):
        parts.append("Schedule was supplied for this slate and has no retrieval time.")
    elif retrieved.get("schedule") is not None:
        parts.append(f"Schedule retrieved {_fmt_ts(retrieved.get('schedule'))}.")
    else:
        parts.append("Schedule retrieval time: unavailable.")
    if retrieved.get("projections") is not None:
        parts.append(f"Projections retrieved {_fmt_ts(retrieved.get('projections'))}.")
    else:
        parts.append("Projections retrieval time: unavailable.")
    if retrieved.get("players") is not None:
        parts.append(f"Players feed retrieved {_fmt_ts(retrieved.get('players'))}.")
    else:
        parts.append("Players feed retrieval time: unavailable.")
    feed = summary.get("projections_feed_updated_at")
    if feed is not None:
        parts.append(f"Projection feed updated_at {_fmt_ts(feed)}.")
    else:
        parts.append("Projection feed updated_at: unavailable.")
    parts.append("Injury update time: unavailable.")
    if summary.get("from_cache"):
        parts.append(f"Saved-copy age {_age_phrase(summary.get('cache_age_seconds'))}.")
    rejected = summary.get("rejected_cache")
    if isinstance(rejected, dict) and rejected.get("reason") == "stale_cache":
        parts.append(
            "The rejected saved copy was retrieved at "
            f"{_fmt_ts((rejected.get('retrieved_at') or {}).get('projections'))} "
            f"and its age is {_age_phrase(rejected.get('age_seconds'))}. It was not used."
        )
    error = str(summary.get("error") or "")
    failed_source = (
        error in {"stale_cache", "wrong_week_cache"}
        or error.startswith("fetch_failed")
        or error.startswith("schedule_fetch_failed")
    )
    if failed_source and not summary.get("applied") and retrieved.get("projections") is None:
        parts.append("No fresh retrieval time was assigned.")
    return " ".join(parts)


SCORING_COLUMNS = ("PPR", "Scoring", "ScoringNote")
MATCH_COLUMNS = ("PreLock", "InjuryNote", "MatchStatus", "StarterFallback")
STARTER_FALLBACK_NOTE = (
    "Confirmed starter has no weekly Sleeper projection. "
    "DraftKings salary-file average kept. This is not a projection match."
)


def scoring_metadata_current(pool: object, summary: object) -> bool:
    """True when this pool carries the current DraftKings scoring contract."""
    if not isinstance(pool, pd.DataFrame):
        return False
    if any(column not in pool.columns for column in SCORING_COLUMNS):
        return False
    if pool.attrs.get("scoring_version") != SCORING_VERSION:
        return False
    return isinstance(summary, dict) and summary.get("scoring") == SCORING_VERSION


def session_contract_current(pool: object, summary: object) -> bool:
    """True when scoring and slate matching are the current pair.

    A wrong slate, a missing version, or a week that was not recorded clears
    the retained pool and its lineups. An explicit failed resolve stays only
    when this overlay wrote both versions and the same slate key.
    """
    if not scoring_metadata_current(pool, summary):
        return False
    if not isinstance(pool, pd.DataFrame) or not isinstance(summary, dict):
        return False
    if summary.get("match_schema") != MATCH_CACHE_VERSION:
        return False
    if pool.attrs.get("match_schema") != MATCH_CACHE_VERSION:
        return False
    if summary.get("applied"):
        if summary.get("season") in (None, "") or summary.get("week") in (None, ""):
            return False
        if summary.get("season_type") not in {"pre", "regular", "post"}:
            return False
    pool_key = ""
    if "Game Info" in pool.columns:
        pool_key = slate_key(slate_games_from_pool(pool))
    if pool_key and pool.attrs.get("slate_key") != pool_key:
        return False
    if summary.get("slate_key") != pool.attrs.get("slate_key"):
        return False
    if summary.get("fingerprint") != pool.attrs.get("fingerprint"):
        return False
    return True


def drop_stale_scoring_session(state: dict) -> bool:
    """Clear a pool and its lineups when scoring or matching is missing or old."""
    pool = state.get("player_pool")
    summary = state.get("sleeper_summary")
    lineups = state.get("lineups") or []
    generated = state.get("generated_run")
    has_work = (
        isinstance(pool, pd.DataFrame)
        or bool(lineups)
        or generated is not None
        or summary is not None
    )
    if not has_work or session_contract_current(pool, summary):
        return False
    state["player_pool"] = None
    state["lineups"] = []
    state["generated_run"] = None
    state["sleeper_summary"] = None
    return True


def scoring_fallback_lines(pool: pd.DataFrame) -> str:
    """Name players whose components were incomplete."""
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


def sleeper_status_caption(summary: Optional[dict], pool: object = None) -> str:
    """One caption for the player pool. Coverage and freshness stay in the text."""
    if pool is not None and not scoring_metadata_current(pool, summary):
        return (
            "Stored projections are missing current DraftKings scoring metadata "
            "and were not used."
        )
    if pool is not None and not session_contract_current(pool, summary):
        return (
            "Stored projections are missing the current slate-matching contract "
            "and were not used."
        )
    return " ".join(sleeper_status_lines(summary))
