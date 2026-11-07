"""Apply cached Vegas lines to a DraftKings player pool."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .optimizer import SKILL_POSITIONS, games_from_pool
from .vegas_cache import implied_team_total, missing_odds_warning
from .weather import apply_weather_projections

HIGH_ITT = 26.0
LOW_ITT = 18.0
LOW_TOTAL = 38.5
ITT_BOOST = 1.05
ITT_PENALTY = 0.95
KICKER_CAPTAIN_MAX_EXPOSURE = 0.10

VEGAS_EDITOR_COLUMNS = ["Game", "Team", "Spread", "Over/Under", "ITT"]
NO_SLATE_ODDS_WARNING = (
    "No Vegas odds cover this slate. Implied team totals were not applied."
)
_SKILL = set(SKILL_POSITIONS)


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


def effective_implied_total(info: object) -> Optional[float]:
    """Implied total derived from the game total and the signed spread.

    ITT is ``(total - signed spread) / 2`` when both numbers are present.
    A stored or typed ``itt`` is not an override. If either input is missing,
    the result is missing too.
    """
    if not isinstance(info, dict):
        return None
    total = _as_float(info.get("total"))
    spread = _as_float(info.get("spread"))
    if total is None or spread is None:
        return None
    return round(implied_team_total(total, spread), 2)


def _same_optional_float(left: Optional[float], right: Optional[float]) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return round(float(left), 2) == round(float(right), 2)


def inconsistent_game_warning(game: str) -> str:
    """A game that does not have one total and opposite spreads."""
    label = game or "This game"
    return (
        f"{label} odds were not used. The teams need one shared total and opposite "
        "spreads, so implied team totals were not applied."
    )


def _distinct_numbers(values: list) -> list:
    found = []
    for value in values:
        if not any(_same_optional_float(value, seen) for seen in found):
            found.append(value)
    return found


def _spreads_oppose(left: Optional[float], right: Optional[float]) -> bool:
    if left is None or right is None:
        return False
    return round(float(left) + float(right), 2) == 0.0


def _game_label(game: str, teams: list[str]) -> str:
    text = str(game or "").strip()
    if text:
        return text
    return "/".join(teams)


def _group_line_items(lines: dict) -> dict[str, list[tuple[str, dict]]]:
    groups: dict[str, list[tuple[str, dict]]] = {}
    for team, info in lines.items():
        if not isinstance(info, dict):
            continue
        game = str(info.get("game") or "").strip()
        groups.setdefault(game, []).append((str(team).upper(), info))
    return groups


def consistent_team_lines(lines: object) -> dict:
    """Team lines from games that share one total and opposite spreads.

    Implied totals are derived. A one-team row, unequal totals, or spreads
    that do not oppose are left out. Nothing is averaged or filled in.
    """
    usable, _warnings = split_game_lines(lines)
    return usable


def split_game_lines(lines: object) -> tuple[dict, list[str]]:
    """Return ``(usable team lines, warnings for rejected games)``."""
    if not isinstance(lines, dict):
        return {}, []
    usable: dict = {}
    warnings: list[str] = []
    for game, rows in _group_line_items(lines).items():
        label = _game_label(game, [team for team, _info in rows])
        if len(rows) != 2:
            warnings.append(inconsistent_game_warning(label))
            continue
        totals = [_as_float(info.get("total")) for _team, info in rows]
        spreads = [_as_float(info.get("spread")) for _team, info in rows]
        if any(value is None for value in totals + spreads):
            warnings.append(inconsistent_game_warning(label))
            continue
        if not _same_optional_float(totals[0], totals[1]) or not _spreads_oppose(
            spreads[0], spreads[1]
        ):
            warnings.append(inconsistent_game_warning(label))
            continue
        total = round(float(totals[0]), 2)
        for (team, info), spread in zip(rows, spreads):
            spread_value = round(float(spread), 2)
            usable[team] = {
                "total": total,
                "spread": spread_value,
                "itt": round(implied_team_total(total, spread_value), 2),
                "game": game,
            }
    return usable, warnings


def _frame_rows(frame: Optional[pd.DataFrame]) -> list[dict]:
    if frame is None or frame.empty:
        return []
    rows = []
    for idx, row in frame.iterrows():
        team = str(row.get("Team", "")).strip().upper()
        if not team or team == "NAN":
            continue
        game = "" if pd.isna(row.get("Game")) else str(row.get("Game") or "").strip()
        rows.append(
            {
                "index": idx,
                "game": game,
                "team": team,
                "total": _as_float(row.get("Over/Under")),
                "spread": _as_float(row.get("Spread")),
                "itt": _as_float(row.get("ITT")),
            }
        )
    return rows


def _resolve_total(previous: list[dict], current: list[dict], edits: list) -> object:
    distinct = _distinct_numbers([value for _team, value in edits])
    if len(distinct) > 1:
        return "ambiguous"
    if len(distinct) == 1:
        return distinct[0]
    baseline = previous or current
    totals = [row["total"] for row in baseline]
    if totals and all(_same_optional_float(totals[0], value) for value in totals):
        return totals[0]
    return "missing"


def _resolve_spreads(teams: list[str], previous: list[dict], current: list[dict], edits: list) -> object:
    if len(edits) > 1:
        edited = {team: value for team, value in edits}
        if set(edited) == set(teams) and _spreads_oppose(edited[teams[0]], edited[teams[1]]):
            return {team: edited[team] for team in teams}
        return "ambiguous"
    if len(edits) == 1:
        team, value = edits[0]
        if value is None:
            return "missing"
        other = teams[1] if team == teams[0] else teams[0]
        return {team: float(value), other: -float(value)}
    baseline = previous or current
    spreads = {row["team"]: row["spread"] for row in baseline}
    if _spreads_oppose(spreads.get(teams[0]), spreads.get(teams[1])):
        return {team: spreads[team] for team in teams}
    return "missing"


def sync_vegas_editor(
    previous: Optional[pd.DataFrame],
    edited: Optional[pd.DataFrame],
    editor_version: int = 0,
) -> dict:
    """Apply one data-editor return as a game, not as two independent rows.

    One edited total becomes the shared total. One edited spread becomes the
    canonical signed spread and the other team receives its opposite. Both
    implied totals are then derived. The editor version increments when the
    grid has to remount so the paired row is visible on the next rerun.
    Passing that paired frame through again does not change it.

    Two conflicting edits in the same return, or a cache pair that does not
    already share a total and opposite spreads, are not averaged. Those games
    are left out of the lines and named in ``warnings``.
    """
    version = int(editor_version or 0)
    if edited is None or edited.empty:
        empty = edited if edited is not None else pd.DataFrame(columns=VEGAS_EDITOR_COLUMNS)
        return {
            "frame": empty,
            "lines": {},
            "warnings": [],
            "editor_version": version,
            "changed": False,
        }
    previous_rows = {(row["game"], row["team"]): row for row in _frame_rows(previous)}
    current_rows = _frame_rows(edited)
    by_game: dict[str, list[dict]] = {}
    game_order: list[str] = []
    for row in current_rows:
        if row["game"] not in by_game:
            by_game[row["game"]] = []
            game_order.append(row["game"])
        by_game[row["game"]].append(row)

    display = edited.copy()
    lines: dict = {}
    warnings: list[str] = []
    changed = False
    for game in game_order:
        rows = by_game[game]
        teams = [row["team"] for row in rows]
        label = _game_label(game, teams)
        previous_pair = [previous_rows.get((game, team)) for team in teams]
        have_previous = all(row is not None for row in previous_pair)
        total_edits = []
        spread_edits = []
        if have_previous:
            for team, before, after in zip(teams, previous_pair, rows):
                if not _same_optional_float(before["total"], after["total"]):
                    total_edits.append((team, after["total"]))
                if not _same_optional_float(before["spread"], after["spread"]):
                    spread_edits.append((team, after["spread"]))
        if len(rows) != 2:
            warnings.append(inconsistent_game_warning(label))
            for row in rows:
                if row["itt"] is not None:
                    display.at[row["index"], "ITT"] = None
                    changed = True
            continue
        total = _resolve_total(previous_pair if have_previous else [], rows, total_edits)
        spreads = _resolve_spreads(
            teams, previous_pair if have_previous else [], rows, spread_edits
        )
        if total is None:
            warnings.append(inconsistent_game_warning(label))
            for row in rows:
                if not _same_optional_float(row["total"], None):
                    display.at[row["index"], "Over/Under"] = None
                    changed = True
                if not _same_optional_float(row["itt"], None):
                    display.at[row["index"], "ITT"] = None
                    changed = True
            continue
        if total in ("ambiguous", "missing") or spreads in ("ambiguous", "missing"):
            warnings.append(inconsistent_game_warning(label))
            for row in rows:
                if not _same_optional_float(row["itt"], None):
                    display.at[row["index"], "ITT"] = None
                    changed = True
            continue
        derived = {}
        total_value = round(float(total), 2)
        for team in teams:
            spread_value = round(float(spreads[team]), 2)
            derived[team] = {
                "total": total_value,
                "spread": spread_value,
                "itt": round(implied_team_total(total_value, spread_value), 2),
            }
            lines[team] = {"game": game, **derived[team]}
        for row in rows:
            values = derived[row["team"]]
            current = {"total": row["total"], "spread": row["spread"], "itt": row["itt"]}
            for column, key in (
                ("Over/Under", "total"),
                ("Spread", "spread"),
                ("ITT", "itt"),
            ):
                if not _same_optional_float(current[key], values[key]):
                    display.at[row["index"], column] = values[key]
                    changed = True
    if changed:
        version += 1
    return {
        "frame": display,
        "lines": lines,
        "warnings": warnings,
        "editor_version": version,
        "changed": changed,
    }


def _matchup_covered(lines: dict, game: str) -> bool:
    """True when both teams have one shared total and opposite spreads."""
    if "@" not in str(game):
        return False
    away, home = str(game).split("@", 1)
    away = away.strip().upper()
    home = home.strip().upper()
    usable = consistent_team_lines(lines)
    if away not in usable or home not in usable:
        return False
    return usable[away].get("game") == game and usable[home].get("game") == game


def slate_odds_warnings(lines: object, pool: Optional[pd.DataFrame]) -> list[str]:
    """Name loaded-slate games that have no usable odds.

    A game counts only when both teams have a total and a signed spread.
    Games left out of the feed entirely are listed along with games the feed
    returned without a market. Nothing is filled in for them.
    """
    games = games_from_pool(pool)
    if not games:
        return []
    line_map = lines if isinstance(lines, dict) else {}
    missing = [game for game in games if not _matchup_covered(line_map, game)]
    if not missing:
        return []
    notes: list[str] = []
    if len(missing) == len(games):
        notes.append(NO_SLATE_ODDS_WARNING)
    notes.extend(missing_odds_warning(game) for game in missing)
    return notes


def visible_odds_warnings(
    lines: object,
    pool: Optional[pd.DataFrame],
    feed_warnings: Optional[list] = None,
    game_warnings: Optional[list] = None,
) -> list[str]:
    """Feed warnings, rejected games, and slate gaps, without duplicates.

    A game rejected for a bad pair is not also called a missing market.
    """
    rejected = set()
    for note in game_warnings or []:
        marker = " odds were not used"
        if marker in str(note):
            rejected.add(str(note).split(marker, 1)[0])
    slate_notes = [
        note
        for note in slate_odds_warnings(lines, pool)
        if not any(note == missing_odds_warning(game) for game in rejected)
    ]
    notes: list[str] = []
    seen = set()
    for item in list(feed_warnings or []) + list(game_warnings or []) + slate_notes:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        notes.append(text)
    return notes


def lines_to_frame(lines: dict) -> pd.DataFrame:
    """Build the sidebar editor table from a Vegas line dictionary.

    A consistent game shows the shared total, opposite spreads, and derived
    implied totals. A contradictory or incomplete game keeps the source
    numbers and leaves ITT blank.
    """
    usable = consistent_team_lines(lines)
    rows = []
    for team, info in lines.items():
        if not isinstance(info, dict):
            continue
        game = str(info.get("game") or "").strip()
        team_key = str(team).upper()
        paired = usable.get(team_key)
        if paired is not None and paired.get("game") == game:
            total = paired["total"]
            spread = paired["spread"]
            itt = paired["itt"]
        else:
            total = _as_float(info.get("total"))
            spread = _as_float(info.get("spread"))
            itt = None
        rows.append(
            {
                "Game": game,
                "Team": team_key,
                "Spread": spread,
                "Over/Under": total,
                "ITT": itt,
            }
        )
    frame = pd.DataFrame(rows, columns=VEGAS_EDITOR_COLUMNS)
    if frame.empty:
        return frame
    return frame.sort_values(["Game", "Team"], kind="mergesort").reset_index(drop=True)


def frame_to_lines(frame: pd.DataFrame) -> dict:
    """Read a Vegas table into team lines, keeping only consistent games."""
    if frame is None or frame.empty:
        return {}
    raw: dict = {}
    for _, row in frame.iterrows():
        team = str(row.get("Team", "")).strip().upper()
        if not team or team == "NAN":
            continue
        raw[team] = {
            "total": _as_float(row.get("Over/Under")),
            "spread": _as_float(row.get("Spread")),
            "game": "" if pd.isna(row.get("Game")) else str(row.get("Game") or ""),
        }
    return consistent_team_lines(raw)


def apply_vegas_projections(pool: pd.DataFrame, lines: dict) -> pd.DataFrame:
    """
    Boost or penalize skill-player projections from implied team totals.

    ITT at or above 26 adds 5%. ITT at or below 18 subtracts 5%. Quarterbacks,
    kickers, and defenses are unchanged.
    """
    adjusted = pool.copy()
    if adjusted.empty or not lines or "Projection" not in adjusted.columns:
        return adjusted
    if "TeamAbbrev" not in adjusted.columns or "Position" not in adjusted.columns:
        return adjusted

    projections = pd.to_numeric(adjusted["Projection"], errors="coerce")
    positions = (
        adjusted["Position"].astype(str).str.split("/").str[0].str.strip().str.upper()
    )
    teams = adjusted["TeamAbbrev"].astype(str).str.strip().str.upper()
    factors = pd.Series(1.0, index=adjusted.index, dtype="float64")
    for team, info in consistent_team_lines(lines).items():
        itt = _as_float(info.get("itt"))
        if itt is None:
            continue
        if itt >= HIGH_ITT:
            factor = ITT_BOOST
        elif itt <= LOW_ITT:
            factor = ITT_PENALTY
        else:
            continue
        mask = (teams == str(team).upper()) & positions.isin(_SKILL)
        factors.loc[mask] = factor
    adjusted["Projection"] = (projections * factors).round(4)
    return adjusted


def low_scoring_showdown(lines: dict, teams: Optional[set] = None) -> bool:
    """True when a slate team's game total is at or under 38.5."""
    if not lines:
        return False
    wanted = {str(team).upper() for team in teams} if teams else None
    if wanted is not None and not wanted:
        return False
    for team, info in lines.items():
        if wanted is not None and str(team).upper() not in wanted:
            continue
        if not isinstance(info, dict):
            continue
        total = _as_float(info.get("total"))
        if total is not None and total <= LOW_TOTAL:
            return True
    return False


def projection_deltas(
    pool: pd.DataFrame,
    lines: dict,
    weather_by_home: Optional[dict] = None,
) -> pd.Series:
    """Signed point change from Vegas ITT and wind, before captain scoring or jitter.

    A positive value is a boost. A negative value is a cut. Quarterbacks,
    kickers, and defenses stay at 0 unless outdoor wind applies to them.
    """
    if pool is None or pool.empty or "Projection" not in pool.columns:
        return pd.Series(dtype="float64")
    base = pd.to_numeric(pool["Projection"], errors="coerce")
    adjusted = apply_vegas_projections(pool, lines or {})
    adjusted = apply_weather_projections(adjusted, weather_by_home or {})
    updated = pd.to_numeric(adjusted["Projection"], errors="coerce")
    return (updated - base).round(2)


def apply_projection_delta(
    pool: pd.DataFrame,
    lines: dict,
    weather_by_home: Optional[dict] = None,
) -> pd.DataFrame:
    """Return a copy where Projection includes Delta and Sleeper stays original.

    The base is the Sleeper weekly number when that cell is filled, otherwise
    the current Projection (the DraftKings average for an unmatched player).
    """
    updated = pool.copy()
    if updated.empty or "Projection" not in updated.columns:
        return updated
    deltas = projection_deltas(updated, lines, weather_by_home)
    base = pd.to_numeric(updated["Projection"], errors="coerce")
    if "Sleeper" in updated.columns:
        sleeper = pd.to_numeric(updated["Sleeper"], errors="coerce")
        base = sleeper.where(sleeper.notna(), base)
    updated["Delta"] = deltas.to_numpy()
    updated["Projection"] = (base + deltas).round(2).to_numpy()
    return updated


def projection_adjustment_labels(pool: pd.DataFrame, lines: dict) -> list:
    """Short labels for teams whose skill players are boosted or penalized."""
    if pool is None or pool.empty or not lines or "TeamAbbrev" not in pool.columns:
        return []
    present = set(pool["TeamAbbrev"].astype(str).str.strip().str.upper())
    labels = []
    for team in sorted(present):
        info = consistent_team_lines(lines).get(team)
        if not isinstance(info, dict):
            continue
        itt = _as_float(info.get("itt"))
        if itt is None:
            continue
        if itt >= HIGH_ITT:
            labels.append(f"{team} +5%")
        elif itt <= LOW_ITT:
            labels.append(f"{team} -5%")
    return labels
