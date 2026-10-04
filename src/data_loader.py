"""Apply cached Vegas lines to a DraftKings player pool."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .optimizer import SKILL_POSITIONS

HIGH_ITT = 26.0
LOW_ITT = 18.0
LOW_TOTAL = 38.5
ITT_BOOST = 1.05
ITT_PENALTY = 0.95
KICKER_CAPTAIN_MAX_EXPOSURE = 0.10

VEGAS_EDITOR_COLUMNS = ["Game", "Team", "Spread", "Over/Under", "ITT"]
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


def lines_to_frame(lines: dict) -> pd.DataFrame:
    """Build the sidebar editor table from a Vegas line dictionary."""
    rows = []
    for team, info in lines.items():
        if not isinstance(info, dict):
            continue
        rows.append(
            {
                "Game": str(info.get("game") or ""),
                "Team": str(team).upper(),
                "Spread": _as_float(info.get("spread")),
                "Over/Under": _as_float(info.get("total")),
                "ITT": _as_float(info.get("itt")),
            }
        )
    frame = pd.DataFrame(rows, columns=VEGAS_EDITOR_COLUMNS)
    if frame.empty:
        return frame
    return frame.sort_values(["Game", "Team"], kind="mergesort").reset_index(drop=True)


def frame_to_lines(frame: pd.DataFrame) -> dict:
    """Read an edited Vegas table back into the optimizer dictionary."""
    if frame is None or frame.empty:
        return {}
    lines: dict = {}
    for _, row in frame.iterrows():
        team = str(row.get("Team", "")).strip().upper()
        if not team or team == "NAN":
            continue
        lines[team] = {
            "total": _as_float(row.get("Over/Under")),
            "spread": _as_float(row.get("Spread")),
            "itt": _as_float(row.get("ITT")),
            "game": "" if pd.isna(row.get("Game")) else str(row.get("Game") or ""),
        }
    return lines


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
    for team, info in lines.items():
        if not isinstance(info, dict):
            continue
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


def projection_adjustment_labels(pool: pd.DataFrame, lines: dict) -> list:
    """Short labels for teams whose skill players are boosted or penalized."""
    if pool is None or pool.empty or not lines or "TeamAbbrev" not in pool.columns:
        return []
    present = set(pool["TeamAbbrev"].astype(str).str.strip().str.upper())
    labels = []
    for team in sorted(present):
        info = lines.get(team)
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
