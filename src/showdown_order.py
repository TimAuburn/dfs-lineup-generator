"""Showdown roster order for display and DraftKings upload rows.

FLEX slots are interchangeable. The solver's row order is not a roster slot,
so every Showdown lineup is arranged the same way before it is shown or
exported. Classic rosters have no CPT row and are returned unchanged.
"""

from __future__ import annotations

import math

import pandas as pd


def order_showdown_roster(lineup: pd.DataFrame) -> pd.DataFrame:
    """CPT first, then FLEX by salary, highest first.

    Equal FLEX salaries use the player name, ascending and case-insensitive.
    A remaining tie uses the player ID, ascending as text, then the original
    row so the order is fully determined. The input frame is not modified.
    Lineup attrs are kept. A frame with no CPT row is returned as given.
    """
    if not isinstance(lineup, pd.DataFrame) or lineup.empty or "Slot" not in lineup.columns:
        return lineup
    slots = lineup["Slot"].astype(str).str.strip().str.upper()
    if not bool(slots.eq("CPT").any()):
        return lineup

    def cell_text(value: object) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        return str(value)

    def salary_rank(value: object) -> float:
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return math.inf
        if not math.isfinite(number):
            return math.inf
        return -number

    def flex_key(pos: int) -> tuple:
        row = lineup.iloc[pos]
        salary = salary_rank(row["Salary"]) if "Salary" in lineup.columns else math.inf
        name = cell_text(row["Name"]).casefold() if "Name" in lineup.columns else ""
        player_id = cell_text(row["ID"]) if "ID" in lineup.columns else ""
        return (salary, name, player_id, pos)

    captain = [i for i, slot in enumerate(slots) if slot == "CPT"]
    flex = [i for i, slot in enumerate(slots) if slot == "FLEX"]
    other = [i for i, slot in enumerate(slots) if slot not in {"CPT", "FLEX"}]
    flex.sort(key=flex_key)
    ordered = lineup.iloc[captain + flex + other].copy()
    ordered.attrs = dict(lineup.attrs)
    return ordered
