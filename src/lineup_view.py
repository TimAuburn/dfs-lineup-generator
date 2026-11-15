"""Display-only sizing and Sleeper projection column for one lineup.

The height matches Streamlit 1.67's dataframe content height so every roster
row is on screen. The Sleeper number is read from a pool that the existing
matcher already scored. It is not written back onto the lineup, and it is not
an optimizer input.
"""

from __future__ import annotations

import pandas as pd

from .names import person_key

# Streamlit's default dataframe metrics at a 16px root:
# row and header are 2.1875rem (35px), and the table border is 1px.
# Content height is rows * row + header + 2 * border, plus 1px so the last
# row is not covered by the inner scrollbar.
_ROW_PX = 35
_HEADER_PX = 35
_BORDER_PX = 1
_CONTENT_SLACK_PX = 1


def dataframe_height(row_count: int) -> int:
    """Pixel height that shows every dataframe row without an inner scrollbar."""
    rows = max(int(row_count), 0)
    return rows * _ROW_PX + _HEADER_PX + 2 * _BORDER_PX + _CONTENT_SLACK_PX


def sleeper_proj_caption(summary: object) -> str:
    """Short note when the Sleeper week was not applied. Empty when it was."""
    if isinstance(summary, dict) and summary.get("applied"):
        return ""
    return "Sleeper projections are unavailable. Sleeper Proj is blank."


def sleeper_proj_for_lineup(
    lineup: pd.DataFrame,
    pool: pd.DataFrame | None,
) -> pd.Series:
    """Sleeper Proj aligned to lineup rows.

    A value is copied only from a pool row whose match status is
    ``projection`` and whose Sleeper number is present. Unmatched, ambiguous,
    unscored, and invented backup zeros stay missing. They are never filled
    with 0. The lineup frame is not modified.
    """
    blank = pd.Series(pd.NA, index=lineup.index, dtype="Float64")
    if not isinstance(pool, pd.DataFrame) or pool.empty:
        return blank
    if "Sleeper" not in pool.columns or "MatchStatus" not in pool.columns:
        return blank
    status = pool["MatchStatus"].astype(str).eq("projection")
    numbers = pd.to_numeric(pool["Sleeper"], errors="coerce")
    matched = pool.loc[status & numbers.notna()].copy()
    if matched.empty:
        return blank
    matched["_proj"] = numbers.loc[matched.index].astype(float)

    out = blank.copy()
    if "ID" in lineup.columns and "ID" in matched.columns:
        by_id = (
            matched.assign(_id=matched["ID"].astype(str))
            .drop_duplicates("_id", keep="first")
            .set_index("_id")["_proj"]
        )
        mapped = lineup["ID"].astype(str).map(by_id)
        out = pd.Series(pd.to_numeric(mapped, errors="coerce"), index=lineup.index, dtype="Float64")

    missing = out.isna()
    if not bool(missing.any()):
        return out
    lineup_team = _team_column(lineup)
    pool_team = _team_column(matched)
    if lineup_team is None or pool_team is None or "Name" not in lineup.columns or "Name" not in matched.columns:
        return out
    matched = matched.copy()
    matched["_key"] = _person_keys(matched, "Name", pool_team)
    usable = matched["_key"].map(lambda key: bool(key[0]))
    by_key = (
        matched.loc[usable]
        .drop_duplicates("_key", keep="first")
        .set_index("_key")["_proj"]
    )
    keyed = pd.Series(_person_keys(lineup, "Name", lineup_team), index=lineup.index).map(by_key)
    out.loc[missing] = pd.to_numeric(keyed.loc[missing], errors="coerce")
    return out.astype("Float64")


def lineup_detail_frame(lineup: pd.DataFrame, pool: pd.DataFrame | None) -> pd.DataFrame:
    """One roster table: slots, the optimizer projection, and Sleeper Proj.

    Salary remaining is a lineup total, so it is not repeated on every player.
    Delta is included when the pool already has that adjustment.
    """
    columns = [name for name in ("Slot", "Name", "Team", "Salary", "Projection") if name in lineup.columns]
    view = lineup.loc[:, columns].copy()
    if (
        isinstance(pool, pd.DataFrame)
        and "ID" in lineup.columns
        and "ID" in pool.columns
        and "Delta" in pool.columns
    ):
        delta_by_id = dict(zip(pool["ID"].astype(str), pool["Delta"], strict=False))
        view["Delta"] = lineup["ID"].astype(str).map(delta_by_id)
    view["Sleeper Proj"] = sleeper_proj_for_lineup(lineup, pool)
    return view


def _team_column(frame: pd.DataFrame) -> str | None:
    if "TeamAbbrev" in frame.columns:
        return "TeamAbbrev"
    if "Team" in frame.columns:
        return "Team"
    return None


def _person_keys(frame: pd.DataFrame, name_col: str, team_col: str) -> list[tuple[str, str]]:
    positions = frame["Position"] if "Position" in frame.columns else [None] * len(frame)
    return [
        person_key(name, team, position)
        for name, team, position in zip(frame[name_col], frame[team_col], positions, strict=False)
    ]
