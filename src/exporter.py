"""Export generated lineups to DraftKings multi-entry upload CSV format."""

from __future__ import annotations

from io import StringIO
from typing import List, Sequence

import pandas as pd

from .presets import ContestFormat

CLASSIC_UPLOAD_COLUMNS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
SHOWDOWN_UPLOAD_COLUMNS = ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]

# Backward-compatible alias.
UPLOAD_COLUMNS = CLASSIC_UPLOAD_COLUMNS


def upload_columns_for(contest_format: ContestFormat | str) -> List[str]:
    fmt = ContestFormat(contest_format)
    if fmt is ContestFormat.SHOWDOWN:
        return list(SHOWDOWN_UPLOAD_COLUMNS)
    return list(CLASSIC_UPLOAD_COLUMNS)


def _lineup_frame_to_row(
    lineup_df: pd.DataFrame,
    upload_columns: Sequence[str],
    *,
    use_name_id: bool = True,
) -> List[str]:
    if "Slot" not in lineup_df.columns:
        raise ValueError("Lineup DataFrame must include a Slot column")

    ordered_values: List[str] = []
    used_index: set[int] = set()
    slots = lineup_df["Slot"].astype(str).tolist()

    for slot in upload_columns:
        match_idx = None
        for i, row_slot in enumerate(slots):
            if i in used_index:
                continue
            if row_slot == slot:
                match_idx = i
                break
        if match_idx is None:
            raise ValueError(f"Lineup is missing required slot: {slot}")
        used_index.add(match_idx)
        row = lineup_df.iloc[match_idx]
        if use_name_id and "Name + ID" in lineup_df.columns:
            ordered_values.append(str(row["Name + ID"]))
        elif "ID" in lineup_df.columns:
            ordered_values.append(str(row["ID"]))
        else:
            raise ValueError("Lineup DataFrame must include 'Name + ID' or 'ID'")
    return ordered_values


def lineups_to_dk_upload_df(
    lineups: Sequence[pd.DataFrame],
    *,
    use_name_id: bool = True,
    contest_format: ContestFormat | str = ContestFormat.CLASSIC,
) -> pd.DataFrame:
    """Build a DraftKings upload DataFrame (duplicate position labels → RB.1, etc.)."""
    upload_columns = upload_columns_for(contest_format)
    # Showdown DK uploads expect numeric Player IDs.
    fmt = ContestFormat(contest_format)
    effective_use_name_id = use_name_id if fmt is ContestFormat.CLASSIC else False
    rows = [
        _lineup_frame_to_row(lu, upload_columns, use_name_id=effective_use_name_id)
        for lu in lineups
    ]
    unique_cols: List[str] = []
    seen: dict[str, int] = {}
    for col in upload_columns:
        count = seen.get(col, 0)
        unique_cols.append(col if count == 0 else f"{col}.{count}")
        seen[col] = count + 1
    return pd.DataFrame(rows, columns=unique_cols)


def lineups_to_dk_upload_csv(
    lineups: Sequence[pd.DataFrame],
    *,
    use_name_id: bool = True,
    contest_format: ContestFormat | str = ContestFormat.CLASSIC,
) -> str:
    """
    Return CSV text for DraftKings multi-entry upload.

    Classic header: QB,RB,RB,WR,WR,WR,TE,FLEX,DST (Name + ID by default).
    Showdown header: CPT,FLEX,FLEX,FLEX,FLEX,FLEX (numeric Player IDs).
    """
    upload_columns = upload_columns_for(contest_format)
    fmt = ContestFormat(contest_format)
    effective_use_name_id = use_name_id if fmt is ContestFormat.CLASSIC else False
    buffer = StringIO()
    buffer.write(",".join(upload_columns) + "\n")
    for lineup in lineups:
        row = _lineup_frame_to_row(
            lineup, upload_columns, use_name_id=effective_use_name_id
        )
        buffer.write(",".join(row) + "\n")
    return buffer.getvalue()


def write_dk_upload_csv(
    lineups: Sequence[pd.DataFrame],
    path: str,
    *,
    use_name_id: bool = True,
    contest_format: ContestFormat | str = ContestFormat.CLASSIC,
) -> None:
    """Write a DraftKings upload CSV to ``path``."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(
            lineups_to_dk_upload_csv(
                lineups,
                use_name_id=use_name_id,
                contest_format=contest_format,
            )
        )
