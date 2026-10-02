"""Parse and validate DraftKings NFL Classic player-pool CSV exports."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import BinaryIO, TextIO, Union

import pandas as pd

REQUIRED_COLUMNS = (
    "ID",
    "Name + ID",
    "Name",
    "Roster Position",
    "Salary",
    "TeamAbbrev",
    "AvgPointsPerGame",
)

_COLUMN_ALIASES = {
    "Player ID": "ID",
    "player id": "ID",
    "Id": "ID",
    "id": "ID",
    "Name and ID": "Name + ID",
    "Name+ID": "Name + ID",
    "RosterPosition": "Roster Position",
    "Team Abbrev": "TeamAbbrev",
    "Team": "TeamAbbrev",
    "Avg Points Per Game": "AvgPointsPerGame",
    "FPPG": "AvgPointsPerGame",
}

_IO = Union[str, Path, BinaryIO, TextIO]


class DKCSVValidationError(ValueError):
    """Raised when a DraftKings CSV is missing required structure or values."""


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        stripped = str(col).strip()
        renamed[col] = _COLUMN_ALIASES.get(stripped, stripped)
    out = df.rename(columns=renamed)
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _derive_position(roster_position: str) -> str:
    """Map DK Roster Position (e.g. WR/FLEX) to a primary Position (WR)."""
    if not isinstance(roster_position, str) or not roster_position.strip():
        raise DKCSVValidationError(f"Invalid Roster Position: {roster_position!r}")
    primary = roster_position.split("/")[0].strip().upper()
    if primary not in {"QB", "RB", "WR", "TE", "DST", "DEF"}:
        raise DKCSVValidationError(f"Unsupported roster position: {roster_position!r}")
    return "DST" if primary == "DEF" else primary


def _read_raw_text(source: _IO) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).read_text(encoding="utf-8-sig")
    if hasattr(source, "seek"):
        source.seek(0)
    data = source.read()
    if isinstance(data, bytes):
        return data.decode("utf-8-sig")
    return str(data)


def _load_dataframe(source: _IO) -> pd.DataFrame:
    text = _read_raw_text(source)
    lines = text.splitlines()
    header_idx = 0
    for i, line in enumerate(lines):
        if "TeamAbbrev" in line and ("Roster Position" in line or "ID" in line):
            header_idx = i
            break
    return pd.read_csv(StringIO("\n".join(lines[header_idx:])), skipinitialspace=True)


def parse_dk_csv(source: _IO) -> pd.DataFrame:
    """
    Load a DraftKings NFL Classic player-pool CSV and validate required fields.

    Returns a DataFrame with normalized columns including ``Position`` (derived
    from ``Roster Position`` when absent) and a ``Projection`` column defaulting
    to ``AvgPointsPerGame``.
    """
    df = _normalize_columns(_load_dataframe(source))

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DKCSVValidationError(
            f"Missing required columns: {', '.join(missing)}. "
            f"Found: {', '.join(map(str, df.columns))}"
        )

    df = df.dropna(subset=["ID", "Name", "Salary", "TeamAbbrev"]).copy()
    if df.empty:
        raise DKCSVValidationError("Player pool is empty after validation.")

    df["ID"] = df["ID"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Name + ID"] = df["Name + ID"].astype(str).str.strip()
    df["TeamAbbrev"] = df["TeamAbbrev"].astype(str).str.strip().str.upper()
    df["Roster Position"] = df["Roster Position"].astype(str).str.strip()
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df["AvgPointsPerGame"] = pd.to_numeric(df["AvgPointsPerGame"], errors="coerce")

    if df["Salary"].isna().any():
        raise DKCSVValidationError("One or more players have invalid Salary values.")
    df["AvgPointsPerGame"] = df["AvgPointsPerGame"].fillna(0.0)

    if "Position" not in df.columns:
        df["Position"] = df["Roster Position"].map(_derive_position)
    else:
        df["Position"] = (
            df["Position"]
            .astype(str)
            .str.split("/")
            .str[0]
            .str.strip()
            .str.upper()
            .replace({"DEF": "DST"})
        )

    if "Game Info" not in df.columns:
        df["Game Info"] = ""

    if "Projection" not in df.columns:
        df["Projection"] = df["AvgPointsPerGame"]
    else:
        df["Projection"] = pd.to_numeric(df["Projection"], errors="coerce").fillna(
            df["AvgPointsPerGame"]
        )

    ordered = [
        "ID",
        "Name + ID",
        "Name",
        "Position",
        "Roster Position",
        "Salary",
        "TeamAbbrev",
        "Game Info",
        "AvgPointsPerGame",
        "Projection",
    ]
    extras = [c for c in df.columns if c not in ordered]
    return df[ordered + extras].reset_index(drop=True)
