"""Parse and validate DraftKings NFL Classic / Showdown player-pool CSV exports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import BinaryIO, Optional, TextIO, Union

import pandas as pd

from .presets import ContestFormat

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
    "Projected Ownership": "Ownership",
    "Own%": "Ownership",
    "Own": "Ownership",
    "Injury Indicator": "Injury",
    "Injury Status": "Injury",
    "Inj": "Injury",
    # Official DKSalaries.csv uses this header for O/Q/D/IR/OUT.
    "Status": "Injury",
    "status": "Injury",
}

# Statuses that mean the player will not take the field. Questionable and
# probable stay in the pool because those players usually play.
_WILL_NOT_PLAY = frozenset({"OUT", "DOUBTFUL", "IR", "PUP", "SUS", "NA"})
_STATUS_CANON = {
    "O": "OUT",
    "OUT": "OUT",
    "D": "DOUBTFUL",
    "DOUBTFUL": "DOUBTFUL",
    "IR": "IR",
    "IR-R": "IR",
    "IRR": "IR",
    "PUP": "PUP",
    "SUS": "SUS",
    "SUSPENDED": "SUS",
    "NA": "NA",
    "NWT": "NA",
    "Q": "QUESTIONABLE",
    "QUESTIONABLE": "QUESTIONABLE",
    "P": "PROBABLE",
    "PROBABLE": "PROBABLE",
}
_NAME_STATUS_RE = re.compile(r"^(?P<name>.*?)\s*\((?P<tag>[A-Za-z][A-Za-z0-9-]*)\)\s*$")

CLASSIC_POSITIONS = {"QB", "RB", "WR", "TE", "DST", "DEF", "K"}
SHOWDOWN_SLOTS = {"CPT", "FLEX"}

_IO = Union[str, Path, BinaryIO, TextIO]


class DKCSVValidationError(ValueError):
    """Raised when a DraftKings CSV is missing required structure or values."""


def canon_injury_status(raw: object) -> str:
    """Normalize a DK injury tag. Empty string means no status."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    text = str(raw).strip().upper()
    if not text or text in {"NAN", "NONE", "-"}:
        return ""
    return _STATUS_CANON.get(text, "")


def injury_status_for_player(name: object, explicit: object = None) -> str:
    """
    Injury status from a DK column, or a trailing tag on the player name.

    ``Name + ID`` values such as ``Patrick Mahomes (1001)`` are not statuses.
    """
    status = canon_injury_status(explicit)
    if status:
        return status
    match = _NAME_STATUS_RE.match(str(name or "").strip())
    if not match:
        return ""
    return canon_injury_status(match.group("tag"))


def is_inactive_injury(status: object) -> bool:
    """True when the player is OUT, IR, Doubtful, PUP, suspended, or inactive."""
    canon = status if status in _WILL_NOT_PLAY else canon_injury_status(status)
    return canon in _WILL_NOT_PLAY


def drop_inactive_players(df: pd.DataFrame) -> pd.DataFrame:
    """Remove every row for players who will not play, including CPT and FLEX."""
    if df.empty or "Injury" not in df.columns or "ID" not in df.columns:
        return df
    inactive = df["Injury"].map(is_inactive_injury)
    if not bool(inactive.any()):
        return df
    inactive_ids = set(df.loc[inactive, "ID"].astype(str))
    kept = df.loc[~df["ID"].astype(str).isin(inactive_ids)].copy()
    return kept.reset_index(drop=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        stripped = str(col).strip()
        renamed[col] = _COLUMN_ALIASES.get(stripped, stripped)
    out = df.rename(columns=renamed)
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _primary_roster_slots(df: pd.DataFrame) -> pd.Series:
    """Roster Position before a slash, uppercased. ``RB/FLEX`` is ``RB``."""
    return (
        df["Roster Position"]
        .astype(str)
        .str.split("/")
        .str[0]
        .str.strip()
        .str.upper()
    )


def detect_contest_format(df: pd.DataFrame) -> ContestFormat:
    """Infer Classic vs Showdown from Roster Position values."""
    if "Roster Position" not in df.columns:
        return ContestFormat.CLASSIC
    if _primary_roster_slots(df).isin(SHOWDOWN_SLOTS).any():
        return ContestFormat.SHOWDOWN
    return ContestFormat.CLASSIC


_UNCHANGED = "Contest format was left unchanged."


@dataclass(frozen=True)
class SalaryFileDetection:
    """Format read from a salary upload, or a reason not to switch."""

    contest_format: Optional[ContestFormat] = None
    caption: Optional[str] = None
    warning: Optional[str] = None


def _salary_warning(text: str) -> SalaryFileDetection:
    return SalaryFileDetection(warning=f"{text} {_UNCHANGED}")


def _detected(fmt: ContestFormat) -> SalaryFileDetection:
    label = "Showdown" if fmt is ContestFormat.SHOWDOWN else "Classic"
    return SalaryFileDetection(contest_format=fmt, caption=f"Detected: {label}")


def _roster_labels(df: pd.DataFrame) -> set[str]:
    labels: set[str] = set()
    for raw in _primary_roster_slots(df).tolist():
        text = str(raw).strip().upper()
        if text and text != "NAN":
            labels.add(text)
    return labels


def detect_salary_file_format(source: _IO | bytes) -> SalaryFileDetection:
    """Classify a DK salary upload as Classic, Showdown, or leave-unchanged.

    Loads the file with the same header scan, column aliases, and required
    columns as ``parse_dk_csv``. Roster positions decide the format: CPT/FLEX
    is Showdown, and QB/RB/WR/TE/DST (including RB/FLEX, WR/FLEX, and TE/FLEX)
    is Classic. A CPT row that is not 1.5x its FLEX row is still Showdown.
    The pre-upload lineup check already blocks a bad captain price.
    """
    if isinstance(source, (bytes, bytearray)):
        source = BytesIO(bytes(source))
    try:
        frame = _normalize_columns(_load_dataframe(source))
    except (
        OSError,
        UnicodeError,
        ValueError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ):
        return _salary_warning(
            "This file could not be read as a DraftKings salary CSV."
        )

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        return _salary_warning(
            "This salary file is missing required columns: " + ", ".join(missing) + "."
        )

    labels = _roster_labels(frame)
    showdown = labels & SHOWDOWN_SLOTS
    classic = labels & CLASSIC_POSITIONS
    other = labels - SHOWDOWN_SLOTS - CLASSIC_POSITIONS
    if showdown and classic:
        return _salary_warning(
            "This salary file mixes Showdown (CPT/FLEX) and Classic "
            "(QB/RB/WR/TE/DST) roster positions."
        )
    if other or not labels:
        return _salary_warning(
            "Could not detect Classic (QB/RB/WR/TE/DST) or Showdown (CPT/FLEX) "
            "from the roster positions in this salary file."
        )
    if showdown:
        return _detected(ContestFormat.SHOWDOWN)
    return _detected(ContestFormat.CLASSIC)


def _derive_classic_position(roster_position: str) -> str:
    """Map DK Roster Position (e.g. WR/FLEX) to a primary Position (WR)."""
    if not isinstance(roster_position, str) or not roster_position.strip():
        raise DKCSVValidationError(f"Invalid Roster Position: {roster_position!r}")
    primary = roster_position.split("/")[0].strip().upper()
    if primary not in CLASSIC_POSITIONS:
        raise DKCSVValidationError(f"Unsupported roster position: {roster_position!r}")
    return "DST" if primary == "DEF" else primary


def _derive_showdown_football_position(row: pd.Series) -> str:
    """
    Resolve the underlying football position for a Showdown CPT/FLEX row.

    Prefers a true skill/DST/K label in ``Position``; falls back to other common
    DK columns when ``Position`` only mirrors the CPT/FLEX slot.
    """
    candidates = []
    for col in ("Position", "Pos", "Player Position", "FootballPosition"):
        if col in row.index and pd.notna(row[col]):
            candidates.append(str(row[col]).strip())
    for raw in candidates:
        primary = raw.split("/")[0].strip().upper()
        if primary in CLASSIC_POSITIONS:
            return "DST" if primary == "DEF" else primary
        if primary in SHOWDOWN_SLOTS:
            continue
    # Last resort: keep a neutral label so the pool still loads.
    return "UTIL"


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


def parse_dk_csv(
    source: _IO,
    *,
    contest_format: Optional[ContestFormat | str] = None,
) -> pd.DataFrame:
    """
    Load a DraftKings NFL player-pool CSV and validate required fields.

    ``contest_format`` may be passed explicitly (``classic`` / ``showdown``).
    When omitted, the format is inferred from ``Roster Position`` values.

    Returns a DataFrame with normalized columns including ``Position`` (football
    position) and a ``Projection`` column defaulting to ``AvgPointsPerGame``.
    Showdown pools also include ``RosterSlot`` (CPT/FLEX).
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

    detected = detect_contest_format(df)
    if contest_format is None:
        fmt = detected
    else:
        fmt = ContestFormat(contest_format)
        if detected is ContestFormat.SHOWDOWN and fmt is ContestFormat.CLASSIC:
            raise DKCSVValidationError(
                "CSV looks like a Showdown (CPT/FLEX) pool, but Classic format was selected."
            )
        if detected is ContestFormat.CLASSIC and fmt is ContestFormat.SHOWDOWN:
            raise DKCSVValidationError(
                "CSV looks like a Classic pool, but Showdown Captain Mode was selected. "
                "Upload a DK Showdown DKSalaries.csv with CPT/FLEX roster rows."
            )

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

    if fmt is ContestFormat.SHOWDOWN:
        slots = df["Roster Position"].str.split("/").str[0].str.strip().str.upper()
        if not slots.isin(SHOWDOWN_SLOTS).all():
            bad = sorted({s for s in slots.unique() if s not in SHOWDOWN_SLOTS})
            raise DKCSVValidationError(
                f"Showdown pools require CPT/FLEX roster rows; found: {', '.join(bad)}"
            )
        df["RosterSlot"] = slots
        df["FootballPosition"] = df.apply(_derive_showdown_football_position, axis=1)
        # ``Position`` is the football position used for game-script / jitter SDs.
        df["Position"] = df["FootballPosition"]
    else:
        if "Position" not in df.columns:
            df["Position"] = df["Roster Position"].map(_derive_classic_position)
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

    if "Ownership" in df.columns:
        df["Ownership"] = df["Ownership"]

    explicit = df["Injury"] if "Injury" in df.columns else pd.Series([""] * len(df), index=df.index)
    df["Injury"] = [
        injury_status_for_player(name, status)
        for name, status in zip(df["Name"], explicit)
    ]

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
        "Injury",
    ]
    if fmt is ContestFormat.SHOWDOWN:
        ordered = [
            "ID",
            "Name + ID",
            "Name",
            "Position",
            "FootballPosition",
            "RosterSlot",
            "Roster Position",
            "Salary",
            "TeamAbbrev",
            "Game Info",
            "AvgPointsPerGame",
            "Projection",
            "Injury",
        ]
    if "Ownership" in df.columns and "Ownership" not in ordered:
        ordered.append("Ownership")
    extras = [c for c in df.columns if c not in ordered]
    return df[[c for c in ordered if c in df.columns] + extras].reset_index(drop=True)
