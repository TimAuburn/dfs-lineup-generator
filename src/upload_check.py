"""Checks a lineup set before the DraftKings upload CSV is offered.

The optimizer is not involved. Projections are not read and lineups are not
rebuilt. Each lineup comes back with blocking reasons and warnings. The
Streamlit page renders that result and withholds the file when anything blocks.

Questionable is a warning. Out, IR, Doubtful, and the other will-not-play
tags already recognized by the salary parser block the file. A salary file
with no status column is reported and is not treated as a silent pass.

``now`` is the clock for kickoff. Tests pass a fixed instant. The default is
the current time in US Eastern, the zone DraftKings uses when a cell names none.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Mapping, Optional, Sequence

import pandas as pd

from .exporter import lineups_to_dk_upload_csv, upload_columns_for
from .optimizer import FLEX_POSITIONS, NFLClassicOptimizer, NFLShowdownOptimizer
from .parser import canon_injury_status, is_inactive_injury
from .player_pool import game_label
from .portfolio import person_key, position_tokens
from .presets import ContestFormat
from .slate_games import EASTERN, game_key, kickoff_datetime, kickoff_text

# First match wins. Parsed pools always carry Injury. The other names are the
# DraftKings headers the parser aliases, so a frame that skipped parsing still
# has a place to look.
_STATUS_COLUMNS = (
    "Injury",
    "Status",
    "Injury Status",
    "Injury Indicator",
    "injury_status",
    "Inj",
)
_STATUS_LABELS = {
    "OUT": "Out",
    "DOUBTFUL": "Doubtful",
    "IR": "IR",
    "PUP": "PUP",
    "SUS": "suspended",
    "NA": "inactive",
    "QUESTIONABLE": "Questionable",
}
_CLASSIC_SLOT_POSITIONS = {
    "QB": frozenset({"QB"}),
    "RB": frozenset({"RB"}),
    "WR": frozenset({"WR"}),
    "TE": frozenset({"TE"}),
    "DST": frozenset({"DST", "DEF"}),
}
_SHOWDOWN_SLOTS = frozenset({"CPT", "FLEX"})


@dataclass(frozen=True)
class LineupFinding:
    """One lineup. ``lineup`` is 1-based. Empty tuples mean that side is clean."""

    lineup: int
    blocks: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class UploadCheck:
    """Every lineup, plus file-level blocks such as a wrong CSV header."""

    findings: tuple[LineupFinding, ...]
    export_blocks: tuple[str, ...]
    status_field: Optional[str]
    status_message: str
    salary_cap: int

    @property
    def blocked(self) -> bool:
        if self.export_blocks:
            return True
        return any(finding.blocks for finding in self.findings)

    @property
    def warnings(self) -> tuple[str, ...]:
        found: list[str] = []
        for finding in self.findings:
            for warning in finding.warnings:
                text = f"Lineup {finding.lineup}: {warning}"
                if text not in found:
                    found.append(text)
        return tuple(found)


def salary_cap_for(contest_format: ContestFormat | str, pool: Optional[pd.DataFrame] = None) -> int:
    """Cap for this slate, or the format's cap when the slate does not carry one.

    A positive ``salary_cap`` on the pool is the slate. Otherwise the figure is
    the Classic or Showdown optimizer cap, which is the DraftKings site cap
    those solvers already use.
    """
    attrs = getattr(pool, "attrs", None)
    if isinstance(attrs, Mapping) and "salary_cap" in attrs:
        raw = attrs.get("salary_cap")
        parsed = _as_int(raw)
        if parsed is not None and parsed > 0:
            return parsed
    fmt = ContestFormat(contest_format)
    if fmt is ContestFormat.SHOWDOWN:
        return int(NFLShowdownOptimizer.salary_cap)
    return int(NFLClassicOptimizer.salary_cap)


def status_column(pool: Optional[pd.DataFrame]) -> Optional[str]:
    """Column the pool uses for injury status, or None when it has none."""
    if pool is None:
        return None
    columns = {str(column) for column in pool.columns}
    for name in _STATUS_COLUMNS:
        if name in columns:
            return name
    return None


def upload_check_key(
    lineups: Sequence[pd.DataFrame],
    pool: Optional[pd.DataFrame],
    *,
    contest_format: ContestFormat | str,
    disabled_games: Sequence[str] = (),
    pool_signature: str = "",
) -> str:
    """Identity of the inputs a stored check was computed from.

    Slate, roster, injury, salary, and game toggles are all in the key. A
    stored check with any other key is stale.
    """
    fmt = ContestFormat(contest_format)
    chunks = [
        "upload-check-v1",
        fmt.value,
        str(pool_signature or ""),
        ",".join(sorted({game_key(game) for game in disabled_games if game_key(game)})),
        str(salary_cap_for(fmt, pool)),
        status_column(pool) or "",
    ]
    if pool is not None and not getattr(pool, "empty", True) and "ID" in pool.columns:
        status = status_column(pool)
        for _, row in pool.iterrows():
            chunks.append(
                "|".join(
                    [
                        _text(row.get("ID")),
                        _text(row.get("Salary")),
                        _text(row.get("Name")),
                        _text(row.get("TeamAbbrev", row.get("Team"))),
                        _text(row.get("Position")),
                        _text(row.get("RosterSlot", row.get("Roster Position"))),
                        _text(row.get("Game Info")),
                        _text(row.get(status)) if status else "",
                    ]
                )
            )
    for index, lineup in enumerate(lineups or [], start=1):
        chunks.append(f"lineup {index}")
        if lineup is None or getattr(lineup, "empty", True):
            chunks.append("empty")
            continue
        for _, row in lineup.iterrows():
            chunks.append(
                "|".join(
                    [
                        _text(row.get("Slot")),
                        _text(row.get("ID")),
                        _text(row.get("Name")),
                        _text(row.get("Team", row.get("TeamAbbrev"))),
                        _text(row.get("Position")),
                    ]
                )
            )
    return hashlib.sha1("\n".join(chunks).encode()).hexdigest()


def fresh_upload_check(result, stored_key: object, current_key: str):
    """The stored check when it belongs to ``current_key``. Otherwise nothing."""
    if result is None or stored_key != current_key:
        return None
    return result


def resolve_upload_check(state: dict, *, key: str, compute: Callable[[], object]):
    """Recompute every time. A stored result from other inputs is dropped first.

    The returned object is the one ``compute`` just produced. A previous
    result is never handed back, including when the key is unchanged, so a
    kickoff that has since passed is not hidden by an older pass.
    """
    if state.get("upload_check_key") != key:
        state.pop("upload_check", None)
        state.pop("upload_check_key", None)
    result = compute()
    state["upload_check"] = result
    state["upload_check_key"] = key
    return fresh_upload_check(state.get("upload_check"), state.get("upload_check_key"), key)


def csv_for_download(check: Optional[UploadCheck], csv_text: str) -> str:
    """The upload file when the check passed. Empty when the download is blocked."""
    if check is None or check.blocked:
        return ""
    return csv_text or ""


def failure_table(check: UploadCheck) -> pd.DataFrame:
    """One row per failing lineup, then any file-level export failures."""
    rows = []
    for finding in check.findings:
        if not finding.blocks:
            continue
        rows.append({"Lineup": finding.lineup, "Reasons": "; ".join(finding.blocks)})
    for reason in check.export_blocks:
        rows.append({"Lineup": "Export", "Reasons": reason})
    return pd.DataFrame(rows, columns=["Lineup", "Reasons"])


def warning_table(check: UploadCheck) -> pd.DataFrame:
    """Questionable players, and any kickoff that could not be read."""
    rows = []
    for finding in check.findings:
        if not finding.warnings:
            continue
        rows.append({"Lineup": finding.lineup, "Warnings": "; ".join(finding.warnings)})
    return pd.DataFrame(rows, columns=["Lineup", "Warnings"])


def validate_lineups(
    lineups: Sequence[pd.DataFrame],
    pool: Optional[pd.DataFrame],
    *,
    contest_format: ContestFormat | str,
    disabled_games: Sequence[str] = (),
    now: Optional[datetime] = None,
    csv_text: Optional[str] = None,
    export_error: Optional[str] = None,
) -> UploadCheck:
    """Check every lineup. ``csv_text`` is the file that would be downloaded.

    When ``csv_text`` is omitted and there is no ``export_error``, the current
    exporter builds the file so the header check covers that output. A passed
    check does not change those bytes.
    """
    fmt = ContestFormat(contest_format)
    cap = salary_cap_for(fmt, pool)
    field = status_column(pool)
    status_message = ""
    if field is None:
        status_message = (
            "This salary file has no injury status field, so Out, IR, and "
            "Doubtful were not checked."
        )
    clock = _clock(now)
    by_id = _index_by_id(pool)
    off = {game_key(game) for game in disabled_games if game_key(game)}
    findings = tuple(
        _check_lineup(
            index,
            lineup,
            by_id=by_id,
            pool=pool,
            fmt=fmt,
            cap=cap,
            status_field=field,
            off=off,
            clock=clock,
        )
        for index, lineup in enumerate(lineups or [], start=1)
    )
    export_blocks = _export_blocks(
        lineups,
        fmt,
        csv_text=csv_text,
        export_error=export_error,
    )
    return UploadCheck(
        findings=findings,
        export_blocks=export_blocks,
        status_field=field,
        status_message=status_message,
        salary_cap=cap,
    )


def _export_blocks(
    lineups: Sequence[pd.DataFrame],
    fmt: ContestFormat,
    *,
    csv_text: Optional[str],
    export_error: Optional[str],
) -> tuple[str, ...]:
    if not lineups:
        return ()
    if export_error:
        return (str(export_error),)
    text = csv_text
    if text is None:
        try:
            text = lineups_to_dk_upload_csv(lineups, contest_format=fmt)
        except ValueError as exc:
            return (str(exc),)
    expected = ",".join(upload_columns_for(fmt))
    first = ""
    if text:
        first = text.splitlines()[0].replace("\r", "")
    if first != expected:
        shown = first if first else "(empty)"
        return (f"CSV header is {shown}, expected {expected}.",)
    return ()


def _check_lineup(
    index: int,
    lineup: pd.DataFrame,
    *,
    by_id: Mapping[str, pd.Series],
    pool: Optional[pd.DataFrame],
    fmt: ContestFormat,
    cap: int,
    status_field: Optional[str],
    off: set[str],
    clock: datetime,
) -> LineupFinding:
    blocks: list[str] = []
    warnings: list[str] = []
    rows = _lineup_rows(lineup)
    if lineup is None or "Slot" not in getattr(lineup, "columns", []):
        return LineupFinding(index, ("Lineup has no roster slots.",), ())

    expected = upload_columns_for(fmt)
    actual = [_slot(row.get("Slot")) for row in rows]
    if Counter(actual) != Counter(expected):
        label = "Showdown" if fmt is ContestFormat.SHOWDOWN else "Classic"
        have = ", ".join(slot for slot in actual if slot) or "none"
        blocks.append(
            f"Roster slots are {have}. {label} needs {', '.join(expected)}."
        )

    seen_ids: dict[str, str] = {}
    seen_people: dict[tuple[str, str], str] = {}
    salaries: list[int] = []
    salary_complete = True
    for row in rows:
        slot = _slot(row.get("Slot"))
        player_id = _text(row.get("ID"))
        pool_row = by_id.get(player_id) if player_id else None
        name, team, positions = _identity_fields(row, pool_row)
        label = _player_label(name, team, player_id)
        if not player_id or pool_row is None:
            blocks.append(f"Player ID {player_id or '(blank)'} is not in the salary file.")
            salary_complete = False
            continue
        blocks.extend(_slot_blocks(fmt, slot, positions, pool_row, label, player_id))
        if fmt is ContestFormat.SHOWDOWN and slot == "CPT" and _pool_slot(pool_row) == "CPT":
            blocks.extend(_captain_price_blocks(pool, pool_row, label))
        person = person_key(name, team, next(iter(positions), None))
        if player_id in seen_ids:
            blocks.append(f"{label} appears twice ({seen_ids[player_id]} and {slot}).")
        elif person[0] and person in seen_people:
            blocks.append(f"{label} appears twice ({seen_people[person]} and {slot}).")
        else:
            if player_id:
                seen_ids[player_id] = slot or "roster"
            if person[0]:
                seen_people[person] = slot or "roster"
        salary = _as_int(pool_row.get("Salary"))
        if salary is None:
            blocks.append(f"Salary for {label} could not be read from the salary file.")
            salary_complete = False
        else:
            salaries.append(salary)
        if status_field is not None:
            blocks.extend(_status_blocks(pool_row.get(status_field), label))
            warnings.extend(_status_warnings(pool_row.get(status_field), label))
        blocks.extend(_game_blocks(pool_row.get("Game Info"), label, off, clock))
        warnings.extend(_game_warnings(pool_row.get("Game Info"), label))

    if salary_complete and salaries and sum(salaries) > cap:
        blocks.append(f"Salary is ${sum(salaries):,}, over the ${cap:,} salary cap.")
    return LineupFinding(index, _unique(blocks), _unique(warnings))


def _slot_blocks(
    fmt: ContestFormat,
    slot: str,
    positions: set[str],
    pool_row: pd.Series,
    label: str,
    player_id: str,
) -> list[str]:
    if fmt is ContestFormat.SHOWDOWN:
        if slot not in _SHOWDOWN_SLOTS:
            return []
        pool_slot = _pool_slot(pool_row)
        if pool_slot == slot:
            return []
        if pool_slot in _SHOWDOWN_SLOTS:
            return [
                f"{slot} uses DraftKings ID {player_id}, which is the {pool_slot} row "
                f"for {label}, not the {slot} row."
            ]
        return [f"{slot} for {label} has no {slot} row on the salary file."]
    allowed = FLEX_POSITIONS if slot == "FLEX" else _CLASSIC_SLOT_POSITIONS.get(slot)
    if allowed is None:
        return []
    if positions & set(allowed):
        return []
    owned = "/".join(sorted(positions)) if positions else "no position"
    return [f"{label} is {owned} and is not eligible for {slot}."]


def _captain_price_blocks(
    pool: Optional[pd.DataFrame],
    captain: pd.Series,
    label: str,
) -> list[str]:
    """CPT salary on the file has to be 1.5x that player's flex salary."""
    captain_salary = _as_int(captain.get("Salary"))
    flex_salary = _flex_salary(pool, captain)
    if captain_salary is None or flex_salary is None:
        return [
            f"{label} has no flex salary on the salary file, so the 1.5x captain price "
            "was not confirmed."
        ]
    if captain_salary * 2 == flex_salary * 3:
        return []
    return [
        f"CPT salary for {label} is ${captain_salary:,}, not 1.5x the ${flex_salary:,} flex salary."
    ]


def _flex_salary(pool: Optional[pd.DataFrame], captain: pd.Series) -> Optional[int]:
    if pool is None or pool.empty:
        return None
    captain_person = person_key(
        captain.get("Name"),
        captain.get("TeamAbbrev", captain.get("Team")),
        captain.get("Position"),
    )
    for _, row in pool.iterrows():
        if _pool_slot(row) != "FLEX":
            continue
        other = person_key(
            row.get("Name"),
            row.get("TeamAbbrev", row.get("Team")),
            row.get("Position"),
        )
        if other == captain_person and other[0]:
            return _as_int(row.get("Salary"))
    return None


def _status_blocks(raw: object, label: str) -> list[str]:
    status = canon_injury_status(raw)
    if not status or not is_inactive_injury(status):
        return []
    name = _STATUS_LABELS.get(status, status.title())
    return [f"{label} is {name}."]


def _status_warnings(raw: object, label: str) -> list[str]:
    status = canon_injury_status(raw)
    if status != "QUESTIONABLE":
        return []
    return [f"{label} is Questionable."]


def _game_blocks(raw: object, label: str, off: set[str], clock: datetime) -> list[str]:
    reasons: list[str] = []
    key = game_key(game_label(raw))
    if key and key in off:
        reasons.append(f"{label} is in {key}, which is off.")
    if _already_started(raw):
        where = key or "this game"
        reasons.append(f"{label} is in {where}, which is already underway.")
        return reasons
    kick = kickoff_datetime(raw)
    if kick is not None and kick <= clock:
        when = kickoff_text(raw)
        where = key or "this game"
        reasons.append(f"{label} is in {where}, and kickoff has passed ({when}).")
    return reasons


def _game_warnings(raw: object, label: str) -> list[str]:
    if _already_started(raw) or kickoff_datetime(raw) is not None:
        return []
    if game_label(raw) == "":
        return [f"Game for {label} could not be read, so kickoff was not checked."]
    return [f"Kickoff time for {label} could not be read, so it was not checked."]


def _already_started(raw: object) -> bool:
    text = " ".join(_text(raw).casefold().split())
    if text in {"in progress", "final"}:
        return True
    return text.endswith(" in progress") or text.endswith(" final")


def _identity_fields(lineup_row: pd.Series, pool_row: Optional[pd.Series]) -> tuple[str, str, set[str]]:
    source = pool_row if pool_row is not None else lineup_row
    name = _text(source.get("Name")) or _text(lineup_row.get("Name"))
    team = _text(source.get("TeamAbbrev")) or _text(source.get("Team"))
    if not team:
        team = _text(lineup_row.get("Team")) or _text(lineup_row.get("TeamAbbrev"))
    position_text = source.get("Position")
    if position_text is None or not _text(position_text):
        position_text = lineup_row.get("Position")
    return name, team, position_tokens(position_text)


def _pool_slot(row: pd.Series) -> str:
    if "RosterSlot" in row.index and _text(row.get("RosterSlot")):
        return _slot(row.get("RosterSlot"))
    return _slot(str(row.get("Roster Position") or "").split("/")[0])


def _index_by_id(pool: Optional[pd.DataFrame]) -> dict[str, pd.Series]:
    found: dict[str, pd.Series] = {}
    if pool is None or pool.empty or "ID" not in pool.columns:
        return found
    for _, row in pool.iterrows():
        player_id = _text(row.get("ID"))
        if player_id:
            found[player_id] = row
    return found


def _lineup_rows(lineup: pd.DataFrame) -> list[pd.Series]:
    if lineup is None or getattr(lineup, "empty", True):
        return []
    return [row for _, row in lineup.iterrows()]


def _clock(now: Optional[datetime]) -> datetime:
    if now is None:
        return datetime.now(EASTERN)
    if now.tzinfo is None:
        return now.replace(tzinfo=EASTERN)
    return now


def _player_label(name: str, team: str, player_id: str) -> str:
    club = team.strip().upper()
    if club in {"NAN", "NONE", "NAT"}:
        club = ""
    if name and club:
        return f"{name} ({club})"
    if name:
        return name
    return f"ID {player_id or '(blank)'}"


def _slot(value: object) -> str:
    return _text(value).upper()


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.casefold() in {"nan", "none", "<na>", "nat"}:
        return ""
    return text


def _as_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _unique(items: Sequence[str]) -> tuple[str, ...]:
    found: list[str] = []
    for item in items:
        if item and item not in found:
            found.append(item)
    return tuple(found)
