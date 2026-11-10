"""Parse DraftKings Game Center standings exports into archive rows."""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import PurePosixPath
from typing import Any, Mapping, Optional, Sequence, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from .names import person_key
from .parser import parse_dk_csv
from .presets import ContestFormat

DK_CONTEST_URL = "https://api.draftkings.com/contests/v1/contests/{contest_id}?format=json"
_USER_AGENT = "dfs-lineup-generator (https://github.com/TimAuburn/dfs-lineup-generator)"
_CONTEST_ID_IN_FILENAME = re.compile(r"contest-standings-(\d+)", re.IGNORECASE)
_LONG_NUMBER = re.compile(r"(?<!\d)(\d{6,})(?!\d)")
_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_PAID_POSITIONS = re.compile(
    r"pays out the top\s+([\d,]+)\s+finishing positions",
    re.IGNORECASE,
)

_IO = Union[str, bytes, io.IOBase]

_STANDING_FIELDS = {
    "rank": ("rank",),
    "entry_id": ("entryid",),
    "entry_name": ("entryname",),
    "points": ("points",),
    "lineup": ("lineup",),
}
_PLAYER_FIELDS = {
    "player": ("player", "playername"),
    "position": ("rosterposition", "position", "pos"),
    "ownership": ("%drafted", "drafted%", "ownership", "own%"),
    "fpts": ("fpts", "actualfpts", "fantasypoints"),
}


def contest_id_from_filename(filename: str) -> str:
    """Pull the numeric contest id from a name like ``contest-standings-197055070.csv``."""
    text = str(filename or "")
    named = _CONTEST_ID_IN_FILENAME.search(text)
    if named:
        return named.group(1)
    numbers = _LONG_NUMBER.findall(text)
    return numbers[-1] if numbers else ""


def fetch_dk_contest_metadata(contest_id: str) -> dict[str, Any]:
    """Load contest name, fee, date, size, and type from the DraftKings contest API.

    Network and schema problems return empty defaults with slate type Classic.
    """
    defaults = _default_contest_metadata()
    contest_key = str(contest_id or "").strip()
    if not re.fullmatch(r"\d+", contest_key):
        return defaults
    try:
        payload = _http_get_json(DK_CONTEST_URL.format(contest_id=contest_key))
        contest = _unwrap_contest(payload)
        name = str(contest.get("name") or "").strip()
        start = contest.get("startDate")
        if start in (None, ""):
            start = contest.get("contestStartTime")
        return {
            "slate_name": name,
            "entry_fee": _optional_float(contest.get("entryFee")),
            "contest_date": _iso_date(start),
            "max_entries": _optional_int(contest.get("maximumEntries")),
            "paid_positions": _paid_positions(contest),
            "slate_type": _slate_type_from_api(contest.get("gameType"), name),
            "contest_state": str(contest.get("contestState") or "").strip(),
        }
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return defaults


def _default_contest_metadata() -> dict[str, Any]:
    return {
        "slate_name": "",
        "entry_fee": None,
        "contest_date": "",
        "max_entries": None,
        "paid_positions": None,
        "slate_type": "Classic",
        "contest_state": "",
    }


def _http_get_json(url: str) -> object:
    request = Request(
        url,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
    )
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _unwrap_contest(payload: object) -> Mapping[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Contest response is not an object.")
    detail = payload.get("contestDetail")
    if isinstance(detail, dict):
        return detail
    contests = payload.get("contests")
    if isinstance(contests, list) and contests and isinstance(contests[0], dict):
        return contests[0]
    nested = payload.get("contest")
    if isinstance(nested, dict):
        return nested
    if "name" in payload or "entryFee" in payload:
        return payload
    raise ValueError("Contest response has no contest record.")


def _slate_type_from_api(game_type: object, contest_name: str) -> str:
    parts = [_game_type_text(game_type), contest_name]
    text = " ".join(part for part in parts if part).casefold()
    if "showdown" in text or "captain" in text:
        return "Showdown"
    return "Classic"


def _game_type_text(value: object) -> str:
    if isinstance(value, dict):
        return " ".join(
            str(value.get(key) or "")
            for key in ("name", "gameTypeName", "description")
        )
    if value is None:
        return ""
    return str(value)


def _optional_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> Optional[int]:
    number = _optional_float(value)
    if number is None:
        return None
    return int(number)


def _paid_positions(contest: Mapping[str, Any]) -> Optional[int]:
    """Last finishing place that receives a prize."""
    summary = contest.get("payoutSummary")
    places: list[int] = []
    if isinstance(summary, list):
        for tier in summary:
            if not isinstance(tier, dict):
                continue
            place = _optional_int(tier.get("maxPosition"))
            if place:
                places.append(place)
    if places:
        return max(places)
    match = _PAID_POSITIONS.search(str(contest.get("contestSummary") or ""))
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def _iso_date(value: object) -> str:
    match = _ISO_DATE.search(str(value or ""))
    if not match:
        return ""
    return "-".join(match.groups())


_MAX_STANDINGS_BYTES = 200 * 1024 * 1024


def open_contest_standings(source: _IO, filename: str = "") -> tuple[str, str]:
    """Return standings CSV text and the name used to find the contest id.

    A ``.zip`` upload uses the contest-standings CSV inside the archive.
    A plain CSV is returned unchanged.
    """
    upload_name = filename or str(getattr(source, "name", "") or "")
    raw = _as_bytes(source)
    if not _looks_like_zip(raw, upload_name):
        return _as_text(raw), upload_name
    text, inner_name = _csv_from_zip(raw)
    id_name = inner_name if contest_id_from_filename(inner_name) else upload_name
    return text, id_name or upload_name


def _looks_like_zip(raw: bytes, filename: str) -> bool:
    if str(filename).lower().endswith(".zip"):
        return True
    return raw.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"))


def _csv_from_zip(raw: bytes) -> tuple[str, str]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise ValueError("Contest standings zip could not be opened.") from exc
    members = []
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        if info.is_dir() or not name.lower().endswith(".csv"):
            continue
        if "__macosx" in name.lower():
            continue
        members.append(info)
    if not members:
        raise ValueError("Contest standings zip does not contain a CSV.")
    preferred = [
        info
        for info in members
        if "contest-standings" in PurePosixPath(info.filename).name.lower()
    ]
    chosen = preferred[0] if preferred else members[0]
    if chosen.file_size > _MAX_STANDINGS_BYTES:
        raise ValueError("Contest standings CSV inside the zip is too large.")
    inner_name = PurePosixPath(chosen.filename).name
    return archive.read(chosen).decode("utf-8-sig"), inner_name


def parse_dk_contest_csv(
    file_stream: _IO,
    contest_id: Optional[str],
    metadata: Mapping[str, Any],
    salary_file_stream: Optional[_IO] = None,
) -> dict[str, Any]:
    """Split a side-by-side contest export into a slate, players, and standings.

    ``ownership_pct`` is a fraction. A blank ``contest_id`` becomes
    ``{date}-{slate_type}-{slug}``.
    """
    text = _as_text(file_stream)
    header, rows = _header_and_rows(text)
    used: set[int] = set()
    standing_idx = {
        name: _column_index(header, aliases, used)
        for name, aliases in _STANDING_FIELDS.items()
    }
    player_idx = {
        name: _column_index(header, aliases, used)
        for name, aliases in _PLAYER_FIELDS.items()
    }

    standings = _standings_rows(rows, standing_idx)
    players = _player_rows(rows, player_idx)
    if not standings:
        raise ValueError("Contest standings CSV has no entry rows.")

    slate_type = _detect_slate_type(standings, players, salary_file_stream)
    if slate_type != "Showdown" and _slate_type_from_api(
        metadata.get("slate_type"),
        str(metadata.get("slate_name") or ""),
    ) == "Showdown":
        slate_type = "Showdown"
    contest_date = str(metadata.get("contest_date") or "").strip()
    slate_name = str(metadata.get("slate_name") or "").strip()
    resolved_id = str(contest_id or "").strip() or _fallback_contest_id(
        contest_date, slate_type, slate_name
    )

    if salary_file_stream is not None:
        _apply_salaries(players, salary_file_stream, slate_type)

    total_entries = len(standings)
    cash_cutline = _cash_cutline(standings, _optional_int(metadata.get("paid_positions")))
    first_place = _points_at_rank(standings, 1)
    entry_fee = metadata.get("entry_fee")
    if entry_fee == "":
        entry_fee = None

    for row in standings:
        row["contest_id"] = resolved_id
    for player in players:
        player["contest_id"] = resolved_id

    slate = {
        "contest_id": resolved_id,
        "slate_name": slate_name,
        "sport": str(metadata.get("sport") or "NFL"),
        "slate_type": slate_type,
        "contest_date": contest_date,
        "total_entries": total_entries,
        "entry_fee": None if entry_fee is None else float(entry_fee),
        "cash_cutline": cash_cutline,
        "first_place_score": first_place,
    }
    return {"slate": slate, "players": players, "standings": standings}


def parse_contest_export(
    source: _IO,
    contest_id: Optional[str],
    metadata: Mapping[str, Any],
    salary_file_stream: Optional[_IO] = None,
) -> dict[str, Any]:
    """Parse a standings CSV or zip. A file without %Drafted still yields ownership.

    Ownership then comes from how often each player appears in the lineups.
    """
    text, _id_name = open_contest_standings(source)
    try:
        return parse_dk_contest_csv(text, contest_id, metadata, salary_file_stream)
    except ValueError as exc:
        if "missing the standings and player headers" not in str(exc):
            raise
        return _parse_standings_only(text, contest_id, metadata)


def ownership_to_fraction(raw: object) -> Optional[float]:
    """Turn ``18.4%`` or ``18.4`` into ``0.184``. Values already at or below 1 stay put."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip().replace(",", "")
    if not text or text.lower() == "nan":
        return None
    has_percent = text.endswith("%")
    text = text[:-1].strip() if has_percent else text
    value = float(text)
    if has_percent or value > 1:
        return value / 100.0
    return value


_CAPTAIN_TOKEN = re.compile(r"\bCPT\b", re.IGNORECASE)


def _detect_slate_type(
    standings: Sequence[Mapping[str, Any]],
    players: Sequence[Mapping[str, Any]],
    salary_file_stream: Optional[_IO],
) -> str:
    """Showdown when a lineup, roster spot, or salary file includes a captain."""
    for row in standings:
        if _CAPTAIN_TOKEN.search(str(row.get("lineup") or "")):
            return "Showdown"
    for player in players:
        if str(player.get("position") or "").strip().upper() == "CPT":
            return "Showdown"
    if salary_file_stream is not None and _CAPTAIN_TOKEN.search(_as_text(salary_file_stream)):
        return "Showdown"
    return "Classic"


def _apply_salaries(players: list[dict[str, Any]], source: _IO, slate_type: str) -> None:
    """Fill salary and team from a DKSalaries file. Showdown uses the flex row."""
    fmt = (
        ContestFormat.SHOWDOWN
        if slate_type.strip().lower() == "showdown"
        else ContestFormat.CLASSIC
    )
    pool = parse_dk_csv(_as_buffer(source), contest_format=fmt)
    if fmt is ContestFormat.SHOWDOWN:
        pool = _without_captain_rows(pool)
    by_identity: dict[tuple[str, str], tuple[Optional[int], Optional[str]]] = {}
    by_name: dict[str, list[tuple[Optional[int], Optional[str]]]] = {}
    for _, row in pool.iterrows():
        salary = pd.to_numeric(row["Salary"], errors="coerce")
        team = str(row.get("TeamAbbrev") or "").strip().upper() or None
        payload = (None if pd.isna(salary) else int(round(float(salary))), team)
        identity = person_key(row.get("Name"), row.get("TeamAbbrev"), row.get("Position"))
        if identity[0] and identity not in by_identity:
            by_identity[identity] = payload
        name = identity[0] if identity[0] != "dst" else person_key(row.get("Name"), row.get("TeamAbbrev"))[0]
        if name:
            by_name.setdefault(name, []).append(payload)
    for player in players:
        position = player.get("position")
        known_team = str(player.get("team") or "").strip()
        if known_team:
            hit = by_identity.get(person_key(player.get("player_name"), known_team, position))
        else:
            # Standings lineups have no team. Attach a salary only when that
            # normalized name belongs to one team, never across two clubs.
            hits = by_name.get(person_key(player.get("player_name"), "")[0], [])
            clubs = {team for _, team in hits if team}
            hit = hits[0] if len(clubs) == 1 and hits else None
        if not hit:
            continue
        player["salary"], player["team"] = hit


def _without_captain_rows(pool: pd.DataFrame) -> pd.DataFrame:
    if "RosterSlot" in pool.columns:
        slot = pool["RosterSlot"]
    elif "Roster Position" in pool.columns:
        slot = pool["Roster Position"].astype(str).str.split("/").str[0]
    elif "Position" in pool.columns:
        slot = pool["Position"]
    else:
        return pool
    return pool.loc[slot.astype(str).str.strip().str.upper() != "CPT"].copy()


_LINEUP_SLOT = re.compile(
    r"(?:^|\s)(CPT|FLEX|QB|RB|WR|TE|DST|DEF|K|UTIL)\s+",
    re.IGNORECASE,
)


def _parse_standings_only(
    text: str,
    contest_id: Optional[str],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Standings without a player block. Ownership is lineup appearances / entries."""
    header, rows = _standings_header(text)
    used: set[int] = set()
    standing_idx = {
        name: _column_index(header, aliases, used)
        for name, aliases in _STANDING_FIELDS.items()
    }
    standings = _standings_rows(rows, standing_idx)
    if not standings:
        raise ValueError("Contest standings CSV has no entry rows.")
    players = _ownership_from_lineups(standings)
    slate_type = _detect_slate_type(standings, players, None)
    if slate_type != "Showdown" and _slate_type_from_api(
        metadata.get("slate_type"),
        str(metadata.get("slate_name") or ""),
    ) == "Showdown":
        slate_type = "Showdown"
    contest_date = str(metadata.get("contest_date") or "").strip()
    slate_name = str(metadata.get("slate_name") or "").strip()
    resolved_id = str(contest_id or "").strip() or _fallback_contest_id(
        contest_date, slate_type, slate_name
    )
    total_entries = len(standings)
    for row in standings:
        row["contest_id"] = resolved_id
    for player in players:
        player["contest_id"] = resolved_id
    entry_fee = metadata.get("entry_fee")
    if entry_fee == "":
        entry_fee = None
    slate = {
        "contest_id": resolved_id,
        "slate_name": slate_name,
        "sport": str(metadata.get("sport") or "NFL"),
        "slate_type": slate_type,
        "contest_date": contest_date,
        "total_entries": total_entries,
        "entry_fee": None if entry_fee is None else float(entry_fee),
        "cash_cutline": _cash_cutline(standings, _optional_int(metadata.get("paid_positions"))),
        "first_place_score": _points_at_rank(standings, 1),
    }
    return {"slate": slate, "players": players, "standings": standings}


def _ownership_from_lineups(standings: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Share of entries whose lineup lists each player. Actual points stay empty."""
    total = len(standings)
    counts: dict[tuple[str, str], int] = {}
    for row in standings:
        seen: set[tuple[str, str]] = set()
        for position, name in _lineup_players(str(row.get("lineup") or "")):
            key = (name, position)
            if key in seen:
                continue
            seen.add(key)
            counts[key] = counts.get(key, 0) + 1
    players: list[dict[str, Any]] = []
    for (name, position), count in sorted(counts.items()):
        players.append(
            {
                "player_name": name,
                "position": position,
                "team": None,
                "salary": None,
                "ownership_pct": None if total == 0 else count / total,
                "actual_fpts": None,
            }
        )
    return players


def _lineup_players(lineup: str) -> list[tuple[str, str]]:
    matches = list(_LINEUP_SLOT.finditer(lineup))
    parsed: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(lineup)
        name = lineup[start:end].strip()
        if not name:
            continue
        position = match.group(1).upper()
        if position == "DEF":
            position = "DST"
        parsed.append((position, name))
    return parsed


def _standings_header(text: str) -> tuple[list[str], list[list[str]]]:
    table = list(csv.reader(io.StringIO(text)))
    for index, row in enumerate(table):
        keys = {_column_key(cell) for cell in row}
        if "rank" in keys and "lineup" in keys:
            return row, table[index + 1 :]
    found = ", ".join(cell.strip() for cell in table[0]) if table else "none"
    raise ValueError(f"Contest CSV is missing standings headers (Rank, Lineup). Found: {found}")


def _standings_rows(rows: Sequence[Sequence[str]], idx: Mapping[str, int]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for row in rows:
        rank_text = _cell(row, idx["rank"])
        if not rank_text:
            continue
        parsed.append(
            {
                "rank": int(float(rank_text)),
                "entry_id": _cell(row, idx["entry_id"]),
                "entry_name": _cell(row, idx["entry_name"]),
                "points": float(_cell(row, idx["points"]) or 0),
                "lineup": _cell(row, idx["lineup"]),
            }
        )
    return parsed


def _player_rows(rows: Sequence[Sequence[str]], idx: Mapping[str, int]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for row in rows:
        name = _cell(row, idx["player"])
        if not name:
            continue
        position = _cell(row, idx["position"]).split("/")[0].strip().upper()
        if position == "DEF":
            position = "DST"
        fpts_text = _cell(row, idx["fpts"])
        parsed.append(
            {
                "player_name": name,
                "position": position,
                "team": None,
                "salary": None,
                "ownership_pct": ownership_to_fraction(_cell(row, idx["ownership"])),
                "actual_fpts": float(fpts_text) if fpts_text else None,
            }
        )
    return parsed


def _cash_cutline(
    standings: Sequence[Mapping[str, Any]],
    paid_positions: Optional[int],
) -> Optional[float]:
    """Lowest score among finishing places that DraftKings paid."""
    if not paid_positions or paid_positions < 1:
        return None
    paid_scores = [
        float(row["points"])
        for row in standings
        if int(row["rank"]) <= paid_positions
    ]
    if not paid_scores:
        return None
    return round(min(paid_scores), 2)


def _points_at_rank(standings: Sequence[Mapping[str, Any]], rank: int) -> float:
    """Points for a finishing place. Tied scores skip rank numbers, so a missing rank uses that place in the ordered field."""
    for row in standings:
        if int(row["rank"]) == rank:
            return float(row["points"])
    ordered = sorted(
        standings,
        key=lambda row: (int(row["rank"]), -float(row["points"])),
    )
    if not ordered:
        raise ValueError(f"No standing row at rank {rank}.")
    index = min(max(rank, 1), len(ordered)) - 1
    return float(ordered[index]["points"])


def _fallback_contest_id(contest_date: str, slate_type: str, slate_name: str) -> str:
    return f"{contest_date}-{slate_type}-{_slug(slate_name)}"


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return cleaned or "slate"


def _header_and_rows(text: str) -> tuple[list[str], list[list[str]]]:
    table = list(csv.reader(io.StringIO(text)))
    for index, row in enumerate(table):
        keys = {_column_key(cell) for cell in row}
        if "rank" in keys and "player" in keys:
            return row, table[index + 1 :]
    found = ", ".join(cell.strip() for cell in table[0]) if table else "none"
    raise ValueError(
        "Contest CSV is missing the standings and player headers "
        f"(Rank, EntryId, Player, %Drafted). Found: {found}"
    )


def _column_index(header: Sequence[str], aliases: Sequence[str], used: set[int]) -> int:
    keys = [_column_key(cell) for cell in header]
    for alias in aliases:
        for index, key in enumerate(keys):
            if index in used or key != alias:
                continue
            used.add(index)
            return index
    raise ValueError(f"Contest CSV is missing a {aliases[0]} column.")


def _column_key(header: str) -> str:
    return re.sub(r"[^a-z0-9%]", "", str(header).strip().lower())


def _cell(row: Sequence[str], index: int) -> str:
    if index >= len(row):
        return ""
    return str(row[index]).strip()


def _as_bytes(source: _IO) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, str):
        return source.encode("utf-8")
    if hasattr(source, "getvalue"):
        data = source.getvalue()
    else:
        if hasattr(source, "seek"):
            source.seek(0)
        data = source.read()
    if isinstance(data, str):
        return data.encode("utf-8")
    return bytes(data)


def _as_text(source: _IO) -> str:
    if isinstance(source, str):
        return source
    if isinstance(source, bytes):
        return source.decode("utf-8-sig")
    if hasattr(source, "getvalue"):
        data = source.getvalue()
    else:
        if hasattr(source, "seek"):
            source.seek(0)
        data = source.read()
    if isinstance(data, bytes):
        return data.decode("utf-8-sig")
    return str(data)


def _as_buffer(source: _IO) -> io.BytesIO:
    text = _as_text(source)
    return io.BytesIO(text.encode("utf-8"))
