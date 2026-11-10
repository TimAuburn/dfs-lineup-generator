"""Local SQLite archive of DraftKings contest standings and player results."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Optional, Sequence

import pandas as pd

from .names import person_key

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "dfs_tracker.db"
_GAME_DATE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")

# Columns added after the first archive schema. Existing rows keep NULL here.
_SLATE_ARCHIVE_COLUMNS = ("content_hash", "imported_at", "archive_note")
_SLATE_HASH_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_slates_content_hash
ON slates(content_hash)
WHERE content_hash IS NOT NULL AND content_hash != ''
"""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS slates (
    contest_id TEXT PRIMARY KEY,
    slate_name TEXT,
    sport TEXT,
    slate_type TEXT,
    contest_date TEXT,
    total_entries INTEGER,
    entry_fee REAL,
    cash_cutline REAL,
    first_place_score REAL,
    content_hash TEXT,
    imported_at TEXT,
    archive_note TEXT
);

CREATE TABLE IF NOT EXISTS player_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contest_id TEXT REFERENCES slates(contest_id),
    player_name TEXT,
    position TEXT,
    team TEXT,
    salary INTEGER,
    ownership_pct REAL,
    actual_fpts REAL,
    UNIQUE(contest_id, player_name, team, position) ON CONFLICT REPLACE
);

CREATE TABLE IF NOT EXISTS contest_standings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contest_id TEXT REFERENCES slates(contest_id),
    entry_id TEXT,
    entry_name TEXT,
    rank INTEGER,
    points REAL,
    lineup TEXT,
    UNIQUE(contest_id, entry_id) ON CONFLICT REPLACE
);

CREATE TABLE IF NOT EXISTS monitored_contests (
    contest_id TEXT PRIMARY KEY,
    slate_name TEXT,
    entry_fee REAL,
    prize_pool REAL,
    draft_group_id TEXT
);
"""


def _migrate_player_identity(conn: sqlite3.Connection) -> None:
    """Keep two same-name players when they are on different teams.

    Older archives uniquely keyed ``(contest_id, player_name, position)``, so
    the second club replaced the first. The key now includes team. Rows that
    were already collapsed cannot be recovered.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'player_performance'"
    ).fetchone()
    if row is None or not row[0]:
        return
    sql = " ".join(str(row[0]).lower().split())
    if "unique(contest_id, player_name, team, position)" in sql:
        return
    # foreign_keys cannot change inside a transaction. The rebuild itself is
    # one transaction so a failure restores player_performance.
    conn.execute("PRAGMA foreign_keys = OFF")
    statements = (
        "ALTER TABLE player_performance RENAME TO player_performance_old",
        """
        CREATE TABLE player_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contest_id TEXT REFERENCES slates(contest_id),
            player_name TEXT,
            position TEXT,
            team TEXT,
            salary INTEGER,
            ownership_pct REAL,
            actual_fpts REAL,
            UNIQUE(contest_id, player_name, team, position) ON CONFLICT REPLACE
        )
        """,
        """
        INSERT INTO player_performance (
            contest_id, player_name, position, team, salary, ownership_pct, actual_fpts
        )
        SELECT contest_id, player_name, position, team, salary, ownership_pct, actual_fpts
        FROM player_performance_old
        """,
        "DROP TABLE player_performance_old",
    )
    try:
        conn.execute("BEGIN")
        for statement in statements:
            conn.execute(statement)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def _slate_column_names(conn: sqlite3.Connection) -> list[str]:
    return [str(row[1]) for row in conn.execute("PRAGMA table_info(slates)")]


def _slate_hash_index_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type = 'index' AND name = 'idx_slates_content_hash'
        """
    ).fetchone()
    return row is not None


def _migrate_slate_archive(conn: sqlite3.Connection) -> None:
    """Add file hash, import time, and note columns without rewriting contests.

    Older archives stored one row per contest id. A standings file with no id
    in the name was saved under ``{date}-{slate_type}-{name}``, so two such
    files shared a key and the later import replaced the earlier one. New
    columns record the file hash used to tell those imports apart. Rows that
    are already stored keep their contest ids and every existing value.

    Columns are added in place. Renaming ``slates`` would rewrite child
    foreign keys to the temporary name, including when foreign keys are off,
    and dropping that temporary table would leave those references dangling.
    """
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'slates'"
    ).fetchone()
    conn.commit()
    if row is None:
        return
    present = set(_slate_column_names(conn))
    has_index = _slate_hash_index_exists(conn)
    # The reads above open a transaction. Close it so BEGIN below is the
    # whole migration, matching _migrate_player_identity.
    conn.commit()
    missing = [name for name in _SLATE_ARCHIVE_COLUMNS if name not in present]
    if not missing and has_index:
        return
    statements = [f"ALTER TABLE slates ADD COLUMN {name} TEXT" for name in missing]
    if not has_index:
        statements.append(_SLATE_HASH_INDEX_SQL)
    try:
        conn.execute("BEGIN")
        for statement in statements:
            conn.execute(statement)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def connect(path: Optional[str | Path] = None) -> sqlite3.Connection:
    """Open the archive database, creating the file and schema when needed."""
    db_path = Path(path) if path else DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    _migrate_player_identity(conn)
    _migrate_slate_archive(conn)
    return conn


DUPLICATE_ARCHIVE_MESSAGE = (
    "This file was already imported. The duplicate was skipped, "
    "and the existing archive entry was left unchanged."
)


class ImportResult(NamedTuple):
    """Outcome of one standings import. ``status`` is imported, duplicate, or kept_both."""

    status: str
    message: str
    contest_id: str
    note: str


def file_content_hash(payload: bytes) -> str:
    """SHA-256 hex digest of an uploaded contest or results file."""
    return hashlib.sha256(payload).hexdigest()


def _import_timestamp(imported_at: str) -> str:
    text = str(imported_at or "").strip()
    if text:
        return text
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _blank_to_none(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _kept_both_note(source_contest_id: str) -> str:
    return (
        f"Kept alongside contest {source_contest_id} because this file's content differs."
    )


def _kept_both_message(source_contest_id: str, archive_key: str) -> str:
    return (
        f"Contest {source_contest_id} is already in the archive with different content. "
        f"The existing entry was kept, and this file was stored separately as {archive_key}."
    )


def salary_contest_id(file_signature: str) -> str:
    """Stable id for one uploaded DKSalaries file."""
    return f"salary-{str(file_signature).strip()}"


def slate_exists(contest_id: str, db_path: Optional[str | Path] = None) -> bool:
    """True when this contest id is already in the archive."""
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM slates WHERE contest_id = ?",
            (str(contest_id).strip(),),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def save_salary_pool_if_new(
    pool: pd.DataFrame,
    slate_type: str,
    file_signature: str,
    slate_name: str = "",
    db_path: Optional[str | Path] = None,
) -> bool:
    """Store salaries and contest type once per salary file.

    Returns False when this file signature is already archived.
    Showdown captain rows are skipped so the stored salary is the flex price.
    """
    signature = str(file_signature or "").strip()
    if not signature:
        raise ValueError("file_signature is required")
    contest_id = salary_contest_id(signature)
    if slate_exists(contest_id, db_path):
        return False

    contest_type = "Showdown" if str(slate_type).strip().lower() == "showdown" else "Classic"
    players = _salary_players(pool, contest_type)
    save_contest(
        {
            "contest_id": contest_id,
            "slate_name": slate_name.strip() or "DK salaries",
            "sport": "NFL",
            "slate_type": contest_type,
            "contest_date": _earliest_game_date(pool),
            "total_entries": None,
            "entry_fee": None,
            "cash_cutline": None,
            "first_place_score": None,
        },
        players,
        [],
        db_path,
    )
    return True


def _salary_players(pool: pd.DataFrame, slate_type: str) -> list[dict[str, Any]]:
    frame = pool.copy()
    if frame.empty:
        return []
    if slate_type == "Showdown":
        if "RosterSlot" in frame.columns:
            slot = frame["RosterSlot"]
        elif "Roster Position" in frame.columns:
            slot = frame["Roster Position"].astype(str).str.split("/").str[0]
        else:
            slot = frame.get("Position")
        if slot is not None:
            frame = frame.loc[slot.astype(str).str.strip().str.upper() != "CPT"]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for _, player in frame.iterrows():
        name = str(player.get("Name") or "").strip()
        position = str(player.get("Position") or "").split("/")[0].strip().upper()
        if position == "DEF":
            position = "DST"
        team = str(player.get("TeamAbbrev") or "").strip().upper() or None
        identity = person_key(name, team, position)
        if not name or not position or identity in seen:
            continue
        salary = pd.to_numeric(player.get("Salary"), errors="coerce")
        seen.add(identity)
        rows.append(
            {
                "player_name": name,
                "position": position,
                "team": team,
                "salary": None if pd.isna(salary) else int(round(float(salary))),
                "ownership_pct": None,
                "actual_fpts": None,
            }
        )
    return rows


def _earliest_game_date(pool: pd.DataFrame) -> str:
    if "Game Info" not in pool.columns:
        return ""
    found: list[str] = []
    for raw in pool["Game Info"].astype(str):
        match = _GAME_DATE.search(raw)
        if not match:
            continue
        month, day, year = (int(part) for part in match.groups())
        found.append(f"{year:04d}-{month:02d}-{day:02d}")
    return min(found) if found else ""


def save_contest(
    slate_meta: Mapping[str, Any],
    players: Sequence[Mapping[str, Any]],
    standings: Sequence[Mapping[str, Any]],
    db_path: Optional[str | Path] = None,
) -> None:
    """Replace one contest's slate, players, and standings in a single transaction."""
    conn = connect(db_path)
    try:
        with conn:
            _write_contest(conn, slate_meta, players, standings, replace=True)
    finally:
        conn.close()


def import_standings(
    slate_meta: Mapping[str, Any],
    players: Sequence[Mapping[str, Any]],
    standings: Sequence[Mapping[str, Any]],
    content_hash: str,
    source_contest_id: str = "",
    imported_at: str = "",
    db_path: Optional[str | Path] = None,
) -> ImportResult:
    """Store one contest or results file without replacing a different file.

    ``source_contest_id`` is the id from the file name. A blank id is stored
    under ``{content_hash}-{imported_at}`` so it does not share the parsed
    fallback key. The same content hash is skipped. A known contest id whose
    stored file differs is kept beside the original under a distinct key.
    """
    digest = str(content_hash or "").strip().lower()
    if not digest:
        raise ValueError("content_hash is required")
    source_id = str(source_contest_id or "").strip()
    stamp = _import_timestamp(imported_at)
    conn = connect(db_path)
    try:
        with conn:
            return _import_standings_in_transaction(
                conn,
                slate_meta,
                players,
                standings,
                digest,
                source_id,
                stamp,
            )
    finally:
        conn.close()


def _import_standings_in_transaction(
    conn: sqlite3.Connection,
    slate_meta: Mapping[str, Any],
    players: Sequence[Mapping[str, Any]],
    standings: Sequence[Mapping[str, Any]],
    digest: str,
    source_id: str,
    stamp: str,
) -> ImportResult:
    existing_hash = conn.execute(
        "SELECT contest_id FROM slates WHERE content_hash = ?",
        (digest,),
    ).fetchone()
    if existing_hash is not None:
        return ImportResult(
            "duplicate",
            DUPLICATE_ARCHIVE_MESSAGE,
            str(existing_hash["contest_id"]),
            "",
        )

    occupant = None
    if source_id:
        occupant = conn.execute(
            "SELECT contest_id, content_hash FROM slates WHERE contest_id = ?",
            (source_id,),
        ).fetchone()

    note = ""
    status = "imported"
    if source_id and occupant is None:
        key = source_id
    elif source_id:
        key = f"{source_id}~{digest}"
        note = _kept_both_note(source_id)
        status = "kept_both"
    else:
        key = f"{digest}-{stamp}"

    if _contest_row(conn, key) is not None:
        key = f"{key}~{stamp}"
        if source_id:
            note = _kept_both_note(source_id)
            status = "kept_both"
    message = _kept_both_message(source_id, key) if status == "kept_both" else ""
    slate = dict(slate_meta)
    slate["contest_id"] = key
    slate["content_hash"] = digest
    slate["imported_at"] = stamp
    slate["archive_note"] = note or None
    _write_contest(conn, slate, players, standings, replace=False)
    return ImportResult(status, message, key, note)


def _contest_row(conn: sqlite3.Connection, contest_id: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT contest_id, content_hash FROM slates WHERE contest_id = ?",
        (contest_id,),
    ).fetchone()


def _write_contest(
    conn: sqlite3.Connection,
    slate_meta: Mapping[str, Any],
    players: Sequence[Mapping[str, Any]],
    standings: Sequence[Mapping[str, Any]],
    replace: bool,
) -> None:
    """Insert one contest. ``replace`` updates an existing contest id in place."""
    contest_id = str(slate_meta.get("contest_id") or "").strip()
    if not contest_id:
        raise ValueError("contest_id is required")
    conflict = ""
    if replace:
        conflict = """
            ON CONFLICT(contest_id) DO UPDATE SET
                slate_name = excluded.slate_name,
                sport = excluded.sport,
                slate_type = excluded.slate_type,
                contest_date = excluded.contest_date,
                total_entries = excluded.total_entries,
                entry_fee = excluded.entry_fee,
                cash_cutline = excluded.cash_cutline,
                first_place_score = excluded.first_place_score,
                content_hash = COALESCE(excluded.content_hash, slates.content_hash),
                imported_at = COALESCE(excluded.imported_at, slates.imported_at),
                archive_note = COALESCE(excluded.archive_note, slates.archive_note)
        """
    conn.execute(
        f"""
        INSERT INTO slates (
            contest_id, slate_name, sport, slate_type, contest_date,
            total_entries, entry_fee, cash_cutline, first_place_score,
            content_hash, imported_at, archive_note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        {conflict}
        """,
        (
            contest_id,
            slate_meta.get("slate_name"),
            slate_meta.get("sport") or "NFL",
            slate_meta.get("slate_type"),
            slate_meta.get("contest_date"),
            slate_meta.get("total_entries"),
            slate_meta.get("entry_fee"),
            slate_meta.get("cash_cutline"),
            slate_meta.get("first_place_score"),
            _blank_to_none(slate_meta.get("content_hash")),
            _blank_to_none(slate_meta.get("imported_at")),
            _blank_to_none(slate_meta.get("archive_note")),
        ),
    )
    if replace:
        conn.execute(
            "DELETE FROM player_performance WHERE contest_id = ?",
            (contest_id,),
        )
        conn.execute(
            "DELETE FROM contest_standings WHERE contest_id = ?",
            (contest_id,),
        )
    conn.executemany(
        """
        INSERT INTO player_performance (
            contest_id, player_name, position, team, salary,
            ownership_pct, actual_fpts
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                contest_id,
                player.get("player_name"),
                player.get("position"),
                player.get("team"),
                player.get("salary"),
                player.get("ownership_pct"),
                player.get("actual_fpts"),
            )
            for player in players
        ],
    )
    conn.executemany(
        """
        INSERT INTO contest_standings (
            contest_id, entry_id, entry_name, rank, points, lineup
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                contest_id,
                str(row.get("entry_id") or ""),
                row.get("entry_name"),
                row.get("rank"),
                row.get("points"),
                row.get("lineup"),
            )
            for row in standings
        ],
    )


def get_recent_slates(
    limit: int = 25,
    db_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Recent contests, newest date first."""
    query = """
        SELECT
            contest_date AS "Date",
            slate_name AS "Slate Name",
            COALESCE(archive_note, '') AS "Note",
            contest_id AS "Contest ID",
            slate_type AS "Slate Type",
            total_entries AS "Total Entries",
            cash_cutline AS "Cash Cutline",
            first_place_score AS "Top Score"
        FROM slates
        ORDER BY contest_date DESC, contest_id
        LIMIT ?
    """
    return _read_frame(query, (int(limit),), db_path)


def search_player_performance(
    player_name: Optional[str] = None,
    contest_id: Optional[str] = None,
    db_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Player results for one slate, one name, or the whole archive.

    Ownership % is the stored fraction multiplied by 100.
    """
    name = (player_name or "").strip()
    contest = (contest_id or "").strip()
    query = """
        SELECT
            s.contest_date AS "Slate Date",
            s.slate_name AS "Slate Name",
            p.player_name AS "Player",
            p.position AS "Pos",
            p.team AS "Team",
            p.salary AS "Salary",
            ROUND(p.ownership_pct * 100.0, 2) AS "Ownership %",
            p.actual_fpts AS "Actual FPTS"
        FROM player_performance AS p
        JOIN slates AS s ON s.contest_id = p.contest_id
        WHERE (? = '' OR instr(lower(p.player_name), lower(?)) > 0)
          AND (? = '' OR p.contest_id = ?)
        ORDER BY s.contest_date DESC, p.ownership_pct DESC, p.player_name
    """
    return _read_frame(query, (name, name, contest, contest), db_path)


def get_chalk_and_cutline_summary(
    db_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Cash cutline, top score, and average points of players owned at 15% or more."""
    query = """
        SELECT
            s.contest_date AS "Date",
            s.contest_id AS "Contest ID",
            s.slate_name AS "Slate Name",
            s.slate_type AS "Game Style",
            s.cash_cutline AS "Cash Cutline",
            s.first_place_score AS "Top Score",
            chalk.avg_fpts AS "Chalk Avg FPTS"
        FROM slates AS s
        LEFT JOIN (
            SELECT contest_id, AVG(actual_fpts) AS avg_fpts
            FROM player_performance
            WHERE ownership_pct >= 0.15
            GROUP BY contest_id
        ) AS chalk ON chalk.contest_id = s.contest_id
        ORDER BY s.contest_date DESC, s.contest_id
    """
    return _read_frame(query, (), db_path)


def _read_frame(
    query: str,
    params: Sequence[Any],
    db_path: Optional[str | Path],
) -> pd.DataFrame:
    conn = connect(db_path)
    try:
        return pd.read_sql_query(query, conn, params=list(params))
    finally:
        conn.close()
