"""Local SQLite archive of DraftKings contest standings and player results."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from .names import person_key

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "dfs_tracker.db"
_GAME_DATE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")

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
    first_place_score REAL
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
    conn.executescript(
        """
        PRAGMA foreign_keys = OFF;
        ALTER TABLE player_performance RENAME TO player_performance_old;
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
        );
        INSERT INTO player_performance (
            contest_id, player_name, position, team, salary, ownership_pct, actual_fpts
        )
        SELECT contest_id, player_name, position, team, salary, ownership_pct, actual_fpts
        FROM player_performance_old;
        DROP TABLE player_performance_old;
        PRAGMA foreign_keys = ON;
        """
    )


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
    return conn


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
    contest_id = str(slate_meta.get("contest_id") or "").strip()
    if not contest_id:
        raise ValueError("contest_id is required")

    conn = connect(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO slates (
                    contest_id, slate_name, sport, slate_type, contest_date,
                    total_entries, entry_fee, cash_cutline, first_place_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(contest_id) DO UPDATE SET
                    slate_name = excluded.slate_name,
                    sport = excluded.sport,
                    slate_type = excluded.slate_type,
                    contest_date = excluded.contest_date,
                    total_entries = excluded.total_entries,
                    entry_fee = excluded.entry_fee,
                    cash_cutline = excluded.cash_cutline,
                    first_place_score = excluded.first_place_score
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
                ),
            )
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
    finally:
        conn.close()


def get_recent_slates(
    limit: int = 25,
    db_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Recent contests, newest date first."""
    query = """
        SELECT
            contest_date AS "Date",
            contest_id AS "Contest ID",
            slate_name AS "Slate Name",
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
