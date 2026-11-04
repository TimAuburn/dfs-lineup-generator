"""Index active NFL tournaments from the DraftKings lobby into the local archive.

Run with the project virtualenv so Task Scheduler does not need Streamlit:

    venv\\Scripts\\python.exe scripts\\sync_contests.py

The database path is resolved from this file, not the working directory.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "dfs_tracker.db"
LOBBY_URL = "https://www.draftkings.com/lobby/getcontests?sport=NFL"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS monitored_contests (
    contest_id TEXT PRIMARY KEY,
    slate_name TEXT,
    entry_fee REAL,
    prize_pool REAL,
    draft_group_id TEXT
);
"""

_NAME_KEEP = re.compile(
    r"millionaire|main event|quarter arcade|jukebox|pylon|fair catch|play-action|first down",
    re.IGNORECASE,
)
_NAME_DROP = re.compile(
    r"satellite|qualifier|ticket|step|booster|supersat",
    re.IGNORECASE,
)
_MAJOR_FORMATS = {"Classic", "Showdown Captain Mode"}
_DOUBLE_UP_FEES = {5.0, 50.0}


def fetch_lobby() -> dict[str, Any]:
    """Download the public NFL contest lobby."""
    request = Request(
        LOBBY_URL,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Lobby response is not an object.")
    return payload


def select_contests(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Major tournaments, plus the $5 and $50 double-up for each showdown game."""
    contests = payload.get("Contests") or []
    if not isinstance(contests, list):
        return []
    groups = {
        group.get("DraftGroupId"): group
        for group in (payload.get("DraftGroups") or [])
        if isinstance(group, dict)
    }
    chosen: dict[str, dict[str, Any]] = {}
    for contest in contests:
        if not isinstance(contest, dict) or not _is_major(contest):
            continue
        row = _row(contest)
        if row is not None:
            chosen[row["contest_id"]] = row
    for row in _double_up_rows(contests, groups):
        chosen[row["contest_id"]] = row
    return list(chosen.values())


def save_contests(
    rows: Sequence[Mapping[str, Any]],
    db_path: Optional[str | Path] = None,
) -> int:
    """Insert or replace the selected contests. Closed contests are left in place."""
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(_SCHEMA)
        conn.executemany(
            """
            INSERT OR REPLACE INTO monitored_contests (
                contest_id, slate_name, entry_fee, prize_pool, draft_group_id
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    row["contest_id"],
                    row["slate_name"],
                    row["entry_fee"],
                    row["prize_pool"],
                    row["draft_group_id"],
                )
                for row in rows
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def _is_major(contest: Mapping[str, Any]) -> bool:
    if contest.get("gameType") not in _MAJOR_FORMATS:
        return False
    name = str(contest.get("n") or "")
    if _excluded(name) or "double up" in name.casefold():
        return False
    lobby = str((contest.get("attr") or {}).get("LobbyClass") or "")
    return bool(_NAME_KEEP.search(name) or "millionaire" in lobby.casefold())


def _double_up_rows(
    contests: Sequence[Any],
    groups: Mapping[Any, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for contest in contests:
        if not isinstance(contest, dict):
            continue
        if contest.get("gameType") != "Showdown Captain Mode":
            continue
        name = str(contest.get("n") or "")
        if "double up" not in name.casefold() or _excluded(name):
            continue
        fee = _number(contest.get("a"))
        if fee not in _DOUBLE_UP_FEES:
            continue
        group = groups.get(contest.get("dg")) or {}
        suffix = str(group.get("ContestStartTimeSuffix") or "").strip()
        if "@" not in suffix:
            continue
        start = str(group.get("StartDate") or "")[:10]
        row = _row(contest)
        if row is None:
            continue
        buckets.setdefault((start, suffix, fee), []).append(row)
    return [_preferred(rows) for rows in buckets.values()]


def _preferred(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Largest prize pool, then the lowest contest id."""
    return min(
        rows,
        key=lambda row: (
            -(row["prize_pool"] if row["prize_pool"] is not None else -1.0),
            _contest_sort_key(str(row["contest_id"])),
        ),
    )


def _contest_sort_key(contest_id: str) -> tuple:
    if contest_id.isdigit():
        return (0, int(contest_id))
    return (1, contest_id)


def _excluded(name: str) -> bool:
    return bool(_NAME_DROP.search(name))


def _row(contest: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    contest_id = str(contest.get("id") or "").strip()
    if not contest_id:
        return None
    draft_group = contest.get("dg")
    return {
        "contest_id": contest_id,
        "slate_name": str(contest.get("n") or "").strip(),
        "entry_fee": _number(contest.get("a")),
        "prize_pool": _number(contest.get("po")),
        "draft_group_id": None if draft_group in (None, "") else str(draft_group),
    }


def _number(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    try:
        saved = save_contests(select_contests(fetch_lobby()))
    except (
        HTTPError,
        URLError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
        sqlite3.Error,
        ValueError,
        TypeError,
        KeyError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Saved {saved} contests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
