"""Lobby selection for the scheduled contest sync."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "sync_contests",
    ROOT / "scripts" / "sync_contests.py",
)
sync = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(sync)


def _contest(**overrides):
    row = {
        "id": 1,
        "n": "NFL $15K Quarter Jukebox",
        "a": 0.25,
        "po": 1500,
        "dg": 100,
        "gameType": "Showdown Captain Mode",
        "attr": {},
    }
    row.update(overrides)
    return row


def _group(draft_group_id, suffix, start="2026-11-01T18:00:00.0000000Z"):
    return {
        "DraftGroupId": draft_group_id,
        "ContestStartTimeSuffix": suffix,
        "StartDate": start,
    }


def _ids(payload):
    return {row["contest_id"] for row in sync.select_contests(payload)}


def test_named_tournaments_and_millionaire_lobby_class_are_kept():
    payload = {
        "Contests": [
            _contest(id=1, n="NFL $15K Quarter Jukebox", po=1500),
            _contest(
                id=2,
                n="Weekly feature",
                gameType="Classic",
                attr={"LobbyClass": "icon-millionaire"},
            ),
        ],
        "DraftGroups": [],
    }
    chosen = {row["contest_id"]: row for row in sync.select_contests(payload)}
    assert set(chosen) == {"1", "2"}
    assert chosen["1"]["entry_fee"] == 0.25
    assert chosen["1"]["prize_pool"] == 1500
    assert chosen["1"]["draft_group_id"] == "100"


def test_other_gpps_satellites_and_boosters_are_excluded():
    payload = {
        "Contests": [
            _contest(id=10, n="NFL $700K Field General", po=700000, gameType="Classic"),
            _contest(id=11, n="NFL SUPERSat to $20 NFL Fantasy Football Millionaire"),
            _contest(id=12, n="NFL $8K Jukebox Booster"),
            _contest(id=13, n=""),
            {"n": "NFL $15K Quarter Jukebox", "gameType": "Classic"},
        ],
        "DraftGroups": [],
    }
    assert _ids(payload) == set()


def test_each_showdown_game_keeps_the_five_and_fifty_dollar_double_up():
    groups = [
        _group(1, " (PHI @ WAS)"),
        _group(2, " (CHI @ SEA)", "2026-11-02T18:00:00.0000000Z"),
    ]
    payload = {
        "DraftGroups": groups,
        "Contests": [
            _contest(id=9, n="NFL Showdown $5 Double Up (PHI @ WAS)", a=5, po=3000, dg=1),
            _contest(id=3, n="NFL Showdown $5 Double Up (PHI @ WAS)", a=5, po=3000, dg=1),
            _contest(id=4, n="NFL Showdown $1 Double Up (PHI @ WAS)", a=1, po=9000, dg=1),
            _contest(id=5, n="NFL Showdown $25 Double Up (PHI @ WAS)", a=25, po=9000, dg=1),
            _contest(id=8, n="NFL Showdown $50 Double Up (PHI @ WAS)", a=50, po=1000, dg=1),
            _contest(id=7, n="NFL Showdown $50 Double Up (PHI @ WAS)", a=50, po=500, dg=1),
            _contest(id=20, n="NFL Showdown $5 Double Up (CHI @ SEA)", a=5, po=400, dg=2),
            _contest(id=21, n="NFL Showdown $50 Double Up (CHI @ SEA)", a=50, po=400, dg=2),
            _contest(id=30, n="NFL $5 Double Up", a=5, po=12000, dg=1, gameType="Classic"),
            _contest(
                id=31,
                n="NFL Showdown $5 Double Up",
                a=5,
                po=9000,
                dg=9,
                gameType="Showdown Captain Mode",
            ),
        ],
    }
    chosen = {row["contest_id"]: row for row in sync.select_contests(payload)}
    assert set(chosen) == {"3", "8", "20", "21"}
    assert chosen["3"]["entry_fee"] == 5
    assert chosen["8"]["entry_fee"] == 50
    assert chosen["3"]["slate_name"].endswith("(PHI @ WAS)")
    assert chosen["20"]["slate_name"].endswith("(CHI @ SEA)")


def test_resave_replaces_the_same_contest(tmp_path):
    db_path = tmp_path / "tracker.db"
    original = {
        "contest_id": "42",
        "slate_name": "NFL $15K Quarter Jukebox",
        "entry_fee": 0.25,
        "prize_pool": 1500,
        "draft_group_id": "100",
    }
    updated = {**original, "prize_pool": 2000, "slate_name": "NFL $20K Quarter Jukebox"}
    assert sync.save_contests([original], db_path) == 1
    assert sync.save_contests([updated], db_path) == 1

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT contest_id, slate_name, prize_pool FROM monitored_contests"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("42", "NFL $20K Quarter Jukebox", 2000)]
