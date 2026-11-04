"""Tuesday harvest of completed contest standings. No live HTTP."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from src.database import connect, get_recent_slates, save_contest, search_player_performance

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "harvest_contests",
    ROOT / "scripts" / "harvest_contests.py",
)
harvest = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(harvest)

SIDE_BY_SIDE = """\
Rank,EntryId,EntryName,TimeRemaining,Points,Lineup,,,,Player,Roster Position,%Drafted,FPTS
1,100,Alpha,0,180.0,CPT Josh Allen FLEX Nico Collins,,,,Josh Allen,QB,18.4%,30.2
2,200,Bravo,0,150.0,CPT Nico Collins FLEX Josh Allen,,,,Nico Collins,WR,10%,22.0
"""

STANDINGS_ONLY = """\
Rank,EntryId,EntryName,TimeRemaining,Points,Lineup
1,100,Alpha,0,180.0,QB Josh Allen RB Joe Mixon
2,200,Bravo,0,150.0,QB Josh Allen WR Nico Collins
"""

COMPLETED = {
    "slate_name": "NFL $15K Quarter Jukebox",
    "entry_fee": 0.25,
    "contest_date": "2026-10-27",
    "paid_positions": 2,
    "slate_type": "Showdown",
    "contest_state": "Completed",
}


def _seed(db_path: Path, contest_id: str, slate_name: str) -> None:
    conn = connect(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO monitored_contests (
                    contest_id, slate_name, entry_fee, prize_pool, draft_group_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (contest_id, slate_name, 0.25, 1500, "100"),
            )
    finally:
        conn.close()


def test_completed_unfinished_and_imported_are_logged(tmp_path):
    db_path = tmp_path / "dfs.db"
    log_path = tmp_path / "data" / "harvest.log"
    _seed(db_path, "1", "NFL $15K Quarter Jukebox")
    _seed(db_path, "2", "Thursday main")
    _seed(db_path, "3", "Already saved")
    save_contest(
        {
            "contest_id": "3",
            "slate_name": "Already saved",
            "slate_type": "Classic",
            "contest_date": "2026-10-01",
            "total_entries": 1,
            "entry_fee": 1,
            "cash_cutline": 10,
            "first_place_score": 20,
        },
        [
            {
                "player_name": "Josh Allen",
                "position": "QB",
                "team": None,
                "salary": None,
                "ownership_pct": 0.5,
                "actual_fpts": 10,
            }
        ],
        [],
        db_path,
    )
    downloaded: list[str] = []

    def metadata(contest_id: str):
        if contest_id == "2":
            return {**COMPLETED, "slate_name": "Thursday main", "contest_state": "Live"}
        return dict(COMPLETED)

    def download(contest_id: str) -> bytes:
        downloaded.append(contest_id)
        return SIDE_BY_SIDE.encode()

    code = harvest.run_harvest(
        db_path,
        log_path,
        "session=abc",
        metadata_loader=metadata,
        downloader=download,
    )
    assert code == 0
    assert downloaded == ["1"]
    recent = get_recent_slates(db_path=db_path)
    jukebox = recent.loc[recent["Contest ID"] == "1"].iloc[0]
    assert float(jukebox["Cash Cutline"]) == 150.0
    players = search_player_performance(contest_id="1", db_path=db_path)
    allen = players.loc[players["Player"] == "Josh Allen"].iloc[0]
    assert float(allen["Ownership %"]) == 18.4
    assert float(allen["Actual FPTS"]) == 30.2

    log = log_path.read_text(encoding="utf-8")
    assert "1 NFL $15K Quarter Jukebox saved entries=2 players=2" in log
    assert "2 Thursday main skipped unfinished" in log
    assert "3 Already saved skipped already imported" in log
    assert "Saved 1. Skipped unfinished 1. Skipped already imported 1. Failed 0." in log


def test_login_page_stops_the_run_and_is_logged(tmp_path):
    db_path = tmp_path / "dfs.db"
    log_path = tmp_path / "data" / "harvest.log"
    _seed(db_path, "9", "NFL $6M Midseason Main Event")

    code = harvest.run_harvest(
        db_path,
        log_path,
        "session=abc",
        metadata_loader=lambda _contest_id: dict(COMPLETED),
        downloader=lambda _contest_id: b"<!doctype html><html><title>Sign Up</title>",
    )
    assert code == 1
    log = log_path.read_text(encoding="utf-8")
    assert "9 NFL $6M Midseason Main Event failed DraftKings login expired." in log
    assert "DraftKings login expired." in log
    assert "Failed 1." in log


def test_standings_without_drafted_column_counts_lineup_ownership(tmp_path):
    db_path = tmp_path / "dfs.db"
    log_path = tmp_path / "data" / "harvest.log"
    _seed(db_path, "5", "NFL $25K Pylon")
    metadata = {
        **COMPLETED,
        "slate_name": "NFL $25K Pylon",
        "slate_type": "Classic",
        "paid_positions": 1,
    }
    code = harvest.run_harvest(
        db_path,
        log_path,
        "session=abc",
        metadata_loader=lambda _contest_id: metadata,
        downloader=lambda _contest_id: STANDINGS_ONLY.encode(),
    )
    assert code == 0
    players = search_player_performance(contest_id="5", db_path=db_path).set_index("Player")
    assert float(players.loc["Josh Allen", "Ownership %"]) == 100.0
    assert float(players.loc["Joe Mixon", "Ownership %"]) == 50.0
    assert float(players.loc["Nico Collins", "Ownership %"]) == 50.0
    assert pd.isna(players.loc["Josh Allen", "Actual FPTS"])
    recent = get_recent_slates(db_path=db_path)
    assert float(recent.loc[recent["Contest ID"] == "5", "Cash Cutline"].iloc[0]) == 180.0


def test_missing_cookie_exits_with_a_log_line(tmp_path):
    log_path = tmp_path / "data" / "harvest.log"
    code = harvest.run_harvest(tmp_path / "dfs.db", log_path, "")
    assert code == 1
    assert "DK_COOKIE is missing." in log_path.read_text(encoding="utf-8")
