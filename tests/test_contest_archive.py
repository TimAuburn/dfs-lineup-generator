"""Contest standings import into a temporary SQLite archive."""

from __future__ import annotations

import json
import zipfile
from io import BytesIO, StringIO

import pytest
from urllib.error import URLError

import pandas as pd

from src.contest_parser import (
    contest_id_from_filename,
    fetch_dk_contest_metadata,
    open_contest_standings,
    ownership_to_fraction,
    parse_dk_contest_csv,
)
from src.database import (
    DUPLICATE_ARCHIVE_MESSAGE,
    _migrate_slate_archive,
    file_content_hash,
    get_chalk_and_cutline_summary,
    get_recent_slates,
    import_standings,
    save_contest,
    save_salary_pool_if_new,
    search_player_performance,
)
from src.parser import parse_dk_csv

STANDINGS = """\
Rank,EntryId,EntryName,TimeRemaining,Points,Lineup,,,,Player,Roster Position,%Drafted,FPTS
1,100,Alpha,0,180.0,CPT Josh Allen FLEX Nico Collins,,,,Josh Allen,QB,18.4%,30.2
2,200,Bravo,0,150.0,CPT Nico Collins FLEX Josh Allen,,,,Nico Collins,WR,18.4,22.0
3,300,Charlie,0,120.0,CPT Joe Mixon FLEX Dalton Schultz,,,,Joe Mixon,RB,4.0%,8.0
4,400,Delta,0,90.0,CPT Dalton Schultz FLEX Joe Mixon,,,,Dalton Schultz,TE,1%,4.0
"""

CLASSIC_STANDINGS = """\
Rank,EntryId,EntryName,TimeRemaining,Points,Lineup,,,,Player,Roster Position,%Drafted,FPTS
1,100,Alpha,0,180.0,QB Josh Allen RB Joe Mixon,,,,Josh Allen,QB,18.4%,30.2
2,200,Bravo,0,150.0,QB Josh Allen WR Nico Collins,,,,Nico Collins,WR,10%,22.0
"""

SALARIES = """\
Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame
QB,Josh Allen (1),Josh Allen,1,CPT,12000,HOU@JAX 10/18/2026 01:00PM ET,HOU,24.0
QB,Josh Allen (2),Josh Allen,2,FLEX,8000,HOU@JAX 10/18/2026 01:00PM ET,HOU,24.0
WR,Nico Collins (3),Nico Collins,3,CPT,9000,HOU@JAX 10/18/2026 01:00PM ET,HOU,18.0
WR,Nico Collins (4),Nico Collins,4,FLEX,6000,HOU@JAX 10/18/2026 01:00PM ET,HOU,18.0
RB,Joe Mixon (5),Joe Mixon,5,CPT,7500,HOU@JAX 10/18/2026 01:00PM ET,HOU,14.0
RB,Joe Mixon (6),Joe Mixon,6,FLEX,5000,HOU@JAX 10/18/2026 01:00PM ET,HOU,14.0
TE,Dalton Schultz (7),Dalton Schultz,7,CPT,4800,HOU@JAX 10/18/2026 01:00PM ET,HOU,8.0
TE,Dalton Schultz (8),Dalton Schultz,8,FLEX,3200,HOU@JAX 10/18/2026 01:00PM ET,HOU,8.0
"""

METADATA = {
    "slate_name": "HOU at JAX",
    "contest_date": "2026-10-18",
    "entry_fee": 20,
}


def test_ownership_percent_and_plain_number_become_a_fraction():
    assert ownership_to_fraction("18.4%") == 0.184
    assert ownership_to_fraction(18.4) == 0.184
    assert ownership_to_fraction(0.184) == 0.184


def test_cash_cutline_is_the_lowest_paid_score(tmp_path):
    db_path = tmp_path / "test_dfs.db"
    metadata = {**METADATA, "paid_positions": 2}
    parsed = parse_dk_contest_csv(STANDINGS, "contest-1", metadata, BytesIO(SALARIES.encode()))
    save_contest(parsed["slate"], parsed["players"], parsed["standings"], db_path)

    assert parsed["slate"]["slate_type"] == "Showdown"
    assert parsed["slate"]["total_entries"] == 4
    assert parsed["slate"]["cash_cutline"] == 150.0
    assert parsed["slate"]["first_place_score"] == 180.0
    by_name = {player["player_name"]: player for player in parsed["players"]}
    assert by_name["Josh Allen"]["ownership_pct"] == 0.184
    assert by_name["Nico Collins"]["ownership_pct"] == 0.184

    recent = get_recent_slates(db_path=db_path)
    assert recent.loc[0, "Cash Cutline"] == 150.0
    assert recent.loc[0, "Top Score"] == 180.0
    assert int(recent.loc[0, "Total Entries"]) == 4


def test_cash_cutline_survives_a_rank_skipped_by_a_tie():
    standings = """\
Rank,EntryId,EntryName,TimeRemaining,Points,Lineup,,,,Player,Roster Position,%Drafted,FPTS
1,100,Alpha,0,200.0,QB Josh Allen RB Joe Mixon,,,,Josh Allen,QB,10%,30
2,200,Bravo,0,180.0,QB Josh Allen WR Nico Collins,,,,Nico Collins,WR,10%,20
2,201,Bravo2,0,180.0,QB Josh Allen WR Nico Collins,,,,
4,300,Charlie,0,150.0,QB Josh Allen RB Joe Mixon,,,,
5,400,Delta,0,120.0,QB Josh Allen RB Joe Mixon,,,,
6,500,Echo,0,90.0,QB Josh Allen RB Joe Mixon,,,,
"""
    parsed = parse_dk_contest_csv(standings, "tied", {**METADATA, "paid_positions": 3})
    assert parsed["slate"]["total_entries"] == 6
    assert parsed["slate"]["cash_cutline"] == 180.0
    assert parsed["slate"]["first_place_score"] == 200.0


def test_reimport_updates_rows_without_duplicating(tmp_path):
    db_path = tmp_path / "test_dfs.db"
    parsed = parse_dk_contest_csv(STANDINGS, "contest-1", METADATA)
    save_contest(parsed["slate"], parsed["players"], parsed["standings"], db_path)

    parsed["standings"][0]["points"] = 200.0
    parsed["standings"][0]["entry_name"] = "Alpha updated"
    parsed["slate"]["first_place_score"] = 200.0
    parsed["players"][0]["actual_fpts"] = 40.0
    save_contest(parsed["slate"], parsed["players"], parsed["standings"], db_path)

    recent = get_recent_slates(db_path=db_path)
    assert len(recent) == 1
    assert recent.loc[0, "Top Score"] == 200.0

    players = search_player_performance(contest_id="contest-1", db_path=db_path)
    assert len(players) == 4
    allen = players.loc[players["Player"] == "Josh Allen"].iloc[0]
    assert float(allen["Actual FPTS"]) == 40.0

    summary = get_chalk_and_cutline_summary(db_path=db_path)
    assert len(summary) == 1
    assert summary.loc[0, "Game Style"] == "Showdown"
    assert summary.loc[0, "Chalk Avg FPTS"] == (40.0 + 22.0) / 2


def test_showdown_salary_file_uses_the_flex_row(tmp_path):
    db_path = tmp_path / "test_dfs.db"
    parsed = parse_dk_contest_csv(
        STANDINGS,
        "",
        METADATA,
        salary_file_stream=SALARIES,
    )
    assert parsed["slate"]["contest_id"] == "2026-10-18-Showdown-hou-at-jax"
    save_contest(parsed["slate"], parsed["players"], parsed["standings"], db_path)

    players = search_player_performance(db_path=db_path).set_index("Player")
    assert int(players.loc["Josh Allen", "Salary"]) == 8000
    assert players.loc["Josh Allen", "Team"] == "HOU"
    assert int(players.loc["Nico Collins", "Salary"]) == 6000
    assert float(players.loc["Josh Allen", "Ownership %"]) == 18.4
    assert isinstance(players, pd.DataFrame)


def test_salary_pool_saves_once_and_keeps_the_flex_price(tmp_path):
    db_path = tmp_path / "test_dfs.db"
    pool = parse_dk_csv(StringIO(SALARIES), contest_format="showdown")
    assert save_salary_pool_if_new(pool, "Showdown", "file-abc", "DKSalaries.csv", db_path)
    assert not save_salary_pool_if_new(pool, "Showdown", "file-abc", "DKSalaries.csv", db_path)

    recent = get_recent_slates(db_path=db_path)
    assert len(recent) == 1
    assert recent.loc[0, "Slate Type"] == "Showdown"
    assert recent.loc[0, "Contest ID"] == "salary-file-abc"
    assert recent.loc[0, "Date"] == "2026-10-18"

    players = search_player_performance(db_path=db_path).set_index("Player")
    assert len(players) == 4
    assert int(players.loc["Josh Allen", "Salary"]) == 8000
    assert players.loc["Josh Allen", "Team"] == "HOU"


def test_classic_lineups_without_a_captain_stay_classic():
    parsed = parse_dk_contest_csv(CLASSIC_STANDINGS, "classic-1", METADATA)
    assert parsed["slate"]["slate_type"] == "Classic"


class _JsonResponse:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _zip_bytes(files: dict[str, str]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return buffer.getvalue()


def test_zip_upload_uses_the_csv_inside_and_keeps_the_contest_id():
    raw = _zip_bytes(
        {
            "__MACOSX/._contest-standings-197055070.csv": "junk",
            "notes.txt": "ignore",
            "contest-standings-197055070.csv": STANDINGS,
        }
    )
    text, id_name = open_contest_standings(raw, "contest-standings-197055070.zip")
    assert id_name == "contest-standings-197055070.csv"
    assert contest_id_from_filename(id_name) == "197055070"
    parsed = parse_dk_contest_csv(text, "197055070", METADATA)
    assert parsed["slate"]["total_entries"] == 4
    assert parsed["slate"]["slate_type"] == "Showdown"


def test_zip_without_a_csv_is_rejected():
    raw = _zip_bytes({"readme.txt": "no standings"})
    try:
        open_contest_standings(raw, "contest-standings-1.zip")
    except ValueError as exc:
        assert "does not contain a CSV" in str(exc)
    else:
        raise AssertionError("expected a missing CSV error")


def test_contest_id_comes_from_the_standings_filename():
    assert contest_id_from_filename("contest-standings-197055070.csv") == "197055070"
    assert contest_id_from_filename("contest-standings-197055070 (1).csv") == "197055070"
    assert contest_id_from_filename("standings.csv") == ""
    assert fetch_dk_contest_metadata("")["slate_type"] == "Classic"


def test_fetch_reads_nested_contest_detail(monkeypatch):
    captured = {}

    def fake_open(request, timeout=20):
        captured["url"] = request.full_url
        captured["agent"] = request.get_header("User-agent")
        captured["timeout"] = timeout
        return _JsonResponse(
            {
                "contestDetail": {
                    "name": "NFL Showdown $15K",
                    "entryFee": 20,
                    "contestStartTime": "2026-10-18T17:00:00.0000000Z",
                    "maximumEntries": 5000,
                    "gameType": None,
                }
            }
        )

    monkeypatch.setattr("src.contest_parser.urlopen", fake_open)
    meta = fetch_dk_contest_metadata("197055070")
    assert captured["url"].endswith("/contests/197055070?format=json")
    assert "dfs-lineup-generator" in captured["agent"]
    assert meta == {
        "slate_name": "NFL Showdown $15K",
        "entry_fee": 20.0,
        "contest_date": "2026-10-18",
        "max_entries": 5000,
        "paid_positions": None,
        "slate_type": "Showdown",
        "contest_state": "",
    }


def test_fetch_prefers_start_date_and_game_type(monkeypatch):
    monkeypatch.setattr(
        "src.contest_parser.urlopen",
        lambda *_args, **_kwargs: _JsonResponse(
            {
                "name": "Thursday main",
                "entryFee": "3.5",
                "startDate": "2026-11-01T18:00:00Z",
                "contestStartTime": "2020-01-01T00:00:00Z",
                "maximumEntries": 100,
                "gameType": {"name": "Classic"},
            }
        ),
    )
    meta = fetch_dk_contest_metadata("42")
    assert meta["contest_date"] == "2026-11-01"
    assert meta["entry_fee"] == 3.5
    assert meta["slate_type"] == "Classic"
    assert meta["max_entries"] == 100


def test_fetch_falls_back_when_the_api_fails(monkeypatch):
    def offline(*_args, **_kwargs):
        raise URLError("offline")

    monkeypatch.setattr("src.contest_parser.urlopen", offline)
    assert fetch_dk_contest_metadata("197055070") == {
        "slate_name": "",
        "entry_fee": None,
        "contest_date": "",
        "max_entries": None,
        "paid_positions": None,
        "slate_type": "Classic",
        "contest_state": "",
    }


def test_paid_positions_come_from_the_last_payout_tier(monkeypatch):
    monkeypatch.setattr(
        "src.contest_parser.urlopen",
        lambda *_args, **_kwargs: _JsonResponse(
            {
                "contestDetail": {
                    "name": "NFL Showdown $7.5K Quarter Jukebox",
                    "entryFee": 0.25,
                    "contestStartTime": "2026-10-27T00:15:00.0000000Z",
                    "maximumEntries": 35671,
                    "contestSummary": "pays out the top 100 finishing positions.",
                    "payoutSummary": [
                        {"minPosition": 1, "maxPosition": 1},
                        {"minPosition": 706, "maxPosition": 8550},
                    ],
                }
            }
        ),
    )
    meta = fetch_dk_contest_metadata("197055078")
    assert meta["paid_positions"] == 8550

    monkeypatch.setattr(
        "src.contest_parser.urlopen",
        lambda *_args, **_kwargs: _JsonResponse(
            {
                "name": "Satellite",
                "contestSummary": "This contest pays out the top 1,545 finishing positions.",
            }
        ),
    )
    assert fetch_dk_contest_metadata("9")["paid_positions"] == 1545

    monkeypatch.setattr(
        "src.contest_parser.urlopen",
        lambda *_args, **_kwargs: _JsonResponse(["not", "a", "contest"]),
    )
    assert fetch_dk_contest_metadata("197055070")["slate_name"] == ""


def test_api_showdown_label_overrides_a_classic_export():
    metadata = {**METADATA, "slate_name": "NFL Showdown $15K", "slate_type": "Showdown"}
    parsed = parse_dk_contest_csv(CLASSIC_STANDINGS, "197055070", metadata)
    assert parsed["slate"]["slate_type"] == "Showdown"
    assert parsed["slate"]["contest_id"] == "197055070"


OTHER_CLASSIC = CLASSIC_STANDINGS.replace("Josh Allen", "Patrick Mahomes").replace("180.0", "175.0")
_IMPORTED_AT = "2026-11-10T17:00:00Z"
_IMPORTED_LATER = "2026-11-10T18:00:00Z"


def _import_file(db_path, text: str, source_contest_id: str, imported_at: str, metadata=None):
    parsed = parse_dk_contest_csv(text, source_contest_id, metadata or {})
    return import_standings(
        parsed["slate"],
        parsed["players"],
        parsed["standings"],
        file_content_hash(text.encode()),
        source_contest_id=source_contest_id,
        imported_at=imported_at,
        db_path=db_path,
    )


def test_two_different_files_without_contest_ids_are_both_kept(tmp_path):
    """A blank contest id used to share ``-Classic-slate`` and the second import replaced the first."""
    db_path = tmp_path / "archive.db"
    first_parsed = parse_dk_contest_csv(CLASSIC_STANDINGS, "", {})
    second_parsed = parse_dk_contest_csv(OTHER_CLASSIC, "", {})
    assert first_parsed["slate"]["contest_id"] == "-Classic-slate"
    assert second_parsed["slate"]["contest_id"] == first_parsed["slate"]["contest_id"]

    first = _import_file(db_path, CLASSIC_STANDINGS, "", _IMPORTED_AT)
    second = _import_file(db_path, OTHER_CLASSIC, "", _IMPORTED_LATER)

    assert first.status == "imported"
    assert second.status == "imported"
    assert first.contest_id != second.contest_id
    assert first.contest_id != "-Classic-slate"
    assert file_content_hash(CLASSIC_STANDINGS.encode()) in first.contest_id
    assert _IMPORTED_AT in first.contest_id
    assert file_content_hash(OTHER_CLASSIC.encode()) in second.contest_id
    assert _IMPORTED_LATER in second.contest_id

    recent = get_recent_slates(db_path=db_path)
    assert set(recent["Contest ID"]) == {first.contest_id, second.contest_id}
    scores = set(recent["Top Score"].astype(float))
    assert scores == {180.0, 175.0}
    names = set(search_player_performance(db_path=db_path)["Player"])
    assert {"Josh Allen", "Patrick Mahomes"} <= names


def test_importing_the_same_file_twice_skips_the_duplicate(tmp_path):
    db_path = tmp_path / "archive.db"
    first = _import_file(db_path, CLASSIC_STANDINGS, "", _IMPORTED_AT)
    second = _import_file(db_path, CLASSIC_STANDINGS, "", _IMPORTED_LATER)
    named_again = _import_file(
        db_path, CLASSIC_STANDINGS, "197055070", _IMPORTED_LATER, METADATA
    )

    assert first.status == "imported"
    assert second.status == "duplicate"
    assert second.message == DUPLICATE_ARCHIVE_MESSAGE
    assert "already imported" in second.message
    assert "duplicate" in second.message
    assert second.contest_id == first.contest_id
    assert named_again.status == "duplicate"
    assert named_again.message == DUPLICATE_ARCHIVE_MESSAGE
    recent = get_recent_slates(db_path=db_path)
    assert list(recent["Contest ID"]) == [first.contest_id]
    assert float(recent.loc[0, "Top Score"]) == 180.0


def test_files_with_contest_ids_keep_that_id(tmp_path):
    db_path = tmp_path / "archive.db"
    direct_path = tmp_path / "direct.db"
    parsed = parse_dk_contest_csv(CLASSIC_STANDINGS, "197055070", METADATA)
    save_contest(parsed["slate"], parsed["players"], parsed["standings"], direct_path)
    result = _import_file(db_path, CLASSIC_STANDINGS, "197055070", _IMPORTED_AT, METADATA)
    other = _import_file(db_path, OTHER_CLASSIC, "197055071", _IMPORTED_LATER, METADATA)

    assert result.status == "imported"
    assert result.contest_id == "197055070"
    assert result.message == ""
    assert other.contest_id == "197055071"

    imported = get_recent_slates(db_path=db_path).set_index("Contest ID")
    direct = get_recent_slates(db_path=direct_path).set_index("Contest ID")
    compared = ["Date", "Slate Name", "Slate Type", "Total Entries", "Cash Cutline", "Top Score"]
    left = imported.loc["197055070", compared].fillna("")
    right = direct.loc["197055070", compared].fillna("")
    assert left.tolist() == right.tolist()
    assert imported.loc["197055070", "Note"] == ""
    players = search_player_performance(contest_id="197055070", db_path=db_path)
    direct_players = search_player_performance(contest_id="197055070", db_path=direct_path)
    assert players["Player"].tolist() == direct_players["Player"].tolist()
    assert len(get_recent_slates(db_path=db_path)) == 2


def test_same_contest_id_with_different_content_keeps_both(tmp_path):
    db_path = tmp_path / "archive.db"
    first = _import_file(db_path, CLASSIC_STANDINGS, "197055070", _IMPORTED_AT, METADATA)
    second = _import_file(db_path, OTHER_CLASSIC, "197055070", _IMPORTED_LATER, METADATA)

    assert first.status == "imported"
    assert first.contest_id == "197055070"
    assert second.status == "kept_both"
    assert second.contest_id != "197055070"
    assert second.contest_id.startswith("197055070~")
    assert file_content_hash(OTHER_CLASSIC.encode()) in second.contest_id
    assert "197055070" in second.message
    assert "different content" in second.message
    assert "existing entry was kept" in second.message
    assert second.note
    assert "197055070" in second.note

    recent = get_recent_slates(db_path=db_path).set_index("Contest ID")
    assert set(recent.index) == {"197055070", second.contest_id}
    assert float(recent.loc["197055070", "Top Score"]) == 180.0
    assert recent.loc["197055070", "Note"] == ""
    assert "differs" in recent.loc[second.contest_id, "Note"]
    assert float(recent.loc[second.contest_id, "Top Score"]) == 175.0
    original = search_player_performance(contest_id="197055070", db_path=db_path)
    assert "Josh Allen" in set(original["Player"])
    assert "Patrick Mahomes" not in set(original["Player"])
    added = search_player_performance(contest_id=second.contest_id, db_path=db_path)
    assert "Patrick Mahomes" in set(added["Player"])


def _legacy_archive(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE slates (
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
        CREATE TABLE contest_standings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contest_id TEXT REFERENCES slates(contest_id),
            entry_id TEXT,
            entry_name TEXT,
            rank INTEGER,
            points REAL,
            lineup TEXT,
            UNIQUE(contest_id, entry_id) ON CONFLICT REPLACE
        );
        CREATE TABLE monitored_contests (
            contest_id TEXT PRIMARY KEY,
            slate_name TEXT,
            entry_fee REAL,
            prize_pool REAL,
            draft_group_id TEXT
        );
        INSERT INTO slates (
            contest_id, slate_name, sport, slate_type, contest_date,
            total_entries, entry_fee, cash_cutline, first_place_score
        ) VALUES
            ('197055070', 'HOU at JAX', 'NFL', 'Showdown', '2026-10-18', 4, 20.0, 150.0, 180.0),
            ('salary-file-abc', 'DK salaries', 'NFL', 'Classic', '2026-10-18', NULL, NULL, NULL, NULL);
        INSERT INTO player_performance (
            id, contest_id, player_name, position, team, salary, ownership_pct, actual_fpts
        ) VALUES
            (1, '197055070', 'Josh Allen', 'QB', 'BUF', 8000, 0.184, 30.2),
            (2, 'salary-file-abc', 'Nico Collins', 'WR', 'HOU', 6000, NULL, NULL);
        INSERT INTO contest_standings (
            id, contest_id, entry_id, entry_name, rank, points, lineup
        ) VALUES
            (7, '197055070', '100', 'Alpha', 1, 180.0, 'QB Josh Allen');
        INSERT INTO monitored_contests (
            contest_id, slate_name, entry_fee, prize_pool, draft_group_id
        ) VALUES
            ('197055070', 'HOU at JAX', 20.0, 15000.0, '55');
        """
    )
    conn.commit()
    conn.close()


def _archive_snapshot(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    snapshot = {
        "slates": [
            tuple(row)
            for row in conn.execute(
                """
                SELECT contest_id, slate_name, sport, slate_type, contest_date,
                       total_entries, entry_fee, cash_cutline, first_place_score
                FROM slates
                ORDER BY contest_id
                """
            )
        ],
        "players": [
            tuple(row)
            for row in conn.execute(
                """
                SELECT id, contest_id, player_name, position, team, salary,
                       ownership_pct, actual_fpts
                FROM player_performance
                ORDER BY id
                """
            )
        ],
        "standings": [
            tuple(row)
            for row in conn.execute(
                """
                SELECT id, contest_id, entry_id, entry_name, rank, points, lineup
                FROM contest_standings
                ORDER BY id
                """
            )
        ],
        "monitored": [
            tuple(row)
            for row in conn.execute(
                """
                SELECT contest_id, slate_name, entry_fee, prize_pool, draft_group_id
                FROM monitored_contests
                ORDER BY contest_id
                """
            )
        ],
    }
    conn.close()
    return snapshot


def test_existing_archive_is_unchanged_after_upgrade(tmp_path):
    db_path = tmp_path / "legacy.db"
    _legacy_archive(db_path)
    before = _archive_snapshot(db_path)

    from src.database import connect

    conn = connect(db_path)
    added = [
        tuple(row)
        for row in conn.execute(
            """
            SELECT contest_id, content_hash, imported_at, archive_note
            FROM slates
            ORDER BY contest_id
            """
        )
    ]
    child_sql = " ".join(
        str(
            conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'player_performance'"
            ).fetchone()[0]
        ).lower().split()
    )
    conn.execute(
        """
        INSERT INTO player_performance (contest_id, player_name, position, team)
        VALUES ('197055070', 'Nico Collins', 'WR', 'HOU')
        """
    )
    conn.commit()
    conn.close()
    after = _archive_snapshot(db_path)
    assert after["slates"] == before["slates"]
    assert after["standings"] == before["standings"]
    assert after["monitored"] == before["monitored"]
    assert after["players"][:2] == before["players"]
    assert "references slates(" in child_sql
    assert "slates_old" not in child_sql
    assert added == [
        ("197055070", None, None, None),
        ("salary-file-abc", None, None, None),
    ]
    kept = _import_file(
        db_path,
        OTHER_CLASSIC,
        "197055070",
        _IMPORTED_AT,
        {"slate_name": "Different file", "contest_date": "2026-10-18"},
    )
    assert kept.status == "kept_both"
    assert kept.contest_id != "197055070"
    upgraded = _archive_snapshot(db_path)
    original_slate = [row for row in upgraded["slates"] if row[0] == "197055070"]
    assert original_slate == [row for row in before["slates"] if row[0] == "197055070"]
    assert any(row[0] == kept.contest_id for row in upgraded["slates"])
    original_players = search_player_performance(contest_id="197055070", db_path=db_path)
    assert "Josh Allen" in set(original_players["Player"])
    assert "Patrick Mahomes" not in set(original_players["Player"])


def test_slate_migration_failure_restores_the_original_archive(tmp_path):
    import sqlite3

    db_path = tmp_path / "legacy.db"
    _legacy_archive(db_path)
    before = _archive_snapshot(db_path)

    class _FailingConnection:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, *args, **kwargs):
            text = " ".join(str(sql).split()).lower()
            if "add column archive_note" in text:
                raise sqlite3.OperationalError("disk full")
            return self._conn.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    held = sqlite3.connect(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError, match="disk full"):
            _migrate_slate_archive(_FailingConnection(held))
    finally:
        held.close()

    check = sqlite3.connect(db_path)
    names = {
        row[0]
        for row in check.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert "slates" in names
    assert "slates_old" not in names
    columns = [row[1] for row in check.execute("PRAGMA table_info(slates)")]
    check.close()
    assert "content_hash" not in columns
    assert _archive_snapshot(db_path) == before
