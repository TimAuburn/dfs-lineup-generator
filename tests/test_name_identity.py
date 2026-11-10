"""Name identity and lineup uniqueness.

These cases failed on the previous normalizer and on pydfs UniquePlayerRule,
which groups by the raw full name.
"""

from __future__ import annotations

import logging
from io import StringIO

import pandas as pd
import pytest

from src.contest_parser import _apply_salaries
from src.database import connect, search_player_performance
from src.database import save_salary_pool_if_new
from src.optimizer import NFLClassicOptimizer, NFLShowdownOptimizer
from src.player_pool import person_key
from src.portfolio import person_key as portfolio_person_key
from src.sleeper import (
    index_projections,
    match_sleeper_pool,
    normalize_player_name,
)


def test_spelling_variants_share_one_normalized_name():
    canonical = "amon ra st brown"
    for spelling in (
        "Amon-Ra St. Brown",
        "Amon Ra St Brown",
        "amon-ra st. brown",
        "AMON-RA ST. BROWN",
        "Amon–Ra St. Brown",
        "  Amon   Ra    St.   Brown  ",
    ):
        assert normalize_player_name(spelling) == canonical
    assert normalize_player_name("D.J. Moore") == normalize_player_name("DJ Moore")
    assert normalize_player_name("Ja'Marr Chase") == normalize_player_name("Jamarr Chase")
    assert normalize_player_name("Ja’Marr Chase") == "jamarr chase"
    assert normalize_player_name("José García") == normalize_player_name("Jose Garcia")
    assert normalize_player_name("Kenneth Walker III") == normalize_player_name("Kenneth Walker")
    assert normalize_player_name("Travis Etienne Jr.") == normalize_player_name("Travis Etienne")
    assert normalize_player_name("Travis Etienne Jr. (Q)") == "travis etienne"
    assert normalize_player_name("Kenneth Walker V") == "kenneth walker"
    assert normalize_player_name("Kenneth Walker Sr") == "kenneth walker"


def test_person_key_is_normalized_name_and_team_and_defense_by_team():
    assert person_key("Josh Allen", "BUF") == ("josh allen", "BUF")
    assert person_key("Josh Allen", " buf ") == portfolio_person_key("Josh Allen", "BUF")
    assert person_key("Amon-Ra St. Brown", "DET") == person_key("Amon Ra St Brown", "det")
    assert person_key("D.J. Moore", "CHI") == person_key("DJ Moore", "CHI")
    assert person_key("Ja'Marr Chase", "CIN") == person_key("Jamarr Chase", "CIN")
    assert person_key("Kenneth Walker III", "SEA") == person_key("Kenneth Walker", "SEA")
    assert person_key("José García", "ARI") == ("jose garcia", "ARI")
    # Same normalized name, different clubs.
    assert person_key("Michael Thomas", "NO") != person_key("Michael Thomas", "LAR")
    # Existing team-abbreviation map only.
    assert person_key("Josh Allen", "JAC") == person_key("Josh Allen", "JAX")
    assert person_key("49ers", "SF", "DST") == person_key("San Francisco", "SF", "DEF")
    assert person_key("49ers", "SF", "DST") == ("dst", "SF")
    assert person_key("49ers", "SEA", "DST") != person_key("49ers", "SF", "DST")


def _projection(first, last, team, position, points, injury=None):
    return {
        "team": team,
        "player": {
            "first_name": first,
            "last_name": last,
            "position": position,
            "team": team,
            "injury_status": injury,
        },
        "stats": {"pts_ppr": points},
    }


def _pool_row(name, team, position, projection=9.0):
    return {
        "ID": "1",
        "Name": name,
        "Position": position,
        "TeamAbbrev": team,
        "AvgPointsPerGame": projection,
        "Projection": projection,
        "Injury": "",
    }


def test_hyphen_space_period_and_suffix_variants_attach_projection_and_injury():
    index = index_projections([
        _projection("Amon-Ra", "St. Brown", "DET", "WR", 18.5, "Questionable"),
        _projection("D.J.", "Moore", "CHI", "WR", 16.0),
        _projection("Ja'Marr", "Chase", "CIN", "WR", 19.0),
        _projection("Kenneth", "Walker III", "SEA", "RB", 15.0, "Questionable"),
        _projection("José", "García", "ARI", "TE", 11.0),
    ])
    pool = pd.DataFrame([
        _pool_row("Amon Ra St Brown", "DET", "WR"),
        _pool_row("DJ Moore", "CHI", "WR"),
        _pool_row("Jamarr Chase", "CIN", "WR"),
        _pool_row("Kenneth Walker", "SEA", "RB"),
        _pool_row("Jose Garcia", "ARI", "TE"),
    ])
    pool["ID"] = [str(i) for i in range(1, len(pool) + 1)]
    updated, report = match_sleeper_pool(pool, index)
    by_id = updated.set_index("ID")
    assert by_id.loc["1", "MatchStatus"] == "projection"
    assert float(by_id.loc["1", "Projection"]) == pytest.approx(18.5)
    assert by_id.loc["1", "Injury"] == "QUESTIONABLE"
    assert by_id.loc["2", "MatchStatus"] == "projection"
    assert by_id.loc["3", "MatchStatus"] == "projection"
    assert by_id.loc["4", "MatchStatus"] == "projection"
    assert by_id.loc["4", "Injury"] == "QUESTIONABLE"
    assert by_id.loc["5", "MatchStatus"] == "projection"
    assert report["unmatched"] == []


def test_same_normalized_name_on_two_teams_does_not_cross_match():
    index = index_projections([
        _projection("Michael", "Thomas", "NO", "WR", 14.0, "Questionable"),
        _projection("Michael", "Thomas", "LAR", "WR", 8.0),
    ])
    pool = pd.DataFrame([
        _pool_row("Michael Thomas", "NO", "WR"),
        _pool_row("Michael Thomas", "LAR", "WR"),
    ])
    pool["ID"] = ["no", "lar"]
    updated, report = match_sleeper_pool(pool, index)
    by_id = updated.set_index("ID")
    assert float(by_id.loc["no", "Projection"]) == pytest.approx(14.0)
    assert by_id.loc["no", "Injury"] == "QUESTIONABLE"
    assert float(by_id.loc["lar", "Projection"]) == pytest.approx(8.0)
    assert by_id.loc["lar", "Injury"] == ""
    assert report["unmatched"] == []


def test_defense_matches_by_team_not_nickname():
    index = index_projections([
        _projection("San Francisco", "49ers", "SF", "DEF", 7.5),
    ])
    pool = pd.DataFrame([_pool_row("49ers", "SF", "DST")])
    updated, report = match_sleeper_pool(pool, index)
    assert updated.loc[0, "MatchStatus"] == "projection"
    assert float(updated.loc[0, "Projection"]) == pytest.approx(7.5)
    assert report["unmatched"] == []


def test_cached_hyphenated_name_is_renormalized_on_match():
    index = {
        "projections": [{
            "player_id": "9",
            "name": "amon-ra st brown",
            "team": "DET",
            "position": "WR",
            "stats": {"pts_ppr": 17.0},
            "injury": "QUESTIONABLE",
            "projection": None,
        }],
        "depth": [],
    }
    pool = pd.DataFrame([_pool_row("Amon-Ra St. Brown", "DET", "WR")])
    updated, report = match_sleeper_pool(pool, index)
    assert updated.loc[0, "MatchStatus"] == "projection"
    assert float(updated.loc[0, "Projection"]) == pytest.approx(17.0)
    assert report["unmatched"] == []


def test_unmatched_player_is_warned_and_logged_without_salary(caplog):
    index = index_projections([_projection("Josh", "Allen", "BUF", "QB", 22.0)])
    pool = pd.DataFrame([
        _pool_row("Josh Allen", "BUF", "QB", projection=22.0),
        {
            "ID": "2",
            "Name": "Not A Real Person",
            "Position": "WR",
            "TeamAbbrev": "BUF",
            "Salary": 9900,
            "AvgPointsPerGame": 3.0,
            "Projection": 3.0,
            "Injury": "",
        },
    ])
    with caplog.at_level(logging.WARNING, logger="src.sleeper"):
        updated, report = match_sleeper_pool(pool, index)
    assert [item["name"] for item in report["unmatched"]] == ["Not A Real Person"]
    from src.sleeper import unmatched_players_warning

    warning = unmatched_players_warning({**report, "matching_ran": True})
    assert "Not A Real Person" in warning
    assert "BUF" in warning
    assert "WR" in warning
    assert "9900" not in warning
    assert "9900" not in caplog.text
    assert "Not A Real Person" in caplog.text
    assert unmatched_players_warning({"matching_ran": False, "unmatched": report["unmatched"]}) == ""
    assert updated.loc[updated["ID"].astype(str).eq("2"), "MatchStatus"].iloc[0] == "unmatched"


def _salary_csv(rows: list[tuple]) -> StringIO:
    lines = ["Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame"]
    for position, name, pid, team, salary in rows:
        lines.append(
            f"{position},{name} ({pid}),{name},{pid},{position},{salary},"
            f"KC@BUF 09/10/2025 08:20PM ET,{team},10"
        )
    return StringIO("\n".join(lines))


def test_salary_join_uses_normalized_name_and_team():
    players = [
        {"player_name": "Amon Ra St Brown", "position": "WR", "team": "DET", "salary": None},
        {"player_name": "Michael Thomas", "position": "WR", "team": None, "salary": None},
        {"player_name": "49ers", "position": "DST", "team": "SF", "salary": None},
    ]
    _apply_salaries(
        players,
        _salary_csv([
            ("WR", "Amon-Ra St. Brown", "1", "DET", 7000),
            ("WR", "Michael Thomas", "2", "NO", 6000),
            ("WR", "Michael Thomas", "3", "LAR", 5000),
            ("DST", "San Francisco", "4", "SF", 3200),
        ]),
        "Classic",
    )
    assert players[0]["salary"] == 7000
    assert players[0]["team"] == "DET"
    assert players[1]["salary"] is None
    assert players[1]["team"] is None
    assert players[2]["salary"] == 3200
    assert players[2]["team"] == "SF"


def test_archive_keeps_same_name_players_on_different_teams(tmp_path):
    db_path = tmp_path / "archive.db"
    pool = pd.DataFrame([
        {
            "Name": "Michael Thomas",
            "Position": "WR",
            "TeamAbbrev": "NO",
            "Salary": 6000,
            "Game Info": "NO@ATL 09/10/2025 01:00PM ET",
        },
        {
            "Name": "Michael Thomas",
            "Position": "WR",
            "TeamAbbrev": "LAR",
            "Salary": 5000,
            "Game Info": "LAR@SEA 09/10/2025 04:25PM ET",
        },
    ])
    assert save_salary_pool_if_new(pool, "Classic", "sig", db_path=db_path) is True
    found = search_player_performance(db_path=db_path)
    teams = set(found["Team"].astype(str))
    assert teams == {"NO", "LAR"}
    assert len(found) == 2


def test_archive_migration_adds_team_to_the_unique_key(tmp_path):
    db_path = tmp_path / "old.db"
    raw = __import__("sqlite3").connect(db_path)
    raw.executescript(
        """
        CREATE TABLE slates (
            contest_id TEXT PRIMARY KEY,
            slate_name TEXT
        );
        CREATE TABLE player_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contest_id TEXT,
            player_name TEXT,
            position TEXT,
            team TEXT,
            salary INTEGER,
            ownership_pct REAL,
            actual_fpts REAL,
            UNIQUE(contest_id, player_name, position) ON CONFLICT REPLACE
        );
        INSERT INTO slates (contest_id, slate_name) VALUES ('salary-sig', 'old');
        INSERT INTO player_performance (
            contest_id, player_name, position, team, salary
        ) VALUES ('salary-sig', 'Michael Thomas', 'WR', 'NO', 6000);
        """
    )
    raw.commit()
    raw.close()
    conn = connect(db_path)
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'player_performance'"
    ).fetchone()[0]
    assert "unique(contest_id, player_name, team, position)" in " ".join(sql.lower().split())
    conn.execute(
        """
        INSERT INTO player_performance (contest_id, player_name, position, team, salary)
        VALUES ('salary-sig', 'Michael Thomas', 'WR', 'LAR', 5000)
        """
    )
    conn.commit()
    teams = {
        row[0]
        for row in conn.execute(
            "SELECT team FROM player_performance WHERE player_name = 'Michael Thomas'"
        )
    }
    conn.close()
    assert teams == {"NO", "LAR"}


def _classic_same_name_pool() -> pd.DataFrame:
    rows = [
        ("QB", "100", "Josh Allen", "BUF", 6000, 20, "KC@BUF 09/10/2025 08:20PM ET"),
        ("RB", "201", "James Cook", "BUF", 5000, 15, "KC@BUF 09/10/2025 08:20PM ET"),
        ("RB", "202", "Saquon Barkley", "PHI", 5000, 15, "PHI@DAL 09/10/2025 04:25PM ET"),
        ("RB", "203", "Derrick Henry", "BAL", 4000, 12, "BAL@CIN 09/10/2025 01:00PM ET"),
        ("WR", "301", "Michael Thomas", "BUF", 5000, 18, "KC@BUF 09/10/2025 08:20PM ET"),
        ("WR", "302", "Michael Thomas", "PHI", 5000, 18, "PHI@DAL 09/10/2025 04:25PM ET"),
        ("WR", "303", "AJ Brown", "PHI", 5000, 13, "PHI@DAL 09/10/2025 04:25PM ET"),
        ("TE", "401", "Travis Kelce", "KC", 4000, 12, "KC@BUF 09/10/2025 08:20PM ET"),
        ("DST", "501", "Bills", "BUF", 3000, 8, "KC@BUF 09/10/2025 08:20PM ET"),
    ]
    return pd.DataFrame(
        rows,
        columns=["Position", "ID", "Name", "TeamAbbrev", "Salary", "Projection", "Game Info"],
    )


def test_classic_lineup_can_roster_same_name_players_on_different_teams():
    optimizer = NFLClassicOptimizer()
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.jitter_seed = 42
    optimizer.load_from_dataframe(_classic_same_name_pool())
    lineups = optimizer.generate_lineups(1)
    assert len(lineups) == 1
    lineup = lineups[0]
    ids = set(lineup["ID"].astype(str))
    assert {"301", "302"} <= ids
    thomas = lineup[lineup["Name"].eq("Michael Thomas")]
    assert set(thomas["Team"]) == {"BUF", "PHI"}


def _showdown_same_name_pool() -> pd.DataFrame:
    people = [
        ("Michael Thomas", "KC", "WR", 8000, 40, "11"),
        ("Michael Thomas", "BUF", "WR", 7500, 36, "22"),
        ("James Cook", "BUF", "RB", 6000, 12, "23"),
        ("Travis Kelce", "KC", "TE", 6000, 11, "14"),
        ("Xavier Worthy", "KC", "WR", 5000, 10, "15"),
        ("Khalil Shakir", "BUF", "WR", 5000, 9, "16"),
    ]
    rows = []
    for name, team, pos, salary, projection, pid in people:
        for slot, multiplier, prefix in (("CPT", 1.5, "c"), ("FLEX", 1.0, "f")):
            rows.append({
                "Position": pos,
                "Roster Position": slot,
                "ID": prefix + pid,
                "Name": name,
                "TeamAbbrev": team,
                "Salary": int(salary * multiplier),
                "Projection": projection,
                "Game Info": "KC@BUF 09/10/2025 08:20PM ET",
            })
    return pd.DataFrame(rows)


def test_showdown_rosters_both_same_name_players_but_not_captain_and_flex():
    optimizer = NFLShowdownOptimizer()
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.jitter_seed = 42
    optimizer.game_script_correlation = False
    optimizer.min_salary_cap = None
    optimizer.max_salary_cap = None
    optimizer.min_teams = None
    optimizer.max_repeating_players = None
    optimizer.load_from_dataframe(_showdown_same_name_pool())
    lineups = optimizer.generate_lineups(1)
    assert len(lineups) == 1
    lineup = lineups[0]
    ids = set(lineup["ID"].astype(str))
    thomas = lineup[lineup["Name"].eq("Michael Thomas")]
    assert set(thomas["Team"]) == {"KC", "BUF"}
    assert not ({"c11", "f11"} <= ids)
    assert not ({"c22", "f22"} <= ids)
    assert len(lineup) == 6
