"""View-lineup height and the display-only Sleeper Proj column."""

from __future__ import annotations

import pandas as pd
import pytest

from src.dk_scoring import score_projection
from src.lineup_view import (
    dataframe_height,
    lineup_detail_frame,
    sleeper_proj_caption,
    sleeper_proj_for_lineup,
)
from src.sleeper import index_projections, match_sleeper_pool


def _record(first, last, team, position, stats, player_id="1"):
    return {
        "team": team,
        "player_id": player_id,
        "player": {
            "first_name": first,
            "last_name": last,
            "position": position,
            "team": team,
            "player_id": player_id,
        },
        "stats": stats,
    }


def _pool_row(**overrides):
    row = {
        "ID": "1",
        "Name": "Josh Allen",
        "Position": "QB",
        "TeamAbbrev": "BUF",
        "Salary": 7400,
        "Roster Position": "QB",
        "AvgPointsPerGame": 9.0,
        "Projection": 9.0,
        "Injury": "",
        "Game Info": "KC@BUF 09/07/2025 01:00PM ET",
    }
    row.update(overrides)
    return row


def _lineup_row(**overrides):
    row = {
        "Slot": "QB",
        "ID": "1",
        "Name": "Josh Allen",
        "Position": "QB",
        "Team": "BUF",
        "Salary": 7400,
        "Projection": 9.0,
    }
    row.update(overrides)
    return row


def test_dataframe_height_fits_classic_and_showdown_rosters():
    assert dataframe_height(9) == 35 * 10 + 3
    assert dataframe_height(6) == 35 * 7 + 3
    assert dataframe_height(0) == 38
    assert dataframe_height(-4) == dataframe_height(0)
    assert dataframe_height(9) > 320
    assert dataframe_height(6) < dataframe_height(9)
    # 320px is one row short of a 9-player roster: header 35, border 2, slack 1.
    visible_rows = (320 - 35 - 2 - 1) // 35
    assert visible_rows == 8


def test_caption_only_when_sleeper_was_not_applied():
    assert sleeper_proj_caption(None).startswith("Sleeper projections are unavailable")
    assert sleeper_proj_caption({}).startswith("Sleeper projections are unavailable")
    assert "blank" in sleeper_proj_caption({"applied": False, "error": "fetch_failed:URLError"})
    assert sleeper_proj_caption({"applied": True, "week": 1}) == ""


def test_component_score_is_shown_and_pts_ppr_is_not_substituted():
    stats = {"pass_yd": 250, "pass_td": 2, "pass_int": 1, "pts_ppr": 99.0}
    expected = score_projection(stats, position="QB", contest="classic")
    assert expected.ok
    assert expected.points != 99.0
    index = index_projections([_record("Josh", "Allen", "BUF", "QB", stats, player_id="4984")])
    pool = pd.DataFrame([_pool_row(ID="4984")])
    original = float(pool.loc[0, "Projection"])
    updated, report = match_sleeper_pool(pool, index)
    lineup = pd.DataFrame([_lineup_row(ID="4984")])
    before = lineup["Projection"].copy()

    shown = sleeper_proj_for_lineup(lineup, updated)

    assert report["projection_rows"] == 1
    assert shown.iloc[0] == pytest.approx(expected.points)
    assert shown.iloc[0] != pytest.approx(99.0)
    assert updated.loc[0, "Scoring"] == "DraftKings"
    assert float(updated.loc[0, "Projection"]) == pytest.approx(expected.points)
    assert lineup["Projection"].equals(before)
    assert float(pool.loc[0, "Projection"]) == original


def test_ppr_fallback_is_the_sleeper_pts_ppr_field():
    index = index_projections(
        [_record("James", "Cook", "BUF", "RB", {"pts_ppr": 14.25}, player_id="3")]
    )
    pool = pd.DataFrame(
        [_pool_row(ID="3", Name="James Cook", Position="RB", Projection=4.0, AvgPointsPerGame=4.0)]
    )
    updated, _report = match_sleeper_pool(pool, index)
    lineup = pd.DataFrame(
        [_lineup_row(Slot="RB", ID="3", Name="James Cook", Position="RB", Projection=4.0)]
    )

    shown = sleeper_proj_for_lineup(lineup, updated)

    assert updated.loc[0, "Scoring"] == "PPR fallback"
    assert shown.iloc[0] == pytest.approx(14.25)


def test_dst_matches_on_team_and_ignores_the_nickname():
    index = index_projections(
        [_record("Buffalo", "Bills", "BUF", "DEF", {"pts_ppr": 7.5}, player_id="BUF")]
    )
    pool = pd.DataFrame(
        [
            _pool_row(
                ID="5001",
                Name="Bills",
                Position="DST",
                TeamAbbrev="BUF",
                Projection=3.0,
                AvgPointsPerGame=3.0,
            )
        ]
    )
    updated, report = match_sleeper_pool(pool, index)
    lineup = pd.DataFrame(
        [
            _lineup_row(
                Slot="DST",
                ID="5001",
                Name="Bills",
                Position="DST",
                Team="BUF",
                Projection=3.0,
            )
        ]
    )

    shown = sleeper_proj_for_lineup(lineup, updated)

    assert report["projection_rows"] == 1
    assert shown.iloc[0] == pytest.approx(7.5)


def test_missing_unmatched_and_ambiguous_stay_blank_and_a_real_zero_stays_zero():
    index = index_projections(
        [
            _record("Josh", "Allen", "BUF", "QB", {"pts_ppr": 0.0}, player_id="1"),
            _record("Josh", "Johnson", "WAS", "QB", {"pts_ppr": 3.0}, player_id="a"),
            _record("Josh", "Johnson", "WAS", "QB", {"pts_ppr": 18.0}, player_id="b"),
        ]
    )
    pool = pd.DataFrame(
        [
            _pool_row(ID="1", Name="Josh Allen", Projection=9.0, AvgPointsPerGame=9.0),
            _pool_row(
                ID="2",
                Name="Josh Johnson",
                TeamAbbrev="WAS",
                Projection=5.0,
                AvgPointsPerGame=5.0,
            ),
            _pool_row(
                ID="3",
                Name="Nobody Matched",
                TeamAbbrev="KC",
                Projection=8.0,
                AvgPointsPerGame=8.0,
            ),
        ]
    )
    updated, report = match_sleeper_pool(pool, index)
    lineup = pd.DataFrame(
        [
            _lineup_row(ID="1", Name="Josh Allen", Team="BUF"),
            _lineup_row(Slot="QB", ID="2", Name="Josh Johnson", Team="WAS", Projection=5.0),
            _lineup_row(Slot="WR", ID="3", Name="Nobody Matched", Position="WR", Team="KC", Projection=8.0),
            _lineup_row(Slot="RB", ID="9", Name="Absent Player", Position="RB", Team="DAL", Projection=6.0),
        ]
    )

    shown = sleeper_proj_for_lineup(lineup, updated)

    assert report["projection_rows"] == 1
    assert shown.iloc[0] == pytest.approx(0.0)
    assert pd.isna(shown.iloc[1])
    assert pd.isna(shown.iloc[2])
    assert pd.isna(shown.iloc[3])
    assert int(shown.fillna(-1).eq(0).sum()) == 1
    assert float(updated.loc[updated["ID"].astype(str).eq("2"), "Projection"].iloc[0]) == 5.0
    assert float(updated.loc[updated["ID"].astype(str).eq("3"), "Projection"].iloc[0]) == 8.0


def test_backup_zero_is_not_shown_as_a_projection():
    pool = pd.DataFrame([_pool_row(ID="9", Name="Backup QB")])
    pool["Sleeper"] = 0.0
    pool["MatchStatus"] = "backup_zero"
    lineup = pd.DataFrame([_lineup_row(ID="9", Name="Backup QB")])

    shown = sleeper_proj_for_lineup(lineup, pool)

    assert pd.isna(shown.iloc[0])


def test_same_player_without_an_id_still_matches_and_captain_stays_at_1x():
    index = index_projections(
        [_record("Patrick", "Mahomes", "KC", "QB", {"pts_ppr": 22.5}, player_id="4046")]
    )
    pool = pd.DataFrame(
        [
            _pool_row(ID="91001", Name="Patrick Mahomes", TeamAbbrev="KC", Projection=22.5),
            _pool_row(ID="91002", Name="Patrick Mahomes", TeamAbbrev="KC", Projection=22.5),
        ]
    )
    updated, _report = match_sleeper_pool(pool, index)
    lineup = pd.DataFrame(
        [
            _lineup_row(Slot="CPT", ID="91001", Name="Patrick Mahomes", Team="KC", Projection=33.75),
        ]
    )
    lineup_without_id = lineup.drop(columns=["ID"])

    by_id = sleeper_proj_for_lineup(lineup, updated)
    by_name = sleeper_proj_for_lineup(lineup_without_id, updated)

    assert by_id.iloc[0] == pytest.approx(22.5)
    assert by_name.iloc[0] == pytest.approx(22.5)
    assert by_id.iloc[0] != pytest.approx(33.75)


def test_detail_frame_drops_salary_remaining_and_keeps_projection():
    pool = pd.DataFrame([_pool_row(ID="1")])
    pool["Sleeper"] = 18.0
    pool["MatchStatus"] = "projection"
    pool["Delta"] = 1.25
    lineup = pd.DataFrame([_lineup_row()])
    lineup.attrs["salary_remaining"] = 0

    view = lineup_detail_frame(lineup, pool)

    assert list(view.columns) == [
        "Slot",
        "Name",
        "Team",
        "Salary",
        "Projection",
        "Delta",
        "Sleeper Proj",
    ]
    assert "Salary Remaining ($)" not in view.columns
    assert view.loc[0, "Projection"] == pytest.approx(9.0)
    assert view.loc[0, "Sleeper Proj"] == pytest.approx(18.0)
    assert view.loc[0, "Delta"] == pytest.approx(1.25)


def test_unavailable_pool_leaves_the_column_blank():
    lineup = pd.DataFrame([_lineup_row()])
    shown = sleeper_proj_for_lineup(lineup, None)
    assert pd.isna(shown.iloc[0])
    view = lineup_detail_frame(lineup, None)
    assert pd.isna(view.loc[0, "Sleeper Proj"])
    assert "Delta" not in view.columns
