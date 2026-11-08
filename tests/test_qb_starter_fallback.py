"""Confirmed starters without a weekly score keep the salary-file average."""

from __future__ import annotations

import pandas as pd

from src.optimizer import NFLShowdownOptimizer, unique_quarterbacks
from src.sleeper import index_depth_chart, match_sleeper_pool


def _depth(name_first, name_last, team, order, position="QB"):
    return {
        "player_id": f"{team}-{order}",
        "first_name": name_first,
        "last_name": name_last,
        "position": "QB",
        "team": team,
        "depth_chart_position": position,
        "depth_chart_order": order,
        "injury_status": None,
    }


def _showdown_rows(name, team, average, ids=("1", "2")):
    return [
        {
            "ID": ids[0],
            "Name": name,
            "Position": "QB",
            "TeamAbbrev": team,
            "Roster Position": "CPT",
            "RosterSlot": "CPT",
            "Salary": 15000,
            "AvgPointsPerGame": average,
            "Projection": average,
            "Injury": "",
        },
        {
            "ID": ids[1],
            "Name": name,
            "Position": "QB",
            "TeamAbbrev": team,
            "Roster Position": "FLEX",
            "RosterSlot": "FLEX",
            "Salary": 10000,
            "AvgPointsPerGame": average,
            "Projection": average,
            "Injury": "",
        },
    ]


def test_confirmed_starter_under_12_keeps_the_average_on_both_rows():
    depth = index_depth_chart({"1": _depth("Josh", "Allen", "BUF", 1)})
    pool = pd.DataFrame(_showdown_rows("Josh Allen", "BUF", 8.0))
    updated, report = match_sleeper_pool(pool, depth)
    assert list(updated["Projection"]) == [8.0, 8.0]
    assert list(updated["Salary"]) == [15000, 10000]
    assert list(updated["Roster Position"]) == ["CPT", "FLEX"]
    assert updated["Sleeper"].isna().all()
    assert set(updated["StarterFallback"]) == {True}
    assert report["projection_rows"] == 0
    assert report["projection_players"] == 0
    assert "not a projection match" in updated.loc[0, "ScoringNote"]
    kept = _without_backup_quarterbacks_host()._without_backup_quarterbacks(updated)
    assert len(kept) == 2
    labels = unique_quarterbacks(updated)
    assert float(labels.loc[0, "Projection"]) == 8.0


class _Host:
    pass


def _without_backup_quarterbacks_host():
    return NFLShowdownOptimizer.__new__(NFLShowdownOptimizer)


def test_unconfirmed_backup_is_not_promoted_by_the_salary_average():
    depth = index_depth_chart(
        {
            "1": _depth("Josh", "Allen", "BUF", 1),
            "2": _depth("Mitchell", "Trubisky", "BUF", 2),
        }
    )
    rows = _showdown_rows("Josh Allen", "BUF", 8.0, ids=("1", "2"))
    rows.extend(_showdown_rows("Mitchell Trubisky", "BUF", 17.0, ids=("3", "4")))
    updated, report = match_sleeper_pool(pd.DataFrame(rows), depth)
    backup = updated.loc[updated["Name"] == "Mitchell Trubisky"]
    assert set(backup["Projection"]) == {0.0}
    assert set(backup["Sleeper"]) == {0.0}
    assert set(backup["StarterFallback"]) == {False}
    assert report["projection_rows"] == 0
    kept = _without_backup_quarterbacks_host()._without_backup_quarterbacks(updated)
    assert set(kept["Name"]) == {"Josh Allen"}


def test_confirmed_starter_with_unscorable_stats_keeps_the_average():
    from src.sleeper import index_projections, merge_sleeper_indexes

    projections = index_projections(
        [
            {
                "team": "BUF",
                "player_id": "BUF-1",
                "player": {
                    "first_name": "Josh",
                    "last_name": "Allen",
                    "position": "QB",
                    "team": "BUF",
                    "player_id": "BUF-1",
                },
                "stats": {"pts_std": 19.0},
            }
        ]
    )
    depth = index_depth_chart({"BUF-1": _depth("Josh", "Allen", "BUF", 1)})
    updated, report = match_sleeper_pool(
        pd.DataFrame(_showdown_rows("Josh Allen", "BUF", 8.0)),
        merge_sleeper_indexes(projections, depth),
    )
    assert list(updated["Projection"]) == [8.0, 8.0]
    assert set(updated["StarterFallback"]) == {True}
    assert report["projection_rows"] == 0
    assert "pts_std" in updated.loc[0, "ScoringNote"] or "no weekly" in updated.loc[0, "ScoringNote"]
    kept = _without_backup_quarterbacks_host()._without_backup_quarterbacks(updated)
    assert len(kept) == 2


def test_real_weekly_projection_under_12_is_still_removed():
    from src.sleeper import index_projections, merge_sleeper_indexes

    projections = index_projections(
        [
            {
                "team": "BUF",
                "player_id": "BUF-1",
                "player": {
                    "first_name": "Josh",
                    "last_name": "Allen",
                    "position": "QB",
                    "team": "BUF",
                    "injury_status": None,
                    "player_id": "BUF-1",
                },
                "stats": {"pts_ppr": 9.0},
            }
        ]
    )
    depth = index_depth_chart({"BUF-1": _depth("Josh", "Allen", "BUF", 1)})
    updated, report = match_sleeper_pool(
        pd.DataFrame(_showdown_rows("Josh Allen", "BUF", 22.0)),
        merge_sleeper_indexes(projections, depth),
    )
    assert set(updated["StarterFallback"]) == {False}
    assert report["projection_rows"] == 2
    assert float(updated.loc[0, "Projection"]) < 12
    kept = _without_backup_quarterbacks_host()._without_backup_quarterbacks(updated)
    assert kept.empty
