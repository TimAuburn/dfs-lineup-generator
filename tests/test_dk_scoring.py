"""DraftKings NFL scoring contract, projection bonuses, and captain scaling.

Realized fixtures follow the rules pages fetched for this change:

- Classic: https://www.draftkings.com/help/rules/nfl/classic
- Showdown Captain Mode: https://www.draftkings.com/help/rules/nfl/showdown-captain-mode

Historical box-score examples below are formula checks against Sleeper's
actual stat lines. They are not official DraftKings contest totals.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.dk_scoring import (
    CLASSIC_RULES_URL,
    HISTORICAL_RECONCILIATION,
    PASS_YARDS_SD,
    SHOWDOWN_RULES_URL,
    expected_milestone_bonus,
    points_allowed_tier,
    score_projection,
    score_realized,
)
from src.exporter import lineups_to_dk_upload_csv
from src.optimizer import NFLShowdownOptimizer, dataframe_to_players
from src.presets import ContestFormat
from src.sleeper import apply_sleeper_index, index_projections


def _qb(**stats):
    base = {"pass_td": 0, "pass_int": 0, "rush_yd": 0, "rush_td": 0}
    base.update(stats)
    return base


def _skill(**stats):
    base = {
        "rush_yd": 0,
        "rush_td": 0,
        "rec": 0,
        "rec_yd": 0,
        "rec_td": 0,
    }
    base.update(stats)
    return base


def test_rules_pages_are_the_official_classic_and_showdown_urls():
    assert CLASSIC_RULES_URL == "https://www.draftkings.com/help/rules/nfl/classic"
    assert SHOWDOWN_RULES_URL == (
        "https://www.draftkings.com/help/rules/nfl/showdown-captain-mode"
    )
    assert HISTORICAL_RECONCILIATION == "unverified"


@pytest.mark.parametrize(
    ("yards", "bonus"),
    [(299, 0.0), (300, 3.0), (301, 3.0)],
)
def test_realized_passing_bonus_boundaries(yards, bonus):
    scored = score_realized(_qb(pass_yd=yards), position="QB", contest="classic")
    assert scored.basis == "draftkings"
    assert scored.points == pytest.approx(yards * 0.04 + bonus)
    assert scored.bonuses["pass_300"] == pytest.approx(bonus)


@pytest.mark.parametrize(
    ("yards", "bonus"),
    [(99, 0.0), (100, 3.0), (101, 3.0)],
)
def test_realized_rushing_and_receiving_bonus_boundaries(yards, bonus):
    rushed = score_realized(_skill(rush_yd=yards), position="RB", contest="classic")
    caught = score_realized(_skill(rec_yd=yards), position="WR", contest="showdown")
    assert rushed.points == pytest.approx(yards * 0.1 + bonus)
    assert caught.points == pytest.approx(yards * 0.1 + bonus)
    assert rushed.bonuses["rush_100"] == pytest.approx(bonus)
    assert caught.bonuses["rec_100"] == pytest.approx(bonus)


def test_projection_milestone_is_an_expected_value_not_a_flat_three():
    """A projected mean on the threshold is not a realized 300-yard game."""
    realized = score_realized(_qb(pass_yd=300), position="QB", contest="classic")
    projected = score_projection(_qb(pass_yd=300), position="QB", contest="classic")
    assert realized.bonuses["pass_300"] == pytest.approx(3.0)
    assert 0.0 < projected.bonuses["pass_300"] < 3.0
    assert projected.points == pytest.approx(300 * 0.04 + projected.bonuses["pass_300"])
    assert projected.points != pytest.approx(realized.points)

    below = score_projection(_qb(pass_yd=299), position="QB", contest="classic")
    above = score_projection(_qb(pass_yd=301), position="QB", contest="classic")
    assert below.bonuses["pass_300"] < projected.bonuses["pass_300"] < above.bonuses["pass_300"]
    assert below.bonuses["pass_300"] != pytest.approx(0.0)
    assert above.bonuses["pass_300"] != pytest.approx(3.0)
    # Far from the threshold the expectation approaches the realized 0 and 3.
    assert expected_milestone_bonus(0, 300, PASS_YARDS_SD) == pytest.approx(0.0, abs=0.05)
    assert expected_milestone_bonus(500, 300, PASS_YARDS_SD) == pytest.approx(3.0, abs=0.05)


def test_projection_rushing_and_receiving_bonuses_stay_below_the_flat_rule():
    for position, key, stat in (
        ("RB", "rush_100", "rush_yd"),
        ("WR", "rec_100", "rec_yd"),
        ("TE", "rec_100", "rec_yd"),
    ):
        line = _skill(**{stat: 100})
        projected = score_projection(line, position=position, contest="classic")
        realized = score_realized(line, position=position, contest="classic")
        assert realized.bonuses[key] == pytest.approx(3.0)
        assert 0.0 < projected.bonuses[key] < 3.0
        assert projected.points < realized.points


@pytest.mark.parametrize(
    ("allowed", "points"),
    [
        (0, 10.0),
        (1, 7.0),
        (6, 7.0),
        (7, 4.0),
        (13, 4.0),
        (14, 1.0),
        (20, 1.0),
        (21, 0.0),
        (27, 0.0),
        (28, -1.0),
        (34, -1.0),
        (35, -4.0),
        (50, -4.0),
    ],
)
def test_dst_points_allowed_tier_boundaries(allowed, points):
    assert points_allowed_tier(allowed) == pytest.approx(points)
    scored = score_realized(
        {"sack": 0, "int": 0, "fum_rec": 0, "pts_allow": allowed},
        position="DST",
        contest="classic",
    )
    assert scored.points == pytest.approx(points)
    assert scored.bonuses["points_allowed"] == pytest.approx(points)


def test_dst_projection_does_not_score_the_average_as_a_realized_tier():
    """Sleeper's single tier flag matches the mean's bucket. That is not a distribution."""
    line = {
        "sack": 2,
        "int": 1,
        "fum_rec": 0,
        "pts_allow": 15,  # 14-20 tier is +1 if it were a realized total
        "pts_allow_14_20": 1,
        "yds_allow": 310,
        "yds_allow_300_349": 1,
        "def_td": 0.16,
        "pass_int_td": 0.11,
        "def_fum_td": 0.05,
        "ff": 0.7,
    }
    scored = score_projection(line, position="DEF", contest="showdown")
    flat = score_realized(
        {"sack": 2, "int": 1, "fum_rec": 0, "pts_allow": 15, "def_td": 0.16},
        position="DST",
        contest="showdown",
    )
    assert scored.basis == "draftkings"
    assert scored.bonuses["points_allowed"] != pytest.approx(1.0)
    assert "points_allowed_distribution" in scored.unavailable
    assert "yds_allow" in scored.unavailable
    # def_td already includes the component touchdowns. Do not add them again.
    assert scored.points == pytest.approx(
        2 + 2 + 0.16 * 6 + scored.bonuses["points_allowed"]
    )
    assert scored.points != pytest.approx(flat.points)
    assert "distribution" in scored.warning.lower()


def test_dst_uses_a_real_bucket_distribution_when_one_is_present():
    scored = score_projection(
        {
            "sack": 0,
            "int": 0,
            "fum_rec": 0,
            "pts_allow": 20,
            "pts_allow_14_20": 0.25,
            "pts_allow_21_27": 0.75,
        },
        position="DST",
        contest="classic",
    )
    assert "points_allowed_distribution" not in scored.unavailable
    assert scored.bonuses["points_allowed"] == pytest.approx(0.25 * 1.0 + 0.75 * 0.0)


def test_showdown_kicker_distance_miss_and_pat():
    made = score_realized(
        {
            "fgm_0_19": 1,  # 0-39 is +3, including a short kick
            "fgm_30_39": 1,  # 39-yard band
            "fgm_40_49": 1,  # 40-49 is +4
            "fgm_50p": 1,  # 50+ is +5
            "fgm": 4,
            "xpm": 2,
            "xpmiss": 1,
            "fgmiss_40_49": 1,
            "rush_td": 1,  # kickers are not eligible for non-kick stats
        },
        position="K",
        contest="showdown",
    )
    assert made.points == pytest.approx(3 + 3 + 4 + 5 + 2)
    assert made.bonuses["missed_kicks"] == pytest.approx(0.0)
    missed_only = score_realized(
        {"fgm": 0, "xpm": 0, "xpmiss": 1, "fgmiss_50p": 2},
        position="K",
        contest="showdown",
    )
    assert missed_only.points == pytest.approx(0.0)
    classic = score_realized(
        {"fgm_40_49": 1, "fgm": 1, "xpm": 1},
        position="K",
        contest="classic",
    )
    assert classic.basis == "unavailable"
    assert classic.points is None


def test_kicker_projection_refuses_a_distance_split_that_does_not_add_up():
    scored = score_projection(
        {
            "fgm": 2.0,
            "fgm_20_29": 0.31,
            "fgm_30_39": 0.46,
            "fgm_40_49": 0.41,
            "fgm_50p": 0.41,
            "xpm": 2.71,
            "xpmiss": 0.13,
            "pts_ppr": 8.57,
        },
        position="K",
        contest="showdown",
    )
    assert scored.basis == "unavailable"
    assert scored.points is None
    assert "fgm" in scored.missing or "field goal" in scored.warning.lower()


def test_reconciled_kicker_projection_scores_makes_and_ignores_misses():
    scored = score_projection(
        {
            "fgm": 2,
            "fgm_30_39": 1,
            "fgm_50p": 1,
            "fgm_50_59": 1,  # overlaps fgm_50p on Sleeper actuals; do not double count
            "xpm": 1,
            "xpmiss": 0.4,
            "fgmiss_40_49": 0.2,
        },
        position="K",
        contest="showdown",
    )
    assert scored.basis == "draftkings"
    assert scored.points == pytest.approx(3 + 5 + 1)
    assert scored.bonuses["missed_kicks"] == pytest.approx(0.0)


def test_partial_components_do_not_become_a_total_and_std_is_not_ppr():
    records = [
        {
            "team": "BUF",
            "player": {
                "first_name": "Josh",
                "last_name": "Allen",
                "position": "QB",
                "team": "BUF",
            },
            "stats": {"pass_yd": 250, "pts_ppr": 17.5, "pts_std": 16.5},
        },
        {
            "team": "DAL",
            "player": {
                "first_name": "Dak",
                "last_name": "Prescott",
                "position": "QB",
                "team": "DAL",
            },
            "stats": {"pts_std": 19.7},
        },
    ]
    index = index_projections(records)
    assert [row["name"] for row in index["projections"]] == ["josh allen", "dak prescott"]

    pool = pd.DataFrame(
        [
            {
                "ID": "1",
                "Name": "Josh Allen",
                "Position": "QB",
                "TeamAbbrev": "BUF",
                "AvgPointsPerGame": 9.0,
                "Projection": 9.0,
            },
            {
                "ID": "2",
                "Name": "Dak Prescott",
                "Position": "QB",
                "TeamAbbrev": "DAL",
                "AvgPointsPerGame": 8.0,
                "Projection": 8.0,
            },
        ]
    )
    updated, matched = apply_sleeper_index(pool, index)
    by_id = updated.set_index("ID")
    assert matched == 1
    assert by_id.loc["1", "Projection"] == pytest.approx(17.5)
    assert by_id.loc["1", "Projection"] != pytest.approx(250 * 0.04)
    assert by_id.loc["1", "Scoring"] == "PPR fallback"
    assert "pass_td" in by_id.loc["1", "ScoringNote"]
    # No component stats and no PPR: keep the salary-file average and say why.
    assert by_id.loc["2", "Projection"] == pytest.approx(8.0)
    assert by_id.loc["2", "Scoring"] == "Unavailable"
    assert "pts_std" in by_id.loc["2", "ScoringNote"]
    assert pd.isna(by_id.loc["2", "Sleeper"])


def test_component_projection_replaces_ppr_and_does_not_scale_the_captain_row():
    stats = _qb(pass_yd=250, pass_td=2, pass_int=0.5, pts_ppr=99.0, pts_std=90.0)
    record = {
        "team": "BUF",
        "player": {
            "first_name": "Josh",
            "last_name": "Allen",
            "position": "QB",
            "team": "BUF",
        },
        "stats": stats,
    }
    index = index_projections([record])
    scored = score_projection(stats, position="QB", contest="showdown")
    assert index["projections"][0]["name"] == "josh allen"
    assert scored.points != pytest.approx(99.0)
    pool = pd.DataFrame(
        [
            {
                "ID": "10",
                "Name": "Josh Allen",
                "Position": "QB",
                "Roster Position": "CPT",
                "TeamAbbrev": "BUF",
                "Salary": 15000,
                "Projection": 1.0,
                "AvgPointsPerGame": 1.0,
            },
            {
                "ID": "11",
                "Name": "Josh Allen",
                "Position": "QB",
                "Roster Position": "FLEX",
                "TeamAbbrev": "BUF",
                "Salary": 10000,
                "Projection": 1.0,
                "AvgPointsPerGame": 1.0,
            },
        ]
    )
    updated, _ = apply_sleeper_index(pool, index)
    assert updated.loc[0, "Projection"] == pytest.approx(scored.points)
    assert updated.loc[0, "PPR"] == pytest.approx(99.0)
    assert updated.loc[0, "PPR"] != pytest.approx(updated.loc[0, "Projection"])
    assert updated.loc[1, "Projection"] == pytest.approx(scored.points)
    assert updated.loc[0, "Scoring"] == "DraftKings"
    players = dataframe_to_players(updated, contest_format=ContestFormat.SHOWDOWN)
    by_id = {player.id: player for player in players}
    assert by_id["10"].fppg == pytest.approx(scored.points * 1.5)
    assert by_id["11"].fppg == pytest.approx(scored.points)
    assert NFLShowdownOptimizer._baseline_projection(by_id["10"]) == pytest.approx(
        scored.points
    )


def _showdown_pair(pos, name, flex_id, team, flex_salary, fppg):
    game = "GB@CHI 09/10/2025 08:20PM ET"
    return [
        {
            "Position": pos,
            "Name + ID": f"{name} ({flex_id + 1})",
            "Name": name,
            "ID": str(flex_id + 1),
            "Roster Position": "CPT",
            "Salary": int(flex_salary * 1.5),
            "Game Info": game,
            "TeamAbbrev": team,
            "AvgPointsPerGame": fppg,
            "Projection": fppg,
        },
        {
            "Position": pos,
            "Name + ID": f"{name} ({flex_id})",
            "Name": name,
            "ID": str(flex_id),
            "Roster Position": "FLEX",
            "Salary": flex_salary,
            "Game Info": game,
            "TeamAbbrev": team,
            "AvgPointsPerGame": fppg,
            "Projection": fppg,
        },
    ]


def test_optimizer_and_export_apply_captain_multiplier_once():
    rows = []
    rows += _showdown_pair("QB", "Jordan Love", 100, "GB", 10000, 40)
    rows += _showdown_pair("WR", "Romeo Doubs", 110, "GB", 8000, 15)
    rows += _showdown_pair("RB", "Josh Jacobs", 120, "GB", 7000, 12)
    rows += _showdown_pair("QB", "Caleb Williams", 200, "CHI", 9000, 18)
    rows += _showdown_pair("WR", "Rome Odunze", 210, "CHI", 6000, 11)
    rows += _showdown_pair("RB", "D'Andre Swift", 220, "CHI", 5000, 9)
    pool = pd.DataFrame(rows)
    optimizer = NFLShowdownOptimizer()
    optimizer.game_script_correlation = False
    optimizer.min_salary_cap = 0
    optimizer.max_salary_cap = 50_000
    optimizer.load_from_dataframe(pool)
    lineup = optimizer.generate_lineups(1)[0]
    bases = pool.groupby("Name")["Projection"].first()
    captain = lineup.loc[lineup["Slot"] == "CPT"].iloc[0]
    captain_base = float(bases[captain["Name"]])
    assert captain["Projection"] == pytest.approx(captain_base * 1.5)
    assert captain["Projection"] != pytest.approx(captain_base * 1.5 * 1.5)
    for _, flex in lineup.loc[lineup["Slot"] == "FLEX"].iterrows():
        assert flex["Projection"] == pytest.approx(float(bases[flex["Name"]]))
    assert lineup.attrs["projection"] == pytest.approx(float(lineup["Projection"].sum()))
    exported = lineups_to_dk_upload_csv([lineup], contest_format=ContestFormat.SHOWDOWN)
    assert exported.splitlines()[0] == "CPT,FLEX,FLEX,FLEX,FLEX,FLEX"
    assert str(captain["ID"]) in exported.splitlines()[1].split(",")


# Real Sleeper stat lines. These formula totals are not official DraftKings
# contest results. Provenance:
# https://api.sleeper.app/v1/stats/nfl/regular/2025/3  player 12497 Tory Horton
# https://api.sleeper.app/v1/stats/nfl/regular/2025/12 player 12540 Chimere Dike
# https://api.sleeper.app/v1/stats/nfl/regular/2025/1  team CHI
# https://api.sleeper.app/projections/nfl/2026/9?season_type=regular  NO and DAL


def test_realized_return_td_uses_st_td_once_for_horton_and_dike():
    horton = score_realized(
        {"rec": 3, "rec_yd": 32, "rec_td": 1, "st_td": 1},
        position="WR",
        contest="classic",
    )
    # 3 receptions + 3.2 receiving yards + 6 receiving TD + 6 return TD.
    assert horton.basis == "draftkings"
    assert horton.points == pytest.approx(18.2)
    dike = score_realized(
        {"rec": 5, "rec_yd": 44, "rec_td": 1, "st_td": 1},
        position="WR",
        contest="showdown",
    )
    assert dike.points == pytest.approx(21.4)


def test_return_td_aliases_are_not_summed_and_inconsistent_totals_are_refused():
    aliased = score_realized(
        {"rec": 0, "rec_yd": 0, "rec_td": 0, "st_td": 1, "kr_td": 1, "pr_td": 1},
        position="WR",
        contest="classic",
    )
    assert aliased.points == pytest.approx(6.0)
    partitioned = score_realized(
        {"rec": 0, "rec_yd": 0, "rec_td": 0, "st_td": 2, "kr_td": 1, "pr_td": 1},
        position="WR",
        contest="classic",
    )
    assert partitioned.points == pytest.approx(12.0)
    inconsistent = score_realized(
        {"rec": 0, "rec_yd": 0, "rec_td": 0, "st_td": 1, "kr_td": 1, "pr_td": 0.4},
        position="WR",
        contest="classic",
    )
    assert inconsistent.basis == "unavailable"
    assert inconsistent.points is None


def test_dst_does_not_drop_special_teams_or_triple_count_aliases():
    """2026 week-9 projection fields. NO repeats one ST TD three ways. DAL's def_td is only the defensive TD."""
    counting = {
        "sack": 2.98,
        "int": 0.76,
        "fum_rec": 0.61,
        "pts_allow": 19.75,
        "blk_kick": 0.08,
    }
    once = score_projection({**counting, "st_td": 0.08}, position="DST", contest="classic")
    aliased = score_projection(
        {**counting, "st_td": 0.08, "def_kr_td": 0.08, "def_pr_td": 0.08},
        position="DST",
        contest="classic",
    )
    assert once.basis == "draftkings"
    assert aliased.points == pytest.approx(once.points)

    defense_only = {
        "sack": 2.48,
        "int": 0.71,
        "fum_rec": 0.53,
        "pts_allow": 23.5,
        "blk_kick": 0.06,
        "def_td": 0.06,
        "pass_int_td": 0.06,
    }
    with_st = score_projection(
        {**defense_only, "st_td": 0.06, "def_kr_td": 0.06, "def_pr_td": 0.06},
        position="DST",
        contest="showdown",
    )
    without_st = score_projection(defense_only, position="DST", contest="showdown")
    assert with_st.points == pytest.approx(without_st.points + 0.36)


def test_dst_ignores_team_td_total_and_refuses_a_def_td_that_disagrees_with_parts():
    chi = score_realized(
        {"sack": 3, "int": 1, "def_td": 1, "pts_allow": 27, "td": 3},
        position="DST",
        contest="classic",
    )
    # Sack 3 + interception 2 + one defensive TD 6 + 21-27 points allowed 0.
    # The team ``td`` of 3 includes offense and is not added.
    assert chi.points == pytest.approx(11.0)
    broken = score_projection(
        {
            "sack": 1,
            "int": 1,
            "fum_rec": 0,
            "pts_allow": 20,
            "def_td": 0.19,
            "pass_int_td": 0.10,
            "def_fum_td": 0.10,
        },
        position="DST",
        contest="classic",
    )
    assert broken.basis == "unavailable"
    assert broken.points is None


def test_points_allowed_weights_must_be_finite_and_between_zero_and_one():
    scored = score_projection(
        {
            "sack": 0,
            "int": 0,
            "fum_rec": 0,
            "pts_allow": 21,
            "pts_allow_0": -1,
            "pts_allow_35p": 2,
        },
        position="DST",
        contest="classic",
    )
    assert scored.basis == "draftkings"
    assert -4.0 <= scored.bonuses["points_allowed"] <= 10.0
    assert scored.bonuses["points_allowed"] != pytest.approx(-18.0)
    nonsense = score_projection(
        {
            "sack": 0,
            "int": 0,
            "fum_rec": 0,
            "pts_allow": 21,
            "pts_allow_0": float("nan"),
            "pts_allow_14_20": 1,
        },
        position="DST",
        contest="classic",
    )
    assert -4.0 <= nonsense.bonuses["points_allowed"] <= 10.0
