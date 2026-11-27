"""Showdown team split can allow more than one construction.

A lineup matches any selected split. The objective is unchanged, so lineups
are not spread across the splits. Any, an empty selection, and a saved
single pair still mean what they meant before.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.optimizer import NFLShowdownOptimizer
from src.presets import ContestFormat, ContestPreset
from src.showdown_rules import (
    RULE_MIN_PER_TEAM,
    RULE_TEAM_SPLIT,
    TEAM_SPLIT_OPTIONS,
    AllowedTeamSplitsRule,
    canonical_team_splits,
    prepare_team_split_widget,
    reconcile_team_split_selection,
    stored_team_split,
    team_split_cause,
    team_split_needs_side,
    team_split_setting_label,
    team_split_supply_cause,
)

MOCK_SHOWDOWN = Path(__file__).parent / "mock_dk_showdown.csv"
# Solver order for the open skill pool with only 3-3, captured before this change.
SHOWDOWN_33_IDS = ["131", "100", "110", "200", "240", "250"]


def _row(pos, name, pid, slot, team, salary, fppg) -> dict:
    return {
        "Position": pos,
        "Name + ID": f"{name} ({pid})",
        "Name": name,
        "ID": str(pid),
        "Roster Position": slot,
        "Salary": salary,
        "Game Info": "HOU@JAX 11/15/2026 01:00PM ET",
        "TeamAbbrev": team,
        "AvgPointsPerGame": fppg,
        "Projection": fppg,
    }


def _pair(pos, name, flex_id, team, flex_salary, fppg) -> list[dict]:
    return [
        _row(pos, name, flex_id + 1, "CPT", team, int(flex_salary * 1.5), fppg),
        _row(pos, name, flex_id, "FLEX", team, flex_salary, fppg),
    ]


def _skill_pool(per_team: int = 6) -> pd.DataFrame:
    """HOU projections are higher, so an open build is 5-1 HOU.

    Every HOU player is 30 and every JAX player is 8, matching the pool the
    existing split tests solve. A 3-3 from that pool is the roster below.
    """
    rows = []
    for index, name in enumerate(list("ABCDEF")[:per_team]):
        rows += _pair("WR", f"HOU {name}", 100 + index * 10, "HOU", 5000, 30)
        rows += _pair("WR", f"JAX {name}", 200 + index * 10, "JAX", 4000, 8)
    return pd.DataFrame(rows)


def _open_rules(**extra):
    rules = dict(
        min_salary_cap=20_000,
        max_salary_cap=50_000,
        min_teams=2,
        game_script_correlation=False,
        max_kickers=None,
        max_k_dst_combined=None,
    )
    rules.update(extra)
    return rules


def _counts(lineup: pd.DataFrame) -> dict[str, int]:
    return {str(team): int(count) for team, count in lineup["Team"].value_counts().items()}


def _solve(split, team=None, n=1, frame=None, **extra):
    optimizer = NFLShowdownOptimizer()
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.load_from_dataframe(frame if frame is not None else _skill_pool())
    optimizer.configure_showdown_rules(
        **_open_rules(team_split=split, team_split_team=team, **extra)
    )
    optimizer.max_repeating_players = None
    optimizer.set_lineup_count(n)
    return optimizer, optimizer.generate_lineups()


def test_saved_single_pair_and_labels_normalize_without_a_widget():
    assert stored_team_split(None) is None
    assert stored_team_split("Any") is None
    assert stored_team_split(["Any"]) is None
    assert stored_team_split([]) is None
    assert stored_team_split(["Any", "4-2"]) is None
    assert stored_team_split((4, 2)) == (4, 2)
    assert stored_team_split([4, 2]) == (4, 2)
    assert stored_team_split("4-2") == (4, 2)
    assert stored_team_split([(4, 2)]) == (4, 2)
    assert stored_team_split(["5-1", "4-2"]) == ((4, 2), (5, 1))
    assert stored_team_split(((5, 1), (4, 2), (4, 2))) == ((4, 2), (5, 1))
    assert canonical_team_splits(("3-3", "5-1")) == ((3, 3), (5, 1))
    assert prepare_team_split_widget("4-2") == ["4-2"]
    assert prepare_team_split_widget("Any") == ["Any"]
    assert prepare_team_split_widget([]) == []
    assert prepare_team_split_widget(["5-1", "4-2", "4-2"]) == ["4-2", "5-1"]
    assert reconcile_team_split_selection(["Any", "4-2"], ["4-2"]) == ["Any"]
    assert reconcile_team_split_selection(["Any", "4-2"], ["Any"]) == ["4-2"]
    assert reconcile_team_split_selection([], ["4-2"]) == []
    assert team_split_needs_side((3, 3)) is False
    assert team_split_needs_side(((3, 3), (4, 2))) is True
    assert team_split_setting_label((4, 2), "BUF") == "4-2, BUF"
    assert team_split_setting_label((4, 2), None) == "4-2, either team"
    assert team_split_setting_label(((4, 2), (5, 1)), "BUF") == "4-2 or 5-1, BUF heavy"
    assert team_split_setting_label(((4, 2), (5, 1)), None) == "4-2 or 5-1, either team"
    assert team_split_setting_label((3, 3), "BUF") == "3-3"
    assert team_split_setting_label(None, "BUF") is None
    assert TEAM_SPLIT_OPTIONS == ("Any", "3-3", "4-2", "5-1")


def test_old_single_tuple_still_loads_and_33_matches_the_previous_roster():
    optimizer = NFLShowdownOptimizer()
    optimizer.configure_showdown_rules(**_open_rules(team_split=(4, 2), team_split_team="HOU"))
    assert optimizer.team_split == (4, 2)
    assert optimizer.team_split_team == "HOU"

    single, single_lineups = _solve((3, 3))
    labeled, labeled_lineups = _solve("3-3")
    listed, listed_lineups = _solve([(3, 3)])
    assert single.team_split == (3, 3)
    assert labeled.team_split == (3, 3)
    assert listed.team_split == (3, 3)
    ids = [str(player_id) for player_id in single_lineups[0]["ID"]]
    assert ids == SHOWDOWN_33_IDS
    assert [str(player_id) for player_id in labeled_lineups[0]["ID"]] == ids
    assert [str(player_id) for player_id in listed_lineups[0]["ID"]] == ids
    assert _counts(single_lineups[0]) == {"HOU": 3, "JAX": 3}


def test_any_and_empty_leave_the_split_open():
    open_split, open_lineups = _solve(None)
    assert open_split.team_split is None
    assert _counts(open_lineups[0]) == {"HOU": 5, "JAX": 1}
    for value in ("Any", [], ["Any"], ["Any", "5-1"]):
        optimizer, lineups = _solve(value)
        assert optimizer.team_split is None
        assert [str(player_id) for player_id in lineups[0]["ID"]] == [
            str(player_id) for player_id in open_lineups[0]["ID"]
        ]


def test_selected_splits_are_an_or_and_are_not_spread_evenly():
    """4-2 or 5-1 never builds 3-3, and the better split takes the lineups."""
    optimizer, lineups = _solve(((4, 2), (5, 1)), n=4)
    assert optimizer.team_split == ((4, 2), (5, 1))
    assert len(lineups) == 4
    shapes = []
    for lineup in lineups:
        counts = _counts(lineup)
        assert 3 not in counts.values()
        assert sorted(counts.values()) in ([1, 5], [2, 4])
        shapes.append(tuple(sorted(counts.values())))
    assert shapes == [(1, 5), (1, 5), (1, 5), (1, 5)]
    assert all(_counts(lineup)["HOU"] == 5 for lineup in lineups)


def test_heavy_team_applies_to_every_uneven_split():
    """The named team takes the larger side of whichever uneven split is used.

    HOU's players score more, so HOU heavy is 5-1 and JAX heavy is 4-2:
    two HOU players beat a fifth JAX player. Neither build is 3-3.
    """
    _optimizer, hou = _solve(["4-2", "5-1"], team="HOU", n=3)
    _optimizer, jax = _solve(["4-2", "5-1"], team="JAX", n=3)
    assert [_counts(lineup) for lineup in hou] == [{"HOU": 5, "JAX": 1}] * 3
    assert [_counts(lineup) for lineup in jax] == [{"JAX": 4, "HOU": 2}] * 3


def test_impossible_split_is_named_instead_of_zero_lineups():
    short = _skill_pool(per_team=3)
    cause = team_split_supply_cause(short, ["4-2", "5-1"], None)
    assert cause is not None
    assert RULE_TEAM_SPLIT in cause
    assert "4-2 or 5-1" in cause
    assert "HOU has 3" in cause and "JAX has 3" in cause
    assert "was not changed" in cause

    optimizer = NFLShowdownOptimizer()
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.load_from_dataframe(short)
    optimizer.configure_showdown_rules(**_open_rules(team_split=["4-2", "5-1"]))
    optimizer.set_lineup_count(2)
    with pytest.raises(RuntimeError) as raised:
        optimizer.generate_lineups()
    message = str(raised.value)
    assert "built 0" in message
    assert RULE_TEAM_SPLIT in message
    assert "4-2 or 5-1" in message
    assert "CBC" not in message
    assert optimizer.team_split == ((4, 2), (5, 1))
    assert optimizer.last_shortfall is not None
    assert optimizer.last_shortfall.built == 0

    minimum = team_split_cause(True, 3, ((4, 2), (5, 1)))
    assert minimum is not None
    assert RULE_MIN_PER_TEAM in minimum
    assert "4-2 or 5-1" in minimum
    assert "put 2 and 1" in minimum
    assert "Neither setting was changed." in minimum
    # A single saved pair keeps the original sentence.
    single = team_split_cause(True, 3, (4, 2))
    assert single == (
        f"{RULE_MIN_PER_TEAM}. The minimum is 3 from each team, "
        "and the team split 4-2 puts 2 on one team. "
        "Neither setting was changed."
    )
    assert team_split_cause(True, 3, ((3, 3), (4, 2))) is None

    blocked = NFLShowdownOptimizer()
    blocked.jitter_pct = (0.0, 0.0)
    blocked.load_from_dataframe(_skill_pool())
    blocked.configure_showdown_rules(
        **_open_rules(
            team_split=((4, 2), (5, 1)),
            min_players_per_team_enabled=True,
            min_players_per_team=3,
        )
    )
    blocked.set_lineup_count(1)
    with pytest.raises(RuntimeError, match="4-2 or 5-1") as blocked_raise:
        blocked.generate_lineups()
    assert "built 0" in str(blocked_raise.value)
    assert "Neither setting was changed." in str(blocked_raise.value)
    assert blocked.team_split == ((4, 2), (5, 1))
    assert blocked.min_players_per_team == 3


def test_one_binary_per_allowed_split_and_no_side_bit_when_the_team_is_named():
    class _Player:
        def __init__(self, team):
            self.team = team

    class _Solver:
        def __init__(self):
            self.variables = []
            self.rows = []

        def add_variable(self, name, min_value=None, max_value=None):
            self.variables.append(name)
            return name

        def add_constraint(self, variables, coefficients, sign, rhs, name=None):
            self.rows.append((name, list(variables), rhs))

    def _rule(heavy):
        rule = AllowedTeamSplitsRule.__new__(AllowedTeamSplitsRule)
        rule.optimizer = type("Opt", (), {})()
        rule.optimizer._showdown_allowed_splits = (
            "BUF",
            ((4, 2), (5, 1)),
            heavy,
        )
        rule.players_dict = {}
        for index in range(3):
            rule.players_dict[_Player("BUF")] = f"b{index}"
            rule.players_dict[_Player("KC")] = f"k{index}"
        return rule

    named = _Solver()
    _rule("BUF").apply(named)
    assert named.variables == ["showdown_team_split_4_2", "showdown_team_split_5_1"]
    one = next(row for row in named.rows if row[0] == "showdown_team_split_one")
    assert one[1] == named.variables
    assert one[2] == 1

    either = _Solver()
    _rule(None).apply(either)
    assert either.variables == [
        "showdown_team_split_4_2",
        "showdown_team_split_side_4_2",
        "showdown_team_split_5_1",
        "showdown_team_split_side_5_1",
    ]


def _showdown_page():
    from src.dk_scoring import SCORING_VERSION
    from src.parser import parse_dk_csv
    from src.slate_week import slate_games_from_pool, slate_key
    from src.sleeper import MATCH_CACHE_VERSION

    pool = parse_dk_csv(MOCK_SHOWDOWN)
    pool["PPR"] = pd.Series(pd.NA, index=pool.index, dtype="Float64")
    pool["Scoring"] = ""
    pool["ScoringNote"] = ""
    key = slate_key(slate_games_from_pool(pool))
    pool.attrs["scoring_version"] = SCORING_VERSION
    pool.attrs["match_schema"] = MATCH_CACHE_VERSION
    pool.attrs["slate_key"] = key
    pool.attrs["fingerprint"] = f"team-split|{key}"
    summary = {
        "ok": True,
        "applied": True,
        "scoring": SCORING_VERSION,
        "match_schema": MATCH_CACHE_VERSION,
        "season": "2025",
        "week": 1,
        "season_type": "regular",
        "fingerprint": pool.attrs["fingerprint"],
        "slate_key": key,
        "matched": 0,
        "rows": len(pool),
        "inactive_rows": 0,
        "ppr_fallback_rows": 0,
    }
    return pool, summary


def _split_control(app):
    return app.segmented_control(key="team_split_showdown_gpp")


def _team_split_row(app):
    for frame in app.dataframe:
        table = frame.value
        if table is None or "Setting" not in getattr(table, "columns", []):
            continue
        match = table.loc[table["Setting"] == "Team split"]
        if not match.empty:
            return match.iloc[0]
    return None


def _heavy_box(app):
    return next((item for item in app.selectbox if item.label == "More players from"), None)


def test_streamlit_multi_select_drives_the_lineups_and_loads_a_saved_string():
    """The segmented control is multi-select. A saved single choice still loads."""
    app_test = pytest.importorskip("streamlit.testing.v1")
    from unittest.mock import patch

    pool, summary = _showdown_page()
    app = app_test.AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app.py"),
        default_timeout=120,
    )
    app.session_state["app_view"] = "Optimizer"
    app.session_state["contest_format"] = ContestFormat.SHOWDOWN.value
    app.session_state["contest_preset"] = ContestPreset.SHOWDOWN_GPP.value
    app.session_state["player_pool"] = pool
    app.session_state["sleeper_summary"] = summary
    app.session_state["pool_signature"] = "team-split"
    app.session_state["weather_by_home"] = {}
    app.session_state["weather_source_json"] = "{}"
    app.session_state["lineup_count_showdown_gpp"] = 3
    # The previous single-select control stored one string under this key.
    app.session_state["team_split_showdown_gpp"] = "4-2"

    with patch("src.weather.fetch_slate_weather", return_value={}), patch(
        "src.vegas_cache.fetch_vegas_lines", return_value={}
    ):
        app.run(timeout=120)
        assert not app.exception, app.exception
        control = _split_control(app)
        assert list(control.value) == ["4-2"]
        row = _team_split_row(app)
        assert row is not None
        assert row["Preset default"] == "Any"
        assert row["Actual"] == "4-2, either team"
        assert _heavy_box(app) is not None
        info = " ".join(str(item.value) for item in app.info)
        assert "Team split" in info

        control.set_value(["3-3"])
        app.run(timeout=120)
        assert not app.exception, app.exception
        assert list(_split_control(app).value) == ["3-3"]
        assert _heavy_box(app) is None
        assert _team_split_row(app)["Actual"] == "3-3"

        _split_control(app).set_value([])
        app.run(timeout=120)
        assert not app.exception, app.exception
        assert list(_split_control(app).value) == []
        assert _team_split_row(app) is None
        assert _heavy_box(app) is None

        _split_control(app).set_value(["Any"])
        app.run(timeout=120)
        assert list(_split_control(app).value) == ["Any"]
        assert _team_split_row(app) is None

        # Any is exclusive. Adding the splits from Any leaves only the splits.
        chosen = _split_control(app)
        chosen.select("4-2").select("5-1")
        app.run(timeout=120)
        assert not app.exception, app.exception
        assert list(_split_control(app).value) == ["4-2", "5-1"]
        heavy = _heavy_box(app)
        assert heavy is not None
        assert "BUF" in heavy.options
        heavy.set_value("BUF")
        app.run(timeout=120)
        assert not app.exception, app.exception
        assert _team_split_row(app)["Actual"] == "4-2 or 5-1, BUF heavy"
        assert "Team split" in " ".join(str(item.value) for item in app.info)

        next(item for item in app.button if item.label == "Generate Lineups").click()
        app.run(timeout=180)
        assert not app.exception, app.exception
        errors = [str(item.value) for item in app.error]
        assert not any("Optimization failed" in value for value in errors), errors
        lineups = list(app.session_state["lineups"] or [])
        assert len(lineups) == 3
        for lineup in lineups:
            counts = _counts(lineup)
            assert counts.get("BUF") in {4, 5}
            assert 3 not in counts.values()
            assert sorted(counts.values()) in ([1, 5], [2, 4])
