"""Showdown lineup rules: captain ban, players per team, and the salary floor.

The checks are the optimizer's. Projections are not an input. Seed 42 with
the three rules off matches the lineup main produced. Classic cash is the
seed-42 result with the same-team QB/RB ban, a 127.5-point tie with the
previous roster.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import apply_low_total_kicker_cap, kicker_captain_exposure_cap
from src.exporter import lineups_to_dk_upload_csv
from src.lineup_capacity import SolvePlan, _cache_key, measure
from src.optimizer import NFLClassicOptimizer, NFLShowdownOptimizer, optimize_tiered_salaries
from src.parser import parse_dk_csv
from src.presets import (
    ContestFormat,
    ContestPreset,
    apply_preset,
    config_with_overrides,
    format_preset_summary,
    format_settings_comparison,
    get_preset_config,
)
from src.player_pool import generation_pool, locked_player_ids
from src.showdown_rules import (
    CONTROL_MAX_PLAYERS_PER_TEAM,
    CONTROL_MIN_PLAYERS_PER_TEAM,
    REQUIRE_MULTIPLE_PLAYERS_LABEL,
    RULE_CASH_CAPTAIN,
    RULE_MIN_PER_TEAM,
    RULE_NO_K_DST_CAPTAIN,
    RULE_SALARY_FLOOR,
    multiple_players_per_team,
    salary_floor_met,
    team_minimum_fit_cause,
    team_minimum_met,
    team_rule_label,
    unused_salary,
)

MOCK_CLASSIC = Path(__file__).parent / "mock_dk_salaries.csv"
MOCK_SHOWDOWN = Path(__file__).parent / "mock_dk_showdown.csv"
GOLDEN_SHOWDOWN = Path(__file__).parent / "golden" / "showdown_cash_seed42.csv"

CLASSIC_IDS = ["1001", "2001", "2003", "3012", "3013", "3009", "4004", "2002", "5006"]
SHOWDOWN_IDS = ["91003", "92006", "95002", "93010", "91002", "92008"]
GAME = "KC@BUF 11/15/2026 01:00PM ET"


def _ids(lineups) -> list[list[str]]:
    return [[str(player_id) for player_id in lineup["ID"].tolist()] for lineup in lineups]


def _pair(pos, name, flex_id, team, flex_salary, fppg) -> list[dict]:
    """CPT salary and the captain projection boost are 1.5x the flex row."""
    return [
        _row(pos, name, flex_id + 1, "CPT", team, int(round(flex_salary * 1.5)), fppg),
        _row(pos, name, flex_id, "FLEX", team, flex_salary, fppg),
    ]


def _row(pos, name, pid, slot, team, salary, fppg) -> dict:
    return {
        "Position": pos,
        "Name + ID": f"{name} ({pid})",
        "Name": name,
        "ID": str(pid),
        "Roster Position": slot,
        "Salary": salary,
        "Game Info": GAME,
        "TeamAbbrev": team,
        "AvgPointsPerGame": fppg,
        "Projection": fppg,
    }


def _solve(rows, **rules):
    frame = pd.DataFrame(rows)
    optimizer = NFLShowdownOptimizer()
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.jitter_seed = 42
    optimizer.load_from_dataframe(frame)
    optimizer.configure_showdown_rules(
        min_salary_cap=rules.pop("min_salary_cap", None),
        max_salary_cap=rules.pop("max_salary_cap", None),
        min_teams=rules.pop("min_teams", 2),
        game_script_correlation=False,
        max_k_dst_combined=None,
        max_kickers=rules.pop("max_kickers", None),
        allow_kicker_captain=rules.pop("allow_kicker_captain", False),
        captain_positions=rules.pop("captain_positions", None),
        ban_k_dst_captain=rules.pop("ban_k_dst_captain", False),
        min_players_per_team_enabled=rules.pop("min_players_per_team_enabled", False),
        min_players_per_team=rules.pop("min_players_per_team", 2),
        salary_floor=rules.pop("salary_floor", False),
        max_unused_salary=rules.pop("max_unused_salary", 1_500),
        team_split=rules.pop("team_split", None),
        flex_min_salary=rules.pop("flex_min_salary", None),
        ban_dst=rules.pop("ban_dst", False),
    )
    if "max_repeating_players" in rules:
        optimizer.max_repeating_players = rules.pop("max_repeating_players")
    locked = rules.pop("locked", None)
    if locked is not None:
        optimizer.set_locked_players(locked)
    if rules:
        raise AssertionError(f"unused solver settings: {sorted(rules)}")
    optimizer.set_lineup_count(1)
    return optimizer


def _generate(rows, n=1, **rules):
    optimizer = _solve(rows, **rules)
    optimizer.set_lineup_count(n)
    return optimizer, optimizer.generate_lineups()


def _skill_pool() -> list[dict]:
    """Two teams of skill players. HOU projections are higher, so 5-1 is the open build."""
    rows = []
    for index, name in enumerate(["A", "B", "C", "D", "E", "F"]):
        rows += _pair("WR", f"HOU {name}", 100 + index * 10, "HOU", 5000, 30 - index)
        rows += _pair("WR", f"JAX {name}", 200 + index * 10, "JAX", 4000, 8 - index * 0.1)
    return rows


def _kicker_pool() -> list[dict]:
    rows = []
    rows += _pair("K", "Harrison Butker", 10, "KC", 5000, 80)
    rows += _pair("DST", "Chiefs", 12, "KC", 4500, 70)
    rows += _pair("WR", "Xavier Worthy", 20, "KC", 7000, 15)
    rows += _pair("RB", "Isiah Pacheco", 30, "KC", 8000, 14)
    rows += _pair("WR", "Rashee Rice", 40, "KC", 6000, 12)
    rows += _pair("WR", "Khalil Shakir", 50, "BUF", 6500, 13)
    rows += _pair("RB", "James Cook", 60, "BUF", 7500, 14)
    rows += _pair("TE", "Dalton Kincaid", 70, "BUF", 5500, 11)
    rows += _pair("WR", "Keon Coleman", 80, "BUF", 5000, 10)
    return rows


def _salary_pool() -> list[dict]:
    """Best lineup leaves $1,501. The next leaves $1,500 and scores one point less.

    Captain salary is 1.5x. KC A is the only captain that reaches the floor,
    and only with BUF B in the flex spots.
    """
    rows = []
    rows += _pair("WR", "KC A", 10, "KC", 10002, 100)
    rows += _pair("WR", "KC B", 20, "KC", 8000, 30)
    rows += _pair("WR", "KC C", 30, "KC", 7000, 28)
    rows += _pair("WR", "KC D", 40, "KC", 6497, 26)
    rows += _pair("WR", "KC E", 50, "KC", 6000, 24)
    rows += _pair("WR", "BUF A", 60, "BUF", 5999, 20)
    rows += _pair("WR", "BUF B", 70, "BUF", 6000, 19)
    return rows


def test_showdown_presets_show_the_three_rules():
    cash = get_preset_config(ContestPreset.SHOWDOWN_CASH)
    gpp = get_preset_config(ContestPreset.SHOWDOWN_GPP)
    assert cash.ban_k_dst_captain is True
    assert gpp.ban_k_dst_captain is True
    assert cash.min_players_per_team_enabled is False
    assert gpp.min_players_per_team_enabled is False
    assert cash.min_players_per_team == 2
    assert gpp.min_players_per_team == 2
    assert cash.min_teams == 2
    assert gpp.min_teams == 2
    assert cash.salary_floor is True
    assert cash.max_unused_salary == 1_500
    assert gpp.salary_floor is False
    assert gpp.max_unused_salary == 1_500

    cash_rows = dict(format_preset_summary(cash))
    gpp_rows = dict(format_preset_summary(gpp))
    assert cash_rows["No kicker or DST captain"].startswith("On")
    assert gpp_rows["No kicker or DST captain"].startswith("On")
    assert cash_rows["Minimum players per team"] == "Off — at least 1 from each team"
    assert gpp_rows["Minimum players per team"] == "Off — at least 1 from each team"
    assert cash_rows["Min teams"] == "2"
    assert gpp_rows["Min teams"] == "2"
    for description in (cash.description, gpp.description):
        assert "at least one player from each team" in description
        assert "Requiring multiple players from each team is off." not in description
        assert "at least 2 players" not in description
    assert "1,500" in cash_rows["Salary floor"]
    assert "49,600" in cash_rows["Salary floor"]
    assert gpp_rows["Salary floor"] == "Off"
    for preset in (
        ContestPreset.CLASSIC_CASH,
        ContestPreset.CLASSIC_SMALL_GPP,
        ContestPreset.CLASSIC_LARGE_GPP,
    ):
        classic = get_preset_config(preset)
        assert classic.ban_k_dst_captain is False
        assert classic.min_players_per_team_enabled is False
        assert classic.salary_floor is False
        assert "Minimum players per team" not in dict(format_preset_summary(classic))


def test_unused_salary_of_1500_passes_and_1501_fails():
    cap = 50_000
    assert unused_salary(48_500, cap) == 1_500
    assert unused_salary(48_499, cap) == 1_501
    assert salary_floor_met(48_500, cap, 1_500) is True
    assert salary_floor_met(48_499, cap, 1_500) is False


def test_team_count_edges_for_the_default_and_the_1_and_3_settings():
    teams = ["HOU", "JAX"]
    two_and_four = {"HOU": 2, "JAX": 4}
    one_and_five = {"HOU": 1, "JAX": 5}
    three_and_three = {"HOU": 3, "JAX": 3}
    assert team_minimum_met(two_and_four, teams, 2) is True
    assert team_minimum_met(one_and_five, teams, 2) is False
    assert team_minimum_met(one_and_five, teams, 1) is True
    assert team_minimum_met(two_and_four, teams, 1) is True
    assert team_minimum_met(three_and_three, teams, 3) is True
    assert team_minimum_met(two_and_four, teams, 3) is False


def test_kicker_and_dst_can_be_flex_and_cannot_be_captain_when_the_rule_is_on():
    rows = _kicker_pool()
    open_rules = dict(
        allow_kicker_captain=True,
        captain_positions=("QB", "RB", "WR", "TE", "K", "DST"),
        ban_k_dst_captain=False,
        min_salary_cap=20_000,
    )
    _, open_lineups = _generate(rows, **open_rules)
    open_captain = open_lineups[0].loc[open_lineups[0]["Slot"] == "CPT"].iloc[0]
    assert open_captain["Position"] in {"K", "DST"}
    assert open_captain["Name"] == "Harrison Butker"

    _, closed = _generate(rows, **{**open_rules, "ban_k_dst_captain": True})
    lineup = closed[0]
    captain = lineup.loc[lineup["Slot"] == "CPT"].iloc[0]
    assert captain["Position"] not in {"K", "DST"}
    assert "Harrison Butker" in set(lineup["Name"])
    flex = lineup.loc[lineup["Slot"] == "FLEX"]
    assert "Harrison Butker" in set(flex["Name"])
    assert "Chiefs" not in set(lineup.loc[lineup["Slot"] == "CPT", "Name"])


def test_locked_kicker_or_dst_captain_blocks_and_names_the_player():
    rows = _kicker_pool()
    frame = pd.DataFrame(rows)
    butker_cpt = "11"
    chiefs_cpt = "13"
    for locked, name in (([butker_cpt], "Harrison Butker"), ([chiefs_cpt], "Chiefs")):
        optimizer = _solve(
            rows,
            allow_kicker_captain=True,
            captain_positions=("QB", "RB", "WR", "TE", "K", "DST"),
            ban_k_dst_captain=True,
            min_salary_cap=20_000,
            locked=locked,
        )
        with pytest.raises(RuntimeError, match=RULE_NO_K_DST_CAPTAIN) as raised:
            optimizer.generate_lineups()
        message = str(raised.value)
        assert name in message
        assert "locked at CPT" in message
        assert "not dropped" in message
        assert "built 0" in message
        assert optimizer.ban_k_dst_captain is True

    kept = _solve(
        rows,
        allow_kicker_captain=True,
        captain_positions=("QB", "RB", "WR", "TE", "K", "DST"),
        ban_k_dst_captain=False,
        min_salary_cap=20_000,
        locked=[butker_cpt],
    )
    lineup = kept.generate_lineups()[0]
    captain = lineup.loc[lineup["Slot"] == "CPT"].iloc[0]
    assert captain["Name"] == "Harrison Butker"
    assert str(captain["ID"]) == butker_cpt

    flex_locked, flex_lineups = _generate(
        rows,
        allow_kicker_captain=True,
        captain_positions=("QB", "RB", "WR", "TE", "K", "DST"),
        ban_k_dst_captain=True,
        min_salary_cap=20_000,
        locked=["10"],
    )
    flex_captain = flex_lineups[0].loc[flex_lineups[0]["Slot"] == "CPT"].iloc[0]
    assert flex_captain["Position"] not in {"K", "DST"}
    butker = flex_lineups[0].loc[flex_lineups[0]["Name"] == "Harrison Butker"].iloc[0]
    assert butker["Slot"] == "FLEX"
    assert flex_locked.locked_player_ids == {"10"}
    assert frame.loc[frame["ID"] == "11", "Name"].iloc[0] == "Harrison Butker"


def test_captain_share_on_a_kicker_is_not_dropped_when_the_rule_is_on():
    rows = _kicker_pool()
    frame = pd.DataFrame(rows)
    frame["CaptainExposure"] = pd.NA
    frame.loc[frame["Name"] == "Harrison Butker", "CaptainExposure"] = 1.0
    optimizer = _solve(
        rows,
        allow_kicker_captain=True,
        captain_positions=("QB", "RB", "WR", "TE", "K", "DST"),
        ban_k_dst_captain=True,
        min_salary_cap=20_000,
    )
    optimizer.load_from_dataframe(frame)
    with pytest.raises(RuntimeError, match=RULE_NO_K_DST_CAPTAIN) as raised:
        optimizer.generate_lineups()
    message = str(raised.value)
    assert "Harrison Butker (KC)" in message
    assert "100%" in message
    assert "not dropped" in message
    assert "built 0" in message


def test_minimum_players_per_team_on_off_and_at_1_and_3():
    rows = _skill_pool()
    _, open_lineups = _generate(rows, min_players_per_team_enabled=False, min_salary_cap=20_000)
    open_counts = open_lineups[0]["Team"].value_counts().to_dict()
    assert open_counts == {"HOU": 5, "JAX": 1}
    assert team_minimum_met(open_counts, ["HOU", "JAX"], 1) is True

    _, at_two = _generate(
        rows,
        min_players_per_team_enabled=True,
        min_players_per_team=2,
        min_salary_cap=20_000,
    )
    two_counts = at_two[0]["Team"].value_counts().to_dict()
    assert team_minimum_met(two_counts, ["HOU", "JAX"], 2) is True
    assert two_counts != {"HOU": 5, "JAX": 1}

    _, at_one = _generate(
        rows,
        min_players_per_team_enabled=True,
        min_players_per_team=1,
        min_salary_cap=20_000,
    )
    one_counts = at_one[0]["Team"].value_counts().to_dict()
    assert one_counts == {"HOU": 5, "JAX": 1}

    _, at_three = _generate(
        rows,
        min_players_per_team_enabled=True,
        min_players_per_team=3,
        min_salary_cap=20_000,
    )
    three_counts = at_three[0]["Team"].value_counts().to_dict()
    assert three_counts == {"HOU": 3, "JAX": 3}


def test_multiple_player_control_is_off_by_default_on_both_showdown_presets():
    assert REQUIRE_MULTIPLE_PLAYERS_LABEL == "Require multiple players from each team"
    assert CONTROL_MIN_PLAYERS_PER_TEAM == 2
    assert CONTROL_MAX_PLAYERS_PER_TEAM == 5
    assert multiple_players_per_team(False, 2) == {
        "min_players_per_team_enabled": False,
        "min_players_per_team": 2,
    }
    assert team_rule_label(False, 2) == "Off — at least 1 from each team"
    assert team_minimum_fit_cause(False, 5) is None
    for preset in (ContestPreset.SHOWDOWN_CASH, ContestPreset.SHOWDOWN_GPP):
        cfg = get_preset_config(preset)
        assert cfg.min_players_per_team_enabled is False
        assert cfg.min_players_per_team == 2
        assert cfg.min_teams == 2
        rows = dict(format_preset_summary(cfg))
        assert rows["Minimum players per team"] == "Off — at least 1 from each team"
        assert "at least one player from each team" in cfg.description
        assert "Requiring multiple players from each team is off." not in cfg.description
        on = config_with_overrides(
            cfg,
            lineup_count=1,
            overrides=multiple_players_per_team(True, 3),
        )
        assert on.min_players_per_team_enabled is True
        assert on.min_players_per_team == 3
        assert on.min_teams == 2
        assert dict(format_preset_summary(on))["Minimum players per team"] == "3 from each team"
        compared = format_settings_comparison(cfg, on)
        assert compared != format_settings_comparison(
            cfg,
            config_with_overrides(cfg, lineup_count=1, overrides={}),
        )


def test_old_session_key_does_not_turn_the_team_minimum_on():
    """A saved True under the previous widget key stays unused."""
    app_test = pytest.importorskip("streamlit.testing.v1")
    app = app_test.AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app.py"),
        default_timeout=90,
    )
    app.session_state["app_view"] = "Optimizer"
    app.session_state["contest_format"] = ContestFormat.SHOWDOWN.value
    app.session_state["contest_preset"] = ContestPreset.SHOWDOWN_CASH.value
    app.session_state["min_players_per_team_enabled_showdown_cash"] = True
    app.session_state["min_players_per_team_showdown_cash"] = 4
    app.session_state["min_players_per_team_enabled_showdown_gpp"] = True
    app.session_state["min_players_per_team_showdown_gpp"] = 3
    app.run()
    assert not app.exception, app.exception

    toggle = next(
        item for item in app.toggle if item.label == REQUIRE_MULTIPLE_PLAYERS_LABEL
    )
    assert toggle.value is False
    count = next(item for item in app.number_input if item.label == "Minimum from each team")
    assert int(count.value) == 2
    assert app.session_state["require_multiple_per_team_showdown_cash"] is False
    assert int(app.session_state["require_multiple_per_team_count_showdown_cash"]) == 2
    assert app.session_state["min_players_per_team_enabled_showdown_cash"] is True

    row = None
    for frame in app.dataframe:
        table = frame.value
        if "Setting" not in getattr(table, "columns", []):
            continue
        match = table.loc[table["Setting"] == "Minimum players per team"]
        if not match.empty:
            row = match.iloc[0]
            break
    assert row is not None
    assert row["Preset default"] == "Off — at least 1 from each team"
    assert row["Actual"] == "Off — at least 1 from each team"
    assert all(
        "Requiring multiple players from each team is off." not in str(item.value)
        for item in app.caption
    )


def test_toggle_on_at_two_and_three_and_off_still_bans_a_six_zero():
    """The control uses the same fields both Showdown presets solve with."""
    rows = _skill_pool()
    for preset in (ContestPreset.SHOWDOWN_CASH, ContestPreset.SHOWDOWN_GPP):
        cfg = get_preset_config(preset)
        assert cfg.min_players_per_team_enabled is False

    _, open_lineups = _generate(
        rows,
        min_players_per_team_enabled=False,
        min_teams=2,
        min_salary_cap=20_000,
    )
    open_counts = open_lineups[0]["Team"].value_counts().to_dict()
    assert open_counts == {"HOU": 5, "JAX": 1}
    assert 6 not in open_counts.values()

    # The site cap of five from one team does not depend on min_teams or this switch.
    _, site_cap = _generate(
        rows,
        min_players_per_team_enabled=False,
        min_teams=None,
        min_salary_cap=20_000,
    )
    site_counts = site_cap[0]["Team"].value_counts().to_dict()
    assert site_counts == {"HOU": 5, "JAX": 1}
    assert max(site_counts.values()) == 5

    _, at_two = _generate(
        rows,
        **multiple_players_per_team(True, 2),
        min_teams=2,
        min_salary_cap=20_000,
    )
    two_counts = at_two[0]["Team"].value_counts().to_dict()
    assert min(two_counts.values()) >= 2
    assert two_counts != {"HOU": 5, "JAX": 1}

    _, at_three = _generate(
        rows,
        **multiple_players_per_team(True, 3),
        min_teams=2,
        min_salary_cap=20_000,
    )
    assert at_three[0]["Team"].value_counts().to_dict() == {"HOU": 3, "JAX": 3}


def test_impossible_team_minimum_is_a_message_not_a_solver_error():
    rows = _skill_pool()
    cases = (
        (4, (5, 1), "5-1"),
        (4, None, "6 spots"),
        (5, None, "6 spots"),
    )
    for minimum, split, needle in cases:
        optimizer = _solve(
            rows,
            min_players_per_team_enabled=True,
            min_players_per_team=minimum,
            team_split=split,
            min_salary_cap=20_000,
        )
        with pytest.raises(RuntimeError, match=RULE_MIN_PER_TEAM) as raised:
            optimizer.generate_lineups()
        message = str(raised.value)
        assert needle in message
        assert "built 0" in message
        assert "The minimum was not lowered." in message or "Neither setting was changed." in message
        for raw in ("CBC", "Pulp", "Traceback", "Infeasible"):
            assert raw not in message
        assert optimizer.min_players_per_team == minimum
        if split is None:
            assert optimizer.team_split is None
        else:
            assert optimizer.team_split == split


def test_capacity_cache_key_includes_the_team_minimum_toggle():
    optimizer = NFLShowdownOptimizer()
    optimizer.min_players_per_team_enabled = False
    optimizer.min_players_per_team = 2
    plan = SolvePlan()
    off = _cache_key(optimizer, plan, 150, 0.4, False)
    optimizer.min_players_per_team_enabled = True
    on_two = _cache_key(optimizer, plan, 150, 0.4, False)
    optimizer.min_players_per_team = 3
    on_three = _cache_key(optimizer, plan, 150, 0.4, False)
    assert len({off, on_two, on_three}) == 3

    frame = pd.DataFrame(_skill_pool())
    solver = NFLShowdownOptimizer()
    solver.jitter_pct = (0.0, 0.0)
    solver.jitter_seed = 42
    solver.load_from_dataframe(frame)
    solver.configure_showdown_rules(
        min_salary_cap=20_000,
        max_salary_cap=None,
        min_teams=2,
        game_script_correlation=False,
        max_k_dst_combined=None,
        max_kickers=None,
        min_players_per_team_enabled=False,
        min_players_per_team=2,
        salary_floor=False,
        flex_min_salary=None,
    )
    cache: dict = {}
    first = measure(solver, requested=1, budget_seconds=None, cache=cache, attribute=False)
    second = measure(solver, requested=1, budget_seconds=None, cache=cache, attribute=False)
    assert first.from_cache is False
    assert second.from_cache is True
    solver.min_players_per_team_enabled = True
    solver.min_players_per_team = 2
    toggled = measure(solver, requested=1, budget_seconds=None, cache=cache, attribute=False)
    assert toggled.from_cache is False


def test_team_split_and_the_minimum_are_not_silently_changed():
    rows = _skill_pool()
    optimizer = _solve(
        rows,
        min_players_per_team_enabled=True,
        min_players_per_team=3,
        team_split=(4, 2),
        min_salary_cap=20_000,
    )
    with pytest.raises(RuntimeError, match=RULE_MIN_PER_TEAM) as raised:
        optimizer.generate_lineups()
    message = str(raised.value)
    assert "4-2" in message
    assert "Neither setting was changed" in message
    assert "built 0" in message
    assert optimizer.min_players_per_team == 3
    assert optimizer.team_split == (4, 2)


def test_salary_floor_keeps_1500_and_rejects_1501_without_changing_projections():
    rows = _salary_pool()
    off, off_lineups = _generate(
        rows,
        salary_floor=False,
        min_salary_cap=None,
        max_salary_cap=None,
        max_repeating_players=None,
    )
    on, on_lineups = _generate(
        rows,
        salary_floor=True,
        max_unused_salary=1_500,
        min_salary_cap=None,
        max_salary_cap=None,
        max_repeating_players=None,
    )
    off_spent = int(off_lineups[0].attrs["salary"])
    on_spent = int(on_lineups[0].attrs["salary"])
    assert unused_salary(off_spent, 50_000) == 1_501
    assert "BUF A" in set(off_lineups[0]["Name"])
    assert unused_salary(on_spent, 50_000) == 1_500
    assert salary_floor_met(on_spent, 50_000, 1_500) is True
    assert "BUF B" in set(on_lineups[0]["Name"])
    assert "BUF A" not in set(on_lineups[0]["Name"])
    assert off.last_projected_frame.to_csv(index=False) == on.last_projected_frame.to_csv(index=False)


def test_salary_floor_that_cannot_be_met_names_the_rule_and_the_count():
    rows = [row for row in _salary_pool() if "BUF B" not in row["Name"]]
    optimizer = _solve(
        rows,
        salary_floor=True,
        max_unused_salary=1_500,
        max_repeating_players=None,
    )
    optimizer.set_lineup_count(2)
    with pytest.raises(RuntimeError, match=RULE_SALARY_FLOOR) as raised:
        optimizer.generate_lineups()
    message = str(raised.value)
    assert "built 0" in message
    assert "not loosened" in message or "Neither setting was changed" in message
    assert optimizer.salary_floor is True
    assert optimizer.max_unused_salary == 1_500


def test_one_legal_lineup_reports_how_many_were_built():
    optimizer = _solve(
        _salary_pool(),
        salary_floor=True,
        max_unused_salary=1_500,
        max_repeating_players=None,
    )
    optimizer.set_lineup_count(2)
    lineups = optimizer.generate_lineups()
    assert len(lineups) == 1
    assert unused_salary(int(lineups[0].attrs["salary"]), 50_000) == 1_500
    assert optimizer.last_shortfall is not None
    assert optimizer.last_shortfall.built == 1
    assert optimizer.last_shortfall.requested == 2
    assert RULE_SALARY_FLOOR in optimizer.last_shortfall.message
    assert "built 1" in optimizer.last_shortfall.message
    assert "not loosened" in optimizer.last_shortfall.message


def test_salary_floor_does_not_silently_replace_a_tighter_window():
    rows = []
    rows += _pair("WR", "KC A", 10, "KC", 10000, 100)
    rows += _pair("WR", "KC B", 20, "KC", 8000, 30)
    rows += _pair("WR", "KC C", 30, "KC", 7000, 28)
    rows += _pair("WR", "KC D", 40, "KC", 6000, 26)
    rows += _pair("WR", "KC E", 50, "KC", 7000, 24)
    rows += _pair("WR", "BUF A", 60, "BUF", 6000, 20)
    rows += _pair("WR", "BUF B", 70, "BUF", 6600, 19)
    optimizer, lineups = _generate(
        rows,
        salary_floor=True,
        max_unused_salary=1_500,
        min_salary_cap=49_600,
        max_salary_cap=50_000,
        max_repeating_players=None,
    )
    spent = int(lineups[0].attrs["salary"])
    assert spent >= 49_600
    assert optimizer.min_salary_cap == 49_600
    assert optimizer.max_unused_salary == 1_500
    assert "BUF B" in set(lineups[0]["Name"])
    bounds = optimizer.salary_resolution()
    assert bounds.conflict is None
    assert bounds.minimum == 49_600
    assert "49,600" in bounds.note


def test_salary_floor_and_the_window_are_not_forced_into_agreement():
    optimizer = _solve(
        _salary_pool(),
        salary_floor=True,
        max_unused_salary=1_500,
        min_salary_cap=40_000,
        max_salary_cap=48_000,
    )
    before = (optimizer.min_salary_cap, optimizer.max_salary_cap, optimizer.max_unused_salary)
    with pytest.raises(RuntimeError, match=RULE_SALARY_FLOOR) as raised:
        optimizer.generate_lineups()
    message = str(raised.value)
    assert "48,000" in message
    assert "48,500" in message
    assert "Neither setting was changed" in message
    assert "built 0" in message
    assert (optimizer.min_salary_cap, optimizer.max_salary_cap, optimizer.max_unused_salary) == before
    assert optimizer.salary_floor is True


def test_shortfall_names_the_rule_and_keeps_the_overlap_detail():
    """One extra lineup is enough to name the rule. The overlap text stays."""
    rows = _salary_pool()
    rows += _pair("WR", "BUF C", 80, "BUF", 6000, 18)
    calls = []
    original = NFLShowdownOptimizer._consume_lineups

    def _record(self, optimizer, count, players, prior):
        calls.append((int(count), len(list(prior or []))))
        return original(self, optimizer, count, players, prior)

    NFLShowdownOptimizer._consume_lineups = _record
    try:
        optimizer, lineups = _generate(
            rows,
            n=4,
            salary_floor=True,
            max_unused_salary=1_500,
            min_salary_cap=None,
            max_salary_cap=None,
            max_repeating_players=5,
        )
    finally:
        NFLShowdownOptimizer._consume_lineups = original
    assert len(lineups) == 3
    message = optimizer.last_shortfall.message
    assert RULE_SALARY_FLOOR in message
    assert "not loosened" in message
    assert "Max overlap 5" in message
    assert "share 5" in message
    assert "at capacity" in message
    # The batch is solved once. Each rule check asks for one lineup on top
    # of the lineups already built, not for the whole request again.
    assert calls[0] == (4, 0)
    assert calls[1:]
    assert all(count == 1 and prior_n == len(lineups) for count, prior_n in calls[1:])


def test_named_rule_drops_contradicting_solver_text_and_keeps_overlap():
    """A named rule keeps overlap detail and drops the generic solver sentence.

    The no-rule shortfall still uses that generic sentence.
    """
    rows = _salary_pool()
    rows += _pair("WR", "BUF C", 80, "BUF", 6000, 18)
    optimizer, lineups = _generate(
        rows,
        n=4,
        salary_floor=True,
        max_unused_salary=1_500,
        min_salary_cap=None,
        max_salary_cap=None,
        max_repeating_players=5,
    )
    assert len(lineups) == 3
    message = optimizer.last_shortfall.message
    assert RULE_SALARY_FLOOR in message
    assert "Binding limit:" in message
    assert "Candidate limits at capacity: Max overlap 5:" in message
    assert "share 5 players" in message
    assert "The solver stopped without naming one binding constraint." not in message
    assert (
        "The solver stopped and did not identify which constraint blocked the next lineup."
        not in message
    )
    assert "Solver-reported" not in message

    # One legal lineup, so there is no overlap pair to report. The salary
    # floor is the cause, so solver names are not attached.
    single = _solve(
        _salary_pool(),
        salary_floor=True,
        max_unused_salary=1_500,
        max_repeating_players=None,
    )
    single.set_lineup_count(2)
    assert len(single.generate_lineups()) == 1
    single_message = single.last_shortfall.message
    assert RULE_SALARY_FLOOR in single_message
    assert "not loosened" in single_message
    assert (
        "The solver stopped and did not identify which constraint blocked the next lineup."
        not in single_message
    )
    assert "Solver-reported" not in single_message

    plain, plain_lineups = _generate(
        rows,
        n=3,
        salary_floor=False,
        min_salary_cap=None,
        max_salary_cap=None,
        max_repeating_players=2,
    )
    assert len(plain_lineups) == 1
    assert plain.last_shortfall is not None
    plain_message = plain.last_shortfall.message
    assert "Binding limit:" not in plain_message
    assert RULE_SALARY_FLOOR not in plain_message
    assert (
        "The solver stopped and did not identify which constraint blocked the next lineup."
        in plain_message
    )
    assert "Solver-reported" in plain_message


def test_tiered_showdown_shortfall_names_the_binding_rule_and_overlap():
    """A short tiered run keeps the rule found in a tier, then the overlap detail."""
    optimizer = NFLShowdownOptimizer()
    optimizer.load_from_dataframe(parse_dk_csv(MOCK_SHOWDOWN))
    apply_preset(optimizer, ContestPreset.SHOWDOWN_GPP, lineup_count=6, jitter_seed=42)
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.max_repeating_players = 1
    lineups = optimize_tiered_salaries(
        optimizer,
        total_n=6,
        leave_money_ratio=0.5,
        standard_cap=50_000,
        discount_cap=47_000,
        discount_floor=40_000,
    )
    assert len(lineups) < 6
    message = optimizer.last_shortfall.message
    assert f"Requested 6 lineups and built {len(lineups)}." in message
    assert "Binding limit:" in message
    assert RULE_NO_K_DST_CAPTAIN in message
    # The extra per-team floor is off, so it is not a binding rule.
    assert RULE_MIN_PER_TEAM not in message
    assert "Candidate limits at capacity: Max overlap 1:" in message
    assert "share 1 players" in message
    assert "The solver stopped without naming one binding constraint." not in message
    assert (
        "The solver stopped and did not identify which constraint blocked the next lineup."
        not in message
    )
    assert "Solver-reported" not in message


def _shortfall_under_hash_seed(seed: str) -> dict:
    """One process, one PYTHONHASHSEED, the shortfall whose names used to move."""
    code = """
import json
from tests.test_showdown_rules import _salary_pool, _pair, _generate
rows = _salary_pool()
rows += _pair("WR", "BUF C", 80, "BUF", 6000, 18)
optimizer, lineups = _generate(
    rows,
    n=3,
    salary_floor=False,
    min_salary_cap=None,
    max_salary_cap=None,
    max_repeating_players=2,
)
print(json.dumps({
    "ids": [[str(player_id) for player_id in lineup["ID"].tolist()] for lineup in lineups],
    "message": optimizer.last_shortfall.message,
}))
"""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = seed
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _gpp_shortfall_under_hash_seed(seed: str) -> dict:
    """150 Showdown GPP lineups. The shortfall text used to follow the hash seed."""
    code = """
import hashlib
import json
from pathlib import Path
from src.optimizer import NFLShowdownOptimizer
from src.parser import parse_dk_csv
from src.presets import ContestPreset, apply_preset

optimizer = NFLShowdownOptimizer()
optimizer.load_from_dataframe(parse_dk_csv(Path("tests/mock_dk_showdown.csv")))
apply_preset(optimizer, ContestPreset.SHOWDOWN_GPP, lineup_count=150, jitter_seed=42)
lineups = optimizer.generate_lineups()
ids = tuple(tuple(str(player_id) for player_id in lineup["ID"].tolist()) for lineup in lineups)
digest = hashlib.sha256("\\n".join(",".join(row) for row in ids).encode()).hexdigest()
print(json.dumps({
    "built": len(lineups),
    "id_hash": digest,
    "message": None if optimizer.last_shortfall is None else optimizer.last_shortfall.message,
}))
"""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = seed
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_showdown_gpp_shortfall_message_is_identical_across_hash_seeds():
    """A 150-lineup Showdown shortfall reads the same under four hash seeds."""
    runs = [_gpp_shortfall_under_hash_seed(seed) for seed in ("0", "1", "2", "42")]
    messages = [run["message"] for run in runs]
    assert all(message == messages[0] for message in messages)
    assert all(run["built"] == runs[0]["built"] for run in runs)
    assert all(run["id_hash"] == runs[0]["id_hash"] for run in runs)
    message = messages[0]
    assert message is not None
    # The extra per-team floor is off, so overlap is the limit this batch names.
    assert RULE_MIN_PER_TEAM not in message
    assert "Candidate limits at capacity: Max overlap 3:" in message
    assert "The solver stopped without naming one binding constraint." in message
    assert "Solver-reported" not in message
    assert (
        "The solver stopped and did not identify which constraint blocked the next lineup."
        not in message
    )


def test_shortfall_message_is_identical_across_hash_seeds():
    """Solver-reported names stay put when PYTHONHASHSEED changes. Lineups do too."""
    first = _shortfall_under_hash_seed("0")
    second = _shortfall_under_hash_seed("1")
    assert first["ids"] == second["ids"]
    assert first["ids"] == [["11", "60", "20", "30", "40", "50"]]
    assert first["message"] == second["message"]
    assert (
        "The solver stopped and did not identify which constraint blocked the next lineup."
        in first["message"]
    )
    assert "Solver-reported" in first["message"]


def test_low_total_kicker_cap_is_skipped_while_the_captain_ban_is_on():
    assert kicker_captain_exposure_cap(
        low_scoring=True,
        ban_k_dst_captain=False,
        cash_preset=False,
    ) == 0.10
    assert kicker_captain_exposure_cap(
        low_scoring=True,
        ban_k_dst_captain=True,
        cash_preset=False,
    ) is None
    assert kicker_captain_exposure_cap(
        low_scoring=True,
        ban_k_dst_captain=False,
        cash_preset=True,
    ) is None
    assert kicker_captain_exposure_cap(
        low_scoring=False,
        ban_k_dst_captain=False,
        cash_preset=False,
    ) is None

    banned = NFLShowdownOptimizer()
    banned.ban_k_dst_captain = True
    banned.allow_kicker_captain = False
    assert apply_low_total_kicker_cap(banned, low_scoring=True, cash_preset=False) is None
    assert banned.kicker_max_exposure is None
    assert banned.allow_kicker_captain is False

    open_captain = NFLShowdownOptimizer()
    open_captain.ban_k_dst_captain = False
    assert apply_low_total_kicker_cap(
        open_captain, low_scoring=True, cash_preset=False
    ) == 0.10
    assert open_captain.kicker_max_exposure == 0.10
    assert open_captain.allow_kicker_captain is True


def test_solver_budget_follows_a_slate_cap_other_than_50000():
    """A cap above or below $50,000 is the solver budget, not the pydfs default."""
    high = parse_dk_csv(MOCK_SHOWDOWN)
    high.attrs["salary_cap"] = 60_000
    optimizer = NFLShowdownOptimizer()
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.jitter_seed = 42
    optimizer.load_from_dataframe(high)
    optimizer.configure_showdown_rules(
        min_salary_cap=55_000,
        max_salary_cap=None,
        game_script_correlation=False,
        max_k_dst_combined=None,
        max_kickers=None,
        ban_k_dst_captain=False,
        min_players_per_team_enabled=False,
        salary_floor=False,
        flex_min_salary=None,
    )
    optimizer.set_lineup_count(1)
    try:
        lineups = optimizer.generate_lineups()
    except Exception as exc:  # noqa: BLE001 - the assertion is the raw solver text
        raise AssertionError(str(exc)) from exc
    spent = int(lineups[0].attrs["salary"])
    assert 55_000 <= spent <= 60_000
    assert "Min salary greater than max budget" not in str(getattr(optimizer.last_shortfall, "message", ""))

    low = parse_dk_csv(MOCK_SHOWDOWN)
    low.attrs["salary_cap"] = 45_000
    capped = NFLShowdownOptimizer()
    capped.jitter_pct = (0.0, 0.0)
    capped.jitter_seed = 42
    capped.load_from_dataframe(low)
    capped.configure_showdown_rules(
        min_salary_cap=40_000,
        max_salary_cap=None,
        game_script_correlation=False,
        max_k_dst_combined=None,
        max_kickers=None,
        ban_k_dst_captain=False,
        min_players_per_team_enabled=False,
        salary_floor=False,
        flex_min_salary=None,
    )
    capped.set_lineup_count(1)
    lineups = capped.generate_lineups()
    spent = int(lineups[0].attrs["salary"])
    assert 40_000 <= spent <= 45_000


def test_seed_42_classic_is_unchanged_and_showdown_rules_off_match_main():
    classic = NFLClassicOptimizer()
    classic.load_from_dataframe(parse_dk_csv(MOCK_CLASSIC))
    apply_preset(classic, ContestPreset.CLASSIC_CASH, jitter_seed=42)
    assert _ids(classic.generate_lineups()) == [CLASSIC_IDS]

    showdown = NFLShowdownOptimizer()
    showdown.load_from_dataframe(parse_dk_csv(MOCK_SHOWDOWN))
    apply_preset(showdown, ContestPreset.SHOWDOWN_CASH, jitter_seed=42)
    showdown.ban_k_dst_captain = False
    showdown.min_players_per_team_enabled = False
    showdown.salary_floor = False
    lineups = showdown.generate_lineups()
    assert _ids(lineups) == [SHOWDOWN_IDS]
    assert lineups_to_dk_upload_csv(lineups, contest_format="showdown") == GOLDEN_SHOWDOWN.read_text(
        encoding="utf-8"
    )


def test_seed_42_showdown_cash_defaults_follow_the_three_rules():
    showdown = NFLShowdownOptimizer()
    showdown.load_from_dataframe(parse_dk_csv(MOCK_SHOWDOWN))
    apply_preset(showdown, ContestPreset.SHOWDOWN_CASH, jitter_seed=42)
    assert showdown.ban_k_dst_captain is True
    assert showdown.min_players_per_team_enabled is False
    assert showdown.min_players_per_team == 2
    assert showdown.min_teams == 2
    assert showdown.salary_floor is True
    assert showdown.max_unused_salary == 1_500
    lineups = showdown.generate_lineups()
    lineup = lineups[0]
    captain = lineup.loc[lineup["Slot"] == "CPT"].iloc[0]
    assert captain["Position"] not in {"K", "DST"}
    counts = lineup["Team"].value_counts().to_dict()
    assert team_minimum_met(counts, ["KC", "BUF"], 1) is True
    assert 6 not in counts.values()
    spent = int(lineup.attrs["salary"])
    assert salary_floor_met(spent, 50_000, 1_500) is True
    # The cash optimum already used both teams, so dropping the extra floor
    # leaves this roster in place.
    assert [str(player_id) for player_id in lineup["ID"]] == SHOWDOWN_IDS


FLEX_LOCK_MESSAGE = "Use 100% to lock at captain, or clear Captain % to lock at FLEX."


def _captain_request(name: str, *, lock: bool, share) -> pd.DataFrame:
    frame = parse_dk_csv(MOCK_SHOWDOWN)
    frame["Include"] = True
    frame["Lock"] = False
    frame["Captain"] = True
    frame["CaptainExposure"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    if lock:
        frame.loc[frame["Name"].eq(name), "Lock"] = True
    if share is not None:
        frame.loc[frame["Name"].eq(name), "CaptainExposure"] = share
    return frame


def _solve_request(frame: pd.DataFrame, preset: ContestPreset, n: int = 3, **overrides):
    eligible = generation_pool(frame, showdown=True)
    optimizer = NFLShowdownOptimizer()
    optimizer.load_from_dataframe(eligible)
    apply_preset(optimizer, preset, lineup_count=n, jitter_seed=42)
    for key, value in overrides.items():
        setattr(optimizer, key, value)
    optimizer.set_lineup_count(n)
    optimizer.set_locked_players(locked_player_ids(eligible, showdown=True))
    return optimizer


def _captains(lineups) -> list[tuple[str, str]]:
    found = []
    for lineup in lineups:
        captain = lineup.loc[lineup["Slot"] == "CPT"].iloc[0]
        found.append((str(captain["Name"]), str(captain["ID"])))
    return found


def test_lock_alone_keeps_the_player_at_flex():
    """Blank and zero Captain % still lock the flex row."""
    for share in (None, 0.0):
        frame = _captain_request("Josh Allen", lock=True, share=share)
        eligible = generation_pool(frame, showdown=True)
        assert locked_player_ids(eligible, showdown=True) == ["91004"]
        for preset in (ContestPreset.SHOWDOWN_CASH, ContestPreset.SHOWDOWN_GPP):
            optimizer = _solve_request(frame, preset, n=2)
            lineups = optimizer.generate_lineups()
            assert len(lineups) == 2
            for lineup in lineups:
                captain = lineup.loc[lineup["Slot"] == "CPT"].iloc[0]
                flex = lineup.loc[lineup["Name"].eq("Josh Allen")]
                assert str(captain["Name"]) != "Josh Allen"
                assert list(flex["Slot"]) == ["FLEX"]
                assert list(flex["ID"].astype(str)) == ["91004"]


def test_captain_percent_100_without_lock_is_captain_every_lineup():
    frame = _captain_request("Josh Allen", lock=False, share=1.0)
    eligible = generation_pool(frame, showdown=True)
    assert locked_player_ids(eligible, showdown=True) == []
    for preset in (ContestPreset.SHOWDOWN_CASH, ContestPreset.SHOWDOWN_GPP):
        lineups = _solve_request(frame, preset, n=3).generate_lineups()
        assert len(lineups) == 3
        assert _captains(lineups) == [("Josh Allen", "91003")] * 3


def test_lock_and_captain_percent_100_puts_an_eligible_player_at_captain():
    """QB, RB, and WR at $8,000+ can be locked at CPT in cash and GPP.

    The team-minimum toggle from the merged Showdown control writes
    ``min_players_per_team_enabled``. Both positions of that switch are solved.
    """
    expected = {
        "Josh Allen": "91003",
        "Isiah Pacheco": "92001",
        "Xavier Worthy": "93001",
    }
    for name, captain_id in expected.items():
        frame = _captain_request(name, lock=True, share=1.0)
        eligible = generation_pool(frame, showdown=True)
        assert locked_player_ids(eligible, showdown=True) == [captain_id]
        for preset in (ContestPreset.SHOWDOWN_CASH, ContestPreset.SHOWDOWN_GPP):
            for enabled in (False, True):
                optimizer = _solve_request(
                    frame,
                    preset,
                    n=3,
                    min_players_per_team_enabled=enabled,
                    min_players_per_team=2,
                )
                lineups = optimizer.generate_lineups()
                assert len(lineups) == 3, (name, preset, enabled, getattr(optimizer.last_shortfall, "message", None))
                assert _captains(lineups) == [(name, captain_id)] * 3
                for lineup in lineups:
                    counts = lineup["Team"].value_counts().to_dict()
                    assert team_minimum_met(counts, ["KC", "BUF"], 1)
                    assert 6 not in counts.values()
                    if enabled:
                        assert team_minimum_met(counts, ["KC", "BUF"], 2)


def test_lock_between_zero_and_100_captain_percent_keeps_the_error():
    frame = _captain_request("Josh Allen", lock=True, share=0.4)
    eligible = generation_pool(frame, showdown=True)
    assert locked_player_ids(eligible, showdown=True) == ["91004"]
    optimizer = _solve_request(frame, ContestPreset.SHOWDOWN_GPP, n=2)
    with pytest.raises(RuntimeError) as raised:
        optimizer.generate_lineups()
    message = str(raised.value)
    assert "Josh Allen" in message
    assert FLEX_LOCK_MESSAGE in message
    assert RULE_CASH_CAPTAIN not in message


def test_kicker_and_dst_captain_lock_uses_the_ban_not_the_flex_error():
    for name in ("Harrison Butker", "Chiefs"):
        for lock in (False, True):
            frame = _captain_request(name, lock=lock, share=1.0)
            optimizer = _solve_request(frame, ContestPreset.SHOWDOWN_CASH, n=1)
            with pytest.raises(RuntimeError) as raised:
                optimizer.generate_lineups()
            message = str(raised.value)
            assert RULE_NO_K_DST_CAPTAIN in message
            assert name in message
            assert "not dropped" in message
            assert "built 0" in message
            assert FLEX_LOCK_MESSAGE not in message
            assert RULE_CASH_CAPTAIN not in message


def test_cash_ineligible_captain_percent_100_is_named_with_or_without_lock():
    cases = (
        ("Travis Kelce", "TE", "$9,200"),
        ("Clyde Edwards-Helaire", "RB", "$5,200"),
        ("Keon Coleman", "WR", "$7,200"),
    )
    for name, position, salary in cases:
        for lock in (False, True):
            frame = _captain_request(name, lock=lock, share=1.0)
            optimizer = _solve_request(frame, ContestPreset.SHOWDOWN_CASH, n=2)
            with pytest.raises(RuntimeError) as raised:
                optimizer.generate_lineups()
            message = str(raised.value)
            assert RULE_CASH_CAPTAIN in message
            assert name in message
            assert position in message
            assert salary in message
            assert "cannot be CPT" in message
            assert "QB, RB, or WR at $8,000+" in message
            assert "not dropped" in message
            assert "built 0" in message
            assert FLEX_LOCK_MESSAGE not in message
            assert "CBC" not in message


def test_gpp_can_lock_a_tight_end_at_captain_when_cash_cannot():
    frame = _captain_request("Travis Kelce", lock=True, share=1.0)
    lineups = _solve_request(frame, ContestPreset.SHOWDOWN_GPP, n=2).generate_lineups()
    assert len(lineups) == 2
    assert _captains(lineups) == [("Travis Kelce", "94001")] * 2


def test_captain_lock_does_not_fight_max_exposure_or_overlap():
    """Overlap of 3 and a 100% cap still build every lineup with the CPT lock.

    A 40% cap can run out of other players. It must not blame the captain lock
    or return an empty batch.
    """
    frame = _captain_request("Josh Allen", lock=True, share=1.0)
    locked = locked_player_ids(generation_pool(frame, showdown=True), showdown=True)

    def _run(exposure):
        optimizer = _solve_request(frame, ContestPreset.SHOWDOWN_GPP, n=5)
        optimizer.set_max_repeating_players(3)
        if exposure is not None:
            optimizer.set_exposure_limits(max_exposure_pct=exposure)
        optimizer.set_locked_players(locked)
        lineups = optimizer.generate_lineups()
        return optimizer, lineups

    for exposure in (None, 100):
        optimizer, lineups = _run(exposure)
        assert optimizer.last_shortfall is None
        assert len(lineups) == 5
        assert _captains(lineups) == [("Josh Allen", "91003")] * 5

    optimizer, lineups = _run(40)
    assert lineups
    assert _captains(lineups) == [("Josh Allen", "91003")] * len(lineups)
    message = "" if optimizer.last_shortfall is None else optimizer.last_shortfall.message
    assert "forces that player into every lineup" not in message
    counts: dict[str, int] = {}
    for lineup in lineups:
        for player_name in set(lineup["Name"].astype(str)):
            counts[player_name] = counts.get(player_name, 0) + 1
    others = {name: count for name, count in counts.items() if name != "Josh Allen"}
    assert others
    assert max(others.values()) <= 2


def test_streamlit_lock_and_captain_percent_100_generates_the_cpt_slot():
    """Edit Lock and Captain % on the pool, then Generate. Josh Allen is CPT."""
    app_test = pytest.importorskip("streamlit.testing.v1")
    from src.dk_scoring import SCORING_VERSION
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
    pool.attrs["fingerprint"] = f"captain-lock|{key}"
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
    app = app_test.AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app.py"),
        default_timeout=120,
    )
    app.session_state["app_view"] = "Player pool"
    app.session_state["contest_format"] = ContestFormat.SHOWDOWN.value
    app.session_state["contest_preset"] = ContestPreset.SHOWDOWN_CASH.value
    app.session_state["player_pool"] = pool
    app.session_state["sleeper_summary"] = summary
    app.session_state["pool_signature"] = "captain-lock"
    app.session_state["weather_by_home"] = {}
    app.session_state["weather_source_json"] = "{}"
    app.run(timeout=120)
    assert not app.exception, app.exception

    editor = next(
        item
        for item in app.data_editor
        if item.value is not None and "Captain %" in getattr(item.value, "columns", [])
    )
    table = editor.value.reset_index(drop=True)
    row = int(table.index[table["Name"].eq("Josh Allen")][0])
    app.session_state[editor.key] = {
        "edited_rows": {row: {"Lock": True, "Captain %": 100}},
        "added_rows": [],
        "deleted_rows": [],
    }
    app.run(timeout=120)
    assert not app.exception, app.exception
    edited = app.session_state["player_pool"]
    allen = edited.loc[edited["Name"].eq("Josh Allen")]
    assert list(allen["Lock"]) == [True, True]
    assert list(allen["CaptainExposure"]) == pytest.approx([1.0, 1.0])
    assert locked_player_ids(generation_pool(edited, showdown=True), showdown=True) == ["91003"]

    app.session_state["app_view"] = "Optimizer"
    app.session_state["lineup_count_showdown_cash"] = 2
    app.run(timeout=120)
    assert not app.exception, app.exception
    next(item for item in app.button if item.label == "Generate Lineups").click()
    app.run(timeout=180)
    assert not app.exception, app.exception
    errors = [item.value for item in app.error]
    assert not any("Optimization failed" in value for value in errors), errors
    lineups = list(app.session_state["lineups"] or [])
    assert len(lineups) == 2
    assert _captains(lineups) == [("Josh Allen", "91003")] * 2
    grid = next(
        frame.value
        for frame in app.dataframe
        if frame.value is not None and "Captain" in getattr(frame.value, "columns", [])
    )
    assert list(grid["Captain"]) == ["Josh Allen (BUF)", "Josh Allen (BUF)"]
