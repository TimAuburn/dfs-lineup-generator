"""Showdown lineup rules: captain ban, players per team, and the salary floor.

The checks are the optimizer's. Projections are not an input. Seed 42 with
the three rules off matches the lineup main produced. Classic cash is the
same lineup main produced.
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
from src.optimizer import NFLClassicOptimizer, NFLShowdownOptimizer, optimize_tiered_salaries
from src.parser import parse_dk_csv
from src.presets import ContestPreset, apply_preset, format_preset_summary, get_preset_config
from src.showdown_rules import (
    RULE_MIN_PER_TEAM,
    RULE_NO_K_DST_CAPTAIN,
    RULE_SALARY_FLOOR,
    salary_floor_met,
    team_minimum_met,
    unused_salary,
)

MOCK_CLASSIC = Path(__file__).parent / "mock_dk_salaries.csv"
MOCK_SHOWDOWN = Path(__file__).parent / "mock_dk_showdown.csv"
GOLDEN_SHOWDOWN = Path(__file__).parent / "golden" / "showdown_cash_seed42.csv"

CLASSIC_IDS = ["1001", "2001", "2003", "3012", "3010", "3013", "4004", "2002", "5004"]
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
    assert cash.min_players_per_team_enabled is True
    assert gpp.min_players_per_team_enabled is True
    assert cash.min_players_per_team == 2
    assert gpp.min_players_per_team == 2
    assert cash.salary_floor is True
    assert cash.max_unused_salary == 1_500
    assert gpp.salary_floor is False
    assert gpp.max_unused_salary == 1_500

    cash_rows = dict(format_preset_summary(cash))
    gpp_rows = dict(format_preset_summary(gpp))
    assert cash_rows["No kicker or DST captain"].startswith("On")
    assert gpp_rows["No kicker or DST captain"].startswith("On")
    assert cash_rows["Minimum players per team"] == "2 from each team"
    assert gpp_rows["Minimum players per team"] == "2 from each team"
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

    # One legal lineup, so there is no overlap pair to report. Solver-reported
    # names used to keep the "did not identify" sentence next to the rule.
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
    assert RULE_MIN_PER_TEAM in message
    assert "Candidate limits at capacity: Max overlap 1:" in message
    assert "share 1 players" in message
    assert "The solver stopped without naming one binding constraint." not in message
    assert (
        "The solver stopped and did not identify which constraint blocked the next lineup."
        not in message
    )


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
    assert showdown.min_players_per_team_enabled is True
    assert showdown.min_players_per_team == 2
    assert showdown.salary_floor is True
    assert showdown.max_unused_salary == 1_500
    lineups = showdown.generate_lineups()
    lineup = lineups[0]
    captain = lineup.loc[lineup["Slot"] == "CPT"].iloc[0]
    assert captain["Position"] not in {"K", "DST"}
    counts = lineup["Team"].value_counts().to_dict()
    assert team_minimum_met(counts, ["KC", "BUF"], 2) is True
    spent = int(lineup.attrs["salary"])
    assert salary_floor_met(spent, 50_000, 1_500) is True
    # The previous cash build already cleared these three rules, so the ids stay.
    assert [str(player_id) for player_id in lineup["ID"]] == SHOWDOWN_IDS
