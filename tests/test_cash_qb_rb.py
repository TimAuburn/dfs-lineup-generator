"""Classic cash ban on a quarterback and a running back from the same team.

The ban is a solver constraint. These tests build lineups without Streamlit.
GPP and Showdown leave the constraint off.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pydfs_lineup_optimizer.solvers.constants import SolverSign

from src.cash_rules import (
    RULE_PASS_CATCHERS,
    RULE_PASS_CATCHERS_STRICT,
    RULE_SAME_TEAM_QB_RB,
    QB_RB_CONSTRAINT_PREFIX,
    SameTeamQbRbRule,
    pass_catcher_violations,
    qb_rb_cause,
    same_team_qb_rb_violations,
)
from src.optimizer import NFLClassicOptimizer, NFLShowdownOptimizer
from src.presets import ContestPreset, apply_preset, format_preset_summary, get_preset_config


def _game(away: str, home: str) -> str:
    return f"{away}@{home} 09/10/2025 01:00PM ET"


def _row(pos, name, pid, team, salary, fppg, game) -> dict:
    roster = pos if pos in {"QB", "DST"} else f"{pos}/FLEX"
    return {
        "Position": pos,
        "Name + ID": f"{name} ({pid})",
        "Name": name,
        "ID": str(pid),
        "Roster Position": roster,
        "Salary": salary,
        "Game Info": game,
        "TeamAbbrev": team,
        "AvgPointsPerGame": fppg,
        "Projection": fppg,
    }


def _showdown_rows(pos, name, pid, team, salary, fppg, game) -> list[dict]:
    flex = _row(pos, name, pid, team, salary, fppg, game)
    flex["Roster Position"] = "FLEX"
    captain = _row(pos, name, pid + 1, team, int(round(salary * 1.5)), fppg, game)
    captain["Roster Position"] = "CPT"
    captain["Name + ID"] = f"{name} ({pid + 1})"
    return [captain, flex]


def _cash(frame, *, n=1, strict=False, ban=True):
    optimizer = NFLClassicOptimizer()
    optimizer.load_from_dataframe(frame)
    apply_preset(optimizer, ContestPreset.CLASSIC_CASH, jitter_seed=42)
    optimizer.set_lineup_count(n)
    optimizer.strict_pass_catcher_cap = strict
    optimizer.ban_same_team_qb_rb = ban
    return optimizer


def _pairs(lineup: pd.DataFrame) -> set[tuple[str, str]]:
    """(team, slot) for every RB who shares a team with the quarterback."""
    quarterback = lineup.loc[lineup["Slot"] == "QB"].iloc[0]
    running_backs = lineup[lineup["Position"].astype(str).str.split("/").str[0].eq("RB")]
    return {
        (str(row["Team"]), str(row["Slot"]))
        for _, row in running_backs.iterrows()
        if str(row["Team"]) == str(quarterback["Team"])
    }


def _flex_rb_pool() -> pd.DataFrame:
    """BUF quarterback plus Z BUF RB is the best core, and Z sorts into FLEX.

    pydfs fills the RB slots from earlier names and leaves the last running
    back for FLEX. A fourth wideout lets a legal lineup fill FLEX without Z.
    """
    game = _game("BUF", "KC")
    other = _game("PHI", "DAL")
    extra = _game("SF", "SEA")
    return pd.DataFrame(
        [
            _row("QB", "BUF QB", 1, "BUF", 7000, 40, game),
            _row("RB", "A PHI RB", 2, "PHI", 5400, 14, other),
            _row("RB", "B DAL RB", 3, "DAL", 5200, 13, other),
            _row("RB", "Z BUF RB", 4, "BUF", 5000, 30, game),
            _row("WR", "C KC WR", 5, "KC", 5000, 12, game),
            _row("WR", "D PHI WR", 6, "PHI", 4900, 11, other),
            _row("WR", "E DAL WR", 7, "DAL", 4800, 10, other),
            _row("WR", "F SEA WR", 8, "SEA", 4700, 9, extra),
            _row("WR", "G SF WR", 9, "SF", 4600, 8, extra),
            _row("TE", "DAL TE", 10, "DAL", 4500, 8, other),
            _row("DST", "PHI DST", 11, "PHI", 2800, 6, other),
            _row("QB", "KC QB", 12, "KC", 6400, 6, game),
            _row("RB", "H SF RB", 13, "SF", 4500, 7, extra),
            _row("WR", "BUF WR", 14, "BUF", 4500, 9, game),
        ]
    )


def _rb_slot_pool() -> pd.DataFrame:
    """The best running back shares the quarterback's team and takes an RB slot.

    Extra skill players from other teams give the ban several legal lineups.
    Pass catchers are split so the cash wideout cap is not what limits them.
    """
    game = _game("BUF", "KC")
    other = _game("PHI", "DAL")
    extra = _game("SF", "SEA")
    return pd.DataFrame(
        [
            _row("QB", "BUF QB", 1, "BUF", 7000, 36, game),
            _row("RB", "A BUF RB", 2, "BUF", 6200, 28, game),
            _row("RB", "B PHI RB", 3, "PHI", 5400, 12, other),
            _row("RB", "C DAL RB", 4, "DAL", 5200, 11, other),
            _row("RB", "H SF RB", 12, "SF", 5000, 10, extra),
            _row("WR", "D KC WR", 5, "KC", 5000, 16, game),
            _row("WR", "E NYJ WR", 6, "NYJ", 4900, 14, other),
            _row("WR", "F SEA WR", 7, "SEA", 4800, 13, extra),
            _row("WR", "G LAR WR", 13, "LAR", 4700, 9, extra),
            _row("WR", "I DEN WR", 14, "DEN", 4600, 8, extra),
            _row("TE", "GB TE", 8, "GB", 4500, 8, other),
            _row("TE", "MIN TE", 15, "MIN", 4400, 7, extra),
            _row("DST", "DAL DST", 9, "DAL", 2800, 6, other),
            _row("QB", "KC QB", 10, "KC", 6000, 8, game),
        ]
    )


def _both_rules_pool() -> pd.DataFrame:
    """Two BUF wideouts and a KC quarterback-running back pair are the points.

    Without either cash rule the lineup takes both wideouts and the pair.
    The Bills' quarterback is not worth rostering for the wideouts alone.
    """
    rows = []
    pid = 1
    game = _game("BUF", "KC")
    other = _game("PHI", "DAL")
    extra = _game("SF", "SEA")
    specs = [
        ("BUF", 4, 40, 36, 8, game),
        ("KC", 30, 12, 9, 26, game),
        ("PHI", 18, 11, 8, 12, other),
        ("DAL", 16, 10, 7, 11, other),
        ("SF", 14, 9, 6, 10, extra),
        ("SEA", 12, 8, 5, 9, extra),
    ]
    for team, qb_points, wr_points, wr2_points, rb_points, matchup in specs:
        rows.append(_row("QB", f"{team} QB", pid, team, 6400, qb_points, matchup))
        pid += 1
        rows.append(_row("RB", f"{team} RB", pid, team, 5600, rb_points, matchup))
        pid += 1
        rows.append(_row("WR", f"{team} WR", pid, team, 5200, wr_points, matchup))
        pid += 1
        rows.append(_row("WR", f"{team} WR2", pid, team, 4800, wr2_points, matchup))
        pid += 1
        rows.append(_row("TE", f"{team} TE", pid, team, 4500, 8, matchup))
        pid += 1
        rows.append(_row("DST", f"{team} DST", pid, team, 2600, 6, matchup))
        pid += 1
    return pd.DataFrame(rows)


def _only_kc_backs() -> pd.DataFrame:
    """The only running backs play for the only quarterback's team."""
    game = _game("KC", "BUF")
    other = _game("PHI", "DAL")
    extra = _game("SF", "SEA")
    return pd.DataFrame(
        [
            _row("QB", "KC QB", 1, "KC", 7000, 24, game),
            _row("RB", "KC RB", 2, "KC", 5600, 18, game),
            _row("RB", "KC RB2", 3, "KC", 5200, 16, game),
            _row("WR", "BUF WR", 4, "BUF", 5000, 14, game),
            _row("WR", "PHI WR", 5, "PHI", 4900, 13, other),
            _row("WR", "DAL WR", 6, "DAL", 4800, 12, other),
            _row("WR", "SF WR", 7, "SF", 4700, 11, extra),
            _row("TE", "SEA TE", 8, "SEA", 4500, 9, extra),
            _row("DST", "PHI DST", 9, "PHI", 2800, 6, other),
        ]
    )


def test_constraint_is_one_row_per_running_back():
    """QB_T + RB_j <= 1, including an RB who is also eligible for FLEX."""

    class _Opt:
        _ban_same_team_qb_rb = True

    class _Solver:
        def __init__(self):
            self.rows = []

        def add_constraint(self, variables, coefficients, sign, rhs, name=None):
            self.rows.append((list(variables), coefficients, sign, rhs, name))

    class _Player:
        def __init__(self, player_id, team, positions):
            self.id = player_id
            self.team = team
            self.positions = positions

        def __hash__(self):
            return hash(self.id)

        def __eq__(self, other):
            return isinstance(other, _Player) and self.id == other.id

    qb = _Player("1", "WSH", ("QB",))
    flex_rb = _Player("7", "WAS", ("RB", "FLEX"))
    other_rb = _Player("8", "WAS", ("RB",))
    no_qb_rb = _Player("9", "BUF", ("RB",))
    wideout = _Player("4", "WAS", ("WR",))
    variables = {
        qb: "qb_var",
        flex_rb: "flex_rb_var",
        other_rb: "rb_var",
        no_qb_rb: "buf_rb_var",
        wideout: "wr_var",
    }
    # The rule holds the optimizer through a weak reference.
    optimizer = _Opt()
    rule = SameTeamQbRbRule(optimizer, variables, None)
    solver = _Solver()
    rule.apply(solver)

    named = {row[4]: row for row in solver.rows}
    assert list(named) == [
        f"{QB_RB_CONSTRAINT_PREFIX}WAS_7",
        f"{QB_RB_CONSTRAINT_PREFIX}WAS_8",
    ]
    for name, expected_rb in (
        (f"{QB_RB_CONSTRAINT_PREFIX}WAS_7", "flex_rb_var"),
        (f"{QB_RB_CONSTRAINT_PREFIX}WAS_8", "rb_var"),
    ):
        used, coefficients, sign, rhs, _name = named[name]
        assert coefficients is None
        assert sign == SolverSign.LTE and rhs == 1
        assert used == ["qb_var", expected_rb]
        assert used.count(expected_rb) == 1

    quiet = _Opt()
    quiet._ban_same_team_qb_rb = False
    skipped = _Solver()
    SameTeamQbRbRule(quiet, variables, None).apply(skipped)
    assert skipped.rows == []


def test_cash_never_pairs_a_quarterback_with_his_running_back_at_flex():
    frame = _flex_rb_pool()
    open_lineup = _cash(frame, ban=False).generate_lineups()[0]
    flex = open_lineup.loc[open_lineup["Slot"] == "FLEX"].iloc[0]
    assert flex["Name"] == "Z BUF RB"
    assert flex["Position"] == "RB"
    assert flex["Team"] == "BUF"
    assert open_lineup.loc[open_lineup["Slot"] == "QB", "Team"].iloc[0] == "BUF"
    assert same_team_qb_rb_violations(open_lineup) == ("BUF rosters a QB with 1 RB",)

    cash = _cash(frame, n=8)
    lineups = cash.generate_lineups()
    assert len(lineups) >= 8
    for lineup in lineups:
        assert same_team_qb_rb_violations(lineup) == ()
        assert _pairs(lineup) == set()
        quarterback = lineup.loc[lineup["Slot"] == "QB"].iloc[0]
        if str(quarterback["Team"]) == "BUF":
            assert "Z BUF RB" not in set(lineup["Name"])
    opened = _cash(frame, ban=False)
    opened.generate_lineups()
    assert cash.last_projected_frame["Projection"].tolist() == (
        opened.last_projected_frame["Projection"].tolist()
    )


def test_cash_never_pairs_a_quarterback_with_his_running_back_in_an_rb_slot():
    frame = _rb_slot_pool()
    open_lineup = _cash(frame, ban=False).generate_lineups()[0]
    assert open_lineup.loc[open_lineup["Slot"] == "QB", "Team"].iloc[0] == "BUF"
    slot = open_lineup.loc[open_lineup["Name"] == "A BUF RB", "Slot"].iloc[0]
    assert slot == "RB"
    assert same_team_qb_rb_violations(open_lineup)

    lineups = _cash(frame, n=5).generate_lineups()
    assert len(lineups) >= 5
    for lineup in lineups:
        assert same_team_qb_rb_violations(lineup) == ()
        if "A BUF RB" in set(lineup["Name"]):
            assert lineup.loc[lineup["Slot"] == "QB", "Team"].iloc[0] != "BUF"


def test_pass_catcher_rule_still_holds_beside_the_running_back_ban():
    frame = _both_rules_pool()
    open_lineup = _cash(frame, ban=False).generate_lineups()[0]
    open_lineup_cap_off = _cash(frame, ban=False)
    open_lineup_cap_off.pass_catcher_team_cap = False
    naked = open_lineup_cap_off.generate_lineups()[0]
    assert pass_catcher_violations(naked, strict=False)
    assert same_team_qb_rb_violations(naked)
    assert naked.loc[naked["Slot"] == "QB", "Team"].iloc[0] == "KC"
    assert len(
        naked[(naked["Team"] == "BUF") & naked["Position"].isin(["WR", "TE"])]
    ) >= 2

    default = _cash(frame, n=8).generate_lineups()
    assert len(default) >= 8
    for lineup in default:
        assert pass_catcher_violations(lineup, strict=False) == ()
        assert same_team_qb_rb_violations(lineup) == ()

    strict = _cash(frame, n=8, strict=True).generate_lineups()
    assert len(strict) >= 8
    for lineup in strict:
        assert pass_catcher_violations(lineup, strict=True) == ()
        assert same_team_qb_rb_violations(lineup) == ()
        quarterback = str(lineup.loc[lineup["Slot"] == "QB", "Team"].iloc[0])
        catchers = lineup[
            lineup["Position"].isin(["WR", "TE"]) & (lineup["Team"] == quarterback)
        ]
        assert len(catchers) <= 1

    # The open lineup above still shows the pass-catcher breach when the ban is off
    # and the cap is on, so the two rules are not the same cut.
    assert pass_catcher_violations(open_lineup, strict=False) or same_team_qb_rb_violations(
        open_lineup
    )


def test_infeasible_cash_lineup_names_the_running_back_rule_and_does_not_drop_it():
    optimizer = _cash(_only_kc_backs())
    with pytest.raises(RuntimeError) as raised:
        optimizer.generate_lineups()
    message = str(raised.value)
    assert message == optimizer.last_shortfall.message
    assert "Requested 1 lineups and built 0." in message
    assert f"Binding limit: {RULE_SAME_TEAM_QB_RB}." in message
    assert "The rule was not loosened." in message
    assert qb_rb_cause() in message
    assert RULE_PASS_CATCHERS not in message

    opened = _cash(_only_kc_backs(), ban=False)
    lineup = opened.generate_lineups()[0]
    assert same_team_qb_rb_violations(lineup)
    assert lineup.loc[lineup["Slot"] == "QB", "Team"].iloc[0] == "KC"


def test_a_short_batch_names_the_rule_when_the_flex_would_be_his_running_back():
    """One legal lineup exists. The only other lineup puts the QB's RB at FLEX."""
    game = _game("KC", "BUF")
    other = _game("PHI", "DAL")
    frame = pd.DataFrame(
        [
            _row("QB", "KC QB", 1, "KC", 7000, 22, game),
            _row("RB", "PHI RB", 2, "PHI", 5400, 12, other),
            _row("RB", "DAL RB", 3, "DAL", 5200, 11, other),
            _row("WR", "BUF WR", 4, "BUF", 5000, 14, game),
            _row("WR", "PHI WR", 5, "PHI", 4900, 13, other),
            _row("WR", "SF WR", 6, "SF", 4800, 10, other),
            _row("TE", "DAL TE", 7, "DAL", 4500, 8, other),
            _row("DST", "DAL DST", 8, "DAL", 2800, 6, other),
            _row("WR", "SEA WR", 9, "SEA", 4500, 7, other),
            _row("RB", "Z KC RB", 10, "KC", 5000, 20, game),
        ]
    )
    locks = ["1", "2", "3", "4", "5", "6", "7", "8"]
    optimizer = _cash(frame, n=2)
    optimizer.set_locked_players(locks)
    lineups = optimizer.generate_lineups()
    assert len(lineups) == 1
    assert "SEA WR" in set(lineups[0]["Name"])
    assert "Z KC RB" not in set(lineups[0]["Name"])
    assert same_team_qb_rb_violations(lineups[0]) == ()
    assert pass_catcher_violations(lineups[0], strict=False) == ()
    message = optimizer.last_shortfall.message
    assert "Requested 2 lineups and built 1." in message
    assert f"Binding limit: {RULE_SAME_TEAM_QB_RB}." in message
    assert "The rule was not loosened." in message
    assert RULE_PASS_CATCHERS not in message
    assert RULE_PASS_CATCHERS_STRICT not in message

    opened = _cash(frame, n=2, ban=False)
    opened.set_locked_players(locks)
    opened_lineups = opened.generate_lineups()
    assert len(opened_lineups) == 2
    flexes = [
        lineup.loc[lineup["Slot"] == "FLEX", "Name"].iloc[0] for lineup in opened_lineups
    ]
    assert "Z KC RB" in flexes


def test_classic_gpp_and_showdown_can_still_pair_them():
    frame = _flex_rb_pool()
    gpp = NFLClassicOptimizer()
    gpp.load_from_dataframe(frame)
    apply_preset(gpp, ContestPreset.CLASSIC_SMALL_GPP, lineup_count=1, jitter_seed=42)
    gpp.jitter_pct = (0.0, 0.0)
    gpp.exclude_vs_dst = False
    assert gpp.ban_same_team_qb_rb is False
    gpp_lineup = gpp.generate_lineups()[0]
    assert gpp_lineup.loc[gpp_lineup["Slot"] == "QB", "Team"].iloc[0] == "BUF"
    assert same_team_qb_rb_violations(gpp_lineup)

    large = get_preset_config(ContestPreset.CLASSIC_LARGE_GPP)
    assert large.ban_same_team_qb_rb is False
    assert "QB and RB" not in dict(format_preset_summary(large))
    showdown_cash = get_preset_config(ContestPreset.SHOWDOWN_CASH)
    showdown_gpp = get_preset_config(ContestPreset.SHOWDOWN_GPP)
    assert showdown_cash.ban_same_team_qb_rb is False
    assert showdown_gpp.ban_same_team_qb_rb is False
    assert "QB and RB" not in dict(format_preset_summary(showdown_cash))
    assert "QB and RB" not in dict(format_preset_summary(showdown_gpp))

    rows = []
    game = _game("BUF", "KC")
    specs = [
        ("QB", "BUF QB", 10, "BUF", 8000, 30),
        ("RB", "BUF RB", 20, "BUF", 7000, 28),
        ("WR", "KC WR", 30, "KC", 6000, 12),
        ("WR", "PHI WR", 40, "PHI", 5500, 11),
        ("TE", "DAL TE", 50, "DAL", 5000, 10),
        ("RB", "SF RB", 60, "SF", 4800, 9),
        ("DST", "KC DST", 80, "KC", 3000, 5),
    ]
    for pos, name, pid, team, salary, points in specs:
        rows.extend(_showdown_rows(pos, name, pid, team, salary, points, game))
    showdown = NFLShowdownOptimizer()
    showdown.jitter_pct = (0.0, 0.0)
    showdown.jitter_seed = 42
    showdown.load_from_dataframe(pd.DataFrame(rows))
    showdown.configure_showdown_rules(
        min_salary_cap=None,
        max_salary_cap=None,
        min_teams=None,
        game_script_correlation=False,
        max_k_dst_combined=None,
        max_kickers=None,
        ban_k_dst_captain=False,
        min_players_per_team_enabled=False,
        salary_floor=False,
        ban_dst=False,
        require_starting_core=False,
        flex_min_salary=None,
        captain_positions=None,
        captain_min_salary=None,
    )
    showdown.set_lineup_count(1)
    assert not hasattr(showdown, "ban_same_team_qb_rb")
    lineup = showdown.generate_lineups()[0]
    buf = lineup[(lineup["Team"] == "BUF") & lineup["Position"].isin(["QB", "RB"])]
    assert set(buf["Position"]) >= {"QB", "RB"}
    assert same_team_qb_rb_violations(lineup)
