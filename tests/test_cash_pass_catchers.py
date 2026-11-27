"""Classic cash cap on same-team wide receivers and tight ends.

The cap is a solver constraint. These tests build lineups without Streamlit.
GPP and Showdown leave the constraint off, so their seed-42 upload files stay
on the bytes main produced.
"""

from __future__ import annotations

import hashlib
import pandas as pd
import pytest
from pydfs_lineup_optimizer.settings import LineupPosition
from pydfs_lineup_optimizer.solvers.constants import SolverSign

from src.cash_rules import (
    RULE_PASS_CATCHERS,
    RULE_PASS_CATCHERS_STRICT,
    PassCatcherCapSettings,
    PassCatcherTeamRule,
    extra_pass_catcher_allowance,
    pass_catcher_cause,
    pass_catcher_violations,
)
from src.exporter import lineups_to_dk_upload_csv
from src.optimizer import NFLClassicOptimizer, NFLShowdownOptimizer
from src.parser import parse_dk_csv
from src.presets import (
    ContestFormat,
    ContestPreset,
    apply_preset,
    format_preset_summary,
    get_preset_config,
)
from pathlib import Path

MOCK_CLASSIC = Path(__file__).parent / "mock_dk_salaries.csv"
MOCK_SHOWDOWN = Path(__file__).parent / "mock_dk_showdown.csv"

# Upload CSV sha256, seed 42, preset defaults. Classic cash is the 127.5-point
# tie CBC returns once the same-team QB/RB ban is in the model.
MAIN_CASH_CSV = "9b1614d798c5819fcf0be8f831d38f1ee25a323aa01c12932446b471049272fd"
MAIN_SHOWDOWN_CASH_CSV = "6980178f01545bef57fc4fdcaa43bc1b7c92459e014f779ba102aae40a437533"
MAIN_SMALL_GPP_CSV = "fa796590470cf9dc48fb533fd6d311e49711384a9c51be93ae2d7616481a6c6d"
MAIN_LARGE_GPP_CSV = "a39e4aea971f8aeb6b44d09f78baeeef06d4cfec480fdc5000deffca8efe1236"
# Showdown GPP moved when the extra per-team floor defaulted off. The previous
# hash was ca656ed4b568b473166bc0fc7da3e49d02e11679e9649bf38da3ea623b2b3d2d.
# Cash stayed put: that seed-42 lineup was already 4-2.
MAIN_SHOWDOWN_GPP_CSV = "f9ede315707544ed6eb97d22954b54c86259e2bda6a932c638d10d6067ecb9ff"


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


def _cash(frame, *, n=1, strict=False, cap=True, flex=None):
    optimizer = NFLClassicOptimizer()
    optimizer.load_from_dataframe(frame)
    apply_preset(optimizer, ContestPreset.CLASSIC_CASH, jitter_seed=42)
    optimizer.set_lineup_count(n)
    optimizer.pass_catcher_team_cap = cap
    optimizer.strict_pass_catcher_cap = strict
    if flex is not None:
        optimizer.flex_positions = flex
    return optimizer


def _buf_catchers(lineup: pd.DataFrame) -> pd.DataFrame:
    positions = lineup["Position"].map(lambda value: str(value).split("/")[0])
    return lineup[(lineup["Team"] == "BUF") & positions.isin(["WR", "TE"])]


def _sha(lineups, contest_format) -> str:
    text = lineups_to_dk_upload_csv(lineups, contest_format=contest_format)
    return hashlib.sha256(text.encode()).hexdigest()


def _stack_pool() -> pd.DataFrame:
    """Two high BUF wideouts, a weak BUF quarterback, and a full surrounding slate.

    Without the cap the points come from both BUF wideouts and a different
    quarterback. Six teams is enough for a 20-lineup batch once that pair is
    blocked.
    """
    rows = []
    pid = 1
    game = _game("BUF", "KC")
    other = _game("PHI", "DAL")
    extra = _game("SF", "SEA")
    specs = [
        ("BUF", 5, 40, 36, game),
        ("KC", 28, 14, 11, game),
        ("PHI", 24, 13, 10, other),
        ("DAL", 22, 12, 9, other),
        ("SF", 20, 12, 9, extra),
        ("SEA", 18, 11, 8, extra),
    ]
    for team, qb_points, wr_points, wr2_points, matchup in specs:
        rows.append(_row("QB", f"{team} QB", pid, team, 6400, qb_points, matchup))
        pid += 1
        rows.append(_row("RB", f"{team} RB", pid, team, 5200, 12, matchup))
        pid += 1
        rows.append(_row("RB", f"{team} RB2", pid, team, 4800, 9, matchup))
        pid += 1
        rows.append(_row("WR", f"{team} WR", pid, team, 5000, wr_points, matchup))
        pid += 1
        rows.append(_row("WR", f"{team} WR2", pid, team, 4600, wr2_points, matchup))
        pid += 1
        rows.append(_row("WR", f"{team} WR3", pid, team, 4400, 7, matchup))
        pid += 1
        rows.append(_row("TE", f"{team} TE", pid, team, 4500, 8, matchup))
        pid += 1
        rows.append(_row("DST", f"{team} DST", pid, team, 2600, 6, matchup))
        pid += 1
    return pd.DataFrame(rows)


def _qb_stack_pool() -> pd.DataFrame:
    """BUF quarterback plus two BUF wideouts is the highest-scoring core."""
    rows = []
    game = _game("BUF", "KC")
    other = _game("PHI", "DAL")
    rows.extend(
        [
            _row("QB", "BUF QB", 1, "BUF", 7000, 40, game),
            _row("WR", "BUF WR", 2, "BUF", 5200, 30, game),
            _row("WR", "BUF WR2", 3, "BUF", 5000, 28, game),
            _row("TE", "BUF TE", 4, "BUF", 4500, 6, game),
            _row("QB", "KC QB", 5, "KC", 6800, 12, game),
            _row("WR", "KC WR", 6, "KC", 5000, 11, game),
            _row("WR", "KC WR2", 7, "KC", 4700, 8, game),
            _row("TE", "KC TE", 8, "KC", 4500, 7, game),
            _row("RB", "PHI RB", 9, "PHI", 5400, 14, other),
            _row("RB", "DAL RB", 10, "DAL", 5200, 13, other),
            _row("WR", "PHI WR", 11, "PHI", 4800, 10, other),
            _row("WR", "DAL WR", 12, "DAL", 4600, 9, other),
            _row("TE", "PHI TE", 13, "PHI", 4500, 8, other),
            _row("DST", "PHI DST", 14, "PHI", 2800, 6, other),
            _row("RB", "DAL RB2", 15, "DAL", 4500, 8, other),
        ]
    )
    return pd.DataFrame(rows)


def _flex_wr_pool() -> pd.DataFrame:
    """The second BUF wideout sorts last, so he is the FLEX when both are used.

    pydfs assigns earlier variable names to the WR slots and the last wideout
    to FLEX. An extra running back lets a legal lineup fill FLEX without him.
    """
    game = _game("BUF", "KC")
    other = _game("PHI", "DAL")
    return pd.DataFrame(
        [
            _row("QB", "KC QB", 1, "KC", 7000, 30, game),
            _row("RB", "PHI RB", 2, "PHI", 5400, 12, other),
            _row("RB", "DAL RB", 3, "DAL", 5200, 11, other),
            _row("RB", "N DAL RB", 11, "DAL", 4500, 15, other),
            _row("WR", "A KC WR", 4, "KC", 5000, 18, game),
            _row("WR", "B PHI WR", 5, "PHI", 4900, 17, other),
            _row("WR", "C BUF WR", 6, "BUF", 4800, 16, game),
            _row("WR", "Z BUF WR", 7, "BUF", 4700, 29, game),
            _row("TE", "DAL TE", 8, "DAL", 4500, 10, other),
            _row("DST", "PHI DST", 9, "PHI", 2800, 6, other),
            _row("QB", "BUF QB", 10, "BUF", 6000, 4, game),
        ]
    )


def _flex_te_pool() -> pd.DataFrame:
    """Two PHI tight ends. The second is the best flex and the PHI quarterback is not."""
    game = _game("PHI", "DAL")
    other = _game("KC", "BUF")
    return pd.DataFrame(
        [
            _row("QB", "KC QB", 1, "KC", 7000, 30, other),
            _row("RB", "KC RB", 2, "KC", 5400, 12, other),
            _row("RB", "BUF RB", 3, "BUF", 5200, 11, other),
            _row("WR", "KC WR", 4, "KC", 5000, 14, other),
            _row("WR", "BUF WR", 5, "BUF", 4900, 13, other),
            _row("WR", "DAL WR", 6, "DAL", 4800, 12, game),
            _row("TE", "PHI TE", 7, "PHI", 4600, 20, game),
            _row("TE", "PHI TE2", 8, "PHI", 4500, 24, game),
            _row("DST", "DAL DST", 9, "DAL", 2800, 6, game),
            _row("QB", "PHI QB", 10, "PHI", 6200, 1, game),
            _row("RB", "DAL RB", 11, "DAL", 4300, 4, game),
            # A spare wideout lets the capped lineup fill FLEX without a second
            # PHI tight end or the quarterback's own running back.
            _row("WR", "SEA WR", 12, "SEA", 4500, 9, game),
        ]
    )


def _only_buf_catchers() -> pd.DataFrame:
    """Every pass catcher is a Bill, and the Bills have no quarterback."""
    game = _game("BUF", "KC")
    return pd.DataFrame(
        [
            _row("QB", "KC QB", 1, "KC", 7000, 20, game),
            _row("RB", "KC RB", 2, "KC", 5400, 12, game),
            _row("RB", "KC RB2", 3, "KC", 5000, 10, game),
            _row("WR", "BUF WR", 4, "BUF", 5000, 18, game),
            _row("WR", "BUF WR2", 5, "BUF", 4800, 17, game),
            _row("WR", "BUF WR3", 6, "BUF", 4600, 16, game),
            _row("WR", "BUF WR4", 7, "BUF", 4500, 15, game),
            _row("TE", "BUF TE", 8, "BUF", 4500, 12, game),
            _row("DST", "KC DST", 9, "KC", 2800, 6, game),
            # Two running backs off the quarterback's team, so the pass-catcher
            # cap is what blocks the lineup. The QB/RB ban stays on for cash.
            _row("RB", "DAL RB", 10, "DAL", 5000, 8, game),
            _row("RB", "DAL RB2", 11, "DAL", 4800, 7, game),
        ]
    )


def _strict_only_pool() -> pd.DataFrame:
    """Three teams of pass catchers. Four slots, so strict mode cannot separate them."""
    game = _game("BUF", "KC")
    other = _game("PHI", "DAL")
    return pd.DataFrame(
        [
            _row("QB", "BUF QB", 1, "BUF", 7000, 25, game),
            _row("WR", "BUF WR", 2, "BUF", 5200, 18, game),
            _row("WR", "BUF WR2", 3, "BUF", 5000, 16, game),
            _row("TE", "BUF TE", 4, "BUF", 4600, 12, game),
            _row("WR", "KC WR", 5, "KC", 5000, 14, game),
            _row("WR", "PHI WR", 6, "PHI", 4900, 13, other),
            _row("RB", "DAL RB", 7, "DAL", 5400, 12, other),
            _row("RB", "DAL RB2", 8, "DAL", 5000, 10, other),
            _row("RB", "PHI RB", 9, "PHI", 4800, 9, other),
            _row("DST", "DAL DST", 10, "DAL", 2800, 6, other),
            _row("TE", "DAL TE", 11, "DAL", 3000, 4, other),
        ]
    )


def test_allowance_matches_the_pass_catcher_slots():
    classic = [
        LineupPosition("QB", ("QB",)),
        LineupPosition("RB", ("RB",)),
        LineupPosition("RB", ("RB",)),
        LineupPosition("WR", ("WR",)),
        LineupPosition("WR", ("WR",)),
        LineupPosition("WR", ("WR",)),
        LineupPosition("TE", ("TE",)),
        LineupPosition("FLEX", ("RB", "TE", "WR")),
        LineupPosition("DST", ("DST",)),
    ]
    assert extra_pass_catcher_allowance(classic) == 4
    assert extra_pass_catcher_allowance(None) == 4
    cash_flex = classic[:-2] + [LineupPosition("FLEX", ("RB", "WR")), classic[-1]]
    assert extra_pass_catcher_allowance(cash_flex) == 4
    rb_flex = classic[:-2] + [LineupPosition("FLEX", ("RB",)), classic[-1]]
    assert extra_pass_catcher_allowance(rb_flex) == 3


def test_wr_flex_eligibility_is_one_variable_in_the_team_sum():
    """A WR/FLEX player is one selection variable, so the cap counts him once."""

    class _Opt:
        pass

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

    wr_flex = _Player("7", "WSH", ("WR", "FLEX"))
    teammate = _Player("8", "WAS", ("TE",))
    quarterback = _Player("1", "WAS", ("QB",))
    other = _Player("9", "BUF", ("WR",))
    variables = {
        wr_flex: "wr_flex_var",
        teammate: "te_var",
        quarterback: "qb_var",
        other: "buf_var",
    }
    optimizer = _Opt()
    optimizer.settings = type("Settings", (), {})()
    optimizer.settings.positions = [
        LineupPosition("WR", ("WR",)),
        LineupPosition("WR", ("WR",)),
        LineupPosition("WR", ("WR",)),
        LineupPosition("TE", ("TE",)),
        LineupPosition("FLEX", ("RB", "WR")),
    ]
    optimizer._pass_catcher_cap = PassCatcherCapSettings(strict=False)
    rule = PassCatcherTeamRule(optimizer, variables, None)
    solver = _Solver()
    rule.apply(solver)

    assert extra_pass_catcher_allowance(optimizer.settings.positions) == 4
    named = {row[4]: row for row in solver.rows}
    assert list(named) == ["cash_pass_catcher_WAS"]
    variables_used, coefficients, sign, rhs, _name = named["cash_pass_catcher_WAS"]
    assert sign == SolverSign.LTE and rhs == 1
    assert variables_used.count("wr_flex_var") == 1
    assert coefficients.count(1.0) == 2
    assert coefficients.count(-4.0) == 1
    paired = dict(zip(variables_used, coefficients))
    assert paired == {"te_var": 1.0, "wr_flex_var": 1.0, "qb_var": -4.0}


def test_twenty_default_lineups_keep_one_pass_catcher_unless_the_qb_is_in():
    frame = _stack_pool()
    open_optimizer = _cash(frame, cap=False)
    open_lineup = open_optimizer.generate_lineups()[0]
    assert len(_buf_catchers(open_lineup)) >= 2
    assert open_lineup.loc[open_lineup["Slot"] == "QB", "Team"].iloc[0] != "BUF"
    assert pass_catcher_violations(open_lineup, strict=False)

    cash = _cash(frame, n=20)
    lineups = cash.generate_lineups()
    assert len(lineups) >= 20
    for lineup in lineups:
        assert pass_catcher_violations(lineup, strict=False) == ()
        counts = {}
        for _, row in lineup.iterrows():
            if str(row["Position"]).split("/")[0] in {"WR", "TE"}:
                counts[row["Team"]] = counts.get(row["Team"], 0) + 1
        quarterback = lineup.loc[lineup["Slot"] == "QB", "Team"].iloc[0]
        for team, count in counts.items():
            if count > 1:
                assert team == quarterback
    assert cash.last_projected_frame["Projection"].tolist() == (
        open_optimizer.last_projected_frame["Projection"].tolist()
    )


def test_twenty_strict_lineups_cap_every_team_even_when_the_qb_is_rostered():
    cash = _cash(_stack_pool(), n=20, strict=True)
    lineups = cash.generate_lineups()
    assert len(lineups) >= 20
    for lineup in lineups:
        assert pass_catcher_violations(lineup, strict=True) == ()
        quarterback = str(lineup.loc[lineup["Slot"] == "QB", "Team"].iloc[0])
        catchers = lineup[
            lineup["Position"].isin(["WR", "TE"]) & (lineup["Team"] == quarterback)
        ]
        assert len(catchers) <= 1


def test_wr_in_flex_counts_toward_the_team_cap():
    frame = _flex_wr_pool()
    open_lineup = _cash(frame, cap=False).generate_lineups()[0]
    flex = open_lineup.loc[open_lineup["Slot"] == "FLEX"].iloc[0]
    assert flex["Position"] == "WR"
    assert flex["Name"] == "Z BUF WR"
    assert flex["Team"] == "BUF"
    assert len(_buf_catchers(open_lineup)) == 2
    assert open_lineup.loc[open_lineup["Slot"] == "QB", "Team"].iloc[0] != "BUF"

    capped = _cash(frame).generate_lineups()[0]
    assert len(_buf_catchers(capped)) == 1
    assert pass_catcher_violations(capped, strict=False) == ()


def test_te_in_flex_counts_toward_the_team_cap():
    frame = _flex_te_pool()
    flex_positions = ("RB", "WR", "TE")
    open_lineup = _cash(frame, cap=False, flex=flex_positions).generate_lineups()[0]
    flex = open_lineup.loc[open_lineup["Slot"] == "FLEX"].iloc[0]
    assert flex["Position"] == "TE"
    assert flex["Team"] == "PHI"
    phi = open_lineup[(open_lineup["Team"] == "PHI") & open_lineup["Position"].isin(["WR", "TE"])]
    assert len(phi) == 2
    assert open_lineup.loc[open_lineup["Slot"] == "QB", "Team"].iloc[0] != "PHI"

    capped = _cash(frame, flex=flex_positions).generate_lineups()[0]
    phi = capped[(capped["Team"] == "PHI") & capped["Position"].isin(["WR", "TE"])]
    assert len(phi) <= 1
    assert pass_catcher_violations(capped, strict=False) == ()


def test_default_mode_allows_two_pass_catchers_with_their_quarterback():
    default = _cash(_qb_stack_pool()).generate_lineups()[0]
    assert default.loc[default["Slot"] == "QB", "Team"].iloc[0] == "BUF"
    assert len(_buf_catchers(default)) >= 2
    assert pass_catcher_violations(default, strict=False) == ()
    assert pass_catcher_violations(default, strict=True)

    strict = _cash(_qb_stack_pool(), strict=True).generate_lineups()[0]
    assert len(_buf_catchers(strict)) <= 1
    assert pass_catcher_violations(strict, strict=True) == ()


def test_infeasible_cash_lineup_names_the_rule_and_does_not_drop_it():
    optimizer = _cash(_only_buf_catchers())
    with pytest.raises(RuntimeError) as raised:
        optimizer.generate_lineups()
    message = str(raised.value)
    assert message == optimizer.last_shortfall.message
    assert "Requested 1 lineups and built 0." in message
    assert f"Binding limit: {RULE_PASS_CATCHERS}." in message
    assert "The rule was not loosened." in message
    assert pass_catcher_cause(False) in message

    opened = _cash(_only_buf_catchers(), cap=False)
    assert opened.generate_lineups()


def test_strict_mode_names_its_own_rule_when_the_quarterback_stack_cannot_split():
    optimizer = _cash(_strict_only_pool(), strict=True, flex=("RB",))
    with pytest.raises(RuntimeError) as raised:
        optimizer.generate_lineups()
    message = str(raised.value)
    assert f"Binding limit: {RULE_PASS_CATCHERS_STRICT}." in message
    assert "The rule was not loosened." in message
    assert RULE_PASS_CATCHERS not in message.replace(RULE_PASS_CATCHERS_STRICT, "")

    default = _cash(_strict_only_pool(), flex=("RB",)).generate_lineups()[0]
    assert default.loc[default["Slot"] == "QB", "Team"].iloc[0] == "BUF"
    assert len(_buf_catchers(default)) >= 2
    assert pass_catcher_violations(default, strict=False) == ()


def test_a_short_batch_names_the_rule_when_the_next_lineup_would_break_it():
    """One legal lineup exists. The only other lineup uses a second BUF wideout."""
    game = _game("BUF", "KC")
    other = _game("PHI", "DAL")
    frame = pd.DataFrame(
        [
            _row("QB", "KC QB", 1, "KC", 7000, 20, game),
            _row("RB", "SF RB", 2, "SF", 5400, 12, game),
            _row("RB", "PHI RB", 3, "PHI", 5200, 11, other),
            _row("RB", "DAL RB", 10, "DAL", 4500, 9, other),
            _row("WR", "BUF WR", 4, "BUF", 5000, 15, game),
            _row("WR", "KC WR", 5, "KC", 4800, 14, game),
            _row("WR", "PHI WR", 6, "PHI", 4700, 8, other),
            _row("WR", "BUF WR2", 7, "BUF", 4600, 19, game),
            _row("TE", "DAL TE", 8, "DAL", 4500, 10, other),
            _row("DST", "DAL DST", 9, "DAL", 2800, 6, other),
        ]
    )
    # FLEX is a running back, so the lineup takes exactly three wideouts.
    # The locked players leave one WR spot, filled by PHI WR or the second Bill.
    locks = ["1", "2", "3", "10", "4", "5", "8", "9"]
    optimizer = _cash(frame, n=2, flex=("RB",))
    optimizer.set_locked_players(locks)
    lineups = optimizer.generate_lineups()
    assert len(lineups) == 1
    assert pass_catcher_violations(lineups[0], strict=False) == ()
    assert "BUF WR2" not in set(lineups[0]["Name"])
    message = optimizer.last_shortfall.message
    assert "Requested 2 lineups and built 1." in message
    assert f"Binding limit: {RULE_PASS_CATCHERS}." in message
    assert "The rule was not loosened." in message

    opened = _cash(frame, n=2, cap=False, flex=("RB",))
    opened.set_locked_players(locks)
    opened_lineups = opened.generate_lineups()
    assert len(opened_lineups) == 2
    assert any("BUF WR2" in set(lineup["Name"]) for lineup in opened_lineups)


def test_classic_gpp_and_showdown_do_not_apply_the_cap():
    frame = _stack_pool()
    gpp = NFLClassicOptimizer()
    gpp.load_from_dataframe(frame)
    apply_preset(gpp, ContestPreset.CLASSIC_SMALL_GPP, lineup_count=1, jitter_seed=42)
    gpp.jitter_pct = (0.0, 0.0)
    gpp.exclude_vs_dst = False
    assert gpp.pass_catcher_team_cap is False
    gpp_lineup = gpp.generate_lineups()[0]
    assert len(_buf_catchers(gpp_lineup)) >= 2
    assert gpp_lineup.loc[gpp_lineup["Slot"] == "QB", "Team"].iloc[0] != "BUF"
    assert pass_catcher_violations(gpp_lineup, strict=False)

    large = get_preset_config(ContestPreset.CLASSIC_LARGE_GPP)
    assert large.pass_catcher_team_cap is False
    assert "Pass catchers" not in dict(format_preset_summary(large))

    rows = []
    game = _game("BUF", "KC")
    specs = [
        ("WR", "BUF WR", 10, "BUF", 8000, 40),
        ("WR", "BUF WR2", 20, "BUF", 7000, 35),
        ("RB", "KC RB", 30, "KC", 6000, 20),
        ("WR", "PHI WR", 40, "PHI", 5500, 18),
        ("TE", "DAL TE", 50, "DAL", 5000, 16),
        ("RB", "SF RB", 60, "SF", 4800, 14),
        ("QB", "BUF QB", 70, "BUF", 9000, 1),
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
    assert not hasattr(showdown, "pass_catcher_team_cap")
    lineup = showdown.generate_lineups()[0]
    buf = lineup[(lineup["Team"] == "BUF") & lineup["Position"].isin(["WR", "TE"])]
    assert len(buf) >= 2
    assert "BUF" not in set(lineup.loc[lineup["Position"] == "QB", "Team"])
    assert pass_catcher_violations(lineup, strict=False)


def test_seed_42_gpp_and_showdown_upload_files_match_main():
    classic = parse_dk_csv(MOCK_CLASSIC)
    showdown = parse_dk_csv(MOCK_SHOWDOWN)

    def classic_csv(preset):
        optimizer = NFLClassicOptimizer()
        optimizer.load_from_dataframe(classic)
        apply_preset(optimizer, preset, jitter_seed=42)
        return _sha(optimizer.generate_lineups(), ContestFormat.CLASSIC)

    def showdown_csv(preset):
        optimizer = NFLShowdownOptimizer()
        optimizer.load_from_dataframe(showdown)
        apply_preset(optimizer, preset, jitter_seed=42)
        return _sha(optimizer.generate_lineups(), ContestFormat.SHOWDOWN)

    assert classic_csv(ContestPreset.CLASSIC_CASH) == MAIN_CASH_CSV
    assert classic_csv(ContestPreset.CLASSIC_SMALL_GPP) == MAIN_SMALL_GPP_CSV
    assert classic_csv(ContestPreset.CLASSIC_LARGE_GPP) == MAIN_LARGE_GPP_CSV
    assert showdown_csv(ContestPreset.SHOWDOWN_CASH) == MAIN_SHOWDOWN_CASH_CSV
    assert showdown_csv(ContestPreset.SHOWDOWN_GPP) == MAIN_SHOWDOWN_GPP_CSV


def _cash_page_pool():
    """Mock Classic pool marked as the current scorer so Generate can run."""
    from src.dk_scoring import SCORING_VERSION
    from src.slate_week import slate_games_from_pool, slate_key
    from src.sleeper import MATCH_CACHE_VERSION

    pool = parse_dk_csv(MOCK_CLASSIC)
    pool["PPR"] = pd.Series(pd.NA, index=pool.index, dtype="Float64")
    pool["Scoring"] = ""
    pool["ScoringNote"] = ""
    key = slate_key(slate_games_from_pool(pool))
    pool.attrs["scoring_version"] = SCORING_VERSION
    pool.attrs["match_schema"] = MATCH_CACHE_VERSION
    pool.attrs["slate_key"] = key
    pool.attrs["fingerprint"] = f"checkbox|{key}"
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


def _pass_catcher_constraints(strict: bool) -> list[tuple]:
    """Constraints the cash rule adds during one Generate click."""
    from unittest.mock import patch

    from src.cash_rules import CONSTRAINT_PREFIX, PassCatcherTeamRule

    app_test = pytest.importorskip("streamlit.testing.v1")
    pool, summary = _cash_page_pool()
    app = app_test.AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app.py"),
        default_timeout=120,
    )
    app.session_state["contest_format"] = ContestFormat.CLASSIC.value
    app.session_state["contest_preset"] = ContestPreset.CLASSIC_CASH.value
    app.session_state["player_pool"] = pool
    app.session_state["sleeper_summary"] = summary
    app.session_state["pool_signature"] = "checkbox"
    recorded: list[tuple] = []
    real_apply = PassCatcherTeamRule.apply

    def spy(self, solver):
        original = solver.add_constraint

        def capture(*args, **kwargs):
            name = kwargs.get("name", args[4] if len(args) > 4 else "")
            coefficients = args[1] if len(args) > 1 else None
            if str(name).startswith(CONSTRAINT_PREFIX):
                spec = getattr(self.optimizer, "_pass_catcher_cap", None)
                recorded.append(
                    (
                        str(name),
                        None if coefficients is None else [float(item) for item in coefficients],
                        bool(getattr(spec, "strict", False)),
                    )
                )
            return original(*args, **kwargs)

        solver.add_constraint = capture
        return real_apply(self, solver)

    with patch("src.weather.fetch_slate_weather", return_value={}), patch.object(
        PassCatcherTeamRule, "apply", spy
    ):
        app.run()
        assert not app.exception, app.exception
        box = next(item for item in app.checkbox if item.label == "Strict one pass catcher per team")
        box.set_value(strict)
        next(item for item in app.button if item.label == "Generate Lineups").click()
        app.run()
    assert not app.exception, app.exception
    return recorded


def test_strict_cash_checkbox_drops_the_qb_exception():
    """Checked: the solver cap is one pass catcher, with no quarterback term."""
    relaxed = _pass_catcher_constraints(False)
    assert relaxed
    assert {item[2] for item in relaxed} == {False}
    assert any(
        coeff < 0
        for _name, coefficients, _strict in relaxed
        if coefficients
        for coeff in coefficients
    )

    strict = _pass_catcher_constraints(True)
    assert strict
    assert {item[2] for item in strict} == {True}
    assert all(coefficients is None for _name, coefficients, _strict in strict)
