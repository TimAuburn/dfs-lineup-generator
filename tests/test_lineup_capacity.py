"""Unique-lineup capacity. Counts below were enumerated by hand, not by the solver.

Overlap pool
------------
Roster: QB, RB, RB, WR, WR, WR, TE, FLEX, DST.
FLEX accepts RB only, so the three running backs fill RB, RB, and FLEX.
Four wide receivers compete for the three WR spots: choose 3 of 4, which is 4
lineups. Each lineup drops a different wide receiver, so every pair shares the
other 8 players.

Max repeating 8 allows a share of 8, so all 4 lineups can be used together.
Max repeating 7 allows a share of 7, so a set can hold only 1 of them. The
distinct feasible lineups are still 4. That maximum of 1 is not reported as
exact: a greedy stop is only a lower bound, and the singles count is only an
upper bound.

Cash pass-catcher pool
----------------------
One KC quarterback, PHI and DAL running backs, SEA tight end, GB defense,
and five wide receivers: BUF A, BUF B, PHI, DAL, SF.
Cash FLEX is WR or RB. Both running-back spots are filled, so FLEX is a
fourth wide receiver: choose 4 of 5, which is 5 salary-legal lineups.
The cash rule allows one WR/TE from a team unless that team's quarterback
is rostered. The quarterback is KC, so the 3 lineups that keep both BUF
wide receivers are illegal. Exactly 2 remain. No running back shares KC
with the quarterback, so the QB/RB ban removes none of those 5.

Cash QB/RB pool
---------------
One BUF quarterback, four running backs (BUF, PHI, DAL, KC), three wide
receivers on three other teams, one tight end, one defense.
Three wide receivers fill the three WR spots, so FLEX is a running back:
choose 3 of 4 running backs, which is 4 salary-legal lineups. The ban
removes every lineup that keeps the BUF running back. Exactly 1 remains.
Each of the three wide receivers is on his own team, so the WR/TE cap
removes none of the 4.
"""

from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path

import pandas as pd
from pydfs_lineup_optimizer.solvers.pulp_solver import PuLPSolver

from src.exporter import lineups_to_dk_upload_csv
import src.lineup_capacity as lineup_capacity
from src.lineup_capacity import (
    CapacityReport,
    format_capacity,
    format_warning,
    measure,
    pool_fingerprint,
)
from src.optimizer import NFLClassicOptimizer, NFLShowdownOptimizer
from src.parser import parse_dk_csv
from src.presets import ContestFormat, ContestPreset, apply_preset

MOCK_CLASSIC = Path(__file__).parent / "mock_dk_salaries.csv"
MOCK_SHOWDOWN = Path(__file__).parent / "mock_dk_showdown.csv"

MAIN_CASH_CSV = "9b1614d798c5819fcf0be8f831d38f1ee25a323aa01c12932446b471049272fd"
# Showdown FLEX columns are salary order. The previous hashes were solver order:
# cash 6980178f01545bef57fc4fdcaa43bc1b7c92459e014f779ba102aae40a437533
# GPP  f9ede315707544ed6eb97d22954b54c86259e2bda6a932c638d10d6067ecb9ff
MAIN_SHOWDOWN_CASH_CSV = "86f97c743ee8fc958a82bac7b4ccec9b519187e6dca0d3f80ed7b637030c489a"
MAIN_SMALL_GPP_CSV = "fa796590470cf9dc48fb533fd6d311e49711384a9c51be93ae2d7616481a6c6d"
MAIN_LARGE_GPP_CSV = "a39e4aea971f8aeb6b44d09f78baeeef06d4cfec480fdc5000deffca8efe1236"
MAIN_SHOWDOWN_GPP_CSV = "ca1c1d108254474e30b53e304eabb040be834045eed4e19e43c72f5030f0578b"


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


def _overlap_pool() -> pd.DataFrame:
    games = {
        "BUF": _game("BUF", "KC"),
        "KC": _game("BUF", "KC"),
        "PHI": _game("PHI", "DAL"),
        "DAL": _game("PHI", "DAL"),
        "SF": _game("SF", "SEA"),
        "SEA": _game("SF", "SEA"),
        "GB": _game("GB", "MIN"),
        "MIN": _game("GB", "MIN"),
    }
    rows = [
        _row("QB", "KC QB", 1, "KC", 6000, 20, games["KC"]),
        _row("RB", "PHI RB", 2, "PHI", 5000, 12, games["PHI"]),
        _row("RB", "DAL RB", 3, "DAL", 5000, 11, games["DAL"]),
        _row("RB", "SF RB", 4, "SF", 5000, 10, games["SF"]),
        _row("WR", "WR1", 5, "BUF", 4000, 9, games["BUF"]),
        _row("WR", "WR2", 6, "KC", 4000, 8, games["KC"]),
        _row("WR", "WR3", 7, "SEA", 4000, 7, games["SEA"]),
        _row("WR", "WR4", 8, "GB", 4000, 6, games["GB"]),
        _row("TE", "MIN TE", 9, "MIN", 4000, 6, games["MIN"]),
        _row("DST", "DAL DST", 10, "DAL", 3000, 5, games["DAL"]),
    ]
    return pd.DataFrame(rows)


def _overlap_optimizer(repeating: int) -> NFLClassicOptimizer:
    optimizer = NFLClassicOptimizer()
    optimizer.load_from_dataframe(_overlap_pool())
    optimizer.flex_positions = ("RB",)
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.jitter_seed = 42
    optimizer.set_max_repeating_players(repeating)
    optimizer.set_lineup_count(1)
    return optimizer


def _omitted_wideouts(lineup: pd.DataFrame) -> frozenset[str]:
    # pydfs prints a one-word name with a trailing space.
    names = {str(name).strip() for name in lineup["Name"].tolist()}
    return frozenset({"WR1", "WR2", "WR3", "WR4"} - names)


def _catcher_pool() -> pd.DataFrame:
    games = {
        "KC": _game("KC", "LV"),
        "LV": _game("KC", "LV"),
        "BUF": _game("BUF", "MIA"),
        "MIA": _game("BUF", "MIA"),
        "PHI": _game("PHI", "DAL"),
        "DAL": _game("PHI", "DAL"),
        "SF": _game("SF", "SEA"),
        "SEA": _game("SF", "SEA"),
        "GB": _game("GB", "MIN"),
    }
    rows = [
        _row("QB", "KC QB", 1, "KC", 7000, 22, games["KC"]),
        _row("RB", "PHI RB", 2, "PHI", 6000, 14, games["PHI"]),
        _row("RB", "DAL RB", 3, "DAL", 6000, 13, games["DAL"]),
        _row("WR", "BUF A", 4, "BUF", 5000, 16, games["BUF"]),
        _row("WR", "BUF B", 5, "BUF", 5000, 15, games["BUF"]),
        _row("WR", "PHI WR", 6, "PHI", 4800, 10, games["PHI"]),
        _row("WR", "DAL WR", 7, "DAL", 4700, 9, games["DAL"]),
        _row("WR", "SF WR", 8, "SF", 4600, 8, games["SF"]),
        _row("TE", "SEA TE", 9, "SEA", 4500, 7, games["SEA"]),
        _row("DST", "GB DST", 10, "GB", 3000, 5, games["GB"]),
    ]
    return pd.DataFrame(rows)


def _ban_pool() -> pd.DataFrame:
    games = {
        "BUF": _game("BUF", "KC"),
        "KC": _game("BUF", "KC"),
        "PHI": _game("PHI", "DAL"),
        "DAL": _game("PHI", "DAL"),
        "SF": _game("SF", "SEA"),
        "SEA": _game("SF", "SEA"),
        "GB": _game("GB", "MIN"),
        "MIN": _game("GB", "MIN"),
        "NYJ": _game("NYJ", "NE"),
    }
    rows = [
        _row("QB", "BUF QB", 1, "BUF", 8000, 24, games["BUF"]),
        _row("RB", "BUF RB", 2, "BUF", 7000, 20, games["BUF"]),
        _row("RB", "PHI RB", 3, "PHI", 6000, 12, games["PHI"]),
        _row("RB", "DAL RB", 4, "DAL", 6000, 11, games["DAL"]),
        _row("RB", "KC RB", 5, "KC", 5500, 10, games["KC"]),
        _row("WR", "PHI WR", 6, "PHI", 5000, 9, games["PHI"]),
        _row("WR", "DAL WR", 7, "DAL", 4800, 8, games["DAL"]),
        _row("WR", "SF WR", 8, "SF", 4600, 7, games["SF"]),
        _row("TE", "SEA TE", 9, "SEA", 4500, 6, games["SEA"]),
        _row("DST", "GB DST", 10, "GB", 3000, 5, games["GB"]),
    ]
    return pd.DataFrame(rows)


def _cash(frame: pd.DataFrame) -> NFLClassicOptimizer:
    optimizer = NFLClassicOptimizer()
    optimizer.load_from_dataframe(frame)
    apply_preset(optimizer, ContestPreset.CLASSIC_CASH, jitter_seed=42)
    optimizer.set_lineup_count(1)
    return optimizer


def _hand_catcher_lineups() -> list[frozenset[str]]:
    """4 of 5 wide receivers, dropping any set that keeps both BUF wideouts."""
    names = ["BUF A", "BUF B", "PHI WR", "DAL WR", "SF WR"]
    legal = []
    for chosen in combinations(names, 4):
        picked = frozenset(chosen)
        if "BUF A" in picked and "BUF B" in picked:
            continue
        legal.append(picked)
    return legal


def _hand_ban_lineups() -> list[frozenset[str]]:
    """3 of 4 running backs, dropping any set that keeps the BUF back."""
    names = ["BUF RB", "PHI RB", "DAL RB", "KC RB"]
    legal = []
    for chosen in combinations(names, 3):
        picked = frozenset(chosen)
        if "BUF RB" in picked:
            continue
        legal.append(picked)
    return legal


def _sha(lineups, contest_format: ContestFormat) -> str:
    text = lineups_to_dk_upload_csv(lineups, contest_format=contest_format)
    return hashlib.sha256(text.encode()).hexdigest()


def test_hand_counts_are_the_numbers_the_comments_describe():
    assert len(list(combinations(["WR1", "WR2", "WR3", "WR4"], 3))) == 4
    assert len(_hand_catcher_lineups()) == 2
    assert len(_hand_ban_lineups()) == 1


def test_overlap_pool_allows_exactly_four_when_share_of_eight_is_legal():
    optimizer = _overlap_optimizer(8)
    report = measure(optimizer, requested=5, budget_seconds=None)
    built = optimizer.generate_lineups(20)
    assert {_omitted_wideouts(lineup) for lineup in built} == {
        frozenset({"WR1"}),
        frozenset({"WR2"}),
        frozenset({"WR3"}),
        frozenset({"WR4"}),
    }
    assert report.exact == 4
    assert report.trivial is True
    assert format_capacity(report, 5) == (
        "Your pool and rules allow up to 4 unique lineups (requested 5)."
    )
    warning = format_warning(report, 5)
    assert warning is not None
    assert "at most 4" in warning
    assert format_warning(report, 4) is None


def test_overlap_of_seven_is_bounds_not_an_exact_maximum():
    # True maximum set size is 1. Distinct lineups are 4. Do not call 1 exact.
    optimizer = _overlap_optimizer(7)
    report = measure(optimizer, requested=5, budget_seconds=None)
    assert report.exact is None
    assert report.lower == 1
    assert report.upper == 4
    text = format_capacity(report, 5)
    assert text == (
        "Your pool and rules allow at least 1 and at most 4 unique lineups (requested 5). "
        "Limited by: overlap/min-unique rule."
    )
    assert "up to" not in text.split("Limited by:")[0]
    assert format_warning(report, 5) is not None
    assert "overlap/min-unique rule" in format_warning(report, 5)
    assert format_warning(report, 4) is None
    assert format_warning(report, 1) is None


def test_cash_pass_catcher_cap_allows_exactly_two():
    optimizer = _cash(_catcher_pool())
    report = measure(optimizer, requested=5, budget_seconds=None)
    built = optimizer.generate_lineups(20)
    chosen = []
    for lineup in built:
        wideouts = lineup.loc[lineup["Position"].astype(str).str.startswith("WR"), "Name"]
        chosen.append(frozenset(str(name).strip() for name in wideouts.tolist()))
    assert chosen_sets(chosen) == set(_hand_catcher_lineups())
    assert report.exact == 2
    assert report.binding_rules == ("pass-catcher cap",)
    notice = format_capacity(report, 5)
    assert notice == (
        "Your pool and rules allow up to 2 unique lineups (requested 5). "
        "Limited by: pass-catcher cap."
    )
    assert "QB/RB ban" not in notice
    assert format_warning(report, 5) is not None
    assert format_warning(report, 2) is None


def chosen_sets(groups) -> set:
    return set(groups)


def test_cash_qb_rb_ban_allows_exactly_one():
    optimizer = _cash(_ban_pool())
    report = measure(optimizer, requested=3, budget_seconds=None)
    built = optimizer.generate_lineups(20)
    backs = []
    for lineup in built:
        names = lineup.loc[lineup["Position"].astype(str).str.startswith("RB"), "Name"]
        backs.append(frozenset(str(name).strip() for name in names.tolist()))
    assert set(backs) == set(_hand_ban_lineups())
    assert report.exact == 1
    assert report.binding_rules == ("QB/RB ban",)
    notice = format_capacity(report, 3)
    assert notice == (
        "Your pool and rules allow up to 1 unique lineup (requested 3). "
        "Limited by: QB/RB ban."
    )
    assert "pass-catcher cap" not in notice
    warning = format_warning(report, 3)
    assert warning is not None
    assert "at most 1 unique lineup" in warning


def test_binding_rule_is_undetermined_when_the_attribution_budget_is_gone():
    optimizer = _cash(_ban_pool())
    report = measure(
        optimizer,
        requested=3,
        budget_seconds=None,
        attribution_budget_seconds=0,
    )
    assert report.exact == 1
    assert report.binding_rules == ()
    assert report.binding_undetermined is True
    text = format_capacity(report, 3)
    assert "The binding rule could not be determined in time." in text
    assert "Limited by:" not in text


def test_cap_above_150_is_clamped_to_150():
    """The enumeration cap cannot be raised. 150 is the hard maximum."""
    optimizer = _overlap_optimizer(8)
    report = measure(optimizer, requested=1, cap=400, budget_seconds=0)
    assert report.cap == 150


def test_cap_hit_is_labeled_with_a_plus_and_is_not_a_ceiling():
    optimizer = _overlap_optimizer(8)
    report = measure(optimizer, requested=1, cap=2, budget_seconds=None)
    assert report.exact is None
    assert report.capped is True
    assert report.cap == 2
    assert format_capacity(report, 1) == (
        "Your pool and rules allow 2+ unique lineups (requested 1)."
    )
    assert format_warning(report, 9) is None
    custom = CapacityReport(capped=True, cap=12, lower=12)
    assert format_capacity(custom, 3) == (
        "Your pool and rules allow 12+ unique lineups (requested 3)."
    )


def test_lower_and_upper_labels_and_the_warning_need_a_proof():
    lower_only = CapacityReport(lower=4, timed_out=True, cap=150)
    text = format_capacity(lower_only, 20)
    assert "at least 4 unique lineups" in text
    assert "time limit" in text
    assert "up to" not in text
    assert format_warning(lower_only, 20) is None

    upper_only = CapacityReport(lower=0, upper=3, upper_exhausted=True, cap=150)
    assert format_capacity(upper_only, 5) == (
        "Your pool and rules allow at most 3 unique lineups (requested 5)."
    )
    assert "at most 3" in format_warning(upper_only, 5)

    empty = CapacityReport(lower=0, timed_out=True, cap=150)
    assert format_capacity(empty, 5) == (
        "The unique-lineup count hit the time limit before a lineup was proved "
        "(requested 5)."
    )
    assert format_warning(empty, 5) is None


def test_zero_budget_does_not_invent_an_exact_count():
    optimizer = NFLClassicOptimizer()
    optimizer.load_from_dataframe(parse_dk_csv(MOCK_CLASSIC))
    apply_preset(optimizer, ContestPreset.CLASSIC_CASH, jitter_seed=42)
    report = measure(optimizer, requested=1, budget_seconds=0)
    assert report.exact is None
    assert report.timed_out is True
    text = format_capacity(report, 1)
    assert "time limit" in text
    assert "up to" not in text


def test_larger_request_reuses_the_count_and_names_the_overlap_rule(monkeypatch):
    """M=4 caches the ceiling. M=5 must name overlap without counting again.

    The ceiling is 4, so attribution is skipped at M=4 and required at M=5.
    The cache key does not include M. The second call still has to attribute.
    """
    calls = []
    real_count = lineup_capacity._count_lineups

    def _record(optimizer, **kwargs):
        calls.append(optimizer.max_repeating_players)
        return real_count(optimizer, **kwargs)

    monkeypatch.setattr(lineup_capacity, "_count_lineups", _record)
    optimizer = _overlap_optimizer(7)
    cache: dict = {}
    first = measure(optimizer, requested=4, budget_seconds=None, cache=cache)
    assert first.exact is None
    assert first.upper == 4
    assert "Limited by:" not in format_capacity(first, 4)
    assert calls == [7]

    second = measure(optimizer, requested=5, budget_seconds=None, cache=cache)
    assert "overlap/min-unique rule" in format_capacity(second, 5)
    assert "overlap/min-unique rule" in format_warning(second, 5)
    # The original count is not solved again. The only new solve clears overlap.
    assert calls == [7, None]
    assert second.from_cache is False

    third = measure(optimizer, requested=5, budget_seconds=None, cache=cache)
    assert third.from_cache is True
    assert "overlap/min-unique rule" in format_capacity(third, 5)
    assert calls == [7, None]


def test_same_request_is_a_cache_hit_without_another_count(monkeypatch):
    calls = []
    real_count = lineup_capacity._count_lineups

    def _record(optimizer, **kwargs):
        calls.append(optimizer.max_repeating_players)
        return real_count(optimizer, **kwargs)

    monkeypatch.setattr(lineup_capacity, "_count_lineups", _record)
    optimizer = _overlap_optimizer(7)
    cache: dict = {}
    first = measure(optimizer, requested=4, budget_seconds=None, cache=cache)
    second = measure(optimizer, requested=4, budget_seconds=None, cache=cache)
    assert second.from_cache is True
    assert second.lower == first.lower
    assert second.upper == first.upper
    assert calls == [7]
    assert "Limited by:" not in format_capacity(second, 4)


def test_cache_repeats_a_proved_count_and_misses_a_different_pool():
    optimizer = _overlap_optimizer(8)
    cache: dict = {}
    first = measure(optimizer, requested=5, budget_seconds=None, cache=cache)
    second = measure(optimizer, requested=9, budget_seconds=None, cache=cache)
    assert first.exact == 4
    assert second.from_cache is True
    assert second.exact == 4
    assert "requested 9" in format_capacity(second, 9)

    changed = _overlap_pool()
    changed.loc[changed["Name"] == "WR1", "Salary"] = 9000
    other = NFLClassicOptimizer()
    other.load_from_dataframe(changed)
    other.flex_positions = ("RB",)
    other.jitter_pct = (0.0, 0.0)
    other.jitter_seed = 42
    other.set_max_repeating_players(8)
    missed = measure(other, requested=5, budget_seconds=0, cache=cache)
    assert missed.from_cache is False
    assert missed.exact is None
    assert pool_fingerprint(optimizer._player_df) != pool_fingerprint(other._player_df)


def test_measure_restores_solver_limit_and_seed_42_upload_bytes():
    before = PuLPSolver.LP_SOLVER.timeLimit
    classic = parse_dk_csv(MOCK_CLASSIC)
    showdown = parse_dk_csv(MOCK_SHOWDOWN)

    def classic_csv(preset):
        optimizer = NFLClassicOptimizer()
        optimizer.load_from_dataframe(classic)
        apply_preset(optimizer, preset, jitter_seed=42)
        measure(optimizer, requested=1, budget_seconds=0.05)
        return _sha(optimizer.generate_lineups(), ContestFormat.CLASSIC)

    def showdown_csv(preset):
        optimizer = NFLShowdownOptimizer()
        optimizer.load_from_dataframe(showdown)
        apply_preset(optimizer, preset, jitter_seed=42)
        measure(optimizer, requested=1, budget_seconds=0.05)
        return _sha(optimizer.generate_lineups(), ContestFormat.SHOWDOWN)

    assert classic_csv(ContestPreset.CLASSIC_CASH) == MAIN_CASH_CSV
    assert classic_csv(ContestPreset.CLASSIC_SMALL_GPP) == MAIN_SMALL_GPP_CSV
    assert classic_csv(ContestPreset.CLASSIC_LARGE_GPP) == MAIN_LARGE_GPP_CSV
    assert showdown_csv(ContestPreset.SHOWDOWN_CASH) == MAIN_SHOWDOWN_CASH_CSV
    assert showdown_csv(ContestPreset.SHOWDOWN_GPP) == MAIN_SHOWDOWN_GPP_CSV
    assert PuLPSolver.LP_SOLVER.timeLimit == before
