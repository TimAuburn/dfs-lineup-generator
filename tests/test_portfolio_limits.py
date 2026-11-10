"""Portfolio overlap and exposure hold on the exported set.

These checks fail on the pre-fix solver: tiered Classic batches did not see
each other, and Showdown captain and flex rows each received their own cap.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from src.exporter import SHOWDOWN_UPLOAD_COLUMNS, lineups_to_dk_upload_csv
from src.optimizer import (
    GameScriptBucket,
    NFLClassicOptimizer,
    NFLShowdownOptimizer,
    exposure_report_csv,
    optimize_game_scripts,
    optimize_tiered_salaries,
    player_exposure_table,
)
from src.parser import parse_dk_csv
from src.player_pool import person_key as pool_person_key
from src.portfolio import (
    appearances_allowed,
    appearances_needed,
    classic_roster_slots,
    exposure_counts,
    min_exposure_conflicts,
    pairwise_overlap,
    person_key,
    team_stack_counts,
)
from src.presets import ContestPreset, apply_preset

MOCK_CLASSIC = Path(__file__).parent / "mock_dk_salaries.csv"
MOCK_SHOWDOWN = Path(__file__).parent / "mock_dk_showdown.csv"


def test_person_key_matches_the_player_pool():
    assert person_key("Josh Allen", " buf ") == pool_person_key("Josh Allen", "buf")
    assert person_key("Josh Allen", "BUF") == ("josh allen", "BUF")


@pytest.mark.parametrize(
    ("rate", "total", "allowed"),
    [
        (0.5, 4, 2),
        (0.4, 6, 2),
        (0.25, 4, 1),
        (0.5, 1, 0),
        (0.1, 4, 0),
        (0.3, 20, 6),
        (0.0, 4, 0),
        (None, 4, 4),
    ],
)
def test_appearances_allowed_is_floor_of_rate_times_requested(rate, total, allowed):
    assert appearances_allowed(rate, total) == allowed
    if rate is not None and rate > 0:
        assert allowed == math.floor(float(rate) * total + 1e-9)


def _assert_limits(lineups, *, requested, overlap, exposure, stack_rate=None, stack_spec=()):
    assert lineups
    if overlap is not None:
        assert pairwise_overlap(lineups) <= int(overlap)
    if exposure is not None:
        allowed = math.floor(float(exposure) * requested + 1e-9)
        assert allowed == appearances_allowed(float(exposure), requested)
        assert all(count <= allowed for count in exposure_counts(lineups).values())
    if stack_rate is not None:
        allowed = math.floor(float(stack_rate) * requested + 1e-9)
        assert allowed == appearances_allowed(float(stack_rate), requested)
        usage = team_stack_counts(lineups, stack_spec)
        assert all(count <= allowed for count in usage.values())


def test_classic_tiered_max_overlap_holds_for_every_pair():
    """Max overlap 4 across salary tiers. On main, a pair shared 7 players."""
    optimizer = NFLClassicOptimizer()
    optimizer.load_from_csv(MOCK_CLASSIC)
    optimizer.set_qb_stack(enabled=False)
    optimizer.set_max_repeating_players(4)
    optimizer.jitter_pct = (0.0, 0.0)
    lineups = optimize_tiered_salaries(
        optimizer,
        total_n=8,
        leave_money_ratio=0.5,
        standard_cap=50_000,
        discount_cap=46_000,
        discount_floor=40_000,
    )
    assert len(lineups) == 8
    assert optimizer.last_shortfall is None
    salaries = [int(lineup.attrs["salary"]) for lineup in lineups]
    assert all(49_800 <= salary <= 50_000 for salary in salaries[:4])
    assert all(40_000 <= salary <= 46_000 for salary in salaries[4:])
    _assert_limits(lineups, requested=8, overlap=4, exposure=None)
    assert pairwise_overlap(lineups) <= 4


def test_showdown_exposure_combines_captain_and_flex_for_josh_allen():
    """50% of 4 is 2. On main, Allen was captain twice and flex twice."""
    pool = parse_dk_csv(MOCK_SHOWDOWN)
    pool.loc[pool["Name"] == "Josh Allen", "AvgPointsPerGame"] = 40
    pool.loc[pool["Name"] == "Josh Allen", "Projection"] = 40
    optimizer = NFLShowdownOptimizer()
    optimizer.load_from_dataframe(pool)
    apply_preset(optimizer, ContestPreset.SHOWDOWN_GPP, lineup_count=4, jitter_seed=1)
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.max_repeating_players = None
    optimizer.set_exposure_limits(max_exposure_pct=50)
    lineups = optimizer.generate_lineups(4)
    assert len(lineups) == 4
    assert optimizer.last_shortfall is None
    counts = exposure_counts(lineups)
    assert counts[person_key("Josh Allen", "BUF")] <= 2
    _assert_limits(lineups, requested=4, overlap=None, exposure=0.5)

    table = player_exposure_table(lineups, requested=4)
    allen = table.loc[table["Name"].eq("Josh Allen")].iloc[0]
    assert int(allen["Lineups"]) == counts[person_key("Josh Allen", "BUF")]
    assert int(allen["CPT count"]) + int(allen["FLEX count"]) == int(allen["Lineups"])
    assert list(table["Name"]).count("Josh Allen") == 1
    assert float(allen["Exported %"]) == pytest.approx(int(allen["Lineups"]) / 4 * 100, abs=0.05)
    assert float(allen["Requested %"]) == pytest.approx(int(allen["Lineups"]) / 4 * 100, abs=0.05)

    upload = lineups_to_dk_upload_csv(lineups, contest_format="showdown")
    assert upload.splitlines()[0].split(",") == SHOWDOWN_UPLOAD_COLUMNS
    report = exposure_report_csv(lineups)
    assert report.splitlines()[0].startswith("Name,Team,Position,Lineups")
    assert "Josh Allen" in report
    assert "Exposure" not in upload


def test_forty_percent_of_six_allows_at_most_two_on_the_single_batch_path():
    """40% of 6 is 2. pydfs rounding would allow a third lineup."""
    pool = parse_dk_csv(MOCK_CLASSIC)
    pool.loc[pool["Name"].eq("Patrick Mahomes"), ["Projection", "AvgPointsPerGame"]] = 80
    optimizer = NFLClassicOptimizer()
    optimizer.load_from_dataframe(pool)
    optimizer.set_qb_stack(enabled=False)
    optimizer.set_max_repeating_players(None)
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.set_exposure_limits(max_exposure_pct=40)
    lineups = optimizer.generate_lineups(6)
    assert len(lineups) == 6
    assert optimizer.last_shortfall is None
    counts = exposure_counts(lineups)
    assert counts[person_key("Patrick Mahomes", "KC")] == 2
    assert all(count <= math.floor(0.40 * 6 + 1e-9) for count in counts.values())


def test_forty_percent_of_six_holds_across_salary_tiers():
    pool = parse_dk_csv(MOCK_CLASSIC)
    pool.loc[pool["Name"].eq("Patrick Mahomes"), ["Projection", "AvgPointsPerGame"]] = 80
    optimizer = NFLClassicOptimizer()
    optimizer.load_from_dataframe(pool)
    optimizer.set_qb_stack(enabled=False)
    optimizer.set_max_repeating_players(None)
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.set_exposure_limits(max_exposure_pct=40)
    lineups = optimize_tiered_salaries(
        optimizer,
        total_n=6,
        leave_money_ratio=0.4,
        standard_cap=50_000,
        discount_cap=46_000,
        discount_floor=40_000,
    )
    assert len(lineups) == 6
    counts = exposure_counts(lineups)
    assert counts[person_key("Patrick Mahomes", "KC")] == 2
    assert all(count <= 2 for count in counts.values())


def test_positive_exposure_below_one_lineup_allows_zero_and_names_the_cap():
    optimizer = NFLClassicOptimizer()
    optimizer.load_from_csv(MOCK_CLASSIC)
    optimizer.set_qb_stack(enabled=False)
    optimizer.set_max_repeating_players(None)
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.set_exposure_limits(max_exposure_pct=40)
    with pytest.raises(RuntimeError, match="0 appearances"):
        optimizer.generate_lineups(2)
    shortfall = optimizer.last_shortfall
    assert shortfall is not None
    assert shortfall.built == 0
    assert shortfall.requested == 2
    assert "40%" in shortfall.message
    assert "below one lineup" in shortfall.message


def test_infeasible_overlap_returns_fewer_lineups_and_names_the_limit():
    rows = []
    roster = (
        ("QB", "Only QB", "KC", 7000, 20),
        ("RB", "Only RB", "KC", 6000, 15),
        ("RB", "Second RB", "BUF", 5500, 12),
        ("WR", "WR One", "KC", 6500, 14),
        ("WR", "WR Two", "KC", 5000, 11),
        ("WR", "WR Three", "BUF", 4800, 10),
        ("TE", "Only TE", "KC", 4500, 9),
        ("RB", "Flex RB", "BUF", 4000, 8),
        ("DST", "Chiefs", "KC", 3000, 7),
    )
    for index, (pos, name, team, salary, points) in enumerate(roster, start=1):
        rows.append(
            {
                "Position": pos,
                "Name + ID": f"{name} ({index})",
                "Name": name,
                "ID": str(index),
                "Roster Position": pos,
                "Salary": salary,
                "Game Info": "KC@BUF 09/10/2025 08:20PM ET",
                "TeamAbbrev": team,
                "AvgPointsPerGame": points,
                "Projection": points,
            }
        )
    optimizer = NFLClassicOptimizer()
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.set_qb_stack(enabled=False)
    optimizer.load_from_dataframe(pd.DataFrame(rows))
    optimizer.set_max_repeating_players(0)
    lineups = optimizer.generate_lineups(2)
    assert len(lineups) == 1
    shortfall = optimizer.last_shortfall
    assert shortfall is not None
    assert shortfall.requested == 2
    assert shortfall.built == 1
    assert "Max overlap 0" in shortfall.message
    assert "Binding limit" in shortfall.message
    assert "Requested 2 lineups and built 1." in shortfall.message
    # The solver names symptoms. Those names are not the binding limit.
    assert "not a proven cause" in shortfall.message
    assert pairwise_overlap(lineups) <= 0 or len(lineups) == 1


@pytest.mark.parametrize("seed", [1, 7, 42])
@pytest.mark.parametrize(
    ("count", "overlap", "exposure", "tiered"),
    [
        (6, 4, 0.50, True),
        (4, 3, 0.40, False),
        (5, 5, None, True),
    ],
)
def test_classic_portfolio_pairs_and_exposure_across_seeds(seed, count, overlap, exposure, tiered):
    optimizer = NFLClassicOptimizer()
    optimizer.load_from_csv(MOCK_CLASSIC)
    optimizer.set_qb_stack(enabled=False)
    optimizer.set_max_repeating_players(overlap)
    optimizer.jitter_pct = (0.04, 0.08)
    optimizer.jitter_seed = seed
    if exposure is not None:
        optimizer.set_exposure_limits(max_exposure_pct=exposure * 100)
    if tiered:
        lineups = optimize_tiered_salaries(
            optimizer,
            total_n=count,
            leave_money_ratio=0.4,
            standard_cap=50_000,
            discount_cap=46_000,
            discount_floor=40_000,
        )
    else:
        lineups = optimizer.generate_lineups(count)
    if len(lineups) < count:
        assert optimizer.last_shortfall is not None
        assert optimizer.last_shortfall.requested == count
        assert "Requested" in optimizer.last_shortfall.message
    _assert_limits(lineups, requested=count, overlap=overlap, exposure=exposure)


@pytest.mark.parametrize("seed", [1, 11, 42])
@pytest.mark.parametrize(
    ("exposure_pct", "overlap"),
    [
        (50, None),
        (50, 3),
        (25, 4),
    ],
)
def test_showdown_portfolio_identity_exposure_across_seeds(seed, exposure_pct, overlap):
    pool = parse_dk_csv(MOCK_SHOWDOWN)
    pool.loc[pool["Name"] == "Josh Allen", ["AvgPointsPerGame", "Projection"]] = 40
    optimizer = NFLShowdownOptimizer()
    optimizer.load_from_dataframe(pool)
    optimizer.jitter_pct = (0.05, 0.10)
    optimizer.jitter_seed = seed
    optimizer.max_repeating_players = overlap
    optimizer.configure_showdown_rules(
        min_salary_cap=20_000,
        max_salary_cap=50_000,
        min_teams=2,
        game_script_correlation=False,
        max_kickers=None,
        max_k_dst_combined=None,
    )
    optimizer.set_exposure_limits(max_exposure_pct=exposure_pct)
    requested = 4
    lineups = optimizer.generate_lineups(requested)
    if len(lineups) < requested:
        assert optimizer.last_shortfall is not None
        assert "exposure" in optimizer.last_shortfall.message.lower()
    _assert_limits(
        lineups,
        requested=requested,
        overlap=overlap,
        exposure=exposure_pct / 100.0,
    )
    allen = exposure_counts(lineups)[person_key("Josh Allen", "BUF")]
    assert allen <= appearances_allowed(exposure_pct / 100.0, requested)
    upload = lineups_to_dk_upload_csv(lineups, contest_format="showdown")
    assert upload.splitlines()[0].split(",") == SHOWDOWN_UPLOAD_COLUMNS


@pytest.mark.parametrize("seed", [2, 9])
def test_team_stack_exposure_holds_across_salary_tiers(seed):
    optimizer = NFLClassicOptimizer()
    optimizer.load_from_csv(MOCK_CLASSIC)
    optimizer.set_qb_stack(enabled=True, pass_catchers=1, bring_back=False)
    optimizer.slate_aware_gpp = True
    optimizer.set_max_repeating_players(7)
    optimizer.jitter_pct = (0.05, 0.10)
    optimizer.jitter_seed = seed
    requested = 6
    lineups = optimize_tiered_salaries(
        optimizer,
        total_n=requested,
        leave_money_ratio=0.5,
        standard_cap=50_000,
        discount_cap=47_000,
        discount_floor=35_000,
    )
    assert optimizer.last_stack_exposure is not None
    spec = (("QB",), ("WR", "TE"))
    if len(lineups) < requested:
        assert optimizer.last_shortfall is not None
    _assert_limits(
        lineups,
        requested=requested,
        overlap=7,
        exposure=None,
        stack_rate=optimizer.last_stack_exposure,
        stack_spec=spec,
    )


def _salary_row(pos, name, team, salary, points, pid, game="KC@BUF 09/10/2025 08:20PM ET"):
    roster = pos if pos in {"QB", "TE", "DST"} else f"{pos}/FLEX"
    return {
        "Position": pos,
        "Name + ID": f"{name} ({pid})",
        "Name": name,
        "ID": str(pid),
        "Roster Position": roster,
        "Salary": salary,
        "Game Info": game,
        "TeamAbbrev": team,
        "AvgPointsPerGame": points,
        "Projection": points,
    }


def _min_shortfall_pool() -> pd.DataFrame:
    """One cheap receiver is outside the natural top lineups.

    50% of 5 lineups needs 3 appearances. pydfs rounds 2.5 to 2 inside one
    batch, and a 4+1 salary split rounds to 2 + 0. The receiver finishes at
    2 on that per-batch rule.
    """
    skill = 6100
    rows = [
        _salary_row("QB", "Only QB", "KC", 5400, 40, 1),
        _salary_row("RB", "RB A", "KC", skill, 30, 2),
        _salary_row("RB", "RB B", "BUF", skill, 29, 3),
        _salary_row("RB", "RB C", "KC", skill, 28, 4),
        _salary_row("WR", "WR A", "KC", skill, 27, 5),
        _salary_row("WR", "WR B", "BUF", skill, 26, 6),
        _salary_row("WR", "WR C", "KC", skill, 25, 7),
        _salary_row("WR", "WR D", "BUF", skill, 24, 8),
        _salary_row("WR", "Punt WR", "BUF", skill, 1, 9),
        _salary_row("TE", "Only TE", "KC", 4500, 15, 10),
        _salary_row("DST", "Chiefs", "KC", 3500, 10, 11),
    ]
    return pd.DataFrame(rows)


def _classic_for_min(pool, *, minimum=50, maximum=None, overlap=None, seed=1, jitter=(0.0, 0.0)):
    optimizer = NFLClassicOptimizer()
    optimizer.jitter_pct = jitter
    optimizer.jitter_seed = seed
    optimizer.set_qb_stack(enabled=False)
    optimizer.set_max_repeating_players(overlap)
    optimizer.load_from_dataframe(pool)
    optimizer.set_exposure_limits(
        min_exposure_pct=minimum,
        max_exposure_pct=maximum,
    )
    return optimizer


def _assert_min_met(lineups, *, rate, requested):
    needed = appearances_needed(rate, requested)
    counts = exposure_counts(lineups)
    short = sorted(name for (name, _), count in counts.items() if count < needed)
    assert not short, f"under the minimum of {needed}: {short}"
    assert all(count >= needed for count in counts.values())


@pytest.mark.parametrize(
    ("rate", "total", "needed"),
    [
        (0.5, 5, 3),
        (0.5, 4, 2),
        (0.4, 3, 2),
        (0.3, 4, 2),
        (0.1, 4, 1),
        (1.0, 4, 4),
        (0.0, 4, 0),
        (None, 4, 0),
    ],
)
def test_appearances_needed_is_ceil_of_rate_times_requested(rate, total, needed):
    assert appearances_needed(rate, total) == needed
    if rate is not None and rate > 0:
        assert needed == math.ceil(float(rate) * total - 1e-9)


def test_single_batch_min_meets_the_ceiling_pydfs_would_round_down():
    """50% of 5 is 3. Per-batch round(2.5) is 2, so Punt WR finished at 2."""
    optimizer = _classic_for_min(_min_shortfall_pool(), minimum=50)
    lineups = optimizer.generate_lineups(5)
    assert len(lineups) == 5
    assert optimizer.last_shortfall is None
    assert optimizer.last_min_report is not None
    assert optimizer.last_min_report.message == ""
    _assert_min_met(lineups, rate=0.5, requested=5)
    assert exposure_counts(lineups)[person_key("Punt WR", "BUF")] >= 3
    table = player_exposure_table(
        lineups,
        requested=5,
        min_players=list(optimizer.last_min_report.rows),
    )
    punt = table.loc[table["Name"].eq("Punt WR")].iloc[0]
    assert int(punt["Lineups"]) >= int(punt["Min needed"])
    assert str(punt["Min met"]).startswith("Yes")
    assert "3 of 3" in str(punt["Min met"]) or f"{int(punt['Lineups'])} of 3" in str(punt["Min met"])
    upload = lineups_to_dk_upload_csv(lineups)
    assert upload.splitlines()[0].split(",") == [
        "QB",
        "RB",
        "RB",
        "WR",
        "WR",
        "WR",
        "TE",
        "FLEX",
        "DST",
    ]


def test_tiered_min_meets_the_ceiling_across_batches():
    """A 4+1 split rounds 50% to 2 and then 0. The portfolio still needs 3."""
    optimizer = _classic_for_min(_min_shortfall_pool(), minimum=50)
    lineups = optimize_tiered_salaries(
        optimizer,
        total_n=5,
        leave_money_ratio=0.2,
        standard_cap=50_000,
        discount_cap=50_000,
        discount_floor=40_000,
    )
    assert len(lineups) == 5
    assert all(int(lineup.attrs["salary"]) == 50_000 for lineup in lineups)
    _assert_min_met(lineups, rate=0.5, requested=5)
    assert exposure_counts(lineups)[person_key("Punt WR", "BUF")] >= 3
    assert optimizer.last_min_report is not None
    assert optimizer.last_min_report.unmet == ()


def test_showdown_min_counts_captain_and_flex_once():
    """Captain and flex are one person. 50% of 5 is 3, not 2 per row."""
    rows = []
    pid = 1
    for pos, name, team, flex_salary, points in (
        ("QB", "QB KC", "KC", 8000, 30),
        ("QB", "QB BUF", "BUF", 7800, 28),
        ("RB", "RB KC", "KC", 6000, 20),
        ("RB", "RB BUF", "BUF", 5800, 18),
        ("WR", "WR KC", "KC", 5600, 16),
        ("WR", "WR BUF", "BUF", 5400, 15),
        ("WR", "WR Low", "BUF", 4000, 4),
        ("TE", "TE KC", "KC", 4200, 10),
        ("TE", "TE BUF", "BUF", 3800, 9),
    ):
        for slot, salary, player_id in (
            ("CPT", int(flex_salary * 1.5), pid),
            ("FLEX", flex_salary, pid + 1),
        ):
            rows.append(
                {
                    "Position": pos,
                    "Name + ID": f"{name} ({player_id})",
                    "Name": name,
                    "ID": str(player_id),
                    "Roster Position": slot,
                    "Salary": salary,
                    "Game Info": "KC@BUF 09/10/2025 08:20PM ET",
                    "TeamAbbrev": team,
                    "AvgPointsPerGame": points,
                    "Projection": points,
                }
            )
        pid += 2
    optimizer = NFLShowdownOptimizer()
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.load_from_dataframe(pd.DataFrame(rows))
    optimizer.configure_showdown_rules(
        min_salary_cap=20_000,
        max_salary_cap=50_000,
        min_teams=2,
        game_script_correlation=False,
        max_kickers=None,
        max_k_dst_combined=None,
    )
    optimizer.max_repeating_players = None
    optimizer.set_exposure_limits(min_exposure_pct=50)
    lineups = optimizer.generate_lineups(5)
    assert len(lineups) == 5
    _assert_min_met(lineups, rate=0.5, requested=5)
    table = player_exposure_table(
        lineups,
        requested=5,
        min_players=list(optimizer.last_min_report.rows),
    )
    low = table.loc[table["Name"].eq("WR Low")].iloc[0]
    assert int(low["CPT count"]) + int(low["FLEX count"]) == int(low["Lineups"])
    assert int(low["Lineups"]) >= 3
    assert str(low["Min met"]).startswith("Yes")
    upload = lineups_to_dk_upload_csv(lineups, contest_format="showdown")
    assert upload.splitlines()[0].split(",") == SHOWDOWN_UPLOAD_COLUMNS


def test_min_above_max_is_named_and_the_max_still_holds():
    optimizer = _classic_for_min(_min_shortfall_pool(), minimum=50, maximum=50)
    lineups = optimizer.generate_lineups(5)
    assert lineups
    allowed = appearances_allowed(0.5, 5)
    assert allowed == 2
    assert appearances_needed(0.5, 5) == 3
    assert all(count <= allowed for count in exposure_counts(lineups).values())
    report = optimizer.last_min_report
    assert report is not None
    assert "needs 3" in report.message
    assert "allows 2" in report.message
    assert report.unmet
    upload = lineups_to_dk_upload_csv(lineups)
    assert upload.splitlines()[0].startswith("QB,RB,RB,WR,WR,WR,TE,FLEX,DST")


def test_min_sum_past_roster_slots_is_named_and_lineups_still_build():
    optimizer = _classic_for_min(_min_shortfall_pool(), minimum=100)
    lineups = optimizer.generate_lineups(5)
    assert len(lineups) == 5
    report = optimizer.last_min_report
    assert report is not None
    assert "roster slots" in report.message
    assert "55" in report.message
    assert "45" in report.message
    assert report.unmet
    # The upload file does not grow a minimum column.
    assert "Min" not in lineups_to_dk_upload_csv(lineups).splitlines()[0]


def test_qb_minimums_past_the_qb_slot_are_named():
    rows = []
    pid = 1
    for name, team, points in (("QB A", "KC", 30), ("QB B", "BUF", 28), ("QB C", "DAL", 26)):
        game = {
            "KC": "KC@BUF 09/10/2025 08:20PM ET",
            "BUF": "KC@BUF 09/10/2025 08:20PM ET",
            "DAL": "DAL@NYG 09/10/2025 01:00PM ET",
        }[team]
        rows.append(_salary_row("QB", name, team, 6000, points, pid, game=game))
        pid += 1
    for pos, name, team, points in (
        ("RB", "RB A", "KC", 20),
        ("RB", "RB B", "BUF", 18),
        ("WR", "WR A", "KC", 16),
        ("WR", "WR B", "BUF", 15),
        ("WR", "WR C", "KC", 14),
        ("TE", "TE A", "KC", 12),
        ("DST", "Chiefs", "KC", 8),
    ):
        rows.append(
            _salary_row(pos, name, team, 5000, points, pid, game="KC@BUF 09/10/2025 08:20PM ET")
        )
        pid += 1
    # DAL skill so the third quarterback's game can fill a lineup on its own.
    for pos, name, points in (
        ("RB", "DAL RB", 11),
        ("WR", "DAL WR", 10),
        ("TE", "DAL TE", 9),
        ("DST", "Dallas", 7),
    ):
        rows.append(
            _salary_row(pos, name, "DAL", 4500, points, pid, game="DAL@NYG 09/10/2025 01:00PM ET")
        )
        pid += 1
    needed = {
        person_key(name, team): 2
        for name, team in (("QB A", "KC"), ("QB B", "BUF"), ("QB C", "DAL"))
    }
    conflicts = min_exposure_conflicts(
        needed=needed,
        allowed={},
        positions={key: frozenset({"QB"}) for key in needed},
        slots=classic_roster_slots(),
        requested=2,
        names={key: key[0] for key in needed},
    )
    assert any("QB" in message and "6" in message for message in conflicts)
    optimizer = _classic_for_min(pd.DataFrame(rows), minimum=100)
    lineups = optimizer.generate_lineups(2)
    assert lineups
    assert optimizer.last_min_report is not None
    assert "QB" in optimizer.last_min_report.message


def test_no_min_leaves_the_exposure_table_unchanged():
    optimizer = NFLClassicOptimizer()
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.set_qb_stack(enabled=False)
    optimizer.load_from_dataframe(_min_shortfall_pool())
    lineups = optimizer.generate_lineups(3)
    assert optimizer.last_min_report is None
    table = player_exposure_table(lineups, requested=3)
    assert list(table.columns) == [
        "Name",
        "Team",
        "Position",
        "Lineups",
        "Exported %",
        "Requested %",
    ]


@pytest.mark.parametrize("seed", [1, 7, 42, 99])
def test_min_exposure_holds_with_max_and_overlap_across_seeds(seed):
    """Jitter does not drop a feasible minimum, and the max and overlap still hold.

    The cap is 100% because this pool has one quarterback. A lower cap cannot
    fill every lineup, which is the separate minimum-above-maximum case.
    """
    requested = 5
    optimizer = _classic_for_min(
        _min_shortfall_pool(),
        minimum=50,
        maximum=100,
        overlap=8,
        seed=seed,
        jitter=(0.05, 0.10),
    )
    lineups = optimizer.generate_lineups(requested)
    assert len(lineups) == requested
    _assert_limits(lineups, requested=requested, overlap=8, exposure=1.0)
    report = optimizer.last_min_report
    assert report is not None
    assert report.conflicts == ()
    assert report.unmet == ()
    _assert_min_met(lineups, rate=0.5, requested=requested)


def test_game_script_buckets_use_the_same_portfolio_minimum():
    """Each bucket is its own solve. The minimum is still on all four lineups."""
    game_a = "DET@ARI 10/11/2026 01:00PM ET"
    game_b = "SF@SEA 10/11/2026 04:25PM ET"
    rows = []
    pid = 1
    for team, game, base in (
        ("DET", game_a, 30),
        ("ARI", game_a, 24),
        ("SF", game_b, 14),
        ("SEA", game_b, 12),
    ):
        for pos, salary, bump in (
            ("QB", 6200, 8),
            ("RB", 5400, 4),
            ("RB", 4800, 2),
            ("WR", 5600, 5),
            ("WR", 5000, 3),
            ("WR", 4400, 1),
            ("TE", 4200, 0),
            ("DST", 2800, -2),
        ):
            rows.append(
                _salary_row(pos, f"{team} {pos} {pid}", team, salary, base + bump, pid, game=game)
            )
            pid += 1
    optimizer = NFLClassicOptimizer()
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.load_from_dataframe(pd.DataFrame(rows))
    optimizer.set_exposure_limits(min_exposure_pct=25)
    requested = 4
    lineups = optimize_game_scripts(
        optimizer,
        [
            GameScriptBucket("DET@ARI", 3, 50_000),
            GameScriptBucket("SF@SEA", 1, 50_000),
        ],
        max_repeating_players=6,
    )
    assert lineups
    assert pairwise_overlap(lineups) <= 6
    needed = appearances_needed(0.25, requested)
    assert needed == 1
    report = optimizer.last_min_report
    assert report is not None
    counts = exposure_counts(lineups)
    assert any(count >= needed for count in counts.values())
    for identity, count in counts.items():
        if count < needed:
            assert any(identity[0] in line.casefold() for line in report.unmet)
    if report.unmet:
        assert "required lineups" in report.message
    else:
        _assert_min_met(lineups, rate=0.25, requested=requested)
    assert lineups_to_dk_upload_csv(lineups).splitlines()[0].startswith(
        "QB,RB,RB,WR,WR,WR,TE,FLEX,DST"
    )
