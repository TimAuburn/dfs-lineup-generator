"""Portfolio overlap and exposure hold on the exported set.

These checks fail on the pre-fix solver: tiered Classic batches did not see
each other, and Showdown captain and flex rows each received their own cap.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.exporter import SHOWDOWN_UPLOAD_COLUMNS, lineups_to_dk_upload_csv
from src.optimizer import (
    NFLClassicOptimizer,
    NFLShowdownOptimizer,
    exposure_report_csv,
    optimize_tiered_salaries,
    player_exposure_table,
)
from src.parser import parse_dk_csv
from src.player_pool import person_key as pool_person_key
from src.portfolio import (
    appearances_allowed,
    exposure_counts,
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
        (0.4, 6, 3),
        (0.5, 1, 1),
        (0.3, 20, 6),
        (0.0, 4, 0),
        (None, 4, 4),
    ],
)
def test_appearances_allowed_matches_total_exposure_strategy(rate, total, allowed):
    assert appearances_allowed(rate, total) == allowed


def _assert_limits(lineups, *, requested, overlap, exposure, stack_rate=None, stack_spec=()):
    assert lineups
    if overlap is not None:
        assert pairwise_overlap(lineups) <= int(overlap)
    if exposure is not None:
        allowed = appearances_allowed(float(exposure), requested)
        assert all(count <= allowed for count in exposure_counts(lineups).values())
    if stack_rate is not None:
        allowed = appearances_allowed(float(stack_rate), requested)
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

    table = player_exposure_table(lineups)
    allen = table.loc[table["Name"].eq("Josh Allen")].iloc[0]
    assert int(allen["Lineups"]) == counts[person_key("Josh Allen", "BUF")]
    assert int(allen["CPT count"]) + int(allen["FLEX count"]) == int(allen["Lineups"])
    assert list(table["Name"]).count("Josh Allen") == 1

    upload = lineups_to_dk_upload_csv(lineups, contest_format="showdown")
    assert upload.splitlines()[0].split(",") == SHOWDOWN_UPLOAD_COLUMNS
    report = exposure_report_csv(lineups)
    assert report.splitlines()[0].startswith("Name,Team,Position,Lineups")
    assert "Josh Allen" in report
    assert "Exposure" not in upload


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
