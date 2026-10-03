"""Integration tests for Classic presets and Showdown Captain Mode."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from src.exporter import (
    CLASSIC_UPLOAD_COLUMNS,
    SHOWDOWN_UPLOAD_COLUMNS,
    UPLOAD_COLUMNS,
    lineups_to_dk_upload_csv,
)
from src.optimizer import (
    DK_SALARY_CAP,
    NFLClassicOptimizer,
    NFLShowdownOptimizer,
    apply_projection_jitter,
    get_nfl_optimizer,
)
from src.parser import DKCSVValidationError, detect_contest_format, parse_dk_csv
from src.presets import ContestFormat, ContestPreset, apply_preset, get_preset_config

MOCK_CSV = Path(__file__).parent / "mock_dk_salaries.csv"
MOCK_SHOWDOWN_CSV = Path(__file__).parent / "mock_dk_showdown.csv"

CLASSIC_SLOTS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
SHOWDOWN_SLOTS = ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]


def test_parse_mock_csv():
    df = parse_dk_csv(MOCK_CSV)
    assert len(df) >= 30
    assert {"ID", "Name", "Position", "Salary", "TeamAbbrev", "Projection"} <= set(df.columns)
    assert set(df["Position"]).issuperset({"QB", "RB", "WR", "TE", "DST"})
    assert detect_contest_format(df) is ContestFormat.CLASSIC


def test_generate_three_lineups_roster_and_salary():
    optimizer = NFLClassicOptimizer()
    optimizer.load_from_csv(MOCK_CSV)
    optimizer.set_lineup_count(3)
    optimizer.set_max_repeating_players(7)
    optimizer.set_qb_stack(enabled=False)

    lineups = optimizer.generate_lineups(3)
    assert len(lineups) == 3

    for lineup in lineups:
        slots = lineup["Slot"].tolist()
        assert slots == CLASSIC_SLOTS

        counts = Counter(slots)
        assert counts["QB"] == 1
        assert counts["RB"] == 2
        assert counts["WR"] == 3
        assert counts["TE"] == 1
        assert counts["FLEX"] == 1
        assert counts["DST"] == 1

        salary = int(lineup.attrs.get("salary", lineup["Salary"].sum()))
        assert salary <= DK_SALARY_CAP
        assert salary == int(lineup["Salary"].sum())

        flex_pos = lineup.loc[lineup["Slot"] == "FLEX", "Position"].iloc[0]
        assert any(p in {"RB", "WR", "TE"} for p in str(flex_pos).split("/"))


def test_export_upload_csv_columns():
    optimizer = NFLClassicOptimizer()
    optimizer.load_from_csv(MOCK_CSV)
    optimizer.set_qb_stack(enabled=False)
    lineups = optimizer.generate_lineups(2)
    csv_text = lineups_to_dk_upload_csv(lineups)
    header = csv_text.strip().splitlines()[0]
    assert header.split(",") == UPLOAD_COLUMNS == CLASSIC_UPLOAD_COLUMNS
    assert len(csv_text.strip().splitlines()) == 3  # header + 2 lineups


def test_classic_cash_preset_deterministic_single_lineup():
    optimizer = NFLClassicOptimizer()
    optimizer.load_from_csv(MOCK_CSV)
    cfg = apply_preset(optimizer, ContestPreset.CLASSIC_CASH, jitter_seed=42)
    assert cfg.default_lineup_count == 1
    assert optimizer.lineup_count == 1
    assert optimizer.enable_qb_stack is False
    assert optimizer.jitter_pct == (0.0, 0.0)

    a = optimizer.generate_lineups()
    b = optimizer.generate_lineups()
    assert len(a) == 1 and len(b) == 1
    assert a[0]["ID"].tolist() == b[0]["ID"].tolist()
    assert a[0]["Slot"].tolist() == CLASSIC_SLOTS


def test_classic_small_gpp_preset_rules():
    optimizer = NFLClassicOptimizer()
    optimizer.load_from_csv(MOCK_CSV)
    apply_preset(optimizer, ContestPreset.CLASSIC_SMALL_GPP, lineup_count=2, jitter_seed=7)
    assert optimizer.lineup_count == 2
    assert optimizer.enable_qb_stack is True
    assert optimizer.stack_pass_catchers == 1
    assert optimizer.enable_bring_back is False
    assert optimizer.exclude_vs_dst is True
    assert optimizer.max_repeating_players == 6
    assert optimizer.jitter_pct == (0.05, 0.08)

    lineups = optimizer.generate_lineups()
    assert len(lineups) == 2
    for lineup in lineups:
        assert lineup["Slot"].tolist() == CLASSIC_SLOTS
        qb_team = lineup.loc[lineup["Slot"] == "QB", "Team"].iloc[0]
        pass_catcher_teams = lineup.loc[
            lineup["Position"].astype(str).str.contains("WR|TE", regex=True), "Team"
        ]
        assert (pass_catcher_teams == qb_team).any()


def test_classic_large_gpp_preset_clamps_lineups_and_sets_ownership_cap():
    optimizer = NFLClassicOptimizer()
    df = parse_dk_csv(MOCK_CSV)
    # Ownership optional — cap should no-op without crashing.
    apply_preset(optimizer, ContestPreset.CLASSIC_LARGE_GPP, lineup_count=5, jitter_seed=1)
    assert optimizer.lineup_count == 20  # clamped up to preset minimum
    assert optimizer.stack_pass_catchers == 2
    assert optimizer.enable_bring_back is True
    assert optimizer.max_repeating_players == 5
    assert optimizer.max_cumulative_ownership == 1.25
    assert optimizer.position_jitter_sds is not None

    optimizer.load_from_dataframe(df)
    # Keep count small for test runtime while still exercising the builder path.
    optimizer.set_lineup_count(2)
    lineups = optimizer.generate_lineups()
    assert len(lineups) == 2


def test_projection_jitter_seed_reproducible():
    df = parse_dk_csv(MOCK_CSV)
    a = apply_projection_jitter(df, jitter_pct=(0.05, 0.08), seed=123)
    b = apply_projection_jitter(df, jitter_pct=(0.05, 0.08), seed=123)
    c = apply_projection_jitter(df, jitter_pct=(0.05, 0.08), seed=999)
    assert a["Projection"].tolist() == b["Projection"].tolist()
    assert a["Projection"].tolist() != c["Projection"].tolist()


def test_parse_showdown_csv():
    df = parse_dk_csv(MOCK_SHOWDOWN_CSV, contest_format=ContestFormat.SHOWDOWN)
    assert detect_contest_format(df) is ContestFormat.SHOWDOWN
    assert set(df["RosterSlot"]) == {"CPT", "FLEX"}
    assert "QB" in set(df["Position"])
    assert df["RosterSlot"].value_counts()["CPT"] == df["RosterSlot"].value_counts()["FLEX"]


def test_showdown_format_mismatch_raises():
    with pytest.raises(DKCSVValidationError):
        parse_dk_csv(MOCK_SHOWDOWN_CSV, contest_format=ContestFormat.CLASSIC)
    with pytest.raises(DKCSVValidationError):
        parse_dk_csv(MOCK_CSV, contest_format=ContestFormat.SHOWDOWN)


def test_showdown_pipeline_roster_salary_and_export():
    optimizer = NFLShowdownOptimizer()
    optimizer.load_from_csv(MOCK_SHOWDOWN_CSV)
    apply_preset(optimizer, ContestPreset.SHOWDOWN_CASH, lineup_count=2, jitter_seed=42)
    # Cash preset allows up to 5; keep 2 for speed.
    assert optimizer.min_salary_cap == 48_000
    assert optimizer.min_teams == 2

    lineups = optimizer.generate_lineups()
    assert len(lineups) == 2
    for lineup in lineups:
        assert lineup["Slot"].tolist() == SHOWDOWN_SLOTS
        assert len(lineup) == 6
        salary = int(lineup.attrs.get("salary", lineup["Salary"].sum()))
        assert 48_000 <= salary <= DK_SALARY_CAP
        teams = set(lineup["Team"])
        assert len(teams) >= 2
        # Same player cannot appear as both CPT and FLEX.
        assert lineup["Name"].nunique() == 6

    csv_text = lineups_to_dk_upload_csv(lineups, contest_format=ContestFormat.SHOWDOWN)
    header = csv_text.strip().splitlines()[0]
    assert header.split(",") == SHOWDOWN_UPLOAD_COLUMNS
    body = csv_text.strip().splitlines()[1]
    cells = body.split(",")
    assert len(cells) == 6
    assert all(cell.isdigit() for cell in cells)


def test_showdown_gpp_preset_with_game_script():
    optimizer = get_nfl_optimizer(ContestFormat.SHOWDOWN)
    optimizer.load_from_csv(MOCK_SHOWDOWN_CSV)
    cfg = apply_preset(optimizer, ContestPreset.SHOWDOWN_GPP, lineup_count=3, jitter_seed=11)
    assert cfg.game_script_correlation is True
    assert optimizer.max_salary_cap == 49_800
    assert optimizer.game_script_correlation is True
    assert optimizer.max_k_dst_combined == 2

    lineups = optimizer.generate_lineups()
    assert len(lineups) == 3
    for lineup in lineups:
        salary = int(lineup.attrs.get("salary", lineup["Salary"].sum()))
        assert 48_000 <= salary <= 49_800
        k_dst = lineup["Position"].isin(["K", "DST", "DEF"]).sum()
        assert k_dst <= 2

        cpt = lineup.loc[lineup["Slot"] == "CPT"].iloc[0]
        flex = lineup.loc[lineup["Slot"] == "FLEX"]
        if cpt["Position"] == "QB":
            same_team_catchers = flex[
                (flex["Team"] == cpt["Team"]) & (flex["Position"].isin(["WR", "TE"]))
            ]
            assert len(same_team_catchers) >= 1
        elif cpt["Position"] in {"WR", "TE"}:
            same_team_qb = flex[(flex["Team"] == cpt["Team"]) & (flex["Position"] == "QB")]
            assert len(same_team_qb) >= 1


def test_preset_format_mismatch_raises():
    optimizer = NFLClassicOptimizer()
    with pytest.raises(ValueError):
        apply_preset(optimizer, ContestPreset.SHOWDOWN_GPP)


def test_get_preset_config_labels():
    assert "Cash" in get_preset_config(ContestPreset.CLASSIC_CASH).label
    assert "Showdown GPP" in get_preset_config(ContestPreset.SHOWDOWN_GPP).label
