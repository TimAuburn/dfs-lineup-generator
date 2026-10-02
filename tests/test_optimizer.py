"""Integration tests for the DraftKings NFL Classic optimizer."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from src.exporter import UPLOAD_COLUMNS, lineups_to_dk_upload_csv
from src.optimizer import DK_SALARY_CAP, NFLClassicOptimizer
from src.parser import parse_dk_csv

MOCK_CSV = Path(__file__).parent / "mock_dk_salaries.csv"

REQUIRED_SLOTS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]


def test_parse_mock_csv():
    df = parse_dk_csv(MOCK_CSV)
    assert len(df) >= 30
    assert {"ID", "Name", "Position", "Salary", "TeamAbbrev", "Projection"} <= set(df.columns)
    assert set(df["Position"]).issuperset({"QB", "RB", "WR", "TE", "DST"})


def test_generate_three_lineups_roster_and_salary():
    optimizer = NFLClassicOptimizer()
    optimizer.load_from_csv(MOCK_CSV)
    optimizer.set_lineup_count(3)
    optimizer.set_max_repeating_players(7)
    # Keep stacking off for a reliable feasibility baseline on the mock pool.
    optimizer.set_qb_stack(enabled=False)

    lineups = optimizer.generate_lineups(3)
    assert len(lineups) == 3

    for lineup in lineups:
        slots = lineup["Slot"].tolist()
        assert slots == REQUIRED_SLOTS

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
    assert header.split(",") == UPLOAD_COLUMNS
    assert len(csv_text.strip().splitlines()) == 3  # header + 2 lineups
