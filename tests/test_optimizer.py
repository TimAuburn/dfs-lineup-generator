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
    STANDARD_TIER_FLOOR,
    GameScriptBucket,
    NFLClassicOptimizer,
    NFLShowdownOptimizer,
    apply_projection_jitter,
    classic_gpp_rule_check,
    default_starting_quarterbacks,
    unique_quarterbacks,
    drop_backup_players,
    even_lineup_counts,
    game_script_summary,
    games_from_pool,
    get_nfl_optimizer,
    optimize_game_scripts,
    optimize_tiered_salaries,
    overlap_cap_for_slate,
    player_exposure_table,
    tiered_lineup_counts,
)
from src.parser import (
    DKCSVValidationError,
    detect_contest_format,
    injury_status_for_player,
    is_inactive_injury,
    parse_dk_csv,
)
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
    assert optimizer.flex_positions == ("RB", "WR")
    assert optimizer.min_skill_salary == 3_800

    a = optimizer.generate_lineups()
    b = optimizer.generate_lineups()
    assert len(a) == 1 and len(b) == 1
    assert a[0]["ID"].tolist() == b[0]["ID"].tolist()
    assert a[0]["Slot"].tolist() == CLASSIC_SLOTS
    lineup = a[0]
    flex_pos = str(lineup.loc[lineup["Slot"] == "FLEX", "Position"].iloc[0])
    assert flex_pos in {"RB", "WR"}
    skill = lineup[lineup["Position"].isin(["RB", "WR", "TE"])]
    assert int(skill["Salary"].min()) >= 3_800
    assert int((lineup["Position"] == "TE").sum()) == 1


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
    assert optimizer.bring_back_position == "SKILL"
    assert optimizer.require_wr_in_stack is True
    assert optimizer.flex_positions == ("RB", "WR")
    assert optimizer.slate_aware_gpp is True
    assert optimizer.max_repeating_players == 5
    assert optimizer.max_cumulative_ownership == 1.25
    assert optimizer.position_jitter_sds is not None

    optimizer.load_from_dataframe(df)
    # Keep count small for test runtime while still exercising the builder path.
    optimizer.set_lineup_count(2)
    lineups = optimizer.generate_lineups()
    assert len(lineups) == 2
    if optimizer.last_game_count <= 4:
        assert optimizer.last_stack_exposure == 0.30
    else:
        assert optimizer.last_stack_exposure == 0.40
    if optimizer.last_game_count < 10:
        assert optimizer.last_overlap_cap == 4
    else:
        assert optimizer.last_overlap_cap == 5
    for lineup in lineups:
        qb = lineup.loc[lineup["Slot"] == "QB"].iloc[0]
        catchers = lineup[
            (lineup["Team"] == qb["Team"]) & lineup["Position"].isin(["WR", "TE"])
        ]
        assert len(catchers) >= 2
        assert (catchers["Position"] == "WR").any()
        bring = lineup[
            (lineup["Team"] == qb["Opponent"])
            & lineup["Position"].isin(["WR", "TE", "RB"])
        ]
        assert len(bring) >= 1
        flex_pos = str(lineup.loc[lineup["Slot"] == "FLEX", "Position"].iloc[0])
        assert flex_pos in {"RB", "WR"}
        assert int((lineup["Position"] == "TE").sum()) == 1


def _classic_player(pos, name, pid, team, salary, fppg, game) -> dict:
    return {
        "Position": pos,
        "Name + ID": f"{name} ({pid})",
        "Name": name,
        "ID": str(pid),
        "Roster Position": pos,
        "Salary": salary,
        "Game Info": game,
        "TeamAbbrev": team,
        "Team": team,
        "AvgPointsPerGame": fppg,
        "Projection": fppg,
    }


def _roster_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.attrs["salary"] = int(frame["Salary"].sum())
    return frame


def test_large_gpp_rule_check_flags_thin_stacks_and_missing_bring_back():
    game = "KC@BUF 09/10/2025 08:20PM ET"
    one_catcher = _roster_frame(
        [
            _classic_player("QB", "Patrick Mahomes", 1, "KC", 8000, 25, game),
            _classic_player("WR", "Xavier Worthy", 2, "KC", 5000, 12, game),
            _classic_player("RB", "James Cook", 3, "BUF", 6000, 14, game),
        ]
    )
    two_tight_ends = _roster_frame(
        [
            _classic_player("QB", "Patrick Mahomes", 1, "KC", 8000, 25, game),
            _classic_player("TE", "Travis Kelce", 2, "KC", 5000, 14, game),
            _classic_player("TE", "Noah Gray", 3, "KC", 3000, 6, game),
            _classic_player("WR", "Khalil Shakir", 4, "BUF", 4000, 10, game),
        ]
    )
    no_bring_back = _roster_frame(
        [
            _classic_player("QB", "Patrick Mahomes", 1, "KC", 8000, 25, game),
            _classic_player("WR", "Xavier Worthy", 2, "KC", 5000, 12, game),
            _classic_player("TE", "Travis Kelce", 3, "KC", 4500, 11, game),
            _classic_player("DST", "Bills", 4, "BUF", 3000, 6, game),
        ]
    )
    good = _roster_frame(
        [
            _classic_player("QB", "Patrick Mahomes", 1, "KC", 8000, 25, game),
            _classic_player("WR", "Xavier Worthy", 2, "KC", 5000, 12, game),
            _classic_player("TE", "Travis Kelce", 3, "KC", 4500, 11, game),
            _classic_player("WR", "Khalil Shakir", 4, "BUF", 4000, 10, game),
        ]
    )
    for frame in (one_catcher, two_tight_ends, no_bring_back, good):
        qb = frame.loc[frame["Position"] == "QB"].iloc[0]
        frame["Slot"] = "FLEX"
        frame.loc[frame["ID"] == qb["ID"], "Slot"] = "QB"
        frame["Opponent"] = frame["Team"].map({"KC": "BUF", "BUF": "KC"})

    thin = classic_gpp_rule_check([one_catcher]).iloc[0]
    assert "Stack needs 2 pass-catchers" in thin["Result"]
    tight = classic_gpp_rule_check([two_tight_ends]).iloc[0]
    assert "Stack needs a WR" in tight["Result"]
    naked = classic_gpp_rule_check([no_bring_back]).iloc[0]
    assert "No bring-back" in naked["Result"]
    passed = classic_gpp_rule_check(
        [good], overlap_cap=4, salary_min=15_000, salary_max=50_000
    ).iloc[0]
    assert passed["Result"] == "Pass"
    assert passed["Bring-back"] == "Khalil Shakir"
    assert overlap_cap_for_slate(5, 8, True) == 4
    assert overlap_cap_for_slate(5, 12, True) == 5


def test_double_stack_rejects_qb_with_only_tight_ends():
    game = "KC@BUF 09/10/2025 08:20PM ET"
    rows = [
        _classic_player("QB", "Patrick Mahomes", 1, "KC", 9000, 40, game),
        _classic_player("TE", "Travis Kelce", 2, "KC", 5000, 16, game),
        _classic_player("TE", "Noah Gray", 3, "KC", 3000, 8, game),
        _classic_player("RB", "Isiah Pacheco", 4, "KC", 6000, 12, game),
        _classic_player("DST", "Chiefs", 5, "KC", 3000, 6, game),
        _classic_player("QB", "Josh Allen", 6, "BUF", 8000, 18, game),
        _classic_player("WR", "Khalil Shakir", 7, "BUF", 4500, 11, game),
        _classic_player("WR", "Keon Coleman", 8, "BUF", 4000, 9, game),
        _classic_player("WR", "Curtis Samuel", 9, "BUF", 3500, 7, game),
        _classic_player("TE", "Dalton Kincaid", 10, "BUF", 4200, 8, game),
        _classic_player("RB", "James Cook", 11, "BUF", 5500, 12, game),
        _classic_player("DST", "Bills", 12, "BUF", 2800, 5, game),
    ]
    optimizer = NFLClassicOptimizer()
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.load_from_dataframe(pd.DataFrame(rows))
    optimizer.set_qb_stack(True, pass_catchers=2, bring_back=True, bring_back_position="SKILL")
    optimizer.require_wr_in_stack = True
    optimizer.set_min_salary_cap(30_000)
    optimizer.set_lineup_count(1)
    lineup = optimizer.generate_lineups()[0]
    qb = lineup.loc[lineup["Slot"] == "QB"].iloc[0]
    assert qb["Name"] == "Josh Allen"
    catchers = lineup[(lineup["Team"] == "BUF") & lineup["Position"].isin(["WR", "TE"])]
    assert len(catchers) >= 2
    assert (catchers["Position"] == "WR").any()
    assert (lineup["Team"] == "KC").any()


def test_flex_wr_or_rb_blocks_a_second_tight_end():
    game_one = "BUF@KC 09/10/2025 08:20PM ET"
    game_two = "PHI@DAL 09/11/2025 01:00PM ET"
    # One spare WR is worse than the second tight end, so an open flex takes the tight end.
    rows = [
        _classic_player("QB", "Josh Allen", 1, "BUF", 7000, 18, game_one),
        _classic_player("RB", "James Cook", 2, "BUF", 5500, 12, game_one),
        _classic_player("RB", "Ray Davis", 3, "BUF", 4000, 8, game_one),
        _classic_player("WR", "Khalil Shakir", 4, "BUF", 4500, 10, game_one),
        _classic_player("WR", "Keon Coleman", 5, "BUF", 4000, 8, game_one),
        _classic_player("WR", "Curtis Samuel", 6, "BUF", 3000, 6, game_one),
        _classic_player("TE", "Dalton Kincaid", 7, "BUF", 4200, 7, game_one),
        _classic_player("DST", "Bills", 8, "BUF", 2800, 5, game_one),
        _classic_player("TE", "Dawson Knox", 9, "PHI", 2500, 40, game_two),
        _classic_player("WR", "Britain Covey", 10, "PHI", 3000, 1, game_two),
        _classic_player("DST", "Eagles", 11, "PHI", 3000, 4, game_two),
    ]
    frame = pd.DataFrame(rows)
    open_flex = NFLClassicOptimizer()
    open_flex.jitter_pct = (0.0, 0.0)
    open_flex.load_from_dataframe(frame)
    open_flex.set_lineup_count(1)
    unrestricted = open_flex.generate_lineups()[0]
    assert "Dawson Knox" in set(unrestricted["Name"])
    assert int((unrestricted["Position"] == "TE").sum()) == 2

    limited = NFLClassicOptimizer()
    limited.jitter_pct = (0.0, 0.0)
    limited.flex_positions = ("RB", "WR")
    limited.load_from_dataframe(frame)
    limited.set_lineup_count(1)
    lineup = limited.generate_lineups()[0]
    flex_pos = str(lineup.loc[lineup["Slot"] == "FLEX", "Position"].iloc[0])
    assert flex_pos in {"RB", "WR"}
    assert int((lineup["Position"] == "TE").sum()) == 1
    checked = classic_gpp_rule_check([unrestricted], flex_positions=("RB", "WR")).iloc[0]
    assert "Flex is TE" in checked["Result"]


def test_cash_skill_floor_drops_punts_unless_locked():
    game_one = "BUF@KC 09/10/2025 08:20PM ET"
    game_two = "PHI@DAL 09/11/2025 01:00PM ET"
    rows = [
        _classic_player("QB", "Josh Allen", 1, "BUF", 7000, 18, game_one),
        _classic_player("RB", "James Cook", 2, "BUF", 5500, 12, game_one),
        _classic_player("RB", "Ray Davis", 3, "BUF", 4000, 8, game_one),
        _classic_player("WR", "Khalil Shakir", 4, "BUF", 4500, 10, game_one),
        _classic_player("WR", "Keon Coleman", 5, "BUF", 4000, 8, game_one),
        _classic_player("WR", "Curtis Samuel", 6, "BUF", 3900, 6, game_one),
        _classic_player("WR", "Punt Receiver", 7, "BUF", 3000, 40, game_one),
        _classic_player("TE", "Dalton Kincaid", 8, "BUF", 4200, 7, game_one),
        _classic_player("DST", "Bills", 9, "BUF", 2800, 5, game_one),
        _classic_player("QB", "Jalen Hurts", 10, "PHI", 7000, 16, game_two),
        _classic_player("RB", "Saquon Barkley", 11, "PHI", 6000, 12, game_two),
        _classic_player("WR", "A.J. Brown", 12, "PHI", 6400, 13, game_two),
        _classic_player("DST", "Eagles", 13, "PHI", 3000, 4, game_two),
    ]
    frame = pd.DataFrame(rows)

    def _solve(locked=None):
        optimizer = NFLClassicOptimizer()
        optimizer.jitter_pct = (0.0, 0.0)
        optimizer.min_skill_salary = 3_800
        optimizer.flex_positions = ("RB", "WR")
        optimizer.load_from_dataframe(frame)
        optimizer.set_locked_players(locked)
        optimizer.set_lineup_count(1)
        return optimizer.generate_lineups()[0]

    blocked = _solve()
    assert "Punt Receiver" not in set(blocked["Name"])
    skill = blocked[blocked["Position"].isin(["RB", "WR", "TE"])]
    assert int(skill["Salary"].min()) >= 3_800

    forced = _solve(["7"])
    assert "Punt Receiver" in set(forced["Name"])


def test_showdown_cash_captain_and_flex_floors():
    rows = []
    specs = [
        ("QB", "Patrick Mahomes", 1, "KC", 10000, 18),
        ("RB", "Isiah Pacheco", 2, "KC", 8000, 12),
        ("WR", "Xavier Worthy", 3, "KC", 7500, 11),
        ("TE", "Travis Kelce", 4, "KC", 9000, 40),
        ("RB", "Cheap Back", 5, "KC", 1500, 30),
        ("DST", "Chiefs", 6, "KC", 4000, 20),
        ("QB", "Josh Allen", 7, "BUF", 9800, 17),
        ("RB", "James Cook", 8, "BUF", 8200, 12),
        ("WR", "Khalil Shakir", 9, "BUF", 7000, 10),
        ("TE", "Dalton Kincaid", 10, "BUF", 6000, 8),
        ("K", "Tyler Bass", 11, "BUF", 4500, 8),
        ("DST", "Bills", 12, "BUF", 4200, 19),
    ]
    for pos, name, pid, team, salary, fppg in specs:
        rows.extend(_showdown_pair(pos, name, pid * 10, team, salary, fppg))

    optimizer = NFLShowdownOptimizer()
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.load_from_dataframe(pd.DataFrame(rows))
    optimizer.configure_showdown_rules(
        min_salary_cap=40_000,
        max_salary_cap=50_000,
        min_teams=2,
        game_script_correlation=False,
        max_kickers=1,
        max_dst=1,
        allow_kicker_captain=False,
        captain_positions=("QB", "RB", "WR"),
        captain_min_salary=8_000,
        flex_min_salary=2_000,
    )
    optimizer.set_lineup_count(1)
    lineup = optimizer.generate_lineups()[0]
    cpt = lineup.loc[lineup["Slot"] == "CPT"].iloc[0]
    assert cpt["Position"] in {"QB", "RB", "WR"}
    assert cpt["Name"] != "Travis Kelce"
    assert int(cpt["Salary"]) >= 12_000
    assert "Cheap Back" not in set(lineup["Name"])
    assert int((lineup["Position"] == "DST").sum()) <= 1
    assert int((lineup["Position"] == "K").sum()) <= 1


def test_showdown_drops_scrubs_unless_flex_punt_is_locked():
    rows = []
    rows += _showdown_pair("WR", "Amon-Ra St. Brown", 100, "DET", 11000, 22)
    rows += _showdown_pair("RB", "Jahmyr Gibbs", 110, "DET", 12000, 18)
    rows += _showdown_pair("WR", "Bench Punt", 120, "DET", 200, 14)
    rows += _showdown_pair("WR", "Other Punt", 130, "DET", 200, 13)
    rows += _showdown_pair("WR", "Zero Projection", 140, "DET", 5000, 0.5)
    rows += _showdown_pair("WR", "Jameson Williams", 150, "DET", 5500, 10)
    rows += _showdown_pair("TE", "Sam LaPorta", 160, "DET", 4800, 9)
    rows += _showdown_pair("K", "Jake Bates", 170, "DET", 4000, 8)
    rows += _showdown_pair("QB", "Jared Goff", 180, "DET", 7000, 16)
    rows += _showdown_pair("QB", "Jordan Love", 200, "GB", 8000, 15)
    rows += _showdown_pair("WR", "Romeo Doubs", 210, "GB", 5200, 9)
    rows += _showdown_pair("TE", "Tucker Kraft", 220, "GB", 4500, 8)
    rows += _showdown_pair("RB", "Josh Jacobs", 230, "GB", 7000, 12)

    frame = pd.DataFrame(rows)

    def _solve(locked=None):
        optimizer = NFLShowdownOptimizer()
        optimizer.jitter_pct = (0.0, 0.0)
        optimizer.load_from_dataframe(frame)
        optimizer.configure_showdown_rules(
            min_salary_cap=20_000,
            max_salary_cap=50_000,
            min_teams=2,
            game_script_correlation=False,
            max_kickers=1,
            allow_kicker_captain=False,
            flex_min_salary=None,
        )
        optimizer.set_locked_players(locked)
        optimizer.set_lineup_count(1)
        return optimizer.generate_lineups()[0]

    lineup = _solve()
    names = set(lineup["Name"])
    assert "Bench Punt" not in names
    assert "Other Punt" not in names
    assert "Zero Projection" not in names
    flex = lineup.loc[lineup["Slot"] == "FLEX"]
    assert int(flex["Salary"].min()) >= 1_200

    forced = _solve(["120"])
    assert "Bench Punt" in set(forced["Name"])
    assert "Zero Projection" not in set(forced["Name"])


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
    assert optimizer.min_salary_cap == 49_600
    assert optimizer.max_salary_cap == 50_000
    assert optimizer.captain_min_salary == 8_000
    assert optimizer.flex_min_salary == 3_500
    assert optimizer.max_kickers == 1
    assert optimizer.ban_dst is True
    assert optimizer.max_below_salary_players == 1
    assert optimizer.below_salary_threshold == 4_500
    assert optimizer.max_rbs_per_team == 1
    assert optimizer.allow_kicker_captain is False
    assert optimizer.min_teams == 2

    lineups = optimizer.generate_lineups()
    assert len(lineups) == 2
    for lineup in lineups:
        assert lineup["Slot"].tolist() == SHOWDOWN_SLOTS
        assert len(lineup) == 6
        salary = int(lineup.attrs.get("salary", lineup["Salary"].sum()))
        assert 49_600 <= salary <= DK_SALARY_CAP
        cpt = lineup.loc[lineup["Slot"] == "CPT"].iloc[0]
        assert cpt["Position"] in {"QB", "RB", "WR"}
        assert int(cpt["Salary"]) >= 12_000
        flex = lineup.loc[lineup["Slot"] == "FLEX"]
        assert int(flex["Salary"].min()) >= 3_500
        assert int((lineup["Position"] == "K").sum()) <= 1
        assert int((lineup["Position"] == "DST").sum()) == 0
        cheap = 0
        for _, row in lineup.iterrows():
            listed = int(row["Salary"])
            if row["Slot"] == "CPT":
                listed = int(round(listed / 1.5))
            if listed < 4_500:
                cheap += 1
        assert cheap <= 1
        rb_counts = lineup.loc[lineup["Position"] == "RB"].groupby("Team").size()
        assert rb_counts.empty or int(rb_counts.max()) <= 1
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


def test_showdown_cash_bans_dst_and_limits_scrubs():
    rows = []
    rows += _showdown_pair("QB", "Patrick Mahomes", 100, "KC", 10000, 18)
    rows += _showdown_pair("RB", "Isiah Pacheco", 110, "KC", 8000, 14)
    rows += _showdown_pair("WR", "Xavier Worthy", 120, "KC", 7000, 12)
    rows += _showdown_pair("WR", "Cheap Worthy", 130, "KC", 4000, 22)
    rows += _showdown_pair("WR", "Deep Punt", 140, "KC", 3000, 20)
    rows += _showdown_pair("TE", "Travis Kelce", 150, "KC", 6500, 11)
    rows += _showdown_pair("K", "Harrison Butker", 160, "KC", 4000, 18)
    rows += _showdown_pair("DST", "Chiefs", 170, "KC", 4600, 40)
    rows += _showdown_pair("QB", "Josh Allen", 200, "BUF", 9800, 17)
    rows += _showdown_pair("RB", "James Cook", 210, "BUF", 7500, 13)
    rows += _showdown_pair("WR", "Khalil Shakir", 220, "BUF", 6800, 12)
    rows += _showdown_pair("WR", "Second Scrub", 230, "BUF", 4200, 21)
    rows += _showdown_pair("K", "Tyler Bass", 240, "BUF", 5000, 17)
    rows += _showdown_pair("DST", "Bills", 250, "BUF", 4400, 35)
    rows += _showdown_pair("TE", "Dalton Kincaid", 260, "BUF", 6000, 10)

    optimizer = NFLShowdownOptimizer()
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.load_from_dataframe(pd.DataFrame(rows))
    optimizer.configure_showdown_rules(
        min_salary_cap=30_000,
        max_salary_cap=50_000,
        min_teams=2,
        game_script_correlation=False,
        max_kickers=1,
        allow_kicker_captain=False,
        ban_dst=True,
        flex_min_salary=3_500,
        max_below_salary_players=1,
        below_salary_threshold=4_500,
        captain_positions=("QB", "RB", "WR"),
        captain_min_salary=8_000,
    )
    optimizer.set_lineup_count(1)
    lineup = optimizer.generate_lineups()[0]
    names = set(lineup["Name"])
    assert "Chiefs" not in names
    assert "Bills" not in names
    assert "Deep Punt" not in names
    assert int((lineup["Position"] == "K").sum()) <= 1
    assert int((lineup["Position"] == "DST").sum()) == 0
    cheap_names = []
    for _, row in lineup.iterrows():
        listed = int(row["Salary"])
        if row["Slot"] == "CPT":
            listed = int(round(listed / 1.5))
        if listed < 4_500:
            cheap_names.append(row["Name"])
    assert len(cheap_names) <= 1
    flex = lineup.loc[lineup["Slot"] == "FLEX"]
    assert int(flex["Salary"].min()) >= 3_500


def test_showdown_cash_blocks_two_running_backs_from_one_team():
    rows = []
    rows += _showdown_pair("QB", "Patrick Mahomes", 100, "KC", 9000, 16)
    rows += _showdown_pair("RB", "Star Back", 110, "KC", 8000, 30)
    rows += _showdown_pair("RB", "Backup Back", 120, "KC", 7000, 28)
    rows += _showdown_pair("WR", "Xavier Worthy", 130, "KC", 6000, 10)
    rows += _showdown_pair("TE", "Travis Kelce", 140, "KC", 5500, 9)
    rows += _showdown_pair("QB", "Josh Allen", 200, "BUF", 8800, 15)
    rows += _showdown_pair("RB", "James Cook", 210, "BUF", 6500, 12)
    rows += _showdown_pair("WR", "Khalil Shakir", 220, "BUF", 6000, 11)
    rows += _showdown_pair("WR", "Keon Coleman", 230, "BUF", 5000, 10)
    rows += _showdown_pair("TE", "Dalton Kincaid", 240, "BUF", 5200, 9)

    def _solve(max_rbs_per_team):
        optimizer = NFLShowdownOptimizer()
        optimizer.jitter_pct = (0.0, 0.0)
        optimizer.load_from_dataframe(pd.DataFrame(rows))
        optimizer.configure_showdown_rules(
            min_salary_cap=30_000,
            max_salary_cap=50_000,
            min_teams=2,
            game_script_correlation=False,
            max_kickers=None,
            allow_kicker_captain=False,
            ban_dst=True,
            flex_min_salary=1_200,
            captain_positions=("QB", "RB", "WR"),
            captain_min_salary=8_000,
            max_rbs_per_team=max_rbs_per_team,
        )
        optimizer.set_lineup_count(1)
        return optimizer.generate_lineups()[0]

    open_lineup = _solve(None)
    open_kc = open_lineup[
        (open_lineup["Team"] == "KC") & (open_lineup["Position"] == "RB")
    ]
    assert set(open_kc["Name"]) == {"Star Back", "Backup Back"}

    capped = _solve(1)
    for team, group in capped.loc[capped["Position"] == "RB"].groupby("Team"):
        assert len(group) <= 1, team
    capped_kc = set(
        capped.loc[
            (capped["Team"] == "KC") & (capped["Position"] == "RB"),
            "Name",
        ]
    )
    assert len(capped_kc) == 1
    assert capped_kc <= {"Star Back", "Backup Back"}


def test_default_starting_quarterbacks_picks_one_per_team():
    rows = []
    rows += _showdown_pair("QB", "Patrick Mahomes", 100, "KC", 10000, 22.5)
    rows += _showdown_pair("QB", "Cooper Rush", 110, "KC", 4000, 5.10)
    rows += _showdown_pair("QB", "Josh Allen", 200, "BUF", 11000, 23.0)
    rows += _showdown_pair("QB", "Mitchell Trubisky", 210, "BUF", 4500, 8.0)
    starters = default_starting_quarterbacks(pd.DataFrame(rows))
    assert starters == [("Josh Allen", "BUF"), ("Patrick Mahomes", "KC")]


def test_default_starting_quarterbacks_uses_sleeper_depth_chart():
    rows = pd.DataFrame(
        [
            {
                "Name": "Sam Darnold",
                "Position": "QB",
                "TeamAbbrev": "SEA",
                "Projection": 14.50,
                "Sleeper": 14.13,
                "DepthOrder": 1,
                "DepthPosition": "QB",
            },
            {
                "Name": "Drew Lock",
                "Position": "QB",
                "TeamAbbrev": "SEA",
                "Projection": 18.40,
                "Sleeper": 8.20,
                "DepthOrder": 2,
                "DepthPosition": "QB",
            },
            {
                "Name": "Bo Nix",
                "Position": "QB",
                "TeamAbbrev": "DEN",
                "Projection": 20.00,
                "Sleeper": 14.28,
                "DepthOrder": 1,
                "DepthPosition": "QB",
            },
            {
                "Name": "Jarrett Stidham",
                "Position": "QB",
                "TeamAbbrev": "DEN",
                "Projection": 16.00,
                "Sleeper": pd.NA,
                "DepthOrder": 2,
                "DepthPosition": "QB",
            },
        ]
    )
    table = unique_quarterbacks(rows)
    by_name = dict(zip(table["Name"], table["Projection"]))
    assert by_name["Sam Darnold"] == pytest.approx(14.13)
    assert by_name["Jarrett Stidham"] == pytest.approx(0.0)
    assert default_starting_quarterbacks(rows) == [
        ("Bo Nix", "DEN"),
        ("Sam Darnold", "SEA"),
    ]


def test_drop_backup_players_keeps_the_leader_at_each_spot():
    pool = pd.DataFrame(
        [
            {"ID": "1", "Name": "Patrick Mahomes", "Position": "QB", "TeamAbbrev": "KC", "Projection": 18, "DepthOrder": 1, "DepthPosition": "QB"},
            {"ID": "2", "Name": "Justin Fields", "Position": "QB", "TeamAbbrev": "KC", "Projection": 40, "DepthOrder": 2, "DepthPosition": "QB"},
            {"ID": "3", "Name": "Kenneth Walker", "Position": "RB", "TeamAbbrev": "KC", "Projection": 14, "DepthOrder": 1, "DepthPosition": "RB"},
            {"ID": "4", "Name": "Emmett Johnson", "Position": "RB", "TeamAbbrev": "KC", "Projection": 30, "DepthOrder": 2, "DepthPosition": "RB"},
            {"ID": "5", "Name": "Rashee Rice", "Position": "WR", "TeamAbbrev": "KC", "Projection": 15, "DepthOrder": 1, "DepthPosition": "SWR"},
            {"ID": "6", "Name": "Xavier Worthy", "Position": "WR", "TeamAbbrev": "KC", "Projection": 12, "DepthOrder": 2, "DepthPosition": "RWR"},
            {"ID": "7", "Name": "Jalen Royals", "Position": "WR", "TeamAbbrev": "KC", "Projection": 8, "DepthOrder": 3, "DepthPosition": "LWR"},
            {"ID": "8", "Name": "Tyquan Thornton", "Position": "WR", "TeamAbbrev": "KC", "Projection": 20, "DepthOrder": 6, "DepthPosition": "LWR"},
            {"ID": "9", "Name": "Travis Kelce", "Position": "TE", "TeamAbbrev": "KC", "Projection": 10, "DepthOrder": 1, "DepthPosition": "TE"},
            {"ID": "10", "Name": "Noah Gray", "Position": "TE", "TeamAbbrev": "KC", "Projection": 12, "DepthOrder": 2, "DepthPosition": "TE"},
            {"ID": "11", "Name": "Harrison Butker", "Position": "K", "TeamAbbrev": "KC", "Projection": 8, "DepthOrder": 1, "DepthPosition": "K"},
            {"ID": "12", "Name": "Mystery Wideout", "Position": "WR", "TeamAbbrev": "KC", "Projection": 25, "DepthOrder": pd.NA, "DepthPosition": ""},
            {"ID": "13", "Name": "Chiefs", "Position": "DST", "TeamAbbrev": "KC", "Projection": 7, "DepthOrder": pd.NA, "DepthPosition": ""},
            {"ID": "14", "Name": "Only QB", "Position": "QB", "TeamAbbrev": "BUF", "Projection": 4, "DepthOrder": pd.NA, "DepthPosition": ""},
            {"ID": "15", "Name": "Other QB", "Position": "QB", "TeamAbbrev": "BUF", "Projection": 19, "DepthOrder": pd.NA, "DepthPosition": ""},
        ]
    )
    kept = set(drop_backup_players(pool)["Name"])
    assert kept >= {
        "Patrick Mahomes",
        "Kenneth Walker",
        "Rashee Rice",
        "Xavier Worthy",
        "Jalen Royals",
        "Travis Kelce",
        "Harrison Butker",
        "Chiefs",
        "Other QB",
    }
    assert {
        "Justin Fields",
        "Emmett Johnson",
        "Tyquan Thornton",
        "Noah Gray",
        "Mystery Wideout",
        "Only QB",
    }.isdisjoint(kept)


def test_classic_lineup_does_not_start_backups():
    game = "KC@BUF 10/11/2026 01:00PM ET"

    def row(pid, name, pos, team, salary, proj, order, slot):
        return {
            "ID": str(pid),
            "Name": name,
            "Position": pos,
            "Roster Position": pos if pos != "DST" else "DST",
            "TeamAbbrev": team,
            "Salary": salary,
            "Game Info": game,
            "AvgPointsPerGame": proj,
            "Projection": proj,
            "Injury": "",
            "DepthOrder": order,
            "DepthPosition": slot,
        }

    pool = pd.DataFrame(
        [
            row(1, "Starter QB", "QB", "KC", 7000, 16, 1, "QB"),
            row(2, "Backup QB", "QB", "KC", 4500, 40, 2, "QB"),
            row(3, "Starter RB", "RB", "KC", 6400, 14, 1, "RB"),
            row(4, "Backup RB", "RB", "KC", 4000, 35, 2, "RB"),
            row(5, "Second RB", "RB", "BUF", 6000, 13, 1, "RB"),
            row(6, "Wide One", "WR", "KC", 5800, 12, 1, "LWR"),
            row(7, "Wide Two", "WR", "KC", 5600, 11, 2, "RWR"),
            row(8, "Wide Three", "WR", "BUF", 5400, 11, 1, "SWR"),
            row(9, "Backup WR", "WR", "KC", 3000, 30, 5, "LWR"),
            row(10, "Starter TE", "TE", "KC", 5000, 10, 1, "TE"),
            row(11, "Backup TE", "TE", "KC", 3000, 28, 2, "TE"),
            row(12, "Flex WR", "WR", "BUF", 5200, 10, 2, "RWR"),
            row(13, "Chiefs", "DST", "KC", 3000, 8, None, ""),
            row(14, "Other QB", "QB", "BUF", 6800, 15, 1, "QB"),
        ]
    )
    optimizer = NFLClassicOptimizer()
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.load_from_dataframe(pool)
    optimizer.set_lineup_count(1)
    lineup = optimizer.generate_lineups()[0]
    names = set(lineup["Name"])
    assert "Backup QB" not in names
    assert "Backup RB" not in names
    assert "Backup WR" not in names
    assert "Backup TE" not in names
    assert "Starter QB" in names or "Other QB" in names
    assert {"Starter RB", "Second RB"} <= names
    assert "Starter TE" in names


def test_showdown_drops_backup_quarterbacks_under_12():
    rows = []
    rows += _showdown_pair("QB", "Patrick Mahomes", 100, "KC", 10000, 18)
    rows += _showdown_pair("QB", "Cooper Rush", 110, "KC", 5000, 10)
    rows += _showdown_pair("RB", "Isiah Pacheco", 120, "KC", 7000, 12)
    rows += _showdown_pair("WR", "Xavier Worthy", 130, "KC", 6000, 11)
    rows += _showdown_pair("TE", "Travis Kelce", 140, "KC", 6500, 10)
    rows += _showdown_pair("QB", "Josh Allen", 200, "BUF", 9800, 17)
    rows += _showdown_pair("RB", "James Cook", 210, "BUF", 6800, 12)
    rows += _showdown_pair("WR", "Khalil Shakir", 220, "BUF", 6000, 11)
    rows += _showdown_pair("TE", "Dalton Kincaid", 230, "BUF", 5500, 9)
    rows += _showdown_pair("WR", "Keon Coleman", 240, "BUF", 5200, 10)

    def _solve(locked_ids=None, confirmed=None):
        optimizer = NFLShowdownOptimizer()
        optimizer.jitter_pct = (0.0, 0.0)
        optimizer.load_from_dataframe(pd.DataFrame(rows))
        optimizer.configure_showdown_rules(
            min_salary_cap=30_000,
            max_salary_cap=50_000,
            min_teams=2,
            game_script_correlation=False,
            max_kickers=None,
            allow_kicker_captain=False,
            ban_dst=True,
            flex_min_salary=1_200,
        )
        if locked_ids is not None:
            optimizer.set_locked_players(locked_ids)
        if confirmed is not None:
            optimizer.set_confirmed_quarterbacks(confirmed)
        optimizer.set_lineup_count(1)
        return optimizer.generate_lineups()[0]

    lineup = _solve()
    assert "Cooper Rush" not in set(lineup["Name"])
    # A 10-point backup's captain row is 15 points and is still removed.
    assert "110" not in set(lineup["ID"].astype(str))
    assert "111" not in set(lineup["ID"].astype(str))

    locked = _solve(locked_ids=["110"])
    assert "Cooper Rush" not in set(locked["Name"])

    kept = _solve(confirmed=[("Patrick Mahomes", "KC"), ("Josh Allen", "BUF")])
    assert "Cooper Rush" not in set(kept["Name"])


def test_showdown_confirmed_quarterbacks_drop_other_qbs():
    rows = []
    rows += _showdown_pair("QB", "Patrick Mahomes", 100, "KC", 9000, 16)
    rows += _showdown_pair("QB", "High Backup", 110, "KC", 7000, 30)
    rows += _showdown_pair("RB", "Isiah Pacheco", 120, "KC", 6000, 10)
    rows += _showdown_pair("WR", "Xavier Worthy", 130, "KC", 5500, 10)
    rows += _showdown_pair("TE", "Travis Kelce", 140, "KC", 5200, 9)
    rows += _showdown_pair("QB", "Josh Allen", 200, "BUF", 8800, 15)
    rows += _showdown_pair("RB", "James Cook", 210, "BUF", 6500, 12)
    rows += _showdown_pair("WR", "Khalil Shakir", 220, "BUF", 6000, 11)
    rows += _showdown_pair("WR", "Keon Coleman", 230, "BUF", 5000, 10)
    rows += _showdown_pair("TE", "Dalton Kincaid", 240, "BUF", 5200, 9)

    def _solve(confirmed):
        optimizer = NFLShowdownOptimizer()
        optimizer.jitter_pct = (0.0, 0.0)
        optimizer.load_from_dataframe(pd.DataFrame(rows))
        optimizer.configure_showdown_rules(
            min_salary_cap=30_000,
            max_salary_cap=50_000,
            min_teams=2,
            game_script_correlation=False,
            max_kickers=None,
            allow_kicker_captain=False,
            ban_dst=True,
            flex_min_salary=1_200,
        )
        if confirmed is not None:
            optimizer.set_confirmed_quarterbacks(confirmed)
        optimizer.set_lineup_count(1)
        return optimizer.generate_lineups()[0]

    open_lineup = _solve(None)
    assert "High Backup" in set(open_lineup["Name"])

    capped = _solve([("Patrick Mahomes", "KC"), ("Josh Allen", "BUF")])
    assert "High Backup" not in set(capped["Name"])
    assert {"Patrick Mahomes", "Josh Allen"} & set(capped["Name"])


def test_showdown_gpp_preset_with_game_script():
    optimizer = get_nfl_optimizer(ContestFormat.SHOWDOWN)
    optimizer.load_from_csv(MOCK_SHOWDOWN_CSV)
    cfg = apply_preset(optimizer, ContestPreset.SHOWDOWN_GPP, lineup_count=3, jitter_seed=11)
    assert cfg.game_script_correlation is True
    assert optimizer.max_salary_cap == 49_800
    assert optimizer.game_script_correlation is True
    assert optimizer.max_k_dst_combined == 2
    assert optimizer.cpt_qb_pass_catchers == 2
    assert optimizer.max_repeating_players == 3
    assert optimizer.max_kickers == 1
    assert optimizer.allow_kicker_captain is False
    assert optimizer.max_rbs_per_team is None

    lineups = optimizer.generate_lineups()
    assert len(lineups) == 3
    for lineup in lineups:
        salary = int(lineup.attrs.get("salary", lineup["Salary"].sum()))
        assert 48_000 <= salary <= 49_800
        k_dst = lineup["Position"].isin(["K", "DST", "DEF"]).sum()
        assert k_dst <= 2
        assert int((lineup["Position"] == "K").sum()) <= 1

        cpt = lineup.loc[lineup["Slot"] == "CPT"].iloc[0]
        flex = lineup.loc[lineup["Slot"] == "FLEX"]
        assert cpt["Position"] in {"QB", "RB", "WR", "TE"}
        if cpt["Position"] == "QB":
            same_team_catchers = flex[
                (flex["Team"] == cpt["Team"]) & (flex["Position"].isin(["WR", "TE"]))
            ]
            assert len(same_team_catchers) >= 2
            bring_backs = flex[
                (flex["Team"] != cpt["Team"]) & (flex["Position"].isin(["WR", "TE", "RB"]))
            ]
            assert len(bring_backs) >= 1
        elif cpt["Position"] in {"WR", "TE"}:
            same_team_qb = flex[(flex["Team"] == cpt["Team"]) & (flex["Position"] == "QB")]
            assert len(same_team_qb) >= 1
        dst_teams = set(lineup.loc[lineup["Position"].isin(["DST", "DEF"]), "Team"])
        for dst_team in dst_teams:
            opponents = lineup[
                (lineup["Team"] != dst_team) & (lineup["Position"].isin(["QB", "K"]))
            ]
            assert opponents.empty


def _showdown_row(pos, name, pid, slot, team, salary, fppg) -> dict:
    return {
        "Position": pos,
        "Name + ID": f"{name} ({pid})",
        "Name": name,
        "ID": str(pid),
        "Roster Position": slot,
        "Salary": salary,
        "Game Info": "GB@CHI 09/10/2025 08:20PM ET",
        "TeamAbbrev": team,
        "AvgPointsPerGame": fppg,
        "Projection": fppg,
    }


def _showdown_pair(pos, name, flex_id, team, flex_salary, fppg) -> list:
    """CPT and FLEX rows for one player. CPT salary is 1.5x the FLEX salary."""
    return [
        _showdown_row(pos, name, flex_id + 1, "CPT", team, int(flex_salary * 1.5), fppg),
        _showdown_row(pos, name, flex_id, "FLEX", team, flex_salary, fppg),
    ]


def test_cpt_qb_pass_catcher_group_excludes_k_and_dst():
    """The quota is same-team WR and TE. RB, K, and DST do not count."""
    rows = []
    rows += _showdown_pair("QB", "Jordan Love", 100, "GB", 8000, 30)
    rows += _showdown_pair("WR", "Romeo Doubs", 110, "GB", 6000, 12)
    rows += _showdown_pair("TE", "Tucker Kraft", 120, "GB", 5000, 10)
    rows += _showdown_pair("RB", "Josh Jacobs", 130, "GB", 7000, 16)
    rows += _showdown_pair("K", "Brandon McManus", 140, "GB", 4000, 9)
    rows += _showdown_pair("DST", "Packers", 150, "GB", 3800, 8)
    rows += _showdown_pair("QB", "Caleb Williams", 200, "CHI", 7600, 18)
    rows += _showdown_pair("WR", "DJ Moore", 210, "CHI", 6400, 14)
    rows += _showdown_pair("TE", "Cole Kmet", 230, "CHI", 4000, 8)

    optimizer = NFLShowdownOptimizer()
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.load_from_dataframe(pd.DataFrame(rows))
    players = optimizer._players_for_optimize()

    captured = []

    class _Sink:
        def add_players_group(self, group):
            captured.append(group)

    optimizer._apply_game_script(_Sink(), players)

    cpt_qb = next(p for p in players if p.full_name == "Jordan Love" and "CPT" in p.positions)
    qb_groups = [g for g in captured if g.depends_on is cpt_qb]
    catchers = next(
        g
        for g in qb_groups
        if {NFLShowdownOptimizer._football_pos(p) for p in g.players} <= {"WR", "TE"}
        and all(p.team == "GB" for p in g.players)
    )
    bring_backs = next(
        g for g in qb_groups if all(p.team != "GB" for p in g.players)
    )
    catcher_names = {p.full_name for p in catchers.players}
    bring_names = {p.full_name for p in bring_backs.players}

    assert catcher_names == {"Romeo Doubs", "Tucker Kraft"}
    assert "Josh Jacobs" not in catcher_names
    assert "Brandon McManus" not in catcher_names
    assert "Packers" not in catcher_names
    assert all("FLEX" in p.positions for p in catchers.players)
    assert catchers.min_from_group == 2
    assert bring_names == {"DJ Moore", "Cole Kmet"}
    assert bring_backs.min_from_group == 1
    assert all(
        NFLShowdownOptimizer._football_pos(p) in {"WR", "TE", "RB"}
        for p in bring_backs.players
    )


def test_cpt_qb_lineup_cannot_use_kicker_or_dst_as_pass_catcher():
    """A Captain QB with no same-team WR, TE, or RB cannot be saved by K or DST."""
    rows = []
    rows += _showdown_pair("QB", "Jordan Love", 100, "GB", 8000, 40)
    rows += _showdown_pair("K", "Brandon McManus", 140, "GB", 3000, 22)
    rows += _showdown_pair("DST", "Packers", 150, "GB", 3000, 20)
    rows += _showdown_pair("QB", "Caleb Williams", 200, "CHI", 7000, 16)
    rows += _showdown_pair("WR", "DJ Moore", 210, "CHI", 6000, 15)
    rows += _showdown_pair("WR", "Rome Odunze", 220, "CHI", 5000, 12)
    rows += _showdown_pair("TE", "Cole Kmet", 230, "CHI", 4000, 9)

    optimizer = NFLShowdownOptimizer()
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.jitter_seed = 1
    optimizer.load_from_dataframe(pd.DataFrame(rows))
    optimizer.configure_showdown_rules(
        min_salary_cap=30_000,
        max_salary_cap=50_000,
        min_teams=2,
        game_script_correlation=True,
        cpt_qb_pass_catchers=1,
        max_k_dst_combined=2,
    )
    optimizer.set_lineup_count(1)

    lineup = optimizer.generate_lineups()[0]
    cpt = lineup.loc[lineup["Slot"] == "CPT"].iloc[0]
    flex = lineup.loc[lineup["Slot"] == "FLEX"]
    # Love's only GB teammates are K and DST, so he cannot be Captain.
    assert cpt["Name"] != "Jordan Love"
    if cpt["Position"] == "QB":
        catchers = flex[
            (flex["Team"] == cpt["Team"]) & (flex["Position"].isin(["WR", "TE"]))
        ]
        assert len(catchers) >= 1


def test_cpt_qb_must_bring_a_wr_or_te_not_kicker_or_dst():
    """A same-team kicker or defense does not satisfy the Captain-QB quota."""
    rows = []
    rows += _showdown_pair("QB", "Jordan Love", 100, "GB", 8000, 40)
    rows += _showdown_pair("WR", "Romeo Doubs", 110, "GB", 4500, 4)
    rows += _showdown_pair("TE", "Tucker Kraft", 120, "GB", 4000, 3)
    rows += _showdown_pair("K", "Brandon McManus", 140, "GB", 3000, 2)
    rows += _showdown_pair("DST", "Packers", 150, "GB", 3000, 2)
    rows += _showdown_pair("RB", "Josh Jacobs", 130, "GB", 5000, 6)
    rows += _showdown_pair("QB", "Caleb Williams", 200, "CHI", 6000, 10)
    rows += _showdown_pair("WR", "DJ Moore", 210, "CHI", 5500, 11)
    rows += _showdown_pair("TE", "Cole Kmet", 230, "CHI", 4000, 8)

    optimizer = NFLShowdownOptimizer()
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.jitter_seed = 1
    optimizer.load_from_dataframe(pd.DataFrame(rows))
    optimizer.configure_showdown_rules(
        min_salary_cap=30_000,
        max_salary_cap=50_000,
        min_teams=2,
        game_script_correlation=True,
        cpt_qb_pass_catchers=2,
        max_k_dst_combined=2,
    )
    optimizer.set_lineup_count(1)

    lineup = optimizer.generate_lineups()[0]
    cpt = lineup.loc[lineup["Slot"] == "CPT"].iloc[0]
    flex = lineup.loc[lineup["Slot"] == "FLEX"]
    assert cpt["Name"] == "Jordan Love"
    gb_names = set(flex.loc[flex["Team"] == "GB", "Name"])
    assert {"Romeo Doubs", "Tucker Kraft"} <= gb_names
    opp_skill = flex[
        (flex["Team"] != "GB") & (flex["Position"].isin(["WR", "TE", "RB"]))
    ]
    assert len(opp_skill) >= 1


def test_defense_excludes_opposing_qb_and_kicker():
    """A rostered defense cannot share the lineup with the other team's QB or K."""
    rows = []
    rows += _showdown_pair("DST", "Packers", 150, "GB", 4000, 80)
    rows += _showdown_pair("QB", "Caleb Williams", 200, "CHI", 7000, 30)
    rows += _showdown_pair("K", "Cairo Santos", 240, "CHI", 3000, 20)
    rows += _showdown_pair("QB", "Jordan Love", 100, "GB", 6000, 12)
    rows += _showdown_pair("WR", "Romeo Doubs", 110, "GB", 5000, 10)
    rows += _showdown_pair("RB", "Josh Jacobs", 130, "GB", 5000, 10)
    rows += _showdown_pair("WR", "DJ Moore", 210, "CHI", 5000, 11)
    rows += _showdown_pair("TE", "Cole Kmet", 230, "CHI", 4000, 8)

    optimizer = NFLShowdownOptimizer()
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.load_from_dataframe(pd.DataFrame(rows))
    optimizer.configure_showdown_rules(
        min_salary_cap=30_000,
        max_salary_cap=50_000,
        min_teams=2,
        game_script_correlation=False,
        allow_kicker_captain=True,
        max_k_dst_combined=2,
    )
    optimizer.set_lineup_count(1)
    lineup = optimizer.generate_lineups()[0]
    names = {str(name).strip() for name in lineup["Name"]}
    assert "Packers" in names
    assert "Caleb Williams" not in names
    assert "Cairo Santos" not in names


def test_showdown_lineup_has_at_most_one_kicker():
    """Opposing kickers cannot share a lineup when the one-kicker cap is on."""
    rows = []
    rows += _showdown_pair("K", "Spencer Shrader", 180, "IND", 5000, 40)
    rows += _showdown_pair("K", "Drew Stevens", 190, "WAS", 4800, 38)
    rows += _showdown_pair("QB", "Marcus Mariota", 100, "WAS", 7000, 12)
    rows += _showdown_pair("RB", "Jonathan Taylor", 110, "IND", 8000, 14)
    rows += _showdown_pair("WR", "Stefon Diggs", 120, "WAS", 6000, 11)
    rows += _showdown_pair("TE", "Tyler Warren", 130, "IND", 5500, 10)
    rows += _showdown_pair("WR", "Josh Downs", 140, "IND", 4000, 8)

    def _solve(max_kickers):
        optimizer = NFLShowdownOptimizer()
        optimizer.jitter_pct = (0.0, 0.0)
        optimizer.load_from_dataframe(pd.DataFrame(rows))
        optimizer.configure_showdown_rules(
            min_salary_cap=20_000,
            max_salary_cap=50_000,
            min_teams=2,
            game_script_correlation=False,
            allow_kicker_captain=True,
            max_k_dst_combined=None,
            max_kickers=max_kickers,
        )
        optimizer.set_lineup_count(1)
        return optimizer.generate_lineups()[0]

    capped = _solve(1)
    kickers = set(capped.loc[capped["Position"] == "K", "Name"])
    assert len(kickers) <= 1
    assert kickers != {"Spencer Shrader", "Drew Stevens"}

    uncapped = _solve(None)
    assert set(uncapped.loc[uncapped["Position"] == "K", "Name"]) == {
        "Spencer Shrader",
        "Drew Stevens",
    }


def test_kicker_captain_is_off_unless_allowed_and_dst_never_captains():
    rows = []
    rows += _showdown_pair("K", "Brandon McManus", 140, "GB", 4000, 40)
    rows += _showdown_pair("DST", "Packers", 150, "GB", 3600, 55)
    rows += _showdown_pair("QB", "Jordan Love", 100, "GB", 7000, 12)
    rows += _showdown_pair("WR", "Romeo Doubs", 110, "GB", 5000, 10)
    rows += _showdown_pair("RB", "Josh Jacobs", 130, "GB", 5500, 11)
    rows += _showdown_pair("QB", "Caleb Williams", 200, "CHI", 6800, 11)
    rows += _showdown_pair("WR", "DJ Moore", 210, "CHI", 5200, 10)
    rows += _showdown_pair("TE", "Cole Kmet", 230, "CHI", 4200, 8)

    def _captain(allow_kicker: bool) -> str:
        optimizer = NFLShowdownOptimizer()
        optimizer.jitter_pct = (0.0, 0.0)
        optimizer.load_from_dataframe(pd.DataFrame(rows))
        optimizer.configure_showdown_rules(
            min_salary_cap=30_000,
            max_salary_cap=50_000,
            min_teams=2,
            game_script_correlation=False,
            allow_kicker_captain=allow_kicker,
            max_k_dst_combined=2,
        )
        optimizer.set_lineup_count(1)
        lineup = optimizer.generate_lineups()[0]
        return str(lineup.loc[lineup["Slot"] == "CPT", "Name"].iloc[0])

    blocked = _captain(False)
    assert blocked != "Brandon McManus"
    assert blocked != "Packers"
    allowed = _captain(True)
    assert allowed == "Brandon McManus"


def test_preset_format_mismatch_raises():
    optimizer = NFLClassicOptimizer()
    with pytest.raises(ValueError):
        apply_preset(optimizer, ContestPreset.SHOWDOWN_GPP)


def test_tiered_lineup_counts_split():
    assert tiered_lineup_counts(5, 0.4) == (3, 2)
    assert tiered_lineup_counts(2, 0.4) == (1, 1)
    assert tiered_lineup_counts(1, 0.0) == (1, 0)
    assert tiered_lineup_counts(1, 1.0) == (0, 1)


def test_optimize_tiered_salaries_uses_both_windows():
    class _FakeOptimizer:
        def __init__(self) -> None:
            self.min_salary_cap = 11
            self.max_salary_cap = 22
            self.calls: list = []

        def set_min_salary_cap(self, value):
            self.min_salary_cap = value
            self.calls.append(("min", value))

        def set_max_salary_cap(self, value):
            self.max_salary_cap = value
            self.calls.append(("max", value))

        def generate_lineups(self, n):
            self.calls.append(("gen", n, self.min_salary_cap, self.max_salary_cap))
            return [f"{self.max_salary_cap}-{i}" for i in range(n)]

    fake = _FakeOptimizer()
    result = optimize_tiered_salaries(
        fake,
        5,
        leave_money_ratio=0.4,
        standard_cap=50_000,
        discount_cap=49_500,
        discount_floor=48_000,
    )
    assert result == ["50000-0", "50000-1", "50000-2", "49500-0", "49500-1"]
    generations = [call for call in fake.calls if call[0] == "gen"]
    assert generations == [
        ("gen", 3, STANDARD_TIER_FLOOR, 50_000),
        ("gen", 2, 48_000, 49_500),
    ]
    discount_calls = fake.calls[fake.calls.index(("gen", 3, STANDARD_TIER_FLOOR, 50_000)) + 1 :]
    assert discount_calls[0] == ("max", 49_500)
    assert discount_calls[1] == ("min", 48_000)
    assert fake.min_salary_cap == 11
    assert fake.max_salary_cap == 22


def test_optimize_tiered_salaries_respects_salary_bands():
    optimizer = NFLClassicOptimizer()
    optimizer.load_from_csv(MOCK_CSV)
    optimizer.set_qb_stack(enabled=False)
    optimizer.set_max_repeating_players(None)
    lineups = optimize_tiered_salaries(optimizer, total_n=4, leave_money_ratio=0.5)
    assert len(lineups) == 4
    salaries = [int(lu.attrs["salary"]) for lu in lineups]
    assert all(STANDARD_TIER_FLOOR <= s <= DK_SALARY_CAP for s in salaries[:2])
    assert all(48_000 <= s <= 49_500 for s in salaries[2:])
    for lineup in lineups:
        assert int(lineup.attrs["salary_remaining"]) == DK_SALARY_CAP - int(lineup.attrs["salary"])
    assert optimizer.min_salary_cap is None
    assert optimizer.max_salary_cap is None


def test_injury_status_ignores_name_plus_id_and_keeps_questionable():
    assert injury_status_for_player("Patrick Mahomes (1001)") == ""
    assert injury_status_for_player("Josh Allen (Q)") == "QUESTIONABLE"
    assert injury_status_for_player("Joe Mixon", "O") == "OUT"
    assert injury_status_for_player("Tee Higgins (IR)") == "IR"
    assert is_inactive_injury("OUT")
    assert is_inactive_injury("DOUBTFUL")
    assert not is_inactive_injury("QUESTIONABLE")
    assert not is_inactive_injury("")


def test_optimizer_excludes_players_who_will_not_play():
    df = parse_dk_csv(MOCK_CSV)
    out_idx = df.index[df["Position"] == "QB"][0]
    out_name = str(df.at[out_idx, "Name"])
    df.at[out_idx, "Injury"] = "OUT"
    df.at[out_idx, "Projection"] = 90.0

    excluded = NFLClassicOptimizer()
    excluded.set_qb_stack(enabled=False)
    excluded.set_exclude_inactive(True)
    excluded.load_from_dataframe(df)
    assert out_name not in set(excluded._player_df["Name"])
    lineup = excluded.generate_lineups(1)[0]
    assert out_name not in set(lineup["Name"])

    included = NFLClassicOptimizer()
    included.set_exclude_inactive(False)
    included.load_from_dataframe(df)
    assert out_name in set(included._player_df["Name"])


def test_parse_dk_status_column_marks_ir_and_out():
    from io import StringIO

    raw = (
        "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,"
        "TeamAbbrev,AvgPointsPerGame,Status\n"
        "WR,Alec Pierce (44331997),Alec Pierce,44331997,CPT,300,"
        "IND@WAS 10/04/2026 09:30AM ET,IND,6.1,IR\n"
        "WR,Alec Pierce (44331942),Alec Pierce,44331942,FLEX,200,"
        "IND@WAS 10/04/2026 09:30AM ET,IND,6.1,IR\n"
        "RB,Jonathan Taylor (44331958),Jonathan Taylor,44331958,CPT,17400,"
        "IND@WAS 10/04/2026 09:30AM ET,IND,21.5,\n"
        "WR,Questionable Guy (44331999),Questionable Guy,44331999,FLEX,5000,"
        "IND@WAS 10/04/2026 09:30AM ET,IND,10,Q\n"
    )
    df = parse_dk_csv(StringIO(raw), contest_format="showdown")
    statuses = dict(zip(df["Name"], df["Injury"]))
    assert statuses["Alec Pierce"] == "IR"
    assert statuses["Jonathan Taylor"] == ""
    assert statuses["Questionable Guy"] == "QUESTIONABLE"
    assert int(df["Injury"].map(is_inactive_injury).sum()) == 2


def test_parse_injury_indicator_column():
    from io import StringIO

    raw = (
        "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,"
        "TeamAbbrev,AvgPointsPerGame,Injury Indicator\n"
        "QB,Patrick Mahomes (1001),Patrick Mahomes,1001,QB,7200,"
        "KC@BUF 09/10/2025 08:20PM ET,KC,22.5,O\n"
        "RB,Josh Allen (1002),Josh Allen (Q),1002,RB,7400,"
        "KC@BUF 09/10/2025 08:20PM ET,BUF,23.0,\n"
    )
    df = parse_dk_csv(StringIO(raw))
    statuses = dict(zip(df["ID"], df["Injury"]))
    assert statuses["1001"] == "OUT"
    assert statuses["1002"] == "QUESTIONABLE"


def test_get_preset_config_labels():
    assert "Cash" in get_preset_config(ContestPreset.CLASSIC_CASH).label
    assert "Showdown GPP" in get_preset_config(ContestPreset.SHOWDOWN_GPP).label


def _game_script_pool() -> pd.DataFrame:
    """Two games, each with a full skill group, so buckets can avoid each other."""
    game_a = "DET@ARI 10/11/2026 01:00PM ET"
    game_b = "SF@SEA 10/11/2026 04:25PM ET"
    rows = []
    pid = 1
    # Game A projections sit above game B so the first lineup stays in DET@ARI
    # except for the defense, which has to come from the other game.
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
                _classic_player(
                    pos,
                    f"{team} {pos} {pid}",
                    pid,
                    team,
                    salary,
                    base + bump,
                    game,
                )
            )
            pid += 1
    return pd.DataFrame(rows)


def test_game_script_portfolio_stacks_caps_and_overlap():
    pool = _game_script_pool()
    assert games_from_pool(pool) == ["DET@ARI", "SF@SEA"]
    assert even_lineup_counts(3, 20) == [7, 7, 6]

    optimizer = NFLClassicOptimizer()
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.load_from_dataframe(pool)
    buckets = [
        GameScriptBucket("DET@ARI", 1, 49_000),
        GameScriptBucket("SF@SEA", 0, 50_000),
        GameScriptBucket("SF@SEA", 1, 47_000),
    ]
    lineups = optimize_game_scripts(optimizer, buckets, max_repeating_players=4)

    assert [lineup.attrs["game_script"] for lineup in lineups] == ["DET@ARI", "SF@SEA"]
    assert optimizer.flex_positions is None
    assert optimizer.exclude_vs_dst is False
    assert optimizer.enable_qb_stack is False
    assert optimizer.max_salary_cap is None
    assert optimizer.min_salary_cap is None

    caps = {"DET@ARI": 49_000, "SF@SEA": 47_000}
    for lineup in lineups:
        game = lineup.attrs["game_script"]
        away, home = game.split("@")
        qb = lineup.loc[lineup["Slot"] == "QB"].iloc[0]
        qb_team = str(qb["Team"])
        assert qb_team in {away, home}
        opponent = home if qb_team == away else away
        same_team = lineup[
            (lineup["Team"] == qb_team) & lineup["Position"].isin(["WR", "TE"])
        ]
        bring_back = lineup[
            (lineup["Team"] == opponent) & lineup["Position"].isin(["RB", "WR", "TE"])
        ]
        assert len(same_team) >= 1
        assert len(bring_back) >= 1
        flex_pos = str(lineup.loc[lineup["Slot"] == "FLEX", "Position"].iloc[0])
        assert flex_pos in {"RB", "WR"}
        dst = lineup.loc[lineup["Position"] == "DST"].iloc[0]
        facing = lineup[
            (lineup["Team"] == dst["Opponent"])
            & lineup["Position"].isin(["RB", "WR", "TE"])
        ]
        assert facing.empty
        assert int(lineup["Salary"].sum()) <= caps[game]

    shared = set(lineups[0]["ID"].astype(str)) & set(lineups[1]["ID"].astype(str))
    assert len(shared) <= 4

    summary = game_script_summary(lineups)
    assert summary["Game"].tolist() == ["DET@ARI", "SF@SEA"]
    assert summary["Lineups"].tolist() == [1, 1]
    exposure = player_exposure_table(lineups)
    assert list(exposure.columns) == [
        "Name",
        "Team",
        "Position",
        "Lineups",
        "Exposure %",
    ]
    assert exposure["Lineups"].is_monotonic_decreasing
    assert float(exposure["Exposure %"].max()) <= 100.0
    assert int(exposure.iloc[0]["Lineups"]) == int(exposure["Lineups"].max())
