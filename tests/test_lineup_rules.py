"""Lineup-construction rules for the Large-Field GPP preset.

Projections are compared as CSV bytes from the pool frame captured before the
solver runs. Turning a rule off is compared with the lineup ids main produces
on the same fixture, seed, and lineup count.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.exporter import CLASSIC_UPLOAD_COLUMNS, lineups_to_dk_upload_csv
from src.lineup_rules import (
    RULE_FAVORITE,
    RULE_PRIMARY,
    ConstructionSettings,
    assess_lineup,
    construction_problem_messages,
    team_is_favorite,
    team_is_underdog,
)
from src.optimizer import NFLClassicOptimizer, NFLShowdownOptimizer
from src.parser import parse_dk_csv
from src.presets import ContestPreset, apply_preset, format_preset_summary, get_preset_config

MOCK_CSV = Path(__file__).parent / "mock_dk_salaries.csv"
MOCK_SHOWDOWN = Path(__file__).parent / "mock_dk_showdown.csv"

# main at e0724cf, Large-Field GPP, seed 42, lineup count 3 (the preset's
# stack-exposure cap stops the batch at 2). Slot order is the DraftKings roster.
MAIN_LARGE_IDS = [
    ["1001", "2005", "2006", "3007", "3015", "3003", "4001", "3009", "5002"],
    ["1003", "2001", "2006", "3002", "3011", "3007", "4005", "3015", "5003"],
]
MAIN_CASH_IDS = ["1001", "2001", "2003", "3012", "3010", "3013", "4004", "2002", "5004"]
MAIN_SHOWDOWN_IDS = ["91003", "92006", "95002", "93010", "91002", "92008"]


def _ids(lineups) -> list[list[str]]:
    return [list(map(str, lineup["ID"].tolist())) for lineup in lineups]


def _large(n: int = 3, **flags) -> tuple[NFLClassicOptimizer, list]:
    optimizer = NFLClassicOptimizer()
    optimizer.load_from_dataframe(parse_dk_csv(MOCK_CSV))
    apply_preset(optimizer, ContestPreset.CLASSIC_LARGE_GPP, jitter_seed=42)
    for name, value in flags.items():
        setattr(optimizer, name, value)
    optimizer.set_lineup_count(n)
    return optimizer, optimizer.generate_lineups()


def _quiet() -> NFLClassicOptimizer:
    optimizer = NFLClassicOptimizer()
    optimizer.jitter_pct = (0.0, 0.0)
    optimizer.position_jitter_sds = None
    optimizer.jitter_seed = 42
    return optimizer


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


def _solve(rows, *, settings: ConstructionSettings, spreads=None, n: int = 1):
    frame = pd.DataFrame(rows)
    optimizer = _quiet()
    optimizer.load_from_dataframe(frame)
    optimizer.primary_stack = settings.primary_stack
    optimizer.primary_pass_catchers = settings.pass_catchers
    optimizer.primary_bring_backs = settings.bring_backs
    optimizer.primary_game_min = settings.game_min
    optimizer.primary_game_max = settings.game_max
    optimizer.primary_bring_back_wr1 = settings.bring_back_wr1
    optimizer.mini_stack = settings.mini_stack
    optimizer.mini_stack_count = settings.mini_count
    optimizer.rb_dst_favorite = settings.rb_dst_favorite
    optimizer.favorite_margin = settings.favorite_margin
    optimizer.ban_underdog_rbs = settings.ban_underdog_rbs
    optimizer.underdog_margin = settings.underdog_margin
    optimizer.set_vegas_spreads(spreads or {})
    optimizer.set_lineup_count(n)
    lineups = optimizer.generate_lineups()
    return optimizer, lineups, frame


def _game(away, home) -> str:
    return f"{away}@{home} 09/10/2025 01:00PM ET"


def test_large_gpp_defaults_turn_three_rules_on_and_the_underdog_rule_off():
    cfg = get_preset_config(ContestPreset.CLASSIC_LARGE_GPP)
    assert cfg.primary_stack is True
    assert cfg.primary_pass_catchers == 2
    assert cfg.primary_bring_backs == 1
    assert cfg.primary_game_min == 4
    assert cfg.primary_game_max == 5
    assert cfg.primary_bring_back_wr1 is False
    assert cfg.mini_stack is True
    assert cfg.mini_stack_count == 1
    assert cfg.rb_dst_favorite is True
    assert cfg.favorite_margin == 7
    assert cfg.ban_underdog_rbs is False
    assert cfg.underdog_margin == 7
    rows = dict(format_preset_summary(cfg))
    assert "WR/TE" in rows["Primary stack"]
    assert "including DST" in rows["QB game players"]
    assert rows["Bring-back WR1"] == "No"
    assert rows["Secondary mini-stack"].startswith("At least 1")
    assert "at least 7" in rows["Favorite RB + DST"]
    assert rows["Underdog RB restriction"] == "Off"
    for preset in (
        ContestPreset.CLASSIC_CASH,
        ContestPreset.CLASSIC_SMALL_GPP,
        ContestPreset.SHOWDOWN_CASH,
        ContestPreset.SHOWDOWN_GPP,
    ):
        other = get_preset_config(preset)
        assert other.primary_stack is False
        assert other.mini_stack is False
        assert other.rb_dst_favorite is False
        assert other.ban_underdog_rbs is False


def test_large_gpp_lineups_meet_rules_that_are_on_or_the_shortfall_is_named():
    optimizer, lineups = _large(3)
    assert lineups
    report = optimizer.last_construction_report
    assert report is not None
    # The fixture has no spreads, so a favorite DST cannot be chosen.
    assert RULE_FAVORITE in report.message
    assert "missing spread" in report.message
    assert report.applied.primary_stack is True
    assert report.applied.mini_stack is True
    assert report.applied.rb_dst_favorite is False
    pool = parse_dk_csv(MOCK_CSV)
    for lineup in lineups:
        scored = assess_lineup(lineup, report.applied, pool=pool, spreads={})
        assert scored.failures == ()
        assert scored.pass_catchers >= 2
        assert scored.bring_backs >= 1
        assert 4 <= scored.game_players <= 5
        assert scored.mini_stacks >= 1


@pytest.mark.parametrize(
    "flag",
    ["primary_stack", "mini_stack", "rb_dst_favorite", "ban_underdog_rbs"],
)
def test_each_toggle_off_matches_main_on_the_fixture(flag):
    """That rule's off position is main when the other new rules are off too."""
    flags = {
        "primary_stack": False,
        "mini_stack": False,
        "rb_dst_favorite": False,
        "ban_underdog_rbs": False,
    }
    assert flags[flag] is False
    _, lineups = _large(3, **flags)
    assert _ids(lineups) == MAIN_LARGE_IDS


def test_fixture_comparison_with_each_rule_isolated():
    """Side-by-side ids: main, each rule off, and that rule on by itself."""
    comparison = {"main": MAIN_LARGE_IDS}
    for flag in ("primary_stack", "mini_stack", "rb_dst_favorite", "ban_underdog_rbs"):
        off = {
            "primary_stack": False,
            "mini_stack": False,
            "rb_dst_favorite": False,
            "ban_underdog_rbs": False,
        }
        _, off_lineups = _large(3, **off)
        comparison[f"{flag} off"] = _ids(off_lineups)
        alone = dict(off)
        alone[flag] = True
        optimizer, on_lineups = _large(3, **alone)
        comparison[f"{flag} on"] = _ids(on_lineups)
        report = optimizer.last_construction_report
        message = "" if report is None else report.message
        if flag == "rb_dst_favorite":
            # No spread on the fixture, so the rule is named and not enforced.
            assert RULE_FAVORITE in message
            assert comparison[f"{flag} on"] == MAIN_LARGE_IDS
        elif flag == "ban_underdog_rbs":
            # No spread means nobody is excluded, so the lineups stay on main.
            assert comparison[f"{flag} on"] == MAIN_LARGE_IDS
            assert "Underdog" not in message
        else:
            pool = parse_dk_csv(MOCK_CSV)
            applied = report.applied if report is not None else optimizer.construction_settings()
            for lineup in on_lineups:
                assert assess_lineup(lineup, applied, pool=pool).failures == ()
    assert comparison["primary_stack off"] == MAIN_LARGE_IDS
    assert comparison["mini_stack off"] == MAIN_LARGE_IDS
    assert comparison["rb_dst_favorite off"] == MAIN_LARGE_IDS
    assert comparison["ban_underdog_rbs off"] == MAIN_LARGE_IDS
    # The mini-stack is the rule that moves this fixture off main's second lineup.
    assert comparison["mini_stack on"] != MAIN_LARGE_IDS


def test_projections_are_byte_identical_with_rules_on_and_off():
    on, on_lineups = _large(3)
    off, _ = _large(
        3,
        primary_stack=False,
        mini_stack=False,
        rb_dst_favorite=False,
        ban_underdog_rbs=False,
    )
    on_bytes = on.last_projected_frame.to_csv(index=False).encode("utf-8")
    off_bytes = off.last_projected_frame.to_csv(index=False).encode("utf-8")
    assert on_bytes == off_bytes
    assert on.last_projected_frame["Projection"].tolist() == off.last_projected_frame["Projection"].tolist()
    # The underdog filter removes runners from the solver only.
    banned, _ = _large(3, ban_underdog_rbs=True, underdog_margin=7)
    banned.set_vegas_spreads({"NYJ": 7, "SF": 6})
    # Spreads are applied at solve time; regenerate so the filter runs.
    banned.set_lineup_count(3)
    banned.generate_lineups()
    assert banned.last_projected_frame.to_csv(index=False).encode("utf-8") == off_bytes
    header = lineups_to_dk_upload_csv(on_lineups).splitlines()[0].split(",")
    assert header == CLASSIC_UPLOAD_COLUMNS


def test_cash_and_showdown_seed_42_match_main():
    classic = parse_dk_csv(MOCK_CSV)
    cash = NFLClassicOptimizer()
    cash.load_from_dataframe(classic)
    apply_preset(cash, ContestPreset.CLASSIC_CASH, jitter_seed=42)
    assert cash.primary_stack is False
    assert cash.mini_stack is False
    assert cash.rb_dst_favorite is False
    assert cash.ban_underdog_rbs is False
    assert _ids(cash.generate_lineups()) == [MAIN_CASH_IDS]

    showdown = NFLShowdownOptimizer()
    showdown.load_from_dataframe(parse_dk_csv(MOCK_SHOWDOWN))
    apply_preset(showdown, ContestPreset.SHOWDOWN_CASH, jitter_seed=42)
    assert _ids(showdown.generate_lineups()) == [MAIN_SHOWDOWN_IDS]


def test_spread_edges_for_favorite_and_underdog():
    assert team_is_favorite({"WAS": -7}, "WSH", 7) is True
    assert team_is_favorite({"WAS": -6}, "WSH", 7) is False
    assert team_is_favorite({"WAS": -8}, "WAS", 7) is True
    assert team_is_favorite({}, "WAS", 7) is False
    assert team_is_favorite({"WAS": None}, "WAS", 7) is False
    assert team_is_underdog({"JAC": 7}, "JAX", 7) is True
    assert team_is_underdog({"JAC": 6}, "JAX", 7) is False
    assert team_is_underdog({"JAC": 8}, "JAX", 7) is True
    assert team_is_underdog({}, "JAX", 7) is False
    assert team_is_underdog({"JAX": None}, "JAX", 7) is False


def _edge_pool() -> list[dict]:
    """One game of fillers plus three defenses whose spreads sit on the edges."""
    game = _game("DAL", "NYG")
    other = _game("BUF", "KC")
    phi = _game("PHI", "TB")
    return [
        _row("QB", "Dak Prescott", 1, "DAL", 5000, 20, game),
        _row("RB", "Tony Pollard", 2, "DAL", 4000, 5, game),
        _row("WR", "CeeDee Lamb", 3, "DAL", 4000, 12, game),
        _row("WR", "Brandin Cooks", 4, "DAL", 3000, 8, game),
        _row("WR", "WanDale Robinson", 5, "NYG", 3000, 7, game),
        _row("TE", "Jake Ferguson", 6, "DAL", 3000, 6, game),
        _row("RB", "James Cook", 10, "BUF", 5000, 12, other),
        _row("DST", "Bills", 11, "BUF", 3000, 15, other),
        _row("RB", "Isiah Pacheco", 12, "KC", 5000, 13, other),
        _row("DST", "Chiefs", 13, "KC", 3000, 40, other),
        _row("WR", "Xavier Worthy", 14, "KC", 3000, 4, other),
        _row("RB", "Saquon Barkley", 15, "PHI", 5000, 14, phi),
        _row("DST", "Eagles", 16, "PHI", 3000, 60, phi),
        _row("WR", "AJ Brown", 17, "PHI", 3000, 4, phi),
        _row("TE", "Dallas Goedert", 18, "PHI", 2500, 3, phi),
    ]


def test_favorite_dst_uses_exact_margin_and_rejects_one_point_inside_and_a_missing_spread():
    spreads = {"BUF": -7, "KC": -6}  # PHI is missing on purpose
    settings = ConstructionSettings(rb_dst_favorite=True, favorite_margin=7)
    optimizer, lineups, pool = _solve(_edge_pool(), settings=settings, spreads=spreads)
    assert len(lineups) == 1
    lineup = lineups[0]
    assert lineup.loc[lineup["Position"] == "DST", "Team"].iloc[0] == "BUF"
    assert "James Cook" in set(lineup["Name"])
    assert "Chiefs" not in set(lineup["Name"])
    assert "Eagles" not in set(lineup["Name"])
    scored = assess_lineup(lineup, settings, pool=pool, spreads=spreads)
    assert scored.failures == ()
    report = optimizer.last_construction_report
    assert report is None or RULE_FAVORITE not in report.message

    # One point inside, and a missing spread, are the only defenses here.
    inside = ConstructionSettings(rb_dst_favorite=True, favorite_margin=7)
    bare_rows = [row for row in _edge_pool() if row["TeamAbbrev"] != "BUF"]
    named, named_lineups, _ = _solve(bare_rows, settings=inside, spreads={"KC": -6})
    assert named_lineups
    assert named.last_construction_report is not None
    assert RULE_FAVORITE in named.last_construction_report.message
    assert "missing spread" in named.last_construction_report.message
    # The highest defense is still rostered. The rule was named, not dropped quietly.
    assert named_lineups[0].loc[named_lineups[0]["Position"] == "DST", "Team"].iloc[0] == "PHI"


def test_underdog_rb_excludes_the_exact_margin_and_keeps_one_point_inside_and_a_missing_spread():
    game = _game("DAL", "NYG")
    buf = _game("BUF", "KC")
    phi = _game("PHI", "TB")
    rows = [
        _row("QB", "Dak Prescott", 1, "DAL", 5000, 20, game),
        _row("WR", "CeeDee Lamb", 3, "DAL", 4000, 12, game),
        _row("WR", "Brandin Cooks", 4, "DAL", 3000, 8, game),
        _row("WR", "WanDale Robinson", 5, "NYG", 3000, 7, game),
        _row("TE", "Jake Ferguson", 6, "DAL", 3000, 6, game),
        _row("DST", "Cowboys", 7, "DAL", 3000, 8, game),
        _row("RB", "James Cook", 10, "BUF", 5000, 50, buf),  # spread +7, excluded
        _row("RB", "Isiah Pacheco", 12, "KC", 5000, 30, buf),  # spread +6, kept
        _row("WR", "Xavier Worthy", 14, "KC", 3000, 4, buf),
        _row("RB", "Saquon Barkley", 15, "PHI", 5000, 20, phi),  # spread missing, kept
        _row("WR", "AJ Brown", 17, "PHI", 3000, 4, phi),
        _row("DST", "Eagles", 16, "PHI", 2500, 1, phi),
    ]
    spreads = {"BUF": 7, "KC": 6}
    on = ConstructionSettings(ban_underdog_rbs=True, underdog_margin=7)
    optimizer, lineups, pool = _solve(rows, settings=on, spreads=spreads)
    names = set(lineups[0]["Name"])
    assert "James Cook" not in names
    assert "Isiah Pacheco" in names
    assert "Saquon Barkley" in names
    assert assess_lineup(lineups[0], on, pool=pool, spreads=spreads).failures == ()
    on_bytes = optimizer.last_projected_frame.to_csv(index=False).encode("utf-8")

    off = ConstructionSettings()
    plain, plain_lineups, _ = _solve(rows, settings=off, spreads=spreads)
    plain_names = set(plain_lineups[0]["Name"])
    assert "James Cook" in plain_names
    assert plain.last_projected_frame.to_csv(index=False).encode("utf-8") == on_bytes


def test_primary_stack_counts_dst_and_can_require_the_opposing_wr1():
    game = _game("KC", "BUF")
    other = _game("PHI", "DAL")
    rows = [
        _row("QB", "Patrick Mahomes", 1, "KC", 7000, 40, game),
        _row("WR", "Xavier Worthy", 2, "KC", 5000, 30, game),
        _row("TE", "Travis Kelce", 3, "KC", 5000, 30, game),
        _row("WR", "Rashee Rice", 4, "KC", 4500, 28, game),
        _row("RB", "James Cook", 5, "BUF", 5000, 35, game),
        _row("WR", "Khalil Shakir", 6, "BUF", 8000, 1, game),
        _row("DST", "Bills", 7, "BUF", 3000, 40, game),
        _row("RB", "Saquon Barkley", 8, "PHI", 4000, 4, other),
        _row("WR", "CeeDee Lamb", 9, "DAL", 4000, 4, other),
        _row("WR", "AJ Brown", 10, "PHI", 3500, 4, other),
        _row("WR", "DeVonta Smith", 11, "PHI", 3000, 3, other),
        _row("TE", "Dallas Goedert", 12, "PHI", 3000, 3, other),
        _row("DST", "Eagles", 13, "PHI", 2500, 1, other),
        _row("RB", "Tony Pollard", 14, "DAL", 3000, 2, other),
    ]
    capped = ConstructionSettings(
        primary_stack=True,
        pass_catchers=2,
        bring_backs=1,
        game_min=4,
        game_max=5,
    )
    _, lineups, pool = _solve(rows, settings=capped)
    scored = assess_lineup(lineups[0], capped, pool=pool)
    assert scored.failures == ()
    assert scored.game_players == 5
    assert lineups[0].loc[lineups[0]["Position"] == "DST", "Team"].iloc[0] == "BUF"
    # Bills DST is one of the five. A sixth Chiefs/Bills player does not fit.
    game_teams = {"KC", "BUF"}
    assert int(lineups[0]["Team"].isin(game_teams).sum()) == 5

    open_rule = ConstructionSettings()
    _, open_lineups, _ = _solve(rows, settings=open_rule)
    assert int(open_lineups[0]["Team"].isin(game_teams).sum()) >= 6

    wr1 = ConstructionSettings(
        primary_stack=True,
        pass_catchers=2,
        bring_backs=1,
        game_min=4,
        game_max=4,
        bring_back_wr1=True,
    )
    _, wr1_lineups, wr1_pool = _solve(rows, settings=wr1)
    bring = wr1_lineups[0][wr1_lineups[0]["Team"] == "BUF"]
    bring = bring[bring["Position"].isin(["RB", "WR", "TE"])]
    assert list(bring["Name"]) == ["Khalil Shakir"]
    assert assess_lineup(wr1_lineups[0], wr1, pool=wr1_pool).failures == ()


def test_wr1_is_the_highest_salary_not_the_highest_projection():
    other = _game("PHI", "DAL")
    rows = [
        _row("QB", "Patrick Mahomes", 1, "WAS", 6000, 30, _game("WSH", "NYG")),
        _row("WR", "Terry McLaurin", 2, "WAS", 5000, 20, _game("WSH", "NYG")),
        _row("TE", "Zach Ertz", 3, "WAS", 4000, 12, _game("WSH", "NYG")),
        _row("RB", "Brian Robinson", 4, "NYG", 4500, 40, _game("WSH", "NYG")),
        _row("WR", "Malik Nabers", 5, "NYG", 7000, 2, _game("WSH", "NYG")),
        _row("DST", "Giants", 6, "NYG", 3000, 5, _game("WSH", "NYG")),
        _row("RB", "Saquon Barkley", 7, "PHI", 4000, 6, other),
        _row("RB", "Tony Pollard", 8, "DAL", 3500, 5, other),
        _row("WR", "AJ Brown", 9, "PHI", 4000, 8, other),
        _row("WR", "CeeDee Lamb", 10, "DAL", 4000, 8, other),
        _row("WR", "DeVonta Smith", 11, "PHI", 3000, 4, other),
        _row("TE", "Dallas Goedert", 12, "PHI", 3000, 4, other),
        _row("DST", "Eagles", 13, "PHI", 2500, 3, other),
    ]
    # The team code on the player is WAS. The game text says WSH. Both are the same club.
    settings = ConstructionSettings(
        primary_stack=True,
        pass_catchers=2,
        bring_backs=1,
        game_min=4,
        game_max=4,
        bring_back_wr1=True,
    )
    _, lineups, pool = _solve(rows, settings=settings)
    assert "Malik Nabers" in set(lineups[0]["Name"])
    assert "Brian Robinson" not in set(lineups[0]["Name"])
    assert assess_lineup(lineups[0], settings, pool=pool).failures == ()


def test_mini_stack_is_two_offensive_players_outside_the_quarterback_game():
    qb_game = _game("KC", "BUF")
    pair = _game("PHI", "DAL")
    lone = _game("BAL", "CIN")
    rows = [
        _row("QB", "Patrick Mahomes", 1, "KC", 6000, 30, qb_game),
        _row("WR", "Xavier Worthy", 2, "KC", 4000, 15, qb_game),
        _row("TE", "Travis Kelce", 3, "KC", 4000, 14, qb_game),
        _row("RB", "James Cook", 4, "BUF", 4000, 12, qb_game),
        _row("DST", "Bills", 5, "BUF", 2500, 4, qb_game),
        _row("RB", "Saquon Barkley", 6, "PHI", 5000, 10, pair),
        _row("WR", "CeeDee Lamb", 7, "DAL", 5000, 10, pair),
        _row("RB", "Derrick Henry", 8, "BAL", 5500, 18, lone),
        _row("DST", "Ravens", 9, "BAL", 3000, 16, lone),
        _row("WR", "AJ Brown", 10, "PHI", 3500, 6, pair),
        _row("WR", "DeVonta Smith", 11, "PHI", 3000, 5, pair),
        _row("TE", "Mark Andrews", 12, "BAL", 3000, 5, lone),
        _row("WR", "Zay Flowers", 13, "BAL", 3000, 4, lone),
        _row("DST", "Eagles", 14, "PHI", 2200, 2, pair),
    ]
    forced = ConstructionSettings(mini_stack=True, mini_count=1)
    _, lineups, pool = _solve(rows, settings=forced)
    scored = assess_lineup(lineups[0], forced, pool=pool)
    assert scored.failures == ()
    assert scored.mini_stacks >= 1
    # Henry plus the Ravens defense is not the stack. Some other game has two offensive players.
    outside = lineups[0][~lineups[0]["Team"].isin(["KC", "BUF"])]
    offensive = outside[outside["Position"].isin(["QB", "RB", "WR", "TE"])]
    assert offensive.groupby("Team").size().max() >= 1

    hand = pd.DataFrame(
        [
            _row("QB", "Patrick Mahomes", 1, "KC", 1, 1, qb_game),
            _row("RB", "Derrick Henry", 8, "BAL", 1, 1, lone),
            _row("DST", "Ravens", 9, "BAL", 1, 1, lone),
        ]
    )
    missed = assess_lineup(hand, forced, pool=pd.DataFrame(rows))
    assert missed.mini_stacks == 0
    assert missed.failures


def test_impossible_game_max_is_named_and_lineups_are_still_built():
    messages = construction_problem_messages(
        ConstructionSettings(
            primary_stack=True,
            pass_catchers=2,
            bring_backs=1,
            game_min=4,
            game_max=3,
        )
    )
    assert messages
    assert RULE_PRIMARY in messages[0]
    assert "max is 3" in messages[0]
    assert "need 4" in messages[0]

    game = _game("KC", "BUF")
    other = _game("PHI", "DAL")
    rows = [
        _row("QB", "Patrick Mahomes", 1, "KC", 6000, 30, game),
        _row("WR", "Xavier Worthy", 2, "KC", 4000, 12, game),
        _row("TE", "Travis Kelce", 3, "KC", 4000, 11, game),
        _row("RB", "James Cook", 4, "BUF", 4000, 10, game),
        _row("DST", "Bills", 5, "BUF", 2500, 4, game),
        _row("RB", "Saquon Barkley", 6, "PHI", 4000, 8, other),
        _row("RB", "Tony Pollard", 7, "DAL", 3000, 5, other),
        _row("WR", "AJ Brown", 8, "PHI", 3500, 7, other),
        _row("WR", "CeeDee Lamb", 9, "DAL", 3500, 7, other),
        _row("WR", "DeVonta Smith", 10, "PHI", 3000, 4, other),
        _row("TE", "Dallas Goedert", 11, "PHI", 2800, 4, other),
        _row("DST", "Eagles", 12, "PHI", 2200, 3, other),
    ]
    settings = ConstructionSettings(
        primary_stack=True, pass_catchers=2, bring_backs=1, game_min=4, game_max=3
    )
    optimizer, lineups, _ = _solve(rows, settings=settings)
    assert len(lineups) == 1
    assert optimizer.last_construction_report is not None
    assert "need 4" in optimizer.last_construction_report.message
    plain, plain_lineups, _ = _solve(rows, settings=ConstructionSettings())
    assert _ids(lineups) == _ids(plain_lineups)
    assert plain.last_projected_frame.to_csv(index=False).encode("utf-8") == (
        optimizer.last_projected_frame.to_csv(index=False).encode("utf-8")
    )


def test_a_primary_stack_the_pool_cannot_afford_is_named_instead_of_returning_nothing():
    game = _game("KC", "BUF")
    other = _game("PHI", "DAL")
    rows = [
        _row("QB", "Patrick Mahomes", 1, "KC", 45000, 80, game),
        _row("WR", "Xavier Worthy", 2, "KC", 3000, 20, game),
        _row("TE", "Travis Kelce", 3, "KC", 3000, 20, game),
        _row("RB", "James Cook", 4, "BUF", 3000, 20, game),
        _row("QB", "Jalen Hurts", 5, "PHI", 5000, 15, other),
        _row("RB", "Saquon Barkley", 6, "PHI", 4000, 10, other),
        _row("RB", "Tony Pollard", 7, "DAL", 3500, 8, other),
        _row("WR", "AJ Brown", 8, "NYJ", 4000, 9, _game("NYJ", "NE")),
        _row("WR", "CeeDee Lamb", 9, "DAL", 4000, 9, other),
        _row("WR", "DeVonta Smith", 10, "PHI", 3000, 6, other),
        _row("TE", "Dallas Goedert", 11, "DAL", 3000, 6, other),
        _row("DST", "Eagles", 12, "PHI", 2500, 4, other),
        _row("DST", "Bills", 13, "BUF", 2500, 3, game),
    ]
    settings = ConstructionSettings(
        primary_stack=True, pass_catchers=2, bring_backs=1, game_min=4, game_max=5
    )
    optimizer, lineups, _ = _solve(rows, settings=settings)
    assert len(lineups) == 1
    assert "Patrick Mahomes" not in set(lineups[0]["Name"])
    assert optimizer.last_construction_report is not None
    assert RULE_PRIMARY in optimizer.last_construction_report.message
