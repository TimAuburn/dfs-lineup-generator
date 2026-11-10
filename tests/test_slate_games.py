"""Game table, on/off pool filter, and pre-solve blocks.

Lineup ids with every game on are the same ones main produces for seed 42.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.exporter import CLASSIC_UPLOAD_COLUMNS, lineups_to_dk_upload_csv
from src.lineup_rules import ConstructionSettings, active_game_shortfall
from src.optimizer import NFLClassicOptimizer, NFLShowdownOptimizer, games_from_pool
from src.parser import parse_dk_csv
from src.player_pool import game_label
from src.presets import ContestPreset, apply_preset
from src.slate_games import (
    MISSING,
    ODDS_SOURCE,
    disabled_for_slate,
    display_spread,
    display_total,
    forecast_source_caption,
    games_table_frame,
    generation_blockers,
    generation_pool_for_games,
    matchups_from_pool,
    odds_source_caption,
    off_game_blocks,
    pool_without_games,
    spreads_for_pool,
    with_disabled,
)

MOCK_CSV = Path(__file__).parent / "mock_dk_salaries.csv"
MOCK_SHOWDOWN = Path(__file__).parent / "mock_dk_showdown.csv"

# main at 6a2649c, seed 42. Cash and Showdown are one lineup. Large-Field GPP
# is compared with the unfiltered solver, which is main's result.
MAIN_CASH_IDS = ["1001", "2001", "2003", "3012", "3010", "3013", "4004", "2002", "5004"]
MAIN_SHOWDOWN_IDS = ["91003", "92006", "95002", "93010", "91002", "92008"]


def _ids(lineups) -> list[list[str]]:
    return [[str(player_id) for player_id in lineup["ID"].tolist()] for lineup in lineups]


def _classic(frame, preset, n=None):
    optimizer = NFLClassicOptimizer()
    optimizer.load_from_dataframe(frame)
    apply_preset(optimizer, preset, jitter_seed=42)
    if n is not None:
        optimizer.set_lineup_count(n)
    return optimizer, optimizer.generate_lineups()


def _player(name, team, pid, game, **extra) -> dict:
    row = {
        "Name": name,
        "TeamAbbrev": team,
        "Position": extra.pop("position", "WR"),
        "ID": str(pid),
        "Game Info": game,
        "Projection": 10.0,
        "Salary": 5000,
        "Include": True,
        "Lock": False,
    }
    row.update(extra)
    return row


def test_table_shows_source_total_and_favorite():
    pool = parse_dk_csv(MOCK_CSV)
    matchups = matchups_from_pool(pool)
    kc = next(game for game in matchups if game.key == "KC@BUF")
    assert kc.matchup == "KC @ BUF"
    assert kc.kickoff == "09/10/2025 08:20PM ET"
    lines = {
        "KC": {"total": 47.5, "spread": -3.5, "game": "KC@BUF"},
        "BUF": {"total": 47.5, "spread": 3.5, "game": "KC@BUF"},
        "PHI": {"total": 51.0, "spread": -7.0, "game": "PHI@DAL"},
        "DAL": {"total": 51.0, "spread": 7.0, "game": "PHI@DAL"},
    }
    assert display_total(lines, "KC", "BUF") == "47.5"
    assert display_spread(lines, "KC", "BUF") == "KC -3.5"
    assert display_spread(lines, "PHI", "DAL") == "PHI -7.0"
    frame = games_table_frame(matchups, lines, {"KC@BUF": "sunny"}, disabled=[])
    kc_row = frame.loc[frame["Matchup"].eq("KC @ BUF")].iloc[0]
    phi_row = frame.loc[frame["Matchup"].eq("PHI @ DAL")].iloc[0]
    assert kc_row["Total"] == "47.5"
    assert kc_row["Spread"] == "KC -3.5"
    assert phi_row["Total"] == "51.0"
    assert phi_row["Spread"] == "PHI -7.0"
    assert bool(kc_row["On"]) is True
    assert ODDS_SOURCE == "The Odds API"
    assert ODDS_SOURCE in odds_source_caption(1_700_000_000.0)
    assert "Fetched:" in odds_source_caption(1_700_000_000.0)
    assert "not fetched" in odds_source_caption(None)
    assert "National Weather Service hourly" in forecast_source_caption(None)


def test_missing_total_or_spread_is_missing_and_not_zero():
    assert display_total({}, "KC", "BUF") == MISSING
    assert display_spread({}, "KC", "BUF") == MISSING
    assert display_total({"KC": {"total": None, "spread": -3}}, "KC", "BUF") == MISSING
    assert display_spread({"KC": {"total": 44, "spread": None}, "BUF": {"total": 44}}, "KC", "BUF") == MISSING
    assert display_total(
        {"KC": {"total": 44.0, "spread": -3}, "BUF": {"spread": 3}},
        "KC",
        "BUF",
    ) == "44.0"
    assert display_spread(
        {"KC": {"total": None, "spread": -3.0}, "BUF": {"total": None, "spread": 3.0}},
        "KC",
        "BUF",
    ) == "KC -3.0"
    assert display_total(
        {"KC": {"total": 40, "spread": -1}, "BUF": {"total": 50, "spread": 1}},
        "KC",
        "BUF",
    ) == MISSING
    assert display_spread(
        {"KC": {"spread": -3}, "BUF": {"spread": -3}},
        "KC",
        "BUF",
    ) == MISSING
    pool = parse_dk_csv(MOCK_CSV)
    frame = games_table_frame(matchups_from_pool(pool), {}, {})
    assert set(frame["Total"]) == {MISSING}
    assert set(frame["Spread"]) == {MISSING}
    assert (frame["Total"] == "0").sum() == 0
    assert (frame["Spread"] == "0").sum() == 0
    assert MISSING not in {"0", "0.0"}


def test_off_game_is_absent_from_lineups_and_the_upload():
    pool = parse_dk_csv(MOCK_CSV)
    off = "KC@BUF"
    off_ids = set(pool.loc[pool["Game Info"].map(game_label).eq(off), "ID"].astype(str))
    assert "5001" in off_ids  # Bills DST
    filtered = pool_without_games(pool, [off])
    assert off_ids.isdisjoint(set(filtered["ID"].astype(str)))
    kept = pool.loc[~pool["Game Info"].map(game_label).eq(off), "Projection"]
    assert list(filtered["Projection"]) == list(kept)
    assert list(pool["Projection"]) == list(parse_dk_csv(MOCK_CSV)["Projection"])
    _optimizer, lineups = _classic(filtered, ContestPreset.CLASSIC_CASH)
    used = {player_id for lineup in _ids(lineups) for player_id in lineup}
    assert off_ids.isdisjoint(used)
    upload = lineups_to_dk_upload_csv(lineups)
    header, *body = upload.splitlines()
    assert header.split(",") == CLASSIC_UPLOAD_COLUMNS
    assert lineups_to_dk_upload_csv([]).splitlines()[0].split(",") == CLASSIC_UPLOAD_COLUMNS
    for player_id in off_ids:
        assert player_id not in upload
        assert f"({player_id})" not in "\n".join(body)


def test_locked_and_min_exposure_players_in_an_off_game_block():
    game = "KC@BUF 09/10/2025 08:20PM ET"
    locked = pd.DataFrame([_player("Patrick Mahomes", "KC", "1001", game, Lock=True, position="QB")])
    messages = off_game_blocks(locked, ["KC@BUF"])
    assert len(messages) == 1
    assert "Patrick Mahomes (KC)" in messages[0]
    assert "KC@BUF" in messages[0]
    assert "locked" in messages[0]

    pinned = pd.DataFrame(
        [_player("Xavier Worthy", "KC", "3009", game, ProjectionOverride=11.5)]
    )
    pin_message = off_game_blocks(pinned, ["KC @ BUF"])[0]
    assert "Xavier Worthy (KC)" in pin_message
    assert "pinned" in pin_message
    assert "KC@BUF" in pin_message

    targeted = pd.DataFrame([_player("Chris Godwin", "TB", "3012", "TB@ATL 09/10/2025 01:00PM ET")])
    minimum = off_game_blocks(targeted, ["TB@ATL"], min_exposure_pct=25)[0]
    assert "Chris Godwin (TB)" in minimum
    assert "TB@ATL" in minimum
    assert "min-exposure" in minimum
    assert off_game_blocks(targeted, ["TB@ATL"], min_exposure_pct=0) == ()
    assert off_game_blocks(targeted, ["TB@ATL"], min_exposure_pct=None) == ()

    removed = locked.copy()
    removed["Include"] = False
    assert off_game_blocks(removed, ["KC@BUF"]) == ()
    assert off_game_blocks(locked, ["PHI@DAL"]) == ()

    blockers = generation_blockers(
        locked,
        ["KC@BUF"],
        showdown=False,
        min_exposure_pct=None,
    )
    assert "Patrick Mahomes (KC)" in blockers[0]


def test_all_games_on_matches_main_seed_42():
    classic = parse_dk_csv(MOCK_CSV)
    assert pool_without_games(classic, []) is classic
    via = generation_pool_for_games(classic, [], showdown=False)
    _cash, cash_lineups = _classic(classic, ContestPreset.CLASSIC_CASH)
    _cash_via, cash_via = _classic(via, ContestPreset.CLASSIC_CASH)
    assert _ids(cash_lineups) == [MAIN_CASH_IDS]
    assert _ids(cash_via) == [MAIN_CASH_IDS]

    showdown = parse_dk_csv(MOCK_SHOWDOWN)
    assert pool_without_games(showdown, ["KC@BUF"], showdown=True) is showdown
    assert generation_blockers(showdown, ["KC@BUF"], showdown=True) == ()
    optimizer = NFLShowdownOptimizer()
    optimizer.load_from_dataframe(showdown)
    apply_preset(optimizer, ContestPreset.SHOWDOWN_CASH, jitter_seed=42)
    assert _ids(optimizer.generate_lineups()) == [MAIN_SHOWDOWN_IDS]

    _large, large_lineups = _classic(classic, ContestPreset.CLASSIC_LARGE_GPP, n=3)
    _large_via, large_via = _classic(
        generation_pool_for_games(classic, [], showdown=False),
        ContestPreset.CLASSIC_LARGE_GPP,
        n=3,
    )
    assert _ids(large_lineups) == _ids(large_via)
    assert large_lineups


def test_too_few_games_for_a_mini_stack_is_named_before_the_solve():
    settings = ConstructionSettings(primary_stack=True, mini_stack=True, mini_count=1)
    message = active_game_shortfall(1, settings)[0]
    assert "Secondary mini-stack" in message
    assert "mini-stack needs a game other than the primary one" in message
    assert active_game_shortfall(2, settings) == ()
    assert active_game_shortfall(1, ConstructionSettings()) == ()

    pool = parse_dk_csv(MOCK_CSV)
    games = games_from_pool(pool)
    kept = games[:1]
    disabled = games[1:]
    blockers = generation_blockers(
        pool,
        disabled,
        showdown=False,
        settings=settings,
    )
    assert any("mini-stack needs a game other than the primary one" in item for item in blockers)
    # A one-game slate with nothing turned off keeps the old path.
    only = pool.loc[pool["Game Info"].map(game_label).eq(kept[0])].copy()
    assert generation_blockers(only, [], showdown=False, settings=settings) == ()


def test_team_stack_limit_counts_only_games_left_on():
    pool = parse_dk_csv(MOCK_CSV)
    games = games_from_pool(pool)
    filtered = pool_without_games(pool, games[4:])
    optimizer = NFLClassicOptimizer()
    optimizer.load_from_dataframe(filtered)
    apply_preset(optimizer, ContestPreset.CLASSIC_LARGE_GPP, jitter_seed=42)
    optimizer.set_lineup_count(1)
    try:
        optimizer.generate_lineups()
    except Exception:
        pass
    assert optimizer.last_game_count == 4
    assert optimizer.last_stack_exposure == pytest.approx(0.30)
    full = NFLClassicOptimizer()
    full.load_from_dataframe(pool)
    apply_preset(full, ContestPreset.CLASSIC_LARGE_GPP, jitter_seed=42)
    full.set_lineup_count(1)
    try:
        full.generate_lineups()
    except Exception:
        pass
    assert full.last_game_count == len(games)
    assert full.last_stack_exposure == pytest.approx(0.40)


def test_game_selection_is_keyed_to_the_slate():
    store = with_disabled({}, "slate-a", ["KC @ BUF", "KC@BUF"])
    assert store["slate-a"] == ["KC@BUF"]
    assert disabled_for_slate(store, "slate-a", ["PHI@DAL", "KC@BUF"]) == ["KC@BUF"]
    assert disabled_for_slate(store, "slate-b", ["PHI@DAL", "KC@BUF"]) == []
    assert disabled_for_slate(store, "slate-a", ["PHI@DAL"]) == []


def test_spreads_for_a_filtered_pool_omit_the_off_game():
    lines = {
        "KC": {"total": 47.5, "spread": -3.5},
        "BUF": {"total": 47.5, "spread": 3.5},
        "PHI": {"total": 51.0, "spread": -7.0},
        "DAL": {"total": 51.0, "spread": 7.0},
    }
    pool = parse_dk_csv(MOCK_CSV)
    filtered = pool_without_games(pool, ["KC@BUF"])
    spreads = spreads_for_pool(lines, filtered)
    assert "KC" not in spreads
    assert "BUF" not in spreads
    assert spreads["PHI"] == pytest.approx(-7.0)
    assert spreads["DAL"] == pytest.approx(7.0)
