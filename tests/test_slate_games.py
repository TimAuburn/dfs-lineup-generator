"""Game table, on/off pool filter, and pre-solve blocks.

Lineup ids with every game on are the same ones main produces for seed 42.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.exporter import CLASSIC_UPLOAD_COLUMNS, lineups_to_dk_upload_csv
from src.kickoff_forecast import forecasts_for_matchups, weather_label
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
    order_game_labels_by_total,
    order_matchups_by_total,
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


def test_sofi_home_game_shows_dome_and_skips_nws(tmp_path):
    """A LAC or LAR home game is enclosed in the games table and does not call NWS."""

    def explode(url: str):
        raise AssertionError(f"NWS was called for a SoFi home game: {url}")

    pool = pd.DataFrame(
        [
            _player("Matthew Stafford", "LAR", "1", "SEA@LAR 11/08/2026 04:05PM ET", position="QB"),
            _player("Justin Herbert", "LAC", "2", "DEN@LAC 11/08/2026 04:25PM ET", position="QB"),
        ]
    )
    matchups = matchups_from_pool(pool)
    forecasts = forecasts_for_matchups(
        matchups,
        slate_key="sofi",
        cache_path=tmp_path / "kickoff.json",
        http_get=explode,
        now=datetime(2026, 11, 8, 13, 0, tzinfo=ZoneInfo("America/New_York")),
    )
    labels = {item.game: weather_label(item) for item in forecasts}
    frame = games_table_frame(matchups, {}, labels)
    assert list(frame["Matchup"]) == ["SEA @ LAR", "DEN @ LAC"]
    assert list(frame["Weather"]) == ["🏟️ Dome", "🏟️ Dome"]
    assert {item.status for item in forecasts} == {"dome"}
    assert {item.summary for item in forecasts} == {"Dome"}


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


def _pair(matchup: str, total: float, spread: float = -3.0) -> dict:
    away, home = matchup.split("@")
    return {
        away: {"total": total, "spread": spread, "game": matchup},
        home: {"total": total, "spread": -spread, "game": matchup},
    }


def _slate_lines() -> dict:
    lines: dict = {}
    lines.update(_pair("KC@BUF", 50.0))
    lines.update(_pair("PHI@DAL", 50.0))
    lines.update(_pair("BAL@CIN", 44.5))
    lines.update(_pair("DET@GB", 44.5))
    lines.update(_pair("TB@ATL", 41.0))
    lines.update(_pair("NYJ@NE", 50.0))
    lines.update(_pair("MIA@IND", 0.0))
    # SEA@LAR is absent. A conflicting total stays missing and is not averaged.
    lines["ARI"] = {"total": 40.0, "spread": -1.0, "game": "ARI@NO"}
    lines["NO"] = {"total": 55.0, "spread": 1.0, "game": "ARI@NO"}
    return lines


def _slate_pool() -> pd.DataFrame:
    """File order is not total order. The highest total is not the first game."""
    specs = [
        ("TB", "1", "TB@ATL 09/10/2025 01:00PM ET"),
        ("PHI", "2", "PHI@DAL 09/10/2025 04:25PM ET"),
        ("KC", "3", "KC@BUF 09/10/2025 01:00PM ET"),
        ("SEA", "4", "SEA@LAR 09/10/2025 08:20PM ET"),
        ("BAL", "5", "BAL@CIN 09/10/2025 01:00PM ET"),
        ("DET", "6", "DET@GB 09/10/2025 01:00PM ET"),
        ("NYJ", "7", "NYJ@NE"),
        ("MIA", "8", "MIA@IND 09/10/2025 01:00PM ET"),
        ("ARI", "9", "ARI@NO 09/07/2025 01:00PM ET"),
    ]
    return pd.DataFrame(
        [_player(team, team, pid, game) for team, pid, game in specs]
    )


_TOTAL_ORDER = [
    "KC @ BUF",
    "PHI @ DAL",
    "NYJ @ NE",
    "BAL @ CIN",
    "DET @ GB",
    "TB @ ATL",
    "MIA @ IND",
    "ARI @ NO",
    "SEA @ LAR",
]


def test_games_order_by_total_then_kickoff_then_matchup():
    """Highest total first. Ties use kickoff, then matchup. Missing totals are last."""
    lines = _slate_lines()
    forward = matchups_from_pool(_slate_pool())
    backward = list(reversed(forward))
    assert [game.key for game in backward] != [game.key for game in forward]

    ordered = order_matchups_by_total(backward, lines)
    assert [game.matchup for game in ordered] == _TOTAL_ORDER
    assert [game.matchup for game in order_matchups_by_total(forward, lines)] == _TOTAL_ORDER
    assert [game.key for game in forward] == [game.key for game in matchups_from_pool(_slate_pool())]

    labels = [game.key for game in forward]
    assert order_game_labels_by_total(labels, backward, lines) == [
        game.key for game in ordered
    ]
    # A label that is not on the slate keeps its place after the slate games.
    assert order_game_labels_by_total(
        ["NOT@SLATE", *labels, "ALSO@OUT"],
        forward,
        lines,
    ) == [game.key for game in ordered] + ["NOT@SLATE", "ALSO@OUT"]

    frame = games_table_frame(backward, lines, disabled=["KC @ BUF"])
    assert list(frame["Matchup"]) == _TOTAL_ORDER
    assert list(frame["Total"]) == [
        "50.0",
        "50.0",
        "50.0",
        "44.5",
        "44.5",
        "41.0",
        "0.0",
        MISSING,
        MISSING,
    ]
    assert MISSING == "missing"
    assert bool(frame.loc[frame["Matchup"].eq("KC @ BUF"), "On"].iloc[0]) is False
    assert bool(frame.iloc[0]["On"]) is False
    assert frame.iloc[0]["Matchup"] == "KC @ BUF"
    # The off game moved to row 0. The game that used to be row 0 stays on.
    assert bool(frame.loc[frame["Matchup"].eq("TB @ ATL"), "On"].iloc[0]) is True
    assert frame.loc[frame["Matchup"].eq("SEA @ LAR"), "Total"].iloc[0] == MISSING
    assert frame.loc[frame["Matchup"].eq("ARI @ NO"), "Total"].iloc[0] == MISSING
    assert (frame["Total"] == "0").sum() == 0
    assert (frame["Total"] == "0.0").sum() == 1


def _games_editor(at):
    for editor in at.data_editor:
        value = getattr(editor, "value", None)
        if value is None:
            continue
        columns = set(getattr(value, "columns", []))
        if {"On", "Matchup", "Total"} <= columns:
            return editor
    raise AssertionError("games table editor was not on the page")


def _stamp_pool(pool: pd.DataFrame) -> pd.DataFrame:
    from src.dk_scoring import SCORING_VERSION
    from src.slate_week import slate_games_from_pool, slate_key
    from src.sleeper import MATCH_CACHE_VERSION

    frame = pool.copy()
    frame["PPR"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    frame["Scoring"] = ""
    frame["ScoringNote"] = ""
    key = slate_key(slate_games_from_pool(frame)) if "Game Info" in frame.columns else ""
    frame.attrs["scoring_version"] = SCORING_VERSION
    frame.attrs["match_schema"] = MATCH_CACHE_VERSION
    frame.attrs["slate_key"] = key
    frame.attrs["fingerprint"] = f"test|{key}"
    return frame


def _sleeper_summary(pool: pd.DataFrame) -> dict:
    from src.dk_scoring import SCORING_VERSION
    from src.sleeper import MATCH_CACHE_VERSION

    return {
        "ok": True,
        "applied": True,
        "scoring": SCORING_VERSION,
        "match_schema": MATCH_CACHE_VERSION,
        "season": "2025",
        "week": 1,
        "season_type": "regular",
        "fingerprint": pool.attrs.get("fingerprint") or "",
        "slate_key": pool.attrs.get("slate_key") or "",
        "matched": 0,
        "rows": int(len(pool)),
        "inactive_rows": 0,
        "ppr_fallback_rows": 0,
    }


def _load_games_page(at, lines: dict) -> None:
    import json

    from src.data_loader import lines_to_frame
    from src.vegas_cache import CACHE_SCHEMA_VERSION

    pool = _stamp_pool(_slate_pool())
    empty = json.dumps({}, sort_keys=True, default=str)
    at.session_state["app_view"] = "Player pool"
    at.session_state["contest_format"] = "classic"
    at.session_state["contest_preset"] = "classic_cash"
    at.session_state["player_pool"] = pool
    at.session_state["sleeper_summary"] = _sleeper_summary(pool)
    at.session_state["pool_signature"] = "total-order"
    at.session_state["vegas_schema"] = CACHE_SCHEMA_VERSION
    at.session_state["vegas_source_json"] = empty
    at.session_state["vegas_editor_version"] = 1
    at.session_state["vegas_frame"] = lines_to_frame(lines)
    at.session_state["weather_by_home"] = {}
    at.session_state["weather_source_json"] = empty


def _on_by_matchup(editor) -> dict[str, bool]:
    frame = editor.value
    return {
        str(row["Matchup"]): bool(row["On"])
        for _, row in frame.iterrows()
    }


def test_game_toggle_stays_on_the_sorted_game_across_reruns(monkeypatch):
    """Unchecking the top row turns off that matchup, not the first file game.

    A later change in total order must not replay the row-0 edit onto the
    game that moves into that row.
    """
    app_test = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setattr("src.vegas_cache.fetch_vegas_lines", lambda *args, **kwargs: {})
    monkeypatch.setattr("src.weather.fetch_slate_weather", lambda *args, **kwargs: {})

    at = app_test.AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app.py"),
        default_timeout=120,
    )
    lines = _slate_lines()
    _load_games_page(at, lines)
    at.run(timeout=120)
    assert not at.exception, at.exception

    editor = _games_editor(at)
    assert list(editor.value["Matchup"]) == _TOTAL_ORDER
    assert list(editor.value["Total"])[:3] == ["50.0", "50.0", "50.0"]
    assert editor.value["Total"].iloc[-1] == MISSING
    assert editor.value["Matchup"].iloc[0] == "KC @ BUF"
    assert editor.value["Matchup"].iloc[0] != "TB @ ATL"
    pills = next(item for item in at.pills if item.label == "Game")
    assert list(pills.options) == [
        "KC@BUF",
        "PHI@DAL",
        "NYJ@NE",
        "BAL@CIN",
        "DET@GB",
        "TB@ATL",
        "MIA@IND",
        "ARI@NO",
        "SEA@LAR",
    ]

    first_key = editor.key
    at.session_state[first_key] = {
        "edited_rows": {0: {"On": False}},
        "added_rows": [],
        "deleted_rows": [],
    }
    at.run(timeout=120)
    assert not at.exception, at.exception
    # The editor's displayed value is the table passed in. The off game is
    # recorded from the edited return, keyed by matchup, during this run.
    assert at.session_state["disabled_games_by_slate"]["total-order"] == ["KC@BUF"]

    at.run(timeout=120)
    assert not at.exception, at.exception
    editor = _games_editor(at)
    assert list(editor.value["Matchup"]) == _TOTAL_ORDER
    assert _on_by_matchup(editor)["KC @ BUF"] is False
    assert _on_by_matchup(editor)["TB @ ATL"] is True
    assert _on_by_matchup(editor)["PHI @ DAL"] is True
    assert at.session_state["disabled_games_by_slate"]["total-order"] == ["KC@BUF"]

    from src.data_loader import lines_to_frame

    moved = _slate_lines()
    moved.update(_pair("TB@ATL", 60.0))
    at.session_state["vegas_frame"] = lines_to_frame(moved)
    at.session_state["vegas_editor_version"] = 2
    at.run(timeout=120)
    assert not at.exception, at.exception
    editor = _games_editor(at)
    assert editor.key != first_key
    assert list(editor.value["Matchup"])[:2] == ["TB @ ATL", "KC @ BUF"]
    assert _on_by_matchup(editor)["TB @ ATL"] is True
    assert _on_by_matchup(editor)["KC @ BUF"] is False
    assert editor.value.loc[editor.value["Matchup"].eq("SEA @ LAR"), "Total"].iloc[0] == MISSING
    assert at.session_state["disabled_games_by_slate"]["total-order"] == ["KC@BUF"]
    assert at.session_state[first_key]["edited_rows"][0]["On"] is False
    pills = next(item for item in at.pills if item.label == "Game")
    assert list(pills.options)[:2] == ["TB@ATL", "KC@BUF"]

    at.run(timeout=120)
    assert not at.exception, at.exception
    editor = _games_editor(at)
    assert _on_by_matchup(editor)["TB @ ATL"] is True
    assert _on_by_matchup(editor)["KC @ BUF"] is False
    assert at.session_state["disabled_games_by_slate"]["total-order"] == ["KC@BUF"]


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
