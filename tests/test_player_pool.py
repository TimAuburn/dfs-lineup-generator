"""Player pool collapse, include, captain, and ownership."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.optimizer import get_nfl_optimizer
from src.player_pool import (
    apply_pool_edits,
    clear_projection_overrides,
    collapse_for_editor,
    filter_editor_frame,
    game_choices,
    generation_pool,
    locked_player_ids,
    position_choices,
    projections_for_solve,
    set_captain,
    stamp_projection_baseline,
    set_included,
)
from src.presets import ContestFormat


def _showdown_pool() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": "1",
                "Name": "Josh Allen",
                "TeamAbbrev": "BUF",
                "Position": "QB",
                "RosterSlot": "CPT",
                "Salary": 12000,
                "Projection": 20.0,
            },
            {
                "ID": "2",
                "Name": "Josh Allen",
                "TeamAbbrev": "BUF",
                "Position": "QB",
                "RosterSlot": "FLEX",
                "Salary": 8000,
                "Projection": 20.0,
            },
            {
                "ID": "3",
                "Name": "Stefon Diggs",
                "TeamAbbrev": "BUF",
                "Position": "WR",
                "RosterSlot": "CPT",
                "Salary": 9000,
                "Projection": 15.0,
            },
            {
                "ID": "4",
                "Name": "Stefon Diggs",
                "TeamAbbrev": "BUF",
                "Position": "WR",
                "RosterSlot": "FLEX",
                "Salary": 6000,
                "Projection": 15.0,
            },
        ]
    )


def test_showdown_collapse_uses_the_flex_salary():
    editor = collapse_for_editor(_showdown_pool(), showdown=True)
    assert list(editor["Name"]) == ["Josh Allen", "Stefon Diggs"]
    assert list(editor["Salary"]) == [8000, 6000]
    assert list(editor["Include"]) == [True, True]
    assert list(editor["Captain"]) == [True, True]
    assert editor["Captain %"].isna().all()
    classic = collapse_for_editor(_showdown_pool().iloc[:1], showdown=False)
    assert "Captain" not in classic.columns
    assert "Captain %" not in classic.columns


def test_include_captain_and_ownership_write_back_onto_both_rows():
    pool = _showdown_pool()
    editor = collapse_for_editor(pool, showdown=True)
    editor.loc[editor["Name"] == "Josh Allen", "Include"] = False
    editor.loc[editor["Name"] == "Josh Allen", "Ownership %"] = 18.4
    editor.loc[editor["Name"] == "Josh Allen", "Captain %"] = 40
    editor.loc[editor["Name"] == "Stefon Diggs", "Captain"] = False
    editor.loc[editor["Name"] == "Stefon Diggs", "Ownership %"] = pd.NA
    editor.loc[editor["Name"] == "Stefon Diggs", "Captain %"] = pd.NA
    editor.loc[editor["Name"] == "Stefon Diggs", "Lock"] = True

    updated = apply_pool_edits(pool, editor, editor, showdown=True)
    allen = updated[updated["Name"] == "Josh Allen"]
    diggs = updated[updated["Name"] == "Stefon Diggs"]
    assert list(allen["Include"]) == [False, False]
    assert list(allen["Ownership"]) == pytest.approx([0.184, 0.184])
    assert list(allen["CaptainExposure"]) == pytest.approx([0.4, 0.4])
    assert list(diggs["Captain"]) == [False, False]
    assert diggs["Ownership"].isna().all()
    assert diggs["CaptainExposure"].isna().all()

    eligible = generation_pool(updated, showdown=True)
    assert set(eligible["Name"]) == {"Stefon Diggs"}
    assert list(eligible["RosterSlot"]) == ["FLEX"]
    assert locked_player_ids(eligible, showdown=True) == ["4"]


def test_classic_include_drops_the_player_and_blank_ownership_stays_empty():
    pool = pd.DataFrame(
        [
            {"ID": "9", "Name": "CeeDee Lamb", "TeamAbbrev": "DAL", "Position": "WR", "Salary": 7000, "Projection": 18.0},
            {"ID": "8", "Name": "Dak Prescott", "TeamAbbrev": "DAL", "Position": "QB", "Salary": 6000, "Projection": 17.0},
        ]
    )
    editor = collapse_for_editor(pool, showdown=False)
    editor.loc[editor["Name"] == "Dak Prescott", "Include"] = False
    updated = apply_pool_edits(pool, editor, editor, showdown=False)
    eligible = generation_pool(updated, showdown=False)
    assert list(eligible["Name"]) == ["CeeDee Lamb"]
    assert updated["Ownership"].isna().all()


def test_name_position_and_game_filters():
    frame = pd.DataFrame(
        [
            {
                "Name": "Josh Allen",
                "TeamAbbrev": "BUF",
                "Position": "QB",
                "Game Info": "BUF@KC 11/01/2026 01:00PM ET",
            },
            {
                "Name": "Travis Kelce",
                "TeamAbbrev": "KC",
                "Position": "TE",
                "Game Info": "BUF@KC 11/01/2026 01:00PM ET",
            },
            {
                "Name": "CeeDee Lamb",
                "TeamAbbrev": "DAL",
                "Position": "WR",
                "Game Info": "DAL@PHI 11/01/2026 04:25PM ET",
            },
        ]
    )
    assert position_choices(frame) == ["QB", "WR", "TE", "FLEX"]
    assert game_choices(frame) == ["BUF@KC", "DAL@PHI"]
    by_name = filter_editor_frame(frame, name_query="allen")
    assert list(by_name["Name"]) == ["Josh Allen"]
    by_position = filter_editor_frame(frame, positions=["WR", "TE"])
    assert list(by_position["Name"]) == ["Travis Kelce", "CeeDee Lamb"]
    by_flex = filter_editor_frame(frame, positions=["FLEX"])
    assert list(by_flex["Name"]) == ["Travis Kelce", "CeeDee Lamb"]
    by_game = filter_editor_frame(frame, games=["DAL@PHI"])
    assert list(by_game["Name"]) == ["CeeDee Lamb"]
    assert list(filter_editor_frame(frame)["Name"]) == list(frame["Name"])


def test_include_shown_updates_only_the_filtered_players():
    pool = _showdown_pool()
    shown = collapse_for_editor(pool, showdown=True)
    shown = shown.loc[shown["Name"] == "Josh Allen"]
    updated = set_included(pool, shown, False)
    assert list(updated.loc[updated["Name"] == "Josh Allen", "Include"]) == [False, False]
    assert list(updated.loc[updated["Name"] == "Stefon Diggs", "Include"]) == [True, True]
    captains = set_captain(pool, shown, False)
    assert list(captains.loc[captains["Name"] == "Josh Allen", "Captain"]) == [False, False]
    assert list(captains.loc[captains["Name"] == "Stefon Diggs", "Captain"]) == [True, True]


def test_showdown_editor_renders_one_row_and_a_captain_column():
    app_test = pytest.importorskip("streamlit.testing.v1")
    from src.presets import ContestFormat

    at = app_test.AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    at.session_state["app_view"] = "Player pool"
    at.session_state["contest_format"] = ContestFormat.SHOWDOWN.value
    shown = _as_current_scoring(_showdown_pool())
    at.session_state["player_pool"] = shown
    at.session_state["sleeper_summary"] = _current_sleeper_summary(shown)
    at.run(timeout=90)
    assert not at.exception
    assert any(item.label == "Player name" for item in at.text_input)
    position_pills = next(item for item in at.pills if item.label == "Position")
    assert "FLEX" in list(position_pills.options)
    button_labels = [item.label for item in at.button]
    assert "Include shown" in button_labels
    assert "Remove shown" in button_labels
    assert "Include captain shown" in button_labels
    assert "Remove captain shown" in button_labels
    assert "Clear filters" in button_labels
    captions = " ".join(item.value for item in at.caption)
    assert "One row per player" in captions
    editor = at.dataframe[0]
    assert "Captain" in editor.columns
    assert "Captain %" in editor.columns
    assert "Include" in editor.columns
    assert "Ownership %" in editor.columns
    assert list(editor.value["Name"]) == ["Josh Allen", "Stefon Diggs"]
    assert list(editor.value["Salary"]) == [8000, 6000]


def test_player_pool_uploader_is_on_the_pool_view_only():
    streamlit = pytest.importorskip("streamlit")
    app_test = pytest.importorskip("streamlit.testing.v1")
    del streamlit
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = app_test.AppTest.from_file(str(app_path))
    at.session_state["app_view"] = "Player pool"
    at.run(timeout=90)
    pool_labels = [item.label for item in at.file_uploader]
    assert "DraftKings player pool CSV" in pool_labels

    optimizer = app_test.AppTest.from_file(str(app_path))
    optimizer.run(timeout=90)
    optimizer_labels = [item.label for item in optimizer.file_uploader]
    assert "DraftKings player pool CSV" not in optimizer_labels
    assert any("Player pool" in item.value for item in optimizer.info)


def _worthy_pool() -> pd.DataFrame:
    """Xavier Worthy at the salary-file base, before any odds or wind."""
    return pd.DataFrame(
        [
            {
                "ID": "3009",
                "Name": "Xavier Worthy",
                "Position": "WR",
                "TeamAbbrev": "KC",
                "Salary": 5000,
                "AvgPointsPerGame": 11.50,
                "Projection": 11.50,
                "Game Info": "KC@BUF 09/10/2025 08:20PM ET",
            }
        ]
    )


def _kc_buf_lines(total: float, kc_spread: float) -> dict:
    return {
        "KC": {"total": total, "spread": kc_spread, "game": "KC@BUF"},
        "BUF": {"total": total, "spread": -kc_spread, "game": "KC@BUF"},
    }


def _solver_fppg(pool: pd.DataFrame, *, showdown: bool = False) -> dict[str, float]:
    optimizer = get_nfl_optimizer(
        ContestFormat.SHOWDOWN if showdown else ContestFormat.CLASSIC
    )
    optimizer.set_exclude_inactive(False)
    optimizer.load_from_dataframe(pool)
    return {player.id: float(player.fppg) for player in optimizer._players}


def test_solve_rederives_worthy_from_the_clean_base_when_odds_change():
    """ITT 30, then 10, then no odds. The stored adjusted float is not the base."""
    lines_30 = _kc_buf_lines(50.0, -10.0)
    lines_10 = _kc_buf_lines(40.0, 20.0)
    boosted = projections_for_solve(_worthy_pool(), lines_30, {})
    assert boosted.loc[0, "ProjectionBaseline"] == pytest.approx(11.50)
    assert boosted.loc[0, "Projection"] == pytest.approx(12.07)
    assert _solver_fppg(boosted)["3009"] == pytest.approx(12.07)

    stale = boosted.copy()
    stale["Projection"] = 12.07
    stale["ProjectionOverride"] = pd.NA
    cut = projections_for_solve(stale, lines_10, {})
    assert cut.loc[0, "ProjectionBaseline"] == pytest.approx(11.50)
    assert cut.loc[0, "Projection"] == pytest.approx(10.93)
    assert _solver_fppg(cut)["3009"] == pytest.approx(10.93)

    legacy = _worthy_pool()
    legacy["Projection"] = 12.07
    cleared = projections_for_solve(legacy, {}, {})
    assert cleared.loc[0, "ProjectionBaseline"] == pytest.approx(11.50)
    assert cleared.loc[0, "Projection"] == pytest.approx(11.50)
    assert _solver_fppg(cleared)["3009"] == pytest.approx(11.50)

    again = projections_for_solve(cleared, {}, {})
    assert again.loc[0, "Projection"] == pytest.approx(11.50)
    assert again.loc[0, "ProjectionBaseline"] == pytest.approx(11.50)


def test_solve_uses_current_weather_and_drops_it_when_the_wind_is_gone():
    pool = _worthy_pool()
    pool["TeamAbbrev"] = "GB"
    pool["Game Info"] = "CHI@GB 10/12/2026 01:00PM ET"
    wind = {"GB": {"wind_mph": 20}}
    windy = projections_for_solve(pool, {}, wind)
    assert windy.loc[0, "Projection"] == pytest.approx(10.35)
    assert _solver_fppg(windy)["3009"] == pytest.approx(10.35)
    calm = projections_for_solve(windy, {}, {})
    assert calm.loc[0, "Projection"] == pytest.approx(11.50)
    assert calm.loc[0, "ProjectionBaseline"] == pytest.approx(11.50)
    repeated = projections_for_solve(calm, {}, {})
    assert repeated.loc[0, "Projection"] == pytest.approx(11.50)


def test_explicit_override_survives_odds_weather_and_a_second_solve():
    pool = _worthy_pool()
    pool["Projection"] = 12.07
    pool["ProjectionOverride"] = 9.25
    lines_30 = _kc_buf_lines(50.0, -10.0)
    lines_10 = _kc_buf_lines(40.0, 20.0)
    wind = {"BUF": {"wind_mph": 25}}
    first = projections_for_solve(pool, lines_30, wind)
    second = projections_for_solve(first, lines_10, {})
    third = projections_for_solve(second, {}, {})
    for frame in (first, second, third):
        assert frame.loc[0, "Projection"] == pytest.approx(9.25)
        assert frame.loc[0, "ProjectionBaseline"] == pytest.approx(11.50)
        assert frame.loc[0, "ProjectionOverride"] == pytest.approx(9.25)
        assert _solver_fppg(frame)["3009"] == pytest.approx(9.25)


def test_showdown_captain_multiplier_is_applied_once_to_the_solve_projection():
    pool = pd.DataFrame(
        [
            {
                "ID": "cpt",
                "Name": "Xavier Worthy",
                "Position": "WR",
                "Roster Position": "CPT",
                "RosterSlot": "CPT",
                "TeamAbbrev": "KC",
                "Salary": 16800,
                "AvgPointsPerGame": 11.50,
                "Projection": 12.07,
                "Game Info": "KC@BUF 09/10/2025 08:20PM ET",
            },
            {
                "ID": "flex",
                "Name": "Xavier Worthy",
                "Position": "WR",
                "Roster Position": "FLEX",
                "RosterSlot": "FLEX",
                "TeamAbbrev": "KC",
                "Salary": 11200,
                "AvgPointsPerGame": 11.50,
                "Projection": 12.07,
                "Game Info": "KC@BUF 09/10/2025 08:20PM ET",
            },
        ]
    )
    solved = projections_for_solve(pool, _kc_buf_lines(50.0, -10.0), {})
    assert list(solved["Projection"]) == pytest.approx([12.07, 12.07])
    fppg = _solver_fppg(solved, showdown=True)
    assert fppg["flex"] == pytest.approx(12.07)
    assert fppg["cpt"] == pytest.approx(18.105)
    repeated = projections_for_solve(solved, _kc_buf_lines(50.0, -10.0), {})
    assert list(repeated["Projection"]) == pytest.approx([12.07, 12.07])
    assert _solver_fppg(repeated, showdown=True)["cpt"] == pytest.approx(18.105)


def test_optional_scoring_columns_and_version_roundtrip():
    pool = _showdown_pool()
    pool["PPR"] = [18.0, 18.0, 12.5, 12.5]
    pool["Scoring"] = ["DraftKings", "DraftKings", "DraftKings", "DraftKings"]
    pool["ScoringNote"] = ["", "", "complete", "complete"]
    pool.attrs["scoring_version"] = "dk-component-v2"
    editor = collapse_for_editor(pool, showdown=True)
    assert list(editor["PPR"]) == [18.0, 12.5]
    assert list(editor["Scoring"]) == ["DraftKings", "DraftKings"]
    assert list(editor["ScoringNote"]) == ["", "complete"]
    assert editor.attrs["scoring_version"] == "dk-component-v2"
    updated = apply_pool_edits(pool, editor, editor, showdown=True)
    assert list(updated["PPR"]) == [18.0, 18.0, 12.5, 12.5]
    assert list(updated["ScoringNote"]) == ["", "", "complete", "complete"]
    assert updated.attrs["scoring_version"] == "dk-component-v2"
    eligible = generation_pool(updated, showdown=True)
    assert set(eligible["Scoring"]) == {"DraftKings"}
    assert eligible.attrs["scoring_version"] == "dk-component-v2"


def _det_cash_pool() -> pd.DataFrame:
    """Nine classic players. Worthy's stored projection is the old ITT 30 number."""
    game = "MIN@DET 11/08/2026 01:00PM ET"
    rows = [
        ("QB", "Jared Goff", "1", 8000, 18.0, 18.0),
        ("RB", "Jahmyr Gibbs", "2", 7000, 16.0, 16.0),
        ("RB", "David Montgomery", "3", 6000, 12.0, 12.0),
        ("WR", "Xavier Worthy", "3009", 5000, 11.50, 12.07),
        ("WR", "Amon-Ra St. Brown", "5", 6400, 15.0, 15.0),
        ("WR", "Jameson Williams", "6", 4800, 10.0, 10.0),
        ("TE", "Sam LaPorta", "7", 4600, 9.0, 9.0),
        ("RB", "Craig Reynolds", "8", 4200, 6.0, 6.0),
        ("DST", "Lions", "9", 3000, 7.0, 7.0),
    ]
    return pd.DataFrame(
        [
            {
                "ID": player_id,
                "Name": name,
                "Position": position,
                "TeamAbbrev": "DET",
                "Salary": salary,
                "AvgPointsPerGame": base,
                "Projection": stored,
                "Game Info": game,
            }
            for position, name, player_id, salary, base, stored in rows
        ]
    )


def _det_lines(total: float, det_spread: float) -> dict:
    return {
        "DET": {"total": total, "spread": det_spread, "game": "MIN@DET"},
        "MIN": {"total": total, "spread": -det_spread, "game": "MIN@DET"},
    }


def _worthy_lineup_projections(at) -> list[float]:
    found: list[float] = []
    for frame in at.dataframe:
        value = getattr(frame, "value", None)
        if value is None or "Name" not in getattr(value, "columns", []):
            continue
        if "Projection" not in value.columns:
            continue
        rows = value.loc[value["Name"].astype(str).eq("Xavier Worthy"), "Projection"]
        found.extend(float(item) for item in rows)
    return found


def test_optimizer_generate_uses_current_odds_not_the_stored_projection():
    """Pool view stored 12.07. Solve still follows the odds on the screen."""
    app_test = pytest.importorskip("streamlit.testing.v1")
    from src.data_loader import lines_to_frame
    from src.vegas_cache import CACHE_SCHEMA_VERSION

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = app_test.AppTest.from_file(str(app_path))
    empty_fetch = json.dumps({}, sort_keys=True, default=str)
    at.session_state["contest_preset"] = "classic_cash"
    scored = _as_current_scoring(_det_cash_pool())
    at.session_state["player_pool"] = scored
    at.session_state["sleeper_summary"] = _current_sleeper_summary(scored)
    at.session_state["vegas_schema"] = CACHE_SCHEMA_VERSION
    at.session_state["vegas_source_json"] = empty_fetch
    at.session_state["vegas_editor_version"] = 1
    at.session_state["vegas_frame"] = lines_to_frame(_det_lines(50.0, -10.0))
    at.session_state["weather_by_home"] = {}
    at.session_state["weather_source_json"] = empty_fetch
    at.run(timeout=90)
    assert not at.exception, at.exception

    def generate(lines: dict | None, version: int) -> list[float]:
        at.session_state["vegas_frame"] = lines_to_frame(lines or {})
        at.session_state["vegas_source_json"] = empty_fetch
        at.session_state["vegas_editor_version"] = version
        at.run(timeout=90)
        assert not at.exception, at.exception
        button = next(item for item in at.button if item.label == "Generate Lineups")
        button.click()
        at.run(timeout=90)
        assert not at.exception, at.exception
        found = _worthy_lineup_projections(at)
        assert found, "Xavier Worthy was not in the solved lineup"
        return found

    assert generate(_det_lines(50.0, -10.0), 2) == pytest.approx([12.07])
    assert generate(_det_lines(40.0, 20.0), 3) == pytest.approx([10.93])
    assert generate({}, 4) == pytest.approx([11.50])


def _clean_det_pool() -> pd.DataFrame:
    pool = _det_cash_pool()
    pool["Projection"] = pool["AvgPointsPerGame"]
    return pool


def _player_editor(at):
    for editor in at.data_editor:
        value = editor.value
        if value is not None and "Name" in value.columns and "Projection" in value.columns:
            return editor
    raise AssertionError("player projection editor was not on the page")


def _worthy_override(pool: pd.DataFrame) -> float:
    rows = pool.loc[pool["Name"].astype(str).eq("Xavier Worthy"), "ProjectionOverride"]
    assert not rows.empty
    return float(rows.iloc[0])


def test_pinned_worthy_survives_odds_removal_editor_remount_and_generate():
    """Pin 11.50 at ITT 30. Removing odds makes the automatic number 11.50 too.

    The remount must not clear that pin. ITT 10 still solves at 11.50.
    """
    app_test = pytest.importorskip("streamlit.testing.v1")
    from src.data_loader import lines_to_frame

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = app_test.AppTest.from_file(str(app_path))
    empty_fetch = json.dumps({}, sort_keys=True, default=str)
    _open_pool(at, _clean_det_pool(), _det_lines(50.0, -10.0))
    at.run(timeout=90)
    assert not at.exception, at.exception

    editor = _player_editor(at)
    table = editor.value.reset_index(drop=True)
    worthy_row = int(table.index[table["Name"].eq("Xavier Worthy")][0])
    assert float(table.loc[worthy_row, "Projection"]) == pytest.approx(12.07)
    at.session_state[editor.key] = {
        "edited_rows": {worthy_row: {"Projection": 11.50}},
        "added_rows": [],
        "deleted_rows": [],
    }
    at.run(timeout=90)
    assert not at.exception, at.exception
    assert _worthy_override(at.session_state["player_pool"]) == pytest.approx(11.50)

    at.session_state["vegas_frame"] = lines_to_frame({})
    at.session_state["vegas_source_json"] = empty_fetch
    at.session_state["vegas_editor_version"] = 8
    at.run(timeout=90)
    assert not at.exception, at.exception
    assert _worthy_override(at.session_state["player_pool"]) == pytest.approx(11.50)

    editor = _player_editor(at)
    table = editor.value.reset_index(drop=True)
    worthy_row = int(table.index[table["Name"].eq("Xavier Worthy")][0])
    at.session_state[editor.key] = {
        "edited_rows": {worthy_row: {"Lock": True}},
        "added_rows": [],
        "deleted_rows": [],
    }
    at.run(timeout=90)
    assert not at.exception, at.exception
    locked = at.session_state["player_pool"]
    worthy_rows = locked.loc[locked["Name"].astype(str).eq("Xavier Worthy")]
    assert _worthy_override(locked) == pytest.approx(11.50)
    assert bool(worthy_rows["Lock"].iloc[0])

    at.session_state["app_view"] = "Optimizer"
    at.session_state["vegas_frame"] = lines_to_frame(_det_lines(40.0, 20.0))
    at.session_state["vegas_source_json"] = empty_fetch
    at.session_state["vegas_editor_version"] = 9
    at.run(timeout=90)
    assert not at.exception, at.exception
    button = next(item for item in at.button if item.label == "Generate Lineups")
    button.click()
    at.run(timeout=90)
    assert not at.exception, at.exception
    assert _worthy_lineup_projections(at) == pytest.approx([11.50])

    at.session_state["app_view"] = "Player pool"
    at.session_state["vegas_frame"] = lines_to_frame(_det_lines(40.0, 20.0))
    at.session_state["vegas_source_json"] = empty_fetch
    at.session_state["vegas_editor_version"] = 10
    at.run(timeout=90)
    assert not at.exception, at.exception
    assert _worthy_override(at.session_state["player_pool"]) == pytest.approx(11.50)
    reset = next(item for item in at.button if item.label == "Reset to automatic")
    reset.click()
    at.run(timeout=90)
    assert not at.exception, at.exception
    assert pd.isna(
        at.session_state["player_pool"]
        .loc[lambda frame: frame["Name"].astype(str).eq("Xavier Worthy"), "ProjectionOverride"]
        .iloc[0]
    )
    at.session_state["app_view"] = "Optimizer"
    at.run(timeout=90)
    assert not at.exception, at.exception
    button = next(item for item in at.button if item.label == "Generate Lineups")
    button.click()
    at.run(timeout=90)
    assert not at.exception, at.exception
    assert _worthy_lineup_projections(at) == pytest.approx([10.93])


def _override_value(pool: pd.DataFrame, name: str, team: str):
    rows = pool.loc[
        pool["Name"].astype(str).eq(name) & pool["TeamAbbrev"].astype(str).eq(team),
        "ProjectionOverride",
    ]
    assert len(rows) >= 1, (name, team)
    return rows.iloc[0]


def _pool_with_hidden_kelce_pins() -> pd.DataFrame:
    """Detroit cash slate plus two Travis Kelce pins the Worthy filter must not clear."""
    pool = _clean_det_pool()
    game = "MIN@DET 11/08/2026 01:00PM ET"
    extra = pd.DataFrame(
        [
            {
                "ID": "kc-te",
                "Name": "Travis Kelce",
                "Position": "TE",
                "TeamAbbrev": "KC",
                "Salary": 5000,
                "AvgPointsPerGame": 8.0,
                "Projection": 8.0,
                "ProjectionOverride": 9.50,
                "Game Info": game,
            },
            {
                "ID": "dal-te",
                "Name": "Travis Kelce",
                "Position": "TE",
                "TeamAbbrev": "DAL",
                "Salary": 4800,
                "AvgPointsPerGame": 8.0,
                "Projection": 8.0,
                "ProjectionOverride": 8.25,
                "Game Info": game,
            },
        ]
    )
    return pd.concat([pool, extra], ignore_index=True)


def _as_current_scoring(pool: pd.DataFrame) -> pd.DataFrame:
    """Mark a fixture as the current scorer and matcher without inventing stats."""
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


def _current_sleeper_summary(pool: pd.DataFrame) -> dict:
    from src.dk_scoring import SCORING_VERSION
    from src.sleeper import MATCH_CACHE_VERSION

    return {
        "ok": True,
        "applied": True,
        "scoring": SCORING_VERSION,
        "match_schema": MATCH_CACHE_VERSION,
        "season": "2026",
        "week": 9,
        "season_type": "regular",
        "fingerprint": pool.attrs.get("fingerprint") or "",
        "slate_key": pool.attrs.get("slate_key") or "",
        "matched": 0,
        "rows": int(len(pool)),
        "inactive_rows": 0,
        "ppr_fallback_rows": 0,
    }


def _open_pool(at, pool: pd.DataFrame, lines: dict, *, name_query: str = "") -> None:
    from src.data_loader import lines_to_frame
    from src.vegas_cache import CACHE_SCHEMA_VERSION

    empty_fetch = json.dumps({}, sort_keys=True, default=str)
    scored = _as_current_scoring(pool)
    at.session_state["app_view"] = "Player pool"
    at.session_state["contest_preset"] = "classic_cash"
    at.session_state["contest_format"] = "classic"
    at.session_state["player_pool"] = scored
    at.session_state["sleeper_summary"] = _current_sleeper_summary(scored)
    at.session_state["pool_signature"] = "pool"
    if name_query:
        at.session_state["pool_name_pool"] = name_query
    at.session_state["vegas_schema"] = CACHE_SCHEMA_VERSION
    at.session_state["vegas_source_json"] = empty_fetch
    at.session_state["vegas_editor_version"] = 1
    at.session_state["vegas_frame"] = lines_to_frame(lines)
    at.session_state["weather_by_home"] = {}
    at.session_state["weather_source_json"] = empty_fetch


def _pin_projection(at, name: str, value: float, *, also: dict | None = None) -> None:
    editor = _player_editor(at)
    table = editor.value.reset_index(drop=True)
    row = int(table.index[table["Name"].eq(name)][0])
    edits = {"Projection": value}
    if also:
        edits.update(also)
    at.session_state[editor.key] = {
        "edited_rows": {row: edits},
        "added_rows": [],
        "deleted_rows": [],
    }
    at.run(timeout=90)
    assert not at.exception, at.exception


def _generate_worthy(at) -> list[float]:
    at.session_state["app_view"] = "Optimizer"
    at.run(timeout=90)
    assert not at.exception, at.exception
    button = next(item for item in at.button if item.label == "Generate Lineups")
    button.click()
    at.run(timeout=90)
    assert not at.exception, at.exception
    found = _worthy_lineup_projections(at)
    assert found, "Xavier Worthy was not in the solved lineup"
    return found


def test_immediate_reset_on_the_same_editor_restores_12_07():
    """Edit 11.50, then Reset with no odds or filter change. Solve must use 12.07."""
    app_test = pytest.importorskip("streamlit.testing.v1")
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = app_test.AppTest.from_file(str(app_path))
    _open_pool(at, _pool_with_hidden_kelce_pins(), _det_lines(50.0, -10.0))
    at.run(timeout=90)
    assert not at.exception, at.exception
    editor = _player_editor(at)
    table = editor.value.reset_index(drop=True)
    assert float(table.loc[table["Name"].eq("Xavier Worthy"), "Projection"].iloc[0]) == pytest.approx(
        12.07
    )
    _pin_projection(at, "Xavier Worthy", 11.50, also={"Lock": True})
    assert _worthy_override(at.session_state["player_pool"]) == pytest.approx(11.50)

    reset = next(item for item in at.button if item.label == "Reset to automatic")
    reset.click()
    at.run(timeout=90)
    assert not at.exception, at.exception
    pool = at.session_state["player_pool"]
    assert pd.isna(_override_value(pool, "Xavier Worthy", "DET"))
    assert bool(pool.loc[pool["Name"].eq("Xavier Worthy"), "Lock"].iloc[0])
    assert _generate_worthy(at) == pytest.approx([12.07])


def test_filtered_immediate_reset_restores_10_93_and_keeps_other_pins():
    """A Worthy-only filter at ITT 10 resets that row and leaves both Kelce pins."""
    app_test = pytest.importorskip("streamlit.testing.v1")
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = app_test.AppTest.from_file(str(app_path))
    _open_pool(
        at,
        _pool_with_hidden_kelce_pins(),
        _det_lines(40.0, 20.0),
        name_query="Worthy",
    )
    at.run(timeout=90)
    assert not at.exception, at.exception
    editor = _player_editor(at)
    assert list(editor.value["Name"]) == ["Xavier Worthy"]
    assert float(editor.value["Projection"].iloc[0]) == pytest.approx(10.93)
    _pin_projection(at, "Xavier Worthy", 11.50, also={"Include": True})
    assert _worthy_override(at.session_state["player_pool"]) == pytest.approx(11.50)

    reset = next(item for item in at.button if item.label == "Reset to automatic")
    reset.click()
    at.run(timeout=90)
    assert not at.exception, at.exception
    pool = at.session_state["player_pool"]
    assert pd.isna(_override_value(pool, "Xavier Worthy", "DET"))
    assert _override_value(pool, "Travis Kelce", "KC") == pytest.approx(9.50)
    assert _override_value(pool, "Travis Kelce", "DAL") == pytest.approx(8.25)
    assert _generate_worthy(at) == pytest.approx([10.93])


def test_showdown_immediate_reset_clears_captain_and_flex_together():
    """One editor row's reset clears both salary rows and the solver sees the automatic number."""
    app_test = pytest.importorskip("streamlit.testing.v1")
    from src.data_loader import lines_to_frame
    from src.vegas_cache import CACHE_SCHEMA_VERSION

    pool = pd.DataFrame(
        [
            {
                "ID": "cpt",
                "Name": "Xavier Worthy",
                "Position": "WR",
                "Roster Position": "CPT",
                "RosterSlot": "CPT",
                "TeamAbbrev": "DET",
                "Salary": 16800,
                "AvgPointsPerGame": 11.50,
                "Projection": 11.50,
                "Game Info": "MIN@DET 11/08/2026 01:00PM ET",
            },
            {
                "ID": "flex",
                "Name": "Xavier Worthy",
                "Position": "WR",
                "Roster Position": "FLEX",
                "RosterSlot": "FLEX",
                "TeamAbbrev": "DET",
                "Salary": 11200,
                "AvgPointsPerGame": 11.50,
                "Projection": 11.50,
                "Game Info": "MIN@DET 11/08/2026 01:00PM ET",
            },
        ]
    )
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = app_test.AppTest.from_file(str(app_path))
    empty_fetch = json.dumps({}, sort_keys=True, default=str)
    at.session_state["app_view"] = "Player pool"
    at.session_state["contest_format"] = "showdown"
    at.session_state["contest_preset"] = "showdown_cash"
    scored = _as_current_scoring(pool)
    at.session_state["player_pool"] = scored
    at.session_state["sleeper_summary"] = _current_sleeper_summary(scored)
    at.session_state["vegas_schema"] = CACHE_SCHEMA_VERSION
    at.session_state["vegas_source_json"] = empty_fetch
    at.session_state["vegas_editor_version"] = 1
    at.session_state["vegas_frame"] = lines_to_frame(_det_lines(50.0, -10.0))
    at.session_state["weather_by_home"] = {}
    at.session_state["weather_source_json"] = empty_fetch
    at.run(timeout=90)
    assert not at.exception, at.exception
    _pin_projection(at, "Xavier Worthy", 11.50, also={"Lock": True})
    pinned = at.session_state["player_pool"]
    assert list(pinned["ProjectionOverride"]) == pytest.approx([11.50, 11.50])

    reset = next(item for item in at.button if item.label == "Reset to automatic")
    reset.click()
    at.run(timeout=90)
    assert not at.exception, at.exception
    cleared = at.session_state["player_pool"]
    assert cleared["ProjectionOverride"].isna().all()
    assert list(cleared["Lock"]) == [True, True]
    solved = projections_for_solve(cleared, _det_lines(50.0, -10.0), {})
    assert list(solved["Projection"]) == pytest.approx([12.07, 12.07])
    fppg = _solver_fppg(solved, showdown=True)
    assert fppg["flex"] == pytest.approx(12.07)
    assert fppg["cpt"] == pytest.approx(18.105)


def test_showdown_pin_stays_on_captain_and_flex_until_reset():
    """One editor row pins both salary rows, and reset restores the current automatic number."""
    pool = pd.DataFrame(
        [
            {
                "ID": "cpt",
                "Name": "Xavier Worthy",
                "Position": "WR",
                "Roster Position": "CPT",
                "RosterSlot": "CPT",
                "TeamAbbrev": "KC",
                "Salary": 16800,
                "AvgPointsPerGame": 11.50,
                "Projection": 11.50,
                "Game Info": "KC@BUF 09/10/2025 08:20PM ET",
            },
            {
                "ID": "flex",
                "Name": "Xavier Worthy",
                "Position": "WR",
                "Roster Position": "FLEX",
                "RosterSlot": "FLEX",
                "TeamAbbrev": "KC",
                "Salary": 11200,
                "AvgPointsPerGame": 11.50,
                "Projection": 11.50,
                "Game Info": "KC@BUF 09/10/2025 08:20PM ET",
            },
        ]
    )
    shown = projections_for_solve(pool, _kc_buf_lines(50.0, -10.0), {})
    editor = collapse_for_editor(shown, showdown=True)
    edited = editor.copy()
    edited.loc[edited["Name"].eq("Xavier Worthy"), "Projection"] = 11.50
    pinned = apply_pool_edits(pool, edited, editor, showdown=True)
    assert list(pinned["ProjectionOverride"]) == pytest.approx([11.50, 11.50])

    displayed = collapse_for_editor(projections_for_solve(pinned, {}, {}), showdown=True)
    returned = displayed.copy()
    returned.loc[returned["Name"].eq("Xavier Worthy"), "Lock"] = True
    kept = apply_pool_edits(pinned, returned, displayed, showdown=True)
    assert list(kept["ProjectionOverride"]) == pytest.approx([11.50, 11.50])
    assert list(kept["Lock"]) == [True, True]
    solved = projections_for_solve(kept, _kc_buf_lines(40.0, 20.0), {})
    assert list(solved["Projection"]) == pytest.approx([11.50, 11.50])
    fppg = _solver_fppg(solved, showdown=True)
    assert fppg["flex"] == pytest.approx(11.50)
    assert fppg["cpt"] == pytest.approx(17.25)

    reset = clear_projection_overrides(kept, displayed)
    assert reset["ProjectionOverride"].isna().all()
    automatic = projections_for_solve(reset, _kc_buf_lines(40.0, 20.0), {})
    assert list(automatic["Projection"]) == pytest.approx([10.93, 10.93])
    reset_fppg = _solver_fppg(automatic, showdown=True)
    assert reset_fppg["flex"] == pytest.approx(10.93)
    assert reset_fppg["cpt"] == pytest.approx(16.395)


def test_baseline_is_the_draftkings_component_total_not_raw_ppr():
    """Vegas and the captain multiplier each apply once, on the DraftKings total."""
    rows = []
    for slot, player_id, salary in (("CPT", "cpt", 16800), ("FLEX", "flex", 11200)):
        rows.append(
            {
                "ID": player_id,
                "Name": "Xavier Worthy",
                "Position": "WR",
                "Roster Position": slot,
                "RosterSlot": slot,
                "TeamAbbrev": "KC",
                "Salary": salary,
                "AvgPointsPerGame": 11.50,
                "Projection": 99.0,
                "Sleeper": 12.00,
                "PPR": 20.00,
                "Scoring": "DraftKings",
                "ScoringNote": "",
                "Game Info": "KC@BUF 09/10/2025 08:20PM ET",
            }
        )
    pool = pd.DataFrame(rows)
    stamped = stamp_projection_baseline(pool)
    assert list(stamped["ProjectionBaseline"]) == pytest.approx([12.00, 12.00])
    solved = projections_for_solve(stamped, _kc_buf_lines(50.0, -10.0), {})
    assert list(solved["Projection"]) == pytest.approx([12.60, 12.60])
    again = projections_for_solve(solved, _kc_buf_lines(50.0, -10.0), {})
    assert list(again["Projection"]) == pytest.approx([12.60, 12.60])
    fppg = _solver_fppg(solved, showdown=True)
    assert fppg["flex"] == pytest.approx(12.60)
    assert fppg["cpt"] == pytest.approx(18.90)

    fallback = pool.copy()
    fallback["Sleeper"] = 10.00
    fallback["PPR"] = 10.00
    fallback["Scoring"] = "PPR fallback"
    fallback = fallback.drop(columns=["ProjectionBaseline"], errors="ignore")
    fallback_base = stamp_projection_baseline(fallback)
    assert list(fallback_base["ProjectionBaseline"]) == pytest.approx([10.00, 10.00])


def test_scoring_version_upgrade_clears_old_baselines_and_lineups():
    from src.sleeper import drop_stale_scoring_session

    pool = _worthy_pool()
    pool["ProjectionBaseline"] = 8.0
    pool.attrs["scoring_version"] = "dk-component-v1"
    state = {
        "player_pool": pool,
        "lineups": [pd.DataFrame({"Name": ["Xavier Worthy"]})],
        "generated_run": {"preset": "classic_cash"},
        "sleeper_summary": {"scoring": "dk-component-v1", "ok": True},
    }
    assert drop_stale_scoring_session(state) is True
    assert state["player_pool"] is None
    assert state["lineups"] == []
    assert state["generated_run"] is None
    assert state["sleeper_summary"] is None
