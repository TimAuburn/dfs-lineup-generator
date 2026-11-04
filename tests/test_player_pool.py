"""Player pool collapse, include, captain, and ownership."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.player_pool import (
    apply_pool_edits,
    collapse_for_editor,
    filter_editor_frame,
    game_choices,
    generation_pool,
    locked_player_ids,
    position_choices,
    set_captain,
    set_included,
)


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
    at.session_state["player_pool"] = _showdown_pool()
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
