"""Contest format detected from an uploaded DraftKings salary file."""

from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import pytest

from src.parser import (
    DKCSVValidationError,
    detect_contest_format,
    detect_salary_file_format,
    parse_dk_csv,
)
from src.presets import ContestFormat, ContestPreset, PRESET_LABELS, presets_for_format

MOCK_CLASSIC = Path(__file__).parent / "mock_dk_salaries.csv"
MOCK_SHOWDOWN = Path(__file__).parent / "mock_dk_showdown.csv"
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
HEADER = (
    "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,"
    "TeamAbbrev,AvgPointsPerGame"
)


def _row(position: str, name: str, player_id: str, slot: str, salary: int) -> str:
    return (
        f"{position},{name} ({player_id}),{name},{player_id},{slot},{salary},"
        f"KC@BUF 09/10/2025 08:20PM ET,KC,10.0"
    )


def _csv(*rows: str) -> bytes:
    return (HEADER + "\n" + "\n".join(rows) + "\n").encode()


def _captain_pairs_are_exact_1_5x(path: Path) -> bool:
    frame = parse_dk_csv(path, contest_format=ContestFormat.SHOWDOWN)
    slots = frame["Roster Position"].str.upper()
    captains = frame.loc[slots.eq("CPT")].set_index("Name")["Salary"]
    flex = frame.loc[slots.eq("FLEX")].set_index("Name")["Salary"]
    if captains.empty or set(captains.index) != set(flex.index):
        return False
    return all(
        int(captains.loc[name]) * 2 == int(flex.loc[name]) * 3 for name in captains.index
    )


def test_classic_fixture_detects_classic_including_flex_forms():
    found = detect_salary_file_format(MOCK_CLASSIC)
    assert found.contest_format is ContestFormat.CLASSIC
    assert found.caption == "Detected: Classic"
    assert found.warning is None
    parsed = parse_dk_csv(MOCK_CLASSIC)
    assert detect_contest_format(parsed) is ContestFormat.CLASSIC
    slots = set(parsed["Roster Position"].str.upper())
    assert {"QB", "RB/FLEX", "WR/FLEX", "TE/FLEX", "DST"} <= slots


def test_showdown_fixture_with_exact_1_5x_captains_detects_showdown():
    assert _captain_pairs_are_exact_1_5x(MOCK_SHOWDOWN)
    found = detect_salary_file_format(MOCK_SHOWDOWN)
    assert found.contest_format is ContestFormat.SHOWDOWN
    assert found.caption == "Detected: Showdown"
    assert found.warning is None
    parsed = parse_dk_csv(MOCK_SHOWDOWN)
    assert detect_contest_format(parsed) is ContestFormat.SHOWDOWN


def test_showdown_without_1_5x_captain_price_is_still_showdown():
    text = MOCK_SHOWDOWN.read_text(encoding="utf-8").replace(",CPT,16800,", ",CPT,16000,", 1)
    found = detect_salary_file_format(text.encode())
    assert found.contest_format is ContestFormat.SHOWDOWN
    assert found.caption == "Detected: Showdown"
    assert found.warning is None


def test_mixed_classic_and_showdown_positions_do_not_auto_switch():
    payload = _csv(
        _row("QB", "Patrick Mahomes", "1001", "QB", 7200),
        _row("QB", "Patrick Mahomes", "91001", "CPT", 10800),
        _row("QB", "Patrick Mahomes", "91002", "FLEX", 7200),
    )
    found = detect_salary_file_format(payload)
    assert found.contest_format is None
    assert found.caption is None
    assert found.warning is not None
    assert "mixes" in found.warning
    assert "left unchanged" in found.warning


def test_garbage_and_unrecognized_files_do_not_auto_switch():
    garbage = detect_salary_file_format(b"\xff\xfe not a salary file")
    assert garbage.contest_format is None
    assert garbage.caption is None
    assert garbage.warning is not None
    assert "could not be read" in garbage.warning

    unknown = detect_salary_file_format(
        _csv(_row("OL", "Someone", "1", "OL", 3000))
    )
    assert unknown.contest_format is None
    assert unknown.warning is not None
    assert "Could not detect" in unknown.warning
    assert "left unchanged" in unknown.warning


def test_missing_columns_do_not_auto_switch_and_name_the_gap():
    header = "Position,Name + ID,Name,ID,Roster Position,Game Info,TeamAbbrev,AvgPointsPerGame"
    payload = header + "\n" + "QB,Patrick Mahomes (1001),Patrick Mahomes,1001,QB,KC@BUF,KC,10\n"
    found = detect_salary_file_format(StringIO(payload))
    assert found.contest_format is None
    assert found.warning is not None
    assert "Salary" in found.warning
    assert "left unchanged" in found.warning
    with pytest.raises(DKCSVValidationError, match="Salary"):
        parse_dk_csv(StringIO(payload))


def test_preamble_uses_the_same_header_scan_as_parse():
    text = "DraftKings instructions\n" + MOCK_CLASSIC.read_text(encoding="utf-8")
    found = detect_salary_file_format(BytesIO(text.encode()))
    assert found.contest_format is ContestFormat.CLASSIC
    parsed = parse_dk_csv(BytesIO(text.encode()))
    assert detect_contest_format(parsed) is ContestFormat.CLASSIC


def _player_pool(timeout: float = 180):
    app_test = pytest.importorskip("streamlit.testing.v1")
    app = app_test.AppTest.from_file(str(APP_PATH), default_timeout=timeout)
    app.session_state["app_view"] = "Player pool"
    return app


def _upload(app, name: str, content: bytes):
    app.run()
    assert not app.exception, app.exception
    return _replace_file(app, name, content)


def _replace_file(app, name: str, content: bytes):
    """Attach a file and run once. No rerun of the file already on the page."""
    uploader = next(
        item for item in app.file_uploader if item.label == "DraftKings player pool CSV"
    )
    uploader.set_value((name, content, "text/csv"))
    app.run()
    assert not app.exception, app.exception
    return app


def _preset(app):
    return next(item for item in app.selectbox if item.label == "Contest Preset")


def _captions(app) -> list[str]:
    return [str(item.value) for item in app.caption]


def _assert_no_format_radio(app) -> None:
    assert all(item.label != "Contest Format" for item in app.radio)
    assert all("Contest Format" not in str(item.value) for item in app.markdown)


def _pool_ids(app) -> set[str]:
    pool = app.session_state["player_pool"]
    assert pool is not None
    return set(pool["ID"].astype(str))


def test_uploading_classic_autosets_classic_and_a_classic_preset():
    app = _player_pool()
    app.session_state["contest_format"] = ContestFormat.SHOWDOWN.value
    app.session_state["contest_preset"] = ContestPreset.SHOWDOWN_GPP.value
    app = _upload(app, "mock_dk_salaries.csv", MOCK_CLASSIC.read_bytes())
    assert app.session_state["contest_format"] == ContestFormat.CLASSIC.value
    assert app.session_state["contest_preset"] == ContestPreset.CLASSIC_CASH.value
    captions = [item.value for item in app.caption]
    assert "Detected: Classic" in captions
    options = list(_preset(app).options)
    classic_labels = {PRESET_LABELS[preset] for preset in presets_for_format(ContestFormat.CLASSIC)}
    assert set(options) == classic_labels
    assert PRESET_LABELS[ContestPreset.SHOWDOWN_CASH] not in options
    assert PRESET_LABELS[ContestPreset.SHOWDOWN_GPP] not in options


def test_uploading_showdown_autosets_showdown_and_shows_the_caption():
    app = _player_pool()
    app = _upload(app, "mock_dk_showdown.csv", MOCK_SHOWDOWN.read_bytes())
    assert app.session_state["contest_format"] == ContestFormat.SHOWDOWN.value
    assert app.session_state["contest_preset"] == ContestPreset.SHOWDOWN_CASH.value
    assert "Detected: Showdown" in [item.value for item in app.caption]
    options = list(_preset(app).options)
    showdown_labels = {
        PRESET_LABELS[preset] for preset in presets_for_format(ContestFormat.SHOWDOWN)
    }
    assert set(options) == showdown_labels


def test_manual_override_survives_reruns_until_a_new_file_is_uploaded():
    app = _player_pool()
    app = _upload(app, "mock_dk_showdown.csv", MOCK_SHOWDOWN.read_bytes())
    _preset(app).set_value(ContestPreset.SHOWDOWN_GPP.value)
    app.run()
    assert not app.exception, app.exception
    assert app.session_state["contest_preset"] == ContestPreset.SHOWDOWN_GPP.value
    assert app.session_state["contest_format"] == ContestFormat.SHOWDOWN.value

    app.run()
    assert not app.exception, app.exception
    assert app.session_state["contest_preset"] == ContestPreset.SHOWDOWN_GPP.value
    assert app.session_state["contest_format"] == ContestFormat.SHOWDOWN.value
    assert "Detected: Showdown" in _captions(app)
    _assert_no_format_radio(app)

    # A file of the other format is what switches. The preset resets and the
    # stored lineups are cleared, then a rerun leaves that choice in place.
    app.session_state["lineups"] = [pd.DataFrame({"ID": ["1"]})]
    app.session_state["generated_run"] = {"kept": True}
    app = _replace_file(app, "mock_dk_salaries.csv", MOCK_CLASSIC.read_bytes())
    assert app.session_state["contest_format"] == ContestFormat.CLASSIC.value
    assert app.session_state["contest_preset"] == ContestPreset.CLASSIC_CASH.value
    assert app.session_state["lineups"] == []
    assert app.session_state["generated_run"] is None
    assert "1001" in _pool_ids(app)
    assert "91001" not in _pool_ids(app)
    assert "Detected: Classic" in _captions(app)
    assert "Format: Classic (from salary file)" in _captions(app)
    app.run()
    assert not app.exception, app.exception
    assert app.session_state["contest_format"] == ContestFormat.CLASSIC.value
    assert app.session_state["contest_preset"] == ContestPreset.CLASSIC_CASH.value

    app = _upload(app, "mock_dk_showdown.csv", MOCK_SHOWDOWN.read_bytes())
    assert app.session_state["contest_format"] == ContestFormat.SHOWDOWN.value
    assert app.session_state["contest_preset"] == ContestPreset.SHOWDOWN_CASH.value
    assert "91001" in _pool_ids(app)
    assert "1001" not in _pool_ids(app)
    assert "Detected: Showdown" in _captions(app)
    assert "Format: Showdown (from salary file)" in _captions(app)


def test_mixed_upload_warns_and_leaves_format_and_preset_unchanged():
    app = _player_pool()
    app.session_state["contest_format"] = ContestFormat.SHOWDOWN.value
    app.session_state["contest_preset"] = ContestPreset.SHOWDOWN_GPP.value
    mixed = _csv(
        _row("QB", "Patrick Mahomes", "1001", "QB", 7200),
        _row("RB", "Isiah Pacheco", "2001", "RB/FLEX", 8000),
        _row("QB", "Josh Allen", "91003", "CPT", 17400),
        _row("QB", "Josh Allen", "91004", "FLEX", 11600),
    )
    app = _upload(app, "mixed.csv", mixed)
    warnings = [item.value for item in app.warning]
    assert any("mixes" in value and "left unchanged" in value for value in warnings)
    assert app.session_state["contest_format"] == ContestFormat.SHOWDOWN.value
    assert app.session_state["contest_preset"] == ContestPreset.SHOWDOWN_GPP.value
    assert "Detected: Showdown" not in [item.value for item in app.caption]
    assert "Detected: Classic" not in [item.value for item in app.caption]
    options = list(_preset(app).options)
    assert PRESET_LABELS[ContestPreset.CLASSIC_CASH] not in options
    assert PRESET_LABELS[ContestPreset.SHOWDOWN_GPP] in options


def test_format_radio_is_gone_and_format_follows_the_uploaded_file():
    """No format radio. Classic, then Showdown, then Classic again, then a rejection."""
    app = _player_pool()
    app.run()
    assert not app.exception, app.exception
    _assert_no_format_radio(app)
    assert app.session_state["contest_format"] == ContestFormat.CLASSIC.value
    assert "Format: detected when you upload a salary file" in _captions(app)
    assert "Format: Classic (from salary file)" not in _captions(app)

    app = _upload(app, "mock_dk_salaries.csv", MOCK_CLASSIC.read_bytes())
    _assert_no_format_radio(app)
    assert app.session_state["contest_format"] == ContestFormat.CLASSIC.value
    assert app.session_state["contest_preset"] == ContestPreset.CLASSIC_SMALL_GPP.value
    assert "1001" in _pool_ids(app)
    assert "Format: Classic (from salary file)" in _captions(app)
    classic_options = list(_preset(app).options)
    assert PRESET_LABELS[ContestPreset.SHOWDOWN_CASH] not in classic_options
    assert PRESET_LABELS[ContestPreset.CLASSIC_CASH] in classic_options

    app.session_state["lineups"] = [pd.DataFrame({"ID": ["1"]})]
    app.session_state["generated_run"] = {"kept": True}
    app = _replace_file(app, "mock_dk_showdown.csv", MOCK_SHOWDOWN.read_bytes())
    _assert_no_format_radio(app)
    assert app.session_state["contest_format"] == ContestFormat.SHOWDOWN.value
    assert app.session_state["contest_preset"] == ContestPreset.SHOWDOWN_CASH.value
    assert app.session_state["lineups"] == []
    assert app.session_state["generated_run"] is None
    assert "91001" in _pool_ids(app)
    assert "1001" not in _pool_ids(app)
    assert "Format: Showdown (from salary file)" in _captions(app)
    showdown_options = list(_preset(app).options)
    assert PRESET_LABELS[ContestPreset.CLASSIC_CASH] not in showdown_options
    assert PRESET_LABELS[ContestPreset.SHOWDOWN_GPP] in showdown_options

    app = _upload(app, "mock_dk_salaries.csv", MOCK_CLASSIC.read_bytes())
    assert app.session_state["contest_format"] == ContestFormat.CLASSIC.value
    assert app.session_state["contest_preset"] == ContestPreset.CLASSIC_CASH.value
    assert "1001" in _pool_ids(app)
    assert "91001" not in _pool_ids(app)
    assert "Format: Classic (from salary file)" in _captions(app)

    kept_preset = app.session_state["contest_preset"]
    app.session_state["lineups"] = [pd.DataFrame({"ID": ["9"]})]
    app.session_state["generated_run"] = {"kept": True}
    mixed = _csv(
        _row("QB", "Patrick Mahomes", "1001", "QB", 7200),
        _row("QB", "Josh Allen", "91003", "CPT", 17400),
        _row("QB", "Josh Allen", "91004", "FLEX", 11600),
    )
    app = _replace_file(app, "mixed.csv", mixed)
    _assert_no_format_radio(app)
    assert app.session_state["contest_format"] == ContestFormat.CLASSIC.value
    assert app.session_state["contest_preset"] == kept_preset
    assert app.session_state["lineups"]
    assert app.session_state["generated_run"] == {"kept": True}
    assert "1001" in _pool_ids(app)
    warnings = [item.value for item in app.warning]
    assert any("mixes" in value and "left unchanged" in value for value in warnings)
    assert "Format: Classic (from salary file)" in _captions(app)
    app.run()
    assert not app.exception, app.exception
    assert app.session_state["contest_format"] == ContestFormat.CLASSIC.value
    assert app.session_state["contest_preset"] == kept_preset
    assert "1001" in _pool_ids(app)
    assert "Format: Classic (from salary file)" in _captions(app)


def test_same_format_upload_keeps_a_preset_that_already_fits():
    app = _player_pool()
    app.session_state["contest_preset"] = ContestPreset.CLASSIC_LARGE_GPP.value
    app = _upload(app, "mock_dk_salaries.csv", MOCK_CLASSIC.read_bytes())
    assert app.session_state["contest_format"] == ContestFormat.CLASSIC.value
    assert app.session_state["contest_preset"] == ContestPreset.CLASSIC_LARGE_GPP.value
    assert "Detected: Classic" in [item.value for item in app.caption]
