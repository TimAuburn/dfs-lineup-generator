"""A rejected salary upload must not replace the pool or the lineups.

Detection and parsing classify the file. The page only swaps the pool when
that decision accepts it. These tests load a real pool, build a lineup, then
upload a file the parser or the detector rejects.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from src.optimizer import NFLClassicOptimizer
from src.parser import (
    DKCSVValidationError,
    assess_salary_upload,
    detect_salary_file_format,
    parse_dk_csv,
)
from src.presets import ContestFormat, ContestPreset, apply_preset

MOCK_CLASSIC = Path(__file__).parent / "mock_dk_salaries.csv"
MOCK_SHOWDOWN = Path(__file__).parent / "mock_dk_showdown.csv"
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
HEADER = (
    "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,"
    "TeamAbbrev,AvgPointsPerGame"
)


def _row(position: str, name: str, player_id: str, slot: str, salary: object) -> str:
    return (
        f"{position},{name} ({player_id}),{name},{player_id},{slot},{salary},"
        f"KC@BUF 09/10/2025 08:20PM ET,KC,10.0"
    )


def _csv(*rows: str) -> bytes:
    return (HEADER + "\n" + "\n".join(rows) + "\n").encode()


def _malformed() -> tuple[str, bytes, str]:
    """Bytes the detector cannot read as a salary CSV."""
    payload = b"\xff\xfe not a salary file"
    found = detect_salary_file_format(payload)
    assert found.contest_format is None
    assert found.warning is not None
    assert "could not be read" in found.warning
    return "not-a-salary.csv", payload, found.warning


def _mixed() -> tuple[str, bytes, str]:
    """Classic and Showdown roster positions in one file."""
    payload = _csv(
        _row("QB", "Patrick Mahomes", "1001", "QB", 7200),
        _row("RB", "Isiah Pacheco", "2001", "RB/FLEX", 8000),
        _row("QB", "Josh Allen", "91003", "CPT", 17400),
        _row("QB", "Josh Allen", "91004", "FLEX", 11600),
    )
    found = detect_salary_file_format(payload)
    assert found.contest_format is None
    assert found.warning is not None
    assert "mixes" in found.warning
    return "mixed-slate.csv", payload, found.warning


def _warning() -> tuple[str, bytes, str]:
    """A roster the detector will not call Classic or Showdown.

    ``parse_dk_csv`` would still load this as Classic. The warning is what
    keeps it from replacing the pool.
    """
    payload = _csv(_row("OL", "Someone", "1", "OL", 3000))
    found = detect_salary_file_format(payload)
    assert found.contest_format is None
    assert found.warning is not None
    assert "Could not detect" in found.warning
    parsed = parse_dk_csv(BytesIO(payload))
    assert detect_contest_format_is_classic(parsed)
    return "ambiguous-roster.csv", payload, found.warning


def detect_contest_format_is_classic(frame) -> bool:
    from src.parser import detect_contest_format

    return detect_contest_format(frame) is ContestFormat.CLASSIC


def _invalid_salary() -> tuple[str, bytes, str]:
    """Columns and a Classic slot that parse rejects."""
    payload = _csv(_row("QB", "Patrick Mahomes", "1001", "QB", "not-a-salary"))
    found = detect_salary_file_format(payload)
    assert found.contest_format is ContestFormat.CLASSIC
    assert found.warning is None
    with pytest.raises(DKCSVValidationError) as raised:
        parse_dk_csv(BytesIO(payload), contest_format=ContestFormat.CLASSIC)
    assert "Salary" in str(raised.value)
    return "bad-salary.csv", payload, str(raised.value)


def _player_pool():
    app_test = pytest.importorskip("streamlit.testing.v1")
    app = app_test.AppTest.from_file(str(APP_PATH), default_timeout=180)
    app.session_state["app_view"] = "Player pool"
    app.session_state["contest_format"] = ContestFormat.CLASSIC.value
    app.session_state["contest_preset"] = ContestPreset.CLASSIC_LARGE_GPP.value
    return app


def _uploader(app):
    return next(item for item in app.file_uploader if item.label == "DraftKings player pool CSV")


def _upload(app, name: str, content: bytes):
    app.run()
    assert not app.exception, app.exception
    _uploader(app).set_value((name, content, "text/csv"))
    app.run()
    assert not app.exception, app.exception
    return app


def _built_lineups():
    pool = parse_dk_csv(MOCK_CLASSIC)
    optimizer = NFLClassicOptimizer()
    optimizer.load_from_dataframe(pool)
    apply_preset(optimizer, ContestPreset.CLASSIC_CASH, jitter_seed=42)
    built = optimizer.generate_lineups()
    assert built
    return built


def _invalid_showdown_salary() -> tuple[str, bytes, str]:
    """Showdown slots with a salary ``parse_dk_csv`` rejects.

    Detection names Showdown. Accepting that format before the salary check
    would switch the page off Classic and clear the lineups.
    """
    payload = _csv(
        _row("QB", "Josh Allen", "91003", "CPT", "nope"),
        _row("QB", "Josh Allen", "91004", "FLEX", 11600),
    )
    found = detect_salary_file_format(payload)
    assert found.contest_format is ContestFormat.SHOWDOWN
    assert found.warning is None
    with pytest.raises(DKCSVValidationError) as raised:
        parse_dk_csv(BytesIO(payload), contest_format=ContestFormat.SHOWDOWN)
    assert "Salary" in str(raised.value)
    return "bad-showdown.csv", payload, str(raised.value)


def _pool_rows(pool) -> tuple:
    columns = [name for name in ("ID", "Name", "TeamAbbrev", "Salary", "Roster Position") if name in pool.columns]
    frame = pool.loc[:, columns]
    return tuple(tuple(str(value) for value in row) for row in frame.itertuples(index=False, name=None))


def _lineup_ids(lineups) -> tuple:
    return tuple(tuple(str(value) for value in lineup["ID"].tolist()) for lineup in lineups)


def _load_pool_and_lineups(app, lineups):
    app = _upload(app, "mock_dk_salaries.csv", MOCK_CLASSIC.read_bytes())
    assert app.session_state["player_pool"] is not None
    assert app.session_state["contest_format"] == ContestFormat.CLASSIC.value
    assert app.session_state["contest_preset"] == ContestPreset.CLASSIC_LARGE_GPP.value
    app.session_state["lineups"] = lineups
    app.session_state["generated_run"] = {"preset": ContestPreset.CLASSIC_LARGE_GPP.value, "kept": True}
    return app


def _messages(app) -> str:
    chunks = [str(item.value) for item in list(app.warning) + list(app.error)]
    return "\n".join(chunks)


@pytest.fixture(scope="module")
def built_lineups():
    return _built_lineups()


@pytest.mark.parametrize(
    "bad_file",
    [_malformed, _mixed, _warning, _invalid_salary, _invalid_showdown_salary],
    ids=["malformed", "mixed", "warning", "invalid-salary", "invalid-showdown"],
)
def test_rejected_upload_keeps_pool_lineups_format_and_preset(built_lineups, bad_file):
    name, payload, problem = bad_file()
    app = _load_pool_and_lineups(_player_pool(), built_lineups)
    pool_rows = _pool_rows(app.session_state["player_pool"])
    lineup_ids = _lineup_ids(app.session_state["lineups"])
    generated = app.session_state["generated_run"]

    _uploader(app).set_value((name, payload, "text/csv"))
    app.run()
    assert not app.exception, app.exception
    _assert_kept(app, pool_rows, lineup_ids, generated, name, problem)

    # The rejected file is still attached. A rerun must not treat it as new.
    app.run()
    assert not app.exception, app.exception
    _assert_kept(app, pool_rows, lineup_ids, generated, name, problem)


def _assert_kept(app, pool_rows, lineup_ids, generated, name, problem) -> None:
    pool = app.session_state["player_pool"]
    assert pool is not None
    assert _pool_rows(pool) == pool_rows
    assert _lineup_ids(app.session_state["lineups"]) == lineup_ids
    assert lineup_ids
    assert app.session_state["contest_format"] == ContestFormat.CLASSIC.value
    assert app.session_state["contest_preset"] == ContestPreset.CLASSIC_LARGE_GPP.value
    assert app.session_state["generated_run"] == generated
    shown = _messages(app)
    assert name in shown
    assert problem in shown
    captions = [str(item.value) for item in app.caption]
    assert not any(value.startswith("Detected:") for value in captions)
    assert not any("Upload a DraftKings" in str(item.value) for item in app.info)


def test_assess_salary_upload_rejects_without_session_state():
    """The accept decision is the parser's, not the page's."""
    name, payload, problem = _mixed()
    mixed = assess_salary_upload(payload, filename=name)
    assert mixed.accepted is False
    assert mixed.level == "warning"
    assert mixed.detection.contest_format is None
    assert mixed.message == f"{name}: {problem}"

    name, payload, problem = _malformed()
    broken = assess_salary_upload(payload, filename=name)
    assert broken.accepted is False
    assert broken.level == "warning"
    assert broken.message == f"{name}: {problem}"

    name, payload, problem = _warning()
    ambiguous = assess_salary_upload(payload, filename=name)
    assert ambiguous.accepted is False
    assert ambiguous.level == "warning"
    assert "Could not detect" in ambiguous.message
    assert ambiguous.message == f"{name}: {problem}"

    name, payload, problem = _invalid_salary()
    invalid = assess_salary_upload(payload, filename=name)
    assert invalid.accepted is False
    assert invalid.level == "error"
    assert invalid.detection.contest_format is ContestFormat.CLASSIC
    assert invalid.message == f"{name}: {problem}"

    name, payload, problem = _invalid_showdown_salary()
    showdown = assess_salary_upload(payload, filename=name)
    assert showdown.accepted is False
    assert showdown.level == "error"
    assert showdown.detection.contest_format is ContestFormat.SHOWDOWN
    assert showdown.message == f"{name}: {problem}"

    classic = assess_salary_upload(MOCK_CLASSIC.read_bytes(), filename="mock_dk_salaries.csv")
    assert classic.accepted is True
    assert classic.level == ""
    assert classic.message == ""
    assert classic.detection.contest_format is ContestFormat.CLASSIC
    assert classic.detection.caption == "Detected: Classic"

    showdown_file = assess_salary_upload(MOCK_SHOWDOWN.read_bytes(), filename="")
    assert showdown_file.accepted is True
    assert showdown_file.detection.contest_format is ContestFormat.SHOWDOWN


def test_valid_replacement_swaps_the_pool_and_clears_lineups(built_lineups):
    app = _load_pool_and_lineups(_player_pool(), built_lineups)
    previous = app.session_state["player_pool"]

    _uploader(app).set_value(("mock_dk_showdown.csv", MOCK_SHOWDOWN.read_bytes(), "text/csv"))
    app.run()
    assert not app.exception, app.exception

    pool = app.session_state["player_pool"]
    assert pool is not None
    assert pool is not previous
    assert "RosterSlot" in pool.columns
    assert set(pool["RosterSlot"].astype(str)) >= {"CPT", "FLEX"}
    assert app.session_state["lineups"] == []
    assert app.session_state["generated_run"] is None
    assert app.session_state["contest_format"] == ContestFormat.SHOWDOWN.value
    assert app.session_state["contest_preset"] == ContestPreset.SHOWDOWN_CASH.value
    assert "Detected: Showdown" in [item.value for item in app.caption]
