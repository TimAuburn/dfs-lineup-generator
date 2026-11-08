"""Stale in-memory pools must not be labeled as current DraftKings scores."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

from scripts.fit_bonus_scales import cohort_scale, median_scale, sample_sd
from src.dk_scoring import (
    PASS_YARDS_SD,
    POINTS_ALLOWED_SD,
    REC_YARDS_RB_SD,
    REC_YARDS_SD,
    REC_YARDS_TE_SD,
    RUSH_YARDS_SD,
    SCORING_VERSION,
)
from src.sleeper import drop_stale_scoring_session, scoring_metadata_current


def _current_pool(points: float) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "Name": ["Josh Allen"],
            "Position": ["QB"],
            "TeamAbbrev": ["BUF"],
            "Salary": [7400],
            "AvgPointsPerGame": [23.0],
            "Sleeper": [points],
            "PPR": [22.03],
            "Projection": [points],
            "Scoring": ["DraftKings"],
            "ScoringNote": [""],
        }
    )
    frame.attrs["scoring_version"] = SCORING_VERSION
    return frame


def _summary() -> dict:
    return {
        "ok": True,
        "scoring": SCORING_VERSION,
        "season": "2026",
        "week": 9,
        "matched": 1,
        "rows": 1,
        "ppr_fallback_rows": 0,
    }


def test_pool_without_scoring_version_is_cleared_with_its_lineups():
    stale = pd.DataFrame(
        {"Name": ["Josh Allen"], "Projection": [22.03], "Sleeper": [22.03]}
    )
    lineup = pd.DataFrame({"Name": ["Josh Allen"], "Projection": [22.03]})
    state = {
        "player_pool": stale,
        "sleeper_summary": {
            "ok": True,
            "season": "2026",
            "week": 9,
            "matched": 1,
            "rows": 1,
            "ppr_fallback_rows": 0,
        },
        "lineups": [lineup],
        "generated_run": {"preset": "classic_small_gpp"},
    }
    assert scoring_metadata_current(stale, state["sleeper_summary"]) is False
    assert drop_stale_scoring_session(state) is True
    assert state["player_pool"] is None
    assert state["lineups"] == []
    assert state["generated_run"] is None
    assert state["sleeper_summary"] is None


def test_current_scoring_version_keeps_the_pool_and_lineups():
    pool = _current_pool(22.89)
    lineup = pd.DataFrame({"Name": ["Josh Allen"], "Projection": [22.89]})
    state = {
        "player_pool": pool,
        "sleeper_summary": _summary(),
        "lineups": [lineup],
        "generated_run": {"preset": "classic_small_gpp"},
    }
    assert drop_stale_scoring_session(state) is False
    assert state["player_pool"] is pool
    assert float(state["player_pool"].loc[0, "Projection"]) == pytest.approx(22.89)
    assert state["lineups"] == [lineup]


@pytest.mark.parametrize("column", ["PPR", "Scoring", "ScoringNote"])
def test_missing_scoring_column_clears_pool_and_lineups(column):
    pool = _current_pool(22.89).drop(columns=[column])
    summary = _summary()
    lineup = pd.DataFrame({"Name": ["Josh Allen"], "Projection": [22.89]})
    state = {
        "player_pool": pool,
        "sleeper_summary": summary,
        "lineups": [lineup],
        "generated_run": {"preset": "classic_small_gpp"},
    }
    assert scoring_metadata_current(pool, summary) is False
    assert drop_stale_scoring_session(state) is True
    assert state["player_pool"] is None
    assert state["lineups"] == []
    from src.sleeper import sleeper_status_caption

    text = sleeper_status_caption(summary, pool)
    assert "DraftKings scoring from Sleeper" not in text
    assert "0 row(s) flagged" not in text
    assert "missing current DraftKings scoring metadata" in text


def test_old_scoring_version_is_not_current():
    pool = _current_pool(22.89)
    pool.attrs["scoring_version"] = "dk-component-v1"
    summary = _summary()
    summary["scoring"] = "dk-component-v1"
    assert scoring_metadata_current(pool, summary) is False


def test_bonus_scale_median_uses_sample_sd_and_the_two_central_values():
    assert sample_sd([0.0, 0.0]) == pytest.approx(0.0)
    assert median_scale([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)
    count, scale = cohort_scale({"a": [0.0, 2.0], "b": [0.0, 4.0, 8.0]}, min_weeks=2)
    assert count == 2
    assert scale == pytest.approx((sample_sd([0.0, 2.0]) + sample_sd([0.0, 4.0, 8.0])) / 2)


def test_checked_in_scales_match_the_2025_refit():
    """Rounded output of scripts/fit_bonus_scales.py on 2026-11-07."""
    assert PASS_YARDS_SD == pytest.approx(63.27)
    assert RUSH_YARDS_SD == pytest.approx(31.57)
    assert REC_YARDS_SD == pytest.approx(28.57)
    assert REC_YARDS_TE_SD == pytest.approx(24.01)
    assert REC_YARDS_RB_SD == pytest.approx(16.94)
    assert POINTS_ALLOWED_SD == pytest.approx(9.28)


def test_streamlit_rerun_drops_a_stale_pool_and_keeps_a_current_one():
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    stale = pd.DataFrame(
        {"Name": ["Josh Allen"], "Projection": [22.03], "Sleeper": [22.03]}
    )
    lineup = pd.DataFrame({"Slot": ["QB"], "Name": ["Josh Allen"], "Projection": [22.03]})
    app = AppTest.from_file(str(APP_PATH), default_timeout=90)
    app.session_state["player_pool"] = stale
    app.session_state["lineups"] = [lineup]
    app.session_state["generated_run"] = {"preset": "old"}
    app.session_state["sleeper_summary"] = {
        "ok": True,
        "season": "2026",
        "week": 9,
        "matched": 1,
        "rows": 1,
        "ppr_fallback_rows": 0,
    }
    app.run()
    assert not app.exception
    assert app.session_state["player_pool"] is None
    assert app.session_state["lineups"] == []
    assert app.session_state["generated_run"] is None
    text = " ".join(getattr(block, "value", "") for block in app.markdown) + " ".join(
        getattr(block, "value", "") for block in getattr(app, "info", [])
    )
    assert "22.03" not in text

    current = _current_pool(22.89)
    kept = AppTest.from_file(str(APP_PATH), default_timeout=90)
    kept.session_state["player_pool"] = current
    kept.session_state["lineups"] = []
    kept.session_state["generated_run"] = None
    kept.session_state["sleeper_summary"] = _summary()
    kept.session_state["contest_format"] = "classic"
    kept.run()
    assert not kept.exception
    assert kept.session_state["player_pool"] is not None
    assert float(kept.session_state["player_pool"].loc[0, "Projection"]) == pytest.approx(22.89)
    assert list(kept.session_state["player_pool"]["Scoring"]) == ["DraftKings"]
    assert kept.session_state["lineups"] == []
