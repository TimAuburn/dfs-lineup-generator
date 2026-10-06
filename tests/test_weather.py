"""Env loading, indoor stadium map, and wind weather downgrades."""

from __future__ import annotations

import json
from datetime import datetime
from urllib.error import URLError

import pandas as pd
import pytest

from src.env_config import (
    ODDS_API_KEY_ENV,
    WEATHER_API_KEY_ENV,
    get_odds_api_key,
    get_weather_api_key,
    key_status,
    load_env,
)
from src.stadiums import INDOOR_TEAM_ABBREVS, STADIUMS, is_outdoor_stadium
from src.weather import (
    WIND_K_FACTOR,
    WIND_PASS_FACTOR,
    WIND_RB_FACTOR,
    WIND_THRESHOLD_MPH,
    apply_weather_projections,
    fetch_weather_for_homes,
    indoor_teams,
    parse_home_team,
    parse_weather_payload,
    should_apply_wind,
    weather_adjustment_labels,
    wind_factor_for_position,
)


def test_load_env_reads_odds_and_weather_keys(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ODDS_API_KEY=odds-secret\nWEATHER_API_KEY=wx-secret\n",
        encoding="utf-8",
    )
    monkeypatch.delenv(ODDS_API_KEY_ENV, raising=False)
    monkeypatch.delenv(WEATHER_API_KEY_ENV, raising=False)
    load_env(env_file, override=True)
    assert get_odds_api_key() == "odds-secret"
    assert get_weather_api_key() == "wx-secret"
    assert "loaded from .env" in key_status(ODDS_API_KEY_ENV)
    assert "secret" not in key_status(WEATHER_API_KEY_ENV)


def test_blank_env_keys_are_none(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("ODDS_API_KEY=\nWEATHER_API_KEY=   \n", encoding="utf-8")
    monkeypatch.delenv(ODDS_API_KEY_ENV, raising=False)
    monkeypatch.delenv(WEATHER_API_KEY_ENV, raising=False)
    load_env(env_file, override=True)
    assert get_odds_api_key() is None
    assert get_weather_api_key() is None
    assert "missing" in key_status(ODDS_API_KEY_ENV)


def test_indoor_stadium_map():
    expected_indoor = {"ARI", "ATL", "DAL", "DET", "HOU", "IND", "LV", "MIN", "NO"}
    assert set(indoor_teams()) == expected_indoor
    assert INDOOR_TEAM_ABBREVS == expected_indoor
    for team in expected_indoor:
        assert is_outdoor_stadium(team) is False
        assert STADIUMS[team].is_outdoor is False
    # SoFi is outdoor for both LA clubs.
    assert is_outdoor_stadium("LAR") is True
    assert is_outdoor_stadium("LAC") is True
    assert is_outdoor_stadium("BUF") is True
    # Unknown abbrev defaults to outdoor (do not skip wind silently).
    assert is_outdoor_stadium("XYZ") is True


def test_parse_home_team_and_wind_gate():
    assert parse_home_team("KC@BUF 09/10/2025 08:20PM ET") == "BUF"
    assert parse_home_team("IND@WAS") == "WAS"
    assert parse_home_team("In Progress") is None
    assert should_apply_wind("BUF", WIND_THRESHOLD_MPH + 0.1) is True
    assert should_apply_wind("BUF", WIND_THRESHOLD_MPH) is False
    assert should_apply_wind("DAL", 40.0) is False  # retractable / enclosed
    assert should_apply_wind("MIN", 25.0) is False


def test_wind_factors_by_position():
    assert wind_factor_for_position("QB") == WIND_PASS_FACTOR
    assert wind_factor_for_position("WR/FLEX") == WIND_PASS_FACTOR
    assert wind_factor_for_position("TE") == WIND_PASS_FACTOR
    assert wind_factor_for_position("RB") == WIND_RB_FACTOR
    assert wind_factor_for_position("K") == WIND_K_FACTOR
    assert wind_factor_for_position("DST") == 1.0


def test_apply_weather_downgrades_outdoor_high_wind_only():
    pool = pd.DataFrame(
        [
            {
                "Name": "Allen",
                "TeamAbbrev": "BUF",
                "Position": "QB",
                "Projection": 20.0,
                "Game Info": "KC@BUF 01:00PM ET",
            },
            {
                "Name": "Diggs",
                "TeamAbbrev": "BUF",
                "Position": "WR",
                "Projection": 12.0,
                "Game Info": "KC@BUF 01:00PM ET",
            },
            {
                "Name": "Cook",
                "TeamAbbrev": "BUF",
                "Position": "RB",
                "Projection": 14.0,
                "Game Info": "KC@BUF 01:00PM ET",
            },
            {
                "Name": "Bass",
                "TeamAbbrev": "BUF",
                "Position": "K",
                "Projection": 8.0,
                "Game Info": "KC@BUF 01:00PM ET",
            },
            {
                "Name": "BillsDST",
                "TeamAbbrev": "BUF",
                "Position": "DST",
                "Projection": 7.0,
                "Game Info": "KC@BUF 01:00PM ET",
            },
            {
                "Name": "Prescott",
                "TeamAbbrev": "DAL",
                "Position": "QB",
                "Projection": 18.0,
                "Game Info": "PHI@DAL 04:25PM ET",
            },
            {
                "Name": "Stafford",
                "TeamAbbrev": "LAR",
                "Position": "QB",
                "Projection": 16.0,
                "Game Info": "SEA@LAR 04:05PM ET",
            },
        ]
    )
    weather = {
        "BUF": {
            "home_team": "BUF",
            "is_outdoor": True,
            "wind_mph": 22.0,
            "applies_wind": True,
        },
        "DAL": {
            "home_team": "DAL",
            "is_outdoor": False,
            "wind_mph": 30.0,
            "applies_wind": False,
        },
        "LAR": {
            "home_team": "LAR",
            "is_outdoor": True,
            "wind_mph": 10.0,
            "applies_wind": False,
        },
    }
    adjusted = apply_weather_projections(pool, weather)
    by_name = dict(zip(adjusted["Name"], adjusted["Projection"]))
    assert by_name["Allen"] == pytest.approx(20.0 * WIND_PASS_FACTOR)
    assert by_name["Diggs"] == pytest.approx(12.0 * WIND_PASS_FACTOR)
    assert by_name["Cook"] == pytest.approx(14.0)  # RB unchanged
    assert by_name["Bass"] == pytest.approx(8.0 * WIND_K_FACTOR)
    assert by_name["BillsDST"] == pytest.approx(7.0)
    assert by_name["Prescott"] == pytest.approx(18.0)  # indoor, ignore wind
    assert by_name["Stafford"] == pytest.approx(16.0)  # outdoor but under threshold

    labels = weather_adjustment_labels(pool, weather)
    assert any(label.startswith("BUF wind 22") for label in labels)
    assert not any(label.startswith("DAL") for label in labels)


def test_parse_weather_payload_sets_applies_wind():
    payload = {
        "windSpeed": 19,
        "windGust": 28,
        "temperature": 48,
        "wxPhraseLong": "Windy",
    }
    outdoor = parse_weather_payload(payload, home_team="BUF")
    assert outdoor["wind_mph"] == 19.0
    assert outdoor["applies_wind"] is True
    indoor = parse_weather_payload(payload, home_team="NO")
    assert indoor["is_outdoor"] is False
    assert indoor["applies_wind"] is False


def test_fetch_weather_skips_indoor_and_mocks_http(tmp_path, monkeypatch):
    cache = tmp_path / "weather_cache.json"
    calls: list[str] = []

    def _ok(url: str, timeout: float = 20):
        calls.append(url)
        assert "api.weather.com/v3/wx/observations/current" in url
        assert "apiKey=wx-secret" in url
        assert "geocode=" in url
        return {
            "windSpeed": 24,
            "windGust": 31,
            "temperature": 55,
            "wxPhraseLong": "Windy",
        }

    monkeypatch.setattr("src.weather._http_get", _ok)
    now = datetime(2026, 10, 5, 12, 0)
    result = fetch_weather_for_homes(
        ["DAL", "BUF"],
        api_key="wx-secret",
        force_refresh=True,
        cache_path=cache,
        now=now,
    )
    assert result["DAL"]["is_outdoor"] is False
    assert result["DAL"]["applies_wind"] is False
    assert result["BUF"]["wind_mph"] == 24.0
    assert result["BUF"]["applies_wind"] is True
    # Only outdoor BUF should hit the network.
    assert len(calls) == 1
    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert "BUF" in saved["data"]
    assert "wx-secret" not in cache.read_text(encoding="utf-8")


def test_fetch_weather_uses_cache_and_survives_errors(tmp_path, monkeypatch):
    cache = tmp_path / "weather_cache.json"
    cached = {
        "BUF": {
            "home_team": "BUF",
            "is_outdoor": True,
            "wind_mph": 20.0,
            "applies_wind": True,
        }
    }
    cache.write_text(
        json.dumps({"timestamp": datetime(2026, 10, 5, 12, 0).timestamp(), "data": cached}),
        encoding="utf-8",
    )

    def _boom(url: str, timeout: float = 20):
        raise URLError("down")

    monkeypatch.setattr("src.weather._http_get", _boom)
    # Fresh cache: no network.
    fresh = fetch_weather_for_homes(
        ["BUF"],
        api_key="wx-secret",
        force_refresh=False,
        cache_path=cache,
        now=datetime(2026, 10, 5, 12, 30),
    )
    assert fresh["BUF"]["wind_mph"] == 20.0

    # Force refresh fails → fall back to cache.
    fallback = fetch_weather_for_homes(
        ["BUF"],
        api_key="wx-secret",
        force_refresh=True,
        cache_path=cache,
        now=datetime(2026, 10, 5, 18, 0),
    )
    assert fallback["BUF"]["wind_mph"] == 20.0

    # Missing key with no cache entry for SEA.
    empty = fetch_weather_for_homes(
        ["SEA"],
        api_key=None,
        force_refresh=True,
        cache_path=tmp_path / "missing.json",
        now=datetime(2026, 10, 5, 18, 0),
    )
    assert empty["SEA"]["error"] == "missing_api_key"
