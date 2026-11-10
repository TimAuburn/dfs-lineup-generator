"""Kickoff-hour forecast mapping. These tests do not call the network."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from zoneinfo import ZoneInfo

import pytest

from src.kickoff_forecast import (
    UNAVAILABLE,
    KickoffForecast,
    describe_hourly_period,
    forecasts_for_matchups,
    hourly_url_from_points,
    nws_request,
    weather_label,
)
from src.slate_games import SlateMatchup
from src.stadiums import get_stadium
from src.weather import NWS_USER_AGENT

EASTERN = ZoneInfo("America/New_York")
KICKOFF = datetime(2026, 11, 8, 13, 0, tzinfo=EASTERN)


def _period(**overrides) -> dict:
    period = {
        "startTime": "2026-11-08T13:00:00-05:00",
        "endTime": "2026-11-08T14:00:00-05:00",
        "temperature": 55,
        "temperatureUnit": "F",
        "windSpeed": "10 mph",
        "shortForecast": "Sunny",
        "probabilityOfPrecipitation": {"unitCode": "wmoUnit:percent", "value": 0},
    }
    period.update(overrides)
    return period


def _hourly(*periods) -> dict:
    return {"properties": {"periods": list(periods)}}


def _points(lat: float, lon: float) -> dict:
    return {
        "properties": {
            "forecastHourly": f"https://api.weather.gov/gridpoints/TEST/1,2/forecast/hourly?lat={lat}&lon={lon}",
        }
    }


def _game(home: str, kickoff=KICKOFF) -> SlateMatchup:
    return SlateMatchup(
        key=f"KC@{home}",
        away="KC",
        home=home,
        kickoff="11/08/2026 01:00PM ET",
        kickoff_at=kickoff,
    )


def _client(payloads):
    calls = []

    def http_get(url: str):
        calls.append(url)
        if "/points/" in url:
            return payloads["points"]
        if "/forecast/hourly" in url:
            body = payloads["hourly"]
            if isinstance(body, Exception):
                raise body
            return body
        raise AssertionError(url)

    return http_get, calls


@pytest.mark.parametrize(
    ("period", "icon", "summary_parts"),
    [
        (
            _period(),
            "sun",
            ["Sunny", "55°F", "0% rain", "wind 10 mph"],
        ),
        (
            _period(shortForecast="Partly Cloudy", temperature=48, windSpeed="8 mph",
                    probabilityOfPrecipitation={"value": 10}),
            "cloud",
            ["Partly Cloudy", "48°F", "10% rain", "wind 8 mph"],
        ),
        (
            _period(shortForecast="Chance Rain Showers", temperature=55, windSpeed="12 mph",
                    probabilityOfPrecipitation={"value": 40}),
            "rain",
            ["Chance Rain Showers", "55°F", "40% rain", "wind 12 mph"],
        ),
        (
            _period(shortForecast="Snow Showers", temperature=30, windSpeed="15 mph",
                    probabilityOfPrecipitation={"value": 60}),
            "snow",
            ["Snow Showers", "30°F", "60% snow", "wind 15 mph"],
        ),
        (
            _period(shortForecast="Breezy", temperature=50, windSpeed="20 mph",
                    probabilityOfPrecipitation={"value": 0}),
            "wind",
            ["Breezy", "50°F", "0% rain", "wind 20 mph"],
        ),
        (
            _period(shortForecast="Sunny", windSpeed="10 to 25 mph"),
            "wind",
            ["Sunny", "wind 10 to 25 mph"],
        ),
    ],
)
def test_icon_and_summary_from_fixture_period(period, icon, summary_parts):
    kind, summary = describe_hourly_period(period)
    assert kind == icon
    for part in summary_parts:
        assert part in summary


def test_missing_weather_fields_stay_missing():
    kind, summary = describe_hourly_period(
        {
            "startTime": "2026-11-08T13:00:00-05:00",
            "endTime": "2026-11-08T14:00:00-05:00",
            "shortForecast": "Sunny",
            "temperature": None,
            "windSpeed": None,
            "probabilityOfPrecipitation": {"value": None},
        }
    )
    assert kind == "sun"
    assert "temperature missing" in summary
    assert "rain chance missing" in summary
    assert "wind missing" in summary
    assert "0°F" not in summary
    assert "0 mph" not in summary


def test_dome_and_fixed_roof_skip_the_forecast():
    def explode(_url: str):
        raise AssertionError("dome games must not call the forecast")

    for home in ("NO", "DET", "MIN", "LV"):
        forecast = forecasts_for_matchups(
            [_game(home)],
            slate_key="slate",
            cache_path=Path("/tmp/does-not-matter.json"),
            http_get=explode,
            now=KICKOFF,
        )[0]
        assert forecast.status == "dome"
        assert forecast.icon == "dome"
        assert forecast.summary == "Dome"
        assert weather_label(forecast) == "🏟️ Dome"


def test_retractable_roof_keeps_the_forecast_and_is_marked(tmp_path):
    stadium = get_stadium("ATL")
    http_get, calls = _client(
        {
            "points": _points(stadium.lat, stadium.lon),
            "hourly": _hourly(_period(shortForecast="Sunny", temperature=70)),
        }
    )
    forecast = forecasts_for_matchups(
        [_game("ATL")],
        slate_key="slate",
        cache_path=tmp_path / "cache.json",
        http_get=http_get,
        now=KICKOFF,
    )[0]
    assert forecast.status == "forecast"
    assert forecast.icon == "sun"
    assert forecast.retractable is True
    assert "retractable roof" in forecast.summary
    assert "70°F" in forecast.summary
    assert calls[0].startswith("https://api.weather.gov/points/")
    assert "/forecast/hourly" in calls[1]
    assert "☀️" in weather_label(forecast)


def test_kickoff_outside_the_hourly_window_is_unavailable(tmp_path):
    stadium = get_stadium("BUF")
    http_get, _calls = _client(
        {
            "points": _points(stadium.lat, stadium.lon),
            "hourly": _hourly(_period()),
        }
    )
    later = datetime(2026, 11, 20, 13, 0, tzinfo=EASTERN)
    forecast = forecasts_for_matchups(
        [_game("BUF", later)],
        slate_key="slate",
        cache_path=tmp_path / "cache.json",
        http_get=http_get,
        now=KICKOFF,
    )[0]
    assert forecast.status == "unavailable"
    assert forecast.summary == UNAVAILABLE
    assert forecast.reason == "outside_window"
    assert weather_label(forecast) == UNAVAILABLE


@pytest.mark.parametrize(
    "failure",
    [
        URLError("timed out"),
        TimeoutError("timed out"),
        HTTPError("https://api.weather.gov/points/1,2", 500, "err", hdrs=None, fp=None),
        json.JSONDecodeError("bad", "", 0),
        ValueError("bad payload"),
    ],
)
def test_api_failure_is_unavailable_and_does_not_raise(tmp_path, failure):
    def http_get(_url: str):
        raise failure

    forecast = forecasts_for_matchups(
        [_game("BUF")],
        slate_key="slate",
        cache_path=tmp_path / "cache.json",
        http_get=http_get,
        now=KICKOFF,
    )[0]
    assert isinstance(forecast, KickoffForecast)
    assert forecast.status == "unavailable"
    assert forecast.summary == UNAVAILABLE
    assert forecast.reason == "fetch_failed"


def test_bad_json_shape_is_unavailable(tmp_path):
    http_get, _calls = _client({"points": _points(1, 2), "hourly": ["not", "an", "object"]})
    forecast = forecasts_for_matchups(
        [_game("BUF")],
        slate_key="slate",
        cache_path=tmp_path / "cache.json",
        http_get=http_get,
        now=KICKOFF,
    )[0]
    assert forecast.summary == UNAVAILABLE


def test_cache_is_per_slate_and_does_not_call_twice(tmp_path):
    stadium = get_stadium("BUF")
    http_get, calls = _client(
        {
            "points": _points(stadium.lat, stadium.lon),
            "hourly": _hourly(_period()),
        }
    )
    kwargs = {
        "cache_path": tmp_path / "cache.json",
        "http_get": http_get,
        "now": KICKOFF,
    }
    first = forecasts_for_matchups([_game("BUF")], slate_key="slate-a", **kwargs)
    second = forecasts_for_matchups([_game("BUF")], slate_key="slate-a", **kwargs)
    assert first[0].summary == second[0].summary
    assert len(calls) == 2
    forecasts_for_matchups([_game("BUF")], slate_key="slate-b", **kwargs)
    assert len(calls) == 4


def test_request_sends_the_required_user_agent():
    request = nws_request("https://api.weather.gov/points/42.7738,-78.7870")
    agent = request.get_header("User-agent")
    assert agent == NWS_USER_AGENT
    assert "dfs-lineup-generator" in agent
    points = {
        "properties": {
            "gridId": "BUF",
            "gridX": 38,
            "gridY": 55,
        }
    }
    assert hourly_url_from_points(points).endswith("/gridpoints/BUF/38,55/forecast/hourly")
