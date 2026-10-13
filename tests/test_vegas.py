"""Vegas cache, implied totals, and projection adjustments."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.data_loader import (
    apply_vegas_projections,
    frame_to_lines,
    lines_to_frame,
    apply_projection_delta,
    low_scoring_showdown,
    projection_deltas,
)
from src.vegas_cache import (
    cache_ttl_seconds,
    fetch_vegas_lines,
    format_cache_status,
    implied_team_total,
    nfl_week_window,
    parse_odds_payload,
    team_abbreviation,
)


def test_ttl_is_short_on_sunday_morning_only():
    sunday_morning = datetime(2026, 10, 4, 8, 0)
    sunday_afternoon = datetime(2026, 10, 4, 13, 0)
    monday = datetime(2026, 10, 5, 9, 0)
    assert cache_ttl_seconds(sunday_morning) == 30 * 60
    assert cache_ttl_seconds(sunday_afternoon) == 6 * 60 * 60
    assert cache_ttl_seconds(monday) == 6 * 60 * 60


def test_implied_totals_and_team_abbreviations():
    assert team_abbreviation("Washington Commanders") == "WAS"
    assert team_abbreviation("Indianapolis Colts") == "IND"
    assert implied_team_total(44.5, -2.5) == pytest.approx(21.0)
    assert implied_team_total(44.5, 2.5) == pytest.approx(23.5)


def _event() -> dict:
    return {
        "home_team": "Washington Commanders",
        "away_team": "Indianapolis Colts",
        "commence_time": "2026-10-11T17:00:00Z",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Indianapolis Colts", "point": -2.5},
                            {"name": "Washington Commanders", "point": 2.5},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": 44.5},
                            {"name": "Under", "point": 44.5},
                        ],
                    },
                ],
            }
        ],
    }


def test_parse_odds_payload_maps_itt():
    lines = parse_odds_payload([_event()])
    assert lines["IND"]["total"] == 44.5
    assert lines["IND"]["spread"] == -2.5
    assert lines["IND"]["itt"] == 21.0
    assert lines["WAS"]["itt"] == 23.5
    assert lines["WAS"]["game"] == "IND@WAS"


def test_fresh_cache_skips_the_network(tmp_path, monkeypatch):
    cache = tmp_path / "vegas_cache.json"
    cached = {"IND": {"total": 44.5, "spread": -2.5, "itt": 21.0}}
    cache.write_text(
        json.dumps({"timestamp": datetime(2026, 10, 5, 12, 0).timestamp(), "data": cached}),
        encoding="utf-8",
    )

    def _boom(url: str, timeout: float = 20):
        raise AssertionError(f"network should not be called: {url}")

    monkeypatch.setattr("src.vegas_cache._http_get", _boom)
    loaded = fetch_vegas_lines(
        api_key="secret",
        force_refresh=False,
        cache_path=cache,
        now=datetime(2026, 10, 5, 12, 30),
    )
    assert loaded == cached
    assert "secret" not in cache.read_text(encoding="utf-8")


def test_missing_key_or_failed_request_returns_cache(tmp_path, monkeypatch):
    cache = tmp_path / "vegas_cache.json"
    cached = {"WAS": {"total": 41.0, "spread": 3.0, "itt": 22.0}}
    cache.write_text(
        json.dumps({"timestamp": 1, "data": cached}),
        encoding="utf-8",
    )

    def _boom(url: str, timeout: float = 20):
        raise TimeoutError("down")

    monkeypatch.setattr("src.vegas_cache._http_get", _boom)
    stale_now = datetime(2026, 10, 5, 18, 0)
    assert fetch_vegas_lines(api_key=None, cache_path=cache, now=stale_now) == cached
    assert (
        fetch_vegas_lines(api_key="secret", force_refresh=True, cache_path=cache, now=stale_now)
        == cached
    )
    assert fetch_vegas_lines(api_key=None, cache_path=tmp_path / "missing.json") == {}
    assert format_cache_status(tmp_path / "missing.json") == "No data cached"


def test_successful_fetch_writes_cache(tmp_path, monkeypatch):
    cache = tmp_path / "vegas_cache.json"

    def _ok(url: str, timeout: float = 20):
        if "sleeper.app" in url:
            return {
                "season": "2026",
                "week": 5,
                "season_type": "regular",
                "season_start_date": "2026-09-09",
            }
        assert "apiKey=secret" in url
        assert "markets=totals,spreads" in url
        return [_event()]

    monkeypatch.setattr("src.vegas_cache._http_get", _ok)
    now = datetime(2026, 10, 5, 15, 0)
    lines = fetch_vegas_lines(api_key="secret", force_refresh=True, cache_path=cache, now=now)
    assert lines["WAS"]["total"] == 44.5
    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert saved["timestamp"] == now.timestamp()
    assert saved["data"]["IND"]["itt"] == 21.0
    status = format_cache_status(cache, now=now.timestamp() + 120)
    assert status == "Vegas Data: Cached (Last updated: 2 minutes ago)"


def _priced(home: str, away: str, commence: str, spread: float, total: float) -> dict:
    return {
        "home_team": home,
        "away_team": away,
        "commence_time": commence,
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": away, "point": -spread},
                            {"name": home, "point": spread},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": total},
                            {"name": "Under", "point": total},
                        ],
                    },
                ],
            }
        ],
    }


def test_current_week_keeps_monday_night_and_drops_next_week():
    begin, end = nfl_week_window("2026-09-09", 5)
    assert begin == datetime(2026, 10, 7, tzinfo=timezone.utc)
    assert end == datetime(2026, 10, 14, tzinfo=timezone.utc)
    payload = [
        _priced("Los Angeles Rams", "Buffalo Bills", "2026-10-13T00:15:00Z", -3.5, 54.5),
        _priced("Los Angeles Rams", "Arizona Cardinals", "2026-10-18T20:05:00Z", -12.5, 52.5),
        _priced("Las Vegas Raiders", "Buffalo Bills", "2026-10-18T20:25:00Z", 6.0, 52.5),
        _priced("Cleveland Browns", "Baltimore Ravens", "2026-10-18T17:00:00Z", 4.5, 40.5),
    ]
    lines = parse_odds_payload(payload, commence_from=begin, commence_to=end)
    assert set(lines) == {"BUF", "LAR"}
    assert lines["BUF"]["game"] == "BUF@LAR"
    assert lines["LAR"]["game"] == "BUF@LAR"
    assert lines["LAR"]["spread"] == pytest.approx(-3.5)
    assert "ARI" not in lines


def test_projection_deltas_show_the_vegas_boost_and_wind_cut():
    pool = pd.DataFrame(
        [
            {
                "Name": "Keon Coleman",
                "Position": "WR",
                "TeamAbbrev": "BUF",
                "Projection": 10.52,
                "Game Info": "BUF@LAR 10/12/2026 08:15PM ET",
            },
            {
                "Name": "Josh Allen",
                "Position": "QB",
                "TeamAbbrev": "BUF",
                "Projection": 20.0,
                "Game Info": "BUF@LAR 10/12/2026 08:15PM ET",
            },
            {
                "Name": "Romeo Doubs",
                "Position": "WR",
                "TeamAbbrev": "GB",
                "Projection": 10.0,
                "Game Info": "CHI@GB 10/12/2026 01:00PM ET",
            },
        ]
    )
    lines = {
        "BUF": {"itt": 29.5, "total": 55.5, "spread": 3.5},
        "GB": {"itt": 22.0, "total": 44.0, "spread": 0.0},
    }
    deltas = projection_deltas(pool, lines, {"GB": {"wind_mph": 20}})
    assert deltas.iloc[0] == pytest.approx(0.53)
    assert deltas.iloc[1] == pytest.approx(0.0)
    assert deltas.iloc[2] == pytest.approx(-1.0)

    shown = pool.copy()
    shown["Sleeper"] = [10.52, pd.NA, 10.0]
    adjusted = apply_projection_delta(shown, lines, {"GB": {"wind_mph": 20}})
    assert adjusted.loc[0, "Sleeper"] == pytest.approx(10.52)
    assert adjusted.loc[0, "Projection"] == pytest.approx(11.05)
    assert adjusted.loc[0, "Delta"] == pytest.approx(0.53)
    assert adjusted.loc[1, "Projection"] == pytest.approx(20.0)
    assert pd.isna(adjusted.loc[1, "Sleeper"])
    assert adjusted.loc[2, "Sleeper"] == pytest.approx(10.0)
    assert adjusted.loc[2, "Projection"] == pytest.approx(9.0)


def test_projection_adjustments_and_low_total():
    pool = pd.DataFrame(
        [
            {"TeamAbbrev": "IND", "Position": "WR", "Projection": 10.0},
            {"TeamAbbrev": "IND", "Position": "QB", "Projection": 20.0},
            {"TeamAbbrev": "WAS", "Position": "RB", "Projection": 8.0},
            {"TeamAbbrev": "WAS", "Position": "K", "Projection": 7.0},
            {"TeamAbbrev": "GB", "Position": "TE", "Projection": 6.0},
        ]
    )
    lines = {
        "IND": {"total": 51.0, "spread": -3.0, "itt": 27.0, "game": "IND@WAS"},
        "WAS": {"total": 51.0, "spread": 3.0, "itt": 18.0, "game": "IND@WAS"},
        "GB": {"total": 44.0, "spread": 0.0, "itt": 22.0, "game": "CHI@GB"},
    }
    adjusted = apply_vegas_projections(pool, lines)
    by_pos = dict(zip(adjusted["Position"], adjusted["Projection"]))
    assert by_pos["WR"] == pytest.approx(10.5)
    assert by_pos["QB"] == pytest.approx(20.0)
    assert by_pos["RB"] == pytest.approx(7.6)
    assert by_pos["K"] == pytest.approx(7.0)
    assert by_pos["TE"] == pytest.approx(6.0)

    assert low_scoring_showdown(
        {"IND": {"total": 37.5}, "WAS": {"total": 37.5}},
        {"IND", "WAS"},
    )
    assert not low_scoring_showdown(lines, {"IND", "WAS"})
    assert not low_scoring_showdown({"GB": {"total": 36.0}}, {"IND"})

    frame = lines_to_frame(lines)
    restored = frame_to_lines(frame)
    assert restored["IND"]["itt"] == 27.0
    assert restored["WAS"]["total"] == 51.0
