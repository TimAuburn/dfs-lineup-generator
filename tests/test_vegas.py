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
    projection_adjustment_labels,
    projection_deltas,
)
from src.vegas_cache import (
    cache_ttl_seconds,
    coverage_warnings,
    fetch_vegas_lines,
    format_cache_status,
    implied_team_total,
    nfl_week_window,
    normalize_vegas_memory,
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
    """Favorite is the negative American line: (total - signed spread) / 2.

    The Odds API publishes that sign on each team's spread point (Tampa Bay
    -6.5, Dallas +6.5 in their example). The old formula subtracted the
    absolute spread for a negative number, so total 44.5 and -2.5 came out
    21.0 for the favorite and 23.5 for the underdog.
    """
    assert team_abbreviation("Washington Commanders") == "WAS"
    assert team_abbreviation("Indianapolis Colts") == "IND"
    assert implied_team_total(44.5, -2.5) == pytest.approx(23.5)
    assert implied_team_total(44.5, 2.5) == pytest.approx(21.0)
    assert implied_team_total(44.5, -2.5) + implied_team_total(44.5, 2.5) == pytest.approx(44.5)
    assert implied_team_total(50, -10) == pytest.approx(30)
    assert implied_team_total(50, 10) == pytest.approx(20)

    # Pick'em, +/-2.5, and +/-10. The pair sums to the game total, and a
    # favorite (nonzero margin) is strictly higher.
    for total, margin in ((47.0, 0.0), (45.0, 2.5), (50.0, 10.0)):
        favorite = implied_team_total(total, -margin)
        underdog = implied_team_total(total, margin)
        assert favorite + underdog == pytest.approx(total)
        if margin == 0:
            assert favorite == pytest.approx(total / 2.0)
            assert underdog == pytest.approx(favorite)
        else:
            assert favorite > underdog


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
    assert lines["IND"]["itt"] == 23.5
    assert lines["WAS"]["itt"] == 21.0
    assert lines["IND"]["itt"] + lines["WAS"]["itt"] == pytest.approx(44.5)
    assert lines["WAS"]["game"] == "IND@WAS"


def test_fresh_cache_skips_the_network(tmp_path, monkeypatch):
    cache = tmp_path / "vegas_cache.json"
    cached = {"IND": {"total": 44.5, "spread": -2.5, "itt": 23.5}}
    cache.write_text(
        json.dumps(
            {
                "schema": 2,
                "timestamp": datetime(2026, 10, 5, 12, 0).timestamp(),
                "data": cached,
                "warnings": [],
            }
        ),
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
    cached = {"WAS": {"total": 41.0, "spread": 3.0, "itt": 19.0}}
    cache.write_text(
        json.dumps({"schema": 2, "timestamp": 1, "data": cached, "warnings": []}),
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
    assert lines["IND"]["itt"] == 23.5
    assert lines["WAS"]["itt"] == 21.0
    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert saved["schema"] == 2
    assert saved["warnings"] == []
    assert saved["timestamp"] == now.timestamp()
    assert saved["data"]["IND"]["itt"] == 23.5
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
        # (55.5 - (-3.5)) / 2 = 29.5. Negative spread is the favorite.
        "BUF": {"itt": 29.5, "total": 55.5, "spread": -3.5},
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
        # ITTs match (total - signed spread) / 2: 27, 18, and 22.
        "IND": {"total": 51.0, "spread": -3.0, "itt": 27.0, "game": "IND@WAS"},
        "WAS": {"total": 40.0, "spread": 4.0, "itt": 18.0, "game": "LAR@WAS"},
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
    assert restored["WAS"]["total"] == 40.0


def _skill_pool() -> pd.DataFrame:
    """Identical baselines so only the implied-total adjustment can separate them."""
    return pd.DataFrame(
        [
            {"Name": "KC WR", "TeamAbbrev": "KC", "Position": "WR", "Projection": 12.0},
            {"Name": "BUF WR", "TeamAbbrev": "BUF", "Position": "WR", "Projection": 12.0},
            {"Name": "KC QB", "TeamAbbrev": "KC", "Position": "QB", "Projection": 12.0},
        ]
    )


def _reversed_fifty_point_lines() -> dict:
    """Total 50 with the favorite saved at 20 and the underdog at 30.

    That is what (total / 2) - (|spread| / 2) wrote for spread -10 / +10.
    """
    return {
        "KC": {"total": 50.0, "spread": -10.0, "itt": 20.0, "game": "KC@BUF"},
        "BUF": {"total": 50.0, "spread": 10.0, "itt": 30.0, "game": "KC@BUF"},
    }


def test_parsed_pairs_sum_to_the_total_at_pickem_and_common_spreads():
    payload = [
        _priced("Green Bay Packers", "Detroit Lions", "2026-10-11T17:00:00Z", 0, 47),
        _priced("Washington Commanders", "Indianapolis Colts", "2026-10-11T17:05:00Z", 2.5, 45),
        _priced("Buffalo Bills", "Kansas City Chiefs", "2026-10-11T17:10:00Z", 10, 50),
    ]
    lines = parse_odds_payload(payload)
    assert lines["DET"]["itt"] == pytest.approx(23.5)
    assert lines["GB"]["itt"] == pytest.approx(23.5)
    assert lines["IND"]["spread"] == pytest.approx(-2.5)
    assert lines["IND"]["itt"] == pytest.approx(23.75)
    assert lines["WAS"]["itt"] == pytest.approx(21.25)
    assert lines["KC"]["spread"] == pytest.approx(-10)
    assert lines["KC"]["itt"] == pytest.approx(30)
    assert lines["BUF"]["itt"] == pytest.approx(20)
    assert lines["DET"]["itt"] + lines["GB"]["itt"] == pytest.approx(47)
    assert lines["IND"]["itt"] + lines["WAS"]["itt"] == pytest.approx(45)
    assert lines["KC"]["itt"] + lines["BUF"]["itt"] == pytest.approx(50)
    assert lines["IND"]["itt"] > lines["WAS"]["itt"]
    assert lines["KC"]["itt"] > lines["BUF"]["itt"]


def test_projection_adjustment_uses_the_signed_spread_not_a_reversed_itt():
    adjusted = apply_vegas_projections(_skill_pool(), _reversed_fifty_point_lines())
    by_name = dict(zip(adjusted["Name"], adjusted["Projection"]))
    assert by_name["KC WR"] == pytest.approx(12.6)
    assert by_name["BUF WR"] == pytest.approx(12.0)
    assert by_name["KC QB"] == pytest.approx(12.0)
    assert by_name["KC WR"] > by_name["BUF WR"]
    labels = projection_adjustment_labels(_skill_pool(), _reversed_fifty_point_lines())
    assert labels == ["KC +5%"]


def test_editor_frame_shows_the_corrected_implied_total():
    frame = lines_to_frame(_reversed_fifty_point_lines())
    shown = dict(zip(frame["Team"], frame["ITT"]))
    assert shown["KC"] == pytest.approx(30)
    assert shown["BUF"] == pytest.approx(20)
    restored = frame_to_lines(frame)
    assert restored["KC"]["itt"] == pytest.approx(30)
    assert restored["BUF"]["itt"] == pytest.approx(20)


def test_parsed_odds_favor_the_favorite_skill_player():
    lines = parse_odds_payload(
        [_priced("Buffalo Bills", "Kansas City Chiefs", "2026-10-11T17:00:00Z", 10, 50)]
    )
    assert lines["KC"]["itt"] == 30.0
    assert lines["BUF"]["itt"] == 20.0
    adjusted = apply_vegas_projections(_skill_pool(), lines)
    by_name = dict(zip(adjusted["Name"], adjusted["Projection"]))
    assert by_name["KC WR"] == pytest.approx(12.6)
    assert by_name["BUF WR"] == pytest.approx(12.0)
    assert by_name["KC QB"] == pytest.approx(12.0)
    deltas = projection_deltas(_skill_pool(), lines)
    assert float(deltas.iloc[0]) > 0
    assert float(deltas.iloc[1]) == pytest.approx(0.0)
    assert float(deltas.iloc[2]) == pytest.approx(0.0)


def test_outdated_odds_cache_is_not_reused(tmp_path, monkeypatch):
    """A pre-schema file still holding favorite 20 / underdog 30 is not a source."""
    cache = tmp_path / "vegas_cache.json"
    now = datetime(2026, 10, 5, 12, 30)
    cache.write_text(
        json.dumps(
            {
                "timestamp": datetime(2026, 10, 5, 12, 0).timestamp(),
                "data": _reversed_fifty_point_lines(),
            }
        ),
        encoding="utf-8",
    )

    def _boom(url: str, timeout: float = 20):
        raise TimeoutError(f"network is down: {url}")

    monkeypatch.setattr("src.vegas_cache._http_get", _boom)
    assert fetch_vegas_lines(api_key=None, cache_path=cache, now=now) == {}
    assert fetch_vegas_lines(api_key="secret", cache_path=cache, now=now) == {}
    assert format_cache_status(cache, now=now.timestamp()) == (
        "Vegas odds cache is out of date and was not used."
    )
    schema_one = tmp_path / "schema_one.json"
    schema_one.write_text(
        json.dumps(
            {
                "schema": 1,
                "timestamp": datetime(2026, 10, 5, 12, 0).timestamp(),
                "data": _reversed_fifty_point_lines(),
            }
        ),
        encoding="utf-8",
    )
    assert fetch_vegas_lines(api_key=None, cache_path=schema_one, now=now) == {}


def test_missing_odds_market_is_reported_and_not_filled_in():
    payload = [
        _priced("Buffalo Bills", "Kansas City Chiefs", "2026-10-11T17:00:00Z", 10, 50),
        {
            "home_team": "Dallas Cowboys",
            "away_team": "Philadelphia Eagles",
            "commence_time": "2026-10-11T20:25:00Z",
            "bookmakers": [],
        },
        {
            "home_team": "Atlanta Falcons",
            "away_team": "Tampa Bay Buccaneers",
            "commence_time": "2026-10-11T17:00:00Z",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [{"name": "Atlanta Falcons", "price": -110}],
                        }
                    ],
                }
            ],
        },
    ]
    lines = parse_odds_payload(payload)
    assert set(lines) == {"KC", "BUF"}
    assert lines["KC"]["total"] == 50
    assert "PHI" not in lines and "DAL" not in lines and "TB" not in lines
    warnings = list(getattr(lines, "warnings", []))
    # Parse walks games by kickoff. TB@ATL is 17:00 and PHI@DAL is 20:25.
    assert warnings == [
        "No odds market for TB@ATL. Implied team totals were not applied to this game.",
        "No odds market for PHI@DAL. Implied team totals were not applied to this game.",
    ]


def test_games_outside_the_week_do_not_raise_a_missing_market_warning():
    lines = parse_odds_payload(
        [
            {
                "home_team": "Dallas Cowboys",
                "away_team": "Philadelphia Eagles",
                "commence_time": "2026-10-20T20:25:00Z",
                "bookmakers": [],
            }
        ],
        commence_from=datetime(2026, 10, 7, tzinfo=timezone.utc),
        commence_to=datetime(2026, 10, 14, tzinfo=timezone.utc),
    )
    assert lines == {}
    assert list(getattr(lines, "warnings", [])) == []


def test_incomplete_market_does_not_invent_an_implied_total():
    pool = pd.DataFrame(
        [
            {"Name": "PHI WR", "TeamAbbrev": "PHI", "Position": "WR", "Projection": 12.0},
            {"Name": "DAL WR", "TeamAbbrev": "DAL", "Position": "WR", "Projection": 12.0},
        ]
    )
    lines = {
        "PHI": {"game": "PHI@DAL"},
        "DAL": {"total": 50.0, "game": "PHI@DAL"},
    }
    adjusted = apply_vegas_projections(pool, lines)
    assert list(adjusted["Projection"]) == pytest.approx([12.0, 12.0])
    assert projection_adjustment_labels(pool, lines) == []


def test_missing_market_warning_is_saved_with_the_cache(tmp_path, monkeypatch):
    cache = tmp_path / "vegas_cache.json"

    def _ok(url: str, timeout: float = 20):
        if "sleeper.app" in url:
            return {
                "season": "2026",
                "week": 5,
                "season_type": "regular",
                "season_start_date": "2026-09-09",
            }
        return [
            _priced("Buffalo Bills", "Kansas City Chiefs", "2026-10-11T17:00:00Z", 10, 50),
            {
                "home_team": "Dallas Cowboys",
                "away_team": "Philadelphia Eagles",
                "commence_time": "2026-10-11T20:25:00Z",
                "bookmakers": [],
            },
        ]

    monkeypatch.setattr("src.vegas_cache._http_get", _ok)
    now = datetime(2026, 10, 5, 15, 0)
    lines = fetch_vegas_lines(api_key="secret", force_refresh=True, cache_path=cache, now=now)
    assert set(lines) == {"KC", "BUF"}
    assert lines["KC"]["itt"] == 30.0
    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert saved["warnings"] == [
        "No odds market for PHI@DAL. Implied team totals were not applied to this game."
    ]
    assert coverage_warnings(cache) == saved["warnings"]
    assert "PHI" not in saved["data"]


def test_in_memory_vegas_lines_from_an_older_schema_are_dropped():
    """Streamlit keeps the editor frame across reruns. An older copy is dropped."""
    frame = {"KC": "reversed"}
    updated = normalize_vegas_memory(
        {
            "vegas_schema": None,
            "vegas_frame": frame,
            "vegas_source_json": "{\"KC\": 20}",
            "vegas_editor_version": 4,
        }
    )
    assert updated["vegas_schema"] == 2
    assert updated["vegas_frame"] is None
    assert updated["vegas_source_json"] is None
    assert updated["vegas_editor_version"] == 5

    kept = normalize_vegas_memory(
        {
            "vegas_schema": 2,
            "vegas_frame": frame,
            "vegas_source_json": "{}",
            "vegas_editor_version": 4,
        }
    )
    assert kept["vegas_frame"] is frame
    assert kept["vegas_source_json"] == "{}"
    assert kept["vegas_editor_version"] == 4

    fresh = normalize_vegas_memory({})
    assert fresh == {
        "vegas_schema": 2,
        "vegas_frame": None,
        "vegas_source_json": None,
        "vegas_editor_version": 0,
    }
