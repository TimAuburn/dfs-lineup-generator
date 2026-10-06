"""Sleeper projection overlay and injury removal."""

from __future__ import annotations

import json
from datetime import datetime
from urllib.error import URLError

import pandas as pd

from src.parser import drop_inactive_players, is_inactive_injury
from src.sleeper import (
    apply_sleeper_index,
    index_projections,
    load_sleeper_index,
    normalize_player_name,
    overlay_sleeper_projections,
)


def _record(first, last, team, position, points, injury=None):
    return {
        "team": team,
        "player": {
            "first_name": first,
            "last_name": last,
            "position": position,
            "team": team,
            "injury_status": injury,
        },
        "stats": {"pts_ppr": points, "pts_std": points - 1},
    }


def test_normalize_player_name_drops_punctuation_and_suffixes():
    assert normalize_player_name("D.J. Moore") == "dj moore"
    assert normalize_player_name("Amon-Ra St. Brown") == "amon-ra st brown"
    assert normalize_player_name("Kenneth Walker III") == "kenneth walker"


def test_apply_sleeper_replaces_projections_and_drops_inactive():
    records = [
        _record("Josh", "Allen", "BUF", "QB", 24.5),
        _record("Cooper", "Rush", "BUF", "QB", 2.0, "Out"),
        _record("James", "Cook", "BUF", "RB", 14.2, "Questionable"),
        _record("Buffalo", "Bills", "BUF", "DEF", 8.4),
        _record("D.J.", "Moore", "CHI", "WR", 16.0, "Doubtful"),
    ]
    index = index_projections(records)
    pool = pd.DataFrame(
        [
            {
                "ID": "1",
                "Name": "Josh Allen",
                "Position": "QB",
                "TeamAbbrev": "BUF",
                "AvgPointsPerGame": 9.0,
                "Projection": 9.0,
                "Injury": "",
            },
            {
                "ID": "2",
                "Name": "Cooper Rush",
                "Position": "QB",
                "TeamAbbrev": "BUF",
                "AvgPointsPerGame": 6.0,
                "Projection": 6.0,
                "Injury": "",
            },
            {
                "ID": "3",
                "Name": "James Cook",
                "Position": "RB",
                "TeamAbbrev": "BUF",
                "AvgPointsPerGame": 4.0,
                "Projection": 4.0,
                "Injury": "OUT",
            },
            {
                "ID": "4",
                "Name": "Bills",
                "Position": "DST",
                "TeamAbbrev": "BUF",
                "AvgPointsPerGame": 3.0,
                "Projection": 3.0,
                "Injury": "",
            },
            {
                "ID": "5",
                "Name": "DJ Moore",
                "Position": "WR",
                "TeamAbbrev": "CHI",
                "AvgPointsPerGame": 11.0,
                "Projection": 11.0,
                "Injury": "",
            },
            {
                "ID": "6",
                "Name": "Unlisted Rookie",
                "Position": "WR",
                "TeamAbbrev": "BUF",
                "AvgPointsPerGame": 1.5,
                "Projection": 1.5,
                "Injury": "",
            },
            {
                "ID": "10",
                "Name": "Josh Allen",
                "Position": "QB",
                "TeamAbbrev": "BUF",
                "Roster Position": "CPT",
                "AvgPointsPerGame": 9.0,
                "Projection": 9.0,
                "Injury": "",
            },
        ]
    )
    updated, matched = apply_sleeper_index(pool, index)
    by_id = updated.set_index("ID")
    assert matched == 6
    assert by_id.loc["1", "Projection"] == 24.5
    assert by_id.loc["10", "Projection"] == 24.5
    assert by_id.loc["2", "Injury"] == "OUT"
    assert is_inactive_injury(by_id.loc["2", "Injury"])
    # Sleeper says Cook is only questionable, so the DK OUT tag is cleared.
    assert by_id.loc["3", "Injury"] == "QUESTIONABLE"
    assert not is_inactive_injury(by_id.loc["3", "Injury"])
    assert by_id.loc["4", "Projection"] == 8.4
    assert by_id.loc["5", "Injury"] == "DOUBTFUL"
    assert by_id.loc["6", "Projection"] == 1.5

    active = drop_inactive_players(updated)
    assert "2" not in set(active["ID"])
    assert "5" not in set(active["ID"])
    assert "1" in set(active["ID"])
    assert "3" in set(active["ID"])


def test_overlay_uses_cache_and_survives_errors(tmp_path, monkeypatch):
    cache = tmp_path / "sleeper_cache.json"
    calls: list[str] = []

    def _ok(url: str, timeout: float = 30):
        calls.append(url)
        if url.endswith("/state/nfl"):
            return {"season": "2026", "week": 4, "season_type": "regular"}
        if "/projections/nfl/2026/4" in url:
            return [_record("Josh", "Allen", "BUF", "QB", 21.0, "Questionable")]
        raise AssertionError(url)

    monkeypatch.setattr("src.sleeper._http_get", _ok)
    pool = pd.DataFrame(
        [
            {
                "ID": "1",
                "Name": "Josh Allen",
                "Position": "QB",
                "TeamAbbrev": "BUF",
                "Projection": 1.0,
                "Injury": "",
            }
        ]
    )
    now = datetime(2026, 10, 5, 18, 0)
    updated, summary = overlay_sleeper_projections(
        pool, force_refresh=True, cache_path=cache, now=now
    )
    assert updated.loc[0, "Projection"] == 21.0
    assert summary["week"] == 4
    assert summary["matched"] == 1
    assert summary["ok"] is True
    assert len(calls) == 2
    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert "josh allen|BUF" in saved["index"]

    def _boom(url: str, timeout: float = 30):
        raise URLError("down")

    monkeypatch.setattr("src.sleeper._http_get", _boom)
    cached_index, meta = load_sleeper_index(
        force_refresh=True,
        cache_path=cache,
        now=datetime(2026, 10, 5, 20, 0),
    )
    assert "josh allen|BUF" in cached_index
    assert meta["from_cache"] is True
    assert str(meta["error"]).startswith("fetch_failed")
