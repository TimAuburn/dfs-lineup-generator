"""Soft TTL must not outlive the hard cache maximum."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from urllib.error import URLError

import pandas as pd

from src.sleeper import overlay_sleeper_projections


def _record():
    return {
        "team": "BUF",
        "player_id": "10",
        "season": "2025",
        "week": 1,
        "season_type": "regular",
        "date": "2025-09-07",
        "opponent": "KC",
        "updated_at": 1759765502622,
        "player": {
            "first_name": "Josh",
            "last_name": "Allen",
            "position": "QB",
            "team": "BUF",
            "injury_status": None,
            "player_id": "10",
        },
        "stats": {"pts_ppr": 21.0},
    }


def _pool(points=9.0):
    return pd.DataFrame(
        [
            {
                "ID": "1",
                "Name": "Josh Allen",
                "Position": "QB",
                "TeamAbbrev": "BUF",
                "Salary": 7400,
                "Roster Position": "QB",
                "AvgPointsPerGame": points,
                "Projection": points,
                "Injury": "",
                "Game Info": "KC@BUF 09/07/2025 01:00PM ET",
            }
        ]
    )


def _schedule():
    return [
        {
            "game_date": "2025-09-07",
            "season": "2025",
            "week": 1,
            "season_type": "regular",
            "away": "KC",
            "home": "BUF",
        }
    ]


def _seed(tmp_path, monkeypatch, now):
    cache = tmp_path / "cache.json"

    def _ok(url: str, timeout: float = 30):
        if "/projections/" in url:
            return [_record()]
        if url.endswith("/players/nfl"):
            return {}
        raise URLError(url)

    monkeypatch.setattr("src.sleeper._http_get", _ok)
    overlay_sleeper_projections(
        _pool(),
        force_refresh=True,
        cache_path=cache,
        now=now,
        schedule_games=_schedule(),
    )
    return cache


def test_soft_ttl_cannot_accept_a_cache_older_than_the_hard_max(tmp_path, monkeypatch):
    now = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
    cache = _seed(tmp_path, monkeypatch, now)
    payload = json.loads(cache.read_text(encoding="utf-8"))
    payload["retrieved_at"]["projections"] = (now - timedelta(hours=7)).timestamp()
    cache.write_text(json.dumps(payload), encoding="utf-8")

    def _boom(url: str, timeout: float = 30):
        raise URLError("down")

    monkeypatch.setattr("src.sleeper._http_get", _boom)
    updated, summary = overlay_sleeper_projections(
        _pool(9.0),
        cache_path=cache,
        now=now,
        schedule_games=_schedule(),
        ttl_seconds=10 * 60 * 60,
        max_cache_age_seconds=6 * 60 * 60,
    )
    assert summary["applied"] is False
    assert updated.loc[0, "Projection"] == 9.0
    assert summary["from_cache"] is False
    assert summary["retrieved_at"]["projections"] is None


def test_age_between_soft_ttl_and_hard_max_is_used_only_after_refresh_fails(tmp_path, monkeypatch):
    now = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
    cache = _seed(tmp_path, monkeypatch, now)
    calls = {"n": 0}

    def _boom(url: str, timeout: float = 30):
        calls["n"] += 1
        raise URLError("down")

    monkeypatch.setattr("src.sleeper._http_get", _boom)
    updated, summary = overlay_sleeper_projections(
        _pool(9.0),
        cache_path=cache,
        now=now + timedelta(hours=2),
        schedule_games=_schedule(),
        ttl_seconds=60 * 60,
        max_cache_age_seconds=6 * 60 * 60,
    )
    assert calls["n"] > 0
    assert summary["from_cache"] is True
    assert summary["applied"] is True
    assert str(summary["error"]).startswith("fetch_failed")
    assert updated.loc[0, "Projection"] == 21.0
    assert summary["retrieved_at"]["projections"] == now.timestamp()


def test_cache_age_equal_to_the_hard_max_is_still_usable(tmp_path, monkeypatch):
    now = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
    cache = _seed(tmp_path, monkeypatch, now)
    payload = json.loads(cache.read_text(encoding="utf-8"))
    payload["retrieved_at"]["projections"] = (now - timedelta(hours=6)).timestamp()
    cache.write_text(json.dumps(payload), encoding="utf-8")

    def _boom(url: str, timeout: float = 30):
        raise URLError("down")

    monkeypatch.setattr("src.sleeper._http_get", _boom)
    updated, summary = overlay_sleeper_projections(
        _pool(9.0),
        force_refresh=True,
        cache_path=cache,
        now=now,
        schedule_games=_schedule(),
        ttl_seconds=10 * 60 * 60,
        max_cache_age_seconds=6 * 60 * 60,
    )
    assert summary["applied"] is True
    assert summary["from_cache"] is True
    assert updated.loc[0, "Projection"] == 21.0


def test_future_or_missing_cache_timestamps_are_not_freshness(tmp_path, monkeypatch):
    now = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
    cache = _seed(tmp_path, monkeypatch, now)
    payload = json.loads(cache.read_text(encoding="utf-8"))
    payload["retrieved_at"]["projections"] = (now + timedelta(hours=2)).timestamp()
    cache.write_text(json.dumps(payload), encoding="utf-8")

    def _boom(url: str, timeout: float = 30):
        raise URLError("down")

    monkeypatch.setattr("src.sleeper._http_get", _boom)
    updated, summary = overlay_sleeper_projections(
        _pool(9.0),
        force_refresh=True,
        cache_path=cache,
        now=now,
        schedule_games=_schedule(),
    )
    assert summary["applied"] is False
    assert updated.loc[0, "Projection"] == 9.0
    assert summary["retrieved_at"]["projections"] is None

    payload["retrieved_at"].pop("projections")
    cache.write_text(json.dumps(payload), encoding="utf-8")
    updated, summary = overlay_sleeper_projections(
        _pool(4.0),
        force_refresh=True,
        cache_path=cache,
        now=now,
        schedule_games=_schedule(),
    )
    assert summary["applied"] is False
    assert updated.loc[0, "Projection"] == 4.0


def test_cache_needs_both_scoring_and_match_schema(tmp_path, monkeypatch):
    now = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
    cache = _seed(tmp_path, monkeypatch, now)
    payload = json.loads(cache.read_text(encoding="utf-8"))
    payload.pop("scoring")
    cache.write_text(json.dumps(payload), encoding="utf-8")

    def _boom(url: str, timeout: float = 30):
        raise URLError("down")

    monkeypatch.setattr("src.sleeper._http_get", _boom)
    updated, summary = overlay_sleeper_projections(
        _pool(9.0),
        force_refresh=True,
        cache_path=cache,
        now=now,
        schedule_games=_schedule(),
    )
    assert summary["applied"] is False
    assert updated.loc[0, "Projection"] == 9.0
    assert "dk-component-v2" not in json.dumps(summary.get("retrieved_at"))
