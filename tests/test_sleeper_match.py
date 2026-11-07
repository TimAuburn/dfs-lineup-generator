"""Slate week, cache age, coverage, and player-matching contract."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from urllib.error import URLError

import pandas as pd

from src.parser import drop_inactive_players, is_inactive_injury
from src.slate_week import (
    parse_espn_scoreboard,
    resolve_slate_week,
    slate_games_from_pool,
    validate_projection_records,
)
from src.sleeper import (
    DEFAULT_TTL_SECONDS,
    MAX_CACHE_AGE_SECONDS,
    apply_sleeper_index,
    index_depth_chart,
    index_projections,
    load_sleeper_index,
    match_sleeper_pool,
    merge_sleeper_indexes,
    normalize_player_name,
    overlay_sleeper_projections,
    sleeper_status_caption,
)


def _record(first, last, team, position, points, injury=None, player_id=None):
    row = {
        "team": team,
        "player_id": player_id,
        "player": {
            "first_name": first,
            "last_name": last,
            "position": position,
            "team": team,
            "injury_status": injury,
            "player_id": player_id,
        },
        "stats": {"pts_ppr": points},
    }
    return row


def _pool_row(**overrides):
    row = {
        "ID": "1",
        "Name": "Josh Allen",
        "Position": "QB",
        "TeamAbbrev": "BUF",
        "Salary": 7400,
        "Roster Position": "QB",
        "AvgPointsPerGame": 9.0,
        "Projection": 9.0,
        "Injury": "",
        "Game Info": "KC@BUF 09/07/2025 01:00PM ET",
    }
    row.update(overrides)
    return row


def test_status_suffix_name_normalizes_to_the_player():
    assert normalize_player_name("D.J. Moore (Q)") == "dj moore"
    assert normalize_player_name("Travis Etienne Jr. (Q)") == "travis etienne"
    assert normalize_player_name("Kenneth Walker III") == "kenneth walker"


def test_blank_sleeper_status_does_not_clear_draftkings_out():
    index = index_projections([_record("James", "Cook", "BUF", "RB", 14.2, None)])
    pool = pd.DataFrame([_pool_row(ID="3", Name="James Cook", Position="RB", Injury="OUT", Projection=4.0)])
    updated, _matched = apply_sleeper_index(pool, index)
    assert updated.loc[0, "Injury"] == "OUT"
    assert is_inactive_injury(updated.loc[0, "Injury"])


def test_questionable_sleeper_status_does_not_clear_draftkings_out():
    index = index_projections([_record("James", "Cook", "BUF", "RB", 14.2, "Questionable")])
    pool = pd.DataFrame([_pool_row(ID="3", Name="James Cook", Position="RB", Injury="OUT", Projection=4.0)])
    updated, _matched = apply_sleeper_index(pool, index)
    assert updated.loc[0, "Injury"] == "OUT"
    assert is_inactive_injury(updated.loc[0, "Injury"])


def test_duplicate_name_does_not_keep_the_highest_projection():
    index = index_projections(
        [
            _record("Josh", "Johnson", "WAS", "QB", 3.0, player_id="1"),
            _record("Josh", "Johnson", "WAS", "QB", 18.0, player_id="2"),
        ]
    )
    pool = pd.DataFrame(
        [_pool_row(Name="Josh Johnson", TeamAbbrev="WAS", Projection=5.0, AvgPointsPerGame=5.0)]
    )
    updated, matched = apply_sleeper_index(pool, index)
    assert matched == 0
    assert updated.loc[0, "Projection"] == 5.0
    assert pd.isna(updated.loc[0, "Sleeper"])


def test_position_mismatch_is_not_a_projection_match():
    index = index_projections([_record("Michael", "Carter", "ARI", "RB", 20.0, player_id="9")])
    pool = pd.DataFrame(
        [
            _pool_row(
                Name="Michael Carter",
                Position="WR",
                TeamAbbrev="ARI",
                Projection=4.0,
                AvgPointsPerGame=4.0,
            )
        ]
    )
    updated, matched = apply_sleeper_index(pool, index)
    assert matched == 0
    assert updated.loc[0, "Projection"] == 4.0


def test_depth_only_hit_is_not_a_projection_match():
    projections = index_projections([])
    depth = index_depth_chart(
        {
            "1": {
                "first_name": "Drew",
                "last_name": "Lock",
                "position": "QB",
                "team": "SEA",
                "depth_chart_position": "QB",
                "depth_chart_order": 2,
                "injury_status": None,
            }
        }
    )
    merged = merge_sleeper_indexes(projections, depth)
    pool = pd.DataFrame(
        [
            _pool_row(
                Name="Drew Lock",
                Position="QB",
                TeamAbbrev="SEA",
                Projection=17.1,
                AvgPointsPerGame=17.1,
            )
        ]
    )
    updated, matched = apply_sleeper_index(pool, merged)
    assert matched == 0
    assert updated.loc[0, "Projection"] == 17.1
    assert updated.loc[0, "AvgPointsPerGame"] == 17.1
    assert pd.isna(updated.loc[0, "Sleeper"])


def test_overlay_uses_slate_dates_not_the_current_week(tmp_path, monkeypatch):
    calls: list[str] = []

    def _http(url: str, timeout: float = 30):
        calls.append(url)
        if url.endswith("/state/nfl"):
            return {"season": "2026", "week": 10, "season_type": "regular"}
        if "/projections/" in url:
            return [_record("Josh", "Allen", "BUF", "QB", 30.0)]
        if url.endswith("/players/nfl"):
            return {}
        raise URLError(url)

    monkeypatch.setattr("src.sleeper._http_get", _http)
    pool = pd.DataFrame([_pool_row(Projection=9.0)])
    updated, summary = overlay_sleeper_projections(
        pool,
        force_refresh=True,
        cache_path=tmp_path / "sleeper_cache.json",
        now=datetime(2026, 11, 7, 18, 0),
    )
    assert not any("/state/nfl" in url for url in calls)
    assert summary.get("week") != 10
    assert updated.loc[0, "Projection"] == 9.0


def _week_record(first, last, team, position, points, injury=None, player_id="10", opponent="KC"):
    row = _record(first, last, team, position, points, injury, player_id=player_id)
    row.update(
        {
            "season": "2025",
            "week": 1,
            "season_type": "regular",
            "date": "2025-09-07",
            "opponent": opponent,
            "updated_at": 1759765502622,
        }
    )
    return row


def _schedule(away="KC", home="BUF", game_date="2025-09-07", season="2025", week=1, season_type="regular"):
    return [
        {
            "game_date": game_date,
            "season": season,
            "week": week,
            "season_type": season_type,
            "away": away,
            "home": home,
        }
    ]


def test_expired_cache_is_not_reused_when_the_fetch_fails(tmp_path, monkeypatch):
    cache = tmp_path / "sleeper_cache.json"
    now = datetime(2026, 11, 7, 18, 0, tzinfo=timezone.utc)
    schedule = _schedule()

    def _ok(url: str, timeout: float = 30):
        if "/projections/" in url:
            return [_week_record("Josh", "Allen", "BUF", "QB", 21.0)]
        if url.endswith("/players/nfl"):
            return {}
        raise URLError(url)

    monkeypatch.setattr("src.sleeper._http_get", _ok)
    overlay_sleeper_projections(
        pd.DataFrame([_pool_row()]),
        force_refresh=True,
        cache_path=cache,
        now=now,
        schedule_games=schedule,
    )
    payload = json.loads(cache.read_text(encoding="utf-8"))
    old = (now - timedelta(hours=10)).timestamp()
    payload["retrieved_at"]["projections"] = old
    cache.write_text(json.dumps(payload), encoding="utf-8")

    def _boom(url: str, timeout: float = 30):
        raise URLError("down")

    monkeypatch.setattr("src.sleeper._http_get", _boom)
    updated, summary = overlay_sleeper_projections(
        pd.DataFrame([_pool_row(Projection=9.0)]),
        force_refresh=True,
        cache_path=cache,
        now=now,
        schedule_games=schedule,
    )
    assert summary["from_cache"] is False
    assert summary["error"] == "stale_cache"
    assert summary["applied"] is False
    assert updated.loc[0, "Projection"] == 9.0
    assert summary["retrieved_at"]["projections"] is None
    caption = sleeper_status_caption(summary)
    assert "was not used" in caption
    assert "Injury update time: unavailable" in caption
    assert "6 hours" in caption


def test_cache_limits_are_explicit_and_configurable():
    assert DEFAULT_TTL_SECONDS == 60 * 60
    assert MAX_CACHE_AGE_SECONDS == 6 * 60 * 60
    assert MAX_CACHE_AGE_SECONDS > DEFAULT_TTL_SECONDS


def test_source_timestamps_stay_distinct_from_injury_update_time(tmp_path, monkeypatch):
    now = datetime(2026, 11, 7, 18, 0, tzinfo=timezone.utc)
    calls: list[str] = []

    def _ok(url: str, timeout: float = 30):
        calls.append(url)
        if "/state/nfl" in url:
            raise AssertionError("current week")
        if "site.api.espn.com" in url and "dates=20250907" in url:
            return {
                "week": {"number": 1},
                "events": [
                    {
                        "date": "2025-09-07T17:00Z",
                        "season": {"year": 2025, "type": 2},
                        "week": {"number": 1},
                        "competitions": [
                            {
                                "date": "2025-09-07T17:00Z",
                                "competitors": [
                                    {"homeAway": "away", "team": {"abbreviation": "KC"}},
                                    {"homeAway": "home", "team": {"abbreviation": "BUF"}},
                                ],
                            }
                        ],
                    }
                ],
            }
        if "/projections/nfl/2025/1" in url:
            return [_week_record("Josh", "Allen", "BUF", "QB", 22.0, player_id="4984")]
        if url.endswith("/players/nfl"):
            return {}
        raise AssertionError(url)

    monkeypatch.setattr("src.sleeper._http_get", _ok)
    updated, summary = overlay_sleeper_projections(
        pd.DataFrame([_pool_row()]),
        force_refresh=True,
        cache_path=tmp_path / "cache.json",
        now=now,
    )
    assert updated.loc[0, "Projection"] == 22.0
    assert summary["dates_confirmed"] is True
    assert summary["season"] == "2025"
    assert summary["week"] == 1
    retrieved = summary["retrieved_at"]
    assert retrieved["schedule"] == now.timestamp()
    assert retrieved["projections"] == now.timestamp()
    assert retrieved["players"] == now.timestamp()
    assert summary["injury_updated_at"] is None
    assert summary["projections_feed_updated_at"] == 1759765502622
    assert summary["projections_feed_updated_at"] != retrieved["projections"]
    caption = sleeper_status_caption(summary)
    assert "Schedule retrieved 2026-11-07 18:00:00 UTC" in caption
    assert "Projections retrieved 2026-11-07 18:00:00 UTC" in caption
    assert "Injury update time: unavailable" in caption
    assert "Projection coverage: 1 of 1 players (1 of 1 rows)" in caption
    assert not any("/state/nfl" in url for url in calls)


def test_wrong_week_cache_is_not_applied(tmp_path, monkeypatch):
    cache = tmp_path / "cache.json"
    now = datetime(2026, 11, 7, 18, 0, tzinfo=timezone.utc)

    def _ok(url: str, timeout: float = 30):
        if "/projections/" in url:
            return [_week_record("Josh", "Allen", "BUF", "QB", 21.0)]
        if url.endswith("/players/nfl"):
            return {}
        raise URLError(url)

    monkeypatch.setattr("src.sleeper._http_get", _ok)
    overlay_sleeper_projections(
        pd.DataFrame([_pool_row()]),
        force_refresh=True,
        cache_path=cache,
        now=now,
        schedule_games=_schedule(),
    )

    def _boom(url: str, timeout: float = 30):
        raise URLError("down")

    monkeypatch.setattr("src.sleeper._http_get", _boom)
    later = pd.DataFrame(
        [
            _pool_row(
                **{
                    "Game Info": "DAL@PHI 09/14/2025 01:00PM ET",
                    "Projection": 8.0,
                    "TeamAbbrev": "PHI",
                    "Name": "Jalen Hurts",
                    "Position": "QB",
                }
            )
        ]
    )
    updated, summary = overlay_sleeper_projections(
        later,
        force_refresh=True,
        cache_path=cache,
        now=now + timedelta(minutes=30),
        schedule_games=_schedule(away="DAL", home="PHI", game_date="2025-09-14", week=2),
    )
    assert updated.loc[0, "Projection"] == 8.0
    assert summary["applied"] is False
    assert summary["error"] == "wrong_week_cache"
    assert summary["from_cache"] is False
    assert "different slate" in summary["message"]


def test_same_slate_cache_within_max_age_is_labeled(tmp_path, monkeypatch):
    cache = tmp_path / "cache.json"
    now = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)

    def _ok(url: str, timeout: float = 30):
        if "/projections/" in url:
            return [_week_record("Josh", "Allen", "BUF", "QB", 21.0)]
        if url.endswith("/players/nfl"):
            return {}
        raise URLError(url)

    monkeypatch.setattr("src.sleeper._http_get", _ok)
    overlay_sleeper_projections(
        pd.DataFrame([_pool_row()]),
        force_refresh=True,
        cache_path=cache,
        now=now,
        schedule_games=_schedule(),
    )

    def _boom(url: str, timeout: float = 30):
        raise URLError("down")

    monkeypatch.setattr("src.sleeper._http_get", _boom)
    updated, summary = overlay_sleeper_projections(
        pd.DataFrame([_pool_row(Projection=9.0)]),
        force_refresh=True,
        cache_path=cache,
        now=now + timedelta(hours=2),
        schedule_games=_schedule(),
    )
    assert updated.loc[0, "Projection"] == 21.0
    assert summary["from_cache"] is True
    assert str(summary["error"]).startswith("fetch_failed")
    assert summary["retrieved_at"]["projections"] == now.timestamp()
    caption = sleeper_status_caption(summary)
    assert "2026-11-07 12:00:00 UTC" in caption
    assert "saved copy" in caption.lower() or "Saved copy" in caption


def test_custom_max_age_rejects_a_younger_copy(tmp_path, monkeypatch):
    cache = tmp_path / "cache.json"
    now = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)

    def _ok(url: str, timeout: float = 30):
        if "/projections/" in url:
            return [_week_record("Josh", "Allen", "BUF", "QB", 21.0)]
        raise URLError(url)

    monkeypatch.setattr("src.sleeper._http_get", _ok)
    overlay_sleeper_projections(
        pd.DataFrame([_pool_row()]),
        force_refresh=True,
        cache_path=cache,
        now=now,
        schedule_games=_schedule(),
    )

    def _boom(url: str, timeout: float = 30):
        raise URLError("down")

    monkeypatch.setattr("src.sleeper._http_get", _boom)
    updated, summary = overlay_sleeper_projections(
        pd.DataFrame([_pool_row(Projection=4.0)]),
        force_refresh=True,
        cache_path=cache,
        now=now + timedelta(seconds=90),
        schedule_games=_schedule(),
        max_cache_age_seconds=60,
    )
    assert updated.loc[0, "Projection"] == 4.0
    assert summary["error"] == "stale_cache"
    assert "60 seconds" in sleeper_status_caption(summary)


def test_blank_and_conflicting_injuries_keep_the_more_severe_status():
    out_blank = index_projections([_record("James", "Cook", "BUF", "RB", 14.0, None, player_id="3")])
    pool = pd.DataFrame([_pool_row(Name="James Cook", Position="RB", Injury="", Projection=4.0)])
    sleeper_out = index_projections([_record("James", "Cook", "BUF", "RB", 14.0, "Out", player_id="3")])
    updated, _matched = apply_sleeper_index(pool, sleeper_out)
    assert updated.loc[0, "Injury"] == "OUT"
    assert "Sleeper" in updated.loc[0, "InjuryNote"]

    dk_out = pd.DataFrame([_pool_row(Name="James Cook", Position="RB", Injury="OUT", Projection=4.0)])
    updated, _matched = apply_sleeper_index(dk_out, out_blank)
    assert updated.loc[0, "Injury"] == "OUT"
    assert "DraftKings" in updated.loc[0, "InjuryNote"]

    questionable = index_projections(
        [_record("James", "Cook", "BUF", "RB", 14.0, "Questionable", player_id="3")]
    )
    dk_q = pd.DataFrame([_pool_row(Name="James Cook", Position="RB", Injury="", Projection=4.0)])
    updated, _matched = apply_sleeper_index(dk_q, questionable)
    assert updated.loc[0, "Injury"] == "QUESTIONABLE"
    assert updated.loc[0, "PreLock"] == "Q"
    assert not is_inactive_injury(updated.loc[0, "Injury"])
    assert len(drop_inactive_players(updated)) == 1

    conflict = pd.DataFrame([_pool_row(Name="James Cook", Position="RB", Injury="Q", Projection=4.0)])
    harder = index_projections([_record("James", "Cook", "BUF", "RB", 14.0, "Out", player_id="3")])
    updated, _matched = apply_sleeper_index(conflict, harder)
    assert updated.loc[0, "Injury"] == "OUT"
    assert updated.loc[0, "PreLock"] == ""
    assert "Conflict" in updated.loc[0, "InjuryNote"]


def test_status_suffix_matches_and_stays_questionable():
    index = index_projections([_record("D.J.", "Moore", "CHI", "WR", 16.0, None, player_id="5")])
    pool = pd.DataFrame(
        [
            _pool_row(
                Name="DJ Moore (Q)",
                Position="WR",
                TeamAbbrev="CHI",
                Projection=11.0,
                AvgPointsPerGame=11.0,
                Injury="",
            )
        ]
    )
    updated, matched = apply_sleeper_index(pool, index)
    assert matched == 1
    assert updated.loc[0, "Projection"] == 16.0
    assert updated.loc[0, "Injury"] == "QUESTIONABLE"
    assert updated.loc[0, "PreLock"] == "Q"
    assert updated.loc[0, "Name"] == "DJ Moore (Q)"


def test_dst_team_alias_and_nickname():
    index = index_projections(
        [_record("Washington", "Commanders", "WAS", "DEF", 6.5, None, player_id="WAS")]
    )
    pool = pd.DataFrame(
        [
            _pool_row(
                Name="Commanders",
                Position="DST",
                TeamAbbrev="WSH",
                Salary=3000,
                **{"Roster Position": "DST"},
                Projection=3.0,
                AvgPointsPerGame=3.0,
            )
        ]
    )
    updated, matched = apply_sleeper_index(pool, index)
    assert matched == 1
    assert updated.loc[0, "Projection"] == 6.5
    assert updated.loc[0, "Salary"] == 3000
    assert updated.loc[0, "Roster Position"] == "DST"


def test_duplicate_names_across_teams_stay_on_their_own_team():
    index = index_projections(
        [
            _record("Josh", "Johnson", "WAS", "QB", 3.0, player_id="1"),
            _record("Josh", "Johnson", "BAL", "QB", 18.0, player_id="2"),
        ]
    )
    pool = pd.DataFrame(
        [
            _pool_row(ID="1", Name="Josh Johnson", TeamAbbrev="WAS", Projection=5.0),
            _pool_row(ID="2", Name="Josh Johnson", TeamAbbrev="BAL", Projection=5.0),
        ]
    )
    updated, matched = apply_sleeper_index(pool, index)
    by_id = updated.set_index("ID")
    assert matched == 2
    assert by_id.loc["1", "Projection"] == 3.0
    assert by_id.loc["2", "Projection"] == 18.0


def test_same_team_different_positions_do_not_share_the_higher_score():
    index = index_projections(
        [
            _record("Michael", "Carter", "ARI", "RB", 20.0, player_id="rb"),
            _record("Michael", "Carter", "ARI", "WR", 4.5, player_id="wr"),
        ]
    )
    pool = pd.DataFrame(
        [
            _pool_row(ID="rb", Name="Michael Carter", Position="RB", TeamAbbrev="ARI", Projection=1.0),
            _pool_row(ID="wr", Name="Michael Carter", Position="WR", TeamAbbrev="ARI", Projection=1.0),
        ]
    )
    updated, _matched = apply_sleeper_index(pool, index)
    by_id = updated.set_index("ID")
    assert by_id.loc["rb", "Projection"] == 20.0
    assert by_id.loc["wr", "Projection"] == 4.5


def test_ambiguous_duplicate_is_reported_and_not_scored():
    index = index_projections(
        [
            _record("Josh", "Johnson", "WAS", "QB", 3.0, player_id="1"),
            _record("Josh", "Johnson", "WAS", "QB", 18.0, player_id="2"),
        ]
    )
    pool = pd.DataFrame(
        [_pool_row(Name="Josh Johnson", TeamAbbrev="WAS", Projection=5.0, AvgPointsPerGame=5.0)]
    )
    updated, report = match_sleeper_pool(pool, index)
    assert updated.loc[0, "Projection"] == 5.0
    assert report["projection_rows"] == 0
    assert len(report["ambiguous"]) == 1
    assert "18" in report["ambiguous"][0]["reason"]
    assert "not" in report["ambiguous"][0]["reason"].lower() or "Neither" in report["ambiguous"][0]["reason"]


def test_sleeper_id_must_agree_on_team_and_position():
    index = index_projections(
        [
            _record("Josh", "Allen", "BUF", "QB", 24.0, player_id="4984"),
            _record("Josh", "Allen", "NYJ", "QB", 9.0, player_id="9999"),
        ]
    )
    matched = pd.DataFrame([_pool_row(SleeperID="4984", Projection=1.0)])
    updated, count = apply_sleeper_index(matched, index)
    assert count == 1
    assert updated.loc[0, "Projection"] == 24.0

    conflict = pd.DataFrame([_pool_row(SleeperID="9999", TeamAbbrev="BUF", Projection=1.0)])
    updated, count = apply_sleeper_index(conflict, index)
    assert count == 0
    assert updated.loc[0, "Projection"] == 1.0
    assert updated.loc[0, "MatchStatus"] == "ambiguous"

    missing = pd.DataFrame([_pool_row(SleeperID="nope", Projection=1.0)])
    updated, count = apply_sleeper_index(missing, index)
    assert count == 0
    assert updated.loc[0, "Projection"] == 1.0
    assert "not on the week feed" in updated.loc[0, "InjuryNote"] or updated.loc[0, "MatchStatus"] == "unmatched"


def test_showdown_captain_and_flex_share_identity_without_double_scoring():
    index = index_projections([_record("Patrick", "Mahomes", "KC", "QB", 22.5, "Questionable", player_id="4046")])
    pool = pd.DataFrame(
        [
            _pool_row(
                ID="91001",
                Name="Patrick Mahomes",
                Position="QB",
                TeamAbbrev="KC",
                Salary=16800,
                **{"Roster Position": "CPT"},
                Projection=10.0,
                AvgPointsPerGame=10.0,
            ),
            _pool_row(
                ID="91002",
                Name="Patrick Mahomes",
                Position="QB",
                TeamAbbrev="KC",
                Salary=11200,
                **{"Roster Position": "FLEX"},
                Projection=10.0,
                AvgPointsPerGame=10.0,
            ),
        ]
    )
    updated, report = match_sleeper_pool(pool, index)
    assert list(updated["Projection"]) == [22.5, 22.5]
    assert float(updated.loc[0, "Projection"]) != 45.0
    assert list(updated["Salary"]) == [16800, 11200]
    assert list(updated["Roster Position"]) == ["CPT", "FLEX"]
    assert report["projection_rows"] == 2
    assert report["projection_players"] == 1
    assert list(updated["PreLock"]) == ["Q", "Q"]
    assert not is_inactive_injury("QUESTIONABLE")
    active = drop_inactive_players(updated)
    assert set(active["Roster Position"]) == {"CPT", "FLEX"}


def test_showdown_out_removes_both_rows_and_keeps_salaries_on_the_source_frame():
    index = index_projections([_record("James", "Cook", "BUF", "RB", 12.0, "Out", player_id="20")])
    pool = pd.DataFrame(
        [
            _pool_row(
                ID="1",
                Name="James Cook",
                Position="RB",
                TeamAbbrev="BUF",
                Salary=12600,
                **{"Roster Position": "CPT"},
                Injury="",
            ),
            _pool_row(
                ID="2",
                Name="James Cook",
                Position="RB",
                TeamAbbrev="BUF",
                Salary=8400,
                **{"Roster Position": "FLEX"},
                Injury="",
            ),
        ]
    )
    updated, _report = match_sleeper_pool(pool, index)
    assert list(updated["Injury"]) == ["OUT", "OUT"]
    assert list(updated["Salary"]) == [12600, 8400]
    assert list(updated["Projection"]) == [12.0, 12.0]
    active = drop_inactive_players(updated)
    assert active.empty


def test_unmatched_players_are_listed_separately_from_projection_coverage():
    index = index_projections([_record("Josh", "Allen", "BUF", "QB", 20.0, player_id="1")])
    depth = index_depth_chart(
        {
            "9": {
                "player_id": "9",
                "first_name": "Drew",
                "last_name": "Lock",
                "position": "QB",
                "team": "SEA",
                "depth_chart_position": "QB",
                "depth_chart_order": 2,
                "injury_status": "Questionable",
            }
        }
    )
    merged = merge_sleeper_indexes(index, depth)
    pool = pd.DataFrame(
        [
            _pool_row(ID="1", Name="Josh Allen", Projection=1.0),
            _pool_row(ID="2", Name="Drew Lock", Position="QB", TeamAbbrev="SEA", Projection=17.0, AvgPointsPerGame=17.0),
            _pool_row(ID="3", Name="Unlisted Rookie", Position="WR", TeamAbbrev="BUF", Projection=1.5),
        ]
    )
    updated, report = match_sleeper_pool(pool, merged)
    by_id = updated.set_index("ID")
    assert by_id.loc["1", "Projection"] == 20.0
    assert by_id.loc["2", "Projection"] == 17.0
    assert pd.isna(by_id.loc["2", "Sleeper"])
    assert by_id.loc["2", "MatchStatus"] == "depth_only"
    assert by_id.loc["2", "Injury"] == "QUESTIONABLE"
    assert by_id.loc["2", "PreLock"] == "Q"
    assert by_id.loc["3", "Projection"] == 1.5
    assert report["projection_players"] == 1
    assert report["injury_players"] == 2
    assert report["depth_only_players"] == 1
    assert [item["name"] for item in report["unmatched"]] == ["Unlisted Rookie"]


def test_schedule_resolution_handles_postseason_year_boundary_and_bad_dates():
    slate = slate_games_from_pool(
        pd.DataFrame({"Game Info": ["NO@ATL 01/04/2026 01:00PM ET", "NO@ATL 01/04/2026 01:00PM ET"]})
    )
    scheduled = coerce_one("2026-01-04", "2025", 18, "regular", "NO", "ATL")
    resolved = resolve_slate_week(slate, scheduled)
    assert resolved["ok"] is True
    assert resolved["season"] == "2025"
    assert resolved["week"] == 18
    assert resolved["season_type"] == "regular"

    post = slate_games_from_pool(pd.DataFrame({"Game Info": ["BUF@JAX 01/11/2026 01:00PM ET"]}))
    post_schedule = coerce_one("2026-01-11", "2025", 1, "post", "BUF", "JAX")
    resolved = resolve_slate_week(post, post_schedule)
    assert resolved["season"] == "2025"
    assert resolved["week"] == 1
    assert resolved["season_type"] == "post"

    multi = slate_games_from_pool(
        pd.DataFrame(
            {
                "Game Info": [
                    "KC@BUF 09/07/2025 01:00PM ET",
                    "DAL@PHI 09/15/2025 01:00PM ET",
                ]
            }
        )
    )
    both = coerce_one("2025-09-07", "2025", 1, "regular", "KC", "BUF") + coerce_one(
        "2025-09-15", "2025", 2, "regular", "DAL", "PHI"
    )
    resolved = resolve_slate_week(multi, both)
    assert resolved["ok"] is False
    assert resolved["error"] == "ambiguous_slate_week"

    same_week = slate_games_from_pool(
        pd.DataFrame(
            {
                "Game Info": [
                    "PHI@DAL 09/04/2025 08:20PM ET",
                    "KC@BUF 09/07/2025 01:00PM ET",
                ]
            }
        )
    )
    week_one = coerce_one("2025-09-04", "2025", 1, "regular", "PHI", "DAL") + coerce_one(
        "2025-09-07", "2025", 1, "regular", "KC", "BUF"
    )
    resolved = resolve_slate_week(same_week, week_one)
    assert resolved["ok"] is True
    assert resolved["week"] == 1

    missing = slate_games_from_pool(pd.DataFrame({"Game Info": ["KC@BUF 01:00PM ET"]}))
    resolved = resolve_slate_week(missing, week_one)
    assert resolved["error"] == "missing_game_dates"

    overlap = slate_games_from_pool(pd.DataFrame({"Game Info": ["KC@BUF 09/07/2025 01:00PM ET"]}))
    doubled = coerce_one("2025-09-07", "2025", 1, "regular", "KC", "BUF") + coerce_one(
        "2025-09-07", "2025", 2, "regular", "KC", "BUF"
    )
    resolved = resolve_slate_week(overlap, doubled)
    assert resolved["error"] == "ambiguous_game_date"


def test_monday_night_kickoff_uses_the_eastern_calendar_date():
    games = parse_espn_scoreboard(
        {
            "week": {"number": 1},
            "events": [
                {
                    "date": "2025-09-09T00:20Z",
                    "season": {"year": 2025, "type": 2},
                    "week": {"number": 1},
                    "competitions": [
                        {
                            "date": "2025-09-09T00:20Z",
                            "competitors": [
                                {"homeAway": "home", "team": {"abbreviation": "WSH"}},
                                {"homeAway": "away", "team": {"abbreviation": "TB"}},
                            ],
                        }
                    ],
                }
            ],
        }
    )
    assert len(games) == 1
    assert games[0].game_date.isoformat() == "2025-09-08"
    assert games[0].home == "WAS"
    assert games[0].season == "2025"
    assert games[0].week == 1


def test_projection_payload_must_match_the_slate_week():
    slate = slate_games_from_pool(pd.DataFrame({"Game Info": ["KC@BUF 09/07/2025 01:00PM ET"]}))
    wrong_week = [
        {
            "season": "2025",
            "season_type": "regular",
            "week": 2,
            "team": "BUF",
            "opponent": "KC",
            "date": "2025-09-14",
            "player": {},
        }
    ]
    result = validate_projection_records(wrong_week, slate, "2025", 1, "regular")
    assert result["ok"] is False
    assert result["error"] == "projection_week_mismatch"

    wrong_game = [
        {
            "season": "2025",
            "season_type": "regular",
            "week": 1,
            "team": "BUF",
            "opponent": "BAL",
            "date": "2025-09-07",
        }
    ]
    result = validate_projection_records(wrong_game, slate, "2025", 1, "regular")
    assert result["error"] == "projection_slate_mismatch"

    unlabeled = [
        {"season": "2025", "season_type": "post", "week": 1, "team": "BUF", "player": {"injury_status": "Out"}}
    ]
    post = slate_games_from_pool(pd.DataFrame({"Game Info": ["BUF@JAX 01/11/2026 01:00PM ET"]}))
    result = validate_projection_records(unlabeled, post, "2025", 1, "post")
    assert result["ok"] is True
    assert result["dates_confirmed"] is False
    assert "unavailable" in result["message"]


def test_overlay_rejects_a_projection_week_that_does_not_match_dates(tmp_path, monkeypatch):
    def _http(url: str, timeout: float = 30):
        if "/projections/" in url:
            row = _week_record("Josh", "Allen", "BUF", "QB", 30.0)
            row["week"] = 2
            row["date"] = "2025-09-14"
            return [row]
        if url.endswith("/players/nfl"):
            return {}
        raise AssertionError(url)

    monkeypatch.setattr("src.sleeper._http_get", _http)
    updated, summary = overlay_sleeper_projections(
        pd.DataFrame([_pool_row(Projection=9.0)]),
        force_refresh=True,
        cache_path=tmp_path / "cache.json",
        now=datetime(2026, 11, 7, tzinfo=timezone.utc),
        schedule_games=_schedule(),
    )
    assert updated.loc[0, "Projection"] == 9.0
    assert summary["error"] == "projection_week_mismatch"
    assert summary["matching_ran"] is False
    assert "Projection coverage: 0" in sleeper_status_caption(summary)


def test_multi_week_slate_does_not_call_the_current_week(tmp_path, monkeypatch):
    calls: list[str] = []

    def _http(url: str, timeout: float = 30):
        calls.append(url)
        raise AssertionError(url)

    monkeypatch.setattr("src.sleeper._http_get", _http)
    pool = pd.DataFrame(
        [
            _pool_row(**{"Game Info": "KC@BUF 09/07/2025 01:00PM ET", "Projection": 9.0}),
            _pool_row(
                ID="2",
                Name="Jalen Hurts",
                TeamAbbrev="PHI",
                **{"Game Info": "DAL@PHI 09/15/2025 01:00PM ET", "Projection": 8.0},
            ),
        ]
    )
    updated, summary = overlay_sleeper_projections(
        pool,
        force_refresh=True,
        cache_path=tmp_path / "cache.json",
        now=datetime(2026, 11, 7, tzinfo=timezone.utc),
        schedule_games=_schedule()
        + _schedule(away="DAL", home="PHI", game_date="2025-09-15", week=2),
    )
    assert calls == []
    assert summary["error"] == "ambiguous_slate_week"
    assert list(updated["Projection"]) == [9.0, 8.0]


def coerce_one(game_date, season, week, season_type, away, home):
    from src.slate_week import coerce_schedule

    return coerce_schedule(
        [
            {
                "game_date": game_date,
                "season": season,
                "week": week,
                "season_type": season_type,
                "away": away,
                "home": home,
            }
        ]
    )
