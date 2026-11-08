"""Refit the DraftKings milestone and points-allowed scales.

The fit uses only the completed 2025 regular season, weeks 1 through 18, from
the public Sleeper stats API. The 2026 season is not included, so a projection
for the current week is not fit on itself.

Source pulls, both unauthenticated:

- ``https://api.sleeper.app/v1/stats/nfl/regular/2025/{week}`` for week 1..18
- ``https://api.sleeper.app/v1/players/nfl`` for the position label

Rules:

- Sample standard deviation uses ``n - 1``.
- A player or defense needs at least 8 qualifying weeks.
- The reported scale is the median of those standard deviations. An even count
  uses the average of the two central values.
- Sleeper omits a zero. On a qualifying week, a missing yardage or
  points-allowed key is 0. Weeks that do not qualify are left out, not filled
  with a zero, so a backup's empty weeks do not shrink a starter's scale.
- Ids that start with ``TEAM`` and 1-3 letter team ids are not players.
- Position is the player's primary Sleeper position.

Qualifying weeks:

- Passing: position QB and at least 15 pass attempts.
- Rushing: position RB and at least 8 rushes.
- RB receiving: those same RB rush weeks. Missing receiving yards are 0.
- WR receiving: position WR and at least 3 receptions.
- TE receiving: position TE and at least 3 receptions.
- Points allowed: a team-defense id that has ``pts_allow``.

A run on 2026-11-07 printed:

- pass 33 63.27
- rush 44 31.57
- rec_wr 53 28.57
- rec_te 29 24.01
- rec_rb 44 16.94
- points_allowed 32 9.28

Those rounded values are ``PASS_YARDS_SD`` and the other scales in
``src/dk_scoring.py``. An earlier fit counted team aggregates as quarterbacks
and dropped receiving weeks that omitted a zero, which is why it reported
65 / 64.24 and 75 / 24.42.
"""

from __future__ import annotations

import json
import math
import urllib.request
from collections import defaultdict
from typing import Iterable, Mapping, Optional

SEASON = "2025"
WEEKS = range(1, 19)
MIN_WEEKS = 8
STATS_URL = "https://api.sleeper.app/v1/stats/nfl/regular/{season}/{week}"
PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"


def sample_sd(values: Iterable[float]) -> Optional[float]:
    """Sample standard deviation. Fewer than two values has no scale."""
    rows = [float(value) for value in values]
    if len(rows) < 2:
        return None
    mean = sum(rows) / len(rows)
    variance = sum((value - mean) ** 2 for value in rows) / (len(rows) - 1)
    return math.sqrt(variance)


def median_scale(deviations: Iterable[float]) -> Optional[float]:
    """Median. An even count averages the two central values."""
    rows = sorted(float(value) for value in deviations)
    count = len(rows)
    if count == 0:
        return None
    middle = count // 2
    if count % 2:
        return rows[middle]
    return (rows[middle - 1] + rows[middle]) / 2.0


def cohort_scale(weeks_by_player: Mapping[str, Iterable[float]], *, min_weeks: int = MIN_WEEKS) -> tuple[int, Optional[float]]:
    deviations = []
    for values in weeks_by_player.values():
        rows = list(values)
        if len(rows) < min_weeks:
            continue
        scale = sample_sd(rows)
        if scale is not None:
            deviations.append(scale)
    return len(deviations), median_scale(deviations)


def _is_team_id(player_id: str) -> bool:
    return player_id.startswith("TEAM") or (player_id.isalpha() and len(player_id) <= 3)


def _primary_position(players: Mapping[str, object], player_id: str) -> str:
    raw = players.get(player_id)
    if not isinstance(raw, dict):
        return ""
    return str(raw.get("position") or "").split("/")[0].strip().upper()


def _get(url: str) -> object:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "dfs-lineup-generator bonus-scale fit", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def fit(stats_by_week: Iterable[Mapping[str, object]], players: Mapping[str, object]) -> dict[str, tuple[int, Optional[float]]]:
    """Scale each cohort from already-downloaded week payloads."""
    passing: dict[str, list[float]] = defaultdict(list)
    rushing: dict[str, list[float]] = defaultdict(list)
    receiving_rb: dict[str, list[float]] = defaultdict(list)
    receiving_wr: dict[str, list[float]] = defaultdict(list)
    receiving_te: dict[str, list[float]] = defaultdict(list)
    points_allowed: dict[str, list[float]] = defaultdict(list)
    for week in stats_by_week:
        for player_id, stats in week.items():
            if not isinstance(stats, dict):
                continue
            if _is_team_id(player_id):
                if "pts_allow" in stats:
                    points_allowed[player_id].append(float(stats["pts_allow"]))
                continue
            position = _primary_position(players, player_id)
            if position == "QB" and float(stats.get("pass_att") or 0) >= 15:
                passing[player_id].append(float(stats["pass_yd"]) if "pass_yd" in stats else 0.0)
            if position == "RB" and float(stats.get("rush_att") or 0) >= 8:
                rushing[player_id].append(float(stats["rush_yd"]) if "rush_yd" in stats else 0.0)
                receiving_rb[player_id].append(float(stats["rec_yd"]) if "rec_yd" in stats else 0.0)
            if position == "WR" and float(stats.get("rec") or 0) >= 3:
                receiving_wr[player_id].append(float(stats["rec_yd"]) if "rec_yd" in stats else 0.0)
            if position == "TE" and float(stats.get("rec") or 0) >= 3:
                receiving_te[player_id].append(float(stats["rec_yd"]) if "rec_yd" in stats else 0.0)
    return {
        "pass": cohort_scale(passing),
        "rush": cohort_scale(rushing),
        "rec_wr": cohort_scale(receiving_wr),
        "rec_te": cohort_scale(receiving_te),
        "rec_rb": cohort_scale(receiving_rb),
        "points_allowed": cohort_scale(points_allowed),
    }


def main() -> None:
    players = _get(PLAYERS_URL)
    if not isinstance(players, dict):
        raise SystemExit("Sleeper players payload was not an object")
    weeks = []
    for week in WEEKS:
        payload = _get(STATS_URL.format(season=SEASON, week=week))
        if not isinstance(payload, dict):
            raise SystemExit(f"Sleeper stats for week {week} were not an object")
        weeks.append(payload)
    scales = fit(weeks, players)
    for name, (count, scale) in scales.items():
        printed = "none" if scale is None else f"{scale:.2f}"
        print(f"{name} {count} {printed}")


if __name__ == "__main__":
    main()
