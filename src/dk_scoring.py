"""DraftKings NFL Classic and Showdown scoring.

Rules pages read for this module on 2026-11-07:

- Classic: https://www.draftkings.com/help/rules/nfl/classic
- Showdown Captain Mode:
  https://www.draftkings.com/help/rules/nfl/showdown-captain-mode

Classic and Showdown use the same offensive and defensive tables. Showdown
adds kickers and says a captain scores 1.5x each statistic. This module scores
one player at 1x. The Showdown optimizer applies the 1.5x once when it builds
the captain row. Export writes the player id and does not scale again.

Realized scoring
----------------
A finished stat line uses the table exactly. Crossing 300 passing yards or
100 rushing or receiving yards adds 3. Points allowed uses the tier of that
integer total. Made kicks score by distance. Missed field goals and missed
extra points are not scoring categories on the Showdown page, so they add 0.
Kickers do not get credit for other stats. Classic has no kicker table, so a
kicker is not scored for a Classic contest.

Projection scoring
------------------
Linear categories (yards, touchdowns, receptions, sacks, interceptions) use
the component mean. That is the expectation of a linear rule.

Milestone bonuses are not linear. The Sleeper weekly feed has the yardage
mean and, on a finished game, a 0/1 ``bonus_pass_yd_300`` flag. It does not
publish P(yards >= threshold) for a projection. ``bonus_rec_wr`` on that feed
matches receptions, which is the PPR point, not the 100-yard bonus.

Choices that were rejected:

- Add 3 whenever the mean is on the threshold. That treats a forecast as a
  box score.
- Treat Sleeper's single ``pts_allow_* = 1`` flag as a probability. On the
  week inspected, every defense had exactly one flag and it was the bucket
  that contains the average. An average is not a realized tier.
- Drop the bonus and leave it at 0. That deletes a real DraftKings term when
  a documented approximation is available.

Choice used for the bonus and, unless a real bucket distribution is present,
for points allowed: a normal approximation. The mean is the Sleeper component.
The scale is the median within-player (or within-team) week-to-week sample
standard deviation from Sleeper actuals for the 2025 regular season, weeks
1-18, among players or defenses with at least 8 games:

- Passing yards: 65 quarterbacks with at least 15 attempts in a game,
  median SD 64.24.
- Rushing yards: 44 players with at least 8 rushes and fewer than 5 pass
  attempts, median SD 31.66.
- Receiving yards for WR and TE: 82 players with at least 3 receptions and
  fewer than 5 rushes, median SD 27.48.
- Receiving yards for RB: 75 players with at least 8 rushes, median SD 24.42.
- Points allowed: 32 defenses, median within-team SD 8.90.

Yards are treated as discrete. P(Y >= threshold) uses a continuity correction,
1 - Phi((threshold - 0.5 - mean) / sd). The bonus expectation is 3 times that
probability. Points allowed uses the same normal, truncated at 0 and
renormalized, then the official tier of each integer. This scale is the
week-to-week spread around a player's own season, which is wider than the
error of a matchup projection. It is not a betting probability.

When the feed has two or more points-allowed bucket weights that sum to about
1, those weights are used instead of the normal approximation.

A defense's yards allowed are on the Sleeper record and are not on either
DraftKings table, so they are reported as unused. ``def_td`` is the total of
the component touchdown fields when both are present, so the components are
not added a second time.

Sleeper omits zeroes. A projection is scored only when the position's required
keys are present. Optional keys that are absent count as zero and are listed
on the result. A partial line does not produce a total. The caller decides
whether to fall back to ``pts_ppr``. ``pts_std`` is never read here.

Kicker projections are scored only when the made-by-distance buckets account
for ``fgm`` within 0.05. On the 2026 week-9 feed those buckets fell short for
almost every kicker, and ``fgm_0_19`` was missing. The gap is not assigned to
a distance.

Historical reconciliation
-------------------------
``HISTORICAL_RECONCILIATION`` is ``unverified``. Sleeper 2025 week 1 actuals
for Josh Allen (id 4984: 394 pass yards, 2 pass TD, 30 rush yards, 2 rush TD)
and Derrick Henry (id 3198: 169 rush yards, 2 rush TD, 1 catch, 13 rec yards,
1 fumble lost) produce 41.76 and 33.2 under this table. Those are formula
results, not official DraftKings totals. The current DraftKings lobby had no
completed slate to read a ``fantasyPoints`` field from, and the historical
draft-group endpoint returned 403. A DraftKings Network week-1 leaders article
lists Allen at 38.76 and Henry at 29.2, which match Sleeper PPR and omit the
+3 the rules pages specify, so that article was not used as the official
ledger. Third-party writeups that say 41.76 were not treated as DraftKings
either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional

CLASSIC_RULES_URL = "https://www.draftkings.com/help/rules/nfl/classic"
SHOWDOWN_RULES_URL = "https://www.draftkings.com/help/rules/nfl/showdown-captain-mode"
SCORING_VERSION = "dk-component-v1"
HISTORICAL_RECONCILIATION = "unverified"

PASS_YARDS_SD = 64.24
RUSH_YARDS_SD = 31.66
REC_YARDS_SD = 27.48
REC_YARDS_RB_SD = 24.42
POINTS_ALLOWED_SD = 8.90
KICKER_SPLIT_TOLERANCE = 0.05

_QB_REQUIRED = ("pass_yd", "pass_td", "pass_int")
_RB_REQUIRED = ("rush_yd", "rush_td")
_RECEIVING_REQUIRED = ("rec", "rec_yd", "rec_td")
_WR_REQUIRED = _RECEIVING_REQUIRED
_DST_REQUIRED = ("sack", "int", "fum_rec", "pts_allow")
_KICKER_REQUIRED = ("xpm",)

_PA_BUCKETS = (
    ("pts_allow_0", 0),
    ("pts_allow_1_6", 1),
    ("pts_allow_7_13", 7),
    ("pts_allow_14_20", 14),
    ("pts_allow_21_27", 21),
    ("pts_allow_28_34", 28),
    ("pts_allow_35", 35),
    ("pts_allow_35p", 35),
)
_DST_TD_PARTS = (
    "def_fum_td",
    "pass_int_td",
    "def_pr_td",
    "def_kr_td",
    "def_st_td",
    "st_td",
)
_SHORT_FG = ("fgm_0_19", "fgm_20_29", "fgm_30_39")
_NOT_SCORED_WHEN_PRESENT = (
    "yds_allow",
    "ff",
    "tkl_loss",
    "fum",
    "pts_std",
    "pts_half_ppr",
    "bonus_rec_wr",
    "bonus_rec_rb",
    "bonus_rec_te",
    "bonus_pass_yd_300",
    "bonus_rush_yd_100",
    "bonus_rec_yd_100",
)


@dataclass
class ScoreResult:
    """One player's DraftKings total, or a refusal to invent one."""

    points: Optional[float]
    basis: str
    warning: str = ""
    missing: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()
    omitted_as_zero: tuple[str, ...] = ()
    bonuses: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.basis == "draftkings" and self.points is not None


def canon_scoring_position(position: object) -> str:
    """Football position used by the scoring table."""
    token = str(position or "").split("/")[0].strip().upper()
    if token in {"DEF", "D", "DST"}:
        return "DST"
    if token == "FB":
        return "RB"
    return token


def points_allowed_tier(points_allowed: float) -> float:
    """Official points-allowed tier. The argument is a realized total."""
    allowed = float(points_allowed)
    if allowed <= 0:
        return 10.0
    if allowed <= 6:
        return 7.0
    if allowed <= 13:
        return 4.0
    if allowed <= 20:
        return 1.0
    if allowed <= 27:
        return 0.0
    if allowed <= 34:
        return -1.0
    return -4.0


def expected_milestone_bonus(mean: float, threshold: float, scale: float) -> float:
    """3 times P(discrete yards >= threshold) under Normal(mean, scale)."""
    if scale <= 0:
        return 3.0 if float(mean) >= float(threshold) else 0.0
    z = (float(threshold) - 0.5 - float(mean)) / float(scale)
    probability = 1.0 - _normal_cdf(z)
    probability = min(1.0, max(0.0, probability))
    return 3.0 * probability


def score_realized(
    stats: Mapping[str, object],
    *,
    position: str,
    contest: str = "classic",
) -> ScoreResult:
    """Exact DraftKings total for one finished stat line."""
    return _score(stats, position=position, contest=contest, realized=True)


def score_projection(
    stats: Mapping[str, object],
    *,
    position: str,
    contest: str = "classic",
) -> ScoreResult:
    """DraftKings expectation from component means. Bonuses are not flat."""
    return _score(stats, position=position, contest=contest, realized=False)


def _score(
    stats: Mapping[str, object],
    *,
    position: str,
    contest: str,
    realized: bool,
) -> ScoreResult:
    pos = canon_scoring_position(position)
    contest_name = str(contest or "classic").strip().lower()
    if pos == "K" and contest_name != "showdown":
        return ScoreResult(
            points=None,
            basis="unavailable",
            warning=(
                "Classic rules do not include kickers. "
                "Showdown is the format that scores them."
            ),
            missing=("kicker",),
        )
    if pos == "K":
        return _score_kicker(stats, realized=realized)
    if pos == "DST":
        return _score_dst(stats, realized=realized)
    if pos == "QB":
        return _score_skill(stats, position=pos, required=_QB_REQUIRED, realized=realized)
    if pos == "RB":
        return _score_skill(stats, position=pos, required=_RB_REQUIRED, realized=realized)
    if pos in {"WR", "TE"}:
        return _score_skill(stats, position=pos, required=_WR_REQUIRED, realized=realized)
    return ScoreResult(
        points=None,
        basis="unavailable",
        warning=f"No DraftKings table for position {pos or '(blank)'}.",
        missing=("position",),
    )


def _score_skill(
    stats: Mapping[str, object],
    *,
    position: str,
    required: tuple[str, ...],
    realized: bool,
) -> ScoreResult:
    required_keys = tuple(required)
    if position == "RB":
        receiving_present = [key for key in _RECEIVING_REQUIRED if key in stats]
        if receiving_present and len(receiving_present) != len(_RECEIVING_REQUIRED):
            missing = [key for key in _RECEIVING_REQUIRED if key not in stats]
            return _refuse(missing, "Receiving stats are only partly present.")
        if not receiving_present:
            required_keys = tuple(required)
        else:
            required_keys = tuple(required) + _RECEIVING_REQUIRED
    missing = _missing(stats, required_keys)
    if missing:
        return _refuse(missing, "Required component stats are missing.")
    omitted: list[str] = []
    points = 0.0
    bonuses: dict[str, float] = {}

    pass_yd = _optional(stats, "pass_yd", omitted)
    pass_td = _optional(stats, "pass_td", omitted) if position != "QB" else _required_num(stats, "pass_td")
    pass_int = _optional(stats, "pass_int", omitted) if position != "QB" else _required_num(stats, "pass_int")
    if position == "QB":
        pass_yd = _required_num(stats, "pass_yd")
    rush_yd = _optional(stats, "rush_yd", omitted) if position != "RB" else _required_num(stats, "rush_yd")
    rush_td = _optional(stats, "rush_td", omitted) if position != "RB" else _required_num(stats, "rush_td")
    if position == "RB" and not any(key in stats for key in _RECEIVING_REQUIRED):
        rec = rec_yd = rec_td = 0.0
        omitted.extend(_RECEIVING_REQUIRED)
    elif position in {"WR", "TE"}:
        rec = _required_num(stats, "rec")
        rec_yd = _required_num(stats, "rec_yd")
        rec_td = _required_num(stats, "rec_td")
    else:
        rec = _optional(stats, "rec", omitted)
        rec_yd = _optional(stats, "rec_yd", omitted)
        rec_td = _optional(stats, "rec_td", omitted)

    points += pass_yd * 0.04
    points += pass_td * 4.0
    points += pass_int * -1.0
    points += rush_yd * 0.1
    points += rush_td * 6.0
    points += rec_yd * 0.1
    points += rec_td * 6.0
    points += rec * 1.0
    points += _optional(stats, "fum_lost", omitted) * -1.0
    for key in ("pass_2pt", "rush_2pt", "rec_2pt"):
        points += _optional(stats, key, omitted) * 2.0
    for key in ("pr_td", "kr_td", "fum_rec_td"):
        points += _optional(stats, key, omitted) * 6.0

    rec_scale = REC_YARDS_RB_SD if position == "RB" else REC_YARDS_SD
    bonuses["pass_300"] = _bonus(pass_yd, 300, PASS_YARDS_SD, realized)
    bonuses["rush_100"] = _bonus(rush_yd, 100, RUSH_YARDS_SD, realized)
    bonuses["rec_100"] = _bonus(rec_yd, 100, rec_scale, realized)
    points += sum(bonuses.values())
    return ScoreResult(
        points=points,
        basis="draftkings",
        omitted_as_zero=tuple(omitted),
        bonuses=bonuses,
        unavailable=_unused(stats),
    )


def _score_dst(stats: Mapping[str, object], *, realized: bool) -> ScoreResult:
    missing = _missing(stats, _DST_REQUIRED)
    if missing:
        return _refuse(missing, "Required defense stats are missing.")
    omitted: list[str] = []
    unavailable: list[str] = list(_unused(stats))
    points = _required_num(stats, "sack") * 1.0
    points += _required_num(stats, "int") * 2.0
    points += _required_num(stats, "fum_rec") * 2.0
    points += _dst_touchdowns(stats) * 6.0
    points += _optional(stats, "safe", omitted) * 2.0
    points += _optional(stats, "blk_kick", omitted) * 2.0
    points += _optional(stats, "def_2pt", omitted) * 2.0

    tier_points, tier_unavailable, warning = _points_allowed(stats, realized=realized)
    unavailable.extend(tier_unavailable)
    points += tier_points
    return ScoreResult(
        points=points,
        basis="draftkings",
        warning=warning,
        unavailable=tuple(dict.fromkeys(unavailable)),
        omitted_as_zero=tuple(omitted),
        bonuses={"points_allowed": tier_points},
    )


def _score_kicker(stats: Mapping[str, object], *, realized: bool) -> ScoreResult:
    del realized  # Makes and misses are linear, so the mean is the expectation.
    missing = _missing(stats, _KICKER_REQUIRED)
    short, mid, long, accounted, split_missing = _field_goals(stats)
    if split_missing:
        missing.extend(split_missing)
    if missing:
        detail = "Field goal distance split does not account for field goals made."
        if "xpm" in missing and len(missing) == 1:
            detail = "Extra points made are missing."
        return _refuse(missing, detail)
    points = short * 3.0 + mid * 4.0 + long * 5.0 + _required_num(stats, "xpm") * 1.0
    return ScoreResult(
        points=points,
        basis="draftkings",
        bonuses={"missed_kicks": 0.0, "field_goals": accounted},
        unavailable=_unused(stats),
    )


def _points_allowed(
    stats: Mapping[str, object],
    *,
    realized: bool,
) -> tuple[float, list[str], str]:
    weights = _bucket_weights(stats)
    if realized or weights is not None:
        if realized:
            return points_allowed_tier(_required_num(stats, "pts_allow")), [], ""
        assert weights is not None
        total = sum(weight for _, weight in weights)
        expected = sum(points_allowed_tier(bucket) * (weight / total) for bucket, weight in weights)
        return expected, [], ""
    expected = _expected_points_allowed(_required_num(stats, "pts_allow"), POINTS_ALLOWED_SD)
    warning = (
        "Points allowed is an average, not a distribution, so the tier "
        "points are an expected value from a normal approximation."
    )
    if any(str(key).startswith("yds_allow") for key in stats):
        warning += " Yards allowed are not a DraftKings category and were not scored."
    return expected, ["points_allowed_distribution"], warning


def _expected_points_allowed(mean: float, scale: float) -> float:
    """E[tier] for a normal points-allowed mean, truncated at zero."""
    if scale <= 0:
        return points_allowed_tier(mean)
    weights: list[float] = []
    tiers: list[float] = []
    for allowed in range(0, 71):
        low = allowed - 0.5
        high = allowed + 0.5
        weights.append(_normal_cdf((high - mean) / scale) - _normal_cdf((low - mean) / scale))
        tiers.append(points_allowed_tier(allowed))
    tail = 1.0 - _normal_cdf((70.5 - mean) / scale)
    usable = sum(weights) + tail
    if usable <= 0:
        return points_allowed_tier(mean)
    expected = sum(weight * tier for weight, tier in zip(weights, tiers))
    expected += tail * points_allowed_tier(70)
    return expected / usable


def _bucket_weights(stats: Mapping[str, object]) -> Optional[list[tuple[int, float]]]:
    present: list[tuple[int, float]] = []
    for key, bucket in _PA_BUCKETS:
        number = _number(stats.get(key)) if key in stats else None
        if number is None:
            continue
        present.append((bucket, number))
    if len(present) < 2:
        return None
    total = sum(weight for _, weight in present)
    if abs(total - 1.0) > 0.05:
        return None
    return present


def _dst_touchdowns(stats: Mapping[str, object]) -> float:
    total = _number(stats.get("def_td")) if "def_td" in stats else None
    if total is not None:
        return total
    return sum(_number(stats.get(key)) or 0.0 for key in _DST_TD_PARTS if key in stats)


def _field_goals(
    stats: Mapping[str, object],
) -> tuple[float, float, float, float, list[str]]:
    bucket_keys = [key for key in (*_SHORT_FG, "fgm_40_49", "fgm_50p", "fgm_50_59", "fgm_60p") if key in stats]
    if not bucket_keys:
        made = _number(stats.get("fgm")) if "fgm" in stats else None
        if made is None:
            return 0.0, 0.0, 0.0, 0.0, ["fgm"]
        # Zero makes have nothing to split. A positive total without distances
        # would hide which point value those kicks earn.
        if abs(made) <= KICKER_SPLIT_TOLERANCE:
            return 0.0, 0.0, 0.0, 0.0, []
        return 0.0, 0.0, 0.0, 0.0, ["fgm_distance"]
    short = sum(_number(stats.get(key)) or 0.0 for key in _SHORT_FG if key in stats)
    mid = _number(stats.get("fgm_40_49")) or 0.0 if "fgm_40_49" in stats else 0.0
    if "fgm_50p" in stats:
        long = _number(stats.get("fgm_50p")) or 0.0
    else:
        long = (_number(stats.get("fgm_50_59")) or 0.0) + (_number(stats.get("fgm_60p")) or 0.0)
    accounted = short + mid + long
    if "fgm" in stats:
        made = _number(stats.get("fgm"))
        if made is None or abs(accounted - made) > KICKER_SPLIT_TOLERANCE:
            # ``fgm`` may be present and still not usable: the distance buckets
            # do not add up to it, so the point value of the missing makes is unknown.
            return short, mid, long, accounted, ["fgm_distance"]
    return short, mid, long, accounted, []


def _bonus(yards: float, threshold: float, scale: float, realized: bool) -> float:
    if yards <= 0:
        # A zero component mean is "none of that stat," not a distribution
        # centered at zero. The yardage scales were fit on players who had
        # the stat, so they are not applied to a zero projection.
        return 0.0
    if realized:
        return 3.0 if yards >= threshold else 0.0
    return expected_milestone_bonus(yards, threshold, scale)


def _refuse(missing: list[str], warning: str) -> ScoreResult:
    names = ", ".join(missing)
    return ScoreResult(
        points=None,
        basis="unavailable",
        warning=f"{warning} Missing {names}.",
        missing=tuple(missing),
    )


def _missing(stats: Mapping[str, object], keys: tuple[str, ...]) -> list[str]:
    missing = []
    for key in keys:
        if key not in stats or _number(stats.get(key)) is None:
            missing.append(key)
    return missing


def _required_num(stats: Mapping[str, object], key: str) -> float:
    number = _number(stats.get(key))
    if number is None:
        return 0.0
    return number


def _optional(stats: Mapping[str, object], key: str, omitted: list[str]) -> float:
    if key not in stats:
        omitted.append(key)
        return 0.0
    number = _number(stats.get(key))
    if number is None:
        omitted.append(key)
        return 0.0
    return number


def _unused(stats: Mapping[str, object]) -> tuple[str, ...]:
    found = [key for key in _NOT_SCORED_WHEN_PRESENT if key in stats]
    if any(key.startswith("yds_allow") for key in stats):
        found.append("yds_allow")
    return tuple(dict.fromkeys(found))


def _number(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
