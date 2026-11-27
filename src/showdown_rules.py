"""Showdown lineup rules: captain eligibility, players per team, and a salary floor.

These choose who can share a lineup and how much salary it spends. They do
not change a projection. The checks here are the same ones the optimizer
enforces, so a test can judge a lineup without rendering Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd
from pydfs_lineup_optimizer.rules import OptimizerRule
from pydfs_lineup_optimizer.solvers.constants import SolverSign

from .names import person_key
from .portfolio import position_tokens

RULE_NO_K_DST_CAPTAIN = "No kicker or DST captain"
RULE_MIN_PER_TEAM = "Minimum players per team"
RULE_SALARY_FLOOR = "Salary floor"
RULE_TEAM_SPLIT = "Team split"

# Showdown control. Any, or nothing selected, adds no split constraint.
# The other three are the only constructions DraftKings allows on six spots.
TEAM_SPLIT_ANY = "Any"
TEAM_SPLIT_OPTIONS = (TEAM_SPLIT_ANY, "3-3", "4-2", "5-1")
_TEAM_SPLIT_ORDER = ((3, 3), (4, 2), (5, 1))
_TEAM_SPLIT_LABELS = {f"{high}-{low}": (high, low) for high, low in _TEAM_SPLIT_ORDER}

# DraftKings Showdown is six spots. pydfs captain mode sets
# max_from_one_team to 5, so a lineup always includes both teams. That cap
# does not look at this switch. The switch raises the floor. The control
# offers 2 through 5. Four or five from each team cannot fit six spots and
# is refused before the solver runs.
MIN_PLAYERS_PER_TEAM_MIN = 1
MIN_PLAYERS_PER_TEAM_MAX = 5
DEFAULT_MIN_PLAYERS_PER_TEAM = 2
CONTROL_MIN_PLAYERS_PER_TEAM = 2
CONTROL_MAX_PLAYERS_PER_TEAM = 5
REQUIRE_MULTIPLE_PLAYERS_LABEL = "Require multiple players from each team"
SHOWDOWN_ROSTER_SIZE = 6
DEFAULT_MAX_UNUSED_SALARY = 1_500
DEFAULT_SHOWDOWN_CAP = 50_000

_K_DST = frozenset({"K", "DST", "DEF"})
_SLOT_COLUMNS = ("RosterSlot", "Roster Position", "Slot")


@dataclass(frozen=True)
class ShowdownRuleSettings:
    """On/off switches and the values next to them."""

    ban_k_dst_captain: bool = False
    min_players_per_team_enabled: bool = False
    min_players_per_team: int = DEFAULT_MIN_PLAYERS_PER_TEAM
    salary_floor: bool = False
    max_unused_salary: int = DEFAULT_MAX_UNUSED_SALARY

    @property
    def any_enabled(self) -> bool:
        return bool(
            self.ban_k_dst_captain
            or self.min_players_per_team_enabled
            or self.salary_floor
        )


@dataclass(frozen=True)
class SalaryBounds:
    """Spend range after the salary window and the unused-salary floor meet.

    ``conflict`` is set when those two cannot both be true. Nothing is
    dropped to make the range nonempty. ``note`` explains which minimum
    is tighter when they can both be true.
    """

    slate_cap: int
    minimum: Optional[int]
    maximum: int
    floor_spend: Optional[int]
    conflict: Optional[str]
    note: str


def shortfall_message(requested: int, built: int, cause: str) -> str:
    """Same shape as the portfolio shortfall: how many, then the binding rule."""
    return (
        f"Requested {int(requested)} lineups and built {int(built)}. "
        f"Binding limit: {cause}"
    )


def football_position(value: object) -> str:
    """Primary football position. CPT and FLEX are roster slots, not positions."""
    tokens = position_tokens(value)
    if "DST" in tokens or "DEF" in tokens:
        return "DST"
    for token in ("K", "QB", "RB", "WR", "TE"):
        if token in tokens:
            return token
    text = str(value or "").split("/")[0].strip().upper()
    if text == "DEF":
        return "DST"
    return text


def is_kicker_or_dst(value: object) -> bool:
    return football_position(value) in {"K", "DST"}


def _player_label(name: object, team: object) -> str:
    label = str(name or "").strip() or "that player"
    club = str(team or "").strip().upper()
    if club and club != "NAN":
        return f"{label} ({club})"
    return label


def _slot(row: pd.Series) -> str:
    for column in _SLOT_COLUMNS:
        if column not in row.index:
            continue
        raw = row.get(column)
        if raw is None:
            continue
        try:
            if raw != raw:  # NaN
                continue
        except TypeError:
            pass
        text = str(raw).split("/")[0].strip().upper()
        if text and text != "NAN":
            return text
    return ""


def _captain_rate(raw: object) -> Optional[float]:
    if raw is None:
        return None
    try:
        if raw != raw:
            return None
    except TypeError:
        pass
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value > 1.0:
        value = value / 100.0
    if value <= 0:
        return None
    return value


def captain_lock_causes(
    pool: Optional[pd.DataFrame],
    locked_ids: Optional[Iterable[str]] = None,
) -> tuple[str, ...]:
    """K or DST forced into CPT. The lock is named and is not dropped.

    A locked CPT row is one case. A captain share above zero is the other:
    that share requires the CPT row for part of the batch. A lock on the
    FLEX row is not a captain lock. K and DST stay eligible at FLEX.
    """
    if pool is None or getattr(pool, "empty", True):
        return ()
    locked = {str(player_id) for player_id in (locked_ids or [])}
    causes: list[str] = []
    seen: set[str] = set()
    for _, row in pool.iterrows():
        if not is_kicker_or_dst(row.get("Position")):
            continue
        label = _player_label(row.get("Name"), row.get("TeamAbbrev", row.get("Team")))
        if label in seen:
            continue
        player_id = str(row.get("ID", "")).strip()
        if player_id in locked and _slot(row) == "CPT":
            seen.add(label)
            causes.append(
                f"{RULE_NO_K_DST_CAPTAIN}. {label} is locked at CPT. "
                "K and DST can be FLEX and cannot be CPT. The lock was not dropped."
            )
            continue
        rate = _captain_rate(row.get("CaptainExposure"))
        if rate is None:
            continue
        seen.add(label)
        causes.append(
            f"{RULE_NO_K_DST_CAPTAIN}. {label} is locked at CPT by a captain share "
            f"of {rate * 100:.0f}%. K and DST can be FLEX and cannot be CPT. "
            "That captain share was not dropped."
        )
    return tuple(causes)


def team_player_counts(lineup: pd.DataFrame) -> dict[str, int]:
    """Players per team. CPT and FLEX each count as one."""
    if lineup is None or getattr(lineup, "empty", True):
        return {}
    team_col = "Team" if "Team" in lineup.columns else "TeamAbbrev"
    if team_col not in lineup.columns:
        return {}
    counts: dict[str, int] = {}
    for team in lineup[team_col]:
        club = str(team or "").strip().upper()
        if not club or club == "NAN":
            continue
        counts[club] = counts.get(club, 0) + 1
    return counts


def team_minimum_met(
    counts: Mapping[str, int],
    teams: Sequence[str],
    minimum: int,
) -> bool:
    """True when every named team has at least ``minimum`` players."""
    floor = int(minimum)
    return all(int(counts.get(str(team).strip().upper(), 0)) >= floor for team in teams)


def team_minimum_cause(
    counts: Mapping[str, int],
    teams: Sequence[str],
    minimum: int,
) -> Optional[str]:
    """Name the teams under the minimum. None when every team clears it."""
    floor = int(minimum)
    short: list[str] = []
    for team in teams:
        club = str(team).strip().upper()
        if not club:
            continue
        have = int(counts.get(club, 0))
        if have < floor:
            short.append(f"{club} has {have}")
    if not short:
        return None
    return (
        f"{RULE_MIN_PER_TEAM}. {', '.join(short)}, and the minimum is {floor} "
        "from each team. CPT and FLEX each count as one player. "
        "The minimum was not lowered."
    )


def team_minimum_fit_cause(
    enabled: bool,
    minimum: int,
    roster_size: int = SHOWDOWN_ROSTER_SIZE,
) -> Optional[str]:
    """At least N from each team cannot fit a Showdown roster."""
    if not enabled:
        return None
    floor = int(minimum)
    spots = int(roster_size)
    needed = floor * 2
    if needed <= spots:
        return None
    return (
        f"{RULE_MIN_PER_TEAM}. The minimum is {floor} from each team, "
        f"and a Showdown lineup has {spots} spots. "
        f"{floor} from each team needs {needed} players. "
        "The minimum was not lowered."
    )


def multiple_players_per_team(enabled: bool, count: int) -> dict[str, int | bool]:
    """Optimizer fields for the Showdown control.

    Off leaves DraftKings' one-from-each-team rule in place. The count is
    stored either way so a later generation can turn the rule on without
    reading a second setting.
    """
    return {
        "min_players_per_team_enabled": bool(enabled),
        "min_players_per_team": int(count),
    }


def _split_number(value: object) -> Optional[int]:
    if isinstance(value, bool) or isinstance(value, str):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _is_split_pair(value: object) -> bool:
    if isinstance(value, (str, bytes, dict)) or not isinstance(value, (list, tuple)):
        return False
    if len(value) != 2:
        return False
    return _split_number(value[0]) is not None and _split_number(value[1]) is not None


def _parse_split_pair(value: object) -> tuple[int, int]:
    """One ``(high, low)`` pair. A saved label such as ``"4-2"`` is accepted."""
    if isinstance(value, str):
        label = value.strip()
        parsed = _TEAM_SPLIT_LABELS.get(label)
        if parsed is None:
            raise ValueError(
                "team split must be 3-3, 4-2, or 5-1"
            )
        return parsed
    if not _is_split_pair(value):
        raise ValueError("team split must be 3-3, 4-2, or 5-1")
    high = _split_number(value[0])
    low = _split_number(value[1])
    assert high is not None and low is not None
    roster = SHOWDOWN_ROSTER_SIZE
    if high < low or low < 0 or high + low != roster:
        raise ValueError(
            f"team split must be two counts that add up to {roster}, larger first"
        )
    if high > 5:
        raise ValueError(
            "DraftKings Showdown allows at most 5 players from one team"
        )
    return (high, low)


def canonical_team_splits(value: object) -> tuple[tuple[int, int], ...]:
    """Allowed splits, in 3-3, 4-2, 5-1 order.

    ``None``, ``Any``, an empty selection, and a selection that includes Any
    add no constraint. A single ``(4, 2)`` pair is the shape older overrides
    saved. A sequence of pairs or labels is the multi-select shape.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text or text == TEAM_SPLIT_ANY:
            return ()
        return (_parse_split_pair(text),)
    if _is_split_pair(value):
        return (_parse_split_pair(value),)
    if isinstance(value, (list, tuple)):
        if any(isinstance(item, str) and item.strip() == TEAM_SPLIT_ANY for item in value):
            return ()
        found: list[tuple[int, int]] = []
        for item in value:
            if item is None or (isinstance(item, str) and not str(item).strip()):
                continue
            pair = _parse_split_pair(item)
            if pair not in found:
                found.append(pair)
        order = {pair: index for index, pair in enumerate(_TEAM_SPLIT_ORDER)}
        found.sort(key=lambda pair: order.get(pair, pair[0]))
        return tuple(found)
    raise ValueError("team split must be 3-3, 4-2, or 5-1")


def stored_team_split(value: object) -> Optional[tuple]:
    """Value stored on the optimizer and in an override dict.

    One split stays a single pair, so a saved ``(4, 2)`` still reads back as
    ``(4, 2)``. Several splits stay a tuple of pairs.
    """
    splits = canonical_team_splits(value)
    if not splits:
        return None
    if len(splits) == 1:
        return splits[0]
    return splits


def prepare_team_split_widget(stored: object) -> list[str]:
    """Multi-select session value.

    The control used to store one string, such as ``"4-2"`` or ``"Any"``.
    A string becomes a one-item list. An empty list stays empty. Anything
    unrecognized becomes Any.
    """
    if isinstance(stored, (list, tuple)):
        chosen: list[str] = []
        for item in stored:
            text = str(item).strip()
            if text in TEAM_SPLIT_OPTIONS and text not in chosen:
                chosen.append(text)
        return [option for option in TEAM_SPLIT_OPTIONS if option in chosen]
    if isinstance(stored, str):
        text = stored.strip()
        if text in TEAM_SPLIT_OPTIONS:
            return [text]
    if stored is None:
        return [TEAM_SPLIT_ANY]
    return [TEAM_SPLIT_ANY]


def reconcile_team_split_selection(current: object, previous: object) -> list[str]:
    """Any does not stay selected next to a split.

    The option that was just added wins. Selecting Any clears the splits.
    Selecting a split while Any is on clears Any. Nothing selected stays
    empty, which is the same as no constraint.
    """
    if current is None:
        selected: list[str] = []
    else:
        selected = prepare_team_split_widget(current)
    if TEAM_SPLIT_ANY not in selected or selected == [TEAM_SPLIT_ANY]:
        return selected
    previous_labels = [] if previous is None else prepare_team_split_widget(previous)
    if TEAM_SPLIT_ANY not in previous_labels:
        return [TEAM_SPLIT_ANY]
    return [option for option in selected if option != TEAM_SPLIT_ANY]


def team_split_needs_side(team_split: object) -> bool:
    """4-2 and 5-1 have a heavy side. 3-3 alone does not."""
    return any(high != low for high, low in canonical_team_splits(team_split))


def team_split_setting_label(team_split: object, team: Optional[str] = None) -> Optional[str]:
    """Settings-summary text. ``None`` when the split is open.

    One uneven split keeps the older wording, ``4-2, BUF``. Several splits
    read ``4-2 or 5-1, BUF heavy``.
    """
    splits = canonical_team_splits(team_split)
    if not splits:
        return None
    body = " or ".join(f"{high}-{low}" for high, low in splits)
    if all(high == low for high, low in splits):
        return body
    side = str(team).strip().upper() if team else ""
    if len(splits) == 1:
        return f"{body}, {side or 'either team'}"
    return f"{body}, {side + ' heavy' if side else 'either team'}"


def _join_and(parts: Sequence[str]) -> str:
    items = [str(part) for part in parts]
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def team_split_cause(
    enabled: bool,
    minimum: int,
    team_split: object = None,
) -> Optional[str]:
    """Name the minimum when every selected split puts fewer than that on one team.

    A split that still fits can be used, so selecting 3-3 and 4-2 with a
    minimum of 3 is not this error. One saved pair keeps the original sentence.
    """
    if not enabled:
        return None
    splits = canonical_team_splits(team_split)
    if not splits:
        return None
    floor = int(minimum)
    if any(low >= floor for _high, low in splits):
        return None
    if len(splits) == 1:
        high, low = splits[0]
        return (
            f"{RULE_MIN_PER_TEAM}. The minimum is {floor} from each team, "
            f"and the team split {high}-{low} puts {low} on one team. "
            "Neither setting was changed."
        )
    labels = " or ".join(f"{high}-{low}" for high, low in splits)
    lows = _join_and([str(low) for _high, low in splits])
    return (
        f"{RULE_MIN_PER_TEAM}. The minimum is {floor} from each team, "
        f"and the team splits {labels} put {lows} on one team. "
        "Neither setting was changed."
    )


def _split_fits_pool(
    counts: Mapping[str, int],
    high: int,
    low: int,
    heavy: str,
) -> bool:
    teams = list(counts)
    if len(teams) != 2:
        return False
    if high == low:
        return all(counts[team] >= high for team in teams)
    if heavy:
        if heavy not in counts:
            return False
        other = next(team for team in teams if team != heavy)
        return counts[heavy] >= high and counts[other] >= low
    first, second = teams
    return (counts[first] >= high and counts[second] >= low) or (
        counts[second] >= high and counts[first] >= low
    )


def team_split_supply_cause(
    pool: Optional[pd.DataFrame],
    team_split: object,
    heavy: Optional[str] = None,
) -> Optional[str]:
    """Name a split the two teams do not have the players to build.

    CPT and FLEX of one person count once. A heavy team that is not on the
    slate is left to the optimizer's own message.
    """
    splits = canonical_team_splits(team_split)
    if not splits:
        return None
    counts = distinct_players_by_team(pool)
    if len(counts) != 2:
        return None
    heavy_key = str(heavy).strip().upper() if heavy else ""
    if heavy_key and heavy_key not in counts and any(high != low for high, low in splits):
        return None
    if any(_split_fits_pool(counts, high, low, heavy_key) for high, low in splits):
        return None
    label = team_split_setting_label(splits, heavy_key or None)
    inventory = _join_and(
        [f"{team} has {count}" for team, count in sorted(counts.items())]
    )
    return (
        f"{RULE_TEAM_SPLIT}. {label} cannot be built from this pool. "
        f"{inventory} different players. The team split was not changed."
    )


def team_split_block_message(team_split: object, team: Optional[str] = None) -> str:
    """The selected splits are infeasible once the other lineup rules are on."""
    label = team_split_setting_label(team_split, team) or "The team split"
    return (
        f"{RULE_TEAM_SPLIT}. {label} cannot be built with the other lineup rules. "
        "The team split was not changed."
    )


def distinct_players_by_team(pool: Optional[pd.DataFrame]) -> dict[str, int]:
    """Unique people per team. CPT and FLEX rows of one person count once."""
    if pool is None or getattr(pool, "empty", True):
        return {}
    team_col = "TeamAbbrev" if "TeamAbbrev" in pool.columns else "Team"
    if team_col not in pool.columns:
        return {}
    people: dict[str, set[tuple[str, str]]] = {}
    for _, row in pool.iterrows():
        club = str(row.get(team_col) or "").strip().upper()
        if not club or club == "NAN":
            continue
        key = person_key(row.get("Name"), club, row.get("Position"))
        if not key[0]:
            continue
        people.setdefault(club, set()).add(key)
    return {club: len(keys) for club, keys in people.items()}


def roster_supply_cause(pool: Optional[pd.DataFrame], minimum: int) -> Optional[str]:
    """A team that does not have enough different players for the minimum."""
    floor = int(minimum)
    short = [
        f"{club} has {count}"
        for club, count in sorted(distinct_players_by_team(pool).items())
        if count < floor
    ]
    if not short:
        return None
    return (
        f"{RULE_MIN_PER_TEAM}. {', '.join(short)} different players, and the "
        f"minimum is {floor} from each team. The minimum was not lowered."
    )


def minimum_spend(cap: int, max_unused: int) -> int:
    """Smallest legal spend: the slate cap minus the maximum unused salary."""
    return int(cap) - int(max_unused)


def unused_salary(spent: int, cap: int) -> int:
    return int(cap) - int(spent)


def salary_floor_met(spent: int, cap: int, max_unused: int) -> bool:
    """Unused salary at the maximum passes. One dollar more fails."""
    return unused_salary(spent, cap) <= int(max_unused)


def lineup_salary(lineup: pd.DataFrame) -> int:
    """Roster spend. CPT rows are already stored at 1.5x the flex salary."""
    if lineup is None or getattr(lineup, "empty", True) or "Salary" not in lineup.columns:
        return 0
    salaries = pd.to_numeric(lineup["Salary"], errors="coerce").fillna(0)
    return int(salaries.sum())


def slate_salary_cap(pool: Optional[pd.DataFrame]) -> int:
    """The cap ``upload_check.salary_cap_for`` uses for this Showdown slate."""
    from .presets import ContestFormat
    from .upload_check import salary_cap_for

    return int(salary_cap_for(ContestFormat.SHOWDOWN, pool))


def resolve_salary_bounds(
    *,
    slate_cap: int,
    window_min: Optional[int],
    window_max: Optional[int],
    salary_floor: bool,
    max_unused: int,
) -> SalaryBounds:
    """Intersect the salary window with the unused-salary floor.

    Both are spend limits. The lineup has to clear the higher minimum and
    stay under the lower maximum. When that range is empty, ``conflict``
    names both settings and neither one is dropped. When the window minimum
    is already higher than the floor, the note says the window is the spend
    floor so the two do not look like they disagree.
    """
    cap = int(slate_cap)
    ceiling = cap if window_max is None else min(cap, int(window_max))
    floor_spend = minimum_spend(cap, max_unused) if salary_floor else None
    floors: list[int] = []
    if window_min is not None:
        floors.append(int(window_min))
    if floor_spend is not None:
        floors.append(int(floor_spend))
    minimum = max(floors) if floors else None
    conflict = _salary_conflict(
        cap=cap,
        ceiling=ceiling,
        window_min=None if window_min is None else int(window_min),
        window_max=None if window_max is None else int(window_max),
        salary_floor=bool(salary_floor),
        max_unused=int(max_unused),
        floor_spend=floor_spend,
        minimum=minimum,
    )
    note = ""
    if conflict is None:
        note = _salary_note(
            cap=cap,
            window_min=None if window_min is None else int(window_min),
            window_max=None if window_max is None else int(window_max),
            salary_floor=bool(salary_floor),
            max_unused=int(max_unused),
            floor_spend=floor_spend,
            minimum=minimum,
        )
    return SalaryBounds(
        slate_cap=cap,
        minimum=None if conflict else minimum,
        maximum=ceiling,
        floor_spend=floor_spend,
        conflict=conflict,
        note=note,
    )


def _money(value: int) -> str:
    return f"${int(value):,}"


def _salary_conflict(
    *,
    cap: int,
    ceiling: int,
    window_min: Optional[int],
    window_max: Optional[int],
    salary_floor: bool,
    max_unused: int,
    floor_spend: Optional[int],
    minimum: Optional[int],
) -> Optional[str]:
    if not salary_floor or floor_spend is None or minimum is None or minimum <= ceiling:
        return None
    window = _window_phrase(window_min, window_max, ceiling)
    if floor_spend > ceiling:
        return (
            f"{RULE_SALARY_FLOOR}. Lineups must spend at least {_money(floor_spend)} "
            f"({_money(cap)} cap minus {_money(max_unused)} unused) and {window}. "
            "Neither setting was changed."
        )
    # The window minimum, not the floor, is what clears the ceiling.
    return (
        f"{RULE_SALARY_FLOOR}. The salary window minimum of {_money(int(window_min or minimum))} "
        f"is above {window}. The salary floor of {_money(floor_spend)} was not used to lower it."
    )


def _window_phrase(
    window_min: Optional[int],
    window_max: Optional[int],
    ceiling: int,
) -> str:
    if window_min is None and window_max is None:
        return f"the salary cap is {_money(ceiling)}"
    parts: list[str] = []
    if window_min is not None:
        parts.append(f"minimum {_money(window_min)}")
    if window_max is not None:
        parts.append(f"maximum {_money(window_max)}")
    return "the salary window has a " + " and a ".join(parts)


def _salary_note(
    *,
    cap: int,
    window_min: Optional[int],
    window_max: Optional[int],
    salary_floor: bool,
    max_unused: int,
    floor_spend: Optional[int],
    minimum: Optional[int],
) -> str:
    if not salary_floor or floor_spend is None:
        return ""
    if window_min is not None and int(window_min) > int(floor_spend):
        return (
            f"At most {_money(max_unused)} unused ({_money(floor_spend)} on a {_money(cap)} cap). "
            f"The salary window minimum of {_money(window_min)} is higher, "
            f"so lineups spend at least {_money(window_min)}."
        )
    if window_min is not None and int(window_min) < int(floor_spend):
        return (
            f"At most {_money(max_unused)} unused. "
            f"This raises the salary window minimum from {_money(window_min)} to {_money(floor_spend)}."
        )
    if window_max is not None and int(window_max) < cap:
        return (
            f"At most {_money(max_unused)} unused (spend at least {_money(floor_spend)}). "
            f"The salary window maximum is {_money(window_max)}."
        )
    if minimum is None:
        return ""
    return (
        f"At most {_money(max_unused)} unused "
        f"(spend at least {_money(minimum)} of the {_money(cap)} cap)."
    )


def captain_rule_label(enabled: bool) -> str:
    if enabled:
        return "On — K and DST can be FLEX, never CPT"
    return "Off"


def team_rule_label(enabled: bool, count: int) -> str:
    if not enabled:
        return "Off — at least 1 from each team"
    return f"{int(count)} from each team"


def salary_floor_label(
    enabled: bool,
    max_unused: int,
    window_min: Optional[int],
    window_max: Optional[int],
    *,
    cap: int = DEFAULT_SHOWDOWN_CAP,
) -> str:
    """Preset-summary text. Uses the standard cap when no slate is loaded."""
    if not enabled:
        return "Off"
    bounds = resolve_salary_bounds(
        slate_cap=cap,
        window_min=window_min,
        window_max=window_max,
        salary_floor=True,
        max_unused=max_unused,
    )
    if bounds.conflict:
        return bounds.conflict
    return bounds.note or f"At most {_money(max_unused)} unused"


def captain_violation(lineup: pd.DataFrame) -> Optional[str]:
    """A K or DST in the CPT slot, or None."""
    if lineup is None or getattr(lineup, "empty", True) or "Slot" not in lineup.columns:
        return None
    slot = lineup["Slot"].astype(str).str.strip().str.upper()
    captains = lineup.loc[slot.eq("CPT")]
    for _, row in captains.iterrows():
        if not is_kicker_or_dst(row.get("Position")):
            continue
        label = _player_label(row.get("Name"), row.get("Team", row.get("TeamAbbrev")))
        return (
            f"{RULE_NO_K_DST_CAPTAIN}. {label} is CPT. "
            "K and DST can be FLEX and cannot be CPT."
        )
    return None


class MinPlayersPerTeamRule(OptimizerRule):
    """At least N roster spots from each team.

    A player group cannot do this. pydfs lets a player sit in only one
    minimum group, and Showdown already uses those groups for game script
    and the starting core. This constraint only counts spots. CPT and FLEX
    are separate rows, and the person rule already keeps a player out of both.
    """

    def apply(self, solver):
        minimum = int(getattr(self.optimizer, "_showdown_min_per_team", 0) or 0)
        if minimum <= 0:
            return
        grouped: dict[str, list] = {}
        for player, variable in self.players_dict.items():
            team = str(getattr(player, "team", "") or "").strip().upper()
            if team:
                grouped.setdefault(team, []).append(variable)
        for team, variables in grouped.items():
            solver.add_constraint(
                variables,
                None,
                SolverSign.GTE,
                minimum,
                name=f"showdown_min_team_{team}",
            )


def _anchor_split_count(anchor: str, high: int, low: int, heavy: Optional[str]) -> int:
    """Players from the anchor team when this split is the one that is used."""
    if high == low or heavy is None or heavy == anchor:
        return high
    return low


class AllowedTeamSplitsRule(OptimizerRule):
    """A lineup matches any one of the selected Showdown splits.

    ``_showdown_allowed_splits`` on the pydfs optimizer is
    ``(anchor_team, splits, heavy_team)``. ``splits`` is ``(high, low)``
    pairs. ``heavy_team`` takes the larger side of an uneven split, or is
    None so either team can. One binary marks which split is used. An
    uneven split with no named team also has a side binary, the same choice
    the single-split rule makes. Exactly one split binary is 1, so lineups
    are not divided evenly across the selected splits.
    """

    def apply(self, solver):
        spec = getattr(self.optimizer, "_showdown_allowed_splits", None)
        if not spec:
            return
        anchor, splits, heavy = spec
        variables = [
            variable
            for player, variable in self.players_dict.items()
            if player.team == anchor
        ]
        if not variables:
            return
        indicators = []
        count_vars = list(variables)
        count_coef = [1.0] * len(variables)
        for high, low in splits:
            indicator = solver.add_variable(f"showdown_team_split_{int(high)}_{int(low)}")
            indicators.append(indicator)
            if heavy is None and int(high) != int(low):
                side = solver.add_variable(
                    f"showdown_team_split_side_{int(high)}_{int(low)}"
                )
                solver.add_constraint(
                    [side, indicator],
                    [1.0, -1.0],
                    SolverSign.LTE,
                    0,
                    name=f"showdown_team_split_side_on_{int(high)}_{int(low)}",
                )
                count_vars.extend([indicator, side])
                count_coef.extend([-float(low), -float(int(high) - int(low))])
            else:
                target = _anchor_split_count(str(anchor), int(high), int(low), heavy)
                count_vars.append(indicator)
                count_coef.append(-float(target))
        solver.add_constraint(
            indicators,
            [1.0] * len(indicators),
            SolverSign.EQ,
            1,
            name="showdown_team_split_one",
        )
        solver.add_constraint(
            count_vars,
            count_coef,
            SolverSign.EQ,
            0,
            name="showdown_team_split_count",
        )


def assess_showdown_lineup(
    lineup: pd.DataFrame,
    settings: ShowdownRuleSettings,
    *,
    cap: int,
    teams: Sequence[str],
) -> tuple[str, ...]:
    """Failures for the rules that are on. An empty tuple is a legal lineup."""
    failures: list[str] = []
    if settings.ban_k_dst_captain:
        captain = captain_violation(lineup)
        if captain:
            failures.append(captain)
    if settings.min_players_per_team_enabled:
        cause = team_minimum_cause(
            team_player_counts(lineup),
            teams,
            settings.min_players_per_team,
        )
        if cause:
            failures.append(cause)
    if settings.salary_floor:
        spent = lineup_salary(lineup)
        if not salary_floor_met(spent, cap, settings.max_unused_salary):
            unused = unused_salary(spent, cap)
            failures.append(
                f"{RULE_SALARY_FLOOR}. Unused salary is {_money(unused)}, "
                f"above the maximum of {_money(settings.max_unused_salary)} "
                f"on the {_money(cap)} cap."
            )
    return tuple(failures)
