"""Portfolio limits for one generated lineup set.

pydfs-lineup-optimizer applies exposure and overlap inside a single
``optimize()`` call, and it keys both by solver variable. A Showdown captain
row and flex row are different variables (different DraftKings ids and
different position labels). A second salary tier or game-script bucket is a
second call, so its counters start over.

This module is the portfolio layer around that solver:

* Identity is ``person_key(name, team)`` from ``names``. That is the key the
  editor already uses to collapse captain and flex, locks, includes, and
  captain share. Classic has one salary row per player, so the key matches the
  DraftKings id. Showdown captain and flex share it. A lineup rosters a person
  once because the optimizer's person-uniqueness rule groups by this key.
  pydfs ``UniquePlayerRule`` groups by full name only and is not used.
* Overlap is the size of the shared identity set for every pair in the final
  export, including pairs that came from different tiers, presets' internal
  batches, or game-script buckets.
* Player exposure and team-stack exposure allow ``floor(rate * requested)``
  appearances. A small epsilon keeps a whole number such as ``0.5 * 4`` from
  being stored as ``1.999999999``. A positive rate that floors to 0 allows no
  appearances: one lineup would already be more than the cap, and the shortfall
  message names that cap. pydfs ``TotalExposureStrategy`` allows one more
  appearance whenever ``rate * n`` is not an integer, and it ignores a rate of
  0. This layer does not use that rounding. ``set_max_repeating_players``
  cannot express an overlap of zero (it rejects values below 1), so overlap of
  zero is enforced here.
* Team-stack exposure, when the Classic solver has one, is the existing
  ``PositionsStack`` (same positions). A later batch passes
  ``max_exposure_per_team`` so that strategy allows only the uses still left
  in the portfolio. A team with nothing left has its quarterback removed.
  That matches the solver's own stack ban, which cuts the quarterback group
  to zero once the stack's exposure is reached. The fraction
  ``(remaining - 0.5) / batch`` is used because ``remaining / batch`` can
  round up and allow one extra lineup. A rate of 0 is ignored by pydfs
  (``if not max_exposure``), so a spent team is removed instead of given a
  zero rate.

* Min exposure uses the same identity and the same requested-lineup basis.
  A player must appear in ``ceil(rate * requested - 1e-9)`` lineups. Showdown
  captain and flex count once. pydfs ``MinExposureRule`` instead requires
  ``round(rate * batch)`` appearances of each solver row inside one
  ``optimize()`` call, so a split portfolio (or a half that rounds to even)
  can finish below the requested minimum. This layer does not use that
  rounding.

  The solver reserves a player for the lineups still ahead: a player is
  forced into the current lineup only when the appearances still needed are
  at least the lineups still to build, including later tiers and game-script
  buckets. A position group is forced only for the overflow that cannot fit
  in the lineups after this one. That avoids spending every early lineup on
  the first minimum and then missing a later one. It is not a joint search
  over the whole portfolio. Shared flex slots, salary windows, stacks,
  overlap, and locks can still make a reserved set impossible. When that
  happens the minimum is dropped for the rest of that batch, the legal
  lineups are kept, and the unmet names are reported. A max is never raised
  to satisfy a min.

The solver does not name a unique infeasible constraint. Shortfall text
reports a limit as binding only when the roster math shows that limit is
enough to block another lineup. Everything else is labeled as a candidate.
Unmet minimums are reported on their own, not guessed as that cause.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from pydfs_lineup_optimizer.rules import OptimizerRule
from pydfs_lineup_optimizer.solvers.constants import SolverSign

from .names import person_key as _person_key
from .names import player_person_key

Identity = Tuple[str, str]


def person_key(name: object, team: object, position: object = None) -> Identity:
    """Same identity the player pool uses for captain and flex.

    The implementation lives in ``names.person_key`` so the editor, the
    Sleeper join, and this portfolio cannot drift. Tests lock the two
    wrappers together. ``position`` selects the defense-by-team key.
    """
    return _person_key(name, team, position)


def appearances_allowed(rate: Optional[float], total: int) -> int:
    """How many lineups one player or stack may occupy.

    ``floor(rate * total)`` after adding ``1e-9``, so ``0.5 * 4`` stays 2 and
    ``0.4 * 6`` stays 2. ``None`` means no cap. A non-positive rate allows
    zero. A positive rate whose product is below 1 also allows zero: the
    first lineup would already exceed the cap. ``total`` is the requested
    portfolio size.
    """
    total = int(total)
    if total <= 0:
        return 0
    if rate is None:
        return total
    if float(rate) <= 0.0:
        return 0
    allowed = math.floor(float(rate) * total + 1e-9)
    return max(0, min(total, allowed))


def appearances_needed(rate: Optional[float], total: int) -> int:
    """How many lineups one minimum requires across the requested set.

    ``ceil(rate * total - 1e-9)``, so ``0.5 * 5`` is 3 and ``0.5 * 4`` stays
    2. ``None`` and non-positive rates require nothing. A rate of 1 requires
    every requested lineup. ``total`` is the requested portfolio size.
    """
    total = int(total)
    if total <= 0 or rate is None or float(rate) <= 0.0:
        return 0
    needed = math.ceil(float(rate) * total - 1e-9)
    return max(0, min(total, needed))


@dataclass(frozen=True)
class RosterSlot:
    """One roster seat and the football positions that can sit there."""

    label: str
    positions: frozenset[str]


def classic_roster_slots(
    flex_positions: Optional[Iterable[str]] = None,
) -> Tuple[RosterSlot, ...]:
    """Classic seats. Flex defaults to RB, WR, and TE."""
    flex = frozenset(
        str(pos).strip().upper()
        for pos in (flex_positions or ("RB", "WR", "TE"))
        if str(pos).strip()
    )
    return (
        RosterSlot("QB", frozenset({"QB"})),
        RosterSlot("RB", frozenset({"RB"})),
        RosterSlot("RB", frozenset({"RB"})),
        RosterSlot("WR", frozenset({"WR"})),
        RosterSlot("WR", frozenset({"WR"})),
        RosterSlot("WR", frozenset({"WR"})),
        RosterSlot("TE", frozenset({"TE"})),
        RosterSlot("FLEX", flex or frozenset({"RB", "WR", "TE"})),
        RosterSlot("DST", frozenset({"DST"})),
    )


def showdown_roster_slots(
    captain_positions: Iterable[str],
) -> Tuple[RosterSlot, ...]:
    """One captain seat and five flex seats. Flex accepts every football spot."""
    captain = frozenset(
        str(pos).strip().upper() for pos in captain_positions if str(pos).strip()
    )
    flex = frozenset({"QB", "RB", "WR", "TE", "K", "DST"})
    return (
        RosterSlot("CPT", captain or flex),
        *(RosterSlot("FLEX", flex) for _ in range(5)),
    )


def _usable_slots(
    player_positions: Iterable[str],
    slots: Sequence[RosterSlot],
) -> frozenset[int]:
    owned = {str(pos).strip().upper() for pos in player_positions if str(pos).strip()}
    if not owned:
        return frozenset()
    return frozenset(
        index
        for index, slot in enumerate(slots)
        if owned & set(slot.positions)
    )


def _slot_phrase(slots: Sequence[RosterSlot], mask: int) -> str:
    labels = [slots[index].label for index in range(len(slots)) if mask & (1 << index)]
    counts = Counter(labels)
    parts: List[str] = []
    for label in dict.fromkeys(labels):
        count = counts[label]
        parts.append(label if count == 1 else f"{count} {label}")
    if not parts:
        return "no roster slots"
    if len(labels) == 1:
        return f"the {labels[0]} slot"
    joined = ", ".join(parts)
    total = len(labels)
    if total == len(slots):
        return f"{total} roster slots"
    return f"{joined} ({total} slots)"


def min_exposure_conflicts(
    *,
    needed: Mapping[Identity, int],
    allowed: Mapping[Identity, Optional[int]],
    positions: Mapping[Identity, Iterable[str]],
    slots: Sequence[RosterSlot],
    requested: int,
    names: Mapping[Identity, str],
    teams: Optional[Mapping[Identity, str]] = None,
) -> Tuple[str, ...]:
    """Name minimums that cannot fit before the solver runs.

    A player's minimum above that player's max is one conflict. The sum of
    minimums is another when those players can only sit in a set of seats
    whose count, times the requested lineups, is smaller. Players with no
    known position are left out of the seat check so a missing label is not
    reported as an empty roster.
    """
    requested = int(requested)
    messages: List[str] = []
    messages.extend(
        _min_above_max_messages(
            needed=needed,
            allowed=allowed,
            requested=requested,
            names=names,
            teams=teams or {},
        )
    )
    if requested <= 0 or not slots:
        return tuple(messages)
    messages.extend(
        _slot_capacity_messages(
            needed=needed,
            positions=positions,
            slots=slots,
            requested=requested,
        )
    )
    return tuple(messages)


def _player_label(
    key: Identity,
    names: Mapping[Identity, str],
    teams: Mapping[Identity, str],
) -> str:
    name = names.get(key) or key[0]
    team = teams.get(key) or key[1]
    if team:
        return f"{name} ({team})"
    return str(name)


def _min_above_max_messages(
    *,
    needed: Mapping[Identity, int],
    allowed: Mapping[Identity, Optional[int]],
    requested: int,
    names: Mapping[Identity, str],
    teams: Mapping[Identity, str],
) -> List[str]:
    bad: List[Tuple[Identity, int, int]] = []
    for key, need in needed.items():
        cap = allowed.get(key)
        if cap is None:
            continue
        if int(need) > int(cap):
            bad.append((key, int(need), int(cap)))
    if not bad:
        return []
    same_need = {need for _, need, _ in bad}
    same_cap = {cap for _, _, cap in bad}
    if len(bad) > 3 and len(same_need) == 1 and len(same_cap) == 1:
        need = next(iter(same_need))
        cap = next(iter(same_cap))
        return [
            f"Min exposure needs {need} appearances in {requested} requested lineups "
            f"but max exposure allows {cap}. That minimum is above the max for "
            f"{len(bad)} players."
        ]
    messages = [
        f"{_player_label(key, names, teams)} min exposure needs {need} appearances "
        f"but max exposure allows {cap}."
        for key, need, cap in bad[:6]
    ]
    if len(bad) > 6:
        messages.append(f"And {len(bad) - 6} more players have a minimum above their max.")
    return messages


def _slot_capacity_messages(
    *,
    needed: Mapping[Identity, int],
    positions: Mapping[Identity, Iterable[str]],
    slots: Sequence[RosterSlot],
    requested: int,
) -> List[str]:
    usable: Dict[Identity, frozenset[int]] = {}
    for key, need in needed.items():
        if int(need) <= 0:
            continue
        seats = _usable_slots(positions.get(key) or (), slots)
        if not seats:
            continue
        usable[key] = seats
    if not usable:
        return []
    width = len(slots)
    failures: List[Tuple[int, int, int, int, int]] = []
    full_mask = (1 << width) - 1
    exact_masks = {_seat_mask(seats) for seats in usable.values() if seats}
    for mask in range(1, full_mask + 1):
        demand = sum(
            int(needed[key])
            for key, seats in usable.items()
            if seats and seats.issubset(_mask_indexes(mask))
        )
        capacity = bin(mask).count("1") * requested
        if demand > capacity:
            failures.append((demand - capacity, bin(mask).count("1"), mask, demand, capacity))
    if not failures:
        return []
    messages: List[str] = []
    for _, _, mask, demand, capacity in failures:
        if mask != full_mask:
            continue
        messages.append(
            f"Min exposure needs {demand} appearances but {requested} lineups "
            f"have {capacity} roster slots."
        )
    # Only the seat sets players actually have. Combined masks restate the same limit.
    specific = [item for item in failures if item[2] in exact_masks and item[2] != full_mask]
    specific.sort(key=lambda item: (item[1], -item[0]))
    for _, _, mask, demand, capacity in specific[:3]:
        messages.append(
            "Min exposure needs "
            f"{demand} appearances from players who can only fill {_slot_phrase(slots, mask)}, "
            f"but {requested} lineups have {capacity} of those slots."
        )
    return messages


def _seat_mask(seats: Iterable[int]) -> int:
    mask = 0
    for index in seats:
        mask |= 1 << int(index)
    return mask


def _mask_indexes(mask: int) -> frozenset[int]:
    return frozenset(index for index in range(mask.bit_length()) if mask & (1 << index))


def min_force_groups(
    positions: Mapping[Identity, Iterable[str]],
    slots: Sequence[RosterSlot],
    extra_caps: Sequence[Tuple[Iterable[str], int]] = (),
) -> Tuple[Tuple[Tuple[Identity, ...], int], ...]:
    """Position groups whose remaining need can overflow a later batch.

    Each group is the players who can only sit in seats that accept that
    position set, plus the number of those seats in one lineup. Extra caps
    (a kicker limit, a defense limit) tighten a position the seats alone
    would leave open.
    """
    normalized = {
        key: frozenset(str(pos).strip().upper() for pos in (owned or ()) if str(pos).strip())
        for key, owned in positions.items()
    }
    tokens: set[str] = set()
    for owned in normalized.values():
        tokens |= set(owned)
    position_sets: List[frozenset[str]] = [frozenset({token}) for token in sorted(tokens)]
    skill = frozenset({"RB", "WR", "TE"}) & tokens
    if len(skill) >= 2:
        position_sets.append(skill)
        ordered = tuple(sorted(skill))
        for left in range(len(ordered)):
            for right in range(left + 1, len(ordered)):
                position_sets.append(frozenset({ordered[left], ordered[right]}))
    if tokens:
        position_sets.append(frozenset(tokens))
    groups: List[Tuple[Tuple[Identity, ...], int]] = []
    seen: set[frozenset[Identity]] = set()
    for position_set in position_sets:
        members = tuple(
            key
            for key, owned in normalized.items()
            if owned and owned <= position_set
        )
        member_set = frozenset(members)
        if not member_set or member_set in seen:
            continue
        if position_set >= tokens and tokens:
            capacity = len(slots)
        else:
            capacity = sum(1 for slot in slots if set(slot.positions) & set(position_set))
        if capacity <= 0:
            continue
        seen.add(member_set)
        groups.append((members, capacity))
    for token_set, cap in extra_caps:
        limited = frozenset(str(pos).strip().upper() for pos in token_set if str(pos).strip())
        if not limited or int(cap) < 0:
            continue
        members = tuple(
            key for key, owned in normalized.items() if owned & limited
        )
        if not members:
            continue
        groups.append((members, int(cap)))
    return tuple(groups)


def position_tokens(value: object) -> set[str]:
    """Football positions from a cell. Captain and flex labels are not positions."""
    tokens: set[str] = set()
    for part in str(value or "").split("/"):
        token = part.strip().upper()
        if token and token not in {"FLEX", "CPT", "UTIL"}:
            tokens.add(token)
    return tokens


def lineup_identities(lineup: pd.DataFrame) -> List[Identity]:
    """Real players in one lineup, once each, in roster order."""
    if lineup is None or lineup.empty or "Name" not in lineup.columns:
        return []
    team_col = "Team" if "Team" in lineup.columns else "TeamAbbrev"
    found: List[Identity] = []
    seen: set[Identity] = set()
    teams = lineup[team_col] if team_col in lineup.columns else [""] * len(lineup)
    positions = lineup["Position"] if "Position" in lineup.columns else [None] * len(lineup)
    for name, team, position in zip(lineup["Name"], teams, positions):
        key = person_key(name, team, position)
        if key in seen or not key[0]:
            continue
        seen.add(key)
        found.append(key)
    return found


def pairwise_overlap(lineups: Sequence[pd.DataFrame]) -> int:
    """Largest number of real players shared by any two lineups."""
    sets = [set(lineup_identities(lineup)) for lineup in lineups]
    largest = 0
    for index, left in enumerate(sets):
        for right in sets[index + 1 :]:
            largest = max(largest, len(left & right))
    return largest


def exposure_counts(lineups: Sequence[pd.DataFrame]) -> Counter[Identity]:
    """Lineup appearances per real player. Captain and flex count once."""
    counts: Counter[Identity] = Counter()
    for lineup in lineups:
        counts.update(lineup_identities(lineup))
    return counts


def normalize_stack_spec(positions: Sequence[object]) -> Tuple[Tuple[str, ...], ...]:
    """Position groups from ``PositionsStack``, one tuple per stacked spot."""
    spec: List[Tuple[str, ...]] = []
    for position in positions:
        if isinstance(position, str):
            spec.append((position.strip().upper(),))
        else:
            spec.append(tuple(str(part).strip().upper() for part in position))
    return tuple(spec)


def meets_positions_stack(
    token_sets: Sequence[set[str]],
    spec: Sequence[Tuple[str, ...]],
) -> bool:
    """Whether these players satisfy one team's ``PositionsStack``.

    Each spot is a count of players whose positions intersect that spot, and
    the union of the spots must be filled by ``len(spec)`` distinct players.
    A wide receiver counts toward both a WR spot and a WR/TE spot. That is how
    pydfs builds the nested group.
    """
    if not spec:
        return False
    counts: Dict[Tuple[str, ...], int] = {}
    union: set[str] = set()
    for group in spec:
        key = tuple(group)
        counts[key] = counts.get(key, 0) + 1
        union.update(group)
    counts[tuple(sorted(union))] = len(spec)
    for group, need in counts.items():
        wanted = set(group)
        have = sum(1 for tokens in token_sets if tokens & wanted)
        if have < need:
            return False
    return True


def teams_stacked(
    lineup: pd.DataFrame,
    spec: Sequence[Tuple[str, ...]],
) -> List[str]:
    """Teams whose rostered players form the configured stack."""
    if not spec or lineup is None or lineup.empty or "Team" not in lineup.columns:
        return []
    position_col = "Position" if "Position" in lineup.columns else None
    if position_col is None:
        return []
    found: List[str] = []
    teams = lineup["Team"].astype(str)
    for team, group in lineup.groupby(teams, sort=False):
        tokens = [position_tokens(pos) for pos in group[position_col]]
        if meets_positions_stack(tokens, spec):
            found.append(str(team))
    return found


def team_stack_counts(
    lineups: Sequence[pd.DataFrame],
    spec: Sequence[Tuple[str, ...]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for lineup in lineups:
        counts.update(teams_stacked(lineup, spec))
    return counts


def solver_rate_for_remaining(remaining: int, batch_size: int) -> float:
    """pydfs exposure fraction that allows at most ``remaining`` uses.

    ``TotalExposureStrategy`` allows another lineup while ``rate > used / batch``.
    ``remaining / batch`` can round up and allow one extra. ``(remaining - 0.5) /
    batch`` allows exactly ``remaining``. A zero return does not ban by itself,
    because pydfs treats a zero rate as uncapped; the caller removes the player
    or adds a hard constraint.
    """
    if remaining <= 0 or batch_size <= 0:
        return 0.0
    if remaining >= batch_size:
        return 1.0
    return (int(remaining) - 0.5) / float(batch_size)


def batch_team_stack_exposures(
    *,
    rate: float,
    portfolio_size: int,
    batch_size: int,
    prior_counts: Mapping[str, int],
    teams: Iterable[str],
) -> Tuple[Dict[str, float], List[str]]:
    """Per-team ``PositionsStack`` rates for one batch, plus teams to block.

    The returned rate is a pydfs ``TotalExposureStrategy`` fraction whose
    allowance is exactly the floor-count still left. ``(remaining - 0.5) /
    batch`` keeps that strategy from rounding up to one extra lineup. Teams
    with nothing left are listed separately; a zero rate would not ban them.
    """
    allowed = appearances_allowed(rate, portfolio_size)
    per_team: Dict[str, float] = {}
    blocked: List[str] = []
    batch_size = max(1, int(batch_size))
    for team in teams:
        remaining = allowed - int(prior_counts.get(str(team), 0))
        if remaining <= 0:
            blocked.append(str(team))
            continue
        per_team[str(team)] = solver_rate_for_remaining(remaining, batch_size)
    return per_team, blocked


@dataclass(frozen=True)
class PortfolioShortfall:
    """Why the export is shorter than the request.

    ``proven`` holds limits that are enough, on their own, to block another
    legal lineup. ``candidates`` are limits that are merely at capacity.
    """

    requested: int
    built: int
    message: str
    proven: Tuple[str, ...]
    candidates: Tuple[str, ...]


def explain_shortfall(
    lineups: Sequence[pd.DataFrame],
    *,
    requested: int,
    overlap_cap: Optional[int],
    max_exposure: Optional[float],
    stack_exposure: Optional[float] = None,
    stack_spec: Sequence[Tuple[str, ...]] = (),
    min_exposure: Optional[float] = None,
    eligible: Sequence[Identity] = (),
    roster_size: int = 9,
    locked: Sequence[Identity] = (),
    solver_constraints: Sequence[str] = (),
    display_names: Optional[Mapping[Identity, str]] = None,
) -> PortfolioShortfall:
    """Describe a short set without inventing a single binding cause."""
    requested = int(requested)
    built = len(lineups)
    names = dict(display_names or {})
    proven: List[str] = []
    candidates: List[str] = []
    eligible_set = set(eligible)
    counts = exposure_counts(lineups)

    if overlap_cap is not None:
        cap = int(overlap_cap)
        pairs_at_cap = _pairs_at_overlap(lineups, cap)
        used: set[Identity] = set()
        for lineup in lineups:
            used.update(lineup_identities(lineup))
        outside = eligible_set - used
        if cap <= 0 and len(outside) < int(roster_size):
            proven.append(
                f"Max overlap {cap} blocks another lineup: lineups may share "
                f"{cap} players, {len(outside)} eligible players are outside the "
                f"exported set, and a lineup needs {int(roster_size)}."
            )
        elif pairs_at_cap:
            candidates.append(
                f"Max overlap {cap}: {pairs_at_cap} exported pair(s) already share {cap} players."
            )

    if max_exposure is not None:
        allowed = appearances_allowed(max_exposure, requested)
        at_cap = [key for key, count in counts.items() if count >= allowed]
        under = [key for key in eligible_set if counts.get(key, 0) < allowed]
        percent = float(max_exposure) * 100
        if allowed <= 0 and eligible_set:
            proven.append(
                f"Max player exposure {percent:.0f}% allows 0 appearances in "
                f"{requested} requested lineups. The cap is below one lineup, "
                "so using a player would exceed it."
            )
        elif eligible_set and len(under) < int(roster_size):
            proven.append(
                f"Max player exposure {percent:.0f}% blocks another lineup: "
                f"a player may appear in {allowed} of {requested} requested lineups, "
                f"{len(under)} eligible players are still under that, and a lineup needs {int(roster_size)}."
            )
        elif at_cap:
            candidates.append(
                "Max player exposure "
                f"{float(max_exposure) * 100:.0f}%: {len(at_cap)} players are at "
                f"{allowed} of {requested} ({_name_list(at_cap, names)})."
            )
        for key in locked:
            # A lock is in every lineup. That conflicts with the cap once the
            # player is already at it, or when the cap allows none.
            if allowed < 1 or counts.get(key, 0) >= allowed:
                label = names.get(key) or key[0]
                proven.append(
                    f"Lock on {label} forces that player into every lineup, "
                    f"which the {float(max_exposure) * 100:.0f}% exposure cap "
                    f"({allowed} of {requested}) does not allow."
                )
                break

    if stack_exposure is not None and stack_spec:
        allowed = appearances_allowed(stack_exposure, requested)
        usage = team_stack_counts(lineups, stack_spec)
        at_cap = sorted(team for team, count in usage.items() if count >= allowed)
        if allowed <= 0:
            proven.append(
                f"Team stack exposure {float(stack_exposure) * 100:.0f}% allows 0 stacks "
                f"in {requested} requested lineups. The cap is below one lineup, "
                "so a team stack would exceed it."
            )
        elif at_cap:
            candidates.append(
                "Team stack exposure "
                f"{float(stack_exposure) * 100:.0f}%: {', '.join(at_cap)} "
                f"already stack in {allowed} of {requested} requested lineups."
            )

    # A lineup-count shortfall does not guess that a minimum caused it.
    # evaluate_min_exposure names anyone still under the requested minimum.
    _ = min_exposure

    notes = [str(name) for name in solver_constraints if str(name).strip()]
    message = _shortfall_message(requested, built, proven, candidates, notes)
    return PortfolioShortfall(
        requested=requested,
        built=built,
        message=message,
        proven=tuple(proven),
        candidates=tuple(candidates),
    )


def _pairs_at_overlap(lineups: Sequence[pd.DataFrame], cap: int) -> int:
    sets = [set(lineup_identities(lineup)) for lineup in lineups]
    pairs = 0
    for index, left in enumerate(sets):
        for right in sets[index + 1 :]:
            if len(left & right) >= cap:
                pairs += 1
    return pairs


def _name_list(keys: Sequence[Identity], names: Mapping[Identity, str]) -> str:
    labels = [names.get(key) or key[0] for key in list(keys)[:6]]
    extra = len(keys) - len(labels)
    text = ", ".join(labels)
    if extra > 0:
        text = f"{text}, and {extra} more"
    return text


def _shortfall_message(
    requested: int,
    built: int,
    proven: Sequence[str],
    candidates: Sequence[str],
    solver_constraints: Sequence[str],
) -> str:
    sentence = f"Requested {requested} lineups and built {built}."
    if len(proven) == 1 and not candidates:
        sentence += f" Binding limit: {proven[0]}"
    elif proven:
        sentence += " Limits sufficient to block another lineup: " + " ".join(proven)
        if candidates:
            sentence += " Also at capacity, not isolated as the sole cause: " + " ".join(candidates)
    elif candidates:
        sentence += (
            " The solver stopped without naming one binding constraint. "
            "Candidate limits at capacity: " + " ".join(candidates)
        )
    else:
        sentence += (
            " The solver stopped and did not identify which constraint blocked the next lineup."
        )
    if solver_constraints:
        sentence += (
            " Solver-reported constraint names (evidence only, not a proven cause): "
            + ", ".join(solver_constraints)
            + "."
        )
    return sentence


@dataclass
class PortfolioRuleState:
    """Mutable counters the portfolio rules read while pydfs solves one batch."""

    portfolio_size: int
    overlap_cap: Optional[int]
    enforce_overlap: bool
    default_allowed: Optional[int]
    allowance: Dict[Identity, int]
    prior_used: Dict[Identity, int]
    prior_history: List[set[Identity]]
    # Remaining appearances still required from this batch forward, after the
    # max has already capped anything a minimum is not allowed to exceed.
    min_remaining: Dict[Identity, int] = field(default_factory=dict)
    min_groups: Tuple[Tuple[Tuple[Identity, ...], int], ...] = ()
    min_after: int = 0
    min_batch: int = 0
    min_relax: bool = False
    # Lower priority is forced first when several players are equally behind.
    min_priority: Dict[Identity, float] = field(default_factory=dict)
    min_roster: int = 0


class PortfolioIdentityRule(OptimizerRule):
    """Cap identity exposure and overlap inside one ``optimize()`` call.

    pydfs copies the base solver for every lineup and discards the copy, so
    this rule re-adds every portfolio constraint on each iteration. Usage from
    earlier batches is seeded before the first lineup.
    """

    def __init__(self, optimizer, players_dict, context):
        super().__init__(optimizer, players_dict, context)
        state: PortfolioRuleState = optimizer._portfolio_limits
        self.overlap_cap = state.overlap_cap
        self.enforce_overlap = bool(state.enforce_overlap and state.overlap_cap is not None)
        self.default_allowed = state.default_allowed
        self.allowance = dict(state.allowance)
        self.used: Counter[Identity] = Counter(state.prior_used)
        self.history: List[set[Identity]] = [set(group) for group in state.prior_history]
        self.variables: Dict[Identity, list] = {}
        self.blocked: set[Identity] = set()
        for player, variable in players_dict.items():
            key = player_person_key(player)
            self.variables.setdefault(key, []).append(variable)
        for key in set(self.variables) | set(self.used):
            allowed = self.allowance.get(key, self.default_allowed)
            if allowed is not None and self.used.get(key, 0) >= allowed:
                self.blocked.add(key)

    def apply_for_iteration(self, solver, result):
        if result is not None:
            self._record(result)
        banned = []
        for key in self.blocked:
            banned.extend(self.variables.get(key, []))
        if banned:
            solver.add_constraint(banned, None, SolverSign.EQ, 0, name="portfolio_exposure")
        if not self.enforce_overlap:
            return
        cap = int(self.overlap_cap)
        for index, group in enumerate(self.history):
            variables = []
            for key in group:
                variables.extend(self.variables.get(key, []))
            if not variables:
                continue
            solver.add_constraint(
                variables,
                None,
                SolverSign.LTE,
                cap,
                name=f"portfolio_overlap_{index}",
            )

    def _record(self, lineup) -> None:
        seen: set[Identity] = set()
        for player in lineup:
            key = player_person_key(player)
            if not key[0] or key in seen:
                continue
            seen.add(key)
            self.used[key] += 1
            allowed = self.allowance.get(key, self.default_allowed)
            if allowed is not None and self.used[key] >= allowed:
                self.blocked.add(key)
        self.history.append(seen)


class PortfolioMinExposureRule(OptimizerRule):
    """Hold identity minimums for the lineups still left in the portfolio.

    A player is required in this lineup when the appearances still needed are
    at least the lineups still to build, including batches that have not
    started. A position group is required only for the overflow that cannot
    fit after this lineup. ``min_relax`` records usage and adds no constraint,
    which is how an impossible reserved set gets out of the way without
    raising a max.
    """

    def __init__(self, optimizer, players_dict, context):
        super().__init__(optimizer, players_dict, context)
        state: PortfolioRuleState = optimizer._portfolio_limits
        self.need: Counter[Identity] = Counter(
            {key: int(count) for key, count in state.min_remaining.items() if int(count) > 0}
        )
        self.after = int(state.min_after)
        self.batch_left = int(state.min_batch)
        self.relax = bool(state.min_relax)
        self.groups = tuple(state.min_groups)
        self.priority = dict(state.min_priority)
        self.roster = int(state.min_roster)
        self.variables: Dict[Identity, list] = {}
        for player, variable in players_dict.items():
            key = player_person_key(player)
            self.variables.setdefault(key, []).append(variable)

    def apply_for_iteration(self, solver, result):
        if result is not None:
            self._record(result)
            self.batch_left -= 1
        if self.relax or self.batch_left <= 0 or not self.need:
            return
        portfolio_left = self.batch_left + self.after
        if portfolio_left <= 0:
            return
        due = [
            key
            for key, need in self.need.items()
            if need >= portfolio_left and need > 0 and self.variables.get(key)
        ]
        due.extend(self._overflow_players(portfolio_left, set(due)))
        forced = 0
        for key in dict.fromkeys(due):
            variables = self.variables.get(key) or []
            if not variables:
                continue
            solver.add_constraint(
                variables,
                None,
                SolverSign.GTE,
                1,
                name=f"portfolio_min_{forced}",
            )
            forced += 1
        for index, (members, slots) in enumerate(self.groups):
            demand = sum(int(self.need.get(key, 0)) for key in members)
            if demand <= 0:
                continue
            overflow = demand - int(slots) * (portfolio_left - 1)
            if overflow <= 0:
                continue
            variables = []
            active = 0
            for key in members:
                if int(self.need.get(key, 0)) <= 0:
                    continue
                owned = self.variables.get(key) or []
                if not owned:
                    continue
                variables.extend(owned)
                active += 1
            force = min(int(overflow), int(slots), active)
            if force <= 0 or not variables:
                continue
            solver.add_constraint(
                variables,
                None,
                SolverSign.GTE,
                force,
                name=f"portfolio_min_group_{index}",
            )

    def _overflow_players(self, portfolio_left: int, already: set) -> List[Identity]:
        """Players who have to start now or the seats ahead cannot hold the need.

        Ties go to the lower projection so a punt is not left until the last
        lineup, where stars who are equally behind would crowd it out.
        """
        if self.roster <= 0 or portfolio_left <= 0:
            return []
        active = [
            key
            for key, need in self.need.items()
            if need > 0 and self.variables.get(key)
        ]
        demand = sum(int(self.need[key]) for key in active)
        overflow = demand - self.roster * (portfolio_left - 1)
        if overflow <= 0:
            return []
        ranked = sorted(
            active,
            key=lambda key: (-int(self.need[key]), self.priority.get(key, 0.0), key),
        )
        chosen: List[Identity] = []
        room = min(int(overflow), self.roster)
        for key in ranked:
            if key in already:
                room -= 1
                if room <= 0:
                    break
                continue
            chosen.append(key)
            room -= 1
            if room <= 0:
                break
        return chosen

    def _record(self, lineup) -> None:
        seen: set[Identity] = set()
        for player in lineup:
            key = player_person_key(player)
            if not key[0] or key in seen:
                continue
            seen.add(key)
            if self.need.get(key, 0) > 0:
                self.need[key] -= 1


@dataclass(frozen=True)
class MinExposureResult:
    """Portfolio minimums for one export.

    ``conflicts`` are obvious impossibilities known before the solve.
    ``unmet`` names players whose final count is still below the requested
    minimum. ``rows`` feed the exposure table.
    """

    requested: int
    conflicts: Tuple[str, ...]
    unmet: Tuple[str, ...]
    message: str
    rows: Tuple[Mapping[str, object], ...]


def evaluate_min_exposure(
    lineups: Sequence[pd.DataFrame],
    *,
    needed: Mapping[Identity, int],
    allowed: Mapping[Identity, Optional[int]],
    positions: Mapping[Identity, Iterable[str]],
    slots: Sequence[RosterSlot],
    requested: int,
    names: Mapping[Identity, str],
    teams: Optional[Mapping[Identity, str]] = None,
    position_labels: Optional[Mapping[Identity, str]] = None,
) -> MinExposureResult:
    """Compare the export with each identity's requested minimum."""
    requested = int(requested)
    team_labels = dict(teams or {})
    conflicts = min_exposure_conflicts(
        needed=needed,
        allowed=allowed,
        positions=positions,
        slots=slots,
        requested=requested,
        names=names,
        teams=team_labels,
    )
    counts = exposure_counts(lineups)
    labels = dict(position_labels or {})
    unmet: List[str] = []
    rows: List[dict] = []
    for key, need in sorted(needed.items(), key=lambda item: (names.get(item[0]) or item[0][0], item[0][1])):
        count = int(counts.get(key, 0))
        required = int(need)
        if required <= 0:
            continue
        label = _player_label(key, names, team_labels)
        if count < required:
            unmet.append(f"{label} is in {count} of {required} required lineups")
        rows.append(
            {
                "Name": names.get(key) or key[0],
                "Team": team_labels.get(key) or key[1],
                "Position": labels.get(key) or "/".join(sorted(positions.get(key) or ())),
                "needed": required,
            }
        )
    message = _min_result_message(conflicts, unmet)
    return MinExposureResult(
        requested=requested,
        conflicts=tuple(conflicts),
        unmet=tuple(unmet),
        message=message,
        rows=tuple(rows),
    )


def _min_result_message(conflicts: Sequence[str], unmet: Sequence[str]) -> str:
    parts: List[str] = []
    if conflicts:
        parts.append("Min exposure cannot all be met. " + " ".join(conflicts))
    if unmet:
        shown = list(unmet[:8])
        text = "Unmet minimums: " + "; ".join(shown)
        extra = len(unmet) - len(shown)
        if extra > 0:
            text += f"; and {extra} more"
        parts.append(text + ".")
    return " ".join(parts)


def exposure_report_frame(
    lineups: Sequence[pd.DataFrame],
    requested: Optional[int] = None,
    min_players: Optional[Sequence[Mapping[str, object]]] = None,
) -> pd.DataFrame:
    """One row per real player. This is not a DraftKings upload file.

    ``Lineups`` is the count. ``Exported %`` divides by the lineups that were
    built. ``Requested %`` divides by the number the user asked for, so a
    short export does not look like a higher cap. ``min_players`` adds
    ``Min needed`` and ``Min met`` and keeps a targeted player who was never
    rostered. Without that argument the columns stay as they are when no
    minimum is set.
    """
    min_rows = [dict(row) for row in (min_players or [])]
    columns = ["Name", "Team", "Position", "Lineups", "Exported %", "Requested %"]
    exported = len(lineups)
    requested_n = exported if requested is None else max(0, int(requested))
    # Classic also has a FLEX roster slot. Captain count columns are Showdown-only.
    show_slots = any(
        "Slot" in lineup.columns and lineup["Slot"].astype(str).str.upper().eq("CPT").any()
        for lineup in lineups
        if lineup is not None and not lineup.empty
    )
    if show_slots:
        columns = [
            "Name",
            "Team",
            "Position",
            "Lineups",
            "CPT count",
            "FLEX count",
            "Exported %",
            "Requested %",
        ]
    if min_rows:
        columns = [*columns, "Min needed", "Min met"]
    if exported == 0 and not min_rows:
        return pd.DataFrame(columns=columns)

    per_lineup: List[dict] = []
    for lineup in lineups:
        if lineup is None or lineup.empty:
            continue
        seen: set[Identity] = set()
        team_col = "Team" if "Team" in lineup.columns else "TeamAbbrev"
        for _, row in lineup.iterrows():
            key = person_key(
                row.get("Name"),
                row.get(team_col) if team_col in lineup.columns else "",
                row.get("Position"),
            )
            if not key[0] or key in seen:
                continue
            seen.add(key)
            slot = str(row.get("Slot") or "").strip().upper()
            per_lineup.append(
                {
                    "key": key,
                    "Name": str(row.get("Name") or "").strip(),
                    "Team": str(row.get(team_col) or "").strip() if team_col in lineup.columns else "",
                    "Position": _football_position(row.get("Position")),
                    "slot": slot,
                }
            )
    if not per_lineup:
        grouped = pd.DataFrame(
            columns=["key", "Name", "Team", "Position", "Lineups", "CPT count", "FLEX count"]
        )
    else:
        frame = pd.DataFrame(per_lineup)
        grouped = (
            frame.groupby(["key", "Name", "Team", "Position"], as_index=False)
            .agg(Lineups=("key", "size"), **(
                {"CPT count": ("slot", lambda s: int(s.eq("CPT").sum())),
                 "FLEX count": ("slot", lambda s: int(s.eq("FLEX").sum()))}
                if show_slots
                else {}
            ))
        )
    if min_rows:
        grouped = _apply_min_targets(
            grouped,
            min_rows,
            show_slots=show_slots,
            exported=exported,
            requested_n=requested_n,
        )
    else:
        if exported:
            grouped["Exported %"] = (grouped["Lineups"] / exported * 100).round(1)
        else:
            grouped["Exported %"] = 0.0
        if requested_n:
            grouped["Requested %"] = (grouped["Lineups"] / requested_n * 100).round(1)
        else:
            grouped["Requested %"] = 0.0
        grouped = grouped.sort_values(
            ["Lineups", "Name"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
    return grouped[columns]


def _apply_min_targets(
    grouped: pd.DataFrame,
    min_rows: Sequence[Mapping[str, object]],
    *,
    show_slots: bool,
    exported: int,
    requested_n: int,
) -> pd.DataFrame:
    """Add minimum columns and players who were targeted but never rostered."""
    needed: Dict[Identity, int] = {}
    meta: Dict[Identity, Mapping[str, object]] = {}
    for row in min_rows:
        key = person_key(row.get("Name"), row.get("Team"), row.get("Position"))
        if not key[0]:
            continue
        needed[key] = int(row.get("needed") or 0)
        meta[key] = row
    if grouped.empty:
        grouped = pd.DataFrame(columns=["key", "Name", "Team", "Position", "Lineups"])
    present = set(grouped["key"]) if "key" in grouped.columns else set()
    extras = []
    for key, row in meta.items():
        if key in present:
            continue
        extra = {
            "key": key,
            "Name": str(row.get("Name") or "").strip(),
            "Team": str(row.get("Team") or "").strip(),
            "Position": _football_position(row.get("Position")),
            "Lineups": 0,
        }
        if show_slots:
            extra["CPT count"] = 0
            extra["FLEX count"] = 0
        extras.append(extra)
    if extras:
        grouped = pd.concat([grouped, pd.DataFrame(extras)], ignore_index=True)
    grouped["Lineups"] = grouped["Lineups"].fillna(0).astype(int)
    if show_slots:
        for column in ("CPT count", "FLEX count"):
            if column not in grouped.columns:
                grouped[column] = 0
            grouped[column] = grouped[column].fillna(0).astype(int)
    if exported:
        grouped["Exported %"] = (grouped["Lineups"] / exported * 100).round(1)
    else:
        grouped["Exported %"] = 0.0
    if requested_n:
        grouped["Requested %"] = (grouped["Lineups"] / requested_n * 100).round(1)
    else:
        grouped["Requested %"] = 0.0

    def _met(key: Identity, count: int) -> str:
        if key not in needed:
            return ""
        required = needed[key]
        state = "Yes" if int(count) >= required else "No"
        return f"{state} ({int(count)} of {required})"

    grouped["Min needed"] = grouped["key"].map(lambda key: needed.get(key, pd.NA))
    grouped["Min met"] = [
        _met(key, count) for key, count in zip(grouped["key"], grouped["Lineups"])
    ]
    grouped["_unmet"] = grouped["Min met"].astype(str).str.startswith("No")
    return (
        grouped.sort_values(
            ["_unmet", "Lineups", "Name"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .drop(columns=["_unmet"])
        .reset_index(drop=True)
    )


def exposure_report_csv(
    lineups: Sequence[pd.DataFrame],
    requested: Optional[int] = None,
    min_players: Optional[Sequence[Mapping[str, object]]] = None,
) -> str:
    """Exposure summary CSV.

    DraftKings rejects extra columns on the upload file, so exposure stays in
    this separate report. Showdown captain and flex rows are one player.
    ``Exported %`` uses the built set. ``Requested %`` uses ``requested``.
    Minimum columns are present only when ``min_players`` is passed.
    """
    return exposure_report_frame(
        lineups,
        requested=requested,
        min_players=min_players,
    ).to_csv(index=False)


def _football_position(value: object) -> str:
    tokens = position_tokens(value)
    if not tokens:
        return str(value or "").strip().upper()
    return "/".join(sorted(tokens))
