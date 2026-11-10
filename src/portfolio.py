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

Min exposure is unchanged and still belongs to pydfs, per batch. The solver
does not name a unique infeasible constraint. Shortfall text reports a limit
as binding only when the roster math shows that limit is enough to block
another lineup. Everything else is labeled as a candidate.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
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

    if min_exposure is not None:
        candidates.append(
            f"Min exposure {float(min_exposure) * 100:.0f}% is active on each solver batch. "
            "It was not isolated as the cause."
        )

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
    """Mutable counters the identity rule reads while pydfs solves one batch."""

    portfolio_size: int
    overlap_cap: Optional[int]
    enforce_overlap: bool
    default_allowed: Optional[int]
    allowance: Dict[Identity, int]
    prior_used: Dict[Identity, int]
    prior_history: List[set[Identity]]


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


def exposure_report_frame(
    lineups: Sequence[pd.DataFrame],
    requested: Optional[int] = None,
) -> pd.DataFrame:
    """One row per real player. This is not a DraftKings upload file.

    ``Lineups`` is the count. ``Exported %`` divides by the lineups that were
    built. ``Requested %`` divides by the number the user asked for, so a
    short export does not look like a higher cap.
    """
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
    if exported == 0:
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
        return pd.DataFrame(columns=columns)
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
    grouped["Exported %"] = (grouped["Lineups"] / exported * 100).round(1)
    if requested_n:
        grouped["Requested %"] = (grouped["Lineups"] / requested_n * 100).round(1)
    else:
        grouped["Requested %"] = 0.0
    grouped = grouped.sort_values(
        ["Lineups", "Name"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    return grouped[columns]


def exposure_report_csv(
    lineups: Sequence[pd.DataFrame],
    requested: Optional[int] = None,
) -> str:
    """Exposure summary CSV.

    DraftKings rejects extra columns on the upload file, so exposure stays in
    this separate report. Showdown captain and flex rows are one player.
    ``Exported %`` uses the built set. ``Requested %`` uses ``requested``.
    """
    return exposure_report_frame(lineups, requested=requested).to_csv(index=False)


def _football_position(value: object) -> str:
    tokens = position_tokens(value)
    if not tokens:
        return str(value or "").strip().upper()
    return "/".join(sorted(tokens))
