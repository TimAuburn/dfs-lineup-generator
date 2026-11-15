"""Classic cash cap on wide receivers and tight ends from one team.

A Classic cash lineup can roster at most one WR or TE from a team unless
that team's quarterback is also in the lineup. Strict mode keeps the cap at
one even when the quarterback is rostered. The rule is a solver constraint.
It does not change a projection, and it is not used for Classic GPP or
Showdown.

Each player is one selection variable. A WR who is also eligible for FLEX
is still that one variable, so the player is counted once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import pandas as pd
from pydfs_lineup_optimizer.rules import OptimizerRule
from pydfs_lineup_optimizer.solvers.constants import SolverSign

from .portfolio import position_tokens
from .slate_week import dk_team

RULE_PASS_CATCHERS = "Pass catchers: max 1 WR/TE per team unless its QB is rostered"
RULE_PASS_CATCHERS_STRICT = "Pass catchers: strict: max 1 WR/TE per team"
PASS_CATCHER_POSITIONS = frozenset({"WR", "TE"})
CONSTRAINT_PREFIX = "cash_pass_catcher_"


@dataclass(frozen=True)
class PassCatcherCapSettings:
    """Strict mode ignores the quarterback exception. The allowance is the roster's."""

    strict: bool = False


def pass_catcher_rule_label(strict: bool) -> str:
    """Caption for the mode that is on."""
    if strict:
        return RULE_PASS_CATCHERS_STRICT
    return RULE_PASS_CATCHERS


def pass_catcher_cause(strict: bool) -> str:
    """Binding-limit sentence. The lineup is not rebuilt without the rule."""
    return f"{pass_catcher_rule_label(strict)}. The rule was not loosened."


def extra_pass_catcher_allowance(positions: Optional[Sequence[object]]) -> int:
    """Tightest M in ``sum(WR/TE of T) <= 1 + M * QB_T``.

    A Classic lineup has three WR slots and one TE slot. FLEX adds one more
    pass catcher when it can be a WR or TE. M is that total minus the one
    pass catcher allowed when the quarterback is not rostered. ``1 + M`` is
    therefore every pass-catcher slot, and the quarterback exception does
    not add a tighter cap of its own.

    ``positions`` are lineup slots (``name`` plus eligible ``positions``).
    None uses the full DraftKings flex, which includes WR and TE, so M is 4.
    An RB-only flex leaves M at 3.
    """
    if not positions:
        return 4
    slots = 0
    for slot in positions:
        name = str(getattr(slot, "name", "") or "").upper()
        eligible = {
            str(pos).upper()
            for pos in (getattr(slot, "positions", ()) or ())
        }
        if name in PASS_CATCHER_POSITIONS or (eligible & PASS_CATCHER_POSITIONS):
            slots += 1
    if slots <= 1:
        return 0
    return slots - 1


def _tokens(value: object) -> set[str]:
    if isinstance(value, (list, tuple, set, frozenset)):
        found: set[str] = set()
        for item in value:
            found |= position_tokens(item)
        return found
    return position_tokens(value)


def solver_positions(player: object) -> set[str]:
    """Football positions on one solver row. FLEX and CPT are slots, not positions."""
    return _tokens(getattr(player, "positions", ()) or ())


def _is_pass_catcher(positions: Iterable[str]) -> bool:
    owned = set(positions)
    if owned & {"DST", "DEF"}:
        return False
    return bool(owned & PASS_CATCHER_POSITIONS)


def _is_quarterback(positions: Iterable[str]) -> bool:
    owned = set(positions)
    if owned & {"DST", "DEF"} or _is_pass_catcher(owned):
        return False
    return "QB" in owned


def pass_catcher_violations(lineup: pd.DataFrame, *, strict: bool) -> tuple[str, ...]:
    """Teams whose WR/TE count breaks the active cash rule.

    Each player id is counted once, in whichever slot they were assigned.
    A missing lineup is not a violation. This is the same count the solver
    constraint enforces.
    """
    if lineup is None or lineup.empty:
        return ()
    id_col = "ID" if "ID" in lineup.columns else None
    team_col = "Team" if "Team" in lineup.columns else "TeamAbbrev"
    if team_col not in lineup.columns or "Position" not in lineup.columns:
        return ()
    catchers: dict[str, int] = {}
    quarterbacks: set[str] = set()
    seen: set[str] = set()
    for index, row in lineup.iterrows():
        player_id = str(row.get(id_col)) if id_col else str(index)
        if player_id in seen:
            continue
        seen.add(player_id)
        positions = _tokens(row.get("Position"))
        team = dk_team(row.get(team_col))
        if not team:
            continue
        if _is_pass_catcher(positions):
            catchers[team] = catchers.get(team, 0) + 1
        elif _is_quarterback(positions):
            quarterbacks.add(team)
    found: list[str] = []
    for team in sorted(catchers):
        count = catchers[team]
        if count <= 1:
            continue
        if not strict and team in quarterbacks:
            continue
        if strict:
            found.append(
                f"{team} has {count} WR/TE and the strict cap is 1"
            )
        else:
            found.append(
                f"{team} has {count} WR/TE and its QB is not rostered"
            )
    return tuple(found)


class PassCatcherTeamRule(OptimizerRule):
    """One linear cap per team that has two or more pass catchers in the pool.

    Default: ``sum(WR/TE of T) <= 1 + M * sum(QB of T)``.
    Strict: ``sum(WR/TE of T) <= 1``.
    A player who is eligible for more than one slot still has one variable.
    """

    def apply(self, solver):
        spec = getattr(self.optimizer, "_pass_catcher_cap", None)
        if spec is None:
            return
        strict = bool(getattr(spec, "strict", False))
        allowance = 0 if strict else extra_pass_catcher_allowance(
            getattr(self.optimizer.settings, "positions", None)
        )
        order: dict[str, list] = {}
        for player, variable in self.players_dict.items():
            positions = solver_positions(player)
            team = dk_team(getattr(player, "team", ""))
            if not team:
                continue
            if _is_pass_catcher(positions):
                kind = "catch"
            elif _is_quarterback(positions):
                kind = "qb"
            else:
                continue
            order.setdefault(team, []).append(
                (str(getattr(player, "id", "")), variable, kind)
            )
        for team in sorted(order):
            ranked = sorted(order.get(team, []), key=lambda item: (item[0], item[2]))
            catchers = []
            quarterbacks = []
            seen_vars = set()
            for _player_id, variable, kind in ranked:
                if id(variable) in seen_vars:
                    continue
                seen_vars.add(id(variable))
                if kind == "catch":
                    catchers.append(variable)
                else:
                    quarterbacks.append(variable)
            if len(catchers) < 2:
                continue
            name = f"{CONSTRAINT_PREFIX}{team}"
            if strict or allowance <= 0 or not quarterbacks:
                solver.add_constraint(catchers, None, SolverSign.LTE, 1, name=name)
                continue
            # LpVariable equality builds an expression, so membership tests
            # collapse distinct players. Key the dict by the variable object
            # and keep insertion order, which is player-id order.
            coefficients: dict = {}
            for variable in catchers:
                coefficients[variable] = coefficients.get(variable, 0.0) + 1.0
            for variable in quarterbacks:
                coefficients[variable] = coefficients.get(variable, 0.0) - float(allowance)
            solver.add_constraint(
                list(coefficients),
                list(coefficients.values()),
                SolverSign.LTE,
                1,
                name=name,
            )
