"""Lineup uniqueness by person, not by the raw full name.

pydfs-lineup-optimizer 3.6.1 ``UniquePlayerRule`` groups roster rows with
``groupby`` on ``player.full_name`` and constrains each group to one spot.
The grouping key is a static method. The library does not take a callable,
and captain/flex rows are different ``Player`` objects, so the rule cannot
tell "the same person at CPT and FLEX" from "a different person who has the
same name on another team."

This module removes that rule and adds one with the same constraint shape:
the sum of the binary variables in a group is at most 1. The group key is
``player_person_key`` (normalized name + team, defense by team). Showdown
captain and flex rows of one person share that key, so they stay mutually
exclusive. Two players who share a normalized name and play for different
teams are different groups and can share a lineup.
"""

from __future__ import annotations

from pydfs_lineup_optimizer.rules import OptimizerRule, UniquePlayerRule
from pydfs_lineup_optimizer.solvers.constants import SolverSign

from .names import player_person_key


class PersonUniquenessRule(OptimizerRule):
    """At most one roster spot per ``person_key``."""

    def apply(self, solver):
        grouped: dict[tuple[str, str], list] = {}
        for player, variable in self.players_dict.items():
            grouped.setdefault(player_person_key(player), []).append(variable)
        for index, variables in enumerate(grouped.values()):
            if len(variables) < 2:
                continue
            solver.add_constraint(
                variables,
                None,
                SolverSign.LTE,
                1,
                name=f"person_unique_{index}",
            )


def install_person_uniqueness(optimizer) -> None:
    """Use person identity instead of pydfs ``UniquePlayerRule``."""
    optimizer.remove_rule(UniquePlayerRule)
    optimizer.add_new_rule(PersonUniquenessRule)
