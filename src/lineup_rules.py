"""Lineup-construction rules for the Large-Field GPP preset.

These rules choose who can share a lineup. They do not change a projection.
A defense is never a pass catcher and never a bring-back. A defense from
either team in the quarterback's game does count toward that game's player
total.

A two-player same-game stack is at least two offensive players (QB, RB, WR,
or TE) from one game other than the quarterback's game. Teammates count, and
so do players on opposite sides of that game. Each such game counts once. A
defense does not count toward the two players.

Team matching uses ``dk_team`` (WSH/WAS, JAC/JAX, LA/LAR, and the older
STL, SD, and OAK codes). Player identity uses ``person_key``.

WR1 is the highest-salary wide receiver on a team. A salary tie counts, so
every wide receiver at that salary is a WR1. The salary is the DraftKings
salary, not the projection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd
from pydfs_lineup_optimizer.rules import OptimizerRule
from pydfs_lineup_optimizer.solvers.constants import SolverSign

from .names import person_key
from .portfolio import position_tokens
from .slate_week import dk_team, parse_game_info

PASS_CATCHER_POSITIONS = frozenset({"WR", "TE"})
BRING_BACK_POSITIONS = frozenset({"RB", "WR", "TE"})
OFFENSIVE_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})
DEFENSE_POSITIONS = frozenset({"DST", "DEF"})
CLASSIC_ROSTER = 9

RULE_PRIMARY = "Primary stack"
RULE_MINI = "Secondary mini-stack"
RULE_FAVORITE = "RB + own DST"
RULE_DOG = "Underdog RB restriction"


@dataclass(frozen=True)
class ConstructionSettings:
    """On/off switches and knobs. Spreads are slate data and stay outside."""

    primary_stack: bool = False
    pass_catchers: int = 2
    bring_backs: int = 1
    game_min: int = 4
    game_max: int = 5
    bring_back_wr1: bool = False
    mini_stack: bool = False
    mini_count: int = 1
    rb_dst_favorite: bool = False
    favorite_margin: float = 7.0
    ban_underdog_rbs: bool = False
    underdog_margin: float = 7.0

    @property
    def any_enabled(self) -> bool:
        return bool(
            self.primary_stack
            or self.mini_stack
            or self.rb_dst_favorite
            or self.ban_underdog_rbs
        )


@dataclass(frozen=True)
class ConstructionPlan:
    """What the solver will enforce, and the rules that were named instead."""

    requested: ConstructionSettings
    applied: ConstructionSettings
    messages: tuple[str, ...] = ()
    dropped: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ConstructionReport:
    """Named gaps for one export. ``message`` is empty when every applied rule held."""

    messages: tuple[str, ...]
    message: str
    applied: ConstructionSettings


@dataclass(frozen=True)
class _Athlete:
    id: str
    name: str
    team: str
    positions: frozenset[str]
    salary: int
    game: Optional[frozenset[str]]
    opponent: Optional[str]
    person: tuple[str, str]


@dataclass(frozen=True)
class LineupAssessment:
    """How one lineup sits against the rules that are on."""

    failures: tuple[str, ...]
    game_players: int
    pass_catchers: int
    bring_backs: int
    mini_stacks: int
    primary_text: str
    mini_text: str
    favorite_text: str
    dog_text: str


def team_code(team: object) -> str:
    return dk_team(team)


def _positions_of(value: object) -> frozenset[str]:
    return frozenset(position_tokens(value))


def is_defense(positions: Iterable[str]) -> bool:
    return bool(set(positions) & DEFENSE_POSITIONS)


def is_pass_catcher(positions: Iterable[str]) -> bool:
    owned = set(positions)
    return bool(owned & PASS_CATCHER_POSITIONS) and not is_defense(owned)


def is_bring_back(positions: Iterable[str]) -> bool:
    owned = set(positions)
    return bool(owned & BRING_BACK_POSITIONS) and not is_defense(owned)


def is_offensive(positions: Iterable[str]) -> bool:
    owned = set(positions)
    return bool(owned & OFFENSIVE_POSITIONS) and not is_defense(owned)


def is_running_back(positions: Iterable[str]) -> bool:
    owned = set(positions)
    return "RB" in owned and not is_defense(owned)


def is_wide_receiver(positions: Iterable[str]) -> bool:
    owned = set(positions)
    return "WR" in owned and not is_defense(owned)


def points_label(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def spread_for(spreads: Optional[Mapping[str, object]], team: object) -> Optional[float]:
    """Signed spread for ``team``, or None when the slate has no number for it.

    A missing team, a blank cell, and a non-numeric cell are all missing.
    Zero is a real pick'em and is returned.
    """
    if not spreads:
        return None
    club = team_code(team)
    if not club:
        return None
    raw = None
    if club in spreads:
        raw = spreads[club]
    else:
        for key, value in spreads.items():
            if team_code(key) == club:
                raw = value
                break
    if raw is None and club not in spreads and not any(team_code(key) == club for key in spreads):
        return None
    if raw is None:
        return None
    try:
        if raw != raw:  # NaN
            return None
    except TypeError:
        pass
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def team_is_favorite(
    spreads: Optional[Mapping[str, object]],
    team: object,
    margin: float,
) -> bool:
    """True when the team is favored by at least ``margin`` points.

    The stored spread is the team's own line, negative for the favorite.
    Exactly ``-margin`` is eligible. A missing spread is not.
    """
    spread = spread_for(spreads, team)
    if spread is None:
        return False
    return spread <= -float(margin)


def team_is_underdog(
    spreads: Optional[Mapping[str, object]],
    team: object,
    margin: float,
) -> bool:
    """True when the team is an underdog by at least ``margin`` points.

    Exactly ``+margin`` is excluded. A missing spread is not an underdog.
    """
    spread = spread_for(spreads, team)
    if spread is None:
        return False
    return spread >= float(margin)


def normalize_spreads(lines: Optional[Mapping[str, object]]) -> dict[str, float]:
    """``{team: signed spread}`` from a Vegas-line map or a flat spread map.

    A line object contributes its ``spread`` key. A missing or unreadable
    spread is left out. Nothing is filled in with zero.
    """
    cleaned: dict[str, float] = {}
    for team, info in (lines or {}).items():
        raw: object
        if isinstance(info, Mapping) and "spread" in info:
            raw = info.get("spread")
        else:
            raw = info
        club = team_code(team)
        if not club:
            continue
        if raw is None:
            continue
        try:
            if raw != raw:
                continue
        except TypeError:
            pass
        try:
            cleaned[club] = float(raw)
        except (TypeError, ValueError):
            continue
    return cleaned


def _unmet(rule: str, detail: str) -> str:
    return f"Unmet lineup rule: {rule}. {detail}"


def stack_floor(settings: ConstructionSettings) -> int:
    """Quarterback plus the pass catchers plus the bring-backs."""
    return 1 + int(settings.pass_catchers) + int(settings.bring_backs)


def construction_problem_messages(settings: ConstructionSettings) -> tuple[str, ...]:
    """Settings that cannot be built, named before the solver runs."""
    messages: list[str] = []
    if settings.primary_stack:
        need = stack_floor(settings)
        if int(settings.game_max) < need:
            messages.append(
                _unmet(
                    RULE_PRIMARY,
                    f"The game-player max is {int(settings.game_max)}, but the quarterback, "
                    f"{int(settings.pass_catchers)} pass catcher(s), and "
                    f"{int(settings.bring_backs)} bring-back(s) need {need} players from that game. "
                    "A defense counts toward that total and is never a pass catcher or a bring-back.",
                )
            )
        elif int(settings.game_min) > int(settings.game_max):
            messages.append(
                _unmet(
                    RULE_PRIMARY,
                    f"The game-player minimum ({int(settings.game_min)}) is higher than the "
                    f"maximum ({int(settings.game_max)}).",
                )
            )
        elif int(settings.game_max) > CLASSIC_ROSTER or int(settings.game_min) > CLASSIC_ROSTER:
            messages.append(
                _unmet(
                    RULE_PRIMARY,
                    f"A Classic lineup has {CLASSIC_ROSTER} players, so the game-player "
                    f"window cannot exceed {CLASSIC_ROSTER}.",
                )
            )
        elif int(settings.game_min) < 1 or int(settings.game_max) < 1:
            messages.append(
                _unmet(RULE_PRIMARY, "The game-player window has to be at least 1.")
            )
        if settings.bring_back_wr1 and int(settings.bring_backs) < 1:
            messages.append(
                _unmet(
                    RULE_PRIMARY,
                    "The bring-back WR1 option needs at least one bring-back. "
                    "WR1 is the highest-salary wide receiver on the opposing team.",
                )
            )
    if settings.mini_stack and not 0 <= int(settings.mini_count) <= 2:
        messages.append(
            _unmet(RULE_MINI, "The extra same-game stack count has to be from 0 to 2.")
        )
    if (
        settings.primary_stack
        and settings.mini_stack
        and 0 <= int(settings.mini_count) <= 2
        and int(settings.game_min) + 2 * int(settings.mini_count) > CLASSIC_ROSTER
    ):
        need = int(settings.game_min) + 2 * int(settings.mini_count)
        messages.append(
            _unmet(
                RULE_MINI,
                f"The quarterback's game asks for at least {int(settings.game_min)} players and "
                f"{int(settings.mini_count)} two-player stack(s) need "
                f"{2 * int(settings.mini_count)} more, {need} in all. "
                f"A lineup has {CLASSIC_ROSTER} spots.",
            )
        )
    return tuple(messages)


def _disable_for_messages(
    settings: ConstructionSettings,
    messages: Sequence[str],
) -> ConstructionSettings:
    """Turn off a rule whose settings message already says it cannot be built."""
    primary = settings.primary_stack
    mini = settings.mini_stack
    wr1 = settings.bring_back_wr1
    for message in messages:
        if message.startswith(f"Unmet lineup rule: {RULE_PRIMARY}."):
            # The WR1 knob can be illegal on its own. The stack itself still stands.
            if "WR1 option" in message:
                wr1 = False
            else:
                primary = False
                wr1 = False
        elif message.startswith(f"Unmet lineup rule: {RULE_MINI}."):
            mini = False
    if wr1 and int(settings.bring_backs) < 1:
        wr1 = False
    return replace(
        settings,
        primary_stack=primary,
        mini_stack=mini,
        bring_back_wr1=wr1 and primary,
    )


def _matchup(away: object, home: object) -> Optional[frozenset[str]]:
    left, right = team_code(away), team_code(home)
    if not left or not right:
        return None
    return frozenset({left, right})


def _game_from_text(raw: object) -> Optional[frozenset[str]]:
    parsed = parse_game_info(raw)
    if parsed is None:
        return None
    return _matchup(parsed.away, parsed.home)


def _opponent(team: str, game: Optional[frozenset[str]]) -> Optional[str]:
    if not game or team not in game or len(game) != 2:
        return None
    return next(iter(game - {team}))


def _athlete_from_player(player: object) -> _Athlete:
    positions = frozenset()
    for pos in getattr(player, "positions", ()) or ():
        positions |= _positions_of(pos)
    info = getattr(player, "game_info", None)
    game = None
    if info is not None and getattr(info, "home_team", None) and getattr(info, "away_team", None):
        game = _matchup(info.away_team, info.home_team)
    team = team_code(getattr(player, "team", ""))
    name = str(getattr(player, "full_name", "") or "").strip()
    try:
        salary = int(getattr(player, "salary", 0) or 0)
    except (TypeError, ValueError):
        salary = 0
    return _Athlete(
        id=str(getattr(player, "id", "")),
        name=name,
        team=team,
        positions=positions,
        salary=salary,
        game=game,
        opponent=_opponent(team, game),
        person=person_key(name, team, next(iter(positions), None)),
    )


def _wr1_ids(players: Sequence[_Athlete], opponent: Optional[str]) -> set[str]:
    if not opponent:
        return set()
    receivers = [
        player
        for player in players
        if player.team == opponent and is_wide_receiver(player.positions)
    ]
    if not receivers:
        return set()
    top = max(player.salary for player in receivers)
    return {player.id for player in receivers if player.salary == top}


def _primary_pieces(
    players: Sequence[_Athlete],
    quarterback: _Athlete,
) -> tuple[list[_Athlete], list[_Athlete], list[_Athlete]]:
    catchers = [
        player
        for player in players
        if player.id != quarterback.id
        and player.team == quarterback.team
        and is_pass_catcher(player.positions)
    ]
    bring = [
        player
        for player in players
        if quarterback.opponent
        and player.team == quarterback.opponent
        and is_bring_back(player.positions)
    ]
    game = [
        player
        for player in players
        if quarterback.opponent and player.team in {quarterback.team, quarterback.opponent}
    ]
    return catchers, bring, game


def pool_supports_primary(
    players: Sequence[_Athlete],
    settings: ConstructionSettings,
) -> bool:
    """Whether some quarterback has the pass catchers, bring-backs, and game size."""
    need = stack_floor(settings)
    quarterbacks = [player for player in players if "QB" in player.positions and not is_defense(player.positions)]
    for quarterback in quarterbacks:
        if quarterback.opponent is None:
            continue
        catchers, bring, game = _primary_pieces(players, quarterback)
        if len(catchers) < int(settings.pass_catchers):
            continue
        if len(bring) < int(settings.bring_backs):
            continue
        if len(game) < max(int(settings.game_min), need):
            continue
        if len(game) < int(settings.game_min):
            continue
        if settings.bring_back_wr1 and int(settings.bring_backs) >= 1:
            if not _wr1_ids(players, quarterback.opponent):
                continue
        return True
    return False


def favorite_is_available(
    players: Sequence[_Athlete],
    spreads: Optional[Mapping[str, object]],
    margin: float,
) -> bool:
    """A defense and a running back on one team favored by at least ``margin``."""
    defenses = {player.team for player in players if is_defense(player.positions)}
    backs = {player.team for player in players if is_running_back(player.positions)}
    for team in defenses & backs:
        if team_is_favorite(spreads, team, margin):
            return True
    return False


def plan_construction(
    players: Sequence[object],
    settings: ConstructionSettings,
    spreads: Optional[Mapping[str, object]] = None,
    locked_ids: Optional[Iterable[str]] = None,
) -> ConstructionPlan:
    """Decide which requested rules can be enforced on this pool.

    ``players`` are pydfs players or ``_Athlete`` rows. Projections are not read.
    """
    roster = [_as_athlete(player) for player in players]
    locked = {str(player_id) for player_id in (locked_ids or [])}
    messages = list(construction_problem_messages(settings))
    applied = _disable_for_messages(settings, messages)
    dropped: set[str] = set()
    if settings.primary_stack and not applied.primary_stack:
        dropped.add("primary")
    if settings.mini_stack and not applied.mini_stack:
        dropped.add("mini")

    if applied.ban_underdog_rbs:
        # Locked running backs stay, matching the salary-floor exception.
        pass
    if applied.primary_stack and not pool_supports_primary(roster, applied):
        messages.append(
            _unmet(
                RULE_PRIMARY,
                f"No quarterback has at least {int(applied.pass_catchers)} pass catcher(s) "
                f"(WR/TE) and {int(applied.bring_backs)} bring-back(s) (RB, WR, or TE) "
                "from the opposing team"
                + (
                    ", including that team's highest-salary wide receiver"
                    if applied.bring_back_wr1
                    else ""
                )
                + ". A defense is neither a pass catcher nor a bring-back, and a missing "
                "opponent leaves that quarterback out.",
            )
        )
        applied = replace(applied, primary_stack=False, bring_back_wr1=False)
        dropped.add("primary")
    if applied.rb_dst_favorite and not favorite_is_available(
        roster, spreads, applied.favorite_margin
    ):
        messages.append(
            _unmet(
                RULE_FAVORITE,
                f"No defense is favored by at least {points_label(applied.favorite_margin)} "
                "points and paired with a running back on that team. Exactly "
                f"{points_label(applied.favorite_margin)} counts. A missing spread does not.",
            )
        )
        applied = replace(applied, rb_dst_favorite=False)
        dropped.add("favorite")
    return ConstructionPlan(
        requested=settings,
        applied=applied,
        messages=tuple(messages),
        dropped=frozenset(dropped),
    )


def _as_athlete(player: object) -> _Athlete:
    if isinstance(player, _Athlete):
        return player
    return _athlete_from_player(player)


def underdog_running_backs(
    players: Sequence[object],
    spreads: Optional[Mapping[str, object]],
    margin: float,
    locked_ids: Optional[Iterable[str]] = None,
) -> list[object]:
    """Running backs the underdog rule removes. Locked ids are kept.

    The returned objects are the same objects that were passed in, so the
    caller can hand them to ``remove_player``. Their projections are not edited.
    """
    locked = {str(player_id) for player_id in (locked_ids or [])}
    banned = []
    for player in players:
        athlete = _as_athlete(player)
        if athlete.id in locked or not is_running_back(athlete.positions):
            continue
        if team_is_underdog(spreads, athlete.team, margin):
            banned.append(player)
    return banned


def _athletes_from_frames(
    lineup: pd.DataFrame,
    pool: Optional[pd.DataFrame],
) -> list[_Athlete]:
    by_id: dict[str, pd.Series] = {}
    if pool is not None and not pool.empty and "ID" in pool.columns:
        for _, row in pool.iterrows():
            by_id[str(row.get("ID"))] = row
    athletes: list[_Athlete] = []
    if lineup is None or lineup.empty:
        return athletes
    team_col = "Team" if "Team" in lineup.columns else "TeamAbbrev"
    for _, row in lineup.iterrows():
        player_id = str(row.get("ID", ""))
        source = by_id.get(player_id)
        positions = _positions_of(row.get("Position"))
        team = team_code(row.get(team_col) if team_col in lineup.columns else "")
        name = str(row.get("Name") or "").strip()
        game = None
        if source is not None and "Game Info" in source.index:
            game = _game_from_text(source.get("Game Info"))
        if game is None and "Game Info" in lineup.columns:
            game = _game_from_text(row.get("Game Info"))
        if game is None:
            opponent = team_code(row.get("Opponent"))
            if opponent:
                game = _matchup(team, opponent)
        try:
            salary = int(row.get("Salary") or 0)
        except (TypeError, ValueError):
            salary = 0
        if source is not None and "Salary" in source.index:
            try:
                salary = int(source.get("Salary") or salary)
            except (TypeError, ValueError):
                pass
        athletes.append(
            _Athlete(
                id=player_id,
                name=name or (str(source.get("Name")).strip() if source is not None else ""),
                team=team,
                positions=positions,
                salary=salary,
                game=game,
                opponent=_opponent(team, game),
                person=person_key(name, team, next(iter(positions), None)),
            )
        )
    return athletes


def _pool_athletes(pool: Optional[pd.DataFrame]) -> list[_Athlete]:
    if pool is None or pool.empty:
        return []
    frame = pool.copy()
    if "Team" not in frame.columns and "TeamAbbrev" in frame.columns:
        frame["Team"] = frame["TeamAbbrev"]
    return _athletes_from_frames(frame, frame)


def assess_lineup(
    lineup: pd.DataFrame,
    settings: ConstructionSettings,
    *,
    pool: Optional[pd.DataFrame] = None,
    spreads: Optional[Mapping[str, object]] = None,
    locked_ids: Optional[Iterable[str]] = None,
) -> LineupAssessment:
    """Score one lineup against the rules that are enabled on ``settings``."""
    roster = _athletes_from_frames(lineup, pool)
    pool_players = _pool_athletes(pool) or roster
    locked = {str(player_id) for player_id in (locked_ids or [])}
    failures: list[str] = []
    game_players = 0
    catcher_count = 0
    bring_count = 0
    mini_count = 0
    primary_text = "Off"
    mini_text = "Off"
    favorite_text = "Off"
    dog_text = "Off"

    quarterbacks = [
        player for player in roster if "QB" in player.positions and not is_defense(player.positions)
    ]
    quarterback = quarterbacks[0] if quarterbacks else None

    if settings.primary_stack:
        if quarterback is None or quarterback.opponent is None:
            failures.append("no quarterback with a game")
            primary_text = "No quarterback game"
        else:
            catchers, bring, game = _primary_pieces(roster, quarterback)
            catcher_count = len(catchers)
            bring_count = len(bring)
            game_players = len(game)
            wr1 = _wr1_ids(pool_players, quarterback.opponent)
            have_wr1 = any(player.id in wr1 for player in bring)
            bits = [
                f"{catcher_count} pass catcher(s)",
                f"{bring_count} bring-back(s)",
                f"{game_players} from the game",
            ]
            if catcher_count < int(settings.pass_catchers):
                failures.append(
                    f"{catcher_count} pass catcher(s), need at least {int(settings.pass_catchers)}"
                )
            if bring_count < int(settings.bring_backs):
                failures.append(
                    f"{bring_count} bring-back(s), need at least {int(settings.bring_backs)}"
                )
            if game_players < int(settings.game_min) or game_players > int(settings.game_max):
                failures.append(
                    f"{game_players} players from the quarterback's game, "
                    f"need {int(settings.game_min)}–{int(settings.game_max)} "
                    "(the defense counts)"
                )
            if settings.bring_back_wr1 and int(settings.bring_backs) >= 1 and not have_wr1:
                failures.append("bring-back is not the opposing WR1 (highest-salary WR)")
            defense_in_game = any(is_defense(player.positions) for player in game)
            primary_text = ", ".join(bits)
            if defense_in_game:
                primary_text += "; DST counts in the game total"
            else:
                primary_text += "; DST is not in this game"
            if settings.bring_back_wr1:
                primary_text += "; WR1 required" if have_wr1 else "; WR1 missing"
    if settings.mini_stack:
        mini_count = _mini_stack_count(roster, quarterback)
        mini_text = f"{mini_count} outside the quarterback's game"
        if mini_count < int(settings.mini_count):
            failures.append(
                f"{mini_count} two-player same-game stack(s) outside the quarterback's game, "
                f"need at least {int(settings.mini_count)}"
            )
    if settings.rb_dst_favorite:
        favorite_text, favorite_failures = _favorite_status(
            roster, spreads, settings.favorite_margin
        )
        failures.extend(favorite_failures)
    if settings.ban_underdog_rbs:
        dog_text, dog_failures = _dog_status(
            roster, spreads, settings.underdog_margin, locked
        )
        failures.extend(dog_failures)
    return LineupAssessment(
        failures=tuple(failures),
        game_players=game_players,
        pass_catchers=catcher_count,
        bring_backs=bring_count,
        mini_stacks=mini_count,
        primary_text=primary_text,
        mini_text=mini_text,
        favorite_text=favorite_text,
        dog_text=dog_text,
    )


def _mini_stack_count(
    roster: Sequence[_Athlete],
    quarterback: Optional[_Athlete],
) -> int:
    primary = quarterback.game if quarterback is not None else None
    grouped: dict[frozenset[str], int] = {}
    for player in roster:
        if player.game is None or not is_offensive(player.positions):
            continue
        if primary is not None and player.game == primary:
            continue
        grouped[player.game] = grouped.get(player.game, 0) + 1
    return sum(1 for count in grouped.values() if count >= 2)


def _favorite_status(
    roster: Sequence[_Athlete],
    spreads: Optional[Mapping[str, object]],
    margin: float,
) -> tuple[str, list[str]]:
    defenses = [player for player in roster if is_defense(player.positions)]
    if not defenses:
        return "No DST", ["no defense to pair with a running back"]
    failures: list[str] = []
    labels: list[str] = []
    for defense in defenses:
        backs = [
            player
            for player in roster
            if player.team == defense.team and is_running_back(player.positions)
        ]
        favored = team_is_favorite(spreads, defense.team, margin)
        if not favored:
            spread = spread_for(spreads, defense.team)
            if spread is None:
                reason = "missing spread"
            else:
                reason = f"favored by {points_label(-spread)}, need {points_label(margin)}"
            failures.append(
                f"{defense.team} DST is not eligible ({reason}). "
                f"Exactly {points_label(margin)} is eligible and a missing spread is not"
            )
        elif not backs:
            failures.append(f"{defense.team} DST has no running back from that team")
        labels.append(defense.team if backs and favored else f"{defense.team} no")
    return ", ".join(labels), failures


def _dog_status(
    roster: Sequence[_Athlete],
    spreads: Optional[Mapping[str, object]],
    margin: float,
    locked: set[str],
) -> tuple[str, list[str]]:
    banned = [
        player
        for player in roster
        if is_running_back(player.positions)
        and player.id not in locked
        and team_is_underdog(spreads, player.team, margin)
    ]
    if not banned:
        return "No excluded RB", []
    names = ", ".join(player.name or player.team for player in banned)
    return names, [
        f"{names} play for a team that is an underdog by at least {points_label(margin)}. "
        f"Exactly {points_label(margin)} is excluded and a missing spread is not"
    ]


def construction_check_frame(
    lineups: Sequence[pd.DataFrame],
    settings: ConstructionSettings,
    *,
    pool: Optional[pd.DataFrame] = None,
    spreads: Optional[Mapping[str, object]] = None,
) -> pd.DataFrame:
    """One row per lineup for the rules that are enabled."""
    rows = []
    for index, lineup in enumerate(lineups, start=1):
        scored = assess_lineup(lineup, settings, pool=pool, spreads=spreads)
        rows.append(
            {
                "Lineup": index,
                "Primary stack": scored.primary_text,
                "Mini-stacks": scored.mini_text,
                "Favorite RB + DST": scored.favorite_text,
                "Underdog RBs": scored.dog_text,
                "Result": "Pass" if not scored.failures else "; ".join(scored.failures),
            }
        )
    return pd.DataFrame(rows)


def report_from_messages(
    messages: Sequence[str],
    applied: ConstructionSettings,
) -> Optional[ConstructionReport]:
    cleaned = tuple(dict.fromkeys(message for message in messages if str(message).strip()))
    if not cleaned and not applied.any_enabled:
        return None
    return ConstructionReport(
        messages=cleaned,
        message=" ".join(cleaned),
        applied=applied,
    )


def verify_lineups(
    lineups: Sequence[pd.DataFrame],
    settings: ConstructionSettings,
    *,
    pool: Optional[pd.DataFrame] = None,
    spreads: Optional[Mapping[str, object]] = None,
    locked_ids: Optional[Iterable[str]] = None,
) -> tuple[str, ...]:
    """Name applied rules a built lineup does not satisfy."""
    found: list[str] = []
    for index, lineup in enumerate(lineups, start=1):
        scored = assess_lineup(
            lineup,
            settings,
            pool=pool,
            spreads=spreads,
            locked_ids=locked_ids,
        )
        for failure in scored.failures:
            if "same-game stack" in failure:
                rule = RULE_MINI
            elif "underdog" in failure:
                rule = RULE_DOG
            elif "DST" in failure or "defense" in failure:
                rule = RULE_FAVORITE
            else:
                rule = RULE_PRIMARY
            found.append(_unmet(rule, f"Lineup {index}: {failure}"))
    return tuple(dict.fromkeys(found))


class LineupConstructionRule(OptimizerRule):
    """PuLP form of the primary stack, the mini-stack, and the favorite DST.

    The underdog running-back rule is a pool filter, applied before this rule,
    so it never edits a projection either. Constraints are static for the
    batch, so they are added once on the base solver and copied with it.
    """

    def apply(self, solver):
        spec: ConstructionSettings = getattr(
            self.optimizer, "_construction_settings", ConstructionSettings()
        )
        spreads = getattr(self.optimizer, "_construction_spreads", {}) or {}
        players = [_athlete_from_player(player) for player in self.players_dict]
        by_id = {player.id: self.players_dict[raw] for raw, player in zip(self.players_dict, players)}
        if spec.primary_stack:
            self._primary(solver, spec, players, by_id)
        if spec.mini_stack and int(spec.mini_count) > 0:
            self._mini(solver, spec, players, by_id)
        if spec.rb_dst_favorite:
            self._favorite(solver, spec, spreads, players, by_id)

    def _primary(self, solver, spec: ConstructionSettings, players, by_id) -> None:
        quarterbacks = [
            player
            for player in players
            if "QB" in player.positions and not is_defense(player.positions) and player.id in by_id
        ]
        for index, quarterback in enumerate(quarterbacks):
            gate = by_id[quarterback.id]
            if quarterback.opponent is None:
                solver.add_constraint([gate], None, SolverSign.EQ, 0, name=f"gpp_qb_game_{index}")
                continue
            catchers, bring, game = _primary_pieces(players, quarterback)
            self._at_least(
                solver,
                [by_id[player.id] for player in catchers if player.id in by_id],
                gate,
                int(spec.pass_catchers),
                f"gpp_catch_{index}",
            )
            self._at_least(
                solver,
                [by_id[player.id] for player in bring if player.id in by_id],
                gate,
                int(spec.bring_backs),
                f"gpp_bring_{index}",
            )
            game_vars = [by_id[player.id] for player in game if player.id in by_id]
            self._at_least(solver, game_vars, gate, int(spec.game_min), f"gpp_game_min_{index}")
            self._at_most(solver, game_vars, gate, int(spec.game_max), f"gpp_game_max_{index}")
            if spec.bring_back_wr1 and int(spec.bring_backs) >= 1:
                wr1 = _wr1_ids(players, quarterback.opponent)
                self._at_least(
                    solver,
                    [by_id[player_id] for player_id in wr1 if player_id in by_id],
                    gate,
                    1,
                    f"gpp_wr1_{index}",
                )

    def _mini(self, solver, spec: ConstructionSettings, players, by_id) -> None:
        grouped: dict[frozenset[str], list[_Athlete]] = {}
        quarterbacks_by_game: dict[frozenset[str], list[_Athlete]] = {}
        for index, player in enumerate(players):
            if player.id not in by_id:
                continue
            is_qb = "QB" in player.positions and not is_defense(player.positions)
            if player.game is None:
                if is_qb:
                    solver.add_constraint(
                        [by_id[player.id]], None, SolverSign.EQ, 0, name=f"gpp_mini_qb_{index}"
                    )
                continue
            if is_qb:
                quarterbacks_by_game.setdefault(player.game, []).append(player)
            if is_offensive(player.positions):
                grouped.setdefault(player.game, []).append(player)
        ys = []
        zs = []
        for index, (game, members) in enumerate(grouped.items()):
            variables = [by_id[player.id] for player in members if player.id in by_id]
            if len(variables) < 2:
                continue
            indicator = solver.add_variable(f"gpp_mini_{index}")
            self._at_least(solver, variables, indicator, 2, f"gpp_mini_min_{index}")
            width = len(variables)
            solver.add_constraint(
                [*variables, indicator],
                [1.0] * width + [-(width - 1)],
                SolverSign.LTE,
                1,
                name=f"gpp_mini_cap_{index}",
            )
            ys.append(indicator)
            gates = [
                by_id[player.id]
                for player in quarterbacks_by_game.get(game, ())
                if player.id in by_id
            ]
            if not gates:
                continue
            paired = solver.add_variable(f"gpp_mini_qb_game_{index}")
            solver.add_constraint([paired, indicator], [1, -1], SolverSign.LTE, 0, name=f"gpp_z_y_{index}")
            solver.add_constraint(
                [paired, *gates],
                [1.0] + [-1.0] * len(gates),
                SolverSign.LTE,
                0,
                name=f"gpp_z_q_{index}",
            )
            solver.add_constraint(
                [paired, indicator, *gates],
                [1.0, -1.0] + [-1.0] * len(gates),
                SolverSign.GTE,
                -1,
                name=f"gpp_z_both_{index}",
            )
            zs.append(paired)
        if not ys:
            blocked = solver.add_variable("gpp_mini_impossible")
            solver.add_constraint([blocked], None, SolverSign.EQ, 1, name="gpp_mini_on")
            solver.add_constraint([blocked], None, SolverSign.EQ, 0, name="gpp_mini_off")
            return
        solver.add_constraint(
            [*ys, *zs],
            [1.0] * len(ys) + [-1.0] * len(zs),
            SolverSign.GTE,
            float(int(spec.mini_count)),
            name="gpp_mini_total",
        )

    def _favorite(self, solver, spec: ConstructionSettings, spreads, players, by_id) -> None:
        defenses = [player for player in players if is_defense(player.positions) and player.id in by_id]
        for index, defense in enumerate(defenses):
            gate = by_id[defense.id]
            if not team_is_favorite(spreads, defense.team, spec.favorite_margin):
                solver.add_constraint([gate], None, SolverSign.EQ, 0, name=f"gpp_dst_ban_{index}")
                continue
            backs = [
                by_id[player.id]
                for player in players
                if player.team == defense.team
                and is_running_back(player.positions)
                and player.id in by_id
            ]
            self._at_least(solver, backs, gate, 1, f"gpp_dst_rb_{index}")

    @staticmethod
    def _at_least(solver, variables, gate, minimum: int, name: str) -> None:
        if minimum <= 0:
            return
        if not variables:
            solver.add_constraint([gate], None, SolverSign.EQ, 0, name=name)
            return
        # The gate can also be one of the players (the quarterback is in his own game).
        coefficients = _gate_coefficients(variables, gate, -float(minimum))
        solver.add_constraint(
            list(coefficients),
            list(coefficients.values()),
            SolverSign.GTE,
            0,
            name=name,
        )

    @staticmethod
    def _at_most(solver, variables, gate, maximum: int, name: str) -> None:
        if not variables:
            return
        coefficients = _gate_coefficients(
            variables, gate, -(float(maximum) - float(CLASSIC_ROSTER))
        )
        solver.add_constraint(
            list(coefficients),
            list(coefficients.values()),
            SolverSign.LTE,
            float(CLASSIC_ROSTER),
            name=name,
        )


def _gate_coefficients(variables, gate, gate_coefficient: float) -> dict:
    """One coefficient per variable. A player who is also the gate is not listed twice."""
    coefficients: dict = {}
    for variable in variables:
        coefficients[variable] = coefficients.get(variable, 0.0) + 1.0
    coefficients[gate] = coefficients.get(gate, 0.0) + float(gate_coefficient)
    return coefficients


def dropped_rule_message(rule_key: str) -> str:
    """Name a rule the solver had to leave off so a legal lineup could be built."""
    labels = {
        "primary": RULE_PRIMARY,
        "mini": RULE_MINI,
        "favorite": RULE_FAVORITE,
        "dogs": RULE_DOG,
    }
    return _unmet(
        labels.get(rule_key, "Lineup construction"),
        "The solver could not build the requested lineups with that rule on, "
        "so these lineups were built without it.",
    )


def install_construction_rule(optimizer, settings: ConstructionSettings, spreads: Mapping[str, object]) -> None:
    """Attach the rule to a pydfs optimizer. Does not edit player projections."""
    if not (settings.primary_stack or settings.mini_stack or settings.rb_dst_favorite):
        return
    optimizer._construction_settings = settings
    optimizer._construction_spreads = dict(spreads or {})
    optimizer.add_new_rule(LineupConstructionRule)
