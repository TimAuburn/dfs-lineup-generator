"""Wrappers around pydfs-lineup-optimizer for DraftKings NFL Classic and Showdown."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pydfs_lineup_optimizer import Player, Site, Sport, get_optimizer
from pydfs_lineup_optimizer.lineup import Lineup
from pydfs_lineup_optimizer.player import GameInfo, LineupPlayer
from pydfs_lineup_optimizer.settings import LineupPosition
from pydfs_lineup_optimizer.stacks import PlayersGroup, PositionsStack

from .parser import drop_inactive_players, is_inactive_injury, parse_dk_csv
from .presets import ContestFormat

DK_SALARY_CAP = 50_000
# Full-cap tier in portfolio salary diversification (leave at most $200).
STANDARD_TIER_FLOOR = 49_800
CLASSIC_ROSTER_SLOTS = ("QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST")
SHOWDOWN_ROSTER_SLOTS = ("CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX")
# Backward-compatible alias used by the UI / older imports.
ROSTER_SLOTS = CLASSIC_ROSTER_SLOTS

# Default seed for projection jitter. Pass ``jitter_seed=None`` for non-deterministic draws.
DEFAULT_JITTER_SEED = 42

SKILL_POSITIONS = ("RB", "WR", "TE")
PASS_CATCHERS = ("WR", "TE")
BRING_BACK_POSITIONS = frozenset({"WR", "TE", "SKILL"})
# DraftKings FLEX accepts RB, WR, or TE. Large-field defaults drop TE.
FLEX_POSITIONS = frozenset({"RB", "WR", "TE"})
# Large-field GPP: short slates cannot share as many players as a main slate.
SMALL_SLATE_GAME_LIMIT = 10
SMALL_SLATE_REPEATING_CAP = 4
SHORT_SLATE_GAME_LIMIT = 4
SHORT_SLATE_STACK_EXPOSURE = 0.30
DEFAULT_STACK_EXPOSURE = 0.40
# Captain-QB FLEX quota is WR and TE only. RB, K, and DST do not count.
CPT_QB_PASS_CATCHER_POSITIONS = frozenset({"WR", "TE"})
_NOT_CPT_QB_PASS_CATCHERS = frozenset({"K", "DST", "DEF", "RB", "QB"})
# Opposing FLEX piece required with a Captain QB.
QB_BRING_BACK_POSITIONS = frozenset({"WR", "TE", "RB"})
# Captain is a skill spot unless a low-total slate explicitly allows a kicker.
CAPTAIN_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})
# Cash classic: no sub-$3,800 WR/RB/TE unless the user locks that player in.
CASH_SKILL_MIN_SALARY = 3_800
CASH_SKILL_POSITIONS = frozenset({"RB", "WR", "TE"})
# Showdown cash spends the cap and captains a high-salary QB, RB, or WR.
SHOWDOWN_CASH_MIN_SALARY = 49_600
SHOWDOWN_CASH_CAPTAIN_POSITIONS = frozenset({"QB", "RB", "WR"})
SHOWDOWN_CASH_CAPTAIN_MIN_SALARY = 8_000
SHOWDOWN_CASH_FLEX_MIN_SALARY = 3_500
# At most one rostered player may be priced under this. Stops a stars-and-scrubs cash build.
SHOWDOWN_CASH_BELOW_SALARY = 4_500
SHOWDOWN_CASH_MAX_BELOW_SALARY = 1
# Every Showdown lineup: no near-zero projections, and no sub-$1,200 FLEX punt.
SHOWDOWN_MIN_PROJECTION = 1.0
SHOWDOWN_FLEX_MIN_SALARY = 1_200
# Showdown backup QBs (Cooper Rush at 5.1) stay out of CPT and FLEX.
SHOWDOWN_QB_MIN_PROJECTION = 12.0


def _parse_game_info(
    raw: object,
    cache: dict,
) -> Optional[GameInfo]:
    """Parse DK Game Info and reuse one GameInfo instance per matchup."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text in {"In Progress", "Final"}:
        key = (None, None)
        if key not in cache:
            cache[key] = GameInfo(
                home_team=None, away_team=None, starts_at=None, game_started=True
            )
        return cache[key]
    try:
        teams, _date, _time, _tz = text.rsplit(" ", 3)
        away_team, home_team = teams.split("@")
        key = (home_team, away_team)
    except ValueError:
        return None
    if key not in cache:
        cache[key] = GameInfo(home_team=home_team, away_team=away_team, starts_at=None)
    return cache[key]


def _split_name(full_name: str) -> tuple[str, str]:
    parts = str(full_name).strip().split(maxsplit=1)
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def _position_tokens(value: object) -> set[str]:
    """Football positions from a slot label such as ``WR`` or ``RB/FLEX``."""
    tokens = set()
    for part in str(value).split("/"):
        token = part.strip().upper()
        if token and token not in {"FLEX", "CPT", "UTIL"}:
            tokens.add(token)
    return tokens


def unique_quarterbacks(df: pd.DataFrame) -> pd.DataFrame:
    """One row per quarterback, using the higher listed projection of CPT and FLEX."""
    empty = pd.DataFrame(columns=["Name", "TeamAbbrev", "Projection"])
    if df is None or df.empty or "Position" not in df.columns or "Name" not in df.columns:
        return empty
    is_qb = df["Position"].map(lambda value: "QB" in _position_tokens(value))
    qbs = df.loc[is_qb].copy()
    if qbs.empty or "TeamAbbrev" not in qbs.columns:
        return empty
    proj_col = "Projection" if "Projection" in qbs.columns else "AvgPointsPerGame"
    if proj_col not in qbs.columns:
        return empty
    qbs["Name"] = qbs["Name"].astype(str).str.strip()
    qbs["TeamAbbrev"] = qbs["TeamAbbrev"].astype(str).str.strip().str.upper()
    qbs["Projection"] = pd.to_numeric(qbs[proj_col], errors="coerce").fillna(0.0)
    grouped = (
        qbs.groupby(["Name", "TeamAbbrev"], as_index=False)["Projection"]
        .max()
        .sort_values(["Projection", "Name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return grouped


def quarterback_label(name: str, team: str, projection: float) -> str:
    """Dropdown label for one quarterback."""
    return f"{name} ({team}) — {float(projection):.2f}"


def default_starting_quarterbacks(
    df: pd.DataFrame,
    *,
    limit: int = 2,
) -> List[Tuple[str, str]]:
    """Highest-projected quarterback on each team, then the top ``limit`` of those.

    A Showdown slate is one game, so this is one starter from each team.
    """
    qbs = unique_quarterbacks(df)
    if qbs.empty:
        return []
    leaders = (
        qbs.sort_values(["Projection", "Name"], ascending=[False, True])
        .drop_duplicates("TeamAbbrev", keep="first")
        .sort_values(["Projection", "Name"], ascending=[False, True])
        .head(max(0, int(limit)))
    )
    return [(str(row.Name), str(row.TeamAbbrev)) for row in leaders.itertuples(index=False)]


def _player_positions(player: Player) -> set[str]:
    tokens: set[str] = set()
    for pos in player.positions:
        tokens.update(_position_tokens(pos))
    return tokens


def _opponent_team(player: Player) -> Optional[str]:
    """The other team in this player's game, when Game Info parsed."""
    info = getattr(player, "game_info", None)
    if info is None or not info.home_team or not info.away_team:
        return None
    if player.team == info.home_team:
        return str(info.away_team)
    if player.team == info.away_team:
        return str(info.home_team)
    return None


def count_slate_games(players: Sequence[Player]) -> int:
    """Unique parsed matchups on the slate."""
    games = set()
    for player in players:
        info = getattr(player, "game_info", None)
        if info is None or not info.home_team or not info.away_team:
            continue
        games.add((str(info.home_team), str(info.away_team)))
    return len(games)


def overlap_cap_for_slate(
    configured: Optional[int],
    n_games: int,
    slate_aware: bool,
) -> Optional[int]:
    """Cap shared players at 4 on slates with fewer than 10 games."""
    if configured is None:
        return None
    cap = int(configured)
    if slate_aware and int(n_games) < SMALL_SLATE_GAME_LIMIT:
        return min(cap, SMALL_SLATE_REPEATING_CAP)
    return cap


def stack_exposure_for_slate(n_games: int, slate_aware: bool) -> Optional[float]:
    """Limit how often one team's double stack can repeat across the batch."""
    if not slate_aware:
        return None
    if int(n_games) <= SHORT_SLATE_GAME_LIMIT:
        return SHORT_SLATE_STACK_EXPOSURE
    return DEFAULT_STACK_EXPOSURE


# Game-script portfolios share at most this many players across every bucket.
GAME_SCRIPT_MAX_REPEATING = 4
GAME_SCRIPT_FLEX = ("RB", "WR")


@dataclass(frozen=True)
class GameScriptBucket:
    """One matchup's share of a Classic portfolio."""

    game: str
    lineups: int
    salary_cap: int


def games_from_pool(pool: Optional[pd.DataFrame]) -> List[str]:
    """Distinct ``AWAY@HOME`` labels in first-seen order."""
    if pool is None or pool.empty or "Game Info" not in pool.columns:
        return []
    labels: List[str] = []
    seen = set()
    cache: Dict[str, Optional[GameInfo]] = {}
    for raw in pool["Game Info"].tolist():
        info = _parse_game_info(raw, cache)
        if info is None or not info.away_team or not info.home_team:
            continue
        label = f"{info.away_team}@{info.home_team}"
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def even_lineup_counts(n_games: int, total: int) -> List[int]:
    """Split ``total`` across games. Remainder lands on the first games."""
    n_games = int(n_games)
    total = max(0, int(total))
    if n_games < 1:
        return []
    base, extra = divmod(total, n_games)
    return [base + (1 if index < extra else 0) for index in range(n_games)]


def _teams_in_game(game: str) -> Tuple[str, str]:
    parts = str(game).split("@")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"game label must look like AWAY@HOME, got {game!r}")
    return parts[0].strip().upper(), parts[1].strip().upper()


def _without_outside_quarterbacks(
    players: Sequence[Player], teams: Tuple[str, str]
) -> List[Player]:
    """Drop quarterbacks who are not in this matchup. Everyone else stays."""
    allowed = {team.upper() for team in teams}
    kept: List[Player] = []
    for player in players:
        if "QB" in _player_positions(player) and str(player.team).upper() not in allowed:
            continue
        kept.append(player)
    return kept


def _lineups_for_overlap(
    frames: Sequence[pd.DataFrame], players: Sequence[Player]
) -> List[Lineup]:
    """Rebuild prior lineups from this bucket's pool so overlap can cross games.

    Players missing from the pool (the other game's quarterback) are skipped.
    They cannot be rostered here, so leaving them out of the cap does not
    let the remaining players overlap more than the limit.
    """
    by_id = {str(player.id): player for player in players}
    built: List[Lineup] = []
    for frame in frames:
        if frame is None or frame.empty or "ID" not in frame.columns:
            continue
        slots = (
            frame["Slot"].astype(str).tolist()
            if "Slot" in frame.columns
            else ["UTIL"] * len(frame)
        )
        chosen: List[LineupPlayer] = []
        for player_id, slot in zip(frame["ID"].astype(str).tolist(), slots):
            player = by_id.get(player_id)
            if player is None:
                continue
            chosen.append(LineupPlayer(player, slot))
        if chosen:
            built.append(Lineup(chosen))
    return built


def _ownership_fraction(raw: object) -> Optional[float]:
    """Normalize ownership values to a 0–1 fraction. Returns None if missing/invalid."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        text = str(raw).strip().replace("%", "")
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
    if value < 0:
        return None
    # Treat values > 1 as percentage points (e.g. 12.5 → 0.125).
    if value > 1.0:
        value = value / 100.0
    return value


def apply_projection_jitter(
    df: pd.DataFrame,
    *,
    jitter_pct: Optional[Tuple[float, float]] = None,
    position_jitter_sds: Optional[Dict[str, float]] = None,
    seed: Optional[int] = DEFAULT_JITTER_SEED,
    projection_col: str = "Projection",
) -> pd.DataFrame:
    """
    Return a copy of ``df`` with projection noise applied.

    Seed behavior
    -------------
    * ``seed=42`` (default): reproducible jitter across runs.
    * ``seed=<int>``: reproducible with that seed.
    * ``seed=None``: non-deterministic (fresh entropy each call).

    Noise modes
    -----------
    * ``position_jitter_sds``: Gaussian noise N(0, sd_pos) as a fraction of projection
      (used by Large-Field GPP). Falls back to the midpoint of ``jitter_pct`` when a
      position is missing from the map.
    * else ``jitter_pct=(lo, hi)``: each player gets a signed uniform multiplier in
      ``[lo, hi]`` (Cash uses ``(0.0, 0.0)`` for a pure deterministic solve).
    """
    out = df.copy()
    if projection_col not in out.columns:
        raise KeyError(f"Missing projection column: {projection_col}")

    use_gaussian = bool(position_jitter_sds)
    use_uniform = bool(jitter_pct) and not use_gaussian
    if not use_gaussian and not use_uniform:
        return out
    if jitter_pct is not None and jitter_pct == (0.0, 0.0) and not use_gaussian:
        return out

    rng = np.random.default_rng(seed)
    base = pd.to_numeric(out[projection_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    noise = np.zeros(len(out), dtype=float)

    if use_gaussian:
        assert position_jitter_sds is not None
        fallback = 0.15
        if jitter_pct is not None:
            fallback = (float(jitter_pct[0]) + float(jitter_pct[1])) / 2.0
        positions = out["Position"].astype(str).str.upper() if "Position" in out.columns else pd.Series([""] * len(out))
        for i, pos in enumerate(positions):
            primary = str(pos).split("/")[0].strip().upper()
            sd = float(position_jitter_sds.get(primary, fallback))
            noise[i] = float(rng.normal(0.0, sd))
    else:
        assert jitter_pct is not None
        lo, hi = float(jitter_pct[0]), float(jitter_pct[1])
        if lo == 0.0 and hi == 0.0:
            return out
        magnitude = rng.uniform(lo, hi, size=len(out))
        sign = rng.choice(np.array([-1.0, 1.0]), size=len(out))
        noise = magnitude * sign

    out[projection_col] = np.maximum(base * (1.0 + noise), 0.0)
    return out


def dataframe_to_players(
    df: pd.DataFrame,
    *,
    contest_format: ContestFormat = ContestFormat.CLASSIC,
    min_exposure: Optional[float] = None,
    max_exposure: Optional[float] = None,
) -> List[Player]:
    """Convert a parsed player-pool DataFrame into pydfs ``Player`` objects."""
    players: List[Player] = []
    game_cache: dict = {}
    for _, row in df.iterrows():
        first, last = _split_name(row["Name"])
        projection = float(row["Projection"] if "Projection" in df.columns else row["AvgPointsPerGame"])
        game_raw = row["Game Info"] if "Game Info" in df.columns else None
        injured = bool(
            "Injury" in df.columns and is_inactive_injury(row.get("Injury", ""))
        )
        ownership = None
        for col in ("Ownership", "Projected Ownership", "Own%", "Own"):
            if col in df.columns:
                ownership = _ownership_fraction(row[col])
                if ownership is not None:
                    break

        if contest_format is ContestFormat.SHOWDOWN:
            roster_pos = str(row.get("Roster Position", row.get("Position", "FLEX"))).strip().upper()
            slot = "CPT" if roster_pos == "CPT" else "FLEX"
            football_pos = str(row["Position"]).split("/")[0].strip().upper()
            if football_pos in {"CPT", "FLEX"}:
                # Fallback when Position only carries the slot label.
                football_pos = str(row.get("FootballPosition", "FLEX")).split("/")[0].strip().upper()
            fppg = projection * (1.5 if slot == "CPT" else 1.0)
            players.append(
                Player(
                    str(row["ID"]),
                    first,
                    last,
                    [slot],
                    str(row["TeamAbbrev"]),
                    float(row["Salary"]),
                    fppg,
                    game_info=_parse_game_info(game_raw, game_cache),
                    min_exposure=min_exposure,
                    max_exposure=max_exposure,
                    projected_ownership=ownership,
                    is_injured=injured,
                    original_positions=[football_pos] if football_pos else None,
                )
            )
        else:
            players.append(
                Player(
                    str(row["ID"]),
                    first,
                    last,
                    [str(row["Position"])],
                    str(row["TeamAbbrev"]),
                    float(row["Salary"]),
                    projection,
                    game_info=_parse_game_info(game_raw, game_cache),
                    min_exposure=min_exposure,
                    max_exposure=max_exposure,
                    projected_ownership=ownership,
                    is_injured=injured,
                )
            )
    return players


class BaseNFLOptimizer:
    """Shared load / exposure / jitter / format helpers."""

    contest_format: str = ContestFormat.CLASSIC.value
    roster_slots: Sequence[str] = CLASSIC_ROSTER_SLOTS
    site = Site.DRAFTKINGS
    sport = Sport.FOOTBALL
    salary_cap: int = DK_SALARY_CAP

    def __init__(self) -> None:
        self._players: List[Player] = []
        self._player_df: Optional[pd.DataFrame] = None
        self.lineup_count: int = 1
        self.min_exposure: Optional[float] = None
        self.max_exposure: Optional[float] = None
        self.max_repeating_players: Optional[int] = None
        self.exclude_vs_dst: bool = False
        self.max_cumulative_ownership: Optional[float] = None
        self.jitter_pct: Optional[Tuple[float, float]] = None
        self.position_jitter_sds: Optional[Dict[str, float]] = None
        self.jitter_seed: Optional[int] = DEFAULT_JITTER_SEED
        self.min_salary_cap: Optional[int] = None
        self.max_salary_cap: Optional[int] = None
        # Drop OUT / IR / Doubtful players. Questionable players stay eligible.
        self.exclude_inactive: bool = True
        # Player IDs the user forced into the lineup. Salary floors skip these.
        self.locked_player_ids: set[str] = set()

    def set_locked_players(self, player_ids: Optional[Sequence[str]]) -> None:
        """Force these player IDs into the lineup. Empty clears the locks."""
        self.locked_player_ids = {str(player_id) for player_id in (player_ids or [])}

    def load_from_csv(self, path: Union[str, Path]) -> pd.DataFrame:
        df = parse_dk_csv(path, contest_format=ContestFormat(self.contest_format))
        return self.load_from_dataframe(df)

    def set_exclude_inactive(self, exclude: bool) -> None:
        """When true, players who will not play are removed before the solve."""
        self.exclude_inactive = bool(exclude)

    def load_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        pool = df.copy()
        if self.exclude_inactive:
            pool = drop_inactive_players(pool)
            if pool.empty:
                raise RuntimeError(
                    "No active players left after excluding players who will not play."
                )
        self._player_df = pool
        self._players = dataframe_to_players(
            self._player_df,
            contest_format=ContestFormat(self.contest_format),
            min_exposure=self.min_exposure,
            max_exposure=self.max_exposure,
        )
        return self._player_df

    def set_lineup_count(self, n: int) -> None:
        if not 1 <= int(n) <= 150:
            raise ValueError("lineup_count must be between 1 and 150")
        self.lineup_count = int(n)

    def set_exposure_limits(
        self,
        min_exposure_pct: Optional[float] = None,
        max_exposure_pct: Optional[float] = None,
    ) -> None:
        """Set global min/max player exposure as percentages (0–100)."""
        self.min_exposure = None if min_exposure_pct is None else float(min_exposure_pct) / 100.0
        self.max_exposure = None if max_exposure_pct is None else float(max_exposure_pct) / 100.0
        if self.min_exposure is not None and not 0.0 <= self.min_exposure <= 1.0:
            raise ValueError("min exposure must be between 0 and 100")
        if self.max_exposure is not None and not 0.0 <= self.max_exposure <= 1.0:
            raise ValueError("max exposure must be between 0 and 100")
        if (
            self.min_exposure is not None
            and self.max_exposure is not None
            and self.min_exposure > self.max_exposure
        ):
            raise ValueError("min exposure cannot exceed max exposure")
        if self._player_df is not None:
            self.load_from_dataframe(self._player_df)

    def set_max_repeating_players(self, n: Optional[int]) -> None:
        if n is not None and n < 0:
            raise ValueError("max_repeating_players must be >= 0")
        self.max_repeating_players = n

    def set_min_salary_cap(self, min_salary: Optional[int]) -> None:
        """Minimum roster salary. ``None`` removes the floor."""
        if min_salary is None:
            self.min_salary_cap = None
            return
        value = int(min_salary)
        if value < 0:
            raise ValueError("min salary cap must be >= 0")
        self.min_salary_cap = value

    def set_max_salary_cap(self, max_salary: Optional[int]) -> None:
        """Maximum roster salary (solver budget). ``None`` uses the site cap."""
        if max_salary is None:
            self.max_salary_cap = None
            return
        value = int(max_salary)
        if value <= 0:
            raise ValueError("max salary cap must be positive")
        self.max_salary_cap = value

    def _apply_salary_caps(self, pydfs_optimizer) -> None:
        """Push the current salary window onto a pydfs optimizer."""
        if self.max_salary_cap is not None:
            pydfs_optimizer.settings.budget = int(self.max_salary_cap)
        if self.min_salary_cap is not None:
            pydfs_optimizer.set_min_salary_cap(float(self.min_salary_cap))

    def _apply_injury_filter(self, pydfs_optimizer) -> None:
        """Keep injured players only when the user has chosen to include them."""
        pydfs_optimizer.player_pool.with_injured = not self.exclude_inactive

    def _apply_locked_players(self, optimizer) -> None:
        """Force user-locked players in after salary floors have removed punts."""
        if not self.locked_player_ids:
            return
        removed = optimizer.player_pool.removed_players
        for player_id in self.locked_player_ids:
            player = optimizer.player_pool.get_player_by_id(player_id)
            if player is None or player in removed:
                continue
            if player in optimizer.player_pool.locked_players:
                continue
            optimizer.player_pool.lock_player(player)

    def _players_for_optimize(self) -> List[Player]:
        if self._player_df is None or not self._players:
            raise RuntimeError("No players loaded. Call load_from_csv or load_from_dataframe first.")
        jittered = apply_projection_jitter(
            self._player_df,
            jitter_pct=self.jitter_pct,
            position_jitter_sds=self.position_jitter_sds,
            seed=self.jitter_seed,
        )
        return dataframe_to_players(
            jittered,
            contest_format=ContestFormat(self.contest_format),
            min_exposure=self.min_exposure,
            max_exposure=self.max_exposure,
        )

    def _apply_ownership_cap(self, optimizer) -> None:
        """Cumulative ownership ≤ cap if ownership is present; otherwise no-op."""
        if self.max_cumulative_ownership is None:
            return
        owned = [p for p in optimizer.players if p.projected_ownership is not None]
        if not owned:
            return
        roster_size = len(self.roster_slots)
        # pydfs enforces average ownership ≤ max_projected_ownership.
        avg_cap = float(self.max_cumulative_ownership) / float(roster_size)
        optimizer.set_projected_ownership(max_projected_ownership=avg_cap)

    def _apply_exclude_vs_dst(self, optimizer) -> None:
        if not self.exclude_vs_dst:
            return
        if not optimizer.player_pool.games:
            return
        optimizer.restrict_positions_for_opposing_team(
            ["DST"],
            list(SKILL_POSITIONS),
            max_allowed=0,
        )

    def format_lineup(self, lineup) -> pd.DataFrame:
        """Format a pydfs Lineup as a roster DataFrame."""
        rows = []
        for player in lineup.lineup:
            football_pos = (
                "/".join(player.original_positions)
                if getattr(player, "original_positions", None)
                else "/".join(player.positions)
            )
            rows.append(
                {
                    "Slot": player.lineup_position,
                    "ID": player.id,
                    "Name": player.full_name,
                    "Name + ID": f"{player.full_name} ({player.id})",
                    "Position": football_pos,
                    "Team": player.team,
                    "Salary": int(player.salary),
                    "Projection": float(player.fppg),
                    "Opponent": _opponent_team(player) or "",
                }
            )
        df = pd.DataFrame(rows)
        spent = int(lineup.salary_costs)
        df.attrs["salary"] = spent
        df.attrs["projection"] = float(lineup.fantasy_points_projection)
        df.attrs["salary_remaining"] = int(self.salary_cap) - spent
        return df

    def generate_lineups_table(self, n: Optional[int] = None) -> pd.DataFrame:
        """Generate lineups and return a single multi-lineup summary table."""
        lineups = self.generate_lineups(n=n)
        frames = []
        for idx, lu in enumerate(lineups, start=1):
            frame = lu.copy()
            frame.insert(0, "Lineup", idx)
            frame["LineupSalary"] = lu.attrs.get("salary")
            frame["LineupProjection"] = lu.attrs.get("projection")
            frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=["Lineup", *self.roster_slots])
        return pd.concat(frames, ignore_index=True)

    def generate_lineups(self, n: Optional[int] = None) -> List[pd.DataFrame]:
        raise NotImplementedError


def drop_backup_players(pool: pd.DataFrame) -> pd.DataFrame:
    """Keep the top Sleeper depth-chart player at each spot.

    Quarterback, running back, tight end, and kicker each keep their
    depth-chart leader. Wide receivers keep the leader at left, right, and
    slot. Defenses stay. When a team has no depth chart, every skill player
    stays except extra quarterbacks, and the highest projection is the one
    who remains.
    """
    if pool.empty or "Position" not in pool.columns or "TeamAbbrev" not in pool.columns:
        return pool
    positions = pool["Position"].map(_fantasy_position)
    if "DepthOrder" not in pool.columns or not pd.to_numeric(pool["DepthOrder"], errors="coerce").notna().any():
        return _one_quarterback_per_team(pool, positions)

    depth_order = pd.to_numeric(pool["DepthOrder"], errors="coerce")
    if "DepthPosition" in pool.columns:
        depth_pos = pool["DepthPosition"].astype(str).str.strip().str.upper()
    else:
        depth_pos = pd.Series("", index=pool.index)
    depth_pos = depth_pos.where(depth_order.notna(), "")
    teams = pool["TeamAbbrev"].astype(str).str.strip().str.upper()
    keep = positions.eq("DST").copy()
    slotted = depth_pos.ne("") & depth_order.notna()
    if slotted.any():
        slotted_frame = pd.DataFrame(
            {
                "_team": teams.loc[slotted],
                "_slot": depth_pos.loc[slotted],
                "_order": depth_order.loc[slotted],
            },
            index=pool.index[slotted],
        )
        for _, group in slotted_frame.groupby(["_team", "_slot"], sort=False):
            best = group["_order"].min()
            keep.loc[group.index[group["_order"] == best]] = True
    unslotted = ~slotted & ~positions.eq("DST")
    for idx in pool.index[unslotted]:
        charted = slotted & teams.eq(teams.loc[idx]) & positions.eq(positions.loc[idx])
        if not bool(charted.any()):
            keep.loc[idx] = True
    return _one_quarterback_per_team(pool.loc[keep].copy(), positions.loc[keep])


def _fantasy_position(value: object) -> str:
    token = str(value or "").split("/")[0].strip().upper()
    if token in {"DEF", "DST"}:
        return "DST"
    if token == "FB":
        return "RB"
    return token


def _one_quarterback_per_team(pool: pd.DataFrame, positions: pd.Series) -> pd.DataFrame:
    """Drop extra quarterbacks on teams that have no Sleeper depth chart."""
    if pool.empty:
        return pool
    is_qb = positions.reindex(pool.index).eq("QB")
    if not bool(is_qb.any()):
        return pool
    if "DepthOrder" in pool.columns:
        depth = pd.to_numeric(pool["DepthOrder"], errors="coerce")
    else:
        depth = pd.Series(pd.NA, index=pool.index, dtype="Float64")
    teams = pool["TeamAbbrev"].astype(str).str.strip().str.upper()
    proj_col = "Projection" if "Projection" in pool.columns else "AvgPointsPerGame"
    if proj_col not in pool.columns:
        return pool
    projection = pd.to_numeric(pool[proj_col], errors="coerce").fillna(0.0)
    drop_indexes: list = []
    quarterbacks = pool.loc[is_qb]
    for _, group_indexes in quarterbacks.groupby(teams.loc[is_qb], sort=False).groups.items():
        indexes = list(group_indexes)
        if depth.loc[indexes].notna().any() or len(indexes) <= 1:
            continue
        best = max(indexes, key=lambda idx: (float(projection.loc[idx]), str(pool.loc[idx, "Name"])))
        drop_indexes.extend(idx for idx in indexes if idx != best)
    if not drop_indexes:
        return pool
    return pool.drop(index=drop_indexes)


class NFLClassicOptimizer(BaseNFLOptimizer):
    """Configurable DraftKings NFL Classic lineup generator."""

    contest_format = ContestFormat.CLASSIC.value
    roster_slots = CLASSIC_ROSTER_SLOTS
    site = Site.DRAFTKINGS

    def __init__(self) -> None:
        super().__init__()
        self.enable_qb_stack: bool = False
        self.stack_pass_catchers: int = 1
        self.enable_bring_back: bool = False
        self.bring_back_position: str = "WR"
        # Double stacks must include a WR. Single stacks stay WR or TE.
        self.require_wr_in_stack: bool = False
        # Short slates tighten overlap and team-stack exposure.
        self.slate_aware_gpp: bool = False
        self.last_game_count: int = 0
        self.last_overlap_cap: Optional[int] = None
        self.last_stack_exposure: Optional[float] = None
        # None keeps the DraftKings flex (RB, WR, or TE).
        self.flex_positions: Optional[Tuple[str, ...]] = None
        # None leaves cheap skill players eligible. Cash sets this to $3,800.
        self.min_skill_salary: Optional[int] = None

    def set_qb_stack(
        self,
        enabled: bool = True,
        pass_catchers: int = 1,
        bring_back: bool = False,
        bring_back_position: str = "WR",
    ) -> None:
        """
        Configure QB + WR/TE primary stacking and optional opposing bring-back.

        ``pass_catchers`` is the number of same-team WR/TE stacked with the QB
        (typically 1 or 2). Bring-back forces an opposing player from the
        quarterback's game. ``SKILL`` accepts WR, TE, or RB.
        """
        if pass_catchers < 1:
            raise ValueError("pass_catchers must be at least 1")
        if bring_back_position not in BRING_BACK_POSITIONS:
            raise ValueError("bring_back_position must be WR, TE, or SKILL")
        self.enable_qb_stack = bool(enabled)
        self.stack_pass_catchers = int(pass_catchers)
        self.enable_bring_back = bool(bring_back)
        self.bring_back_position = bring_back_position

    def _qb_stack_positions(self) -> List[Union[str, tuple]]:
        """QB + WR + remaining WR/TE when a double stack must include a WR."""
        if self.require_wr_in_stack and self.stack_pass_catchers >= 2:
            positions: List[Union[str, tuple]] = ["QB", "WR"]
            for _ in range(self.stack_pass_catchers - 1):
                positions.append(("WR", "TE"))
            return positions
        positions = ["QB"]
        for _ in range(self.stack_pass_catchers):
            positions.append(("WR", "TE"))
        return positions

    def _apply_bring_back(self, optimizer, players: List[Player]) -> None:
        """Pair the rostered QB with an opposing skill player from his game."""
        if self.bring_back_position != "SKILL":
            optimizer.force_positions_for_opposing_team(("QB", self.bring_back_position))
            return
        allowed = set(SKILL_POSITIONS)
        for qb in players:
            if "QB" not in _player_positions(qb):
                continue
            opponent = _opponent_team(qb)
            bring_backs = [
                player
                for player in players
                if opponent
                and player.team == opponent
                and _player_positions(player) & allowed
            ]
            if not bring_backs:
                optimizer.player_pool.remove_player(qb)
                continue
            optimizer.add_players_group(
                PlayersGroup(
                    bring_backs,
                    min_from_group=1,
                    max_from_group=len(bring_backs),
                    depends_on=qb,
                    strict_depend=False,
                )
            )

    def _build_optimizer(self, players: List[Player]):
        optimizer = get_optimizer(self.site, self.sport)
        self._apply_salary_caps(optimizer)
        self._apply_injury_filter(optimizer)
        optimizer.player_pool.load_players(players)
        n_games = count_slate_games(players)
        overlap_cap = overlap_cap_for_slate(
            self.max_repeating_players, n_games, self.slate_aware_gpp
        )
        stack_exposure = stack_exposure_for_slate(n_games, self.slate_aware_gpp)
        self.last_game_count = n_games
        self.last_overlap_cap = overlap_cap
        self.last_stack_exposure = stack_exposure
        if overlap_cap is not None:
            optimizer.set_max_repeating_players(overlap_cap)
        if self.enable_qb_stack:
            optimizer.add_stack(
                PositionsStack(
                    self._qb_stack_positions(),
                    max_exposure=stack_exposure,
                )
            )
        if self.enable_bring_back:
            self._apply_bring_back(optimizer, players)
        self._apply_flex_positions(optimizer)
        self._apply_skill_salary_floor(optimizer, players)
        self._apply_locked_players(optimizer)
        self._apply_exclude_vs_dst(optimizer)
        self._apply_ownership_cap(optimizer)
        return optimizer

    def _apply_skill_salary_floor(self, optimizer, players: List[Player]) -> None:
        """Drop WR, RB, and TE priced under the cash floor. Locked players stay."""
        floor = self.min_skill_salary
        if floor is None:
            return
        locked = self.locked_player_ids
        for player in players:
            if str(player.id) in locked:
                continue
            if not (_player_positions(player) & CASH_SKILL_POSITIONS):
                continue
            if float(player.salary) < int(floor):
                optimizer.player_pool.remove_player(player)

    def _apply_flex_positions(self, optimizer) -> None:
        """Limit the FLEX slot so a second tight end cannot fill it."""
        allowed = self.flex_positions
        if not allowed:
            return
        chosen = tuple(dict.fromkeys(str(pos).upper() for pos in allowed))
        if not chosen or not set(chosen) <= FLEX_POSITIONS:
            raise ValueError("flex positions must be RB, WR, or TE")
        if set(chosen) == FLEX_POSITIONS:
            return
        flex = LineupPosition("FLEX", chosen)
        positions = [
            flex if pos.name == "FLEX" else pos
            for pos in list(optimizer.settings.positions)
        ]
        # Instance assignment leaves the shared DraftKings settings list alone.
        optimizer.settings.positions = positions
        optimizer.player_pool._remaining_positions = list(positions)

    def _players_for_optimize(self) -> List[Player]:
        original = self._player_df
        if original is None:
            return super()._players_for_optimize()
        trimmed = drop_backup_players(original)
        if trimmed.empty:
            raise RuntimeError("No starters left after removing backup players.")
        self._player_df = trimmed
        try:
            return super()._players_for_optimize()
        finally:
            self._player_df = original

    def generate_lineups(
        self,
        n: Optional[int] = None,
        *,
        existing_lineups: Optional[Sequence[pd.DataFrame]] = None,
        quarterback_teams: Optional[Tuple[str, str]] = None,
        game_script: Optional[str] = None,
    ) -> List[pd.DataFrame]:
        """Generate lineups and return each as a DataFrame of rostered players.

        ``existing_lineups`` are earlier lineups from this portfolio. Shared
        players with those lineups stay at or under ``max_repeating_players``.
        ``quarterback_teams`` keeps only that game's quarterbacks.
        """
        count = self.lineup_count if n is None else int(n)
        if not 1 <= count <= 150:
            raise ValueError("lineup count must be between 1 and 150")
        players = self._players_for_optimize()
        if quarterback_teams is not None:
            players = _without_outside_quarterbacks(players, quarterback_teams)
            if not any("QB" in _player_positions(player) for player in players):
                label = game_script or "@".join(quarterback_teams)
                raise RuntimeError(f"No quarterbacks left for {label}.")
        optimizer = self._build_optimizer(players)
        exclude = _lineups_for_overlap(existing_lineups or [], players)
        raw_lineups = list(
            optimizer.optimize(
                count,
                max_exposure=self.max_exposure,
                exclude_lineups=exclude or None,
            )
        )
        frames = [self.format_lineup(lineup) for lineup in raw_lineups]
        if game_script:
            for frame in frames:
                frame.attrs["game_script"] = game_script
        return frames


class NFLShowdownOptimizer(BaseNFLOptimizer):
    """DraftKings NFL Showdown (Captain Mode) lineup generator."""

    contest_format = ContestFormat.SHOWDOWN.value
    roster_slots = SHOWDOWN_ROSTER_SLOTS
    site = Site.DRAFTKINGS_CAPTAIN_MODE
    salary_cap = DK_SALARY_CAP

    def __init__(self) -> None:
        super().__init__()
        self.min_salary_cap: Optional[int] = 48_000
        self.max_salary_cap: Optional[int] = 49_800
        self.min_teams: Optional[int] = 2
        self.game_script_correlation: bool = True
        self.cpt_qb_pass_catchers: int = 2
        self.max_k_dst_combined: Optional[int] = 2
        # CPT + FLEX combined. None leaves the kicker count uncapped.
        self.max_kickers: Optional[int] = 1
        self.allow_kicker_captain: bool = False
        # Fraction of the batch a kicker may occupy. None leaves exposure uncapped.
        self.kicker_max_exposure: Optional[float] = None
        # Showdown cash narrows CPT to QB/RB/WR at a base-salary floor.
        self.captain_positions: Optional[frozenset] = None
        self.captain_min_salary: Optional[int] = None
        # FLEX rows under this salary are punts. None leaves them eligible.
        self.flex_min_salary: Optional[int] = None
        # None leaves the defense count uncapped. Cash bans defenses instead.
        self.max_dst: Optional[int] = None
        self.ban_dst: bool = False
        # Cash: at most one player whose base salary is under the threshold.
        self.max_below_salary_players: Optional[int] = None
        self.below_salary_threshold: Optional[int] = None
        # Cash: at most this many running backs from one team. None leaves it uncapped.
        self.max_rbs_per_team: Optional[int] = None
        # Showdown batches share at most 3 players so lineups do not collapse together.
        self.max_repeating_players = 3
        # None keeps every QB who clears the projection floor. A set keeps only those.
        self.confirmed_quarterbacks: Optional[set[Tuple[str, str]]] = None

    def configure_showdown_rules(
        self,
        *,
        min_salary_cap: Optional[int] = 48_000,
        max_salary_cap: Optional[int] = 49_800,
        min_teams: Optional[int] = 2,
        game_script_correlation: bool = True,
        cpt_qb_pass_catchers: int = 2,
        max_k_dst_combined: Optional[int] = 2,
        max_kickers: Optional[int] = 1,
        allow_kicker_captain: bool = False,
        captain_positions: Optional[Sequence[str]] = None,
        captain_min_salary: Optional[int] = None,
        flex_min_salary: Optional[int] = None,
        max_dst: Optional[int] = None,
        ban_dst: bool = False,
        max_below_salary_players: Optional[int] = None,
        below_salary_threshold: Optional[int] = None,
        max_rbs_per_team: Optional[int] = None,
    ) -> None:
        if cpt_qb_pass_catchers < 1:
            raise ValueError("cpt_qb_pass_catchers must be at least 1")
        if max_kickers is not None and int(max_kickers) < 0:
            raise ValueError("max_kickers must be at least 0")
        if max_dst is not None and int(max_dst) < 0:
            raise ValueError("max_dst must be at least 0")
        if captain_min_salary is not None and int(captain_min_salary) < 0:
            raise ValueError("captain_min_salary must be at least 0")
        if flex_min_salary is not None and int(flex_min_salary) < 0:
            raise ValueError("flex_min_salary must be at least 0")
        if max_below_salary_players is not None and int(max_below_salary_players) < 0:
            raise ValueError("max_below_salary_players must be at least 0")
        if below_salary_threshold is not None and int(below_salary_threshold) < 0:
            raise ValueError("below_salary_threshold must be at least 0")
        if max_rbs_per_team is not None and int(max_rbs_per_team) < 0:
            raise ValueError("max_rbs_per_team must be at least 0")
        self.min_salary_cap = min_salary_cap
        self.max_salary_cap = max_salary_cap
        self.min_teams = min_teams
        self.game_script_correlation = bool(game_script_correlation)
        self.cpt_qb_pass_catchers = int(cpt_qb_pass_catchers)
        self.max_k_dst_combined = max_k_dst_combined
        self.max_kickers = None if max_kickers is None else int(max_kickers)
        self.allow_kicker_captain = bool(allow_kicker_captain)
        self.captain_positions = (
            None
            if captain_positions is None
            else frozenset(str(pos).upper() for pos in captain_positions)
        )
        self.captain_min_salary = (
            None if captain_min_salary is None else int(captain_min_salary)
        )
        self.flex_min_salary = None if flex_min_salary is None else int(flex_min_salary)
        self.max_dst = None if max_dst is None else int(max_dst)
        self.ban_dst = bool(ban_dst)
        self.max_below_salary_players = (
            None if max_below_salary_players is None else int(max_below_salary_players)
        )
        self.below_salary_threshold = (
            None if below_salary_threshold is None else int(below_salary_threshold)
        )
        self.max_rbs_per_team = None if max_rbs_per_team is None else int(max_rbs_per_team)

    def set_confirmed_quarterbacks(
        self,
        players: Optional[Sequence[Tuple[str, str]]],
    ) -> None:
        """Keep only these (name, team) quarterbacks. ``None`` clears the name filter.

        Quarterbacks under 12.0 points are still removed.
        """
        if players is None:
            self.confirmed_quarterbacks = None
            return
        self.confirmed_quarterbacks = {
            (str(name).strip(), str(team).strip().upper()) for name, team in players
        }

    def _without_backup_quarterbacks(self, pool: pd.DataFrame) -> pd.DataFrame:
        """Drop Showdown QBs whose listed projection is under 12.0.

        The check uses the projection on the salary file, before jitter and
        before the 1.5x captain boost, so a 10-point backup cannot survive as
        a 15-point captain row.
        """
        if pool.empty or "Position" not in pool.columns:
            return pool
        proj_col = "Projection" if "Projection" in pool.columns else "AvgPointsPerGame"
        if proj_col not in pool.columns:
            return pool
        projection = pd.to_numeric(pool[proj_col], errors="coerce")
        is_qb = pool["Position"].map(lambda value: "QB" in _position_tokens(value))
        return pool.loc[~(is_qb & (projection < SHOWDOWN_QB_MIN_PROJECTION))].copy()

    def _players_for_optimize(self) -> List[Player]:
        original = self._player_df
        if original is None:
            return super()._players_for_optimize()
        self._player_df = self._without_backup_quarterbacks(original)
        try:
            if self._player_df.empty:
                raise RuntimeError(
                    "No players left after removing quarterbacks under "
                    f"{SHOWDOWN_QB_MIN_PROJECTION:.1f} points."
                )
            return super()._players_for_optimize()
        finally:
            self._player_df = original

    @staticmethod
    def _football_pos(player: Player) -> str:
        """Primary football position, ignoring the CPT/FLEX roster slot."""
        stored = getattr(player, "_original_positions", None)
        raw = str(stored[0]) if stored else ""
        primary = raw.split("/")[0].strip().upper()
        if primary in {"CPT", "FLEX", "UTIL"}:
            return ""
        if primary == "DEF":
            return "DST"
        return primary

    def _counts_as_cpt_qb_pass_catcher(self, player: Player) -> bool:
        """FLEX WR or TE only. Kickers, defenses, and running backs do not count."""
        if "FLEX" not in player.positions:
            return False
        pos = self._football_pos(player)
        if pos in _NOT_CPT_QB_PASS_CATCHERS:
            return False
        return pos in CPT_QB_PASS_CATCHER_POSITIONS

    def _counts_as_qb_bring_back(self, player: Player, captain_team: str) -> bool:
        """Opposing FLEX WR, TE, or RB."""
        if "FLEX" not in player.positions or player.team == captain_team:
            return False
        return self._football_pos(player) in QB_BRING_BACK_POSITIONS

    def _captain_positions_allowed(self) -> frozenset:
        if self.captain_positions is not None:
            return frozenset(self.captain_positions)
        allowed = set(CAPTAIN_POSITIONS)
        if self.allow_kicker_captain:
            allowed.add("K")
        return frozenset(allowed)

    @staticmethod
    def _listed_salary(player: Player) -> int:
        """DraftKings base salary. Captain rows are stored at 1.5x."""
        salary = float(player.salary)
        if "CPT" in player.positions:
            salary = salary / 1.5
        return int(round(salary))

    @staticmethod
    def _baseline_projection(player: Player) -> float:
        """Projection before the 1.5x captain boost."""
        points = float(player.fppg)
        if "CPT" in player.positions:
            return points / 1.5
        return points

    def _apply_kicker_exposure_cap(self, players: List[Player]) -> None:
        cap = self.kicker_max_exposure
        if cap is None:
            return
        for player in players:
            if self._football_pos(player) != "K":
                continue
            if player.max_exposure is None or player.max_exposure > cap:
                player.max_exposure = cap

    def _apply_captain_eligibility(self, optimizer, players: List[Player]) -> None:
        """Keep CPT on the allowed positions. Cash also requires a base-salary floor."""
        allowed = self._captain_positions_allowed()
        minimum = self.captain_min_salary
        for player in players:
            if "CPT" not in player.positions:
                continue
            if self._football_pos(player) not in allowed:
                optimizer.player_pool.remove_player(player)
                continue
            if minimum is not None and self._listed_salary(player) < int(minimum):
                optimizer.player_pool.remove_player(player)

    def _apply_projection_floor(self, optimizer, players: List[Player]) -> None:
        """Drop players whose own projection is 1.0 or less, before the captain boost."""
        for player in players:
            if self._baseline_projection(player) <= SHOWDOWN_MIN_PROJECTION:
                optimizer.player_pool.remove_player(player)

    def _apply_confirmed_quarterbacks(self, optimizer, players: List[Player]) -> None:
        """Drop quarterbacks the user did not confirm as starters."""
        allowed = self.confirmed_quarterbacks
        if allowed is None:
            return
        for player in players:
            if self._football_pos(player) != "QB":
                continue
            key = (player.full_name.strip(), str(player.team).strip().upper())
            if key not in allowed:
                optimizer.player_pool.remove_player(player)

    def _flex_salary_floor(self) -> int:
        """Showdown FLEX floor. Cash can raise it. A lock is the only way under $1,200."""
        configured = self.flex_min_salary
        floor = SHOWDOWN_FLEX_MIN_SALARY if configured is None else int(configured)
        return max(SHOWDOWN_FLEX_MIN_SALARY, floor)

    def _apply_flex_salary_floor(self, optimizer, players: List[Player]) -> None:
        """Drop punts priced under the floor. A locked row is the explicit exception.

        The check uses base salary, so a $200 player's $300 captain row is the
        same punt and is removed with the FLEX row.
        """
        floor = self._flex_salary_floor()
        locked = self.locked_player_ids
        for player in players:
            if self._listed_salary(player) >= floor:
                continue
            if str(player.id) in locked:
                continue
            optimizer.player_pool.remove_player(player)

    def _apply_dst_ban(self, optimizer, players: List[Player]) -> None:
        """Cash lineups are QB, RB, WR, TE, and K. Defenses are removed."""
        if not self.ban_dst:
            return
        for player in players:
            if self._football_pos(player) == "DST":
                optimizer.player_pool.remove_player(player)

    def _apply_cheap_player_cap(self, optimizer, players: List[Player]) -> None:
        """At most one player priced under the cash threshold on the six-man roster.

        Both the captain and flex rows of a cheap player count, so a $4,000
        player cannot be paired with a second player under the threshold.
        """
        if self.max_below_salary_players is None or self.below_salary_threshold is None:
            return
        threshold = int(self.below_salary_threshold)
        cheap = [
            player
            for player in players
            if self._listed_salary(player) < threshold
        ]
        cap = int(self.max_below_salary_players)
        if len(cheap) <= cap:
            return
        optimizer.add_players_group(PlayersGroup(cheap, max_from_group=cap))

    def _apply_same_team_rb_limit(self, optimizer, players: List[Player]) -> None:
        """At most ``max_rbs_per_team`` running backs from one team.

        Captain and flex rows of the same back are separate players in the
        pool, and a lineup can take only one of those rows. Capping the rows
        from a team at the same number caps the backs.
        """
        if self.max_rbs_per_team is None:
            return
        cap = int(self.max_rbs_per_team)
        by_team: Dict[str, List[Player]] = {}
        for player in players:
            if self._football_pos(player) != "RB":
                continue
            by_team.setdefault(player.team, []).append(player)
        for team_rbs in by_team.values():
            distinct = {player.full_name for player in team_rbs}
            if len(distinct) <= cap:
                continue
            optimizer.add_players_group(PlayersGroup(team_rbs, max_from_group=cap))

    def _apply_game_script(self, optimizer, players: List[Player]) -> None:
        if not self.game_script_correlation:
            return

        cpt_players = [p for p in players if "CPT" in p.positions]
        flex_players = [p for p in players if "FLEX" in p.positions]

        for cpt in cpt_players:
            pos = self._football_pos(cpt)
            if pos == "QB":
                catchers = [
                    p
                    for p in flex_players
                    if p.team == cpt.team and self._counts_as_cpt_qb_pass_catcher(p)
                ]
                needed = int(self.cpt_qb_pass_catchers)
                bring_backs = [
                    p
                    for p in flex_players
                    if self._counts_as_qb_bring_back(p, cpt.team)
                ]
                if len(catchers) < needed or not bring_backs:
                    # K and DST are not catchers, and a QB captain needs an opposing skill player.
                    optimizer.player_pool.remove_player(cpt)
                    continue
                optimizer.add_players_group(
                    PlayersGroup(
                        catchers,
                        min_from_group=needed,
                        max_from_group=len(catchers),
                        depends_on=cpt,
                        strict_depend=False,
                    )
                )
                optimizer.add_players_group(
                    PlayersGroup(
                        bring_backs,
                        min_from_group=1,
                        max_from_group=len(bring_backs),
                        depends_on=cpt,
                        strict_depend=False,
                    )
                )
            elif pos in PASS_CATCHERS:
                qbs = [
                    p
                    for p in flex_players
                    if p.team == cpt.team and self._football_pos(p) == "QB"
                ]
                if qbs:
                    optimizer.add_players_group(
                        PlayersGroup(
                            qbs,
                            min_from_group=1,
                            max_from_group=len(qbs),
                            depends_on=cpt,
                            strict_depend=False,
                        )
                    )

        if self.max_k_dst_combined is not None:
            k_dst = [
                p
                for p in players
                if self._football_pos(p) in {"K", "DST", "DEF"}
            ]
            if k_dst:
                optimizer.add_players_group(
                    PlayersGroup(
                        k_dst,
                        min_from_group=0,
                        max_from_group=self.max_k_dst_combined,
                    )
                )

    def _apply_kicker_limit(self, optimizer, players: List[Player]) -> None:
        """Keep CPT and FLEX to at most ``max_kickers`` kickers combined.

        Captain mode roster slots are CPT and FLEX, so a football-position
        cap is a players group over every kicker row. ``max_from_group``
        with no minimum encodes sum(kickers) <= max.
        """
        if self.max_kickers is None:
            return
        kickers = [p for p in players if self._football_pos(p) == "K"]
        if len(kickers) <= int(self.max_kickers):
            return
        optimizer.add_players_group(
            PlayersGroup(kickers, max_from_group=int(self.max_kickers))
        )

    def _apply_dst_limit(self, optimizer, players: List[Player]) -> None:
        """Keep CPT and FLEX to at most ``max_dst`` defenses combined."""
        if self.max_dst is None:
            return
        defenses = [p for p in players if self._football_pos(p) == "DST"]
        if len(defenses) <= int(self.max_dst):
            return
        optimizer.add_players_group(
            PlayersGroup(defenses, max_from_group=int(self.max_dst))
        )

    def _apply_dst_negative_correlation(self, optimizer, players: List[Player]) -> None:
        """A defense cannot share a lineup with the opposing QB or kicker."""
        by_team_pos: Dict[str, Dict[str, List[Player]]] = {}
        for player in players:
            pos = self._football_pos(player)
            if pos not in {"DST", "K", "QB"}:
                continue
            by_team_pos.setdefault(player.team, {}).setdefault(pos, []).append(player)

        for team, positions in by_team_pos.items():
            defenses = positions.get("DST") or []
            if not defenses:
                continue
            for other_team, other_positions in by_team_pos.items():
                if other_team == team:
                    continue
                for blocked_pos in ("K", "QB"):
                    blocked = other_positions.get(blocked_pos) or []
                    if not blocked:
                        continue
                    # Sum of this defense and the opposing piece is at most 1.
                    optimizer.add_players_group(
                        PlayersGroup(defenses + blocked, max_from_group=1)
                    )

    def _build_optimizer(self, players: List[Player]):
        optimizer = get_optimizer(self.site, self.sport)
        # Optional window. Showdown presets leave $200–$2,000 under the DK cap.
        self._apply_salary_caps(optimizer)
        self._apply_injury_filter(optimizer)
        self._apply_kicker_exposure_cap(players)
        optimizer.player_pool.load_players(players)
        if self.min_teams is not None:
            optimizer.set_total_teams(min_teams=int(self.min_teams))
        if self.max_repeating_players is not None:
            optimizer.set_max_repeating_players(self.max_repeating_players)
        self._apply_captain_eligibility(optimizer, players)
        self._apply_dst_ban(optimizer, players)
        self._apply_projection_floor(optimizer, players)
        self._apply_confirmed_quarterbacks(optimizer, players)
        self._apply_flex_salary_floor(optimizer, players)
        self._apply_locked_players(optimizer)
        self._apply_cheap_player_cap(optimizer, players)
        self._apply_same_team_rb_limit(optimizer, players)
        self._apply_game_script(optimizer, players)
        self._apply_kicker_limit(optimizer, players)
        self._apply_dst_limit(optimizer, players)
        self._apply_dst_negative_correlation(optimizer, players)
        self._apply_ownership_cap(optimizer)
        return optimizer

    def generate_lineups(self, n: Optional[int] = None) -> List[pd.DataFrame]:
        count = self.lineup_count if n is None else int(n)
        if not 1 <= count <= 150:
            raise ValueError("lineup count must be between 1 and 150")
        players = self._players_for_optimize()
        optimizer = self._build_optimizer(players)
        raw_lineups = list(
            optimizer.optimize(
                count,
                max_exposure=self.max_exposure,
            )
        )
        return [self.format_lineup(lineup) for lineup in raw_lineups]


def classic_gpp_rule_check(
    lineups: Sequence[pd.DataFrame],
    *,
    overlap_cap: Optional[int] = None,
    salary_min: Optional[int] = None,
    salary_max: int = DK_SALARY_CAP,
    flex_positions: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Score each Classic lineup against the large-field stack, bring-back, salary, and overlap rules."""
    rows = []
    id_sets = []
    for lineup in lineups:
        ids = set(lineup["ID"].astype(str)) if "ID" in lineup.columns else set(lineup["Name"].astype(str))
        id_sets.append(ids)
    for index, lineup in enumerate(lineups):
        reasons: List[str] = []
        qb_rows = lineup.loc[lineup["Slot"] == "QB"] if "Slot" in lineup.columns else lineup.iloc[0:0]
        if qb_rows.empty:
            stack = "No QB"
            bring_back = ""
            reasons.append("No QB")
        else:
            qb = qb_rows.iloc[0]
            qb_team = str(qb["Team"])
            opponent = str(qb.get("Opponent", "") or "")
            catchers = lineup[
                (lineup["Team"].astype(str) == qb_team)
                & lineup["Position"].map(lambda pos: bool(_position_tokens(pos) & set(PASS_CATCHERS)))
            ]
            wideouts = [
                name
                for name, pos in zip(catchers["Name"], catchers["Position"])
                if "WR" in _position_tokens(pos)
            ]
            catcher_names = [str(name).strip() for name in catchers["Name"]]
            stack = " + ".join([str(qb["Name"]).strip(), *catcher_names]) if catcher_names else str(qb["Name"]).strip()
            if len(catchers) < 2:
                reasons.append("Stack needs 2 pass-catchers")
            if not wideouts:
                reasons.append("Stack needs a WR")
            if opponent:
                bring_rows = lineup[
                    (lineup["Team"].astype(str) == opponent)
                    & lineup["Position"].map(
                        lambda pos: bool(_position_tokens(pos) & set(SKILL_POSITIONS))
                    )
                ]
            else:
                bring_rows = lineup.iloc[0:0]
                reasons.append("No bring-back")
            bring_names = [str(name).strip() for name in bring_rows["Name"]] if len(bring_rows) else []
            bring_back = ", ".join(bring_names)
            if opponent and not bring_names:
                reasons.append("No bring-back")
        spent = int(lineup.attrs.get("salary", int(lineup["Salary"].sum())))
        if salary_min is not None and not salary_min <= spent <= int(salary_max):
            reasons.append(f"Salary outside ${int(salary_min):,}–${int(salary_max):,}")
        shared = 0
        for other_index, other_ids in enumerate(id_sets):
            if other_index == index:
                continue
            shared = max(shared, len(id_sets[index] & other_ids))
        if overlap_cap is not None and shared > int(overlap_cap):
            reasons.append(f"Shares {shared} players")
        flex_label = ""
        if "Slot" in lineup.columns:
            flex_rows = lineup.loc[lineup["Slot"] == "FLEX"]
            if not flex_rows.empty:
                flex_label = str(flex_rows.iloc[0]["Position"])
                allowed_flex = {pos.upper() for pos in flex_positions} if flex_positions else set()
                if allowed_flex and not (_position_tokens(flex_label) & allowed_flex):
                    reasons.append(f"Flex is {flex_label}")
        rows.append(
            {
                "Lineup": index + 1,
                "Stack": stack,
                "Bring-back": bring_back,
                "Flex": flex_label,
                "Salary remaining": int(DK_SALARY_CAP - spent),
                "Max overlap": shared,
                "Result": "Pass" if not reasons else "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def tiered_lineup_counts(total_n: int, leave_money_ratio: float) -> Tuple[int, int]:
    """Split ``total_n`` into ``(full-cap count, leave-money count)``."""
    total_n = int(total_n)
    ratio = float(leave_money_ratio)
    if total_n < 1:
        raise ValueError("total_n must be at least 1")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("leave_money_ratio must be between 0 and 1")
    standard_n = int(round((1.0 - ratio) * total_n))
    standard_n = max(0, min(total_n, standard_n))
    return standard_n, total_n - standard_n


def optimize_tiered_salaries(
    optimizer: BaseNFLOptimizer,
    total_n: int,
    leave_money_ratio: float = 0.4,
    standard_cap: int = DK_SALARY_CAP,
    discount_cap: int = 49_500,
    discount_floor: int = 48_000,
) -> List[pd.DataFrame]:
    """
    Build a portfolio where only ``leave_money_ratio`` of lineups spend under the cap.

    The first batch uses salary bounds of $49,800–``standard_cap`` and contains
    ``(1 - leave_money_ratio) * total_n`` lineups. The rest are solved after
    ``set_max_salary_cap(discount_cap)`` and ``set_min_salary_cap(discount_floor)``.
    Salary settings on ``optimizer`` are restored when the call finishes.
    """
    standard_n, discount_n = tiered_lineup_counts(total_n, leave_money_ratio)
    standard_cap = int(standard_cap)
    discount_cap = int(discount_cap)
    discount_floor = int(discount_floor)
    if STANDARD_TIER_FLOOR > standard_cap:
        raise ValueError(
            f"standard_cap must be at least ${STANDARD_TIER_FLOOR:,}"
        )
    if discount_floor > discount_cap:
        raise ValueError("discount_floor cannot exceed discount_cap")

    previous_min = optimizer.min_salary_cap
    previous_max = optimizer.max_salary_cap
    lineups: List[pd.DataFrame] = []
    try:
        if standard_n:
            optimizer.set_max_salary_cap(standard_cap)
            optimizer.set_min_salary_cap(STANDARD_TIER_FLOOR)
            lineups.extend(optimizer.generate_lineups(standard_n))
        if discount_n:
            optimizer.set_max_salary_cap(discount_cap)
            optimizer.set_min_salary_cap(discount_floor)
            lineups.extend(optimizer.generate_lineups(discount_n))
    finally:
        optimizer.set_min_salary_cap(previous_min)
        optimizer.set_max_salary_cap(previous_max)
    return lineups


_GAME_SCRIPT_FIELDS = (
    "min_salary_cap",
    "max_salary_cap",
    "enable_qb_stack",
    "stack_pass_catchers",
    "enable_bring_back",
    "bring_back_position",
    "require_wr_in_stack",
    "slate_aware_gpp",
    "flex_positions",
    "exclude_vs_dst",
    "max_repeating_players",
)


def optimize_game_scripts(
    optimizer: "NFLClassicOptimizer",
    buckets: Sequence[GameScriptBucket],
    max_repeating_players: int = GAME_SCRIPT_MAX_REPEATING,
) -> List[pd.DataFrame]:
    """Solve one Classic bucket per game, then cap overlap across the portfolio.

    Each bucket uses that game's quarterback, one same-team pass catcher, an
    opposing skill player, a WR/RB flex, no skill player against the defense,
    and that bucket's salary cap. Salary, stack, flex, and overlap settings
    are restored when the portfolio finishes.
    """
    if not isinstance(optimizer, NFLClassicOptimizer):
        raise TypeError("Game-script allocation is for Classic lineups.")
    repeating = int(max_repeating_players)
    if repeating < 1:
        raise ValueError("max repeating players must be at least 1")
    saved = {name: getattr(optimizer, name) for name in _GAME_SCRIPT_FIELDS}
    prior: List[pd.DataFrame] = []
    try:
        optimizer.set_qb_stack(
            True, pass_catchers=1, bring_back=True, bring_back_position="SKILL"
        )
        optimizer.require_wr_in_stack = False
        optimizer.slate_aware_gpp = False
        optimizer.flex_positions = GAME_SCRIPT_FLEX
        optimizer.exclude_vs_dst = True
        optimizer.set_max_repeating_players(repeating)
        optimizer.set_min_salary_cap(None)
        for bucket in buckets:
            count = int(bucket.lineups)
            if count < 0:
                raise ValueError("lineup count must be >= 0")
            if count == 0:
                continue
            away, home = _teams_in_game(bucket.game)
            optimizer.set_max_salary_cap(int(bucket.salary_cap))
            prior.extend(
                optimizer.generate_lineups(
                    count,
                    existing_lineups=prior,
                    quarterback_teams=(away, home),
                    game_script=f"{away}@{home}",
                )
            )
        return prior
    finally:
        for name, value in saved.items():
            setattr(optimizer, name, value)


def game_script_summary(lineups: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Lineup counts per game, in the order the games were solved."""
    counts: Dict[str, int] = {}
    order: List[str] = []
    for lineup in lineups:
        game = str(lineup.attrs.get("game_script") or "")
        if game not in counts:
            order.append(game)
            counts[game] = 0
        counts[game] += 1
    if not order:
        return pd.DataFrame(columns=["Game", "Lineups"])
    return pd.DataFrame(
        {"Game": order, "Lineups": [counts[game] for game in order]}
    )


def player_exposure_table(lineups: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """How often each player appears in the batch, highest exposure first."""
    columns = ["Name", "Team", "Position", "Lineups", "Exposure %"]
    total = len(lineups)
    if total == 0:
        return pd.DataFrame(columns=columns)
    pieces = [
        lineup[["ID", "Name", "Team", "Position"]].copy()
        for lineup in lineups
        if not lineup.empty
    ]
    if not pieces:
        return pd.DataFrame(columns=columns)
    stacked = pd.concat(pieces, ignore_index=True)
    grouped = (
        stacked.groupby(["ID", "Name", "Team", "Position"], as_index=False)
        .size()
        .rename(columns={"size": "Lineups"})
    )
    grouped["Exposure %"] = (grouped["Lineups"] / total * 100).round(1)
    grouped = grouped.sort_values(
        ["Lineups", "Name"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    return grouped.drop(columns=["ID"])


def get_nfl_optimizer(contest_format: ContestFormat | str) -> BaseNFLOptimizer:
    """Factory for Classic vs Showdown optimizer instances."""
    fmt = ContestFormat(contest_format)
    if fmt is ContestFormat.SHOWDOWN:
        return NFLShowdownOptimizer()
    return NFLClassicOptimizer()
