"""Wrappers around pydfs-lineup-optimizer for DraftKings NFL Classic and Showdown."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pydfs_lineup_optimizer import Player, Site, Sport, get_optimizer
from pydfs_lineup_optimizer.player import GameInfo
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
# Captain-QB FLEX quota is WR and TE only. RB, K, and DST do not count.
CPT_QB_PASS_CATCHER_POSITIONS = frozenset({"WR", "TE"})
_NOT_CPT_QB_PASS_CATCHERS = frozenset({"K", "DST", "DEF", "RB", "QB"})
# Opposing FLEX piece required with a Captain QB.
QB_BRING_BACK_POSITIONS = frozenset({"WR", "TE", "RB"})
# Captain is a skill spot unless a low-total slate explicitly allows a kicker.
CAPTAIN_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})


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
        (typically 1 or 2). Bring-back forces a player from the opposing team
        at ``bring_back_position`` (WR or TE).
        """
        if pass_catchers < 1:
            raise ValueError("pass_catchers must be at least 1")
        if bring_back_position not in {"WR", "TE"}:
            raise ValueError("bring_back_position must be WR or TE")
        self.enable_qb_stack = bool(enabled)
        self.stack_pass_catchers = int(pass_catchers)
        self.enable_bring_back = bool(bring_back)
        self.bring_back_position = bring_back_position

    def _build_optimizer(self, players: List[Player]):
        optimizer = get_optimizer(self.site, self.sport)
        self._apply_salary_caps(optimizer)
        self._apply_injury_filter(optimizer)
        optimizer.player_pool.load_players(players)
        if self.max_repeating_players is not None:
            optimizer.set_max_repeating_players(self.max_repeating_players)
        if self.enable_qb_stack:
            stack_positions: List[Union[str, tuple]] = ["QB"]
            for _ in range(self.stack_pass_catchers):
                stack_positions.append(("WR", "TE"))
            optimizer.add_stack(PositionsStack(stack_positions))
        if self.enable_bring_back:
            optimizer.force_positions_for_opposing_team(("QB", self.bring_back_position))
        self._apply_exclude_vs_dst(optimizer)
        self._apply_ownership_cap(optimizer)
        return optimizer

    def generate_lineups(self, n: Optional[int] = None) -> List[pd.DataFrame]:
        """Generate lineups and return each as a DataFrame of rostered players."""
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
        # Showdown batches share at most 3 players so lineups do not collapse together.
        self.max_repeating_players = 3

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
    ) -> None:
        if cpt_qb_pass_catchers < 1:
            raise ValueError("cpt_qb_pass_catchers must be at least 1")
        if max_kickers is not None and int(max_kickers) < 0:
            raise ValueError("max_kickers must be at least 0")
        self.min_salary_cap = min_salary_cap
        self.max_salary_cap = max_salary_cap
        self.min_teams = min_teams
        self.game_script_correlation = bool(game_script_correlation)
        self.cpt_qb_pass_catchers = int(cpt_qb_pass_catchers)
        self.max_k_dst_combined = max_k_dst_combined
        self.max_kickers = None if max_kickers is None else int(max_kickers)
        self.allow_kicker_captain = bool(allow_kicker_captain)

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
        allowed = set(CAPTAIN_POSITIONS)
        if self.allow_kicker_captain:
            allowed.add("K")
        return frozenset(allowed)

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
        """Keep CPT on QB/RB/WR/TE. A kicker CPT is allowed only when the flag is on."""
        allowed = self._captain_positions_allowed()
        for player in players:
            if "CPT" not in player.positions:
                continue
            if self._football_pos(player) not in allowed:
                optimizer.player_pool.remove_player(player)

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
        self._apply_game_script(optimizer, players)
        self._apply_kicker_limit(optimizer, players)
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


def get_nfl_optimizer(contest_format: ContestFormat | str) -> BaseNFLOptimizer:
    """Factory for Classic vs Showdown optimizer instances."""
    fmt = ContestFormat(contest_format)
    if fmt is ContestFormat.SHOWDOWN:
        return NFLShowdownOptimizer()
    return NFLClassicOptimizer()
