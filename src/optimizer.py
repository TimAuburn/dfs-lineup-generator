"""Wrapper around pydfs-lineup-optimizer for DraftKings NFL Classic."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from pydfs_lineup_optimizer import Player, Site, Sport, get_optimizer
from pydfs_lineup_optimizer.player import GameInfo
from pydfs_lineup_optimizer.stacks import PositionsStack

from .parser import parse_dk_csv

DK_SALARY_CAP = 50_000
ROSTER_SLOTS = ("QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST")


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
        if key not in cache:
            cache[key] = GameInfo(home_team=home_team, away_team=away_team, starts_at=None)
        return cache[key]
    except ValueError:
        return None


def _split_name(full_name: str) -> tuple[str, str]:
    parts = str(full_name).strip().split(maxsplit=1)
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def dataframe_to_players(
    df: pd.DataFrame,
    *,
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
            )
        )
    return players


class NFLClassicOptimizer:
    """Configurable DraftKings NFL Classic lineup generator."""

    def __init__(self) -> None:
        self._players: List[Player] = []
        self._player_df: Optional[pd.DataFrame] = None
        self.lineup_count: int = 1
        self.min_exposure: Optional[float] = None
        self.max_exposure: Optional[float] = None
        self.max_repeating_players: Optional[int] = None
        self.enable_qb_stack: bool = False
        self.stack_pass_catchers: int = 1
        self.enable_bring_back: bool = False
        self.bring_back_position: str = "WR"

    def load_from_csv(self, path: Union[str, Path]) -> pd.DataFrame:
        df = parse_dk_csv(path)
        return self.load_from_dataframe(df)

    def load_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        self._player_df = df.copy()
        self._players = dataframe_to_players(
            self._player_df,
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

    def _build_optimizer(self):
        if not self._players:
            raise RuntimeError("No players loaded. Call load_from_csv or load_from_dataframe first.")
        optimizer = get_optimizer(Site.DRAFTKINGS, Sport.FOOTBALL)
        optimizer.player_pool.load_players(self._players)
        if self.max_repeating_players is not None:
            optimizer.set_max_repeating_players(self.max_repeating_players)
        if self.enable_qb_stack:
            stack_positions: List[Union[str, tuple]] = ["QB"]
            for _ in range(self.stack_pass_catchers):
                stack_positions.append(("WR", "TE"))
            optimizer.add_stack(PositionsStack(stack_positions))
        if self.enable_bring_back:
            optimizer.force_positions_for_opposing_team(("QB", self.bring_back_position))
        return optimizer

    def generate_lineups(self, n: Optional[int] = None) -> List[pd.DataFrame]:
        """Generate lineups and return each as a DataFrame of rostered players."""
        count = self.lineup_count if n is None else int(n)
        if not 1 <= count <= 150:
            raise ValueError("lineup count must be between 1 and 150")
        optimizer = self._build_optimizer()
        raw_lineups = list(
            optimizer.optimize(
                count,
                max_exposure=self.max_exposure,
            )
        )
        return [self.format_lineup(lineup) for lineup in raw_lineups]

    def format_lineup(self, lineup) -> pd.DataFrame:
        """Format a pydfs Lineup as a roster DataFrame."""
        rows = []
        for player in lineup.lineup:
            rows.append(
                {
                    "Slot": player.lineup_position,
                    "ID": player.id,
                    "Name": player.full_name,
                    "Name + ID": f"{player.full_name} ({player.id})",
                    "Position": "/".join(player.positions),
                    "Team": player.team,
                    "Salary": int(player.salary),
                    "Projection": float(player.fppg),
                }
            )
        df = pd.DataFrame(rows)
        df.attrs["salary"] = int(lineup.salary_costs)
        df.attrs["projection"] = float(lineup.fantasy_points_projection)
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
            return pd.DataFrame(columns=["Lineup", *ROSTER_SLOTS])
        return pd.concat(frames, ignore_index=True)
