"""Contest preset definitions and solver configuration for Classic / Showdown."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .optimizer import BaseNFLOptimizer


class ContestFormat(str, Enum):
    CLASSIC = "classic"
    SHOWDOWN = "showdown"


class ContestPreset(str, Enum):
    CLASSIC_CASH = "classic_cash"
    CLASSIC_SMALL_GPP = "classic_small_gpp"
    CLASSIC_LARGE_GPP = "classic_large_gpp"
    SHOWDOWN_GPP = "showdown_gpp"
    SHOWDOWN_CASH = "showdown_cash"


CLASSIC_PRESETS = (
    ContestPreset.CLASSIC_CASH,
    ContestPreset.CLASSIC_SMALL_GPP,
    ContestPreset.CLASSIC_LARGE_GPP,
)

SHOWDOWN_PRESETS = (
    ContestPreset.SHOWDOWN_GPP,
    ContestPreset.SHOWDOWN_CASH,
)

PRESET_LABELS = {
    ContestPreset.CLASSIC_CASH: "Cash (50/50s & Double-Ups)",
    ContestPreset.CLASSIC_SMALL_GPP: "Small-Field / Single-Entry GPP",
    ContestPreset.CLASSIC_LARGE_GPP: "Large-Field Multi-Entry GPP (20–150 Max)",
    ContestPreset.SHOWDOWN_GPP: "Showdown GPP",
    ContestPreset.SHOWDOWN_CASH: "Showdown Cash",
}

# Position-specific relative SDs for large-field Classic GPP (within ~12–18%).
LARGE_FIELD_POSITION_SDS = {
    "QB": 0.12,
    "RB": 0.15,
    "WR": 0.18,
    "TE": 0.16,
    "DST": 0.14,
    "K": 0.14,
}


@dataclass(frozen=True)
class PresetConfig:
    """Declarative solver rules for a contest preset."""

    format: ContestFormat
    label: str
    lineup_count_range: Tuple[int, int]
    default_lineup_count: int
    # Jitter: either fixed 0.0, a (min, max) uniform relative band, or position SDs.
    jitter_pct: Optional[Tuple[float, float]] = None
    position_jitter_sds: Optional[Dict[str, float]] = None
    enable_qb_stack: bool = False
    stack_pass_catchers: int = 1
    enable_bring_back: bool = False
    bring_back_position: str = "WR"
    exclude_vs_dst: bool = False
    max_repeating_players: Optional[int] = None
    # Cumulative ownership cap as a fraction (1.25 == 125%). None = do not apply.
    max_cumulative_ownership: Optional[float] = None
    # Showdown-specific
    min_salary_cap: Optional[int] = None
    max_salary_cap: Optional[int] = None
    min_teams: Optional[int] = None
    game_script_correlation: bool = False
    cpt_qb_pass_catchers: int = 1
    max_k_dst_combined: Optional[int] = None
    description: str = ""


PRESET_CONFIGS: Dict[ContestPreset, PresetConfig] = {
    ContestPreset.CLASSIC_CASH: PresetConfig(
        format=ContestFormat.CLASSIC,
        label=PRESET_LABELS[ContestPreset.CLASSIC_CASH],
        lineup_count_range=(1, 1),
        default_lineup_count=1,
        jitter_pct=(0.0, 0.0),
        enable_qb_stack=False,
        exclude_vs_dst=False,
        max_repeating_players=None,
        description="Single deterministic cash lineup; maximize projected points, no stacks.",
    ),
    ContestPreset.CLASSIC_SMALL_GPP: PresetConfig(
        format=ContestFormat.CLASSIC,
        label=PRESET_LABELS[ContestPreset.CLASSIC_SMALL_GPP],
        lineup_count_range=(1, 5),
        default_lineup_count=3,
        jitter_pct=(0.05, 0.08),
        enable_qb_stack=True,
        stack_pass_catchers=1,
        enable_bring_back=False,
        exclude_vs_dst=True,
        max_repeating_players=6,
        description="Small-field GPP: light jitter, QB + 1 WR/TE, exclude vs DST, max 6 repeats.",
    ),
    ContestPreset.CLASSIC_LARGE_GPP: PresetConfig(
        format=ContestFormat.CLASSIC,
        label=PRESET_LABELS[ContestPreset.CLASSIC_LARGE_GPP],
        lineup_count_range=(20, 150),
        default_lineup_count=20,
        jitter_pct=(0.12, 0.18),
        position_jitter_sds=LARGE_FIELD_POSITION_SDS,
        enable_qb_stack=True,
        stack_pass_catchers=2,
        enable_bring_back=True,
        bring_back_position="WR",
        exclude_vs_dst=True,
        max_repeating_players=5,
        max_cumulative_ownership=1.25,
        description="Large-field multi-entry: position jitter, QB double stack + bring-back, ownership cap.",
    ),
    ContestPreset.SHOWDOWN_GPP: PresetConfig(
        format=ContestFormat.SHOWDOWN,
        label=PRESET_LABELS[ContestPreset.SHOWDOWN_GPP],
        lineup_count_range=(1, 150),
        default_lineup_count=10,
        jitter_pct=(0.08, 0.14),
        min_salary_cap=48_000,
        max_salary_cap=49_800,
        min_teams=2,
        game_script_correlation=True,
        cpt_qb_pass_catchers=1,
        max_k_dst_combined=2,
        max_repeating_players=4,
        description="Showdown GPP: salary window, both teams, CPT game-script stacks, K/DST ≤ 2.",
    ),
    ContestPreset.SHOWDOWN_CASH: PresetConfig(
        format=ContestFormat.SHOWDOWN,
        label=PRESET_LABELS[ContestPreset.SHOWDOWN_CASH],
        lineup_count_range=(1, 5),
        default_lineup_count=1,
        jitter_pct=(0.0, 0.0),
        min_salary_cap=48_000,
        max_salary_cap=50_000,
        min_teams=2,
        game_script_correlation=False,
        max_k_dst_combined=2,
        max_repeating_players=None,
        description="Showdown Cash: low/no jitter, few lineups, both teams required.",
    ),
}


def presets_for_format(contest_format: ContestFormat) -> Tuple[ContestPreset, ...]:
    if contest_format is ContestFormat.SHOWDOWN:
        return SHOWDOWN_PRESETS
    return CLASSIC_PRESETS


def get_preset_config(preset: ContestPreset) -> PresetConfig:
    return PRESET_CONFIGS[preset]


def apply_preset(
    optimizer: "BaseNFLOptimizer",
    preset: ContestPreset,
    *,
    lineup_count: Optional[int] = None,
    jitter_seed: Optional[int] = 42,
) -> PresetConfig:
    """
    Configure ``optimizer`` from a contest preset.

    Jitter is applied later (at generate time) using ``optimizer.jitter_*`` fields
    so editable projections in the UI remain the baseline.
    """
    config = get_preset_config(preset)
    if config.format.value != optimizer.contest_format:
        raise ValueError(
            f"Preset {preset.value} is for {config.format.value}, "
            f"but optimizer is {optimizer.contest_format}"
        )

    lo, hi = config.lineup_count_range
    count = config.default_lineup_count if lineup_count is None else int(lineup_count)
    count = max(lo, min(hi, count))
    optimizer.set_lineup_count(count)

    optimizer.jitter_pct = config.jitter_pct
    optimizer.position_jitter_sds = (
        dict(config.position_jitter_sds) if config.position_jitter_sds else None
    )
    optimizer.jitter_seed = jitter_seed

    optimizer.set_max_repeating_players(config.max_repeating_players)
    optimizer.exclude_vs_dst = config.exclude_vs_dst
    optimizer.max_cumulative_ownership = config.max_cumulative_ownership

    if hasattr(optimizer, "set_qb_stack"):
        optimizer.set_qb_stack(
            enabled=config.enable_qb_stack,
            pass_catchers=config.stack_pass_catchers,
            bring_back=config.enable_bring_back,
            bring_back_position=config.bring_back_position,
        )

    if hasattr(optimizer, "configure_showdown_rules"):
        optimizer.configure_showdown_rules(
            min_salary_cap=config.min_salary_cap,
            max_salary_cap=config.max_salary_cap,
            min_teams=config.min_teams,
            game_script_correlation=config.game_script_correlation,
            cpt_qb_pass_catchers=config.cpt_qb_pass_catchers,
            max_k_dst_combined=config.max_k_dst_combined,
        )

    return config
