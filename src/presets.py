"""Contest preset definitions and solver configuration for Classic / Showdown."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

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
    cpt_qb_pass_catchers: int = 2
    max_k_dst_combined: Optional[int] = None
    # Showdown roster cap. 1 blocks both teams' kickers in one lineup. None = no cap.
    max_kickers: Optional[int] = None
    allow_kicker_captain: bool = False
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
        cpt_qb_pass_catchers=2,
        max_k_dst_combined=2,
        max_kickers=1,
        max_repeating_players=3,
        allow_kicker_captain=False,
        description=(
            "Showdown GPP: salary window, both teams, CPT QB needs 2 WR/TE "
            "plus an opposing skill player, no K or DST captains, at most one kicker, "
            "max 3 repeats."
        ),
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
        max_kickers=1,
        max_repeating_players=None,
        allow_kicker_captain=True,
        description="Showdown Cash: low/no jitter, few lineups, both teams required, at most one kicker.",
    ),
}


def presets_for_format(contest_format: ContestFormat) -> Tuple[ContestPreset, ...]:
    if contest_format is ContestFormat.SHOWDOWN:
        return SHOWDOWN_PRESETS
    return CLASSIC_PRESETS


def get_preset_config(preset: ContestPreset) -> PresetConfig:
    return PRESET_CONFIGS[preset]


# Identity fields stay on the preset. Lineup count is applied separately.
_LOCKED_OVERRIDE_FIELDS = frozenset(
    {"format", "label", "description", "lineup_count_range", "default_lineup_count"}
)


def config_with_overrides(
    config: PresetConfig,
    *,
    lineup_count: int,
    overrides: Dict[str, Any],
) -> PresetConfig:
    """
    Copy ``config`` with solver fields replaced by UI overrides.

    Unknown keys are ignored. ``lineup_count`` becomes the count shown as the
    actual lineup count (the allowed range stays the preset's).
    """
    allowed = {f.name for f in fields(PresetConfig)} - _LOCKED_OVERRIDE_FIELDS
    updates = {key: overrides[key] for key in allowed if key in overrides}
    updates["default_lineup_count"] = int(lineup_count)
    return replace(config, **updates)


def _fmt_pct_band(band: Optional[Tuple[float, float]]) -> str:
    if band is None:
        return "None"
    lo, hi = float(band[0]), float(band[1])
    if lo == 0.0 and hi == 0.0:
        return "None (deterministic)"
    if lo == hi:
        return f"{lo * 100:.0f}%"
    return f"{lo * 100:.0f}–{hi * 100:.0f}%"


def format_preset_summary(config: PresetConfig) -> List[Tuple[str, str]]:
    """
    Build readable (setting, value) rows for UI display from a PresetConfig.

    Single source of truth: derive labels from the config fields / description,
    do not hardcode per-preset copy beyond PresetConfig itself.
    """
    lo, hi = config.lineup_count_range
    rows: List[Tuple[str, str]] = [
        (
            "Lineup count",
            f"{config.default_lineup_count} (allowed {lo}–{hi})",
        ),
    ]

    if config.position_jitter_sds:
        band = _fmt_pct_band(config.jitter_pct)
        rows.append(
            (
                "Projection jitter",
                f"Position SDs (~{band} relative)",
            )
        )
    else:
        rows.append(("Projection jitter", _fmt_pct_band(config.jitter_pct)))

    if config.format is ContestFormat.CLASSIC:
        if config.enable_qb_stack:
            stack = f"QB + {config.stack_pass_catchers} WR/TE"
            if config.enable_bring_back:
                stack += f" + {config.bring_back_position} bring-back"
            rows.append(("Stacks", stack))
        else:
            rows.append(("Stacks", "None"))
        rows.append(("Exclude vs DST", "Yes" if config.exclude_vs_dst else "No"))
    else:
        min_sal = config.min_salary_cap
        max_sal = config.max_salary_cap
        if min_sal is not None and max_sal is not None:
            rows.append(("Salary window", f"${min_sal:,}–${max_sal:,}"))
        elif min_sal is not None:
            rows.append(("Salary window", f"min ${min_sal:,}"))
        elif max_sal is not None:
            rows.append(("Salary window", f"max ${max_sal:,}"))
        else:
            rows.append(("Salary window", "Full cap"))

        rows.append(
            (
                "Min teams",
                str(config.min_teams) if config.min_teams is not None else "No minimum",
            )
        )
        rows.append(
            (
                "Game-script correlation",
                "Yes" if config.game_script_correlation else "No",
            )
        )
        if config.game_script_correlation:
            rows.append(
                ("CPT QB pass-catchers", str(config.cpt_qb_pass_catchers))
            )
        rows.append(
            (
                "Kicker captain",
                "Yes" if config.allow_kicker_captain else "No",
            )
        )
        rows.append(
            (
                "Max K+DST combined",
                (
                    str(config.max_k_dst_combined)
                    if config.max_k_dst_combined is not None
                    else "No limit"
                ),
            )
        )
        rows.append(
            (
                "Max kickers",
                str(config.max_kickers) if config.max_kickers is not None else "No limit",
            )
        )

    rows.append(
        (
            "Max repeating players",
            (
                str(config.max_repeating_players)
                if config.max_repeating_players is not None
                else "No limit"
            ),
        )
    )

    if config.max_cumulative_ownership is not None:
        rows.append(
            (
                "Ownership cap",
                f"{config.max_cumulative_ownership * 100:.0f}% cumulative",
            )
        )
    else:
        rows.append(("Ownership cap", "None"))

    if config.description:
        rows.append(("Strategy notes", config.description))

    return rows


def format_settings_comparison(
    preset: PresetConfig,
    actual: PresetConfig,
    extra_rows: Optional[List[Tuple[str, str, str]]] = None,
) -> List[Tuple[str, str, str]]:
    """
    Align preset defaults with the settings that will actually be solved.

    Each row is ``(setting, preset default, actual)``. Strategy notes stay on
    the preset description, not in this comparison. ``extra_rows`` appends
    UI-only settings (exposure, non-default jitter seed) in the same shape.
    """
    actual_map = dict(format_preset_summary(actual))
    rows = [
        (setting, default, actual_map.get(setting, "—"))
        for setting, default in format_preset_summary(preset)
        if setting != "Strategy notes"
    ]
    if extra_rows:
        rows.extend(extra_rows)
    return rows


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
            max_kickers=config.max_kickers,
            allow_kicker_captain=config.allow_kicker_captain,
        )

    return config
